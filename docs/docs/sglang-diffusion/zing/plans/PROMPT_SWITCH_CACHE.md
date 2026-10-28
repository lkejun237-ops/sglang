# Zing H3 prompt switch Block0 cache

2026-10-27，当前 `feat/zing-preparation-cache` 工作区实现。基于 `df0f677e97` 的初始
Block0 cache 接入继续修改；本版按评审收敛到同一个 cache、一个内容 key、原位恢复和串行历史重建。

## 实现

启动 TTFF 和流式 prompt switch 共用 stage 持有的同一个 `Block0PreparationCache`。
CPU 存储在启动时一次预分配，使用 `OrderedDict` 管理 LRU。默认 **6 个条目/rank**，
通过 `SGLANG_ZING_BLOCK0_CACHE_ENTRIES` 调整。淘汰复用原 CPU slot，不在请求中重新 pin 数 GiB。
沿用 `SGLANG_ZING_BLOCK0_CACHE_MODE=off|late|early|overlap`；启用示例：

```bash
export SGLANG_ZING_BLOCK0_CACHE_MODE=overlap
export SGLANG_ZING_BLOCK0_CACHE_ENTRIES=6
```

模式默认仍为 off。late/early 保留初始请求的时序消融含义；prompt switch 命中均跳过 Qwen 编码，
并恢复 Block0 后走原有三段串行 `_commit_clean_prefix`。early 会同步等待恢复，overlap 在消费端等待 event。
缓存仅支持现有固定部署、单活跃会话及 GPU KV pool，不新增跨进程持久化或热重载协议。

| 文件 | 本次职责 |
| --- | --- |
| `preparation_cache.py` | 多条目 pinned LRU、单内容 key、复用 slot、现有 H2D/D2H |
| `stage.py` | 读取条目数，构造唯一 cache 实例 |
| `source_session.py` | 复用启动请求内容生成切换 key，命中恢复准备产物，当前 pool admission，首输出后回写 |
| `incremental.py` | 在真正切换边界选择恢复路径，捕获已计算的新 prefix |
| `sampling.py` | clear 后恢复或 forward Block0；原有 clean history 循环不变 |

## 单一内容 key

初始消息经现有 parser 得到 prompt、media、target、conditions、source；媒体 payload/URI 内容以摘要
保存为本会话接受的请求快照。`preparation_key` 对该快照、S0、source 几何和冻结的 policy 计算唯一 SHA256。
切换只替换 prompt；reference 更新按当前有效 reference 集合替换对应媒体及 conditions。

没有第二级索引或新票据系统。启动 A 和之后切回的 A 使用同一个 key；不包含 event_id、生成 seed、
recent_visual 或全局输出位置。模型及编码配置由 stage 实例固定，重启部署重新分配缓存。
同一会话继续使用已接受的媒体内容；新会话重新读取 URI，识别同路径换图。
缓存快照包含配套 condition/reference latent，恢复时不把重新编码的 latent 与旧 KV 混用。

## 切换与回写

```text
_update_prompt：查询 → 准备/复用 condition → 当前容量准入 → 保存最新候选
完整 block 边界：等待旧 fill → 查询实际候选 → clear 当前 GPU cache
  hit：同槽 H2D + prefix commit
  miss：正常 Block0 forward（或原部署的六分支 miss 路径）
接着原有 clean history：1 + 2 + 2，t=1
正常生成、输出 → 下一 dispatch 对本 epoch 的 Block0 发起 D2H
```

固定五个 latent 来自最新 recent_visual，不进入静态 key。切回 A 复用最初的 A Block0，
但历史仍按本次画面重建；不恢复以前 A 的历史 KV，不改变 RNG、event 标签、PTS 或输出进度。

切换继续使用本会话已经持有的 GPU slot，不申请第二槽。恢复复用原有 copy stream/event 和 prefix commit。
CPU H2D/D2H 共用同一 copy stream，因此槽位复用的写入排在先前恢复读取之后；下一次 clear 和 dispose 前
调用已有 `finish_fill()`，避免未完成的 D2H 读取被覆盖。不引入源保护事务机或额外 GPU 前缀副本。

新 prefix 捕获时记录 `committed_pixel`；只有 `ledger.output_frames` 越过这个边界才开始 fill，
排除 Decoder 队列里旧 prompt 的输出。prefix 已被替换时跳过回写。所有 rank 统一命中或重算；
部分 SP miss 后，即使本 rank 已有同 key 条目也覆盖该槽，确保所有 shard 属于同次准备。

本版不修改五分支 Graph/Joint。命中直接选择串行 history，绕过要求空 cache 的六分支入口；
miss 保留原有 prompt-joint 配置。清除前的候选失败保留 active 状态，清除后的执行故障沿用关闭会话语义。

## 容量与验证

预分配 KV 字节数/rank：`entries × layers × 2 × max_prefix_rows × local_heads × head_dim × dtype_bytes`。
CPU 不存 rotated K。capacity 日志计全部 pinned slab；准备快照另占少量 CPU 内存。
历史 11596 行实例为 7.741 GiB/rank/条目；六条同规模有效 KV 约 46.446 GiB/rank，实际预分配还取决于
max_prefix_rows。SP2 主机须按两个 rank 汇总预算。

单元测试入口：

- `test/unit/realtime/test_zing_preparation_cache.py`：保留原启动缓存测试。
- `test/unit/realtime/test_zing_prompt_switch_cache.py`：参数化容量、LRU 地址复用、SP 部分命中、
  单 key、候选失败、首输出后回写、dispose 顺序；使用真实小型 CPU DiT 比较 A→B→A→B→A→B 的
  输出、全部层 prefix/history KV、事件和 GPU pool 模拟存储地址。
- 回归覆盖既有 prompt 生命周期、reference media、continuous source session。

本次本地 Python 3.11 / CPU 回归：上述五个文件 **836 passed**；既有 denoise-graph 中
prompt-joint、prefix replay、prompt transition 定向回归 **7 passed、94 deselected**，合计 **843 passed**。
新增 switch cache 文件含 15 个用例（含参数化）；原启动缓存六项文件内容保持不变。
Ruff F/E9/UP037/import 检查、format check、7 个 Python 文件 AST 检查及 `git diff --check` 均通过。
测试环境复用现有 CPU venv 和已缓存 PyAV；首次缺 PyAV 的七项媒体回归补齐环境后全部通过。

CPU 测试仅模拟 CUDA transport，模型、packing、cache commit、串行 history 和采样均执行真实逻辑。
在阿里云灵骏 H100 裸金属节点（832×480，120 帧，SP2）的 A→B→A 实测中已验证统一条目复用：
切回已知 Prompt 时直接跳过 Block0 计算，通过异步 PCIe DMA（~150 ms）与原有 3 段 clean history（~215 ms）流水化衔接，
将 Prefix 重建总耗时从 1.269 秒压缩至 0.370 秒（-70.9%），播放卡顿由 1.66 秒降至 0.48 秒（-70.7%）。
50 层 KV 与全流程 120 帧 RGB 均逐位对齐（SHA256 完全一致）。
实测性能指标详见 [Prompt Switch 缓存结果报告](/benchmark/zing_h3_causal/PROMPT_SWITCH_CACHE_RESULTS) 与 [部署文档](../deployment_zing_h3.mdx)。
