# 同任务 Qwen 前缀 KV 与输出特征复用

2026-10-23；实现基线 `72dbb651d0e1f81d5b2d95f925554ce70d4f0018`。
状态：实现、CPU 回归、B300 Qwen 组件验证和三例 480/736 真实测量完成；Qwen 特征通过容差，跨实例视频等价性尚未建立、人工画质待验收；尚未上线。

## 使用场景与边界

一个世界生成任务中，客户端每次提交完整 prompt 与完整有序媒体。
例如原输入为 `…world… events: N/A …camera constraints…`，下一次替换为
`…world… events: rain …camera constraints…`。服务按**完整 token 序列的共同前缀**
决定复用范围；从第一个变化 token 到结尾全部重算，包括事件之后文本未变的字段。
不会把新事件追加到旧事件之后，也不会按字符串字符位置直接裁剪 KV。

加速的前提是本次切换保持 conditioning 参考媒体不变。参考图内容、数量或顺序变化时，
当前实现清空旧 KV 和输出特征，完整编码并建立新缓存；新参考图固定后的 event 修改
才可再次复用。如果每次 event 都同时换参考图，本版没有前缀复用收益。
这是当前保守的失效策略，不代表理论上无法保留第一处变化之前的序列；本 PR 不实现
按图片定位的部分复用。source 视频帧持续推进不等于 Qwen 参考媒体变化：
`source_session._update_prompt` 未收到 `references` 字段时沿用当前 `condition_media`。

本功能只服务当前任务，部署采用单活跃会话。第二个任务在当前租约存续时得到
`CONDITION_BUSY`；不同任务即使输入相同，也不共享本功能的 KV 或输出特征。
原有无任务完整结果 CPU 缓存继续用于旧客户端/无会话请求；任务请求绕过它。

适用范围是现有独立 conditioning 服务：NVIDIA、TP1、BF16、常驻权重、单编码线程，
输入仍为文本和有序单帧 RGB image/video。模型、processor、tokenizer、projection
在服务启动时固定。暂不增加多帧/音频、跨任务 radix tree、分页 KV、CPU KV offload、
并发活跃任务、DiT KV 复用或 KV 网络传输。

## 接入和回退

服务启动命令沿用现有入口：

```bash
python -m sglang.multimodal_gen.runtime.conditioning.cli serve \
  --config conditioning-service.json --profile conditioning-profile.json
```

服务 `limits` 增加 `prefix_cache_bytes`（默认 `4294967296`，4 GiB）和
`prefix_task_idle_s`（默认 `1800` 秒）。字节预算限制保留的 KV、投影后特征和 token/位置元数据，
不等于总显存或峰值上限；权重、视觉塔、激活、拼接临时张量和 CUDA allocator 另计。
部署时须根据实际 Qwen token 数和共卡余量设置容量。

Worker 的 conditioning 配置可加 `"prefix_reuse": false` 恢复无任务的全量路径。
默认 `true`，且必须在 `/v1/conditioning/profile` 握手中收到
`task_prefix_cache_v1: true` 才使用新协议；旧服务不支持时维持原路径。
本轮扩展了 profile 的代码身份覆盖：Qwen wrapper、backbone、prefix 状态文件也参与哈希。
部署新服务须重新生成 profile 并同步 Worker 的 expected profile，不能沿用旧代码身份。

调用方不需要传 token 差量。`ZingH3GenerationStage.conditioner` 将真实
`RealtimeSession` 的 `WorkerConditioningState` 传给远程 conditioner；无 session 时传 `None`，
不会创建无人负责释放的任务。SP 仅 rank 0 发 RPC/持有服务 handle，其他 rank 沿用既有
验证、结果广播和 tensor 分配流程。会话 dispose 释放 rank 0 租约。

## 状态归属与接口

| 状态 | 所有者 | 内容和生命周期 |
| --- | --- | --- |
| Worker 状态 | `RealtimeSession` | 随机 registration nonce、服务 handle、最近完整输出；dispose 清空并 DELETE |
| 单任务租约 | `SinglePrefixTask` | tenant、nonce、随机 handle、活动请求计数、最后活动时间、closing 标记 |
| 数值缓存 | `QwenPrefixCache` | 一个模型实例、媒体摘要、CPU token IDs/三轴位置、所有选中 LM 层的 DynamicCache KV、GPU BF16 完整输出特征 |
| 编码执行 | 既有 `EncodeQueue` | 单线程串行；只有该线程能执行模型、裁剪/写入/清空数值缓存 |

新接口仍在现有鉴权边界内：

1. `POST /v1/conditioning/tasks/{nonce}`：nonce 是 32 位小写 UUID hex。
   同 tenant、同 nonce 重试返回同一个随机 `task_handle`；其他任务返回 BUSY。
2. `/encode` 的 v1 JSON header 可选增加 `task_handle`。未带该字段的请求格式和结果契约不变。
   服务在解码后验证 tenant/handle，并在排队和返回前检查是否已关闭。
3. `DELETE /v1/conditioning/tasks/{handle}`：幂等释放。陈旧 handle 永远不会释放新任务。
   服务先标记关闭，再在编码 executor 中排队清空缓存；正在运行的 GPU forward 先 drain，
   尚未执行的旧任务请求跳过。清理完成前拒绝新任务。

新任务接口复用请求槽限额；没有新增多 owner 容量或调度队列。
任务接口在同一请求期限内消费空请求体、等待释放及发送响应，断连或超时后释放请求槽；
底层编码 drain 保持 shield，不因 DELETE 的 HTTP 等待结束而提前开放新任务。
任务 singleflight key 包含 handle，避免同内容的新旧任务合并。
新任务请求不写入共享 CPU LRU，只保留一份当前 token/KV/输出状态。

断连或超时仅结束当前 RPC 的等待，运行中的 kernel 必须继续 drain，不能从 HTTP
线程释放其张量。DELETE 调用者断连也不会取消服务端清理。Worker 崩溃时，无活动 RPC
且超过 idle timeout 的租约由后台清理回收；最多额外约 30 秒检查间隔。
原 Worker 在下次编码收到 `CONDITION_TASK_EXPIRED`（HTTP 410）时重新注册并冷编码；
另一任务已占用服务时仍返回 BUSY。服务关闭先终止请求、释放任务，再关闭 executor。
首次注册、编码和租约过期恢复共享一个 `conditioning_timeout_s` 截止时间，
防止多个 RPC 分别重置预算，使等待 rank 0 结果的 SP CPU collective 超时。

## 编码算法与失效

每次完整 processor/presentation 仍运行，得到新 token IDs、row tags 和媒体网格。
本轮不缓存媒体预处理结果，因此请求传输、RGB 哈希和 processor 开销仍存在。
服务对验证后的单帧媒体按内容、形状、类型、顺序计算独立摘要；单帧 FPS 的现有
等价/规范化语义沿用 `inputs.py`，不把媒体地址或形状相同当作内容相同。

1. 对完整新输入调用 `get_rope_index`，得到显式三轴 `position_ids`。
   纯文本也走这个步骤，避免依赖模型全局 `rope_deltas`。
2. 模型实例和媒体摘要相同、缓存有效且容量允许时，比较 token IDs **以及位置**，
   得到最长共同前缀长度 `p`。若待计算部分还含视觉 token，保守全量重算，
   不尝试恢复半个视觉块的 DeepStack 特征。
3. 完全相同或新输入正好是旧输入前缀时，裁剪并返回缓存输出的独立拷贝。
   其他情况将每层 KV 裁剪到 `p`，输入只含 `[p:N]`，其 cache positions 为 `[p:N]`，
   三轴位置取完整输入对应切片。Q 长度 `N-p`，K/V 长度 `N`。
4. 非空复用前缀时不再运行视觉塔；其信息已经在前缀每层 KV 中。
   当前后端只允许既有 FA / TORCH_SDPA；SDPA 已实现非方形的右下对齐因果掩码，
   不另外实现一套 attention。具体 backend/版本仍须真实 GPU 对照。
5. 后缀的选层 hidden state 经原 projection 和 BF16 转换，与缓存的投影后前缀拼接。
   projection 的 `sink_out` 只覆盖**全局 token 0**，后缀 `token_offset>0` 时不能覆盖其首行。
   校验完整形状后保存最新状态。返回与缓存隔离的张量，消费者修改结果不会污染缓存。

| 条件 | 处理 |
| --- | --- |
| 修改已有 event、追加、删除/缩短、改回旧 event、Unicode 导致 token 边界变化 | 按实际 token/位置 LCP 裁剪重算，不维护事件历史分支 |
| 媒体内容、顺序、类型或形状改变 | 丢弃该任务旧数值状态，完整编码 |
| 分叉落入视觉 token 区间 | 完整编码 |
| 需要保留的状态超过预算 | 清除旧状态，本次走不保存 KV 的全量路径；请求仍受原 max rows 限额 |
| 模型实例改变、服务重启/配置或权重 profile 变化 | 冷编码；profile 不符按既有契约拒绝 |
| 模型/processor/传输到 CPU 过程中失败 | Native 清空任务缓存后 drain；下次从全量开始 |
| 取消、EOS、断连后的会话 dispose | Worker DELETE；失败时由服务端 idle lease 回收 |
| 旧 handle 重放或其他 tenant 访问 | 410，不读取或修改当前任务缓存 |

DynamicCache 是可变对象，失败时不能把部分层已追加的状态再次使用；因此异常清空属于
正确性要求。裁剪可能暂时保留旧 tensor storage，预算是当前任务曾允许的状态上限，
不是承诺每次缩短后 driver 显存立即下降；CUDA reserved 也不会随着 tensor 释放立即归还。

## 为什么不直接搬入 SRT RadixCache

官方 [HiCache/RadixAttention 设计](https://docs.sglang.ai/advanced_features/hicache_design.html)
采用 token 前缀匹配和受控 KV 所有权。设计阶段核实官方 SGLang main
`e43f6128a0d0bf98c8c73c42ba69773987b069fe`；这里的目标是私有 Zing multimodal 分支，
二者运行时不能混同。当前 H3 需要指定 LM 中间层、Identity norm、可选 projection/sink，
以及每个 token 的完整输出特征；仅给出前缀 KV 不能补回这些输出行。

同一任务只有最新状态，现有单线程 encoder 已提供顺序。直接使用 DynamicCache 的
update/crop 并保存投影后特征即可覆盖需求。迁移到 SRT 需要额外适配中间层输出、
缓存命中行特征和调度器，当前没有跨任务共享的收益来抵消该代价。
这是复用现有 backbone 的 KV 接口，并非宣称已经接上 SRT RadixCache 服务。

上游 reviewer 应重点核对：可选协议与旧服务兼容、单任务 lease 的关闭/drain、
非方形 attention 在不同 FA 版本的语义、projection 的全局 offset、额外显存、
以及 microbenchmark 与视频端到端证据的边界。

## 验证方法与当前证据

数值和速度对照固定同一 GPU、权重、精度、输入和进程。基线执行从
`72dbb651d0` 提取的原 `encode_ids` 方法，绑定同一已加载模型；不是引用旧报告数字。
对比首次带 KV 的完整编码、已有 event 的短/长/Unicode 替换、恢复 N/A、保留 event 后字段，
以及完整 native conditioning 入口。缓存完全命中不能混入“改变事件后的增量编码”速度。

通过标准事先设置为：输出 finite；复用部分相对上次缓存逐位相等；整体和重新计算后缀
分别 `relative_l2 <= 0.01`；全局 cosine `>=0.9999`、最差行 cosine `>=0.999`；
`row_role` 逐位相等。另记录最大绝对差、平均绝对差、分叉边界误差。BF16 矩阵形状变化
可能改变舍入，不能将高 cosine 写成逐位相等；这些容差是组件数值门槛，不证明视频观感一致。
尤其不能让很长的相同前缀稀释后缀误差，后缀必须单独验收。

计时以 CUDA synchronize 围住 `perf_counter`，另记录 CUDA event elapsed。
本轮使用 TP1/BF16、`torch_sdpa` 的独立 Qwen 组件；已读取基线的
`PERFORMANCE_DEFAULTS.md`。其中 FA4、AdaLN、DiT CUDA Graph、window21/sink3、
RoPE KV 池与 VAE tiling 作用于视频管线，本轮不执行这些组件，也不改变其默认配置。
Qwen 的 FA 后端没有本轮测量证据，不能把 SDPA 结果直接推广到该后端。

Qwen 组件排除 processor/tokenizer；native 入口包含媒体摘要、processor/tokenizer、
H2D、Qwen、完整特征拼接和 D2H。两者都不包含网络、队列、Worker 广播、DiT 重建或视频输出。
预热后报告 N、median、P95，480/736 分开。记录 peak allocated/reserved；成对基线
测量时任务 KV 仍常驻，这个 peak 不能冒称旧服务的独占显存，另采无缓存基线。

复现工具：`benchmark/zing_h3_causal/bench_conditioning_prefix.py`，接受完整媒体和
prompt/update 的 JSON manifest、模型目录、原始 baseline wrapper 文件和输出目录。
结果保存 `metadata.json`、逐样本 `samples.json` / `native_samples.json`、`checks.json`、`RESULT.json`。

| 验证层 | 覆盖 | 实际状态 |
| --- | --- | --- |
| CPU 缓存行为 | token 替换/缩短、输出隔离、媒体摘要、预算、失败清空、sink offset | B300 受测版 111 passed、1 deselected；PR 整理未改数值算法 |
| CPU 生命周期/RPC | 单任务拒绝、tenant/handle、idempotent release、在途 drain、失联超时、Worker dispose、真实 ASGI/wire、总 deadline、慢发送释放 | PR 修复版本地 83 passed、1 deselected |
| B300 Qwen 480/736 | 同权重旧全量/新增量的数值与耗时 | B300 r5 完成，两档全部通过预设组件数值门槛 |
| H100/H200、Qwen FA 后端、非 32B projection 权重 | 相同数值/耗时矩阵 | 未测 |
| DiT、VAE Encoder、Decoder、TTFF/E2E/ICL、RGB | 三例真实 SP2 视频链路，统一性能基线 | 两档 72/72 请求完成，见末节；DiT FA4 与 Qwen SDPA 分别记录 |
| 人工画质、真实 SP 故障路径 | 事件效果与故障恢复 | 画质待用户检查；本轮成功请求不覆盖 SP 故障注入 |
| 跨实例视频等价性 | 原始 RGB 与重复样本对照 | 尚未建立：第 0 帧已存在跨组差异，见末节；不能仅据 Qwen 容差认定视频无回归 |

排除的唯一旧测试为 `test_qwen3vl_auxiliary_component_falls_back_from_global_backend`，
其 monkeypatch 指向当前主线已移除的 selector 接口，本次未改该测试及接口。

CPU 测试：111 passed、1 deselected，17.81 秒（`dlc1ra3eh72j9xxl`）。
该作业在测试通过后因重复创建已有素材目录退出；GPU 启动前另校验两组已准备素材 manifest 一致。
真实 backbone/config 边界回归覆盖原生 ArchConfig 缺少 HF 输出默认项的情况：缓存分支
显式传入 `output_attentions=False` / `output_hidden_states=False`，与旧 wrapper 一致。

PR 前独立 review 修复了多 RPC 超时累加和任务接口慢发送占槽；随后在 Windows、
torch 2.14.1+cpu 上运行 `test_prefix_task.py`、`test_conditioning_rpc_core.py`、
`test_conditioning_rpc_transport_policy.py`，83 passed、1 deselected（4.72 秒）。
唯一排除项 `test_configuration_requires_every_limit_and_no_artifact_secrets` 依赖 Linux
`os.O_NOFOLLOW`。本地隔离 harness 仅绕过 eager GPU pipeline 包初始化及全局 GPU conftest，
保留真实 Torch、ASGI、client/service/worker/presentation；不验证真实 SP collective 或 GPU。
独立 reviewer 另复查正常 POST/DELETE、慢发送、断连及 DELETE 超时后持续 drain。

PR #400 补齐 `test_prefix_task.py` 的 CPU inventory 登记，纳入完整
`conditioning-contracts` recipe。首轮 CI 各分组均在 inventory 阶段被该遗漏阻断，
不能将首轮失败视作业务测试结果。
三个 registered RPC 测试入口改为先通过统一 runner 初始化 CPU 环境，再显式调用
`pytest.main()` 并返回退出码；沿用检查器现有规则，无需跳过 `check-registered-tests`。
新增 9 个子进程用例覆盖三个入口的成功、断言失败、空选择及 `-f` 参数转发；
本地 9 passed，原有缓存/RPC 回归复跑 83 passed、1 deselected（同上 Windows 限制）。
修复后的 Linux CI `39893950052` 全部 13 项通过，包含 Realtime、4 个 DiT 分组、
conditioning、decoder、lint、inventory 和工具测试；工具测试合计 247 项通过。

本轮原始产物留在仓库外；源码以完整 base SHA 加逐文件 SHA256 冻结，并记录作业
ID、镜像 digest、输入素材身份。目标后端及画质的未验证范围分别以上表与真实链路结果为准。

## B300 组件实测结果（2026-10-23）

最终作业 `dlc1xdyolnnhjos4` Succeeded；基线和候选在同一进程、同一权重、同一张卡上运行。
泰国 B300 配额 `ml.gx9df.8.62xlarge`，驱动报告 NVIDIA L20D、CC 10.3、274113 MiB；
TP1/BF16/torch_sdpa，torch 2.13.0+cu130、CUDA 13.0、transformers 5.12.1。
镜像 digest 为 `sha256:4b9bca7dd61784cd84e43ac760db4c458c08a2d6bf22887e3740455fffe9e015`。
14 个权重分片均校验 SHA256。32B 使用第 50 层特征、Identity norm，无额外 projection 权重。

同一 coast 世界，6 张原尺寸 reference（1×1280×720、5×1024×854），source 首帧分别为
832×480 / 1280×736。不缩小 reference。每档纯 Qwen 5 次重复 × 5 种事件变更，共 25 对；
Native 入口 3 次重复 × 5 种变更，共 15 对。顺序固定全量后增量，单卡无并行样本；
结果是这组长视觉前缀、短事件后缀的热态收益，不代表任意世界/前缀长度的平均收益。

| 档位 | Qwen 全量 / 增量 median | Qwen 全量 / 增量 P95 | 中位加速 | Native 全量 / 增量 median | Native 全量 / 增量 P95 |
| --- | --- | --- | --- | --- | --- |
| 832x480 | 527.94 / 72.04 ms | 529.10 / 84.74 ms | 7.33× | 595.56 / 150.59 ms | 600.93 / 164.33 ms |
| 1280x736 | 576.20 / 75.59 ms | 578.44 / 88.49 ms | 7.62× | 641.54 / 157.49 ms | 654.48 / 173.19 ms |

以下误差统计合并 Qwen 与 Native 样本；验证在 CPU FP64 中计算，不计入编码耗时。
所有输出 finite、全部缓存前缀逐位相等、全部 row_role 逐位相等；冷缓存及强制媒体身份失效后
的全量输出均与旧方法逐位相等。事件变更后整体输出不是逐位一致，误差仍在事先制定的门槛内。

| 档位 | 最大整体 relative L2 | 最大后缀 relative L2 | 最低整体 cosine | 最低行 cosine | 最大绝对差 / 平均绝对差 |
| --- | --- | --- | --- | --- | --- |
| 832x480 | 0.0370% | 0.8870% | 0.9999999316 | 0.9999021671 | 4.0000 / 0.00014054 |
| 1280x736 | 0.0358% | 0.8547% | 0.9999999358 | 0.9999159955 | 2.0000 / 0.00013015 |

| 档位 | 复用 / 重算 tokens | 无缓存全量 peak allocated | 增量 peak allocated / reserved | 增量相对无缓存 resident 增量 |
| --- | --- | --- | --- | --- |
| 832x480 | 6714–6714 / 14–47 | 49.798 GiB | 50.248 / 52.592 GiB | 1.356 GiB |
| 1280x736 | 7244–7244 / 14–47 | 49.934 GiB | 50.411 / 53.289 GiB | 1.460 GiB |

resident 差值包含缓存和测量时存活输出，不能视为纯 KV 精确大小。成对全量的 peak 已包含
候选缓存常驻，因此单独保留无缓存基线；allocated / reserved 都不等于 nvidia-smi 整卡用量。
首轮冷启动耗时未纳入热态统计，不承诺首次构建更快。

早期 r4 报告发现 FP32 对约 3500 万元素求整体 cosine 得到大于 1 的无效值；
其数值验收结论作废。改为 FP64 并增加 cosine 上界检查后重新完成 r5 全量测试，
没有放宽 relative L2/cosine 门槛。r4 和 r5 原始数据都保留，最终结论只取 r5。

源代码：base `72dbb651d0e1f81d5b2d95f925554ce70d4f0018` 加 r5 `SOURCE.json` 的逐文件哈希，
r5 产品代码与 111 项 CPU 测试受测版本相同；r5 仅修正 benchmark 的统计精度和验收上界。
PR 整理随后新增总 deadline/任务响应生命周期修复和导入格式整理，未改变 Qwen 数值算法；
新增协议改动当时以 83 项本地 CPU 回归验证；之后当前 PR 提交的真实链路测试见下一节。
原始产物：`oss://leap-worldmodel-thailand-single-az/world-model/sglang/evals/conditioning-prefix-b300-20261022-r5/`，
包含源码 manifest、输入哈希、配置、CPU 日志、每对耗时/误差、无缓存显存探针及最终结果。
原始数据不随产品代码提交；测试脚本和本设计保留复现入口及验收口径。

结构自审追踪了 Session → rank 0 Worker → RPC/租约 → 编码线程 → Qwen → 完整特征返回；
保留一份最新状态避免跨任务调度。上述组件测试的 CPU 覆盖协议和生命周期，GPU 覆盖
真实 Native 编码入口，其本身不包含 HTTP + Worker + DiT。目标部署若切换 Qwen 到 FA、
增加 projection 或改变输入分布，仍需补相应组件/视频验收；后续真实链路证据单独列在下节，
不将组件倍速或数值容差当成视频画质结论。

## 三例真实视频链路验证（2026-10-23）

受测提交 `569baf7d24fa799dcfd5f7b4bdede7b77dd219c9`，在同一现有 conditioning
服务和真实 HTTP/WebSocket → SP2 Worker → DiT → Lynn 输出链路中，仅切换
客户端 `prefix_reuse=false/true`。从 clean 15-event 用例中选择黄雨衣、彩色玻璃、
青蓝发光树，分别使用 5/6/6 张原参考图；将既有 `events: N/A` 替换为目标事件，
没有更换参考图。每个请求都是新任务，先冷编码，再在该任务内复用，完整结果 CPU
缓存预算设为 1 byte，防止跨请求完整结果命中掩盖测量。

平台分配 4 张 B300，实际使用 3 张（DiT SP2/Ulysses2 + 独立 Qwen TP1），第 4 张闲置。
两组使用同一镜像、checkpoint、seed 42、360 帧、24 FPS；source 使用 bilinear
生成 832×480 / 1280×736 两档输入，原参考图不缩小。遵循统一性能配置的
window21/sink3、block_relative、gap12、stable decode、FA4、RoPE 融合/grouped/chunk
reuse、stable shapes、compiled AdaLN、metadata reuse、CUDA Graph/SP Graph；
`HISTORY_ROPE_PLAN=0`。480 专用 fixed FA4/current RoPE-KV 只在 480 开启。
保留原用例前 3 个 block 四步、之后两步，以及显式 `source_work_quantum=1`。
source progressive encoder 实际选择 SP2 tile parallel。旧共享模型 profile 是
window6，本实验只在独立模型 overlay 改为 window21，并以真实 API `on_init` 验证；
没有修改共享模型或其他作业。

每档每组正式轮每例预热 1 次、测量 3 次，另外独立运行预热 1 次、测量 1 次的诊断轮。
每档按全量后复用的固定顺序运行，未随机交错；这是小样本描述性结果。
正式轮使用客户端实际到达时间，不保存特征、不增加 CUDA 同步；诊断轮记录阶段
host/CUDA stream spans 并保存实际 Qwen 输出，在 CPU 用 FP64 比较。
下面的完整切换耗时指**发送 prompt 到完整收到首批归属于新 event 的 RGB**，
不等于肉眼语义生效。正式 P95 仅是 3 个样本插值，不代表生产尾延迟。

| 档位 | 用例 | 切换中位数：全量 → 复用 | 降幅 | Qwen CUDA：全量 → 复用（诊断 n=1） | 后缀 relative L2 |
| --- | --- | --- | --- | --- | --- |
| 480 | 黄雨衣 | 2327.10 → 2075.62 ms | 10.81% | 411.99 → 114.03 ms | 0.83304% |
| 480 | 彩色玻璃 | 2656.75 → 2277.57 ms | 14.27% | 507.43 → 129.83 ms | 0.87487% |
| 480 | 青蓝发光树 | 2628.44 → 2298.13 ms | 12.57% | 507.15 → 133.85 ms | 0.88928% |
| 736 | 黄雨衣 | 3708.66 → 3405.85 ms | 8.17% | 459.05 → 121.08 ms | 0.87174% |
| 736 | 彩色玻璃 | 4043.38 → 3669.18 ms | 9.25% | 554.95 → 138.41 ms | 0.88062% |
| 736 | 青蓝发光树 | 4018.91 → 3641.28 ms | 9.40% | 556.10 → 141.57 ms | 0.85294% |

736 彩色玻璃复用组三次分别为 4045.44、3669.18、3631.36 ms，其中一次与全量
基本持平；全部保留，不能把中位收益解释成每一次请求都得到相同改善。
持平样本仍命中 6976 行前缀、重算 129 行，条件更新为 571.14 ms（对应全量
1017.48 ms）；正式轮没有足够细的 spans 定位这一次净收益被抵消的具体环节。

两档 12 组初始/更新特征对照均通过门槛，6 组初始输出逐位一致；更新后整体
relative L2 最大 0.06910%、后缀最大 0.88928%，最小行 cosine
0.99913479，整体/后缀均通过既定门槛，完整更新输出并非逐位相同。缓存前缀与本任务
**上一次输出**逐位一致；这不等于其与新 prompt 全量重算的前缀逐位相同，两类比较
分别保留。480 真实复用 5284/6446/6464 行，736 复用 5814/6976/6994 行，
均重算 129/129/142 行；不能直接沿用此前
重算 14–47 行的组件 benchmark 倍速。

两档全部 36 个正式测量请求都是收到 128 帧后发送，首批新 event 从第 136 帧开始，两组一致。
条件更新的正式中位数分别为 721.74→449.98、914.45→527.75、886.66→567.44 ms；
此口径含 RPC、Worker 检查与 pending 更新。诊断中的黄雨衣 DiT KV 重建仍约
708→703 ms，本改动不省去它。完整诊断时间轴拆为 17 段，嵌套 spans 不相加；
诊断 RPC 包含特征落盘等待，不能用来替代正式切换耗时，Qwen/Native 计时在落盘前结束。

代价是首次建立缓存：480 三例的正式 TTFF 中位数增加约 95–111 ms，
切换前 Qwen resident allocated 增加约 1.06–1.30 GiB；736 TTFF 中位数增加约
34–54 ms、resident 增加约 1.17–1.40 GiB。频繁换图场景没有测量加速收益；
当前会完整重编码，不能推广本表结论。DiT/VAE/Lynn 共进程显存按共享范围报告。
Graph replay、RoPE 等路径保留逐请求计数；FA4/编译/metadata 等配置的实际解析值
已保存，没有把开关为 1 当成逐 kernel 验收。
每例只预热一次，480 正式第 1 次仍有一次 init Graph capture、后两次为 replay，
两组一致；TTFF/E2E/P95 包含这部分启动成本，没有事后剔除较慢样本，
也不宣称所有 Graph 路径已经完全热态。
DiT/VAE/Lynn 共享进程的两档最大 peak allocated 为 141.789 / 203.927 GiB，
peak reserved 为 153.750 / 215.496 GiB；包括加载/预热的整卡采样峰值分别约
163.201 / 224.467 GiB。逐 rank、逐 case、独立 Qwen 与各阶段原始计时见完整报告。

**跨实例视觉基线尚未对齐，不能宣称整链路等价。** 所选两组原始 RGB 从第 0 帧
就有差异；在请求前已收到的 0–127 帧，六个用例/档位的 MAE 为 3.3080–4.7502
（0–255 通道值）。全量组第 1/2、2/3 次与复用组第 2/3 次的重复输出全部逐位相同，
因此组内重复本身不能解释跨组差异。480 黄雨衣的全量正式/全量诊断对照在请求前
MAE 为 6.2083，复用正式/复用诊断为 5.6023；这些对照同时改变服务实例和观测方式，
没有定位根因，也不能归因于其中某一个因素。初始 Qwen 特征的逐位相等是在诊断轮
取得的局部证据，不能代替正式视频的等价性证明。

这不改变已测得的延迟和 Qwen 特征误差，但限制视频差异的归因。上线前应进一步对齐
跨实例/同实例基线，必要时沿 Worker 收到的 context、VAE moments、DiT latent 到 RGB
定位第一处差异，并完成人工质量检查。本次不把视频可解码或 Qwen 容差通过写成画质通过。

原始产物目录为
`oss://leap-worldmodel-thailand-single-az/world-model/sglang/evals/sglang400-real-switch-20261023-r4/`，
GPU 作业 `dlcf6c7bfwwv5zpd` Succeeded（UTC 04:04:56–05:39:00，72/72 请求），
输入源码、素材及冻结脚本哈希保存在 `LOCK.json`。`review.tar.gz` 包含 `report.md`、
`results.json`、原始日志、冻结 harness、CPU 分析回执及 12 段对照视频。
视频取每例正式第 1 次（未按效果挑选），全部通过 360 帧、24 FPS、尺寸与完整解码检查，
但人工画质待用户检查，不据数值容差宣称通过。对照页按 24 FPS 回放，不是实时等待录屏。
受测实现固定在上述 SHA；随后更新本设计和部署说明不改变受测代码。

## Reviewer 阅读路线

1. `stage.py::conditioner` → `worker.py::RemoteConditioner`：真实会话和 rank 0 所有权。
2. `client.py` / `wire.py` → `service.py::_encode`：可选 handle、旧路径兼容、singleflight 隔离。
3. `prefix_task.py::SinglePrefixTask`：一个活跃任务，关闭标记先于 drain，旧 handle 无法释放新任务。
4. `native.py::__call__` → `encoder.py::encode_condition`：完整媒体身份、异常失效与 CUDA drain。
5. `qwen_prefix.py` / `minimax_h3_qwen3vl.py::encode_ids`：LCP、完整位置、视觉边界、KV crop、输出与 sink。
6. CPU `test_prefix_task.py` / `test_qwen_prefix.py`、registered `test_conditioning_rpc_core.py`，
   最后看 GPU benchmark 的逐样本后缀误差和计时边界。


## 2026-11-05：同步 main 冲突处理

基线更新至 main `a36defd471`，保留其 NCCL 结果传输、后台预取、输入预算校验与 CPU 测试清单。同步请求的任务句柄可随 HTTP/NCCL 请求传入，注册、过期恢复与编码共享超时预算；清理先处理预取，再释放任务。握手同时核对 profile、传输协议及前缀能力。

后台预取仍走 main 的无任务句柄路径，优先使用预取结果；该组合尚未接入前缀复用，不能外推既有同步 HTTP 加速数据。历史 B300 数据对应 `569baf7d24`，合并后 GPU 数值、NCCL 传输及性能未复测，跨实例视频等价性未建立的限制继续保留。

本轮本地验证：任务生命周期与 HTTP/NCCL 请求控制回归 9 项通过；相关文件 pre-commit 和 diff 检查通过。扩展 NCCL 控制测试 42 项通过、2 项真实 HTTP 流时限测试失败（尚未证明原因），另 9 项未执行；Windows 不支持预取测试的 ZeroMQ IPC，标准完整 CPU harness 也受 Unix resource 模块缺失限制。完整 Linux 回归以本次合并提交的 CI 为准，不能引用旧提交 CI 代替。


## 2026-11-05：后台预取接入任务前缀缓存

此次接入替代上节“后台预取尚未接入”的状态。开启现有 remote conditioning prefetch 和 `prefix_reuse` 后，初始同步编码、后台预取、切换边界未命中后的同步编码共用当前会话的任务句柄。关闭 `prefix_reuse` 或服务不宣告能力时，仍执行全量编码，不新增使用参数。

任务句柄及注册/过期恢复放在 `WorkerConditioningState.encode_remote`，沿用原会话所有权；`ConditioningPrefetch` 只接受一个编码回调，不持有第二份任务，不向后台搬移 SP collective 或有效 prompt 提交。`source_session._start_conditioning_prefetch` 在启动线程前绑定该回调。预取结果继续按完整输入 digest 判断命中，换图触发服务端已有媒体身份失效；被后续 prompt 覆盖的投机 KV 只影响可复用长度，不直接成为有效会话输出。

预取锁串行化后台编码和边界回退；任务锁覆盖注册、编码、过期恢复与释放，防止清理时登记出的句柄遗漏。锁等待、注册、编码和一次过期恢复共用 deadline。dispose 首先禁止新请求、停止预取，再在有界等待后释放任务；无法完成 drain 时报告失败并保留句柄供再次清理/服务端 idle 回收，不假装已释放。

NCCL 仅把收到 HTTP 410、尚未收到任何结果 metadata 的明确拒绝视为可恢复任务过期；其余不明确或中途传输错误继续标记通道不可用，禁止在原通信器上重试。HTTP 任务恢复沿用原语义。

验证覆盖：HTTP/NCCL 参数化真实 IPC 预取→边界命中、与初始编码共享句柄、任务过期重注册、换媒体仍携带同一任务、dispose 释放；生产 source session 回调接线；并发注册与清理、锁等待预算；NCCL 410 的控制流及不明确错误不可重试。CPU 证据不证明 CUDA 传输或真实 Qwen 数值/加速收益。以下补充当前提交的上海 GPU 证据；本次未部署。

### 当前提交的宽矩阵与真实输入探针

受测源码固定 `530cf2400c6e7a604645525de914aee889e71fbe`，优化基线来自
main `a36defd471881b2c6720c833bb3d129121d3fc32`。资源清单称上海 H100，
实际设备报告 NVIDIA L20Z、SM90、81559 MiB，报告以实际身份为准。
本节与上面的历史 B300/同步路径数据分开，不混算绝对耗时。

Qwen-only 矩阵共 79 格：参考图 0/1/4/8/16/32/64 张，文本
512/2K/8K/16K/32K/64K token，并覆盖修改位置、替换比例、插入/删除、
换图/换序/增删图及完整重复。参考图使用真实素材和生产预处理，额外 resize=None；
32/64 图循环使用 20 张真实素材，不能代表同等数量的独立世界资产。
组件缓存预算 64 GiB、max_rows=None，用于探测边界；不代表服务的 4 GiB、
16384 行准入已扩大。

结果为 68 格逐位一致、10 格数值门槛失败、1 格 OOM；可运行的 78 格均保留
5 对交替 full/prefix 计时，共 780 个样本。64 图加 64K 文本在激活分配时 OOM，
没有截断输入或填入虚构时间。前缀与旧输出全部逐位一致，失败集中在 8/26 行短后缀；
最大后缀 relative L2 为 8.22042%，不因整体误差被长前缀稀释而判为通过。
默认 FP8 路径保留，诊断用线性层补行没有作为生产修复。

**短后缀误差原因：同一个 FP8 GEMM 因 M 不同分派了不同的 CUDA 内核。**
这里 M 是 token 行数，K 是输入通道数；全量与后缀计算的每行 K 都是 5120。
线性投影仍逐行独立，不能用“行数减少使每行归约长度改变”解释误差。
在已检查的 26 行后缀案例中，原始输入、FP8 量化字节和 scale 逐位相同，
全量使用 tile=128×128×128、cluster=2×1×1 的
`KernelTmaWarpSpecializedPingpongFP8FastAccum`，短后缀使用
tile=64×16×256、cluster=1×4×1 的 `KernelTmaWarpSpecializedFP8FastAccum`。
首层最早观察到差异的位置为 K 投影输出，max-abs=6.103515625e-5；
两个 512 文本 token 案例的最终后缀 relative L2 为 6.68297% / 8.22042%。

控制实验中，投影补到 32/64 行仍不同，补到 65/128/256/512 行后该投影差异消失；
仅将 FP8 线性层补到 128 行时，两个失败案例及一个 2048 文本 token 控制案例的
最终输出均逐位一致。补行未向 attention 添加有效 token。这些证据定位到短行内核
路径，但尚未逐条确认内核内部浮点累加或舍入指令，也没有对 79 格逐格执行该消融。
原始 kernel 名称、量化值/scale 比较及补行结果保存在
`sglang400-prefetch-matrix-20261105-r4/r4-evidence/diagnostic/results.json`。
正式矩阵与完整视频仍使用默认 FP8 路径，不包含补行改动；缓存前缀逐位一致，
不应将重算后缀的数值差异解释为前缀看到了未来 token。

例如 4 图、8K 文本、95% 处修改的 Qwen wall 中位数为
897.03→62.97 ms（14.25 倍、逐位一致）；16 图、32K 为
6187.06→355.67 ms（17.40 倍、逐位一致）。换媒体的用例复用行数为 0，
输出逐位一致，但有约 11–19 ms 的额外开销。计时包含 encode_ids 内的输入搬运、
MRoPE 和输出组装，结束时同步 GPU；CPU 图像处理及 FP64 输出对照单列。
每次 prefix 测量前清空缓存、编码旧输入，然后只计新输入，完整结果缓存不参与。

从实际 prompt 切换素材筛查 24 格后，冻结以下三类输入，各补 30 对 Qwen 测量：

| 场景 | 480 Qwen wall 中位数，ms | 736 Qwen wall 中位数，ms | 更新特征结果 |
| --- | --- | --- | --- |
| garden 原始 event，6 图 | 528.57→42.32 | 572.33→42.57 | 逐位一致，重算 129 行 |
| forest 短 event，6 图 | 527.10→42.18 | 574.74→42.28 | 后缀 L2 2.98% / 3.06%，重算 8 行，失败 |
| hiker 参考图换序，5 图 | 440.58→449.59 | 482.91→497.83 | 0 行复用，全量回退，逐位一致 |

这些是独立 Qwen 探针，不能当作用户可见切换耗时。完整推理采用当前
1027 step10000 模型、W6/S3、block_relative/gap12，适用优化开启，
按用户要求关闭 DiT CUDA Graph 与 SP Graph，两侧均开启后台预取。
480 采用 DiT SP2/四卡整体布局；用户随后取消 736 完整推理，
736 仅保留上表已完成的 Qwen 组件测量。正式计时每臂三次，细阶段诊断另做一次并保存全部 360 帧 RGB；
诊断与正式样本分开，嵌套 Host spans 与 CUDA stream elapsed 不直接相加。
480 完整推理已完成 18 个正式请求和 6 个诊断请求，均输出全部 360 帧、实际采用
一次新 event；另有 24 个预热请求，不混入测量。W6/S3 与 Graph 对象关闭已从
24 个请求的 48 份 DiT rank 运行时状态核验，未以环境变量代替生效证据。

| 案例 | 正式切换中位数，full→prefix ms | 节省 ms | 降幅 | 实际首个新 event 帧 |
| --- | --- | --- | --- | --- |
| garden 原始 event | 1821.29→1367.00 | 454.29 | 24.94% | 两侧均 68 |
| forest 短 event | 1820.93→1362.71 | 458.22 | 25.16% | 两侧均 68 |
| hiker 参考图换序 | 1960.43→1988.50 | -28.06 | -1.43% | 两侧均 85 |

切换从客户端发送 prompt 前计到首批新 event 完整 RGB 到达，包含序列化、
writer 等待、上传及接收；不是屏幕渲染或语义改变的人工判定。每臂 n=3，
所有原始值和 P95 保留，三点 P95 仅为描述统计。发送时 source_frame_start=120，
不代表第 120 个输出帧才开始使用新条件，流内仍有未处理输入。

诊断中 garden 的 Qwen CUDA 为 529.19→42.32 ms，Native Host 为
563.04→80.72 ms；激活开始到首个新条件 latent 提交仍为约 985.13→985.70 ms，
说明此项缓存没有省掉 DiT 的联合前缀重建/预测。后者是相邻 Host 时间点区间，
不是独占 GPU 内核耗时。已保留 API、预取准备/任务 RPC、服务队列、Native/Qwen、
NCCL、边界等待、候选条件、激活、联合预测、RGB 与客户端到达时间轴。
诊断 encode_ids 的 Host 调用可异步返回，不能把该 Host 数字当成 GPU 完成时间；
独立探针 wall 则在 GPU 同步后结束。远端解码调用的 Host 记录不证明独占 CUDA 时间。

**本轮视频基线已对齐，但短后缀仍有差异，人工画质尚未验收。** 两侧各自的正式
三次与对应诊断的完整 RGB 哈希全部一致。garden、hiker 的 full/prefix 全部
360 帧逐位相同；forest 的切换前 68 帧逐位相同，切换后 292 帧均有差异，
平均每帧 MAE 为 11.258589（0–255 通道值）。这与上面的历史 B300 跨实例
第 0 帧差异是不同冻结配置的证据，不能混为一谈。forest 的数值失败保持原标签，
不因完成推理就判定等价。六段原尺寸全帧视频经 CPU 编码和完整解码检查，
生成三对并排视频，留待人工检查。

正式请求期间约 200 ms 驱动采样的最大显存，按四张逻辑卡分别为
77.126 / 76.042 / 35.569 / 39.792 GiB（full），
77.126 / 76.042 / 35.569 / 40.505 GiB（prefix）。最后一卡共享 Qwen 和
Source VAE，不称为 Qwen 独占显存；各卡峰值时刻不同，不相加。
实际观察到 grouped history RoPE 和 fused QK norm，current RoPE 形状回退保留；
Block0 虽开启但未请求 cache_write，不宣称 warm cache 命中。

原始完整推理来源为 `sglang400-full-switch-20261106-r11` 的五个成功正式 cell
和 `sglang400-full-switch-20261106-r14-resume480` 的一个正式、两个诊断 cell。
r11 在最后正式 cell 的启动探测中失败，尚未发请求；失败单独保留，未覆写为成功。
补跑保持同一受测 SHA、模型、输入和四张物理卡，仅修复启动健康探测的临时 HTTP
异常处理，没有推理请求重试。合并回执 `480_DONE.json` 包含 24 个请求哈希；
视频与日志归档 `full-evidence-media.tar.gz` 的 SHA256 为
`5106b6c895eeea386f7a34017ba90fa1ba7256b9e38220d516b9a0e18ff249fa`。
736 完整推理按用户要求取消，未提交 GPU 作业；本轮 GPU 已释放。

原始证据位于上海评测存储的 `world-model/sglang/evals/` 下：
`sglang400-prefetch-matrix-20261105-r5`、`sglang400-real-probe-20261105-r2`
和 `sglang400-real-probe-20261105-r3`。门槛沿用 relative L2≤1%、
cosine≥0.9999、min-row cosine≥0.999；失败标签、原始耗时和 OOM 均保留。

### 2026-11-06：同步 main 与数值差异补测的版本边界

本次同步 main `b9e4bc95fc5d0869829749a27b81c23a9c942dc9`，解决两处冲突：
`source_session.py` 保留通过当前 `WorkerConditioningState.encode_remote`
提交后台预取的回调，同步请求与预取继续共享同一个任务前缀句柄；同时保留 main
的 Block0 缓存命中判断、NCCL device 选择及资源释放路径。
CPU 测试清单同时保留任务前缀测试和 main 新增的 VAE 融合测试，未覆盖任一组条目。

六例数值差异补测使用独立冻结的 `530cf2400c` 源码包，
W6S3、832×480、DiT Graph 关闭；该作业未热更新为此次合并代码，结果见下一节。
main 的共享 KV 预算、准入、VAE 融合等变更已保留在分支中，但旧冻结作业的
性能与视频结果不能作为合并后版本的 GPU 验证。736 完整推理仍按用户要求不跑。

合并验证：CPU inventory 完整性及冲突文件 Ruff 检查通过；Linux CPU 回归首轮
2054 项通过、9 项因本机平台探测/Windows worktree 路径失败。显式选择 CPU
平台并设置 WSL Git 路径后，复跑这 9 项及 CLI、Block0 切换、prompt 准入相关测试，
56 项全部通过。未把 CPU 回归作为 NCCL/CUDA 正确性或合并版本性能结论。
全仓 `check_registered_tests.py` 仍因 main 已有的
`test/registered/kernels/ops/test_fp8_scaled_mm_routing.py` 缺少注册入口失败；
该文件与上述 main SHA 完全一致，本次未修改。

### 2026-11-06：六个数值失败事件的完整视频补测

`sglang400-numeric-video-20261106-r16` 已完成：3 个真实世界、6 种事件，
不是 6 个独立世界。产品源码仍为 `530cf2400c`，沿用上文真实参考图、模型、
832×480、W6S3 和四卡布局，关闭 DiT/SP Graph。每例每臂各 1 次正式、1 次诊断，
共 24 个测量请求，另 24 次预热；全部交付 360 帧且更新实际生效。
48 份逐 rank 最终状态确认 W6S3、block_relative/gap12 及 Graph 对象未创建。
正式 n=1 只报告单次配对收益，不据此宣称稳定 P95。

|事件|Qwen 后缀 L2|独立 Qwen wall，全量→复用 ms（n=3 中位数）|正式完整切换，全量→复用 ms（各 n=1）|切换后 RGB MAE（0–255）|
|---|---:|---:|---:|---:|
|黄色雨衣|2.791%|419.72→41.48|1876.48→1517.15|3.906|
|彩色玻璃|2.923%|536.29→41.67|1818.41→1367.09|13.582|
|降雪|2.756%|418.34→41.76|1895.53→1548.59|7.973|
|夜晚极光|2.893%|418.23→41.88|1890.91→1503.51|5.903|
|棋盘格地面|3.050%|535.75→42.49|1832.77→1340.40|18.480|
|森林浓雾|5.635%|532.27→42.30|1824.73→1366.85|17.307|

正式六例切换前 RGB 全部逐位一致，切换后均有差异。两臂实际采用帧相同：
garden/forest 为第 68 帧，hiker 为第 85 帧。夜晚极光的两臂诊断运行都在第 68 帧
切换，与正式运行第 85 帧不同，因此不能拿其正式/诊断哈希不同作为数值不稳定证据；
其他五例的同臂正式/诊断全部 360 帧一致。MAE 仅描述像素差异，**人工质量仍待检查**。
本轮没有逐例重新采集 kernel profiling，数值归因不超出上文已有 FP8 探针证据。

保留 12 条诊断时间线、108 个相邻墙钟区间、Qwen CUDA 分项、Encoder/DiT/Decoder
Host/CUDA 及逐卡显存记录；嵌套区间不累加为独占组件耗时。
24 段原尺寸视频及六对并排视频均完成全帧解码校验。
归档位于 `oss://leap-world-shanghai/world-model/sglang/evals/sglang400-numeric-video-20261106-r16/`，
`evidence-media.tar.gz` SHA256：
`6e9b7021d7610a4aaffd9a231cb9671b194c386547866bf24e9ff6b632e8a163`。
GPU 与 CPU 后处理作业已完成并清理，原始 RGB 保留。

冲突修复提交 `05f1b32f0a` 的
[Zing CPU CI](https://github.com/seedleap/sglang/actions/runs/42184171840) 和
[Block0 cache contracts](https://github.com/seedleap/sglang/actions/runs/42184171904) 均通过；
本节只是补记冻结版本结果，不替代合并版本 GPU 验收。

### 2026-11-10：单卡 Qwen 关闭 FP8 的数值对照

**关闭 FP8 明显减小了这批案例的复用误差，但没有恢复逐位一致。**
本轮按用户要求只测 Qwen，不跑完整推理。冻结产品 `cd7299eb9841613f37b5b7e570fc5175c7776415`，
在上海 H200 资源池的同一张独占 GPU（实际 NVIDIA L20X / SM90）上，
串行运行 FP8 OFF、ON 两个独立进程；每个进程内部比较同精度的 full/prefix，
不把 BF16 与 FP8 输出之间的差异算成缓存误差。

输入为 r16 的 3 个真实世界、6 个事件，各 short/medium，共 12 份原始张量。
每例 5 或 6 张参考图，总输入 5292–6494 行、重算后缀 8–34 行。
参考图原始尺寸为 1280×720 / 1024×854，生产预处理记录为
1280×704 / 1024×832，`reference_resize_override=None`；本轮直接复用
已冻结的 Qwen 张量并核对 SHA，没有额外缩图。案例名称中的 480 是来源标签，
不代表本轮生成了视频。

|精度|后缀 relative L2|原数值门槛通过|全量 wall 中位数范围 ms|复用 wall 中位数范围 ms|同精度复用加速|编码样本最大 allocated GiB|
|---|---:|---:|---:|---:|---:|---:|
|FP8 ON|2.755549%–5.634973%|0/12|390.43–498.68|49.30–51.62|7.57–10.04×|27.11|
|FP8 OFF / BF16|0.728334%–0.943696%|12/12|530.75–663.78|29.29–30.68|17.30–22.49×|49.80|

两组整段/后缀逐位一致均为 0/12；关闭 FP8 后，同例后缀 L2 下降
69.93%–83.62%，最大绝对差仍为 1–3。门槛仍为 L2≤1%、cosine≥0.9999、
min-row cosine≥0.999，没有放宽。两种精度的冷缓存与全量、完全相同输入的
缓存命中、未变化前缀与旧结果、未变化前缀与新输入全量结果，均为 12/12 逐位一致；
全量重复测量也全部与各自基准逐位一致。观测到的复用误差集中在重算后缀。

这修正了“误差全部来自 FP8”的过强表述：已有 FP8 探针证明 M 行数变化触发不同
GEMM tile，本轮消融证明关闭 FP8 后仍有残余差异。BF16 Linear/FA4 在全量与
短后缀形状下可能采用不同分块或累加顺序，但本轮没有重新逐层 profiling，
**r20 当时尚未把残余差异定位到具体 BF16 算子**；随后 r22 的逐层与干预结果见下一节。
因果性是数学依赖关系，不自动保证不同形状的有限精度实现逐位相同；
本轮未变化前缀实际保持逐位一致。

同卡逐例比较，关闭 FP8 后全量编码慢 32.21%–37.05%，短后缀复用却快
38.85%–42.11%；BF16 内部 full→prefix 节省 500.07–634.14 ms。
这些是 Qwen-only 时间，不能当作完整 prompt 切换时间。每例每精度先预热，
再交替顺序测 3 对 full/prefix，共 144 个正式计时样本；n=3 的 P95 仅为样本分位数。
计时同步 GPU，包含 `encode_ids` 内部位置准备/H2D，不含预处理、D2H 数值回读、
CPU FP64 比较及种子预填充；seed 时间/显存另存。表中显存是正式编码样本峰值，
不是模型加载峰值，关闭 FP8 增加约 22.69 GiB。

两组均为 TP1、`output_hidden_layer=50`、缓存 4 GiB，torch 2.13.0+cu130 /
transformers 5.12.1；HF RMSNorm、fused MRoPE、fused gate/up、text/vision FA4
由 native runtime receipt 核验开启。OFF 实际零 float8 参数/FP8 量化方法，
ON 实际有 300 个 float8 参数及 300 个 `Fp8LinearMethod`。模型目录索引派生为
只声明 text_encoder/tokenizer/processor，原 Qwen 权重/配置/输入 SHA 均保持一致。
产品实现及生产默认值未修改；完整推理、人工画质、79 格宽矩阵及任意形状仍未由本轮验收。

CPU `dlc9tfocr9lms61s`、GPU `dlc12o9g5n0o4ieh` 均成功，24 份数值结果、
144 个计时样本及两臂 DONE 均已回读，单卡资源已释放。执行 harness 为
`ea1bb9cbcbd18b1d40df8f17795a2123d2f4b07a`；分析脚本为
`e4907938b52b4a143a7f2657778016da7de9a0e9`，备份分支
`codex/conditioning-prefix-fp8-off-h200-r20`。
此前 r18 CPU 打包失败、r19 模型目录完整性失败均未进入编码，保留失败回执，
不计入本轮样本。逐例误差、P50/P95、CUDA 时间、显存和原始日志归档于
`oss://leap-world-shanghai/world-model/sglang/evals/sglang400-qwen-bf16-h200-20261110-r20/`，
`qwen-fp8-off-evidence.tar.gz` SHA256 为
`d7691002beff6b692db7ebf1eacef73b2b7cbbc3d389787f64087beefd8a11c1`，上传后读回校验通过。

### 2026-11-10：BF16 残余误差定位到 Linear 形状相关计算路径

同一冻结产品 `cd7299eb9841613f37b5b7e570fc5175c7776415` 在上海单卡 H200 资源池
（实际 L20X / SM90）完成 r22 三例诊断。FP8 实际关闭，其余五项 Qwen 优化保持开启，
仍为 TP1、输出第 50 层。沿用 r20 的原始权重和输入张量，5/6 张真实参考图、生产尺寸，
没有额外缩图。测试前重新核对 main `36fb7bf21c264e157a401531b55d1e09cf5ceec6`
的优化基线；未把 main 更新混入冻结产品。本轮只测 Qwen，没有完整推理。

**三例第一处分歧均出现在第 0 层 K 投影输出；只固定全部文本 Linear 形状，最终
整段特征均恢复逐位一致。只固定 attention 形状，误差不变。** 这把本轮残余差异定位到
Linear 随 token 行数变化而选择不同计算路径，不再仅是对 FP8 或 FA4 的推测。

|案例|全量/前缀/后缀行|第一层 K 不等元素|正常后缀 relative L2|正常全量→复用 wall ms|加速|
|---|---:|---:|---:|---:|---:|
|黄色雨衣 short|5295/5284/11|20/11264|0.728334%|542.07→28.84|18.80×|
|黄色雨衣 medium|5314/5284/30|46/30720|0.819371%|544.78→31.68|17.20×|
|棋盘格地面 short|6456/6446/10|22/10240|0.943696%|675.95→32.23|20.97×|

正常计时先预热再交替次序测 3 对/例，共 18 个样本；表中各臂中位数，没有 hook 或 padding。
同步墙钟包含 `encode_ids` 内部位置准备/H2D，不含预处理、种子编码、D2H 数值校验。
三例分别节省 513.23/513.10/643.72 ms，正式编码样本峰值 allocated 最大 49.80 GiB；
均为 Qwen-only 口径，不代表完整切换时间。r20 与 r22 为不同运行，不能混合成同一组采样。

#### 逐层与孤立算子证据

每例 50 层的 `position_ids`、`cache_position`、`attention_mask` 均一致。
50 层旧种子前缀 K/V 与新输入全量前缀逐位一致，另 50 层增量组装后的前缀 K/V
也与全量一致。第 0 层输入、input norm、Q 投影以及 K 投影输入均逐位一致；
K 投影输出第一次出现差异，三例最大绝对差均为 0.00048828125。
插桩前后结果一致，撤销诊断 wrapper 后恢复原始结果，排除 hook 改变输出。

固定同一份 BF16 输入与权重，单独重放第一层 Linear 即可复现差异。
K 投影可写为 `Y[M,1024] = X[M,5120] × Wᵀ[5120,1024]`；变的是 M（token 行数），
**输入通道/归约长度 K=5120 没有变化**。split-K 的 K 是矩阵乘法归约维度，
不是 attention 的 Key。实际 profiler 记录：

|案例|全量 K GEMM|短后缀 K GEMM|
|---|---|---|
|雨衣 short/medium|`nvjet_sm90_tst_128x168_64x5_2x1_v_bz_TNT`|short：`nvjet_sm90_tst_64x8_64x16_4x2_h_bz_splitK_TNT`；medium：`nvjet_sm90_tst_64x16_64x16_4x2_h_bz_splitK_TNT`|
|棋盘格 short|`nvjet_sm90_tst_128x200_64x5_2x1_v_bz_coopA_TNT`|`nvjet_sm90_tst_64x8_64x16_4x2_h_bz_splitK_TNT`|

三例 short 还实际执行 `cublasLt::splitKreduce_kernel<32,16,...>`，full 未出现该独立归约。
因此，“行数减少但每行点积长度不变”仍可能触发不同的分块和部分和归约路径。
[cuBLAS 数值精度说明](https://docs.nvidia.com/cuda/cublas/index.html#gemm-algorithms-numerical-behavior)
解释 split-K 按归约维度拆分后再求和的机制；
[PyTorch 数值精度说明](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)
说明不同形状/计算路径不保证逐位相同。这里的具体归因以 profiler 与干预实测为依据。

同一 K 算子转 FP32、关闭 TF32 后，full/short 仍不逐位一致，但 relative L2
仅为 2.9941e-7/3.1514e-7/1.6810e-7（比例，不是百分比）。未跑全模型 FP32。
第一层六种 Linear 中 Q、fused gate/up 相同，K/V/O/down 均可独立产生差异；
**不能把第一处 K 分歧等同于全部最终误差都由 K 单独造成**。

#### 干预结果与性能代价

仅在诊断 wrapper 中把全部文本 Linear 的 token 行数补到当前全量 N：真实后缀保持原值，
其余行补零，计算后裁掉补行；未修改 token/位置/缓存/视觉结果。attention 对照只补 Q，
不添加 KV，保留真实 KV 长度及 causal 语义。

|案例|只固定 Linear 后缀 L2|只固定 attention 后缀 L2|两者同时固定 L2|Linear-only / attention-only / both 诊断 ms|
|---|---:|---:|---:|---:|
|雨衣 short|0，逐位一致|0.728334%，未变|0，逐位一致|327.51 / 63.23 / 361.82|
|雨衣 medium|0，逐位一致|0.819371%，未变|0，逐位一致|328.90 / 62.39 / 364.24|
|棋盘格 short|0，逐位一致|0.943696%，未变|0，逐位一致|409.95 / 79.96 / 464.20|

诊断计时每格 n=1，包含补行与额外计算，不作为生产性能。三例各干预的未变化前缀也一致，
因此 Linear-only 与 both 均为整段逐位一致。**补到全量形状把约 29–32 ms 的正常复用
增至约 328–410 ms，只能作为定位手段，不能直接当作已完成的低延迟修复。**

第一层 attention 使用完全相同 Q/K/V，full Q 和 short Q 输出逐位一致；
显式 `pack_gqa=False/True` 也一致。二者相对显式 bottom-right causal FP64 reference
（最终转 BF16）的 L2 相同，三例分别 0.088643%/0.084537%/0.075196%，
这是共有的算子精度差异，不是复用新增差异。full/short 均执行 FlashAttentionForwardSm90。

K/V 在已测补行点中 512 行即可一致，O 为 128 行，但 down 补到 512 行仍不同；
不能把这些第一层离散探针当作全模型 padding 阈值。本轮没有单独禁用 split-K，
没有追踪内部 SASS 或精确舍入位置，不能断言关掉某个归约开关就能全模型逐位一致。
若后续要求兼顾严格一致性和速度，需要另验稳定 GEMM 算法/归约配置及全模型代价。

范围仅为这三例：正常 BF16 均通过原数值门槛但非逐位一致；不扩大为 12 例、79 格、
任意形状或画质验收。生产实现与默认值未改，没有把诊断 padding 并入产品。

CPU `dlcjtf3t79e988fu`、GPU `dlc1e26uizd9l1v8` 均 Succeeded，GPU 用时 417 秒；
三例 trace/profiler、18 个正常计时样本及 DONE 已回读。GPU 已释放，配额恢复 7/8，
未修改其他服务。r21 因诊断字典重复 `rows` 关键字失败，保留失败身份；
r22 改为 `padded_rows` 并经过 CPU 预检后重跑，不把 r21 计为成功。

执行 harness `39125cf2cb8686bea962b2719323b8f9e0e0b164`，备份分支
`codex/conditioning-prefix-bf16-diagnosis-r22`。报告、CSV、逐层/孤立算子/干预结果、
Chrome traces、日志及冻结源码归档于
`oss://leap-world-shanghai/world-model/sglang/evals/sglang400-qwen-bf16-diagnosis-20261110-r22/qwen-bf16-diagnosis-evidence.tar.gz`，
SHA256 `1381b8d9ff54a147e01537fda63d8b98f841b850ec61c01116a21b92bf956cbf`，
上传读回校验通过。每例第一处 Linear 原始操作数 `.pt` 单独保存在同一 OSS 根目录，
不在此归档中，也未声称本轮下载校验这些张量文件。

### 2026-11-10：同步 main 的 prompt token 准入

同步 main `caf4a56f7faf9ae435b3aad3f8c198c8d028b19f` 时，合并
`encode_condition` 与 `NativeConditionEncoder` 的参数冲突：同时传递部署侧
`max_prompt_tokens` 和任务侧 `prefix_cache/media_identity`。长度检查仍在媒体预处理和
模型执行之前；启用任务缓存不能绕过 prompt 准入，异常仍清空任务缓存并 drain 设备。
新增 CPU 回归覆盖任务缓存开启时的精确边界通过、超限拒绝、缓存清理与设备 drain。

后台预取继续使用同一 `WorkerConditioningState.encode_remote` 回调，与同步编码共用
任务句柄；保留 main 的 Block0 判断和 NCCL 更新。CPU inventory 保留 main 全部条目，
额外保留本 PR 的 `test_prefix_task.py`；部署文档保留双方结果和各自的版本边界。
此次是合并兼容修复，没有采用 r22 的诊断 padding，没有修改生产精度默认值。
前述 GPU 数值与性能证据仍对应各节冻结提交，不能作为本次合并后版本的 GPU 验收。

合并工作树的完整 conditioning CPU runner 首轮为 2214 passed、12 failed（失败计数包含
子测试）。失败来自本地 CPU 轮子误选 CUDA 平台、WSL Git 路径、以及新增测试夹具缺少
`image_processor`；修正测试夹具并设置 `SGLANG_DIFFUSION_PLATFORM_OVERRIDE=cpu`、
使用 Windows Git 访问工作树后，包含全部首轮失败项的文件复跑为 191 passed、3 subtests passed，
零失败、零跳过。未将这组复跑表述为第二轮全套通过。CPU inventory、冲突标记、PR diff
空白检查与相关 lint 通过；Windows 缺 `python3` 的三个 hook 用同一脚本/规则显式执行通过。
额外 Qwen 模型 helper 测试受本地缺 `sgl_kernel` 的导入边界阻断，未执行，也不冒称 GPU 验证。


## 2026-11-10 审查后精简与回归补齐

本轮保持用户入口、RPC schema、任务生命周期和数值精度配置。服务端请求解码产生结构化 `DecodedRequest`，把服务端计算的媒体摘要与客户端 header 分开；逐图像素只哈希一次，同批可信摘要分别生成完整请求身份与纯媒体身份。完整请求 digest 保持原格式，仍验证客户端声明；媒体身份覆盖顺序、类型、shape 和像素。native 的本地直接调用没有解码元数据时仍自行计算身份。不得将客户端提供的媒体摘要当作可信值，也不得跨请求复用未经验证的哈希。

Qwen 同长度完全命中时保留内部 KV、token、位置及特征，仅克隆返回结果；缩短命中仍裁剪 KV 和压紧特征存储，后续扩展重算后缀。输出与缓存保持不同所有权。没有增加媒体 processor 缓存，也不改变两种锁的职责。

Qwen 缓存契约测试迁入 `test/unit/conditioning/test_qwen_prefix.py`，随 conditioning 默认目录发现并登记 manifest。CPU runner 只隔离 SRT 的三个 GPU 视觉组件构造，误用即抛错；真实 `encode_ids`、投影、缓存与 Qwen backbone forward 保留。验证计划：完全命中/缩短后扩展/输出隔离、逐图哈希次数与篡改拒绝、真实 ASGI 到 native 的摘要传递，以及现有 conditioning 套件、CI inventory 和 lint。CPU 证据不覆盖 GPU 算子、完整视频或优化延迟；历史 r20/r22 仍对应旧受测提交。

本轮实测对应实现 `ed582176611cfcd2f347f4d5f9a32e96114fc309`（随后仅补验证文档）：

- WSL Linux、Python 3.12.3、torch 2.14.1+cpu；CPU 平台覆盖、OMP/MKL/OPENBLAS 线程 1、HF 离线，沿用本地 Windows 工作树的 Git 适配。完整 `run_conditioning_cpu_tests.py` 默认套件 **2229 passed、59 subtests passed、0 failed、0 skipped**，553.10 秒。针对性回归另为 68 passed，不相加重复计算。
- 正式 `scripts/ci/zing/run_cpu_tests.py --suite conditioning-contracts --triage --file python/sglang/multimodal_gen/test/unit/conditioning/test_qwen_prefix.py --trials 2` 两个独立进程各 **5 passed、0 skipped**；源文件 SHA 与暂存提交一致，manifest 据此设为 approved。`--check-inventory` 成功确认收录。
- RPC 集成测试经真实 ASGI、解码、排队与 native/预处理边界，核验两张图每请求仅两次像素哈希；文字变化保留媒体身份，像素/顺序/类型/shape 变化更新身份。旧 digest 配新像素、客户端伪造媒体身份均拒绝，未进入编码。仅模型计算及 CUDA 设备操作使用 CPU 测试替代，不能代替真实 GPU。
- 完全命中验证缓存存储不重建、返回结果独立；缩短后输出存储压紧、再次扩展与全量 CPU 因果 oracle 相等。真实 Qwen backbone 的 native config forward 测试也通过。
- Ruff、isort、语法、bare-pytest、包内注册限制和 diff 检查通过；全仓注册检查唯一报告 main 未改的 `test/registered/kernels/ops/test_fp8_scaled_mm_routing.py` 缺注册，不在本轮扩修。Mintlify CLI 不可用，未做全站构建。

原始日志、JUnit、CI receipt 与文件哈希保存于 `D:/sglang400-review-fixes-20261110/`。新增 RPC 测试早期缺少 Content-Type 被正确拒绝，另修正同批图像 shape 夹具；失败记录保留，未放宽产品校验。没有新增 GPU/端到端实验，因此不报告本轮减少了多少毫秒，也不宣称既有 FP8/BF16 数值差异已修复。

后续监控扩展见 [Qwen 前缀复用：Prometheus / Grafana 指标](conditioning-prefix-api.zh-CN.md)，不扩展业务 API 响应。
