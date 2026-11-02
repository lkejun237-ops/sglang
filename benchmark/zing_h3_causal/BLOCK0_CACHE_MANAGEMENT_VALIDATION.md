# Block0 共享 L2 管理与强制 prefill 验收

对应 PR：[seedleap/sglang#467](https://github.com/seedleap/sglang/pull/467)。
业务和接口契约见[开发方案](/docs/sglang-diffusion/zing/plans/BLOCK0_CACHE_MANAGEMENT_SHARED_L2)。

## 2026-11-01：SP=2 完整视频输出对照

候选源码 `37e76af6fdad9b03d0718c5f92b440184ad06c67`（运行时代码 `4f6af40030`）
通过两组完整 480p 输出对照。[同步播放与原始结果](https://review-cdn.loopit.com.cn/reports/pr467-cache-20261101/r5/index.html)
需办公网或公司 VPN；左侧关闭缓存重算，右侧显式 prefill 后命中 L1。

| 原始 product case | 每组帧数 | 更新结果（off / on） | 原始 RGB 对照 |
| --- | ---: | --- | --- |
| few-images | 403 | 无 prompt 更新 | 全帧逐字节一致，MAE / RMSE / 最大差均为 0 |
| reference-switch | 764 | 各 6 次更新全部 applied | 全帧逐字节一致，MAE / RMSE / 最大差均为 0 |

每个 case、每组仅一次请求；保留完整 prompt、参考图、更新边界和全部输出。
两组输入 SHA256 相同，seed=42，832×480、24 fps；视频播放时长不作为生成速度。
使用上海 `e01-cn-2zd4xi8m505` 的 NVIDIA L20Z × 4（每卡 81,559 MiB），
DiT SP=2 / FA4、Encoder 与 Decoder 各一卡、remote Qwen BF16 / SDPA，
eager、cross-chunk joint、W6/sink3，初始三块四步、其后两步。
模型为 1027 step010000 AR+DMD diff，checkpoint SHA256
`89c3a67d77214e57d810e6a4e5554c7c919ae929db0174e6439dfadb53a1d731`，
`num_frame_per_block=[3,2]`、`split_first_latent=false`；按修正后配置重新生成 AdaLN 表。

显式 prefill 8 条记录形成 5 个不同 key，未淘汰。通过 realtime_trace 的 request/session ID
关联缓存 operation_id，固定场景观察到 1 条查询，换图场景 13 条查询；全部为
`hit / l1 / allow_disk=false`，覆盖初始请求和六次换图。
没有直接采集 KV Tensor hash；数值结论限定为这些输入的完整 RGB 输出。

关闭缓存的基线来自 Pod `pr467-cache-video-gpu-20261101-r4` 已完成的 off 子组；
该作业后续 on 子组因重复使用 conditioning NCCL bridge 失败，整作业标为 Failed。
修正服务生命周期后，Pod `pr467-cache-video-gpu-20261101-r5` 使用全新 Qwen/worker 配对完成 on 子组，
Succeeded / exit 0；复用前逐条核验基线原始 RGB 大小和 SHA256，并保留原失败日志。
两组使用相同源码、模型、配置和物理 GPU。
测试编排使用现有 role_launcher 启动 Encoder/Decoder，再 spawn 未修改的 DiT/HTTP 入口；
无模型观测/替换 hook。本轮不验收原生 CLI 的 role 分卡启动入口。

CPU 导出完整解码核对 6 个 MP4 的帧数。13 个发布对象的源站完整 SHA/MIME 及 CDN 内容/MIME/视频 Range
均核验通过。Firefox 对本地发布树的四个原始 MP4 完成元数据、同步播放和同步拖动检查，
无页面脚本错误；CDN 浏览器访问及人工画质验收待完成。早先 Chromium 镜像缺少 H.264 解码器，
其失败回执保留，不当作媒体损坏。
结果包 66,759,677 bytes，SHA256
`20e0dddda230de139b144ca1348bb4a50bf4f5c72e516b7793cce59b7ca59ece`，
已回读并逐文件核验；本地 `exports/pr467-gpu-20261101-r5/`，
远端 `/mnt/zing-nvme/pr467-gpu-20261101-r5/`。

这是 480p 输出与命中路径验证，不提供性能排名。736 输入尚未冻结，本轮未执行；
480/736 同条件重复 benchmark、阶段耗时和显存采样仍需补齐。
同机双实例真实 API 的独立验收状态和回执在 PR #467 中单独记录。PR 保持 Draft。

## 2026-11-01：缓存日志与 source FPS 去重 CPU 复验

候选提交 `b005e65fe9bad1329e14255fa1afbb4616b846a6`（运行时代码 `4f6af40030`，
随后仅修正测试断言与 fixture 路径）已通过：

- [缓存专项](https://github.com/seedleap/sglang/actions/runs/41284470554)：两轮合计 **396 pass，零失败/跳过**。
- [完整 CPU CI](https://github.com/seedleap/sglang/actions/runs/41284470598)：全部通过，
  JUnit **12,762 项记录、零失败、432 项跳过**，跳过数与前次完整回归一致。
- 两轮精确 head 与 JUnit 已回读核验，完整 CI 的 13 个 artifacts 保存在
  `exports/pr467-cache-logs-20261101-r1/`，汇总为 `FINAL_VERIFIED.json`。

覆盖 key 中重复 source.fps 去除、模型/媒体身份隔离、缓存事件级别和关联字段、
异步落盘成功/失败、prefill 复用/force、批量删除/卸载、管理部分失败/拒绝及 GC/轮询。
API 接入文档见[缓存管理 API 与部署契约](/docs/sglang-diffusion/zing/specs/block0-cache-management-api)。
本轮文档纠错不改变运行时代码；仅做 JSON、Shell、参数及链接静态核对。

本 CPU 复验之后的 GPU 输出对照见上节；其余验收缺口以最新记录为准。

## 2026-11-01：同步 main 后的 CPU 复验

提交 `c54e9268153818edcc9338f8ce8c8a8850699321` 已同步 main `2242cc4fa7`，
解决部署文档与旧身份测试的冲突，保留 main 的 source 缩放、首块调度和解码生命周期改动。
参考图 resize 上限会改变有效模型输入，因此参与 preparation key；新增用例验证不同上限隔离、
恢复默认设置后 key 一致，继续使用简化配置身份，不恢复旧 runtime_policy。

[缓存专项](https://github.com/seedleap/sglang/actions/runs/41279424307) 通过，
两轮合计 **380 pass，零失败/跳过**；JUnit 已回读核验。
[完整 CPU CI](https://github.com/seedleap/sglang/actions/runs/41279424329) 的全部业务套件通过，
JUnit 合计 12,746 项记录，零失败、432 项跳过；跳过数与此前完整回归一致。本轮未新增 GPU 或 NVMe 性能验收；旧性能证据的版本边界保持不变。
本地回执：`exports/pr467-ci-20261101-r1/`。

## 用户管理缓存版本：新目录与索引验收

2026-10-30，新 key、正常视频 L1-only、写入开关、game/group 索引与 preload/status/unload
已进入候选代码；提交 `65ae9daf7421b1a6ac5648c6e7c9058a797aefd6` 的
[缓存专项 CI](https://github.com/seedleap/sglang/actions/runs/41034480612) 已通过，
两轮合计 **378 pass，零失败/跳过**：每轮 156 项缓存/KV pool、6 项 Graph provider、27 项 prefill/管理 API。
JUnit 已回读逐项核验，artifact `12967617242`，本地 `CPU_FINAL_VERIFIED.json`。
[完整 CPU CI](https://github.com/seedleap/sglang/actions/runs/41033440960) 已通过，源码提交 `87ec8e38f3`。
以下 NVMe 结果仅验证文件/索引并发，不覆盖模型或 HTTP 到 worker 的完整路径。

Pod `pr467-managed-cache-20261030-r1` Succeeded / exit 0，上海同一 `/dev/nvme0n1` / ext4，
2 CPU / 1 GiB / 零 GPU，上限 330 秒；预检可用约 1.1 TiB。实际使用同一 boot_id 的 PID 14/15。
候选 disk_cache.py SHA256 `e78ee70dd23f61fe253113af51ad8b4ef4732ea330e662d2e20c78201ddd7fdd`，
index 与 probe 的精确源码摘要见产物 SOURCE.json。

- 两进程分别发布 182/120 次，完整读取 72/43 次，允许竞争 miss 3/31 次。
- 新 `entries/<key>` 目录的逐文件发布、读删恢复、mtime 与进程退出残留 GC 通过。
- 两进程重复登记各自组及共同 key，游戏级结果去重为三个 key，单组两个 key；无关联丢失或临时文件残留。
- 重建索引管理对象后关联保持可读，允许索引指向尚不存在的磁盘实体。
- 复用真实文件管理与索引类定义；conditions 用 JSON 替身。真实 Tensor 编解码和 preload/unload
  由 CPU CI 验证，本轮未重跑 16.406 GiB IO 或 GPU benchmark。

实际原始 KV IO 三个函数 `_write_kv` / `_readinto` / `_read_kv` 与历史验收提交
`1889a04f78` 逐字一致，记录于本地 IO_REUSE.json；复用大 KV IO 结果，不把新目录/索引
探针当作新的带宽或 GPU 性能测量。测试 Pod 已删除，262,144 字节合成 KV 已清理，回执与源码保留。

本地回读产物：`exports/pr467-managed-cache-20261030-r1/receipts.tar.gz`，
SHA256 `5a106428c616f9d57b4abd1e46bc86c79ec0211aae853df8380933457e8c2722`；
`VERIFIED.json` 已逐项核验两个进程的通过回执。测试的实际源码、Pod 清单和提交意图在同一目录。

复现时为两个同机进程使用全新隔离目录，在原有探针参数后增加
`--index-source CACHE_INDEX_PY --index-sha256 CACHE_INDEX_SHA256`。

## 历史版本本地 NVMe 验收


2026-10-30 确认保留 #467 现有运行时，只补同机本地 NVMe 验收与文档。
**状态：本地 NVMe 验收通过。** Pod `pr467-nvme-20261030-r4` Succeeded / exit 0，
节点 `e01-cn-2zd4xi8m505` / 上海，8 CPU / 80 GiB / 零 GPU，最长运行 20 分钟。
盘型 `SAMSUNG MZQL23T8HCLS-00B7C`、`/dev/nvme0n1`、ext4，挂载 `/mnt/zing-nvme`，预检可用约 1.3 TiB。
工具/测试源码提交 `08883300b3d4d4b0d55939ea72937dc04bb5c48f`，运行时仍为 `e42b951c15`；本轮没有运行时代码变更。
候选 `disk_cache.py` SHA256 为 `ce27db880810855db6a6dbfa7cc1778179171d81782343a2f87bfad448babf87`。
跨机器 CPFS 的历史失败仍未通过。

- NVMe 临时目录中的既有 CPU 合约 **181 项通过、零失败/跳过**；JUnit 已回读逐项核验。
- 两进程 PID 171/170、相同 boot_id、相同源码，均输出 ok；分别完成 182/120 次 rank 发布，
  完整读取 69/45 次，miss 7/31 次。读删竞争后的恢复、退出残留 GC、mtime 可见性均通过。
  182 包含 120 次并发阶段发布、60 次读删阶段发布和 2 次恢复发布；miss 是允许的竞争结果。
- size/mtime 在收到写完标记后的首次 stat 轮询即可见，未产生额外轮询等待；不将它解释成跨进程延迟的精确上界。
- Python 3.12.3 / PyTorch 2.13.0+cu130，CUDA available=false；pytest 8.4.2、pytest-asyncio 1.4.0、
  Diffusers 0.41.0 从仓库锁定版本的 wheel 离线加载，仅放在验收目录，未修改基础镜像。
  实际 imageID 为 `docker.io/library/import-2026-10-22@sha256:eab963c4ca51cdcb349810996203ba28e14058d69b905d4f6a5473872493177b`。

三轮合成 SP2 KV、整 key 17,616,076,800 bytes，两个 rank 并行 IO：

| Host 文件 IO，秒 | 第 1 轮 | 第 2 轮 | 第 3 轮 | 中位数 |
| --- | ---: | ---: | ---: | ---: |
| 写入并 fsync 整 key wall time | 7.911 | 7.831 | 8.073 | 7.911 |
| 冷读到预分配 CPU Tensor | 2.813 | 2.840 | 3.397 | 2.840 |
| 页缓存热读到预分配 CPU Tensor | 2.435 | 2.093 | 1.902 | 2.093 |

每轮冷读前两个文件驻留率均 0%、热读前均 100%，每次全量 Tensor 精确相等。
逐 rank 写入关闭耗时中位数为 4.630/4.788 秒，随后 fsync 中位数 2.890/3.261 秒；
分项允许重叠，不能相加推导整 key wall time。该测量不覆盖完整 lookup、conditions 加载、pinned 分配、
H2D 或 TTFF，不作为运行时性能提升的 baseline/candidate 对照。

早期环境准备回执均保留：r1 缺 pytest-asyncio、r2 节点无法从 PyPI 获取包（均未执行用例）；
r3 为 180 pass、1 个 Decoder 用例在导入缺失 H3 Diffusers 模块时失败。
r4 补齐锁定的离线 wheel 并通过导入预检后，181 项全部通过；没有删除或跳过失败用例。

完整日志、JUnit、逐 rank IO、页驻留率、实际源码、Pod/硬件回执归档：
`oss://leap-world-shanghai/world-model/sglang/evals/pr467-nvme-20261030-r4/receipts.tar.gz`，
SHA256 `326de5129525ea1f24445ea9bfeb4809377400a46800ac2f88a34a6636292f03`，已上传并回读核验。
本地产物在 `exports/pr467-nvme-20261030-r4/`，自动核验汇总为 `VERIFIED.json`；独立测试路径为
`/mnt/zing-nvme/pr467-nvme-20261030-r4/`。

当前工具为 `bench_block0_local_disk.py`：同一全新 NVMe 隔离目录启动两个进程，各自传入
`--root ROOT --source DISK_CACHE_PY --source-sha256 SHA256`，要求相同 boot_id、不同 PID。
已移除跨节点模式、网络文件系统的 60 秒可见性/恢复等待、周期栈转储及逐轮删除诊断输出。
上面的原始验收对应固定提交中的
[旧探针](https://github.com/seedleap/sglang/blob/08883300b3d4d4b0d55939ea72937dc04bb5c48f/benchmark/zing_h3_causal/bench_block0_shared_fs.py)。
该探针使用实际发布/删除/GC 类定义，conditions 使用 JSON 替身；真实 Tensor、force 和管理 API
由同盘临时目录中的既有 CPU 用例验证。

精简后的本地盘探针提交 `73a5d15898733c9d2e7da1729c99a89f809d4cf1` 已在同一 NVMe 重跑通过：
Pod `pr467-nvme-cleanup-20261030-r1` Succeeded / exit 0，2 CPU / 1 GiB / 零 GPU，上限 330 秒。
两进程 PID 15/14、同一 boot_id，发布 182/120 次，完整读取 71/44 次、允许 miss 5/31 次；
本地文件关闭后的 size/mtime、删除后恢复和退出残留 GC 均直接检查通过。
结果包 `oss://leap-world-shanghai/world-model/sglang/evals/pr467-nvme-cleanup-20261030-r1/receipts.tar.gz`，
SHA256 `c27d52b0c017355034b020854cce7adc69350bc3cf983dce6f8eaf86d57ac7e1`，已回读核验。
该次只验证探针精简；原始 181 项 CPU 与大 KV IO 结果保持上述版本边界。
缓存身份会哈希 stage 目录的源码，因此该目录保持原始字节；运行时代码仅调整缓存目录参数的 help 文本。


大 KV IO 工具 `bench_block0_disk_io.py` 调用实际 `_write_kv`、`_read_kv`，示例：

```bash
SGLANG_LIGHTWEIGHT_RUNTIME=1 PYTHONPATH=python python benchmark/zing_h3_causal/bench_block0_disk_io.py \
  --root /LOCAL_NVME/ISOLATED_RUN/io-data --output /LOCAL_NVME/ISOLATED_RUN/io.json \
  --source-sha256 CANDIDATE_DISK_CACHE_SHA256 --ranks 2 --trials 3
```

默认每 rank `(50, 2, 12288, 28, 128)` BF16，8,808,038,400 bytes；SP2 单 key 共 17,616,076,800 bytes。
这是接近历史最大 prefix 槽位的合成数据体量，不是本轮模型生成的真实 KV。
每 rank 一个 IO 线程并行读写，目标为预分配且已触页的 pageable CPU Tensor；不含 pinned 分配或 H2D。
写入关闭耗时与后续 fsync 分开记录；fsync 仅为本次冷读准备，不属于运行时持久化保证。
只对测试文件调用 DONTNEED，再用 mincore 检查文件页驻留率；<=1% 才标记 cold_verified，
热读前 >=99% 才标记 warm_verified。不满足时保留观测值，不冒称冷/热读，也不清空整机缓存。
每轮全量核对 Tensor 精确相等，记录逐 rank 与整 key wall time。三次样本不推导稳定尾延迟或在线 TTFF。

## 历史版本 CPU 回归

合并最新 main（#466）后的运行时与测试提交：`e42b951c15a81a8809b8706a1d9135c3f129d9a0`。
合并保留 main 的流水线默认值与 CUDA stream 设备修复；本 PR 提取的共享 CPU fixture 同步增加 stream.device。
磁盘缓存源码与先前 `7d5441a6f0` 完全相同，下文 CPFS 回执仍对应这份磁盘源码。

- [缓存专项 CI](https://github.com/seedleap/sglang/actions/runs/40890624545)：两轮合计 **362 项通过，零失败/跳过**。
  每轮缓存/池 153 项、Graph provider 6 项、prefill/管理 API 22 项。
  [artifact 12905336078](https://github.com/seedleap/sglang/actions/runs/40890624545/artifacts/12905336078) 已下载并逐份核验 JUnit。
- [完整 CPU CI](https://github.com/seedleap/sglang/actions/runs/40890631416)：全部通过，包含四个 DiT 分片、conditioning、Decoder、AdaLN、realtime、benchmark、tooling、lint 和最终汇总。
  conditioning 两轮各 **1951 项通过，零失败/跳过**；
  [artifact 12904863901](https://github.com/seedleap/sglang/actions/runs/40890631416/artifacts/12904863901) 已回读核验。
- 合并前 `7d5441a6f0a7b4f52057b0926caf6b3ed26ddc74` 的
  [缓存专项](https://github.com/seedleap/sglang/actions/runs/40885879182) 两轮 360 项、
  [完整 CPU CI](https://github.com/seedleap/sglang/actions/runs/40885887510) 也均通过，conditioning 两轮各 1944 项；
  旧回执保存在 `exports/pr467-cache-management-20261029-r2/`。
- 新覆盖：同 key 跨批次 SP 命中、进程并发单文件发布、打开文件后的 inode 替换、读删竞争、崩溃临时残留、子文件 mtime、指定 GC 执行者、近似水位与快照、删除部分失败，以及创建/写入/关闭/替换等阶段 ENOSPC 降级。
- 当前回执、CI ZIP 和逐份计数保存在 `exports/pr467-cache-management-20261029-merge/` 的
  `CACHE_CI_VERIFIED.json`、`CONDITIONING_CI_VERIFIED.json`；WSL 只做静态检查。

## 历史网络存储验收

CPFS 跨机器验收未通过，不在本 PR 当前支持或待交付范围。
原始作业号、日志、flock 失败与 stat 阻塞证据见
[固定提交中的历史记录](https://github.com/seedleap/sglang/blob/ae21c3f42351fb31907856eb3d83eb261872e256/benchmark/zing_h3_causal/BLOCK0_CACHE_MANAGEMENT_VALIDATION.md)。
当前工具仅用于同机本地盘，旧记录中的跨节点命令应使用其对应固定提交复现。

## 未验证范围

已完成的 480p GPU 输出对照见首节。480/736 的正式 TTFF、E2E、FPS 和 Encoder/DiT/Decoder
耗时、allocated/reserved/driver 显存对照均未测。旧版本性能记录不能作为本版验收。
旧 #455/#467 中间版本混跑明确不支持；旧磁盘格式只按 miss 重建，不提供迁移。
