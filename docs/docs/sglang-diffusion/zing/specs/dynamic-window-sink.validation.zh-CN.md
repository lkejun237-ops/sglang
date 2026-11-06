# Dynamic Sink 验收记录

2026-11-03：实现和本轮自动验收完成。CPU CI、SP1/SP2 数值隔离、H100 独立角色五段完整 480 视频、B300 十段完整 480/736 视频均通过，包含真实全分辨率 Graph、同卡串行对照与取消恢复。两种分辨率均有完整可播放 30 秒视觉对照，已检查关键帧和覆盖全片的接触表；完整视频供人工最终观看。

本文 `exports/` 相对工作区 `/home/ss80194/work/sglang-zing`。下列 GPU 结果冻结于 `d87d75fb16`，本次 PR 的后续主线整合边界见下节。

## 2026-11-06 主线冲突修复

合入 `main@b9e4bc95fc`，保留共享静态 KV 容量／准入、独立服务流控与取消收尾、unsplit prompt joint 和 VAE 优化。两处文本冲突分别保留部署文档双方新增记录、移除 Encoder 入口空行冲突；自动合并后仍在主线 `_forward_source()` 内传递 sink 控制，并保留首次联合 forward 前应用 pending sink 的顺序。

本次整合仅在 WSL 检查 AST、Ruff、isort、CI inventory 和 diff；运行时 CPU 回归以本次 PR checks 为准。上述主线改动与 sink 的组合尚未重跑 GPU，以下冻结视频与数值结果不代表本次整合已通过 GPU 验收。

## 提 PR 时的主线整合

提 PR 前整合 `main@a9a4b2f93e8ceb30120d7f374660c8e3d8e066f9`，包含首次 condition/source/target 联合 forward、Prompt/参考图 API 预算，以及分页 Block0 收尾修复。保留原提交历史，以 merge commit 解决两处重叠：

- `session.ready` 同时发布 `sink_updates` 和 `input_budget`。
- 单块 `step()` 先在拥有权边界应用 pending sink，再进入主线 `_initial_prediction()`；其首次联合 forward 和串行 fallback 行为均保留。

新增 CPU 回归比较 off/refresh 在首块前提交时，开启/关闭首次 prefix 联合 forward 的输出、全部留存 raw KV、块序和 sink 回执；同时覆盖 `step()` / `drain_ready()`。最新结果以 PR checks 为准。

以下真实 GPU、显存、forward 次数与视频结果均属于原冻结源码；整合主线后的首次联合 forward 与动态 sink 组合尚未重跑 GPU，不能将旧结果冒称最新 head 的 GPU 验收。本次提交以既有 GPU 证明和新增 CPU 集成回归供 review。

## 代码与输入身份

| 项目 | 冻结身份 |
| --- | --- |
| 同步的 main | `0a4623e1299efc386a81473ed03d6a79ed234eaa` |
| GPU runtime | `d87d75fb16b30e4cc732a6f29bc0271cf3284bd1`，逐项校验 967 个运行相关 Python 文件 |
| 全绿 CI head | `fbef5b6dba0d82c040b2baef9656cc9eb1d764d0`；后两次提交仅修正 CPU 跳过清单，runtime 与 GPU 冻结源码相同 |
| 分支 | `docs/zing-sink-prefix-design` |
| Checkpoint SHA-256 | `89c3a67d77214e57d810e6a4e5554c7c919ae929db0174e6439dfadb53a1d731` |
| `zing_config.json` SHA-256 | `21cf5379e60ac8554a59a1aa4400f776329337f6f2847fe9659c6a34cf6bf8f4` |
| 服务语义 | unsplit `[3,2]`，W6/S2，视频；H100 完整视频关闭 Graph，数学隔离使用真实 Graph |

PAI CPU 作业 `dlc18gxjij5rkiqt` 核对了完整模型归档、41 个权重文件和 transformer index。H100 使用已核对 checkpoint 的 weights-free AdaLN 表，DiT 权重 37.44 GiB/卡；B300 使用完整 resident 权重，77.88 GiB/卡。两种部署分别记录，不跨配置比较性能。

| 完整 RGB 输入 | 帧数 | SHA-256 |
| --- | --- | --- |
| 室内 832×480 | 720 | `75ef65248be2bc8bb611eb1798925dfcf7820b9aaa3e9282990e8374e28fd01e` |
| reference-switch 832×480 | 764 | `b9fb587ee3dc1335b11bd78bc3a5e1a5f959461cc9484da7e0b2fb8555e808b2` |
| 室内 1280×736 | 720 | `5133cb4995951d97f0d70b100d974271e1da7fd40236b3c1d0f706a765104a1f` |
| reference-switch 1280×736 | 764 | `395ce5ac5047900e1282aa9da05ee110668aee6bf73d043be7aef4a1995fb254` |

使用正式客户端相同的 FFmpeg bilinear、`-vsync 0`。736 是显式派生输入；原 1280×720 source 不作为 736 验收。图片引用、prompt/sink 脚本、全部输入帧及部署文件在零 GPU 作业中组成冻结闭包，见 r19 `DATA_CLOSURE.json`。

## 正确性证据

| 风险 | 判据与结果 |
| --- | --- |
| KV 切错、读到丢弃历史 | CPU 以已知数值的 source/target KV 检查实际行内容、角色、块序和 prefix；同时逐 forward 核对可见历史和全局范围，已通过 |
| pending prompt、已预执行 cross-chunk、未提交 final、同边界组合 | CPU 控制拥有权、排队次序和条件就绪时点，检查快照与 latest-wins，已通过；奇偶相位作为低成本参数覆盖 |
| Prefix cache 复用 | CPU 比较缓存恢复和未缓存路径，检查 sink 操作保留条件 prefix、prompt 切换保留 S0/S2 模式；已通过。GPU cache hit/miss 性能矩阵不在本轮范围 |
| Graph 执行差异 | 精确 main/feature 控制及同融合路径 Graph/eager 的 26 块混合脚本，SP1、SP2 均逐值相同，见下节 |
| ACK 与实际输出错位 | H100 为 off 第 366 帧、refresh 第 519 帧；B300 两分辨率均为 357/502 帧。首次变化均与 sink 快照一致，此前原始 RGB 逐帧相同 |
| 混合控制 | 原生独立角色完整 764 帧通过；8 条 sink 控制中 7 applied、同提交帧的旧 off 1 superseded，6 次原始 prompt/reference 更新保留 |
| 取消后状态污染 | 观察到 sink 生效后取消；同服务下一条完整 720 帧通过，生成 MP4 与固定基线逐字节相同 |
| EOS 与媒体完整性 | H100 五段 3,644 帧、B300 十段 7,332 帧；输入哈希、RGB 字节数、输出帧序和全片解码均通过；多个 EOS 后新会话完成 |

[CPU CI 41597077017](https://github.com/seedleap/sglang/actions/runs/41597077017) 全绿。四个 DiT 分片、两轮 JUnit 为 6,690 条执行记录、486 条条件跳过，0 failure/error；不是独立新增用例数。

实测发现并修复了 graph source 的 CPU 位置元数据与 standalone eager final 的 CUDA 元数据无法拼接的问题。`d87d75f` 优先复用已校验 host positions；零 GPU red/green 作业 `dlcw0c4crk9lupdd` 复现旧代码、验证修复，真实 GPU 作业 `dlc1uz6sjvrcldva` 的 pooled/unpooled × meta/CUDA 四项均通过，无失败或跳过。

WSL 仅做编辑、静态检查及媒体产物分析；没有安装项目依赖或运行项目测试。

## 数值隔离

完整 checkpoint，小空间 4×4 latent 几何，synthetic Qwen 条件行数 12→18→6→12，26 原生块/65 latents。冻结输入 SHA-256：`085b90cdf8dd50f2301036169a3e394951c553a488b2149644649ce657548460`。

| 比较 | B300 SP1 resident | H100 SP2 weights-free，每个 rank |
| --- | --- | --- |
| main/feature 固定 S2，串行初始 3 块 | exact，max 0 | exact，max 0 |
| main/feature 固定 S2，融合 eager 初始 3 块 | exact，max 0 | exact，max 0 |
| feature 混合 26 块，同 cross-chunk/deferred 路径的 eager/真实 Graph | exact，max 0、MAE 0；trace 相同 | 两 rank 均 exact，max 0、MAE 0；trace 相同 |
| main 串行/融合 eager，事件前初始 3 块 | max 0.78140485，MAE 0.09972900 | max 1.59424043，MAE 0.17439051 |

feature 在最后一行得到与 main 相同的差异，因此串行对融合的 0.02 gate 不能独立归因 sink，也不能声称两条路径整体数值等价。数学证明采用 main/feature 同路径控制，以及同融合路径的 Graph 开关隔离。

两种平台均真实 replay：deferred 19、resume 14、recovery 3；各为 72 次逻辑 DiT forward：prefix 4、denoise 32、commit 10、source_first 8、cross_chunk 18。SP2 的 72 是每个 rank 执行同一逻辑计划，不能与 SP1 的单 rank 计数直接相加比较性能。

SP2 两 rank 数值断言完成后，验收脚本末尾的默认 NCCL barrier 卡住，r9 因此没有记为整作业成功。项目 `GroupCoordinator.barrier()` 明确要求 CPU group。r10 的两 rank 小 Graph 均完成 3 次 replay、Gloo barrier 和正常 group teardown 后才启动视频。

数值证明路径：

- SP1：`exports/sink-prefix-20261103-b300-r18/retrieved/results/numeric/feature/rank0/NUMERICAL_GATE.json`。
- SP2：`exports/sink-prefix-20261103-r9/retrieved/results/numeric/feature/rank{0,1}/NUMERICAL_GATE.json`。

这些是模型执行路径证据，完整视频另行验证。synthetic 条件下 prompt joint 走 layout fallback；实际产品混合视频也记录该分支，不能将其写成 prompt-joint 成功融合的 GPU 证据。

## 视觉对照与实际边界

H100 四卡独立角色：Encoder/Qwen 逻辑卡 3、DiT SP2 卡 0–1、Decoder 卡 2。完整视频为 eager；模型、硬件、种子、输入与 prompt 在室内三组间一致。H100 pool 的 prefix 上限为 12,288 行、容量 26,530 行/slot；不能与 B300 16,384 行上限的部署混算显存。

| 操作 | source 提交帧 | 首次匹配输出帧 | 实际锁定 source | send 到首个匹配 RGB |
| --- | --- | --- | --- | --- |
| off | 416 | 366，即 15.25 s | 无锚点 | 844.89 ms |
| refresh | 560 | 519，即 21.625 s | chunk 59、60，对应 `[502,519)`，20.917–21.625 s | 939.82 ms |

提交进度可领先模型已拥有的 native 块；事件在拥有权边界应用，因此提交帧不是生效帧。off 比首轮 14.5–15 s 候选略晚，但仍在 15.5 s 转向前，保留该自然边界。refresh 两块完全位于 `[450,522)` 稳定段，满足选点条件；这组自然覆盖 2+3 相位，共 5 latents。

`RAW_PREFIX_COMPARISON.json` 对原始 RGB 比较得出：off 前 366 帧完全相同，后 354 帧有变化；refresh 前 519 帧完全相同，后 201 帧有变化。首次像素变化与首次 sink 快照逐帧对齐。

已查看 14、16、20、22、24、28 秒对照帧：14 秒三组一致；后续墙面、柱位、走廊和顶棚出现可观察变化，人物动作仍对应 source。该视觉证据用于观察影响，不把布局变化直接评为质量提升。

完整可播放 30 秒四宫格：`exports/sink-prefix-20261103-r10/retrieved/480-sink-comparison.mp4`；源/固定/off/refresh 分别显示，画面标注实际生效时间。混合脚本和取消恢复另有完整视频。所有下载均核对字节数、SHA-256，元数据先于视频回收。

## B300 全分辨率补验

r21 使用同一 B300、同一 resident checkpoint：480 优化模式开启真实 whole-chunk CUDA Graph（max 4）；480 串行混合对照关闭 Graph、cross_chunk、deferred final 和 prompt joint；736 优化模式保留 cross_chunk/deferred final，关闭 Graph。

480 的固定/off/refresh/混合/取消后新会话、同卡串行混合对照，以及 736 固定/off/refresh/混合，共十段完整视频、7,332 帧均通过。三组混合脚本均为 8 条 sink 控制中 7 applied、1 superseded，6 条 prompt/reference 更新全部 applied。取消后新会话的原始 RGB SHA-256 与固定基线相同，MP4 也相同。两种分辨率的 off 第一次变化均为第 357 帧（14.875 s）；refresh 第一次变化均为第 502 帧（20.917 s），锁定 native chunks 57、58，对应 source `[485,502)`，位于选定稳定区间内。零 GPU observer 对四组原始 RGB 全片逐帧比较，四组的第一次像素变化都与首次生效快照精确相同。

已查看 736 的 14、22、28 秒四宫格：生效前一致，生效后柱位、墙面与走廊布局有可见变化。完整对照分别为 r21 `retrieved/480-sink-comparison.mp4` 和 `retrieved/736-sink-comparison.mp4`。480 串行混合全片接触表也已查看；视觉采样与完整媒体解码分别记录，不能将接触表检查表述为人工逐帧看完全片。

480 优化服务日志记录 56 次 Graph capture snapshot：deferred 20、resume 20、init 4、recovery 12。capture 失败会抛错；成功 capture 后实际调用 replay，完整请求成功，因而不是仅打开配置但未执行 Graph。完整视频未导出逐 route replay 计数，精确 replay 次数仅以上面的数学隔离为准。

串行与优化模式的已拥有 native 块进度不同，生效帧可以不同；对照检查各自的拥有权边界、事件结果和帧数。数值一致性仍使用上面的同路径隔离，不要求已有数学差异的两种路径像素相同。

## 融合拆分成本与资源

H100 eager 的单次完整请求，完整 EOS 日志的逻辑 forward 统计：

| 用例 | 帧数 | DiT forward | source 块计算 | prefix forward | TTFF | 请求 E2E |
| --- | --- | --- | --- | --- | --- | --- |
| 固定 S2 | 720 | 177 | 85 | 1 | 64.372 s，含首次冷编译 | 94.205 s |
| off | 720 | 178 | 85 | 1 | 2.432 s | 33.099 s |
| refresh | 720 | 178 | 85 | 1 | 2.464 s | 32.235 s |
| 混合 reference-switch | 764 | 211 | 90 | 7 | 4.611 s | 39.031 s |
| 取消后新会话 | 720 | 177 | 85 | 1 | 2.500 s | 32.526 s |

单次 off/refresh 各多 1 次完成 deferred final 的 forward；source 块和 prefix 计算次数不增加。混合脚本的 7 次 prefix 对应初始 1 次与 6 次 prompt 更新。没有重算已提交历史。

原生多进程 stdout 有 7 行 JSON 被交错写入破坏，off 的 rank 0 最后观测到 714 帧，因此表中使用完整 EOS 的 rank 1；该处不能把 rank 0 的部分计数 176 写作总数。原始回执、完整/部分 trace 标记均保留在 `VALIDATION_SUMMARY.json`。

同步 driver 显存每秒采样四张选中卡，共 741 次：同一采样点总峰值 245.408 GiB；各卡峰值 75.690、74.438、33.622、61.657 GiB。包含测试 wrapper 的 CUDA context/保留张量，可能漏掉亚秒瞬时峰值。DiT worker 的 live allocated 和 lifetime peak 另存，不把 allocator 保留量、逻辑 KV 行数与全卡显存混作同一指标。

以上均为功能试跑的一次测量。冷启动/编译、控制到首 RGB、请求 E2E 的口径分别记录；未执行完整产品四 case × 三次性能矩阵。

B300 同硬件一次试跑，详细回执为 r21 `retrieved/VALIDATION_SUMMARY.json`：

| 用例 | 帧数 | 逻辑 forward | TTFF | 请求 E2E |
| --- | --- | --- | --- | --- |
| 480-fixed | 720 | 177 | 28.704 s | 92.929 s |
| 480-off | 720 | 178 | 3.473 s | 66.270 s |
| 480-refresh | 720 | 178 | 3.134 s | 67.664 s |
| 480-mixed | 764 | 212 | 3.923 s | 75.982 s |
| 480-mixed-serial | 764 | 未导出 | 10.824 s | 79.988 s |
| after-cancel | 720 | 177 | 2.630 s | 66.627 s |
| 736-fixed | 720 | 177 | 10.263 s | 114.358 s |
| 736-off | 720 | 178 | 2.683 s | 106.629 s |
| 736-refresh | 720 | 178 | 2.625 s | 103.783 s |
| 736-mixed | 764 | 212 | 4.140 s | 110.767 s |

480 首条含冷编译；736 首条也有新尺寸启动成本。串行日志没有逻辑 forward 计数，明确留空而非推算。B300 两种分辨率的单次 off/refresh 仍各多 1 次 forward，source 块 85、prefix 1 保持不变；混合为 source 90、prefix 7。两种模式的 mixed 单次 E2E 不能当作稳定性能提升比例。

B300 pool prefix 上限 16,384 行；实际室内 prefix 为 480 的 7,917 行、736 的 8,447 行，slot 容量分别为 30,626、49,706 行。monolithic worker 报告的 lifetime peak allocated 上限分别为 161.63、194.62 GiB，包含 Encoder/DiT/Decoder；Qwen 为另一进程。该计数包含此前会话峰值，既不是每 case 独立增量，也不是全卡 driver 峰值。B300 全卡 driver 指标未采到；不能与 H100 四卡同步总峰值直接比较。

## 作业与最终状态

| 作业 | 范围与终态 |
| --- | --- |
| PAI `dlc1uz6sjvrcldva` | 四项 CUDA/meta 元数据回归通过；后续脚本配置缺失，未进入视频 |
| PAI `dlcrbcirvtzggi7c` | SP1 数学隔离通过；视觉资产文件漏打包，整作业 Failed |
| H100 `sink-prefix-20261103-r9-gpu` | SP2 两 rank 数值断言通过；末尾测试 barrier 卡住，保存证明后删除实验 Pod |
| H100 `sink-prefix-20261103-r10-gpu` | 五段完整 480 视频、混合与取消恢复通过；额外串行对照继承 cross_chunk 专用 `joint_wait_ms=50`，握手拒绝，整作业 Failed |
| PAI `dlc1rf0gzvs3cj8t` | r21 零 GPU 预检 Succeeded；用实际 adapter guard 复现数值等待 + off 的拒绝，验证修正后的所有模式，完整哈希比较 conditioning profile ID 一致 |
| PAI `dlctkgtfvigwoamf` | Succeeded，10:48:46–11:15:30 UTC；r21 单卡完整补验十段视频均通过，7,332 帧 |
| PAI `dlc8rgw0bvjse0wp` | Succeeded，零 GPU 原始 RGB observer，四组首次变化均与 sink 快照精确一致 |
| PAI `dlc189vaq3vhp67y` | Succeeded，零 GPU 视频验证与私有产物回收，十段视频及全部元数据 SHA-256/字节数已核对 |

单卡 Qwen/DiT bridge 显式配置 `NCCL_MULTI_RANK_GPU_ENABLE=1`，对应 [NCCL 2.30.7 的实验性同卡多 rank 能力](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2307/user-guide/docs/env.html#nccl-multi-rank-gpu-enable)。只用于本次部署，feature 不设置该变量。真实小张量验证先于完整模型加载。

全部已结束作业累计 4.5334 GPU·小时，包含环境失败与重试；GPU 活动作业为零。r21 按 GPU Pod 创建到作业结束保守计 1,633 秒，即 0.4536 GPU·小时。批准总预算同时最多 4 卡、累计 16 GPU·小时；排队未计 GPU 用量。完整成本、较早环境失败及零分配队列见 `exports/sink-prefix-20261103-r4/GPU_BUDGET_LEDGER.json`。AWS 凭证未重试，未创建 AWS 资源；现有资源已完成本轮批准范围，无需继续等 Spot。

本轮范围已闭环。边界：H100 全分辨率 Graph 曾容量不足，本轮以 B300 完成 480 全 Graph；736 完整视频采用 eager。prompt-joint 的 unsupported_layout fallback 已显式记录。未执行完整产品性能矩阵，也未将串行/融合之间原有数学差异写成数值等价。
