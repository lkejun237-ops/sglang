---
title: "首次 condition/source/target 联合 forward"
description: "首次无缓存去噪的联合事务、执行边界和验证证据"
---

# 首次 condition/source/target 联合 forward

当前整合基线：seedleap/sglang 的 main `450080f8ca`（包含 #481）。实现最初基于 `0267732e6f`，此前在 main `4adab592c602` 上验证；历史结果保留各自提交边界。目标是将首次会话的独立 prefix forward 与 source/target step0 合并。入口沿用现有 joint_forward，关闭联合、CFG、音频及不支持的布局保持串行。缓存命中直接复用 prefix；prompt 切换的五/六分支契约保持原状。

## 决策与边界

复用 PrefixJointTransaction，增加显式 INITIAL_PREFIX_PHASES=(prefix, source, active)，不能根据三分支数量识别，因为跨块 final/source/active 也是三分支。首次 block 为 0/1/2，每个分支保留串行 timestep、modality、attention 可见范围和 RoPE 坐标。事务拥有 KV 发布：先 prefix，再 begin_chunk(target)，最后 source/active；失败清空会话缓存。调度器仅负责选择和回退，模型不依赖服务入口。

同时扩展 eager 和 fixed Graph 计划；SP1 或 stable shapes 的 Graph 初始化从两个 forward 计划变成一个；SP 未启用 stable shapes 保留原 prefix 独立计划和 source/target 联合计划。已有 mixed-prefix modulation 保留参考图与文本的混合时间，并正确计算纯视频输出边界。独立环境变量 `SGLANG_ZING_INITIAL_PREFIX_JOINT` 默认 `1`，在 sampler 构造时读取。设为 `0` 仅恢复首次独立 prefix + source/target 联合，不关闭 source-first、cross-chunk 或 prompt joint；仍须满足联合配置及 stable shapes/SP 条件。该变量没有 profile/CLI 同名字段。各 worker 必须设置相同值，修改后重启服务。

## 验证计划与状态

对比联合关闭/开启的首次预测、最终 latent、prefix/source KV、active 几何；检查一次联合代替 prefix+source/target 两次调用。覆盖 absolute/block_relative、参考条件、Graph 计划/回放、缓存命中与 prompt 切换回归、联合关闭回退。CPU 小模型用于验证数值与状态契约，不能证明真实 CUDA/SP 或性能。

用户后续授权并明确指定 W6/S3、两卡 SP2、仅 forward。已完成该范围的真实输入 480/736 A/B，见下方实测；完整端到端与 Encoder/Decoder 性能仍未测。

## 社区依据与 review 导读

核查当前分支既有 zing_h3_joint.py 的五/六分支事务、zing_h3_denoise_graph.py 的 planning_copy/publish_plan 和测试中的串行 oracle；本改动扩展这些已有机制，不声称私有 Zing 契约已被公共上游采用。review 重点是 prefix 是否只能看自身、媒体分支 RoPE 是否与串行一致、Graph 计划是否只在成功后发布，以及现有三分支跨块路径是否仍可区分。

阅读顺序：本文 → incremental.py 的首次选择 → zing_h3_joint.py 的事务校验/提交 → zing_h3.py 的 mixed-prefix modulation → zing_h3_denoise_graph.py 的初始化计划 → test_zing_h3_denoise_graph.py 和 test_zing_h3_joint.py 的 oracle。


## 已执行结果

### 原开发基线（2026-10-28，0267732e6f + 本功能增量）

本地 WSL / PyTorch 2.8.0+cpu，`SGLANG_LIGHTWEIGHT_RUNTIME=1`、`SGLANG_DIFFUSION_PLATFORM_OVERRIDE=cpu`、`SGLANG_DIFFUSION_PERF_BASELINE_PLATFORM=h100`（仅避免测试工具在导入时探测 CUDA，不是 H100 性能测量）、`PYTHONPATH=python`、OMP/MKL 单线程。执行 `python -m pytest python/sglang/multimodal_gen/test/unit/test_zing_h3_denoise_graph.py python/sglang/multimodal_gen/test/unit/test_zing_h3_joint.py -q -x --disable-warnings`：243 passed、4 skipped，223.14 秒。4 个跳过均为测试要求真实 CUDA；通过不代表真实 CUDA/SP 已验收。

新增测试覆盖 step/drain/固定槽 Graph、带/不带参考图；与关闭联合的串行最终 latent 和全部存活 raw KV 在已有 CPU BF16 容差 atol/rtol=0.02 下对齐。首次计数 initial_joint=1、独立 prefix=0；Graph 通过复用相同固定槽地址覆盖 warmup/capture/replay。双 rank CPU collective 故障注入覆盖 stable shapes 开/关，检查准备失败同步退出且未写 KV。原有缓存命中、prompt 五/六分支、跨块、窗口、RNG 和失败清理回归一并通过。Ruff check/format、compileall 和 git diff --check 通过。

文档构建工具 mint 未安装，未运行站点完整构建；已检查 docs.json 的 JSON、导航条目、frontmatter 和新增链接目标。

最终测试文件补强后，串行重跑新增首次联合与双 rank 故障用例：8 passed、135 deselected，27.23 秒；相同固定槽的 Graph capture/replay 与首次固定执行输出逐位一致。

### PR 主线基线复验（2026-10-29）

将本功能增量移到 main `4adab592c602f89e87717277a3c0b29af667b35f` 后，在代码提交 `7c4e618763` 上用同一环境和完整命令重新执行：**243 passed、4 skipped、1 warning，225.92 秒**。4 个跳过均需真实 CUDA；覆盖包括最终补强的 Graph 逐位回放和双 rank 准备失败用例。Ruff check/format、compileall、docs.json JSON 与 git diff --check 再次通过。后续提交仅整理本设计文档，未改变受测代码。当时真实 GPU 与 480/736 性能仍未验证；后续 forward 专项结果如下。

## 2026-10-29：两卡 SP2 / W6 S3 的真实输入 forward A/B

受测代码提交 `be7f1f7a6c5f58fd80f98578929540c0e5196338`。按用户指定将统一基线的 W21 改为 **W6/S3**，仅使用物理 GPU 0/1、纯 Ulysses SP2；没有运行 SP4 或服务端端到端实验。H100 目标节点实际上报 **NVIDIA L20Z / SM90 / 132 SM**，驱动 580.105.08，PyTorch 2.13.0+cu130、CUDA 13.0。真实 50 层、56 heads、head_dim128 的 BF16 权重；每 rank 28 个 KV heads，weights-free AdaLN 表通过 checkpoint 校验。

### 输入与计时口径

真实样本 `zing-whitebox-video-five-triview-0001`（海岸）。复用生产准备后的真实 condition/reference 张量，文件 SHA256 `2f1f797cf6a49a59d67c305dfd486f7f0d26b97cb15d1f2c0cfec73206ad2cd9`，prefix 11,605 行。匹配原始 source 视频的首帧分别缩放到 832×480、1280×736，经实际 progressive VAE 和 indexed posterior 得到首个 source latent；target 使用相同 seed42/indexed noise。两 rank 的 source/context/noise 哈希一致，两臂共享冻结输入。两档复用同一份原 480 档 condition/reference，不证明 736 独立 conditioning 准备正确性。

A 为独立 condition prefix forward，再 source/target 联合 forward；B 为 prefix/source/target 三分支一次 forward。两臂调用受测源码的相同 fixed-forward planning/execution，共享同一个 KV 槽；host packing、拷贝、编码、权重加载、预热和 Graph capture 不入计时。eager 指固定计划的普通执行，Graph 使用原生 `CudaGraphBackend`，两 rank 实际捕获并回放。每臂预热后做 **11 次 AB/BA 交替**；每次 barrier/synchronize，再以 CUDA event 测完整 forward，按匹配样本取两 rank 最大值后汇总中位数/P95。P95 为 nearest-rank（11 个样本中的最大值）。这是内核预热后的首次 step0 组件耗时，不是冷请求 TTFF。

| 输出档位 | 执行 | 原路径 P50 / P95 (ms) | 联合 P50 / P95 (ms) | P50 耗时下降 |
| --- | --- | --- | --- | --- |
| 832×480 | eager | 1112.86 / 1118.14 | 657.03 / 665.83 | 40.96% |
| 832×480 | graph | 1110.69 / 1114.11 | 674.51 / 681.16 | 39.27% |
| 1280×736 | eager | 1173.94 / 1551.76 | 712.92 / 723.67 | 39.27% |
| 1280×736 | graph | 1172.55 / 1176.85 | 734.37 / 739.93 | 37.37% |

736 eager 原路径有一个 1551.76 ms 样本，已保留，没有剔除。Graph 在本轮没有比 eager 更快，不能由该实验宣称 Graph 本身带来收益。

### 数值与显存

两个输出档位、两个 rank 的 target step0，及全部 **50 层 condition/source K、V（共 800 组张量比较）逐位一致**：max_abs、RMSE、relative L2 均为 0，全部有限。两臂 Graph 输出也逐位一致，且各自与 eager 的输出 SHA256 相同。没有测后续去噪步或最终 RGB。

| 档位 | rank0/1 公共 allocated 常驻 (GiB) | eager A / B allocated 峰值 (GiB，取较大 rank) | 公共 reserved 峰值 (GiB) |
| --- | --- | --- | --- |
| 480 | 50.706 / 50.706 | 52.722 / 51.948 | 56.049 |
| 736 | 52.066 / 52.066 | 54.082 / 53.415 | 57.010 |

两臂的计划和 Graph 在同一进程共存，reserved/常驻包含共享权重、KV、两臂计划与 Graph pool；Graph replay 没有新增 allocator 峰值。此表不能作为分别独立部署的显存对比，也不包括进程外 NCCL/驱动开销。KV 容量仅为首次事务所需的 12,416 / 13,568 行，策略为 W6/S3；不能推出完整 736 长流服务显存足够。

### 实际配置、回退与范围

遵循 PERFORMANCE_DEFAULTS 的 FA4、stable shapes、compiled AdaLN、FA4 metadata reuse、fused current/history RoPE、grouped history RoPE、chunk reuse；`HISTORY_ROPE_PLAN=0`。当前源码允许纯 Ulysses SP Graph，故 `DENOISE_CUDA_GRAPH=1`、`DENOISE_CUDA_GRAPH_SP=1`，本轮实际 capture/replay 成功。SM90 不属于文档中 CC10.3 专用固定 FA4 / current RoPE-KV 已验证硬件，故这两项显式为 0，保留 varlen FA4 与普通当前 RoPE/KV 写入。其余各项未单独采集 kernel 命中计数，不将环境变量为 1 当作逐项加速证据；首次事务也不代表稳态 history reuse 效果。source 编码为单 rank replicated progressive tiling，进程退出后再加载两卡 DiT，其耗时和显存不入表。

未验证：完整 SourceReadySession 服务入口、后续多步、长流/窗口饱和、最终 RGB/质量、Encoder/Decoder 性能、TTFF/E2E/FPS/TPOC/ICL、完整服务峰值显存。用户本轮明确只要求 forward。

### 证据与失败记录

本地结果：`D:/pr464-initial-joint-h100-20261029-r4/{SUMMARY.json,readback/results/rank0.json,readback/results/rank1.json}`，同目录保留完整测试控制脚本、预检、源码 hash 和输入记录。远端根目录同名位于 `/mnt/zing-nvme/`，仅本任务 Pod 已删除，GPU 0/1 显存读回 0 MiB，原始目录保留。

私有 OSS 原始记录：`oss://leap-world-shanghai/world-model/sglang/evals/pr464-initial-joint-h100-20261029-r4/readback.tar.gz`，SHA256 `cc09f90a39d1213097e4e719abdb81a1a109287b7eed1b99302b860d84473496`，260,176 bytes，下载后校验一致。源码压缩包 SHA256 `4e322bf2328f06b30247607d390fb21b0317f0b693080d6243dae6d10cfd8083`。

r2 因测试脚本未初始化项目 SP group，在 source posterior 广播时退出；r3 已完成两档真实 source 编码和双 rank 权重加载，但测试脚本缺少 sampler.device，在首次 forward 前退出。两轮均无有效 A/B；修正仅涉及控制脚本，按约定换新根目录并重新 CPU 冻结。r4 新增实际小模型执行 benchmark 的同一 plans_for，验证预测/KV 发布后才启动 GPU，最终 Pod exit0、两 rank complete=true。

### CI 计数回归修正

首次 joint 的独立 prefix 次数从 1 变为 0、source_first 次数减少 1；更新 prompt switch cache 与 initial blocks 的旧计数断言，并显式检查 initial_joint=1。原预测、全部 KV、RNG、缓存命中和资源释放断言保留。两个完整文件本地复验 **148 passed、1 warning，62.63 秒**；只改测试与文档，GPU 受测 runtime 与 be7f1f7a6c 相同。旧 PR CI 中对应失败来自这些旧计数断言，新提交的 GitHub CI 状态单独核实，不把本地通过当作远端 CI 通过。

## 2026-10-30 主线整合与四卡服务验证

保留 main `450080f8ca` 的 target-grid 优化、unsplit prompt 回退测试和部署证据；本 PR 的首次联合与独立开关保持原职责。受测代码 `aae0f99d85e55ede94552e091d8aa2697c0558bb` 的四套 CPU 回归 **406 passed、4 skipped（CUDA）、1 warning，261.29 秒**。包含开关关闭/开启的 step、drain、固定 Graph 数值及全 KV 对照；CPU 结果不代替真实 CUDA/SP。

按用户指示，四卡实验沿用原 PR 模型和 15 case 中的 `hiker-clay-04-amethyst-boulders`，不切换新的 1027/unsplit 默认模型。**两臂 Block0 cache off、CUDA Graph off**，仅 `SGLANG_ZING_INITIAL_PREFIX_JOINT=0/1` 不同。E1+DiT SP2+D1，W6/S3（与统一 W21 基线不同），source832×480，分别输出832×480和1280×736；每臂一次预热、一次正式、一次独立阶段诊断。正式 n=1 只支持本次观察，不证明稳定分位数收益；阶段诊断不计入正式 FPS。

先前 Graph 开启的对照组预热完成，第二会话在 prompt joint Graph 的 KV 快照克隆处 OOM，未形成有效 A/B；用户随后要求关闭 Graph。该失败属于旧轮次，不作为当前 eager 结果。最终有效实验目录 `D:/pr464-service-ab-h100-20261030-r7`，2026-10-31 已读回并校验归档。每组独立启动 Qwen/DiT，避免复用绑定旧 worker 的 Qwen NCCL 通道。

### 四卡实际结果与限制

480 两臂的预热、正式和独立诊断共六个请求均完成360帧，prompt均在输出第136帧生效。两rank的A：prefix=1、initial_joint=0；B：prefix=0、initial_joint=1；cross_chunk=42、prompt_joint=1，证明独立开关没有关闭其他联合路径。Graph owner未创建。

| 正式指标（各臂n=1） | A关闭 | B开启 |
| --- | ---: | ---: |
| 客户端开始→完整首批RGB TTFF / s | 2.031585 | 3.476349 |
| E2E / s | 18.877477 | 20.592146 |
| 全程 / 首包后 FPS（paced输入） | 19.070 / 21.311 | 17.482 / 20.975 |
| Prompt-switch TTFF / ms | 1645.031 | 1639.078 |

本次正式样本TTFF反而上升71.12%，**尚未证明端到端收益**。独立诊断轮TTFF为2.133836→1.791463s，首个DiT输出host调用按较慢rank为1108.238→838.155ms；诊断数据不替换正式样本，不与CUDA耗时混称。正式首个Decoder append host区间为84.151→782.583ms（含收包/等待，非纯GPU计算），只定位到部分波动，完整归因未完成。

正式请求driver采样峰值（0.5s间隔，MiB）：共享Qwen/Encoder卡39441→39441，DiT rank0 72371→72251、rank1 71089→70869，Decoder16971→16971。Qwen/reference/source共卡不能独占归因；未完整采集各组件allocated/reserved及DiT/Decoder CUDA分项。完整host分项P50/P95/样本数、请求前常驻、ICL、TPOC、source-to-output与原始数据见运行目录REPORT.md/ANALYSIS-480.json。

736在KV池预分配阶段OOM（单rank约79.09/79.18GiB，继续分配260MiB失败），未进入请求，故没有736 A/B；不能用480或历史首次forward数字外推。当前池仍按最大target面积保守预留source/target，实际source固定480；该池规划问题不混入本PR。新默认1027/unsplit模型、完整质量和稳定多次正式收益未验证，PR保留draft。

原始归档：`oss://leap-world-shanghai/world-model/sglang/evals/pr464-service-ab-h100-20261030-r7/readback.tar.gz`，973328bytes，SHA256 `89b0973934a885cc4e8629464b79c30e2f575d9804514ef3fd7f5ea085ced587`，回读校验一致。受测runtime为`aae0f99d85`，后续交付只更新文档。
