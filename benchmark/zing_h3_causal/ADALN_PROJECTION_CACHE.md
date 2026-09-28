# Zing AdaLN projection cache：实验与验收

关联 #135。自 #207 起**默认开启**：单卡 60 s 白盒会话稳态 DiT 514.1 → 492.2 ms/chunk（−4.3%），
同进程 GPU 逐位检查 12,954 次命中全部相等（见 #207）。不代表达到持续 24 FPS。

## 实现边界

- `SGLANG_ZING_ADALN_PROJECTION_CACHE=1`（默认）开启；`0` 走原投影。
- `SGLANG_ZING_ADALN_PROJECTION_CACHE_BYTES=134217728`：每请求、每 rank
  上限 128 MiB，LRU 淘汰，单 entry 超预算直接原算，不减少 timestep / 模态。
- 绑定 `ZingH3KVCache`；会话 clear/close 释放，模型计算阶段异常清空结果；输入校验失败仍由会话 owner 清理。
  禁用开关后的下一次 forward 释放已有结果；不跨请求预热。
- resident 原权重保留，不用普通 H3 sidecar，不拆分原 `unique_t` GEMM。
  所有主层四模态六组参数与 final layer 两组参数保持原返回视图和 token 索引。
- key 使用完整有序 unique-t 批次原始字节、dtype/shape、embedding dtype/shape/stride，
  embedder 参数/频率 buffer 身份与版本、projection 参数身份/版本/存储、权重 generation、
  设备/stream 和算术设置。实际 t 值体现 noise/schedule；prompt/source/reference 像素
  不进入 projection，所以不盲目把它们加入 key。不同批形状绝不共用结果。
- 保留 timestep/SiLU、norm/modulation/gate；这不是 compiled AdaLN 融合。
  有 host 行元数据时用 host 上的 unique-t 建键，不读回设备；否则读回紧凑 timestep。
- 训练/autocast、无版本参数、runtime LoRA wrapper 回退原投影；LoRA wrapper
  即使 disabled/merged 也回退，避免 scalar strength/enable 状态漏入 key。
  checkpoint loader 与公共 tensor/disk 权重刷新回调均提升 generation；外部 `.data` 原地修改必须提升 generation
  并清理 session。现有 KV 契约本就拒绝带旧 history 的权重更新。
- `cache.adaln_projection_cache.stats()` 提供 hits/misses/bytes/evictions/bypasses、
  key_host_ns/build_host_ns。后两项是 CPU 调用时间，**不是 CUDA 同步 kernel 时间**。

## 冻结门槛（看 GPU 结果前确定）

基线是此 PR 的同一 SHA、开关 0，候选仅改开关 1；基于 main `5b1b93db8c`。
选择当前默认六图 `zing-whitebox-video-five-triview-0001`，真实 source，
使用当前 typed media / 实际 source 首帧派生契约。冻结 checkpoint、四步 schedule、
832×480、音频、SP、窗口/sink/rebase、原始输入与显式噪声 hash。
旧 a34 r8 的完整条件结果只能用于寻找 scope，不能替代新 main 验收。
当前 source 原片仅 120 帧/5 秒，r8 的 30 秒是重复六次；它不满足非循环长流门槛。
已由协调任务核验同 parent_id 的 15 秒非循环父片，可在独立新 case/revision 中用于 ≥10 秒质量扩展；不得替换默认 5 秒基线。≥60 秒真实长流仍无已验收素材。

精确标准：同 SP 下紧凑参数、每步 visual/audio latent、最终输出、KV 内容和位置
全部 `torch.equal` 且 finite / shape / dtype 正确。不得改用近似阈值。
缓存持久 storage 不超过 128 MiB/rank/session；CPU 测试不替代 CUDA。

主性能指标：无 profiler 同业务服务 wall time（API、DiT 与 source/decode 分列）。
至少两轮 A→B/B→A，每轮每配置完整预热后至少五个样本，保留全部样本。
单项净收益的 paired speedup 95% bootstrap CI 下界须大于 1；不通过则保持默认关闭，
报告无结论/回归，不挑最好样本。冷 session、请求内热 cache、clear 后分别报告。
独立短 trace 证明热命中跳过 projection GEMM，记录采集扰动与查找/索引开销。

先 SP1 最小正确性，再目标 SP2/SP4，单项通过后才与 #136 已验收路径组合。
无 LoRA 缓存加速承诺；LoRA 路径须验证安全回退。prefix/source/active/final、首尾块、
多 session、异常、权重/embedder 更新、prompt/source/reference 更新、窗口淘汰/rebase/扩容
均需覆盖；未跑配置明确未验收。

至少 10 秒有效视频 native / 原 SG / 候选比较 source 动作、姿态、运动、参考外观和连续性；
没有独立 action 输入。video/video_audio、离线/持续 RGB 分开。
持续 24 FPS 系统门槛另列：≥60 秒真实 paced source 且≥3 完整窗口周转，取更长者，
无循环/补帧/丢帧，交付帧数/墙钟≥24 且积压不增长，报告 TTFF、source-to-output、
chunk p50/p95、超预算比例、逐 rank 显存、GPU 秒。未测指标写未测。

## 同请求复现

先把已验收的六图请求和显式噪声快照冻结到独立 run；不能用旧五图变体或纯文本替代。
以下是离线 harness 的 A/B 启动方式（路径由该 run 提供，不能代替持续输入验收）：

```bash
for enabled in 0 1 1 0; do
  SGLANG_ZING_ADALN_PROJECTION_CACHE="$enabled" \
  SGLANG_ZING_ADALN_PROJECTION_CACHE_BYTES=134217728 \
  PYTHONPATH=python python benchmark/zing_h3_causal/validate_real_checkpoint.py sglang \
    --request "$FROZEN_REQUEST" --model "$FROZEN_MODEL" --sglang-root "$PWD" \
    --output "$RUN_ROOT/$(date +%s)-cache-$enabled" --sp-size 1 \
    --solver few_step --sigma-points 4 --denoising-step-list 1000 750 500 250 \
    --warmup-runs 1 --measured-runs 5 --save-all-raw --collect-rank-memory
done
```

命令须追加该冻结 run 的 decoder/window 配置；不能把 harness 默认配置冒充业务基线。
同 seed 不等于已核验噪声相同，必须比较实际快照。

诊断 worker 在模型加载后，用 `adaln_projection_probe.ProjectionProbe(model, rank_path)`
包住**一个缓存关闭的 session**；文件使用独占创建。它读回 embedding/output hash，
有同步扰动，不能用于计时。然后运行：

```bash
python benchmark/zing_h3_causal/adaln_projection_probe.py "$RANK_LEDGER"
```

输出按阶段的调用数/唯一 key/潜在命中数、无淘汰紧凑预算、同 key 输出冲突。
探针不包含完整权重 hash，运行前的资产与代码 manifest 必须独立冻结。

## 当前证据与未完成项

- 本机独立 CPU 缓存/探针 13 项通过；复用已安装依赖环境，小模型每 forward / KV exact 与 reset 集成测试 4 项通过（包含实际 hit 断言）；开启缓存后既有模型/window/SP 布局/失败清理回归 103 项通过。
- 已读 r8 rank0 原始 ledger：SP1/2/4/8 各含 prefix 2、source 256、active 1024、
  final 256 个 forward；这是诊断进程总数，不能当单请求次数。旧 ledger 没有 t 值，
  因而无法从旧 ledger 得到实际 t/key 分布或复用率。
- GPU 排第三，前序 #136 → #163 已交接。单卡组件 replay 已提交，代码冻结在
  `8cbac2c479fec68734a221af299defd66bc87b64`；已成功完成并释放 GPU。
- 原权重 projection replay exact 已通过；真实业务逐步 latent/KV exact、冷/热 E2E A/B、媒体质量、持续输入、组合与完整生命周期验收未完成。
- 历史 0.95–2.33% 是累计 CUDA kernel 占比，不是墙钟收益预测。

媒体与 trace 仅保存到内部访问的工件位置；不提交运行媒体、大 JSON 或一次性云脚本。

## 单卡投影 replay

`benchmark_adaln_projection.py` 只接受单张 SM103，加载 resident 原 time embedder 与全部
50 层/final 原投影。输入是完整有序 unique-t JSON 批次列表；`--batches-origin` 必须明确
为 `captured` 或 `schedule-derived`。派生 schedule 的组件诊断不能称真实请求采集。
运行会保留每个被加载权重的 hash、输入 hash、exact 检查、A/B/B/A 全部样本、冷 cache
样本与预算统计；`--profile` 的热路径 trace 单独采集，不参与计时。
此脚本保留 timestep/SiLU 开销，但不包含 token 索引、norm/modulation、KV 或媒体，因此
不能用于 E2E FPS 或最终业务判断。组件结果如下，尚未覆盖业务路径。

## 离线 shape 线索（非业务验收）

已流式检查一份本地历史 r5 SP2/rank0 trace，SHA256
`ebf6ca35c0de5467f064143237e2f6368abbbd8199ed8a40a6b3376721b527ff`。
它缺 VLM 条件，仅作为算子形状线索。该 trace 的 3 次 source、12 次 active、3 次 final
共记录 918 次 projection linear：主层分别 150/600/150 次，final 分别 3/12/3 次。
所有输入 `[1,2688]`，主层权重 `[129024,2688]`，final 权重 `[10752,2688]`，均 BF16。
该 trace 不含 prefix，也没有实际 t 值，不能给出真实 key 命中率或完整请求调用次数。

以源码四步 grid 和默认 video shift=5 / audio shift=3 推导的紧凑预算如下，
这是 schedule 推导，尚待真实业务 probe 核验：

| 路径 | 完整 unique-t 批次的行数总和 | 全 50 层及 final 紧凑字节数 |
| --- | ---: | ---: |
| video，clean + 四步 | 5 | 64,619,520 |
| video_audio，clean + 四步 | 8 | 103,391,232 |

每个 unique-t 行是 `(50×24+2)×5376×2 = 12,923,904` 字节。
video_audio 的两个 t 必须留在同一批 GEMM，不能为复用而拆开。128 MiB 预算可容纳
上述任一单一模式的理论集合；混合模式、额外 schedule 或特殊批形状仍按 LRU 淘汰。
这些字节数不含 allocator 保留池、resident 权重、token 展开临时量与 KV；它们分别测量。

## 当前六图请求的形状证据

复用 #163 的 SP1、Graph 关闭短 trace（作业 `dlc1f8plaigxn38g`），
7 份 trace 的主层 AdaLN `aten::linear` 共 6,650 次：active 4,400、source 1,100、
final 阶段 1,100、prefix 50。输入均 `[1,2688]`、权重 `[129024,2688]`、
bias `[129024]`，对应 `addmm` 记录一致。这里的 final 是 forward 阶段，
不是 final output projection；本摘要未统计后者。

分析来源 `zing163-20260924-r3/cpu-analyze-r1/launch-gaps.json` 的 SHA256：
`e1cebd24509921325765000a4ec233caa6be7ae2048e0b01def4f7777d24153c`。
这是实际六图请求的形状/调用证据，支持 video 单行 replay 的形状选择；
未记录实际 t 值和内容 hash，不能据此计算唯一 key 或真实命中率，
也不覆盖 video_audio 双行形状或本缓存候选的正确性/性能。

## 单卡组件实测（不是端到端收益）

作业 `dlc2ou9l671kjkq7` 于 2026-09-24 08:40:20 UTC 成功结束，worker exit 0，
无本任务排队/运行 GPU。源码为上述冻结 8cb，GPU UUID
`646b29ad-7af0-c36f-3971-e02609f57bce`，SM103、PyTorch 可用显存 274113 MiB，
驱动 580.105.08、torch 2.13.0+cu130、CUDA 13.0。两模式独立进程，原 50 层及 final
resident 权重，每模式记录 106 个权重 tensor 的 hash。输入均为 schedule-derived。

| 模式 | 基线均值 ms | 热缓存均值 ms | 冷缓存均值 ms | 局部均值比 | paired bootstrap 95% CI | 缓存 MiB |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| video | 25.580 | 3.785 | 26.117 | 6.758 | [6.634, 6.867] | 61.626 |
| video_audio | 25.804 | 4.126 | 26.307 | 6.254 | [5.739, 6.720] | 98.602 |

每个样本包含整个 5 批 schedule 的 timestep/SiLU、投影或缓存查找；
A/B/B/A 每组预热一次、计时五次，另测五次冷缓存。CI 来自顺序组件样本的
10,000 次配对重采样（seed 135），不是独立业务请求的置信区间。
两模式分别 510 次 miss/hit tuple 比较全部 `torch.equal`、finite、shape/dtype 一致。
每模式最终 255 个 entry，无淘汰/回退；累计 hit 3315、miss 2040 包含 correctness、
warmup、cold 和 profiler，不能作为业务命中率。冷缓存均值比基线增加约 2.1% / 2.0%。

独立热 trace 各保留 10 次 time-embedder linear，目标主层/final projection linear
均为 0，CUDA kernel events 分别 45 / 51。这证明本 replay 的热命中跳过目标 GEMM，
不包含后续 token 索引、norm/modulation/gate、KV 或媒体。
allocated 约 34.91 / 34.95 GB（十进制），含 resident 组件权重；报告里的内存峰值与
计数器包含 correctness/profiler 阶段，不能当无 profiler 的请求显存增量。

0GPU 导出作业 `dlc88vz7uhj1z9be` 成功，内部工件：
`oss://leap-worldmodel-thailand-single-az/world-model/evals/zing135-20260924-r2/evidence.tar.gz`，
SHA256 `8e346c81a7eacb681f0fa57e8aaf4124c5d4e340e801f2aa27cf3d7e2e9ddf4e`。
包含全部计时样本、权重/输入 hash、依赖、两份原始 hot trace 与分析摘要。
这些结果支持进入实际业务 exact/A-B 阶段，尚不能决定启用默认开关或声称 24 FPS。

## 有界业务正确性诊断合同（首次采集已超时）

真实请求 exact 阶段使用既有 `SGLANG_ZING_DIAGNOSTICS_DIR` 与
`SGLANG_ZING_DIAGNOSTICS_ALL_CACHE_LAYERS=1`，保留实际 conditioning、initial noise、
scheduler noise、每个 forward 输入/输出、各层 KV/位置与最终 latent。两条路径的实际
噪声 tensor 和从 `seeded` 事件开始的 RNG 状态须一致；conditioning 之前的独立 worker
启动 RNG 不属于请求合同，但所有 conditioning tensor 仍须 exact。同 seed 本身不算证据。

新增可选 `SGLANG_ZING_DIAGNOSTICS_TENSOR_STORE=1` 将同一请求中重复的原始 tensor
按 dtype/shape/内容存成独立 blob，事件仍引用每一次出现。每条路径各自建 store，
不跨 baseline/candidate 共用。`SGLANG_ZING_DIAGNOSTICS_TENSOR_STORE_BYTES` 默认
192 GiB/请求，超限明确失败，不截断 KV 或跳过事件。它仍会同步复制/hash，每次 GPU
诊断必须另设时间上限；不能参与计时。inline 默认存储保持可用。

用 `compare_adaln_diagnostics.py LEFT_RANK_DIR RIGHT_RANK_DIR` 在零 GPU CPU 作业中
核对每个事件、原布局 metadata、RNG 与 blob 完整性，然后对两路径各自保存的 raw tensor
做 shape/dtype/finite/`torch.equal`。重复 blob pair 复用已通过的精确比较，报告全部引用数
和独立比较数。此工具本身不证明捕获完整：业务验收还必须核验两个运行成功、
`initial_noise` / `final_latents` 存在，所有 forward 均有全层 cache 记录，最终媒体一致。

首次窗口配置为单卡诊断开关 0/1，各一次真实 5 秒六图请求（不计时），每路径最多 192 GiB，
两路径合计最多 384 GiB raw blob，外加小型事件文件；单次进程上限 20 分钟，作业上限
45 分钟。释放 GPU 后以 CPU 比较为准入；通过后另排无 profiler A/B 窗口，所有诊断环境
变量关闭。parent15 质量扩展和真实 paced 长流分别验收，不混入此诊断。

业务采集固定 `128b38087b515319c8432b9b36e579dc0cb7512f`。CPU 比较合同在
2026-09-24 09:12:12 UTC、GPU 运行中但尚未查看诊断张量结果前，以
`72a6b3c0f4770214ba37ad3f7a0f634303185bc1` 修订了上述 seed 边界；保留原比较器、
差异和 7 项 CPU 测试。原因是 worker 启动 RNG 与请求实际噪声无关，不能误判成差异。
不忽略任何 conditioning 张量、seed 后 RNG 或数值差异，不根据测量结果放宽门槛。
新比较器 SHA256 `5c4b39369d4817965084002a61743f0896207fbf7647df6d0033a20e7bc001e1`。

## 业务采集超时与 CPU 采集提效验证

`dlckgmjgqwjaq2ds` 在基线路径的 20 分钟进程上限退出（124），该上限包含约 6.5 分钟
模型加载；GPU 已释放，候选没有启动。最后完整事件是 `block9.source/forward_output`，
101 个事件含 26 次 forward input/output、25 次 cache、16 次 scheduler.video 和 4 次
block_complete；缺少 final_latents，不能验收业务 exact，也不是缓存数值失败。

零 GPU 归因作业 `dlc1of8hzenutmss`：完整记录累计处理约 609.66 GB 逻辑张量，
去重后最后完整记录占 36.19 GB。25 个 forward_output 文件写完至 cache 文件写完的
间隔合计 716.16 秒、中位 27.78 秒，包含复制等待、hash、去重和写盘；原采集没有
分项计时，不能将其拆成 GPU、D2H 或 hash 时间。原回执 SHA256：
`3bddc37aa7c4804018561d9c8959accd78ae214ce4d182c22d90af546d26ce66`。

后续仅 CPU 验证采集提效：SHA256 直接读取连续 NumPy buffer，避免额外 bytes 副本，
hash 值不变。可选 `SGLANG_ZING_DIAGNOSTICS_TENSOR_STORE_WORKERS=1..8`（默认 1）
只并行 CPU hash；blob 写入和预算检查串行锁保护。CUDA→CPU 复制仍由原调用线程及
原 stream 执行，最多暂存 workers 个 CPU tensor；每事件等待所有结果，异常也清理
线程池。每次出现的 tensor 都仍完整复制/hash，不根据 identity/version 跳过观察。

新事件提供 `capture_costs`：同步复制的主机耗时、worker hash/IO 耗时和主机 wall time。
worker 耗时可重叠，不能相加推导墙钟；复制计时含 CUDA 等待，不能当纯 D2H kernel
时间。CPU 验证不覆盖 CUDA copy 或完整业务吞吐，未据此自动重提 GPU。


零 GPU 完整 prefix 对照 `dlc1h7n9qi9mh8ek` 已 Succeeded / exit 0。
读取实际全 50 层 prefix KV（17,556,159,504 bytes），分别在独立 store 中采集三次：

| CPU 采集器 | 首次写入 | 第二次完整观察 | 第三次完整观察 |
| --- | ---: | ---: | ---: |
| 原 `128b380` | 46.650 s | 17.410 s | 16.890 s |
| 新 `c4b8719`，8 workers | 32.083 s | 2.479 s | 2.487 s |

315 次 tensor 出现、101 对独立 raw blob、3 次 RNG 快照均通过 finite / exact。
这只是单份真实 prefix 的 CPU 复制/hash/存储对照，CPU 输入的 `.cpu()` 可为无操作，
不覆盖 D2H、增长中的完整历史 KV、GPU forward 或业务 FPS。三次样本也不足以给出
稳定性能分布。报告 SHA256：
`d365b586a2dd8e4b5259ddd17dd7f97c1ce652324a53286b8148592697410533`，
内部对象 `oss://leap-worldmodel-thailand-single-az/world-model/evals/zing135-20260924-r3/cpu-capture-bench-r5.json`。

同时导出部分 baseline 的 26 次真实 forward 元数据（完整 packed 长度、dtype、phase、
seq_lens 和 unique_t），供 #136 使用；prefix visual rows=5040，packed rows=12246，
不能只用 visual shape 构造所有层输入，也不能由输入 latent dtype 推定 FC 输入 dtype。
元数据 SHA256 `c43237982e5228bdc7241b749b8e9b2276bedd83a11cb954bb64a1c8ed9eb943`。

下一 GPU 窗口尚未申请：可用预算必须包含约 6.5 分钟模型加载、全部 forward、每次同步
复制、增长 KV 的 hash/写盘和最终媒体。目前 CPU 对照仅覆盖其中一部分，尚无足够证据
保证完整双路径能在原 45 分钟上限内完成；不得把 2.48 秒乘事件数当完整预算。
新 collector 的 CUDA 路径和业务完整性仍待有界验证，PR 保持 draft、功能默认关闭。


## 完整请求静态预算（未执行）

实际 step1000 chunk 配置为 `[1,2,2]`、不 split first；冻结 ChunkPlan 对实际
36 latent frames 输出 22 chunks。四步、guidance1、video-only 对应 133 forward，
133 全层 cache、88 scheduler.video、22 block_complete；含准备和 final 共 514 事件。
以真实 prefix/source/target 元数据推导，并逐项复现已有 101 事件的名称、phase 和
609,655,332,040 逻辑字节。全请求累计需观察约 **7.02 TB**，含 layer0/layer_last
重复引用；每次出现仍复制/hash。去重 raw payload 上界约 **120.15 GiB/path**，
不含序列化头、事件/RNG索引和最终媒体；192 GiB/path 硬限保持不变。

最大有效 KV 为 56,696 行、75.70 GiB；但 raw K/V 与 rotated K 的存储容量按
12246→24492→48984→97968 行翻倍，三份 store 本身达 **196.20 GiB**，尚未包含
模型/外围权重、activation、workspace、扩容临时内存与 allocator。不能仅凭有效 KV
大小证明实测 267.69 GiB GPU 能承载完整请求。collector 八路 CPU pending 张量
上界约 1.31 GiB，亦不等于整个进程 RSS。

因此不直接重跑完整 baseline/candidate。待协调的 collector 校准最多 1GPU/8CPU/
32GiB、进程10分钟/作业15分钟，不加载模型；以相同布局的确定性合成最大有效 KV
执行 prefix冷、最大KV冷/热三次完整捕获，记录复制/hash/IO/wall及RSS/CUDA峰值，
释放GPU后CPU核验。该方案尚未提交；不能替代完整模型显存验证、业务exact或吞吐。


## KV 同时驻留审计：当前完整离线配置不适合直接重跑

零GPU使用实际cache类、50层和真实行数，将heads×dim缩至1并按7168恢复字节，
验证raw K、V、rotated K共150份独立storage，chunk视图复用raw storage。133forward
结束时各层capacity一致；block35.source触发97968行档位。窗口只compact不收缩
capacity。store稳态196.202GiB，逐层扩容的对象存活峰值196.856GiB，后者不包括
CUDA allocator和其他模型张量。诊断仅持有CPU副本，没有延长跨forward的GPU生命周期。

本任务原始加载日志显示四个常驻模块VRAM合计约128.16GiB（日志GB实际按1024³），
加上store即约324.36GiB，已超过设备267.69GiB。**原作业实际终止原因仍是timeout，
并未测到OOM**；这里是源码容量与已实测驻留量结合的不可行性判断。

#163成功120帧采用预缩放832×480的progressive source，source/target均390行/latent，
prefix11756；本次offline source880行/latent、target390、prefix12246。相同22chunks
下前者capacity停在47024，后者越过48984而翻倍；两版KV实现仅clear中的AdaLN清理
有差异，不能把成功baseline当本次offline几何的fit证据。

暂停collector CUDA校准。最小前置修正候选是保持全部输入/窗口/常驻/SP1，按有限
请求已打包的真实chunk行数和window推导必要capacity上限（本case56696），限制
几何扩容的过量预留；不裁剪KV，超出验证上限明确失败。尚未实施，应单独review并
做生命周期/窗口/尾块/在线fallback回归。理论常驻+三store约241.71GiB，仍须验证
瞬态余量，不能作为已通过的GPU fit或业务exact。


## 生产 transport revision 准备（不覆盖原文件请求）

核对 current main `02101643277d9da9056bd552943afed7bf215572`：
`SOURCE_PREVIEW_ACCEPTANCE.md` 既定生产验收是原 source bilinear 缩放到832×480
RGB，再无损FFV1；`CONTINUOUS_SOURCE_VALIDATION.md` 明确它是独立streaming
variant。原文件/native入口独立bucket source也符合合同，不能称880行路径为bug。
因此暂停KV修正，改为独立transport-r6诊断准备，保留原失败与预算历史。

新的CPU准备核验全部120帧、首帧、六refs与prompt，并回解码FFV1验证完整RGB hash；
随后实际CPU parser/processor校验source/target390行、prefix11756及预计36latent
的几何。准备尚未成功取得回执，不能声称新输入冻结完成。该几何下静态预算为
514事件、约5.109TB逻辑观察、raw payload上界72.377GiB/path、KV store94.176GiB；
常驻+store约222.336GiB。时间和新collector完整GPU峰值仍待实测。

唯一拟议GPU窗口为单卡、cache0/1各一次同一新revision正确性捕获，35分钟/path、
75分钟/job硬上限，192GiB/path存储；这些是待协调的新预算建议，未提交GPU。
只有CPU预处理/素材/代码全部冻结通过后才具备窗口申请条件。全层KV/实际noise/RNG
与最终输出仍exact，不把新revision称作旧原始文件请求通过，也不产出吞吐结论。


补充 native967 原始 source 的实测CPU回执：`dlckhvhmyf005l7f` 成功，报告SHA256
`9ab2a027de5cbdca9ca5832c6f57bbd627397ff7105c20194d2c59a19704b95d` 已本地复核。
原5秒source由 `[3,120,720,1280]` 归一化为 `[3,120,704,1280]`（880行/latent），
归一化RGB SHA256 `5b90d30329a9c785bb140c67edc668b760299afc11f2a7b9bdb1288ee333961e`。
这进一步确认旧文件路径符合native预处理；报告不含VAE/Qwen/完整模型执行，且native
依赖版本门槛仍未通过。15秒parent原始/transport的预计latent长度均为107，不能复用
本任务5秒36latent预算。该回执不替代本任务transport-r6待完成的输入冻结。


## 2026-09-28 transport-r7 完整 A/B：严格一致性失败

原 CPFS 临时输入清理后，从保留的 OSS 原始资产重新构建独立 r7；旧请求重建
SHA256 仍为 `7862f437e5e6586b8d5c0e24bcc815b6d99bd06d45ca00599f2d9c989ce7283e`。
新请求 SHA256 `2e15bf3c722aca20cec46f7d9c8b5f6556800323bcdd4224ab4e3ced21dd1277`，
120 帧 RGB 和 FFV1 回解码一致，source/target 各390行/latent，prefix11756。
CPU 素材/几何/decoder 预检及9项诊断采集测试通过。

源码 `c4b871978097e796e4bffe3c9b86ac1bf491a0dd`，单卡作业
`dlc11x93wqthgdce` 成功退出；缓存关闭/开启两组均完成514事件：133次前向输入、
133次前向输出、133份全50层KV、88次scheduler、22个完成块及最终latent。
这证明完整采集跑完，**不代表业务exact通过**。

严格比较在event1的conditioning失败：Picture2–6参考图的VAE latent不同，
每图约5.2–5.4%元素不等，最大绝对差最高0.1275634765625。
source_visual、Picture1、文本条件的记录hash相同；实际初始noise、全部88次
采样noise及seeded之后RNG记录hash相同。最终帧与latent均finite但不exact：
帧MAE约0.05989054，latent MAE约0.31574574。差异已发生在AdaLN缓存介入之前，
不能据此归因缓存，也不能放宽或忽略conditioning门槛。

源码为每次visual posterior调用创建固定CPU generator(seed=42)；运行配置采用
VAE performance模式。编码算法的非确定性是待验证假设，尚无根因结论。
性能、SP2/SP4及完整GPU生命周期依赖阶段未提交；缓存仍默认关闭，PR保持草稿。

[查看真实A/B同步视频及失败差异报告](https://review-cdn.loopit.com.cn/world-model/evals/zing135-20260928-r7/failed-exact-export/index.html)
（办公网或公司VPN）。两份120帧视频均完整解码通过，浏览器一键同步播放已验证；
页面明确标为失败样例，不替代native画质或性能验收。差异报告SHA256：
`befa38abb42b1e28c81469ea662e589a3a5a18fc6640b8ef0fcec9262e09480a`。
