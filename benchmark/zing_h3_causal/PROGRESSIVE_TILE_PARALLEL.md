# Progressive source encoder tile parallel（issue #177）

状态：显式并行路径的真实 SP1/2/4 验证已完成，用户于 2026-09-24 确认画质通过。
按用户要求，默认策略现为 `auto`，对已验收的数值/分卡组合自动启用。
自动默认路径的最终提交复验与最新结果以 [PR #178](https://github.com/seedleap/sglang/pull/178)
为准；下列冻结结果不能冒充后续提交的测量。

2026-09-23 用户更新了本轮范围：**仅 SP1/2/4，最多 4 GPU，重点为 encoder
加速比、正确性与画质 A/B**；画质由用户通过一键同步播放页面判断。
原始 issue 的 SP8 和完整服务性能矩阵不属于本轮执行范围，不能记作通过。
为减少 GPU 空等，本轮 encoder 性能采用 `--delivery offline`，10 秒/60 秒
各 1 次预热 + 3 次正式测量；paced 全矩阵另列未运行。画质生成各 SP 的
replicated/parallel 各一条 10 秒视频，由用户判断。
旧 #148 的 clip17 差异及性能结果不用于放行本改动。

设计、数学关系、旧 H3 SP 对照和 `agree`/错误汇合/执行记录的用途见
[设计文档](PROGRESSIVE_TILE_PARALLEL_DESIGN.md)。

## 配置与实际分工

`MINIMAX_H3_PROGRESSIVE_ENCODER_TILING` 默认为 `auto`（不设置变量即可）：

- `auto`：deterministic、BF16、非 stack tiling、实际 `vae_decode` 组为 2/4 卡时
  自动选择 tile-parallel；其他组合保持 replicated，并记录具体回退原因。
- `replicated`：显式关闭，便于回退和 A/B 基线。
- `parallel`：显式实验覆盖；未验收组合的成功运行不代表 exact 或性能达标。

配置在 session 创建时冻结，与旧 `parallel_tiling` 独立。`encode_mode` 和
`encode_dtype` 也在创建时冻结；中途修改会拒绝继续编码，防止已自动启用的
session 被改回未验收的数值策略。改变配置须创建新 session。

`deterministic` 关闭 cuDNN benchmark、启用确定性算法；`performance` 开启
benchmark。它们不改变 BF16 精度、权重或采样步数。当前 performance 的
SP2/SP4 exact 失败，自动选择不会暗中修改此策略，也不会自动启用并行。
SP1、SP8、stack-on、其他精度不继承 SP2/SP4 BF16 的验收结论。

使用 runtime 的 `vae_decode` coordinator：device group 搬运 moments，Gloo
CPU group 核对状态/错误。该组按 TP/SP/PP/CFG 组合，**不跨 DP replica**；
只有 TP=PP=CFG=DP=1 时其 world size 才等于 DiT SP。实际 global ranks、
rank-in-group、world size 与 encoder ranks 记录于 `cache.last_execution`。
`encoder_ranks` 指一份编码的分片数；fallback 时为 1，但组内仍有
`participating_group_ranks` 张卡各自重复编码，`replicated_copies` 明确记录份数，
不能把 fallback 当成只占一张卡。
所有参与者必须持有相同 source、权重、dtype、后端配置；使用同样的 session
创建与 pop 顺序，不能在多个线程并发使用同一组的 collective。

沿用 `split_tiles`、`_local_tile_indices` 与 `_assemble_tiles`。tile 的全局
ID 为 row-major 下标，归属为 `tile_id % world_size`。非 stack cache key
使用全局 ID；stack key 使用稳定的本地 ID 元组，batch slot 对应该元组顺序。
每个 clip 内 layout 冻结，17 帧边界清历史，不重置绝对 latent 索引。
逐 pop 的 RGB 增量仍为 1/8/8，latent 增量仍为 1/2/2，真实水位为 1/9/17。

| 情况 | 行为 |
|---|---|
| `auto` 且为已验收组合 | 默认 tile-parallel，无需额外开关 |
| 显式 `replicated` / 未验收自动组合 | replicated，不增加通信，记录关闭/回退原因 |
| 单 rank / 未初始化 distributed | replicated，明确记录原因 |
| 非 tiling encoder | replicated，不把画面切成新的 tile |
| tile 数少于组内 ranks | 全组 replicated，无 moments gather，无空任务 rank |
| tile 数不可整除 | round-robin，本地结果 pad 到 ceil(N/P)，gather 后丢弃 pad |
| stack_tiling 关闭 | 每个本地 tile 独立增量 forward |
| stack_tiling 开启 | 只 batch 本地 tile；batch 改变可能影响算子数值，exact 失败不得放行 |

每次成功 pop 的 `last_execution` 只存标量/列表，不持有 tensor：实际 tile IDs、
输入输出 shape/dtype、水位、逐层 temporal positions、本次 pending cache bytes、
逻辑 gather 次数、每 rank padded 输入及汇总输出字节。字节数是张量 payload，
不是 NCCL 链路实际传输量。`state_usage()` 是实际存储字节，不能替代 CUDA
allocated/reserved。profiler 标记 `H3ProgressiveTile(s)/...` 和
`H3ProgressiveTileGather` 用于逐 rank 的 CPU/CUDA 归因。

## 失败与生命周期

本地 forward、gather 前 tensor 分配、拼接、返回 slice 分配均在状态汇合后
才前进。失败不提交 pending history。每次 pop 检查 session 顺序、水位和
生命周期状态，encode 检查几何与输出格式，防止错误 rank 进入不同 collective。
Python 层失败对所有参与者抛错；调用方应取消整个分布式请求并 dispose。
未进入 pop 的取消/退出 rank 由 30 秒 Gloo monitored barrier 检出。
进程丢失、设备异步错误、collective 中断仍依赖进程组 timeout 与 launcher
终止全部 worker；不能捕获异常后把损坏进程组当作健康服务继续用。

CPU 单元测试证明的边界不等于 NCCL/真实服务故障恢复通过。完整服务取消、
慢消费者和异步 CUDA tensor 生命周期仍需实际多卡验收。

## 可复现命令

CPU/Gloo 回归（macOS 可加 `GLOO_SOCKET_IFNAME=lo0`；Linux 按实际接口）：

```bash
OMP_NUM_THREADS=1 PYTHONPATH=python python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_zing_h3_progressive_kernel.py \
  python/sglang/multimodal_gen/test/unit/test_zing_h3_progressive_tiles.py \
  python/sglang/multimodal_gen/test/unit/test_zing_h3_streaming_codecs.py \
  benchmark/zing_h3_causal/test_progressive_encoder.py
```

真实 GPU 使用 clean 最终 PR commit。`MODEL` 为已核验权重包，`RGB` 为冻结
source 832×480 RGB24，至少 1440 帧；不在 worker 临时生成/缩放/循环素材。
`RGB_SHA256` 和 `SOURCE_SHA` 由准备阶段冻结。每组独立进程、独立输出目录：

```bash
PYTHONPATH=python torchrun --standalone --nproc_per_node=4 \
  benchmark/zing_h3_causal/validate_progressive_tiles.py \
  --model "$MODEL" --rgb "$RGB" --rgb-sha256 "$RGB_SHA256" \
  --source-sha "$SOURCE_SHA" --stack-tiling off --parallel-mode auto \
  --sp1-reference "$RESULTS/sp1-reference" --phase correctness \
  --output "$RESULTS/sp4-stackoff-correctness"
```

使用 `--sp1-reference` 时必须先运行 ranks=1 生成完整基准，再运行 ranks=2/4。
该目录包含私有 tensor，不作为公开媒体发布；跨 SP 检查结果独立列出，不能沿用
此前仅在同 SP 内对照的通过结论。未传入该参数时仅检查同 SP 对照。

本轮授权范围为 ranks=1/2/4、stack=off；`--parallel-mode auto` 实际清除环境变量，
验证无需显式设置时的默认选择。`--parallel-mode parallel` 保留显式路径验证。
更广的 ranks=8、stack=on、profile/paced 另行执行，不混作本轮通过项。
correctness 严格比较 replicated 与 parallel 的 moments、indexed posterior；
覆盖 EOS 1..34 和 120 帧、单帧/1-2-2/17/固定随机分包，以及未来帧、交错、
reset 和 dispose。无 tolerance 或 report-only 开关。失败保存 exact-failure.pt。
实际少 tile 的 SP8 行单列 fallback，不能视为成功加速。

计时对比 clip17/replicated/parallel，在 240/1440 帧及 offline/paced 下各
预热一次、测量三次。`pop_seconds` 与 `encode_host_seconds` 是包含 GPU
同步的 wall time（含归并/控制通信），不是纯 kernel 时间。各 rank 保存完整
rows、CUDA allocated/reserved、状态 bytes 与 temporal positions；按输出
1 latent/2 latents 分组分析 p50/p95。profile 另外运行，不混入三次计时。

服务 A/B 需另用现有服务验证入口，严格冻结
`zing-whitebox-video-five-triview-0001` 的 **source + Picture1 全局图 +
Picture2–6 三视图 + source 第0帧 prompt_video_frame**，四步、seed42、
video-only；禁止替换为 target-frame0 或旧五图变体。encoder-only 工具的
完成状态明确不是 service acceptance。

## 冻结 GPU 结果与验收边界

冻结代码 `2979374e436943e320740e199a62ff346a4569fa`；832×480/24 FPS、真实六图
业务输入、source 首帧、BF16、四步/seed42/video-only。120 帧 source 循环成
10/60 秒输入，每 5 秒存在循环边界。每组 1 次预热 + 3 次正式 offline 测量。

| deterministic | 10 秒输入加速比 | 60 秒输入加速比 | Exact / encoder 门槛 |
|---|---:|---:|---|
| SP1 | 0.998× | 1.000× | exact 通过；单卡回退 |
| SP2 | 1.911× | 1.925× | 均通过 |
| SP4 | 3.507× | 3.477× | 均通过 |

SP1/2/4 每 rank 均通过 140 项 EOS/分包用例、完整 moments/indexed posterior exact、
未来帧隔离与交错/reset/输出所有权/dispose 检查。SP2/4 编码时间改善超过10%，
首 latent 和输出间隔 p95 均无超过5%的退化。12 个 tile 的所有正式测量均无重复
无遗漏；10/60 秒状态峰值相同，释放后归零，CUDA 分配/预留另列。

performance 的 SP2/SP4 exact 失败保留，未接受数值容差；其 replicated 基线跨卡
本身已有差异，parallel gather 后各卡输出一致。两种策略使用了不同的物理卡集，
只能比较各自策略内的同卡加速比，不能据此量化跨策略的绝对速度差。

| 条目 | 状态 / 边界 |
|---|---|
| deterministic SP1/2/4 数值、offline encoder 性能与10秒服务输出 | 已通过；新默认提交复验见 PR |
| 六图 source 业务画质 A/B | 用户于2026-09-24确认通过 |
| 独立逐 rank trace、通信 CUDA 归因、paced 矩阵 | 本轮未运行；执行计数器不代替 trace |
| SP8、GPU stack-on/少 tile/非 tiling fallback、异常注入 | 本轮未运行；有 CPU 回归，单卡 fallback 已做 GPU 验证 |
| 完整10/60秒端到端 FPS/p95、音频 | 本轮未运行；encoder 加速不代表完整服务性能达标 |
| 默认启用 | 仅已验收组合自动选择；显式 replicated 可回退 |

[一键同步 A/B 与原始证据](https://review-cdn.loopit.com.cn/reports/zing-progressive-tiles/20260923-r1/index.html)
仅办公网/公司 VPN 可访问，包含数值策略与 SP 选择、计时/输入 provenance、逐 rank JSON。
未运行项不能记为原始 issue 全范围完成；本轮按用户缩减范围及默认启用要求交付。

## GPU 执行资源约束

已只读核验 shengdong / ap-southeast-7 / workspace 384，以及现有 quota
`quota1betfyi1ilu`。不接管或停止其他独立实验作业。

本轮使用单节点最多 4 GPU，矩阵串行执行，GPU 作业只挂现有 CPFS；准备、
校验素材、结果汇总使用零 GPU CPU 作业。预计先给最多 4 小时自动超时，
该值是成本上限而非完成时间保证。SP1/2/4 复用同一物理节点的卡子集，
所有输入与代码冻结后才提交。**用户已授权本轮最多 4 GPU，SP8 不执行。**
此方案不增加 quota，不授权停止他人作业或自动合并 PR。

## 本地执行记录（2026-09-23）

- 基线 main：`a34cdbef61f8ca6c73edbf1b05b8cac6711f9f1f`。
- infra-skills main：`f74c315adb4b71fced9c22ae79bd4e58f83141cd`；六目录已核对。
- macOS / Python 3.11 / PyTorch 2.14.0 / CPU Gloo，不是 CUDA 验收环境。
- 原 progressive kernel、streaming codecs、validator：566 项通过。
- 增加模式冻结/非 tiling 回退、布局冻结、严格字节 gate 后，kernel + tile gate 46 项通过（含重复的原 kernel 测试）。
- 新增 14 项真实多进程 Gloo/CNN 测试全部通过：SP1/2/4/8、9 tiles 的不可整除分工、4 tiles 对 8 ranks 的 fallback、stack 开/关、单 rank kernel/dtype 失败和 reset/dispose 分歧。
- 全部独立测试合计 583 项。未将 CPU exact 成功外推为 GPU exact 成功。
- 初次 macOS fork 测试在 Gloo hostname 自动探测处崩溃；设置 loopback 接口并将 CPU 线程数固定为 1 后重跑 14 项通过。测试没有跳过此失败或放宽数值门槛。

### 冻结源码校验

`--source-manifest` 在 CUDA 初始化前与 `--source-sha` 的完整 tracked 路径和 Git
blob ID 核对。空清单、额外/缺失/修改文件即使重写 SHA256 也会失败。归档 worker
须同时取得冻结提交的 Git commit/tree 对象；仅提供 JSON 清单不足以证明源码。
清单覆盖每个 tracked 相对路径，符号链接的哈希覆盖链接文本。不能在已修改的
worker 上重建清单并冒充冻结版本；旧的自声明清单证据不能代替该校验。
