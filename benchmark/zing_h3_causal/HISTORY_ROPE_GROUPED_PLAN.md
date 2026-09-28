# History RoPE grouped runtime：实现与验证记录

此文记录实现范围与独立验证：严格数值、180 个正式样本和 21 份 trace 技术验收已完成；
画质仍待人工验收，原 performance encoder 的重复性问题不由确定性结果覆盖。比较基线为 PR #182 的
`85ad572ab935cd56346302a1efa1ea38a3bf56fb`。前三版探索仍使用原
`e6537184cdcd9de05e04da6201be70446847b5be` 外围基线；最新 main
`5b1b93db8cb7cf6bd3008adc3de1830d8e524bc4` 已经通过本地 merge
`54df8ef3f197a2992294f979f271adc25b1d7d83` 纳入，后续完整验证重新冻结源码、
encoder 配置、source 和六参考图契约，不跨外围基线计算增益。原验证证据见
[History RoPE 验证](HISTORY_ROPE_VALIDATION.md)，不得把原 14–18% 相对 eager
收益写成新候选相对逐片段 fused 的增量收益。

## 借鉴 MinWM 的部分

- 调用方持有 rotated-K 输出，融合算子直接写入；保留 authoritative raw K。
- 同一 forward 的可见片段、有效位置 view、长度和连续 runs 可跨层复用。
  Zing 原有系数缓存、raw K/有效位置签名和 denoise 复用保持不变，不计作新增优化。
- 连续历史可构造视图；多片段历史可批量处理，不能为了单 launch 默认拼接整个 K。
- 小型 launch metadata 从 CPU 已知 shape/stride 推导，不新增 device-to-host 同步。
  本轮不引入跨会话缓存池。

不移植 MinWM 的数学实现：本快照 MinWM fast path 是 SM90 interleaved、FP32
中间算术；本任务实际设备为 CC 10.3，Zing 为 rotate-half、实际 D128/R96、
BF16 逐乘舍入和尾部原始位模式。Q/K 各自有融合 RoPE 调用不等于单 kernel 融合
QK 或 norm。当前 Q norm+RoPE、K norm 双输出以及失效策略分别留给 #160、#161、
#180 / PR #185；跨会话大池留给 #171。

## 独立开关与对照

| 对照 | `SGLANG_ZING_FUSED_HISTORY_ROPE` | `SGLANG_ZING_GROUPED_HISTORY_ROPE` | `SGLANG_ZING_HISTORY_ROPE_PLAN` |
| --- | --- | --- | --- |
| Eager | 0 | 0 | 0 |
| 已有逐片段 fused | 1 | 0 | 0 |
| Grouped 多指针 | 1 | 1 | 0 |
| 连续 raw-K 视图、拼接系数 | 1 | contiguous | 0 |
| Plan 单独消融 | 保持所比较路径相同 | 保持所比较路径相同 | 0 / 1 |

所有开关默认关闭。Grouped 只消费现有签名比较后确定的失效 suffix，按最多 32 个
片段分批，容量按 2 的幂取整；不主动将完整动态几何声明为 constexpr。实测 Triton
3.7.1 的嵌套 tuple 仍会特化值 1 和 divisibility16 类别，顶层 do_not_specialize
及 alignment 开关没有消除这些嵌套特化。必须分别记录
首次编译、不同签名的有限变体与同一签名预热后的复用，不能仅凭源码声称不会重新编译。
连续路线只在同一 storage、相邻 row、兼容 stride 时使用 raw-K view；系数 concat
在计时内执行，其他布局回退 grouped。不会丢掉这项分配、搬运或 fallback 成本。
Plan 仅在当前 forward 内存在，校验 owner/state/position signatures；不延长到其他
会话或请求，不改变任何缓存失效判定。失败写入前撤销 rotated-K publication 的语义保留。

## 验证顺序

1. CPU：布局、fallback、plan 归属和失效；CUDA：真实 head 14/28/56、动态片段、
   非零 row/head offset、stride、partial RoPE、NaN/signed-zero 尾部位模式、alias、
   prefix 更新、窗口淘汰、store growth、跨层复用、异常未发布与 clear。
2. 单卡真实几何：Eager / chunked / grouped / contiguous / pack 原型分列。
   首次调用编译/启动成本单列，20 次预热，保留全部计时样本；CUDA-event 区间、
   同步 host 时间、独立 kernel 时间和 metadata/pack 分配分别报告。
3. 实际 FA4/Ulysses SP1/2/4：四种路径的最终 attention、捕获 KV 与 raw K
   全部逐位一致；额外路线必须进入相同契约后才参与端到端比较。
4. 复用已核验模型与真实输入，在新的 revision 冻结 deterministic 配置。
   同 SP 长窗口逐位检查通过后，才执行无 profiler 10/15 秒正式样本。基本三路按
   E→F→G、G→F→E 对称顺序，各进程完整预热，每时长各五次；如此 F/G 相对顺序为
   F→G→G→F。Plan 与连续路线先独立消融，再冻结最终候选，不挑选/混用重复样本。
5. 报告新候选相对 Eager **及**相对已验收 chunked fused 的增量、所有样本与 spread、
   逐 rank allocated/reserved 峰值。预设 spread ≤10% 保留；长尾不删除，收益未达标
   如实报告，不用 kernel sum 推算 API wall。
6. 独立 profile 采集满窗实际淘汰，各 rank trace 单独保留、核验并在 Perfetto 打开。
   不将 profiler 样本混入正式性能。技术解码、严格数值比较与人工画质判断分列。

## 空间与证据预算

不重新复制权重或下载大型张量到 Mac。第一次准备实测 CPFS free 3666.625 GiB，
第二次 3662.660 GiB；这是当次测量，不表示之后仍有相同空间。每阶段重新核验可用
空间，保守输出峰值预算 200 GiB，开始新阶段要求至少 500 GiB 可用，不把其他作业
预计释放计入。正确性 PT、每次 RGB、trace、压缩及临时副本均计入；达到预算时停止
开始下一阶段并记录原因，不静默减少正确性覆盖或删除其他任务数据。

保存源码 overlay、文件 SHA256、配置/输入/模型回执、完整命令、实际 worker 导入
路径、Job ID、硬件与所有原始计时。旧失败与通过证据保留。原 performance encoder
A/A 不重复性没有由本轮 deterministic 结果覆盖；source 跟随画质最终待人工验收。

## 单卡探索记录（不是正式端到端验收）

- `grouped-r1` 的 139 项测试通过，但 B512 分支实现仅 H14 的实际调用微基准优于
  逐片段 fused；H28/H56 多数组合退化。没有把相对 eager 的收益当作增量收益。
- `grouped-r3` 的 172 项测试通过；随后“不同几何首次调用也不应新增 cubin”的
  严格探针失败，后续 collective/micro 没有执行。这不证明预热后反复编译。
- `tuning-r3` 对 35 种配置、两种顶层 alignment 设置，各执行首次和两轮重复，
  共 210 条编译记录；已预热组合两轮均没有新增缓存键。保存实际 specialization、
  key、metadata、cubin/PTX SHA256 和调用耗时。部分不同 key 对应相同 cubin；
  两个设置各有 25 个内存缓存键、14 种 cubin，35 个 case 的 key 对照完全相同。
  第二设置的首次调用复用了第一设置的磁盘缓存，不能将 49.839 秒与 0.175 秒解释为
  编译加速；顶层 alignment 开关未改变这些嵌套特化，候选不采用该改动。
- 在同一旧基线、同一 GPU 上，B4096 的统一算术体是下一轮候选。探索包含完整
  逐片段/grouped dispatcher 和单独 kernel 调用两类数据；后者的 host 计时未经过
  正式 dispatcher，图重放只表示 GPU 组件耗时，均不能替代真实模型路径收益。
  后续 `grouped-r5` 已完成下述单卡门槛和 SP1/2/4 严格诊断；正式计时仍在执行。

## 冻结候选 grouped-r5 的单卡结果

模型外围基线固定为 main `5b1b93db`，运行源码与普通仓库源码分别保存完整
SHA256 清单；两者的唯一实验 observer 差异单独记录。以下不与旧外围基线横比。

| 门槛 | 当前证据 |
| --- | --- |
| CPU 回归 | 合计 289 passed / 94 CUDA skips；多进程 tiling 模块单独进程运行 |
| 带 GPU 环境的完整单元测试 | 178 passed = 94 个 CUDA 用例 + 84 个 CPU 用例；包含非对齐指针、stride、位模式尾部、alias 和失败未发布。84 个 CPU 用例已包含在上行 289 内，不重复累计 |
| 编译特化 | 35 cases × 两种顶层 alignment 设置 × 三轮，共 210 条；全部数值正确 |
| 同一签名预热复用 | 两轮重复均无新内存 key、无新磁盘 cubin |
| 编译变体数量 | 每种设置 25 个内存 key、11 种不同 cubin；第二设置复用首设置磁盘缓存 |
| SP1 实际 attention | Eager / chunked / grouped / contiguous 的 KV、输出与 raw K 保留全部逐位通过 |
| 完整 dispatcher micro | 60 条记录齐全；30 次计时原样保留，11/60 的 spread 超过 10% |
| 后续 SP1/2/4 严格诊断 | full-r2 的 E/F、F/G 及 180 个正式端到端样本全部通过，见文末；独立 profile 单独验收 |

作业 `dlc1omaqru07xqka`（单 GPU）已成功；实际设备 CC 10.3、274113 MiB。
小产物包 SHA256 为
`9bfc3e564376b9e480cc4bf0cc30e8ac801b75469c85d7038ece018b3974947c`。
此处 11 种 cubin 属于最终候选，上一节 tuning-r3 的 14 种属于探索实现。

CPU-r31（`dlc19mtuyy9vdtlt`）已逐文件核验全部 67 份模型文件，但将 model 和
progressive-tiles 测试放在同一 pytest 进程，导致后者 fork 继承前者初始化的 Gloo：
原结果为 273 passed / 94 skipped / 16 failed，16 项错误均为默认进程组重复初始化。
CPU-r33（`dlc17p6j9mr3d8qe`）在新的进程中运行完整 tiling 模块，16/16 通过。
原失败保留，没有为此修改生产源码；接受回执关联两份日志、源码清单与模型哈希。

这批 micro 从真实 token 几何重建，但输入 row stride 为 `2 × heads × 128`，
属于 head-shard 压力布局；实际 raw KV store 分配为本 rank heads 连续布局。
因此不能将其称为实际缓存布局的性能。四组 workload 的 CUDA-event median 范围：

| Heads | Chunked（ms） | Grouped（ms） | Contiguous（ms） |
| --- | --- | --- | --- |
| 14 | 0.554–0.608 | 0.310–0.323 | 0.306–0.315 |
| 28 | 0.565–0.609 | 0.406–0.422 | 0.402–0.418 |
| 56 | 0.567–0.607 | 0.592–0.616 | 0.607–0.645 |

H56 未表现出一致增量收益。下节进一步分列连续 store 与压力布局，并记录真实
模型调用布局及 SP1 筛选。短矩阵和组件计时均不作为正式端到端加速结论。

## SP1 完整模型诊断与候选筛选（已通过）

单 GPU 作业 `dlctts8p6vz7w7yo` 已分别完成 Eager、chunked、grouped、连续视图、
grouped + plan 五条路线的 15 秒诊断。四个候选各对 Eager 的 65 个块、六份 prefix
快照、cache/noise/source/effective positions/latents 和最终 360 帧 RGB 均逐位一致。
最终 RGB SHA256 为
`1160535e35e15dd598a0fcf0aca920888334407592e25046acbdc9b8177edef3`。
六参考图、source S0、30×52 source latent 几何和 deterministic encoder 配置的
输入契约也一致。诊断中的 clone、CPU 拷贝和逐位检查不进入正式性能计时。

每条路线实际检查 118,750 个历史 chunk、219,680,500 行，原始 K 保留。
Eager/chunked 分别调用 118,750 次；grouped 与 grouped + plan 为 6,500 次批量调用；
连续视图为 6,500 次连续路径调用，没有 fallback。67 种实际 raw K 布局均为
`stride=(7168,128,1)`，即 H56 的连续 local-head store，指针均 16-byte 对齐。

真实模型首次累计 33 个 grouped 内存缓存键；这是其输入组合的覆盖，不能套用上节
35 个 characterization cases 的 25 个键。正反向 grouped / grouped + plan 的
完整预热与计时请求均维持 33 个键；全部 16 个计时请求均无新增磁盘 cubin。
正式 SP1/2/4 矩阵继续记录实际 key、metadata、cubin/PTX 哈希及预热后的变化。

本轮另完成 120 条双布局微基准（连续 store 与 2×head row-stride 压力布局）。
连续 store 下 H14/H28 的 grouped 中位数较 chunked 低约 43.5–44.5% / 27.0–28.2%，
但 H56 反而高 3.3–7.3%；部分计时 spread 仍超过 10%。这仍是合成输入的完整
dispatcher 测量，不能推算实际请求的收益。

速度筛选固定四路线正反顺序 `F → G → C → G+plan` / `G+plan → C → G → F`，
每进程完整预热 15/10 秒后各测一次 10/15 秒，共 16 个计时样本。
它只用于选候选；完整正式矩阵仍为三路线 × SP1/2/4 × 两顺序 × 两时长 × 五次
= 180 个样本。不得把筛选中每组单样本的零 spread 当作稳定性已通过。


下表为全部筛选样本的 API wall 秒数；每个单元格依次为正向 / 反向顺序，
各仅一次，不据此宣称已通过正式稳定性门槛。

| 路线 | 10 秒请求 | 15 秒请求 |
| --- | --- | --- |
| F：逐片 fused | 37.1446 / 36.9076 | 54.8164 / 54.5016 |
| G：grouped | 35.6315 / 36.3739 | 53.2859 / 53.4812 |
| C：连续视图 | 35.9208 / 36.0986 | 53.4801 / 53.5042 |
| G + plan | 35.8598 / 36.7722 | 54.3240 / 53.4010 |

G 相对 F 的四项降幅为 4.073%、1.446%、2.792%、1.872%，逐 rank
allocated/reserved 峰值均未增加。G + plan 没有稳定额外收益，C 与 G 接近，
因此正式矩阵选最小的 **G（mode 2，不启用 plan / contiguous）**。这是候选选择，
不是最终加速结论；保留其余路线与所有原始样本，不将旧基线收益计入本轮增量。

CPU-r35（`dlca4z47erbd2fux`）已核验全部 16 份 raw RGB 的大小及 SHA256，
同长度输出完全一致，两个唯一视频完整解码通过，人工画质仍待验收。
912 份小产物逐文件校验通过，归档 SHA256 为
`a13a548bc63988009aab8f1b463f59d3b0d8f2bf937a776af1dfe089659f4c28`。

第一次正式作业 `dlcz49t6ycwjw2ly`（full-r1，单 worker 4 GPU）失败：
SP1 E/F 单路诊断完成，G 已运行块 0–64，但新增采集器在 dispose 保存编译
metadata 时，未处理 Triton `GPUTarget` 对象的 JSON 序列化。没有观察到数值
assert 失败，但该轮严格对照未完成，正式计时尚未开始，不能称为通过。
全部失败产物保留，未覆盖或删除。

修正仅作用于实验 observer：将 dataclass metadata 转为保留字段的结构，未知类型
仍明确报错，不改生产 r5 源码。零 GPU 作业 `dlc91n7re8f8zp2h` 使用实际 Triton
3.7.1 对 210 份 metadata、嵌套 target 与未知类型拒绝行为完成检查，backend、arch、
warp_size 可完整读回。新 GPU 启动前核对同一 serializer 哈希并执行实际类型检查，
在加载模型前拦截采集器问题。

完整重跑 full-r2 `dlc69soi2bxlfqt0` 使用单 worker 4 GPU，固定 r5 生产源码、
main `5b1b93db` 外围与相同输入。SP1/2/4 的 E/F、F/G 严格对照全部通过：
每组共 7 个 rank，每 rank 65 个块与六份 prefix 快照，cache、noise、source、
有效位置、latents 和同 SP 最终 RGB 逐位一致。SP1/2 RGB SHA256 为
`1160535e35e15dd598a0fcf0aca920888334407592e25046acbdc9b8177edef3`，
SP4 为 `dbbac3fa3698b34d73b441bd7d9a44c33e15e2792c5761fe534280c2a2834abe`；
这里不宣称跨 SP 逐位一致，也不替代人工画质验收。

G 的实际模型编译记录已成功保存全部 7 个 rank 的结构化 target、key 与
cubin/PTX 哈希。SP1/2 每 rank 为 33 个内存 key，SP4 为 35 个；各 rank 均为
6 份不同 cubin、47 种 batch signature。它们是实际模型覆盖，与上文微基准的
25 key / 11 cubin 不是同一组输入。跨 SP 首次出现不同 key 不等同预热后反复编译；
后者由各 rank 的 15/10 秒预热与正式请求对照另行验收。

首次 source 编码的执行快照中，SP2 的 auto tiling 使用 `vae_decode` 组的两个 rank
并行执行；rank0/1 分别处理偶数/奇数 tile。SP4 快照使用四个 rank，每个 rank
处理 12 个 tile 中的 3 个。各快照均记录一次 gather、无 fallback；这不是整个
请求累计只有一次 gather 的结论。
`--encoder-parallel replicate` 不代表 progressive source 强制复制；严格对照包含
实际执行的输入 contract，要求同 SP 三路线一致。

## 180 个正式样本与 CPU 验收（已通过）

full-r2 `dlc69soi2bxlfqt0` 已 Succeeded，完整矩阵标记与退出码 0 均核对。
零 GPU 后处理 `dlcd099fuukpwaj8` 已 Succeeded；全部 180 份 raw RGB 重新核验
大小与 SHA256，四种唯一输出完整解码通过，人工画质仍待验收。
归档 4,162 个文件逐一核验；压缩包 SHA256 为
`59c3f7995eeb5ba5f75a652711c106fddd4718e981895d3837806cb651e9b7e8`。

本表仅比较相同冻结源码外围下的 E（eager）、F（逐片段 fused）、G（grouped）；不借用历史基线的收益。API wall 包含编码、解码与传输，独立 profiler 样本不计入这里的 180 个正式样本。

每个表格单元来自 5 次请求，中位数按顺序分别统计；全部样本与长尾保留。降低为正表示 G 更快，为负表示退化。画质最终仍待人工验收。

| SP | 顺序 | 视频秒数 | E 中位数/s | F 中位数/s | G 中位数/s | G 相对 E 降低 | G 相对 F 降低 | E/F/G spread | F→G 显存峰值未增加 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | EFG | 10 | 45.0143 | 37.0292 | 36.5719 | 18.755% | 1.235% | 1.01%/4.84%/2.15% | 是 |
| 1 | EFG | 15 | 68.7514 | 55.3999 | 54.8382 | 20.237% | 1.014% | 1.89%/4.15%/1.98% | 是 |
| 1 | GFE | 10 | 45.9529 | 38.1038 | 36.6645 | 20.213% | 3.777% | 3.70%/2.09%/1.19% | 是 |
| 1 | GFE | 15 | 68.5351 | 56.2719 | 54.6767 | 20.221% | 2.835% | 1.81%/1.16%/1.55% | 是 |
| 2 | EFG | 10 | 40.6558 | 38.0534 | 36.8965 | 9.247% | 3.040% | 2.10%/2.30%/3.24% | 是 |
| 2 | EFG | 15 | 60.4835 | 56.6002 | 54.5257 | 9.850% | 3.665% | 2.10%/2.06%/0.89% | 是 |
| 2 | GFE | 10 | 40.3880 | 38.4373 | 36.8498 | 8.760% | 4.130% | 1.27%/2.26%/2.75% | 是 |
| 2 | GFE | 15 | 60.0085 | 57.8198 | 54.1703 | 9.729% | 6.312% | 2.68%/2.49%/3.39% | 是 |
| 4 | EFG | 10 | 41.5182 | 39.4932 | 37.7099 | 9.173% | 4.515% | 2.70%/3.02%/2.86% | 是 |
| 4 | EFG | 15 | 63.3353 | 57.8326 | 55.0977 | 13.006% | 4.729% | 2.89%/8.47%/3.09% | 是 |
| 4 | GFE | 10 | 42.1518 | 39.2063 | 37.7197 | 10.515% | 3.792% | 3.56%/1.17%/1.51% | 是 |
| 4 | GFE | 15 | 62.6196 | 59.6166 | 55.7663 | 10.944% | 6.458% | 2.41%/7.99%/2.57% | 是 |

36 组各 5 次计时中，0 组 spread 超过 10%；12 组 G/F 对照中，0 组未显示正增量。此计数如实保留，不因验收预期改变阈值。

原始 RGB 哈希已核验全部 180 个请求；各 SP/时长输入与输出一致，15 秒输出同时匹配同 SP 严格诊断。预热后实际内存编译 key 与磁盘 cubin 复用检查通过。跨 SP 不要求逐位相同。

各 SP 下，E/F/G 记录的最大进程峰值相同。下表在同一 SP 的所有 rank、顺序和正式请求中
取最大值；计数器包含预热，原始逐 rank 记录见 `perf-analysis.json`，不是将多个
rank 的显存相加。

| SP | E/F/G 最大 allocated GiB | E/F/G 最大 reserved GiB |
|---|---:|---:|
| 1 | 229.413 | 229.941 |
| 2 | 179.226 | 179.881 |
| 4 | 154.090 | 154.891 |

实际编译验收 `actual-compile-acceptance.json` 保留每 rank 两次预热的 key 和 cubin
记录；每次正式请求均无新增磁盘 cubin，内存 key 数保持其完整预热后的值。
媒体索引 `media-review/index.json` 将全部 180 个请求映射到四种唯一 RGB 输出。
失败的 full-r1 observer 记录和 CPU 序列化修正证据另存，未计入本轮通过样本。

## 独立 profile（已完成）

GPU `dlcki2ih6zkx6230`、CPU monitor `dlcwfl1xmpri0s2s` 和零 GPU 后处理
`dlc1h94nbebcjsd7` 均 Succeeded。E/F/G × SP1/2/4 共九个独立进程组、21 份逐 rank
trace。各进程完整预热 15 秒后独立采集 15 秒请求，截取三次 Session.step 与三次真实
淘汰的窗口；profiler 请求不计入上述 180 个正式性能样本。

每份 trace 的历史行数均为 11,424,300。E/F 各重建 6,450 个片段，G 批为 300 次；
实际关联的 history CUDA kernel 数分别为 51,600 / 6,450 / 300。以下累计值可能
重叠，不是关键路径 wall time，也不将每次调用中位数直接横比。

| SP/rank | E/F/G CPU scope 总计 ms | E/F/G CUDA kernel 总计 ms | G/F CUDA 累计降低 |
|---|---|---|---:|
| 1/0 | 843.553/249.195/64.641 | 875.225/142.958/117.807 | 17.593% |
| 2/0 | 841.116/249.447/63.552 | 495.557/77.957/59.439 | 23.754% |
| 2/1 | 850.749/254.871/64.564 | 495.228/78.460/59.518 | 24.142% |
| 4/0 | 873.907/251.991/68.418 | 308.076/45.311/30.196 | 33.360% |
| 4/1 | 839.418/254.143/65.896 | 306.280/45.422/30.231 | 33.444% |
| 4/2 | 880.000/241.209/67.071 | 307.695/45.306/30.205 | 33.330% |
| 4/3 | 894.247/255.094/67.176 | 306.731/45.332/30.327 | 33.100% |

九份 profile 输出的 raw RGB 与同 SP 严格诊断一致，两种唯一输出完整解码通过。
所有 profile 请求无新增 grouped cubin；七个 G rank 在预热与 profile 后的内存 key
集合和 cubin 集合完全一致（SP1/2 为 33 key，SP4 为 35 key，各 6 份 cubin）。
全部 21 份 trace 已实际导入 Perfetto UI；SQL 的 scope/kernel 计数与原始 trace 核对，
保留可见 CUDA stream 时间线和计数截图。原始导入重叠事件警告保留，不宣称全部事件
都经过穷举验证或跨 rank 时钟已对齐。CPU scope、CUDA kernel、GPU memcpy/memset
分别归因，不能因某个 scope 没有关联 kernel 就认定其没有设备操作。

[交互报告、逐 rank trace 与 UI 截图](https://review-cdn.loopit.com.cn/reports/zing179-20260923/grouped-review-r1/index.html)
（办公网 / 公司 VPN）。[一键同步视频对照](https://review-cdn.loopit.com.cn/reports/zing179-20260923/playback-r1/index.html)
沿用逐字节一致的四种 MP4；G 与同 SP/时长 E/F 相同，共用同一组播放素材。

Profile 归档的 339 个文件逐一校验（另有清单自身）；CDN 的 42 个原始/压缩 trace 链接
核验 206、总大小和首 1,024 字节，完整 SHA 另由 OSS 归档与 UI 导入文件核验。
Profile 归档 SHA256：`24dd07ec960e2ebfee461f29586125d24710f6b51d264e9a2afb6ee2e8e69823`。
原始明细见 `profile-summary.json`、`profile-output-check.json`、`trace-manifest.json`
和 `perfetto-ui-checks.json`。预算包含旧失败轮次和后处理副本；新增路径继续默认关闭，
不自动合并 PR。

## 关于 performance 模式默认开启

本轮数据支持将最小 G 配置（`FUSED_HISTORY_ROPE=1`、
`GROUPED_HISTORY_ROPE=1`、`HISTORY_ROPE_PLAN=0`，均带 `SGLANG_ZING_` 前缀）
作为已测配置的后续默认候选，不支持把所有实验开关一起默认打开。
Plan 未显示稳定额外收益；contiguous 与 G 接近，不纳入正式选择。

当前完整端到端证据使用 deterministic encoder，不能替代原 performance encoder
下的验收。后者 eager A/A 在首个 prefix 前已有输入差异，因此其旧 A/B 不能直接
作为候选画质差异的因果证据。performance preset 的默认变更还需在实际部署输入、
encoder、硬件和其他已启用优化的组合下单独验证；本轮仅验证固定 832×480、
SP1/2/4、CC 10.3 环境，不外推全部尺寸或设备。尤其不把 #182 与 #185 的收益相乘。
原 source 跟随画质仍留给人工通过同步视频判断。本 PR 保持所有新增开关默认关闭。
