# Prompt joint 适配 split_first_latent

PR：[seedleap/sglang #501](https://github.com/seedleap/sglang/pull/501)（草稿）。

## 使用与范围

现有 source 会话开启 prompt joint、`joint_forward=cross_chunk` 时，按照模型的
`split_first_latent` 打包 prompt 切换的五个历史 latent，无新增开关。
`num_frame_per_block=[3,2]`、`split_first_latent=false` 使用两个历史块 3/2：

| condition | 联合分支 | 数量 |
| --- | --- | --- |
| miss / cache off | condition、history3、history2、source、target step0 | 5 |
| hit | 恢复 condition 后，history3、history2、source、target step0 | 4 |

`true` 继续采用 1/2/2 历史，分别六／五分支。音频、CFG、非 cross_chunk 等原有
限制仍适用。不改变缓存 key、采样和窗口策略。
本 PR 扩展 prompt 切换；合入主线 #464 后，首次未命中由已有
`INITIAL_PREFIX_JOINT` 控制 condition/source/target step0 三分支，
与首块拆分开关独立。首次命中恢复 condition KV 后走 source/target 两分支。

## 决策与不变量

复用 PrefixJointTransaction、分段 attention 和 PrefixJointGraphs，不新增计算后端。
单纯放宽入口不足够：旧代码按分支数量识别 condition hit，五分支同时可能是
split hit 和 unsplit miss。改为显式构造 phases，Graph 根据 cache 的 prefix 状态
和 pack 数确定历史数量；事务按 phases 校验 block 序列与历史 frame 数。

两种布局的 target block 分别为 8 和 6；source 为其前一 block。
每个 query 分支只读取已恢复 condition、之前分支和当前分支，不能读未来分支。
保留逐分支 RoPE 视图校验与 AdaLN timestep 分组，避免不同 GEMM batch 导致舍入变化。
KV 按 condition → history active/final → source → active 顺序发布，active target
不发布噪声 KV；source begin_chunk 从实际 target block 获取。失败清空会话 cache。
Graph 准备只在 shadow cache 上提交，执行成功后发布；hit 不能覆盖恢复的 condition。

## 验证方案与状态

计划：沿真实 SourceReadySession 的 prompt 切换，分别测试 split true/false、
cache hit/miss、eager/fixed/记录式 capture-replay、动态 AdaLN/静态表。
比较串行预测、各层 live KV、事件采用边界、RNG；验证实际分支数和 joint 命中，
以及失败后的释放和槽复用。小模型 BF16 沿用既有 atol/rtol=0.02，fixed 与记录式
replay 要逐位一致。记录式 backend 只验证 CPU 控制和张量路径，不是 CUDA Graph。

480 的真实四卡 SP2、cache on/off 正式请求结果见下文；736 按用户要求暂停。
独立细分采集仍需区分 Encoder（Qwen/参考图 VAE/source VAE）、DiT、Decoder
的 CUDA 耗时与 Host RPC/控制范围，不能用 forward 次数推断加速。

## 已执行结果（2026-11-02）

基线 `e4d8047acd9ee854cd023b677674f91ef7e78179` 加本工作区修改，
WSL Python3.11.15 / PyTorch2.8.0+cpu。定向矩阵37通过（72.80秒），最终完整三文件
`test_zing_h3_denoise_graph.py`、`test_zing_h3_joint.py`、
`realtime/test_zing_prompt_switch_cache.py` 共 **301 passed、4 skipped、1 warning**，
274.65秒。跳过项需要CUDA。
输出/live KV沿用BF16 atol/rtol=0.02，fixed与记录式replay逐位一致；静态表零投影权重、
动态投影缓存命中、condition恢复字节、事件/RNG和失败后槽复用均由对应断言覆盖。
新增四／五分支attention对照完整block mask，并检查未来KV污染不影响早期分支。

命令环境为 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SGLANG_LIGHTWEIGHT_RUNTIME=1
SGLANG_DIFFUSION_PLATFORM_OVERRIDE=cpu SGLANG_DIFFUSION_PERF_BASELINE_PLATFORM=h100
PYTHONPATH=python`；h100选项仅用于导入性能基线数据，不是硬件测量。
以 `python -m pytest <上述三个文件> -q -x --disable-warnings` 执行。
CPU fixture显式用torch_sdpa、stable shapes=0、compiled AdaLN=0，与CUDA性能默认配置不同。
Ruff check/format、isort、git diff --check通过。日志/JUnit/最终源码hash位于
本地 `outputs/unsplit-prompt-joint-20261102/`，不提交原始测试产物。
上述 CPU 测试不替代下述真实 CUDA Graph、NCCL 和模型验证；未部署。

## H100 实验中的缓存准入修正

四卡、SP2、W6/S3、未拆分首块的完整实验中，发现原 benchmark 客户端
没有传 `cache_write=true`，因此其 cache on 请求均为 miss，不能作为四分支命中
性能结果。客户端新增显式 `--cache-write`，默认保持不写入；真实 WebSocket
测试覆盖 RGB/video 与开关 on/off，4 项通过。重跑时先完整写入，再用相同数据
核实实际 hit/miss 和恢复记录，随后测正式请求。host 预算按全 SP 总额计算。
本数据每请求使用 condition 七次，对应四个唯一 key；全 SP host 80 GB、每 rank
40 GB，按最大 prefix 槽预算可容纳七槽，为保守超配。实际 L1 四条，无淘汰。

当前 480 candidate cache off 的独立数值诊断发现少量 live K 超出原有
atol/rtol=0.02；尚未完成原因核查，保留原阈值及失败记录，不据此宣称正确性
或性能验收通过。736 按用户要求不再补测；只补 cache on 的详细分段计时。

## 480 四卡正式结果（2026-11-02）

模型/runtime：旧版 `e4d8047acd9ee854cd023b677674f91ef7e78179`，新版
`e31b0e8b8ac37fb22f14e527b1ecbf90783adeb5`；cache on 两臂使用相同的
`f8d8d3e0caac3a9b891156274a02e09405a185aa` 原生客户端工具。工具 SHA
与模型/runtime SHA 分开记录。off 复用修正客户端前 r7 的两臂正式数据，on 为 r11
后续独占四卡时段，不能把两时段直接当作严格配对的 cache 开关消融。

H100 测试节点的设备实际名称为 L20Z、SM90、单卡 CUDA 容量 79.18 GiB；
布局为 Qwen/Encoder 一卡、DiT Ulysses SP2 两卡、独立 Lynn Decoder 一卡。
模型 1027 AR+DMD step010000，checkpoint SHA256
`89c3a67d77214e57d810e6a4e5554c7c919ae929db0174e6439dfadb53a1d731`。
BF16 全 50 层、weights-free AdaLN 表，split=false、history=3/2、quantum=1；
W6/S3 为用户对 window21/sink3 基线的显式覆盖；block_relative、gap12、stable 解码。
FA4、融合/grouped/chunk-reuse RoPE、stable shapes、编译 AdaLN、metadata reuse、
CUDA Graph/SP Graph 开启；HISTORY_ROPE_PLAN=0；SM90 不启用未验证的 fixed FA4
及 current RoPE/KV 融合。使用真实 conditioning 容量 7421 prefix rows。
不能沿用旧文档中“Graph 仅 SP1”描述来判断本提交的实际 SP2 路径。

输入为完整 reference-switch：764 帧、源帧 34/289/374/544/561/629 六次更新。
每臂三次正式请求，全部生成完整 RGB 并采用六次更新。on 每臂先完整 fill 一遍，
再完整 hit warmup 一遍；第二遍及正式请求均逐 rank 7 hit、0 miss；新版每请求
六次四分支 prefix joint。正式组不加载细分计时 hook 或串行数值 oracle。

| 路径 | TTFT P50/P95（秒） | 全程 FPS P50（范围） |
| --- | --- | --- |
| 旧版 off | 3.2306 / 3.4081 | 19.0243（18.5559–19.3484） |
| 新版 off | 2.5012 / 3.3105 | 18.6301（17.9051–18.9876） |
| 旧版 on | 1.6249 / 1.7093 | 20.0524（19.9331–20.1514） |
| 新版 on | 1.5464 / 1.7310 | 20.1568（19.9824–20.2680） |

TTFT 为客户端开始至完整有效首个 RGB，未包含浏览器渲染；P95 按三次请求线性
插值。on 中位 TTFT 降约 4.83%、FPS 升约 0.52%，样本范围重叠，不能断言稳定
整体收益。启动路径未改，TTFT 差异不能直接归因于 prompt joint。
off FPS 中位回退约 2.07%；同样不能仅据 n=3 给出稳定性能回退的因果结论。
提示提交至首个匹配事件完整 RGB，旧/新 on 的 P50/P95 分别为
1410.24/1554.87 ms、1338.68/1553.69 ms；每臂 18 个事件来自三次独立请求。

新版正式 on 每请求/每 rank 六次 prefix replay，无新 prefix capture；但首次正式
请求仍有一次尾块/恢复块 denoise Graph capture。该块不能并入稳态 replay 计时，
不能声称“两遍准备后所有 Graph capture 为零”。

独立 cache on 数值诊断 n=1：两 rank 的串行 A/B 与四分支联合预测 112320 个
元素逐位一致，全部 50 层 live K/V（100 张量）逐位一致，50 层恢复 prefix 字节
均保持；通过原 atol/rtol=0.02。这是 hit 路径证据，不能覆盖 off 诊断失败。
off 失败为一个 21034496 元素 K 张量中 13 个元素超差，max abs=0.05102539；
诊断当时即停止，没有完整各层误差分布。没有扩大容差或用完整视频替代数值验收。

off 原生正式日志已有同口径 Host 分块：各臂 rank0 的三 latent n=84、二 latent
n=81，按实际采用的 prompt epoch 排除前六个生成块及不完整尾块。
Encoder/DiT/Decoder P50（ms），旧版三 latent 为 217.08/235.55/2.54、二 latent
13.20/213.17/2.46；新版分别为 215.63/236.80/2.49、13.36/214.59/2.43。
这些是调用、RPC/预取等待与提交范围，不能标为 GPU kernel 或直接相加。
新版 off 数值诊断未进入稳态，缺少与旧版可比的稳态 CUDA 明细。

正式原始数据位于 `pr501-unsplit-joint-h100-20261102-r7`（off）及
`pr501-unsplit-joint-h100-20261102-r11`（on），节点目录均在 `/mnt/zing-nvme/`。
本地核验下载及分析位于 `D:/pr501-report-20261102-r1/formal-analysis/`。
详细 r11 因观测 hook 不支持 Decoder 的 `-m` 入口而在请求前失败，未影响正式组；
修正后 r12 的真实模块入口及 CPU 解码线程测试 2 项、prefetch 39 项、原生客户端
4 项均通过，已完成两臂各三次完整 cache on 详细请求，每次七 hit/零 miss，
新版每次六次四分支。详细组不充当正式吞吐数据。

### Cache on 独立分段计时

单位 ms、P50/P95；CUDA 为当前 stream 两事件间 elapsed（包括 stream 等待），
不是纯 kernel busy time。DiT/Decoder 按真实采用 epoch 排除前六生成块、尾块及
观测到的 capture/warmup；source VAE 连续运行，排除最初十五 source latent，
其输出批次为 1/2，不能与 DiT 的 3/2 块混同。各臂三次诊断请求，块样本不视为
独立重复，两个 SP rank 分开；下表 DiT 为 rank0，rank1 同口径原始数据保留。

| 服务块 | 旧版 CUDA P50/P95 | 新版 CUDA P50/P95 | 样本数/臂 |
| --- | --- | --- | --- |
| source VAE 输出 1 latent | 86.75 / 87.20 | 86.82 / 87.13 | 126 |
| source VAE 输出 2 latent | 263.01 / 264.60 | 259.42 / 264.68 | 252 |
| DiT 3 latent | 319.15 / 338.48 | 319.74 / 341.59 | 84 |
| DiT 2 latent | 267.24 / 280.89 | 267.52 / 282.83 | 81 |
| 独立 Lynn Decoder 9 帧 | 105.34 / 105.92 | 106.08 / 106.22 | 84 |
| 独立 Lynn Decoder 8 帧 | 70.61 / 71.03 | 70.90 / 71.17 | 81 |

命中请求的 Qwen/reference VAE 跳过，三个请求区间内无 encode 调用；fill 有
调用和绑定记录，不能将“无调用”填成一次零耗时。Host 控制范围 P50/P95：
前端 prepare_next_request 旧0.56/0.70、新0.55/0.88；scheduler 排队均0.02/0.03；
return_result 旧0.18/0.22、新0.18/0.21；Decoder append 旧1.53/1.90、新1.51/1.85；
shared-memory publish 旧38.92/40.50、新39.03/39.53。各 scope 包含相应请求内调用，
存在嵌套/并发，不可相加作为独占控制时间。相邻稳态 3→2 DiT Host 调用间隔旧
105.88/115.25、新106.50/115.15（各81对），还包含真实组件工作和等待。

详细组原始数据为 `pr501-unsplit-detail-h100-20261102-r12`；本地
`D:/pr501-report-20261102-r1/detail-analysis/report.md` 给出逐 rank Host/CUDA、
控制边界及两臂三请求全部36次 prompt 时间线，CSV 为 `prompt-switch.csv`。
各事件分列提交→入口、cache 恢复开始、恢复提交 Host/原生 copy stream CUDA、
prefix prediction/joint、ACK、首个匹配完整 RGB，不将 ACK 当作事件采用。
只选提交至首匹配 RGB 的记录，排除后续检查遗留旧 event 元数据。
时钟按节点 boot ID 与客户端 CLOCK_MONOTONIC 核对，无 session 关联冲突；
CUDA elapsed 不映射成 Host 时间戳。所有预期 CUDA span 均有完成记录、无错误，
原生 profile complete 且 health 无 dropped/error。服务 SIGTERM 退出未生成 hook
终态 detail_health，因此不声称取得进程最终 dropped 计数。

### 显存与媒体核验

NVML 在三次正式客户端区间内采样的设备峰值，单位 MiB；只计实际选中四张卡，
不含独立 fill/warmup/diagnostic。采样峰不等于连续观测的绝对峰值。

| 路径 | Encoder+Qwen 共卡 | DiT rank0 | DiT rank1 | Decoder |
| --- | --- | --- | --- | --- |
| 旧版 off | 38961 | 77639 | 76519 | 25543 |
| 新版 off | 38961 | 80211 | 79995 | 25543 |
| 旧版 on | 38961 | 70677 | 69373 | 29829 |
| 新版 on | 38961 | 72731 | 71427 | 29829 |

表为设备总占用，不把共卡 Encoder/Qwen 拆成组件独占显存。off/on 来自不同时段与
不同客户端版本，不能用该表直接归因缓存的显存收益；同一 cache 的旧/新比较保留。
每臂正式首请求均编码为 24fps/764帧 MP4，经 ffprobe 及全片解码核验；全部正式
RGB 字节长度、SHA256 已核验。on 正式第0次旧/新 RGB SHA256 相同，但其他请求
可能因实际 prompt 采用边界不同产生不同输出，不能宣称所有请求逐位相同。
抽查六个时间点未发现明显解码损坏，未据此宣称完整画质等价或覆盖 off 数值失败。
原始 RGB、失败和中断实验均保留；私有归档已校验读回，不提交原始媒体/日志。

## 2026-11-03：与主线首次联合执行兼容

合并主线 `772927dc45a03616e0f26219143d021aa6df414e`，保留 #464 的首次三分支，
同时接纳 split/unsplit 的 prompt miss/hit 四、五、六分支。
`PrefixJointTransaction` 统一校验合法 phases；首次没有 history，source/target
block 为 1/2，prompt 切换按历史块数推导为 5/6 或 7/8。
eager 使用事务类型识别 prefix AdaLN 分组，fixed 路径同时识别首次 prefix 与
四/五分支命中布局；恢复 condition 的坐标映射仍应用于全部 prefix 事务。

补充首次联合开关 on/off × split true/false × step/drain/记录式 Graph ×
有无参考图，以及 Block0 首次 miss/hit 后两次 prompt 切换的组合回归。
模型相关三个完整测试文件 332 项通过、4 项 CUDA 跳过（280.05 秒）；
上述组合回归另跑 36 项通过，客户端两组回归 127 项通过。
记录式 Graph 只验证 CPU 控制/张量路径，不代表真实 CUDA Graph。
前述 H100 结果仍属于 `e31b0e8b` runtime；本次合并后的 runtime 未重跑 GPU，
不能将历史结果当作合并版本的端到端或性能结论。

## 社区与 review 导读

实现基于 seedleap/sglang `e4d8047acd9ee854cd023b677674f91ef7e78179`，
延续该分支已有 mixed transaction、host metadata 和 fixed KV planning 机制。
这是 Zing 分支的现有机制，未据此声称官方 sgl-project/sglang 已采用本特性。
已核对本提交的 `docs/docs/sglang-diffusion/contributing.mdx`：要求控制抽象与异常
处理、提交前做端到端验证，并为影响延迟／吞吐／显存的改动提供同条件性能报告。
本改动沿用已有事务边界，没有新增后端或异常屏蔽；已补真实四卡性能报告，但
off 数值失败及未覆盖的 off 稳态 CUDA 对照仍阻止完整验收，保持草稿。

阅读顺序：incremental.py `_prefix_prediction`（真实开关/入口与回退）；
zing_h3_joint.py `PrefixJointTransaction`（合法布局、attention 坐标、KV 提交）；
zing_h3_prefix_graph.py 和 zing_h3_denoise_graph.py（hit/miss 准备与 prefix 映射）；
zing_h3.py（eager/fixed 的 AdaLN 分组）；test_zing_h3_denoise_graph.py（真实会话回归）。
Review 重点是五分支歧义、source/target block、恢复 condition 的所有权、
以及四分支是否确实进入 prefix modulation 而非普通 joint。
