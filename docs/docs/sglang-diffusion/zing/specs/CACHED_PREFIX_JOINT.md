# 外部前缀 KV 的联合执行适配

2026-11-02 布局扩展：下文六／五分支对应 `split_first_latent=true` 的历史实现与证据。
当前 `false` 使用 3/2 两个历史块，condition miss / cache off 为五分支，hit 为四分支；
两种布局共用分段 attention、AdaLN 分组和按序 KV 提交机制。
新布局的独立方案、验证与 GPU 缺口见 [Prompt joint 开关适配](../plans/UNSPLIT_PROMPT_JOINT.md)。

依赖：PR #445 `2d70a4575efe18b446e1fcf5007f1b785ac09a39`（替代 #441 的单条目提供方）；本地分支 `codex/zing-cached-prefix-joint`。状态：消费者接入新版 L1/L2 缓存，联合执行与缓存开关独立。历史 r8 的 480/SP2/L1 单输入 GPU A/B 使用修正前的开关绑定；当前解耦版本尚未重测 GPU，不能沿用旧数据证明六分支与五分支的性能差异；未上线。

## 使用与分工

其他模块负责 prompt/reference 缓存的 key、命中、淘汰及 KV 写池；本改动只消费当前会话池中已就绪的静态前缀。启动命中后联合执行 source + target.step0；切换命中后联合执行最近 5 个 target latent（1/2/2 三块）+ source + target.step0，后续去噪沿用现有调度。

连续 source 入口在联合开关启用、#445 的 `stage.block0_cache` 已创建且采样为 `cross_chunk` 时安装启动 `_load_block0_prefix`；缓存开启时切换复用 #445 的 `prepare_prompt_prefix`。回调只负责缓存恢复，不决定是否启用六分支。无需另外注入 stage 回调。直接使用 `SourceReadySession` 的内部调用方仍可传入 `prefix_loader`，回调签名为 `loader(cache, inputs, prompt, prefix)`：

- `cache` 是本 rank 当前会话租用的实际 KV 池包装；不得换池、释放租约或改其他会话。
- `inputs` 是候选的完整 H3 输入；`prompt` 包含 condition/event_id/prompt_version；`prefix` 是对应静态前缀 packed layout，可用于核实位置/块布局。
- 返回 `False` 表示未命中，必须完全不修改 cache；返回 `True` 时必须已将所有层的 raw K/V 写入池前端，并通过既有 cache 发布契约建立完整的 prefix-only 状态（含 CPU 元数据和模型身份）。不得保留旧 committed/active/source 历史；跨 stream 写入必须在回调返回前建立消费 stream 依赖。
- 每个 SP rank 都在同一边界执行回调；各 rank 的命中与布局必须一致。异常、状态不完整或分歧终止该会话，沿用外层释放逻辑；不尝试在已被覆盖的池上继续旧 prompt。

联合执行由 `SGLANG_ZING_SOURCE_PROMPT_JOINT=1` 独立启用（默认 `0`），显式 source command 的布尔字段 `prompt_joint` 优先于环境变量，与 main 的入口契约一致。受支持的 `cross_chunk` 路径中，联合开启后：缓存关闭走六分支；缓存开启但未命中也走六分支；命中恢复 KV 后走五分支。联合关闭时沿用串行切换，缓存仍可独立使用。`step()` 与 `drain_ready()` 遵循同一规则。

缓存仍须配置 KV pool 并设置 `SGLANG_ZING_BLOCK0_CACHE_MODE=overlap`；默认 `off`，与 #445 一致。`early/late` 保留其恢复时序对照，条目数和可选磁盘目录也由 #445 配置。关缓存仅不安装恢复回调，不关闭联合调度。启动未命中保持 prefix 单独计算再执行二分支，命中直接执行二分支。直接使用 `SourceReadySession` 时默认 `prefix_loader=None`。

```bash
export SGLANG_ZING_SOURCE_PROMPT_JOINT=1
# 对照六分支无缓存时，仅把下面的 overlap 改成 off。
export SGLANG_ZING_BLOCK0_CACHE_MODE=overlap
export SGLANG_ZING_BLOCK0_CACHE_ENTRIES=6
# 可选 L2；目录须遵守 #445 的 checkpoint/部署配置隔离要求。
export SGLANG_ZING_DISK_CACHE_DIR=/mnt/zing-block0-cache
```

## 状态与执行

此前适配将 `prompt_joint` 绑定到缓存对象存在性，错误覆盖了独立开关；本次恢复 main 的 command/env 配置优先级。`_source_key_policy` 同时用于 prefill 和 live，确保实际联合策略属于一致的缓存 key。缓存对象只控制查找、恢复及填充，`incremental._activate_pending_prompt` 根据联合开关和实际命中选择六／五分支；SP 路由一致性检查包含联合开关。无缓存时不需要创建空缓存提供方或伪造恢复回调。

启动在首块打包、池租用完成后尝试装载；切换在五 latent 和 codec 周期边界满足、历史打包完成后尝试装载。候选仅排队时不调用装载，已准备的旧块先排空。每次条件只尝试一次，命中后不再 clear 或 forward prefix。

复用现有 source/active 联合事务处理启动；扩展现有 prefix/history 联合事务支持省去 prefix 的五分支路径。prefix KV 作为只读历史参与所有层 attention；新历史仅提交本次 clean 行，target.step0 保持 active。Graph 复用现有前缀图管理器，固定计划的历史长度包含恢复的前缀，捕获/回放不得写入这些行。

继续沿用现有联合路径的 video-only、raw pooled KV、few_step、after_step0、SP1/2/4 等支持条件；五／六分支支持 AdaLN 静态表和动态投影缓存，SP 仍要求 stable shapes。命中但不支持联合执行时明确失败，不悄悄串行；未命中的不支持布局沿用既有回退。不新增 source 历史语义，也不包含 PR #426。

### 五分支 AdaLN

五分支保留串行 timestep 分组，各组分别查询静态表或会话 LRU，再拼接投影，不查询不存在的混合 `[clean, active]` 表项。静态表仍要求完整 schedule 准入，缺项直接失败，不恢复投影权重或在线建表。启动和跨 chunk 联合也按 singleton 表项消费。

fixed/Graph 在 host 准备阶段解析投影，将各层及 final 的小张量加入固定输入；replay 更新投影输入，capture 内不执行 Python 缓存操作。动态缓存归属于 KV 会话，沿用预算、权重/流签名及清理规则。六分支包含静态 prefix 的混合 timestep GEMM，按 `prefix_modulation` 的完整分组查询精确表项，不拆成 singleton 改变舍入行为。

验证计划覆盖 dynamic/table × eager/fixed/recording，包含文本与参考图、重复 prompt 切换、输出及 KV 对照、prefix bytes 和 RNG 精确比较，并核查实际命中。此处为五分支 AdaLN 更新时的计划；后续 480/SP2/静态表实测见末节，真实 CUDA Graph、动态投影缓存 GPU 和 736 仍待验证。

2026-10-27 五分支 AdaLN 更新实测：基于 `081f57e85d` 的本轮工作树，在相同 WSL / Torch CPU 环境运行原五个相关文件，另加 `test_zing_h3_adaln_projection.py` 和 `benchmark/zing_h3_causal/tests/test_prefix_joint_graph.py`，**452 passed、10 skipped、0 failed，285.28 秒**。最终单组投影避免额外复制、消费端断言由五分支六组对照及完整 `test_zing_h3_fixed_forward.py` 复跑覆盖，**33 passed、0 failed，53.76 秒**，与前者有重复，不相加。日志为仓库外 `D:/adaln-five-full-tests.log`、`D:/adaln-five-final-focus2.log`。静态表模型的投影参数存储为零；动态缓存有实际命中；五分支模型主体检查 eager 缓存接口或 fixed/Graph 预解析投影确实被消费。Ruff、isort、格式、diff 和 CPU inventory 通过。

简单地跳过 prefix forward 不足以满足需求：现有切换事务要求空缓存、六分支，并在消费前 clear。这里扩展同一事务和图 owner，避免另建一套 scheduler 或存储。

## 正确性和性能验证

2026-10-28 独立开关修正：四个完整文件 `test_zing_preparation_cache.py`、`test_zing_progressive_source_session.py`、`test_zing_prompt_switch_cache.py`、`test_zing_h3_denoise_graph.py`，WSL / Torch 2.8 CPU 既有导入适配 runner 实测 **898 passed、4 skipped、0 failed，83.81 秒**（跳过 CUDA 路径）。覆盖 command/env 优先级、缓存开关独立性、early/late/overlap 启动接线、prefill/live key 一致及策略变化；真实小 DiT 在 step/drain × L1/L2 下，关缓存的五次切换实际分支数为 `[6,6,6,6,6]`，开缓存为 `[6,5,5,5,5]`。对照输出、完整 prefix/动态 KV、RNG、prompt 事件和原池地址；联合关闭也保留串行对照。首轮测试发现 step 入口仍依赖恢复回调，修正后通过，未放宽数值容差。Ruff 检查/格式与 diff 检查通过。日志在仓库外 `D:/pr439-prefetch-ci-20261028/independent-gate-tests-final.log`。这是 CPU 调用链与数值回归，当前解耦版本未执行新的 GPU/NCCL 性能对照。

计划：用真实小 DiT 与真实 KV 池对照串行重建，覆盖启动、重复切换、前缀长度变化、eager/fixed/记录后端捕获回放；核对输出容差、完整 KV、RNG、事件采用、prefix 不重算且不被覆盖。覆盖装载异常、不完整状态、miss 和会话释放/复用。CPU 记录后端不等同真实 CUDA Graph/NCCL。

2026-10-27 原计划（480/SP2 的后续实测见末节）：遵循 PERFORMANCE_DEFAULTS、CLIENT_BENCHMARK 和 STAGE_TIMING，在同硬件/模型/输入上分别比较冷路径与热命中路径，记录 TTFF/E2E/FPS/TPOC/ICL、样本量、逐 rank 峰值以及 Qwen/参考 VAE/source VAE/DiT/Decoder 时间与显存。必须单列装载时间和缓存常驻成本；本实现不宣称现成加速比。

2026-10-27 实测：WSL Linux / Python 3.11 / Torch 2.8.0+cpu，五个完整文件共 **436 passed、10 skipped、0 failed**（257.37 秒）。文件为 `test_zing_h3_denoise_graph.py`、`test_zing_h3_joint.py`、`test_zing_h3_chunks.py`、`test_zing_initial_blocks.py`、`test_zing_h3_kv_pool_cache.py`；跳过项均为 CUDA 路径，完整 nodeid 在日志中。两项环境 warning 是 pytest 已导入插件与可选 torch-c-dlpack-ext JIT，不作为 CUDA 能力证明。

新增覆盖包含真实小 DiT + 文本/参考图 KV 的串行对照、五分支 eager/fixed/记录后端 capture/replay、prefix bytes 精确保持、重复长度/内容切换、RNG/事件、SP2/SP4 CPU 线程 collective、step/drain 两入口、miss 保持原路由、最新候选覆盖、装载异常和同槽复用。输出与动态 KV 沿用测试既有 BF16 `rtol=atol=0.02` 容差，静态 prefix 和 RNG 为精确比较。

复现入口：仓库根运行 `pytest python/sglang/multimodal_gen/test/unit/{test_zing_h3_denoise_graph,test_zing_h3_joint,test_zing_h3_chunks,test_zing_initial_blocks,test_zing_h3_kv_pool_cache}.py -q`，独占或显式分配 `MASTER_PORT`，CPU 线程设为 1。此次使用仓库外 `/mnt/d/cached-prefix-cpu-run.py` 设置 lightweight/CPU 平台并适配本机无 SGL CPU 扩展的 RoPE 导入，未替代模型/KV/通信计算。正式 CI 仍应使用仓库固定 runner。

Ruff 0.15.1 检查/格式、isort 7.0.0、`git diff --check`、Windows 原生 `run_cpu_tests.py --check-inventory` 通过。完整日志 `D:/cached-prefix-final-tests.log`，runner 和清单回执均存仓库外。此前失败保留在 `D:/cached-prefix-*-tests*.log`：包含装载校验误把 visual rows 当总 rows（已修正）、测试夹具的 timestep/通信配置、记录后端断言、并行进程端口冲突，以及默认参数透传破坏覆写（已修正）；不将失败轮计入最终通过数。WSL inventory 的 Git diff 因挂载盘扫描超时，改用同仓库 Windows 原生运行后通过。

## 仓库实践与 review 导读

### 与 PR #445 多级缓存适配（2026-10-28）

缓存实现以 #445 为准：`Block0PreparationCache` 拥有完整请求 key、L1 pinned LRU、L2 提升、跨 rank 命中投票、异步恢复和刷盘；本 PR 不另设条目或淘汰策略。启动复用其提前提交的恢复；late 对照在首个消费边界才提交。切换复用 `_prepare_block0_switch(candidate)`，在实际采用最新候选时更新缓存 epoch 元数据并返回恢复回调，不再读取单条 `.entry`，不再用 Python condition 对象身份推断命中。

执行方先完成滚动 prefix 打包与 SP 边界检查，再清理当前逻辑缓存并调用恢复回调；恢复仍写入原 GPU KV slot。命中只保留三段 history，加 source/target 进入五分支；未命中保留完整六分支。无 cross_chunk 的原生调用者继续使用 #445 的串行恢复入口。miss 生成的 prefix 在 `_finish` 发布 capture hook，`_remember_block0` 记录新 epoch 对应的像素起点；只在该 epoch 首帧真正输出后填充缓存，避免旧 Decoder 队列输出提前触发写回。

关键不变量是恢复后 prefix-only 的完整 raw KV、跨 rank 分支一致、prefix 只读，以及候选取代／失败时不改活跃 epoch。沿用 #439 的 AdaLN 静态表／动态缓存和 history RoPE 路径，保留 #445 的预热和 L2 重启语义。合并时模型固定执行保留本 PR 对动态投影缓存的支持，未退回仅静态表支持。

验证计划：真实小 DiT 对照 serial，覆盖 cold/warm 启动、A→B miss→A hit、B 首帧后写回、L1 淘汰及 L2 恢复、最新候选／拒绝、step/drain、输出和逐层 KV、RNG、eager/fixed/记录后端；回归静态表和动态 AdaLN。本段 CPU 适配验收不覆盖 GPU；后续本联合路径的 480/SP2 实测见末节，#445 自身的性能结果不能视为本联合路径的验收。

本轮组合工作树（父提交 `f39981ea48` 与 #445 `2d70a4575e`）实测：

- 四个完整 source/cache/denoise 文件 **835 passed、4 skipped、0 failed，109.51 秒**；跳过为 CUDA 路径。
- cache、joint、AdaLN 与 prefix graph 七个文件 **302 passed、0 failed，186.32 秒**；真实小 DiT 的 step/drain × serial/joint × L1/L2 八组 A/B 往返对照，检查 B 首帧后写回、命中／未命中路由及完整 KV；L2 使用一槽 L1 强制淘汰，实际文件写入／提升与进程签名重新绑定，不将 DMA 传输 stand-in 当作真实 CUDA。
- 最终启动接线文件 **14 passed、0 failed，27.50 秒**，覆盖 early/late/overlap、总开关策略和已提交恢复复用。

以上组有重复覆盖，不相加。环境为 WSL Python 3.11／Torch 2.8.0+cpu，沿用仓库外 `D:/cached-prefix-cpu-run.py` 的 CPU 导入适配；首次直接 pytest 因 CPU 平台错误触发 CUDA 初始化而收集失败，改用该 runner 后通过，未替代模型／KV 数值运算。日志为 `D:/pr445-adapter-regression.log`、`D:/pr445-adapter-cache-adaln.log`、`D:/pr445-adapter-final-wiring.log`。Ruff、格式、diff 和 CPU inventory 通过；合并产生的重复测试登记已移除，按 #445 的独立 DiT 套件运行。该轮仅执行 CPU 测试；后续 GPU 实测见末节。未部署或合并 PR。

#445 已于本轮适配期间合入 main（`5e66d5e2c321e3f5b188b2aebe292f0371586a8c`），#439 改以 main 为基分支。随后同步 main 的 Source 调度增量；相对已测 `7aaf3ac2ee`，本适配器仅新增主线的 remote source 输入额度 usage 字段，DiT、KV、AdaLN 与联合路由不变。Source adapter/progressive/Decoder overlap/Encoder transport/work quantum 五个完整文件 **982 passed、0 failed，225.40 秒**，日志 `D:/pr445-adapter-main-integration.log`。该组与前述覆盖也有重叠，不相加；不将 CPU 调度测试作为 GPU 性能证据。

### 历史：与 PR #441 的缓存总开关适配（2026-10-28，现由 #445 替代）

依赖 #441 `7031fd76befa71b3f5ea62978fa175889137ab7a` 的 `Block0PreparationCache`。KV pool 启用且 `SGLANG_ZING_BLOCK0_CACHE_MODE` 未设为 `off/0/false` 时创建缓存对象；连续 source 的 cross_chunk 路径据此同时安装消费者和启用切换联合，不再额外要求 `SOURCE_PROMPT_JOINT`。关闭总开关时两者均不安装，保留原执行方式。启动未命中保持 prefix 单独计算后执行 source + target，不新增三分支。

启动命中继续使用 #441 提前发起的复制，在消费者边界等待事件并发布 prefix。切换时按已接纳候选的 conditioning 身份、当前 packed prefix 的 CPU 位置与缓存条目匹配；通过缓存方的跨 rank 一致命中投票后才清理逻辑历史并恢复当前池。命中五分支、未命中六分支。静态表和动态 AdaLN 缓存按每个串行 timestep 分组解析，六分支的混合 prefix 分组使用现有精确表项。

不新增缓存容量、条目或淘汰策略。#441 的单条目仍由启动未命中请求在首个输出后回填；切换未命中不会另建条目，切换回该条目可命中。conditioning 身份包含 prompt 和预处理媒体（含 S0），布局再由实际 prefix 核对；不把不同输入仅因 token 数相同视作命中。

本地 `LocalConditioner` 与远程 conditioner 一样在成功编码／缓存命中后发布 `WorkerConditioningState.identity/positive`；否则本地路径的空身份不足以区分 prompt。编码失败保留原状态，成功 bypass 清除可用身份，消费者遇到缺失身份按 miss 处理，不允许 `None == None` 成为命中依据。

验证计划：组合 #441 的 CPU 缓存恢复与真实小 DiT，覆盖开／关、启动 hit/miss、A→B miss→A hit、身份/位置不匹配、失败释放及后续复用，分别验证 eager/fixed/记录后端和静态表／动态投影模式。GPU 双分辨率、CUDA Graph/NCCL、性能与显存仍未验证。

本轮组合工作树（父提交 `b45af7054a` 与依赖 `7031fd76be`）实测：10 个完整相关文件 **1170 passed、10 skipped、0 failed，317.92 秒**；最终身份发布、缺失身份、bypass/编码失败及 conditioning 回归三个文件 **62 passed、0 failed，44.06 秒**。两组存在重复覆盖，不相加。新增真实 `Block0PreparationCache` CPU copy/commit 与小 DiT 的 cold/warm × eager/fixed/recording 六组 A→B→A 对照通过；static/dynamic AdaLN 的六组未命中六分支对照通过。检查启动 prefix forward 次数、hit/miss、五／六分支实际消费、输出、完整 KV、RNG 与事件；关闭缓存时即使旧 `SOURCE_PROMPT_JOINT=1` 也不安装消费者或启用联合切换。

日志为仓库外 `D:/block0-adapter-full.log`、`D:/block0-adapter-identity2.log`；前置夹具错误（简化输入缺少快照字段、测试 RGB dtype 不符合身份契约）已修正，不计入通过结果。Ruff、isort、格式、diff 和 CPU inventory 检查通过。CPU 记录后端与组合测试不替代真实 CUDA stream/Graph/NCCL 和 480/736 端到端性能证据。

本仓库已有 `ZingH3KVCache` 单写者、`SourceReadySession` 会话所有权、`PrefixJointTransaction`、`PrefixJointGraphs` 和 SP 边界投票；本改动扩展这些入口，保持其他模型、通用调度器和 wire 协议不变。参考本基线 `docs/docs/developer_guide/contribution_guide.mdx` / `scripts/ci/zing/README.md`。这属于 seedleap Zing 扩展，不声称是上游 SGLang 已支持的跨任务缓存接口。

2026-10-27 核对上游 main `e4d4a9a7de5454cd8b854b21af723c2155a55a3d` 的 [BasePrefixCache](https://github.com/sgl-project/sglang/blob/e4d4a9a7de5454cd8b854b21af723c2155a55a3d/python/sglang/srt/mem_cache/base_prefix_cache.py)：其 `match_prefix` 返回命中位置/长度，`init_load_back` 区分装载，pool allocator 与请求生命周期负责资源归属。本次同样保留“缓存方负责命中与放置，执行方消费已就绪状态”的边界，但 Zing 是 raw KV 固定槽，未照搬 LLM 的分页/radix 数据结构。上游合入仍需讨论回调契约；末节已补480/SP2单输入证据，双分辨率和其他GPU路径仍有缺口。

阅读顺序：`source_session.py`/`stage.py` 注入 → `incremental.py` 装载边界与命中路由 → `zing_h3_joint.py` 五分支事务 → `zing_h3_prefix_graph.py` 固定计划 → 对应回归。重点确认回调对条件版本和池租约的绑定、旧状态何时可覆盖、prefix 始终只读、位置映射与串行一致，以及多 rank 失败是否能在进入模型前一致退出。

## 2026-10-28：H100 资源池上的 r8 配置 cache A/B 实测

运行源码为 `0267732e6f947cfb495db7e5db8308e99ff0bc50`，叠加 r8 的 eager merged-attention 实验 overlay；本节后续文档提交未改变受测运行代码。**下列性能属于该组合，不是纯 PR HEAD 默认 attention 的性能。**

### 配置与真实路径

- 资源池称 H100，运行时实际报告 `NVIDIA L20Z` / SM90；Torch `2.13.0+cu130`、CUDA `13.0`。共四卡：Qwen/source 共卡、DiT SP2/Ulysses2、独立 Lynn decoder。
- 用户指定复现 T038 r8-a2：同一输入 `hiker-clay-01-yellow-raincoat`、五张参考图，832×480、360 帧、seed42、24 FPS paced；收到120帧后发 prompt 更新。window6/sink3、block_relative/gap12、defer_final=1、静态 weights_free AdaLN、Qwen六项优化开启，Graph/SP Graph均关闭。window6及Graph关闭是用户指定的历史配置覆盖，与统一默认不同。
- 两组分别 `SGLANG_ZING_BLOCK0_CACHE_MODE=off/overlap`，每组先完整预热一遍，再正式一遍；开启组 L1 两条、无 L2，第一遍写入，第二遍复用同进程条目。
- 两个 DiT rank 的真实日志均确认：开启组第一遍启动 miss 为单独 prefix 后二分支、切换 miss 为六分支；第二遍启动和切换均 hit，无 prefix 重算，分别走二分支和五分支。普通块保留三分支联合。五分支中的三段 history 共五个 latent（1/2/2）。
- 四次请求都成功输出360帧，prompt均在输出帧136生效。两组正式输出各431308800字节，360帧逐帧字节一致，完整 RGB SHA-256 均为 `efbc3ad61ad6b1b17269a4438951866fb6ab9ed003fe8c512cb816b60a83284f`。

### 正式遍整体结果

每组只有一个正式样本；不能据此推为稳定总体加速比。

| 指标 | cache off | cache on 第二遍命中 |
| --- | ---: | ---: |
| TTFF | 2413.30 ms | 1440.71 ms |
| E2E | 18912.76 ms | 16651.42 ms |
| 客户端开始准备更新 → 新 prompt 首帧收到 | 1726.44 ms | 954.55 ms |
| 最大输出批间隔 | 1510.51 ms | 988.68 ms |

### 稳态逐块

输出帧窗口 `[32,112)`、`[208,336)`，排除启动、切换和尾块。每组8帧块11个、9帧块12个；两个rank完整保存，下表取rank0，单位ms。

| 输出块 | off DiT Host均值 / P95 | on DiT Host均值 / P95 | off DiT CUDA均值 / P95 | on DiT CUDA均值 / P95 |
| --- | ---: | ---: | ---: | ---: |
| 8帧 | 265.18 / 268.62 | 266.72 / 274.48 | 280.27 / 285.56 | 281.88 / 290.57 |
| 9帧 | 289.97 / 295.49 | 291.99 / 301.07 | 331.02 / 337.43 | 333.32 / 343.30 |

本次稳态 DiT 近似持平，收益主要发生在启动和切换。source worker Host均值8/9帧块约211.75/292.6 ms，与DiT异步重叠；主侧encoder等待约11.6–16.8 ms，decoder提交约2.5–2.8 ms。未覆盖source worker CUDA和实际decoder worker CUDA，不能把提交时间当作解码计算时间。

### Prompt 切换时间线与内部工作

下表为同一 `CLOCK_MONOTONIC` 域上的Host时间线，三段可组成更新到首帧总延迟。起点为客户端 `prepare_begin`，本例开始准备到提交仅约0.003 ms；第一段结束于采用新prompt的切换DiT dispatch调用进入，覆盖请求传递、候选准备及等待可切换块边界，其间也可能并行发生conditioning预取RPC；**不是纯网络、纯排队或DiT计算时间，也不表示命中路径必须等待RPC**。最后一段仍包含GPU完成、解码、发布/传输及接收剩余工作，不能称为独立decoder耗时。

| Host区间 | off | on |
| --- | ---: | ---: |
| 客户端开始准备更新 → 切换DiT dispatch调用进入 | 323.313 ms | 325.384 ms |
| 切换DiT dispatch Host调用 | 1239.924 ms | 464.513 ms |
| Host返回 → 新prompt首帧收到 | 163.205 ms | 164.651 ms |

嵌套观测：conditioning RPC为246.983/250.923 ms，prompt prepare为39.197/20.194 ms，activate为1026.556/13.111 ms（off/on）。conditioning RPC是DiT侧请求prompt/reference条件张量并取得返回值的调用方等待，包含传输和服务端处理，**不是独立Qwen计算时间**；正式遍没有新增Qwen `encode_device`事件，无法细分其中Qwen计算与传输占比。这些观测与上表有嵌套/重叠，不能再次相加。

已确认的冗余预取：`SourceReadySession._update_prompt`先查询KV缓存，命中分支直接复用`entry.inputs.positive`，只有miss才调用条件编码；主路径不依赖RPC来判定KV命中。但本次配置开启`conditioning_prefetch`，`ConditioningPrefetch._receive`收到更新后直接执行`client.encode`，未先检查KV缓存，因而第二遍命中仍发出一笔推测性RPC。正式遍没有主线程`conditioning_total`调用，后台`conditioning_rpc_encode`事件与此调用链一致。参考受测版本的[source_session.py命中分支](https://github.com/seedleap/sglang/blob/0267732e6f947cfb495db7e5db8308e99ff0bc50/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/zing_h3/source_session.py#L642)和[prefetch.py直接编码](https://github.com/seedleap/sglang/blob/0267732e6f947cfb495db7e5db8308e99ff0bc50/python/sglang/multimodal_gen/runtime/conditioning/prefetch.py#L158)。

这属于prefetch与KV缓存协同的已确认缺口；本轮结果包含该冗余工作，未修复该路径，也未重测消除后的收益。RPC与旧块计算重叠，不能把约251 ms视作命中必需开销，也不能直接从954.55 ms中扣除。后续处理需让预取入口识别可复用准备状态，同时保持SP collective和GPU消费的线程/状态所有权。

后续修复设计：通用`ConditioningPrefetch`接受可选的CPU判定回调；收到更新后，先用与正式路径相同的完整request key检查本rank已发布L1准备或完整L2文件提示，预计命中则跳过推测性RPC，miss保留原先的提前编码。提示不修改LRU、不等待CUDA、不提升磁盘KV、不执行SP collective，也不代替正式命中投票。若提示后发生淘汰、L2损坏或部分rank miss，切换边界仍按原路径查找、验证并回退RPC。共享key构造由source会话持有，缓存方只暴露只读提示，避免后台线程接触活跃KV池。测试需覆盖命中零RPC、miss提前RPC、媒体/策略不同不跳过、淘汰后回退、L2不完整与部分rank缺失；旧r4性能保留为修复前结果，新代码的GPU收益另行验证。


CUDA event细分（rank0，当前stream口径）：

| 工作 | off | on |
| --- | ---: | ---: |
| prefix重算 | 814.736 ms | 无 |
| KV恢复 | 无 | 124.636 ms |
| 串行history提交（三段） | 46.076 + 79.155 + 80.542 ms | 合入五分支 |
| source/active首步 | 二分支143.862 ms | history/source/active五分支260.728 ms |
| 后续active去噪 | 86.250 ms | 89.458 ms |
| 切换dispatch完整CUDA span | 1277.409 ms | 503.123 ms |

KV恢复CPU提交1.615 ms、消费边界Host调用0.709 ms，实际恢复CUDA124.636 ms；`ready_at_consumer=false`，消费stream通过依赖等待恢复，因此不能声称传输已完全隐藏。启动恢复CUDA123.379 ms，提前提交后在消费边界`ready=true`。Host时间线与CUDA event不能跨口径直接累加。

### 证据、运行适配与剩余范围

两组各rank的DiT峰值allocated均72206033920字节（约67.25 GiB）；逐rank reserved和逐块结果保存在原始记录。stage timing健康回执无丢弃、积压或错误。完整RGB一致只覆盖本输入的端到端输出，不替代GPU内部完整KV/latent容差验证。

为复现r8注意力实现，只覆盖下列两文件；规范化Python换行后核对SHA-256，启动前后完整源码哈希一致：

| overlay文件 | SHA-256（运行字节） |
| --- | --- |
| `python/sglang/multimodal_gen/runtime/models/dits/zing_h3.py` | `6ea8e798edb2b45f8f9dc6e741c95399802376b272d063cf37567b397a327622` |
| `python/sglang/multimodal_gen/runtime/models/dits/zing_h3_joint_attention.py`（新增） | `3d787ee218748148c9b5116cf6e02d78dde21e3a9cd4b7ff66cb4c960f9f93fa` |

实验启动入口在导入前限定DiT可见卡，并使用现有launcher单独启动decoder，处理导入期CUDA初始化先于角色分卡的冲突；未修改PR模型或缓存逻辑。前两轮启动失败及第三轮测试标签环境读取错误均保留，不计推理通过。最终脚本补全请求循环CPU预检后才重测成功。

证据标识：`pr439-cache-ab-h100-20261028-r4`；`blocks.csv/json`保留每块及rank，`switches.json`保留完整切换时间线，另有`cache-routes.json`、`ROUTE_ASSERTIONS.json`、`RGB_COMPARISON.json`、`allocator-memory.json`。私有原始归档为 `oss://leap-world-shanghai/world-model/sglang/evals/pr439-cache-ab-h100-20261028-r4/readback.tar.gz`，886702字节，SHA-256 `b663d0c3a0e87cdbdf388dadb53005e9c6b5c01d4f03fdd3b0a8de710bc40255`；RGB及比较回执单独保留在实验目录。无secret、模型权重或RGB大文件提交到仓库。测试Pod已清理，所用4张卡显存归零，未停止其他服务。

仍未验证：纯HEAD默认attention性能、736、CUDA Graph、动态AdaLN的真实GPU路径、L2 GPU恢复、内部完整KV/latent数值、独立Qwen/source/decoder CUDA细分，以及重复样本和长期稳定性。本次结果不能外推这些路径。

## 2026-10-28：跳过命中后的推测性 conditioning RPC

修复提交 `558229554f` 已实现上述 CPU 提示方案；合并 main `d89e38d75c` 后，运行代码受测版本为 `bf65c97431`。用户沿用原缓存总开关，无需新增配置：`off` 保留提前编码；缓存开启且本 rank 存在可复用提示时，后台预取跳过 RPC，正式切换仍由缓存方执行跨 rank 投票、恢复和验证。

`ConditioningPrefetch` 只接受可选 `should_prefetch` 回调，不依赖 Zing 缓存实现。source 会话持有共享 `_block0_update_key`，缓存提供 `may_contain` 只读提示；后台线程不修改 LRU、不等待 CUDA、不执行 collective、不提升 L2。提示失效、淘汰、磁盘条件损坏或部分 rank miss 时，消费边界仍回退条件编码；提示不发布命中状态。关闭会话先结束线程，再释放回调引用。

合并保留 main #452 的 NCCL 私有流预取和 `device` 参数、#454 的 `defer_final` 命令语义与 prefill/live key 一致性；`prompt_joint` 仍受真实缓存和 `cross_chunk` 消费者约束。三个冲突分别位于预取构造参数、source 会话接线及部署文档，未改为 main 的串行缓存消费路径。

CI 的失联 idle rank 测试曾出现一次失败、重试通过。控制面在 IDLE 广播后补充现有 CPU idle group 的接收确认，避免将广播返回当作所有接收者确认；不改测试超时或放宽断言。磁盘条件读取补充损坏 pickle/EOF 的 miss 回退。

`bf65c97431` 本地四个完整文件（prefetch、progressive source、prompt switch cache、disaggregation transport）**777 passed、0 failed，26.58秒**；WSL/Python3.11/Torch2.8 CPU，使用既有仓库外导入适配 runner。覆盖真实 IPC 预取接线、命中零 RPC、miss 提前编码、淘汰/部分 rank 回退、L2 提示不修改状态、损坏条件回退和 CPU idle 确认；不替代实际 NCCL/GPU 性能证明。相关日志：`D:/pr439-prefetch-ci-20261028/bf65-focused.log`。

GitHub run `40693469069` 的 conditioning 套件已通过；DiT shard0 的四个参数化用例因旧测试漏传新增 `command` 参数失败。补齐测试调用后，该 preparation cache 完整文件 **14 passed、0 failed，8.65秒**；未修改生产签名、期望或断言以迁就测试。其余尚未结束的分片不计通过。

GPU 复测 `pr439-cache-ab-h100-20261028-r5` 保留 r8 attention overlay、运行源码 `bf65c97431`，首个请求在 conditioning bridge 握手失败，零输出，不计推理或性能通过。合并主线后，服务端和客户端的缺省 result transport 均由 HTTP 改为 NCCL；旧测试配置客户端 `conditioning_timeout_s=600`、服务端 `request_timeout_s=10`，不满足 NCCL 的连接预算约束而返回400。用户明确继续使用 NCCL，后续显式设置共享配置 `result_transport=nccl`，两端预算统一为600秒；不改生产默认。

测试版本 `r6` 的 HTTP 方案已按用户指示停止，不计结果；`r7` 在推理前停止以修正测试驱动的 NCCL 通道生命周期。NCCL 服务当前绑定单一 DiT bridge，故 `r8` 每组一起启动/停止 Qwen 与 DiT，两遍请求组内复用相同进程，source encoder 继续共用。冻结运行源码哈希一致；`57c180bed7` 相对受测运行版本只改测试和设计文档。NCCL 服务禁用其自身条件结果缓存，两组均使用同一传输配置；不能将本轮与 HTTP r4 的时间差全部归因于本次预取修复。修复前 r4 结果保持原样，新结果待请求完成后补充。

## 2026-10-28：NCCL 与跳过命中 RPC 后的实测

运行源码 bf65c97431 + 原r8 eager attention overlay；57c180bed7 只改测试/设计。硬件运行时报告 NVIDIA L20Z/SM90，Torch2.13.0+cu130。

沿用 T038 r8-a2、同一五参考图输入、832×480/360帧/seed42/24FPS、SP2、window6/sink3、Graph关闭、静态AdaLN。NCCL显式开启，客户端/服务端均600秒。每组 Qwen+DiT 同启停，组内预热一次、正式一次；on=overlap/L1两槽/无L2。

| 指标（ms） | cache off | cache on 第二遍 |
| --- | ---: | ---: |
| TTFF | 2705.206 | 1415.765 |
| E2E | 19733.948 | 16647.684 |
| 更新开始准备→新prompt首帧 | 2208.979 | 894.188 |
| 最大批间隔 | 1953.482 | 1014.361 |
| 准备更新→切换DiT dispatch进入 | 807.765 | 204.241 |
| 切换dispatch Host | 1240.358 | 530.918 |
| Host返回→新prompt首帧 | 160.856 | 159.029 |
| 切换dispatch CUDA | 1277.512 | 568.009 |

Host三段属于同一CLOCK_MONOTONIC域，可组成更新到首帧总延迟；第一段包括请求/候选准备和旧块边界等待，不是纯RPC时间。CUDA与Host分开，嵌套工作不可再次相加。

| rank0 稳态块 | off Host均值/P95 | on Host均值/P95 | off CUDA均值/P95 | on CUDA均值/P95 |
| --- | ---: | ---: | ---: | ---: |
| 8帧 | 265.181/270.739 | 264.676/270.393 | 280.521/287.129 | 279.788/286.473 |
| 9帧 | 290.566/295.809 | 290.984/296.431 | 331.788/338.148 | 332.213/338.752 |

稳态仅取完整块落在[32,112)、[208,336)内，排除启动/切换/尾块；逐块/两rank见blocks.csv。source worker Host、主侧encoder等待、decoder提交分别记录；独立source/decoder CUDA未覆盖。

RPC观测：[{"arm": "cache-off", "case": "cache-off-A-paced-r1", "count": 2}, {"arm": "cache-off", "case": "cache-off-A-paced-r2", "count": 2}, {"arm": "cache-on", "case": "cache-on-A-paced-r1", "count": 2}, {"arm": "cache-on", "case": "cache-on-A-paced-r2", "count": 0}]

完整切换CUDA/恢复细分见FORMAL_BREAKDOWN.json，分支断言见ROUTE_ASSERTIONS.json，内存见allocator-memory.json。

限制：一个输入、每组一个正式样本；未验收736、Graph、动态AdaLN GPU、L2 GPU或内部完整KV/latent容差。NCCL禁用Qwen服务自身结果缓存，与旧HTTP r4不能直接做单因素因果比较。


四次请求均完整360帧，更新都在输出帧136生效。两rank的8组路由断言通过：off保留原串行prefix/history与二分支；on第一遍启动miss先prefix后二分支、切换miss六分支；on第二遍启动/切换两次L1 hit，无prefix重算，二/五分支，普通块仍三分支。NCCL使用`ConditioningClient.encode_device`，统计同时覆盖encode和encode_device：off两遍、on第一遍各2次，on第二遍**0次**，主侧conditioning_total也为0。不是仅根据Qwen未计算推断跳过RPC。

正式两组360帧完整RGB逐字节一致，各431308800字节，SHA-256均`efbc3ad61ad6b1b17269a4438951866fb6ab9ed003fe8c512cb816b60a83284f`。这个输入也与旧r4输出hash一致；不能据此替代内部完整KV/latent容差验证。

切换CUDA细分，rank0，单位ms；当前stream事件span可能包含依赖等待，不当作纯kernel耗时：

| 工作 | off | on第二遍 |
| --- | ---: | ---: |
| prefix重算 | 813.550 | 无 |
| KV恢复 | 无 | 124.638 |
| 三段history串行 | 46.306 + 79.924 + 81.058 | 合入五分支 |
| source/active首步 | 二分支143.669 | 五分支324.301 |
| 后续active | 86.313 | 95.122 |
| 完整dispatch CUDA span | 1277.512 | 568.009 |

off切换conditioning RPC为772.516ms，on无RPC；该数包含实际条件编码和NCCL传输等等待，**不是NCCL握手耗时**。每组只建立一次bridge，组内正式请求复用；HTTP桥接握手和首次NCCL communicator惰性初始化未分别计时，不据此报告握手毫秒数。两端600秒只是失败预算。

on切换KV恢复CPU提交1.722ms、消费调用0.663ms、提前量2.007ms，消费时`ready=false`，实际恢复CUDA124.638ms，仍有依赖等待；启动恢复提前539.068ms，消费时`ready=true`。不能声称恢复传输已完全隐藏。

稳态8/9帧块source worker Host均值off211.360/291.882ms、on211.506/292.045ms；主侧encoder等待均值off12.722/19.669ms、on12.136/23.384ms；decoder提交均值off2.618/2.847ms、on2.491/2.704ms。均与其他阶段异步重叠，独立source/decoder CUDA未计，不以提交时间代替解码计算。稳态DiT近似持平，主要收益在启动/切换。每组8帧块11个、9帧块12个；两rank逐块完整保留。

两组/两rank DiT峰值allocated均72206033920字节（约67.25GiB）；各rank reserved及decoder内存见记录。正式stage timing回执无丢弃、积压或错误。

证据根`D:/pr439-cache-ab-h100-20261028-r8/`：`blocks.csv/json`、`steady-summary.json`、`switches.json`、`FORMAL_BREAKDOWN.json`、`ROUTE_ASSERTIONS.json`、`RPC_ASSERTIONS.json`、`allocator-memory.json`及`readback/RGB_COMPARISON.json`。私有原始包`oss://leap-world-shanghai/world-model/sglang/evals/pr439-cache-ab-h100-20261028-r8/readback.tar.gz`，76文件、886998字节，SHA-256 `23ce59ef14e46ffd587f75909ed7746160a1971d940d1ef529f6d7111dbc7774`。r5失败包已保留；r6/r7取消不计通过。CPU/GPU测试Pod已清理、物理4–7显存0，未动0–3服务。
