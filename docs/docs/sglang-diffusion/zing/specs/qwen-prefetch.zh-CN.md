# Qwen 提前编码与独立 Decoder 的跨块重叠

本PR提供两个默认关闭、可分别A/B的优化：`--zing-conditioning-prefetch`提前处理已经收到的prompt/reference，要求remote conditioning与单活跃会话；`--zing-source-decoder-overlap`让独立Decoder/输出与下一DiT块重叠，要求独立Decoder与source quantum=1。新目标四卡布局为DiT两卡、Source Encoder＋Qwen一卡、Lynn Decoder一卡；不预测未来prompt，不自动迁移或拉起服务。前一阶段的独立Qwen一卡、Decoder仍在DiT卡上的布局也可仅开启Qwen预取。

## 使用

沿用已有 conditioning profile 和服务配置。GPU编号均为进程可见编号，示例假设本节点四卡；Source Encoder 沿用已配置的独立 worker/profile（GPU2，1个rank），Lynn仍在DiT进程组。

```bash
# 当前 remote native profile 仅允许显式全关的 BF16 / torch_sdpa 基线。
# 创建 profile 与启动 Qwen 服务的 shell 均须使用这六项设置。
export SGLANG_H3_QWEN_HF_RMSNORM=0
export SGLANG_H3_QWEN_FUSED_MROPE=0
export SGLANG_H3_QWEN_FUSED_GATE_UP=0
export SGLANG_H3_QWEN_TEXT_FA4=0
export SGLANG_H3_QWEN_VISION_FA4=0
export SGLANG_H3_QWEN_FP8=0

# Qwen 服务：profile backend 必须是 torch_sdpa，TP1。
CUDA_VISIBLE_DEVICES=3 python -m sglang.multimodal_gen.runtime.conditioning.cli serve \
  --config conditioning-service.json --profile conditioning-profile.json

# DiT＋Decoder：保留实际部署的其余参数与优化设置。
CUDA_VISIBLE_DEVICES=0,1 sglang serve --model-path "$MODEL" \
  --num-gpus 2 --sp-degree 2 --ulysses-degree 2 \
  --zing-conditioning-mode remote --zing-conditioning-prefetch \
  --conditioning-service-url http://127.0.0.1:8093 \
  --conditioning-config conditioning-client.json \
  --conditioning-expected-profile conditioning-profile.json \
  --realtime-max-sessions 1 --realtime-max-sessions-per-worker 1
```

main #404 将本地H3 Qwen六项优化改为默认开启，但当前remote profile尚未包含这些执行选项。上例显式全关是现有RPC身份契约的要求；变量未设置或任意项开启会以CONDITION_PROFILE_MISMATCH拒绝启动。须按当前源码重建/核对BF16、torch_sdpa profile，不沿用旧implementation identity，也不将六项优化混入本PR的remote对照。

示例不替代模型所需的其他部署参数。conditioning服务、client配置和profile按已有RPC文档创建；配置路径由管理员提供，令牌不出现在日志/请求。Source Encoder也不由新开关自动启动。未加开关时保持原同步行为；移除开关即可在新会话恢复同步RPC。

## 为什么需要一条私有控制通道

原路径是 API 接收prompt→下一次dispatch→worker归一化媒体→同步Qwen RPC→SP广播→prefix重建。仅把Qwen搬到别的GPU，仍要在dispatch内等待，无法利用当前DiT块剩余时间。

API现有pending command负责更新顺序；原始reference字节是会话内存资源，归一化后的active媒体和S0由worker持有。若直接在API编码，必须重复引入媒体归一化与active媒体发布/确认状态。为避免这份重复状态，本实现用会话临时目录下随机IPC端点，把完整prompt command（单帧MessagePack）提前发给leader的后台线程；线程复用worker既有媒体和Qwen client，返回结果不经过API、不写临时文件或结果服务。

私有IPC端点随内部source command送到worker，不是公网协议参数。输入最大字节数沿用source transport预算，发送中的包和唯一最新待发包分别有RAM credit。ZMQ CONFLATE只保留最新等待消息，单次GPU编码不强行取消；过时工作可能完成，但不能绕过正式采用检查。

## 调用链及所有权

1. `ZingH3SourceStreamAdapter._ingest_prompt_event` 完成既有校验/媒体准入，保留现有pending/selected command与ACK语义，额外提交私有预取包。Publisher不阻塞接收事件，至多一个在途发送和一个最新待发包。
2. `ZingSourceStreamState.initialize` 在首轮条件准备后，仅leader为该会话创建一个 `ConditioningPrefetch` 接收线程。不会让尚无S0/active媒体的初始请求偷跑条件编码。初始Qwen仍有首块依赖。
3. 后台对仅文本更新读取现有active condition_media；换reference时复用 `ZingH3Codecs.prepare_reference_media` 的CPU解码、bucket和几何逻辑，再调用既有client.encode。这里没有VAE/DiT/CUDA计算、SP collective或active状态更新。
4. `RemoteConditioner.__call__` 仍在原dispatch边界校验真实归一化输入的digest并进行SP共识。leader经同一个锁取预取结果；同一digest命中就复用，已在途的编码只等剩余工作。无匹配结果时走原同步encode。digest绑定profile、prompt、媒体内容及顺序，不能只按event_id命中。
5. CPU结果只缓存一份；H2D、SP广播、容量准入和既有 `update_prompt/_activate_pending_prompt` 仍由主线程执行。旧DiT块保持旧条件，输出仍带旧event标识；不额外生成旧块来隐藏等待。
6. 新reference的VAE编码仍在原主线程路径，且与Qwen使用同一CPU媒体准备方法；因此本优化只隐藏Qwen及CPU准备，不声称隐藏全部reference更新成本。

从输入到达时算，最多可隐藏当前在途DiT块的剩余耗时。first-new-RGB还需要剩余Qwen/交付、prefix重建、新块去噪及解码，不能把RPC提前提交或命中次数当作等量端到端收益。

## 失效、失败与清理

不新增prompt版本计数器、active媒体副本或候选注册表。边界重算真实输入digest，背景使用旧active reference或过时prompt的结果不会被误采用。后台准备失败只记录内部日志，正式边界继续既有校验与非致命更新错误行为；不吞掉正式路径错误。

后台和边界fallback共用RPC锁，避免同会话并发两次client.encode。每个会话只有一个接收/编码线程和一份CPU结果；不跨会话共享结果。state dispose等待有截止时间的RPC排空，随后释放CPU结果与媒体引用，API取消发送并归还credit、关闭socket。若预取排空超时，仍执行Source内部资源清理与远端Decoder abort路径，再报告清理异常；Conditioning状态的active特征/hash也在finally释放。已进入CPU prepare的调用持有局部active媒体引用，停止后不再启动RPC或提交结果。上次线程尚未退出时拒绝启动新的预取owner。Decoder/KV/Source Encoder仍按已有顺序清理；正常退出先等待预取线程；超时时保留live owner拒绝新的预取会话，不让这次失败跳过其余资源释放。

通过result usage的 `zing_conditioning_prefetch` 查看leader累计encodes/hits/failures/fallbacks；非leader为null。这里的hit只表示复用了正确CPU结果，不代表完整GPU重叠或性能验收。

## CPU验证与未覆盖路径

在main `a99560cf90` 上，Linux WSL（Python3.12、Torch2.14.1+cpu）conditioning整组35项通过（124.50s，无skip）。新增12项预取测试在两个独立进程通过（5.54s、4.51s），覆盖真实IPC、提前RPC、边界等待无重复RPC、错媒体不命中、最新消息合并、总超时预算、清理和失败保留旧条件。API/source adapter接线检查提前发布、清理与配置准入。较早rebase前的882项组合不作为当前完整回归证据。

Decoder测试及480全异步结果见下文。真实reference替换的CPU归一化→预取→边界digest整链、SP2取消/故障后下一会话恢复、极慢媒体准备超过close等待上限仍未端到端验收。CPU stand-in不替代真实GPU与分布式生命周期验证。两个开关均默认关闭；提交评审不代表这些缺项已通过。

本次提交收尾的独立review发现预取排空失败会跳过Source内部资源释放，已用finally保证后续清理并保留live owner和异常。增加3项针对性回归后，在同一WSL CPU环境重跑prefetch、progressive source session、Decoder overlap三个文件：689项通过（33.02s，0失败/0跳过；1条anyio已导入的pytest改写警告）。使用仓库`benchmark/zing_h3_causal/run_conditioning_cpu_tests.py`，JUnit保留于实验根目录`release-regression.xml`。这次清理修复只完成CPU验证，未重跑GPU；独立review复核后未发现新的阻塞问题。

## 社区与review路线

沿用SGLang已有独立conditioning RPC和单会话生命周期，不增加第二套部署编排。官方disaggregation已有按角色拆分与P2P传输；本次IPC只提前运送prompt控制与图片输入，结果复用现有RPC路径。它是H3持续source的专用优化，不把通用role拆分宣称为已支持此mid-stream重叠。参考：[官方disaggregation文档](https://github.com/sgl-project/sglang/blob/main/docs/docs/sglang-diffusion/disaggregation.mdx)。

建议阅读：本文→source adapter（输入/预算）→conditioning/prefetch.py（有限通道与CPU结果）→conditioning/worker.py（digest与SP边界）→source_session.py/codecs.py（媒体共享及生命周期）→新增IPC和现有source/reference测试。review重点是预算覆盖真实持有、归一化一致性、停止后无迟到采用、RPC deadline与SP超时、以及后台没有SP collective/GPU状态修改。

## 独立 Decoder 与下一 DiT 块重叠（2026-10-25）

本次在同一PR新增`--zing-source-decoder-overlap`，默认关闭，可与Qwen预取分别A/B。目标布局为GPU0/1 DiT SP2、GPU2 Source Encoder＋Qwen、GPU3独立Lynn Decoder；该开关不迁移GPU或拉起服务。要求单活跃会话、`SGLANG_ZING_EXPERIMENTAL_SOURCE_WORK_QUANTUM=1`，并通过现有`--zing-decoder-gpus`启动独立Decoder，或在部署profile配置`source_decoder.remote`。非独立Decoder或quantum非1在source初始化时拒绝，不改变其他模型/本地Decoder路径。

保留实际部署其余参数，DiT/API服务添加`--zing-source-decoder-overlap --realtime-max-sessions 1 --realtime-max-sessions-per-worker 1`；如需同时提前Qwen，再加前文remote配置及`--zing-conditioning-prefetch`。移除Decoder重叠开关即可恢复q1下先发完RGB再dispatch的路径。显式非零`realtime_interactive_event_grace_ms`仍遵守客户端约定：先完成前块输出，再等待事件窗口，不静默绕过它来伪造重叠。

### 为什么改 API，为什么不删除所有等待

T033的独立Decoder已接受latent后返回共享内存引用；但API每轮开始的`_wait_for_source_chunk_output`等待该引用变成RGB并完成发送，导致下一块DiT串行。新增开关仅让符合条件的source adapter跳过这一次前置等待，保留每轮结束发布下一个send任务前的等待。

因此任意时刻最多拥有一个旧块decode/send任务和一个正在执行或刚完成的下一DiT结果，DiT/KV仍串行、输出仍严格按块序发送。慢客户端会阻塞第二块结果的交付，不能继续第三块DiT；不新增队列、缓存容量或预先生成的旧prompt块。Qwen预取与Decoder重叠没有强制依赖，可独立比较。

RemoteLynnDecoder.append经bridge发送latent并收到服务端接收确认后返回；独立worker持有接收到的tensor、记录stream事件并在自己的decode线程消费。下一append仍遵守`wait_for_transfer`，publisher的两个共享内存slot保持现有容量。API不能因为自己的worker RPC返回就取消旧输出引用；旧引用由旧send任务及其done cleanup持有，新结果在等待旧send时由`current_result`持有。取消先排空已提交worker RPC，再取消旧输出和未交付新引用，沿用既有会话清理。

### 新增的失败路径

旧输出失败可能无法归还source credit。开启重叠后若API正等新source输入，不能无限等待该credit：`_wait_for_source_input`同时观察旧输出任务和输入就绪，旧输出失败时取消输入等待并交回原错误清理路径。旧输出成功时只结束对错误的监视，仍等真正输入就绪。主loop不创建第二个worker RPC，也不在输入等待结束后吞掉旧输出异常。

Prompt更新在下次真正dispatch时按现有pending/selected语义选择，结果使用各自batch/event/provenance，不从可变active prompt重写旧输出；在途GPU工作不虚假取消。一次事件可能在旧RGB发送前生效于下一块，这是重叠带来的预期边界变化，而不是重排旧帧。是否应额外限制客户端交互前视是独立产品语义，现有显式grace仍保留。

### 验证与阅读入口

CPU测试通过受控阻塞的旧输出调用真实`_generate_loop_local`，检查开/关行为、旧输出尚未完成时下一DiT已开始、发送顺序/event身份、慢客户端下最多提前一块、失败唤醒source输入等待，以及取消排空worker并释放两份输出。Source adapter测试覆盖配置拒绝和独立Decoder/q1正例。测试不把fake worker视为GPU重叠证据；真实bridge、SP2、480/736吞吐/切换延迟及显存仍须同条件GPU验证，T033只是修改前的证据。

阅读顺序：server_args新开关→source adapter的部署准入→API `_wait_for_source_chunk_output/_wait_for_source_input/_generate_loop_local`→既有`RemoteLynnDecoder.append`、`_DecodeService.wait_for_transfer`和publisher→`test_zing_source_decoder_overlap.py`（已注册Zing CPU清单）。2026-10-25：API/source adapter/runtime组合290项通过（121.70s），扩展后的关键路径7项通过（76.84s）；并将其中6项重叠测试整理到最终独立文件，重跑6项通过（97.12s，含加强的最多提前一块断言），均无skip。CPU清单检查通过，不能替代GPU性能验收。当前新增逻辑集中于API调度及配置，未改Decoder模型/算子、SP collective或latent传输协议。

## 布局选择及范围

首版采用DiT两卡、Source Encoder＋Qwen一卡、独立Decoder一卡。T033修改前的四卡测量发现独立Decoder与下一DiT没有Host重叠，推动本PR将等待移动到下一次输出交付前。该历史测量使用L20X/SM90与step2500模型，不能与下面L20D/Thai r10结果直接计算收益。

Qwen继续与Encoder共卡，复用已有部署。Qwen与Decoder共卡尚未测量，当前也没有解码优先、可暂停Qwen的调度契约；不能因Decoder执行较短就保证整次Qwen不推迟RGB交付。本PR不新增共卡抢占策略，也不扩大缓存隐藏切换停顿。

本PR只交付Qwen预取与Decoder跨块重叠。Source结果槽位、正式输入水位推进、零RGB轮次和重复原始帧传输的调度优化留待独立PR，不在本PR实施或作为通过条件。

## 2026-10-25 B300四卡全异步实测

用户将本轮范围收敛为所有异步全开，沿上次三个案例各一次预热、一次正式FPS、一次独立Host/CUDA诊断，不执行上述OFF/ON或新布局矩阵。GPU作业`dlc111jbbfwzu2y1`和零卡导出`dlc1gkwkqzoblzja`均Succeeded；CPU预检`dlc1uqagp7g6s068`18/18无skip，新增真实prefix挂点调用通过。前次r3观测脚本闭包误绑导致首次warmup失败，已修复并完整重跑，不计入性能样本。

受测产品SHA `428cd4829515c2af9befe2bb59488e163f97a141`，受测时使用完整PR源码；后续收尾新增预取超时清理修复及CPU回归，未重跑GPU；额外包含T033部署覆盖：H3 Qwen config/model、native六项执行profile扩展、CUDA UUID映射、decoder_worker、remote_decoder、benchmark包init。逐文件前后SHA256保留在`SOURCE_DIFF.json`，这不是纯HEAD无覆盖GPU验收。实际设备NVIDIA L20D/CC10.3，每卡267.69GiB；Thai r10模型，不与H200 step2500直接计算加速。

布局GPU0/1 DiT SP2、GPU2 Source Encoder＋Qwen、GPU3独立Lynn Decoder。832×480/360帧/seed42；三个异步开关、Qwen六项、生产prompt_joint全开。Qwen服务实际加载300个FP8 linear层。Graph OFF、W6/S3、q1/defer-final1沿前次条件，与统一基线W21/Graph默认不同；block_relative/gap12/after_step0、prefix容量12288；初始三块4步，随后2步。CURRENT_KV/HISTORY_ROPE/CURRENT_ROPE/QK_NORM fused计数实际增加，capture/replay/warmup均0。

|案例|首批后FPS（含切换）|含TTFF整体FPS|TTFF秒|prompt到首个新RGB秒|最大批间隔秒|0.5秒虚拟缓冲下累计停顿秒|
|---|---:|---:|---:|---:|---:|---:|
|雨衣|24.10|20.28|2.856|1.244|1.031|0.095|
|温室|23.55|19.58|3.146|1.470|1.264|0.291|
|树林|23.60|19.57|3.186|1.393|1.180|0.289|

FPS为客户端完整批到达率，paced输入上限24FPS，首批后口径359/(末批−首批)，含prompt停顿和尾排空。三例在已收128帧时发prompt，新event从136帧采用；全部360帧提交/接纳/消费/收到一致，协议检查通过。三个正式预取均encodes=hits=1、failures=fallbacks=0。虚拟播放沿用原0.5秒缓冲，未扩大；不是实际浏览器测量，不能称无卡顿。

稳态按8帧/9帧分别统计，排除初始4步、首新prompt和尾padding；三例合并n=60/57，以下为P50/P95 ms：

|阶段|8帧|9帧|
|---|---:|---:|
|Source CUDA按块求和|158.50/158.79|238.38/239.18|
|Source服务含传输按块求和|159.71/160.18|240.66/242.04|
|DiT rank0 CUDA块驻留|219.58/229.24|230.24/238.69|
|Decoder CUDA核心|69.27/70.81|103.34/104.97|
|Decoder开始到RGB发布|75.09/77.93|109.54/113.60|
|Decoder队列|0.27/0.34|0.28/0.39|

两段Source编码组成一个9帧块；跨块DiT执行预算含下一块首预测，不相加各重叠阶段。Source服务17帧一对P50合计约400ms，相对24FPS的708ms供给周期有余量。正式193条leader队列记录pending_results=0–2、result_bytes=0–599040，此前将source_wait_host_ms最大8.977ms解释为稳态Encoder等待的结论撤回：它实际是会话首帧接收等待。具体空泡应按source_encoder_wait_and_receive和编码完成/消费时间判断；prefetched_frames不等于ready latent，最低队列为0，不能声称切换永远有冗余。

Decoder与下一块DiT的Host区间交集为22/43、21/43、21/43个转换，已真实发生重叠，其余转换无交集，不宣称全隐藏。Qwen后台完整RPC约532/690/632ms；Qwen服务总Host约296/342/342ms，encode CUDA约233/280/278ms；服务与旧DiT Host交集只有43/26/36ms。边界condition仍等待/广播426/587/526ms，说明hit并不等于已完成，更不等于零等待。

第一例诊断实测关键路径：发送/排队128.21ms→边界condition等待/广播425.73ms→到prefix准备80.64ms→prefix＋首预测220.14ms→剩余DiT245.05ms→提交/异步解码/返回138.44ms，总1238.21ms。后台RPC约发送后12–544ms，与边界等待嵌套。末段包含交接7.79、Decoder队列0.29、解码到RGB113.94、发布到客户端16.42ms。Host调用起止与CUDA时长分开记录；不拿CUDA时长伪造跨GPU起止。

reference VAE初始调用n=34，Host P50/P95=45.61/60.92ms，CUDA=45.75/61.01ms。三例只切文本，未覆盖reference切换。NVML1Hz逻辑GPU0/1/2/3峰值114.81/113.86/38.37/46.44GiB，GPU2为共享合计；采样从Encoder/Qwen ready后开始，不覆盖其完整加载峰值。

结论：暂保留Qwen＋Encoder、独立Decoder。本次480首批后到达率约23.55–24.10FPS，Decoder确实与下一DiT重叠；稳态仍有周期空泡，prompt切换仍有可见等待。Qwen服务期间没有Source服务区间交集，不能推出同卡无争用；Qwen＋Decoder尚未测，其Qwen长工作可能推迟旧RGB交付，不能仅凭Decoder较短就保证更优。当前数据支持该布局作为首版，不构成两种布局胜负或性能提升比例。

证据根目录`D:/sglang-all-async-b300-20261025-r4/completed/`：REPORT.md、summary.json、timeline.json、四份逐段CSV、原始日志/counters/events、steady-pipeline和prompt-pipeline PNG/SVG及逐条figure-data。OSS `oss://leap-worldmodel-thailand-single-az/world-model/sglang/evals/sglang-all-async-b300-20261025-r4/`，98归档文件逐项hash通过，三视频完整解码各360帧并检查联系表。原始RGB保留云端，不提交仓库。

验证限制：每例仅一次正式和诊断；块P95不是重复请求P95。无A/B、不报加速比；736、reference切换、事件早晚扫描、真实GPU取消/故障/下一会话恢复和纯HEAD数值等价仍未验收。单组视频不能证明画质等价。该次记录不授权扩大矩阵。

## 已知调度限制

诊断中相邻DiT块仍出现约174ms和171ms空泡。一个样例中所需Source latent在411ms已编码完成，前一DiT于467ms结束，但下一DiT直到641ms开始：正式接纳水位停在52帧，一次无新增输入的续推只消费1帧对应latent，随后零RGB结果仍在API轮尾等待旧输出；下一轮接纳后才能消费余下8帧。这里既有正式水位/调度/复制开销，也有约71ms旧RGB等待，不能归因为输入24FPS或Encoder计算不足。

Encoder两份slice额度也会在8帧＋1帧占满后暂停生产；本PR保留现有额度与VAE分片。`source_wait_host_ms`实际计量会话首帧接收等待，不能据其最大8.977ms判断每块Encoder等待或稳态供给冗余。正确归因需关联编码完成、正式接纳、消费与DiT开始时间。

上述限制不否定已观测到的部分Qwen/Decoder重叠，也不意味着当前流水线已完全打通。后续调度修正另行设计、实现与验收；本PR不包含三槽、零RGB快进或metadata接纳协议。
