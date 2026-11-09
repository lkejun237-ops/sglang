---
title: "上海 main、输出空档与 S0 v2 整体发布回归"
description: "整合 S0 v2、主线性能与当前块优先输出；单 slot 协调切换和完整回归。"
---

## 当前范围

2026-10-27，将远端 `main@387f693f306e601586afdfebae938e823317620a` 合入
`codex/shanghai-output-gap-20261027`。合并前开发提交为 `e58fbcf9f6`，发布分支基线由 `white_box@f2894cda2e` 同步至
`white_box@e09d3ee7c5`（已批准 Atlas CLI pin 与 CI 清单）。本次只做代码整合、本地测试和发布准备；**未推送发布分支、未部署、
未重启模型或共享入口，也未创建云作业**。部署须等待用户明确指令。

保留当前块就绪唤醒、DiT/D1 优先交接、单 latent 渐进 RGB、完整尾帧、source receipt、
启动 prepare/attach（升级为独立尺寸 S0 v2）、CUDA Graph capture 生命周期、三路归档和单实例单活跃会话。
共享内存发布器采用主线通用实现位置，同时保留分片 reserve/publish/ACK/CANCEL 和关闭回收。

## 合入功能及启用策略

| 主线功能 | 生效条件 | 首轮策略与验证边界 |
| --- | --- | --- |
| #405 跨 Graph history RoPE 复用 | `SGLANG_ZING_HISTORY_ROPE_CROSS_GRAPH` 默认 `1`；需满足 fixed/pooled 与 chunk reuse 路由 | 单独比较 `0/1`，新进程预热；历史四步结果不能外推到当前双步 |
| #442 单卡 Lynn fused preparation | SM90/SM103、BF16 及严格形状门控命中即启用 | 当前 D1 会受影响。没有现成独立环境开关；组件 A/B 用父版本与候选模块，不伪造关闭参数，也不将未量化 DiT 当作“所有新算子都关闭” |
| #438 统一分卡启动与传输 | 新 `--encoder-gpus/--denoiser-gpus/--decoder-gpus` 和运行时 role port | 必须迁移 launcher 并核对 GPU UUID/进程归属、端口、清理。框架统一本身不承诺 FPS 提升 |
| #430 prompt codec 边界 | 与条件切换和块规划有关的正确性修复 | 保留完整 codec 边界、请求生效边界、早/晚 prompt 与尾块测试，不能只跑不切换的稳态 |
| #413 SM90 FP8 GEMM 路由、#414 前序融合量化 | DiT 显式采用 online FP8；满足 kernel、精度和布局门控 | 首轮保持 DiT BF16，不隐式开 FP8。FP8 独立实验，固定选层范围、scale 与 kernel 路由；微算子加速不能当整链路收益 |
| #410 Qwen 异步预取 | `--zing-conditioning-prefetch` 默认关闭，remote 单活跃 HTTP 条件路径 | 在基础候选稳定后单项比较 prompt/reference 更新；不是稳态 DiT 每帧加速 |
| #410 source Decoder overlap | `--zing-source-decoder-overlap` 默认关闭，独立 Decoder 且 source quantum 必须 `1` | 现有 q0 渐进模式已有部分并行；先比较 q0/q1，再在 q1 上只切该开关，避免把 quantum 变化混作 overlap 收益 |
| #415 Qwen NCCL 结果通道 | 服务端和客户端 `result_transport: "nccl"`，同机单接收方 | 与 HTTP 预取分开比较。当前 NCCL 路径不会启动 HTTP/CPU 预取器，两项同时配置不代表叠加生效 |
| #440 六分支 prompt joint 与静态/动态 AdaLN 缓存 | `SGLANG_ZING_SOURCE_PROMPT_JOINT=1`；静态表完整性与动态缓存独立门控 | 首轮保留现网关闭状态；后续单项验证 prompt 切换、表项/每层KV、Graph与LoRA边界，不能把冷填充当暖缓存收益 |
| #437 remote Qwen 六项执行契约 | 六项 `SGLANG_H3_QWEN_*` 默认开启，执行选项写入 profile | 升级初轮显式保留原部署六项值（现有成功配方为全关），重建 profile；随后分组验证，FP8 与 BF16 不宣称画质等价 |

实际配置以合并代码为准；#410/#415 的早期说明中“remote 六项必须全关”的限制已由 #437
更新。代码合入不意味着所有开关应同时开启。特别是 NCCL 结果通道与 HTTP 预取不是可直接叠加的组合。

发布分支 #444 的 Room CLI pin 为 `d451ed29a320eea6fa4b8c1699b552590de7ca7c`，已保留在候选源码。
本次 S0 v2 必须同时更新 Web/Node、Room、Model Worker。Room 镜像保留该批准 pin，不顺带升级 SDK；
WTP 当前 main 的 SDK 与播放器 pin 为 `3a7ee0f487ca839c8068544d145076d46052776f`，与 Room pin **不是同一个 SHA**。
本次没有改动两端既有 pin，但不能因此断言与现网完全相同或两端已验收兼容。发布前核对实际线上值与两端媒体/UI/输入协议、World publication/运行时契约。
若 WTP main 的这些批准输入尚未在线运行，须完成仓库 sealed-release 的真实消费者门禁，再纳入整体切换；不能仅为让 SHA 相同而擅改 pin 或绕过边界。

## S0 v2 合并与版本协同

从 `cli和推理并行` 任务的本地未提交改动提取 S0 v2，未修改其工作区：
后端基于 `f2894cda2e` 的 14 文件差异，前端基于 `eec4cab706` 的 5 文件差异。
前端在 WTP `main@8652e940f9` 上整合，保留 UI 在开流前准备、AREA 输入回执与新版 SDK pin。
后端保留 main 的 role port、预取与通用 SHM 生命周期；测试冲突采用两侧测试的并集。

| 组件 | 本次要求 | 保留的边界 |
| --- | --- | --- |
| WTP Web + Node | 声明 `whitebox-startup-prepare-v2`，按图片真实宽高上传 S0；点击时图片未就绪则省略 | UI 预准备、输入/显示回执、历史 World/runtime 身份及原发布资产不变 |
| CPU Room | 校验 v2 S0；条件准备和 CLI 并行；只借一个模型租约；真实 CLI Fk 固定一次再选最新 | 独立 S0 不进入 source 视频；不伪造 PTS、不重录、三路归档与网络编码预算保持 |
| GPU Model Worker | 接收独立尺寸 S0，沿已有 Qwen 选桶/resize；首次 CLI append 不覆盖 S0 | prepare/attach 不推进 RNG、VAE、latent 或 AR 游标，真实 source 仍按固定流几何 |
| Gateway / Coordinator | 无 S0 feature-code 修改；发布前检查准入、路由、epoch 与版本元数据 | 只做切换所需配置，不扩大容量、不改其他服务 |
| Worldkit CLI / SDK | 本次无 S0 feature-code 修改 | 不能借本次发布更换已冻结世界或重建历史产物 |

S0 v2 与 v1 **不能混合启用**。v1 请求应明确拒绝，旧浏览器须刷新后重新申请会话；不带 startup 的 legacy 请求继续受支持。
仅设置前端 `WHITEBOX_MODEL_STARTUP_PREPARE=0` 不等于回滚全部后端性能改动；可作为关闭启动 prepare 的独立降级方式，但正式回滚仍使用冻结的完整旧版本组合。
外部 S0 像素不同不再触发租约释放和二次准入；`condition_preview` 与 `cli_frame_rgb_sha256` 分别证明条件与真实视频起点。

## 资源事实与执行边界

2026-10-27 20:01:25 CST 的其他任务回执已确认：上海只保留 **1 个 slot / 4 GPU**，另一台四卡实例已按用户要求停止。
这是一条历史资源证据，部署前仍须 fresh-read 实际状态。仓库通用清单的“两台 H200 demo”不再代表当前上海容量。
不恢复已释放实例，不假定另一个健康 slot 可以承接用户。既有单 slot 在用户试玩自然结束前不得重启。
用户最新指定：后续只在当前上海 H200 四卡上就地重启和测试，**不使用 H100、B300、其他卡或第二个 GPU 作业**。此要求覆盖下文原先的独立测试资源建议。当前仍仅准备，等待用户明确部署命令后才改优先级、接管进程、部署和在线测试。

准备阶段 GetJob 只读核实 `dlcn1aux7lbh0fff` 仍 Running、RequestGPU=4、Priority=9、EnablePreemptibleJob=false，worker ready且无 active/reserved session。PAI最高优先级为9，故当前无需写入调整；部署前再次核对，若降低则只更新本作业Priority。不开竞价，不修改其他作业、Quota抢占策略或账号权限。
[优先级API](https://help.aliyun.com/en/pai/developer-reference/api-pai-dlc-2020-12-03-updatejob)和[Quota抢占规则](https://help.aliyun.com/zh/pai/resource-quota-quota-preemption-policy)说明：最高优先级与关闭竞价不等于不可被父Quota、平台故障或管理员回收。应保留当前DLC作业/Pod的四卡分配，避免StopJob后重新争抢；无法保证平台级绝对独占。

现有外层controller会在serve子进程退出后结束DLC作业。**不能直接kill模型或使用旧controller从头重跑**。执行窗口内先关闭新准入并确认idle，再用明确归属/start-time校验建立维护监督：暂停原外层controller的退出检查、保留其作业主链存活，然后由新监督者管理当前四卡的模型停启、串行测试、失败回滚及最终常驻；只处理本作业已验证的子进程组。原controller的暂停/接管身份必须留档，不能误恢复其旧退出检查或让旧heartbeat重复注册。若不能证明作业主链会保留，停止部署动作，保留健康旧服务；不要以StopJob重提作为自动回退。维护监督接管与GPU UUID一致性只能在授权执行窗口实机验证，本轮未操作。

## 发布前冻结与兼容预检

1. 重新读取上海实际 CPU/GPU 运行源码、镜像 digest、worker id/epoch、配置、健康、活跃会话与归档状态。
   历史记录只能帮助定位，不能据旧 Job ID 或卡号操作。
2. 冻结合并提交、镜像、PyTorch/CUDA/driver、`seedleap-kernels` wheel/源码与哈希、checkpoint、
   Lynn decoder、参考图和输入 hash。旧权重可复用；不把现有显存常驻进程做原地代码热换。
3. 重建 Qwen conditioning profile，并给服务端与 DiT 使用同一份。即使六项全关，新代码身份也不能
   继续沿用旧 profile。核对 `backend`、`execution_options`、timeout 和实际算子/精度回执；不删字段绕过检查。
4. 在最终 GPU 镜像解释器与 `PYTHONPATH` 上做 CPU 预检、依赖导入和 CLI 参数解析，验证需要的
   kernel 可用以及 BF16 回退路径。Mac CPU 测试不能替代镜像或 GPU 验证。
5. 保留当前 step2500 权重、稳态 `[1000,500]`、初始三块 `[1000,750,500,250]`、17帧 codec/原生 AR
   计划、W6/sink3、SP2、E1+SP2+D1、CLI30、Room输入25帧及网络码率。以发布前实际读回值确认；
   未在本次对照中改变的设置均不随主线默认值漂移。

### Launcher 迁移

原来的 `--zing-dit-gpus` / `--zing-decoder-gpus` 不能直接复用。新增配置使用调用者
`CUDA_VISIBLE_DEVICES` 快照内的索引，并在任何 CUDA 初始化前完成映射。例如保持四卡布局：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 sglang serve --model-path MODEL_PATH \
  --num-gpus 2 --sp-degree 2 --ulysses-degree 2 --ring-degree 1 \
  --encoder-gpus 2 --denoiser-gpus 0,1 --decoder-gpus 3
```

这里只展示角色映射差异，不是完整生产启动命令。其余模型、网络、sampling、KV、conditioning 和
启动准备参数从上次成功配方逐项保留。Qwen 服务仍需明确独立监督者及其 GPU2 归属。
主线 launcher 启动 E1/D1 时，旧外层 supervisor 不得再各自启动一套 E1/D1；同一角色只能有一个
进程 owner。若暂由外层监督者保留 E1，应省略自动 Encoder 启动，并验证 profile 中实际 E1 port；
两种方式分别冻结，不能拼接执行。所有角色的 readiness、SIGTERM/异常清理和下一会话都需验收。

## 性能配置矩阵（仅当前四卡，未启用项留待后续）

本次按用户指定仅使用现有上海 H200 四卡。默认资源建议不适用于本次，不能申请另一台CPU/GPU作业作为前置。依赖与镜像输入先在本机准备；最终解释器/内核预检、GPU与浏览器回归在授权后的同一四卡维护窗执行。构建服务不作为额外推理测试资源。

下表是可选归因目录，不是本次全量开启清单。一次整体发布优先采用保守基础候选：原BF16、原Qwen选项、已实现的渐进交接+S0 v2；FP8/NCCL/额外overlap等未验证项保持关闭，不为试遍开关占用其他卡。基础候选自身的main默认优化按实际门控记录。

| 阶段 | 比较 | 要回答的问题 |
| --- | --- | --- |
| A | 当前已验收发布基线 vs `e58fbcf9f6`，同参数 | 本次 ready-block/D1 交接修复是否缩短约 50/70ms 候选空档，输入/吞吐是否退化 |
| B | A 的胜出项 vs 本次合并基础候选，DiT BF16，Qwen显式原值/HTTP，两个新增异步开关关闭 | 主线启动/传输/正确性与默认优化整合后能否稳定运行；这是整合比较，不能归因于某一个新算子 |
| C | 同一合并候选只切 history cross-graph `0/1`；Lynn另做同latent组件A/B | 分别判断 RoPE 复用和 D1融合的收益与数值边界 |
| D | q0 vs q1；再在 q1 下 overlap关/开 | 新 API overlap 是否比已有渐进路径更优，是否引入零RGB续推或输出背压等待 |
| E | HTTP无预取 vs HTTP预取；另做 HTTP vs NCCL（预取关闭） | 缩短的是初始条件/切换路径还是稳态；不同传输下首次与热态成本分别记录 |
| F | Qwen六项与DiT FP8分别试验，单项通过后再组合 | 精度与画质是否可接受，实际kernel是否命中，端到端收益是否存在 |

六分支 prompt joint/AdaLN 作为额外独立试验：固定其余胜出配置，仅切该功能，并分别记录静态表、动态缓存冷/热态；未通过不影响基础候选发布。

本次只在当前四卡上串行测旧基线与一套整体候选，不执行上表所有可选试验。失败在同一资源上回滚，不通过新申请卡补救。

所有阶段先测 832×480，再测 1280×736；每项沿用英伦 continuous-source benchmark，至少两次预热、
三次正式720帧。模型加载、Graph capture、profiler/阶段计时另行统计，正式性能只用无profiler结果。
每次更改 Graph/精度/路由配置使用新进程与独立cache目录，重新预热，不跨版本复用捕获对象。

FP8以原BF16 checkpoint做online量化，先限定既有 `ffn` 选层范围，另与同FP8配置下
`SGLANG_ZING_FUSED_PRE_LINEAR_QUANT=0/1` 比较融合收益。量化带来的质量变化与融合调度带来的
速度变化分别报告；检查workspace预分配/冻结、scale形状、Graph后零额外重分配和NCCL路由一致。
若开启FP8，不能沿用仅支持未量化checkpoint的 direct GPU weight loading 配方。

## 验收指标与门槛

- 请求完整性：输出帧数、PTS、source/model配对、首尾帧正确；正常链路不新增模型输出丢帧。
- 交互：同动作脚本、同输入序列，按键到对应模型可见帧P50/P95/max与样本数。优先保留P95至少下降10%的
  改动；只有微算子收益而整链路无收益的选项不自动上线。
- 性能：TTFF、E2E、FPS（全程及首包后）、TPOC、ICL；两处模型RGB长间隔的P50/P95与17帧周期。
  显示FPS不应明显下降（初始门槛≥同条件基线97%），冻结次数/时长及显示间隔P95不恶化，队列年龄不持续增长。
- 组件：Encoder内Qwen/参考VAE/sourceVAE、DiT、Decoder各自Host/CUDA中位数/P95/n；逐rank
  allocated/reserved常驻和请求峰值，driver整卡峰值单列。共享卡不冒称组件独占，重叠时长不相加作E2E。
- 浏览器：真实生产播放器至少每场180秒，至少覆盖海藻林、UI复杂场景、运动/纹理复杂场景；每场结束确认
  Slot、GPU会话及归档释放再开下一场。记录实际两路码率、raw/decoded/presented FPS、RTT、buffer、UI跳帧/耗时。
- 画质：同输入同噪声的可比数值/接触表及视频检查，特别关注动作连续、首分钟、切换边界与马赛克；不能只用相同seed证明相同输入。
- 生命周期：prepare有/无S0、S0独立尺寸/不同像素、旧v1拒绝、无startup legacy、prompt/reference早晚更新、部分EOS、取消/断连、慢消费者、
  D1/Encoder故障、超时清理、下一会话。单实例第二会话拒绝且不影响当前会话；资源释放后下一会话可用。

## 一次性整体发布顺序（等待用户指令）

1. **冻结发布单**：SGLang 与 WTP 两个准确提交、三种组件镜像 digest、旧/新 profile、GPU角色/端口、全部有效参数、SDK/归档/资产 pin，保留旧版本完整回滚包。模型权重无需重新训练或复制；旧显存进程不能原地换 Python/CUDA Graph。
2. **窗口外准备**：完成适用 CI、构建并验证镜像中的代码身份、依赖、核函数门控、启动参数与 profile。构建/拉取不算在服务停机目标里。云构建或测试作业同样等待授权，不借这份文档自动提交。
3. **同条件旧基线（仅当前四卡）**：使用现存证据仅作历史参照；正式比较要在授权测试窗口按下文同游戏、动作、时长、网络重新采旧版。若用户选择跳过，候选只能做功能/绝对指标验收，不宣称延迟下降百分比。
4. **排空唯一 slot**：暂停售卖/新准入，等待现有用户自然结束；读回 active session、模型租约、Room 清理和异步归档状态。未知状态先查询，不重复发布或强杀他人会话。
5. **统一更新**：先按上述维护监督步骤保留DLC作业与四卡GPU UUID，仅停止归属明确的模型角色；在准入关闭期间更新 Model Worker + profile，启动并预热；再更新兼容 v2 的 Room、WTP Web/Node 与必要路由元数据。三者顺序可因平台编排调整，但全部就绪前不得开放 v2 公开入口。使用新的 worker epoch 和 revision，旧 ticket/epoch 不得串到新 worker。
6. **私有准入验证**：确认实际源码 SHA、镜像 digest、唯一 worker、4 卡角色映射、profile、健康、v2 session.ready；带独立尺寸 S0、无 S0、legacy 各跑一轮短会话，验证首个真实模型画面、按键和停止释放。仍不开公共准入。
7. **正式候选验收**：完成下节模型两档与浏览器固定矩阵。全部关键门禁通过后恢复公开准入并保持健康常驻；只清理本任务临时资源。不能把 HTTP 200、model.prepared 或旧版本录像当作新版本出帧证明。

**停机估算**：这是单 slot 协调重启，不承诺热更新/零停机。旧版本已有约 8 分钟模型加载及准备记录，仅供维护窗参考；新版本 Graph/profile 和两档回归时间需实测。构建、拉取、资源排队、正式测试另计，不能承诺固定 10 分钟完成。

### 用户指定的四项必测与首帧对照

1. **传首帧/不传首帧均可用**：同一游戏、同一配置分别执行。传图覆盖原尺寸及960×540独立尺寸、S0与真实CLI画面不同；不传图只省略initial_frame，保持prepare v2开启，不能通过关闭整个prepare功能冒充对照。额外保留legacy无startup、取消/重连/下一会话与三路归档检查。
2. **用户提前看到模型画面多久**：以浏览器同一单调时钟，从点击开始（含鉴权、建连）到首次实际呈现模型画面；同时记录首个白模可见帧、model.prepared、首RGB和UI准备。`model.prepared`和解码完成均不能替代实际呈现。主游戏至少5组成对试验，交替provided/omitted顺序；两种模式都允许预取同一张已发布截图，只在发送时决定是否带S0，防止把本地图片加载差异当模型并行收益。必须读回startup request/receipt证明实际分组；强制provided但图片未就绪的失败不能算成omitted成功。
3. **按键→对应可见模型帧**：长测每场至少30次有效按下，按input sequence/source身份匹配，报告P50/P95/max/n和未匹配数；包含首分钟、后半程、长按/短按与松开，人工复核真实动作，不以输入回执声称语义动作已完成。
4. **白模/模型FPS**：分别报告CLI源端、Room实际入模、模型raw输出、浏览器decoded/presented。以同一首模型画面后190秒窗口计数除实际时长，并列1秒窗口P50/P95/min和卡顿/帧年龄；不能直接比较不同3秒刷新窗口的末值。保留实际两路码率、RTT、UI耗时与跳帧。

首帧收益逐对计算 `omitted首模型可见ms - provided首模型可见ms`，给出每对、P50、均值、范围和样本数；正数表示传图更早，负数如实报告。P95仅作带样本数的描述，不能把5次当高置信估计。世界资源冷/热、首张Qwen条件冷/热分别分组；以缓存命中/实际下载字节和运行回执判定。不能把一个冷启动与一个热启动相减，不清理别人缓存来制造“冷”样本；若无可比冷组，报告缺项而不编造收益。

浏览器使用生产 `WhiteboxPlaytestPage`、原UI预准备、正式输入/合成/呈现链路。测试bundle仅增加观察点与`?s0=provided|omitted`，不改正式页面默认；单页单场，必须人工/工具点击“开始试玩”，每场结束确认slot释放再进行下一场。主游戏首组与至少三个其他游戏的两种模式各运行190秒以上，其余启动重复可单列短测，短测不计入稳定性验收。历史10场自动循环不恢复。

本轮已在本地准备v2 GPU benchmark adapter，传递preview宽高并保留首个真实source字节；旧R8硬编码v1脚本不得使用。更换分辨率所需的profile/Graph重建在同一四卡串行执行，完成736后恢复480常驻配置并再次验证。

### 整体回归清单与采样

| 层级 | 用例与口径 | 放行要求 |
| --- | --- | --- |
| S0 协议 | 有图：832×480、960×540、竖图；无图/图片点击时未加载；不同 CLI 像素；非法尺寸、2 MiB/像素上限、hash/epoch/profile冲突；prepare重试/attach前append/v1 | 合法条件只准备一次；异常明确拒绝；真实 CLI 首帧与首 PTS 保留；无 RNG/帧游标提前推进；每轮释放成功 |
| GPU 功能 | E1+SP2+D1、legacy/provided/late 三路径；17帧完整周期、短尾块、EOS同包/后送、prompt/reference切换、D1/Encoder失败及恢复 | 正常输出完整，PTS/配对连续；取消不残留 SHM/credit/租约；下一会话可用；故障注入仅维护窗内本任务私有会话 |
| GPU 性能 | 832×480、1280×736各两次预热+三次720帧正式；与同硬件旧版固定权重/输入/噪声/参数比较 | 表格列 TTFF/E2E/FPS/TPOC/ICL、source→RGB、输出间隔、credit wait、tail drain；FPS≥基线97%，ICL/长空档不恶化；736独立判定，不要求假装达到24fps |
| 分组件 | Encoder(Qwen/参考VAE/sourceVAE)、DiT、D1，Host/CUDA P50/P95/n，逐rank常驻/峰值allocated/reserved，driver整卡峰值 | 正式性能不挂profiler；独立诊断保留原trace；共享/嵌套/重叠项不相加，不把无法归因填0 |
| 浏览器主矩阵 | 海藻林、庭院、UI复杂、纹理/快速运动至少4个不同游戏；旧/新使用相同版本、相同动作，每场从首个模型画面起190秒，按键样本≥30 | CLI/raw/received/decoded/presented各自FPS、两路实际码率min/mean/max、按键到匹配可见模型帧P50/P95/max/n、RTT、buffer/帧年龄、UI跳帧/耗时、freeze与缺帧；未匹配数单列 |
| 启动/画质 | provided/late各至少3次；冷世界资源与热缓存分组，模型常驻启动单列；首分钟及后半程可见截图/录像 | 分开记录资源准备、CLI首帧、prepare完成、attach、首RGB、首次模型呈现；检查预览图和实际世界相符、动作/镜头/首尾帧、马赛克；不存在S0被插入录像 |
| 网络/生命周期 | 正常网络；受控客户端弱网短抖动后恢复；断连/重连；B→A→B；另一次模型段；第二用户被拒绝 | 不改Mac代理/共享网关路由；弱网测试只限本任务获准的客户端会话；恢复后帧年龄回落，无随时间增长的队列；真实输入回执/UI/三路OSS归档正确 |

同一次按键必须使用 input id/source frame id/呈现回执关联；UI 指标刷新或不同窗口末次值不能拼接成耗时总和。按键应发生在首分钟与后半程，含短按、长按、转向与松开。常规路径不新增丢帧；原有弱网追赶丢帧策略按原因单列，不算作模型漏输出。
正式收益目标为按键 P95 至少下降 10%，同时有效模型/呈现 FPS ≥基线97%、显示间隔 P95与freeze不恶化、画质无新增退化。S0只改变启动时序，不能将稳态收益归给S0；未达性能收益目标但功能通过时明确报告，不自动宣称“最佳配置”。
用户之前停止过10场压测，本轮不擅自恢复它；上述4游戏为整体回归建议，未来执行按本次部署指令确定时段和范围。每场后核对唯一slot释放再开下一场。

### 回滚

任一协议不匹配、无首帧、帧丢失/错配、资源不能释放或明显性能/画质回归，立即停止候选新准入，保存本轮证据。等或清理本任务自身会话后，恢复冻结的 **Web/Node + Room + Worker + profile + routing** 兼容旧组合，再验证健康、真实出帧和下一会话，最后恢复准入。不得只回滚Worker却让Node继续宣告v2。
旧权重、旧镜像、缓存、OSS录像和其他任务保持。NCCL故障按契约重启对应组，不在坏communicator里盲重试。回滚和切换结果未知时先查询实际镜像、epoch和部署记录，不重复提交。

## 本次证据边界

本轮完成的是 main + 输出空档 + S0 v2 本地整合。后端 35 文件首轮 2,376 通过 / 190 CUDA 跳过，新增 S0 与渐进调度组合后该文件 728 通过（其中仅 1 个为新增用例），额外 32 个不重叠 Room 文件 379 通过 / 27 原生环境跳过；去重合计 **2,756 通过 / 217 跳过**。前端试玩相关 **230 通过**，TypeScript、生产构建、sealed SDK 输入只读检查通过；真实 Node builder → MessagePack → Room 解析的 4 个本地协议样例通过。

最初 Room 扩展测试被 Mac 模型测试 bootstrap 的 spawn 重入拖慢，导致录像子进程触发原有时限；采用父进程加载模型测试依赖、子进程使用正式 lightweight Room 入口后 379 项全部通过，生产超时/录像代码未修改。不能把初次失败隐藏成一次全绿，也不能把此修正当作生产性能优化。

所有实际命令、原始日志、来源补丁哈希和限制存于本地 `reports/20261027-shanghai-s0-v2-integration/` 验收包。GPU测试已准备本地v2客户端派生件（原尺寸S0/无S0/独立预览文件）；4项工具单测通过，但仍须在最终镜像验证真实握手。浏览器验收bundle已更新到正式页面和v2分组、包含鉴权的点击计时。旧R8历史报表不作为本次结果。

CUDA/NCCL、SM90 FP8、Lynn融合、两档GPU性能与显存、
生产浏览器端到端收益均须按上述矩阵补测。历史PR中的单项测试、不同配置/硬件或算子倍率不作为合并版上线证明。
