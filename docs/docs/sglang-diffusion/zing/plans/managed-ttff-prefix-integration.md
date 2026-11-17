# 并行启动与 TTFF / prefix 综合测试版本

目标仅 test-shanghai-copy。用户要求在克君当前已部署开发分支上整合 #662/#687，另建分支后部署。
现网 Room/Gateway 镜像 source 标签为 `5ff10650577a8d05e8a0dd8f6d07c2fba327a66d`，
branch `codex/managed-parallel-startup-20261112`，SDK `d67107d90421d3645abd83bd2628874487e93268`。
镜像和即时实例读回吻合，冻结该基线。模型 cachetiers `3d83b8cd...` 属于另外分支，未把它误当此 Room 分支。

新分支 `codex/managed-parallel-startup-ttff-prefix` 合入 `0a2e5e2bd4ccf8ccb671863d6f472e1917aa6002`
（#662/#687 的 main 整合及 deferred-KV 修复）。不修改克君原分支、main 或原 PR head。
不是旧 baseline 单独计时开关 A/B；保留并行 CLI/condition 准备和原子 prepared child handoff。

## 使用和接口

Room 环境 `WHITEBOX_MODEL_STARTUP_PREPARE=1` 与 `WHITEBOX_DETAILED_TIMING=1`。
Encoder `SGLANG_ZING_QWEN_PREFIX_BYTES=4294967296` 和已挂载的 metrics token file；
prefix 指标独立于 TTFF。保持现网 SDK 与模型1027、480档；736未验证。
正式采集仍须在部署后确认目标与认证、真实 /metrics 样本及 Grafana 入库。

提前准备在 Render 创建前已有真实 init，不能仅给后面的 Render 设置详细计时。
PlayStartup.launch 采用源会话 journal 的 detailed_timing_default（init 显式值优先）；
source.prepared/source.attached 回执可携带可选 timing_reports，附模型会话和 generation。
PlayStartup 持有有界、过滤后的准备分时；原模型连接交接并 attach 完成后，经既有
RenderSegment.model_control → _archive_timing → TimingMetrics 消费一次，并清空缓存。
不新增 exporter、调度或模型会话，也不把准备期 Host/CUDA 分项相加成用户端 TTFF。
准备已发生的计时不能被后来的 render 请求追溯开关；运行中切换仍用 #662 原有控制协议。

## 合并和 review 重点

RenderJournal 保留 prepared_model_ids 原子身份与 detailed_timing_default；SDK pin沿现网。
复用原 source.prepared 回执传递可选分时，避免在准备等待期间增加未识别的独立消息类型。
仅保留匹配模型 session/generation 的 public_timing；最多256条，错误身份不进入下一玩家。
准备未采用/释放时清空分时，不混入其他渲染。发布前检查取消、fallback、原连接接续及
手动开启输出门禁；指标开关不能让模型画面提前发布。
阅读顺序：本说明 → RenderJournal.start → PlayStartup.launch/_event →
ZingEncoderDiTAdapter._send_connected_output → InteractiveSession._generation → RenderSegment.model_control。

## 验证计划与当前状态

静态检查、组合 CPU 套件、原生完整 Room 镜像检查、隔离三角色建组、真实 source 回放、
空闲门禁切换和实际 TTFF/prefix 指标读回分别记录。已有共享分支 CPU 结果不是本组合验收。
新增开关传递、prepared 回执分时保留与跨 generation 隔离回归；组合结果待运行。
GPU性能/显存及480/736整体A/B未完成；不承诺无开销或固定秒数收益。

2026-11-13：运行时 dde1817b2b 的模型 CPU 套件493通过、49跳过、1排除；
并行准备/交接专项73通过。Room独立回归发现绕过构造函数的fixture缺少TimingControl，
70b4ff6fdd已补齐；旧child-lease用例显式清理绑定模式环境。受影响启动/切换控制103项通过。
按文件隔离的组合覆盖累计450通过、2排除；未改文件沿用dde1817b2b回执。
70b4ff6fdd全套单进程混跑332通过后，source_conditions旧mock缺player_expires_at失败；
该文件独立151通过，混跑限制保留，不能宣称任意混跑或全仓CI通过。
完整 Room 镜像已构建并通过原生测试、源码/SDK逐文件检查和镜像层检查。
三角色 deploy-20261113T075937Z-2e80269e 已通过原生建组/解组并挂载测试池，平台未切换。
真实回放先暴露 Coordinator/Room 到新 Encoder、Decoder 的网络规则缺项，修正后已连通；
原失败租约由原生恢复流程确认释放，不强清账本。

实际预填充进一步发现 EncoderClient 在无 realtime_handoff 的成功 OutputBatch 上直接
调用 setdefault，导致计时逻辑抛出 AttributeError 并使 Encoder 进入故障隔离。
bac29ff82f 修正为先 drain 计时缓冲，存在报告时才初始化可选 handoff；空报告不改变
预填充结果，也不增加同步。真实 OutputBatch 的四组回归覆盖有/无 handoff、有/无报告，
保留 usage 和已有 handoff，并验证不触发故障隔离。该提交 Encoder API CPU 整文件
88 项通过；尚未完成修复后的真实 GPU 回放，不能据此宣称平台可用。

用户确认本次保持全程两步：initial_block_count=0、regular=[1000,500]，
继续使用匹配的原 AdaLN 表，不启用旧三块混合计划。
旧 split_first_latent=true 时三块1/2/2等于五latent；当前false时两块3/2才等于五latent，
不能把旧“三块”解释套到当前模型。本次不修改分块或增加混合计划支持。


最新测试环境 Room/Gateway 更新为克君分支 40df6be3200b 后，综合分支 f60d65078c4f
已保留其 reset 恢复和可选模型准入拒绝后的手动启动行为。5c888d0f2338 修正合并测试
夹具：保留 journal.initialize_inference，只设置详细计时属性；StrictLeases 补齐
player_expires_at。组合 CPU 510通过、6失败、2排除；剩余6项是完整 Source 场景
ModelBinding.close 超时并返回 RELEASE_UNKNOWN，原因尚未确定，不视为通过。
新完整 Room 镜像构建仍在进行；平台未切换。r6真实回放使用旧镜像导入路径，
下一案例 USER_SESSION_LIMIT 后超时；三个GPU角色释放，但 allocation awaiting_sdk。
该轮没有最终报告，不能作为当前新Room源码验收，不强清SDK账本或继续追加测试。

## 2026-11-13 测试采集链路修复

Room/Gateway 已于北京时间17:52切换至 0e81d6db80cd2d7b74754f5f02c692ae60f21885，
完整CPU组合540通过、2排除。运行时保持全程两步、详细计时开启。
先前6项混跑失败源于fixture继承WHITEBOX_BOUND_COMPUTE，已显式隔离；
旧Nov10 awaiting_sdk记录不是本轮释放失败。r7/r8真实回放仍有多参考图prefill失败，
不能宣称四案例全通过或性能A/B完成。

Qwen采集地址10.0.0.213:31000与认证已配置，源端HTTP 200返回296条样本。
入库被旧采集命令阻塞：Room/Gateway容器名固定、模型池端口清单缺28510/43310。
`prepare_test_runtime_collector.py`在校验部署归档SHA后生成测试专属包和只读命令：
按固定测试端口定位PID/cgroup容器，验证固定Docker socket、容器ID、角色、测试名称前缀、
host网络及启动身份，保留既有Room→模型池路由与前后身份校验。它不改变模型、
生产采集器或身份校验失败时拒绝写入的行为。host reader与FC同步补齐已核实的角色端口。
本地7项身份回归通过，云端采集与Grafana入库待验证；单独更新采集不会重启游戏。

2026-11-13 21:57（北京时间）：管理员更新RAM授权后，测试FC切换至bd4ca640539f
构建的采集包（SHA256 486c9fd1b7163aa50e6ce8316db76dfeda19ec96bb17b36bf504ab0a9589f2a0）。
FC包CRC64与本地上传包一致，环境变量已回读；两条主机只读命令的全部数据源
complete=true。连续采集conditioning_metrics/metrics/registry均true，Room写入180条序列。
Grafana实际查询已取得12种conditioning指标名：当前实例累计5次miss、0次hit，
编码耗时count=5，不等于试玩次数。日志全覆盖仍false。
首帧/切换详细分时源端暂无样本；验收trace 00ea0ba50c854bdb92d047184426153e
在admission遭CAPACITY_EXHAUSTED，whitebox/model均0帧，managed_closed已确认。
因此采集入库通过，但TTFF业务样本验收未通过；不强清共享容量账本，准入问题另查。

### 2026-11-13 提前准备能力配置缺口

线上缺少TTFF分时不能仅用验收客户端CAPACITY_EXHAUSTED解释。进一步对照同轮Encoder
日志，server.init_invalid明确为`This worker does not support early preparation.`：
Room WHITEBOX_MODEL_STARTUP_PREPARE=1，但独立Encoder未设置SGLANG_ZING_SOURCE_PREPARE。
Room日志有12次模型init失败；所查新Room RTC流的model media_stream_id为空、handoff为0。
这证明当前部署有模型初始化阻塞，不能据此否定用户在其他时段/路由的成功首帧和切换。

独立Encoder launcher默认启用该已实现的协商能力，显式0仍保留。它不主动执行预热，
请求仍必须协商source_prepare_protocol，固定seed及session身份校验保持不变。
实际三角色配置与平台业务验收分开记录；修复后须重新验证首帧/切换及Grafana真实样本。

新 Encoder df71aafd39d77c21a98b0de8fe6576f50f4889b2 已 ready、自测和注册，
但平台 trace e19f3002e4c9467f92088def42e8f0fd 的实际 allocation
5f0f858718a142e1a20013bd66acd3ff 仍选择旧 cachetiers Encoder（10.0.0.121:28510）。
模型池有两个兼容 Encoder，每会话只选一个；因此更换自己的 Encoder 不等于平台已使用它。
保留他人服务，Room 复用已有 WHITEBOX_PREFERRED_SERVICE_IDS / WHITEBOX_AVOID_SERVICE_IDS
选用提示。它们影响既有 admission 排序，并非容量不可用时的强隔离；验收必须回读每次 allocation。
docker_cutover 仅允许这两个 Room 配置按 expected/value 变更，同时更新运行环境、发布 bundle
及配置 hash，继续保留身份漂移、闲置与回滚检查。相同镜像仅替换 Room，不重建或更新 Gateway。

45e818ea9e 的发布工具在 Linux 下35项测试通过；测试 Room 已复用0e81镜像切换上述选用配置。
下一验收 trace cbc137991bac42fc9d0707c1f09cd124 尚未取得模型组：22:30 的
`cachetiers-test-slot-controlled-drain` 将旧cachetiers Encoder及本次DiT、Decoder排空，
22:45云端GetJob记录本次DiT被克君账号205206788242572972停止（StoppedByUser）。
此时失败是模型池没有完整可用组，不再是此前已选中旧Encoder后的init拒绝。

恢复部署deploy-20261113T145233Z-6952dd34复用新Encoder，DiT为dlc45lqffwf0hv7q，
Decoder为dlce5bbwrthvazn6，源码均为df71aafd39d77c21a98b0de8fe6576f50f4889b2。
PAI实测配额184CPU/8GPU，已分配168CPU/7GPU；Decoder的24核请求无法调度。
UpdateJob返回成功但CPU回读仍为24，已停止该排队任务，按原生render生成16核替代请求。
GPU1、160Gi内存、模型及采样不变。16核仅用于当前功能验收，不与原24核配置直接做计时开销A/B。
私有state保留替代请求及精确Job身份；模型加载、业务出帧及Grafana真实分时仍分别待验收。


### 2026-11-14 当前共享入口复核

旧失败实验的 GPU 注册残留已根据云端退出状态和精确 worker/GPU 日志通过正常退役接口清理；本次自有三角色建组、释放和 attach 均通过。但共享入口随后使用 cachetiers-shcopy-06d-sp1，不能把自有组自检冒称平台验收。

当前共享 Encoder 仍缺 SGLANG_ZING_SOURCE_PREPARE=1；北京时间10:01/10:04两次实测均在模型 init 明确拒绝 early preparation。named capacity route 同时有一个空闲完整组，客户端 CAPACITY_EXHAUSTED 不是显卡已满的证明。第一轮世界发布身份也过期；刷新 publication 和素材后该问题消失，但 Encoder 初始化仍失败。当前0模型帧，首帧 / 切换分时尚无有效新样本。

Qwen prefix 采集目标已按当前注册拓扑更新为10.0.0.121:28510，并仅补齐测试 FC 两个 ENI 到该端口的规则；coverage.conditioning_metrics=true，Grafana查询验证12类 conditioning 指标。当前共享Encoder的单开关修复manifest和回滚备份已准备，未重启他人服务；业务出帧与分时验证继续单独跟踪。

### 2026-11-14 按用户授权归并回原共享组

综合运行源码06d04b805c96已包含f4a136f9a284合并提交，将克君cachetiers改动和0e81d6db80cd的TTFF/prefix改动合并。当前任务分支进一步合入06d，保留运维修复和验收工具。
用户明确授权更新原共享组，已在原H100卡重启Encoder并加SGLANG_ZING_SOURCE_PREPARE=1；原DiT/Decoder仍用两张H200。误沿用的独立三张H200作业dlclcjx1nihcrfij、dlc45lqffwf0hv7q、dlce5bbwrthvazn6已正常down、退役并确认Stopped。

trace 10c4e8d887324693a0ac795f76759518对应world-d94f64601bff9a3b867f82adb3f55123，实际分配为cachetiers组。Encoder init已通过；但有两个独立阻塞：验收工具直接访问Gateway时缺whitebox_control=v2，误发旧/v1/sessions/admit并收到429；DiT提前条件确认读取尚未由source初始化的state.device，抛出AttributeError。后者使Encoder/DiT进入quarantined/restart_required，Decoder已释放。当前不能声称可试玩或已有有效TTFF样本。

客户端修复仅在实际WebSocket URL添加路由参数，保持SDK不含query的execution descriptor契约；不改公共SDK或生产路由。DiT条件屏障使用已接纳condition的设备作为首source前的设备依据，仍等待CUDA完成并执行跨rank一致性确认，不以跳过屏障修复。新增真实queued conditioning回归覆盖详细计时开/关和0source输入；CPU/GPU验证及恢复部署结果待补。

### 2026-11-14 原共享组故障恢复回执

原组为 H100 Encoder（10.0.0.121:28510）+ H200 DiT + H200 Decoder，并非全 H200。
Encoder 固定原 GPU；Decoder 保持作业 dlc1s2z097ix14ej。DiT 原作业
dlc1r9083vt0f6nn 已停止并按真实退出证明退役；替换作业仍在原 H200 节点，PAI
重新分配具体 GPU，不宣称与旧作业是同一 UUID。

新增 CPU 回归修正 V2 identity 夹具后2通过、2 GPU项跳过；此前外围组合167通过、
49跳过，原新增两项失败已由修正后的定向回归覆盖。这不代表 GPU 或性能验收通过。

完整修复镜像运行源码 bb3c04c39b54876230864a51c4411b4e8eddee6a，digest
sha256:c479ad08b9794e32b70418b2e5af801947a22b7d50ed44c2ed7702f960105864。
沿用完整原生 Dockerfile 与锁定依赖，未在运行实例覆盖应用源码。Encoder 重启沿用
06d完整镜像（设备修复仅在DiT），保留 SOURCE_PREPARE=1；新 Pod UID
0ab59d4f-b0bd-4d48-8953-e5b4024d674b 已完成预热注册。
DiT 首次替换 dlc9m0tfv8pb2e1f 因部署包缺密钥文件在注册前失败；补齐原私有bundle
两项密钥对象后提交 dlc1ks3si32iwpqg，10.0.0.238，GPU UUID
GPU-8e0b8658-a79c-282f-e02f-c31b57f8fbb8。前次失败任务不再占用运行卡。

实际采样回执仍为全程[1000,500]、initial_block_count=0、cross_chunk、
source_prompt_joint=false；不将launcher初始解析日志里的true误认为覆盖后的值。
窗口6/sink0、mem_fraction_static=0.875保持原组值，相对main的21/2属于冻结复现差异。
详细计时、Qwen prefix保持开启；业务分时样本尚待验收。

恢复脚本核验原boot/Pod UID和本次失败smoke allocation，仅恢复Encoder/DiT；终止后
才confirm-stopped/retire。重试要求前次作业Failed且未注册新epoch。仅为新DiT地址
补齐测试Room/Coordinator到33000的两条精确网络规则。完整组就绪、首AI帧、prompt
切换与Grafana分时仍分别验证，不以Running状态代替验收。
