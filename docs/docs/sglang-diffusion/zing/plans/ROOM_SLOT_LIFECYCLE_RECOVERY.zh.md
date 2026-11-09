# 多人试玩与每玩家 Model Slot 生命周期最终方案

状态：最终设计稿，尚未实施、合并、部署或通过线上验收。更新：2026-11-02（Asia/Shanghai）。本文取代此前关于平台统一整局最长时间的提案；其他历史报告用于取证，不作为目标行为规范。

配套：[目标架构拓扑](/docs/sglang-diffusion/zing/plans/ROOM_SLOT_TARGET_TOPOLOGY.zh)。

## 1. 已确认的范围与原则

1. 平台支持真正多人联机、多 GameRoom、多 Worker 和多机器。每个 Zing Worker 同时只服务一个活跃 PlayerSession，不实现同 Worker 多玩家并发。
2. 每位已准入玩家独占一个 Model Slot，纯白模、等待用户、上传参考图、Prefill 和换段期间都保留。模型子操作结束不归还父 Slot。
3. 只限制每位玩家的最长试玩时间；平台不再给 GameRoom 设置统一最长时间。晚加入的玩家拥有自己的完整配置时长，不受房间创建时间截断。
4. 正常退出、个人到期、浏览器断线超时都进入同一玩家回收流程。其他健康玩家继续。
5. GameRoom 仅在空房收尾、明确授权解散、游戏已有的正常结束事件或共享执行故障时关闭。未来游戏可以自行决定何时结束，本次不新增游戏时限策略引擎、配置体系或插件机制。
6. Prefill 是可选的、每玩家独立的模型计算。没有调用 Prefill 的游戏也能正常试玩；一人的 Prefill 不暂停共享 World。
7. CUDA、模型进程或节点故障导致清理无法确认时，隔离、告警、人工恢复，不自动整组重启。用户已接受无剩余兼容容量时其他用户可能等待。
8. 本次只整理方案，不实现代码、不申请资源、不部署、不重启，也不发起新的线上试玩。

“停止玩家使用权”和“Worker 已安全空闲”是两个事实。到期必须撤销使用权并触发回收；不可响应的硬故障不能靠删除租约伪装为已释放。

## 2. 对象、部署与容量关系

| 对象 | 职责与归属 |
| --- | --- |
| Gateway | 鉴权、查找游戏/玩家归属、转发控制与媒体；不运行权威世界、不持有模型计算状态 |
| Coordinator | 现有 Redis 上的目录、玩家期限、Worker 预约及清理事实；协调准入和回收 |
| CPU 侧 Room 服务 | 托管 GameRoom、CLI 执行、玩家视角和源流到模型的连接；可部署多个服务实例 |
| GameRoom | 一个 game_id/game_epoch 对应一个 SDK 权威世界、CLI 房间执行组和成员集合；没有平台业务截止时间 |
| PlayerSession | 一位玩家的成员、输入身份、私有视角/媒体、浏览器连接、个人截止时间及 ModelBinding |
| ModelBinding | 该玩家固定 Worker 上的一个 Slot 和长期 bound 父 WS |
| 模型子操作 | 一次 Prefill 或模型 Segment；同一父绑定内串行运行，具有独立 operation/session/generation |

N 名正在试玩的玩家需要 N 个 Worker Slot，无论他们处于白模还是模型模式。一个 Worker 可以由多张 GPU 组成；同一物理节点可在资源隔离明确时承载多个 Worker。多机器扩展不要求共享 GPU 内存或跨机迁移正在运行的 KV。

SDK 继续拥有成员、actor、输入权、复制和唯一世界模拟时钟。各玩家使用自己的视角，不能为多人联机分别创建彼此不共享状态的权威世界。CLI 执行组可以包含 Node、权威浏览器和各玩家视角进程，不能把“一份 CLI”机械等同于“一个 PID”。

本次可新增玩家数受剩余游戏成员位、CPU/视角资源和兼容空闲 Worker 共同限制。游戏声明的最大人数是内容能力，不等于当时空闲 GPU 数量。没有容量时明确返回容量不足，保留现有重试方式；不新增排队系统，也不默认为新玩家提供没有 Model Slot 的正式试玩。

## 3. 已核对现状与需要补齐的差距

以下为 2026-11-02 调查结果，不代表未来发布时仍然相同。

| 项目 | 已有实现/证据 | 本次改造 |
| --- | --- | --- |
| CLI 多人 | 上海安装的 CLI 有 room create/join/resume/leave/close；有 SDK 双人测试源码和交接记录 | 复用既有 SDK，多人选择受控 WorldkitRoom 驱动 |
| 默认试玩驱动 | cli_room → worldkit stream 明确 max_players=1 | 不靠调大人数绕过单视角限制；接通多人玩家视角 |
| Gateway 控制链 | 上海现网已挂载 room-control 和玩家媒体路由；createRoom/admitPlayer/leavePlayer/retireRoom 执行代码已部署 | 保留鉴权、原操作身份和目录归属；加入时联合预留每玩家 Slot |
| CLI 成员 resume | WorldkitRoom.resume 已实现；控制 action 和运行时调用缺失 | 保留底层能力和扩展位置，本次不做 Gateway/业务接口适配；浏览器 WS 重新 attach 是另一能力 |
| 当前 Prefill 入口 | 加入已有 game_id 与 renderSegments 互斥；F0 准备使用独占源时钟 hold | 已有成员视角绑定独立模型；多人使用不暂停世界的 F0 快照 |
| 个人结束 | 普通 source.unbind、普通 WS detach 到期存在只停子段、遗漏父绑定的问题；WebRTC 已有整会话关闭路径 | 统一玩家关闭入口，保留并整合正确的 WebRTC 行为 |
| 清理真实性 | dispose 异常被吞掉、release 仍返回成功的路径已做 CPU 故障注入复现 | 向上传递失败，全角色确认后才归还 Slot |
| 固定房间寿命 | WHITEBOX_GAME_SECONDS 创建固定 expires_at；到期续租失败触发整局清理 | 删除平台固定房间寿命，保留短租约和故障检测 |
| 自动重启 | 曾将长时间无模型计算视为异常并触发维护重启 | 有效白模玩家不判僵尸；运行期回收不触发 PlannedRestart/SIGUSR1 |

上海现网 Gateway 的 RoomProxy 挂载与当前文档工作树存在差异，已通过运行代码和 OpenAPI 核对；相关 Room 控制/驱动模块与本工作树 SHA256 一致。不能按某个本地文件缺少调用就判定线上未部署，也不能把历史覆盖文件的方式继续用于发布。

本轮未新建联机游戏、未跑现网双人模型验收；接口、源码和旧测试记录不等于新版本端到端通过。

## 4. 每玩家期限与服务存活

### 4.1 唯一试玩截止时间

成功预留 Model Slot 时记录 admitted_at，并固定：

```text
player_expires_at = admitted_at + region.player_session_ttl_seconds
```

计时从资源占用开始，包含后续启动、白模、Prefill、模型生成和重连；点击按钮、尚未取得 Slot 的等待不算已准入。前端在入场时展示实际个人截止时间。重连、重试、Prompt 热更新、换段和多组 Prefill 都不能重置它。

拟采用 WHITEBOX_PLAYER_SESSION_TTL_SECONDS 表达这一明确语义，名称尚未上线。初始值承接已调查的试玩配置：上海 1800 秒、泰国 7200 秒。上海为进程读回；泰国为平台配置，进程读回当时返回 502，发布前必须再次核对。旧“有效游玩计时”不能作为延长 Slot 占用期限的第二套时钟。

例如 A 在 10:00 准入、B 在 10:20 准入，配置均为 30 分钟，则 A 在 10:30 结束，B 在 10:50 结束；A 到期不关闭 B 的共享世界。

### 4.2 移除固定 GameRoom 寿命

现有 WHITEBOX_GAME_SECONDS 的上海 3600、泰国平台 9000 秒仅作为迁移前历史配置保留，不沿用为目标上限。

- Coordinator 不再为正常 GameRoom 从创建时刻计算固定业务 expires_at，不再以该字段拒绝续租、加入或媒体路由；玩家期限也不与它取最小值。
- 保留 owner 的短租约、心跳与 epoch 校验；活着的 owner 可持续续租，租约丢失仍阻止旧 owner 继续服务或被静默替换。
- 活跃目录采用随有效存活续期的存储 TTL；清理账本和未确认资源不依赖普通 TTL 自动删除。终态记录按既有留存要求回收，不能把没有业务寿命变成无限增长的历史记录。
- 清查 Room、CLI/SDK、WTP 和目录的隐藏固定时限及文案，防止 Coordinator 改完后仍被旧上限关闭。CLI 命令超时、个人恢复期限与服务租约仍保留。
- 新建房间尚无人加入、加入失败或创建结果未知，都必须有启动/补偿期限；pending join 不能永久充当“房间有人”。超时查询原操作并清理迟到结果，已确认无人使用时关闭世界，未知占用保留隔离证据。

未来游戏若有自身结束规则，走游戏已有的明确结束事件。本次不预设 game_expires_at，不新增平台整局定时器。

### 4.3 断线与回收时间

| 时间 | 本方案口径 |
| --- | --- |
| 启动 S | 沿用已调查的 480 秒上限，且不越过个人截止时间；只约束本次创建/加入 |
| 浏览器探测 H | 目标不超过 15 秒，需弱网、半开连接、后台页面实测 |
| 重连宽限 G | 初始沿用 15 秒；主动退出不走宽限 |
| 已受理重连协商 R | 最多 13 秒，不延长个人截止时间；同次断线不能重复重置 |
| 回收预算 C | 初始建议 90 秒；需故障注入校准，不是现网承诺 |
| 软排空 P | 初始建议最多 10 秒，之后转取消，包含在 C 内 |

检测到断线的时间为 d：没有合格重连时，撤销使用权并进入 CLOSING 的时刻为 min(player_expires_at, d+G)。第一次合格重连在 a 受理且 a 早于基础截止时，最多保留到 min(player_expires_at, max(d+G, a+R))。SDK 原成员有效期更短时，以其真实能力为限；SDK 已退休则不能继续恢复旧成员。探测耗时、重连窗口与清理耗时分别报告，不能只用“15 秒”代表完整资源回收时间。

期限由可信服务端签发，各端以本地 monotonic 剩余时间执行；业务准入同步检查，后台计时独立于帧发送、录像和 Prefill future。超时或取消后先停止新工作，不能声称已提交的 CUDA 工作立即消失。

## 5. 创建、加入、恢复与离开

### 5.1 沿用 Gateway → Room → CLI

受信任业务服务调用 POST /v1/whitebox/room-control。Gateway 校验签名、主体、action、请求摘要和有效期，查目录转发到指定 Room 的 /internal/room-control；CLI 命令由 CPU 侧 WorldkitRoom 适配器执行。

| 操作 | 当前控制 action | 目标行为 |
| --- | --- | --- |
| 创建共享世界 | createRoom；getOperation 查询 | 复用冻结发布物及幂等创建；创建本身不代表某玩家已准入 |
| 玩家加入 | admitPlayer；getAdmission 查询 | 个人成员/视角与 Model Slot 联合准入，全部就绪才发布可试玩 |
| 发布玩家媒体连接 | getPlayerSession | 只发布本人现存成员视角，不因接入模型再次 join |
| CLI 恢复原成员 | 尚无公开控制 action | 仅保留现有驱动扩展位置，本次不新增接口或恢复流程 |
| 玩家退出 | leavePlayer | 进入统一玩家回收，CLI room leave 只影响该成员 |
| 明确解散/清理共享世界 | retireRoom | 冻结加入，关闭成员与世界，关联各玩家模型回收 |

查询不创造资源。相同操作重试沿用原 operation/join 身份；结果未知先查询原创建，不能盲目再生成成员。

### 5.2 玩家准入顺序

1. 验证用户、房间开放状态、发布物能力及 CPU 成员/视角资源，登记有期限的 pending join。
2. 预留兼容 Worker 的唯一 Slot，固定个人 deadline；没有容量时不发布正式试玩，不影响既有玩家。
3. 通过原 join_id 创建或复用一次 SDK 成员/视角，同时建立该玩家长期模型父绑定；登记所有在途创建。
4. SDK 视角和 ModelBinding 均就绪，才提交 ACTIVE、返回可试玩。普通游戏直接开始；游戏可按自己的流程选择 Prefill。
5. 失败、取消或到期只补偿本次玩家资源。迟到成功也归原关闭流程；若这次新建世界已无人且无未决引用，再进行空房收尾。

Slot 预留后仍可能等待 CLI、视角、模型父连接建立；采用 Prefill 的游戏还可能等待自己的预热结果。不能把“点击开始”或“预留成功”当作第一帧已经可见。

### 5.3 浏览器重连与 CLI resume 扩展边界

普通浏览器重连只重新 attach 仍有效的同一 PlayerSession，校验主体、恢复凭据、owner/连接 generation，替换旧连接，不重置期限、不再占第二个 Slot。

本次不适配 CLI room resume：保留已有 WorldkitRoom.resume 方法作为未来扩展位置，不新增 Gateway action、业务 API、恢复状态机或预先触发调用。浏览器只能重连仍有效的原成员/视角；其已经失效或需要 CLI 成员恢复时，结束并清理旧 PlayerSession，明确告知本次试玩已结束，不偷偷新 join、不伪装为恢复成功。

未来若启用该扩展，必须保持原玩家和 Slot 的归属、原个人 deadline，校验更新后的连接/视角 epoch，并使旧视角 Prefill context、帧和回执失效。这些是扩展约束，本轮不实现、不将 CLI 成员 resume 列为验收通过条件。

## 6. 可选 Prefill、换段和缓存隔离

每位玩家使用自己的 Prompt、有序参考图、seed、视角条件和模型配置。多玩家在各自 Worker 上可以同时预热；同一玩家的 Prefill/Generate 子操作串行，共用其长期父 WS 和 Slot。

多组 Prefill 按完整输入 key/版本管理，分别返回准确请求的 memory_ready；落盘完成是另一状态。正式使用须核对 F0、条件、模型身份与 key。L1/L2 为 Worker 本地缓存，不承诺跨机器自动可见；合法相同静态前缀可复用，不能共享玩家动态历史。

多人固定 F0 时，从该玩家已有视角帧流取真实快照，记录游戏/成员/视角/source epoch、原始 tick/PTS 和像素摘要。共享 World、其他玩家输入和媒体继续运行。保留 SDK CaptureHold 的独占世界检查，不用移除保护来强行停钟；单人独占源可保留现有 hold 兼容路径。

消费预热时使用同一 F0 和匹配条件，再按有界选帧规则接后续实时帧，保留真实时间间隔。视角/源身份改变、场景或条件改变、快照过旧时，仅重做该玩家准备；不能给旧 key 偷换新 F0，也不能每个世界 tick 都让全部玩家缓存失效。具体快照新鲜度、转入实时流的视觉连续性必须在实现时确定并用真实联机画面验收。

模型 Segment 切换不关闭长期父 WS、不归还 Slot：取消或结束旧子操作，等关键资源清理确认，再创建新 operation/session/generation。各段动态 KV 有效长度/索引、近期 latent、RNG、source bridge 和编解码状态独立；底层池内存可复用，旧有效数据不能被新段读取。Prompt 同段热更新按现行版本协议执行，不把所有更新强行变成新 Segment；回执与第一张新条件帧可追溯。

## 7. 一个玩家的统一回收

主动退出、个人到期、断线超时、启动失败及不可恢复的个人错误，都进入同一幂等 close_player_session。兼容 source.close/source.unbind；render.end、取消单次 Prefill 仅结束子操作。

沿用 Coordinator/Redis 的权威资源状态，不另造资源调度平台。正常流程为 STARTING → ACTIVE/RECONNECTING → CLOSING → RELEASING → RELEASED，清理未知为 QUARANTINED。

1. CAS 冻结 close_id、个人身份、原因、在途操作和 cleanup_deadline，撤销新输入/模型工作；重复关闭归同一结果。
2. 并行关闭两个分支：SDK 成员/视角/输入；模型父子操作与 Encoder、DiT 各 SP rank、Decoder、动态 KV/latent 和关键异步使用者。
3. SDK 确认成员退休后，保存终态收据，立即独立核销成员位、视角预算和 GameRoom 引用。共享 CLI 为 B 继续运行不代表 A 未清理，A 的坏 GPU 也不能永久锁住已清理的成员位。
4. 模型清理确认后，Worker 保持 CLEANED_PENDING_COMMIT，仍禁止新预约。Coordinator 收齐 SDK 终态与模型证据，进入 RELEASING，向原预约提交 finalize_release。
5. Worker 返回最终 ACK 或可查询终态后，Coordinator 才标记 RELEASED/free。ACK 丢失查询原 close_id，不重复影响新 owner；已经销毁的 SDK 成员以持久收据确认，不要求再次响应。
6. 关键清理失败或结果未知，保留对象/占用/证据并隔离告警，不返回假成功。用户已无使用权，未知 Worker 仍不算空闲。

收据使用原 game/player/member/view、Worker/预约、父子模型 session、epoch/generation 和 close/join_id 定位。一个用户 ID、裸 actorId、404、WS 已断或 Redis TTL 到期都不能单独证明真实资源已经停止。

整局已被确认关闭时，覆盖准确 game_epoch、成员与在途 join、且阻止迟到有效创建的持久 GameStopProof 可以替代 SDK 侧证明；不能替代模型 Worker 清理和最终 ACK。

已转交独立所有者的录像封装/上传、L2 写盘不阻塞 Slot 归还；仍依赖本次 GPU 的计算、D2H 和缓冲使用必须停止或完成。close 并发调用等待同一清理任务，不能因 closing 标记提前返回成功。

## 8. 共享世界与故障兜底

### 8.1 空房和明确结束

SDK 成员退休后，原子检查是否还有有效/重连中的成员、有期限的 pending join 或未确认 SDK 清理。确认无引用即进入空房关闭；初始不新增额外空房保留时长。新建但无人加入的世界使用启动/入场期限收尾，不能因取消固定寿命而常驻。

最后退出与新加入使用同一房间版本竞争：关闭获胜则拒绝/补偿加入，加入获胜则保留世界。GameRoom 锁只保护状态提交，不能持锁等待 Prefill、生成或 GPU 清理。

空房世界停止与玩家模型清理互不等待：SDK 已退休且模型证据齐全即可释放玩家 Slot；空房也不必等待坏 GPU 恢复才关闭。world_stop_complete 与各玩家容量释放分别记录。

明确授权解散、已有游戏结束事件、共享权威故障或 owner 租约丢失时，先冻结新加入/恢复，取消在途创建，并发结束该局所有玩家和共享 SDK 世界。创建者退出仅退出自己，普通返回页面不能被解释为解散。

### 8.2 独立兜底和不自动重启

- Gateway 的模型 WS、Room 心跳或共享世界推进不能替代某位玩家的浏览器存活证据。
- Room 与 Coordinator 独立检查个人期限；Worker 父绑定也持有同一期限，到期取消已阻塞的子操作，而不仅拒绝下一次请求。CLI/SDK owner 的存活租约丢失要停止旧世界，不静默创建替代世界。
- 取消入口不得排在卡住的生成队列之后。控制面不可达时，各 owner 按已签发期限和租约停止业务，保留原身份结果，恢复后归并关闭记录；不能自行宣称 free。
- cleanup 执行权可有界续租，但不能延长玩家试玩；CLOSING/QUARANTINED 和资源占用不能被普通 TTL 清掉。清理晚到确认仍按原最终释放协议提交。
- 修复 RealtimeSessionCache/RealtimeSession/ZingSourceStreamState 的吞错、提前丢对象和提前归还 KV lease；Scheduler 聚合全部 SP rank 与远端角色的结构化结果，Encoder 线程停止和 Decoder abort 必须真实确认。
- 独立模型故障只隔离该 Worker；共享世界严重故障按该 GameRoom 级联；节点故障按实际归属处理，不能影响其他健康局、Worker 或训练任务。
- 删除运行期“无模型计算/长时间无帧即重启”的维护触发条件；审计 guard、重试/rollback 和信号入口，运行期回收不能落入整组重载。仅修改 AGENTS 或磁盘文件不等于运行策略已经修复。
- 硬故障人工恢复；计划重启须针对该次动作得到明确授权。不通过自动换进程、强删占用或清空缓存制造验收成功。

## 9. 最小实施顺序与迁移

| 顺序 | 改造位置 | 交付结果 |
| --- | --- | --- |
| 1. 身份与期限 | Coordinator/rooms、配置、Gateway/Room | 个人 deadline 与资源占用统一；取消固定游戏寿命，保留租约及空房/启动回收 |
| 2. 联机准入与浏览器重连 | WTP、room-control/runtime、WorldkitRoom、玩家媒体 | 既有成员/私有视角与一个 ModelBinding 绑定；不重复 join；CLI resume 仅保留扩展位置 |
| 3. 模型接入 | render_host/source、ControlPrefill、SourceFrames | 多人加入可用本人的可选 Prefill；不暂停世界；同父连接热更新/换段 |
| 4. 真实清理与兜底 | Room/RTC/WS、bound API、Scheduler、Encoder/DiT/Decoder、Coordinator | 全退出路径统一、失败不误释放、晚到结果隔离、无自动重启 |
| 5. 前端与观测 | 试玩入口、玩家状态与运维查询 | 个人倒计时、退出/解散区别、可定位占用/关闭/隔离、真实容量提示 |

步骤可并行开发，必须作为相容能力整合验收后再启用。保留现有 Coordinator/Redis、SDK 成员系统、Gateway 路由和 bound 协议；不新增跨机缓存服务、同 Worker 并发、自动迁移、排队平台或游戏时限框架。

旧记录迁移保留真实 admitted_at、个人 deadline、成员与预约身份，不用迁移时刻重新增加一个 T。缺乏可靠身份/起点的旧会话在明确维护窗口排空，不能把创建整个 GameRoom 的时间冒作所有玩家起点。固定 game expires 字段只在确认新版本理解个人期限、租约和清理账本后停用；迁移期间不能混用新旧解释。

回退必须理解未完成创建、CLOSING/RELEASING/QUARANTINED 和占用；不能回退到忽略这些记录的旧分配器或通过清空隔离回退。按实例排空，不终止其他有效玩家。

## 10. 发布与验收

遵循完整规范 [/Users/kejun/.codex/policies/whitebox-release-and-e2e.md](/Users/kejun/.codex/policies/whitebox-release-and-e2e.md) 和[报告模板](/Users/kejun/.codex/templates/whitebox-e2e-report.md)。本方案不授予合并、生产部署、重启或新增 GPU 的授权。

所有开发在独立 worktree 完成。SGLang、Worldkit、World Test 的部署代码都来自各自远端 white_box 的准确提交；Worldkit 所需 dev、World Test 所需 main 变更先纳入 white_box。内嵌 CLI/SDK 同样固定来源。上海、泰国使用同一组件代码身份，差异只由配置表达；禁止热补丁、容器内改源码、旧镜像仅覆盖几个文件及运行脏代码。

### 生命周期专项验收

先做虚拟时钟与 CPU 故障注入，再在已有授权容量下做真实联机与模型验收；短 TTL 用隔离环境，不缩短线上玩家期限。

| 用例 | 必须结果 |
| --- | --- |
| 纯白模、只上传参考图、从未生成 | 有效期内保留 Slot；个人到期进入真实清理 |
| 无 Prefill、单组/多组 Prefill、Prompt 热更新、跨 Scene/Segment | 普通路径可用；各输入/版本/F0/key/回执一致，换段不归还父 Slot |
| A/B 同局且错峰加入 | 两人共享世界、各自视角/Slot/个人 deadline；A 到期 B 继续 |
| 联机公开接口创建/加入、状态同步、输入与视角隔离 | 从实际业务/Gateway 入口创建并加入同一局；A 动作在 B 视角中按共享状态出现，输入只控制授权角色，媒体/凭据不能串用；退出与重连均正常 |
| 活跃房间跨越旧 3600/9000 秒边界 | 通过可控时钟专项验证没有平台统一关局；新加入玩家仍有自己的完整 T |
| A Prefill 时 B 移动/生成 | World tick、B 输入及模型持续，A 使用真实 F0 和后续帧时间 |
| 浏览器重连与原成员失效 | 成员仍有效则同玩家同 Slot 重连、不续时，旧连接失效；成员已失效则结束回收，不调用未接通的 CLI resume 或重复 join |
| A 主动退出、关闭浏览器、网络半开、个人到期 | 仅 A 回收，B 正常；漏发关闭也有兜底 |
| 最后离开与并发 join；创建后无人入场 | 无孤儿世界/人物，无无限 pending；不误关成功加入者 |
| 明确解散/共享世界或节点故障 | 精确影响范围，关联全部未决资源；不重启无关 Worker |
| dispose、Encoder、Decoder、SP 清理失败/ACK 丢失 | 不发假成功、不提前复用；证据重查幂等，硬故障隔离人工处理 |
| A GPU 隔离但成员已退休；空房关闭缓慢 | SDK 资源独立核销；已有证据的模型释放不等待世界关闭 |
| Coordinator/Room 丢失、迟到创建与旧回执 | 原身份回收，旧结果不能影响新 owner；未知占用不当作 free |
| 录像/L2 非计算 I/O 缓慢 | 不无限阻止 Slot 归还；录像和持久化结果仍如实可查 |
| 新玩家使用释放容量 | 新身份真实出模型帧、输入正确、无前玩家状态串用；不靠重启通过 |

### 正式浏览器与性能验收

部署得到授权后固定同条件旧版基线和 5 个当前正常的动态 case；记录发布物、输入、模型/源码/SDK/镜像/配置身份。Prefill、Prompt、多组缓存和多 Scene/Segment 优先使用同事本地可控 case，普通世界覆盖无 Prefill 和复杂运动。

每个正式基线/候选玩家会话从首个可见模型帧开始连续至少 180 秒；A 退出后 B 接续时两场各满足。短 TTL 故障注入是专项测试，不能代替这 180 秒。失败、不足时长和未覆盖单列，补测不能抹掉旧失败。

分别对比 CLI/入模/raw model/浏览器呈现 FPS、按键到白模/模型可见帧延迟、实际 L1/L2/miss、读盘与 GPU 恢复时间、RTT、卡顿/队列年龄和画质。至少检查开头、约 90 秒、最后 30 秒和每次切换；异常画质保存截图，录像核对实际三路及分段归属和可解码性。

任何观察到的恶化逐项显示；不以总体均值或阈值隐藏，不以缺少基线证明“无退化”。未真实观察 L2、双人联机或任何能力就写未覆盖，不为补测试擅自清缓存、重启或新增资源。模型正常换段/玩家交接保持 Worker epoch，Slot 恢复以新玩家真实模型输出证明。

联机接口正常是本方案必验项，包含创建/加入、共享状态同步、输入与视角隔离、退出/浏览器重连。CLI 成员 resume 按用户确认仅保留扩展位置，明确标为本次范围外，不能混写成已验收。现有容量不足时如实标记验收未完成，不能用单人测试或 CLI 命令存在替代，也不能将整体验收写为通过。

## 11. 观测和证据

查询至少关联 GameRoom、PlayerSession、SDK 成员/视角与 Worker，展示个人 deadline、服务租约、pending join 期限、模型绑定、Prefill/Segment 状态、关闭原因、关键清理收据及隔离范围。区分等待用户、预热、生成、重连、清理、隔离；无模型计算不计为 free。

原始调查保存在工作区 reports/20261102-shanghai-slot-incident：REPORT.txt、EVIDENCE.json、lifecycle-policy-baseline.json、cpu-reproduction.json、cleanup-failure-reproduction.json、physical-deployment.json、multiplayer-capability.json、gateway-room-control-capability.json。泰国未完成的进程配置读回与未执行的端到端验证不得冒称通过。

SDK 命令核对基于当时 main 936f3306a9fd4606efc733a413e821f3eb95aac3 的对应模块及上海安装文件；这不是完整包一致性证明，也不是未来允许从 main 直接发布。历史 SGLang SHA 和容器文件差异只是基线证据，实施时重新核对并纳入 white_box 的完整版本。

本稿完成后运行服务仍保持原状：固定游戏上限、未接通的 CLI resume 和其他缺陷不会因文档修改自动消失。
