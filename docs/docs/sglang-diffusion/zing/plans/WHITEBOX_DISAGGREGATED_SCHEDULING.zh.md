# 试玩链路三角色分离部署与分布式调度设计

状态：设计及本地实现说明。分支已开始整合下述 main 并实现 managed role pool；运行与配置合同见[实现指南](/docs/sglang-diffusion/zing/specs/MANAGED_ROLE_POOL)。本次仅本地验证，未部署或执行 GPU 验收。日期：2026-11-03（Asia/Shanghai）。

工作分支：`codex/whitebox-disaggregated-design-20261103`。基线为本轮获取的远端 `white_box@5d7e71a87ab81cf61bcea60e03c606a2121ea3a0`；参考远端 `main@0a4623e1299efc386a81473ed03d6a79ed234eaa`，即分离部署 PR #509。该 main 提交不在原 white_box 基线中，本次在工作分支中整合。来源固定到 SHA，不以之后移动的分支内容替代本轮判断。

配套：[角色池与会话绑定拓扑](/docs/sglang-diffusion/zing/plans/WHITEBOX_DISAGGREGATED_TOPOLOGY.zh)。沿用[玩家生命周期约束](/docs/sglang-diffusion/zing/plans/ROOM_SLOT_LIFECYCLE_RECOVERY.zh)，已有实现以本轮 white_box 源码为准，不能把历史设计中的缺陷清单直接当作今天仍未修复。

## 1. 决策与范围

建立独立的 Encoder、DiT、Decoder 资源池。玩家准入时，服务端选取三个相容 service，联合预留容量并建立通信组；此后同一玩家固定使用这组资源，到退出或到期才释放。每个 Encoder/Decoder 进程可用多个 service 对接不同 DiT，数量由实际服务数和吞吐预算决定，不写死 1:2，也不预先绑死机器配对。

“动态”首先指新玩家入场时组合、空闲资源重新配对、新进程加入和故障摘流；不意味着把正在生成的有状态会话逐请求随机转发或自动迁移 KV/latent。健康会话不因新增进程或负载变动被搬迁。共享角色硬故障影响其关联的所有组，不能承诺任意故障下无中断。

本设计沿用已确认的产品边界：每玩家限时，GameRoom 无平台统一最长时间；纯白模也保证其模型容量；可选 Prefill、多组预热和 Segment 切换保留；CLI `room resume` 仅保留扩展点；浏览器对仍有效成员的重连保留；硬故障隔离告警、人工恢复，不自动重启或强删占用。

最新需求允许 Encoder/Decoder 进程被多个玩家共享，细化此前“每 Worker 一名玩家”的口径：**每个逻辑 Model Slot/通信组仅属于一位玩家，每个 service 同时最多一个活跃会话，DiT service 不并发服务多玩家；E/D 进程可以包含多个分别独占的 service。** 不把一个 service 的 capacity 调成 N 来绕过隔离。

## 2. 已有能力与适配缺口

| 层 | 本轮源码证据 | 设计结论 |
| --- | --- | --- |
| main 独立角色 | E/D 支持启动参数 `--service-count N`，共享模型、独立 service 连接与状态；DiT 一个 service，当前独立入口 SP1/SP2 | 以 service 为绑定单位、进程为共享容量和故障域；SP rank 不是额外 Slot |
| main 建组 | `/get_connection_status` 空闲时会取得 10 秒预留；三个 `/build_group`；三个 `/destroy_group` | 原语可复用，但没有生产注册目录、分片联合调度和可靠跨控制器接管 |
| main 业务接口 | Encoder 输入 WS；Decoder 输出 WS 和 source.consumed；Encoder 返回输入回执和 prefill.completed | Room 单 WS 模型客户端不能只换 URL，需双通道适配 |
| main 缓存 | Prefill/缓存管理入口在 Encoder，操作组内 DiT 的原 L1/L2；拆组保留缓存 | 缓存归属固定到 DiT，建组与业务会话身份分开 |
| main 旧共享 API | 自动挑选多个 DiT 的旧候选池入口已停用 | 不启用旧动态重试路径，不把测试 Master 当生产调度服务 |
| white_box Coordinator | WorkerHeartbeat 角色仍为 denoiser/vae；SessionAssignment 是 denoiser 加可选 vae；Redis 单状态 JSON 配合全局锁 | 需要三角色 assignment、service/进程双重容量约束及分片存储；不能照搬旧全局锁实现高并发 |
| white_box 试玩 | ModelBinding 使用 `/v1/realtime_video/bound`，串行复用父连接；已有玩家期限、验证清理与隔离记录 | 保留逻辑父绑定，增加分离式后端，不能让旧 bound endpoint 承担未实现协议 |
| white_box 世界寿命 | 目录校验已去掉固定游戏截止条件，owner 短租约仍在 | 沿用现有方向，不重新增加整局 TTL；核对存储留存与 pending join 兜底 |

main 有历史 H100 双路回归记录，但异构卡型组合、当前 white_box 整合版、生产服务发现、调度吞吐以及完整浏览器链路仍须验收。旧 H200 池化/SP=k/在线迁移数据不能作为当前独立入口的通用支持声明。

## 3. 资源对象与所有权

| 对象 | 关键内容 | 所有者 |
| --- | --- | --- |
| RoleProcess | process_id、boot_epoch、角色、GPU/MIG 资源身份、模型/代码指纹、service 清单、共享显存/吞吐预算 | 原生角色进程；RoleReporter 上报 |
| RoleService | process_id + boot_epoch + service_id、管理/数据端点、生命周期、当前组、局部执行/清理状态 | 原生 service，容量为 1 |
| ServingPool | 区域、网络互通域、兼容模型/协议集合、调度分片 | Coordinator；不以卡型分池，异构卡可在同池 |
| GroupBinding | allocation_id、binding_epoch、worker_group_id、固定 E/T/D service 描述、玩家 deadline、owner lease、构建/清理进度 | Coordinator 权威账本，控制器执行 |
| PlayerSession | 游戏成员/私有视角与一份 GroupBinding；浏览器授权和输入归属 | 现有 WTP/Room 合同 |
| 子操作 | operation_id、session_id、generation_id、Prefill context/Segment、输入输出序列 | Room 模型适配器与三角色执行端 |

`worker_group_id` 标识通信拓扑；业务 `session_id/generation_id` 标识组内一次子操作，不能互代。业务完成不自动销毁通信组。所有子操作带完整玩家及绑定上下文，旧输出/回执不能进入新段或新玩家。

每个物理 GPU/MIG slice 只能归属于一个明确资源预算域；若同卡部署多个角色进程，必须额外有共同资源上限和经过验证的共置配方。首版不默认允许无约束同卡混部。一个 E/D 进程的所有 service 与共享预算归同一调度分片，避免跨分片分别把同一进程预算用满。

部署方式可以是 ACK Pod、VM/DLC 进程，发现合同不依赖扫描某个 Kubernetes 集群。业务数据使用实际端点直连；上游固定入口只路由到 CPU Room/控制服务，不用随机 Service VIP 转发 stateful E/T/D 流。

## 4. 注册、健康和动态摘除

### 4.1 进程自注册

在现有 worker_heartbeat 能力上扩展 RoleReporter，随角色进程运行，使用服务端部署身份向 Coordinator 注册完整进程与 service 清单。Kubernetes watch 可以作为地址/退出提示，不能成为唯一事实来源，也不遍历云账号自动接管未知机器。

注册记录至少包括：角色和实际端点；boot_epoch；代码/模型/codec/协议版本；SP/shape/dtype 能力；可达网络域；service 数和可调度数；已验证容量 profile；当前组/操作摘要与健康状态。端点必须由可信部署身份上报，不能由浏览器指定；管理与业务连接使用现有内部认证/私网边界，生产 managed 模式须验证角色身份和绑定凭证。

启动流程为 STARTING → WARMING → READY。模型完成加载/基础 warmup、清单完整、端点互通且具备本方案管理能力后才可分配。`/health ready` 不等于空闲，也不表示本玩家的 Prefix 已热；扩容加入期间不影响已有组。

首版采用启动时配置 service-count；**当前 main 不提供任意热增 service 数的完整生产合同**。动态能力通过已启动空闲 service、新进程加入和排空旧进程实现；运行时修改 service-count 不属于首版。

### 4.2 心跳、状态与数据留存

- RoleReporter 周期上报进程和 service 状态；建议初始周期 2 秒、失联摘流阈值 10 秒，加抖动，均为待实测参数，不是已经达成的 SLA。
- 新心跳必须匹配 boot_epoch；旧 epoch、旧序列和乱序回包不覆盖当前状态。登记“可调度健康状态”和“实际占用”分别更新，心跳不能冲掉清理隔离。
- 正常状态 READY，主动排空 DRAINING，失联 SUSPECT/LOST，无法确认清理 QUARANTINED。SUSPECT/DRAINING/LOST/QUARANTINED 均不接受新组，已有归属留在账本。
- `/health` 用于轻量探测；管理扩展提供只读服务状态。**禁止定时轮询 `/get_connection_status` 统计空闲**，它会占有预留；`available` 表示此次已经领取 token。
- 摘除仅指从候选和容量中剔除，不能删除未清理 assignment。已退出旧进程留下 tombstone、关联组及必要证据，完成核验后按留存策略回收。
- Reporter 自身失联不能被当作 CUDA 已停止。新进程即使复用地址也用新 boot_epoch；相同 GPU 的旧执行未确认退出时，不能把换 PID/重新注册当作已恢复。

角色数量没有固定配比。新增 E 只增加 E 可用量；如果 D 或 T 已满，完整可试玩容量不会因此增加。管理端分别展示角色余量和能够组成完整组的容量估计。

## 5. 容量模型与负载均衡

### 5.1 同时约束 service 和共享进程

一个玩家的配置 profile 包含 source/target 几何、FPS、模型配方、参考图/Prompt 上限、Prefill 类型等。对每种通过验证的“GPU + 模型 + profile”，测量 Encoder 条件准备/持续 source VAE、DiT 生成、Decoder 解码，以及峰值显存、网络、队列等待。

每个进程有共享预算向量 B（持续计算、突发工作、动态显存、传输/输出），每个玩家有该 profile 的保守需求向量 b。调度必须同时满足：

```text
每个 service 的已持有绑定数 <= 1
对每个进程、每个预算维度 j：Σ 已预留玩家需求 b[i,j] <= 安全预算 B[j]
DiT service 固定一位玩家；SP2 的两个 rank 合起来仍只有一个 service
```

`service_count` 只是可建立几条隔离连接，不是吞吐倍数。main 的 Encoder 共用单执行线程，Decoder 也共享角色执行循环；扩到多个 service 可能增加公平排队而非加速单卡。Qwen/参考图准备和持续 source 编码、解码之间要做有界工作调度，不能让一次超长 Prefill 饿死其他玩家。

容量 profile 以满足每玩家延迟/FPS SLO 的混合并发实测为准；不能仅用单会话 FPS 线性相加。上线前确定 SLO 和实际 profile；本稿不给未测卡型编造吞吐或承诺固定并发数。未知组合不进入保证容量池；测试池可单独验证。

运行指标可带滞后阈值地减少新准入预算，恢复时只能回升到该 profile 已验证的上限；不能因短期 GPU 利用率低就自动提高保证并发。预算下调低于现有承诺时停止新增、按原玩家期限自然排空，不追溯抢占已准入玩家。

纯白模玩家也预留其 profile 的整套 b 和三个 service，不因为当前利用率为零超卖。多组预热受本人串行子操作和进程突发预算限制。业务参数扩大到原 profile 之外须先获得新增预算，否则明确拒绝，不能静默挤占其他玩家的保证。

### 5.2 任意配比的例子

以下单位仅为演示算法，不代表真实硬件 FPS：

| 进程 | service 数 | 实测后允许的标准玩家预算 |
| --- | ---: | ---: |
| E1 | 4 | 3 |
| E2 | 2 | 1 |
| T1…T5 | 各 1 | 各 1 |
| D1 | 4 | 2 |
| D2 | 3 | 2 |

同一标准 profile、无网络/拓扑限制时最多 4 个组：G1=E1/s0+T1+D1/s0，G2=E1/s1+T2+D1/s1，G3=E1/s2+T3+D2/s0，G4=E2/s0+T4+D2/s1。虽然 E1/s3、D1/s2 仍未建组，它们的进程预算已满，不能再准入；T5 暂时无完整上下游组合。

同质 profile 的粗略上界为 min(E有效容量总和，DiT空闲数，D有效容量总和)。异构 profile 和受限网络下，这个 min 只是上界，实际由容量与兼容性匹配决定。空闲 service 可以下一次选择不同伙伴，不需要手工维护固定的 E→T→D 对照表。

### 5.3 筛选与选择

先做硬筛选，再做负载评分：

1. 同区域/许可网络域、READY 且未隔离，service 和进程预算都足够。
2. 编码条件/latent schema、checkpoint 兼容集、Qwen/VAE 指纹、source/target 几何、precision、ChunkPlan/采样合同相容；软件协议能力和完整版本可核对。
3. CUDA/NCCL/驱动及卡型组合、rank 拓扑、端口与实际带宽已验证。不同 GPU 型号不是排除条件；“都是 CUDA”也不是互通与性能证明。异构首先发生在不同角色之间，SP2 内部不默认允许未经验证的混卡。
4. 从有可行 E/D 邻居的空闲 DiT 中取有限候选，结合 locality、预计队列等待、进程预算占用与已验证 cache hint 打分，再比较完整三元组的预计端到端代价。不能贪心选到没有上下游余量的 DiT 后就错误报告全池已满。

候选索引按 pool/profile/role 和网络邻接维护；首轮每类取有限 K（初值 8），失败时有界扩大（例如 64）或重新抽样，不做每次全池 Cartesian product。控制预算耗尽返回可重试繁忙，只有确证无可行容量才报告容量不足。打散同分候选，冲突后抖动重试并设次数上限，避免惊群。

负载数据是排序建议，最终原子预算核销和服务本地预留才是准入依据。DiT L1 命中可加权，但不得为了缓存无限等待忙 DiT；缓存管理接口要求已建组，不能为搜遍缓存而预留所有 DiT。hint 来自可信最近回执，实际命中以本次执行结果确认。

## 6. 分布式准入和建组事务

### 6.1 控制面组织

保留现有 Coordinator 服务，对三角色引入版本化的 `GroupAssignment` 分配路径，不把 encoder/decoder 硬塞进旧 `vae` 字段。多个 API 实例无状态处理请求；分片控制器领取有期限的执行权，负责持久任务的建组、续租、回收与故障协调。

按区域和网络/模型兼容 ServingPool 分片。每个进程唯一归属一个分片，其所有 service、预算和该组的资源记录使用同一 Redis Cluster hash tag。三角色预留以短 Lua/CAS 在该分片内原子完成；不跨 Redis hash slot 假装有事务。用户配额/同一入场请求跨池重试由稳定的 admission_id 和用户入口 claim 保护，原池结果未查明前不在另一个池再建组；claim 已产生或可能产生外部副作用时不能因普通 TTL 消失而允许重试。跨分片 claim 与组账本分步提交，通过原 admission 查询完成或补偿，不宣称跨分片原子事务。

移除新路径对单个全局 JSON 大锁的依赖：service、进程预算、assignment、close 记录分 key；状态变更写版本事件，心跳可批量刷新。控制器不持锁等待模型加载、HTTP、NCCL 建连或清理。一个分片暂时单写或 Lua 串行提交不等于整个集群只有一个调度线程。

### 6.2 准入状态机

`REQUESTED → HELD → BUILDING → BOUND → CLOSING → RELEASING → RELEASED`；失败且清理不明为 QUARANTINED。服务注册状态与业务绑定状态分别存储。

1. Room/WTP 校验玩家/发布物/成员位，登记有期限的 pending join。调度选择候选并原子写 HELD：占用 E/T/D 三个 service、各共享进程预算、admission_id。此时固定 admitted_at 和 player_expires_at，重试不重置。
2. 建组控制器在账本登记每个即将发出的调用，再并发请求三个 `/get_connection_status`。保留各自 token、有效期和完整描述；10 秒为 worker 本地原生预留，与玩家期限、控制执行租约不同。
3. 校验返回描述与登记的 service/boot_epoch 相符，生成唯一 group_id；并发对三者发相同拓扑、各自 token 的 build_group。三个请求都须在各自预留期限内被接纳；建连耗时由单独 startup/build deadline 约束，不把 10 秒理解成 NCCL 必须完成的耗时。
4. 三方均确认相同 group/topology、原身份和管理代次有效，才提交 BOUND。加入/取得 SDK 私有视角并绑定该组，只有游戏视角与完整模型容量均就绪才发布可试玩。任何一步晚于截止或已关闭，不发布成功，进入原补偿。
5. 任一预留失败，已发起的预留都在账本可见。未发送 build 的原生 token 可按实际到期解除；未知是否发送/接纳 build 的资源必须先核验或清理，不能凭短 TTL 释放控制面占用。
6. 部分成功或超时，对所有“已成功或可能已接纳”的 service 协调原组清理。不能只销毁收到成功回包的两个而遗漏第三个迟到成功。清理完成的子资源可独立归还；未知部分隔离，整次玩家入场失败，不发布半组。

原生 main 的 destroy_group 在 connecting 时可能 409；build 回包丢失也不代表失败。因此必须补下节的只读状态、构建取消/期限和清理确认，不能仅复制测试 Master 的轮询逻辑。未知 CUDA/NCCL 状态不无限重试配对、不自动重启。

### 6.3 控制器丢失和存储故障

每次外部副作用之前持久记录 operation_id、目标 epoch、group_id、执行代次。正常同一存储主实例内，继任控制器通过 CAS 获得新的有界执行租约，由受信任 Coordinator 签发含 allocation_id、执行代次和有效期的操作凭证；凭证有效期不得超过对应执行租约，执行器自己不能无限续签。各角色本地拒绝过期凭证和落后于已接收代次的操作。继任者查询原操作接管，相关原生端均确认安装新执行代次前不发起新的业务副作用，不能重发一组新身份；无法完成接管时保持隔离，原有本地超时清理继续有效。

**首版对控制存储灾难采取明确的暂停准入、人工恢复合同，不承诺 Redis 回滚后的自动无损接管。** 普通 API/执行器副本故障仍可在同一权威存储下自动接管；以下限制仅针对 Redis 主身份变化、回滚或账本丢失：

1. 准入权威为部署控制配置中的 control_epoch（每次恢复新建唯一值），并固定获准写入的各 Redis 分片主实例身份，身份包含进程启动/恢复代次，不能只用地址或持久节点 ID；同节点从旧持久化重启也按新实例处理。这个授权配置不从可能回滚的 Redis 数据读取。所有 Coordinator 的写连接建立/重连时必须核对实际主实例身份；禁止透明跟随 Sentinel/DNS/Cluster failover 后直接恢复准入。新主即使继承了旧 OPEN 键，也没有新准入授权。发现变化、身份不明或账本缺失，相关分片和依赖它的用户 admission 入口默认 CLOSED。
2. 旧主在网络分区中可能还活着。**新管理代次解锁前必须先撤销旧入口的写路由/管理身份，隔离旧 Coordinator→角色的管理路径，并确认所有合法入口关闸。** 对无法确认已 fence 的旧路径保持关闭，不仅依赖旧进程自报退出。届时旧角色只能在原短租约内继续，到期自行停止，不能无限续租。
3. 人工恢复建立新的 control_epoch，经可信配置通道安装到所有拟重新使用的角色；角色只接受当前被安装代次及其签发的操作。未确认旧执行停止的原生占用继续隔离；不可达旧角色涉及的准入归属未能恢复时，阻塞对应 admission home/用户范围，无法界定范围时整个相关入口继续关闭。
4. 原生只读状态必须能恢复 `admission_id → 玩家/PlayerSession → allocation_id/group_id → 全拓扑 → profile/已占预算 → admitted_at/player_expires_at`，包括在途创建及清理终态。对账重建 service 占用、进程总预算和跨分片用户/admission claim，保留旧 deadline；不能只恢复“service occupied”。原池 claim 未查明，禁止转到另一池创建第二组。
5. 所有受影响分片、用户入口 claim 和角色均通过同一恢复屏障，才由人工显式打开新代次准入。事件缓存从新快照重建，所有旧代次管理请求被拒绝。无法完成屏障则保持隔离和告警，不降级为猜测空闲。

该合同用保守的恢复可用性换取不重复分配，避免在异步复制 Redis 上虚构不可回滚的计数器。未来若要求存储故障也自动接管，需另行引入能提供不可回滚 admission/epoch 权威的共识合同并验收，不作为本次前提。普通读缓存、丢失事件或 Redis TTL 都不是 free 的证据。

## 7. 原生协议需要的最小管理扩展

以下为本分支新增 managed 能力，**原 main 接口未全部提供**。先与 #509 整合到 white_box，再启用生产三角色池；不通过 sidecar 伪造 native 执行确认。

| 能力 | 最小合同与目的 |
| --- | --- |
| 无副作用观测 | 返回原生 idle/reserved/connecting/connected/cleaning/quarantined、group/operation、boot_epoch、清理结果及第6.3节完整 admission 恢复元数据；与会领取 token 的 get_connection_status 分开 |
| 精确服务和进程身份 | 注册/建组校验 boot_epoch、service_id、兼容指纹；地址复用不复活旧绑定 |
| 管理 owner 与 fencing | bind/renew/abort/destroy/finalize 绑定 control_epoch、allocation_id、group_id、执行代次及有界凭证；本地校验安装的管理代次、当前 owner 和过期时间；存储恢复遵循第6.3节关闸及人工屏障 |
| 个人 deadline 与 owner lease | 每个 service 本地保存固定玩家 deadline 和短租约，独立 watchdog 到期取消自己的组/子操作；白模仍保留预约；不无限续期 |
| 有界构建和取消 | connecting 也可按原 operation 请求停止并查询终态，或在本地 deadline 收尾；未确认则隔离，不强制宣布断开 |
| 清理证据与最终提交 | 原生完成计算/通信使用者清理后保留 CLEANED_PENDING_COMMIT；返回原组及 SP ranks/进程受影响范围的证据；最终确认后才开放本地 service |
| 进程共享容量/排空 | 本地校验 profile/预算，拒绝超出进程限额；drain 原子停止新组，不影响其他 service 的有效会话；记录进程级与 service 级故障 |

managed 模式下，原生 `destroy_group` 清理完成不能绕过控制面最终确认直接接受外部新 owner。需复用 white_box 的“验证清理后释放”思想扩展 group 管理，提供显式能力协商；旧 standalone 测试模式保持其现有协议，不能混入保证回收的生产池。

原生管理端仅接受获授权控制服务，业务输入/输出凭证限定玩家、绑定和子操作；组内传输也验证拓扑和身份。管理 token 不传浏览器、不进日志。`restart_required=true` 只触发隔离与告警，不被控制器转换为重启指令。

## 8. Room 试玩和双通道数据适配

### 8.1 保留逻辑父绑定

在现有 ModelBinding 后端增加 `disaggregated` 实现，与旧 bound 后端通过声明能力区分。逻辑父绑定持有 GroupAssignment，不再假设存在一条覆盖全部角色的物理父 WS。一个玩家的组长期保留，子操作完成只结束自身会话，不拆通信组；正常换段前须验证旧动态状态清理。

三角色数据路径为 Room → Encoder 输入 WS → ZMQ/NCCL → DiT → ZMQ/NCCL → Decoder 输出 WS → Room → 浏览器。HTTP/Redis/Coordinator 不转发 RGB、latent 或逐帧输入；Room 沿用现有 WebRTC/WS、UI 对齐及录像能力。浏览器和 CLI 不接触角色选择、服务端口及 NCCL。

### 8.2 并行收发与顺序

- 普通生成先连接 Encoder 并 init，持续读其回执；取得 native session.ready 中原 session/generation 和 output_attachment token 后连接 Decoder，发送 output.attach，等待匹配 output.ready。之后按窗口和消费 credit 供帧，不能把 session.ready 当作 Prefix 已预热。
- Encoder 回执读取、Decoder 输出读取与 source 发送独立推进；慢浏览器不能堵住控制/取消/credit 的读循环。使用有界 task/字节队列，不按帧创建无限后台任务。
- 两通道没有全局到达顺序：Decoder credit 可早于 Encoder accepted。分别保存 submitted/accepted/consumed 水位，保留包边界直到两端确认；不能因跨通道先后顺序错误退还窗口或重复送帧。
- Encoder 的 accepted/conditioning 只说明输入已接收/准备，不等于画面已应用新 Prompt。以输出 event_id 和实际条件版本关联首个新条件帧，保留现有热更新回执语义。
- Encoder 输入完成不等于最终模型帧全部返回。普通成功要求双通道终态、源/帧边界一致及子会话清理确认；异常关闭按取消清理，不伪装为成功 EOS。
- output WS 的绑定凭证、group/boot epoch、session/generation 必须相符。重连只接回有效原玩家；模型组故障后不重放半个 Segment 到另一 DiT。

### 8.3 Prefill、缓存与多玩家

Prefill 仍可选，来自各玩家私有视角；多人 F0 使用真实快照，不暂停共享 World。同一玩家在原组内顺序执行多组预热和生成。显式 prefill 调 Encoder 的原接口，由该组 DiT 保存前缀；按 native `prefill.completed` 的精确语义映射现有 memory_ready/子操作终态，不能按字面相似就提前归还 Slot。L2 持久化异步完成另计。

拆组保留 DiT L1/L2，下一玩家只按完整合法 key 使用静态前缀；动态 KV 有效范围、latent、RNG 和 encoder/decoder 会话状态不能跨玩家或跨段复用。cache hint 与查中结果分开，多个 DiT 的本地 L1 不保证相同；共享 L2 若已有受支持配置可沿用，不能假定自动复制或本轮引入全局 KV 服务。

多人共享 GameRoom/CLI，玩家各有不同 GroupBinding。E/D 进程共享仅复用模型权重和执行预算，不共享成员、输入、F0、会话缓存或 UI 归属。公共建房/加入/退出与有效成员浏览器重连要验收；CLI 成员 resume 本次仍只保留扩展位置。

## 9. 退出、故障和弹性操作

### 9.1 玩家结束

继续使用个人 deadline、主动退出、断线超时的统一关闭入口。并行撤销 SDK 成员/视角/输入和模型子操作；SDK 退休后独立归还成员资源，最后玩家空房收尾不等待故障 GPU。

模型分支取消输入、DiT/SP 计算、Decoder 输出和关键缓冲使用，关闭子操作，然后对 E/T/D 原 group 分别 destroy。202 只是受理，必须等原生终态及最终释放确认；不能用健康 200、available 一次回包或“socket 已关闭”代替。每个角色完成且相关跨角色通信使用者已停止后，可独立释放该 service 和其进程预算；共享进程其他 service 不销毁。

GroupBinding 保留逐角色结果：全部释放且 SDK 证据齐全才为 RELEASED；有未知角色则保持部分已释放、部分隔离的关闭记录，不能把已释放容量反复归还或把隔离容量计为 free。L2/录像等独立归档不阻塞计算回收；仍在引用本次 GPU/通信缓冲的工作必须真实完成。

### 9.2 故障矩阵

| 事件 | 对新玩家 | 对已有组 |
| --- | --- | --- |
| 新角色进程注册 | READY 且容量/兼容性通过后自动进入候选 | 不改已有配对 |
| 空闲 service 下线 | 立即停止分配，保留身份 tombstone | 无业务影响 |
| 主动 drain | 停止新组；空闲服务可清理下线 | 默认等各玩家结束/到期；不为缩容提前抢占，强制终止须明确操作 |
| 单 service 的连接故障 | 隔离该 service | 关闭所关联组的玩家并核验其他角色；确认隔离范围后同进程其他 service 可继续 |
| E/D 共用 CUDA/模型/进程故障 | 摘除该进程全部 service | 通过反向索引通知全部关联组；不把它误报成单用户故障 |
| DiT/SP 任一 rank 故障 | 整个 DiT service 不可用 | 关联玩家结束；SP2 不能留下半组继续 |
| 网络分区/控制器丢失 | 过 lease 阈值停止准入 | 各端按个人期限/owner lease 撤销旧工作，未知清理隔离；不双重分配 |
| Redis 故障/切主 | 拒绝新准入；禁止透明自动切主后重新分配 | 原生短租约到期清理；先 fence 旧入口、重建角色占用和 admission 归属，再人工开启新 control_epoch |
| restart_required | 剔除并告警 | 人工恢复；不自动发 SIGUSR1/PlannedRestart 或启动替代进程 |

“自动剔除/注册”是资源发现和调度行为，不是故障模型自动重启。运行中任一角色真正丢失后，首版不承诺热替换三元组或无缝续玩；保留其他健康玩家，结束受影响 PlayerSession 并释放/隔离原资源。下一次明确的新试玩走新身份准入，不以重试延长旧个人时长。

## 10. 高并发与性能边界

1. **分离控制面和数据面。** 调度只在入场、续租、退出和状态变化时执行；不每帧查 Redis，不经中心服务器转发 GPU 数据。
2. **本地候选缓存加原子提交。** API 实例订阅增量变更、保存有版本快照；事件丢失重拉。心跳合并更新、周期抖动；最终原子校验不相信陈旧排序结果。
3. **按池横向扩展。** 多分片各管独立资源预算域，控制器以 per-group 有界执行权并发建组。每进程限制并发 NCCL 建连数，避免大量准入同时压垮 TCPStore/GPU。
4. **多级背压。** Room 的帧/字节窗口、native 有界队列、进程 profile 限额和建组信号量共同限流；已准入持续 source 优先于可推迟的预热突发，但不能通过无界优先级让任一合法会话永远饿死。
5. **共享角色公平性。** 对 service 轮转/工作量配额，基于执行时间而非仅请求数核算 Qwen、source VAE、decoder 工作。长 CUDA kernel 不可假定可抢占；Prefill 最大形状、burst 限额和实际最差阻塞需压测，必要时将不相容 profile 分池。
6. **网络与内存成本显式计入。** 测真实 E→T、T→D 带宽/时延和 Room→E、D→Room RGB 传输；同可用区优先，跨区默认禁止。NCCL/RDMA 可用性和实际回退路径分别报告，不依赖云产品名推断。
7. **不宣称相乘加速。** 满载吞吐受共享进程和网络瓶颈限制；service 增加也可能增加尾延迟。对照以每玩家 FPS/输入延迟/帧年龄和整体可承载玩家数同时评估。

空闲通信组预建池作为后续基于建组耗时证据的优化，首版按需建组，避免预配关系占死灵活容量。活动组不因负载变化自动重排；在线扩容对新组立即生效，缩容按 drain 和个人期限推进。

## 11. 实施拆分与版本迁移

本节保留实施分解。当前分支实现内容、显式能力边界与配置以[实现指南](/docs/sglang-diffusion/zing/specs/MANAGED_ROLE_POOL)为准；自动性能调参、缓存感知择机、增量事件订阅与按实际 GPU 执行时间调度尚未实现；GPU 性能验收仍需后续证据，不能把设计意图当作已验证结果。

| 阶段 | 模块/交付 | 必须通过的门槛 |
| --- | --- | --- |
| A. 源码整合 | 将 #509 所需变更完整整合到 white_box，保留 #508 的玩家期限、清理和分页 L1；核对共享文件冲突 | 旧单体路径与新 standalone 合同并存，不覆盖旧修复；禁用的旧多 DiT shared API 不恢复 |
| B. 管理原语 | worker_connections/services/http、角色入口、worker_heartbeat | 自注册、只读状态、epoch/fencing、deadline/drain、进程容量和真实清理回执 |
| C. 分布式调度 | coordinator/assignment/schema、分片存储、建组执行器 | E/T/D 联合占用、部分失败补偿、controller 接管、无全局大锁、新旧模式容量隔离 |
| D. 试玩适配 | ModelBinding/ModelLeases、ControlPrefill、RenderSegment/Room 输出处理 | 双 WS、有界背压、精确条件回执、同组换段、每玩家 TTL、真实回收 |
| E. 生产接入 | WTP 容量/错误提示、部署元数据、监控和运维 | 浏览器合同尽量保持，显示完整组容量；联机/录制/输入/接续和故障验收 |

旧 assignment 保留旧读写解释，新模式使用明确版本和 backend 字段；所有读者先识别新状态后启用新准入。不将旧 denoiser 与新 dit 同时登记成同一 GPU 的两份可用资源。实例切换须排空，回退不能丢弃新模式未完成组/隔离账本，不能从数据库删除记录伪造回退完成。

部署代码、内嵌 Worldkit CLI/SDK 均来自对应远端 white_box 准确 SHA；Worldkit dev、World Test main 变更先纳入各自 white_box。上海/泰国同一组件同一代码身份，仅配置不同。构建完整镜像，禁止容器热改、旧镜像覆盖文件或运行脏代码。发布/计划重启另行明确授权，本稿不申请资源或执行云操作。

## 12. 验收与观测

### 正确性与故障注入

| 场景 | 核验结果 |
| --- | --- |
| E1 多 service 对接不同 DiT，D1 同时解码不同 DiT | 多组输入、条件、KV、latent、输出与 credit 不串；进程总预算不超限 |
| 任意 E/T/D 数量、不同卡型/规格和 SP1/SP2 | 无固定比例；不兼容组合拒绝；健康余量可组合；rank 不被当独立 Slot |
| 新进程加入、重复注册、乱序心跳、地址复用 | 按 boot_epoch 区分，旧回包不复活；不影响已有组 |
| 并发多控制器争抢、重试、跨池重试 | 同一 service/预算和玩家只分配一次；原操作可查询，不产生双人物/双组 |
| 原生预留响应丢失、10 秒边界、build 部分成功/迟到成功 | 未知占用保留并补偿，不漏第三个可能成功的组端 |
| 控制器崩溃、Redis 切主/记录丢失、网络分区 | 同主副本接管有 fencing；存储变代默认关闸，旧主仍存活时新主不得给同 admission 在另一套 E/T/D 建组；完成旧路径隔离、完整 admission 对账与人工屏障后再开放 |
| 角色 drain、共享进程故障、单 service 故障、SP rank 故障 | 正确传播影响范围；原组释放/隔离，不重放到新 DiT、不自动重启 |
| 三角色 destroy 202、ACK 丢失、迟到 finalize、cleanup 超时 | 不假 free；已确认资源可独立归还，旧回执不释放新 owner |
| 纯白模、个人到期、退出、浏览器失联、有效成员重连 | 保留完整容量保证、不续时；同局其他健康玩家继续；无整局上限 |
| Prefill L1/L2/miss、多组预热、Prompt 热更新和跨 Segment | 输入/key/F0/版本一致，命中实际可查；不暂停 World，旧段状态不串 |

控制面规模测试用模拟角色覆盖 10²/10³/10⁴ service 与多 API/控制器，记录准入/清理 p50/p95/p99、冲突和超时、Redis 操作量、CPU/内存及队列年龄；再测多分片扩展，不把模拟吞吐当作 GPU 性能。故障用受控环境，不破坏真实玩家或训练任务。

### 正式端到端与性能

遵循[发布与验收规范](/Users/kejun/.codex/policies/whitebox-release-and-e2e.md)及[报告模板](/Users/kejun/.codex/templates/whitebox-e2e-report.md)。固定 5 个当前正常动态 case 与旧版同条件基线，每个正式玩家从首个可见模型画面起至少 180 秒，A/B 交接两场分别满足。覆盖联机建房/加入、同步、输入与视角隔离、退出/浏览器重连；CLI 成员 resume 明确不在本次范围。

角色共享至少测 1→2→N 组混合 Prefill/持续生成；每种拟上线异构组合分别测，480 与 736 配方分开，缺少资源或组合未验证就标未覆盖，不用其他卡型推断通过。记录各角色/各 rank 常驻及峰值显存、Host/CUDA 执行、共享线程等待、NCCL 建组/拆组和传输、TTFF/输入到可见帧、FPS、RTT、credit/队列年龄、L1/L2恢复耗时及画质。

监控按 process/service/group/player/segment 关联：注册健康、候选剔除原因、完整组容量、各角色预算、组状态年龄、10秒预留失效率、半组数量、清理证据、受故障影响的组数。不要在 Prometheus 上无限使用 session_id 作为高基数标签；会话明细放可查询事件/日志，聚合指标按池、角色、profile、原因统计。

画质异常保存截图；全部观察到的退化、失败、不足时长及未覆盖逐项报告，不能用聚合吞吐掩盖某玩家 FPS 降低或卡顿。真正的 Slot 回收由新玩家用新身份完成输入和模型输出证明，不能通过重启让测试通过。

## 13. 代码依据

- [main 分离接入指南（固定 SHA）](https://github.com/seedleap/sglang/blob/0a4623e1299efc386a81473ed03d6a79ed234eaa/docs/docs/sglang-diffusion/zing/DISAGGREGATED_SERVING_GUIDE.md)：service 复用、双 WS、cache/Prefill、当前 SP 支持。
- [Worker 通信组协议](https://github.com/seedleap/sglang/blob/0a4623e1299efc386a81473ed03d6a79ed234eaa/docs/docs/sglang-diffusion/zing/specs/WORKER_CONNECTION_API.md)、[ServiceConnections](https://github.com/seedleap/sglang/blob/0a4623e1299efc386a81473ed03d6a79ed234eaa/python/sglang/multimodal_gen/runtime/realtime/worker_connections.py)：查询取得预留、build shield、connecting/destroy 边界和清理状态。
- [worker_services](https://github.com/seedleap/sglang/blob/0a4623e1299efc386a81473ed03d6a79ed234eaa/python/sglang/multimodal_gen/runtime/realtime/worker_services.py)、[Encoder server](https://github.com/seedleap/sglang/blob/0a4623e1299efc386a81473ed03d6a79ed234eaa/python/sglang/multimodal_gen/runtime/entrypoints/zing_source_encoder_server.py)：service 公平接入和共享模型执行循环。
- [测试 Master](https://github.com/seedleap/sglang/blob/0a4623e1299efc386a81473ed03d6a79ed234eaa/benchmark/zing_h3_causal/worker_group_master.py)、[双通道客户端](https://github.com/seedleap/sglang/blob/0a4623e1299efc386a81473ed03d6a79ed234eaa/benchmark/zing_h3_causal/decoder_output_client.py)：显式地址建组，输出 attachment 和跨通道 credit；仅作为协议参考。
- [white_box Coordinator](https://github.com/seedleap/sglang/blob/5d7e71a87ab81cf61bcea60e03c606a2121ea3a0/python/sglang/multimodal_gen/runtime/realtime/coordinator.py)、[ModelBinding](https://github.com/seedleap/sglang/blob/5d7e71a87ab81cf61bcea60e03c606a2121ea3a0/python/sglang/multimodal_gen/runtime/realtime/whitebox/model_binding.py)、[worker_reservation](https://github.com/seedleap/sglang/blob/5d7e71a87ab81cf61bcea60e03c606a2121ea3a0/python/sglang/multimodal_gen/runtime/realtime/worker_reservation.py)：旧角色分配、bound 父绑定、玩家期限及验证释放基线。
- [跨平台变更核对入口](https://github.com/seedleap/worldkit-monorepo/blob/dev/docs/development/cross-platform-change-checklist.md)：WTP 业务归属、SDK 世界所有权、Room/Gateway/Coordinator 与部署依赖。

本轮固定源码、工作树和技能来源的取证材料在本机 `/Users/kejun/.codex/tmp/whitebox-disagg-design-20261103/`。线上资源和配置没有在本轮重新检查；本文不把源码已有能力标为线上已启用。
