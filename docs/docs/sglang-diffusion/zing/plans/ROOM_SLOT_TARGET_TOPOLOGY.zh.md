# 共享联机世界与每玩家 Prefill 的目标拓扑

状态：最终方案对应的目标架构，尚未实现或部署。更新：2026-11-02。以[多人试玩与每玩家 Model Slot 生命周期最终方案](/docs/sglang-diffusion/zing/plans/ROOM_SLOT_LIFECYCLE_RECOVERY.zh)为准。

一个 GameRoom 共享一个 SDK 权威世界，可以容纳多个玩家；平台不设置整局最长时间。每个玩家拥有独立 PlayerSession、私有视角、输入授权和一个 Zing Worker 的 Model Slot。Prefill 是各玩家独立的模型计算，不共享玩家生命周期、不暂停共享世界，也不要求所有玩家同时预热。

## 1. 总体拓扑

下图为一个区域内的一局双人联机。多个 GameRoom 可重复此结构；玩家模型 Worker 可以位于不同 GPU 机器，每 Worker 同时最多服务一位玩家。

```mermaid
flowchart TB
    A[玩家 A 浏览器]
    B[玩家 B 浏览器]
    GW[WTP 鉴权与 Gateway<br/>加入资格 控制与信令]
    C[Coordinator 与回收控制<br/>每玩家期限 服务租约与模型容量]
    DB[(Redis<br/>游戏和玩家生命周期 占用与清理收据)]

    subgraph CPU[CPU 侧 Room 服务]
        subgraph GAME[共享 GameRoom G]
            WORLD[SDK 权威 World<br/>唯一游戏状态 物理与模拟时钟]
            VA[玩家 A 的 Replica 与视角<br/>私有输入绑定 媒体与UI]
            VB[玩家 B 的 Replica 与视角<br/>私有输入绑定 媒体与UI]
            WORLD -->|状态复制| VA
            WORLD -->|状态复制| VB
            VA -->|A的授权输入| WORLD
            VB -->|B的授权输入| WORLD
        end
        PA[PlayerSession A<br/>独立 Prefill context 与模型父连接]
        PB[PlayerSession B<br/>独立 Prefill context 与模型父连接]
        VA <--> PA
        VB <--> PB
    end

    subgraph GPUA[GPU 机器 1]
        WA[Zing Worker A<br/>一个 Slot 属于玩家 A]
        KA[(A所在Worker的 L1 / 本地L2)]
        WA <--> KA
    end
    subgraph GPUB[GPU 机器 2]
        WB[Zing Worker B<br/>一个 Slot 属于玩家 B]
        KB[(B所在Worker的 L1 / 本地L2)]
        WB <--> KB
    end

    A <--> GW
    B <--> GW
    GW <--> PA
    GW <--> PB
    PA <-->|长期 bound WS| WA
    PB <-->|长期 bound WS| WB
    PA -->|白模或模型媒体| A
    PB -->|白模或模型媒体| B
    PA -. 申请和续租 .-> C
    PB -. 申请和续租 .-> C
    C <--> DB
    C -. 预约 期限与精确清理 .-> WA
    C -. 预约 期限与精确清理 .-> WB
    WA -. 健康和清理证据 .-> C
    WB -. 健康和清理证据 .-> C
    C -. 游戏或玩家级生命周期 .-> GAME
```

GameRoom是共享世界，不是每玩家的推理会话。SDK管理权威模拟和成员，各视角使用只读副本/独立相机渲染，不推进第二个玩法时钟。CLI托管执行可以包含Node、权威浏览器和逐玩家视角资源，不能只根据PID数量定义一个世界实例。

浏览器使用已为自己建立的私有媒体/control连接；不能为接入模型再创建一个RoomClient或重复join。WebRTC媒体逻辑上从Room媒体端到浏览器，不经过Coordinator/Redis；具体ICE和中继沿现有部署。录像和诊断异步归档到OSS，非计算归档不阻止Slot回收。

同一游戏的A和B分别拥有一个Worker，因此两名可随时切换模型的玩家需要两份可用模型容量。CLI支持多人，不代表单个Zing Worker可以服务多人。

## 2. Prefill 是逐玩家独立的

```mermaid
sequenceDiagram
    participant G as 共享 GameRoom
    participant VA as A的私有视角
    participant PA as PlayerSession A
    participant WA as Worker A
    participant VB as B的私有视角
    participant PB as PlayerSession B
    participant WB as Worker B

    Note over G,WB: 两名玩家已各自完成成员及模型Slot准入
    G-->>VA: A视角对应的真实状态和画面
    PA->>VA: 从已有帧流固定A的F0快照
    VA-->>PA: 玩家绑定 视角epoch tick PTS 像素摘要
    PA->>WA: A的Prompt 参考图 F0 执行Prefill

    par A独立预热
        WA-->>PA: prefill.memory_ready
        WA-->>PA: bound.completed cleaned=true
    and B继续游戏
        VB->>G: B的输入与动作
        G-->>VB: 新的共享状态和B视角
        VB-->>PB: B的白模帧
        PB->>WB: B的模型输入
        WB-->>PB: B的模型画面
    end

    Note over G,WA: A的Prefill不停止共享World 不关闭B的输入或媒体
    PA->>WA: 新模型段使用同一F0和匹配条件
    VA-->>PA: 后续实时帧
    PA->>WA: 按有界选帧规则继续输入
    WA-->>PA: A的模型画面
    Note over PA,WA: A换段清理自己的动态状态 父WS和Slot保持
```

A、B均可在各自Worker上预热，只有同一玩家的子计算串行。每位玩家的source/model session、context_id、视角绑定、帧身份和请求结果都隔离。共享的是游戏状态，不是一个所有玩家共用的Prefill任务。

现有单人capture hold只能用于独占世界。多人使用逐玩家F0快照，保留真实capture时间；过旧、视角重建、场景或条件身份改变时，只重做该玩家的准备。不能把更晚的新帧悄悄替换旧key的F0，也不能因World每一tick变化就让所有缓存全失效。

静态前缀可按合法完整内容key复用；动态KV、近期latent、RNG、编码/解码状态按每段创建和清理。一个玩家预热失败不应影响另一个玩家的模型Slot。

## 3. 玩家离开与整局关闭是两个作用域

```mermaid
sequenceDiagram
    participant E as A退出或个人到期或失联
    participant C as Coordinator与回收控制
    participant D as 生命周期账本
    participant SDK as GameRoom及A的SDK成员owner
    participant W as Worker A
    participant B as 玩家B及Worker B

    E->>C: begin_close scope=player A的完整身份
    C->>D: A进入CLOSING 冻结close_id
    par 关闭A的游戏接入
        C->>SDK: 撤销A输入 leave A成员 关闭A视角
        SDK-->>C: A的SDK终态收据
        C->>D: 独立核销A的成员位 视角预算 游戏引用
        alt B仍在局内
            Note over SDK,B: 共享World继续 B的身份 输入和Slot不变
        else 无成员且无未决join
            C-)SDK: 启动独立空房收尾 不等待结果
            Note over SDK,D: 世界停止结果独立记录 不作为A模型释放屏障
        end
    and 关闭A的模型计算
        C->>W: 取消A子操作 清理A父绑定
        W-->>C: CLEANED_PENDING_COMMIT或清理失败
    end

    alt A的关键清理证据齐全
        Note over C,D: SDK侧引用已保存的退休收据或覆盖A的GameStopProof
        C->>D: A进入RELEASING
        C->>W: finalize_release A的预约
        W-->>C: A的ACK
        C->>D: A进入RELEASED 归还A的Slot
    else A的模型清理无法确认
        C->>D: 隔离A的Worker 告警 人工恢复
    end
```

个人清理需要“A成员、输入与视角已退出”的证据，不要求共享CLI进程停止。SDK确认退休后成员位/视角预算可独立归还，故障GPU不能将它们永久锁住；模型Slot仍按自己的确认协议释放。空房收尾独立运行和记录world_stop_complete；A已有个人SDK终态收据时，不等待共享世界关闭或GameStopProof才释放A的模型Slot。一个玩家返回页面、个人期限到达或浏览器关闭，都不能被解释为解散整局。

明确授权解散、游戏已有结束事件、SDK共享权威故障或owner租约丢失时，GameRoom先关闭新成员准入，再并发关闭全部PlayerSession和共享世界。world_stop_complete与所有玩家模型容量已释放分别记录；个别Worker隔离不伪装为已恢复，也不要求继续运行没有玩家的世界。GameStopProof可作为SDK侧终态确认，不能代替各模型Worker的清理和最终释放ACK。

创建者离开默认只退出自己。最后玩家离开与新join竞争时，使用同一GameRoom状态版本和pending join登记，防止世界刚关闭又出现孤儿玩家。

## 4. 每玩家限时，世界按成员存活

```text
player_expires_at = 该玩家成功预留 Model Slot 的时间 + 个人配置 T
```

| 对象 | 目标规则 |
| --- | --- |
| 上海 PlayerSession | 初值 1800 秒；只结束该玩家 |
| 泰国 PlayerSession | 初值 7200 秒，发布前核对进程配置；只结束该玩家 |
| GameRoom | 没有平台统一最长时间；还有有效玩家就可持续存在 |
| 服务 owner | 短租约和心跳持续续期，用于发现失联和隔离旧 owner |
| 新建世界/pending join | 有界启动与补偿；无人加入不能永久常驻 |
| 浏览器重连 | 外层宽限初值 15 秒，首次合格恢复协商最多 13 秒；不越过个人期限和 SDK 原成员有效期 |

A 10:00、B 10:20 准入且 T=30 分钟时，分别在 10:30、10:50 结束。A 到期不关闭 B；最后玩家真正离开且没有未决加入时，才关闭共享 CLI。当前代码的 WHITEBOX_GAME_SECONDS 固定房间上限需移除，服务租约和清理账本仍保留，不能把 Redis 记录消失当作 GPU 已释放。

Prefill 是可选能力，多组预热、白模、热更新、换段和重连都不续个人期限。未来游戏可自行触发结束，本次不新增游戏时限配置或通用策略体系。

可新增玩家数受剩余成员位、CPU 视角容量和兼容空闲 Worker 共同限制；每 Worker 一个玩家，以多 Worker、多机器支持多人。硬故障仍隔离告警、人工恢复，不自动重启；健康玩家和 Worker 继续服务。

## 5. CLI现状与需要接通的部分

- 当前上海安装包包含worldkit room多人命令；room命令模块与本轮SDK main对应模块SHA一致。
- 默认worldkit stream适配器仍为一个玩家视角，不能据此否定CLI本体的多人房间能力。
- 仓库已有WorldkitRoom、成员/视角适配以及实际双玩家CLI测试源码和交接记录。
- 上海现网Gateway已挂载room-control和玩家媒体路由，createRoom/admitPlayer/leavePlayer的服务端执行链已部署。CLI成员resume尚无Gateway控制action和运行时调用，按用户确认本次只保留底层扩展位置，不适配接口；当前试玩WS重连不能等同于该能力。此现网挂载状态与本工作树RoomProxy存在差异，须以部署读回为准。
- 当前Prefill分段客户端的已有game_id加入限制需要改造；把已认证的玩家视角接到其自己的ModelBinding，不重造Prefill算法。
- 多人F0获取不能使用独占World的时钟暂停；这属于源输入桥接调整，不是Prefill要跨玩家协调。
- 本轮没有运行现网双人CLI＋两台Zing Worker的完整验收，不能把命令存在或源码测试当作生产链路已验收。

本文只描述目标架构；未修改运行代码、并发配置或云资源，未启动新的联机/GPU测试。

## 6. 实施边界

复用现有 Gateway/Coordinator/Room/SDK/bound 协议，补齐联合准入、浏览器重连、每玩家期限及真实回收。CLI resume 仅保留已有扩展位置，不新增接口适配。成员仍有效时恢复原连接；成员已失效则结束旧试玩并回收。开发在 worktree 中完成，所有部署源码和内嵌 CLI/SDK 统一来自各自远端 white_box 的准确提交；地区只用配置区分，禁止热补丁。

部署须另行明确授权。验收遵循统一发布规范：5 个当前正常 case、每个正式玩家会话从首个可见模型画面起至少 180 秒，与旧版同条件逐项比较性能、实际缓存召回、延迟及画质，记录失败/未覆盖和所有恶化。联机创建/加入、状态同步、输入与视角隔离、退出/浏览器重连必须通过；CLI 成员 resume 明确为本次范围外。容量不足就标验收未完成，不能以单人测试替代。短 TTL 专项不能代替正式时长。
