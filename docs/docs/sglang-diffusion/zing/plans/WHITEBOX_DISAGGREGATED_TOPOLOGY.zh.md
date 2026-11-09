# 最新服务拓扑：Managed 三角色试玩链路

对应本地实现提交 `a0492805f13eb79eceeaf0f54a54ddd701ceaa07`。代码已完成本地测试，尚未部署到线上。本图描述已实现的组件关系，不是上海或泰国当前线上机器清单。角色数量、GPU 类型和配对仅作示例。

配置与能力边界见 [Managed role pool 实现指南](/docs/sglang-diffusion/zing/specs/MANAGED_ROLE_POOL)。

## 1. 总体服务拓扑

实线表示业务调用或数据流，虚线表示模型资源管理。图中服务可以独立部署，分组不表示必须位于同一台机器。

```mermaid
flowchart TB
    B[玩家浏览器]
    WTP[World Test / Console 后端<br/>身份、发布物、建房与加入资格]
    GW[Gateway<br/>鉴权、控制和信令代理]

    subgraph ROOM[CPU Room 服务：可有多个实例]
        WORLD[GameRoom / CLI / SDK<br/>共享世界、成员、物理和模拟时钟]
        PLAYER[PlayerSession<br/>私有视角、输入、个人期限]
        BIND[ModelBinding<br/>固定三角色组、子操作串行、双 WS 适配]
        WORLD <-->|成员状态 / 私有视角 / 输入| PLAYER
        PLAYER <--> BIND
    end

    subgraph COORD[Coordinator 服务：同池可多副本]
        API[内部 API<br/>房间目录与角色池入口]
        GROUP[GroupController + 回收循环<br/>准入、建组、续租、排空、清理]
        API <--> GROUP
    end

    ROOMDB[(Room 目录 Redis<br/>GameRoom 到 Room owner 的映射)]
    POOLDB[(Role pool Redis<br/>进程、service、预算、组和清理记录)]
    HOME[(Admission Redis<br/>跨池用户与入场请求去重)]

    E[Encoder 实例池<br/>每进程多个 service + 一个 Reporter]
    T[DiT 实例池<br/>每实例一个 service / SP1 或 SP2]
    D[Decoder 实例池<br/>每进程多个 service + 一个 Reporter]

    B -->|业务 HTTP| WTP
    WTP -->|房间控制 HTTP| GW
    B <-->|WS 控制 / 信令 / 可选 WS 媒体| GW
    GW <-->|路由后的控制 / 媒体| PLAYER
    GW -->|查找 Room owner| API
    GW -->|创建 / 加入 / 离开| WORLD
    API <--> ROOMDB
    BIND -->|申请完整组 / 续租 / 关闭| API
    API -->|返回 GroupBinding 与实际端点| BIND
    GROUP <--> POOLDB
    GROUP <--> HOME

    E -. 注册进程与全部 service / 心跳 .-> API
    T -. rank 0 注册 / 心跳 .-> API
    D -. 注册进程与全部 service / 心跳 .-> API
    GROUP -. 签名 managed HTTP .-> E
    GROUP -. 签名 managed HTTP .-> T
    GROUP -. 签名 managed HTTP .-> D

    BIND <-->|Encoder 输入 WS / 输入回执| E
    E -->|ZMQ 控制描述 + NCCL 条件与 source latent| T
    T -->|ZMQ 控制描述 + NCCL latent| D
    D -->|Decoder 输出 WS：RGB / source.consumed| BIND
    PLAYER -->|WebRTC 视频：直达或经中继| B
```

关键边界：

- GroupController 和回收循环运行在 Coordinator 副本内，当前不是额外部署的独立 Master 服务。
- Encoder/DiT/Decoder 通过 Coordinator 注册，由 Coordinator 写入 Redis；角色进程不直接修改分配账本。
- Redis 不保存逐帧 RGB、latent 或模型 KV。Gateway/WTP 不转发 GPU 间张量；Room 直接连接已绑定的 Encoder 和 Decoder。
- 三类 Redis 是逻辑职责，可以按配置共用实例或使用不同实例；同一准入范围内各池共用 admission home。当前客户端固定 Redis primary 身份，不提供透明切主或在线重分片。
- 一个 Coordinator 配置一个 ServingPool；该池的多个副本共享同一权威账本。增加独立池时配置对应 Coordinator/资源域，不把同一进程预算重复登记到不同池。

## 2. 每位玩家如何绑定三种角色

示例中 A、B 属于同一 GameRoom，C 可以属于另一局。所有玩家各有私有视角和逻辑 Model Slot；共享世界不等于共享模型会话。

```mermaid
flowchart LR
    A[玩家 A 的 ModelBinding]
    B[玩家 B 的 ModelBinding]
    C[玩家 C 的 ModelBinding]

    subgraph E1[Encoder 进程 E1：共享权重和总预算]
        EA[service 0：属于 A]
        EB[service 1：属于 B]
        EF[service 2：尚未绑定]
    end
    subgraph E2[Encoder 进程 E2：可用不同卡型]
        EC[service 0：属于 C]
    end

    T1[DiT T1：一个 service<br/>SP2 / 两个 rank / 属于 A]
    T2[DiT T2：一个 service<br/>SP1 / 属于 B]
    T3[DiT T3：一个 service<br/>SP1 / 属于 C]

    subgraph D1[Decoder 进程 D1：共享权重和总预算]
        DA[service 0：属于 A]
        DB[service 1：属于 B]
    end
    subgraph D2[Decoder 进程 D2：可用不同卡型]
        DC[service 0：属于 C]
    end

    A -->|输入 WS| EA -->|组 A| T1 -->|组 A| DA -->|输出 WS| A
    B -->|输入 WS| EB -->|组 B| T2 -->|组 B| DB -->|输出 WS| B
    C -->|输入 WS| EC -->|组 C| T3 -->|组 C| DC -->|输出 WS| C

    classDef playerA fill:#e8f1ff,stroke:#3975c6,color:#173d72
    classDef playerB fill:#e9f7ef,stroke:#3d9664,color:#20543a
    classDef playerC fill:#fff3e3,stroke:#c38b39,color:#704816
    class A,EA,T1,DA playerA
    class B,EB,T2,DB playerB
    class C,EC,T3,DC playerC
```

E1 有三个 service，就有三个 service 注册记录和一份进程记录。如果 E1 的已验证预算只能承载两位玩家，service 2 即使没连接，也不能继续分配。

- 每个 service 同时只承载一个玩家；E/D 进程可以共享模型权重和执行资源，但其 service 各有独立连接与状态。
- SP2 的两张卡/两个 rank 共同形成一个 DiT service；只由 rank 0 注册，不能计为两个 Model Slot。
- 配对在准入时动态选择，不固定 1:2；可以有任意相容且预算足够的进程/service 数量。新增节点参与后续分配，健康活动组保持原配对。
- L1/L2 前缀缓存归 DiT 管理。换段清理动态 KV、latent 和编解码状态；静态前缀只按完整合法 key 复用。

## 3. 一位玩家的建立、使用与释放

```mermaid
sequenceDiagram
    participant R as Room / PlayerSession
    participant C as Coordinator / GroupController
    participant L as Redis 归属与预算
    participant E as Encoder service
    participant T as DiT service
    participant D as Decoder service

    R->>C: admit：原 admission_id / 玩家 / profile
    C->>L: 用户入场去重 + 联合 HELD 三服务和预算
    Note over C,L: 固定个人 deadline，不设置整局最长时间
    par 各 service 预留
        C->>E: managed/reserve
        E-->>C: 原生 10 秒预留
    and
        C->>T: managed/reserve
        T-->>C: 原生 10 秒预留
    and
        C->>D: managed/reserve
        D-->>C: 原生 10 秒预留
    end
    Note over C,D: 相同 group 和拓扑；原生 token 在各自 service 内保留
    par 建立通信组
        C->>E: managed/build
        E-->>C: connected
    and
        C->>T: managed/build
        T-->>C: connected
    and
        C->>D: managed/build
        D-->>C: connected
    end
    C->>L: 三方确认后 BOUND
    C-->>R: GroupBinding / 端点 / 个人期限
    Note over R,D: SDK 私有视角和模型组都就绪，才发布可试玩
    R->>E: 当前生成子操作：输入 WS + init
    E-->>R: session.ready + output attachment
    R->>D: 当前子操作：输出 WS + output.attach
    D-->>R: output.ready
    par 持续输入
        R->>E: source / Prompt / 参考图更新
        E-->>R: 输入 accepted / 条件回执
    and 持续输出
        D-->>R: RGB / 实际 event_id / source.consumed
    end
    Note over R,D: 子操作结束关闭自己的 WS；通信组和预算继续保留
    R->>C: 个人退出 / 到期 / 断线超时
    C->>L: CLOSING / 原组和执行代次
    Note over R,D: SDK 成员清理独立推进；模型取消不依赖浏览器 ACK
    par 精确组清理
        C->>E: managed/destroy
        C->>T: managed/destroy
        C->>D: managed/destroy
    end
    Note over C,D: 等 native CLEANED；再逐角色 finalize，收到确切回执后归还容量
    C->>L: 全部角色释放 + SDK 终态确认后 RELEASED
```

ModelBinding 是长期逻辑父绑定，不是一条覆盖全部角色的物理父 WS。当前实现为每次生成子操作建立 Encoder 输入 WS 和 Decoder 输出 WS；换段可重新建立业务 WS，但不重新分配三角色，也不自动拆掉 NCCL 通信组。

显式 Prefill 使用 Encoder 的 Prefill WS，计算/恢复 DiT 前缀缓存，不需要 Decoder 出图；Decoder service 和预算仍为玩家保留。`memory_ready` 与 L2 持久化确认分开，持久化检查在 DiT 侧进行。

## 4. 扩缩与故障处理位置

| 事件 | 负责组件 | 行为 |
| --- | --- | --- |
| 新进程加入 | 角色 Reporter → Coordinator | 模型就绪、manifest 和预算校验后加入候选 |
| 心跳失联 | Coordinator 的候选过滤 | 剔除候选，保留占用和未确认清理记录 |
| 缩减 Encoder/Decoder | Operator → Coordinator → 指定进程 | DRAINING 禁新组；等待现有玩家及在途操作收尾，再由部署管理者停止进程 |
| 用户退出或个人到期 | Room + Coordinator + 各角色本地 watchdog | 精确取消并确认三个角色；SDK 成员独立清理 |
| E/D 共享进程硬故障 | 进程内保护 + Coordinator | 影响其关联组，隔离告警；不自动重启或迁移活动会话 |
| Redis 主身份变化 | 固定 primary 检查与准入屏障 | 停止新操作；外部证据和原生恢复查询完成核对后人工恢复 |

角色数量由部署管理决定；注册/摘流/绑定自动化不等于自动购买 GPU 或启动云实例。CLI `room resume` 仍仅保留扩展位置，浏览器对有效原成员的重连正常保留。
