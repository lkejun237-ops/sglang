---
title: "Room 手动扩容与动态接入"
description: "新增 Room 的动态登记、容量预留、原会话路由和人工排空，不重启既有服务。"
---

本实现基于 main `940ba7625b0f36d935755a2a946fa12058760260`，独立任务分支，不依赖尚未合入的 SQLite 下线 PR。新增 Room/CLI 容量不重启已有 Gateway、Coordinator 或 Room；首次启用这套能力仍需要一次正常版本升级。没有自动伸缩、活跃世界迁移或 WTP 接口变更。

## 使用与范围

1. 在现有发布 bindings 增加 `room_pool_compatibility`（同一池固定同一验证过的镜像/SDK/协议组合标识）。该字段为 opt-in；不填写时保留原静态配置和路由。
2. Gateway、目录 Coordinator、Room 首次完整构建/部署到这个版本。Coordinator 另配 `WHITEBOX_ROOM_ADMIN_TOKEN`，必须与内部服务 token 不同；Room 不持有 operator token。
3. 每个 Room 使用独立 owner、进程代次、直连 HTTP 地址、私有数据目录，以及准确到达该实例的 RTC 公网 IP/端口。多源必须配置独立媒体端点；更改 `max_bridges` 不等于新增模型 Slot。
4. SDK、原生编解码器、媒体入口与预留资源检查完成后，操作者登记新实例。登记会实时读取受内部鉴权保护的 Room 描述，拒绝非空实例、错误池、重复 owner、地址或媒体端口。
5. Coordinator 原子预留房间/视图容量。新 GameRoom 才分配到新 owner，已有游戏和重连继续走原 owner。Gateway 不再为增加 owner 重启。

初次上线只在已排空的测试/发布窗口建立新池；本轮不实现把正在运行的旧静态 Room 自动转换为受登记容量管理的 Room。

## 操作入口

`deploy/whitebox/room_instance.py` 从已有非秘密发布 bundle 生成一个新容器的 `room.env` 和 `plan.json`，只写本地文件，不调用 Docker 或修改现有实例。示例参数是占位输入，不是生产配置：

```sh
python deploy/whitebox/room_instance.py \
  --bundle /private/pool-bundle.json \
  --name room-b \
  --image registry.example/room@sha256:REPLACE_WITH_VERIFIED_DIGEST \
  --room-url http://10.0.1.34:28092 \
  --public-ip 192.0.2.1 --public-port 4444 --listen-port 40001 \
  --state-dir /var/lib/room-b --output-dir /private/room-b \
  --secret-file /private/room-runtime-secrets.env --cpus 4 --memory 8g
```

该模板一次配置一个源；多视图 Room 应使用审查过的多 RTC endpoint 配置。CPU/内存为示例，需实测。执行前核对主机和所有监听端口、私有目录归属、磁盘余量、镜像来源及资源预算；准备目录供镜像用户写入。环境文件优先级为秘密文件在前、生成的非秘密运行参数在后。分离模型池等运行接线仍从原私有环境文件继承；需要的数据配置文件显式用 `--config-mount /private/role-pool-room.json=/etc/whitebox/role-pool-room.json` 挂载，并与环境变量中的路径一致。只允许只读数据配置，禁止覆盖 `/app`、SDK 或发布身份文件。运行中热更新策略也需先读回有效值，再纳入新实例配置；模板不会偷偷复制老实例的全部可写目录。运行 `plan.json` 的 docker argv 仅创建新容器；公网 NAT/GA/NLB 端口必须先接到该实例，不能随机负载均衡到不同 owner。

手动登记与检查（token 仅从环境变量读取）：

```sh
# 只有全新且无活跃资源的控制命名空间，第一次才允许这个初始化选项。
python deploy/whitebox/room_pool.py --coordinator http://coordinator:18081 \
  enroll --room-url http://room-a:8092 --initialize-empty-pool
# 后续扩容不用初始化，不修改旧实例。
python deploy/whitebox/room_pool.py --coordinator http://coordinator:18081 \
  enroll --room-url http://room-b:8092
python deploy/whitebox/room_pool.py --coordinator http://coordinator:18081 inspect
```

禁止将 `--initialize-empty-pool` 当作丢失控制状态后的恢复按钮。人工恢复仍需核对旧 owner/SDK/GPU 的实际归属。没有隐式重试建房或自动重新登记。

## 容量与身份

- 登记信息的 owner/epoch/地址/池兼容身份和容量不可原地替换。同一 owner 的心跳只能续租，不能清掉已有预留。
- 单人 GameRoom 预留一个视图；多人 GameRoom 首版保守预留发布声明的全部 `maxPlayers` 视图容量，避免之后加入时超卖 Room。只给实际玩家申请模型 Slot，不提前抢占未加入玩家的 GPU。
- Room 总视图容量来自现有 `WHITEBOX_ROOM_PIPELINE_CAPACITY`；模型 Worker 容量由原模型 Coordinator 管理。CPU Room 扩容不会提高 GPU Slot 数量，也不修改 main 已有手动预览的模型绑定时机。
- 首次分配原子检查所有就绪 owner 的已占视图/房间与待启动预留，选择相对空闲实例。并发 Coordinator 通过原有 CAS 校验原值提交；Room 启动 CLI 前必须认领原预留。
- 心跳每 5 秒、可分配租约 30 秒。租约过期只禁止新分配，不能把已认领容量算空闲。
- 未认领预留 60 秒后可回收，迟到的认领必须失败。认领成功后只能通过原 owner 的结束路由和完整 SDK/媒体清理确认释放；未知清理继续占用。
- 动态 GameRoom 路由使用 120 秒 TTL，由原 owner 心跳续期；结束回执保留 24 小时。路由丢失时只能凭原预留与原 owner 的实际清理完成写入终态回执，不能恢复活跃世界。
- 注册目录最大 128 个 owner、4096 个游戏预留、4 MiB，整体 24 小时 TTL，真实控制变更/心跳续期。已移除 owner 墓碑最长 24 小时；未确认的活跃占用不因目录里的时间估计而释放。控制状态丢失禁止新分配；活跃游戏的原路由心跳不因注册表缺失而被强制结束。
- 本 main 中原 Room 控制存储仍有 SQLite；本次新增目录均有 TTL，不能把此次扩容测试当作 SQLite 下线验收。

## 缩容与管理边界

```sh
python deploy/whitebox/room_pool.py --coordinator http://coordinator:18081 \
  drain --owner-id room-a --owner-epoch ORIGINAL_EPOCH
# 等待 reserved_rooms/reserved_views 为零，原 Room 报告无玩家、启动/清理任务和归档任务。
python deploy/whitebox/room_pool.py --coordinator http://coordinator:18081 \
  remove --owner-id room-a --owner-epoch ORIGINAL_EPOCH
```

`drain` 只停止新增 GameRoom，已有多人世界仍可按原规则加入成员，不保证两小时内排空。`remove` 再次读取准确原实例的空闲证明，并要求目录没有预留；超时/404/无法确认均拒绝。命令不停止容器，也不释放模型组，后续停止必须针对已确认排空的准确容器。

既有不带 owner 的全机缓存/预热管理接口仍以发布配置中的原主 Room 为作用域，不代表整个动态池。带 owner 的状态/录像、单人/多人控制、SDK Session WS/HTTP 和原试玩 WS 均走动态寻址。Room 被移除后不保证其本地历史查询，归档按原存储策略保留。

## 验证

本地使用独立临时 Redis，覆盖并发分配不超卖、原会话重复创建、原连接在新增 owner 前后持续转发、媒体地址冲突、原 epoch 隔离、未认领迟到、已认领租约过期、注册丢失、排空和确认释放。

本地 WS 测试是实际 HTTP/WebSocket 连接配合受控源，不是 GPU/浏览器性能证明。完整 Room 镜像另运行 opt-in 真实 CLI 测试：A 持续取原生白模帧时登记并启动 B，检查 A 的 CLI PID 和连续帧保持、两局身份独立；结果须以镜像门禁实际输出为准。正式测试环境需要完整镜像、固定 SDK、准确媒体端口、前后容器 ID/StartedAt、浏览器帧与输入、原会话续存、新会话接入和退出收据。无足够模型 Slot 时不抢占其他用户或训练，也不声称已覆盖并发 GPU 试玩。
