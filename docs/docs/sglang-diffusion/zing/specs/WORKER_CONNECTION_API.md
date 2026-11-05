# Worker 通信组接口协议

本文定义 Encoder、DiT、Decoder 每个 service 的健康检查和通信组管理 HTTP 接口。示例地址仅作格式说明，实际地址以查询响应为准。

每个 service 有独立的管理 HTTP 地址，请求哪个地址就操作哪个 service，无需额外指定本地 service index。`--http-port` 为管理基础端口，service 从 `0` 编号，service `i` 的管理端口为 `http-port + i`。数据连接地址通过查询获取，调用方不推算数据端口。

部署背景、业务链路和启动命令见[Zing 分离部署接入指南](../DISAGGREGATED_SERVING_GUIDE.md)。

## 通信组与业务会话的生命周期

一个 `worker_group_id` 配对三个角色的具体 service、外部 ZMQ 控制连接和 NCCL 张量链路；
worker 进程及其常驻模型、DiT 内部 SP 组的生命周期更长。业务 `session_id`／
`generation_id` 标识这组连接上当前请求的输入、输出及生成代次，不代表新建通信组。

同一通信组可以顺序服务不同 session。部署同时只允许一个活跃会话；已有会话时拒绝第二个，
不修改当前 owner。正常 EOS 或取消完成后释放该会话的 KV lease、源流、输出及 Decoder 状态，
保留模型和外部连接，下一合法会话使用新的 session／generation 身份。故障隔离仍可能要求重建，
不能把正常顺序复用理解为模型／NCCL 故障后在线恢复。

`quiesce_group` 关闭新会话准入，等待初始化和在途任务排空、取得取消确认，保持传输连接；
`destroy_group` 才拆除外部通信组。销组不卸载模型、不销毁 DiT 内部 SP 组，也不清空其缓存。
要复用已 quiesce 的服务，须完成销组并重新建组；正常会话结束无需 quiesce／destroy。

## Master 管理调用与数据链路

`WorkerGroupMaster.rpc(role, path, payload=None, *, timeout=None)` 是异步封装的 **HTTP＋JSON
管理调用**，并非 gRPC 或可切换的独立传输协议。`role` 从 `self.addresses` 选取 Encoder、
DiT 或 Decoder 的管理地址，`path` 是该服务的 HTTP 路径；有 `payload` 时 POST JSON，
否则 GET。默认超时为 `control_timeout`（当前实现中 `None`／0 均回退该值）；
`asyncio.to_thread` 把阻塞的 HTTP 操作移出事件循环。当前 Master 实现支持此 HTTP 管理接口，
未实现其他管理协议。

例如 `await master.rpc("encoder", "/quiesce_group", {"worker_group_id": master.group})`
调用的是 Master → Encoder 管理服务；它与客户端 → Encoder 输入 WS、客户端 → Decoder
输出 WS，以及角色间 ZMQ 控制／NCCL 张量链路分开。Master 仅在 Encoder 排空成功后并行拆除
三个角色的外部连接；各角色共同参与 NCCL 退出，不能逐个拆连接。

销组实现分为 `_quiesce_encoder()`、`_destroy_role_until_disconnected(role)` 和入口 `destroy()`：
入口判断是否有本组需要清理的连接，先排空，再并行运行三个角色的幂等销组／轮询，最后汇总错误。
部分建组收到 409 时只用 `reserve=false` 查看状态，并校验 group owner，不能拆除别人的组或
重新取得空闲服务的预留。超时和真实通信故障保留为失败，不自动重试业务生成。

## 目录

- [1. 接口清单与通用返回规则](#1-接口清单与通用返回规则)
- [2. 健康检查](#2-健康检查)
- [3. 查询连接状态及取得预留](#3-查询连接状态及取得预留)
- [4. 建立通信组](#4-建立通信组)
- [5. 销毁通信组](#5-销毁通信组)

## 1. 接口清单与通用返回规则

| 方法 | 路径 | 功能 |
|---|---|---|
| GET | `/liveness` | 存活检查 |
| GET | `/health` | 健康检查 |
| GET | `/get_connection_status` | 查询连接状态；默认空闲时取得预留及连接信息；`?reserve=false` 只查看状态、不取得预留 |
| POST | `/build_group` | 建立通信组，返回最终结果 |
| POST | `/quiesce_group` | Encoder 排空当前会话，保留通信组 |
| POST | `/destroy_group` | 启动通信组清理，立即返回清理中 |

`worker_group_id` 标识一组 encoder、DiT、decoder，区别于业务请求中的 `session_id`。

| HTTP 状态码 | 含义 |
|---|---|
| `200` | 接口调用成功 |
| `202` | 已受理销毁请求，正在清理 |
| `409` | 预留失效、身份不匹配、占用或状态冲突 |
| `422` | 请求参数不合法 |
| `503` | worker 不可用或连接操作失败 |
| `501` | 当前角色不支持所请求的会话排空操作 |

新增接口的错误使用稳定的英文 `code` 和英文 `message`：

```json
{
  "success": false,
  "code": "RESERVATION_INVALID",
  "message": "The reservation is invalid or expired."
}
```

## 2. 健康检查

沿用现有行为。

### 2.1 存活检查

```http
GET /liveness
```

HTTP `200`：

```json
{
  "status": "ok"
}
```

### 2.2 健康检查

```http
GET /health
```

响应：

```json
{
  "role": "decoder",
  "ready": true
}
```

`ready=true` 返回 HTTP `200`，否则返回 `503`。

健康状态不代表 service 是否空闲。客户端 申请使用时需调用 `get_connection_status`。

## 3. 查询连接状态及取得预留

```http
GET /get_connection_status
```

无请求体。

本接口在空闲时会取得预留，响应设置 `Cache-Control: no-store`。客户端 应使用 `/health` 做定期健康检查。

### 3.1 成功取得预留

HTTP `200`：

```json
{
  "state": "available",
  "reservation_token": "<opaque-token>",
  "expires_in_seconds": 10,
  "service": {
    "worker_id": "decoder-worker-1",
    "service_id": "decoder-service-0",
    "role": "decoder",
    "endpoints": {
      "http": "http://10.0.0.13:36000",
      "zmq": "tcp://10.0.0.13:37000",
      "output_ws": "ws://10.0.0.13:38000/v1/realtime_video/output"
    }
  }
}
```

| 字段 | 含义 |
|---|---|
| `state` | `available` 表示本次调用成功取得预留 |
| `reservation_token` | 调用该 service 建连接口时使用的凭证 |
| `expires_in_seconds` | 预留有效时间，当前为 10 秒，由 worker 决定，不可指定 |
| `service` | 完整节点描述，客户端 用于构建拓扑 |
| `worker_id` | worker 身份 |
| `service_id` | service 身份 |
| `role` | `encoder`、`dit` 或 `decoder` |
| `endpoints` | 该 service 对外公布的实际接口地址 |

各角色的 `endpoints` 字段：

| 角色 | 字段 |
|---|---|
| encoder | `http`、`input_ws`、`rendezvous` |
| DiT | `http`、`zmq`、`rendezvous` |
| decoder | `http`、`zmq`、`output_ws` |

`rendezvous` 是 NCCL 初始化使用的 TCPStore 地址。

再次查询不会续期，也不会重复返回 token。调用超时后重试，可能收到 `reserved`，需等待预留到期后重新申请。

### 3.2 已被预留或暂不可申请

HTTP `200`：

```json
{
  "state": "reserved"
}
```

不返回预留 token。

### 3.3 已建立通信组

HTTP `200`，其中 `topology` 为建连请求中确认的完整拓扑：

```json
{
  "state": "connected",
  "worker_group_id": "group-a",
  "topology": {
    "encoder": { "...": "encoder 节点描述" },
    "dit": { "...": "DiT 节点描述" },
    "decoder": { "...": "decoder 节点描述" }
  }
}
```

### 3.4 worker 不可用

HTTP `503`，以下为需要重启的示例：

```json
{
  "state": "unavailable",
  "restart_required": true,
  "code": "WORKER_RESTART_REQUIRED",
  "message": "The worker must be restarted."
}
```

## 4. 建立通信组

```http
POST /build_group
Content-Type: application/json
```

### 4.1 使用查询结果组装请求

假设成功取得预留的三个响应分别为：

```text
encoder_response
dit_response
decoder_response
```

客户端 生成一个 `worker_group_id`，并按下面的对应关系组成拓扑：

```text
topology.encoder = encoder_response.service
topology.dit     = dit_response.service
topology.decoder = decoder_response.service
```

三个请求使用**相同的 `worker_group_id` 和 `topology`，分别使用接收方自己的 token**：

| 请求地址 | 请求中的 `reservation_token` |
|---|---|
| `encoder_response.service.endpoints.http + "/build_group"` | `encoder_response.reservation_token` |
| `dit_response.service.endpoints.http + "/build_group"` | `dit_response.reservation_token` |
| `decoder_response.service.endpoints.http + "/build_group"` | `decoder_response.reservation_token` |

### 4.2 请求格式

发送给 decoder 的示例：

```json
{
  "worker_group_id": "group-a",
  "reservation_token": "<decoder_response.reservation_token>",
  "topology": {
    "encoder": {
      "worker_id": "encoder-worker-1",
      "service_id": "encoder-service-0",
      "role": "encoder",
      "endpoints": {
        "http": "http://10.0.0.11:31000",
        "input_ws": "ws://10.0.0.11:30000/v1/realtime_video/generate",
        "rendezvous": "tcp://10.0.0.11:32000"
      }
    },
    "dit": {
      "worker_id": "dit-worker-1",
      "service_id": "dit-service-0",
      "role": "dit",
      "endpoints": {
        "http": "http://10.0.0.12:33000",
        "zmq": "tcp://10.0.0.12:34000",
        "rendezvous": "tcp://10.0.0.12:35000"
      }
    },
    "decoder": {
      "worker_id": "decoder-worker-1",
      "service_id": "decoder-service-0",
      "role": "decoder",
      "endpoints": {
        "http": "http://10.0.0.13:36000",
        "zmq": "tcp://10.0.0.13:37000",
        "output_ws": "ws://10.0.0.13:38000/v1/realtime_video/output"
      }
    }
  }
}
```

所有节点描述均来自查询响应。示例地址不代表固定端口规则。

**客户端 必须在各 service 的预留期内发送建连请求。** 接口等待本 service 建连完成后返回最终结果。

`connected` 表示该 service 的每条链路已完成本端 CUDA 建连工作，并收到对端对同一次连接的完成确认。
仅创建 communicator、提交 GPU 工作或 `Work.wait()` 返回均不足以发布 `connected`。
确认使用本次 bridge epoch 隔离，旧连接确认不能复用；完成确认超时或失败时走建连失败与回滚路径。
Master 仍须收到三侧成功，才能将整个通信组交给业务客户端。
三角色须配套升级；旧版对端未提供完成确认时，新版按超时失败处理，不降级发布成功。

### 4.3 成功响应

HTTP `200`：

```json
{
  "success": true,
  "state": "connected",
  "worker_group_id": "group-a"
}
```

### 4.4 失败响应

预留失效、身份或状态冲突时，返回 HTTP `409`：

```json
{
  "success": false,
  "code": "RESERVATION_INVALID",
  "message": "The reservation is invalid or expired."
}
```

建连失败且已恢复空闲时，返回 HTTP `503`：

```json
{
  "success": false,
  "state": "idle",
  "worker_group_id": "group-a",
  "restart_required": false,
  "code": "CONNECTION_FAILED",
  "message": "Failed to establish the worker connections."
}
```

失败后原预留 token 失效。如果返回 `restart_required=true`，该 worker 不能继续申请使用。

## 5. 销毁通信组

会话取消与通信组销毁分开处理。输出连接断开或输入 owner 取消时，先回收该会话的
输出邮箱和 RGB 引用；Decoder 继续接收已发送的载荷，直到输入 owner 按顺序取消会话。
正常输出关闭不能触发整个通信组回收，也不能影响下一会话。模型或传输的实际故障
仍按故障处理，不能用丢弃输出掩盖。

Decoder 接收的 `source_stream.width/height` 是原始输入流尺寸，独立于输出 `width/height`；
`source_records` 中的累计字节和 `source.consumed` 均按输入 RGB24 字节计算。
例如 source 832×480、输出 1280×736 时，一帧输入 credit 仍是 `832×480×3`，不能按输出面积计算。
旧的同尺寸调用方省略 `source_stream` 时沿用原 config 的宽高。

独立 Decoder 的像素不经过输入 WS。Encoder 在验证首个正帧数的 Decoder 调度回执后，
启动既有会话硬时限；这是服务端 dispatch 起点，不是客户端接收或播放确认，也不增加
输入 WS 的像素发送统计。仅断开输出、继续发送输入心跳或不发送 playback ACK 都不能
延长该期限；到期仍由输入 owner 沿既有有序取消路径回收会话，保留通信组供下一会话使用。
独立 Encoder 可用 `--session-max-lifetime-s` 明确设置有限上限，省略时沿用标准 realtime
缺省值；`0` 保持管理员显式关闭硬时限的既有语义。长会话基准必须为对照两侧配置相同、
足够长的有限期限，并单独验证短期限下的单边断连回收。

### 5.1 先排空 Encoder 会话

成功建组后，先向 Encoder 的管理地址发送：

```http
POST /quiesce_group
Content-Type: application/json
```

请求为 `{"worker_group_id":"group-a"}`。服务核对当前 group 后停止接收新会话和输入，
排空已接收的工作，并等待 DiT 的取消确认；确认前 Decoder 和全部通信通道保持可用。
请求方断连不会取消已启动的排空。重复请求共用同一次操作。

尚在异步输入校验或模型 owner 队列中的初始化也属于排空范围。初始化在首次异步校验前
登记，登记与 connected 检查和 quiesce 状态切换同步；排空等待该初始化完成或被拒绝。
`link.open` 执行前在 owner 队列内再次检查准入。尚未执行的初始化遇到排空时返回
`CONDITION_BUSY`，公开消息为 `The service is busy. Please try again shortly.`；
这是普通请求拒绝，不标记模型／通信故障。已开始的初始化由同一 owner 队列在完成后
执行取消，取消确认前不返回 `quiesced`，因此成功排空后不能新建会话 owner。

排空完成返回 HTTP `200`：

```json
{"success":true,"state":"quiesced","worker_group_id":"group-a"}
```

此时 group 仍归原 owner，查询返回 `reserved`，不会发放新预留；下一步销毁通信组后才能重新申请。
不匹配的 group 返回 `409`，未提供排空能力的角色返回 `501 / QUIESCE_UNSUPPORTED`。
排空失败返回 `503 / QUIESCE_FAILED`，消息为 `Failed to finish the current session.`，
worker 标记为需要重启，内部异常仅保留在服务端日志。

### 5.2 再并行销毁三个角色的通信组

收到 Encoder 的 `quiesced` 后，并行向三个 service 发送：

```http
POST /destroy_group
Content-Type: application/json
```

请求：

```json
{
  "worker_group_id": "group-a"
}
```

正常受理后立即启动后台清理，不等待断连完成。若同一 group 正在排空，销毁请求等待排空结束后再受理。
NCCL 通信组退出需要两端共同参与，三个角色的销毁请求必须并行发出。

受理响应，HTTP `202`：

```json
{
  "success": true,
  "state": "cleaning",
  "worker_group_id": "group-a"
}
```

group 与当前通信组不匹配时返回 HTTP `409`。同一组正在清理时，重复请求返回相同的 `202` 清理中响应，不重复启动清理；同一组已经清理完成且 service 仍为空闲时，重复请求返回 `200`，状态为 `disconnected`。

客户端通过重复 `destroy_group` 轮询到 `disconnected`，不会取得新预留。清理期间拒绝新任务，
`get_connection_status` 返回 `reserved`；清理完成后，下一次查询会取得新的 10 秒预留并返回 `available`。
部分建组尚未受理而返回 `409` 时，Master 仅以 `get_connection_status?reserve=false`
确认是否空闲；过期预留可以释放，但不会新取预留，已有有效预留或其他 owner 不受影响。
如果后台清理失败且需要重启，查询返回 `503`、`unavailable` 和 `restart_required: true`。

上述恢复 `available` 仅适用于正常拆组。运行或通信故障发生后，worker 在尝试清理前即隔离，查询返回 `503`、`unavailable` 和 `restart_required: true`；即使后续本机清理成功，也不据此复用出错的 CUDA/NCCL 上下文。建组或拆组完成不能覆盖并发出现的故障状态。

故障关闭中止本地 bridge，不再等待对端补齐在途数据。独立于 CUDA 工作线程的进程监督器为清理设置绝对截止时间；正常清理完成后解除监督，清理异常或超时则保持监督并终止所属 worker 进程。Encoder 清理预算为 120 秒，DiT/Decoder 使用各自已有的 worker timeout；不依赖 NCCL watchdog 或 Python 线程取消保证退出。若多个 service 共用一个 worker 进程，该进程整体是故障边界；不主动终止其他独立服务进程。底层不支持 bridge abort 时不降级为正常排空，而是要求重启并保留退出兜底。部署层负责后续重建，本接口不自动重放业务请求。

Encoder→DiT bridge 的 abort 与 close 在同一锁下取得关闭权，原生调用在锁外执行。abort 先发布故障状态，再调用底层中止；并发或后续 close 不再启动 graceful shutdown，中止失败也不回退排空。正常 close 已先取得关闭权时，不再并发发起原生 abort，其阻塞仍由清理截止时间兜底。模型线程先调用 transport abort，再通知 I/O 线程退出，避免退出通知提前触发正常关闭。

本协议定义通信组管理接口；业务 WS 消息格式和缓存接口仍以各自既有协议为准。
