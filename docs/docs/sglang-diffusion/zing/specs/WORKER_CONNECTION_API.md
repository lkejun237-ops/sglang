# Worker 通信组接口协议

本文定义 Encoder、DiT、Decoder 每个 service 的健康检查和通信组管理 HTTP 接口。示例地址仅作格式说明，实际地址以查询响应为准。

每个 service 有独立的管理 HTTP 地址，请求哪个地址就操作哪个 service，无需额外指定本地 service index。`--http-port` 为管理基础端口，service 从 `0` 编号，service `i` 的管理端口为 `http-port + i`。数据连接地址通过查询获取，调用方不推算数据端口。

部署背景、业务链路和启动命令见[Zing 分离部署接入指南](../DISAGGREGATED_SERVING_GUIDE.md)。

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
| GET | `/get_connection_status` | 查询连接状态；空闲时取得预留及连接信息 |
| POST | `/build_group` | 建立通信组，返回最终结果 |
| POST | `/destroy_group` | 启动通信组清理，立即返回清理中 |

`worker_group_id` 标识一组 encoder、DiT、decoder，区别于业务请求中的 `session_id`。

| HTTP 状态码 | 含义 |
|---|---|
| `200` | 接口调用成功 |
| `202` | 已受理销毁请求，正在清理 |
| `409` | 预留失效、身份不匹配、占用或状态冲突 |
| `422` | 请求参数不合法 |
| `503` | worker 不可用或连接操作失败 |

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

接口受理后立即启动后台清理，不等待断连完成。

受理响应，HTTP `202`：

```json
{
  "success": true,
  "state": "cleaning",
  "worker_group_id": "group-a"
}
```

group 与当前通信组不匹配时返回 HTTP `409`。同一组正在清理时，重复请求返回相同的 `202` 清理中响应，不重复启动清理；同一组已经清理完成且 service 仍为空闲时，重复请求返回 `200`，状态为 `disconnected`。

客户端 需要分别调用组内三个 service 的接口。清理期间拒绝新任务，`get_connection_status` 返回 `reserved`；清理完成后，下一次查询取得新的 10 秒预留并返回 `available`。如果后台清理失败且需要重启，查询返回 `503`、`unavailable` 和 `restart_required: true`。

上述恢复 `available` 仅适用于正常拆组。运行或通信故障发生后，worker 在尝试清理前即隔离，查询返回 `503`、`unavailable` 和 `restart_required: true`；即使后续本机清理成功，也不据此复用出错的 CUDA/NCCL 上下文。建组或拆组完成不能覆盖并发出现的故障状态。

故障关闭中止本地 bridge，不再等待对端补齐在途数据。独立于 CUDA 工作线程的进程监督器为清理设置绝对截止时间；正常清理完成后解除监督，清理异常或超时则保持监督并终止所属 worker 进程。Encoder 清理预算为 120 秒，DiT/Decoder 使用各自已有的 worker timeout；不依赖 NCCL watchdog 或 Python 线程取消保证退出。若多个 service 共用一个 worker 进程，该进程整体是故障边界；不主动终止其他独立服务进程。底层不支持 bridge abort 时不降级为正常排空，而是要求重启并保留退出兜底。部署层负责后续重建，本接口不自动重放业务请求。

Encoder→DiT bridge 的 abort 与 close 在同一锁下取得关闭权，原生调用在锁外执行。abort 先发布故障状态，再调用底层中止；并发或后续 close 不再启动 graceful shutdown，中止失败也不回退排空。正常 close 已先取得关闭权时，不再并发发起原生 abort，其阻塞仍由清理截止时间兜底。模型线程先调用 transport abort，再通知 I/O 线程退出，避免退出通知提前触发正常关闭。

本协议定义通信组管理接口；业务 WS 消息格式和缓存接口仍以各自既有协议为准。
