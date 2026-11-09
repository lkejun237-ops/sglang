---
title: "Block0 缓存管理 API 与部署契约"
description: "默认 local 与可选 Mooncake 后端的缓存管理行为对比、preload/unload/status、L2 管理接口及部署契约。"
---

# Block0 缓存管理 API 与部署契约

2026-11-02 分页变更：管理响应删除 `capacity`，统一报告页数与字节容量；中台需同步修改。
页预算取代固定条目上限，接口字段见下文。设计与验证边界见 [分页方案](/docs/sglang-diffusion/zing/plans/BLOCK0_PAGED_L1)。

本文定义 Zing H3 受管 Block0 前缀缓存的外部接口契约（原版 local 沿用 PR #467 的 schema `zing-block0-v3`、`disk_format=4`；Mooncake 使用独立的内存对象协议，不读取这些磁盘格式）、HTTP 状态码、完整返回值结构、结构化日志与部署边界。
设计背景与验证记录见[确认方案](/docs/sglang-diffusion/zing/plans/BLOCK0_CACHE_MANAGEMENT_SHARED_L2)，串行预热协议见[prefill WebSocket 契约](/docs/sglang-diffusion/zing/specs/block0-prefill-websocket)。

---

## 1. 架构与读写行为总览

### 1.1 后端选择与 API 行为对比

**Mooncake 路径暂时默认关闭，业务适配与启用尚未确认。** 默认后端为 `local`，保留原版实例私有 pinned L1、磁盘 L2 和文件业务索引。Mooncake 是可选后端，需将以下参数追加到服务启动命令中显式选择；HTTP 请求不切换后端：

```bash
--zing-block0-cache-mode on \
--zing-block0-cache-backend mooncake \
--zing-block0-mooncake-config /path/client.json
```

Mooncake 由独立 Store 持有共享 pinned 内存，同模型、SP 布局和 namespace 的兼容实例可复用缓存。**当前 Mooncake 后端仅使用共享内存 L1，不启用 L2，也不读取、写入或清理原版 L2 文件。** Master/Store 与客户端配置见[部署说明](/docs/sglang-diffusion/zing/deployment_zing_h3)。

| 接口或行为 | 原版 `local`（默认） | `mooncake`（可选） |
| --- | --- | --- |
| 正常视频会话，`cache_write=false`（默认） | 只查本实例 L1；未命中重算，不写回 L1/L2 | 只查共享 L1；未命中重算，不写回共享缓存 |
| 正常视频会话，`cache_write=true` | 重算产生新前缀后写入本实例 L1，后台异步落盘 L2；携带业务标签时登记关联 | 重算产生新前缀后尝试写入共享 L1；携带业务标签时登记共享关联，无落盘阶段 |
| `WS /v1/realtime_video/prefill`，`force=false` | 依次尝试本实例 L1、L2；均未命中则重算，填充 L1 并异步落盘 L2 | 尝试共享 L1；未命中则重算并发布共享条目，所有 SP 分片成功后才发布 manifest |
| `prefill`，`force=true` | 跳过命中，重算并替换本实例 L1，后台覆盖 L2 | 跳过命中并重算；同 key 内容一致，已存在的不可变对象可复用，不引入内容版本覆盖语义 |
| `POST /v1/zing/cache/preload` | 复用本实例 L1，或从 L2 加载到本实例 L1；缺失返回 `missing`，不重算 | 检查共享条目完整性；存在返回 `ready`，缺失返回 `missing`。不从盘加载、不重算，不提供常驻保证 |
| `POST /v1/zing/cache/unload` | 仅卸载本实例 L1，等待该条目写盘与 DMA 完成；保留 L2 和业务索引 | **删除共享条目的 manifest、conditions 和所有 SP rank 的 KV 分片，影响共享它的所有实例**；保留业务关联键，读租约未结束或无法安全删除时可返回 `409 cache_busy`，不强删 |
| `GET /v1/zing/cache/status` | 查询本实例 L1 的驻留状态及页池容量 | 查询当前兼容 namespace 中共享条目的完整可用状态；返回 `backend="mooncake"`、`scope="shared"`，`budget_bytes` / `reserved_bytes` 为 `null`，不返回页池字段 |
| `GET /v1/zing/cache` | 列出 L2 磁盘条目及容量快照 | `HTTP 501`，`cache_operation_unsupported`；不改作共享内存列表接口 |
| `DELETE /v1/zing/cache/entries/{cache_key}` | 仅删除该 key 的 L2 文件，保留各实例已有 L1 和业务索引 | `HTTP 501`，`cache_operation_unsupported`；删除共享条目使用 `unload` |
| `POST /v1/zing/cache/clear` | 仅清空配置目录中的 L2 条目，保留各实例已有 L1 和业务索引 | `HTTP 501`，`cache_operation_unsupported`；当前没有清空整个共享池的 HTTP 接口 |
| Worker 重启 | 本实例 L1 消失；已成功落盘的数据可通过 preload/prefill 从 L2 恢复 | Store 数据与 Master 元数据仍在时可复用；本期没有 L2 恢复路径 |
| 满池与淘汰 | 按实际行数占页，整条 LRU 淘汰；preload 总页需求超限可返回 `cache_capacity_exceeded` | 使用 Mooncake 原生 group、读租约和水位策略，可能批量超量淘汰或写入失败；分组是 best-effort，不保证精确保留条目数 |

两种后端都保留相同的业务选择器，以及“有实时会话时拒绝 preload/unload、允许 status”的规则。Mooncake 的 `ready` 表示检查时完整可用；status 会通过 SDK 读取并可能刷新读租约，随后 unload 仍可能暂时返回 `busy`。普通生成中的缓存写入失败不使生成失败，显式 prefill 写入失败会报告失败。

原版没有同时删除 L1 和 L2 的单一接口。需要两层都删除时，先对持有该 key 的实例完成 `unload`，再调用 L2 `DELETE`；这不是跨实例原子失效操作。Mooncake 的 `unload` 虽沿用旧名称，当前承担的是共享条目的删除作用；它不会清理迁移前留下的 L2 文件。

### 业务接入待确认项

本 PR 提供可显式启用的后端，不包含中台、持久化流程或部署平台的自动适配。以下差异需由业务方确认后再启用 Mooncake，不能把旧接口调用成功当作兼容验收：

- **L2 与持久化**：不落盘、无 SSD offload、无持久化写回或重启后磁盘恢复；`cache-dir`、host/disk budget 和 L2 GC 参数不用于 Mooncake。旧 L2 文件不迁移、不清理。若业务仍需离线预热结果长期保存，继续使用 `local`，另行确认迁移方案。
- **管理操作**：L2 list/delete/clear 明确返回 `501 cache_operation_unsupported`。不自动改写为共享内存清空或删除。`preload` 缺失时仍返回 missing；需由业务决定是否通过 prefill 重算补热。
- **共享删除范围**：Mooncake 的 unload 影响该 namespace 内所有共享实例；需确认业务权限、调度和重试策略，不能沿用“仅释放本实例内存”的假设。普通读租约未结束时可返回 busy，服务不强删。
- **就绪与恢复**：Worker 重启可在 Store/Master 仍持有数据与元数据时复用；Store/Master 丢失或条目淘汰后没有磁盘恢复路径。业务索引也是易失缓存，关联登记采用 best-effort，失败会记录 index_bind failed；prefill.completed 不承诺索引长期存在。调用方应保留预期 key/场景清单，并处理 missing、空选择和重新预热。
- **响应字段**：Mooncake 不提供页池字段，budget_bytes/reserved_bytes 为 null；status 的 used_bytes 仅是本次选中且完整的 KV 字节数，不含 conditions/索引，也不是 Store 全局占用。prefill 的 entries 是本客户端已知 key 数的 CPU 本地提示，可能包含已被原生淘汰的 key，不是共享池总条目数或就绪数；evictions 不包含原生 Store 淘汰，不能据其为 0 判断没有淘汰。

共同的 WebSocket 请求字段、业务标签校验和实时会话 busy 规则沿用；不新增 pin/unpin、常驻承诺或客户端兼容兜底。业务确认后再单独提交适配与开启配置。

**下文的页池容量、分页响应示例和磁盘部署契约适用于 `local`。Mooncake 的接口差异以本节为准；通用请求字段和业务标签规则继续适用。**

### 1.2 原版 local 的双层读写流程

Block0 缓存采用 **“实例独占 L1（Host Pinned Memory） + 同机共享 L2（本地 NVMe 磁盘） + 业务标签索引（`game_id` / `group_id`）”** 的双层受管架构：

- **L1（实例内存层）**：每个推理实例（SP 组）独立持有预分配的 Pinned CPU 内存页。命中时通过异步 DMA 直接恢复 50 层 Block0 KV，跳过 `codecs.prepare` 与首块前缀计算。
- **L2（同机磁盘层）**：同一台物理机上的多个实例可共享同一个本地 NVMe 目录（`--zing-block0-cache-dir`），存储 `entries/<cache_key>/` 实体与 `index/<game_id>/<group_id>/` 业务关联索引。
- **在线生成与磁盘解耦**：正常视频会话（初始请求与后续 prompt 切换，含 `may_contain` 预取提示）**严格只查本实例 L1，永不阻塞读盘**。

| 调用入口 | L1 行为 | L2（本地磁盘）行为 | 业务索引（`game_id` / `group_id`） |
| --- | --- | --- | --- |
| **正常视频会话**<br/>`cache_write=false`（**默认**） | **只读本实例 L1**；L1 未命中直接走 GPU 重算 | **不读盘、不写盘**（即使 L2 有该条目也不读） | 不写入索引 |
| **正常视频会话**<br/>`cache_write=true` | 只读本实例 L1；L1 未命中走 GPU 重算，并在新前缀首次输出后写回 L1 | 重算产生新条目后，后台异步落盘 L2 | 若携带 `game_id`（及可选 `group_id`），写回时登记关联 |
| **预热接口 (`prefill`)**<br/>`force=false`（**默认**） | 优先复用 L1；L1 未命中时查 L2 并恢复至 L1；L1+L2 均未命中则重算并写回 L1 | L1+L2 均未命中时，后台异步落盘 L2 | 无论命中或重算，只要携带标签即登记关联 |
| **预热接口 (`prefill`)**<br/>`force=true` | **跳过 L1/L2 命中**，强制重算并覆盖本实例 L1（不影响其他实例已有 L1） | 后台异步覆盖 L2 对应 key | 携带标签时登记关联 |
| **显式预加载 (`preload`)** | 已在 L1 则复用；不在 L1 则同步从 L2 加载校验并填入本实例 L1（满时按 LRU 淘汰） | 只读 L2；若 L2 缺失或损坏，逐项返回 `missing`，**不触发 GPU 重算** | 只读索引解析目标 `cache_key` 集合 |
| **显式卸载 (`unload`)** | 从本实例 L1 移除选中条目并归还空闲页（等待该条目写盘与 DMA 完成） | 保留 L2 磁盘实体 | 保留业务索引 |
| **磁盘清理 (`DELETE` / `clear`)** | 保留各实例已有 L1（不跨实例广播失效） | 立即删除目标 L2 磁盘目录及分片文件 | 保留索引（后续对被删 key 执行 `preload` 将返回 `missing`） |

---

## 2. 容量规格与淘汰边界

以下是与历史合成 IO 验收一致的容量示例：**SP=2、50 层 BF16 KV、每 rank 28 个 head、head_dim=128、12,288 prefix rows**。实际页容量取决于模型、SP 布局、页行数与 host 预算，应以实例启动时的页数与字节日志为准；单条前缀须满足共享 GPU token 容量，L1 还受 host 页预算限制：

- **单分片（每张卡 / 每 rank）**：`50 × 2 × 12288 × 28 × 128 × 2 Bytes` = **8.81 GB**（二进制 **8.203 GiB**）
- **单完整条目（SP=2 双卡合计）**：**17.62 GB**（二进制 **16.406 GiB**）

| 层级 | 配置参数 | 默认值与示例配置 | 按上述布局估算的完整条目数 | 满容后的淘汰与边界行为 |
| --- | --- | --- | --- | --- |
| **L1 内存缓存** | `--zing-block0-cache-host-gb` | **无，启用缓存时必填**；示例 `800`（整实例十进制 GB，SP=2 下每 rank 400 GB） | **每 rank 272 页（2048 行/页）** | **按实际行数占页，整条 LRU 淘汰**：页不足时逐条淘汰至足够；单次 `preload` 的有效 key 总页需求超过实例总页数时，在加载前返回 `HTTP 409 / cache_capacity_exceeded`。L2 缺失的 key 不计需求页；已在全部 SP rank 的 L1 hit 即使 L2 已删仍可复用。 |
| **L2 磁盘缓存** | `--zing-block0-cache-disk-gb` | **无，启用缓存时必填**；示例 `4000`（十进制 GB，清理高水位） | 高水位 **≈ 227 条**<br/>低水位(`0.9`) **≈ 204 条** | **周期性后台 GC**：开启 `--zing-block0-cache-gc` 的实例每 60 秒扫描一次元数据：先清理超时未完成的临时文件（默认 `600s`），若总占用超过高水位，按 `.access` 的近似 LRU 顺序删除磁盘条目直至降至低水位（默认 `90%`）。 |

---

## 3. 业务标签与 HTTP 状态码规范

### 3.1 业务标签校验规则

- `game_id` / `group_id`：类型必须为非空 `str`，UTF-8 字节长度 **≤ 128**。服务端按原值保存和匹配，不去除首尾空白，也不额外拒绝控制字符；调用方应保持 ID 一致。
- **层级约束**：传入 `group_id` 时**必须同时提供** `game_id`；登记关联时，仅传 `game_id` 表示归入该游戏的“未分组（ungrouped）”集合；查询、预加载和卸载时省略 `group_id` 则选择该游戏的所有分组及未分组条目。
- `cache_key`：固定为 **64 位小写十六进制字符串**（`^[0-9a-f]{64}$`），由服务端根据 schema、模型配置（排除 `checkpoint` 路径）、有效输入（prompt、参考图内容与条件参数、宽高、参考图 resize 上限、S0 像素）、SP 拓扑和 KV dtype 归一化计算 SHA-256 生成。`game_id`、`group_id`、`event_id`、`request_id`、`session_id`、`force`、`cache_write` 均在计算前剥离，不改变 `cache_key`。

### 3.2 HTTP 状态码与错误结构体约定

所有 `/v1/zing/cache/*` HTTP 接口遵循统一的工程状态码契约：

- **为什么查不到游戏或磁盘缺文件时返回 `HTTP 200` 而不是 `404`？**
  1. `preload` / `unload` / `status` 是**按业务标签集合选择器执行的批处理操作**（一个 `game_id` 下往往关联多个 `cache_key`，可能 10 个里有 8 个 `ready`、2 个 `missing`）。HTTP 状态码无法同时表达部分成功与部分缺失，因此请求被服务端成功执行时统一返回 `HTTP 200 OK`，由响应体顶层的 **`"all_ready": true/false`** 和逐项 **`"status": "ready" | "missing" | "unloaded"`** 表达业务就绪态。
  2. `local` 后端的 `DELETE /v1/zing/cache/entries/{cache_key}` 遵循幂等清理语义，均返回 `HTTP 200 OK`，通过响应体区分 `"deleted": [...]` 与 `"missing": [...]`。Mooncake 后端不支持这些 L2 管理操作，返回 `HTTP 501`、`{"detail": {"code": "cache_operation_unsupported"}}`。
- **HTTP 非 200 状态码用于整次请求被拒绝或服务异常**。响应体有两种结构，调用方需要分别处理：
  - 业务错误：`detail` 为对象，通过 `detail.code` 判断错误码。
  - FastAPI/Pydantic 参数校验错误：`detail` 为列表，包含 `loc`、`type`、`msg` 等字段，没有统一的 `detail.code`。例如缺少 `game_id`、请求体含额外字段，或 Query 中的 key/分页参数不合法。

业务错误示例：

```json
{"detail":{"code":"cache_busy"}}
```

参数校验错误示例（`POST /v1/zing/cache/preload` 请求体为 `{}`）：

```json
{"detail":[{"type":"missing","loc":["body","game_id"],"msg":"Field required","input":{}}]}
```

下表中 `detail.code` 只适用于业务错误；框架校验错误按 `detail[]` 解析：

| HTTP 状态码 | 错误码 (`detail.code`) | 触发条件 | 调用方处理建议 |
| --- | --- | --- | --- |
| **`200 OK`** | 无（正常返回业务 JSON） | 请求执行完成（**注：需检查 JSON 内的 `all_ready`、`results[].status`、`failed` 判断条目级结果**） | 解析返回字段，按需路由或补做 `prefill` |
| **`409 Conflict`** | `cache_busy` | 目标实例正在执行视频生成、`prefill` 或其他 `preload`/`unload` 操作 | 稍后重试，或调度到其他空闲实例 |
| **`409 Conflict`** | `cache_capacity_exceeded` | 单次 `preload` 有效 key 的总页需求超过 SP 共同页容量（返回 `page_rows`、`total_pages`、`requested_pages`） | 拆分 `group_id` 控制单组大小，或调大 `--zing-block0-cache-host-gb` |
| **`409 Conflict`** | `cache_requires_dp1` | 当前服务实例 `dp_size != 1`，不支持实例级 L1 定向管理 | 部署确保单实例 `dp_size=1` |
| **`422 Unprocessable Entity`** | `invalid_cache_selection` | 标签格式非法、传了 `group_id` 但未传 `game_id`、或指定的 `cache_key` 未绑定在该 `game_id`/`group_id` 下 | 修正请求参数，或先通过 `prefill` 绑定标签关联 |
| **`422 Unprocessable Entity`** | DELETE 路径参数为 `invalid_cache_key`；框架校验为 `detail[]` | `cache_key` 不是 64 位小写 hex，`offset < 0`，`limit` 不在 `1..1000`，或传入了未定义的额外字段 | 修正请求参数格式 |
| **`503 Service Unavailable`** | `cache_unavailable` | L1 管理时 Worker 没有缓存 owner（如模式为 `off` 或没有 GPU KV pool）、磁盘管理时未配置目录，或操作遇到磁盘/索引/RPC 错误 | 检查实例启动参数与本地 NVMe 挂载状态 |

---

## 4. 接口详细参考（入参、状态码与返回值）

### 4.1 串行预热 WebSocket：`WS /v1/realtime_video/prefill`

用于在流量接入前计算固定 case 的 Block0 KV，填充本实例 L1、异步落盘 L2，并登记 `game_id` / `group_id` 关联。同一连接内必须**串行发送**（收到上一条回执后再发下一条）。完整连接生命周期见[prefill WebSocket 契约](/docs/sglang-diffusion/zing/specs/block0-prefill-websocket)。

#### 建连握手回执（Server → Client，MessagePack）

```json
{
  "type": "prefill.ready",
  "max_message_bytes": 67112960
}
```

#### 请求帧（Client → Server，MessagePack）

在常规 `init` 请求字段基础上将 `type` 设为 `"prefill"`，并包含以下控制字段：

| 字段名 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `type` | `str` | 是 | `"prefill"` | 固定为 `"prefill"` |
| `request_id` | `str` | 是 | 无 | `1..128` 字符的请求标识，回执原样带回（不参与 `cache_key` 计算） |
| `source_frame` | `bytes` | 是 | 无 | 恰好 1 帧行优先 `RGB24` 原始字节，长度 `width × height × 3` |
| `force` | `bool` | 否 | `false` | 严格布尔值；`true` 时跳过 L1/L2 命中强制重算并覆盖 |
| `game_id` | `str` | 否 | `null` | 登记该 `cache_key` 归属的游戏 ID |
| `group_id` | `str` | 否 | `null` | 登记该 `cache_key` 归属的游戏内分组（如关卡/阶段），须同时传 `game_id` |

#### 成功回执帧（Server → Client，MessagePack）

```json
{
  "type": "prefill.completed",
  "request_id": "wasteland-480-a",
  "session_id": "8a3f0c1e9b2d4e5f8a3f0c1e9b2d4e5f",
  "cache_key": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "hit": false,
  "entries": 1,
  "evictions": 0
}
```

| 返回字段 | 类型 | 说明 |
| --- | --- | --- |
| `type` | `str` | 成功固定为 `"prefill.completed"` |
| `request_id` | `str` | 对应请求的 `request_id` |
| `session_id` | `str` | 当前 WebSocket 绑定的内部会话 ID |
| `cache_key` | `str` | 64 位小写 hex 缓存键（中台可保存用于后续校验或单条管理） |
| `hit` | `bool` | `true` 表示复用了已有 L1/L2 缓存；`false` 表示本次执行了 GPU 重算 |
| `entries` | `int` | 操作完成后本实例 L1 当前驻留的条目总数 |
| `evictions` | `int` | 本实例自启动以来发生的 L1 LRU 累计淘汰次数 |

> **完成边界说明**：收到 `prefill.completed` 时，本实例所有 SP rank 的 **L1 已就绪**且会话租约已释放，但**后台异步写 L2 磁盘可能仍在进行中**。`l2_write completed` 仅表示日志中该 rank 的写入完成，默认 INFO 只显示 rank 0；`GET /v1/zing/cache` 也只是可能过期的 GC 快照，两者均不作为跨实例就绪屏障。跨实例使用时，对目标空闲实例执行 `preload`，以返回的 `all_ready=true` 确认其 L1 就绪；`missing` 时结合写盘失败日志判断并重试或补热。

#### 失败回执帧（Server → Client，MessagePack）

```json
{
  "type": "prefill.failed",
  "request_id": "wasteland-480-a",
  "code": "prefill_busy",
  "content": "The service is busy. Please try again.",
  "fatal": false
}
```

| 返回字段 | 类型 | 说明 |
| --- | --- | --- |
| `type` | `str` | 失败固定为 `"prefill.failed"` |
| `request_id` | `str \| null` | 对应请求的 `request_id`（若请求体解码前即失败则为 `null`） |
| `code` | `str` | 错误码：`prefill_invalid`（参数/格式错误）、`prefill_busy`（实例正忙或同连接并发提交）、`prefill_too_large`（超出消息字节上限）、`prefill_failed`（内部执行失败）或 `CONDITION_*`（条件编码异常） |
| `content` | `str` | 英文错误描述 |
| `fatal` | `bool` | `false` 时连接保持打开可继续发送下一条；`true` 时服务端将以 WebSocket `1011` 关闭连接 |

---

### 4.2 正常实时视频会话扩展字段

在现有的实时视频会话初始化请求（`type="init"`）中新增以下可选字段，其余视频流事件保持不变：

| 字段名 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `cache_write` | `bool` | 否 | `false` | 严格布尔值。默认 `false`（在线生成只查 L1，未命中重算后**不写回** L1/L2，避免在线流量冲刷预热缓存）；设为 `true` 时允许把会话初始及后续 prompt 切换产生的新前缀写入 L1 并异步落盘 L2 |
| `game_id` | `str` | 否 | `null` | 仅当 `cache_write=true` 且实际产生了缓存写入时，登记游戏关联 |
| `group_id` | `str` | 否 | `null` | 仅当 `cache_write=true` 时生效，须同时提供 `game_id` |

---

### 4.3 显式预加载到实例 L1：`POST /v1/zing/cache/preload`

`local` 后端将指定游戏/分组/单键的缓存从共享 L2 磁盘同步加载到**当前目标实例的 L1 内存**中。不触发模型重算。Mooncake 后端仅检查共享条目完整性，行为与响应差异见 1.1 节。

#### 请求体（JSON）

```json
{
  "game_id": "game-a",
  "group_id": "stage-1",
  "cache_key": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
}
```

| 字段名 | 类型 | 必填 | 选择器语义 |
| --- | --- | --- | --- |
| `game_id` | `str` | **是** | 仅传 `game_id` 时，选中该游戏下所有分组及未分组的全部关联 `cache_key`（自动去重并排序） |
| `group_id` | `str` | 否 | 传 `game_id` + `group_id` 时，仅选中该游戏下指定分组关联的 `cache_key` |
| `cache_key` | `str` | 否 | 额外指定单个 64 位 hex `cache_key`，服务端会校验该 key 必须已绑定在所选 `game_id`（及 `group_id`）下，否则返回 `422 invalid_cache_selection` |

#### 成功响应（`HTTP 200 OK`）

```json
{
  "page_rows": 2048,
  "total_pages": 272,
  "free_pages": 254,
  "budget_bytes": 800000000000,
  "reserved_bytes": 798595481600,
  "used_bytes": 52848230400,
  "entries": 3,
  "cache_keys": [
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    "1111111111111111111111111111111111111111111111111111111111111111",
    "2222222222222222222222222222222222222222222222222222222222222222"
  ],
  "selected": [
    {
      "cache_key": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "ready": true
    }
  ],
  "all_ready": true,
  "busy": false,
  "results": [
    {
      "cache_key": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "status": "ready"
    }
  ]
}
```

#### 返回字段字典

| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| `page_rows` | `int` | 每页行数，当前为 2048 |
| `total_pages` | `int` | SP 各 rank 总页数的最小值，共同保证的逻辑页容量 |
| `free_pages` | `int` | SP 各 rank 空闲页数的最小值 |
| `budget_bytes` | `int` | 整实例 KV 数据预算，跨 SP rank 求和 |
| `reserved_bytes` | `int` | 实际预分配 pinned KV 字节，跨 rank 求和；预算余数不分配 |
| `used_bytes` | `int` | 已分配给驻留条目和 pending fill 的页字节，跨 rank 求和；不含 conditioning/元数据 |
| `entries` | `int` | 操作完成后，本实例所有 SP rank 均驻留的 L1 条目总数 |
| `cache_keys` | `list[str]` | 本实例当前 L1 全部已就绪的 `cache_key` 升序列表 |
| `selected` | `list[object]` | 本次选择器命中的目标 `cache_key` 在操作完成后的最终 L1 驻留状态：`{"cache_key": "...", "ready": true/false}` |
| `all_ready` | `bool` | **核心判定字段**：仅当 `selected` 非空且其中**每一项的 `ready` 均为 `true`** 时为 `true`。若传入不存在的 `game_id`（空选择）或任一选中条目最终未在所有 SP rank 的 L1 驻留，则为 `false`；已有 L1 条目即使 L2 已被删，也可以为 `true` |
| `busy` | `bool` | 执行时实例是否处于忙碌态（`preload` 成功返回时恒为 `false`） |
| `results` | `list[object]` | 本次逐条加载的执行结果：`status` 为 `"ready"`（已在 L1 或从 L2 加载校验成功）或 `"missing"`（L2 磁盘缺失、未写完或校验不兼容） |

#### 错误响应（HTTP 非 200）

- `409 Conflict`：
  - 实例忙：`{"detail": {"code": "cache_busy"}}`
  - 请求有效 key 的总页需求超 L1 总页容量：`{"detail": {"code": "cache_capacity_exceeded", "page_rows": 2048, "total_pages": 272, "requested_pages": 300}}`
  - DP 配置不支持：`{"detail": {"code": "cache_requires_dp1"}}`
- `422 Unprocessable Entity`：`{"detail": {"code": "invalid_cache_selection"}}`
- `503 Service Unavailable`：`{"detail": {"code": "cache_unavailable"}}`

---

### 4.4 显式从实例 L1 卸载：`POST /v1/zing/cache/unload`

`local` 后端将指定游戏/分组/单键从**当前实例 L1** 中移除并归还页面，**不删除 L2 磁盘文件和业务索引**。Mooncake 后端会删除共享条目、影响所有共享实例，行为与响应差异见 1.1 节。

#### 请求体（JSON）

与 `preload` 完全一致（`game_id` 必填，`group_id`、`cache_key` 可选）。

#### 成功响应（`HTTP 200 OK`）

```json
{
  "page_rows": 2048,
  "total_pages": 272,
  "free_pages": 260,
  "budget_bytes": 800000000000,
  "reserved_bytes": 798595481600,
  "used_bytes": 35232153600,
  "entries": 2,
  "cache_keys": [
    "1111111111111111111111111111111111111111111111111111111111111111",
    "2222222222222222222222222222222222222222222222222222222222222222"
  ],
  "selected": [
    {
      "cache_key": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "ready": false
    }
  ],
  "all_ready": false,
  "busy": false,
  "results": [
    {
      "cache_key": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "status": "unloaded"
    }
  ]
}
```

- `results[].status`：原先在 L1 中并被成功移除为 `"unloaded"`；原先就不在 L1 中为 `"missing"`。
- 卸载完成后，`selected[].ready` 变为 `false`，`all_ready` 变为 `false`。
- 错误响应为 `409 cache_busy/cache_requires_dp1`、`422 invalid_cache_selection` 或 `503 cache_unavailable`；卸载不检查选择数量是否超过 L1 容量，不返回 `cache_capacity_exceeded`。

---

### 4.5 查询实例 L1 驻留状态：`GET /v1/zing/cache/status`

`local` 后端查询当前目标实例的 L1 真实驻留状态，Mooncake 后端查询共享条目的完整可用状态（响应差异见 1.1 节）。**不抢占活跃会话，实例在生成视频期间（`busy=true`）也可正常返回 `HTTP 200`**。

#### Query 参数（全部可选）

| 参数名 | 类型 | 必填 | 选择器组合规则 |
| --- | --- | --- | --- |
| `game_id` | `str` | 否 | 传 `game_id`（可搭配 `group_id` / `cache_key`）按业务索引检查指定集合的 L1 就绪状态 |
| `group_id` | `str` | 否 | 必须搭配 `game_id` 使用；单独传 `group_id` 返回 `422 invalid_cache_selection` |
| `cache_key` | `str` | 否 | 可单独传入（不传 `game_id`）直接检查某一个 `cache_key` 是否在当前实例 L1 中；也可搭配 `game_id` 使用 |

*(注：若不传任何参数，`selected` 默认等于当前已在 L1 中的 `cache_keys` 列表)*

#### 成功响应（`HTTP 200 OK`）

```json
{
  "page_rows": 2048,
  "total_pages": 272,
  "free_pages": 260,
  "budget_bytes": 800000000000,
  "reserved_bytes": 798595481600,
  "used_bytes": 35232153600,
  "entries": 2,
  "cache_keys": [
    "1111111111111111111111111111111111111111111111111111111111111111",
    "2222222222222222222222222222222222222222222222222222222222222222"
  ],
  "selected": [
    {
      "cache_key": "1111111111111111111111111111111111111111111111111111111111111111",
      "ready": true
    },
    {
      "cache_key": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "ready": false
    }
  ],
  "all_ready": false,
  "busy": false
}
```

- 字段含义与 `preload` 响应一致（无 `results` 字段）。`busy` 反映 Worker 是否持有实时会话；维护操作不创建实时会话，不能将该字段当作实时维护锁或空闲预留。以实际 `preload` / `unload` 的准入结果为准。
- 错误响应：`409 cache_requires_dp1`、`422 invalid_cache_selection`、`503 cache_unavailable`。

---

### 4.6 列出 L2 磁盘条目与容量快照：`GET /v1/zing/cache`

读取共享 L2 磁盘目录最近一次 GC 采样生成的容量与条目快照（`cache_usage.json`），不在每次 HTTP 请求时全盘遍历文件。

#### Query 参数

| 参数名 | 类型 | 默认值 | 校验范围 | 说明 |
| --- | --- | --- | --- | --- |
| `cache_key` | `str` | `null` | `^[0-9a-f]{64}$` | 可选，按单个 `cache_key` 过滤 |
| `offset` | `int` | `0` | `>= 0` | 分页起始偏移 |
| `limit` | `int` | `100` | `1..1000` | 单页最大返回条数 |

#### 成功响应（`HTTP 200 OK`）

```json
{
  "max_bytes": 4000000000000,
  "root_entries": 12,
  "root_bytes": 211459250000,
  "sampled_at": 1761984000.125,
  "temporary_bytes": 0,
  "total_entries": 12,
  "total_bytes": 211459250000,
  "complete_entries": 12,
  "offset": 0,
  "entries": [
    {
      "dir": "entries/0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "key": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "size_bytes": 17621606912,
      "modified_at": 1761983950.412,
      "last_accessed_at": 1761983990.102,
      "complete": true,
      "prompt_preview": "Render the scene."
    }
  ],
  "next_offset": null
}
```

#### 返回字段字典

| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| `max_bytes` | `int \| null` | 配置的磁盘清理高水位字节数 |
| `root_entries` | `int \| null` | 快照采样时整个缓存根目录下的条目总数 |
| `root_bytes` | `int \| null` | 快照采样时整个缓存根目录占用的总字节数（含临时文件） |
| `sampled_at` | `float \| null` | 快照采样的 Unix 时间戳（秒）。**若当前尚无任何 GC 采样快照，则为 `null`，且上述容量字段也为 `null`（不可将 `null` 当作 `0`）** |
| `temporary_bytes` | `int \| null` | 快照采样时在途写入临时文件占用的字节数 |
| `total_entries` | `int \| null` | 满足当前过滤条件（如指定 `cache_key`）的条目总数 |
| `total_bytes` | `int \| null` | 满足当前过滤条件的条目总字节数 |
| `complete_entries` | `int \| null` | 快照中满足当前过滤条件，且 manifest 与各 rank 文件元数据校验完整的条目数；不代表当前时刻仍可加载 |
| `offset` | `int` | 当前分页偏移 |
| `entries` | `list[object]` | 当前页条目列表，包含 `dir`、`key`、`size_bytes`、`modified_at`、`last_accessed_at`、`complete`、`prompt_preview` |
| `next_offset` | `int \| null` | 下一页起始偏移；若已到末页则为 `null` |

---

### 4.7 删除单条 L2 磁盘缓存：`DELETE /v1/zing/cache/entries/{cache_key}`

实时删除共享 L2 磁盘上指定 `cache_key` 的实体目录（不依赖 GC 快照，不影响各实例已有 L1，保留业务索引）。

- **路径参数**：`cache_key`（64 位小写 hex；格式非法返回 `HTTP 422` + `{"detail": {"code": "invalid_cache_key"}}`）。
- **成功响应（`HTTP 200 OK`）**：

```json
{
  "deleted": [
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  ],
  "missing": [],
  "failed": [],
  "reclaimed_bytes": 17621606912,
  "remaining_bytes": 211459250000,
  "sampled_at": 1761984000.125
}
```

| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| `deleted` | `list[str]` | 本次实际删除成功的 `cache_key` 列表 |
| `missing` | `list[str]` | 磁盘上原本就不存在的 `cache_key` 列表（幂等返回 `HTTP 200`） |
| `failed` | `list[object]` | 删除失败的条目及原因：`[{"key": "...", "code": "delete_failed"}]` |
| `reclaimed_bytes` | `int` | 本次删除实际回收的字节数 |
| `remaining_bytes` | `int \| null` | 最近一次 GC 快照记录的 `root_bytes`（非实时扣减值） |
| `sampled_at` | `float \| null` | 最近一次 GC 快照的时间戳 |

---

### 4.8 清空全部 L2 磁盘缓存：`POST /v1/zing/cache/clear`

实时删除共享 L2 磁盘目录下 `entries/` 中的全部缓存实体（保留 `index/` 业务索引，不影响各实例已有 L1）。

- **请求参数**：无。
- **成功响应（`HTTP 200 OK`）**：返回结构与 `DELETE /v1/zing/cache/entries/{cache_key}` 完全相同（`deleted` 列出所有被清除的 key，`missing` 为空列表 `[]`）。

---

### 4.9 离线磁盘管理 CLI：`sglang.multimodal_gen.tools.zing_block0_cache`

运维脚本可在不加载模型权重、不启动推理服务的情况下，直接管理已挂载的本地缓存目录：

```bash
# 1. 查看容量快照与分页列表
python -m sglang.multimodal_gen.tools.zing_block0_cache \
  --cache-dir /mnt/nvme/zing-block0-cache \
  list --offset 0 --limit 100

# 2. 将 CACHE_KEY_1 / CACHE_KEY_2 设置为 prefill 回执中的实际 key 后删除
python -m sglang.multimodal_gen.tools.zing_block0_cache \
  --cache-dir /mnt/nvme/zing-block0-cache \
  delete "$CACHE_KEY_1" "$CACHE_KEY_2"

# 3. 清空全部磁盘缓存实体
python -m sglang.multimodal_gen.tools.zing_block0_cache \
  --cache-dir /mnt/nvme/zing-block0-cache \
  clear
```

- **标准输出 (`stdout`)**：与对应 HTTP 接口完全一致的单行 JSON。
- **退出码 (`exit code`)**：全部成功返回 `0`；若 `failed` 非空则返回 `1`；若目录不存在或操作参数非法，向 `stderr` 输出 `{"code": "cache_management_failed", "message": "..."}` 并返回 `1`；缺少必填 CLI 参数或命令等 argparse 解析错误输出 usage 文本并返回 `2`。

---

## 5. 结构化缓存日志（`zing_cache_event`）

服务端沿用标准 logger 输出前缀为 `zing_cache_event` 的单行 JSON，用于串联请求、异步写盘与故障定责：

```text
zing_cache_event {"operation_id":"wasteland-480-a","session_id":"...","game_id":"game-a","group_id":"stage-1","instance":"node1:12345","rank":0,"event":"l2_write","cache_key":"0123...","result":"completed","reason":null,"elapsed_host_ms":7842.1}
```

- **公共字段**：`operation_id`、`instance`（`<hostname>:<pid>`）、`rank`、`event`、`cache_key`、`result`、`reason`、`elapsed_host_ms`（纯 Host 侧耗时），有业务上下文时附带 `session_id`、`event_id`、`game_id`、`group_id`。
- **日志级别规则**：
  - 默认 `--log-level info` 下，各实例仅由 **SP rank 0** 输出正常生命周期事件（`lookup`、`prefill`、`l1_fill`、`l1_evict`、`l2_write`、`preload`/`unload`/`delete`/`clear`、发生实际删除的 `gc`）。
  - 异常或部分失败（如写盘空间不足、跨进程覆盖竞争、`preload` 未全部就绪、管理请求被拒绝）在**实际发生故障的 rank** 输出 **`WARNING`**。
  - 高频状态轮询（`status`、`list`）、空转 `gc` 及非零 rank 的正常分片事件仅在 `--log-level debug` 下输出。

---

## 6. 部署红线与标准接入流程

### 6.1 三条部署红线

1. **每个共享磁盘目录必须且只需 1 个实例开启 `--zing-block0-cache-gc`**
   - 同一台物理机共用同一个 `--zing-block0-cache-dir` 的多个实例中，**必须指定 1 个实例加上 `--zing-block0-cache-gc`**（内部仅 `local_rank == 0` 启动后台 GC 线程）。
   - 若所有实例均未开启 GC：磁盘满后不会自动淘汰旧条目，容量快照也不再更新。新目录尚无快照时，`GET /v1/zing/cache` 的 `sampled_at` / `root_bytes` 为 `null`；已有快照则可能持续返回旧值。
   - 物理磁盘需预留 10%–15% 余量（如 4.5 TB 分区配置 `--zing-block0-cache-disk-gb 4000`），承载并发写入临时文件；即使极端写满，`OSError` 也会被捕获并降级为 L2 写入跳过，**不会导致推理进程崩溃**。
2. **严禁直接挂载 OSS / S3 FUSE（如 ossfs）或跨机 NAS（CPFS/NFS）作为 `--zing-block0-cache-dir`**
   - 当前 L2 存储引擎针对**同机本地 NVMe（ext4/xfs）**设计：每个条目包含多层目录、`.access` 访问时间戳 `touch()`、`index/<game_id>/<group_id>/<key>.json` 小文件，以及依赖本地 POSIX 语义的临时文件写完后 `os.replace` 原子替换与 `rglob` 目录树遍历。
   - 对象存储 FUSE 挂载不支持真正的原子覆盖写，且高频元数据 `stat`/`utime` 会触发严重延迟甚至 I/O 报错。若后续需接入 OSS 冷备，需另行通过独立 PR 实现“单键归档打包 + 本地 NVMe 落盘中转”，**禁止在部署层直接将缓存目录指向 OSS 挂载点**。
3. **实例级路由亲和与 `dp_size=1` 约束**
   - L1 Pinned 内存驻留在具体 Worker 实例（SP 组）内，要求 `--dp-size 1`（否则 `preload`/`unload`/`status` 返回 `HTTP 409 / cache_requires_dp1`）。
   - 中台对**实例 A** 调用 `POST /v1/zing/cache/preload` 并确认 `"all_ready": true` 后，**必须将该局游戏的后续视频 WebSocket 会话定向路由到实例 A**。

### 6.2 同机双实例参数示例

以下展示四张可用 GPU 上两个 SP=2 实例的卡位、端口和缓存参数，不作为已验收的完整模型部署配方。
同一网络命名空间内，每个实例还会占用 `HTTP port + 1` 作为离线 broker 端口；
两个实例不能使用相邻 HTTP 端口，且应与 scheduler、master、conditioning 和 role 服务端口错开。
先按[部署 recipe](/docs/sglang-diffusion/zing/deployment_zing_h3)准备模型、codec、AdaLN 等配置，并使用已有的 `--pipeline-config-path` JSON。
服务必须通过 `--mem-fraction-static` 启用静态 KV 池，且每个实例只允许一个活跃会话。
比例的含义、容量计算和验证状态见
[容量契约](/docs/sglang-diffusion/zing/specs/PERFORMANCE_DEFAULTS#预分配-kv-池的容量字段)。
未设置 KV pool 时，Worker 不会创建 Block0 cache owner，仅传 cache 参数不能启用缓存。
若沿用 Encoder/Decoder 分卡部署，保留 recipe 中的角色参数，并为两个实例分配互不重叠的 GPU。

先创建位于已挂载本地 NVMe 下的缓存目录。下方 800/4000 仅为容量示例：两个实例各 800 GB
将合计预分配约 1.6 TB pinned 内存，部署方须按可用主机内存与模型占用调整预算。

```bash
# 两个终端都先设置以下变量，再分别执行对应实例的命令。
MODEL_PATH=/models/zing-h3
PIPELINE_CONFIG_PATH=/models/zing-h3/pipeline.json
CACHE_DIR=/mnt/nvme0/zing-block0-cache
# 使用该硬件与输入集合实测通过的比例；当前迁移版本尚无推荐值。
: "${MEM_FRACTION_STATIC:?Set the validated static memory fraction}"

# 在独立终端启动实例 0：承担共享目录的 GC。
CUDA_VISIBLE_DEVICES=0,1 sglang serve \
  --model-path "$MODEL_PATH" --pipeline-config-path "$PIPELINE_CONFIG_PATH" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --num-gpus 2 --sp-degree 2 --dp-size 1 \
  --port 30000 --master-port 30005 --scheduler-port 30555 \
  --zing-block0-cache-mode on \
  --zing-block0-cache-dir "$CACHE_DIR" \
  --zing-block0-cache-host-gb 800 \
  --zing-block0-cache-disk-gb 4000 \
  --zing-block0-cache-gc \
  --zing-block0-cache-gc-interval-s 60 \
  --zing-block0-cache-gc-stale-s 600 \
  --zing-block0-cache-gc-low-watermark 0.9

# 在另一终端启动实例 1：共享 NVMe，不开启 GC。
CUDA_VISIBLE_DEVICES=2,3 sglang serve \
  --model-path "$MODEL_PATH" --pipeline-config-path "$PIPELINE_CONFIG_PATH" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --num-gpus 2 --sp-degree 2 --dp-size 1 \
  --port 40000 --master-port 40005 --scheduler-port 40555 \
  --zing-block0-cache-mode on \
  --zing-block0-cache-dir "$CACHE_DIR" \
  --zing-block0-cache-host-gb 800 \
  --zing-block0-cache-disk-gb 4000
```

### 6.3 中台标准调用闭环

1. **离线/开服预热**：连接空闲实例的 `WS /v1/realtime_video/prefill`，携带 `game_id`、`group_id` 串行灌入固定场景，收到 `prefill.completed` 后，该实例的 L1 已可使用；跨实例使用时，后续以目标实例的 `preload` 结果确认就绪。
2. **开局前装载 L1**：选定目标空闲实例，调用 `POST /v1/zing/cache/preload`（传 `{"game_id": "...", "group_id": "..."}`），检查返回 `HTTP 200` 且 **`"all_ready": true`**（`missing` 也可能来自未完成、损坏或不兼容的 L2；先检查日志并按需重试或补热）。若在仍持有该 key 的原实例补写 L2，使用 `force=true`；默认 `force=false` 命中 L1 后不会重新写盘。也可在没有该 L1 条目的目标实例执行 `prefill`。
3. **接入玩家流量**：将玩家视频会话定向路由至该实例（默认 `cache_write=false`，Block0 只查 L1、不读取 L2；未命中则重算）。
4. **切关/退房轮换**：`preload` / `unload` 要求实例空闲。需在同一视频会话内切关时，提前装载所需分组并保证能同时驻留；要动态装载或卸载，则先结束当前会话并等待释放，再调用 `preload` / `unload`，完成后重新接入视频会话。活跃会话期间调用会返回 `409 cache_busy`。

自动缓存写入遇到前缀超过 L1 host 页预算时跳过该条缓存，当前生成继续；显式预填充返回 `cache_capacity_exceeded`。这不会扩大 GPU 容量，也不会驱逐原有缓存来尝试存放无法容纳的条目。
