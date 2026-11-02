---
title: "串行 Block0 prefill WebSocket"
description: "固定 case 预热、强制覆盖与异步磁盘写回的接口契约。"
---

# 固定 case 的串行 Block0 prefill WebSocket

2026-10-30 候选代码增加可选 game_id/group_id，并简化 key；`65ae9daf74` 的缓存专项两轮 378 项已通过。
显式 preload/status/unload 的参数、返回值结构、HTTP 状态码与业务语义见
[缓存管理 API 与部署契约](/docs/sglang-diffusion/zing/specs/block0-cache-management-api)与
[最新确认方案](/docs/sglang-diffusion/zing/plans/BLOCK0_CACHE_MANAGEMENT_SHARED_L2)。

启动预热客户端在服务就绪后、真实用户进入前，连接
`/v1/realtime_video/prefill`。同一连接逐个提交完整 case，收到本条完成或可恢复失败
回执后才提交下一条。服务端复用普通 realtime 的模型请求校验、sampling 参数构造、
scheduler RPC、worker 会话与取消清理；worker 在条件编码、参考图 VAE 和 Block0
KV 填充后提前返回，不运行 source VAE、目标扩散采样或视频 Decoder。

## 请求与回执

所有消息都是二进制 MessagePack。建立连接后返回：

```json
{"type":"prefill.ready","max_message_bytes":67112960}
```

`max_message_bytes` 以服务端实际回执为准。每个 case 的字段沿用现有 source `init`
请求，`type` 改为 `prefill`，增加字符串 `request_id`（1–128 字符）和 `source_frame`。
后者为恰好一帧、行优先 RGB24 bytes，长度为 `width × height × 3`；参考图的
`payload` 仍为 PNG/JPEG/WebP 编码 bytes。以下为 Python 请求示意：

```python
case = {
    "type": "prefill",
    "request_id": "wasteland-480-a",
    "force": False,
    "game_id": "game-a",
    "group_id": "stage-1",
    "zing_realtime_api_version": 1,
    "prompt": "Render the scene.",
    "size": "832x480",
    "source_stream": {"format": "rgb24", "width": 832, "height": 480, "fps": 24},
    "source_frame": s0_rgb_bytes,
    "references": [{"id": "ref-0", "format": "png", "payload": reference_png_bytes}],
}
await websocket.send(msgpack.packb(case, use_bin_type=True))
receipt = msgpack.unpackb(await websocket.recv(), raw=False)
```

成功回执的形状为：

```json
{"type":"prefill.completed","request_id":"wasteland-480-a","session_id":"connection-id","cache_key":"content-key","hit":false,"entries":1,"evictions":0}
```

回执发送前，L1 条目已发布，临时 KV 租约已释放，普通 worker 会话清理已得到确认，
本条准入租约也已释放。异步 L2 写盘不纳入完成边界。命中已有条目也返回成功，
`hit=true`。不会发送视频帧。重复 request_id 不提供幂等保证，客户端负责关联回执。

可选 `force` 必须为布尔值，默认 false。true 时跳过 Block0 L1/L2 命中，重新计算
Block0，更新本实例 L1 并允许覆盖已有 L2；Qwen/参考图编码缓存仍可正常复用。
该字段不进入模型请求或 cache key。强制覆盖继续使用异步写盘，成功回执不承诺 L2 已完成
覆盖；后台失败记录日志，允许后续 miss/重算。其他实例的已有 L1 和活跃会话不受影响。
并发覆盖允许竞争 miss，不保证按请求开始顺序发布。PR #467 本轮确认同 key 结果等价，
允许各 rank 来自不同缓存写入批次，但必须全部成功加载并通过兼容性校验才命中。
缓存 generation 一致性检查已移除；会话生命周期身份校验不受本次调整影响。

可选 game_id/group_id 登记本次 cache_key 的业务关联，均为最多 128 UTF-8 字节的非空字符串；
group_id 要求同时提供 game_id，省略 group_id 表示未分组。命中已有缓存也登记关联。
它们通过操作元数据传入，不进入模型请求或 key。索引发布失败记录日志，不回滚成功 L1；
磁盘异步写入期间，从其他实例 preload 仍可能返回 missing。
正常视频默认 cache_write=false，只查询 L1；这不改变 prefill 的计算/复用/force 语义。

## 顺序、错误与生命周期

- 一次只执行一个 case；不维护待执行队列。提前提交的新 case 返回
  `prefill.failed / prefill_busy`，不会中断当前 case，也不会在之后偷偷执行。
- 输入错误返回 `prefill.failed / prefill_invalid`，`fatal=false`，允许修正后重试。
  服务被其他生成会话占用时同样返回可恢复的 busy；空闲连接不占生成名额。
- 不可恢复的 worker/通信故障返回友好的英文错误，`fatal=true`，关闭连接。
  内部异常只写服务端日志。尚未收到成功回执的 case 不能视为已完成。
- 客户端断连或处理被取消时，沿用普通请求的 drain：等已提交 worker 工作结束，
  再释放输入、worker 状态及准入资源。缓存保留于 worker，连接可重新建立。
- 同一连接各 case 使用同一 session routing key，保持在同一 DP replica；每个 case
  重新创建 generation 身份并清理。多副本部署仍需由部署方覆盖各目标 worker，
  单个连接成功不能证明整个副本集已预热。

## 部署边界

复用已有 KV pool 及 `--zing-block0-cache-mode` 配置；缓存关闭时预热失败。
prompt、参考图内容/ID/顺序、S0 RGB 像素、几何、zing_config、SP 和 KV dtype 须与实际请求一致。
现有容量与 LRU 规则保持适用；`entries` 和 `evictions` 是当前 worker 的累计状态，
单个 case 曾成功不保证之后仍驻留。启动客户端负责完成所有必需 case、检查淘汰并
开放业务流量；此接口不自动修改服务 readiness。

现有 benchmark 参数 `--prewarm-only` 仍表示完整生成 warmup 视频、跳过正式测量。
此 WebSocket 只做 Block0 prefill，两者执行范围不同。

共享磁盘管理与删除接口见[管理方案](/docs/sglang-diffusion/zing/plans/BLOCK0_CACHE_MANAGEMENT_SHARED_L2)。

本次使用零 GPU CPU 协议/生命周期验证，并以同机本地 NVMe 作为磁盘验收目标；跨机器网络存储不在支持范围。
当前预热可同步从磁盘加载到本实例 L1，条目之后仍受 LRU 淘汰；后台写盘中的槽位复用可能等待。
不声明 CUDA、真实模型精度、性能或
480/736 GPU 验收。作业与回执见部署文档的当前验证状态。
