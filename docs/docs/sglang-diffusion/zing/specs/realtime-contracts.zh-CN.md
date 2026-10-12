# Zing 共享像素 provenance 与 ID（#111 / #118）

本 PR 只提供 codec 和 chunk planner 共享的数据类型，直接基于 main。入口为 `python/sglang/multimodal_gen/runtime/realtime/zing_contracts.py`，仅依赖 Python 标准库。当前主服务没有接入持续 source 协议；此 PR 不实现或宣称协议已可用。

## 数据接口与调用方

| 接口 | 语义 | 消费者 |
| --- | --- | --- |
| `PixelRange(start, end)` | 绝对像素帧半开区间，不是 latent 区间 | #120 codec、#121 分块输入 |
| `SourceSpan(source_id, source_frame_start, output_frame_start, num_frames)` | 一个 source 的连续像素映射，首版同 FPS、同时间轴 | #120 输出、#114 服务对齐 |
| `Provenance(pixels, source_alignment, source_dependency_end)` | 不可变完整映射与实际计算依赖上界 | #120 codec、#121 ready chunk |
| `slice(pixels)` / `concatenate(parts)` | 拆分保留原依赖，连续合并取最大实际依赖，不读取 latest accepted | 网络重分批和 codec/chunk bridge |
| `TemporalChunkId` / `PackedRoleId` | 不同静态类型；相同时间范围 source+target/audio 算一个 temporal chunk | #121 chunk planner |

#120/#121 目前为未合并消费者，迁移通知已通过对应 issue/PR 发布。这里不把下游分支引用写成 main 已有运行调用链。数据类型只验证自身结构；source_id 是否真正 accepted、是否先发送 ACK、是否超出可用依赖，由服务所有者检查。

像素位置保持绝对时间轴，不使用 `latent_index*4`，不将 append、VAE clip 或 packed role 计数当作 temporal chunk。AR `[1,2,2]` 与 DMD `5` 的 checkpoint 分块规则属于 #121；DMD `5` 不是 solver 步数。窗口包含关系和 block-relative RoPE 不在本模块定义。

## #114 交接与旧路径迁移

原 `runtime/entrypoints/openai/realtime/zing_contracts.py` 已删除，不保留没有生产调用方的兼容封装。调用方改从 `runtime.realtime.zing_contracts` 导入上述类型。

`ModelProfile`、`Init`、`Limits`、`parse_init`、`Accepted`、`Phase`、`SourceLedger`、`ContractSession`、`OutputUnit` 一并从本 PR 移除。它们属于部署验证、接收预算、协议事件、生命周期和传输层，应随 #114 的真实调用方一起实现、测试和审查，不能仅通过 fake-worker 验收提前作为生产代码合入。

旧原型可在提交 `a295939cda47cde67cfda213f26aa09b289b5c67` 查看，仅作为设计参考，不要求原样搬回。#114 需要在真实 adapter/worker/writer 调用链实现：

- init/ready 版本协商，核验实际 checkpoint/solver/schedule；静态 reference 与持续 source 分离。
- 帧数、data 字节和完整 MessagePack 三种额度；完整 accepted 前缀 consumed；必须在真实有界下游接管后释放预算。
- 为 codec/chunk bridge 提供 `Provenance`，以实际 encoder/AR/decoder 依赖维护水位，先验证再提交游标。不得按 latest accepted 伪造依赖。
- prompt 安全边界快照；单 writer 完整发送 header/payload 后才推进输出；source.end flush 有效尾帧后 completed。
- 真实队列背压、失败回滚、在途取消和所有 owner 清理；错误映射使用友好英文，不泄露内部异常；若改 gateway 合约则同步 AsyncAPI。

#120 原先可选 `source_ledger` 依赖不能只改 import：应由 #114 拥有实际账本、协商额度和 provenance 解析，在 codec/服务边界完成交接。不要在 codec 内另造一个假账本。#121 的数据类型 import 可直接迁移。

## 验证范围

运行 `python3.11 -m pytest -q test/registered/unit/realtime/test_zing_realtime_contracts.py`，验证区间/映射合法性、不可变性、跨 source_id fixture、切片、连续合并和 10 个固定随机种子的独立逐帧对照。

共享 fixture：`test/registered/unit/realtime/fixtures/zing_source_provenance.json`，输出 `[15,19)` 跨两个 source_id，实际依赖上界 34；切片保留依赖。测试通过文件路径加载实际模块，绕过 GPU package 初始化。

旧版本的 103 项协议原型测试及其 GPU 联合 harness 不再作为当前精简版本的验收数量或生产证据。真实服务协议与网络/codec/GPU 联调由 #114 连同调用方交付；本 PR 仅验收共享数据契约。

## #256 conditioning RPC 的实时错误与版本字段

本节描述 #256 RPC v6 本地实现增加的字段，前文 #111/#118 的范围说明是历史共享数据类型交接。CPU 契约验证不能证明在线 GPU 服务或前端已经完成部署验收；本轮不修改前端代码。

`remote` 的条件媒体只允许图片、自动 source S0 和合法单帧 video presentation（`clip.start == clip.end`，可以是非零时间）。多帧 video、audio、video_audio 在初始化或 prompt 更新的元数据预检中拒绝。`session.ready` 原有 `{start:0,end:0}` 是默认单帧位置提示，不能解释为只支持零时间。local 保留原生输入域。

实时错误通过既有 MessagePack 消息发送，`type:"error"`、`content` 保留，追加 `code` 和 `fatal`：

| code | content（公开英文信息） |
| --- | --- |
| `CONDITION_INVALID` | `We could not prepare this request. Please try again.` |
| `CONDITION_UNAUTHORIZED` | `We could not prepare this request. Please try again.` |
| `CONDITION_PROFILE_MISMATCH` | `We could not prepare this request. Please try again.` |
| `CONDITION_BUSY` | `The service is busy. Please try again shortly.` |
| `CONDITION_TIMEOUT` | `Preparing your request took too long. Please try again.` |
| `CONDITION_UNAVAILABLE` | `The service is unavailable. Please try again shortly.` |

首次 conditioning 失败时发送 `fatal:true` 并结束会话。中途 prompt 更新失败时发送 `fatal:false` 和该失败更新的 `event_id`，继续使用旧 conditioning 出帧；同一 `(prompt,event_id)` 只通知一次，新事件可再次尝试。内部异常文本只写服务日志，不放入公开 content。

```json
{"type":"error","code":"CONDITION_BUSY","content":"The service is busy. Please try again shortly.","fatal":false,"event_id":11}
```

帧 header 的 `event_id`、`prompt_version` 表示该 block 实际使用的 prompt。失败更新不会推进它们；延迟输出的旧 block 也保留其旧版本。有限/block 输出逐项使用 `usage.zing_realtime_blocks` 的快照；source 输出优先使用原 `source_blocks` 的事件与 provenance，并追加实际 prompt_version。source_alignment、source_dependency_end 与 accepted/consumed 水位的原含义不变。

worker IPC 的 `OutputBatch.error_code` 传递稳定错误码，`conditioning_notice` 传递非致命通知，`realtime_effective_prompt` 传递 block 的生效版本；这些字段不是供浏览器提交的输入。有限 HTTP 请求的失败按公开码映射为 400/503，消息仍使用上述公共英文文案。完整 frontend AsyncAPI 不在本仓库，未同步或发布。
