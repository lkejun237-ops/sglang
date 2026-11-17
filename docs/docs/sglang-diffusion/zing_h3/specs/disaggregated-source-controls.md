# Source controls on managed Encoder / DiT / Decoder groups

The existing player and native protocols remain the public contract. A player
owns one parent Model Slot; Prefill, generation and scene children use its serial
compute lock. Child completion never releases the player's parent allocation.
The separate upload Prefill endpoint is a model-only admission: it never creates
an SDK member and cannot supply or bypass a real player's SDK cleanup receipt.

## Conditions and startup

Room and the split Encoder use `clip-barrier-v2`. Room finishes the old clip at
boundary B. After preparing the candidate and submitting it in FIFO order,
Encoder returns `source_ready` so Room can resume with a fresh capture. DiT
drains the old source, verifies B, and uses the original admission and
`activate_prompt_at_source_boundary` before processing source at or beyond B.
It confirms CUDA completion and identical SP receipts before `kv_ready`.
Identical complete conditions can reuse the resident prefix and report
`bypass`. Different conditions use their own Block0 and candidate references.
The first RGB bearing the new version confirms application.

The optional deferred-KV protocol keeps `kv_ready` unchanged. An Encoder advertising
`source_condition_deferred_kv=true` accepts `allow_deferred_kv=true` on a clip-barrier
prompt update. When the DiT's prompt-joint policy is enabled, it freezes history at
the boundary and returns `conditioning.ready(kv_deferred=true)`, translated to
`control_ack(state=source_ready)`. Room resumes fresh source on this receipt; KV
materialization and the joint first step happen after that source arrives.
`source_ready` does not mean KV completion or prompt application. Only the matching
new-version output confirms application. CUDA dependencies and SP receipt agreement
still complete before readiness. No capability/request means the original serial
barrier, including mixed-version Room/Encoder/DiT deployments. Resident-prefix
reuse may finish synchronously and still return `kv_ready`.

Standalone DiT CLI `--zing-source-prompt-joint` / `--no-zing-source-prompt-joint`
overrides `SGLANG_ZING_SOURCE_PROMPT_JOINT`; its unconfigured default is false.
This differs from integrated ServerArgs defaults. Real joint execution still
requires supported source/target layout; unsupported layouts use existing fallback.
See [design and validation](../../zing/plans/serving-ttff-breakdown.md).

`source.prepare` v1/v2 uses the existing immutable preview/identity contract.
Encoder prepares the original Qwen/reference condition and waits for the connected
DiT's confirmation. `source.prepared(level=condition)` is not an assertion that a
future generation's GPU KV has already been restored. Duplicate preparation must
have identical pixels **and dimensions**. v1 retains the exact first-frame check;
v2 retains its independently sized preview semantics. Old peers lacking the
negotiated capabilities fail explicitly instead of silently omitting the fence.

Room sends changes to its hot-reloaded auto sink configuration over the existing
model connection. DiT schedules the refresh in committed 5-latent cycles. A due
refresh at the prompt boundary is absorbed by the prompt rebuild. Manual Sink
on/refresh/off controls keep their existing session and event ownership.

## Public upload Prefill

The same signed ticket, binary upload, status/idempotency and persistence messages
remain in use. The Gateway selects the configured model backend through the
existing lease factory. A model-only group records its operation at admission;
that immutable identity cannot be converted into a player allocation.

`memory_ready` is exposed after positive child-cleanup evidence. The caller can
then release all native roles without an invented SDK receipt. Pending disk
writeback is observed through the original allocation and boot identity, within
the original bounded deadline. A terminal Prefill may only inspect persistence;
it cannot renew capacity, access another generation, open a generation socket,
or change a later player's owner. Ordinary players still require their actual
SDK retirement proof before terminal release.

## Cache evidence and acceptance

Prefill reuses the original Block0 cache and model key. L1 lookup, L2 promotion,
Host publication and GPU restoration remain distinct. A second bookkeeping L1
lookup must not erase a first L2 promotion. Public `model.cache` uses the existing
sanitizer; native per-rank `DIT_CACHE_RECEIPT` retains transfer-event timing without
synchronizing merely to collect a metric. An unavailable CUDA timing stays absent.
Expected cache admission failure is a request rejection, not permission to
restart or quarantine a healthy model group.

Required validation uses real input identities and request/worker epochs: both
user WebSocket and CLI-originated Prefill; ten distinct keys warmed **before**
recall, all ten classified as L1 hits; separate L2 read/promotion and GPU restore;
auto sink configure, refresh and disable; Sink on/refresh/off; scene changes, cancellation, expiry,
reconnect, next-player reuse, recording and sustained visual quality. CPU protocol
tests do not prove CUDA, checkpoint quality or real-game CLI output. No runtime
configuration switch substitutes for the missing native protocol implementation.

Current implementation work has CPU protocol evidence. GPU/SP2, real-game media
and ten-entry resident-hit acceptance remain pending until recorded in the PR.


### 按模型会话开启详细计时

模型 init 可选 `detailed_timing`（严格 boolean，默认 false）。Encoder 将该值写入自己的
source config 和 DiT session.open；DiT 的 Decoder output_session 原样携带该值。每个
角色在会话内固定读取，下一会话重新取值；不允许 prompt.update 临时改变采样策略。
关闭时不创建本 PR 详细 timing/CUDA events，不发送 model.timing；已有 cache 回执与
执行 fence 保持原逻辑。Room 的基础成功 TTFF 不依赖此开关。开启不保证旧角色具备埋点，
缺失保持显式，部署需配套三角色。见 [TTFF 方案](../../zing/plans/serving-ttff-breakdown.md)。

## Request-scoped frame latency diagnostics

`return_stage_timing` is the existing model init field, defaulting to `false`.
The managed Encoder forwards its choice to the connected DiT and Decoder for
that session. Detailed frame timing is opt-in: the disabled path does not record
its sampling/denoising CUDA events or publish `chunk_timing` / `chunk_telemetry`.
Changing it requires a new session, not a worker restart or an in-flight toggle.
The flag also enables existing stage timing; comparisons must describe the cost
of the whole switch rather than attributing every difference to CUDA events.

The dedicated source group's data path is Room → Encoder → DiT → Decoder → Room.
The Decoder process owns `DecoderOutputBroker`: it combines the DiT's timing
sideband with local decode timing and sends `chunk_telemetry` on its output WS.
Room receives and sanitizes it into `whitebox.model_timing`; Room does not send
`chunk_telemetry`. This direct-output route predates frame timing. The legacy
non-attached path instead returns a shared-memory reference through the DiT /
Encoder output path. Neither is a claim about every community diffusion pipeline;
a normal integrated pipeline returns its `OutputBatch` through GPUWorker /
Scheduler / API. Timing this dedicated group does not validate that other route.

CUDA event recording enqueues stream work; it is not a stream synchronize.
The diagnostic implementation resolves elapsed times at transport completion
boundaries and preserves tensor ownership synchronization even when disabled.
The Host interval around metadata send, contiguous conversion, NCCL submission /
wait and stream completion is a sender handoff latency, not isolated NCCL kernel
or link-transfer time. `dit.return_ms` is a broader residual and should not be
labelled pure network time. See the [replay guide](../../../../../benchmark/whitebox/LATENCY_BREAKDOWN.md)
for measurement boundaries and same-code off/on experiments.
