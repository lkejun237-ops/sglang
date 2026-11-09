# Optional Prefill on the play control connection

Status (2026-10-31): deployed on Shanghai, with real-browser acceptance in Dream Walk S01/S02/S03 plus ordinary no-Prefill and default single-Prefill sessions. Model endpoint overlay `bb72f98872`, Room fixes `375b044e2d` / `bc6a39a452`, and the minimal frozen-CLI input-lease fix from SDK `dd5d8fc4d` are in service. This supersedes the temporary browser RGB export / `prepared_input` flow. The existing independent `/v1/realtime_video/prefill` endpoint remains available for offline callers.

## Ownership and compatibility

`WHITEBOX_BOUND_COMPUTE=1` enables one model reservation owned by the Room source session. The reservation is acquired before the CLI is opened. It survives Prefill, completed video segments, and scene changes until the source session closes. Every operation is routed through the same worker connection and epoch; a child frees only its verified scheduler state. Unknown cleanup quarantines the reservation. The model's internal `/v1/realtime_video/bound` socket requires an existing gateway reservation and is not a browser endpoint.

Old clients do not need to send Prefill or an extra attach command. Their normal `render.start`/automatic startup path still works. WebRTC video, WASD, Prompt controls, source/session fencing and reconnect grace are retained. The API does not contain game IDs, scene names or style logic.

New clients check `session.ready.capabilities.control_prefill_protocol == "whitebox-control-prefill-v1"`. They may use the existing asset upload API before freezing the source. Commands below use the existing reliable control WebSocket, binary MessagePack and version 2. Do not place raw RGB, encoded reference bytes, arbitrary worker addresses or a caller-selected cache key in Prefill controls.

## Prepare one context and its real video F0

After ending the previous model segment, send:

```json
{
  "version": 2, "type": "command", "op": "render.prepare",
  "request_id": "context-01",
  "source_session_id": "SOURCE", "source_epoch": "EPOCH",
  "expected_render_revision": 0
}
```

The Room acknowledges the operation, stops the dedicated CLI World's simulation clock, drains already-started captures, and captures exactly one new RGB frame. It retains that frame as this model context's F0. An `operation.result` with `state: "completed"` contains `context_id`, `source_frame_index`, `source_pts_us`, and `source_frame_sha256`; it never exports RGB to the browser.

The CLI's authenticated `POST /v1/sessions/{id}/capture` is an internal source-control endpoint (`action: hold|resume`, `requestId`). Shared/multiplayer views cannot pause their authoritative World through this endpoint. The same SDK clock is resumed, without replacing the World or replaying repeated still frames. Source capture, UI and full-source archive timestamps keep the real hold gap. Only the new model segment places F0 near resume on a separate `model_presentation_pts_us` timeline; its selected-source/model pair uses that same playback time. Original capture timestamps, source frame IDs, input receipts and wall latency remain intact. The hold is bounded (10-second first-capture timeout, 300-second safety release); the Room context expires at 270 seconds.

The Room's ordinary 15-second media-read deadline follows capture state. A
pending authenticated hold has a bounded acknowledgement wait; a confirmed
hold permits media silence until the SDK safety deadline. This also updates
reads already in progress. Control connection failure still ends the read.
After a confirmed resume, the source has a fresh 15-second frame budget.

Player input inactivity starts on the first validated source frame, not source
readiness or input received before that frame. Exclusive foreground preparation
and Prefill suspend input inactivity without releasing the player's parent Slot.
The first `memory_ready` after verified child cleanup restores a full inactivity
budget for the user's start command, while F0 remains held. Duplicate replies
do not extend that budget. A cancelled/resumed source restores the budget only
when a new source frame arrives. Shared-world snapshot Prefill neither pauses
the World nor suspends or resets input inactivity.

The context deadline starts at accepted preparation and is shared by all its
serial Prefills; each wait uses the remaining context budget. Expiry retires F0
before source recovery and child cancellation. Unconfirmed child cleanup is a
player failure, never free capacity. Player lifetime, connection recovery and
normal cleanup deadlines remain independent and cannot be extended by Prefill.
These rules use the existing control messages and require no client API change.

## Multiple optional conditions under that S0

```json
{
  "version": 2, "type": "event", "eventType": "prefill",
  "request_id": "prefill-A",
  "source_session_id": "SOURCE", "source_epoch": "EPOCH",
  "payload": {
    "context_id": "context-01",
    "prompt": "Complete current condition text", "seed": 42,
    "references": [{"id": "<Picture 1>", "asset_id": "sha256:CONTENT_HASH"}]
  }
}
```

A, B and C can be submitted serially with distinct request IDs and inputs. Each receives `prefill.accepted`, `prefill.computing`, and `prefill.memory_ready` or `prefill.failed`. The native cache key and cache/locking implementation are unchanged. When the live source needs model-grid padding, Prefill uses exactly the same padding and model geometry as normal video; the public F0 hash still describes the original source pixels. Replies carry the context, model session, worker identity/epoch, key and native timing details. `memory_ready` is emitted only after native compute cleanup has completed; it does not wait for disk persistence.

`prefill.persisted` arrives separately on the same reliable connection. Storage failure is nonfatal and does not invalidate an already-published L1 entry. Each request keeps its own state; the last prepared condition never becomes the active Prompt automatically. L1 may evict earlier entries according to its existing byte budget; several historical `memory_ready` receipts do not promise simultaneous permanent residency.

Only one child computes at a time; at most 256 Prefill request identities are retained per source session. Another new Prefill while computing or while a video segment is active receives nonfatal `PREFILL_BUSY`. Existing request ID plus identical content replays its receipt; different content receives `REQUEST_ID_CONFLICT`. This is bounded request idempotency, not another model/cache scheduler. Reconnection uses the same existing source session; worker loss invalidates the old binding instead of silently moving its L1 guarantee to another worker.

Native CPU writeback buffers can remain occupied after `memory_ready`. If a later Prefill child fails with verified cleanup while earlier persistence is pending, Room allows at most two retries (500 ms then 1 s backoff), emitting `prefill.retrying`. The request ID, conditions and F0 remain unchanged; each attempt gets fresh disposable compute state. Unknown cleanup and protocol/input validation failures are not retried. Normal success still returns at `memory_ready`, without awaiting disk persistence.

## Start the selected condition

A client may start without any Prefill, after only A is ready, or after A/B/C are ready. To use prepared F0, the original render command includes the returned context and exact boundary:

```json
{
  "version": 2, "type": "command", "op": "render.start",
  "request_id": "start-A",
  "source_session_id": "SOURCE", "source_epoch": "EPOCH",
  "expected_render_revision": 0,
  "context_id": "context-01",
  "start_boundary": {"source_frame_index": 123},
  "prompt": "Complete current condition text", "seed": 42,
  "references": [{"id": "<Picture 1>", "asset_id": "sha256:CONTENT_HASH"}]
}
```

The first selected RGB is exactly the pinned F0. The source resumes before subsequent input frames are required. Native lookup decides L1/L2/miss from the actual inputs, regardless of which group was prepared last. No selected Prompt is required to have been prefilled. Subsequent normal `prompt.replace` uses the same context's S0. A new scene ends the child segment and prepares a new context/F0 while retaining the parent worker reservation.

`render.prepare.cancel` with `context_id` cancels outstanding prefill compute, verifies cleanup, resumes the source and releases its pinned RGB. Source close performs the same cleanup. Heartbeats and cancellation do not wait behind GPU Prefill work in the connection receive loop. Normal WASD/action handling resumes with the source clock; inputs cannot mutate a held F0.

## Acceptance and metrics

Local tests cover: parent reservation ownership across three Prefills and later segments, unknown-cleanup quarantine, bounded pending-frame draining on cancellation, real-F0 header selection, independent A/B/C keys and no implicit activation, duplicate/conflicting IDs, busy during generation, legacy startup, native cache receipts, HTTP capture authentication, one-frame hold and continuous resumed presentation time.

Shanghai acceptance must separately report:

- Prepared-input coverage and actual online L1/L2/mixed/miss counts, per scene and Prompt version. Missing receipts are not hits.
- Native lookup, disk/promotion and GPU restore timing with sample count and Host/CUDA/rank scope. Prefill cache lookup is not the same measurement as restoring GPU KV for online generation.
- Source/raw model/decoded/presented FPS over time, display gaps, visible-input P50/P95/max/count, RTT and lifecycle resource release.
- Screenshots before/after cache activation and scene/Prompt switches; compare backend recordings when browser blur/mosaic may be transport-related.

Shanghai 480 acceptance: nine prepared groups across three scenes reached `memory_ready` and persisted; all three scene starts recalled the originally selected key from L2. A separate default single-group browser run recalled L1 and remained playable. The three-scene session completed after 616.4 seconds with 12,967 model frames and no recorded session error. Browser telemetry was sampled separately from native cache timings; the local journal reached its 2 MiB bound at 395.2 seconds, so later FPS/latency samples are not included in its summaries. The final browser diagnostic and screenshots cover completion and later visual observations. This is three scenes within one game, not ten different games. No same-input pixel-exact cold/hit comparison or 736 acceptance is claimed.

The dedicated hold retires the input lease revoked by `World.stop()` and reacquires it after `World.start()`. Source/UI timestamps, worker affinity, native cache identity and locks remain unchanged. Without this lease renewal, the first post-resume input could fail with `WORLD_REMOTE_INPUT_RELEASED` despite successful cache recall.

## white_box 合并兼容范围（2026-10-31）

共享首帧接收端继续支持显式 v1 和 v2。v1 保留源帧尺寸与 video zero 一致性及原有
fallback；v2 保留独立条件预览的尺寸和像素，不因 CLI 首帧不同重新申请模型租约。
缓存/Pipeline 的输入冻结与阶段计时共用原生路径，native Prefill 不改缓存锁或读写格式。

同连接控制入口接受只有 `eventType` 的 Prefill 消息，同时保留已准入 SDK member
的媒体、输入和冻结模型所有权；这类 member 不借新的 eventType 发起额外模型操作。
当前远端 Player/Room、原操作恢复、模型池容量与错误来源保持原合同。

## Admission-frozen member Prefill

The trusted member admission may include the exact optional outer field
`mediaLaunch.memberPrefill = {"protocol":"worldkit-member-prefill-v1"}`.
The original signed admission digest includes this field and its complete
`create.manual` launch bytes. The startup manifest contains the authorized
prompt, seed and ordered inline reference images. A browser cannot enable this
mode in `init`, change that manifest, or upload another asset.

This mode keeps the original SDK member input and retirement owners. Room loads
the existing `SourceInference` assets without automatic model startup and returns
`session.ready.member_prefill` with the protocol, prompt SHA256, seed and exact
`initial_references` descriptors. The member may prepare a context, prefill only
that complete frozen condition, and start it only after its matching
`prefill.memory_ready` from the same worker epoch. `render.start` must name the
same context and exact `source_frame_index`; `next_available` is rejected.
Checks repeat after asynchronous asset resolution and before admission.

For a shared World, the existing `private_view_snapshot` path pins a future real
member frame for 120 seconds. It never sends a World hold/resume or creates a
second simulation clock. Other members continue to run. Asset upload,
prompt/segment replacement, inference toggles, browser gameplay on the media
channel and source closure remain forbidden. The normal SDK input port owns
gameplay; the original member leave operation owns full cleanup. A model segment
may end without retiring its member. Worlds declaring their own model-condition
channel cannot also select this external frozen-condition mode.

Members without the marker retain their existing manual inference behavior and
control allowlist. Ordinary non-member Prefill is unchanged. This is a CPU
control-plane candidate, not a declaration of regional deployment or GPU/cache
acceptance. Validate the actual Room image, signed routing, live capability,
F0/cache identities, matching model pixels and source HUD, and member release
before enabling a regional player.

An operator may set the optional exact
`WHITEBOX_MEMBER_PREFILL_PLAYER_ORIGIN` on the existing Room and Gateway.
The ordinary `WHITEBOX_PLAYER_ORIGIN` remains unchanged. This second origin
can read or control only a member whose trusted admission carries the Prefill
marker, after the original viewer capability and owner checks. Gateway forwards
the real Origin to Room; it does not rewrite it to the ordinary allowed origin.
HTTP, WebSocket and CORS use the same two exact origins, never a wildcard.
The SDK's public member routes remain rooted at `/v1/sessions/`; deployment must
route that exact path prefix to the owning regional Gateway. A separate
`/whitebox/` signalling rewrite does not publish those SDK routes automatically.

本次是源码分支整合；两地已部署验收对应各自明确的历史 SHA，不能当成新合并提交的
GPU/浏览器验收。合并候选增加双版本首帧、控制归属与 Prefill 的 CPU 回归；
发布新镜像仍需按固定 SDK、模型入口、运行配置与实际部署逐项核验。
