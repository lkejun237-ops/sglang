# Cloud whitebox render segments

The opt-in `whitebox-render-v2` protocol keeps one authoritative cloud CLI source
alive while independently starting, draining, cancelling and replacing H3
render segments. Selecting display A or B never changes the model lease. This
implementation retains the single-active-user deployment constraint; v1 and v2
share the Room admission lock and capacity.

Optional signed, source-only RTC encoder selection and its default-preserving
admission/recovery rules are defined in [Session source encoding](whitebox-source-encoding.md).

Runtime Sink controls (default OFF) share this same connection; see
[Sink control API](whitebox-sink-control.md), including the optional original SDK
control-packet tunnel for admitted multiplayer members.

Server-owned periodic reuse of the current condition is documented in
[Condition refresh and resident GPU Prefix reuse](whitebox-condition-refresh.md).
It adds no player control endpoint.

## Ownership and controls

Connect to the existing generate WebSocket with `whitebox_control=v2`, then send
an `init` with `control_protocol=whitebox-render-v2`, `source_operation=create`
and the existing `whitebox`, prompt and reference fields. The source starts in A
with no model allocation. A later `source.bind` or reconnect uses the original
source session/epoch, game, owner epoch and resume capability; it does not join
another CLI view. Read-only operation recovery is available through
`/v1/whitebox/owners/{owner_id}/render/sources/{source_session_id}` using the resume
capability as a bearer token. A lost Room epoch returns lost/unknown state and
never reconstructs gameplay from a render journal.

Commands use `type=command`, a unique `request_id`, `source_session_id` and
`source_epoch`. Start/end/replace require `expected_render_revision`; display
requires its independent `expected_display_revision`. Segment operations also
carry `segment_id` and `segment_epoch`. Identical retries return the durable
operation result; changed content under the same ID is rejected. The Room PVC
stores a separate FULL-synchronous SQLite WAL control journal. Accepted replies
follow commits; frame transport does not write this journal per frame.

- `render.start`: full prompt, seed, immutable asset references and start boundary.
- `render.end`: `drain` or `cancel`; release confirmation is independent of video
  drain, operation completion, archive finalization and upload.
- `render.replace`: validate the new conditions first, persist replacement
  intent, release the old segment, then acquire a new generation. An ordinary end
  does not silently delete a pending replacement.
- `display.configure`: `mode=whitebox|model`, with no model allocation/reset.
- `prompt.replace`: full text, `expected_prompt_version`, next `prompt_version`,
  an effective global source frame or PTS, optional full `references`, and
  optional one-shot event IDs. Omitted references preserve the applied set;
  `[]` clears it. A supplied set uses the same immutable asset descriptors as
  `render.start`; partial patches and `null` are invalid.
- `source.unbind`: stop the model binding; retain the source/game.
- `source.close`: explicitly finish the source and its model child.

Model-frame delivery keeps its archive-pair age deadline in the render task.
Cancellation must survive an archive source AU becoming available in the same
event-loop turn; otherwise Python 3.11 can leave the render loop waiting for a
next frame during segment cleanup. The lifecycle regression in
`test_whitebox_render_lifecycle.py` exercises this ordering with a real render loop
and a controlled encoder. Manual inference disable still waits for confirmed
parent and child cleanup before reporting `disabled`; its World, input and UI
remain alive, and a later enable selects a fresh source frame. CPU coverage does
not establish deployed Room, GPU or visible-player behavior.

### World-owned model conditions and recovery

When `session.ready.world_driven` is true, the authenticated SDK source supplies
the complete prompt and ordered reference handles through
`worldkit-model-context-v1`. Browser `render.start`, `render.replace`,
`prompt.replace` and asset uploads remain rejected. Image preparation is pure;
only the subsequent fixed-boundary commit enters the original RenderJournal.

After EOS/cancel/failure and confirmed model resource release (`none` or
`released`), the original source lifecycle owner can explicitly request:

```json
{
  "type": "command",
  "op": "render.context.start",
  "request_id": "next-scene-segment",
  "source_session_id": "ORIGINAL_SOURCE_SESSION_ID",
  "source_epoch": "ORIGINAL_SOURCE_EPOCH",
  "expected_render_revision": 2
}
```

Use the latest observed render revision rather than copying the example value.
This lifecycle-only command accepts no prompt, references, seed or client-chosen
model identities. The original journal reserves fresh segment/session/generation
identities idempotently; the existing Host consumer settles the old candidate,
then supplies the current full condition for init at version 1. A different
request during an active segment, unknown release or pending handoff is rejected.
Events do not restart segments automatically. Desired World state and consumer
execution timers survive; consumed onset text is not replayed. Old segment
receipts remain queryable and cannot update the successor adapter.

Normal `render.end` with drain freezes new dispatch while an already dispatched
condition may still receive real source frames and settle. Its wait shares the
existing cleanup deadline; explicit cancel remains a separate termination path.
The same normal-end boundary also applies to browser-owned prompt updates. A
terminal segment with released resources is insufficient for `render.context.start`
when an original model-context execution is still unknown: it rejects with
`RENDER_TRANSITION_IN_PROGRESS`, retaining the original journal/query identity.
Read that original result before a later explicit start; do not turn this rejection
into a new-condition loop, report unknown as failed, or promise immediate restart.
See [Worldkit condition ownership and startup](/docs/sglang-diffusion/whitebox/specs/world-model-context#owners-and-startup)
for dispatch-lock, deadline and unknown-result details.

End/restart changes require acceptance against the exact deployed Room/CLI and
SDK/Player bundle identities. A successful run on an earlier source revision does
not validate a successor's drain timing, first frame or performance. Keep CPU
lifecycle checks, real model output, actual Player presentation and resource release
as separate evidence; source push does not deploy these changes.

The native CLI consumer reconnects control and pixels together within a bounded
15-second transport attempt, using the same authenticated descriptor and logical
client ID. It keeps the consumer epoch, adapter, control cursor and original
candidate. Subscription queries that candidate's original journal operation,
then requests the contiguous control prefix or existing checkpoint. Frames still
require their exact sample and prefix mapping; reconnect does not replay input,
UI actions or model commits, and does not create another World or model session.
An expired consumer/source identity is an explicit failure, never a new binding.

A missing sample waits at most 30 seconds, requests one transport recovery, and
waits for exact mapping again. A second missing-sample timeout suspends only that
model route, emits `context_recovery_required` with
`MODEL_CONTEXT_SAMPLE_RECOVERY_EXPIRED`, stops its model child, and switches the
same source transport to preview mode. Preview frames carry no fabricated model
context evidence. A later explicit lifecycle request reopens the model gate and
discards preview-only buffered packets before resuming model delivery.

CPU coverage lives in `test_whitebox_source_conditions.py`: actual SourceSession
release/new-segment/model-init with one lease, old-result fencing, lifecycle-only
idempotency, real authenticated local WebSocket loss after durable commit,
original request/cursor recovery and preview-only degradation. These checks use a
controlled CPU model and do not establish GPU/visual or full SDK-browser parity.

### Player text supplements for ordinary Worlds

Manual-inference ordinary sources advertise `model_context_instruction:true`
without claiming `worldkit-model-context-v1` or binding a Harness compiler.
After actual inference state is `enabled`, the existing
`model.context.intervention.get` and `model.context.intervention` commands accept
the exact source session/epoch and current segment/epoch. Writes additionally
require an explicit `expectedRevision` and a nonempty `prompt` or `null` to clear.

The original RenderJournal atomically checks the revision, retains the segment's
initial prompt, and queues the initial prompt plus the current supplement through
its existing prompt-update owner. Clear restores the initial prompt; neither
operation replaces references, allocates compute nor starts a segment. The
ordinary source uses its source/member lifetime fences; only Harness sources run
the bound context-engine check and original compiler. Raw browser `prompt.replace`
and render-start commands remain rejected for manual inference.

The member RenderPipeline admits these same two instruction commands for manual
sources without member prefill. Its member/generation fence remains in place;
ordinary instructions do not gain `render.context.start`, raw prompt or input
authority. The actual member-pipeline CPU regression checks read/set/clear and
that the other member and worker count remain unchanged.

Same-segment reconnect reads the retained supplement and revision; a new segment
starts empty. Identical request IDs return their original result, changed reuse
rejects, and conflicts preserve the current value. Reads do not grow the command
journal; writes share its existing history and pending-prompt limits. Acceptance
is distinct from worker applied and visible output. The ordinary source queues
its correlated instruction receipt before waking the model prompt owner. Worker
progress for the same request does not replace or retire that instruction receipt.
CPU regressions live in
`test_whitebox_inference_control.py`, including the actual `LocalRoomEngine`
consumer; real GPU and formal Player acceptance remain separate checks.

## Public source and private model resolution

For a new source, select `whitebox.resolution="480p"` (832×480) or `"720p"`
(1280×720) in the existing init. Omission preserves the configured host default;
there are no width×height string aliases. Invalid values return
`INVALID_RESOLUTION`, `stage=validation`. A Room pins its source `VideoSpec` when
the driver starts. Later joins and resume inherit that canvas; an override that
conflicts with the pinned canvas returns `RESOLUTION_UNSUPPORTED`. No request
changes `RoomRuntime.spec` or the shared host configuration.

Source `session.ready.source_stream`, CLI capture/descriptor/native frame proofs,
Canvas/UI coordinates, live H.264 and archived MP4 remain at the public dimensions.
720p is real 1280×720. The Whitebox model connector alone uses
`whitebox-model-geometry-v1`: it replicates the last source row 16 times to create
1280×736, sends that private `size` and `source_stream`, validates private frame
and credit byte counts, and crops the bottom 16 rows from each decoded frame
before composition, playback or archive. No image resize, frame duplication,
new clock, new codec, or GPU worker adaptation is involved. Aligned 480p pixels
pass through unchanged. One private 720p model frame is 2,826,240 bytes; one public
source/output frame is 2,764,800 bytes. Private clip17 capacity is 48,050,176 bytes
including the existing 4,096-byte transport allowance. Pair reservations use the
larger private canvas rather than undercharging padded copies.

Startup v1 hashes and sends the padded model S0 with private source geometry, so
the backend's prepared-frame/first-append check still matches. Startup v2 keeps
the independent condition preview at its native dimensions, while its fingerprint
also binds the private source geometry. The pinned CLI frame remains public in
both versions. Startup receipts retain public `effective_rgb_sha256` and
`cli_frame_rgb_sha256`, and separately report `model_condition_rgb_sha256`,
`model_cli_frame_rgb_sha256` and the geometry policy. Segment recordings include
`model_geometry` with source/model dimensions, replicate padding and output crop.
The original model profile fingerprint and source/session/epoch identities remain
unchanged in ownership and meaning.

Interactive WS and render-v2 WS/native RTC share this connector. The classic
Whitebox generate path rejects 720p before coordinator admission or CLI startup.
Only that classic route preflights bounded init bytes (30 seconds, at most 17
messages and 8 MiB + 4 KiB total, including optional pre-init client metrics), then
replays each original message exactly once. Other model and authorized world
routes retain their existing handshake and authentication paths.

Native RTC currently negotiates H.264 Baseline Level 3.1. Its macroblock budget
supports 832×480 through 60 FPS and 1280×720 through 30 FPS. The existing 480p
encoder keeps its 60 FPS hint. For a larger canvas, live source, paired archive
and model encoders use the source's confirmed `engine.capture_fps`; missing
metadata falls back to the explicit CLI request, `WHITEBOX_CLI_FPS`, then model
FPS. This changes the real SPS chosen by libx264, including after bitrate
rollover; it does not rewrite codec strings, change capture timestamps, or drop
frames. The public model FPS remains 24.

720p RTC at an effective CLI rate above 30 returns `RTC_CODEC_UNSUPPORTED`,
`stage=validation`, before any startup reservation, CLI/model allocation or
existing attachment replacement. This release does not negotiate Level 3.2;
WebSocket supports the selected CLI rate through 60 FPS and advertises the real
encoder SPS. Default system Chrome 156's two-video receive offer was checked
locally: it advertised H.264 Level 3.1, so raising the sender to 3.2 would not
make that browser compatible. An actual browser/media session remains a
separate deployment acceptance check.

CPU coverage lives in `test_whitebox_model_geometry.py` and
`test_whitebox_resolution.py`, registered in `run_area_tests.py`. It checks exact
rows and packed batches, native 480p/720p H.264 and MP4 dimensions, private model
init/append/credit, v1/v2 prepare identities, different simultaneous Rooms and
cleanup, plus classic rejection before any model or CLI allocation. It does not
prove deployed CLI/browser pixels or GPU quality, throughput or memory. The
[implementation plan](../plans/whitebox-true-720.md) tracks WTP consumption and
online acceptance separately.

## Trusted single-player launch reconciliation

The service-only `POST /v1/whitebox/single-player-control` (also mounted under
`/backends/minwm`) adds launch recovery to the original RenderHost, not to the
multiplayer Room driver. WTP remains the durable business-operation owner. The
Gateway uses the existing directory's immutable caller/operation association;
the original RenderHost SQLiteRenderStore fences creation and retains the
original source/session/owner identities. No additional runtime or scheduler is
created. Normal generate/init/resume clients retain their existing behavior.

Failed single-player results may include `failureCode` with the reviewed values
`CAPACITY_EXHAUSTED` or `PLAYER_IDLE_TIMEOUT`. The original journal retains the
reason across queries; raw exception text and other internal codes remain private.
The latter also identifies an inactivity-ended source when reading its attachment.
The native source close path persists this terminal reason after its original
cleanup completes; registry removal during that write remains a pending outcome.
Neither reason proves cleanup or authorizes a replacement source. Deploy the WTP
reader and Player error catalog before enabling these optional result fields.

Configure `WHITEBOX_SINGLE_CONTROL_SECRET` (at least 32 UTF-8 bytes) and
`WHITEBOX_SINGLE_CONTROL_SERVICES` on the Gateway. Missing configuration denies
service access with 503. The existing signed-byte service-ticket codec uses a
separate `aud=whitebox-single-control`, `scope=single-player-control`, version 3,
allowlisted `service_id`, original `sub`, action, exact HTTP
`request_sha256`, one-use 32-hex `jti` and integer `iat/exp` (maximum 120 seconds).
The explicit signed `origin` must equal the request's Origin header; use JSON
null when a backend request has no Origin. Every retry gets a fresh ticket but
keeps its original operation/hash. The ticket's service/subject must equal the
request caller. The original internal token authorizes the Gateway-to-owner hop
at `/internal/single-player-control`; browsers never receive that token.

The request shape is:

```ts
{
  schemaVersion: 1,
  action: 'start' | 'getOperation' | 'getAttachment' | 'close',
  singlePlayerOperation: {
    operationId: string, parametersSha256: string,
    caller: { serviceId: string, subject: string },
    frozenPublication: FrozenPublication,
    frozenLaunchParameters: {
      transport: 'websocket' | 'webrtc-rtp',
      publicationIdentity: {sourceId: string, worldId: string, publicationId: string,
        sourceManifestSha256: string, renditionInputSha256: string, manifestSha256: string,
        attempt?: number},
      initMsgpackBase64: string
    },
    controlSequence?: number
  }
}
```

`FrozenPublication` must have verified/world-frozen-publication status/scope and
its original `world.sourceId/worldId/publicationId`. When playConfiguration is
present, its SDK runtime `playMode` must be `single-player`. The public Player
API target label `single` is not that runtime value and is rejected here.
A genuinely absent playConfiguration is accepted for a verified historical
single publication, after the same existing signed caller, scope, world and
publication/init checks. WTP's verified-delivery owner rejects omitted metadata
for multiplayer; explicit null, unknown modes and multiplayer are rejected by
SG. The owner never inserts a default, rewrites frozen claims, resolves today's
SDK metadata, or changes the retained operation fingerprint. WTP derives and freezes the **original**
init after its existing authorization. `initMsgpackBase64` is standard Base64 of
that binary-safe MsgPack map, preserving startup frames/references and all
original prompt/seed/startup settings. It must request `type=init`,
`control_protocol=whitebox-render-v2`, `source_operation=create` and matching
`whitebox.source_id/world_id`, with no game ID; the directory supplies that ID.
`publicationIdentity` comes from WTP's original verified PublishedIdentity, not
from browser input or the init builder. All six strings are required; hashes are
64 lowercase hexadecimal characters. Optional attempt is a safe nonnegative
integer and must match the original frozen world when it declares an attempt.
Overlapping identity fields in FrozenPublication must agree. The owner passes
this expected identity separately as `expected_publication` through the original
SourceSession, LocalRoomFactory, RoomRuntime and `cli_room` path; it is not an init
option the browser can overwrite.

Before SourceSession sends any init or starts PlayStartup/model conditioning,
RenderPipeline invokes the existing publication helper in read-only frozen mode.
It requests the fixed publication/source-manifest/rendition/manifest selectors
from the original authenticated playable locator and compares its actual
location. Missing, changed or unavailable metadata rejects the original source;
this mode never requests reconciliation. The helper returns the actual checked
identity, which the Python owner compares again. Ordinary preparation behavior
for unmanaged sources is unchanged.

The original CLI keeps its `--wtp` and configured CDN/OSS transport. Controlled
sources use synchronous cache selection so a later background update cannot
replace the frozen body. The final real CLI descriptor's artifact must contain
all six expected identity fields plus matching artifact/top-level worldBuildHash;
missing fields also fail. `cli_room` performs this fence before returning the
CLI or loading its Python UI package; RoomRuntime checks its actual configured driver's
descriptor before admitting a view. A changed descriptor closes that original
CLI/driver and source before failure is retained. Managed model startup waits
until this final descriptor fence passes, using the original model pipeline
thereafter. No new CLI flag, alternate driver, archive fallback or operation is
invented. Existing CLI versions that cannot prove this identity fail closed.
Same-world publication replacement that cannot reopen the frozen body therefore
fails explicitly rather than silently selecting the replacement. Native/model
startup timing with this managed ordering has not been measured.

This trusted parameter envelope is not a browser options API. Decoded init is
limited to 8 MiB; HTTP requests to 12 MiB. Existing RTC/startup feature flags and
one-active-model capacity remain authoritative. `run_service.py` already passes
the role's environment unchanged; no separate launcher or model path is added.

The async `SinglePlayerControl.start/get_operation/get_attachment/close`
methods always echo `operationId`, `parametersSha256` and `state`. States are
unknown, starting, ready, cancelling, cancelled, failed, lost and expired. A
reserved source additionally returns `execution={gameId, sourceSessionId,
sourceEpoch, ownerId, ownerEpoch, publicationId}`. Ready means the original
source emitted ready, not that model pixels, ICE or presentation were verified.
A starting source can expose its private attachment once its identity exists,
so the original RTC startup can complete its own readiness gates. While frozen
preflight or the original init has not yet completed, starting has no attachment;
poll the same operation. This avoids finishing an early RTC handshake before
init creates its original startup readiness events.

Reservation commits precede original factory/startup work. Lost replies and
request cancellation leave that owner task available for same-operation query.
Changed hash, publication or original launch bytes conflict, even when the
caller repeats an unchanged WTP hash. Query and attachment never allocate a
source. Close before start stores a cancellation tombstone; a late init observes
cancellation and releases only its original pipeline. During init, close returns
cancelling until the original source close completes. Concurrent closes share
one shielded teardown task per original operation; a second request remains
cancelling while that task is active. Only an actually closed original pipeline
can persist cancelled, and no global lock is held through teardown. Cancelled does not invent
a model-release receipt: resource-release certainty remains in the original
segment journal. It never closes a multiplayer room.
Owner restart or missing source returns lost/unknown/expired and never rebuilds
an execution from its journal. Upstream network uncertainty returns 200 unknown;
query the same operation before deciding what happened.

The directory retains a small digest/route association without expiry in its
existing store. After ordinary game-route expiry it returns the same original
owner/game tombstone, not another allocation. This retention intentionally fails
closed rather than recycling an old operation. SQLite query retention is seven
days; expired operation rows remain non-reusable tombstones. Recovery requires
the original retained SQLite PVC and persistent production directory. Explicit
storage reset/migration is an operator action, not an automatic retry strategy.

`getAttachment` returns `{attachment:{expiresAt,transport,message}}` only for the
original live source. The 60-second message capability binds user, source/owner
epochs, game, connection generation and replay cursor. Send that message unchanged
as the first message on the existing media/signalling route; its type is resume
or source.bind. Obtain another short capability after expiry or a successful
connection-generation change. Managed sources do not expose the long-lived
owner resume token in attached/ready events. The original resume/bind, RTC
reservation, source budget, replay-history and generation checks still apply.
`controlSequence` optionally selects the client's last received control cursor;
its first-attachment default is zero so original startup controls replay before
ready; later attachments default to the current sequence plus the ready snapshot.
WebSocket media retains its existing matching authenticated user requirement;
RTC relay carries `resume_user_id` and the owner verifies the signed attachment.
A member attachment signs only `source_id`, `world_id`, `game_id`, and the
optional `scene_rendering` in its `whitebox` object. CLI FPS and other startup
settings stay on the bound source and are applied during initial startup.
Distinct admitted members may negotiate RTC concurrently. The existing
per-source reconnect reservation rejects concurrent attempts for the same
member. New source startup continues to use the Room-wide reservation.



Malformed/duplicate-key requests return 400, oversized requests 413, denied
service tickets 403 and changed/unavailable operations 409 with English public
details. Responses use Cache-Control: no-store. WTP owns business error mapping,
SessionDescriptor assembly, original-init derivation and consumer activation.
The focused CPU gate is `test_whitebox_player_control.py`, alongside startup,
Room pipeline, RTC lifecycle, Room control and directory regression tests in the
existing Whitebox Docker checks stage. Mock peers and a real original
RenderPipeline/SourceSession with a CPU driver verify control ownership; they do
not verify GPU generation, native ICE/playback or Shanghai performance. Protected
private deployed patches are a separate baseline; never deploy this source
branch by replacing an entire live runtime directory.

Canonical and historical single producer regressions retain byte-identical WTP
captures in `fixtures/player-single-wtp-{canonical,legacy}-v1.json`. The original
service verifier checks their exact raw bodies and fixture-only signatures;
original RenderHost/SourceSession/RoomRuntime then preserve the frozen operation
and attachment execution tuple. Locator/CLI/model peers and prior SDK verification
records are synthetic. These CPU tests do not establish native sealing, live world,
model pixels, GPU or ICE acceptance. The producer scope contains either the real
SDK `single-player` value or no configuration key; no test relabels it to `single`.

## Explicit area sources and visible handoff

An AREA source must request `whitebox.loading=areas` with an optional full
`initial_groups` selection. It requires a verified v2 publication and the actual
CLI descriptor's `area.protocol=worldkit-area-presented-v1`. Archive delivery,
asynchronous complete-cache updates and the v1 pipeline remain unsupported for
this source. Partial area files never become a complete game/cache record.
The formal SDK pin and compatible frozen author toolkit remain separate release
prerequisites; a browser selector does not upgrade either one.

The existing Room attachment and ordered command journal own the permission.
`area.context` reports a pending or current interactive native World/controller
operation, runtime and accepted display/condition authority. Current interactive
identity may bind an immutable model intent, while presentation offers still
require pending held input. The context carries no frame or presentation
verdict. `area.offer` additionally binds an actual accepted binary source frame
to its producer sample and current output. Its challenge belongs to the current
source/connection generation. A model offer uses an opaque `model.mapping_id`
retained in the same bounded reliable control history. The mapping comes from
the actual model output event, segment epoch, applied prompt version/hash,
reference-set hash, seed and effective source boundary. Accepted-but-unapplied
conditions and local display choices cannot authorize it.

`render.start`, `render.replace` and `prompt.replace` for an AREA source include
`area_binding={protocol: worldkit-area-condition-v1, identity: ...}` from that
pending native operation. Room validates it before accepting the immutable
condition. Display authority remains the existing `display.configure` CAS and
its accepted revision. Complete sources preserve the absence of this field.

Only an explicitly negotiated WebSocket or native `webrtc-rtp` capability can submit the existing
`area.presented` command with `{challenge, receipt, display, model?}`. The receipt
contains the exact unchanged native `frame`, full `identity`, `uiRevision` and
`uiStatus=synchronized`. `display` is the accepted mode/revision; `model` is the
exact offered opaque reference and is omitted for whitebox output. The page
must compose those pixels and UI, wait for outer visibility, revalidate the
actual inner image/font/layout readiness, then check current authority again.
Retained/restored/hidden frames, unsupported transports and stale operation,
runtime/World epoch, selection, attachment, condition, segment or display
revision cannot release the native input hold. The SDK nonce stays inside its
native control owner.

Room journals the original request before native dispatch. Post-send unknown
results stay reconciling; reconnect queries that original `operation.get`
without sending another confirmation. Restarted/lost owners remain unknown.
The source, controller, clock, cache, player and model owner are unchanged.
Local CPU contracts and mocked-model native fixtures do not prove deployed
model quality, GPU performance or real-game acceptance. Current implementation
receipts and remaining native acceptance retain their exact release identities.
Local AREA RTP contracts do not establish production deployment or performance.

### AREA native RTP frame mapping

RTC attachments initialize the same AREA offer resolver and an independent
per-attachment challenge. When the actual CLI descriptor supports AREA, ready
capabilities advertise only this attachment's transport. AREA context, model
condition binding and native dispatch remain with the existing source owner.

An RTC frame carries the exact `area_offer` already retained in reliable control
history as part of its native RTP-mapped metadata. The producer envelope is copied,
not mutated. Newer control offers can overtake video; a client must match the
actual RTP timestamp's original stream, media epoch, producer frame index and
source PTS before using the attached offer. RTP playout retiming does not replace
source PTS. Missing, stale or unknown pixel/UI mappings never authorize input.
All existing media/control byte, age and queue bounds remain unchanged.

The WTP pre-start `nativeRtc.areaPresented` declaration must be explicitly enabled
only for a validated matching Room deployment; generic RTC availability is not
AREA support. Actual session capabilities are checked again. Capability mismatch
must not start a second source or silently migrate an active session to WS.
This source change requires paired Room/Web validation before live use. No new
SDK/runtime, model allocation or protocol performance claim follows from local
contract tests.

### Maintained CPU and native consumers

The independent `area-contracts` Zing CPU job executes the maintained
AREA/CLI/journal/model-lifecycle/Room-pipeline/RTP-attachment test owners twice through the
existing child-process/JUnit receipt validator. It uses the existing pinned
Linux CPU recipe and is mandatory in the aggregate result. Historical CPU
approval manifests remain unchanged; registration alone is not execution proof.
`run_area_tests.py` records the actual source provenance and fails on a failed,
empty, skipped or missing required owner receipt.

Local cross-repository native acceptance additionally uses WTP's explicit
`test:area-author-room` gate with exact clean SDK inputs and the assigned
SGLang source. It executes the real author compiler, publisher, standalone and
resident CLI, Room and React UI; only storage/cloud transport and the model
worker are fixtures. It covers five return trips, optional groups, hidden
composition, actual model-output conditions and a dropped completed reply that
reconnects and queries the original operation without another confirmation.
The existing worker-boundary test also remains required in the full CPU recipe;
the lightweight macOS interpreter does not provide its Torch dependency.
These receipts establish local ownership/control semantics, not source CI,
formal pin acceptance, deployment, model quality or GPU performance.

## Conditions and media evidence

The Room accepts at most 16 pending prompts/1 MiB, preserving every accepted
version. Only one handoff is outstanding at the Worker. Version 1 must produce a
nonempty output before an update is handed over. The Worker reserves a native
model-block boundary beyond all admitted source work, including Encoder
lookahead. SourceReadySession activates the new condition only at that boundary;
it does not reset visual history, RNG or the KV timeline. Joint lookahead cannot
cross a pending condition boundary using the previous condition.

`strict` remains validating until the Worker reserves the exact legal boundary.
`at_or_after` reports the actual boundary from decoded model output provenance.
Accepted and applied versions are distinct. A known rejected condition fails its
queued descendants and requires explicit rebase to the last applied version;
lost replies remain unknown until reconciled. Prompt changes do not introduce
audio generation.

### Prompt/reference switching with source skipping

Workers that advertise `source_condition_switch_protocol=clip-barrier-v1` and
`prompt_reference_updates=true` accept a full text/reference change in the same
segment. Use `effective.policy=at_or_after` and
`source_policy=skip_during_switch`; strict boundaries are rejected for this
protocol. Configure `--zing-reference-max-images 21` explicitly on H3. The
Room accepts up to 21 images, 8 MiB per uploaded image and 64 MiB per reference
set; the Worker's negotiated limits and actual KV capacity also apply. A count
limit is not a promise that every image resolution fits the GPU pool.

The Room confirms new reference files in the recorder before handing the
candidate to H3. Text-only changes retain the existing image latents and archive
paths. The Worker reserves a complete 17-frame source clip boundary B. The Room
may finish that old clip, then admits no local source index at or beyond B until
the Worker returns `kv_ready` after candidate preparation, five-latent history
rebuilding and rank agreement. This control dispatch does not need future RGB.

The world, input handling, source preview and source archive continue during
the pause. On readiness, the Room takes a newly captured frame, retaining its
real source index/PTS while the model's contiguous local index resumes at B.
It never catches up by replaying captures from the pause. The segment and media
stream remain unchanged. Model output provenance, rather than the ready ACK,
marks the combined version applied and selects that version's reference files
for recording. The manifest's `condition_switches` records B, the previous and
resumed source identities, skipped captures and host-side switch duration.

One source retains at most 192 MiB/512 uploaded assets, including historical
assets; the client reuses uploads with the same content hash. A timeout or an
unconfirmed destructive rebuild ends the model segment with failure while the
source remains available. This implementation does not query/replay an unknown
Worker switch. CPU protocol tests do not establish GPU timing or image quality.

Every segment has fresh model session/generation, segment epoch and H.264 stream
identity. Model ACKs restart at -1; old stream ACKs and packets cannot advance or
display in a new stream. Whitebox counters and source capture timestamps
continue. The browser resets only the model decoder/playout when a segment
changes. The browser uses adaptive smoothing and retains a 120-frame compressed
queue as a hard memory limit. Control and state messages have priority but share a WebSocket with
media, so TCP head-of-line blocking remains possible.

Room delivery enforces both the negotiated frame window and a 256 KiB encoded
byte window independently for each media stream, plus a shared 512 KiB window.
Cumulative playback ACKs free both windows; duplicate or old-stream ACKs cannot
release new media. One access unit larger than the per-stream window may be sent
only when the entire attachment has no outstanding media, with a hard 1 MiB
serialized-AU limit. Larger units are discarded with their dependent queue and
request a fresh keyframe. Segment changes
clear the model byte window without changing source credit.

Unsent media expires after one second. Expiration discards the dependent queued
access units and requests a fresh keyframe; it does not skip arbitrary H.264
reference frames. Reliable state/input evidence and archived frames retain their
existing lifecycle. `playback_delivery.limits.inflight_bytes_per_stream` and each
stream's `max_inflight_bytes` expose the encoded-media budget and its high water
mark (including the single-access-unit exception). They do not measure TCP or
proxy buffers. These bounds limit media bursts and stale recovery; they cannot
guarantee target FPS or a latency ceiling on an undersized network connection.

The report also exposes current in-flight bytes/frames, oldest unacknowledged
age, oldest queued age, maximum observed ACK age, and byte-credit wait counts.
These are Room-clock observations; they are not wire-level retransmission
measurements. The one-second expiry applies only to media not yet sent.

The browser can proactively resume the same source over a fresh WebSocket when
clock-error-adjusted media age exceeds 1500 ms for at least 400 ms and three
samples, or both media and a pending probe have stalled for 3000 ms. A 10-second
cooldown prevents repeated active refreshes. Startup, hidden pages, finishing
and final decoder drain are excluded. Socket identity fences queued callbacks;
the existing owner/source/generation/capability and reliable-control cursor
protect resume. Old live input is discarded and the resumed input state is
neutral; render operations replay their original idempotency IDs. This does
not create a new world, restart a model segment, or truncate an archive.

The live player first repays short-lived debt through bounded rate adjustment,
subject to UI paint capacity. If locally queued media remains over one second
old when newer media arrives, it recovers from the newest queued IDR chain,
counting skipped display frames. Raw compressed delta frames are never fed
across a missing reference. EOS drain and old decoder callbacks are generation
fenced, including delayed flush errors. Transport refreshes are counted in the
performance panel. These mechanisms do not guarantee service on a link whose
sustained capacity is below the complete stream's demand.

Source preview and source-only archive run without a model. Each segment has a
separate paired archive using actual selected source frames, prompt text/hash,
source frame/PTS and segment identity. Upload/finalization does not gate the next
model lease. Archive failures retain incomplete status; this feature does not
automatically promote records into training data.

CLI UI history remains ordered and independent of preview replacement. Encoded
and native CLI frames carry at most 512 original UI events each, matching
[`BrowserUiTimeline`'s public batch budget](https://github.com/seedleap/worldkit-monorepo/blob/dev/packages/stream-player/src/browser-ui-config.ts).
A slow startup retains the remaining events for subsequent real frames under the
existing 4096-event / 8 MiB history limit; byte accounting uses canonical UTF-8 JSON
for both admission and retained events. `whitebox.states` publishes every batch
reliably before preview replacement. Frame headers, UI revisions and event payloads
are unchanged; native source reset replaces retired UI history with the new reset
event. Missing historical UI mapping still requires pixels alone, never a fabricated
synchronized-UI receipt.

### Project audio on the source clock

A complete SDK session may advertise sealed project audio in its authenticated
`SessionDescriptor.audio`. Room reads only that session's fixed
`/v1/sessions/{sessionId}/audio/` route with its existing capability, rejects
redirects, validates the manifest SHA-256, and admits only sealed
`assets/audio/` entries with exact byte lengths and hashes. It does not fetch
arbitrary source URLs or accept browser-provided paths. Bounds are 256 KiB per
manifest, 64 resources, 32 MiB encoded audio and 128 MiB declared decoded audio;
the existing combined recording/UI asset budget also applies. This one-time
resource prefetch precedes viewer readiness, has a 30-second total deadline and
adds startup latency, recorded as `engine.audio_resource_prefetch` (elapsed
milliseconds, encoded bytes and resource count). No-audio sessions skip it.

The ready message adds optional `project_audio: {manifestSha256}`. The same
recording report capability protects
`/v1/whitebox/recordings/{recording_id}/audio/project-audio.json` and declared
`audio/assets/audio/...` resources, including the existing owner-qualified proxy
route. Sealed bytes reuse the recording archive's asset ownership and upload
lifecycle under reserved internal `ui_bundle/__project_audio/` names. Missing,
changed or unlisted resources fail closed; no live-source fallback is used after
archive closure.

Original `audio.snapshot` messages become `states.audio_snapshot`; source session,
epoch, presentation identity, SDK epoch, history interval, event times and voices
are preserved. A bounded 128-snapshot / 8 MiB history selects exact source-frame
proof synchronously; missing or late proof leaves that presentation silent and
never waits for audio on the video/inference source path. Optional listener position/forward/up vectors describe the SDK's
single active camera listener. The adapter never rebases audio to generated media
PTS. Source reset clears retired audio evidence; media-generation replacement
alone does not. Audio-bearing source states bypass sparse UI replay compaction because each
accepted presentation needs its exact historical audio frame proof. Project audio remains independent of model inference and
reuses the existing single-player world owner.

### Best-effort, diskless Room archives

Room uses `StreamingArchiveManager` for new recordings. Source creation returns
without waiting for an archive child to boot. Frame offers only enqueue bounded
snapshots of existing H.264 access units and metadata. The child remuxes those
same bytes to fragmented MP4 and uploads from memory through two background
threads. No second video encoder is used, so live resolution, bitrate and pixels
are unchanged. Input/state JSONL, reference images, UI assets, report capability
hashes and optional browser-recording verification also avoid new archive files.

The game-level `RoomLog` is part of this archive boundary too. Ordered join,
input and leave receipts use at most 8 MiB of RAM, with a 64 KiB per-event limit.
It uploads its small JSONL and game manifest asynchronously after the room ends.
Overflow drops that archive buffer and reports failure while receipt sequencing,
input delivery and the next room continue. It does not start the old disk journal
process. This receipt archive is distinct from the durable SQLite control journal.

The source, model lease and live transport never wait for OSS or archive
completion. A missing store, full/expired queue, upload failure or child timeout
fails that archive and releases its buffers and worker. It does not stop
gameplay, block the next model segment or fall back to a disk spool. A failed
archive is not advertised as complete. If OSS itself is unavailable, its failure
manifest is also best effort; RAM status can be lost on Room restart.

Defaults are explicit per recording:

| Resource | Bound |
| --- | --- |
| Parent queue | 256 operations / 32 MiB / oldest age 5 seconds |
| Multipart part | 8 MiB |
| Queued/running upload payload | 32 MiB / 16 jobs / 2 threads |
| Object accumulation buffers | 96 MiB total |
| Total archive bytes | 2 GiB |
| Archive process | 2 GiB Linux address-space limit, background CPU priority |
| Boot / completion | 15 / 120 seconds |

`WHITEBOX_STREAM_ARCHIVE_QUEUE_ITEMS` and
`WHITEBOX_STREAM_ARCHIVE_QUEUE_BYTES` configure the parent queue. Existing
disk-recording queue variables do not control this Room path. The recorder slot
limit remains two, or three with render segments enabled. These are per-process
and per-recording bounds, not a promise that the whole Room RSS equals one queue.
Reference images retain the source's 64 MiB/128-item bound; live UI assets have a
32 MiB/128-item memory cache. Optional browser-composed video verification runs
in a separate bounded child with 256 MiB candidate and 512 MiB model limits.

The muxer uses `delay_moov+default_base_moof+frag_keyframe`, microsecond clocks,
edit lists and disabled negative-timestamp shifting to retain nonzero absolute
source PTS. Multipart parts carry MD5; each completed object has a SHA-256 entry.
An ambiguous completion response is reconciled through exact key, length and
generation nonce, without creating another upload. The capability index lives
under `<object_prefix>/.archive-index/v1/<recording_id>.json`, outside each
recording's data prefix. The final manifest is published after its objects and
index are confirmed. `upload_status=complete` means the bytes were uploaded;
`recording_status=partial` still identifies an incomplete/aborted timeline.

Reports and replay inputs read active state from memory and finished state from
private OSS objects. Existing local recordings remain readable. Live UI serving
does not depend on successful archiving. `recording_queue` distinguishes
`produced_bytes`, `confirmed_upload_bytes` (including accepted multipart parts),
`upload_buffer_bytes`, `object_buffer_bytes` and `last_processed_frame`.
Only a committed manifest makes frames successfully archived; accepted parts
alone do not constitute a playable object.

Local mandatory UI asset reads share the existing two-worker report I/O bound.
A normal browser may request all four required bundle files concurrently; these
reads wait for a worker instead of failing merely because two checks overlap.
There are at most 128 waiting UI operations. Each `run_io` keeps its original
three-second deadline **including slot wait and execution**. One HTTP request may
perform authorization, manifest and path validation as three separate operations;
this is not a three-second whole-request deadline. Cancellation releases its
waiting position; a timed-out disk thread retains its worker until it actually
finishes. Ordinary diagnostic report reads still fail fast when both workers are
busy. Original capability, declared-path and verified-loader hash checks remain
unchanged. The maintained AREA gate includes the concurrent UI resource cases.

Known incomplete uploads are aborted best effort. A killed process or lost
initiation reply can leave parts whose upload ID is unknown; the deployment
must provide a recording-prefix lifecycle rule for incomplete multipart uploads.
Check Create/Upload/Complete/Abort/Head/Get/Put permissions and report-index access
against the actual OSS bucket before rollout. No broad bucket cleanup is implied.
The CLI world-resource cache and SQLite control journal retain their existing
storage contract; they are separate from recording media and are not removed.

Local tests exercise real H.264 pixels/absolute PTS, real SDK and spawned children
against a loopback object store, multipart transfer before EOS, slow/failing
uploads, cleanup/next-session admission, report/replay recovery after RAM eviction,
optional UI video verification and four model segments with unavailable archives.
They do not establish real OSS permissions, production throughput or crash
cleanup policy.

### Immediate capture-clock H.264 output

`WHITEBOX_H264_IMMEDIATE=1` selects the Room model-output encoder's native x264 bridge. Source preview and paired-source encoding remain unchanged. The default is
off for comparison and rollback. It requires `WHITEBOX_CAPTURE_CLOCK=1`; a
missing or incompatible library fails server startup before a source is admitted.
The Room image compiles this bridge through x264's public API, statically links
the codec with its symbols hidden, and records its package version. It does not
patch x264 or access private encoder structures.

The existing x264 VFR path retains one frame to discover its duration, even with
B-frames and lookahead disabled. The immediate path uses a fixed internal codec
30 Hz tick, while each picture's VBV replenishment represents source capture
time. When the next selected source timestamp is already known, it supplies the
exact interval without waiting for the next model RGB. Otherwise the previous
source interval is used (including EOS). Both the configured bits-per-second limit and 250 ms bucket capacity are
preserved. A long source gap cannot grant more than one bucket of credit. This
avoids both the artificial bitrate reduction from a fixed 60 FPS budget and an
oversized budget when output exceeds a fixed 24 FPS encoder assumption.

Each successful `push()` returns exactly its current frame's access unit.
Prediction, GOP/keyframe requests, CRF 18, baseline profile and YUV420P remain;
the mode neither inserts synthetic frames nor drops model outputs. Codec PTS
identity is checked on every call. The protocol continues to carry the original
microsecond source PTS and nominal model clock unchanged.

Live `duration_us` uses the known source interval, or a previous-interval
estimate, explicitly marked `duration_provisional`. `duration_hint_exact`
identifies the known-interval case. It does not drive presentation ahead of the actual
PTS. Only the independent archive muxer retains one compressed access unit until
the next PTS fixes its exact duration. At EOS it writes the last access unit once
using the last observed interval, matching the existing tail convention. Video
payload bytes are never re-encoded for archival.

The stream reports `encoder_retained_frames=0`,
`implementation=room-immediate-au-v1`, `rate_control_clock=elapsed_source_capture`
and the native encoder version. These identify a codec mode, not a claim of
end-to-end latency improvement. Validate source-clock byte envelopes, decoded
quality and PTS/durations before comparing identical scripted browser sessions.

## Optional model RTP presentation cadence

`WHITEBOX_RTC_MODEL_CADENCE=1` opts one Room deployment into a model-only RTP
presentation clock. The default is `0`; WS, source video, encoded picture
identity, archive timestamps and reliable UI mappings retain their source PTS.
The native sender receives a separate timestamp while `rtc.frame.meta` still
carries the original source identity and PTS.

Latest-source admission creates consecutive captures separated by selection
holes. Once a hole is detected, the clock smooths the most recent 17 source
intervals. Every interval contributes, including partial groups; it does not
integrate a sparsely sampled FPS estimate. Regular continuous source cadence
is unchanged. Source gaps over two seconds reset the filter and retain the
discontinuity. A one-second phase boundary limits exceptional cadence changes;
this is a timestamp boundary, not a playback buffer or an added wait.
New model epochs and RTC attachments start with independent clocks. There is
no added timer, pixel queue, synthetic frame or application playback ACK.

The opted-in model AU handoff holds at most 17 frames rather than eight, enough
for one input clip's burst. Its existing 200 ms enqueue-age and 1 MiB byte
limits still apply; expiry and capacity recovery retain only a complete IDR
chain. The source track keeps its original eight-frame budget. Benchmark
model production separately from browser decoded/presented frames; this switch
does not increase GPU throughput or guarantee a browser FPS.

## Limits and release proof

The binding allows 300 seconds of effective play, 600 seconds of cumulative
startup/wait time, 120 seconds to a segment's first decodable frame, and 1800
seconds absolute lifetime. A time-budget expiry unbinds rather than finishing
the game. The source owner has a separate 60-second heartbeat lease. A viewer
disconnect gives 15 seconds to reconnect before cancelling its model child.

Model cleanup uses Coordinator `release-verified`. Only an epoch-matched Worker
204, or a stored release receipt for the exact assignment, frees the lease.
Owner mismatch, missing endpoints and timeout are not release proof. Unknown
cleanup quarantines the slot. Worker cleanup failure retains the consumed owner
and advertises failed lifecycle. Heartbeat occupancy prevents lease expiry alone
from making an occupied slot available.

Closing a browser tab does not require an explicit Exit action for cleanup.
The Room retains the original source for its configured reconnect grace period
(15 seconds by default), then closes that source and its model binding. A source
close that reports `RELEASE_UNKNOWN` for an admitted player with the strict
SDK/worker release contract automatically retries the same SDK owner and model
assignment twice, after 1 and 3 seconds. Concurrent close callers share this
cleanup flight; cancelling a caller does not cancel it. These delays are in
addition to each owner's existing bounded cleanup waits. A confirmed release
still requires the original SDK terminal receipt and worker release; retries do
not admit a replacement session or free a slot based on lease expiry. Persistent
uncertainty remains quarantined after the bounded retries and requires operator
reconciliation. The original close/unbind operation remains pending during retry
and reaches its terminal result after recovery or the final failure. Generic
engine/renderer failures without that player release contract are not retried.
Restarting the Room cannot reconstruct a lost owner receipt.
If cancellation races with admission, cleanup first drains the original admission
reply. Retry eligibility then uses that same binding's acquired or cleanup
assignment; it does not depend on admission having finished before close began.

For the frozen SDK's `stream` service route, `SERVICE_QUEUE_TIMEOUT` means the
request was rejected before dispatch or withdrawn from the queue. Its structured
CLI failure plus completed client cleanup can establish `not_created` through the
original factory and Room owners. This does not apply to arbitrary Room commands,
`STREAM_STARTUP_TIMEOUT`, service disconnects, or an unknown CLI exit after
dispatch.

Resident CLI cancellation additionally uses the native
`worldkit-service-cleanup-v1` receipt. The CLI first identifies its private stream
request, then cancels that exact request over the existing authenticated service
connection without closing the reply channel. The original SDK World owner emits
`retired` only after its Host, source and registrations close; a request rejected
before dispatch may instead emit `not_created`. The native client may attest
`not_created` only when it never dispatched the request. A lost reply, timeout,
failed Host cleanup, or process exit alone is not a receipt.

Room consumes these structured records from the original CLI process, matches
the request ID/protocol/schema, drains stderr and verifies process completion
before issuing its existing factory/Room cleanup receipts. Mismatched IDs,
conflicting receipts and client-issued `retired` claims remain unconfirmed.
The resident browser is retained for other requests; this path neither stops a
shared service nor restarts GPU roles. Deployment requires a matching pinned SDK
with this protocol; older SDKs retain their conservative quarantine behavior.
The CLI's cancellation reply wait is bounded at 20 seconds, inside Room's
35-second process termination grace. Before TERM, an early startup cancellation
allows up to five seconds for the original CLI to install its signal handler;
the request marker itself is never cleanup proof. Cancellation racing with the
spawn reply still drains the same child and its receipt. Normal cleanup proceeds
as soon as the native owner acknowledges it.

Other bounds: source history 56 MiB plus one start pin up to 8 MiB; immutable
reference assets 64 MiB/128 items per source, active references 9/8 MiB; journal
64 MiB and 1024 operations/2 MiB per source; control rate 2/s with a burst of 4.
The deployed 38-frame model input window remains unchanged.

## Verification and scope

Contract tests cover CAS/idempotency, fencing, prompt ordering/rebase and durable
reopen. Lifecycle tests exercise four segments on the same source, two prompt
versions, reconnection without a second join and stale stream ACK rejection.
CPU sampler tests exercise a pending condition across older ready blocks while
preserving history. Existing Room, Coordinator, worker reservation, source
adapter, decoder/playout and v1 tests remain regression gates. Native deployment
and 300-second real-media validation are recorded separately; local tests alone
do not establish GPU performance or visual instruction-following quality.

This release implements the cloud source path. Local source uploads, NPC command
integration, multi-Worker capacity expansion and separate control/media sockets
are outside this implementation. Reference image upload within an existing cloud
source is supported. Model semantic conditioning retains ordinary reference
image behavior; role metadata does not imply a new model architecture.

## Optional resource preparation before source creation

Candidate contract; cloud startup performance is not yet accepted. No H3 kernels,
model configuration, source clock, game session or model lease are created by this
operation. Existing active-user capacity remains one.

With `WHITEBOX_PREFETCH_ENABLED=1` on both Gateway and Room, a single-owner Gateway
accepts a first MessagePack message `world.prepare` on the existing playtest
WebSocket, containing `whitebox.source_id` and `whitebox.world_id` (the normal
`whitebox_control=v2` / `control_protocol=whitebox-render-v2` configuration applies).
It forwards to the configured owner's internal `POST /internal/worlds/prepare`
with the existing internal token; the browser cannot choose an upstream URL.
This reuses the current direct playtest entry. It does **not** add authentication
to the legacy public route or claim to implement authorized_generate integration.

Room requires CDN transport, a valid `WHITEBOX_CLI_ENTRYPOINT`, and an absolute
`WHITEBOX_CLI_CACHE_DIRECTORY`. The installed SDK must export
`prefetchWtpPlayable` from `@worldkit/stream-host/wtp` and support the matching cache
options on `downloadWtpPlayable`. Pin both revisions in a release; installing the
Gateway alone does not enable preparation. `WHITEBOX_CLI_DOWNLOAD_CONCURRENCY`
defaults to 6 (SDK accepts 1–16). Mount the cache on persistent, Room-owned storage
if it must survive process/container replacement. Normal source startup uses this
same cache, with fresh authorization/publication validation.

Progress messages have `type=world.preparation`, state `preparing`, `ready` or
`unavailable`, and bounded file/byte/elapsed metrics. Ready means all current
publication files were verified and retained, not that a browser or model has
started. Room permits one preparation process at a time; a second receives
`PREPARATION_BUSY`. Disconnect cancels the HTTP stream and terminates the child.
Node's preparation budget is 300 seconds; Room bounds the stream at 310 seconds
and Gateway at 320 seconds. Unsupported/disabled preparation must fall back to
ordinary source startup, without retrying in a tight loop.

Prepare a fixed world after its WTP playable publication is ready and before
inviting users; selection-screen preparation can confirm or refresh the cache.
A new publication or replaced cache volume can require a cold transfer again.
Do not count either prewarming time or publication time as model inference.

The CDN CLI adapter records `prepared.startupMetrics`: `preparationMs` includes
metadata, verified download/copy and cache storage; `hostStartMs` measures Host
creation; `worldReadyMs` measures session creation and the real stream-ready
barrier. Room stores these in engine metadata and observes their durations.
These are server measurements; the client must separately measure entry click
to matching model pixels plus UI. A cache mock or `ready` response alone cannot
prove ten-second interactive entry.

### Capture-rate debugging and diagnostics

`init.whitebox.cli_fps` is an optional integer from 1 through 60. It only controls the newly created source's CLI capture rate. Omit it to use `WHITEBOX_CLI_FPS` (currently deployment-configured); existing sources, joins and reconnects retain their capture rate. It does not change the model's 24fps clock, H.264 nominal descriptor or captured source timestamps. `session.ready.engine.capture_fps` reports the resolved rate.

Known startup failures include an English public message, stable `code`, `stage` and `trace_id`. Raw exceptions, credentials and internal paths remain server-side. The browser should retain the stable code/trace rather than replacing every error with a generic message.

For manual inference, the original `model.inference.configure` result and
`source.status.inference.code` preserve allowlisted coordinator admission reasons
such as `CAPACITY_EXHAUSTED`, `USER_SESSION_LIMIT` and `USER_ALREADY_ACTIVE`.
Unknown exceptions remain `INITIALIZATION_FAILED`; an unconfirmed cleanup takes
precedence as `RELEASE_UNKNOWN`. A failed admission keeps whitebox play alive.
An explicit retry uses the same source and a new inference revision, and cannot
reserve another model while the prior release is unknown. The owning inference
and bound-model tests cover these receipts without GPU computation; deployment
and live recovery remain separate acceptance steps.

The owning Room pipeline reports a new game rejected by the world limit as `CAPACITY_EXHAUSTED`,
`stage=admission`, with the original trace ID. Rejection happens before the
Room starts another CLI or replaces existing players. Joining an existing game
still follows that game's player, geometry and ownership rules. Model-slot
availability alone does not guarantee Room availability: a live whitebox source
can occupy the Room while holding no model lease.

When diagnosing a full Room, correlate source/game identity, source-close
receipts and directory lease/end times. A recording-status poll alone does not
prove live play or failed cleanup. In the 2026-10-30 incident, a background 480p
source was still occupying the one Room during two rejected starts; it later
closed and completed its archive. A separate 720p model failure had confirmed
1280×720 public / 1280×736 private geometry and released its model resource;
its cause was not established by the Room-capacity diagnosis. Keep these
failure identities and evidence boundaries separate.

Optional `whitebox.model_timing` events associate numeric stage timings with a contiguous accepted model-source frame range and a model/segment identity. They are diagnostic only; pixels never wait for timing delivery. Missing/mismatched events must be shown as unavailable, not zero. Per-frame `source.model_timing` separates source selection, submission preparation and dispatch-to-raw return. These are elapsed spans, not additive GPU utilization measurements; source collection and cross-chunk work can overlap.

## Cache management

The separate [Room cache maintenance](whitebox-runtime-maintenance.md) surface
uses the existing owner and admission lock. It does not participate in frame
processing, render revisions or model leases.


## Optional connection Prefill

See [the control-connection Prefill contract](whitebox-control-prefill.md) for parent worker ownership, optional `render.prepare`, real F0, multiple conditions and native cache receipts. The former `prepared_input` / browser RGB roundtrip is superseded.

### Player instruction business outcomes

`model.context.instruction.receipt` is reliable control traffic, independent of
`WHITEBOX_PROMPT_DEBUG` and the browser debug subscription. Its payload contains
`source_session_id`, `source_epoch`, `segment_id`, `segment_epoch`, the user
`requestId`, instruction `revision`, terminal `phase` (`applied`, `failed`,
`rejected`), and optional `reason`. It contains no prompt text or reference bytes.
`applied` confirms model output for the condition, not browser presentation.

A Harness `model.condition` may carry `instruction: {requestId, revision}` frozen
by Stream Host. The journal stores this separately as `model_context_instruction`
on the original operation so reconciliation can recover the same identity after
adapter recreation. These are user instruction identifiers, not the model request
ID or model prompt version. Plain sources use their original `instruction_result`.
Consumers fence source and segment lifetimes and never infer success from a timer.

Install the accepting SG parser before deploying a Host which emits this metadata.
Conditions without an instruction remain supported. This does not change model
payload hashing, model prompt contents, prefill input, or the encoded video protocol.
Existing prefill/cache business observation remains independent of the Debug UI;
detailed compiled prompts and reference-image reads remain opt-in diagnostics.
Test-lane acceptance verifies ordinary non-inference playback; real model application
requires a model-enabled environment and is not established by a whitebox-only run.
