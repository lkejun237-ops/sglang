# Whitebox playtest integration

This branch connects an existing world to H3 through the main_ali gateway and
coordinator lifecycle, with independent regional deployment settings. The engine CLI and the world's UI implementation remain
independent plugins. The local mock is an executable integration fixture, **not
H3 inference and not training data**.

## Session source encoding

The opt-in production contract, ownership, signed WTP carriers, codec semantics
and staged validation boundary live in
[Session source encoding](../../docs/docs/sglang-diffusion/zing_h3/specs/whitebox-source-encoding.md).
Omitting the selection preserves the deployment default. This is separate from
isolated local RTC controllers and does not update a frozen World runtime.

## Scoped model-context evidence

[2026-10-31 single-player GPU regression](MODEL-CONTEXT-20261031.md) records the
exact overlaid sources, four runs, timing samples and remaining deployment/Prefill
acceptance. It is separate from historical non-model and formal Console results.

## Source and compatibility

- H3/runtime base: main `0cfc3d6f79b707fd25e396d02882fd101a1cff57`.
- Beijing gateway/coordinator/business modules: main_ali
  `b2852cbc546d144d9c3ab3e61c614ff7244b0b02`, selectively ported with regression tests.
- H3 model kernels, Qwen conditioning and denoising remain from main. The source
  adapter additionally accepts the trusted two-step deployment grid `[1000, 500]`;
  requests must match their deployed profile. Four-step deployments are retained.
  The worker's gateway-assignment parser now also accepts a fenced local decode
  reservation. H3 continuous source requires local decoding, so this assignment
  reserves one H3 worker, without a fictitious remote VAE slot.
- Redis and the in-memory test store support this new assignment. DynamoDB
  explicitly rejects it; existing remote denoiser/VAE assignments remain intact.
- Existing direct `/v1/realtime_video/generate` is the playtest ingress. The
  authorized routes, token verification, callbacks and legacy behavior remain
  available for their existing clients. They do **not** yet wrap whitebox media:
  current whitebox activation is on the direct route only. Future official-site
  integration can select the same adapter after its outer authorization flow.
- `WHITEBOX_ENABLED` is off by default. When enabled the gateway does not mount
  its bundled WebUI. The actual `/tasks` platform is `seedleap/world-test-platform`,
  while the public player client lives in `agent-whitebox-world-sdk`.

## Official CLI compositor integration

### Manual inference candidate

The CPU image installs the shared Redis prefill client from `requirements.txt`.
Its actual-image check constructs the configured Gateway prefill store and closes
the application without network access; Coordinator's minimal Redis fallback
does not satisfy this Gateway dependency.
The prefill lane also exercises repeated authorized observation against the
shared request claim; ordinary RTC create capabilities remain one-use.

The external Player manual-inference candidate selects exact SDK
81171edcaa0fd6cc86b101ca5913a73f9ee50c75 through sdk-revision.txt. This reviewed `white_box` integration contains the reviewed dev snapshot,
the Harness CLI producer and historical-console dependency compatibility, plus
trusted subscription pause/inactive capability negotiation and the public
confirmation/control port, schema-5 optional-playtest delivery support and the
original capture hold/resume/input-lease fixes. CPU source pairing must use this exact source; an npm
version string or changing a deployment variable is not capability proof.

Historical World runtimes are immutable. A Harness World without the negotiated
pause capability is refused before model acquisition; existing auto callers keep
their original wire. New World deliveries require the paired runtime. This source
pin does not publish runtime bytes, promote a shared branch, assemble WTP's
image-verified SDK source bundle or deploy a Room/model service.

### Upstream optional-playtest baseline

The preceding optional-playtest Room CLI release selected exact SDK
0b239470c54d54e9ed0efd0f82be9a91ae45969f. Its shared Stream Host reader accepts
schema-5 deliveries explicitly sealed with requirePlaytest: false and
playtest.status: not-run; their original episode may be absent. Default-required
deliveries still require genuine native recording. Room consumes the frozen
delivery/runtime and does not manufacture an input or video baseline. Baseline
replay remains unavailable when no recording exists. The actual Room image,
schema-5 start/stop, video, input acknowledgement and UI acceptance must be
verified after deployment. That white_box baseline includes the capture
hold/resume and input-lease fixes; a later CLI pin must retain those consumers.

### Historical frozen CLI baseline

The prior unified Player baseline selected exact SDK
`e99e03be52410a37732e2007969de862ecfb75b5` through `sdk-revision.txt`.
It is the frozen, identical SDK `main` and `dev` source at release.
Its compiler/materialization and Atlas preset inputs require the paired new
cache identity and dependency lock; equal ordinary runtime payload bytes do
not make the prior release interchangeable. The client renews through its original heartbeat owner before
decoding a canonical large resource and again before handing it to the visual
consumer. It also adds Creator player-control tools while retaining admission,
input-lease and cleanup owners, including the closed-stderr logger correction.
The ordinary runtime now has hash
`c07c4dbd81d043a9373a4549ccf77d13517727c07fe33c93abc42d51bb753d39`,
with cache identity `763edefe3db5db107b71d52a0427b6218495dd40509a6840932c7f6984686bab`.
The public release is `0.2.20-e99e03be`, with release-manifest SHA-256
`8b94cf613b228eafa8ba3162bae4295fc2dc81c8885eb9c7abca8bbb0f35c3e8`.
The exact dependency lock SHA-256 is
`dc9d0795225038fc2058e6a47b879374dd47e2edbd704c1b87eb895aeb4b55c3`.
The compiler/materialization input change requires this source and runtime
cache identity even though the ordinary JavaScript payload hash is unchanged
from `0.2.19-619b236f`.
The paired WTP source selection and the actual Room image remain separate
from the ordinary runtime publication.

The prior `0.2.18-a019b6a4` release retains its actual A019 producer and public
manifest `1d4d533845f78f48a9278a4cfd5f890cc67bceb3b39f6f9dd4a2f548dd8b7d77`;
it is not an alias for the new runtime. An ordinary-rate native pass does not
establish liveness under arbitrary CPU pressure. Diagnostic slowdown scenarios
retain their original failure boundaries and do not increase heartbeat timeouts.

The prior f83 release remains `0.2.17-f83b370a`, with public manifest SHA-256
`2817c90cc8ae1d2b0f80ec7a6e8f75d8acd032489090d465bb039f7708e09e04`.
Its anonymous raw/gzip verification and original native starter, queued
cancellation and member-resume evidence retain the exact f83 identity. The
cooperative large-definition copy remains included, with original heartbeat
and preparation limits. Neither that evidence nor equal runtime bytes prove
that the newer Host/CLI source has been deployed.

The candidate retains shared `white_box` Player/media and cleanup integration
from PR #460 / `bc882c640a9030ad189dd86c6a86b47e5ec579c3`, including the
frozen publication/admission/subject fences and native/model media union.
Startup failure requires the original factory and Room's positive cleanup
receipts; explicit same-operation leave/retire can continue retained cleanup,
while queries do not restart it. Member model children must close before the
original native source acknowledges cleanup. The added optional negative
`authority` evidence preserves existing successful action, session and media
protocols. SGLang continues to forward the original receipt rather than
inventing a completion or retrying the action.

Exact SDK/SG CI, the actual Room image and paired single/member native,
RTC/model/ICE acceptance remain separate release gates. Existing published
worlds, stable-channel choices and historical runtime references retain their
original identities. This pin update does not deploy services.

### Published single-player baseline

The previously published single-player integration used exact SDK
`e0cf37beb1fc9b02e1053260622b3b9b12a503ef`, the reviewed shared `main`
CLI source. The following runtime and acceptance notes retain that historical
identity. Published AREA preparation verifies its declared startup context
before initial and deferred runtime closure reads. Cache misses for immutable
Atlas resources honor the selected CDN/OSS transport.

Public runtime 0.2.15 is independently published from this exact source. It
retains immutable rounded matrix records during geometry-guard snapshots without
changing matrix refreshes, mutation checks or ownership. It also includes the
previously merged camera instance-scaling and full gameplay-view recovery fixes.
The guard-owner microbenchmark is local CPU evidence only; indoor source-capture
performance and the new runtime's camera behavior still require acceptance in the
same original LUMINA project. Authored shadows, scene geometry and assets remain
unchanged. Existing public 0.2.14 remains bound to its original source
`3a7ee0f487ca839c8068544d145076d46052776f`; this pin does not rewrite it.
Implicit AREA still captures retain the controlled subject's semantic heading;
explicit Episode starts retain their behavior. Bounded navigation separation,
native obstacle avoidance, NPC and composed-character checks, and authored
keyboard, pointer/button and HUD action delivery remain in the shared source.

Jenkins verifies the pinned SDK against `origin/dev` and
installs the pinned checkout through the existing image build. The reviewed main
source is reconciled into trusted SDK history without changing the dev file tree;
this pin does not bypass the ancestry or clean-checkout checks. Before activation,
verify exact-pin SGLang CI, the official Room build, native image/configuration
health and the original-world input/model consumers. Source promotion, immutable
runtime publication, this Room pin and deployed gameplay are separate gates.
Each world's immutable runtime remains bound to its own publication; changing a
Room CLI pin alone does not replace an older frozen World runtime. Historical
SDK242 source/CI `40234552446`, pin #422, and earlier 219c/f494 receipts retain
their original identity and do not validate this new pin. See the
[integration plan](../../docs/docs/sglang-diffusion/whitebox/plans/2026-10-24-worldkit-cli-adaptation.md).
The default Room renderer uses the public `@worldkit/stream-compositor` Node API. It returns
the starting descriptor before model output so the Room can subscribe without
losing the first composed frame. The CLI `compose` command waits for first decode
and therefore cannot be used as a blocking prerequisite to model input.

The adapter converts model RGB to local VP8 and composed VP8 back to RGB; playback
and all three archived tracks remain H.264. This adds a lossy local codec boundary
and its measured cost must be reported. Only one composition frame is in flight
per player, preventing the realtime player from dropping intermediate frames.
CLI capture, H3, recording and upload still run in independent bounded stages.
Returned source identity, UI revision and PTS must match; EOS must acknowledge the
exact final frame before a session is reported complete. The SDK owns UI history,
React rendering and native self-tab capture. No SDK source/OSS routing is patched.

Local 832x480 integration passed once; the AMD64 release check currently reproduces
`COMPOSE_CAPTURE_INVALID`. See `deployment/THAILAND-READINESS.md`; the new image is
not accepted for deployment until this failure is resolved.

## Debian build downloads

The official `docker/whitebox.Dockerfile` uses the shared build-only
[`debian_main_mirror.py`](../../deploy/whitebox/debian_main_mirror.py) for all four APT
consumers, including Playwright's dependency installer. Shanghai defaults to the
Aliyun Bookworm main mirror only when its complete `InRelease` bytes match the
current Debian origin and verify with the original `Signed-By` keyring. No fixed
Release hash is carried forward between builds. `bookworm-updates` and
`bookworm-security` keep their official sources over HTTPS to the same
`deb.debian.org` host; a valid but older mirror is not
an equivalent release.

The wrapper uses the existing Python/urllib, CA certificates and Debian archive
keyring/GPG tool in the shared Python Bookworm base. It does not install its own
bootstrap dependencies. Both complete Release reads run concurrently with a
15-second process deadline and 1 MiB limit each; signature verification is
limited to five seconds. Missing verification support, an unsupported source
layout, a download error or different Release bytes selects the original sources
over HTTPS on the two known Debian origin paths and prints the reason. Unknown
repositories and snapshots are not rewritten. APT still performs its ordinary signature, expiry and
package checks. Nothing sets `trusted=yes`, disables `Valid-Until`, or changes
package selection and install arguments.

The helper is mounted read-only during a build command. It restores the original
source file in `finally`, including failed commands; it is not copied into the
runtime image. The command runs once and retains its exit status. A package or
browser-install failure is a failed build, not permission to repeat a partially
executed build command. JSON log events report source selection, comparison
latency and command elapsed time. They do not claim package throughput from a
small Release download. Cancellation that discards a build step does not create
a deployable modified image.

Shanghai's separately reviewed preserved-base overlay gate must reuse this same
public helper when preparing future build contexts, instead of copying a fixed
Release hash or another mirror-selection implementation. Its one-off build
receipt is separate from the official full image, and this repository does not
own or reproduce private role configurations. Export the helper from the same
reviewed SG commit, mount it for the gate's existing command, and retain the
existing immutable-context/CAS checks. For an existing gate command the call is:

```dockerfile
RUN --mount=type=bind,source=deploy/whitebox/debian_main_mirror.py,target=/tmp/debian_main_mirror.py,readonly \
    python /tmp/debian_main_mirror.py -- sh -ec 'apt-get update && apt-get install -y --no-install-recommends procps ffmpeg'
```

Mock-network/signature regression tests live beside the helper and run in the
existing official control-plane image checks. No cloud job or gameplay
acceptance step is introduced by download acceleration.

## Release dependency decoupling

The [design](../../docs/docs/sglang-diffusion/whitebox/specs/2026-11-01-release-dependency-decoupling.md)
and [two-task execution plan](../../docs/docs/sglang-diffusion/whitebox/plans/2026-11-01-release-dependency-decoupling.md)
cover separating test-only runtime inputs and connecting fixed-digest cross-commit
layer candidates to actual Jenkins builds, based on source `509dc194`.

The Dockerfile's build-only `runtime-source` stage now projects `python/sglang`
through [project_runtime_source.py](../../deploy/whitebox/project_runtime_source.py), using
read-only source/helper mounts. Both `runtime` and `room-runtime` copy this output.
Only `multimodal_gen/test/` is excluded; runtime resources, dynamic plugins,
`sglang/test` and paths named demo or testset remain. The standard-library helper
preserves bytes, paths, executable modes and safe relative internal symlinks,
sorts traversal and normalizes timestamps. It rejects escaping, dangling or
excluded-target links and unsafe/nonempty destinations without changing source.
The complete excluded test subtree, including Node fixtures, is explicitly
restored in both `checks` and `room-checks`. Existing CPU, real SDK CLI and native
codec commands and their release markers remain in place. The SDK pin, model,
media and public protocol inputs are unchanged.
The `checks` stage copies the binary from the shared Node 24 stage to execute
the real JavaScript fixtures; the CPU `runtime` and `release` payloads remain
Node-free. Room keeps its existing Node/SDK dependencies.

Run the local projection and stage-consumer regression on CPU:

```bash
python -m pytest deploy/whitebox/test_runtime_source_projection.py \
  deploy/whitebox/test_runtime_image_contract.py \
  --confcutdir=deploy/whitebox -q
```

The tests execute the actual projection-stage command on repository source,
compare complete test fixture restoration, and check stage dependency edges.
They prove equal projected inventories after test-only edits and changed output
after runtime edits. The cheap projection step can still rerun when tests change;
downstream BuildKit cache reuse and build-time improvement require actual image
measurements. To inspect separately built final image payloads and resolve the
configured engine, CLI/Worldkit Room and UI factories, set `WHITEBOX_RUNTIME_IMAGE`
and `WHITEBOX_ROOM_IMAGE` to the exact candidate references before this command.
Without them, those two image checks explicitly skip. These payload/import checks
do not replace real launcher lifecycle, SDK/native-codec or GPU/playback acceptance.
The Jenkinsfile now computes a layer recipe after the managed Dockerfile is
available and discovers up to three verified fixed-digest candidates from at
most 20 recent records in the existing trusted ECR repository. Branch, logical
target, recipe, platform, full producer revision and manifest/config bytes must
match the cache contract. Source SHA and general source/test bytes remain
BuildKit inputs; they do not enter the recipe identity. A new SHA still builds
with its true revision unless its original immutable final tag already exists.
The actual build command consumes accepted candidates through the original
imports file and retains inline export, same-SHA imports, regional push routes
and callback verification. See [cache eligibility and local tests](CI-REGISTRIES.md#cross-commit-layer-candidates).

Candidate image builds, launcher startup, actual cached vertices, BuildKit
cold/warm comparison and hosted configuration reconciliation remain pending.
Repository source does not update installed inline jobs or managed Dockerfiles;
the non-executing Jenkins model converter and exact hosted-byte reconciliation
are required before a later authorized installation. No image/cache hit, native,
GPU/model/playback, performance or production activation is established by the
local CPU tests.

## Service interaction

### Room process supervision

For persistent model-context recovery, run the accepted/applied SIGKILL cases in
`python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_render_contract.py`
(`-k room_process_sigkill`). They launch separate production RenderHost control
processes, kill the first after durable SQLite acceptance, then query the original
HTTP status route after restart. The model evidence is deterministic and no GPU
is allocated. Recovery preserves the original journal, fences the lost owner,
and reports `release_unknown`; it does not recreate the lost CLI/GPU session.

For browser disconnect and bounded original-player cleanup recovery, see the
[release proof contract](../../docs/docs/sglang-diffusion/zing_h3/specs/whitebox-render-segments.md#limits-and-release-proof).
The CPU regressions in
`python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_startup_retirement.py`
cover failed-start evidence, cancellation, cleanup retries, quarantine and
next-player admission. Run them with the lightweight pytest entry below. They do
not establish live GPU/RTC or production recovery acceptance.

The Room image runs its existing Python launcher under `tini` as PID 1.
Chromium crash handlers and compiler subprocesses may become orphaned when a
CLI exits; init reaps those descendants while Python continues to await its
direct children. Do not add a global Python `waitpid(-1)` handler: it could
consume exit statuses owned by asyncio. Gateway and other role entrypoints
remain unchanged. Deployments must retain the Room image's init entrypoint;
an explicit command override that starts Python directly bypasses this behavior.

The maintained `scripts/ci/zing/check_room_init.py --image <room-image>
--output <receipt.json>` check mounts a CPU process probe through the image's
actual entrypoint. It verifies orphan reaping, direct child exit statuses,
SIGTERM forwarding and exit-code propagation twice, then removes only its own
isolated containers. It uses no network, GPU, production environment or cache.
CI checks the `room-process-init` stage; release acceptance must run the same
check against the exact final Room image. Source/CI success does not confirm
that an existing container has adopted the new entrypoint.

### Negotiated input receipt deltas

Clients may request `input_receipt_protocol: "state-delta-v1"` in `init`.
Room echoes the accepted protocol in `session.ready` and retains new or changed
`input.receipt` evidence in reliable `whitebox.states` control messages. Evidence
already retained there is omitted from subsequent states and media envelopes;
unseen media evidence is preserved without advancing the reliable receipt index.
The index is bounded to 256 `(epoch, clientSequence)` identities per pipeline.
Reconnect replays the stored deltas after the applied control cursor exactly
once. Clients retain pending receipt evidence until a displayed frame reaches
its source epoch/tick. Unknown receipt types and clients without negotiation
retain their original envelopes. Producer data and complete recording evidence
are never rewritten; input acknowledgements and video credit windows are unchanged.

```mermaid
sequenceDiagram
    actor U as Player
    participant P as Test platform / SDK client
    participant G as Gateway
    participant C as Coordinator + Redis
    participant E as World CLI plugin
    participant H as Main H3 worker (local decode)
    participant R as Backend UI plugin
    participant A as Recording / archive tasks
    participant O as OSS
    U->>P: 试玩 → select references / prompt → 开始试玩
    P->>G: generate WebSocket, MessagePack init(world_id,...)
    G->>C: admit / reserve real H3 slot
    C-->>G: fenced assignment
    G->>H: consume reservation, H3 source init
    G->>E: resolve world_id / connect CLI WebSocket
    H-->>G: ready + source credits
    G-->>P: ready (dimensions, FPS, duration)
    loop Active session
        P->>G: ordered keyboard/mouse input; optional prompt update
        G->>E: input through engine adapter
        E-->>G: RGB24 + frame/PTS + opaque states + effective input
        G->>H: source.append (credit controlled); separate prompt event
        G->>A: enqueue whitebox frame
        H-->>G: source consumed + generated RGB with source/prompt ownership
        G->>A: enqueue model frame
        G->>R: generated RGB + matching states/frame
        R-->>G: composited RGB
        G->>A: enqueue UI frame + aligned metadata
        G-->>P: H.264 access unit + frame/PTS/states
        P->>P: WebCodecs decode, then draw matching VideoFrame
        P-->>U: display final frame
        G->>C: renew lease
    end
    P->>G: finish (or configured play deadline)
    G->>E: finish, drain frames to explicit EOF
    G->>H: source.end(actual frame count)
    H-->>G: final frames, session.completed
    G-->>P: final frames, completed
    G->>C: release lease / slot
    G->>A: finalize in background
    A->>O: parallel H264 MP4 / JSONL / reference uploads
    A->>O: publish manifest last
```

The CLI and gateway-side adapter should be near H3: raw 832×480 RGB at 24 FPS is
about 28.8 MB/s in each RGB direction. H3 sees RGB24, never the archived H.264.
The native CLI reader grants at most two frames and 64 MiB of pixel credit,
including the frame held by the consumer. This lets the next capture overlap
downstream work. Credit returns only when the consumer advances. Native pixels
do not wait for UI snapshots or clock watermarks; late UI events travel in a
later reliable state message with their original timestamps. Presentation uses
exact historical UI when available and otherwise shows video alone. Pixel
identity, generation, payload size and source-clock checks remain mandatory.
Session admission waits once for the archive child to bootstrap within its
existing deadline, before starting the live frame pump. A failed archive still
degrades independently; frame offers never wait on archive IPC or disk.
Source ingestion, inference, UI processing and the three encoders overlap through
bounded queues. UI necessarily waits for its matching model frame; inference of
subsequent frames can proceed while that frame is being rendered/encoded.
Browser playback uses a continuous H.264 Annex B stream over the existing
MessagePack WebSocket. Each `frame_batch` contains one access unit (one picture),
with `encoding=h264`, `content_type=video/h264`, `bitstream_format=annexb`,
`codec`, `keyframe`, `duration_us`, and the existing `output_frame_start`,
`pts_us`, `width`, `height`, `states`, `event_id` and `pipeline_ms` fields.
`session.ready` advertises `format=h264` and `video_stream` containing the actual
SPS-derived `avc1.*` codec string, dimensions, FPS and Annex B format.

The live encoder uses libx264 constrained baseline, yuv420p, CRF 18, `veryfast`,
`zerolatency`, no B frames/lookahead, and an IDR every 24 frames. Each IDR includes
SPS/PPS. The SDK configures WebCodecs without `description`, as required for
[Annex B H.264](https://www.w3.org/TR/webcodecs-avc-codec-registration/).
It decodes continuously, matches metadata by PTS, and flushes only at completion
before closing the session. Decoded VideoFrames are released after display;
decode/render backlog and stalled decoders terminate explicitly without silent
frame dropping. A WebCodecs-capable browser with H.264 support and HTTPS (or
localhost for local validation) is required; capability errors are shown before
playback. There is no JPEG fallback. This is elementary H.264 transport, not MP4
fragments, HLS or WebRTC. Only the final backend-composited video goes to players.

The live encoder is independent of the three archive encoders to avoid making
playback wait for MP4 finalization. It adds one CPU encoder per active session;
include its CPU/RAM usage in deployment sizing. H3 input remains RGB24, and all
three persisted videos remain H.264 MP4 files.

## CLI and UI ports

Server configuration supplies Python callables, never client-supplied module names:

```python
# Your package; its actual CLI wire schema is intentionally not prescribed here.
from sglang.multimodal_gen.runtime.realtime.whitebox.engine import WebSocketEngineFactory
from sglang.multimodal_gen.runtime.realtime.whitebox.ui import RgbaOverlayRenderer

engine_factory = WebSocketEngineFactory(resolve_cli_endpoint, make_cli_codec)

def ui_factory(*, world_id, trace_id):
    # overlay returns exactly width * height * 4 RGBA bytes; it can call another
    # service or an offscreen renderer. Backend composition is implemented here.
    return RgbaOverlayRenderer(make_overlay(world_id), metadata={"bundle": "ui-v1"})
```

Implementations of these example callbacks are supplied by the engine/UI owners:

1. `resolve_cli_endpoint(world_id, spec, trace_id)` → `EngineEndpoint(url,
   headers, cleanup)`. A CLI subprocess may be launched here using argument
   arrays. `cleanup` is async and owns that subprocess/session's release.
2. `make_cli_codec(spec)` → `EngineCodec`: serialize input/finish, translate
   each received CLI message into `EngineFrame`, `EngineEnd`, or an acknowledgement.
   The transport enforces contiguous frames and an explicit EOF count.
3. `EngineFrame(index, pts_us, rgb, states, effective_input)` uses RGB24 HWC,
   zero-based contiguous frames, `pts_us = index * 1000000 // 24`. `states` is
   opaque JSON-compatible application data. No HUD fields are imposed.
4. `effective_input` is the full input snapshot actually consumed for that frame,
   including transient events for the appropriate frame only. The plugin must
   establish that mapping from the CLI's receipts/state, not gateway arrival time.
   If unavailable, pass `None`: received events are preserved as approximate
   evidence and `quality.unknown_input_frames` prevents claiming exact input alignment.
5. `finish()` stops new production, drains already committed frames, and yields
   `EngineEnd(total_frames)`. Socket disconnect is never successful EOF. Producer
   backpressure must pause capture/retain ordered frames, not discard frames.
6. `UIRenderer.render(model_rgb, frame, spec)` returns same-size RGB; its metadata
   identifies the UI bundle/resources. `RgbaOverlayRenderer` implements actual
   alpha composition. `contracts:no_ui_renderer` is an explicit opt-in only for
   worlds with no UI. A real world's state-to-overlay logic is still required.

The UI adapter may move to another process/service without changing admission or
the engine contract. Unsupported dynamic references are not silently accepted:
the current H3 source session fixes ordered references at init; select new
references by starting another session. Prompt updates use H3's existing path,
including its own initialization media/S0 conditioning semantics.

## Configuration and launch contract

Region-specific runtime environments are rendered offline with
[统一发布入口](../../deploy/whitebox/README.md) 负责当前区域策略、模型资产和启动命令；地区只保留接线与资源差异。

Use an existing working main H3 deployment (checkpoint, bound realtime profile,
TAEH3/native decoder and Qwen setup) from [the H3 docs](../../docs/docs/sglang-diffusion/zing/realtime-streaming.mdx).
This branch does not invent a new model launch or GPU sizing result.

| Setting | Meaning / default |
| --- | --- |
| `WHITEBOX_ENABLED=1` | Opt in to the new direct-route adapter |
| `WHITEBOX_ENGINE_FACTORY=your_plugin:engine_factory` | Required CLI factory |
| `WHITEBOX_UI_FACTORY=your_plugin:ui_factory` | Required UI renderer factory |
| `WHITEBOX_WIDTH`, `WHITEBOX_HEIGHT` | 832×480, both divisible by 32; 24 FPS |
| `WHITEBOX_STARTUP_SECONDS` | 90; includes init, engine start, conditioning and first delivered preview |
| `WHITEBOX_PLAY_SECONDS` | 60; begins at the first preview successfully sent by gateway |
| `WHITEBOX_RENDER_PLAY_SECONDS` | Render-v2 effective-play budget: integer 1–7200 seconds, default 300; frozen per source at creation, separate from the direct adapter budget above |
| `WHITEBOX_DRAIN_SECONDS` | 30; manual/automatic finish must drain within this bound |
| `WHITEBOX_PENDING_FRAMES` | 96 source/state entries; tune with H3 codec lookahead |
| `WHITEBOX_MAX_MESSAGE_BYTES` | 67108864; must equal H3's source transport budget |
| `WHITEBOX_SPOOL` | `/var/lib/sglang/whitebox`; use persistent disk and monitor space |
| `WHITEBOX_MODEL_METADATA` | JSON object with checkpoint/profile hashes, worker image/build version |
| `WHITEBOX_OSS_BUCKET`, `WHITEBOX_OSS_ENDPOINT`, `WHITEBOX_OSS_REGION` | Optional OSS S3-compatible publisher; default region `cn-beijing` |
| `WHITEBOX_OSS_PREFIX` | `whitebox`; each recording appends a random immutable ID |

Install the gateway's existing dependencies plus PyAV with `libx264`, Pillow,
NumPy, and boto3 for OSS. Credentials use the boto3 credential provider chain;
no credential is sent to the browser or written to the manifest.

Coordinator must use `--backend redis` in a shared deployment, with a dedicated
`--redis-key-prefix` for the whitebox service. For local tests use memory. Keep
existing `--ttl-s`, `--worker-ttl-s`, `--wait-timeout-s` and reservation timeouts.
Worker heartbeats register `role=denoiser`, `vae_fingerprint=local`, their actual
reservation endpoint/epoch and matching model/routing revision.

Configure simultaneous users through ready worker capacity and
`--denoiser-capacity-limit` (a **per-worker cap**, not a global count). Match this
with H3 `--realtime-max-sessions-per-worker`; overall available slots are the sum
across eligible live workers. One browser identity retains one active lease.
Direct `user_id` is an untrusted browser identity, not account authentication.
Existing gateway wait limits, worker fencing, lease renewal and cleanup remain
authoritative; no second session allocator is introduced.

Example gateway arguments (deployment values are placeholders, not commands run
as part of development):

```bash
python -m sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server \
  --coordinator-url http://coordinator:18081 \
  --model-revision YOUR_H3_ROUTING_REVISION --vae-fingerprint local \
  --internal-output-url ws://gateway:18080/v1/internal/realtime_output \
  --lease-renew-interval-s 10 --max-admission-waiters 64
```

Keep direct routes enabled; `--disable-legacy-routes` also disables direct generate.
The internal-output URL remains a required shared gateway flag, but local H3
does not use a remote VAE/output route. Do not expose worker reservation APIs to
browsers. Set the platform's `WORLD_TEST_WHITEBOX_GATEWAY_URL` to its TLS `wss://`
direct generate ingress; ordinary platform login remains outside this protocol.

The legacy direct route was limited by finite `max_chunks`, while signed sessions
have business play timers. Continuous whitebox therefore supplies the explicit
startup/play/drain bounds above without changing either legacy path. Configure
the worker's lifetime and stale-reservation watchdog to exceed startup + play +
drain + cleanup margin. Backend encoding/OSS completion does not extend the GPU lease.

Each session adds raw frame caches, bounded encoder queues and UI memory on top
of H3's transport/KV budget. Size gateway CPU and RAM independently of GPU slots;
do not raise capacity based only on the number of installed GPUs.

## Recording package

For the capability-scoped play-report context and artifact access API, including
exact segment selection and archive-frame mapping limits, see the
[owning contract](../../docs/docs/sglang-diffusion/zing/specs/WHITEBOX_CLIENT_UI_ACTIONS.zh.md#游玩投诉证据读取).

```text
whitebox/<recording_id>/
  manifest.json
  inputs.jsonl
  states.jsonl
  report-frames.jsonl    # compact archive/source-frame → actual per-stream PTS
  output-contexts.jsonl  # actual model-output prompt/reference transitions
  whitebox.mp4
  model.mp4
  ui_rendered.mp4
  references/00.png      # extension/order follows initial references
  references/01.jpeg
```

- Three MP4s: H.264, 24 FPS, same dimensions/index/PTS; libx264 CRF 18,
  veryfast, yuv420p, no B frames. They are compressed training candidates, not
  byte-exact RGB originals. No independent silent frame dropping occurs.
- `inputs.jsonl`: six fields: `input`, `prompt`, `start_time_us`, `end_time_us`,
  `start_frame`, `end_frame`. Intervals are `[start,end)`, full snapshots, split
  whenever input/prompt/reference changes. Prompt ownership comes from actual H3
  output `event_id`, not the time the user typed it. `input.references` contains
  ordered `{id,path,sha256}` entries; no separate asset-ID index is needed.
- `states.jsonl`: `{frame,pts_us,states}`; states contents belong to the engine.
- `manifest.json`: world/recording/trace identity, UTC creation/finalization,
  inference config and deployment metadata, UI identity, timeline, video frame
  counts/encoder settings, object sizes/SHA256, input quality, execution/recording/
  upload statuses. Gateway fills actual selected worker model/routing revision.
- Only `recording_status=complete`, one-to-one timeline and acceptable input/UI
  quality are candidates for training. Completion is not automatic dataset
  admission. Aborted/failed sessions remain marked partial; no fabricated frames.

Three encoders run concurrently; finalization and uploads run after session cleanup.
Uploads are bounded to three files concurrently per archive, retry three times,
and publish the manifest only after every object succeeds. The local spool is
retained, including when upload capacity is exhausted or a transfer fails. An
explicit retry is available through `python -m
sglang.multimodal_gen.runtime.realtime.whitebox.publish <finalized-directory>`;
it verifies object hashes before upload. The gateway now retries finalized pending/failed spools on startup and every 30 seconds; no external queue is required. However, process-killed incomplete MP4s require recovery and must
not be treated as complete archives. See [ACCEPTANCE.md](ACCEPTANCE.md) for current local and online validation status.

## Observability and validation

All stages share `trace_id`: gateway admission/reservation/renewal/send metrics,
engine startup/input/frame wait, source credit/send wait, H3's existing source /
Qwen / denoise / decode traces, UI render, three recording queues/encoders,
metadata, preview encoding/queue and OSS upload. Gateway `/metrics` exports
`world_model_critical_path_stage_duration_seconds` histograms; new stages start with
`whitebox_`. Set `WORLD_MODEL_METRIC_MODEL=zing_h3` consistently to distinguish
this service from legacy Zing. Browser decode/draw metrics are best-effort and
do not enter `inputs.jsonl`. `pipeline_ms` is gateway receipt-to-preview-queue,
not network RTT or display latency.

The existing realtime trace helper emits structured logs and optional OTLP logs
using `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT`/`OTEL_EXPORTER_OTLP_ENDPOINT` and
`OTEL_SERVICE_NAME`. Install its OpenTelemetry exporter dependencies when used.
This is not a claim that generic LLM production-request tracing automatically
instruments every diffusion/CLI stage. CLI-internal timings and true browser
display/input-to-photon timing require corresponding plugin/client instrumentation.

CPU fixture, no cloud writes/GPU:

```bash
PYTHONPATH=python SGLANG_LIGHTWEIGHT_RUNTIME=1 \
  python benchmark/whitebox/mock_gateway.py
# WS: ws://127.0.0.1:18087/v1/realtime_video/generate
# Read-only fixture events: http://127.0.0.1:18087/mock/events
```

Run `test_whitebox.py` and the gateway/coordinator/world regression tests under
`python/sglang/multimodal_gen/test/unit/realtime` with `--confcutdir` set to that
directory in the lightweight CPU environment. Tests cover local reservations,
lease fences, deadlines, actual H.264 PTS/pixels, delayed prompt application,
unknown input evidence, arbitrary states, UI composition and archive retries.
For manual native-reset inference recovery, include
`test_whitebox_inference_reset.py`, `test_whitebox_inference_control.py`,
`test_whitebox_source_reset_recovery.py` and `test_whitebox_source_conditions.py`.
The [owning lifecycle contract](../../docs/docs/sglang-diffusion/zing_h3/specs/whitebox-render-segments.md#world-owned-model-conditions-and-recovery)
states the supported reset scope and separate GPU/Player validation boundary.
RTC integration clients must independently consume each subscription's reliable
control channel while member actions await cleanup; completing mock RTP handoffs
does not drain those controls. The member-isolation regression sends 513 real
status snapshots with the original 512-message queue limit and verifies the other
member's status on the delivered wire. Queue-overflow and stale-attachment guards
remain separate assertions; background test readers are stopped and checked.
The actual SDK member-control fixture also queries `operation.get` after display
changes, checks the original source/generation and retained operation, and proves
the read leaves the journal unchanged. Member input and model-launch commands
remain restricted to their original owners.
The platform's `local-validation/whitebox-preview.mjs` mounts its actual task
detail against this fixture. H.264 browser validation used real libx264/WebCodecs:
1,154 access units, 49 keyframes, matching frame/PTS/states, final decoder drain
and released slot; all three archived MP4s had 1,154 frames and identical PTS.
Engine/H3 content in this validation was synthetic. Real CLI wire, real world UI, GPU throughput,
production ingress and OSS credentials remain deployment integration checks.

## Two-slot platform acceptance

See [ACCEPTANCE.md](ACCEPTANCE.md) for the real Redis/control-plane test, browser-world mock, recording status API, CPU image and deployment constraints.

### Cross-region CLI delivery startup

The Room adapter downloads through the public `@worldkit/stream-host/wtp` API,
then checks the receipt and sealed delivery through `@worldkit/stream-host/delivery`.
`WHITEBOX_CLI_DELIVERY_TIMEOUT_SECONDS` (default 300, maximum 900) is a bounded
whole-download deadline, replacing only the SDK transport's 30-second body
request deadline. The SDK still controls URLs, authentication, redirects, archive
limits, receipt hashes and UI resource validation. Its source and Beijing OSS
access are unchanged. The verified archive is passed to `worldkit stream
--delivery`, with all artifact identities checked against its resulting descriptor.
The stream manifest preserves both the CLI's local artifact identity and the
original WTP attempt/receipt under `engine.wtp_delivery`. Temporary archives are
removed after session cleanup or startup failure; this is not a persistent cache.

Regional CLI deployments allow 480 seconds for startup (download 300 seconds,
CLI startup 120 seconds, plus join overhead). The Room join HTTP deadline follows
the Gateway startup configuration. The 60-second play clock still starts at first
media. `cli_delivery_prepare` is reported separately from `cli_room_start`.

On platforms that generate every Service port as HTTP, the dedicated Redis
Service must specify `spec.ports[].appProtocol: tcp` for port 6379. Verify it after
platform reconciliation and before checking `/v1/slots`; do not alter other
services or use a proxy bypass. Mock deployments should set
`WORLD_MODEL_METRIC_MODEL=whitebox_mock` in both Gateway and Room so CPU mock
measurements do not appear as real H3 measurements.

### Archive workers and live timing

Production gateways now create a recorder process per session (at most two,
including a recorder that is finishing), plus one separate OSS publisher process.
Whitebox/model RGB are each encoded once in the recorder. The UI H.264 access
units used by WebCodecs are remuxed into `ui_rendered.mp4` without another encoder.
All three files retain the same frame-derived 24 Hz PTS; abnormal termination
produces `partial`, or `failed` if the process cannot finalize the files. Such
files must not be treated as complete training samples.

`WHITEBOX_RECORDING_QUEUE_DEPTH` defaults to 24 queued operations, plus at most
one executing operation. `WHITEBOX_RECORDING_QUEUE_BYTES` defaults to 64 MiB
including the executing payload. Frames and metadata share the ordered queue.
Full queues apply measured backpressure; frames are never silently dropped.
Each codec uses one thread. On Linux with at least three available CPUs,
recorder and publisher are pinned to different CPUs with a lower process priority
(`nice +10`). The existing gateway pod CPU/memory limits still apply; these are
process/CPU-affinity bounds, not separate cgroups or disk/network quotas.

The capability-scoped recording report includes queue depth, bytes, high-water
mark, oldest pending age, and pending uploads. Trace and Prometheus expose
`recording_backpressure`, `recording_worker_queue`, `encode_whitebox`,
`encode_model`, `mux_ui_rendered`, `recording_metadata_write`,
`upload_worker_queue` and `oss_upload`. Prometheus also exposes bounded aggregate
`world_model_whitebox_recording_pending_operations` / `_pending_bytes` gauges.
Queue wait may rise without delaying playback; sustained **backpressure** is the
measurement of actual waiting in the live producer.

H3 numeric chunk telemetry is forwarded into the report via an allowlist. H3
`*_host_ms` spans measure host calls without CUDA synchronization and must be
labelled Host, never presented as GPU kernel time. Browser receive/render FPS
and H.264 payload bandwidth are rolling measurements; precise input-to-display
or one-way network latency is unavailable until consumption mapping and clock
alignment are supplied.


The real CLI acceptance fixture is owned by the serving tests at
`python/sglang/multimodal_gen/test/unit/realtime/whitebox_wtp_fixture.mjs`.
It compiles the SDK's streaming UI example through the public
`@worldkit/creator-host/compiler` export, seals a local hash-checked archive,
and serves only loopback WTP endpoints. It does not depend on private SDK test
filenames or contact a cloud store. Run both direct and Room-owned paths with
`WHITEBOX_TEST_CLI_SDK` pointing at the pinned SDK checkout; the existing
`test_whitebox_cli_integration.py` remains the image-build acceptance test.

The optional manual-inference CPU lane uses that same real SDK/CLI/SourceSession
fixture and the explicitly mocked model peers. It proves preview before any model
lease, original complete prompt/ordered PNG conditions, member isolation, actual
model cleanup with a live world/peer, and fresh-segment re-enable:

```sh
env -u WHITEBOX_TEST_GPU_COORDINATOR_URL SGLANG_LIGHTWEIGHT_RUNTIME=1 \
  WHITEBOX_TEST_CLI_SDK=/absolute/path/to/worldkit \
  WHITEBOX_TEST_MANUAL_INFERENCE=1 PYTHONPATH=python \
  python -m pytest --confcutdir=python/sglang/multimodal_gen/test/unit/realtime \
  'python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_room_real_cli.py::test_real_sealed_room_cli_private_native_views_and_cleanup[trusted-model-members]' -q
```

Use a checkout containing the exact paired Harness producer and manual-inference
protocol; an older SDK/Room pin is not equivalent. This lane refuses a configured
GPU coordinator, does not deploy anything, and does not prove GPU throughput,
production seat capacity or shared-image rollout. The maintained CPU owner tests
are `test_whitebox_inference_control.py`, registered in `scripts/ci/zing/run_area_tests.py`.
Their manual init pairing and shutdown/journal contract is documented in the
[owning protocol](../../docs/docs/sglang-diffusion/zing/specs/WHITEBOX_CLIENT_UI_ACTIONS.zh.md#外部-player-按需推理).


Render-v2 sources use the same frozen `session.config.play_seconds` for ready
capabilities, remaining-time status, effective-play expiry and rebind admission.
Changing `WHITEBOX_RENDER_PLAY_SECONDS` affects only newly created sources, not
retained sources or their elapsed budget. Startup/model waiting still consumes the
separate 600-second wait budget. The absolute binding lifetime is
`max(1800, play_seconds + 600)` seconds: 1800 at the default play budget and 7800
at the 7200-second maximum. The shared owner does not add a browser override or
another clock. Owner regressions are
`test_whitebox_render_contract.py` and `test_whitebox_render_lifecycle.py`.

The maintained contract tests cover the default, the former 1800-second boundary,
1801 and the current 7200-second maximum; zero, negative, over-limit and non-integer
configuration remain rejected. This aligns the Harness contract with the regional
runtime merged from `white_box`; it does not establish a two-hour GPU soak result.

Historical validation boundary (2026-10-29): all 14 focused play-budget/lifecycle cases
passed. The complete two-file local run had 37 passes and one import failure in
`test_worker_prompt_boundary_includes_admitted_prefetch` because the borrowed
environment lacks `zmq`; that full gate is not green. Run the unchanged worker
case in the existing complete pinned CPU/image environment before claiming full
owner-file or release acceptance. No dependency stub or test skip was added.

### Standalone role cancellation and NCCL policy

Standalone Encoder, DiT (every rank), and Decoder apply upstream PR #582's
NCCL startup policy before initializing communicators: MIN_NCHANNELS=2 and
MAX_NCHANNELS/MAX_NRINGS/MIN_CTAS/MAX_CTAS unset. Network/interface settings
remain deployment-owned. Startup logs record the inherited values. The integrated
`--static-connection` Decoder retains its launcher policy to match the integrated
Encoder/DiT; managed startup normalizes before validating the role configuration.

An output disconnect cancels only that request. Decoder drains already-sent
tensors and releases late RGB shared-memory references before owner cleanup;
protocol errors and model/transport failures still quarantine the worker.
