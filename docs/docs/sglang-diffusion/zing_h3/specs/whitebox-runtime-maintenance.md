# Whitebox Room cache maintenance

The playtest Web and the independent `/runtime-cache` page observe the configured
Room owner through WTP authentication and the existing Gateway. This does not
inspect arbitrary host processes or browser/CDN caches. No model lease, new world,
resident daemon, cache index or process supervisor is created by management.

## Request and trust boundaries

WTP signs a one-use, 120-second `whitebox-runtime` capability. It binds the
operation, authenticated user and either exact source/world or read-only machine
scope. The existing RTC verifier owns signature, expiry and replay checks; RTC
create tokens cannot authorize maintenance. Cookie claims include the exact
browser origin. Explicit API Tokens never fall back to cookies.

Gateway `POST /v1/whitebox/runtime` (and its established MinWM prefix) requires one
configured Room owner and forwards only to that owner's fixed
`POST /internal/worlds/runtime` with the existing internal token. It accepts no
paths, upstream URLs, PIDs, service names, shell commands or arbitrary CLI flags.
WTP checks each inventory row against catalog world access before displaying it.
No credentials, raw CLI options or object path lists are returned.

Operations: `status`, `inventory` (machine or world); `inspect`, `prepare`,
`remove-preview`, `remove` (world required). Requests use `source_id`, `world_id`,
`operation`, `page`; Room receives the canonical source/world reference. Both
removal operations require the last observed `expected_owner_epoch`; a changed
owner rejects them. Preview is advisory: native leases and pinned/shared object
references are checked again during actual removal. There is no force/prune-all
or stop-process operation. Original deliveries and source files remain outside
cache ownership.

## Managed SDK selection and public prewarm candidate

The newer current-binding SDK/Room observation and standalone public runtime
prewarm are LOCAL extensions; prior management/cache-UI release evidence does not
prove these extensions are installed. Track their consumer contract in the
[WTP runtime/cache integration](https://github.com/seedleap/world-test-platform/blob/main/docs/game-runtime-integration.md)
and [SDK integration](https://github.com/seedleap/worldkit-monorepo/blob/dev/packages/cli/docs/runtime-cache-integration.md).
Actual complete-world managed-context Room/CLI standalone and resident execution
now passes locally with real source pixels and controlled model/storage transport.
Final SDK pin, genuine candidate compatibility evidence and paired live release
remain pending.

Runtime dependencies and World assets have different logical records/categories
but share the configured CLI content store, native reservations, process lock,
capacity and hash deduplication. World preparation already fills the selected
public SDK/Three.js closure. Older embedded World packages do not create reusable
independent SDK retention. Creator release staging, browser profiles and CDN
edges are separate caches; warming one Room does not warm other machines.

For the new SDK candidate on Linux/macOS with Node >=22, the explicit public CLI
operation is `worldkit cache runtime-prefetch --preset /absolute/approved-runtimes.json --pin --json`.
The operator preset supplies fixed complete runtime references (version, manifest
SHA and public URL), not latest/history discovery. It starts no World, browser,
model or GPU job and needs no WTP token. Use the same authorized CLI user/cache
root; this command is NOT a new Gateway/Room management operation. Do not forward
arbitrary CLI flags or invent an HTTP runtime-preset payload.

Independent runtime retention upgrades the same index to schema2. New owners read
schema1; older owners reject schema2 instead of deleting unknown references.
Deploy consistent SDK/CLI/cache owners before explicitly prewarming a shared root.
Pinned runtime or active World references can legitimately keep physical bytes
present after a World record is removed. Existing world inventory is not a list
of every standalone runtime record.

Resolve current authority before a new online World starts. Pointer updates
select a new immutable runtime/projection and reuse unchanged asset hashes;
changed entry/UI files may still transfer. An active session remains on its
prepared runtime. A prior complete World cache cannot claim the new SDK is cached
until its exact current binding is complete. Missing authority remains unknown.
Context belongs to the actual source World/Room CLI startup; H.264/RTC video
players do not request it and no second simulation is started for status.

## Ownership, performance and cancellation

The Node bridge consumes the pinned CLI's public game-cache APIs and official
`serve status` for the existing configured service. Status and inventory use file
existence/length checks; explicit `inspect` verifies SHA. Shared physical disk
bytes and world logical references are separate. Missing CLI/owner observations
are unknown, never zero. RSS, where available, covers only the resident daemon,
not Worker/browser/model memory. The daemon name is a configured owner, not a
claim that all machine processes were enumerated.

For a selected world's `status` or `inspect`, the pinned SDK's public
`readWtpGameCacheSelection` also authenticates the current immutable rendition.
`cache.selection` reports `current`, `stale`, `missing` or `unavailable`, separately
from local file integrity. The selected manifest and canonical runtime binding
must match the retained cache record; a complete older record is not a current
SDK cache hit. Failure to read current authority stays unknown. This bounded
metadata read prepares no resources or World and starts no model. Inventory and
removal remain local owner operations without querying every historical world.
The Web cached-state button requires both complete files and this current-binding
match; the online CLI independently revalidates before a new execution.

Resource-only preparation reuses the existing prefetch callable, budget and busy
lock. It never starts StreamHost or obtains a model. Heavy maintenance uses the
existing Room/pipeline admission boundary: active players/sources and pending
Room joins and admitted RTC startup reservations block maintenance; a new play admission cancels and drains maintenance
before continuing. The per-game join lock also covers the period before a driver
or room exists. No frame-loop, decoder, input or UI presentation callback changes.

Responses are no-store, at most 2 MiB. Prepare progress retains native bounded
NDJSON (8 KiB/line), and no ready claim is accepted without the final native ready
record. Subprocesses are task-owned process groups. Disconnect, timeout and play
preemption terminate and drain only that task's group. Repeated cancellation waits
for the same shielded termination/escalation/reap completion before returning play
admission; prefetch releases its busy lock unconditionally. Shared resident and model
processes are never stopped. A non-idempotent operation is not automatically
retried; an unknown outcome must be reconciled by fresh status.

## Validation and release boundary

`test_whitebox_runtime_management.py` covers capability boundaries, actual SDK
cache/lease/pinned/shared/corrupt objects, factory integration and pending native
Room admission. The Room image's native SDK check stage requires
`WHITEBOX_TEST_CLI_SDK=/opt/worldkit`; generic CPU tests explicitly skip native SDK
cases if the SDK is absent. Existing CLI integration covers real worlds/Chromium
without a GPU. The Room browser image also installs Debian's `ffmpeg` package
(`ffmpeg` and `ffprobe`). Multiplayer delivery admission validates the original
recorded segments through the frozen SDK's `ffprobe` reader before starting a
browser. The image build encodes and re-probes a tiny MP4 using the same duration,
frame-count and dimensions fields; this also checks libx264 and MP4 decoding.
A missing probe can surface as `CREATOR_MULTIPLAYER_EVIDENCE_INVALID`; check the
Room image's media tools before treating that code as corrupted author evidence.
For deployment acceptance, run the frozen SDK's public `loadDelivery` against the
unchanged sealed multiplayer archive in the candidate image, then verify native
startup. Build-tool checks alone do not establish complete delivery acceptance.
These tests are not GPU FPS, E2E presentation or multihour memory proof. Both 480
and 736 model performance/memory remain unverified for this added management
surface; no model kernel or steady frame path changed.

Deployment requires the reviewed shared SDK pin, SGLang white_box merge and the
existing Room/Gateway image builds. WTP's matching management Web is a separate
shared-source deployment. A local passing UI or source merge is not deployed
cache/stream acceptance.

## Original control operation cleanup

A failed controlled Room startup becomes terminal `failed` with
`cleanupComplete: true` only after the original factory acknowledges its private
CLI process/view/temporary-file cleanup and RoomRuntime finishes the original
RoomLog. Typed receipts bind that proof to the operation ID and owner epoch;
only the existing public diagnostic allowlist supplies `errorCode`. Local log
flush/reap shares one task with background archiving and does not wait for OSS
publication. An unconfirmed factory, cancellation, failed local cleanup or
changed owner remains `unknown`.

An explicit replay of the same `leavePlayer` or `retireRoom` may retry cleanup on
the retained original entry/Room. It cannot repeat create/join, change the caller
or binding, or start another cleanup while the original task runs. A successful
receipt is replayed without another resource action. `getOperation` and
`getAdmission` remain observations. WTP must resend the exact original leave
invocation for an explicit leave retry; an admission query alone does not retry
cleanup. Missing process-local entries never establish a cleanup receipt.

These receipts are process-local. After an owner restart, the old directory and
operation binding still fence replacement creation, but the new process cannot
turn an old `unknown` into failure from absence or infer the old factory's
cleanup. Source deployment alone does not settle a historical unknown operation.
Preserve that operation and obtain proof from its original cleanup owner; do not
rewrite platform operation state or replay create to bypass reconciliation.

### Explicit retirement of a failed original create

The directory now has a narrow maintenance path for a stopped original owner
whose **reviewed sealed-loader refusal occurred before the driver returned and
before any player admission or model lease**. It does not recover arbitrary
lost executions. A diagnostic code, expired lease, `ended` route, empty process
list, or zero capacity is insufficient. If the original caller/operation record
or its exact game/owner-epoch association is unavailable, this path refuses to
create a substitute record. Unknown execution or remote-resource responsibility
remains unknown.

After a managed Room confirms SDK, media and local archive cleanup, the Room
records its exact execution on the original create operation in the coordinator.
The Gateway can replay the positive close response after the Room process or
its routing record disappears. An ended route or an existing archive alone
cannot create this receipt. Roll out support on the coordinator and Gateway
before depending on persistence from the Room.

`owner_retirement.verify_retired_create` validates one explicitly approved audit
against the exact original invocation, source bytes, timestamped startup
diagnostic and stopped Docker instance. The independent source review must
establish the failed-before-driver boundary for those exact bytes and frozen
input; a hash check cannot establish this semantic fact. The reviewed closure
includes the SG entrypoint/driver/runtime/control/diagnostic code and the SDK
bin/main/room-command/room-runner/sealed-loader/delivery-verifier chain, listed in
`STARTUP_SOURCE_PATHS`. Do not infer the original failed archive's byte hash from
a later successful download when its temporary copy is gone.

The deployment host tool
`deploy/whitebox/confirm_retired_create.py` reads the retained old
container using Docker inspect, bounded timestamped logs and exact-file tar
streams. It never starts or stops it. Only the existing coordinator container
receives these observations on stdin; its existing parsed Redis configuration
stays private. No browser, WTP or internal HTTP endpoint accepts retirement
evidence. The original privileged deployment operator is the trust boundary;
possession of an arbitrary JSON file or a business capability is not approval.

Prepare a private audit with these exact fields, then independently review its
bytes and record their SHA-256 before invoking the tool:

- `schema_version: 1`, `kind: "native-create-owner-retirement"`, `caller`
  (`serviceId`, `subject`), `operation_id`, `parameters_sha256`, `game_id`,
  `owner_id`, `owner_epoch`, `container_id`, `image_id`, `started_at`, `stopped_at`.
- `invocation_sha256`: canonical JSON of the original stored `createContext`,
  `videoSpec`, and `cliFps`; `control_value_sha256` and `route_value_sha256`:
  hashes of the exact original stored JSON bytes. Preserve each caller's distinct
  operation and game even when their publication and video parameters match.
- `epoch_observation`: `container_id`, `owner_id`, `owner_epoch`, `at`,
  `evidence_sha256`, binding an observation within that container lifetime.
- `stop_receipt`: `container_id`, `action: "stop"`, `at`, `finished_at`,
  `evidence_sha256`. A positive completed lifecycle action is required; the tool
  separately verifies the same instance is exited, PID zero and restart disabled.
- `startup_boundary`: `classification: "sealed-loader-before-driver"`,
  `error_code: "CREATOR_MULTIPLAYER_EVIDENCE_INVALID"`,
  `review_evidence_sha256`, and `source_paths`/`source_sha256` maps containing
  exactly `STARTUP_SOURCE_PATHS`. Paths identify actual files in the original
  container; do not guess from another checkout. The three referenced evidence
  files must be present and match their hashes.

The host's dry run is read-only and publishes no business result:

```bash
python3 deploy/whitebox/confirm_retired_create.py \
  --audit /private/original-create-retirement.json \
  --approved-audit-sha256 APPROVED_AUDIT_SHA256 \
  --epoch-evidence /private/original-epoch-evidence.json \
  --stop-evidence /private/completed-stop-receipt.json \
  --source-review /private/reviewed-startup-boundary.md \
  --coordinator-id EXACT_EXISTING_COORDINATOR_CONTAINER_ID
```

Only after the exact audit and deployment are approved, the same invocation with
`--apply` asks the original `RoomDirectory.confirm_retired_create` to compare the
control record **and route** atomically and append a terminal proof to the
original control key. This introduces no registry, cancellation API or capacity
counter. The terminal control key has no automatic expiry; routing/session TTLs
are unchanged. Redis must support the two original keys in one atomic script.
A cross-slot or other store error remains unconfirmed; never split the compare
and write or rename the namespace to bypass it. The tool does not retry an
uncertain write. An explicit repeat with the same reviewed audit recognizes an
already accepted identical proof; changed proofs conflict.

Authenticated `getOperation` and a repeated original `createRoom` then return
the existing exact operation/hash contract: `failed`, `cleanupComplete: true`,
`errorCode: "ROOM_OWNER_RETIRED"`, without dispatching to a new Room. WTP's
existing reconciliation settles its original journal and confirms retirement;
the Console observes that original request. No session-storage clearing or new
request is required. This proof describes retired execution responsibility,
not successful factory cleanup, completed archives, or gameplay acceptance.

Focused tests are in `test_whitebox_owner_retirement.py` alongside the original
`test_whitebox_room_control.py`. They cover malformed/absent evidence, caller and
source isolation, route changes during CAS, repeated proof idempotence, original
Redis-adapter arguments, internal HTTP directory consumption and no runtime
dispatch. Host Docker reads and cloud storage are mocked locally; these results
do not establish production Redis topology or authorize a production proof write.
Deploying source alone still does not settle old unknown operations. The reviewed
Gateway and Coordinator must both understand the stored proof before an explicit
maintenance apply; the Room runtime and SDK pin need no change for this path.

## Multiplayer CLI source adapter

`sglang.multimodal_gen.runtime.realtime.whitebox.worldkit_room:worldkit_room`
provides an opt-in RoomRuntime driver for sealed multiplayer deliveries. The
existing single-player `cli_room:cli_room` driver remains available. Select the
driver in the trusted Room deployment configuration; browsers cannot choose a
plugin, source path or CLI flags.

The existing `game_id` directory still selects one Room Host/pod. WTP's playtest
and “加入联机” entry points and the planned independent game console share this
routing. A game creates one SDK Room and one authoritative world. Each admitted
player gets a private SDK member, media connection, UI state and input binding;
starting a player does not start another authoritative simulation.

For Linux/macOS CPU Room hosts, configure Node >=22 and a compatible Worldkit
installation containing `room create/join/inspect/resume/leave/close`,
`@worldkit/stream-host/session`, and capability-based player streams. Its Node
runtime must resolve `tsx` from the configured SDK checkout. The historical
Worldkit 0.2.15 deployment pin does not establish these capabilities. Upgrade and
release pairing remain separate from this source change.

```bash
export WHITEBOX_ROOM_DRIVER=sglang.multimodal_gen.runtime.realtime.whitebox.worldkit_room:worldkit_room
export WHITEBOX_CLI_ENTRYPOINT=/absolute/worldkit/packages/cli/bin/worldkit.mjs
export WHITEBOX_MAX_PLAYERS=8
```

`WHITEBOX_MAX_PLAYERS` is the existing trusted deployment admission bound and
private authentication-pool size (1–128, within the SDK authorization limit).
The driver reports the sealed game's exact `maxPlayers`; insufficient deployment
capacity rejects room startup instead of truncating the game. Random subjects,
actor IDs and credentials belong to the Host and never become a game metadata
roster. Existing `player_id`, `stream_id`, `game_id` and owner epoch contracts keep
their original meanings.

The adapter reuses the existing receipt-verified `prepare_delivery` downloader
and its `WHITEBOX_CLI_PLATFORM_URL`, `WORLD_TEST_API_TOKEN` and delivery-deadline
configuration. It starts the foreground CLI with that exact local archive and
`--service off`, then compares `archiveSha256`, `worldBuildHash`, `runtimeHash`
and `sourceHash` with the downloaded receipt. The room's `contentHash` matches
`sourceHash`. This first adapter uses sealed archives, not the separate CDN
playable/game-cache or resident-room path. Area source options are rejected.
Source selection is frozen at room creation; a later publication cannot replace
an active room. An upstream publication-selector extension must reuse the agreed
WTP/SDK reference contract rather than inventing another content record here.

Private authorization, member and session records stay in a canonical, private
temporary directory. Member files contain a minimal SessionConnection; the Node
bridge uses authenticated public `readSession` and `streamEndpoints` to obtain
live descriptors. Player WebSockets use capability subprotocols, with no query
tokens. Controller admission and the exact `input.binding.applied` ACK precede
input. Native pixels, source receipts, UI events, frame credit and composition
continue through the existing NativeCLIView/renderer consumers.

An interrupted join retains its original request ID, credential and actor slot.
RoomRuntime calls the driver's `cancel_join` rollback when startup fails before
publishing a stream entry. Rollback reconciles that same journal request before
leaving. A `cleaning`, failed or unknown cleanup remains reserved; another player
cannot take its slot. Direct leave closes local media/UI before retirement, and
room close cancels outstanding startup/recovery work and terminates its owned
foreground CLI. One member's normal leave does not close a peer's view.
The CLI can initially acknowledge leave with `cleaning`. Within the existing
operation deadline, the adapter queries the original Room's public inspection
and releases capacity only after the exact Room identity, member session and
connection epoch report `left`. Blocked cleanup, a changed epoch, a missing Room
or an unknown result retains the reservation. A missing member record or an
automatic empty-Room shutdown alone is not a terminal cleanup receipt.

A failed join can reserve its model parent before RoomRuntime publishes a
player entry. Its original rollback now retains a bounded, per-stream retirement
receipt when either the driver confirms that exact attempted member has left,
or the same failed join's original empty Room completes its owned shutdown.
The latter requires a successful cleanup result from the same Room object; an
absent/replaced Room or a failed shutdown cannot grant capacity. The model
parent still completes its normal verified release using that receipt. With a
live peer, rollback never closes the shared Room and requires the exact member's
driver receipt. Repeated cancellation drains this cleanup before returning.
The failed admission remains unsuccessful, while an explicit leave can confirm
its retirement without inventing a successfully published member ID.
Rejections before any SDK create/join dispatch retain `not_created`. Concurrent
duplicates and retries of a dispatched but unconfirmed join cannot acquire that
receipt; the original stream remains fenced until its cleanup is confirmed.

The driver's explicit `resume(player_id)` replaces only a suspended member's
media session and keeps its actor/member/viewport identity. A lost resume reply
is reconciled through the atomically replaced private member file and live SDK
session, without automatically issuing another resume. An unchanged or
unavailable file remains unconfirmed. Closing an old view cannot leave its
successor. This is a driver-level recovery contract: the existing public gateway
connection grace is unchanged. No new RoomRuntime recovery endpoint or automatic
suspended-media recovery routing is introduced by this adapter.

Three integration items remain open. First, the current CLI join journal does not
expose a request-specific terminal cleanup result for a rejected admission whose
member file was never published. Replaying the failed request returns the same
failure. A failed join in an otherwise empty Room can use the original owned
shutdown receipt described above. When peers keep that Room alive, the adapter
still retains the unresolved reservation; a public CLI/Room reconciliation
contract is needed for request-specific release in that case.
Second, a service-level suspended-media recovery entry must agree with the
existing source/model epoch and late-frame fences before invoking driver resume.
Third, the service has no separate trusted frozen-publication create context or
request-specific create-result lookup before a `game_id` is available. The
current factory accepts only `world_id`, `spec` (VideoSpec), `trace_id`, optional
`cli_fps` and rejected nonempty `source_options`; those fields must not be
repurposed as a trusted content envelope. WTP service authorization, invitation
claims and this create-operation contract still require a shared fixture before
wiring. Existing internal Gateway-to-Room Bearer authentication alone does not
establish those business permissions. These gaps must not be presented as a
complete production integration or production fault recovery.

The model coordinator and renderer/GPU leases keep their existing policy. A
shared multiplayer room does not increase model-worker concurrency: one deployed
model slot still serves at most one active model session, and exhausted capacity
must be rejected. This change does not allocate GPUs, publish a runtime, change
WTP pins or deploy a service.

Focused CPU checks use the established lightweight runtime and realtime test
boundary:

```bash
SGLANG_LIGHTWEIGHT_RUNTIME=1 PYTHONPATH=python python -m pytest \
  --confcutdir=python/sglang/multimodal_gen/test/unit/realtime \
  python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_room_cli.py \
  python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_room_transport.py
```

The CLI-boundary tests use subprocesses and sockets with a fake CLI plus a real
RoomRuntime consumer. Transport tests resolve the actual SDK's public Node APIs
when `WHITEBOX_TEST_CLI_SDK` points to a compatible checkout. Keep those results
separate from real sealed-world/Chromium pixels, real model/GPU output, two
physical devices and 300-second performance acceptance. Read the driver lifecycle,
then the Session bridge/NativeCLIView changes, then these direct-consumer tests
when reviewing this integration.

The additional opt-in `test_whitebox_room_real_cli.py` uses the SDK's exported
compiled-room fixture, real multiplayer playtest/sealing, the actual room CLI,
Chromium, authenticated Session reads and NativeCLIView. Only WTP download and
the UI renderer are substituted: it observes per-player UI events but does not
prove composed output. It checks independent native pixels/input receipts,
continued peer execution after leave, and owned-process teardown. Run it with
`WHITEBOX_TEST_CLI_SDK=/absolute/worldkit` and the same lightweight pytest command.
Record an actual failure as an acceptance blocker with the tested SDK checkout
identity, including local changes; do not skip it or equate the fake-CLI suite
with successful real admission. Current candidate results belong in the PR.


## Multiplayer failure logs

Room control and member model failures reuse Python container logging with the
JSON event `whitebox.player.failed`. The existing SDK
`worldkit.production.blocked` stderr/FD path remains separate and unchanged.

The originating boundary reports `room.start`, `room.retire`, `member.admit`,
`member.rollback`, `member.leave`, `member.media-start`, `model.connection` or
`model.release`. Querying a blocked operation does not reissue work or produce a
new failure event. Admission failures retain the existing bounded `phase`:
`model_reserve`, `sdk_join` or `member_identity`. Expected cancellation and normal
model shutdown produce no failure event. Diagnostic sink failures cannot replace business results or
interrupt cleanup.

Correlate WTP `admissionOperationId` with the model log's `operationId`. Available
fields retain the original game/owner/publication, member/player, parent model
session/generation and worker ID/epoch. `operationId` identifies the admission,
create-context operation or lease trace. Before allocation or SDK admission,
unavailable identities are omitted. A model reservation failure
must not invent a worker or imply that cleanup succeeded. Each worker's existing
single-active-session policy remains unchanged.

Logs retain the safe error code, exception class, two cause levels and the
original player/model identities. The existing logger still supplies the enabled
level and native call-site record; this diagnostic does not invoke its shared
Handlers. The sanitized record goes to the same container stderr collector as a
single nonblocking pipe write. Credentials, caller subject, reference/init bytes,
prompt, raw exception text and GPU connection URLs are not serialized.

This is best-effort diagnostics, not a delivery contract. A private marker avoids
reporting the same exception again across owners. Only FIFO pipe sinks are
supported. An already nonblocking pipe uses a duplicate; a blocking Linux stderr
uses an independently opened `/proc/self/fd/2` with `O_NONBLOCK`. The original
stderr flags are never changed. Missing/unsupported sinks, broken/full pipes and
records larger than `PIPE_BUF` are dropped, without partial rows or retries.
Blocking local terminals/files and platforms without a safe nonblocking
reference are deliberately skipped rather than falling back to synchronous
logging. This rule only changes the added member diagnostic, not existing model
error, CLI or access logging.

There is no diagnostic writer thread, shared Handler lock, queue, cache, timer,
flush, shutdown hook or new dependency. Disabled logging returns before reading
member/model context. Sink errors cannot replace the original operation result
or interrupt cleanup. Normal frames and inputs do not call this helper.

CPU regressions in `test_whitebox_room_control.py`,
`test_whitebox_model_binding.py` and `test_whitebox_player_media.py` cover returned
blocked/unknown states, failure identity/redaction, two independent model owners,
normal shutdown, duplicate capture, disabled logging and broken/full pipes. A native
Linux subprocess regression also verifies unchanged stderr flags, intact output,
unsupported sinks and process exit while an existing Handler would block. They use controlled
model transport; they do not prove concurrent physical GPU execution or deployed
cloud collection. Verify the actual deployed sources and cloud log visibility
separately before claiming online coverage.

## Playback capacity and resident CLI workers

Room playback admission uses the shared `WHITEBOX_ROOM_PIPELINE_CAPACITY`; this
is a per-owner source bound, not a per-user quota. A larger authorized CPU test
profile also needs enough `WHITEBOX_MAX_ROOMS`, dedicated native RTC endpoints
and public UDP ports. The managed CLI stream adapter passes the same capacity
as `--workers` when resident service mode is enabled. A second source must not
wait behind the first game's entire lifetime in a one-worker CLI queue.
Foreground CLI mode is unchanged. Existing resident services must be drained
before changing this configuration; restarting a source must not interrupt peers.

Keep model-slot capacity and its per-user lease policy unchanged. More CPU
playback sources do not create additional GPU or role-pool model slots. The
default single-session release policy is unchanged; a test profile with multiple
native sources is an explicit deployment override. Update its mounted policy,
configuration digest and actual Room environment together, preserving release
identity checks. Verify simultaneous video in two pages, independent input,
exit of one while the other continues, and a capacity refusal without eviction.

## Player input inactivity

Room ends an individually idle PlayerSession after 60 seconds without effective
input. Owner heartbeats, media, neutral input refreshes and Prefill replies do not
renew this timer. Held-key renewals, camera input and completed game UI actions
count; native Player controls are correlated with their SDK input receipts.
Timing starts when the player source is ready; it never extends the independent
absolute player deadline (for example 1800 seconds or 7200 seconds).

Operators can atomically replace `WHITEBOX_ROOM_SPOOL/.runtime/player-idle.json`
on the existing persistent Room volume:

```json
{"player_idle_timeout_seconds": 60}
```

The Room process rereads this one value every five seconds, including for existing
players. `0` disables input inactivity expiry; it does not disable the absolute
player deadline. Invalid/missing updates retain the last valid value (60 on first
boot). This is runtime configuration, not a code overlay; no image rebuild, Room
restart, GPU restart or frontend change is required. A shorter value applies to
existing idle players on the next check. Already-ending players cannot be revived.
Use the same value for all Room owners in one regional lane.

An expiry enters the existing verified player cleanup path. Only confirmed SDK
and model retirement makes the Slot free; failures remain quarantined. Other
players in the shared GameRoom, model caches and background recording uploads are
not cancelled by the player's inactivity. `PLAYER_IDLE_TIMEOUT` and its original
stream/trace identity are logged for diagnosis.

New native WebRTC connections allow up to 45 seconds for signaling and ICE
negotiation; the native ICE failure threshold is 40 seconds. Successful
connections proceed immediately. This is separate from player input inactivity
and does not extend the existing reconnect lease. A permanently blocked media
path still fails; this budget only tolerates slower connection establishment.


### Retired single-player source reconciliation

A source may finish and leave the live registry before its Player operation is
explicitly closed. Registry absence alone remains `lost`. When the original owner's
per-source SQLite journal proves the same user, source session, source
epoch and game, with `source_state=ended` and `cleanup_state=released`, an explicit
close can durably acknowledge cancellation without recreating or restarting it.
The retained operation keeps the source outcome separately; a failed source is
not reclassified as successful gameplay. Missing, mismatched or unknown cleanup
receipts cannot confirm release. Reads use the original journal key and existing
bounded CAS, without scanning the history or changing publication identities.

Regression: `test_retired_source_close_requires_original_durable_release` in
`test_whitebox_player_control.py` exercises released, unknown, missing and foreign
scope receipts with real SQLite and concurrent close calls. This CPU evidence does
not prove GPU/model behavior or a deployed runtime update.

A Room restart does not invalidate an already durable release receipt for the
original operation: terminal reads and explicit close may use that receipt after
validating the stored original owner epoch, caller, source epoch and game. They
retain the original execution and source outcome; they never recreate, attach
to or resume that source. Missing, foreign or uncertain release evidence still
returns lost across a restart. The durable SQLite regression above runs both
within one owner lifetime and after closing/reopening the owner store.

### Prepared-area audio SDK release input

Room builds pin Worldkit dev commit `53539878684a23ebebac3305d8e133e120c392ba`
(reviewed SDK PR #1067). The Stream Host exposes the sealed audio manifest and
resources for prepared areas through the selected playable-file allowlist and
length/hash checks, matching the existing complete-world audio transport. This
pin change does not upgrade frozen authored worlds or WTP task SDK identities.
Image build, test-lane deployment and actual playback acceptance remain separate
release steps; source integration alone does not establish remote audibility.

### Project-defined input (`player-input/1`)

The native SDK remains the semantic schema, authorization and consumption owner.
Room's existing SDK relay preserves `input.semantic.open/applied/batch/cancel` and
binding/receipt/error bytes, on its original generation-fenced control socket.
Open/applied negotiate a lease; neither is gameplay activity. Gameplay packets
(`input.state`, `input.action`, `input.semantic.batch`) have a 200ms relay residence
and send budget. Expiry closes the old socket/lease; nothing is resequenced or
replayed after backpressure. This budget is independent of the SDK's own queue and
lease, and does not expire a slow initial binding negotiation.
The public control relay reads ingress independently, retains at most 64 messages
and 1 MiB, and timestamps gameplay on receipt. A queued gameplay deadline still
expires while a preceding handshake send or archive observation is blocked. Sends
use the remaining residence budget, including the observation before dispatch.
Both control relays register pending intent before sending so a fast committed
receipt cannot overtake its correlation record. This records an observed intent,
not successful delivery: only a matching SDK receipt can count as activity.
Both relays independently expire queued gameplay while an earlier handshake or
archive write is blocked. A handshake without queued gameplay keeps its original
negotiation timeout.
Handshake-only negotiation and media forwarding
remain outside that gameplay timer.

Activity requires a matching SDK committed receipt: client, generation, sequence
and state revision. Only the SDK's `meaningful:true` decision counts, using the
original correlated send time; neutral renewal, cancellation, unmatched, duplicate
and late receipts cannot extend that time. Effective sustained controls are
confirmed by their SDK owner, without Room interpreting custom game fields.
All seven semantic envelope types enter the existing viewer input archive with
source/session/epoch identity. That network observation is not a complete World
execution recording; exact replay uses committed SDK/Episode evidence.

The existing public SDK session route also proxies only frozen `player-input/` resources, using the original viewer capability on both Gateway and Room hops. Session publication rewrites the input resource root to the public origin; no standalone CLI administrative token is published. Resource reads reject unsafe paths, wrong/revoked capabilities and a changed source epoch, use bounded reads and never follow redirects. This route is transport support, not a claim that replica/multiplayer semantic consumption is implemented.

The ordinary single-player CLI now returns a separate scoped `inputSession` in
ready when its sealed World declares input. The original CLIRoom publishes this
session through its source route, and SourceSession forwards `sdk_input_session`
on ready. The public Player attaches the same SDK input owner while retaining its
original media and action owners. It does not fabricate a multiplayer binding or
start another World. The CLI administrative token stays private. An input viewer
cannot reset the original session or read private model-context resources.
Its input-only public descriptor omits the separate SDK audio resource descriptor;
audio continues through the original Whitebox audio owner. This permits audio and
custom controls to coexist without publishing private loopback resource URLs.

CLIRoom renews the same viewer grant every 60 seconds while the original view is
alive, and cancels/drains that task on close. Renewal failure closes the view.
Renewal teardown runs inside the original cleanup guarantees, so cancellation there
still reaches view and CLI cleanup. View close callers share one shielded cleanup
task: cancelling renewal cannot interrupt it, and Room retirement waits for that
same task. Model-context cleanup failure still drains sockets and the UI renderer.
A directly consumed CLI ready input grant is
separately renewed by its original CLI owner until that stream ends.
This administrative grant lifetime is independent of semantic observations,
gameplay activity and the SDK's short input lease. Both the generic public control
tunnel and SDKControlRelay share the 200ms gameplay send budget above; handshake
messages are excluded. SDK input-resource reads are bounded to 32 MiB per resource,
do not follow redirects and recheck the original viewer/epoch after reading.

WTP accepts input-declared content only when the selected SDK facade exports the
matching `PLAYER_INPUT_PROTOCOL`; an older facade returns
`PLAYER_INPUT_UNSUPPORTED` before allocation. Local verification includes an actual
CLI child process through the original Gateway/Room capability/resource/control
routes to the original SDK World committed receipt. Public React/RTC fixture tests
verify input-owner discovery on ready. These are source/protocol proofs, not a
live GPU/model presentation or production deployment result. Multiplayer semantic
World consumption remains unsupported.

CPU verification uses the existing lightweight environment, without model imports:
`SGLANG_LIGHTWEIGHT_RUNTIME=1 PYTHONPATH=python python -m pytest --confcutdir=python/sglang/multimodal_gen/test/unit/realtime -o asyncio_default_fixture_loop_scope=function -q python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_player_idle.py python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_sdk_control_relay.py python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_player_transport.py python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_cli_lifetime.py`.
The paired SDK's `packages/cli/tests/player-input.browser.test.ts` additionally
uses `WORLDKIT_PLAYER_INPUT_SGLANG` and `WORLDKIT_PLAYER_INPUT_PYTHON` to verify the
actual cross-repository CLI tunnel. Use the existing shared Python environment;
point `PYTHONPATH` at this selected source rather than installing another copy.
This source support requires the paired SDK input protocol. It does not update a
frozen SDK/browser pin, publish an image, or establish live model acceptance.

### Live input archival and backpressure

Room-owned game logs always use the existing bounded fail-open owner. Streaming
Rooms retain their diskless memory journal; non-diskless Rooms use
`AsyncRoomJournal` even when the media archive manager selects synchronous or
strict recording. Direct offline `RoomLog` users keep their selected writer.
Media recording configuration must not delay live SDK control.

All Room events retain one original sequence and ordered journal. Existing
limits remain 64 KiB per event, 128 pending events and 8 MiB queued bytes. Full
or failed archival marks the recording failed through its original manifest,
without waiting for disk or queue capacity on the input route. Normal finish
still drains that owner, seals the original input file and reaps its child.

Intent correlation still precedes forwarding, so an immediate SDK receipt can
confirm it; raw input alone is not activity. Both relays retain their 200ms
gameplay deadline: an actually stalled SDK send closes and retires its queue
without replay. There is no unbounded per-packet task or parallel log writer.

`test_whitebox_live_input_archive.py` covers both old manager modes, forwarding
with a full archive, fast receipt correlation, real ordered journal finalization
and explicit failed overflow manifests. Run it with the lightweight CPU pytest
entry above. The existing control relay expiry tests verify the unchanged
transport protection; these local tests do not measure production disk latency.
