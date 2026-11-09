# Official Worldkit CLI integration

Status: implemented and independently reviewed; both review findings are fixed
with native regression coverage. Shared SDK pin, exact CI and `white_box`
integration are pending. Source changes do not deploy a model or Room.

The Room should invoke `worldkit stream` for current playable publications. The
CLI owns preparation, verified runtime/assets, managed game cache, resident
browser worker, readiness, connection liveness and teardown. SGLang continues to
own its single Room/player, two-frame native credit and model/reconnect lifecycle.

- Default resource transport is `auto`; explicit CDN and OSS routes use the same
  CLI. Explicit `archive` downloads a sealed archive, then the CLI verifies and
  extracts it once. The download bridge compares the actual CLI archive identity
  with the original receipt; it no longer extracts a second UI copy.
- Explicit CDN requests that fail with `WTP_PLAYABLE_NOT_READY` retain bounded
  platform publication preparation: one reconciliation POST and identity-fenced
  polling under the original startup deadline, followed by one CLI attempt. An
  unknown POST result fails without resubmission. Warm starts add no metadata
  request, and resource-only prefetch remains read-only.
- CLI ready JSON supplies a same-host `localResources.uiRoot`, valid for the
  session. SGLang verifies its manifest SHA against `uiBundleHash`, then the
  existing UI owner verifies declared assets. Missing handoff fails clearly;
  there is no guessed SDK-private directory or unverified fallback.
- Default resident mode is `auto`, name `whitebox`; its existing idle retention
  is independent of session lifetime. Managed game cache defaults to the SDK
  root. An old `WHITEBOX_CLI_CACHE_DIRECTORY` is preserved and the new game cache
  uses its `games/` child. Explicit `WHITEBOX_CLI_GAME_CACHE_DIRECTORY` or
  `WORLDKIT_GAME_CACHE_DIR` selects the same root for prefetch and streaming.
- Startup deadline defaults to 420 seconds; direct stream idle grace is 60,
  protocol liveness is 30, hard duration is unlimited. Environment overrides are
  validated before launch. Zero disables the corresponding runtime policy.
  A healthy connection without input stays connected; Room's browser heartbeat,
  owner-idle and reconnect grace remain independent policies.
- Epoch-less normal `session.ended` (`stopped`, `idle-timeout`) ends the source
  iterator normally. Explicit wrong epoch, reset, source failure and missing data
  epoch still fail. Independent media/control socket closing has a bounded
  one-second terminal-message grace; ordinary unacknowledged socket loss fails.
- Startup metadata retains supported CLI preparation/stage timings plus actual
  CLI-ready elapsed time. These spans are not player-presentation latency.
- `WHITEBOX_CLI_DOWNLOAD_CONCURRENCY` controls resource-only prefetch; stream
  transfer uses the SDK default. The previous legacy cache is preserved, so the
  first new game-cache preparation may redownload objects and retain old disk
  usage. Transport establishment retry policy follows the shared SDK rather
  than the removed private downloader; some transient failures may now fail.
- Normal Room shutdown uses the authenticated public session DELETE, then waits
  for CLI completion so resident request/temp cleanup finishes before reuse.
  SIGTERM/SIGKILL remain the owned-child fallback; the shared daemon is not killed.

Verification includes real local sockets, compiled native World/Chromium, trusted
local HTTPS publication, prefetch followed by zero-download stream starts, same
resident instance across two Rooms, verified UI assets, removed session files and
per-request UI roots, and existing native two-game Room integration. These are CPU
source/integration checks, not GPU throughput or player-presentation measurements.

The SDK must first expose the handoff on an ordinary CI-validated shared revision.
A candidate may be staged in `benchmark/whitebox/sdk-revision.txt` before that
release prerequisite, but it must not be pushed for integration or deployed until
its shared ancestry and complete SDK CI are verified. Then run exact SGLang PR CI
before merging. The old pin must not be used to deploy
these new flags. Explicit v2 area resources/author transitions remain separate;
this adapter starts complete worlds and does not convert partial resources into a
complete-world descriptor.

## Service-controlled multiplayer integration (2026-10-28)

The follow-up `room_control.py` and `room_control_runtime.py` implement the
service receiver agreed in WorldKit's
[production console plan, section 3.2](https://github.com/seedleap/worldkit-monorepo/blob/51c90d210/docs/superpowers/plans/2026-10-28-multiplayer-production-console.md).
This is a source candidate, not a deployed or accepted multiplayer console.

- Gateway accepts only `POST /v1/whitebox/room-control` and its fixed
  `/backends/minwm` alias. Configure `WHITEBOX_ROOM_CONTROL_SECRET` (at least
  32 UTF-8 bytes) and comma-separated `WHITEBOX_ROOM_CONTROL_SERVICES` on the
  Gateway. With neither configured, control returns 503; partial configuration
  fails startup. Existing RTC v1/v2 purposes and keys are unchanged.
- v3 tickets bind the exact request bytes, action, allowed service, internal
  subject, audience, expiry and nonce. Nonce replay protection has the original
  process-local scope; cross-Gateway retries are additionally fenced by the
  original directory association and Room operation receipts. Browser fields
  cannot supply the verified caller or choose an upstream origin.
- The original coordinator's Room store uses CAS to associate the immutable
  WTP invocation with one random UUIDv4 game ID. Query never allocates a route.
  This association is retained for the configured Room maximum plus 24 hours;
  WTP's original journal must reject expired operations before issuing another
  control ticket. This does not add a second business-operation journal.
- Room control reuses `_start_room`, Room log, heartbeat, player join/leave and
  `_end`. The production Room entrypoint selects `worldkit_room` only when the
  original control owner supplies its verified `create_context`; ordinary joins
  retain the existing `WHITEBOX_ROOM_DRIVER` factory, arguments and lifecycle.
  Keep that deployment default rather than globally replacing the single-player
  driver. No player field or game name selects a factory. Create consumes no
  player slot. HTTP disconnects do not cancel or reissue accepted work. Queries do not dispatch work. A new
  process cannot start the previous owner's game after its epoch was committed.
  Room release images must copy the tracked `whitebox_room_server.py` entrypoint
  and `benchmark/whitebox/run_service.py`, in addition to the `whitebox/` modules;
  inheriting an older entrypoint would omit this production composition.
- `admitPlayer`, `getAdmission`, `leavePlayer` and `retireRoom` retain original
  IDs/digests and exact caller/member/owner bindings. Leave fences Session
  publication first; unknown cleanup stays blocked. Failed Room cleanup keeps
  the existing Room and each unconfirmed player view available to the original cleanup owner, including repeated cleanup attempts. A retired
  execution cannot publish an old admission as claimed.
- `WorldkitRoom` receives `create_context` and separate runtime-owned
  `verified_caller`. Controlled creation exclusively calls the public
  `@worldkit/stream-host/wtp` `downloadWtpOperationDelivery` extension. Configure
  `WHITEBOX_FROZEN_READ_SECRET`, `WHITEBOX_FROZEN_READ_SERVICE_ID` and the existing
  `WHITEBOX_CLI_PLATFORM_URL` on the Room. The helper signs each exact
  receipt/archive request with its own nonce and verifies frozen receipt/archive/
  runtime expectations through the SDK. Missing public SDK support or service
  configuration fails closed; it never switches to inventory/latest or a
  personal WTP token. Existing non-controlled callers retain their sealed
  publication download path. CLI ready must match the frozen play configuration.

The WTP sender byte fixtures come from
`seedleap/world-test-platform@fa979d90e2046d41b98453aeeaf86e9e728704a4`.
CPU tests exercise original bytes/digests, authorization, atomic routing,
Gateway/coordinator/Room ASGI hops, asynchronous create/query/admission/leave/
retirement, restart fencing, failed cleanup and the actual Node service-download
bridge against a public-package test double. Register the tests in the existing
maintained CPU gate (`scripts/ci/zing/run_area_tests.py`).

## Native public player transport follow-up

The same Gateway and Room now relay the original SDK Session/control/native
pixels/UI wire. No encoder, renderer, physics clock or membership owner is added.
`player_transport.py` installs only fixed `/v1/sessions/<SDK-session-id>` viewer
routes and their original authenticated Room hop. The original Room directory
store retains the SDK session -> game/owner-epoch/stream association with CAS;
ended, expired or replaced owners cannot resolve it. Producer, reset, settings,
DELETE and arbitrary upstream paths are not exposed.

Configure the same exact canonical origins on Gateway and Room:

- `WHITEBOX_ROOM_PUBLIC_ORIGIN`: reachable Gateway authority for SDK transport.
- `WHITEBOX_PLAYER_ORIGIN`: the independent Next execution origin (`/frame`).
- `WHITEBOX_CONTROL_ORIGIN`: the WTP trusted control origin. Its hostname must
  differ from both public transport and execution hosts so a viewer connection
  cannot carry the host-only WTP management cookie. Public transport and frame
  may share their dedicated origin through explicit ingress routes.

Gateway's outer CORS middleware uses the exact deduplicated pair of the existing
`WHITEBOX_BROWSER_ORIGIN` (legacy WTP browser) and the installed, validated
`PlayerTransportConfig.player_origin`. Execution-origin canonical validation and
legacy browser-origin configuration semantics remain unchanged; no new origin
setting is required. This lets configured execution browsers pass preflight
before the route handler while preserving GET/POST, Authorization/Content-Type,
X-Content-SHA256 exposure, Vary Origin and credentials disabled. CORS acceptance
never bypasses a Session's exact execution Origin/Host or viewer capability.
Absent native configuration contributes no execution origin; absent Whitebox
routes install no CORS middleware. The actual ASGI regression lives in
`test_realtime_gateway_server.py::test_whitebox_cors_uses_installed_player_config`.

With no configuration the service Session action remains 503; partial/unsafe
configuration fails startup. Public WebSockets require exact Host/Origin and the
original `worldkit.stream.v1` + private viewer capability handshake. The Room
checks the original live member, admission/leave fence and exact capability before
opening its original SDK socket. HTTP Session reads require that capability;
only immutable, manifest-declared UI assets are anonymous, matching the original
SDK contract. Requests never forward browser Cookie/Authorization to arbitrary
owners, follow redirects or accept client URLs. Responses are bounded and private.

`WorldkitRoom.public_session` reads current truth through the SDK's public
Session APIs, checks original member/render identity, and uses its public parser
to return only a rewritten reachable descriptor. Private `streams`, filesystem
paths, CLI metadata and `localResources` are excluded. S2 verifies all immutable
source/member fields before and after directory publication. Existing direct CLI
consumers retain their original controller role. In controlled Rooms the SGLang
source capture is a spectator, leaving the SDK's input lease to the browser.
Browser intents and SDK receipts are observed in the original Room log as
`viewer_input_observed` / `viewer_input_receipt`, with client/session/epoch identity;
observed intent is not an applied-input claim. Recording failures mark the original
manifest and do not submit input twice or take over live control.

CPU checks cover fixed routing, ownership/Cookie/Origin/capability separation,
source packet and credit forwarding, late leave fences and original input-log
observations. An opt-in real sealed CLI/Chromium test also passed with two private
players through real Gateway/Room TCP hops: public parsed Sessions, original UI
bytes, independent input grants/receipts and native packets, wrong-player token
rejection, A leaving while B continues, and owned-process cleanup. This is native
source/transport evidence using a synthetic sealed author fixture, not WTP
production import, a React-composed browser page, GPU output or deployment.

The extended regression exposed an existing recording fail-open race: live model
delivery can retire a pair before its queued archive-source encode completes.
The late encode now drops its AU after archive admission closes, without restoring
pairs or failing live playback. Active archive admission still rejects a missing
reservation. A deterministic regression preserves both conditions.

S4's public frozen-read download extension is now available in WorldKit
`a99628fd3814e5bcf515e7090069bb4fa939042b`; WTP's browser/member/frozen-read
composition is `19221d5d6f01526cad1642073fa0cc43eb5d361f`. Their
[handoff](https://github.com/seedleap/worldkit-monorepo/blob/b08a9d216a229b8986ba2d1f3a76e45f3f0d8a62/docs/development/game-console-integration.md)
remains the cross-repository entry. A reviewed shared SDK release/pin, actual
WTP service composition and Next ConsolePort integration remain pending. No
runtime publication, cloud deployment, GPU/model performance or full production
acceptance is claimed.

### Original-stage SDK 84e pin and CPU release boundary (historical snapshot)

This retained original-stage snapshot preserves its exact source/test identities
and then-pending validation status. The active future Player pin and its own
validation boundary are stated in the future-candidate section below.

The follow-up candidate pin is WorldKit
`84e2e4232155e201c21e9e4cea2114aee552c171`, replacing
`dfb5a913d3d2ba529ef7278b474b039745e204f2`. This change legitimately emits a new
browser runtime, `2a2a4914df58239d373590b05fa5416004f5b19fbcba00efe165fd559b7199a5`;
`runtime-manifest.json` SHA-256 is
`ed8e4ce71ff5233ddc870b2de18da27e0ffe12738c7d3d0a75fbe5a05e38a0bb`.
The SDK review verified changes only to multiplayer-authoring/authority output;
the other six runtime modules are unchanged. The combined fixes scope Console exit
cleanup to its current binding and retire Creator member lifecycle records through
the original World reset hook. They do not grant a new Console/CLI reset capability.

The retained Node Host resource repair uses capability-probed native Blob IO,
with the existing two-reader FIFO, resource byte caps, aggregate deadline and exact
public JSON/Session contracts. Generic browsers without that capability keep the
existing transport; resource failure does not retry through another transport.
The SDK's owning browser-upgrade gate covers this deprecated CDP dependency.
Source identity remains separate from runtime publication or an old sealed archive.

The earlier SDK pin's complete
[CI 40761644100](https://github.com/seedleap/worldkit-monorepo/actions/runs/40761644100)
failed, as did maintenance source `766c2fcd43d1ae0a8f7b26935acc1b71e4957017`
[CI 40771283636](https://github.com/seedleap/worldkit-monorepo/actions/runs/40771283636).
At this pin-update validation snapshot, exact `84e2e423` source
[CI 40786814550](https://github.com/seedleap/worldkit-monorepo/actions/runs/40786814550)
has not completed, and shared `origin/dev` ancestry has not been established. Keep
the existing Jenkins `dev`/`white_box` ancestry guard unchanged. Verify complete
successful validation of the pinned SDK source, SDK shared ancestry, exact SGLang
PR CI and `white_box` integration before release.

The earlier exact `dfb5a913` native check remains **2 passed in 160.94 seconds**,
with runtime `71ccab185446824290f4f23d00e63e028afd215e450af0c25e45e747aeb17e08`
and manifest `88a4eee1f7bc3841120eccad43c3dd972ff7a3b91a83d94afcc57b1d96a03de5`.
Those receipts are historical and have not been relabelled as the new runtime.
The 8 public Session helpers and original single-player CLI parametrizations retain
their earlier `e872244f` evidence; neither unrelated gate was repeated for this pin.

Both required native consumers were run once directly against the clean, exact
`84e2e4232155e201c21e9e4cea2114aee552c171` SDK checkout on macOS/Node 23.11:
**2 passed in 64.98 seconds**, with 12 upstream websockets/uvicorn deprecation
warnings. Fresh compiled/playtested/sealed fixtures actually report the new runtime
above and new worldBuildHash
`636876a54d1ec263d3da48ad25345cc642010cfffaf853c28020c28dea0bdc36`.
The authored fixture sourceHash remains
`3d6d16a5aff5318cda12ce2507d628e60979ff3e94661452537c05e1db39bc70`.
Both CLI and public browser controllers retain real private Sessions/views/pixels,
independent input, remaining-peer survival after leave and owned-process cleanup.
Only cloud retrieval was substituted; the original single Gateway remains.
These checks validate original consumers of the new runtime, not separate Console
reset authorization, Linux image closure, real WTP import, composed model output
or GPU performance. Earlier `f36d455f`, `816c3ea0`, `e872244f` and `2e468e7d`
evidence keeps its original identity.

The derived Linux image must still run the affected control/transport contracts,
original single-player CLI integration and both sealed native Room cases in its
actual runtime base. No Linux deployment, GPU/model acceptance or production
WTP/Next browser composition is established by the local source checks.

For the existing Shanghai CPU deployment, derive each role from its freshly
verified immutable image. Export an explicit role-specific overlay from the
merged SG source and record both source and base hashes. Preserve deployed
startup/render/RTC/model owners and the Coordinator's role launcher unless a
separately reviewed change owns them. Copy the tracked Room entrypoint and Room
launcher; preserve the existing Node/Chromium/RTC binaries and prove the pinned
SDK production dependency closure in that same Linux base. Do not replace the
entire deployed Whitebox directory with an unrelated checkout.

For each candidate, retain the exact shared SG commit, CLI pin, complete tracked
SDK blob/mode index and verified dependency lock. Execute the affected contracts
and original native gates in the actual role images; an unchanged browser
runtime hash does not establish unchanged Host source. Keep image readiness and
live native/UI/model acceptance as separate claims. Temporary transfer services
must expose only their owned inputs, exit after verified transfer, and leave no
listener. Remove unused transfer/cache copies only after checking their ownership
and references; retain source identities and formal evidence.

Configure the control ticket secret/service allowlist on Gateway, the frozen-read
secret/service ID and canonical WTP platform URL on Room, and the three exact
origins above on Gateway and Room. Keep the ordinary configured Room driver;
only the verified control context chooses the multiplayer adapter. Merge runtime
configuration privately from a fresh capture of the existing service. Before
replacement, use the original maintenance owner to drain, check both native and
model activity, and compare container/image/config identities again. Drift stops
the release for reconciliation. Retain the current recordings/cache, old node,
model routing and GPU jobs; native source video/UI acceptance does not establish
model-video or performance acceptance.

## Same-member native and model media

`getPlayerMedia` is an additive read-only action on the existing signed
`whitebox-room-control` service route. It accepts `schemaVersion:1`, the original
`memberBinding`, and optional nonnegative safe-integer `controlSequence` for
replay. The native response contains `status:'ready'`, `kind:'sdk-session'`,
`memberBinding`, exact frozen public `publication`, and the public SDK `session`.
Retained admissions without a model launch report
`model:{status:'unavailable',reason:'MEMBER_MODEL_SOURCE_UNAVAILABLE'}`.
`getPlayerSession` retains its original `{status:'ready',session}` response.
Neither observation starts or retries model work.

New room creates may freeze `frozenPublication.publicationIdentity` alongside the
unchanged `world` binding. The public field is the original complete six-field
PublishedIdentity with optional attempt; overlapping source/world/publication and
attempt must match `world`. World bindings may also contain platform/SDK run or
interactive-import identities and are never used as a public identity. The media
response projects only the separate complete public field, preserving attempt.
A media launch must match that exact frozen value before native admission/model
allocation. Older creates without it and `getPlayerSession` remain supported;
new media/model paths fail closed if it is absent, with no historical rewrite or
latest-publication lookup.

To request model output, the trusted original `admitPlayer` operation optionally
includes `mediaLaunch:{transport:'webrtc-rtp',publicationIdentity,
initMsgpackBase64}`. WTP must freeze its original validated prompt/reference/profile
startup bytes and complete PublishedIdentity in the original admission parameters
hash, then sign the exact request bytes. The same `getAdmission`/admission replay
must carry the same envelope. Omitted envelopes preserve native-only behavior.
The decoded original MsgPack init is at most 8 MiB and includes the original
`whitebox-render-v2` create/startup fields and source/world identity; it cannot
supply `game_id`. The Room supplies only the original admitted game ID. Only
`admitPlayer`/`getAdmission` requests with this envelope may use the bounded
12 MiB + 16 KiB HTTP budget; other Room actions retain 256 KiB. Public media and
WebSocket budgets are unchanged. Wrong publication, changed replay bytes and
oversized envelopes fail before model allocation.

The original native admission remains claimed when model startup is unavailable.
Capacity exhaustion reports `CAPACITY_EXHAUSTED`; configuration/startup failure
reports `MEMBER_MEDIA_UNAVAILABLE`. A retained failure never becomes a hidden
retry or queue when capacity frees. A future explicit retry requires a separately
agreed operation identity. Source/owner restart cannot recreate an old admission.

`player_media.py` adds a trusted `MemberSource` factory over the exact existing
RoomRuntime entry, retained original join receipt, NativeCLIView iterator and
RemoteRenderer. It performs no second join, capture, renderer or input setup.
`RoomRuntime.next_frame` rejects a competing reader while the child borrows it.
Original RenderHost/SourceSession/RenderSegment, source clock, H.264 encoder,
model lease, single active capacity, RTCSubscription and playout remain the
owners. Factory publication checks use the original frozen admission instead of
re-running independent publication preparation. Media cleanup releases the reader
and model resources; individual member leave and whole-room retirement drain the
child before original native teardown. One member cannot retire a peer.

The same room game ID is shared by its admitted members. RenderHost permits
that overlap only for distinct live `MemberSource` instances belonging to its
original MemberMedia owner: original room binding, admission entry, engine,
member/player/stream, native session and viewport remain fenced. A duplicate
member, copied or replaced admission, legacy/member mix or a caller-supplied
alternate game ID cannot acquire another source. Legacy same-game protection
remains unchanged; stale children retain their reservation until normal cleanup.

This does not increase deployment capacity. Each member owns an independent
model session/lease and RTC source. Concurrent model video for two members
requires two real eligible model slots, Room pipeline capacity for both sources
and dedicated configured RTC endpoints. A capacity-one deployment still serves
one model session; it must not advertise extra capacity without deployment-owner
validation. The CPU regression exercises two explicit test source slots and
separate protocol peers, not live GPU concurrency or deployed media acceptance.

While a media child exists, `getPlayerMedia` changes the explicit union tag to
`kind:'whitebox'` and adds the original `execution` plus short RTC `attachment`,
while retaining the same native `session`, `memberBinding` and `publication`.
`model.status` stays `starting` until the original current model segment produces
its first decodable AU and remains active; only then it becomes `ready`.
`status:'ready'` at the top level describes the usable native Session. Issuing an
RTC capability or observing source-ready does not prove model-ready. An ended
child returns native media with `MEMBER_MEDIA_ENDED`.

Member attachment uses the distinct `whitebox-member-attachment` purpose and
`member_authorization` field, valid for 60 seconds. The exact signed original
resume/bind message includes memberBinding and
`sdkSession:{sessionId,epoch,worldBuildHash,uiBundleHash,playerBinding}`, source/owner
epochs, current connection generation, subject and replay cursor. Existing
RenderHost/reserve_rtc validate it before replacing an old subscriber. No browser
long resume token, URL capability, new public route or global API key is added.
SinglePlayerControl source creation/authorization remains unchanged.

**Gameplay input remains the original SDK controller channel in `session`.**
Member media explicitly allows only `status.get` and `display.configure` commands
needed by the shared client's polling and source/model presentation selection.
They retain the original ordered control queue, source identity, request replay
and display revision validation. Other commands (including model/parameter or
source lifecycle mutations), Whitebox held-keys input, legacy prompt/finish/mock
mutations, new init and asset uploads remain rejected. A consumer must reuse the
original SDK input port without mounting
a second capture/player/decoder. The existing frozen startup asset protocol is
unchanged. Native SDK session/epoch replacement fences the old media child and
attachment; the adapter does not guess a member recovery or resume its successor.

Original native-v2 envelopes retain complete SDK FrameIdentity and build/UI hashes.
Source/model media and `rtc.frame.meta.frame.source.cli_frame_header` retain that
same FrameIdentity. The adapter separately validates opaque SG source epoch/view
identity, numeric SDK render epoch and member binding. It never derives SDK
frame/tick/UI identity from H.264 output indices, PTS, RTP or arrival counters.

The maintained AREA CPU gate includes `test_whitebox_player_media.py`. Golden
vectors live in its sibling `fixtures/player-media-native-v1.json` and
`fixtures/player-media-model-v1.json`; the model vector captures the original CPU
owner path and a decodable H.264 AU with external model/device protocol peers.
Tests cover original signed routing, native/model identities, first decodable
readiness, source/SDK epoch and capability fences, exclusive iterator/input,
capacity/no-retry, reconnect, peer leave and retirement cleanup. These are CPU
contract checks, not live GPU, native ICE, physical-device, 300-second throughput
or deployed acceptance. The two-model lifecycle regression starts B's model before
A leaves, ends its original model segment, cancels its prepared pre-RTC media, or
invalidates its frozen SDK epoch. It checks A's owned reader/session/lease cleanup,
B's model frames generated after cleanup/input and their UI identity, original
SDK input and status scope, and reconnect on B's unchanged source/model lease.
Both post-cleanup and reconnect checks fence against the fixture's next source
index, so already queued frames cannot satisfy new-frame evidence. Original leave/cleanup replays
are idempotent and A's ended attachment cannot adopt B. This proves CPU ownership
and identity isolation; it does not prove browser HUD painting or deployed resets. Matching WTP frozen admission production and SDK/Player
consumption remain separate integration work; this source does not deploy them.

The current SDK/WTP producer's unchanged `whitebox-startup-prepare-v2` bytes
now cross the original member owner. The retained `room-member-media-wtp-v1.json`
filename identifies the fixture envelope, not the startup version. Its test
reaches a model frame through the original RTC subscription using CPU protocol
peers, one worker lease and attempt zero. V1 model vectors remain separate.

The shared receiver supports explicit v1/v2 semantics from the original S0 v2
implementation (`bf347fa6108f612240238e92781a5942d740fc32`, parent
`c7afcef474087bc8945b8c2479847415a2a220d3`), reconciled with the subsequently
captured Room reference-capacity routing. V2 preserves native preview dimensions
and the `condition-preview-rgb24-v2` fingerprint; real CLI frame zero can differ
and remains the pinned live source. V1 keeps its original equal-size/pixel check
and explicit one-shot fallback. Capability echoes must match exactly. The
adapter passes the selected version to the original worker state; there is no
second pipeline or version downgrade. See the [startup boundary](../../zing/plans/shanghai-model-startup-prepare.mdx).

This is source/CPU compatibility evidence. Known Git worker provenance and a
read-only Room capture do not prove fresh deployed GPU process bytes. Physical
model inference, native ICE, 480/736 performance and live acceptance remain
unverified for this candidate; historical deployments do not extend those claims.

CPU verification boundary: Room/member tests 60 passed, 1 Node SDK envelope-case skip (SDK checkout absent);
source_prepare/schema/Torch cache tests 6 passed. A bounded original worker run
with `--continue-on-collection-errors` executed 12 real-Torch source_session
cases successfully (685 deselected), but the adapter test file failed collection
on the unrelated VLA → SRT function-call `xgrammar` import. That run exited 1:
worker adapter execution remains **unverified**, and the full worker gate is not
claimed green. No missing module/Torch stub or protocol downgrade was used.

The actual shared SDK polling/display regression belongs to
`test_whitebox_cli_integration.py`, executed by the existing `room-checks` image
stage with `WHITEBOX_TEST_CLI_SDK=/opt/worldkit`. That case fails if its required
SDK checkout is missing. It retains real Node client envelopes, the original
member/RTC owner, the first two-second poll interval and input/model-mutation
rejections. The pure CPU AREA member file keeps all30 protocol tests and no SDK
skip. Generic approved CPU inventory does not select the SDK integration file.
Earlier checkpoint wording calling this skip a native RTC binary requirement
was incorrect: this particular skip was the external Node SDK client case.

An earlier never-live Player candidate selected reviewed SDK integration
`137d6f716d9244b5cbc6e71f45c8cdf6b55d45f8` for the
shared WhiteboxClient used by this required test and the Task3/3A/6 consumers.
The older e0cf/766 SDK lacks that module; it is not an alternative client for
this case. This is a draft integration pin pending shared SDK dev ancestry and
CI, not a released runtime. Existing Jenkins ancestry/checkout guards remain
unchanged. The separate original minimal SG release's SDK pin is managed independently;
source merge, candidate CI, image build and live acceptance remain separate.


### Earlier Player source alignment (historical)

That earlier Player Docker SDK input selected
`137d6f716d9244b5cbc6e71f45c8cdf6b55d45f8`, incorporating the reviewed native Blob
resource reader and member-reset/player-exit corrections. The source owner's
prebuild records runtime
`2a2a4914df58239d373590b05fa5416004f5b19fbcba00efe165fd559b7199a5` and manifest
`ed8e4ce71ff5233ddc870b2de18da27e0ffe12738c7d3d0a75fbe5a05e38a0bb`. These identify
a candidate build, not a shared release, public runtime activation or live SG
image. Earlier source/runtime evidence keeps its original identity.

This alignment changes only the existing SDK input and owning documentation.
Jenkins SDK dev and SG white_box ancestry guards, exact checkout checks and Docker
acceptance stages remain unchanged. The separate original minimal SG release is
not modified. The focused actual Node shared-client poll/display case uses this
exact SDK; it does not run a browser, physical RTC peer, GPU model or image build.
Exact new-source CI/shared ancestry, paired WTP facade selection and real
single/member native/model/ICE/performance acceptance remain release gates.


The final source137 alignment changes only ESLint/test fixtures/tests/documentation
relative to the reviewed e4c source. Production runtime, public facade/API and lock
inputs are unchanged; the existing exact-e4c shared-client poll/display test covers
those same implementation bytes. This is source equivalence, not a new test run or
current-head CI success. The reviewed Gateway CORS correction remains included.

### Current shared cleanup integration

The current candidate selects exact SDK input
`e99e03be52410a37732e2007969de862ecfb75b5`, the frozen SDK
`main` and `dev` source at release, on top of
shared `white_box` Player/media
integration `bc882c640a9030ad189dd86c6a86b47e5ec579c3` (PR #460). It retains
the frozen publication, admission, subject and owner fences and the original
cleanup-continuation contract. New Host/CLI code adds trusted negative action
evidence, action-input anchoring, released-input lease fencing and cleanup
isolation without changing the successful action or session/media protocol.
Its logger additionally retains a stable owner for closed diagnostic pipes:
an `EPIPE` must not terminate the Host or displace another error listener.
The client now fences large-resource decoding and visual handoff with the
original heartbeat RPC. Creator controls use the existing player/input owner;
these changes do not create a second runtime or expand heartbeat timeouts.

The paired ordinary runtime is `0.2.20-e99e03be`, with runtime hash
`c07c4dbd81d043a9373a4549ccf77d13517727c07fe33c93abc42d51bb753d39`,
cache identity `763edefe3db5db107b71d52a0427b6218495dd40509a6840932c7f6984686bab`
and public manifest SHA-256
`8b94cf613b228eafa8ba3162bae4295fc2dc81c8885eb9c7abca8bbb0f35c3e8`.
The exact dependency lock SHA-256 is
`dc9d0795225038fc2058e6a47b879374dd47e2edbd704c1b87eb895aeb4b55c3`.
The compiler/materialization input change requires this source and runtime
cache identity even though the ordinary JavaScript payload hash is unchanged
from `0.2.19-619b236f`.
Advance the complete Room CLI source, matching WTP provenance and public runtime
together. Retained control-role artifacts keep their true shared producer when
their owning business and dependency bytes are unchanged; never relabel them as
new builds. Ordinary-rate native evidence does not prove arbitrary CPU-stress
liveness. The prior `0.2.18-a019b6a4` and its A019 public manifest
`1d4d533845f78f48a9278a4cfd5f890cc67bceb3b39f6f9dd4a2f548dd8b7d77`,
and the f83 ordinary release
`0.2.17-f83b370a` and its public manifest
`2817c90cc8ae1d2b0f80ec7a6e8f75d8acd032489090d465bb039f7708e09e04`
remain historical identities rather than aliases for e99.

The existing actual SDK shared-client poll/display and private-member session
consumer tests cover the public integration seams. Exact-source CI, the final
Room image and paired public browser/RTC/model acceptance remain separate;
this source integration does not deploy services or certify the historical
member-failure cases as fixed.
