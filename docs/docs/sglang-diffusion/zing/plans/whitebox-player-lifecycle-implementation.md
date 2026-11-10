# Player lifecycle and paged Block0 integration

This source change implements the [accepted lifecycle design](ROOM_SLOT_LIFECYCLE_RECOVERY.zh.md).
Deployment and GPU/browser acceptance are separate evidence; no capacity number in
this document is a measured production result.

## Ownership and deadlines

Custom-input recovery retains `input.semantic.available` alongside the binding,
receipt and error envelopes in the existing native input archive. This SDK hint
announces that a dedicated source has finished resuming capture admission; it
does not renew gameplay activity or restore old input. The SDK control relay
forwards it unchanged. Only an original send correlated with a meaningful
committed receipt renews the player's gameplay activity; neutral recovery
receipts still do not count as activity. SDK input lease recovery and model/player
TTL remain independent.

`WHITEBOX_BOUND_COMPUTE=1` uses one parent model connection per player. A managed
assignment contains immutable `admitted_at` and `player_expires_at` Unix timestamps.
`WHITEBOX_PLAYER_SESSION_TTL_SECONDS` is rendered from regional `play_seconds`:
Shanghai 1800 seconds, Thailand 7200 seconds. The shorter renewable assignment
lease remains a service-health signal. Renew, reconnect, Prefill and segment
transitions do not change the player deadline. Room and Worker enforce it
independently; Coordinator retains and quarantines overdue unknown ownership.

Automatic and member-prefill admission reserve this parent before SDK join and reuse it for
member media. It does not join a second view. Legacy non-bound protocols retain
their original wire contract; both regional release bundles explicitly enable
the bound protocol. A failed, ambiguous SDK join keeps the reservation quarantined
unless the original owner can provide positive cleanup evidence.

A GameRoom has no platform business expiry. The existing owner lease still fences
owner loss. Cold startup keeps renewing that lease independently of the SDK join
lock. Empty controlled rooms close after bounded grace (120 seconds before the
first member, 15 seconds after occupancy), excluding current pending joins. The
last-leave/next-join fence never waits on model cleanup.

## Release confirmation

The worker retains a managed parent in `cleaned_pending_commit` after all model
ranks and asynchronous RGB copies finish. Room retires the SDK member/view/input
independently and supplies an owner-issued terminal receipt to Coordinator.
Coordinator persists that receipt in its existing Redis state, prepares/finalizes
the exact original reservation, and only then marks capacity free. A deterministic
close ID and exact session/generation/worker epoch fence retries and lost replies.

The worker control plane adds `POST /v1/realtime_worker/reservations/{token}/prepare-release`
and `/finalize`. These are internal Coordinator operations, not browser APIs.
The existing `DELETE /v1/sessions/release-verified` accepts an internal
`sdk_cleanup` receipt for managed assignments. Ordinary release cannot bypass this
contract, including by omitting the player deadline from a replayed payload.

Redis expiry, HTTP 404, disconnected sockets and absence of a local object are
not cleanup evidence. Unknown objects and GPU occupancy remain retained.
A failed player's cleanup does not classify the shared world as owner-lost.
The model launcher and regional outer supervisor must not automatically restart
the GPU group; hard failure requires quarantine, diagnosis and manual recovery.

Startup cancellation carries a cleanup receipt only after the original factory
has reaped its owned process group and the original Room has finished cleanup.
The exception remains the exact `asyncio.CancelledError` type so timeout and
caller cancellation semantics are preserved. Subprocess creation retains its
handle and inherited log descriptors until fork completes; repeated cancellation
cannot interrupt reaping. Cancellation before SDK invocation is acknowledged by
the original local join scope, not inferred from a missing SDK record. Resident
service uncertainty, failed cleanup and owner-epoch changes remain quarantined.
Failed close attempts can retry the same original owners and update the terminal
cleanup receipt without recreating a player, view or model binding. The
`test_whitebox_startup_retirement.py` regressions exercise these boundaries with
real subprocesses and verify that the following player can reserve released
capacity only after both SDK and model cleanup are confirmed.

## Prefill and cache

A private member view supplies its next genuine native frame as Prefill F0 without
sending shared-clock hold/resume. Its context expires after 120 seconds. Dedicated
single-world sources retain capture hold compatibility. Both paths pin the same
F0 bytes, native identity and source PTS across multiple Prefill conditions and
subsequent consumption. Display placement may account for snapshot age; source
capture identity, input receipts and archived source time remain unchanged.

Main PR467's cache management prerequisite and PR499's paged L1 are integrated.
An entry consumes `ceil(prefix_rows / 2048)` pages, with whole-entry LRU and native
DMA/writer fencing. The host budget is still pinned at startup. Entry packing is
on demand; this is not a promise of lower startup RSS or a fixed entry count.

Native disk schema is `zing-block0-v3`, format 4, under
`<cache_root>/entries/<key>/rank-N`. Old v2 files remain untouched. Native online
lookup defaults to L1; explicit Prefill/preload can promote L2. The existing
explicit persistence receipt uses small native metadata, never a full KV scan.
Native-sized S0 v2 previews additionally bind shape/dtype in the key; main's
fixed-size preview keys are unchanged. Cache timings distinguish L1/L2/miss,
lookup, disk reads and GPU restoration; forced recompute never reports an old hit.

For each region, measure real resident keys, prefix rows/pages, evictions, warm
recall, cgroup memory/peak/limit and host headroom with representative inputs.
A target of 20+ groups must be established by loading and recalling distinct real
keys. Changing a configured count or dividing the budget is not acceptance.

## Validation and release

All production changes first merge to remote `white_box`, followed by complete
source builds with the exact SDK pin. Rollout order is Thailand then Shanghai.
The same source/config compiler serves both regions. Preserve original Mac proxy,
existing four-card allocations, unrelated training, old caches and recordings.

CPU tests cover retained cleanup owners, independent SDK retirement, original
identity retries, immutable expiry, cancellation with queued source messages,
real two-rank Gloo cleanup aggregation, and paged-cache contracts. Production
acceptance additionally covers actual Prefill/recall/Prompt/segment behavior,
inputs, released Slot reuse, recordings and five current browser cases of at
least 180 seconds per player. Actual two-player acceptance is explicitly waived
for this rollout because a second compatible Worker is not available; this is
not a waiver of isolation contracts or future multiplayer acceptance.

## Resource preparation compatibility

The regional bundle selects the SDK's native `auto` resource transport. It tries
the current playable publication and uses the sealed archive only for the SDK's
existing not-ready/404/405 cases; it does not fall back on authorization or integrity
failures. This replaces the observed Thailand `cdn`-only setting, which rejected
a source with `WTP_PLAYABLE_NOT_READY` before any CLI frame. Preserve this failed
baseline separately; it provides no FPS or latency comparison. Resource/cache
preparation and publication identity are checked independently of model health.

## Trusted World context and manual inference integration

Ordinary manual preview opens the SDK source without reserving a model parent or
child, including shared-room members. The first explicit enable obtains the
parent through the existing source owner. Automatic inference and the internal
member-prefill profile retain admission-time reservation. After enable, disabling
inference closes the child and resets optional Prefill context while retaining
the same managed parent and its original deadline. Only player teardown supplies
SDK retirement and the final verified model release; disabling inference is not
a parent-slot release receipt. Legacy unbound source behavior remains supported.
An explicit cleanup retry reconciles the original stopped assignment; it cannot
open a new reservation.

Empty-room collection observes the actual unfinished admission task, independently
of the externally blocked admission status. Failed, cancelled and leaving
admissions do not retain an empty room. Admission failures log the original
exception, operation ID and stage on the server; public responses retain stable
codes and exclude exception text.

The signed original `getOperation` query can observe directory `ended` after
verifying its caller, complete original invocation, world and owner route. It
returns `status=failed`, `errorCode=ROOM_ENDED`, `cleanupComplete=false`. An exact
retained `getAdmission` record with the same caller, admission, publication,
game and owner epoch returns `status=blocked`, `errorCode=ROOM_ENDED` unless it
already retains a verified `retired` result. That positive cleanup receipt takes
precedence, including recovery with the exact original `leaveOperationId` after
a lost leave response; an unknown or different admission's leave ID is rejected.
Directory termination alone never releases WTP member seats or claims SDK/model cleanup. Missing or
expired routes, mismatched identities and admissions lost on restart remain
unknown or unavailable. These CPU contracts do not establish deployed versions,
real browser join success or GPU acceptance.
