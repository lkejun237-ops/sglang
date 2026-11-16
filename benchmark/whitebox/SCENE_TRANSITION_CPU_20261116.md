# Scene transition Workbench CPU acceptance — 2026-11-16

This record validates the actual CLI → Room → native RTC → React Player lifecycle
against explicit loopback CPU role fixtures. It is not GPU/model-cache acceptance,
a deployment receipt, or a performance benchmark.

## Boundary and identities

- Only Workbench ECS `i-uf6hh7iq2z2jcjyt7h3a`, Shanghai, `10.0.1.32`.
- No production endpoints, model workers, Coordinator allocation or GPU requests.
  The CPU fixture uses production `PooledBoundSocket` and separate loopback
  Encoder input / Decoder output WebSockets; allocation, auth and compute are fixtures.
- Workbench root: `/opt/worldkit-scene-validation-20261115`.
- SGLang main baseline `0892103b2c36d425921785f4c5088bd8c98533dd`;
  Worldkit dev baseline `8f0f4f1ecfd31182ceb247dbe81c8a0f76a017bb`;
  WTP main `f96faa2461a86e07f8b81bdc889d13d647601b04` (not modified).
  Exact pushed execution SHAs and commands are retained in each run's launch JSON.
- Real Creator-sealed fixture: `cpu-chain-fixture-2/case.json`;
  delivery SHA256 `05167c18e2acc26b9e11f48e042bbfc5a4d052b37290daec4add7dc43a58a2de`;
  world build `b9e1d66a25a0c7421bb29cfdfbb07aefcce0afbac6493ba9b3e4382f84274202`.
  Live video `832×480`, independent condition preview `960×540`, two distinct
  ordered `256×256` references. This is not validation of the 736 model profile.
- Native RTC rebuilt from the matching SGLang source; binary SHA256
  `814c9c71ae4a44a74caac7dc592e8c277cf82de6d8265a0f90af982da2460eac`.
  Reused x264 C source matches SHA256
  `14daee3672fddacf4f4104bf96cac208d8bea69c23f404bfcfd9f2a312dad59d`.

## Root failures reproduced and fixed

| Failure | RED | Fix / boundary |
| --- | --- | --- |
| Normal old-segment terminal receipt races with scene prefill and marks inference failed | `ed2f089fc5`, 1 failed / 2 negative controls passed | `35dcda59b7`: only exact retiring segment + original inference revision + no error bypasses the failure reducer; real errors and newer owners still fail |
| Old Host condition reconciliation discards the successor preparation before segment validation | `5b09ead849`, 2 failed / 1 passed | `3251c52daf`: validate reserved segment before matching prompt/references; same-segment mismatches still reject |
| Condition commit blocks the control reader waiting for source frames whose samples need that reader | `f4ed1e67dd`, 2 failed | `dd5df30604`: adapter owns asynchronous submission through close; reader can advance; original journal CAS remains authoritative |
| CLI treats condition commit and gameplay activation as the same fixed tick | `65ab3c0787`, 2 failed / 4 passed | `b6de9181b6`: activation must follow both condition commit and presented target tick; reject early activation |

Stream Host also separates `commitCondition` from gameplay `activate`; actual
Chromium/World red-green evidence is recorded in the Worldkit spec section 13.5.
Final review additionally checks author-method receivers, conflicting in-flight
commit boundaries, and identity of the held child used for refresh/exit acceptance.

## Evidence and current results

- `cpu-chain-run-9`: 11 real browser assertions passed.
- `cpu-chain-run-10`: 12 passed, adding no-model video → target presentation →
  gameplay activation → reentry without allocating any model lease.
- Both runs: actual source/Player, authored story event, decoded local video,
  video-first and prefill-first ordering, held gameplay timers/damage under a
  visible mask, identical S0/Prompt/seed/reference bytes at prefill and generate,
  one parent through A→B→A→B, fresh Player refresh, final exit.
- Cleanup: `cleanup_errors=[]`, `cli_exit_code=0`, no forced kill,
  HTTP 19968 and UDP 42071–42073 released, fixture parent `released`.
- `scene-race-green.xml`: 151 affected checks passed (overlapping other suites).
- `scene-commit-green-2.xml`: 157 source-condition checks passed in 74.87s.
- `scene-activate-green.xml`: 113 passed, 16 optional CLI checks skipped, 23.18s.
- `scene-reconcile-green.xml`: 294 passed, 1 blocked:
  `test_worker_prompt_boundary_includes_admitted_prefetch` cannot import Torch
  in this interpreter. It is not a passing test; the pre-existing dependency
  blocker is retained without another installation attempt.

Final post-review evidence is listed below.
Do not add overlapping suite totals to claim unique coverage.

## Reproduction

Use the Worldkit task branch's
`packages/stream-player/tests/integration/scene-transition/README.md` to build a
fresh fixture with the real Creator, then run the existing S06 integration runner
with `--case`, `--role-pool`, matching native binaries and fresh output/ports.
`--case` explicitly enables scene model transitions only inside the CPU process;
the browser checks Room admission rather than removing production capability gates.

Read both `browser/result.json` and `run/runtime-result.json`, plus launch JSON,
not only the shell exit. The fixture hashes actual bytes received on the model
wire; synthetic `model.cache` hits prove request identity only. Its deliberately
absent persistence descriptor returns `CACHE_PERSISTENCE_UNAVAILABLE`; it neither
claims nor requires durable cache persistence for memory-ready handoff.

Same environment issue failing twice: retain both errors, stop that dependent
retry, continue independent coverage. Existing WTP recording closure, asset-library
packaging, telemetry dependencies and producer timeouts stay listed in Worldkit
spec 13.3–13.5; no third retry, production fallback or silent environment switch.

## Not established

No real Encoder/DiT/Decoder compute, GPU L1/L2 hit or preload recovery, quality,
480/736 performance or memory comparison, deployed WTP/CDN authorization,
all-phase network/TTL failures, two-player allocation isolation, recording/audio
integration, or production acceptance. This task neither deploys nor registers workers.

## Final review fix pass and receipts

Three Important findings were fixed in one pass: preserve the author binding
receiver for both fixed hooks (`callback.call(binding, …)`); freeze the first
in-flight commit effective boundary and reject conflicts before its CAS settles;
replace historical held-prefill lookup with a fresh operation identity that
remains live through Player refresh and is cancelled by exit.

- World receiver RED: 1 failed; final real scene subset: **7 passed, 5 unselected**.
- In-flight boundary RED: **2 failed, 2 controls passed**.
- Final SGLang five-file closure: **272 passed, 16 skipped, 0 failed**, 98.323s;
  `evidence/review-sg-green.xml` and `review-sg-green-exit.json` (exit0).
- Scoped Worldkit ESLint exit0: `review-world-lint-exit.json`.
- Final `cpu-chain-run-12`: **13 passed**. In addition to run10, checks actual
  old generate cleaned → prefill → prefill cleaned → successor ordering,
  first source append at index0 with real F0 distinct from S0, and current
  held operation surviving refresh immediately before exit.
- Execution code: Worldkit `891c31a714926c4c87775a0110671cfc9944d42f`,
  SGLang `2899a45416728646bf80e00804cb353c6d254deb`; both pushed before execution.
- Cleanup: no sources/children, parent released, CLI exit 0, no forced kill,
  all four ports released; final process audit `processes=[]`.
- `scene-validation-cpu-chain-final-evidence.tar.gz`, SHA256
  `fd30bf1b3e675bfda9055d62f4eef2770e115773ebd42606b393259d15c3be8b`,
  contains raw JSON/JUnit/logs, every failed CPU run and frozen input identity.

Run11 read the previous HUD immediately after Room reached revealing under CPU
load. The test now waits for the actual target HUD before asserting; it does not
add fixed delay or fabricate target presentation. Workbench exec returned blank
output/timeouts twice; subsequent evidence reads use persisted files/download
on the same machine, without retrying completed tests or switching environments.

The independent reviewer did not independently rerun Workbench or evaluate GPU
quality/cache/performance/deployment. Actual receipts were checked by the executor.
No remaining Important finding or deferred Minor; the explicit unvalidated
product/environment combinations above remain open.

## Follow-up: failed World transaction rollback (2026-11-16)

Latest main 248c4510e52313e9aff3b5cfbea97e7c9f712c19 was merged without conflicts
as 3fc4b76efee1223db3da7ba930d5124f38b79187 before this validation. The same
five-file closure passed 272, skipped 16, failed 0 in 98.223s (`p2-merged-sg-green.xml`).

Worldkit deep review confirmed a separate P2: the fixed hook wrapper rejected
the scene promise but swallowed the author exception, allowing queued state and
model intents to commit. New condition/activation tests failed 2 before the fix.
Worldkit c88272fe5 forwards the original failure to the existing World transaction
owner. A subsequent cleanup test failed 1 because the failed World blocked an
existing input-hold token's release. Worldkit f29d8c92b permits owned-token cleanup
after a fault while keeping new work, reentry and capture ownership fenced.

Final World scene subset: 9 passed, 5 unselected; fixed-commit/local-input source
full files: 37 passed; scoped lint exit0. All are fixed Workbench CPU checks.
The earlier 5 passed / 4 failed intermediate run is retained (`p2-rollback-green.json`),
not relabeled green. Final normal-cutscene evidence uses a rebuilt Creator-sealed
runtime, with full identities in Worldkit spec 13.7 and `p2-runtime-identity-proof.json`.

## Original Room SDK pin: incompatible review baseline

At the review baseline, the standard builder consumed `benchmark/whitebox/sdk-revision.txt` =
`2457d5ea2f8b86f3f668656b8c0e5d6ad5cb635b`. That SDK executes activation during
commit and returns the cached commit tick again during activate. Current Room
requires activation after the condition commit. The enabled model-cutscene
combination is incompatible when it reaches activation receipt validation.

The CPU runner explicitly sets `WHITEBOX_CLI_ENTRYPOINT` from `--sdk-root`, so
task-source success does not establish the standard pinned-build combination.
Jenkins SDK checkout and `deploy/whitebox/release.py` require dev ancestry.
A matching Worldkit revision must first become dev-reachable, then this pin and
the pinned CPU integration must be updated. At that reproduction stage no dev merge, SDK pin update or
production deployment was performed. The task-branch pin fix and its remaining
release prerequisite are recorded below.
The failure-rollback fix and this SDK delivery gap are distinct findings.

Final follow-up `cpu-chain-p2-rollback-14`: 13 assertions passed with rebuilt
runtime 3875fd052a976acd3c490211d07f19f331c720f8968f98b23fbbde4520de36b7,
Worldkit af09f5d800c4a85d943766727cd455e2c37fdc60 and
SGLang 3fc4b76efee1223db3da7ba930d5124f38b79187. Cleanup errors [],
CLI exit 0, no forced kill, parent released, all four ports free, task processes [].
Archive `scene-p2-rollback-final-evidence.tar.gz` SHA256
`f6723c42083f9f83e475585b4c25b95035cf68211ccbe7ee91c7f370d2b1d720`
retains the REDs, intermediate cleanup failure, final scoped GREENs, rebuilt
sealed input and native CPU chain. These results preceded the pin update; they do not accept the old
2457d5ea standard-build combination.

## Targeted fixed SDK receipt verification

On the fixed Workbench `.32`, `sdk-pin-contract-run-3` executed the real SDK
scene producer with a real World fixed clock and delivered its receipts over
loopback HTTP to current `NativeCLIView.scene_entry_control`. No model, GPU,
Coordinator or media connection was used. Results: **1 failed, 3 passed**, exit 1.

| SDK source | Cutscene path | Actual commit / activation tick | Room result |
| --- | --- | --- | --- |
| Configured pin `2457d5ea…` | Model: commit then activate | 1 / 1 | `SCENE_ENTRY_SOURCE_RECEIPT_INVALID` |
| Candidate `52429626…` | Model: commit then activate | 1 / 2 | Passed |
| Configured pin `2457d5ea…` | Whitebox: activate only | No separate commit / 1 | Passed |
| Candidate `52429626…` | Whitebox: activate only | No separate commit / 2 | Passed |

The pinned producer invoked the author activation hook during commit at tick 1.
When its activate response was rejected, SDK input was already released while
Room retained preparation `contract` with no accepted activation. The candidate
called only the condition hook at commit, retained the hold until activation,
and activated once at tick 2. Successful cases also passed repeated activation
without another SDK command. The acceptance assertion deliberately remains red
for the incompatible standard pin; it is not xfail or a compatibility fix.

Execution identities: SGLang harness
`82a46d8f87b2f85928e84f34024bbae56de8b1d4`, candidate Worldkit
`52429626abb51d9e6921ba24038d16ff2a380a13`, configured SDK pin
`2457d5ea2f8b86f3f668656b8c0e5d6ad5cb635b`.
The pinned scene module was exported byte-for-byte from Git; runtime domain
imports were checked identical between revisions and hashed. Both producers
received the same candidate World clock. This proves the receipt contract
mismatch, not an entire frozen SDK installation, Room image or Player/GPU chain.
The one-off probe is retained with the execution source bundle in the evidence
archive; it was removed from the final branch after diagnosis. The maintained
CLI/Room/Player integration below is the acceptance path.

The first two attempts failed in newly added fixture code (package-relative
module resolution, then non-atomic readiness file). They did not reach product
assertions and remain archived. The final fixture closed normally; no fixture
process remained and port 40189 returned connection-refused after teardown.
The narrow existing Room receipt gates plus unconfigured opt-in probe returned
**7 passed, 4 skipped, 20 deselected**. The four skips require explicit frozen
inputs and do not count as successful compatibility tests.

Archive in the Workbench task root:
`sdk-pin-contract-evidence-final-20261116.tar.gz`, SHA256
`5d3326991dbbf5a48b1b9e6675645421af0c549852848b977824dc44678921d6`.
It retains inputs, wire/hook/hold state, JUnit, cleanup, failed attempts and the
pushed execution source bundle. A subsequent Python format/import-order-only
change passed isort 7.0.0, Ruff 0.15.1 format and F401/F821/UP037 checks; normalized
AST comparison confirmed unchanged executable definitions and import set.

## Fixed SDK pairing on the original task branches

The task branch replaces the incompatible pin with
`c03318030337a09817983e13d74a2cc135a401a6`. Room's strict later-activation tick
check is unchanged. The one-off SDK module probe (four files) and the task-only
`/test/tasks` stack endpoint were removed; the pre-existing S06 logger is unchanged.

Validation uses `deploy/whitebox/room-sdk-archive.sh` at the configured pin.
The canonical export SHA256 is
`221d0e2c11120c02f29ca2f56946086904f5e6b596199d7a776db25a08ab700d`.
On the same Workbench `.32`, 6,835 matching regular files were reused and the
84 missing/different files supplied from that immutable commit. All **6,940**
exported file/symlink entries and **126** SGLang source inputs were checked.
Installed dependencies were reused with an identical pnpm lock; actual Host,
World and Player resolution points into `/dev/shm/worldkit-scene-sdk-c0331803`.
This is the complete standard-export SDK tree, not the earlier isolated-module
probe. No Room image was built and no GPU or production service was used.

Execution: SGLang `144c30105b7c2087d7d6a940d7913fea24fb9e37`, SDK pin above,
existing Creator-sealed case `cpu-chain-p2-fixture-4/case.json` without resealing.
The direct Workbench CPU checks returned:

- SDK scene transaction / rollback regressions: **9 passed**, 5 unselected.
- Room scene receipt boundaries: **7 passed**, no skips; same/early ticks still reject.
- Real CLI → Room → RTC → Player, loopback CPU role pool: **13 passed** in
  `cpu-sdk-pin-c0331803-run-1`. Both model and whitebox cutscenes, presentation
  gating, serial child cleanup, same parent/condition, distinct F0, reentry,
  refresh and exit cancellation pass. Exit 0, cleanup errors [], CLI exit 0,
  no forced kill, parent released, TCP 19978 / UDP 42171–42173 all released.

Archive `sdk-pin-fix-final-evidence-20261116.tar.gz` SHA256
`638e2aa6a6bba92c6de42294c960a5d19bc5e7cb00817f7a0d60a8e355bf9956`
retains the original old-pin RED, current raw reports, native run, frozen inputs
and complete source manifests. Local work only exported/read Git and verified
returned evidence; no local CI was run.

**Release prerequisite remains blocked:** the user explicitly declined merging
Worldkit into `dev`. Observed dev `111545e1eaaf6d6cbff87b71bf991de19affefca`
does not contain the paired SDK change. The Jenkins/release dev ancestry gate
is unchanged and must reject this pin until it becomes dev-reachable. This
fix is committed to the original task branches only; it is not an accepted
standard release build, deployment or GPU acceptance.

## P3: independent condition-commit tick boundary

The existing receipt matrix now includes `commit=8`, `presented=7`: activation
at tick 8 rejects while retaining the preparation/commit; tick 9 succeeds and
repeated activation returns the original receipt. The prior six cases remain.
Only the test matrix changed; runtime and SDK pin are unchanged.

Fixed Workbench `.32`: current implementation **9 passed, 20 deselected**.
A separate snapshot using only `cli_room.py` from main
`248c4510e52313e9aff3b5cfbea97e7c9f712c19` failed both new cases as expected
(incorrectly accepts tick 8 and rejects tick 9); no setup errors. Current runtime
source is `e1a763d1bf825446b1ffaf20e331d38ef1dc8870`. The tested test-file SHA256 is
`d0886aa13c4ae82a840352ec1cdb6dbd48a8ecacdb2d5e506655d85f661f183f`.
Evidence: Workbench task root `p3-tick-evidence-20261116.tar.gz`, SHA256
`4d8da970d099c9199c853d2f85cf95c1b845bf7247b31d5177e2461f7ae3e3c5`.
No local CI, GPU use or dev merge. Inputs and both JUnit reports are archived.

## Worldkit dev admission

Worldkit #1250 merged the SDK changes and regression tests into dev. Worldkit
#1251 moved the cross-package fixture to
`scripts/integration/build-scene-transition-fixture.test-support.ts` so it uses
the root workspace's declared test dependencies. The Room pin now points to
the latter dev commit, `ba45dffc9dc42f2ec8bd435068555489eb687aa0`.
Use the pinned checkout's scene integration README for the fixture command.
The production Room build also requires SGLang source reachable from main.
