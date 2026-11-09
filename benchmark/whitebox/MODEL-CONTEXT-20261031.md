# Single-player model-context GPU regression — 2026-10-31

These are task-owned real-GPU regressions of the actual WTP React **Playtest**,
Creator delivery, CLI, SGLang RenderPipeline and original journal. Route/auth,
storage and publication admission are explicit private fixtures; assets use the
approved fixed local legacy mode. This record does not replace formal external
Console `/play` / signed attachment / RTC acceptance, or shared online deployment.

## Exact tested sources and inputs

The four priority-p1–p4 runs predate the latest Prefill merge. Their SDK base is
`e3c1fc9132274bcb1bb5b45f04a96d6284499caa` plus the consumer retry correction
(`consumer.ts` SHA-256 `5f5a4b7cd9f4d4dd0a3312c88dc22b9cad02e73a057bc2a403e218b644defcab`).
Their SGLang base is `a2aca6da95651be91f697c9c266ee18a0cd03a49` plus the
nonblocking end-admission correction (`render_source.py` SHA-256
`8ae4f5f7ed1cbda345b3e4e2252cff237c2b4dcd1200e937ea64d5f9c66eacb0`).
WTP is `8241de0361b64f61e5bdc3a25b7cb5f18621bf77`. These base SHAs alone do
not identify the tested overlaid source. Workbench task copies were restored
using hash guards after testing; shared services were not deployed or restarted.

The sealed waterhall delivery is
`235ce0ea7dc40a33297bb9035afb15abe9b6ae9018ce8c2db959c650570c1dd0`, archive
SHA-256 `eccd6b5df6fdc1ff8884d513201e1297e016c6094fb28dcf6db1784c8f7ce7ee`.
Native world hash and CDN CLI hash are distinct, both recorded in the
[machine-readable summary](model-context-20261031-summary.json).
Existing model revision `whitebox-sh-s0v2-main-bf347fa610-r1`, seed42,
832×480 at24fps was used serially, one active lease at a time. The service
revision label is not proof of deployed worker source or checkpoint bytes.

## Results

| Run | Real output | Verified scope |
|---|---:|---|
| priority-p1 | 1500 frames /106.019s | 17-stage weather/outfit/manual text/reference/override/door/reconnect/new-segment composition; 4 distinct required onset IDs submitted once each; 2 initializations for the explicit two segments. Final white shirt and sampled door open/close frames manually checked. |
| priority-p2 | 4124 frames /200.661s | Accepted receipt lost before actual CLI control disconnect; same original request once. Bad reference hash never commits; old applied state survives and the same payload can be re-armed. Withdrawn preparation completing late never commits; a later valid condition succeeds. |
| priority-p3 | 361 frames /41.725s | Required onset receipt-loss reconciliation without replay. Real source dispatch held; drain returns to the ordered queue, cancel is admitted and confirmed release takes2895ms. Same source starts a new segment, retains persistent red outfit, does not replay the onset, and accepts the next weather event. |
| priority-p4 | 8043 frames /360.245s | One model initialization;340.05s continuous session including ten complete30s sample windows and ten applied/presented event updates. |

`priority-p2`'s overall browser result remains **failed**: the private driver
constructed WhiteboxConfig directly with its180s play default, then terminated
normally with `BINDING_BUDGET_EXHAUSTED`. Fault subchecks passed; five-minute
stability did not. The separate `priority-p4` run uses the existing
`configured_render_play_seconds()` with test-only
`WHITEBOX_RENDER_PLAY_SECONDS=600`; production/shared configuration is unchanged.
The supported render factory default is300s, range1–1800s.

Every stage cross-checks full prompt SHA, original request and journal applied,
ordered reference bytes, segment/epoch/version and actual presented frame scope.
Faults are injected only in private receipt/resource/send wrappers. The journal,
model acknowledgements and pixels remain real. Withdrawal is not World.reset.

## Performance scope

- Ten30s window FPS: min23.49, max24.27, mean23.94. Maximum of the window display
  interval P95 values:63.5ms. This is descriptive playback sampling, not a
  baseline/candidate speed comparison or a global FPS calculation.
- Event-to-presented samples range3946–4078ms; exact ten-sample median3982.5ms.
  Timing includes clicking, matching presented version, UI readiness and
  screenshot. It is not Encoder/DiT/Decoder or pure-model latency. The original
  archived verifier labelled sorted sample index5 (3992ms) as median; this record
  recomputes the conventional median without changing the original archive.
- Five-second context queue sampling: observed peak3077bytes, final0. This is not
  an all-instant high-water mark. Process RSS samples and their scope are in the
  summary: Python includes retained harness diagnostics; Node is the CLI main
  process only. No hour-scale stability or memory-leak conclusion is made.
- All four runs ended with CLI exit0, no wire/cleanup errors, confirmed lease
  release and final active0/queued0/free1; no task listeners remained.

## Current integration and remaining acceptance

After synchronizing SDK `7c994c4b46c26fbeb43badf3f1670686cdfd2f9d` and SGLang
`92939d91fe75fd06a2890cbf81fbf131d1e9851f`, the current consumer tests passed54,
and the scoped end lifecycle tests passed10 (23 deselected). Those are CPU/source
regressions; **the newly merged Prefill/runtime path has not had GPU revalidation**.
The unavailable torch-importing worker boundary case is not counted as passing.
Current SDK typecheck/targeted ESLint, test census (inventory only), runtime
prebuild, introduced document-target checks, summary/original-data assertions,
Python AST and diff checks pass. Mintlify site-wide validation was not run:
its CLI is not installed. Runtime bytes remain
`4fce263390d2263e72f9e9069745b145771f1114f62eb027ac9ba756033ce949`;
this check does not publish a runtime.


Shared online publication/auth/storage/admission and actual deployment pins,
World.reset during image preparation, Room-process/persistent-journal recovery,
original ZingShow/Dream input/visual equivalence, worker condition fingerprints,
hour-scale resources, and the480/736 matched benchmark with per-stage timing and
GPU memory remain unverified. Multiplayer remains deferred. Trusted protocol
`visible` is unsupported; the design's explicitly negotiated applied-only mode
is supported, and actual screenshots do not fabricate a visible receipt.

## Retained evidence and reproduction

The original tracked JSON and this full record are preserved byte-for-byte in the
[private evidence index](history/20261109-evidence-index.json). The index records
the authenticated OSS URI, archive/member SHA256, byte counts, format and access
requirements. Upload/download and every member hash were verified on 2026-11-09.
No temporary or expiring signed URL is used. The compact JSON retains outcomes,
source/patch identities, inputs, aggregates and limitations; per-event receipts,
prompt/reference diagnostics and allocation stacks live in the original member.

This archive preserves the repository evidence only. It does not claim to contain
the older 137804792-byte video/journal archive (SHA256
`1b144babd7f7676aafeaf3cd5c885900cf7c1cc22f58634da4dcd4ab70a285f0`) or
263992576-byte follow-up archive (SHA256
`70d40b899c631bf3078c14913b8bdd8170d880645aef8d688a363dbb29d8196d`). Their
current durable locations have not been independently verified in this cleanup.
Use the original indexed record for harness inputs and the
[SDK local model testing entry](https://github.com/seedleap/worldkit-monorepo/blob/7386ddaf03cb0b1eaaa70bff0dca638d7edbb7d8/docs/development/local-model-testing.md)
for reproduction. A historical workload is not permission to start a new GPU run.

## Follow-up scopes, retained results and limitations

- The ten-minute single-player run lasted 600008ms (626.557s process duration),
  produced 13950 model frames and used one model lease. Eighteen sampling windows
  ranged from 22.0647 to 24.1004 FPS; the first/last windows were 23.7406/22.1800.
  These are descriptive windows, not a baseline/candidate speed comparison.
- Frozen follow-up sources: SDK `7386ddaf03cb0b1eaaa70bff0dca638d7edbb7d8`, SGLang
  `b4cc51e2c519f8ba814936cf8adc987c3e632516`, WTP
  `7808e33885fa9ef0021bdd1cc9069b75cf96837c`. Input delivery and model identities
  remain in the JSON. Later fixes are not retroactively included in that GPU run.
- Real World reset regression used PNG/filesystem and a delayed model peer;
  82 source/lifecycle/consumer cases passed. It was not a GPU reset test.
- RenderHost process-kill/restart used the original HTTP/SQLite path and
  deterministic model records, without a GPU lease: 37 journal tests and 16
  preparation-close tests passed; 1 worker-boundary and 72 other-condition cases
  were deselected. A later source-level normal-end/cancel regression does not
  establish current-source real-GPU cancellation or every lifecycle transition.
- Ten-minute stability replaces the historical hour-scale requirement for this
  experiment only. Multiplayer, matched 480/736 performance and per-stage GPU
  memory remain unverified. RSS includes retained harness diagnostics; no
  memory-leak or current-version GPU acceptance conclusion follows.

All failure/unknown flags, measured aggregate values, test counts and follow-up
limits remain in the JSON. The archived complete record supplies the detailed
chronology and source boundaries without becoming a current deployment recipe.
