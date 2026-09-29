# Issue #158 — Ulysses KV writer head stride

## Decision

Draft; no default-enable or overall optimization acceptance. Final implementation
requires `SGLANG_ZING_EXPERIMENTAL_STRIDED_KV_WRITE=1` before process startup;
without it, the preexisting contiguous-head writer remains enabled and new
strided inputs keep the old fallback. No retired switch is restored.

The deterministic SP1, SP2 and SP4 numerical comparisons pass, but the shared baseline and
candidate fail long-video visual quality and source-motion adherence. Numerical
parity, decodable media, writer microbenchmarks, and business quality are separate
gates. SP2 and SP4 each fail all four registered denoise gain gates across the
complete 120-sample matrix. Final default-off GPU checks, all 174 media-file
integrity/decode checks, and all 14 trace audits are complete. This is a deferred
optimization with an opt-in implementation and a complete negative acceptance
record, not a default-enable recommendation.

## Implementation and revisions

| Role | Revision |
|---|---|
| Frozen baseline | `e6537184cdcd9de05e04da6201be70446847b5be` |
| Measured opt-in prototype | `1bcc0b0393d077b6904d2e05de113a27bdcd5261` |
| Final default-off implementation | `7e3dc7e715b37ba98973ef2dc7c1829199110e64` |

The baseline includes #150, #170, #169 and #175. Subsequent changes to the
benchmark add the actual 11,756-row prefix and correct the description of its
fixed local measurement order. The full GPU matrix is frozen on the prototype;
it does not measure the added startup-option predicate in the final revision.
Final guard validation runs separately and is not a full-model performance test.

Ordinary Ulysses returns contiguous `[1,S,Hlocal,3D]`; splitting Q/K/V leaves a
`3D` input head stride. QK normalization precedes this exchange. Independent K/V
row and head strides now reach the existing Triton writer. The new input layout
is restricted to H28/H14, D128, rotary96. Unit inner stride, nonoverlapping rows,
conservative int32 address bounds, BF16/CUDA/no-grad requirements and disjoint
output storage remain required. The existing D-stride specialization adds no
packing. H7/SP8 strided geometry retains fallback. USP, attention math, raw-K
ownership and cache transactions are unchanged. Hardware validation is NVIDIA
CUDA only; HIP remains rejected by the guard and no AMD-device result is claimed.

The competing local batch3 benchmark includes packing all Q/K/V. It does not
change production USP returns and is not a full-model batch3 implementation.

## Frozen business request and environment

- Case: `zing-whitebox-video-five-triview-0001`; six real references plus the
  real source video; 832×480 at 24fps, 10/15/60s, video-only. Audio is not covered.
- Seed 42, CFG 1, ODE step 1000, schedule 1000/750/500/250, indexed-cpu-v1 noise,
  progressive source `h3-clip17-progressive-v1`, window 21 / sink 3 / block-relative / gap 12.
- Verified 5s/120-frame source explicitly repeated to the requested duration.
  These are looped workloads, not independent long source footage; 5s input
  seams are not automatically candidate artifacts.
- Resident model, no offload/FSDP; same FA4/stable-shapes/compiled-AdaLN settings
  and same physical GPU subsets for both trees, SP1/2/4 masks 0/0,1/0,1,2,3.
- PAI Thailand `ap-southeast-7`; actual four NVIDIA L20D, CC10.3, 274,113MiB each;
  preflight memory gate 240,640MiB. Hardware receipts include UUIDs and topology.
- Image digest `sha256:4b9bca7dd61784cd84e43ac760db4c458c08a2d6bf22887e3740455fffe9e015`.
  Torch 2.13.0+cu130, CUDA 13.0, Triton 3.7.1, transformers 5.12.1, FA4 4.0.0b19.
  `torch.cuda.nccl.version()` reports 2.29.7; collective logs report 2.30.7+cuda13.3.
  Both observations are preserved; their cause is not established.
- Model receipt hashes all 66 files (163,140,333,068 bytes). Full baseline source
  receipt verifies 9,455 files; workers verify actual imported critical modules.

Source SHA256: `6579ebd3f1d80455b71e30f9795aa494bceb615ab1302a812430f1199ec80f85`.
Resized source RGB SHA256:
`d917780f9c0bb002fbd646b77f23d442d73b5a42d840c4c911cef74f5af7adbb`.
Detailed checkpoint, six-reference and component provenance remains in the
frozen preparation manifest, including original historical bucket references.
Execution uses `leap-worldmodel-thailand-single-az`.

## Registered acceptance and measurement scope

- Frequency-weighted writer cost, including all actual shapes and any fallback
  or packing: ≥1.2×. Fixed-order microbenchmarks are not end-to-end evidence.
- SP2 and SP4 full denoise interval: ≥2% gain for both 10s and 15s in each A→B/B→A
  round. Five samples per length/configuration/round after complete warmups;
  all samples retained. Independent percentile bootstrap of medians,
  10,000 draws, seed 42; 95% lower bound >0.
- Primary CUDA events wrap complete `SourceReadySession.step`, including prefix,
  packing, eviction, source/final and sampler updates. Rank maximum, never sum.
  Model-only, active-only, host timing and profiler kernel sums are diagnostics.
  API timing includes raw RGB writes to CPFS. TTFF is measured at the client.
- Same-SP actual intermediate tensor and RGB equality; correct shapes/dtypes and
  finite values; no tolerance changes. Correctness observers save actual tensors;
  formal timed requests do not copy/hash intermediate tensors or serialize routes.
- Allocated-peak delta ≤max(256MiB,1%); released-allocation growth over ten
  repeated sessions ≤32MiB. Reserved memory is reported separately. No confirmed
  >2% denoise/API regression in other covered requests.
- Independent full-window traces capture three complete model forwards starting
  before cache preparation/compaction, with separate writer, history rotation,
  coefficient, reserve and compact scopes. CUDA sums do not imply wall-time gain.
- Supplementary audit, registered before formal samples: all 120 formal RGB
  digests/frame counts must match the same-SP/same-length correctness reference.

## Numerical experiments and results

### Original performance encoder: baseline self-comparison fails

GPU `dlcw50du25clyf2v` passed 49 layout cases (25 CUDA + 24 CPU), real SP1/2/4 writer
bitwise checks and 281 transactions per rank under each of three cache policies.
Its full SP1 source/candidate 10/15/60 gate failed before formal performance.
First prefix inputs/outputs match; first source input x `[390,4,24]` FP32 differs
in 10,436/37,440 values (max abs 0.03515625). First active explicit noise is equal,
but source KV already differs. All three durations reproduce this boundary.

Independent same-GPU baseline A/A `dlctngoi2x3umtcz` also fails: first source x
has 3,106/37,440 differing values (max abs 0.033203125), while prefix and initial
active noise match. Zero-GPU audits read the saved tensors directly. This proves
baseline cross-process nonreproducibility under these conditions; it does not
identify a specific backend cause or establish a candidate regression.

### Separate deterministic encoder experiment

Only the model-config semantic field `vae_config.encode_mode` changes from
`performance` to `deterministic`. Real source and references still traverse VAE
encoding. BF16/posterior seed 42/weights remain unchanged. Existing encode policy
disables cuDNN benchmarking and enables deterministic execution within its
context, restoring previous flags afterwards. Both processes additionally use
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, idle 3600s and lifetime 7200s.

| File | SHA256 |
|---|---|
| Original model config | `ab74752bd3f625e2e3d8bf9a1cf16bb0fbf4692f5058c83a581fde8fc60d5333` |
| Deterministic model config | `fbf94deb5906835337d5d41698c9a731be6f099ce6cdfc9a1331ba212caacc57` |
| Deterministic realtime profile | `a9343e430af9cc7ce18e311aca3e68a897cc69dc20c1297d51deecb45caa8a2b` |

GPU-r2 `dlcmpxurukupbrrd` passed the primitive matrix, including 11,756-row prefix,
then stopped before generation: an old experiment-bundle test incorrectly
expected H56/stride384 fusion (4 failed / 45 passed). Retry corrects only the bundled
test to the already committed H14/H28 guard. No production or tolerance change.

GPU-r3 `dlc19r4d8qns3sk5`:

| Scope | Result |
|---|---|
| Layout cases, real worker | 49 passed = 25 CUDA + 24 CPU |
| Real SP1/2/4 primitives | Bitwise/cache checks pass, including actual prefix |
| Independent baseline 15s A/A | 391 actual forward pairs exact; RGB exact |
| SP1 candidate 10/15/60 | 2,187 actual forward pairs exact; RGB exact |
| SP2 candidate 10/15/60 | 4,374 actual forward pairs exact across two ranks; RGB exact |
| SP4 candidate 10/15/60 | 8,748 actual forward pairs exact across four ranks; RGB exact |
| SP1 both-order performance/memory | 40 samples; all denoise CIs cross zero, no confirmed >2% regression; allocated gates pass |
| SP2 performance/memory | 40 samples; all four denoise gain gates fail; allocated gates pass |
| SP4 performance/memory | 40 samples; all four denoise gain gates fail; allocated gates pass |
| Lifecycle receipts | All 30 cases pass: five scenarios × source/candidate × SP1/2/4 |
| Independent full-window profiles | All 14 captures and scope/byte audits pass; three forwards per rank at retained history 20 |

The 15s RGB SHA256 is
`1160535e35e15dd598a0fcf0aca920888334407592e25046acbdc9b8177edef3`;
the 60s RGB SHA256 is
`5a72a3f7050387ca4b401b5a15faf597054516d29805823e5f424b29cda9e318`.
This experiment does not certify the original performance-encoder configuration.

SP2 first 10s real path evidence: both ranks have 13,200 ordinary writer calls
(H28/D128, row/head strides 10752/384; 390 rows / 4,800 calls and 780 rows / 8,400 calls).
All change from baseline head-stride rejection to candidate fusion with identical
layouts: source 2,200, active 8,800 and final 2,200 each rank. Replicated prefix has
52 attention calls with append disabled and does not invoke this writer.
All three SP2 RGB digests also match the corresponding SP1 outputs.

SP1's four denoise median gains (A→B 10s/15s, B→A 10s/15s) are +0.457%,+0.234%,
+1.016% and −0.508%. Their 95% intervals are respectively [−1.053,+1.062]%,
[−0.598,+2.058]%,[−0.647,+1.592]% and[−0.849,+0.656]%. None establishes an
incremental SP1 speedup. API intervals also show no confirmed >2% regression.
All allocated-peak gates pass; each ten-session sequence ends 2,396,160 bytes
(2.29MiB) above its first released allocation, below 32MiB. All 40 SP1 formal RGB
digests and frame counts match the numerical reference. Full raw samples remain
in the complete `analysis.json` alongside all other SP configurations.


SP2 results use all five samples per side and the frozen bootstrap procedure.
None meets both the 2% denoise threshold and positive lower confidence bound.

| Order / frames | Baseline denoise, s | Candidate denoise, s | Median gain | 95% interval |
|---|---|---|---|---|
| A→B / 240 | 32.230 | 32.186 | +0.137% | [−2.707%, +1.588%] |
| A→B / 360 | 47.894 | 48.974 | −2.256% | [−3.699%, +0.175%] |
| B→A / 240 | 32.836 | 33.371 | −1.627% | [−4.761%, −1.122%] |
| B→A / 360 | 48.943 | 49.234 | −0.596% | [−3.520%, +0.785%] |

Corresponding API median gains are +0.303%, −1.966%, −1.426% and −0.621%.
The B→A 240-frame result supports a slowdown; its confidence interval does not
establish a true regression larger than 2%. All eight SP2 per-rank allocated-peak
deltas are zero. Each ten-session sequence has 2,396,160 bytes released-allocation
growth, below the 32 MiB gate. All 120 formal RGB/count checks match the
corresponding numerical reference. The complete frozen analyzer retains its
original random draw sequence; no samples are excluded.

SP4 first 10s has the same writer call frequencies on every rank, with H14/D128
and row/head strides 5376/384. All 13,200 calls per rank fuse; replicated prefix
still makes no writer call. Same-SP 10/15/60s RGB hashes are respectively
`6628f2b0435923d936271502ad5eb14d6ce942409d26cd2b30619d351a6aed2b`,
`dbbac3fa3698b34d73b441bd7d9a44c33e15e2792c5761fe534280c2a2834abe`, and
`6b57c2d6611ecf0182a2c3427314d37421829e5d99a744cc8716e16cffa10165`.
Cross-SP equality is not an acceptance substitute.

Using each rank's actual 390/780-row call frequencies, fixed-order local writer
speedups are 2.119–2.191× for SP2 and 2.158–2.209× for SP4. Batch3 includes all QKV
packing and reaches 1.274–1.365× / 1.348–1.367× respectively, below direct stride
access. This supports choosing A for the prototype; B was not implemented or
measured as a full-model candidate, so no end-to-end A-versus-B winner is claimed.
SP1 microbenchmarks use row stride 7168 while its actual model writer uses 21504
with head stride 128. Their results only diagnose the preexisting writer and do
not establish an incremental SP1 gain.

SP4 A→B uses all five samples per length/configuration. Denoise medians are
34.948680 / 35.842710 s for 10s (gain −2.558%, 95% interval [−4.065%, −0.248%])
and 52.123635 / 53.194763 s for 15s (−2.055%, [−4.255%, −1.106%]). Both support
a slowdown and fail the gain gate; neither interval establishes a true regression
larger than 2%. API median gains are −1.893% and −1.836%, with intervals crossing
zero. Both allocated-peak gates pass. B→A also fails both gain gates: +0.84%
and +0.23%, with both confidence intervals crossing zero. The complete analysis
keeps every prior draw in the frozen order; all 120 formal RGB/count checks match
their same-SP numerical reference.

## Complete formal timing matrix

All 120 formal RGB digests and frame counts match their same-SP, same-length
correctness references. SP1 uses its registered non-regression gate; passing it
does not establish a speedup. SP2/SP4 use the ≥2% gain and positive 95% lower-bound
gate. Both orders retain all five samples per configuration.

| SP | Order | Frames | Denoise baseline/candidate, s | Denoise gain [95% CI] | API baseline/candidate, s | Denoise gate |
|---|---|---|---|---|---|---|
| 1 | A→B | 240 | 33.288 / 33.136 | +0.46% [-1.05, +1.06] | 45.401 / 45.409 | pass |
| 1 | A→B | 360 | 50.478 / 50.359 | +0.23% [-0.60, +2.06] | 67.983 / 67.829 | pass |
| 1 | B→A | 240 | 33.313 / 32.975 | +1.02% [-0.65, +1.59] | 45.438 / 45.462 | pass |
| 1 | B→A | 360 | 50.384 / 50.640 | -0.51% [-0.85, +0.66] | 67.974 / 68.297 | pass |
| 2 | A→B | 240 | 32.230 / 32.186 | +0.14% [-2.71, +1.59] | 44.847 / 44.711 | fail |
| 2 | A→B | 360 | 47.894 / 48.974 | -2.26% [-3.70, +0.17] | 65.867 / 67.162 | fail |
| 2 | B→A | 240 | 32.836 / 33.371 | -1.63% [-4.76, -1.12] | 45.349 / 45.996 | fail |
| 2 | B→A | 360 | 48.943 / 49.234 | -0.60% [-3.52, +0.79] | 67.069 / 67.486 | fail |
| 4 | A→B | 240 | 34.949 / 35.843 | -2.56% [-4.07, -0.25] | 48.179 / 49.091 | fail |
| 4 | A→B | 360 | 52.124 / 53.195 | -2.05% [-4.25, -1.11] | 70.636 / 71.933 | fail |
| 4 | B→A | 240 | 35.483 / 35.184 | +0.84% [-1.52, +1.80] | 48.107 / 48.515 | fail |
| 4 | B→A | 360 | 52.782 / 52.661 | +0.23% [-0.43, +1.34] | 71.073 / 71.161 | fail |

Median values above; complete min/max and all five samples per group remain in `analysis.json`.

| SP | Order | Frames | Baseline peak allocated, MiB | Candidate peak allocated, MiB | Delta, MiB | Memory gate |
|---|---|---|---|---|---|---|
| 1 | A→B | 240 | 234918.7 | 234918.7 | +0.0 | pass |
| 1 | A→B | 360 | 234918.7 | 234918.7 | +0.0 | pass |
| 1 | B→A | 240 | 234918.7 | 234918.7 | +0.0 | pass |
| 1 | B→A | 360 | 234918.7 | 234918.7 | +0.0 | pass |
| 2 | A→B | 240 | 186755.6 | 186755.6 | +0.0 | pass |
| 2 | A→B | 360 | 186755.6 | 186755.6 | +0.0 | pass |
| 2 | B→A | 240 | 186755.6 | 186755.6 | +0.0 | pass |
| 2 | B→A | 360 | 186755.6 | 186755.6 | +0.0 | pass |
| 4 | A→B | 240 | 162630.6 | 162630.6 | +0.0 | pass |
| 4 | A→B | 360 | 162630.6 | 162630.6 | +0.0 | pass |
| 4 | B→A | 240 | 162630.6 | 162630.6 | +0.0 | pass |
| 4 | B→A | 360 | 162630.6 | 162630.6 | +0.0 | pass |

## Independent trace diagnosis (SP1/SP2/SP4)

The final zero-GPU analysis covers all 14 original per-rank traces. Each contains
three full-window forwards (127–129, retained history 20 with current completing
window 21), beginning before prepare/compaction. The earlier six-trace analysis
from job `dlcnvt2buvh89k6s` is retained under `trace-preaudit-r1`; all six raw trace
SHA256 values match the final audit. All 14 also match the original export manifest.
`final-supplemental-r1/trace-summary.json` and `copy-byte-audit-all.json` retain
per-rank host, kernel, memory, coefficient, reserve and attribution details.

SP2 rank 0 current writer changes from 1,350 kernels / 5.463 ms to 150 kernels /
0.651 ms. Its inclusive CPU scopes change from 32.568 to 18.083 ms. In the same
captures, history RoPE remains 17,200 kernels / approximately 167 ms. SP1 writer
stays at 150 kernels / approximately 1.03 ms on both sides. These observations
confirm the local mechanism, not a formal end-to-end gain or a causal explanation
for the measured slowdown.

Ulysses output exchange contains `spin_wait_kernel` (rank 0 source 166.649 ms,
candidate 368.533 ms in these particular captures). The generic NCCL-name subtotal
does not include that kernel and is not a complete communication accounting.
Do not sum overlapping ranks or use these independently profiled captures as
unprofiled A/B performance estimates.

Original SP2 rank 0 source and candidate files were also opened locally in Perfetto
v58.3. UI SQL confirms 150 candidate `_zing_rope_kv_write_kernel` events, while the
source has none. Import diagnostics report respectively 400/812 overlapping
complete-event spills: Perfetto retains them on overflow tracks but nesting is
ambiguous. Correlation therefore uses launch identifiers and host scope ownership,
not rendered nesting or GPU timestamp clipping. Unassigned/ambiguous kernels
remain explicit. CPU intervals, GPU annotations, correlated kernel sums and memory
operations are separate views; annotation spans include waits and differ from
kernel sums. Detailed UI observations are in `perfetto-ui-review.json`.

The SP2 rank 0 byte audit attributes 3,400 DtoD memcpy events and 32,428,032,000
payload bytes to the single compaction in each capture (source 13.281 ms;
candidate 13.276 ms). Those copies remain unchanged. Baseline current writer has
300 `aten::copy_` calls with 524,160,000 inferred destination bytes; candidate has
none in that scope. History RoPE retains 2,150 pass-through copies with
6,894,003,200 inferred destination bytes. Memcpy payload bytes are recorded trace
arguments; tensor-copy bytes derive from shape/dtype. The two views overlap and
must not be added, and neither measures actual DRAM traffic or bandwidth.
`copy-byte-audit.json` preserves unassigned copies and the counting semantics.

SP4 confirms the same local mechanism on all four ranks: current-writer kernels
fall from 1,350 / 4.439–4.476 ms to 150 / 0.461–0.463 ms, while history RoPE stays
at 17,200 kernels / 103.2–104.2 ms. Every SP4 capture retains one compact with
3,400 DtoD copies and 16,214,016,000 payload bytes. Rank 0 compact is 12.464 ms
baseline / 12.502 ms candidate. Its history pass-through copies retain
3,447,001,600 inferred destination bytes; current-writer inferred copy bytes fall
from 262,080,000 to zero. These independent trace timings are diagnostic only.

Across all 14 traces, the history pass-through32 copy payload corresponds to
3,847,100 rows summed across layer calls per capture. This is not a count of
unique cache rows. The SP2 rank 0 compact has 1,700 clones, with 1,700 internal
copies plus 1,700 explicit copies; each group contains 1,200 BF16 copies shaped
[1560,28,128] and 500 shaped [780,28,128]. K/V are already included, so do not
multiply the reported payload by another K/V factor. Compaction precedes the
captured source forward; no enclosing causal block/forward is recorded, so its
block/forward attribution remains null rather than being invented.

## Supplemental latency, memory and startup scope

All 28 per-rank allocated-peak comparisons pass with zero delta. All 28 ten-session
released-allocation sequences pass with 2,396,160 bytes (2.29 MiB) last-minus-first
growth, below 32 MiB; reserved allocations remain separately recorded. All 30
lifecycle scenarios pass across the six source/candidate SP processes. The 120
formal requests have correctness/profile observers disabled and no saved raw
intermediates, layout/route maps or profiler captures; raw output RGB is still
written for the separately registered output-integrity check.

The supplemental artifact retains all 24 warmup requests, every per-step
rank-maximum CUDA duration and every full-window model forward. For orientation,
the following values pool both durations and orders within each row; they are
descriptive and do not replace the registered paired configuration comparisons.

| SP / tree | TTFF median [min, max], s | Step median, ms | Full-window forward median, ms |
|---|---|---|---|
| 1 / baseline | 4.418 [4.350, 5.227] | 767.219 | 88.656 |
| 1 / candidate | 4.431 [4.368, 4.800] | 765.682 | 88.648 |
| 2 / baseline | 4.795 [4.692, 5.668] | 756.907 | 98.097 |
| 2 / candidate | 4.844 [4.724, 5.250] | 769.610 | 99.808 |
| 4 / baseline | 5.049 [4.991, 5.395] | 825.258 | 108.133 |
| 4 / candidate | 5.108 [5.021, 5.678] | 833.048 | 109.787 |

The first 240-frame warmup starts after process health readiness. Its API times
range from 50.988–51.638 s at SP1, 52.339–53.930 s at SP2 and 57.497–58.445 s at
SP4 across trees/orders. These include first-request work but exclude process/model
load: exact launch-to-readiness wall time was not instrumented. Screening all 12
formal server logs finds no compilation/recompilation messages after the first
formal session; absence of such messages is not hardware-level proof of no JIT.

Formal requests total 16,338.864 assigned-SP GPU-seconds, or 27,639.590 GPU-seconds
when charging the full four-GPU allocation during those requests. These are GPU
count × API wall-time accounting views, not measured SM activity or full-job
billing. `supplemental-metrics.json` retains their definitions and all underlying
rows. No denoise/API confidence interval lies entirely below −2%; this does not
rescue the failed positive-speedup gates.

## Media and business quality: failed independently of numerical parity

The final audit covers exactly 174 outputs: 18 correctness, 120 formal, 24
performance warmups and 12 profile/warmup outputs. Every raw RGB size/hash, frame
count, 832×480/24fps specification and full MP4 decode passes. All 174 MP4 hashes
match the original export manifest. The previously reviewed SP1/SP4 raw and MP4
hashes are unchanged; SP2 correctness RGB matches SP1 for all three durations.
This complete file audit is separate from the manual quality review below; it
does not claim frame-by-frame manual inspection of all 174 videos.

Original six outputs and deterministic SP1 six outputs pass raw digest, expected
240/360/1,440-frame counts, 832×480/24fps and full decode checks. Zero-GPU media
jobs `dlc1nmk2dmu88zsr` / `dlcdkpatm3zid80b` exported 17/23 verified artifacts.

Full-duration1fps sheet review found severe original60s defects on both sides.
Dense8Hz checks separate input-loop transitions from persistent corruption.
Original-mode candidate causality remains unresolved because source inputs differ.

Deterministic baseline/candidate MP4 and corresponding sheets are byte-identical.
Review of all five unique full-duration sheets finds geometry drift around10–19s,
ranger shrink/disappearance20–29s, oversized wheels/vehicles and horizontal neon
sky bands30–49s. Scene reset near50s coincides with an explicit source-loop seam.
The persistent defects before this seam fail long-video quality.

Independent #181 review of the identical15s output, sampled at4fps across its
full duration, also finds failed source-motion adherence: the running source is
rendered mostly standing/moving little around2.5–7.5s, then walking. #158 inspected
this same sheet. These shared defects are not attributable to compaction or the
strided writer. Sampled review is not frame-by-frame manual certification.

SP4's own zero-GPU media job `dlc1um9k3umt83hq` verifies all six outputs, including
full decode. All 23 exported artifacts were SHA256/size-verified locally. Three
MP4 pairs and five corresponding sheet pairs are byte-identical. Manual review
of all five unique 1fps sheets across 10/15/60s finds exaggerated terrain ridges
and geometry stretching around 20–34s, followed by broad sky/horizon color bands
around 35–49s. A reset near 50s coincides with the explicit input-loop seam and is
not counted alone as a candidate defect. Distorted ground/platform and banded
sky recur afterwards. The identical 15s raw hash also matches #181's independently
reviewed 4fps sheet, showing limited movement around 2.5–6s despite a continuously
running source. SP4 business quality therefore fails on both sides. Evidence:
`media-deterministic-sp4-r2/visual-review.json`; same-SP equality does not repair
this quality failure.

## Reproduction and evidence

```sh
SGLANG_ZING_EXPERIMENTAL_STRIDED_KV_WRITE=1 \
  torchrun --standalone --nproc-per-node=4 \
  benchmark/zing_h3_causal/validate_ulysses_kv_stride.py --output /results/sp4
python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_zing_rope_kv_stride.py -q
```

The exact measured whole-model entry point is
`/cpfs/world-model/sglang/evals/zing158-20260923/scripts-deterministic-r2/gpu-r3.sh`.
It launches each baseline/candidate in its own process, selects the same GPU mask,
then invokes frozen `matrix.py` for correctness, two performance orders, lifecycle
probes and independent profiles. `deterministic-r2/manifest.json` pins every script
and input config. Reproduction requires a fresh output namespace; the archived
experiment directory is immutable. This command measures prototype `1bcc0b0`,
not the final default-off predicate.

The final CPU regression command covers `test_zing_rope_kv_stride.py`,
`test_zing_h3_model.py`, `test_zing_h3_window.py`, and
`realtime/test_zing_progressive_source_session.py`, after importing `zing_h3`
for the repository's existing circular-import bootstrap. Exact stage/probe
commands and all file hashes are retained under `default-off-r1`.

Run the collective harness with 1/2/4 ranks. Each rank checks 70 chunks and 281
operations per cache configuration: window2/sink0/relative, window21/sink3/relative,
and full-history/absolute, including prefix replacement, eviction, source/active/
final, nonzero offsets, sentinel protection, raw-K preservation and failure cleanup.
Final implementation has 51 layout cases (27 CUDA + 24 CPU). Its independent zero-GPU
regression job `dlcs0cm4q924bqce` passed 706 tests with 27 CUDA skips; all 8 staged
file hashes and the 7e3dc7e revision receipt were verified. Final CUDA guard checks
ran in job `dlczk6wwa6tlldu6`, which succeeded: all 51 layout cases passed
(27 CUDA + 24 CPU, 4.17s). Actual unset/0/1 probe output confirms the intended
routes and bitwise results for H7/H14/H28/H56. The real SP1 collective harness
passed all three policies with 281 transactions each, including prefix replacement
and failure cleanup. Short zero-GPU export job `dlc185zy16rf26y1` succeeded; all
nine output files passed local size/SHA256 verification, and all eight staged
code files match the final revision manifest. Evidence is also retained in
`final-inspection-r1/gpu` and `final-guard-verification.json`.

Evidence root on CPFS: `/cpfs/world-model/sglang/evals/zing158-20260923`.
OSS results: `oss://leap-worldmodel-thailand-single-az/world-model/sglang/evals/zing158-20260923/`.
Frozen code/manifests: same bucket, `world-model/code/sglang/zing158-20260923/`.
Local control: `/tmp/zing158-20260923/`.

Key subdirectories: `diag-r1`, `aa-audit-r1`, `deterministic-r2`, `gpu-r3/gates`,
`media-original-r1`, `media-deterministic-sp1-r1`, `media-deterministic-sp4-r2`,
`default-off-r1`, `final-inspection-r1` and `final-supplemental-r1`. Raw tensors/RGB
remain on CPFS; the original export manifest records 1,985 artifacts. Local
verification checks all 12 small final-summary files and nine final-GPU files by
size/SHA256, cross-checks 174 MP4 and 14 raw-trace hashes against the original
manifest, and verifies controller actions 005/006/007 against their frozen script
hashes. `final-local-verification.json` records the final checks. Media reviews
retain explicit scope/attribution limits. No optimization success, issue closure
or merge is claimed.

All 21 jobs created under this experiment prefix are terminal, including preserved
failed attempts. Controller `dlc1jgex3eyo3byc` exited successfully after the explicit
sequence-008 stop request; main GPU, final GPU and short inspection jobs succeeded.
`final-job-statuses.json` records the final inventory. No unrelated job was stopped.
