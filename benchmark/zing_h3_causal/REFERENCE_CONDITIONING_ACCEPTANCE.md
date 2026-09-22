# PR #164 reference conditioning acceptance

## Corrected contract

- Up to nine references; a tenth is rejected at initialization.
- Preserve client reference IDs, order and original bytes, including PNG.
- The confirmed global rendered reference remains Picture 1 and the first reference.
- Original paired sheets retain Picture 2–6 and original prompt/condition mapping.
- RGB source frames are never implicitly injected into prompt media.
- Existing eight-MiB aggregate and decoded pixel budgets remain enforced.

## Frozen input

Case ID remains `zing-whitebox-video-five-triview-0001` for compatibility; it now
contains six references and seven original assets. The legacy name does not
mean that the global style reference is omitted.

Frozen single-case manifest SHA256: `a97154b3010064e846fe0ad7b6cc21de3704885c7f0eba5c618231de395e2947`.
First reference PNG: 1280×720, 2186467 bytes, SHA256
`03475dd7769de747f16b0d13709495cce426c310a62822b93eb38db6eb92fa05`.
Original prompt and reference conditions exactly match source manifest line 10,
whose manifest SHA256 is `d24cddb259eb7e22dc9b763648934394e98e0818c80a744f79b2fc947176cbed`.

## Standalone PR CPU validation

Runtime SHA: `5d95d9011b51a7a54bfb1dfa88fda2ab100fac73`.
PAI job `dlc02gdfuu2f5006`: Succeeded, exit 0, **1170 passed / 55.84 seconds**.
This is the standalone PR runtime, not the combined #148 result.
The source overlay includes every changed executable/test/case file over frozen
base `74d19dde391717b5f57c8690a6027768e2cd3f6d`; docs are excluded from the runtime overlay.

Coverage includes nine-image mixed-PNG acceptance, ten-image rejection, ordered
reference bytes/IDs, 5/9-image codec routing with no injected prompt video,
original Picture 1 and six-reference case mapping, shared source/runtime
lifecycle and TAEH3 contracts. All seven original asset hashes were verified.

GitHub `taeh3-contracts` for this head passed in 41 seconds:
https://github.com/seedleap/sglang/actions/runs/35721743685/job/106725994622

CPU logs, exit receipt and asset receipt are in private OSS:
`world-model/sglang/evals/zing-h3-dmd/zing164-acceptance-20260922/cpu-r1/`.

## Client follow-up

Commit `75032959029b896364df1e760ee42c42f14fde2a` adds actual-format detection
for hash-named PNG assets and explicit reference IDs to the standalone client,
using the same media helper already tested in the combined GPU deployment.
Server adapter/codec/source-session files are byte-identical to the standalone
CPU-tested `5d95d9011b` versions (`runtime-equivalence.json`). Local client,
lifecycle and case tests: 16 passed / 0.54 seconds. This head's CI passed in
45 seconds: https://github.com/seedleap/sglang/actions/runs/35723301648/job/106730937629.

## Combined regression

#148 combined SHA `386543ac257d7fd190707a1a1796ad6006a4873d`, including equivalent
PNG allowlist and this reference correction: CPU `dlct76xhmx0dlt7o`, Succeeded,
2001 passed / 64.13 seconds. This additionally covers its client MIME sniffing
and progressive encoder changes; it is not relabeled as a standalone #164 run.

## GPU acceptance

SP1 corrected six-reference `clip17-120-r1` completed in existing job
`dlccjwp3efr86pqz` (combined SHA `386543ac257d7fd190707a1a1796ad6006a4873d`).
The individual case passed; the larger #148 matrix is tracked separately.
Actual request IDs are Picture 1–6 in original order, `prompt_video_frame` is
null and ready advertises nine references. Submitted/accepted/consumed/output
all equal 120, maximum pending is 60. Independent replay verified 23 append
batches and 29 output headers, consecutive alignment and PTS, complete consumed
boundaries, explicit completion and cleared EOS state.

Read-only CPU export `dlc1xzngga298c7j` Succeeded / exit 0. The output RGB hash
matches the session receipt; the MP4 was fully decoded and checked as 832×480,
120 frames, 24 FPS media time base, 5 seconds. This does not claim 24 FPS inference.
MP4 SHA256: `7eca7e6db56a2cd0e38ff4ec23961379c9d5edd7961bc9ca77b2d86c76f28af0`.
Private OSS evidence: `world-model/sglang/evals/zing-h3-dmd/zing164-acceptance-20260922/media-r1/`.
Four sampled frames (0/24/60/119) show colored-pencil texture and sandy palette
consistent with the global reference; this is only a spot check.

SP2 corrected six-reference `clip17-120-r1` completed in existing job
`dlcsnhkp621hpxuc`, with the same combined runtime and frozen input. All four
counts equal 120; independent replay verified 23 append batches, 29 output
headers, consecutive alignment and PTS, and consumed boundaries. Both ranks
ended with empty EOS state. Actual server cleanup traces confirm scheduler
release and input-file removal for SP1 and SP2.

Read-only SP2 export `dlc10z9lvug43k2i` Succeeded / exit 0. Its logged receipt
confirms full FFmpeg decode and 832×480 / 120 frames / 24 FPS media time base.
Raw RGB SHA256: `955d1f2b14cb7677a4c6e9d5af564b8887b3005062a4d98ab40fbc00c2de49b2`.
MP4 SHA256: `91e9f88b7535bd76c6ed093db9b838de77c9b3d3592b5a517100f08c772fb227`.
CPFS export: `/cpfs/world-model/sglang/evals/zing-h3-dmd/zing164-acceptance-20260922/media-r2/`.
The corresponding private OSS prefix is `world-model/sglang/evals/zing-h3-dmd/zing164-acceptance-20260922/media-r2/`.
After export, OSS returned `403 AccessDenied: Access denied by bucket policy`
on local reads (2026-09-22 12:19 UTC). Therefore local SP2 media readback and
visual spot review remain incomplete; job success is not claimed as verified
OSS delivery. The control-plane export receipt and already downloaded GPU
protocol evidence remain available. SP1 media was downloaded and reviewed
before this access change.

### Live SP2 reference limits

In the same SP2 job, `reference-limit-nine` accepted nine ordered unique IDs,
with the original PNG first, and completed one submitted/accepted/consumed/output
frame. Both ranks drained and actual session cleanup completed. The additional
three paired images repeat existing content under unique IDs: this is a limit
test, not a nine-subject quality test.

`reference-limit-ten.json` records rejection before worker generation. Public
error content was `invalid generate request`; the server validation reason was
`Please provide at most nine reference images.` The session cleanup trace
confirms scheduler release and input-file removal. No worker step was dispatched.
Raw protocol receipts and server snapshots are under private OSS prefix
`world-model/sglang/evals/zing148-final-20260922/sp2-r2/` and were downloaded
before the access change.

No duplicate GPU matrix was submitted. Individual acceptance cases are complete;
the broader #148 performance jobs are owned and tracked separately. Historical
five-reference/injected-RGB results are not evidence of corrected conditioning.

## Main integration and final scope

PR #168 landed the two independently selected cases on main. This PR merged
main at `cb49dcd0cdd0b4fe3d7d8bbc05d966c3aaaa30b7`, preserving the default
six-image case and the separate explicit-video case
`zing-whitebox-video-target-frame0-five-triview-0001`. The latter is not the
input used by this acceptance and must be selected explicitly.

At integration head `5f17bfbdf2775f02bb758f43168ae2f21456cf05`, every file under
`python/` is unchanged from the standalone CPU-tested head. The default case
JSON object is identical to the GPU-frozen input. Adding the second case and
reserializing changed the whole manifest SHA256 to
`db1086690d7dbe31c58029384f5f5928ee26191b02b36e4b857362a29d59f1e0`;
the default record SHA256 is
`adca3c24e264fcca2df68e56dbc4893346a408df83e57a994d4255820740916e`.
These must not replace the frozen GPU manifest hash in its receipt.

Post-integration case, media-client, lifecycle-probe and profile-tool tests:
**38 passed / 53.37 seconds**. Ruff and diff checks passed. Integration CI:
https://github.com/seedleap/sglang/actions/runs/35725322805/job/106737470808.

The functional gate passes: original first-reference identity/order, nine/ten
limits, PNG byte preservation, no implicit source-frame prompt, real SP1/SP2
output, protocol alignment, EOS and cleanup. Media decode passes for both.
SP2 manual visual review and remote delivery are not marked passed.

Performance/24 FPS and complete subjective quality remain separate follow-ups.
