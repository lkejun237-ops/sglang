> Current deployment and client size limits: [bounded source transport](SOURCE_TRANSPORT.md).
> This contract explicitly rejects exact 1280x720 for the H3 32-pixel grid.
> High-resolution results: [three-resolution source transport acceptance](SOURCE_TRANSPORT_ACCEPTANCE_20260928.md).

> Conditioning correction: the current contract accepts up to nine references,
> preserves their supplied order, and never injects the first source RGB frame
> into prompt media. Historical runs below that used this injection are not
> evidence of the corrected conditioning or visual quality.

# Current acceptance

The earlier runs below are historical and retain their own SHAs. See
[final shared-runtime acceptance](CONTINUOUS_SOURCE_ACCEPTANCE.md) for the
new frozen runtime, complete SP1/SP2 gates, and terminal-job evidence.

# Zing H3 continuous RGB validation

The source-stream entry point is opt-in. A request must include
`zing_realtime_api_version=1` and an explicit `source_stream` map; finite-video
requests keep the existing adapter. The service returns `session.ready` before
the first source append, accepts RGB24 batches with consecutive frame indices,
and returns `source.accepted`, `source.consumed`, frame provenance, and
`session.completed`.

The production client is:

```bash
python benchmark/zing_h3_causal/continuous_source_client.py \
  --url ws://HOST:30000/v1/realtime_video/generate \
  --source-rgb - --output /tmp/zing-h3-run \
  --prompt 'Preserve the game character and motion' \
  --width 832 --height 480 --paced --batch-pattern 1 3 17 2 5
```

`--source-rgb -` reads a live RGB24 pipe. A file may be used for a finite
reproduction, but it must contain real frames and is not evidence of unknown
duration by itself. The client waits for negotiated frame and byte credit,
checks accepted/consumed boundaries, validates output alignment and timestamps,
and writes raw frames, event trace, hashes, TTFF, source-to-output p50/p95/max,
effective delivery FPS, and playback startup-buffer stalls.

The continuous profile requires a deployment-attested
`zing_realtime_profile.json` beside `zing_config.json`:

```json
{
  "solver": "few_step",
  "denoising_step_list": [1000, 750, 500, 250],
  "guidance_scale": 1,
  "noise_version": "indexed-cpu-v1",
  "model_config_sha256": "<sha256 of zing_config.json>"
}
```

The server rejects an override that changes checkpoint sampling semantics. The
initial implementation is video-only, RGB24 input, local decode, stable output,
and bounded source/codec/AR/decoder queues. It does not support H.264 input,
audio streaming, session resumption, implicit frame duplication, or a
client-selected attention window. A fresh connection starts a new generation
after the previous session releases its admission lease; a client reconnecting
during cleanup can receive a transient admission rejection. A failure sends a friendly English error and releases the
session; a normal close without `session.completed` is incomplete.

## Evidence boundary

The CPU integration bundle and test result are recorded in issue #131. CPU
tests verify protocol accounting, codec-to-chunk geometry, transport metadata,
window lifecycle, and existing Zing realtime regressions. GPU tests must use the
same source commit, checkpoint, four-step profile, 832x480 canvas, real RGB
source, and no profiler for the throughput runs. Browser presentation and WAN
latency are separate measurements from local WebSocket delivery.

## Frozen baseline evidence (2026-09-21)

The native-decoder source baseline at `8392845efd` ran in job
`dlczr6brew2wiciy`, on measured NVIDIA L20D hardware (CC 10.3, 274113 MiB),
Torch 2.13.0+cu130, with SP1 and the four-step profile above. The input was a
non-looping 240-frame montage of distinct UniEdit clips, delivered progressively
through an RGB pipe without declaring total length. This tests protocol duration,
not continuous-scene visual quality or a live game-engine connection.

All five sessions (warmup, three measured runs, paced input) accepted, consumed,
and delivered all 240 frames and completed EOS. The three measured runs had
2.1610 / 2.1619 / 2.1627 effective delivery FPS, with median TTFF 5.581 s.
Their input and output hashes agreed across runs. The paced run had 103.384 s
p95 source-to-output latency and 97.045 s playback stall time with a 0.5 s startup
buffer. Maximum submitted-but-unconsumed input was 68 frames. This baseline
**fails** the 24 FPS gate. It uses absolute RoPE and native decode; these numbers
must not be attributed to the later relative-RoPE/TAEH3 combination.

Per-dispatch `*_host_ms` values are wall-clock host observations, not CUDA-event
or profiler-exclusive kernel times. Raw output and full traces remain in the run
directory; summary JSON, logs and encoded media are exported under
`world-model/sglang/evals/zing-h3-dmd/zing131-20260921/gpu-sp1-r2/`.


## Relative-RoPE / TAEH3 source evidence (2026-09-21)

Commit `5ac5eee1e6b78d291deae8699fc723e06f9e6f3d` fixes window/sink to
21/3 and block-relative RoPE with maximum frame gap 12. Both runs use the same
four-step checkpoint, TAEH3 stable decoder, 832x480 RGB pipe, irregular append
pattern 1/3/17/2/5, and measured hardware described above.

| Measurement | SP1 (`dlcjnzhu29iahxc3`) | SP2 (`dlcprtde29qiecax`) |
| --- | ---: | ---: |
| Three unprofiled 240-frame delivery FPS | 5.5940 / 5.5805 / 5.5762 | 8.9014 / 8.9829 / 8.9235 |
| Paced 1440-frame delivery FPS | 3.2565 | 5.5534 |
| Paced 1440-frame elapsed seconds | 442.192 | 259.299 |
| Paced 1440-frame TTFF seconds | 2.389 | 2.019 |
| Paced 1440-frame source-to-output p95 seconds | 373.070 | 194.754 |
| Paced 1440-frame playback stall seconds | 380.963 | 198.433 |
| Maximum pending source frames, long run | 67 | 67 |

Both sizes accepted, consumed and delivered every frame. The 240-frame output
SHA-256 is `616be0870bc87ec67fa9b8edc697c32bd025596cebaa6a4f14abdcde11a7e4b6`;
the 1440-frame output SHA-256 is
`01705a49499fb2a8ec4cf68ec9c2270518ece2b28b0964780ec0a8a9bc5c67f4`.
These hashes agree between SP1 and SP2. Both fail the 24 FPS acceptance gate.
The long run is one paced measurement per size, not three warmed long-run
measurements. The 60-second source is a non-looping montage, not a continuous
scene or live game-engine capture. Visual quality remains unaccepted.

SP1's 39 full-window dispatches retained exactly 117445345328 bytes of KV.
Host RSS varied by 58.14 MiB; allocated CUDA memory varied by 38.85 MiB.
Reserved CUDA memory grew by approximately 1 GiB. EOS cleared per-session KV,
queues, metadata and static state; model allocations and allocator reservations
remained. This is bounded-state evidence, not a claim that EOS frees all CUDA
memory. The r4 logger emitted rank 0 only: SP2 has no independent rank-1 memory
record. The subsequent revision enables every-rank logging.

SP2 also completed 1-, 17- and 18-frame EOS cases and a 51-frame slow-consumer
case with a two-second delay per output batch. Its job nevertheless exited 1:
the lifecycle probe received the expected invalid-sequence error, then opened
the next connection before the previous admission lease had been released.
The second connection received `USER_SESSION_LIMIT` during the approximately
2 ms cleanup interval. Cancellation and reconnect-after-cancellation were
therefore **not executed** by that probe. This failed probe must not be counted
as a passing lifecycle test.

Artifacts are exported under
`world-model/sglang/evals/zing-h3-dmd/zing131-20260921/gpu-sp1-r4/` and
`gpu-sp2-r4/`. The legacy r4 summary field `max_latency` was unused and always
zero; use `source_to_output_max_seconds` instead. Later client revisions remove
the unused field.


### Visual failure observed

Six sampled source frames from the 10-second input show a hard cut from a woman
in a desert to a vase/rose scene in the second half. At corresponding sample
times, both the native-decoder and TAEH3-decoder **SGLang source** outputs keep
the woman/desert scene. This is a source-following failure for this montage.
“Native decoder” here does not mean an independent native-Zing pipeline result.
The failure appears with both decoders; that observation does not isolate its
cause to source encoding, conditioning, autoregressive history, or model
behavior. A native-Zing comparison with the same request is still required.


## Lifecycle probe

`continuous_source_lifecycle_probe.py` takes at least 51 real RGB24 frames and
a JPEG reference via `--rgb`, `--reference`, and an empty `--output` directory.
Run it against an otherwise idle worker using the same `--url` as the source
client. It records every admission attempt, including transient rejections, and
allows at most 30 seconds to acquire a fresh session. This tests fresh-session
recovery, not continuation of a cancelled generation.

Cases cover invalid sequence, disconnect during work, disconnect just after
EOS, recovery after each cancellation, same-user concurrent admission rejection,
a two-second source pause followed by a prompt update and resumed input, and a
reference-image request. Passing the same-user rejection case establishes only
that admission boundary, not independent concurrent users or scheduler fairness.
The prompt case checks output event provenance; it does not score visual prompt
following. Keep the failed-attempt logs even when a later fresh connection
succeeds. The SP8 run uses the separately frozen probe hash recorded in its
job command; adding this tool is not evidence that its GPU cases have passed.


## SP4 observability revision

Frozen source `31f935812a920e50a08a293714b5d22a49786aed` merges main's
per-rank profiling evidence and enables every-rank source logs. Job
`dlciu953et479vwq` passed 889 CPU regression tests in the official image.
Its three unprofiled 240-frame runs delivered 13.1679 / 13.1944 / 13.1317 FPS
(median 13.1679), with TTFF 1.1685 / 1.1667 / 1.1698 seconds. All three outputs
have the same 240-frame hash recorded above. This still fails 24 FPS. The three unprofiled 1440-frame runs delivered 9.5077 / 9.4773 / 9.5070 FPS
(median 9.5070), all with the same long-output hash recorded above. Paced
1440-frame delivery was 9.4105 FPS, with 91.228 s source-to-output p95 and
92.278 s playback stalls. All accepted/consumed/output counts are 1440.
The first nine encoded videos passed full decoding and geometry checks; paced
long media and profile validation are recorded separately below. Compare with r4 as diagnostic
scaling evidence: r5 also adds per-rank memory collection, so it is not an
otherwise byte-identical source revision.


The SP4 job completed with exit 0. Five long sessions (warmup, three measured,
paced) provide 20 rank/session records. Each full-window KV storage is fixed
at 29362810928 bytes. Maximum within-session RSS spans for ranks 0/1/2/3 were
63.90 / 31.91 / 51.30 / 82.15 MiB. Every long-session EOS cleared its retained
KV, source queues, metadata and static condition storage.

The separate profiled 240-frame run took 268.004 seconds and retained the same
output hash. Its overhead is not corrected or used as throughput evidence.
There are 36 exported raw traces (nine dispatches, four ranks). The trace covers
the 10-second request, not the fully occupied 60-second attention window.
Raw traces include CPU scopes, CUDA kernels and launch correlation IDs.
Perfetto opened a raw rank-2 trace but reported 88 depth and 5454 overlapping
complete-event import errors. A UI-only view retains selected CPU annotations,
CUDA runtime/memory/kernel events and places overlaps on separate sublanes,
without changing timestamps or durations. That view was opened and visually
checked with visible Zing stage/block/denoise scopes and CUDA streams and no
import warning displayed. Thread nesting and flows are not preserved in this
view; numerical attribution must continue to use the unchanged raw traces.


## SP8 deferred for shared capacity

At the user's request, SP8 job `dlcqc2e4q9w1lm0z` was stopped while still
Queuing and confirmed Stopped. It did not execute GPU work. Do not automatically
resubmit it. SP4 is Succeeded; remaining trace analysis runs locally on CPU.
SP8 performance and the extended lifecycle probe remain unvalidated.

## ODE step1000 SP2 visual comparison

At the user's request, a single 240-frame visual-quality run used
`h3-ode-sp4-01-0920-test-507a242e6`, step1000, with SP2. CPU export job
`dlc1didxmdei2den` and GPU job `dlcnkq496j5f4iwz` both Succeeded / exit 0.
The original generator SHA-256 is
`09b0348a66f567c3f708cf6298a794950bb7e3e6e506fc00aa6824db5e75b0ba`;
the training export configuration SHA-256 is
`e639e1bb3d107fd501fa150cebc45f09fe19519ea23bd1a1fec9084915ff98da`.
The export contains 638 tensors across 19 shards. It preserves checkpoint
chunk lengths `[1,2,2]` and `vae_config.encode_mode=performance`. The encoding
policy support is committed in `90971200f5`; the actual run overlays its two
files on frozen source `31f935812a92`, using archive SHA-256
`8f8a795cca1ca300857d5693f4588d007b2606960425c2f29104e95a828edf8f`.

The run retains the previous 10-second input, preservation prompt, seed 42,
four-step schedule, TAEH3 fp16 stable decoding, window 21 / sink 3 and
block-relative RoPE / maximum gap 12. Compared with step3500, checkpoint
chunk geometry and encoder policy change along with the weights; this is not
a weights-only ablation. There was no warmup matrix, repeated measurement or
profiling, and no performance conclusion is drawn.

All 889 image-local CPU regressions passed. Submitted, accepted, consumed and
received frame counts are all 240, with normal EOS. The input RGB hash matches
the previous run; output RGB SHA-256 is
`936d083258e2e1c5279a3e13a628e61f507d7aed27aa94fe05026154b6364296`.
The source, old SP2 and new SP2 MP4s pass complete decoding and 832x480 / 240
frames / 24 FPS checks. All three played to 10 seconds in the local comparison
page. Sampled frames still show the source cutting to a vase/rose while both
outputs retain the desert person. This remains a source-following failure;
full subjective quality judgment is left to review of the delivered videos.

Run evidence is under OSS/CPFS run prefix
`zing131-20260921/ode-sp2-quality-r1`. User-facing videos and the synchronized
comparison page are in local Downloads directory
`zing131-ode-step1000-sp2-20260921`. The GPU job has ended and SP8 remains deferred.


## Whitebox acceptance revision (2026-09-22)

The acceptance case is `zing-whitebox-video-five-triview-0001`: the original
prompt and all five reference sheets, with 120 real source frames. Streaming
historically supported five references under the unchanged total byte budget.
That run retained the first RGB frame as a prompt-only video condition; it was not a sixth VAE
reference. Subsequent prompt updates reuse this first frame and the references.
The source is resized to the negotiated 832x480 RGB canvas before transport;
thus the prompt first frame is also 832x480, unlike the original file request's
1280x720 prompt frame. This is an explicitly recorded streaming variant, not
an identical offline/native request comparison. No future source frame is used
in initial prompt conditioning. Existing results above predate this correction.


### Current source-path acceptance

Frozen runtime `3212f3e59b` ran in SP2 job `dlchhvdaiix6jow5` using the
same ODE step1000 export and image digest as above. Both measured devices were
NVIDIA L20D, CC 10.3, 274113 MiB; Torch 2.13.0+cu130. All 954 image-local CPU
regressions passed (47.88 s). The preceding zero-GPU asset/test job
`dlcvx8umicltfjgt` Succeeded / exit 0, with 954 tests and all six original media
hashes verified. Source video SHA256:
`6579ebd3f1d80455b71e30f9795aa494bceb615ab1302a812430f1199ec80f85`.

| Case | Submitted / accepted / consumed / delivered | Completion |
| --- | --- | --- |
| Full whitebox case | 120 / 120 / 120 / 120 | passed |
| Tail prefixes | 1, 2, 16, 17, 18, 34; all four counts equal each length | passed |
| Slow consumer, 2 s delay per output batch | 51 / 51 / 51 / 51 | passed |

Every case uses all five references and the original prompt. The client verifies
ordered output ranges, source alignment, timestamps, accepted/consumed boundaries
and an explicit `session.completed`. Every case has server cleanup completion.
All 16 rank/session EOS records clear KV, static condition storage, codec pixels,
bridge/decoder latents, metadata and pending blocks. This short matrix is not a
new long-run memory plateau or performance benchmark. The full case reached a
maximum of 60 submitted-but-unconsumed frames, below its 68-frame budget.

Full-case input RGB SHA256: `96dcb9c925fcc04f214d975277086a5d8444ae0dfcb31f138f9f8061b8098c6c`.
Output RGB SHA256: `8258ad197b07f9959e36adeca2a07a9ea63285a0a023b1472fdf7848e9986b03`.
The generated MP4 fully decodes as 120 frames, 832x480, 24 FPS / 5 seconds.
Sampled frames show changing character poses, but background geometry drift and
ghosting remain; quality is pending under #153, with no independent native baseline.

The SP2 job as a whole **Failed / exit 1**, despite the eight passing matrix
cases: the lifecycle probe completed invalid-sequence and disconnect initiation,
then did not catch an admission close occurring before its init send. This is
not evidence of passing disconnect cleanup. `5090743fcf` fixes only that probe:
retry exact close code 1008 / reason USER_SESSION_LIMIT under the existing
30-second deadline, log every attempt, and propagate unrelated failures. Two
local real-WebSocket regressions passed. A separate lifecycle-only rerun is
recorded below; successful video cases are not regenerated.

Artifacts: `world-model/sglang/evals/zing-h3-dmd/zing140-20260922/`, revisions
`cpu-r1`, `sp2-r1`, and `sp2-r2`. No SP8 job was submitted.


### Lifecycle-only recovery evidence

Job `dlc15n1nuycwuqnl`, frozen probe `5090743fcf` and unchanged `3212f3e59b`
model-serving code, passed six cases: invalid sequence, in-flight disconnect,
fresh generation after disconnect, EOS/disconnect race, fresh generation after
that race, and same-user concurrent admission rejection. Server logs confirm
cleanup completion for all six. In-flight and EOS-race cancellation released
scheduler session state after 3.871 / 3.901 seconds respectively; new sessions
then generated all 17 requested frames. The 74 transient admission rejections
are retained among 81 negotiation attempts; the longest connect recovery was
3.902 seconds, within the 30-second probe deadline. This is fresh-session
recovery, not resumption or independent-user isolation.

The job as a whole still **Failed / exit 1** in the later pause/prompt case:
the probe rounded frame-34 PTS to 1416667 us instead of the protocol's floored
1416666 us. The server correctly rejected that input and cleaned up its session.
`cac0fa21e3` fixes probe timestamps and adds selectable cases; all three local
real-WebSocket/server-ledger contract tests pass. Serving code is unchanged.
The remaining pause/prompt and reference cases are isolated in `sp2-r3` below.


The first targeted run `dlc1993eoanp46ty` (`sp2-r3`) also **Failed / exit 1**:
the pause probe sent its first 17 frames and waited through cold compilation
without an application heartbeat, so the unchanged 60-second idle watchdog
closed the connection. No remaining case passed in that run. `d3577b2e9f`
adds the existing protocol heartbeat to the probe, matching the production
source client's keepalive behavior; it does not relax server idle/lifetime
limits. Four local probe regression tests pass. Only the two pending cases
are retried in `sp2-r4`; all failed attempts remain in the evidence bundle.


### Final targeted acceptance and report

`dlcyzmn7u2f53eub` (`sp2-r4`, probe `d3577b2e9f`, unchanged serving
runtime `3212f3e59b`) **Succeeded / exit 0**. Four probe regression tests
passed in the job. Pause/prompt/resume delivered all 51 frames in order, with
the first output preceding prompt event 2 and later outputs carrying that event.
The reference case delivered all 17 frames. Both sessions emitted explicit
completion and server cleanup completion; all four rank/EOS records cleared
the session-owned state fields listed above. The cold pause case remained
connected through the existing heartbeat protocol; no timeout was relaxed.

Together, the eight video matrix cases and eight lifecycle cases now have
passing evidence. Three earlier Failed jobs are retained and explained above;
their status is not rewritten as success. All GPU jobs from this acceptance
sequence are terminal. No SP8 was submitted. This does not establish native
quality parity, 24 FPS inference, independent-user isolation, or a long-run
memory plateau for the revised five-reference configuration.

[Playable source/output comparison and evidence bundle](https://review-cdn.loopit.com.cn/reports/zing140-whitebox/20260922-r1/index.html)
requires the office network or company VPN and is hosted independently of any
local server. The report includes original asset hashes, source manifests,
per-case results, failed-run logs, job receipts and reproduction scripts.
Performance remains under #131 / #134 / #142 / #144 / #145; extended lifecycle
under #114 / #116; visual quality under #153.
