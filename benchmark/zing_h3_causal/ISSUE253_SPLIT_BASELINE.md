# Issue #253 independent Encoder baseline

This checkpoint makes the existing same-node Encoder service measurable for
Issue #253. It reuses the continuous-source transport and Encoder overlap
implementation described in [ENCODER_OVERLAP.md](ENCODER_OVERLAP.md):

- one dedicated source VAE Encoder GPU;
- an SP2 DiT group whose two GPUs also host the TAEH3 Decoder;
- concurrency one, sequential requests, resident processes within each cell;
- the frozen 5s coast case and the two frozen 30s cases from PR #261;
- paired `serial` and `overlap` source Encoder modes.

The paired split runner replays source through `source.append` as quickly as
credit permits. Its preparation exports the complete source to RGB24 at the case's output geometry;
it does not loop, crop, shorten, or replace the video. The client sends the text
and ordered reference images. Initial conditioning uses the first accepted source
frame as `<Video 1>`, not the entire original prompt video. The deployment profile
must select progressive encoding, indexed source noise and a bounded attention
window. Consequently this is a **continuous-source baseline using the same case
assets**, not a numerically equivalent replay of the finite-video request.

Only the source VAE is placed on the dedicated Encoder GPU; this entrypoint does
not introduce a separate Qwen or reference-image Encoder deployment. `overlap`
uses the existing prefetch within submitted work; cross-dispatch prefetch and
asynchronous source output remain off in the prepared baseline. This stage adds
measurement, not a new TTFF optimization or proof of GPU compute overlap.

The 5s case is the gate: a request, media identity, output completeness,
or per-session stage-collection failure stops the job before either 30s case
runs. Serial/overlap hashes are retained as strong diagnostic evidence. A hash
mismatch preserves both outputs for numeric diagnostics and decoded-video
review; it does not alone decide business quality.

Each layout/case cell starts resident model processes, runs one warmup followed
by three formal requests, and then stops those owned processes. Formal runs do
not enable Torch profiler. TTFF and E2E start at the client's `started` timestamp,
before connection and init preparation, and end at first/last complete RGB batch
receipt. FPS is delivered frames divided by E2E. TPOC is the mean gap between
output batches; ICL includes gap distribution statistics. Raw output writes can
affect subsequent receives. These are client delivery metrics, not browser
presentation metrics, and their start boundary differs from the finite client's
pre-send boundary. IFL remains unavailable because the protocol does not expose
individual frame delivery timestamps.

Remote source Encoder spans use the client session ID and record source
dependency plus latent ranges. DiT and Decoder spans use scheduler dispatch IDs.
Every completed request emits a final per-rank Host completion record. The
client receives and saves the Host profile, then acknowledges it so server
files can be removed. The harness checks latent/frame coverage and collector
health from client-owned profiles before starting the next run. Serving
creates one deployment temporary directory automatically and propagates it
to SP workers and the dedicated Encoder. No timing environment variables are
required; ordinary requests are timed only with `return_stage_timing=true`.
All processes on the node record CLOCK_MONOTONIC and boot identity. Host
intervals are CPU call boundaries and are not additive E2E.
Encoder worker spans cover source VAE
encoding and streaming cache updates. They do not yet measure consumer-observed
Encoder RPC latency or DiT's blocking wait for the next latent.

## Run

Use a deployment manifest containing `realtime_profile` and `taeh3_sha256`, as
described in [ENCODER_OVERLAP.md](ENCODER_OVERLAP.md#prepare-inputs-without-a-gpu).
The profile binds the checkpoint's `zing_config.json` hash and must select a
four-step progressive streaming deployment. Model and TAEH3 weights must already
be available; preparation does not convert checkpoints.

Prepare each case without GPUs:

```bash
python benchmark/zing_h3_causal/prepare_issue253_split_case.py \
  --deployment-manifest deployment.json \
  --case-manifest benchmark/zing_h3_causal/whitebox_video_smoke_cases.jsonl \
  --case zing-whitebox-video-five-triview-0001 \
  --model MODEL --media-root MEDIA --taeh3-weights TAEH3 \
  --output PREPARED/5s
```

Repeat preparation with `whitebox_video_long_cases.jsonl`, selecting
`zing-whitebox-video-30s-six-triview-0001` for `PREPARED/30s-wasteland` and
`zing-whitebox-video-30s-six-triview-0002` for `PREPARED/30s-town`. These two cases
have seven references; the 5s coast case has six. Run them in the fixed order on
one node with at least three visible GPUs (DiT/Decoder: 0,1; Encoder: 2):

```bash
python benchmark/zing_h3_causal/run_issue253_split_baseline.py \
  --case PREPARED/5s \
  --case PREPARED/30s-wasteland \
  --case PREPARED/30s-town \
  --output RESULTS
```

Each case contains `enc1-sp2-{serial,overlap}-unpaced/` directories with
`warmup/`, `measured-0/` through `measured-2/`, client summaries/events/raw RGB,
role logs and `timing/`. `CASE_COMPLETE.json` records both layouts;
`FIVE_SECOND_GATE.json` permits the long cases, and `EXPERIMENT_COMPLETE.json`
records runner completion. These receipts check delivery and timing contracts;
they do not certify visual quality. A transient shared-filesystem `EIO` or
`ESTALE` read receives bounded retry; other I/O failures are propagated.

## Review and validation

Read the preparation and baseline runner first, then `endpoint_cell` and
`session_timing_ready` in `run_encoder_overlap_matrix.py`. Runtime changes carry
the client session identity into `source_encoder`/`remote_encoder`/
`encoder_worker`, and attach asynchronous spans in `source_session` and
`zing_stage_timing`.

The benchmark tests cover frozen reference order, repetition policy, transient
read handling and timing completeness by session/rank/range. Runtime tests cover
identity propagation and DiT/Decoder/completion spans, including the existing
joint-drain path. Changed test files are registered in main's unified Zing CPU
manifest. They do not replace GPU transport, media or performance validation.

The saved 2026-10-12 r7 cloud receipt (`dlc1eojiff7blb66`) reports runner success
for all three cases and both layouts on E1+SP2, with one warmup and three formal
requests per cell. Its full result directory has not yet been recovered and
independently reviewed locally. No serial/overlap speedup or visual acceptance
is claimed here. The frozen r7 source predates the subsequent multi-rank timing
validator correction and formatting cleanup. Recover results before publishing
performance comparisons; retain numerical differences for diagnostics and human
video review.

## Incremental source client and measured delivery

For an already-running service, `bench_streaming_source.py` decodes video through
a bounded FFmpeg pipe instead of requiring a full RGB export. It supports the
same three cases, lossless paced/unpaced input and JSON export. See
[SOURCE_TRANSPORT.md](SOURCE_TRANSPORT.md#incremental-video-benchmark) for commands
and metric definitions. This entrypoint checks client delivery and waits for
normal server closure, but does not own the server processes or inspect timing
files. The paired runner above additionally waits for stage timing completeness.

The 2026-10-13 Alibaba Cloud run `dlc1lf4b9nyd3b8d` completed three unpaced cases
and a paced 5s case on one resident E1+SP2 overlap deployment, concurrency one.
Four GPUs were allocated, with GPU 2 for source Encoder, GPUs 0/1 for DiT/Decoder,
and GPU 3 idle. Source was incrementally decoded at 832×480, 24 FPS, with batches
of one frame and seed 123. Each group had one warmup and three measured requests.
Asynchronous CUDA stage timing was enabled; Torch profiler was not enabled.

| Input | Median TTFF (s) | Median E2E (s) | Median delivered FPS |
| --- | ---: | ---: | ---: |
| 5s coast, unpaced | 4.34 | 15.17 | 7.91 |
| 30s wasteland, unpaced | 4.64 | 86.85 | 8.29 |
| 30s town, unpaced | 4.65 | 86.38 | 8.34 |
| 5s coast, paced | 4.61 | 15.44 | 7.77 |

The full worker stdout contains all 16 client summaries. The 12 measured requests
closed their submitted/accepted/consumed/received ledgers at 120 or 720 frames.
Every request delivered output before sending EOS. Each case's measured output
hash was repeatable; the paced and unpaced 5s hashes also matched. The first
warmup's 133-second TTFF is excluded from the table.

Delivery is bursty: per-run batch-interval P95 is approximately 1.7–2.1 seconds.
The paced 5s source actually took about 8 seconds to send because of backpressure.
These results establish lossless incremental input, not a service that can keep
up with an unthrottled 24 FPS camera. Same-case hash equality is a diagnostic,
not visual quality acceptance. There is no matched serial or finite-input A/B
in this run, so the table is not an overlap speedup claim.

The measured source was `b75447e3a448502f6b60566e67b41223876bf142` plus the
frozen streaming-client overlay; its freeze manifest SHA256 is
`f5b30327b38e985c8043e7fdb72fb9f065c80f18c3c3c9fd1be7132f05d08c9b`.
Results remain under
`/cpfs/world-model/sglang/evals/issue253-stream-source-20261013-r1/gpu`.
The table was checked against full stdout and client summary arithmetic.
Per-rank timing, health and the original client records were recovered and
hash-checked on 2026-10-14. The 12 formal sessions have complete dedicated Encoder
latent ranges and both SP ranks' DiT/Decoder ranges and completion records.
SP rank 0's cumulative drop count is 97; matching SP span identities locates
all 97 missing records in the wasteland warmup. That warmup is excluded from
formal aggregates. Thus collection is not globally lossless, while the formal
stage coverage is complete. The raw RGB files were not recovered in this
metadata-only export; visual acceptance remains pending.

| Input | Source Encoder CUDA sum (s) | DiT rank 0 / 1 (s) | Decoder rank 0 / 1 (s) |
| --- | ---: | ---: | ---: |
| 5s coast, unpaced | 3.683 | 10.320 / 10.319 | 0.042 / 0.048 |
| 30s wasteland, unpaced | 22.018 | 66.793 / 66.733 | 0.272 / 0.267 |
| 30s town, unpaced | 22.020 | 66.112 / 66.103 | 0.269 / 0.266 |
| 5s coast, paced | 3.685 | 10.439 / 10.410 | 0.049 / 0.044 |

Each cell is the median of three per-request sums, not a sum across ranks.
These CUDA stream intervals overlap and do not form an additive E2E breakdown.
The offline streaming report presents these totals alongside separate Host
measurements and a Host timeline aligned to the client start on the same node. Later main integration and JSON metadata
changes have CPU coverage but have not been rerun on GPU.

Remaining work includes report-viewer integration, consumer-side Encoder service
latency and blocking waits, and measured TTFF/Condition optimization. This PR
does not close Issue #253.
