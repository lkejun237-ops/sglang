# Bounded high-resolution RGB source transport

The direct `/v1/realtime_video/generate` endpoint supports a configurable,
finite MessagePack budget. Deployment, source admission and the accompanying
`continuous_source_client.py` share `runtime/realtime/source_transport.py`.
This contract is video-only; source audio is not transmitted or generated.

## Deployment and client configuration

Add these arguments to the existing Zing server command:

```bash
--realtime-max-message-bytes 67108864 \
--realtime-transport-host-bytes 8589934592 \
--realtime-max-sessions 1 \
--realtime-max-sessions-per-worker 1
```

The defaults are 64 MiB per complete message and 8 GiB for transport buffers.
Neither accepts zero, negative values, or an unlimited setting. The host budget
must cover the admitted session limit. Zing deployments enforce one active
session at both API and GPU worker admission; higher session settings are
normalized to one. It is a conservative admission
reservation for transport-owned buffers, **not a cap on process RSS**: model,
codec tensors, KV cache, interpreter, socket/OS buffers and other endpoints need
additional deployment memory. Each serving process needs its own reservation;
multiply by process count when sizing a host/container.

Per session the reservation is `14 * max_message_bytes + 9 MiB`: four pending
source batches, one queued incoming message, two receive assembly/prefetch buffers, current encoded/decoded input,
two dispatch/IPC copies, three output staging/encoding/send buffers, 8 MiB
reference payloads and 1 MiB metadata allowance. At defaults this is 905 MiB,
for the single admitted Zing session. Increasing the message
limit without sufficient host budget fails startup. GPU/KV capacity must still
be checked independently.

The Uvicorn `websockets` backend enforces `ws_max_queue=1`, with compression
disabled. Do not substitute `websockets-sansio`: the version used here does not
implement Uvicorn's message-count queue limit. Dependencies retain the existing
supported version ranges (`uvicorn>=0.35.0,<0.40`, `websockets>=12.0,<16`),
checked before GPU worker startup. Images with newer packages must install the
repository dependencies: unsupported websockets 17.1 can assert when a pong
overlaps a backpressured video write. This backend uses the upstream legacy protocol API;
a future backend upgrade must repeat the real queue/backpressure regression.

Run the client with the **same** complete-message limit:

```bash
PYTHONPATH=python SGLANG_LIGHTWEIGHT_RUNTIME=1 \
python benchmark/zing_h3_causal/continuous_source_client.py \
  --url ws://HOST:30000/v1/realtime_video/generate \
  --source-rgb /path/to/source-1280x736.rgb --output /path/to/new-result \
  --prompt 'Preserve the characters and motion' --width 1280 --height 736 \
  --max-message-bytes 67108864 --batch-pattern 1 3 17 2 5 \
  --reference /path/to/reference.png
```

Repeat `--reference` for each reference (up to nine, 8 MiB combined). RGB inputs
must already have the stated dimensions and FPS. The service never silently
rescales a source to make it pass admission. A six-reference case is validated
separately; one reference in this command is only a usage example.

The client sends `source_message_budget_bytes` in init. A mismatch is rejected
before reference decoding or temporary-file allocation. Older clients may omit
this optional field but must implement the negotiated limits themselves. The
paired client requires `session.ready.max_message_bytes` to equal its configured
limit and fails clearly against an older server. Roll out this client and server
together. `max_source_message_bytes` includes a 4,096-byte protocol allowance;
clients must check encoded bytes, not just RGB payload length. Source init is
limited to `8 MiB + 4096` complete encoded bytes. Each append is limited to both
its negotiated maximum and `len(RGB) + 4096`; unsupported payload fields are
rejected so small batches cannot retain large hidden metadata.

## Incremental video benchmark

`bench_streaming_source.py` runs the frozen `5s`, `30s-wasteland` and `30s-town`
cases against an already-running resident service. Use Python 3.11+, the client
dependencies, and FFmpeg on PATH (or pass `--ffmpeg /path/to/ffmpeg`). The media
root must contain the paths expected by the frozen case manifests; preflight
checks asset sizes and SHA-256 before measurement.

```bash
PYTHONPATH=python SGLANG_LIGHTWEIGHT_RUNTIME=1 \
python benchmark/zing_h3_causal/bench_streaming_source.py \
  --url ws://HOST:30000/v1/realtime_video/generate \
  --case 5s --media-root /path/to/media --output-dir /path/to/new-result \
  --warmup-runs 1 --measured-runs 3
```

Select `--case all` for all three cases. Requests are strictly serial. Each case
gets its own warmup, excluded from statistics. The next request waits for normal
server closure after session cleanup; any failure stops the run and preserves a
failed receipt. Model loading is outside this benchmark: keep the service alive
throughout the run.

FFmpeg decodes incrementally into a bounded pipe and scales to the case's target
dimensions. It does not first decode the complete video. Source is sent as
`source.append` batches within negotiated frame/byte credits; EOF sends
`source.end`. Source length is checked locally, not announced in init. The
canonical videos are 24 FPS; arbitrary inputs must also be 24 FPS because this
client does not resample FPS. The low-level client also accepts `--source-video`,
or `--source-rgb -` for a live RGB24 producer.

Default replay sends as quickly as credits allow. Add `--paced` to simulate a
24 FPS capture clock; credit stalls do not reset this clock. Paced and unpaced
runs are different workloads and should not be pooled. Paced replay is lossless
and backpressured: it may take longer than the video duration to send and is not
an unthrottled camera/ingress test. References retain the
canonical IDs and order, but continuous-source conditioning uses the first
accepted source frame, not the offline prompt-video preview. This is a separate
baseline from whole-video input; equal seeds do not establish identical output.

The result directory contains `config.json`, `requests.jsonl`, `chunks.jsonl`,
`summary.json`, a self-contained `results.json` bundle and an offline
`results.html` report. Per-request directories
retain the negotiated request, event trace, RGB output, hashes and decoder log.
Successful request rows retain the server session ID and input/output hashes
for correlation with stage logs; failed rows retain the underlying exception
traceback. `stall_seconds` and unavailable `ifl_seconds` are preserved per request.
Open `results.html` directly in a browser. The light report compares cases,
shows formal repetitions and warmups separately, and visualizes batch intervals
and source backpressure. Use **打开结果 JSON** to append multiple result files;
select a file to inspect it, or export the selected bundle. All reading happens
locally, without a server or network request. Runs from separate files are not
pooled into one performance statistic.

To rebuild a page from existing streaming results:

```bash
python benchmark/zing_h3_causal/build_streaming_report.py \
  --results RUN/results.json --output RUN/results.html
```

For historical raw CUDA archives, add `--server-results GPU_RESULT_DIRECTORY`.
That archive contains `timing/`, `hardware.json` and the per-request client
summaries. New benchmarks already include received Host profiles in their
results JSON; do not point this option at a live temporary timing directory. The report checks collector health and per-session/rank latent/frame
coverage, then shows component Host totals and a single-request Host
timeline. Timeline alignment requires the client clock domain to match the
server, or recorded evidence that both ran in the same worker. Newly collected
client summaries include the local boot-based clock domain. Host totals remain
separate from the aligned Host timeline. Exporting
the current JSON from the page retains the server evidence as well.

Finite-input and continuous-source reports now share `benchmark_report.html`,
the file selector and the `zing-benchmark-results` version 1 envelope. Streaming
report data has `data.kind="continuous_source"` and retains its measurement
bundle under `data.streaming`; the existing finite report representation is
unchanged. The common page opens both, including old #275/#283 JSON reports.
Raw `bench_streaming_source.py` result bundles are also accepted as an input
format. To save the shared report envelope directly, use `--json-output FILE`.
Scene names come from case metadata; request selectors include scene, input
mode, measured repetition and session ID. The selected request also shows its
full client request ID, server session ID, source and prompt.
Warmups and failures are excluded from timing distributions.

All intervals use the client's monotonic clock:

| Field | Meaning |
| --- | --- |
| `ttff_seconds` | Client start, before connection and decoder startup, to first complete RGB batch |
| `e2e_seconds` / `fps` | Client start to last RGB batch / delivered frames divided by that duration |
| `tpoc_seconds` / `icl_seconds` | Mean / distribution of intervals between complete delivered RGB batches |
| `source_to_first_output_seconds` | First source send start to first complete output batch |
| `dependency_to_output_seconds` | Send start of the batch containing each output's last source dependency to output receipt |
| `source_send_span_seconds` (result bundle) | First source send start to last source send start |
| `tail_drain_seconds` | Last source send start to last output receipt |
| `credit_wait_seconds` | Total time waiting for source credit after a batch is ready |
| `source_to_output_seconds` | Per-frame simulated capture-to-batch-delivery in paced mode; decoded batch availability-to-delivery in unpaced mode |

`stall_seconds` estimates accumulated lateness against a 24 FPS playback clock
with a 0.5-second startup buffer; it is not observed browser playback.
`IFL` stays unmeasured: a batch arrival supplies no individual frame presentation
timestamps. `first_output_before_source_end` records whether output arrived
before sending EOS; it is useful evidence, not a universal pass condition for
very short inputs. These metrics include client decoding, transport and output
handling; dependency latency is not isolated Encoder RPC time or CUDA time.

CPU tests in `tests/test_bench_client.py` exercise real FFmpeg decoding and a local
WebSocket protocol peer, including early output, credit backpressure, EOF,
frame-count failure, disconnect cleanup, warmup exclusion and frozen case
preflight. They do not run the model or establish TTFF improvement, visual
quality, or GPU Encoder/DiT overlap. Those require a real resident service run.

## Dimensions and credits

Admission uses the loaded model's VAE spatial compression and DiT patch size
independently for width and height. For H3 these multiply to 32 on both axes.

| Requested dimensions | Grid status | 17-frame append budget (bytes) |
| --- | --- | ---: |
| 832 × 480 | aligned | 20,371,456 |
| 896 × 512 | aligned | 23,400,448 |
| 1280 × 736 | aligned | 48,050,176 |
| **1280 × 720** | **rejected; height is not divisible by 32** | not admitted |

Grid/admission pass does not establish GPU capacity, successful generation or
visual quality. 1280 × 736 results are not evidence for exact 1280 × 720.

`session.ready` reports `max_message_bytes`, `max_source_message_bytes`,
`max_source_batch_frames`, `max_pending_source_frames` and
`max_pending_source_bytes`. Source credit remains four clips (68 frames for
clip 17), bounded by actual RGB bytes. A batch also occupies a bounded ledger
record until both ownership transfer and output delivery complete. Waiting for
`source.consumed` is required before exhausting frame or byte credit; clients
must not confuse `source.accepted` with released credit. Slow output continues
to backpressure source input.

## Direct endpoint errors

Errors contain English `content` and the following stable ASCII `code`:

| Code | Meaning |
| --- | --- |
| `source_dimensions_unaligned` | Request is outside the model's spatial grid |
| `source_transport_mismatch` | Client and server message budgets differ |
| `source_message_too_large` | Dimension admission or encoded append exceeds budget |
| `source_init_too_large` | Initialization including references exceeds its budget |
| `source_credit_exceeded` | Pending frame/byte or metadata credit exhausted |
| `invalid_source_event` | Other malformed source event |

A message larger than the WebSocket transport limit is closed with standard
code **1009 before MessagePack parsing**; an application error object cannot be
promised in that case. Invalid initialization can be corrected on the same
connection; invalid source events terminate generation through existing cleanup.
Only curated `SourceTransportError` content is exposed, never arbitrary exception
text. The world-platform `authorized_generate` error contract and AsyncAPI are
outside this direct-source change and are unchanged.

## Validation

CPU boundary and real-socket queue tests:

```bash
SGLANG_LIGHTWEIGHT_RUNTIME=1 PYTHONPATH=python \
python -m pytest -q benchmark/zing_h3_causal/test_source_transport.py
```

Adapter regressions also cover pre-allocation rejection and all three aligned
sizes. GPU evidence, source/material SHA-256, repeats, audio status, actual
frames/dimensions and per-rank memory must be reported with the exact frozen
source. Transport acceptance alone does not establish KV capacity or quality.

The [2026-09-28 high-resolution acceptance report](../../../../../benchmark/zing_h3_causal/SOURCE_TRANSPORT_ACCEPTANCE_20260928.md)
records three real-source, six-reference repeats at each aligned resolution,
including timing, per-rank memory and frozen-source/media checksums.

### First-stage profiling and result retention

The first-stage report uses `host_call_ms` and Host timestamps. No timing
environment variables are required. Ordinary generation does not collect
stage data; `return_stage_timing=true` enables Host collection and delivery
for that request, including the dedicated Encoder. CUDA events are not
created on this path. The existing `profile=true` still means Torch profiler.

The serving launcher creates one temporary `zing-stage-timing-*` directory
for each deployment. SP workers inherit it, and the Source Encoder receives
the same absolute path through its internal open-session RPC.
`SGLANG_ZING_STAGE_TIMING_DIR` is an optional parent-directory override;
the launcher creates and owns a unique child, preserving other files in the
parent. The old `SGLANG_ZING_STAGE_TIMING` mode variable is no longer needed.

Both clients request results with `return_stage_timing=true`
(`--receive-stage-timing`). Both finite and continuous-source benchmarks request this by
default; `--no-stage-timing` skips it. After generation and session cleanup,
the server sends `profile_result`, including dedicated Encoder records for
continuous source. The client saves the result, sends one
`{type: "profile_ack", session_id: ...}`, and waits for normal close. This
wait is outside TTFF/E2E. No acknowledgement retries or transaction log.

Worker timing rows are stored in separate files per session. An ACK deletes
only that session's files. A disconnect, malformed ACK, or a five-second ACK
timeout also removes the delivered session's expendable timing files.
Aborted sessions are cleaned after worker release; deployment shutdown
removes the automatically created directory, including health snapshots.
Per-worker health snapshots are overwritten in place rather than appended.
Client profiles (`profiles/<request_id>.json` for finite input,
`<case>/<phase>-<index>/profile.json` for continuous source) and exported
result JSON remain available for reopening the report. The continuous
benchmark assembles its server view from these client files; it no longer
requires recovery of the server timing directory. The split baseline checks client-owned profiles after server cleanup.
Historical raw worker files remain readable by offline analysis tools.
