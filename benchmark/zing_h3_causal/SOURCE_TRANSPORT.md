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
--realtime-max-sessions 8 \
--realtime-max-sessions-per-worker 8
```

The defaults are 64 MiB per complete message and 8 GiB for transport buffers.
Neither accepts zero, negative values, or an unlimited setting. The host budget
must cover the larger configured session limit. It is a conservative admission
reservation for transport-owned buffers, **not a cap on process RSS**: model,
codec tensors, KV cache, interpreter, socket/OS buffers and other endpoints need
additional deployment memory. Each serving process needs its own reservation;
multiply by process count when sizing a host/container.

Per session the reservation is `14 * max_message_bytes + 9 MiB`: four pending
source batches, one queued incoming message, two receive assembly/prefetch buffers, current encoded/decoded input,
two dispatch/IPC copies, three output staging/encoding/send buffers, 8 MiB
reference payloads and 1 MiB metadata allowance. At defaults this is 905 MiB,
or 7,240 MiB for eight admitted sessions. Increasing the message or session
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

The [2026-09-28 high-resolution acceptance report](SOURCE_TRANSPORT_ACCEPTANCE_20260928.md)
records three real-source, six-reference repeats at each aligned resolution,
including timing, per-rank memory and frozen-source/media checksums.
