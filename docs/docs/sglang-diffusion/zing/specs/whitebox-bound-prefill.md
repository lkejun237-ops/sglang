# Bound Prefill inputs and per-Prompt cache receipts

Local implementation, 2026-10-30. Not deployed or GPU-benchmarked by this change.

## Input binding

A source advertises `capabilities.bound_prefill_protocol=whitebox-bound-prefill-v1`.
After reference uploads, before any model segment is active, the source owner sends
`render.input.prepare` through the existing authenticated control connection with
`request_id`, source identity, `expected_render_revision`, prompt, seed, ordered
reference descriptors and `region_id`. Room resolves references and pins one actual
RGB24 frame in the existing 8 MiB pinned-start allowance. No GPU lease is taken.

The ephemeral `render.input.prepared` response includes `input_id`,
`input_sha256`, source identity, `world_id`, `source_frame_index`, `source_pts_us`,
`source_frame` bytes, `source_frame_sha256`, `model_init` and ordered reference
SHA256 descriptors. `ttl_ms` is at most 180000. Pixels are never stored in the
Redis control journal or reconnect history. Retrying the identical request ID
returns the same snapshot while it is valid; it never captures a replacement.
Only one unconsumed snapshot is allowed. `render.input.release` with `input_id`,
expiration, consumption, or source close releases it. AREA sources are not covered.

Use the snapshot's exact prompt, seed, source geometry/FPS, RGB and original
ordered reference bytes with the existing Prefill API. The bound descriptor is
not a caller-chosen cache key. The native worker still computes its own key.

After `memory_ready`, computation and the model lease have been released by the
Gateway. The client may start a segment without waiting for `persisted`:

```json
{
  "op": "render.start",
  "start_boundary": {"mode": "next_available"},
  "prepared_input": {
    "protocol": "whitebox-bound-prefill-v1",
    "input_id": "original-prepare-request-id",
    "input_sha256": "64-lowercase-hex-digits"
  }
}
```

The normal source identity/revision, prompt, seed, references and region are also
required. Room compares them with its frozen inputs. It uses native S0 v2
`source.prepare` with the exact cached condition RGB, then `source.attach` with a
fresh video start. The exported reply declares
`start_policy=condition-preview-then-live`. The frozen snapshot is the **condition
preview**, not an old frame inserted into the live video. This prevents the warmup
interval from becoming a large capture/RTP timestamp gap. Video timestamps remain
unmodified. Any condition mismatch or expired snapshot is rejected;
`next_available` selects only fresh live video, never a replacement S0 condition.
A successfully journaled start remains idempotent after its snapshot is consumed.

In the single-Slot deployment, warmup computation and live model inference are
serial. Disk persistence may overlap playback. Snapshot creation requires no live
model segment. A failed or unknown Prefill is queried with the original request ID
before any new input is submitted. This protocol does not promise L1 across worker
restarts or different instances, nor hits for a different S0 or dynamic prompt.

## Cache telemetry

Native source output emits `model.cache` on the first output for each activated
Prompt version. Its `usage.zing_block0_cache` contains only bounded public fields:
`cache_key`, `prompt_version`, `cache_result`, `miss_reason`, and host timings.
The model internal usage retains `key` as a compatibility alias. A promotion from
L2 during preparation remains classified as L2 even if activation finds that entry
in L1. A miss at activation overrides an earlier preparation hit.

Room validates model session/generation and attaches source/segment identities.
The existing reliable control channel carries this receipt to the browser; up to
64 version receipts are retained in the segment's archive manifest. No arbitrary
model usage, prompt text, reference bytes or private identity metadata is forwarded.

Clients accept `cache_key` and `key`, index evidence by segment + Prompt version,
and clear the displayed hit to `unreported` when a newer Prompt is accepted. Late
older-version receipts cannot replace newer evidence. Unversioned legacy receipts
can describe opening v1 only. Missing telemetry is never a cache miss or hit.

## Validation and rollout boundary

CPU tests cover input equality, changed references/prompt/seed, ring eviction,
expiration, duplicate requests, cancellation, original-ID recovery, early memory
readiness, generation fencing and version-scoped receipts. Frontend build and
isolated tests do not establish production L1 hit rates or latency improvement.
Both 480 and 736 GPU end-to-end/per-component latency and memory remain unmeasured.

Deploy the matching model/API and Room code before enabling this frontend flow.
Older clients ignore the added telemetry; the new client retains normal playback
when the source does not advertise snapshot support. Model-role updates require a
coordinated full-role restart, not a single DiT restart. Source code participates in
the execution fingerprint, so prefill again after the model update; old cache files
and all existing cache locks must remain unchanged. Room snapshot support itself
does not change GPU configuration, storage roots, bitrate, or recording policy.
