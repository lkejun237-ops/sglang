# Sink controls on the existing player connection

For client integration, see the [Chinese Sink API guide](/docs/sglang-diffusion/zing_h3/specs/whitebox-sink-api.zh-CN).

Implementation candidate, not a deployment or GPU acceptance report. Whitebox
sessions start with Sink **off**. The model export's
`source_attention_window.sink_chunks` and pipeline's
`zing_kv_pool_sink_chunks` must both be `0`; window sizes must match and be at
least four. `deploy/whitebox` checks these prepared assets before starting any
role. Model weights are unchanged. Prepare a new serving revision; do not edit
running containers or shared exports. Generic Zing profiles retain their own
explicit defaults, including historical S3 profiles which cannot use this API.

The supported model is video-only unsplit `[3,2]`, `split_first_latent=false`.
Runtime on/refresh pins up to two recent completed chunks (five latents when
full). The S0 pool must cover the on/off peak; no new allocation is requested.

## Existing connection and command

Use the original authenticated player control WebSocket and binary MessagePack,
alongside existing input and optional Prefill messages. No Sink socket, HTTP
endpoint, new Slot, or UI change is required. The original Room routes the
command to this player's current model connection. This does not merge different
players' identities or remove internal SDK/Worker transports.

Wait for the current segment to be streaming and for its `sink.supported=true`
status. This capability comes from the actual model's `ready.sink_updates`.
Absent/false means unavailable, including on old workers.

```json
{
  "type": "command",
  "op": "model.sink",
  "request_id": "sink-on-001",
  "source_session_id": "SOURCE",
  "source_epoch": "SOURCE_EPOCH",
  "segment_id": "SEGMENT",
  "segment_epoch": "SEGMENT_EPOCH",
  "action": "on"
}
```

| action | Effect |
| --- | --- |
| `on` | If off, establish a new anchor from recent completed history. If already on, no-op. |
| `off` | Remove the fixed anchor and retain Recent history. If already off, no-op. |
| `refresh` | Select the latest up-to-two committed chunks as the anchor and turn on. |

Refresh does not promise the exact frame currently displayed by a remote client.
It does not recover evicted initial history or run Prefill. An incomplete initial
anchor fills as native chunks complete. Prompt/reference changes in the same
segment retain on/off state and may rebuild the anchor. Every new segment starts
off; callers explicitly turn it on again if desired. No player preference store
or automatic camera/scene trigger is introduced.

Use a new request ID for each intended operation. Identical retries replay the
original durable result; changed reuse returns `REQUEST_ID_CONFLICT`. Old
source/segment identities reject. Only one Sink operation is in flight per
segment. `SINK_UPDATE_BUSY` means a Sink or condition transition is pending;
`MODEL_UNAVAILABLE` means no active eligible segment (including Prefill-only).
`CAPABILITY_UNAVAILABLE` means this source/model does not support the operation.
These are nonfatal `operation.rejected` replies on the same connection.

## Receipts and recovery

The existing `operation.result` first reports `state=accepted, applied=false`.
Native acceptance does not prove the change has affected generation. On the
first matching RGB output it reports `state=completed, applied=true`, including:

```json
{
  "request_id": "sink-on-001",
  "state": "completed",
  "applied": true,
  "native_event_id": 42,
  "first_output_frame": 357,
  "sink": {
    "enabled": true,
    "sink_chunks": 2,
    "sink_chunk_ids": [38, 39],
    "sink_event_id": 42
  }
}
```

The example omits the existing operation/segment envelope fields. The frame
number proves model output, not browser presentation. No-op completes without a
fabricated first output frame. Query through existing `status.get`/
`operation.get`; current segment status includes `sink.enabled`, anchor IDs and
`pending_request_id`. Frame `rendering.sink` also preserves this snapshot through
normal WS/RTC metadata and recording.

Timeout/send uncertainty reports `reconciling/SINK_UPDATE_UNKNOWN`. Query the
original request; never submit a second refresh to compensate for a missing ACK.
Late output can still complete the original operation. A segment ending cancels
unsent operations and retains uncertainty for sent, unobserved ones. New segments
cannot consume old receipts. No automatic model restart or forced Slot release.

The original ingress queue and model wire sequence provide ordering. Compute
and output waits run asynchronously without holding the source send lock, so
input/heartbeat/exit can continue. Sink does not pause the shared World. Existing
Prefill/active-segment mutual exclusion is unchanged.

## SDK member input on the same connection

Ordinary whitebox input already uses this connection. SDK-admitted multiplayer
members retain the original SDK controller/binding/lease protocol: raw legacy
`kind=input` remains forbidden on their model channel.

An API caller can optionally tunnel the **existing SDK JSON control packets**
over its already authorized member model WebSocket:

```json
{"type":"sdk.control.open","client_id":"stable-client-id"}
```

After `{"type":"sdk.control","state":"ready", ...}`, the SDK's original
binding offers and other messages arrive as strings in `sdk.control.payload`.
Respond with the unchanged SDK JSON string:

```json
{"type":"sdk.control","payload":"{\"type\":\"SDK_MESSAGE_TYPE\",\"...\":\"original SDK fields\"}"}
```

This is an envelope example, not a valid SDK input packet. Follow the SDK's real
binding offer/ack and input schema, including its epoch, binding sequence and
input sequence. Do not send input before SDK authorization. `ready` means the
relay socket opened, not that an input lease has been granted. Use either this
tunnel or the separate SDK control socket for a controller, not both.

The server selects the original member's SDK endpoint and viewer capability;
callers cannot supply addresses, tokens or other member IDs. The SDK still
validates every original packet. Replies carry attachment generation, are bounded,
and are not replayed into a later attachment. Disconnect/replace/leave closes the
relay, letting the original SDK revoke its input lease. Reconnect reopens and
performs the original SDK handshake; old key events are never resent. Input
activity accounting uses the existing `observe_player_control` owner.

A relay `rejected` or `closed/CONTROL_UNAVAILABLE` event is distinct from a model
operation failure. The bounded input queue rejects overflow. Prefill and Sink
use their original MessagePack messages on this same connection; they are never
forwarded into the SDK JSON stream. Existing clients can keep their old transport.

## Validation boundary

CPU tests cover controller state/receipts, native KV/graph paths, same-source
input during control, duplicate/conflicting requests, timeout/late output,
retirement and relay identity/byte preservation. Real SDK/browser multiplayer,
CUDA/SP collectives, model quality and live FPS need separate authorized
acceptance on the final built revision. No production deployment is implied.
