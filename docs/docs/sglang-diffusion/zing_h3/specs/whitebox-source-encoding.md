# Session source encoding

You can opt a new native RTC whitebox source into an explicitly signed software
encoder configuration. Omit the selection to inherit the deployment's current
configuration. Omission does not select a historical bitrate or GOP, enable a
codec, change a native command line, or rewrite a shared service configuration.
This contract extends [render segments](whitebox-render-segments.md); it does
not change the model encoder, frozen World runtime or engine CLI version.

## Admission and authorization

The client selects the source encoding for each new native RTC session; Gateway
and Room do not require a separate environment feature gate. Omitting the override
preserves the current deployment configuration, including its original native
arguments, and does not run an additional codec probe.

WTP advertises the requestable debug schema in
`config.nativeRtc.sourceEncoding={enabled,capabilities}` without requiring a
capability environment variable. This declaration lists settings you may request;
it does not prove that a software encoder is installed, that a candidate passed
a runtime probe or that the browser can decode it. Room checks the exact selected
settings with a bounded real encode/decode probe before each new explicit source.
An unsupported candidate fails before source allocation. There is no public
arbitrary-parameter configuration endpoint or global encoder mutation endpoint.

`source_encoding.encoding_capabilities()` remains a schema and range template
with every `available:false`; it is not a probe result. For deployment validation,
you can independently run `preflight_source_encoding()` against the exact Python
codec libraries and native binary that will be deployed. Preserve candidate
settings, library identity, binary hash and probe results. A native binary's codec
list proves RTP support only. A successful CPU probe does not establish browser
acceptance, all settings in the schema ranges or network smoothness.

The three carriers have the same normalized value:

| Carrier | Optional field |
| --- | --- |
| WTP ticket HTTP request | `sourceEncoding` |
| Signed `whitebox-rtc` ticket claim | `source_encoding` |
| Initial `init.whitebox` | `source_encoding` |

All three omit the field for the deployment default. `null` is invalid. The
Gateway verifies presence and exact canonical contents against the signed claim
before consuming the ticket nonce. An unsigned value, omitted signed selection,
changed setting or noncanonical signed object fails authorization. Existing
world, source, user, Origin, expiry and one-use checks remain in force.

The formal single-player service uses the existing
[trusted launch reconciliation](whitebox-render-segments.md#trusted-single-player-launch-reconciliation)
instead of a browser RTC ticket. Its canonical `init.whitebox.source_encoding`
is inside `frozenLaunchParameters.initMsgpackBase64`; the service signature binds
the complete request bytes, including that frozen init. No separate encoding
field is added to the control operation or its attachment capability. WTP must
freeze the selected init before signing, keep `create.manual` with
`inference_control:manual`, and reuse that original operation for recovery.
The formal Player initially requests the client-owned VP9 preset and freezes its
explicit selection in the encoded init. The exact preset still requires server
validation and encoder preflight; selecting it does not establish browser media
or performance acceptance. Other callers that omit the whole override retain
the existing server defaults; Gateway and Room do not change their
deployment-wide encoder configuration.

Canonical H.264 has exactly seven fields: `codec`, `bitrate`, `gop`, `vbvMs`,
`preset`, `quality`, `intraRefresh`. VP9 and AV1 have exactly those fields plus
`minQp`, including an explicit zero. An entry normalizer may remove H.264
`minQp:0` or add omitted VP9/AV1 `minQp:0` before signing. Signed and init objects
must already be canonical; verification never repairs them. Unknown keys,
booleans in integer fields and fractional numbers fail. Cross-language examples
are frozen in
[`source-encoding-contract.json`](../../../../../python/sglang/multimodal_gen/test/unit/realtime/fixtures/source-encoding-contract.json).

Overrides apply only to a new `whitebox-render-v2` native RTC source at 832×480
or 1280×720 (`whitebox.resolution: "480p"` or `"720p"`), using the existing `create` or manual `create.manual` lifecycle. AREA, existing
Room/source joins and WebSocket media reject an override. Default calls retain
their existing resolution and transport support. Admission checks run before
source allocation or replacing an attachment.

## Settings and encoder ownership

| Setting | Contract |
| --- | --- |
| `bitrate` | Integer 100,000–50,000,000 bits/s; source ceiling, subject to existing feedback and transport limits |
| `gop` | Integer 1–24; source GOP only |
| `vbvMs` | Integer 50–5,000 ms |
| `preset` | H.264 x264 preset; VP9 string `0`–`8`; AV1 string `0`–`12` |
| `quality` | H.264 CRF 1–51; VP9 max QP 0–63; AV1 max QP 1–63 |
| `minQp` | VP9/AV1 integer 0–`quality`; zero retains the prior option set; nonzero H.264 is invalid |
| `intraRefresh` | Boolean, true allowed only for H.264 |

H.264 uses libx264 baseline/zerolatency. VP9 uses libvpx-vp9 realtime CBR,
lag zero, alternate reference off, row multithreading, no intentional frame
dropping, two software threads and maximum intra rate 300%. AV1 uses libsvtav1
CBR with `rtc=1`, `pred-struct=1`, lookahead zero and `lp=2` (SVT's parallelism
level, not a promise of two threads). `quality` bounds quantization in the CBR
codecs; it is not their former CRF setting. A nonzero `minQp` constrains the
minimum quantizer and may reduce refresh peaks at a visible quality cost.
Neither CBR nor a minimum QP guarantees exact bitrate, AU size or smooth playback.

Room copies the source settings into the session configuration. It does not
mutate host configuration, environment, model `playback_gop`, model bitrate or
archive settings. Source encoding, feedback ceilings, scheduling and native RTP
negotiation read that session's selection. The model track remains H.264. For
VP9/AV1 the encoder's nominal cadence follows the source CLI capture descriptor;
model cadence remains 24 FPS. H.264 retains its existing nominal-rate path.
Actual source PTS, frame production and browser presentation remain measured
quantities; a nominal 30 FPS does not establish 30 presented frames/s.

Before reserving a new explicit source, Room checks the selected native binary's
`--capabilities` (`native-video-rtp-v1`, `sessionSourceEncoding:true`, selected
source codec and H.264 model support). It then encodes and decodes three real
AUs at the selected source width, height and encoder rate. H.264's probe also
freezes its actual SPS profile for negotiation. Each new explicit session probes
again; there is no success cache that can reuse a 480p proof for 720p.
The two subprocess limits are 5 and 15 seconds. Probes run off the event loop,
outside the host admission lock, one at a time; cancellation retains that bound
until the subprocess work finishes. Probe failure refuses the candidate without
starting a World, CLI, model or listener. CPU probes do not acquire a GPU.

The native sender receives `--source-codec` and, for H.264, the verified
`--source-h264-profile` only for an explicit override. With omission, its original
arguments and defaults remain unchanged. The new sender must be deployed with
the matching Python implementation. The resolution selection does not reset or disable codec, bitrate, GOP, QP, VBV,
preset or intra-refresh settings. Source capture uses the common integer 1–60
FPS contract at both resolutions. VP9 and AV1 are not constrained by H.264
macroblock limits. H.264 uses the selected rate at 720p and retains the existing
60-FPS hint at 480p. The software encoder selects its actual SPS profile/level;
native negotiation uses the verified value, without forcing Level 3.1. VP9 keeps
profile 0; AV1 keeps profile 0, level-idx 5, tier 0. Model H.264 negotiation and
model geometry are unchanged.

Managed single-player launch runs the same native reservation and bounded
preflight before its detached source allocation. It stores the verified H.264
profile on the source pipeline before startup, because its native peer is only
created at the later browser attachment. Failed preflight starts no source;
failed or cancelled startup releases its own reservation. The first attachment
and reconnect both reuse the accepted session settings and profile without a
second probe. The omitted/default managed launch retains its existing path.

## Receipt and recovery

`session.ready.source_encoding` echoes a copy of the server-accepted canonical
selection only for an explicit override. The default omits this field. The WTP
client compares the receipt with its signed request; a ticket alone does not
prove that an encoder was applied.
Formal Player clients compare the same field with their frozen operation's
selection after attachment. The operation's `ready` state alone is not an encoder
receipt. Native `rtc.configuration.video_codecs` reports `whitebox` as the chosen
source codec and `model` as `h264`; the source stream descriptor and negotiated
SDP provide separate codec evidence.

Resume normally omits the selection and reuses the source session's configuration
and preflight profile, including before its first encoder output is ready. An
identical canonical echo is accepted; a changed value or a new override on a
default source is rejected before detaching its current subscriber. Recovery does
not re-read deployment defaults or run another encoder capability probe. A new
Room epoch still loses the source under the existing recovery contract.

## Validation and release boundary

CPU checks live in `test_whitebox_session_encoding.py`,
`test_whitebox_source_encoding.py`, `test_whitebox_capture_clock.py`, native RTC
and lifecycle tests, `test_whitebox_player_control.py`, and `rtc_sender` Go tests.
Managed-launch CPU regressions exercise signed manual init with all three
canonical codecs through the real `RenderHost`/`SourceSession` factory, frozen
receipt/profile recovery, and rejected/interrupted startup cleanup. Their native
preflight is stubbed; real codec and native checks remain separate gates.
The checks cover strict signed
normalization, nonce preservation on tampering, per-session ownership, default
arguments, pre-allocation/off-loop probes, early recovery, real elementary
encode/decode, and RTP depayload identity. Run the Python tests with
`SGLANG_LIGHTWEIGHT_RUNTIME=1 PYTHONPATH=python` and the realtime test directory as
pytest's `--confcutdir`; run `go test ./...` from `rtc_sender`. The real binary
preflight test accepts `WHITEBOX_TEST_RTC_BINARY` pointing at that exact build.
PyAV must provide libx264/libvpx-vp9/libsvtav1 and an AV1 decoder for all codec
checks; a skipped codec check is not validation of that codec.

Release the compatible Gateway/Room Python and native sender first. Verify the
omitted/default path, then release WTP with its default-visible debug selector
and issue explicitly bound candidate tickets. No environment switch is required.
An older Room still enforcing the removed switch will reject explicit selections,
so update the backend before WTP. Roll back the matched images to undo this
change; the removed switch is not a rollback control. Confirm each actual Room image,
software codec library, native binary and WTP client together. Formal 480p and 720p
source playback, input, model H.264 start/stop, reconnect, authorization failures
and the next session's unchanged default all require deployment acceptance.
This implementation and CPU evidence do not establish production playability,
current online encoder parameters, all-scene quality or stable browser FPS.


The 720p override requires the compatible Python admission/probe code in both
Gateway and Room, plus SDK/WTP clients that allow resolution and encoding together.
The existing session-source-codec native sender is compatible; no Go protocol
change is required. Upgrade an older native binary lacking
`sessionSourceEncoding` before enabling overrides. CPU encode/decode evidence
does not validate 720p model output, GPU latency/memory or browser playability.


When `source_encoding` is omitted, 480p retains its existing encoder and native
arguments. A default H.264 source outside the historical Level 3.1 envelope
(such as 720p at 36/60 FPS) probes the existing paired source settings before
allocation and freezes the actual SPS for initial attachment and reconnect.
This internal probe does not add an override to signed requests or ready DTOs.
An unavailable encoder or incompatible native binary fails admission; no codec
or rate fallback is implied. Model geometry, bitrate, native immediate encoder,
and model negotiation are unchanged by the source-capture rate change.

CPU regression evidence covers real 1280×720 encode/decode at 36 and 60 FPS for
H.264, VP9 and AV1. The optional native transport fixture input
`WHITEBOX_SOURCE_CODEC_FIXTURE` lets the existing TCP negotiation/elementary-AU
roundtrip test consume freshly generated AUs and their real H.264 SPS. See
[the high-rate CPU validation record](../../../../../benchmark/whitebox/rtc-source-high-fps-20261111.md).
These checks do not establish browser throughput, GPU/model performance or
production deployment acceptance. GOP remains capped at 24 by the existing
source-encoding contract; this change does not extend that separate policy.
