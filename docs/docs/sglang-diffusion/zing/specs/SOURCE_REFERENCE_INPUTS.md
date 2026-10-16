# Continuous source reference inputs

The streaming adapter accepts at most nine reference images and preserves their
input order. The combined 8 MiB payload budget and 16,777,216-pixel per-image
limit remain enforced. Reference IDs retain their order in user media and target
visual-reference conditions. Keep IDs unique and match the prompt's Picture IDs.
The client accepts repeated `--reference` / `--reference-id` arguments and sends
original image bytes, detecting their actual format even for extensionless files.

For causal source inference, send source pixels, references and prompt; no preview
option or duplicate first-frame upload is required. Ready is sent before S0 exists.
The worker reads S0 directly from the first accepted source packet into a CPU RGB
tensor. There is no S0 PNG encode/decode, temporary file, or separate media transfer.
The full packet still enters source VAE once. Qwen uses the declared source FPS;
it cannot recover pixels or metadata already lost on the client. Later packets
do not change S0. Prompt updates reuse retained media at the existing chunk boundary.

For independent presentation, use ordered `user_media` instead of `references`:

```python
user_media = [
    {"type": "video", "id": "<Video 1>", "format": "png", "payload": png_bytes,
     "fps": 24, "clip": {"start": 0, "end": 0}},
    {"type": "image", "id": "<Picture 1>", "format": "jpeg",
     "payload": jpeg_bytes, "reference": True},
]
```

The streaming subset accepts image bytes and single-frame video presentation,
not arbitrary paths/URLs or encoded video uploads. Video FPS is required, finite,
positive and at most 240; its clip is zero-time. `reference: true` selects image
media for VAE conditions. Other media are presentation-only. Image/video numbering
is independent and follows declaration order. At most ten media (including derived
S0) and nine references share the 8 MiB uploaded-byte budget. Explicit video supplies
the presentation instead of automatic S0, retaining its type, clip and order.
An image cannot claim `<Video 1>` when S0 is automatic. `references` remains
shorthand for ordered image references and cannot be mixed with `user_media`.

`session.ready` advertises `user_media_formats`, `max_user_media`, and the
single-frame `user_media_video_clip` contract. The old `prompt_video_frame` field
and CLI flags are removed. The proposed `source_preview` request field and
`--source-preview` flag are also removed, rather than introducing a new caller
switch. Requests using these removed fields fail validation. Use source alone
for its own first frame, or `--user-media manifest.json` for independent explicit
media (each item uses a local `path` in place of `payload`). There is no alias.

For complete video inference, supply the full video as `target.source`. With the
model's existing `causal=true` configuration, a request without explicit video
presentation automatically prepends native `type=video, clip={start:0,end:0}`
using that source URI. Native itself has no `ar-first-frame` mode: the training
converter constructs this ordinary video descriptor and native inference consumes
it. SGLang request preparation supplies that descriptor for convenience.
Canonical explicit video media (including independent target preview) is retained.
Image targets and noncausal configurations do not acquire a source first-frame
preview. Clipped sources need an explicit presentation descriptor. Neither full
source VAE input nor reference conditions are rewritten.

All aligned source pixels must still arrive through `source.append`, followed
by `source.end`. Arbitrary `zing_request` and complete source objects remain
rejected on the public streaming API. Temporary media remains session-owned.

For `zing-whitebox-video-five-triview-0001`, the historical sample ID is retained,
but the corrected case contains six references and seven assets. Preserve the
original `<Picture 1>` through `<Picture 6>` order and original prompt. Picture 1
is the user-confirmed global rendered reference (PNG SHA256
`03475dd7769de747f16b0d13709495cce426c310a62822b93eb38db6eb92fa05`),
followed by the five paired reference sheets. The case manifest SHA256 is
`a97154b3010064e846fe0ad7b6cc21de3704885c7f0eba5c618231de395e2947`.
Use only references, prompt, and source_stream at init, followed by paced source
pixels. Current causal inference derives source Video 1 automatically, while all
six images remain Picture 1–6 and all six visual reference conditions remain.
For native comparison, include the equivalent source `type=video` plus zero-time
clip in native user content; omitting it would compare different Qwen inputs.
The historical no-preview measurements keep their frozen revision and are not
current AR parity evidence. The separate source-first-frame case ID remains for
traceability, and the target-first-frame variant remains an explicit video case.

Validation messages are plain English. This direct source protocol does not change
world-platform error codes, close codes or callback errors.
