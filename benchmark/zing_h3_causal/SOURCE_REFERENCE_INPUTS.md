# Continuous source reference inputs

The streaming adapter accepts at most nine reference images and preserves their
input order. The combined 8 MiB payload budget and 16,777,216-pixel per-image
limit remain enforced. `session.ready` advertises `max_references=9` and
`max_reference_bytes`.

Reference JPEG/WebP/PNG `id` values are preserved in the internal user media and
target visual-reference conditions. Clients must keep IDs unique and match
the IDs named in their prompt. The benchmark client accepts repeated
`--reference` and matching `--reference-id` arguments. Both clients detect the
actual image format from its bytes, including extensionless content-addressed
files, and send the original bytes without transcoding.

An optional `prompt_video_frame` init field contains `{id, format: "png",
payload: <bytes>}`. It is a single lossless snapshot, decoded as a one-frame
video with clip `{start: 0, end: 0}` for the prompt encoder only. It does not
become a visual reference condition or an aligned source. Its ID must be
distinct from reference IDs; its bytes count toward the same 8 MiB budget.
The client uses `--prompt-video-frame` and `--prompt-video-id` for this field.

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
pixels. Do not send `prompt_video_frame` for this case or automatically extract
source RGB into prompt media. An explicitly requested prompt-video extension is
separate from the corrected reference semantics.

Validation messages added by this protocol extension are English:
`Please provide at most nine reference images.` and
`Please provide a uniquely named PNG prompt video frame.`
