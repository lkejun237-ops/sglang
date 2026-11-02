# Continuous source reference inputs

The streaming adapter preserves reference order and uses the deployment setting
`zing_reference_max_images` (default 21, nonnegative integer). A reference image
may contain at most 64 MiB of encoded bytes and 16,777,216 pixels; complete input
and transport capacities derive from the configured count. Reference IDs retain their order in user media and target
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

The public streaming media inputs are `source_stream` plus `references`, with
`prompt` describing their use. Reference entries contain exactly `id`, `format`
and `payload`; every image becomes a VAE reference. `<Video 1>` is reserved for
the internally derived source presentation. There is no caller-provided video
preview or presentation-only image. The removed `user_media` field is rejected,
even when null or empty, and the client no longer offers `--user-media`.

## Block0 cache controls

The session `init` accepts `cache_write` (strict boolean, default `false`) and
optional `game_id` / `group_id` labels. Labels are nonempty strings of at most
128 UTF-8 bytes; a group requires a game. They describe cache ownership and do
not enter the model input or content key. The write setting applies to the
initial prompt and subsequent prompt changes for the whole session.

Normal video requests only query L1. A miss recomputes the prefix without
checking disk; new L1/L2 entries and business associations are created only when
`cache_write=true`. Existing L1 entries remain readable with writes disabled.
Use prefill or explicit preload before routing video to the prepared instance.
See the [managed cache contract](/docs/sglang-diffusion/zing/plans/BLOCK0_CACHE_MANAGEMENT_SHARED_L2)
for preload/status/unload and the game/group selection parameters.

## Source presentation and admission

`session.ready` advertises reference count, image/upload/decoded-byte limits and
source transport limits. It does not advertise the removed independent-media
capabilities. `prompt_video_frame`, `source_preview` and their former client flags
remain rejected. Source preview always comes from the first accepted RGB frame;
changing prompt reuses that preview, while `references` replaces the reference
image list when supplied in a prompt event.

For complete video inference, supply the full video as `target.source`. With the
model's existing `causal=true` configuration, a request without explicit video
presentation automatically prepends native `type=video, clip={start:0,end:0}`
using that source URI. Native itself has no `ar-first-frame` mode: the training
converter constructs this ordinary video descriptor and native inference consumes
it. SGLang request preparation supplies that descriptor for convenience.
Canonical explicit video presentation must match the source URI or be selected
as a reference condition; unrelated visual media is rejected before Qwen/VAE.
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

## Replace references during a continuous session

The server requires `prompt_switch_prefix_latent_frames=5`. With checkpoint plan
`num_frame_per_block=[3,2], split_first_latent=True`, `session.ready` advertises
`prompt_reference_updates: true`. Check this capability before sending images;
older services may silently ignore additional event fields.

The existing MessagePack `kind="prompt"` event accepts an optional top-level
`references` list. Each item uses the same `id`, `format` (`png`, `jpeg`, `webp`)
and image-byte `payload` fields as initial references. `payload` on the event
itself remains the complete prompt string. Use a new increasing `event_id`, even
when only the images change and the prompt text stays identical.

- Missing `references`: retain the worker's currently applied reference set.
- `references: []`: remove all references, retaining source preview and other
  presentation media.
- A nonempty list: replace the entire reference set, in the supplied order.
- `references: null` or invalid images: reject the entire update with
  `CONDITION_INVALID`, `fatal: false`, and its `event_id`; continue the session.

The combined retained presentation and new references must fit the initial media
count, image-dimension and upload-byte limits. A `control_ack` acknowledges
admission only. Encoding or KV capacity can still reject an update. Observe
`prompt_event_id` on output blocks to determine when it actually applies.
The newest worker-admitted update supersedes older pending updates, including
when the newest update is rejected; the previously applied epoch stays active.

All explicitly supplied references are decoded and VAE-encoded again. The new
condition/reference prefix and the latest five clean target latents (`1+2+2`)
share one KV rebuild at the existing safe switching boundary. Source input and
output timestamps continue without resetting the stream. Identical-image VAE
latent reuse is a future optimization.

For direct service testing, `continuous_source_client.py --prompt-updates updates.json`
accepts a JSON list, with local image paths relative to the manifest:

```json
[
  {"source_frame_start": 24, "prompt": "Use <Picture 1>.",
   "references": [{"id": "<Picture 1>", "path": "new-reference.png"}]},
  {"source_frame_start": 48, "prompt": "Keep that appearance in the snow."},
  {"source_frame_start": 72, "prompt": "Use no reference images.", "references": []}
]
```

Positions must increase. The client splits source packets as needed and sends
the update before the source packet starting at that position. This schedules
ingress, not the exact output frame where the model switches; use output event
metadata for that. The client fails negotiation if a reference update is
scheduled but the service lacks the capability.

This change does not extend the authorized world-platform rewrite contract or
prove that an upstream coding agent already produces complete reference lists.
See the [implementation design](/docs/sglang-diffusion/zing/specs/prompt-reference-switch.zh-CN)
for state ownership, cleanup failure handling, measured results and remaining GPU validation.

## Count and memory budgets

Init and complete reference replacement share the configured N-reference /
N+1-media limits. Each image is bounded by 64 MiB encoded and 48 MiB RGB
(16,777,216 pixels). Reference RGB defaults to N × 48 MiB; encoded upload and
transport budgets also derive from count, reserving one extra presentation slot.
`session.ready` advertises the actual count, per-image, reference, total-media,
RGB and negotiated message limits. Retained non-reference media use total slots.

Zing derives its maximum WS message and transport host budgets unless explicitly
overridden. A client may negotiate a smaller message budget, which ready returns
and later messages must respect. Increase the client's `--max-message-bytes` to
send a larger complete snapshot; smaller existing clients continue to work.
Other models retain the generic transport defaults.

Remote Qwen requires `limits.max_media=N+1` in its separate service configuration.
Null byte/pixel/inflight fields select count-derived sizing; existing explicit
numeric limits remain enforced. Identity calculation no longer hides a second
22-item or 512 MiB ceiling. Remote workers check configured media limits and
the raw RGB request-size lower bound before hashing or serialization. RPC
request framing includes a count-dependent bound for safetensors metadata;
HTTP admission and automatic request sizing share this calculation. The
actual encoded body is still checked against explicit request limits. Model condition rows and GPU KV admission remain
separate; these budgets do not resize the KV pool or imply maximum-sized GPU
inference has passed. Formulae, defaults, migration and validation scope are in
[the feature design §9](prompt-reference-switch.zh-CN.md#9-启动时配置-reference-张数与容量2026-10-21).



## Encoded reference ownership and update latency

Reference upload means caller-to-API encoded PNG/JPEG/WebP file bytes. The API
verifies and reserves RAM/inflight capacity, then passes the original bytes to
the worker's shared init/update memory decoder. No reference media is spooled,
uploaded to OSS, forcibly compressed or transcoded. URI media remains supported
for existing canonical requests. IPC and SP broadcast serialize/copy bytes;
this path is not zero-copy. Initial images travel only in the first dispatch;
updates carry a complete new set. Pending bytes are released on supersession or
rejection, in-flight bytes after dispatch, and initial bytes at session cleanup.

The client writes per-event outcomes to `prompt-updates.json` and summary JSON.
Timing begins before `send(event)` and ends at the complete first valid payload
with the corresponding event ID. It includes sending/upload waiting, excludes
local image reads and browser rendering, and uses one client monotonic clock.
Each row contains submission/arrival times, output frame/PTS and latency in ms.
Rejected, superseded, unobserved-at-EOS and interrupted events retain a null
latency. An ACK is not activation; event adoption is not semantic editing success.

Pass `--prompt-updates <manifest>` to `bench_streaming_source.py` to include the
same schedule in serial warmup/measured runs. The shared report displays each
measured event and aggregates only applied events in successful measured runs,
with median/P95/count. Controlled GPU performance at 480/736 is still unmeasured;
existing diagnostic timings and CPU checks do not constitute that acceptance.

### 可选的 reference 解码大小限制

部署可在 `--pipeline-config-path` 对应 JSON 中设置 `zing_reference_max_decoded_bytes`（正整数，单位字节）。默认 `null` 按 `zing_reference_max_images × 48 MiB` 自动计算；显式值覆盖。实时 API 在 RGB 解码及模型编码前，从图片头按 `sum(width * height * 3)` 检查每组 reference；初始化和后续完整集合更新共用该口径，等于上限允许，超过则拒绝。初始化的全部 `references` 图片计入；内部 source 展示和旧 active 集合不计入。超限更新为非 fatal 错误，旧状态保留，后续合法更新可以继续。

这不是压缩上传大小、整个 worker 的峰值内存或 KV/显存容量限制；单图64 MiB压缩上限、按数量推导的完整上传预算及现有模型容量检查仍独立生效。该配置不覆盖直接调用原生 URI/offline worker 的接口。完整边界见[设计方案 §8.7](prompt-reference-switch.zh-CN.md#87-显式配置-reference-解码大小上限)。

The startup flag `--zing-reference-max-images 21` overrides the pipeline JSON
`zing_reference_max_images` value. If omitted, the JSON value remains in effect;
if neither is supplied, Zing defaults to 21. Zero disables references.
