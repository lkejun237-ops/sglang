# Zing baseline validation and review handoff

## Scope and acceptance

This PR implements the single-GPU Zing H3 causal generation baseline. Production
regression tests belong in this PR; the three-way native/SGLang comparison harness,
cloud job manifests, frozen bundles, model weights and generated media do not.

The current experiment request evaluates end-to-end generation effects against
native Zing rather than enforcing layerwise numerical parity. This change of
experiment objective does **not** establish numerical equivalence or resolve the
previous review's numerical acceptance concern. Maintainers must explicitly decide
whether the baseline can merge with that known limitation; no approval or waiver is
claimed here.

## Real-checkpoint experiment: e2e-threeway-r2

[Internal three-way media report](https://review-cdn.loopit.com.cn/world-model/evals/zing-h3-causal/b300-baseline-20260916/layer1/e2e-threeway-r2/review/index.html)
(office network or company VPN required).

- Native source: `0a1fdc897cbf8cf76264c7259c78f3ee1339739f`.
- SGLang source: `9042c558325f38ee8ac3c6f662e12506511ab4ed`.
- Experiment lock SHA-256:
  `979243083a553f2928cc8408d56eacf91da4f6b3969a1bb8ee7f288d7251f92b`.
- Same step3500 model, requests and source/reference media; native, SGLang with
  frozen native surroundings, and independent SGLang production-speed groups.
- Frozen group reuses native conditions, noise, schedule, packing and sampling
  update; SGLang runs the DiT and transfers lossless latents to the native image
  for decoding. Independent SGLang uses its own full pipeline.
- 16 GPU stages succeeded, producing 12 final media across four cases. Peak
  concurrency was eight single-GPU jobs on one node. The runtime reported CUDA
  capability 10.3 and 274113 MiB; product-name metadata was inconsistent.
- CPU export verified 74 files; all 27 published files were read back through the
  CDN with matching hashes, and MP4 range reads returned HTTP 206.

| Case | Input and requested edit | Output |
| --- | --- | --- |
| image-edit | Source hilltop image; add a person beside the man | One image |
| multi-reference-image | Source blockout plus three reference images; render ranger and volcanic landscape | One image |
| video-edit | Source desert video; add a walking camel beside the woman; no additional reference image | 125 frames, 24 FPS, 5.208 seconds, no audio |
| video-audio-edit | Same source and edit, with video/audio target | 125 frames plus an audio stream |

Generation, media decoding, dimensions/frame counts, required stream presence and
file integrity passed. **Human quality review remains pending.** These results are
not a speed benchmark: the groups have different initialization/processing costs.

### Audio limitation

The three video-audio outputs have 32 kHz stereo AAC streams, but measured mean
volume is approximately -91 dBFS (peaks approximately -84 to -90 dBFS), effectively
inaudible during normal playback. The prompt used `N/A` for both soundscape and
music. This is an inadequate test of meaningful sound generation, not evidence of
correct audible output. It does not prove a unique model/decoder root cause.
The linked report's original technical `passed` checks stream presence; it predates
this loudness follow-up. A future audible-quality case needs explicit sound content,
waveform/silence checks and listening review.

## Version coverage

The r2 generation results cover `9042c558`, not the newer PR source baseline
`effd0b47c3749f549deabab100bd66869a024320`. Subsequent changes include request
preparation, media timestamps, validation order, output FPS pinning, rejection of
unsupported post-processing, transformer validation refactoring and lightweight
encoder imports. The old generated media cannot validate these changes.

The CPU tests cover contracts and synthetic execution on the newer source. Tests
with mocked media IO or encoders do not validate real Qwen/VAE numerics or actual
codec output. A real-checkpoint run on the final merge candidate remains required
before claiming final-head end-to-end validation.

## CPU regression result

On 2026-09-17, the six test files listed in the README completed with **71 passed,
2 skipped** using Python 3.13 and PyTorch 2.14.0 on macOS CPU, with
`SGLANG_DIFFUSION_PLATFORM_OVERRIDE=cpu`. The skipped tests require CUDA. The
seven new media tests cover four request modes, source-length rejection and two
video timestamp cases. This is synthetic/contract evidence, not GPU validation.
Ruff 0.15.1 format and selected lint checks passed for all 36 Python files in the
existing PR plus the new test; isort 7.0.0 passed for the new test.

## Review and merge gates

- Review production tests and the source-version distinction above.
- Obtain a maintainer decision on the historical strict numerical acceptance
  concern; do not reinterpret media generation success as passing that threshold.
- Complete human quality review and a meaningful audio case before claiming full
  accuracy/quality acceptance.
- Check current CI separately. On `effd0b47`, base tests were gated by the missing
  `run-ci` label; other jobs reported integration permissions failures. Whole-tree
  lint also included unrelated baseline files, in addition to two PR-owned format
  issues corrected in this follow-up. These are not successful CI results.
- Do not bypass branch protection, fabricate approvals, or resolve reviewers'
  acceptance concerns merely by changing documentation.

See the benchmark README for the reproducible CPU command. Runtime receipts,
cloud-specific scripts and generated assets remain outside this PR.
