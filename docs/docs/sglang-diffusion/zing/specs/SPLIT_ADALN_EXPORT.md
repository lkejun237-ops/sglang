# Zing split-AdaLN export contract

This foundation is extracted from [PR #242](https://github.com/seedleap/sglang/pull/242)
for [issue #257](https://github.com/seedleap/sglang/issues/257). It retains full
resident AdaLN weights and current denoise CUDA Graph execution. Persistent
projection tables and weights-free loading remain separate work.

## Accepted checkpoints

`export_checkpoint.py` accepts a complete, native-merged `backbone.*` state dict:

- Legacy single `adaln_proj.linear` tensors.
- Split `video_linear`, `text_linear`, `audio_linear`, `source_visual_linear`
  tensors, plus `condition_visual_linear` when condition has an independent tag.

Base tags must be unique and contiguous. Any permutation is supported.
Condition may share the video tag or add a fifth, distinct tag. Omitted
`condition_visual` means shared video; no other alias is accepted. An optional
`num_modalities` must agree with the actual projections.

Split weights and biases are concatenated in ascending native tag order. No
branch is initialized from another, and LoRA merging stays in native Zing.
Mixed layouts, incomplete branches, unexpected keys, incompatible dtypes and
three-modality initialization checkpoints are rejected before writing output.
The main-block projection width is `6 * R * H`; the final projection is always
`2 * H`, independent of the four/five main-block branches.

## Provenance and runtime metadata

Split input and LoRA-enabled configs require `--provenance <file.json>`:

```json
{
  "native_sha": "4c01c82d17db366bd82c81541b42bbada16d0e43",
  "checkpoint_source": "generator",
  "component_versions": {"h3": "<actual component revision>"},
  "lora_merged": true
}
```

Use the actual native revision and component revisions; `checkpoint_source`
must explicitly identify `generator` or `ema`. Unversioned legacy single exports
remain supported with a warning. Version-2 metadata records input/runtime
layouts, projection order, source-file SHA256, provenance and a redacted effective
config. `transformer/config.json` carries the actual tags and branch count.
The runtime rejects disagreements in projection order or condition timesteps
between these two config files. Version-1 canonical routing remains supported.

Text and condition rows use `ar_condition_timestep` (default `1.0`); reference
visual rows use `reference_visual_timestep` (split default `0.999`, matching the
native snapshot above; legacy single default `1.0`). Explicit config values
override these defaults. Both must be finite within `[0, 1]`. Prefix KV writes
must match those values; source and committed target writes still require clean
`t=1`. Reference visuals retain the video projection, even with an independent
condition projection. Native rational RoPE scales such as `"5/3"` are accepted.

## Verification boundary

Run the standalone conversion contracts on CPU:

```bash
SGLANG_LIGHTWEIGHT_RUNTIME=1 PYTHONPATH=python \
  python -m pytest -q test/registered/unit/diffusion/test_zing_adaln_export.py
```

The dedicated GitHub workflow runs these contracts. Existing diffusion unit
tests additionally exercise model routing, persisted KV, mocked SP collectives
and a CPU recording backend for graph capture/replay. Exact converted tensors
and same-shape routing comparisons are separate from the split-versus-concat
FP32 GEMM oracle (`atol=2e-6`, `rtol=1e-5`): changing GEMM shape can change floating
point rounding. These tests do not validate full trained checkpoints on CUDA,
real CUDA Graphs, image/video quality or performance. GPU acceptance is pending.
