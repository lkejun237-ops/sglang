# Zing fixed-version AdaLN tables

This implements [issue #257](https://github.com/seedleap/sglang/issues/257), based on `f43a7f97e0a68a09dcb6956f08b4e275b5120df8` (#264). It supports normalized four/five-branch serving exports, shared or independent condition tags, and valid tag permutations. CPU synthetic acceptance is recorded in [ISSUE_257_EXPERIMENT_PLAN.md](../../../../../benchmark/zing_h3_causal/ISSUE_257_EXPERIMENT_PLAN.md). Real-checkpoint SP1 GPU acceptance is done (see Status). SP2 fixed-conditioning/RNG resident-versus-table acceptance is done on two L20D GPUs (see Status). A stable speedup conclusion is not established.

## Deployment contract

The offline builder consumes the effective generator/EMA checkpoint exported by [export_checkpoint.py](../../../../../benchmark/zing_h3_causal/export_checkpoint.py). Any LoRA must already be merged and its provenance explicit. It streams one main/final projection module at a time, executes each complete ordered unique-t batch with the serving module's original GEMM, and writes compact BF16 shift/scale/gate values. It does not combine singleton outputs to simulate a different batch shape.

The worker selects `weights_free` by setting `zing_adaln_table_path`; omitting the path selects the existing resident deployment. The loader verifies the export config and every selected shard's SHA256, skips main/final projection keys before `get_tensor`, and constructs these modules as `None` even on meta. Other weights, including TimeMLP and token refiner, remain resident. The table is copied to each rank's device at startup and fenced with a CUDA event. No builder runs inside a worker, and lookup adds no collective.

The first deployment is fixed-version, video-only, TP1 with SP1 or SP2, unquantized, and without DiT FSDP or layerwise/CPU offload. Changing weights, LoRA or schedule requires an offline rebuild and redeployment. Weight update and LoRA hooks reject before writes. No table miss invokes a deleted projection or switches to resident mode.

The worker holds its immutable table for its entire lifetime. Complete request plans acquire references before KV pool acquisition or prefix writes; failure, prompt update and close release references. There is no eviction or online insertion: the entire table must fit `zing_adaln_table_max_bytes` (default 512 MiB). This guarantees stable table addresses across sessions and graph lifetime. Graph static inputs still copy compact projections by value through the existing provider. SP1 and SP2 use the existing eager and graph projection provider. Graph capture/replay remains subject to the existing topology and eligibility rules. SP4 and higher are rejected; see Status for validation coverage.

## Status and support scope

| Item | State |
| --- | --- |
| SP1, real DMD step800 checkpoint, source plus six references, 480x864, 120 frames, 4 steps | Per-step tensors, KV and final latents are `torch.equal` and finite against resident; startup receipt reports zero projection parameters and storage; zero misses and fallbacks |
| Memory (SP1, allocated) | About 41 GiB lower after load and at request peak (236,495 to 195,312 MiB); table is 231 MiB |
| Latency | SP1 median 21.50 s resident vs 20.90 s table over 3 runs, not interleaved: treat as "no slowdown", not as a speedup |
| SP2, real step1000 four-branch checkpoint, source plus six references, 832x480, 120 frames, 4 steps | Eager saved FP32 final latents are `torch.equal` (max error/RMSE 0). Pooled `before_source` denoise CUDA Graph: 7 captures and 52 replays per rank in each mode; all 7,164 projection/step/RNG/final events per rank match, and saved final latents are `torch.equal` and finite |
| Memory (SP2, ready allocated) | 127.902475 to 95.582048 GiB per rank: 32.320426 GiB saved; 32.365674 GiB of projection storage omitted, 61.626 MiB table; 6,783 hits, zero misses/fallbacks per rank |
| Not covered | Chunk-level p50/p95, CUDA multi-stream fencing on GPU, audio, TP>1, quantized or offloaded DiT |

SP2 acceptance replays one conditioning snapshot and saved CPU/rank-local CUDA RNG states in both modes. Eager outputs reuse the completed `after_step0` run; the separate Graph comparison uses the production KV pool and `before_source` in both arms, preserving Graph eligibility rules. Graph routes also report 22 `rope_rebuild` eager forwards per rank; this is not full-request Graph coverage. Table projection parameters/storage are empty/zero, checkpoint identity is verified, table references and the pooled slot are released after the request. No KV tensors were scanned. The [acceptance record](../../../../../benchmark/zing_h3_causal/issue257_sp2_adaln_acceptance.json) contains counts, memory bytes, job IDs and checksummed artifact locations. This covers offline inference with the pooled production denoise Graph path; chunk Graph, streaming ingress, prompt updates, mixed schedules, five-branch checkpoints and H200 are not covered by this SP2 run. Earlier runs with unfrozen conditioning do not establish acceptance.

A table is only valid for one exact combination. Rebuild it (do not reuse the old one) when any of these change: checkpoint weights or export config; merged LoRA or the effective checkpoint source; the solver, timestep or sigma grid, shift, condition or reference timesteps; the GPU model or PyTorch, CUDA or matmul flags used to build. `build_adaln_table.py check` reports the first three; the worker also rejects the rest at startup or admission.

## Enable, verify and roll back

1. `plan`: write `plan.json` from the effective inference configuration of the release (solver, step list, shift, condition and reference timesteps). Do not infer these from the bare checkpoint or training config.
2. `build` (or `ensure`): run on one GPU of the target hardware and software, writing a new directory.
3. `check --model-path ... --table ... --plan ...`: must exit 0.
4. Merge `zing_adaln_table_path` into the `--pipeline-config-path` JSON (see below) and start the server.
5. Confirm the startup log `adaln_deployment_receipt`: `mode` weights-free, empty `projection_parameters`, zero `projection_storage_bytes`, `checkpoint_verified=true`. Request metrics should show zero misses.
6. Roll back by removing `zing_adaln_table_path` from the config and restarting; the worker then loads the resident projections from the same export. A wrong or stale table fails startup or admission and never falls back silently.

## Artifact schema

An artifact directory contains `manifest.json` and `table.safetensors`.

| Field | Contract |
| --- | --- |
| `schema` | `zing-adaln-table-v1`; unknown versions reject |
| `identity` | Exact export config SHA256 and basename→SHA256 for all serving shards |
| `provenance` | Exported checkpoint hash, generator/EMA, native/component versions, effective configuration and merged-LoRA evidence |
| `layout` | Actual tags/count, hidden/time dimensions, layer count; main six-part and independent final two-part widths |
| `batches` | Sorted unique FP32 values; raw bytes, dtype, shape and SHA256 for each complete batch |
| `schedule` | Video solver, exact ordered evaluation and sigma grids, condition/reference timesteps; no audio schedule |
| `arithmetic` | Full-batch BF16 strategy, PyTorch/CUDA versions, build device, CUDA device name/capability and matmul/determinism flags |
| `time_embedder`, `frequencies` | Individual parameter identities and generated frequency constants |
| `tensors`, `table_sha256` | Every compact tensor's shape/dtype/checksum and serialized table checksum |
| `build` | Synchronized embedding/projection/output-transfer wall time; build device and CUDA allocated/reserved/peak counters when built on CUDA |

`batch.<index>.slot.<layer>` stores a contiguous `[M * R, 6 * H]` block. Final stores `[M, 2 * H]`. The consumer reconstructs the original chunk views and strides, then uses the original token indices, normalization, modulation and gates. Token-expanded activations are not cached.

CPU and CUDA tables are different arithmetic contracts. A CPU build is useful for contract testing and is rejected by a CUDA worker. A CUDA build must match the worker's recorded software, GPU capability/name and arithmetic flags. The builder and resident oracle must use the same full-batch shape. Device/software identity alone is not a proof of GPU bitwise equivalence: the real-checkpoint acceptance gate must pass before rollout.

## Build and deployment

Use a unique output directory. Match the actual exported checkpoint's solver, grid, shift, condition and reference values; the numbers below are an example of the historical DMD configuration, not a new asset validation.

```bash
PYTHONPATH=python python benchmark/zing_h3_causal/build_adaln_table.py plan \
  --solver few_step --denoising-step-list 1000 750 500 250 \
  --num-train-timesteps 1000 --video-shift 5 \
  --condition 0 --reference 0.999 --output /path/to/run/plan.json

PYTHONPATH=python torchrun --standalone --nproc-per-node=1 \
  benchmark/zing_h3_causal/build_adaln_table.py build \
  --model-path /path/to/serving-export --plan /path/to/run/plan.json \
  --output /path/to/run/adaln-table --device cuda
```

`plan` covers active singleton batches, source/final clean batches, and complete nonempty combinations of condition/reference/clean prefix timesteps. Request admission additionally enumerates the actual packed prefix, source, active and final rows for every chunk, including tails and CFG. Unsupported audio roles and missing batches reject before KV acquisition. Overlap and arbitrary solvers remain unsupported by this first version.

### Keeping table and export in sync

Realtime initial-block mixed sampling uses a separate composite plan; see the
[design and migration guide](../../docs/docs/sglang-diffusion/zing/specs/initial-block-static-adaln.md).
An existing single-schedule table is not admitted as a mixed plan.

The table is bound to the export, so rebuild whenever weights, merged LoRA or schedule change. Two subcommands automate that decision:

```bash
# Preflight (CPU only): exit 0 = current, exit 2 = rebuild required, reasons listed as JSON.
python benchmark/zing_h3_causal/build_adaln_table.py check \
  --model-path /path/to/serving-export --table /path/to/adaln-table --plan /path/to/plan.json

# Run after every export: keeps a current table, otherwise builds a fresh one (needs one GPU).
torchrun --standalone --nproc-per-node=1 benchmark/zing_h3_causal/build_adaln_table.py ensure \
  --model-path /path/to/serving-export --plan /path/to/plan.json --device cuda
```

`check` compares the manifest with the export's config and shard hashes, the effective provenance in `zing_config.json` (including merged-LoRA evidence) and, with `--plan`, the schedule. It names each cause. `ensure` defaults the table to `<export>/adaln-table-<key>`, where the key hashes the checkpoint identity, provenance, schedule and the build device's arithmetic contract (GPU, PyTorch/CUDA, matmul flags), so any change gives a new directory and a repeated `ensure` reuses it instead of rebuilding. An existing table that no longer matches is never overwritten; `ensure` fails and asks for removal or a new `--table`. The worker still validates identity and arithmetic at startup, so a skipped preflight fails closed.

For the programmatic worker API, add these fields to the existing pipeline configuration:

```python
pipeline_config = {
    "video_decoder_backend": "taeh3",
    "taeh3_checkpoint_path": "/path/to/taeh3-checkpoint",
    "zing_adaln_table_path": "/path/to/run/adaln-table",
    "zing_adaln_table_max_bytes": 536870912,
}
```

For a server, merge those fields into the existing JSON passed to `--pipeline-config-path`. Preserve checkpoint, codec, KV pool, SP and other deployment settings. The existing real-checkpoint harness now accepts `--adaln-table PATH --adaln-table-max-bytes N` on its `sglang` subcommand; omitting `--adaln-table` runs the resident control. Read its `--help` and use the frozen request/options from the experiment manifest.

The startup log records `adaln_deployment_receipt`: weights-free mode must have an empty `projection_parameters` list, zero `projection_storage_bytes`, and `checkpoint_verified=true`. Request metrics also contain `adaln_deployment`. Reader auditing and model parameter inspection are separate evidence from CUDA `allocated`/`reserved`; never use `empty_cache` as the proof that projection weights were omitted.

## Local verification

The dedicated `zing-adaln-cpu.yml` workflow checks exporter and standalone table integrity/admission contracts with CPU-only dependencies. Full-model tests additionally exercise meta construction, actual streaming builder keys, strict loading of remaining parameters, exact per-chunk latent/KV equality for Euler and few-step with six synthetic references/CFG, reader filtering, prompt updates and exception cleanup. A CPU recording graph backend verifies capture/replay and interleaved session isolation; this does not claim actual CUDA graph acceptance.

The initial source analysis is retained as a historical snapshot in [ISSUE_257_ANALYSIS.md](../plans/ISSUE_257_ANALYSIS.md). GPU submissions require separate authorization and the project-mandated read-only submission review.
