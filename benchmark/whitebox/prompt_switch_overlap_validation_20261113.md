# Prompt switch overlap: development validation

Scope: code only. No service was deployed/restarted, no playtest Slot was acquired,
and no training or GPU task was touched. Base: main
`cd14fa7e38b2a6ae327beea4a3c3c57cfef9067a`.

Implementation and protocol behavior are described in
[the deployment guide](../../docs/docs/sglang-diffusion/zing/deployment_zing_h3.mdx) and
[the prompt/reference contract](../../docs/docs/sglang-diffusion/zing/specs/prompt-reference-switch.zh-CN.md).
This file records checks; it is not a GPU performance result.

## Local CPU checks

Python 3.11.15 on macOS arm64, Torch 2.8.0 CPU execution. A private development
venv reads the existing local dependency environment and adds xgrammar 0.2.1
and apache-tvm-ffi 0.1.11. This is not the pinned Linux CI recipe.

| Existing owner | Passed | Skipped | Coverage |
|---|---:|---:|---|
| `realtime/test_zing_source_encoder_api.py` | 83 | 0 | Same/different key release; L1/miss; reference snapshots; active identity; preparation failure; EOS/release; stale candidates; source-idle gate and control sequence budget |
| `realtime/test_whitebox_render_lifecycle.py` | 52 | 0 | Room clip fencing, source_ready versus kv_ready, lifecycle |
| `test_zing_encoder_dit_link.py` | 31 | 17 | CPU transport/FIFO/cleanup; CUDA/NCCL cases skipped |
| `test_zing_h3_queued_dit.py` | 30 | 17 | CPU queue/condition boundary cases; CUDA cases skipped |
| `realtime/test_zing_source_decoder_overlap.py` | 16 | 0 | Decoder/output overlap and condition receipts |

212 local CPU checks passed. The 34 skipped cases are not claimed to pass.
Owners were run in separate pytest processes with `SGLANG_LIGHTWEIGHT_RUNTIME=1`,
`PYTHONPATH=python`, `HF_HUB_OFFLINE=1`, and `OMP_NUM_THREADS=1`.
Model kernels are controlled CPU fixtures where the existing tests require them;
the new checks prove orchestration and ownership, not real model values.

Local collection first failed because xgrammar and then tvm_ffi were absent.
Private dependency installation resolved those failures. Two new fixture errors
(missing peer release and missing ledger records) were corrected. Ordinary Ruff
reported two pre-existing F811 fixture-name warnings; repository-selected
F401/F821/UP037, isort and formatting are the applicable checks.

## Fixed Linux CPU CI

PR CPU CI is required for the final source. The workflow run and per-owner
receipts are linked from [PR #715](https://github.com/seedleap/sglang/pull/715). Do not treat the local table above as Linux
approval or replace historical inventory evidence with a macOS run.

## Uncovered

- 480p and 736p GPU numeric/visual parity, FPS, prompt-to-visible latency and
  Encoder/DiT/Decoder peak memory were not measured.
- NCCL, CUDA Graph replay, independent candidate copies under real allocation
  pressure and compatibility with periodic Sink need the original fixed-input
  baseline/candidate benchmark before deployment.
- Preparation may outlast an initially idle source window. No FPS improvement or
  absence of regression is claimed. Only one immutable candidate is retained;
  the extra snapshot lifetime must be included in memory measurement.
- Browser, multiplayer, recording, 180-second sessions and production acceptance
  were not run because the user requested code development only.

The existing negotiated capability gates the optimization; legacy/non-async
peers retain synchronous boundary completion. A preparation control RPC cannot
consume a source command number or `max_chunks`. Ready source work wins over
opportunistic preparation, and the 17-frame activation boundary is unchanged.
