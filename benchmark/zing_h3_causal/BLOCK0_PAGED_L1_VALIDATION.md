# Block0 paged L1 validation

## CPU evidence

2026-11-02, PR #499 head `367a32adc65542ad52742ba2d851f7ff90406508`,
checkout merge `5ee8422b44671d87800981c6bfe09bda90b47255`.
[Cache contracts run 41409448193](https://github.com/seedleap/sglang/actions/runs/41409448193)
succeeded, including two fresh CPU trials of each file:

| File | Passed per trial | Failures/skips |
| --- | ---: | ---: |
| test_zing_paged_cache.py | 13 | 0/0 |
| test_zing_block0_cache_contract.py | 65 | 0/0 |
| test_zing_preparation_cache.py | 38 | 0/0 |
| test_zing_prompt_switch_cache.py | 37 | 0/0 |
| test_zing_kv_pool_runtime.py | 15 | 0/0 |
| test_zing_lynn_startup_warmup.py | 10 | 0/0 |

178 cases per trial, not 356 distinct cases. The same job also passed two Graph
provider trials and the prefill/management API trials. The new 13-case candidate
is promoted in the CPU manifest only after checking both records and its exact
source hash. The status-byte accounting regression was added after this snapshot. For its
subsequent cache trials and the complete CPU matrix, use [PR #499 checks](https://github.com/seedleap/sglang/pull/499/checks).
This table describes the frozen first head, not unexecuted later changes.

[Artifact 13124494918](https://github.com/seedleap/sglang/actions/runs/41409448193/artifacts/13124494918),
ZIP SHA256 `c18c1be2038eee5e4c045f33e0835705c3c6e44bf18e7687725999d162a3f03a`.
Local copy: workspace `exports/pr499-paged-l1-20261102-r1/ci-first/`.

Local checks: Ruff/isort, Python AST/JSON, CPU inventory and an isolated PagedKV
module reproduction. Full project imports on WSL encountered the absent runtime
(torchvision); no dependency chain was installed and no local project regression
suite was run.

## Capacity inputs and performance boundaries

Two deidentified reconstructed inputs are approximately 2855/4886 prefix rows.
They are shape examples, not full production traffic. For 2048 rows/page they
occupy 2/3 pages, with 1241/1258 unused tail rows. Combined tail waste is 24.40%
of allocated rows; at 1024 it is 5.51%. With a 40960-row fixed-slot comparator,
shape allocations are reduced by 90%/85%; these are arithmetic capacity results,
not process RSS savings or measured performance. Original observed deployment
used max_prefix_rows=57344; comparisons normalize this parameter explicitly.

`bench_block0_paged_l1.py` can compare dense/paged H2D and warm-cache L2 reads
on one GPU without loading models. No PAI job was submitted for this PR; the later user-selected H100 execution
is recorded below. The component result does not establish
480/736 E2E/Encoder/DiT/Decoder metrics. CPU copies with fake
CUDA streams/events do not establish GPU correctness or performance.

Design, API and stopping conditions:
[BLOCK0_PAGED_L1.md](../../docs/docs/sglang-diffusion/zing/plans/BLOCK0_PAGED_L1.md).

## Frozen implementation: complete CPU matrix and H100 component execution

Full [CPU CI run 41410737009](https://github.com/seedleap/sglang/actions/runs/41410737009)
passed all 13 jobs on `fa171b20eea925f1119e6d9a57f06bffc8eda072`.
[Cache run 41410736946](https://github.com/seedleap/sglang/actions/runs/41410736946)
also passed: 179 cache/pool cases per trial (including pending/SP byte accounting),
plus 6 Graph provider and 27 prefill/API cases per trial, zero failures/skips.
Subsequent changes in this update are documentation only.

The user selected the Shanghai H100 target instead of Thailand. Read-only probe
confirmed the node `e01-cn-2zd4xi8m505`, actual L20Z 80GB / driver580.105.08,
and local NVMe ext4 on `/dev/nvme0n1` mounted at `/mnt/zing-nvme`.
This component run uses GPU0 UUID `GPU-6d5005a8-3721-3368-3cb1-893d25094c1c`;
another task owns GPUs2–7, so this is not an exclusive-node measurement.

- Zero-GPU precheck Pod `pr499-paged-load-cpu-20261102-r1` completed / exit0;
  staging invocation `t-sh071vrp2kr2s5c`.
- Single-GPU Pod `pr499-paged-load-gpu-20261102-r1`, staging invocation
  `t-sh071vsb4qremf4`, starts eight independent-process cells. Active deadline is
  1200 seconds; shell deadline1050. It loads no model and sends no service requests.
- Requested cached image digest894eb66a6f96755d439e07045c27dac2dd5087d7caa5ea5f4d534bf7a500740a;
  containerd reports resolved import digest eab963c4ca51cdcb349810996203ba28e14058d69b905d4f6a5473872493177b.
  Precheck observed Torch2.13.0+cu130 / CUDA13.0.
- Each cell uses the same BF16/50-layer/28-local-head/head_dim128 layout,
  max_prefix_rows57344, warmup2 + measured5; rows2855/4886/12288/40960,
  page_rows1024/2048. Independent processes prevent host allocator caches from
  accumulating across cells. Pod RAM limit192Gi accommodates the dense baseline,
  paged buffer and warm filesystem cache. Pinned allocation is outside load timing.
- Source bundle SHA256 `f7162aaa5a3ca2b02a013620b81f6d6690972a80b5a07f0ca3d8d057a4379827`.
  Reports/receipts are retained under workspace `exports/pr499-paged-l1-20261102-r1/h100-load/`
  and node `/mnt/zing-nvme/pr499-paged-l1-20261102-r1/`.

The component outcome, timing table and cleanup receipt are maintained in
[PR #499](https://github.com/seedleap/sglang/pull/499); creating a Pod is not a pass.
The Thailand PAI candidates were never submitted. A public OSS download failed
before any precheck Pod was created; the successful stage used Shanghai OSS's
internal endpoint. All failed discovery/staging receipts are retained.
This execution does not validate API preload latency, concurrent SP transfers,
attention/Graph numerics, or the 480/736 E2E/Encoder/DiT/Decoder matrix.

## Continuous-run follow-up (2026-11-02; validation pending)

The approved follow-up keeps 2048 rows/page and one copy stream, uses layer/part-major
backing, scans a free-page occupancy table in ascending order, and coalesces adjacent
logical/physical pages. No extra eviction is used to obtain contiguous pages.
The earlier component evidence above describes the unmerged page-major implementation.

`bench_block0_paged_runs.py` compares that frozen page-major copy loop with the
coalesced layout, using identical page IDs and payload bytes. Each cell runs in a
fresh process, warmup2 + measured5, in both H2D and D2H directions. The approved
cells are 4886/contiguous and 40960/{contiguous,churn,scattered}. Churn and scattered
page maps are generated by actual allocator/release/LRU methods with small payloads;
these synthetic operation sequences are reproducible scenarios, not a measured traffic distribution.
Main results report contiguous/churn separately; fully scattered is a stress case.
The tool checks all copied values, unused pages, tail padding, and the GPU guard.
CPU CI precedes a one-GPU/no-model job with a maximum 600-second lifetime.
