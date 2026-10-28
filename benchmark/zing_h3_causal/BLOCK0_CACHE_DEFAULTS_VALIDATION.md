# Block0 cache defaults: CPU validation

## Scope and status

Base: `5e66d5e2c3` (PR #445). Date: 2026-10-28.
This draft changes cache startup, addressing, persisted identity, SP shard publication,
and byte budgets. It does not change the DiT calculation or GPU restore algorithm.
The user explicitly selected CPU cache-correctness validation for this change.

| Check | Status |
| --- | --- |
| AST, Ruff/isort, whitespace | Passed before the first draft commit (11 Python files); workflow YAML/JSON and static inventory also passed |
| New CPU contracts | 25/25 passed in each of two fresh GitHub Actions trials; admitted with artifact/source-hash evidence |
| Existing preparation/switch/pool CPU regressions | 4/24/15 passed per trial, two trials each; zero failures/skips |
| 800 GB pinned allocation, cgroup/memlock limits | Not run; no large allocation on WSL |
| CPFS mount, locking semantics, throughput | Not run on CPFS; unit tests use private temporary directories |
| 480/736 GPU output/performance, GPU A→B→A | Not run, outside the agreed CPU validation scope |

Runtime configuration and identity design:
[Block0 cache defaults](/docs/sglang-diffusion/zing/plans/BLOCK0_CACHE_DEFAULTS).
Historical #445 GPU results remain version/workload-specific; they are not acceptance for this draft.

## Minimal closed loop

The dedicated `Zing Block0 cache contracts` workflow uses the existing pinned Linux CPU
recipe on `ubuntu-24.04` / Python 3.11, `SGLANG_LIGHTWEIGHT_RUNTIME=1`, offline assets,
and two fresh trials per file through `scripts/ci/zing/run_cpu_tests.py --triage`.
Artifacts include commands, source hashes, logs, JUnit and trial receipts.
The normal Zing CI also runs the admitted regression files. The first broad run rejected the new UNRUN file as designed; the follow-up metadata commit admits it from these two successful trials.

| Contract | Coverage |
| --- | --- |
| Startup | No pool skips caching without disk/host arguments; off needs no cache parameters; missing/invalid budgets and missing/unmounted roots reject |
| Pinned budget | Small CPU stand-ins verify slot bytes, floor/remainder, and insufficient capacity; no 800 GB allocation |
| Key/identity | DiT/Qwen/VAE and layout changes separate keys; content changes with the same filename/length reject; mount-path changes preserve asset identity |
| AdaLN reuse | Already verified identity is consumed without a second DiT scan; unverified tables reject |
| SP2 persistence | Rank 0 alone is invisible; both shards publish; per-rank conditioning and KV read back correctly; truncated/missing shards reject |
| Competing writers | Different fill generations cannot complete one mixed entry |
| Restart/migration | Fresh cache instance recovers tensors; model/SP mismatch, legacy identity-free metadata and tampering miss |
| Geometry | Same-byte-size wrong dtype/shape rejects before restoring to L1 |
| Disk capacity | More than 50 tiny cases survive with sufficient bytes; a smaller root budget evicts across model namespaces; broken index rebuilds all namespaces |
| IO errors | Corrupt conditioning after startup is a miss; startup invalid roots fail |
| Existing lifecycle | Preparation bypass, CPU KV restore, slot reuse, switch semantics and disposal remain covered by existing files |

Run in the CI/PAI CPU environment, not on the WSL editor:

```bash
python scripts/ci/zing/run_cpu_tests.py --suite dit-pipeline-contracts --triage --trials 2 --timeout 500 \
  --file python/sglang/multimodal_gen/test/unit/realtime/test_zing_block0_cache_contract.py \
  --file python/sglang/multimodal_gen/test/unit/realtime/test_zing_preparation_cache.py \
  --file python/sglang/multimodal_gen/test/unit/realtime/test_zing_prompt_switch_cache.py \
  --file python/sglang/multimodal_gen/test/unit/test_zing_kv_pool_runtime.py \
  --file python/sglang/multimodal_gen/test/unit/test_zing_lynn_startup_warmup.py \
  --output-dir artifacts/block0-cache
```

Stop after two passing CPU trials and appropriate static checks. Expand tests only
for a failing contract or a newly changed behavior; GPU work requires a separate agreed scope.
CPU tests prove the cache's addressing/content/ownership rules, not GPU bitwise equivalence,
800 GB allocation feasibility, or CPFS production behavior.

## Two-trial CPU receipt

Runtime/test source: `88ffc192a0372f9c4085bd2867adb9b1509cfc67`.
GitHub merge checkout: `ca6bf9aa46cb4b549af8137e5cbed452dff74b07`.
[Workflow run](https://github.com/seedleap/sglang/actions/runs/40686237526) and
[artifact 12815765941](https://github.com/seedleap/sglang/actions/runs/40686237526/artifacts/12815765941).
Artifact ZIP SHA256: `8ac47d4c11b18417ae019dff0f77a35a25c0e1b265a42bf650e4dece6c4a6894`.
The archive was downloaded and its checksum and all four test source hashes were verified before admission.

| File | Trial 1 | Trial 2 |
| --- | --- | --- |
| preparation cache | 4 passed | 4 passed |
| prompt switch cache | 24 passed | 24 passed |
| GPU pool CPU runtime | 15 passed | 15 passed |
| new Block0 contracts | 25 passed | 25 passed |

Total: **68 cases/trial, 136 passes across two trials, zero failures/skips**.
Environment: Ubuntu 24.04 runner, Python 3.11.17, Torch 2.8.0+cpu, no CUDA runtime.
The admission follow-up changes only documentation and the CPU inventory; runtime and test source remain unchanged.
The complete Zing CI rerun after admission is separate from this targeted passing evidence.

## CI compatibility follow-up

After main merged PR #452 (`6f53b2e31d`), the PR merge checkout exposed two
fixture mismatches: the source-policy helper now requires the deployment command,
the mocked pipeline used by Lynn startup tests lacked the resolved model path,
and the new prefill/live policy test's fake cache lacked its persistent identity.
Main's subsequent PR #454 (`d89e38d75c`) also adds an internal-prefill fixture;
that fake cache receives the same explicit model/layout identity.
The fix passes an empty command in the key-isolation test and gives the pipeline
fixture its model path and the fake cache a concrete model/layout identity,
preserving key equality, command-policy isolation, warmup-failure and readiness-order assertions.

The branch is synchronized with both main commits, preserving both cache-contract
and prefill-API inventory records. The targeted two-trial workflow
now also includes the 10 Lynn startup cases: **78 cases per trial** across five files.
The earlier 68-case receipt above remains evidence for its original source only.
Latest validation is reported by the [PR #455 checks](https://github.com/seedleap/sglang/pull/455/checks);
the follow-up does not relax assertions, skip failing tests, or change cache behavior.

## Restricted conditioning serialization follow-up

The user deferred LRU policy changes to a separate PR. This follow-up only replaces
unrestricted conditioning pickle loading with a versioned plain-data envelope and
`weights_only=True`; the two fixed record types are rebuilt explicitly.
No custom class is allowlisted, and a rejected file never retries unrestricted loading.

Nine added CPU cases cover complete nested conditioning/pool-request round trips,
a pickle execution marker that must remain absent, five legacy/schema/record/tensor/shape
rejections, legacy-manifest rejection before loading, and writer-side object rejection.
The existing restart/signature-rebinding cases retain all KV and model-mutation assertions;
their optional persisted signature is now plain text in the new data protocol.
The dedicated workflow passed **87 cases per trial, 174 passes across two trials,
zero failures/skips** on runtime/test head `e17d9f166b0f117b30911aaf428d7a34da648517`.
[Run 40705343014](https://github.com/seedleap/sglang/actions/runs/40705343014),
[artifact 12823402662](https://github.com/seedleap/sglang/actions/runs/40705343014/artifacts/12823402662),
merge checkout `0b599cae1c50ec9eef00daa832688639fa975add`.
ZIP SHA256: `77c16639b64ce052b7e6265373e730e1a713399e63095052d3d2d64c4bd1dbc8`.
The ZIP checksum, all five test source hashes, and both trials were checked before
refreshing the two modified files' CPU inventory evidence. The follow-up commit
changes only evidence/documentation. Full CI status is available in the PR checks.
Earlier 68/78-case results apply only to their recorded sources.
WSL runs static checks only. No GPU, large pinned allocation or CPFS job is added.

## Main PR #439 integration

Main `9986ea5feece36834f828129602865e16f77a663` adds cached-prefix joint
consumption and a CPU-only speculative prefetch hint. The merge keeps those paths,
adapts the hint to `rank-N/kv.bin` in the identity namespace, and preserves
restricted conditioning loading and coordinated SP publication.
Startup tests cover joint scheduling with the cache enabled/disabled; obsolete
late/early/overlap ablation combinations are replaced by the agreed production
on/off behavior, while invalid legacy modes remain covered by startup rejection tests.
The upstream L1/L2 joint-switch and speculative-RPC fallback cases remain enabled.
The five-file targeted workflow now collects **131 cases per trial** (34/36/36/15/10).
The earlier 87-case artifact remains evidence for its recorded pre-merge source;
the merged source's result is reported by the PR checks. LRU policy is unchanged.

The first merged full-CI run exposed one additional upstream Graph-provider test
still calling the old mode/count constructor. It now uses the same byte-budget
CPU fixture as the cache tests, retaining all six backend/warm-start combinations
and their output, cache-state and model-mutation assertions. The dedicated workflow
runs this exact six-case node in two fresh CPU processes, with separate JUnit/logs
alongside the five-file receipt (131 cases/trial plus 6 Graph-provider cases/trial).
The full Graph file remains in normal CI with its existing reviewed skip policy.
