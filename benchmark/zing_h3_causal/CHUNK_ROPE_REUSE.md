# Chunk-specific historical RoPE invalidation (#180)

The experimental `SGLANG_ZING_CHUNK_ROPE_REUSE=1` switch preserves independent
historical rotation publications. It does not change the history RoPE operator,
relative position mapping, current KV writer, QK norm, or eviction copy operator.
The established path remains the default pending real-checkpoint acceptance.

## Chunk meaning and size

A temporal chunk is one source + target/audio transaction for a latent time
interval. Source and target are separate packed role blocks; merging them at
commit preserves their separate RoPE part identities. Reference-prefix rows
are cached separately. A cache part contains token rows, so its row count also
depends on spatial resolution and modality; it is not a fixed-size GPU tile.

The frozen #180 checkpoint configuration uses `num_frame_per_block=[1, 2, 2]`
and `split_first_latent=false`: the latent-frame lengths repeat
`1, 2, 2, 1, 2, 2, ...`, with the final interval clipped at EOS. These are latent
frames, not output video frames. `ChunkPlan` reads the model configuration;
the cache does not hard-code this cycle. `window21/sink3` counts temporal chunks,
not frames. The unit-test fixture's two token rows are deliberately synthetic.

## Exact reuse condition

Within the same model/policy binding, reuse requires an exact match with the
last successfully published entry for that layer: the same source/target part
identity, raw K content version and layout, effective `(t, h, w)` coordinates,
raw-store offset, and rotated-output range. Unchanged coordinates alone are
insufficient. Compaction that moves a part rebuilds it; unchanged parts at the
same ranges remain reusable. Current-step K/V is always written anew.

A publication includes an owned chunk identity, exact effective coordinate token,
raw K tensor identity/version and layout, authoritative store offset, and rotated
output range. A moved chunk is rebuilt; an unchanged chunk at the same range can
survive changes elsewhere. Buffer growth copies only valid published ranges.
Cache-owned raw writes acknowledge shared tensor version increments; unaccounted
writes advance a per-layer raw epoch before further owned writes. Thus a normal
`no_grad` scratch append does not invalidate immutable historical rows.
Every layer is unpublished before any write, including a potentially failing
current-tail write. Policy/model changes and clear discard all publications.

Coordinate comparison uses CPU shape/device/dtype/identity guards and one batched
CUDA-to-host boolean read per forward when live chunks can be compared. It never
performs a blocking read per chunk. This conservative path still launches
comparison/reduction operations for each comparable chunk; it does **not** claim
zero metadata or synchronization overhead. Candidate raw K buffers retain version counters even inside `inference_mode`;
unaccounted in-place edits therefore invalidate rotations without GPU reads.

`rope_reuse_stats` counts rebuilt and reused history rows across layers and map
synchronizations. Session `clear()` resets these counters. The correctness oracle
can run on CPU or CUDA:

```sh
PYTHONPATH=python SGLANG_LIGHTWEIGHT_RUNTIME=1 \
  python -m pytest -q benchmark/zing_h3_causal/test_chunk_rope_reuse.py
ZING_TEST_DEVICE=cuda PYTHONPATH=python SGLANG_LIGHTWEIGHT_RUNTIME=1 \
  python -m pytest -q benchmark/zing_h3_causal/test_chunk_rope_reuse.py
```

`validate_chunk_rope_cuda.py` adds H56/D128 BF16 attention and real uneven-row
Ulysses collectives. Run each SP degree in a fresh process with `BASELINE_CACHE`
pointing to the frozen main cache file and `OUT` pointing to a new evidence
directory:

```sh
torchrun --standalone --nproc-per-node=4 \
  benchmark/zing_h3_causal/validate_chunk_rope_cuda.py
```

Repeat with one and two processes. The harness covers at least 63 evictions,
small windows, full-history absolute positions, prefix replacement, storage
growth, failed publication, inference-mode raw edits, and model/session cleanup.
Its synthetic inputs supplement the required real-checkpoint comparison; they
do not establish model quality or performance.

Acceptance is against main `e6537184cdcd9de05e04da6201be70446847b5be` with the
source-video + six-reference case `zing-whitebox-video-five-triview-0001`.
Same-SP numerical comparisons require bitwise equality. Performance uses 832×480,
24 FPS, 10s/15s, window21/sink3/block_relative/gap12, SP1/2/4, independent warm
processes in A→B and B→A order with five measured requests per arm/round.
Each SP1 10s/15s round requires at least 5% lower median denoise latency and
a positive lower bound of the 95% bootstrap confidence interval (10,000 independent
resamples of each arm, fixed seed 180). SP2/SP4 denoise and all API configurations
must have no confirmed regression exceeding 2%. Unstable or inconclusive samples
require a new documented attempt or additional samples. Independent long requests
must cover at least three full window turnovers (at least 63 evictions for window21),
and separate traces cover a representative full-window interval. Peak allocated
growth is limited to max(256 MiB, 1% of baseline); reserved is reported separately.
After ten sessions, released allocated memory must not grow by more than 32 MiB
from the first session. Profiler samples never enter latency medians. Existing long-video
quality defects must remain explicitly separate from candidate regressions.


## minWM comparison

Reviewed minWM causal attention, raw position-aware cache, and RoPE-out at
`9b7c118183c12e19793d5902453474d952da4726`; these two source files are unchanged
in the frozen main baseline. Adopted: reuse rotated history during repeated
current denoising, caller-owned output storage, and one layer-independent position
plan. Source/target merge preserves each original part identity, so appending a
target does not itself invalidate its source rows.

Not adopted: minWM's global rotated-valid flag and position mathematics. Zing
requires part-specific effective coordinates, gap clamping, raw-content versions,
and physical mapping guards. Zing's existing BF16 products, accumulation, and
pass-through dimensions remain untouched. No new kernel, preallocation policy,
compact algorithm, or Ulysses packing is introduced.
