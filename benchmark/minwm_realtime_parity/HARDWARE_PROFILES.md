# MinWM hardware deployment profiles

This is the stable source for humans and Codex when generating MinWM benchmark or
deployment manifests. Do not use old dated YAML as a template.

## Generation workflow

1. Read the actual GPU name, total memory, and compute capability:

   ```bash
   nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
   ```

2. Select a profile by **physical memory first**. A product label never overrides
   the reported memory.
3. Copy `k8s/minwm_hardware_job.template.yaml` to `k8s/generated/<run>.yaml`
   and replace every `REPLACE_WITH_...` token.
4. Keep the generated file untracked. Validate it with:

   ```bash
   kubectl apply --dry-run=server -f k8s/generated/<run>.yaml
   ```

5. Record the detected hardware, selected profile, SGLang/model commits, image
   digest, and result URI with the run artifacts.

## Hardware gates and `vae_cpu_offload`

Do not derive memory policy from the string `6000`. NVIDIA's full-card published
capacities differ by generation: RTX 6000 Ada has 48 GB, while the RTX PRO 6000
Blackwell Server, Workstation, and Max-Q editions have 96 GB. A cloud device or
partition labelled `6000` can expose a different amount, including about 32 GiB.
The value reported inside the workload is authoritative.

Use this decision table when Codex renders a job:

| Visible memory and architecture | `vae_cpu_offload` | Decision |
| --- | --- | --- |
| SM120 and <=36,864 MiB, conservative fit validation | `true` | `blackwell-32g` |
| SM120 and 28,672-36,864 MiB, explicit speed validation | `false` | `sm120-32g-speed`; use only through the image-bundled launcher and apply the speed gates below |
| SM120 and >=65,536 MiB | omitted / `false` | High-memory speed default |
| SM100 and >=180,000 MiB | `false` | `experimental-sm100-high-memory`; B200 validation only |
| SM103 and >=250,000 MiB | `false` | `experimental-sm103-high-memory`; B300 validation only |
| SM90 in an explicitly bounded H100/H200 validation Job | `false` only after the required same-process fit smoke passes | device-specific `experimental-sm90-*-no-offload`; never a deployment default |
| Any other capacity or compute capability, including a 48 GB Ada card | `true` initially | Experimental; do not disable until the same device passes the acceptance gates |

Always record the queried GPU name, total MiB, compute capability, and the final
offload value. A familiar product name is not evidence that offload can be
disabled. The conservative `true` setting for unclassified hardware is a first
validation candidate, not a claim that the hardware profile is production-ready.

## Checkpoint and request invariants

These values align the current MinWM checkpoint with the Tianpeng setting. They
are not hardware tuning knobs and must remain identical across profiles:

| Setting | Required value |
| --- | --- |
| local attention / sliding window | 32 frames |
| attention sink | 8 frames |
| RoPE position mode | `block_relative` |
| RoPE maximum frame gap | 12 |
| prompt first-frame pin | enabled |
| request cache window / sink | 32 / 8 |
| cache growth | disabled for the bounded request |

Use these conversion arguments:

```text
--local-attn-size 32
--sink-size 8
--sliding-window-num-frames 32
--rope-position-mode block_relative
--rope-max-frame-gap 12
--prompt-first-frame-pin-enabled
```

The request must set `realtime_causal_kv_cache_num_frames=32` and
`realtime_causal_sink_size=8`. At runtime, verify the one-time
`MINWM_RUNTIME_ALIGNMENT` log rather than assuming the requested values took
effect.

## Realtime startup KV-cache resolution buckets

Launch profiles own the supported-resolution contract. Requests cannot add a
bucket or trigger an unplanned allocation after worker readiness.

| Compatibility alias | Canonical bucket / exact output (width x height) | Latent resolution (width x height) | Patch tokens / frame | 32-frame cache tokens |
| --- | --- | --- | --- | --- |
| `480p` | `832x480` | `52x30` | 390 | 12,480 |
| `720p` | `1248x704` | `78x44` | 858 | 27,456 |
| `720p-wide` | `1280x704` | `80x44` | 880 | 28,160 |

Canonical bucket names are the exact output dimensions: `832x480`,
`1248x704`, and `1280x704`. The compatibility aliases in the first column are
accepted at the CLI boundary, but startup logs, leases, and launch contracts use
the canonical names. Existing `720p` continues to mean `1248x704`; it is not
silently redefined when the wider variant is enabled.

The image-bundled launcher selects the buckets as follows:

| Launch profile / hardware | Startup buckets | Service behavior |
| --- | --- | --- |
| `sm120-32g-speed` / RTX 5090 32 GiB | `832x480` | Initialize and accept 480p only; reject both 720p shapes during request adjustment |
| `resident-speed` / H100, H200, B200, B300 | `832x480,1248x704,1280x704` | Accept 480p and both exact 720p resolutions |
| `sm120-highmem-speed` / SM120 >=64 GiB | `832x480,1248x704,1280x704` | Accept 480p and both exact 720p resolutions |

Each pool slot owns one physical self-attention K/V/rotated-K allocation sized
for the largest enabled bucket. Acquiring a slot applies the selected bucket's
logical cache and sink bounds; releasing it resets cursors and makes the same
backing allocation reusable by a later non-overlapping request. Do not allocate
separate resident 480p and 720p buffers for one slot.

For the current 30-layer, 24-head BF16 cache, increasing the largest bucket
from `1248x704` to `1280x704` adds 704 cache tokens: approximately 371 MiB per
SP1 slot or 186 MiB per rank at SP2. This is much smaller than allocating a
second independent 720p slot.

The current launcher fixes both global and per-worker realtime admission to one
active session and allocates one slot per rank. Increasing admission requires an
equal pool-size increase and therefore multiplies the resident cache memory.

## Hardware profiles

### `experimental-sm90-h100-no-offload` and `experimental-sm90-h200-no-offload`

This profile is a benchmark-only exception to the conservative unclassified-
hardware default. It does not classify all SM90 devices as safe for no-offload
deployment. A generated Job must first gate on the reported compute capability
and a bounded physical-memory range, then use the exact server process intended
for the formal run to complete a local-TAEHV no-offload protocol smoke.

For Hopper, the smoke uses 8 warmup chunks. Baseline adds 2 measured smoke
chunks, while NSYS retains its 1 measured smoke chunk. Four latent frames are
generated per chunk, so the warmup alone crosses the complete
32-latent-frame causal-cache window. The smoke must prove all of the following
before the 20+200 headline segment may start:

- the one-GPU server is still alive with `vae_cpu_offload=false`;
- the reported GPU name, SM version, and total MiB match the rendered bounds;
- local StreamingTAEHV loaded in the server process;
- all expected raw-RGB payload and server-timing chunks completed;
- the Tianpeng runtime-alignment assertions passed.

The runner records this evidence in
`NO_OFFLOAD_PROTOCOL_FIT_PASS.json` and reads the marker back immediately before
the formal segment. A failed or missing marker is fail-closed. A successful
marker authorizes only the following formal segment in that same server process;
it does not promote the hardware to a stable no-offload profile.

The current isolated Spot validation targets are managed node groups in
`aws03-usw2/default`, both using the verified RWX `s3-claim`:

| Requested device | Exact instance / managed node group | Required physical memory | Compute capability |
| --- | --- | --- | --- |
| H100 | `p5.48xlarge` / `minwm-spot-p5-h100-sglang-0718` | 80,000-90,000 MiB | `9.0` |
| H200 | `p5en.48xlarge` / `minwm-spot-p5en-h200-sglang-0718` | 140,000-150,000 MiB | `9.0` |

Both manifests use exact node-group and instance selectors plus the
`seedleap.ai/workload=wan22-ti2v` taint/toleration. The actual in-workload
`nvidia-smi` result remains authoritative; sharing SM90 does not allow one
device to inherit the other's memory range.

### `blackwell-32g`

Select when an SM120 GPU reports at most 36,864 MiB. This includes the 32 GiB
5090/production target described for this work. If a machine is labelled
"6000" but reports 32 GiB, this profile still wins.

Add these explicit server settings:

```text
--vae-config.taehv-checkpoint-path <taehv-checkpoint>
--vae-cpu-offload true
--text-encoder-cpu-offload true
--dit-cpu-offload true
--dit-layerwise-offload true
--dit-offload-prefetch-size 0.0
```

Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. This is the
lowest-memory candidate, not a speed recommendation.

Validation status:

- Physical fit passed on an RTX PRO 4500 Blackwell Server Edition reporting
  32,623 MiB: 17,411 MiB peak at 832x480 and 25,505 MiB at 1248x704.
- The same-SKU performance gate has **not** passed for the 32 GiB 5090/6000
  production target. Do not claim production readiness from the fit result.

### `sm120-32g-speed`

Select this explicit speed-validation profile only when an SM120 GPU reports
28,672-36,864 MiB, or on a larger SM120 GPU when qualifying the exact low-memory
policy. Start the server through
`sglang.multimodal_gen.tools.minwm_profile_launcher`; do not reproduce the
managed arguments by hand. The launcher fixes the following residency policy:

```text
--performance-mode speed
--text-encoder-cpu-offload true
--vae-cpu-offload false
--dit-cpu-offload false
--dit-layerwise-offload false
```

The DiT and TAEHV remain resident to preserve throughput. Promotion requires an
832x480 same-digest run with client FPS >=24 and peak process memory <30,720 MiB.
This profile passes
`--realtime-causal-kv-cache-pool-buckets 832x480`; it intentionally does not
initialize or serve the 720p bucket.

Validation status:

- The exact policy passed on an RTX PRO 6000 Blackwell Server Edition at
  30.944 FPS with a 27,751 MiB peak. This qualifies the policy and proves
  32 GiB memory feasibility on SM120.
- RTX 5090 uses the same SM120 dispatch but has not completed the same-SKU
  acceptance gate. Treat it as a validation candidate, not a production-ready
  profile, until that run passes.

### `blackwell-high-memory`

Select when an SM120 GPU reports at least 65,536 MiB. The measured
RTX PRO 6000 Blackwell Server Edition node reported 96 GiB.

Add:

```text
--performance-mode speed
--vae-config.taehv-checkpoint-path <taehv-checkpoint>
```

Do not add CPU/layerwise offload unless a separate memory goal requires it. The
96 GiB characterization measured 30.045 FPS at 832x480 and 11.356 FPS at
1248x704 for local TAEHV, but those numbers do not prove a 32 GiB deployment.

### `experimental-sm100-high-memory`

Select this experimental profile only when `nvidia-smi` reports compute
capability `10.0` and at least 180,000 MiB of visible physical memory. Record the
reported GPU name as well; `B200` in the name confirms the intended device but
does not override either gate.

Add these explicit server settings:

```text
--performance-mode speed
--vae-config.taehv-checkpoint-path <taehv-checkpoint>
--vae-cpu-offload false
```

Validation status:

- The current B200 `a3` observation is only no-OOM and run-progress evidence for
  running without VAE CPU offload. It is not a formal successful benchmark and
  does not validate local TAEHV at 24 FPS.
- This profile remains experimental and must not be described as
  production-ready.

### `experimental-sm103-high-memory`

Select this experimental profile only when `nvidia-smi` reports compute
capability `10.3` and at least 250,000 MiB of visible physical memory. Record the
reported GPU name as well; `B300` in the name confirms the intended device but
does not override either gate.

Add these explicit server settings:

```text
--performance-mode speed
--vae-config.taehv-checkpoint-path <taehv-checkpoint>
--vae-cpu-offload false
```

Validation status:

- The exact-720 SP1 result recorded in `README.md` completed at 1248x704 with a
  51,588 MiB per-GPU peak. This is same-device physical-fit evidence for the
  high-memory B300 profile.
- That historical result does not validate the current local TAEHV path at
  24 FPS. This profile remains experimental and must not be described as
  production-ready.

### Unclassified hardware

Do not silently choose a nearby profile when the reported compute capability and
visible memory do not match one of the exact gates above. Generate a manifest
only after marking the profile experimental and defining an explicit
memory/performance validation.

## Acceptance gates

For a 32 GiB production candidate:

- physical process peak <= 30,720 MiB;
- complete 1089 ordered frames at 832x480;
- at least three repetitions with no OOM;
- headline FPS CV <= 3%;
- <= 5% throughput regression against the accepted baseline on the same GPU SKU
  and software environment.

For H100/H200/B200/B300 and high-memory SM120 profiles, run the same stability
gates at `832x480`, `1248x704`, and `1280x704` because all three buckets are
exposed.

Fit on a different SKU is evidence for memory feasibility only.
