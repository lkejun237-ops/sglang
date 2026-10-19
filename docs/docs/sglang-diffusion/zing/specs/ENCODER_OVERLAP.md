# Zing source Encoder / DiT overlap

## Deployment contract

This is an experimental same-node source Encoder service. Encoder weights are
resident on a dedicated GPU group; DiT and the Lynn Decoder share
another group. It is not integrated with the generic `runtime/disaggregation`
orchestrator; integration is tracked in #267. Independent instances and request routing are separate concerns;
do not treat this private service as an Omni deployment API.

Encoder ranks divide existing spatial tiles, preserving geometry, ownership,
temporal caches and assembly order. They do not partition consecutive time blocks.
The Encoder leader sends posterior moments (mean and log-variance before latent
sampling, not metadata) to the DiT leader through an isolated two-rank NCCL group. Sampling
remains on the consumer with the existing indexed RNG and source watermarks.
The DiT network receives sampled latents, not moments directly. This loopback
service does not support cross-host H20/H200 deployment; non-IB transport needs
separate backend support and bandwidth/latency/staging validation (#267).

`serial` computes on demand; `overlap` prefetches already submitted source work in
order, with at most two pending results and a byte limit. New input is available
only after its dispatch has handed it to the Encoder. This does not implement
continuous input delivery during an in-flight DiT dispatch.

## Start the roles

Use disjoint GPU sets. E2 + SP4, for example, needs six GPUs:

```bash
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.run --standalone --nproc-per-node 2 \
  -m sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.encoder_worker \
  --model /prepared/model --port 31001
```

Configure the DiT service on GPUs `0,1,2,3` with its normal SP4 launcher. Add this
inside the existing progressive `source_encoder` profile:

```json
{"remote": {"port": 31001, "mode": "overlap", "capacity_bytes": 268435456,
            "timeout_seconds": 120, "profile_dir": "/results/encoder-traces"}}
```

Keep the existing mode, contract version and factory fields. `profile_dir` is
used only while Torch profiling is enabled, not during formal timing. Freeze the
same dependencies for worker, service and client. Import checks alone do not
prove production WebSocket version requirements are satisfied.

## Prepare inputs without a GPU

Preparation uses the existing whitebox case manifest and asset verifier. The
default case contains a real source video and six ordered references. Choose a
different case explicitly; do not substitute an older five-image case.

Provide a deployment manifest with the full realtime profile and expected Lynn checkpoint
SHA256. The profile's `model_config_sha256` binds `zing_config.json`:

```json
{"realtime_profile": {
  "model_config_sha256": "<expected config SHA256>",
  "solver": "few_step", "denoising_step_list": [1000,750,500,250],
  "guidance_scale": 1, "noise_version": "indexed-cpu-v1",
  "source_encoder": {
    "mode": "progressive", "contract_version": "h3-clip17-progressive-v1",
    "factory": "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.progressive_encoder:create_session"
  }
}, "decoder_checkpoint_sha256": "<expected checkpoint SHA256>",
 "decoder_architecture_sha256": "<model_dir/config.json SHA256>",
 "decoder_config": {"type": "lynn_streaming", "checkpoint_path": "/models/lynn.pt",
                    "model_dir": "/models/lynn", "precision": "bf16"}}
```

These values are experiment inputs, not hashes hardcoded into the tool:

```bash
python benchmark/zing_h3_causal/prepare_encoder_overlap.py \
  --deployment-manifest deployment.json --model CHECKPOINT --media-root MEDIA \
  --decoder-checkpoint LYNN_WEIGHTS --width 832 --height 480 --output NEW_PREP_DIR
```

Use `--case-manifest` and `--case` for another validated six-reference case.
Preparation emits an existing-checkpoint deployment copy, request JSON, RGB
streams and integrity receipts; it does not convert or train checkpoint weights.
The same preparation applies to shared and dedicated baselines.
Preparation checks manifest media identities, model/Lynn identity, source frames
and FPS, and records geometry and the preparation tool hash. Git commit identity
is recorded when preparation runs in a checkout; source archives must retain their
separate frozen-source manifest. Long inputs loop
the original source; loop boundaries are synthetic and must be reported.

## Select and run an experiment

Select layouts explicitly; the runner does not recommend a deployment. Layouts
can also be listed in the prepared input JSON's `layouts` field:

```bash
python benchmark/zing_h3_causal/run_encoder_overlap_matrix.py \
  --input NEW_PREP_DIR/input.json --output NEW_RESULTS \
  --stage overlap --layout enc1-sp1-serial --layout enc1-sp1-overlap
```

Supported Encoder sizes are 1/2/4; DiT SP is 1/2/4, plus shared-only SP8. All
configurations run serially with at most eight GPUs. The runner starts and cleans
up only its own process groups. CPU preparation/analysis must not occupy GPUs.
Cloud resource availability must be checked separately before submission.

Stages `encoder`, `transport`, `capture`, `correctness`, `dit`, `overlap` and
`soak` separate computation, lifecycle and measurement. Component replay inputs
are captured by an untimed request. Capture copies tensors to CPU and writes
files; never enable it for formal performance requests.

Formal collection includes a full warmup and five runs; profile is separate.
`--collection-phase formal` and `profile` permit separate jobs, but profile must
bind identical source/configuration and completed formal evidence through
`--performance-root`. Missing profile evidence is not a pass. For 120-second
stability, select one overlap layout with `--stage soak` and its completed paired
`--performance-root`.

## Correctness and measurement

Encoder moments and indexed latents must exactly match the same-version single
rank reference. Every pop must prove actual group, unique tile ownership, gather
and backend policy; replicated fallback is not parallel success. Component and
matrix checks share `encoder_execution_evidence.py`.

`encoder_tile_mode` defaults to `parallel` for dedicated Encoder experiments;
shared baseline and consumers keep `auto`. Explicit `auto` retains the shared
implementation’s validated selection/fallback policy, not guaranteed sharding.
`cudnn_benchmark_limit` defaults to `1` in the matrix (explicit `null` retains the
model policy); it applies only inside Encoder contexts and is restored.
Reference and candidate must share the frozen policy; do not mix timing or outputs
across policies.

End-to-end correctness compares shared/dedicated captures, source/noise watermarks
and complete raw RGB. Cancellation, reconnect, old-token isolation, busy-owner
protection, EOS and cross-rank cleanup failures remain mandatory.

Maximize rank time per repetition before statistics. Report wall time, first
frame, paced block p50/p95, EOS drain, FPS and stalls separately. Kernel sums do
not replace complete-stage time. Verify wall anchors before aligning trace clocks;
measure interval-union intersections and exclude communication from compute.
Missing or unaligned traces remain unknown.

`analyze_encoder_overlap.py` is CPU-only postprocessing and reads selected layouts
from `matrix-plan.json`. Review media/seams, queue/VRAM/latency growth and actual
CUDA overlap before recommending a deployment. CPU tests or platform job success
do not prove GPU numerical, performance or native quality acceptance.

Explicit `--exploratory` retains failed numerical gates while permitting authorized
diagnostic collection. Results remain non-merge-eligible, not validated speedups.
Communication, ownership, completeness and resource safety are never bypassed.

Incremental Decoder and per-dispatch chunk yielding are separate features. This
Encoder PR retains the original behavior and rejects incremental Decoder selection.
Historical experiment details and one-off first-frame diagnostics remain at frozen
Git revisions; they are not maintained benchmark entrypoints.
