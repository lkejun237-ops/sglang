# TAEH3 provenance

`upstream.py` comes from [madebyollin/taehv](https://github.com/madebyollin/taehv/blob/011dfc2112197741c540e0bdd5b7b67bcc930771/taehv.py),
commit `011dfc2112197741c540e0bdd5b7b67bcc930771`. Only repository formatting
has been applied. The accompanying MIT license is retained.

## Why vendor the upstream module?

The pinned upstream **does** have a `pyproject.toml` and can be installed and
imported as `taehv`; vendoring is a deployment choice, not an import limitation.
This adapter depends on the internal `MemBlock`, `SuperMemBlock`, and `TPool`
classes as well as `TAEHV`. Keeping the exact source alongside the adapter makes
those internals available in the existing SGLang image without adding a Git
dependency or requiring a separate installation. It also keeps the reference
implementation used by parity tests fixed. The tradeoff is maintaining this
copy: upgrades must update the pinned commit, license/provenance and rerun
partition/reference parity. A separately installed dependency pinned to the
same commit is a valid future alternative; an unpinned import is not equivalent.

Download `taeh3.pth` from that same commit separately. It is not stored in Git.
SHA256: `af92965c2d7986a89a757e7cccd26f9eeeff0c3f0d5495eb168aeb2d6d9be9ba`.
Loading rejects a different digest, including a different architecture with the
same filename. No network download occurs during inference.

The adapter takes diffusion-normalized NCTHW latents directly, without native
latent mean/std or ImageNet pixel conversion. It returns RGB NCTHW in [0, 1].
Request-level last-latent extension follows the native H3 frame-count contract;
the reference comparison must apply the identical extension before calling
upstream `decode_video`. This includes the single-frame case.

Each request owns its layer memories and 12-frame output tail. Execution uses
fixed batches of five latents, preserving the previous input of each MemBlock.
Every 20 raw frames lose their first three; EOS zero-pads the final raw group
and discards the final 12 frames, exactly as upstream H3 decoding does.
The generic upstream StreamingTAEHV is deliberately not used.

Set pipeline config `video_decoder_backend=taeh3`, `taeh3_checkpoint_path` to
the downloaded file and `taeh3_decode_precision` to `fp32`, `fp16`, or `bf16`.
Native remains the default. Realtime TAEH3 requires `zing_realtime_decode_mode=stable`.
FP32 decoding disables cuDNN TF32 inside the decoder and restores the caller's
flags afterward; compare upstream FP32 with the same IEEE convolution policy.
Weights are replicated on participating ranks for now; SP is not a TAE speedup
claim. No temporal partitioning or asynchronous overlap is implemented.

Validation status must be read from the experiment report, not inferred from
these configuration options. This is a lossy decoder, and numeric agreement
with the upstream TAE model does not establish native visual quality.

## Reproduce

### Existing Thailand weights

The 2026-09-21 validation used these copies (mounts are environment-specific):

- OSS: `oss://leap-worldmodel-thailand/world-model/code/taeh3/011dfc2112197741c540e0bdd5b7b67bcc930771/taeh3.pth`
- OSS mount: `/mnt/world-model/code/taeh3/011dfc2112197741c540e0bdd5b7b67bcc930771/taeh3.pth`
- Validated CPFS copy: `/cpfs/world-model/sglang/evals/zing-h3-dmd/taeh3-main-20260921-r1/taeh3.pth`

The CPFS path is an experiment asset, not a cross-region default. Use the OSS
copy when provisioning a new mount; loading verifies the SHA256 above.

### Precision selection

On the measured CC 10.3 device, 480p / 240-frame decode medians were FP32
313.274 ms, FP16 66.376 ms and BF16 66.030 ms. Complete streaming chunk p50
was 21.546 / 4.662 / 4.644 ms respectively. BF16's approximately 0.5% advantage
over FP16 in this three-repeat test does not establish a meaningful speedup.
Both matched their same-dtype upstream reference exactly; that is not a
cross-dtype or native-quality claim. Keep the conservative FP32 API default;
use explicit FP16 below to reproduce the full API/streaming matrix. BF16 is
supported, but the complete matrix was not measured with BF16.

Create a pipeline config JSON:

```json
{
  "video_decoder_backend": "taeh3",
  "taeh3_checkpoint_path": "/cpfs/world-model/sglang/evals/zing-h3-dmd/taeh3-main-20260921-r1/taeh3.pth",
  "taeh3_decode_precision": "fp16"
}
```

Pass this file as `--pipeline-config-path` when launching SGLang, or pass the
same dictionary as `pipeline_config` to `DiffGenerator.from_pretrained`.
Defaults are 832×480, 24 FPS; explicit request dimensions are preserved.
The four-step benchmark uses `[1000, 750, 500, 250]`, guidance 1, seed 42.

Decoder-only correctness and timing (CUDA, official SGLang image):

```bash
python benchmark/zing_h3_causal/validate_taeh3_decoder.py \
  --latent /path/to/sglang.pt --weights /path/to/taeh3.pth \
  --frames 240 --output /path/to/decoder-results
```

End-to-end: use `benchmark/zing_h3_causal/validate_real_checkpoint.py sglang`
with `--video-decoder-backend native` or `--video-decoder-backend taeh3`,
`--taeh3-checkpoint-path`, `--taeh3-decode-precision`, `--warmup-runs 1`,
`--measured-runs 3`, and the same explicit request/model/SP configuration.
Profile in a separate run with `--profile-run`.

Sustained streaming through the actual scheduler and RGB/IPC transport:

```bash
python benchmark/zing_h3_causal/benchmark_taeh3_stream.py \
  --model /path/to/zing-ar-export --weights /path/to/taeh3.pth \
  --request benchmark/zing_h3_causal/request_10s.json \
  --frames 1440 --sp-size 4 --dtype fp16 --output /path/to/stream-results
```

This benchmark records a cold 10-second session and a fresh warm 60-second
session, including first-frame latency, per-block p50/p95, buffer trajectory
and playback stalls. It excludes WebSocket/network/browser overhead and
must not be presented as a browser end-to-end measurement.

Add `--transport websocket` to exercise the real HTTP/WebSocket adapter using
the same loaded scheduler. That mode receives full-resolution RGB over loopback,
checks payload sizes and total frames, and tests disconnect/reconnect after the
cold and sustained runs. It includes server framing and socket delivery, but
still excludes WAN latency and browser rendering.
