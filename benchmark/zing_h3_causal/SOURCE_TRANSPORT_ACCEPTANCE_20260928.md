# Issue 208: bounded high-resolution source transport acceptance

All three aligned resolutions passed the final real-source cloud run, including
three 720-frame formal repeats per size. Exact 1280x720 is **unsupported** by the
current model grid and is rejected before reference decoding or temporary-file
allocation; 1280x736 is a separate size, not a substitute acceptance for 720.

## Tested implementation

Implementation commit: `0bfc1236872dd18a63312800d48bfd4af1ecef34`.
Base main: `4da9d907d76b2a9627e18db7cd05d1d84bcecba0`.
The cloud run used a frozen source bundle before the implementation was committed;
its 5,145 archived source files were checked against the implementation commit.
Runtime code and paired clients are unchanged from that tested bundle. Later
acceptance-report edits and evidence links are documentation-only.

The change replaces the fixed 32 MiB admission limit with a shared, configurable
64 MiB default, retaining bounded queues, source credit and concurrent host-buffer
reservations. It rejects unsupported dimensions before media allocation and
checks transport dependency compatibility before loading GPU workers.
See [bounded source transport](SOURCE_TRANSPORT.md) for deployment, client
negotiation, memory accounting and direct-endpoint error codes.

## Final validation

Local CPU regression: **182 passed**. Final cloud regression: **88 passed**.
Coverage includes exact message boundary, overflow, hidden metadata, concurrent
host budget, pre-allocation grid rejection, real-socket receive queue/backpressure,
and overlapping heartbeat/video flow-control waiters. Formatting and diff checks
passed. Tests use explicit CPU/CUDA platforms respectively.

GPU job `dlcr8ykuivzamibs`: **Succeeded**. Independent CPU verification/export job
`dlcwsqgi2fa8uftd`: **Succeeded**. Thailand `ap-southeast-7`, workspace 384, existing
quota `quota1betfyi1ilu`, one worker with two GPUs; bounded 90-minute GPU run.
Runtime reports two NVIDIA L20D devices with compute capability 10.3; device
properties and nvidia-smi observations are retained rather than inferred from a
quota product label. Image digest:
`sha256:4b9bca7dd61784cd84e43ac760db4c458c08a2d6bf22887e3740455fffe9e015`.
Transport packages are Uvicorn 0.39.0 / websockets 15.0.1, installed from frozen
upstream wheels within repository constraints. Remaining runtime: Python 3.12,
Torch 2.13.0+cu130; actual environment evidence is in the recovered archive.

Each size used a 240-frame warmup followed by three formal 720-frame runs at
24 FPS. SP2, resident weights, four steps, window21/sink3, block-relative RoPE,
TAEH3 FP16 decoder. The launch wrapper selects the H3 FA4 attention path, stable
shapes and compiled AdaLN; the generic server argument remains `torch_sdpa`.
This is configuration/source-path evidence, not a GPU kernel profiler trace.
The server log records successful two-rank NCCL initialization, and both ranks
produce matching terminal source counts. Per run,
submitted = accepted = consumed = received = 720;
output file size, SHA256 and per-rank terminal source records were verified.

| Actual dimensions | Formal elapsed seconds | Median seconds | Max sampled allocated GiB, rank 0 / 1 | Lifetime peak allocated GiB, rank 0 / 1 |
| --- | --- | ---: | --- | --- |
| 832x480 | 91.701, 91.955, 92.080 | 91.955 | 153.62 / 153.66 | 156.07 / 156.11 |
| 896x512 | 98.299, 98.943, 98.800 | 98.800 | 154.41 / 154.19 | 157.39 / 156.91 |
| 1280x736 | 143.655, 144.135, 143.412 | 143.655 | 207.29 / 207.38 | 211.89 / 211.98 |

Elapsed time spans connection/init through `session.completed`, including
paired-client transport and output writes; it excludes connection teardown and
the independent output hash reread. It is not pure GPU latency or browser/WAN
performance. Memory samples come from per-dispatch rank records. Lifetime peak counters are not reset between runs;
reserved memory and nvidia-smi samples are retained in `analysis-r4.json` and
`gpu-memory.jsonl`. Only one active model session was cloud-tested; concurrent
host accounting was regression-tested, not multi-session GPU capacity.

## Source and integrity

Real case `zing-whitebox-video-five-triview-0001`: original source video plus six
reference images, including the global image and five tri-view references. All
seven asset SHA256 values and the prompt match the frozen main case manifest;
`main-case-verification.json` records that independent comparison. Zero-GPU job
`dlcpv4tmyu4xszzl` independently regenerated all three RGB files from the original
video; every hash matched. Its successful receipt was downloaded and SHA256
verified in `source-provenance-r4.json`. No private prompt or media is included
in this report.

720 frames are a sixfold replay of a short real video, not 30 seconds of unique
footage. The resolution-specific RGB files were explicitly resized before the
experiment; the API never silently resizes inputs. Audio was disabled, is absent
from the source RGB transport, and was not generated.

| Actual dimensions | Source RGB SHA256 | Output RGB SHA256 (all three repeats) |
| --- | --- | --- |
| 832x480 | `ce3359a55e17116c136d8847a451b43d717f383da5de868ae3b4036a571cc7bd` | `f2ca735b3aabf7fb16456f99901894f7fb97349c8d05bcd06cdeb4f86ec1a5e9` |
| 896x512 | `02dc0c3ba16b2320f285ca04179492f351f4e4ac3649fad6da729fdedc62ed17` | `cb1bca50836aab199c75abea0efdd8590ced1d6bd6fda94fd092404a6cd02a0b` |
| 1280x736 | `d6a50bf655fc1c81d5aed76458bf04e2392d164974f2ceb1fb60b83a0afae911` | `53a7feec14ad485ce8b53b7f0bc47512d8b4d512fe46070ee1d239d542fb4318` |

Frozen source archive SHA256: `bc9b218e552a4cd698329c2007761bf944f87a2e3f6b92e7061c2cbb9b8c2709`.

Frozen runner/wheels archive SHA256: `163acf77aa979c2b58c94cb88dd0c0754a6263d5245381417bafce489422ff8e`.

Recovered evidence archive SHA256: `f6907f76c8f8ae3370f06900992a43aeda7fd4e7adaafc80f0ce54bd40f78875`; CRC64: `2639282863166844378`; bytes: 938277.

Per-file source hashes are in recovered `run/assets.json` under `freeze`; all
archived evidence files were checked against the manifest after download.
The independent zero-GPU exporter read and hashed all twelve completed RGB
outputs on CPFS. Raw RGB remains there; it was not downloaded locally or made
public. The private OSS archive contains logs, summaries and manifests.

- Cloud raw/results: `/cpfs/world-model/sglang/evals/issue208-independent-20260928-gpu-r4`.
- Frozen source and runner bundles: `oss://leap-worldmodel-thailand-single-az/world-model/code/sglang/issue208-independent-20260928/` (`source-r4.tar.gz`, `tools-r4.tar.gz`, `bundles-r4.json`).
- Private archive: `oss://leap-worldmodel-thailand-single-az/codex-private/issue208-independent-20260928/r4/evidence.tar.gz`.
- Local evidence and reproduction scripts: `/home/ss80194/work/artifacts/issue208-independent`.
- Local recovered evidence: `/home/ss80194/work/artifacts/issue208-independent/recovered-r4`.

## Dependency compatibility and limits

R2 (`dlc17lzdvurpwtms`) failed during the second 1280x736 repeat. Its image had
Uvicorn 0.52.4 / websockets 17.1, outside the repository constraints. Websockets
17.1 removed the legacy drain lock, and overlapping pong/video writes triggered
`_drain_helper` AssertionError. A deterministic CPU reproduction confirmed it;
the supported version and added regression pass. R4 reran the whole matrix with
the supported dependencies. The failed R2 evidence is preserved and is not
counted as a stability pass. Setup attempts and their terminal states are retained
in `job-ledger.json` and the original local `REPORT.md` in the evidence directory.

These results establish the bounded transport loop under the recorded settings.
They do not establish visual quality, other KV policies/capacities, SP1 fit,
exact 1280x720 support, or production multi-session throughput. Output hashes
agree across repeats; that is a reproducibility check, not a quality metric.
