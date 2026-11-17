# Sparse attention with Block0 on the Mooncake branch

Development baseline: `codex/mooncake-tiered-cache-20261112` at
`002f9f0641e1f32d4709173bc05c6b7683aa6304`. The adaptation lives on
`codex/sparse-block0-mooncake-20261113`. PR #701's prepared-prefix retention is
included. This document describes the candidate; it does not authorize or record
a production rollout.

## Identity and payload

For trusted world-scoped requests on Mooncake, S0 pixels **and native-preview
shape** are excluded from key equality. World, effective prompt, ordered
references, model/precision, SP layout, and actual model/source geometry still
participate. Non-world/local requests retain their existing exact-S0 policy.
S0 provenance remains in the artifact; excluding it from the key does not remove
its conditioning from the cached payload. Cross-S0 reuse therefore needs its own
visual acceptance, separate from same-S0 numerical parity.

The shared Mooncake `execution_identity` includes `zing-compressed-kv-v2` for
three-tier recipes. Older layouts are cold misses, including those made with the
same weight dtype. No SDK input or business cache-version field is added.

Mooncake supports two payload formats selected by the resolved execution recipe:

| Worker recipe | Persistent payload | Hit path |
| --- | --- | --- |
| Three-tier off (including the current Thailand B300 recipe) | `bf16-raw-v1`, canonical raw BF16 KV | Restore raw K/V |
| Three-tier on with layout v2 (Shanghai H200 candidate) | `fp8-residual-v1`, complete resident compressed state | Copy directly into private FP8 stores; no dequantization or requantization |

The FP8 payload includes residuals, block/sub-block means, scales, router
statistics, valid lengths, ownership and shift metadata. A fixed versioned schema
packs one aligned byte buffer per layer. Geometry in the manifest describes the
logical KV; `kv_encoding` and exact `layer_bytes` describe its physical storage.
The wire format is not inferred from dtype or GPU marketing name. It follows the
actual recipe, so a future B300 sparse implementation can use the same codec.

An FP8 namespace adds its codec version to the storage identity, uses manifest
v4, and rejects BF16 or unknown payloads. BF16 retains manifest v3 and can read
older raw manifests without an explicit encoding field. A world may have both
variants, but a request writes only its own variant and never transcodes a hit.
L1/L2/L3 carry the same immutable encoded bytes. Local legacy storage remains raw
BF16, including its existing reconstruction path for sparse consumers.

The native transfer event and compressed-ready event are distinct:
`PrefixRestore.wait()` after commit covers the unpack/copy into private arrays.
Packed restore buffers are released after stream-safe ownership handoff.
Direct FP8 persistence removes restore-time quantization; the per-forward
attention-view repacking described below is still present and must be measured.
Existing native-transfer timing is not relabelled as complete preparation time.

## Three different layouts

1. **Resident history:** FP8 residuals, means and statistics. Every committed
   part starts at a fresh block. Prefix ownership is -1; source block `2u+1`
   and target block `2u+2` both belong to temporal chunk `u`.
2. **Current scratch:** raw current clean K plus current rotated K/V. Its row
   count is independent of full history length. No history-sized bounds are
   passed to this buffer.
3. **Attention view:** a temporary complete layer view, including every current
   noisy row. History is reconstructed from its immutable unrotated FP8 base,
   rotated using the current full position map, then packed into frame-local 2D
   tiles. Partial blocks carry valid lengths. Queries use the same tile order
   and outputs are restored to their original row order. The native all-FP8
   kernel receives `k=None, v=None` and a complete compressed view.

This is a correctness-first adapter for the current kernels API. It performs
per-layer dequantization, RoPE, repacking and fresh routing. It does **not** claim
the constant-cost temporal-mean-shift optimization: arbitrary sink/window maps
cannot be represented by the kernel's single global query shift. Fresh routing
also avoids retaining an older denoising step's current K/V in a cached mask.
Canonical prefix/history branches remain dense; sparse source/target branches
use Refine/Skim/Skip, including 100% Refine protection. An unavailable native
compressed CUDA kernel fails explicitly rather than running a Python fallback.

## Ownership, writes and graphs

Planning copies share GPU allocations but own their counters. Planning clear,
eviction and prefix updates cannot mutate live bytes. Execution compacts at an
owner boundary. Clean writes use frozen destination offsets and device owner
IDs; Python counters are published once after successful execution, not during
graph replay. Capture rollback covers physical scratch rows starting at zero
and every overwritten compressed field, including statistics and ownership.
The graph identity includes compressed block-role classifications, so equal-sized
prompt/sink changes cannot replay an older routing policy. Such changes may
require a new graph warmup/capture; this cost remains part of GPU acceptance.

Raw Block0 exports own snapshots until a writer takes them. Local D2H holds
references until completion. Mooncake registers owned snapshots for transfer,
unregisters after transfer, and does not clone an already-owned snapshot again.
FP8 writers pack directly from the immutable compressed prefix; their preparation
and ordinary cache-write paths do not retain a whole BF16 prefix snapshot. The
optional snapshot budget uses actual encoded bytes and is checked before packing.
The generation cache drops raw exports after handoff or optional-write skips.
Raw-export boundaries execute eagerly; steady-state graph support remains enabled.

The pooled eager path uses the same fixed-forward implementation as graph plans.
This candidate targets the finite-window pooled serving topology; standalone
unpooled compressed generation is not supported by this adapter. Multi-rank
native execution and actual CUDA capture still require hardware validation.

## Deployment inputs

Use `deploy/sglang/examples/dit.sparse-block0-mooncake.json` for the sparse
Shanghai candidate and `deploy/sglang/examples/dit.bf16-block0-mooncake.json`
for the non-sparse Thailand B300 recipe. The latter explicitly pins `dense` and
`THREE_TIER=0`, so a launching shell cannot accidentally enable sparse attention.
The sparse opt-in example pins these required process settings without changing the ordinary DiT
example or any existing two-H200 Slot:

```sh
export SGLANG_ZING_SPARSE_ATTN_MODE=sparse_095
export SGLANG_ZING_SPARSE_THREE_TIER=1
# Native DiT arguments:
# --zing-block0-cache-mode on --zing-block0-cache-backend mooncake
# --zing-block0-mooncake-config /absolute/path/to/verified-cache.json
```

Retain the Mooncake host's existing verified L1/L2/L3 configuration, workload
credentials and regional cache namespace. Replace example asset paths and
receipt hashes with verified inputs. Freeze the source SHA and build a complete
image. The Shanghai dependency base supplied for this work is:

`whitebox-shanghai-registry-vpc.cn-shanghai.cr.aliyuncs.com/whitebox/sglang:v0.5.19-cu130-hub1.33-seedleap-kernels-c9bb3c89@sha256:0433130f16790ecb2b483820d4b947c7b65197261c264e286fa7939b509428e5`

It is a **dependency base**, not a built application image or proof of running
identity. Do not change old Slot settings when selecting this candidate.

## Verification and limits

CPU contracts cover real small-model prefix/source/denoise/final execution and
window movement, source/target ownership, speculative-plan isolation, device-op
write replay and rollback, 480/736 tile bijections and partial blocks, current-V
sensitivity, exact compressed round trips through local memory/disk and the
Mooncake byte-transport double (both BF16 and FP8), a trap proving FP8 hits never call the quantizer, encoding/truncation rejection, owned snapshot lifetime, world/S0 equality and
recipe flag forwarding. The existing Block0 workflow runs the new tests twice
using its pinned Linux recipe. Local Mac tests use a namespace harness to avoid
the unrelated eager pipeline registry; they are not Linux CI evidence.

No native GPU, live Mooncake/RDMA/OSS, browser, SP2/SP4, or 480/736 throughput and
visual acceptance is claimed here. The additional temporary layer workspace,
raw export snapshot, and routing/repacking cost require same-device baseline and
candidate measurements. Report Encoder/DiT/Decoder latency and resident/peak
memory separately, plus TTFF, FPS, TPOC, ICL and long-session quality. Historical
single-H200 results do not certify this candidate. Do not promote it based only
on CPU tests or container health.
