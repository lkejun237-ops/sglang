# SPDX-License-Identifier: Apache-2.0
"""Zing's four/five-modality causal execution over the shared H3 parameter modules.

The checkpoint has separate Q/K/V and value|gate SwiGLU matrices. The H3
loader converts these once; execution uses the same projections, RMSNorms,
time embedding, refiner, MLP and output heads as native H3.
"""

import collections
import dataclasses
import functools
import math
import os
import warnings
from dataclasses import dataclass, field
from functools import lru_cache

import msgspec
import torch
import torch.nn.functional as F

from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm
from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
    resolve_zing_modality_tags,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_group,
    get_tp_world_size,
    get_ulysses_ctx,
)
from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import (
    compute_sequence_splits,
    gather_sequence_varlen,
)
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all_varlen_qkv,
    _usp_output_all_to_all_varlen,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3DiTModel,
    _apply_rope,
    _apply_rope_kv_into,
    _diffusers_h3_checkpoint,
    _fused_qk_norm_rejection,
    _rope_cos_sin_cache,
)
from sglang.multimodal_gen.runtime.models.dits.zing_h3_adaln_cache import (
    ZingH3AdalnCache,
)
from sglang.multimodal_gen.runtime.models.dits.zing_h3_adaln_table import (
    ZingAdalnTable,
    layout_contract,
)
from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
    RouteCoordinator,
    ZingDenoiseGraphs,
)
from sglang.multimodal_gen.runtime.models.dits.zing_h3_joint import (
    PrefixJointTransaction,
    joint_transaction,
    prefix_modulation,
)
from sglang.multimodal_gen.runtime.models.dits.zing_h3_model_signature import (
    model_signatures,
)
from sglang.multimodal_gen.runtime.models.dits.zing_h3_rope import (
    _validated_fixed_device,
    apply_contiguous_history_rope_batch,
    apply_current_qknorm_rope_kv,
    apply_current_rope,
    apply_current_rope_kv,
    apply_history_rope_batch,
    apply_history_rope_into,
    use_current_qknorm_rope_kv,
    validated_fixed_480p,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.profiler import maybe_record_function


@lru_cache(maxsize=1)
def _has_seedleap_kernels() -> bool:
    try:
        import seedleap_kernels.ops as ops

        required_ops = (
            "fused_rmsnorm_indexed_scale_shift_quant_fp8",
            "fused_gated_residual_rmsnorm_indexed_scale_shift_quant_fp8",
            "fused_indexed_gated_residual_add",
            "fused_swiglu_quant_fp8",
        )
        return all(callable(getattr(ops, name, None)) for name in required_ops)
    except (ImportError, ModuleNotFoundError, AttributeError, ValueError):
        return False


def storage_masks(roles):
    return (
        (roles == 0) | (roles == 1),
        (roles == 4) | (roles == 7) | (roles == 10),
        (roles == 5) | (roles == 11),
    )


def host_to_device(tensor, device):
    """Copy a host-built tensor without a pageable-copy stream synchronization.

    The pinned staging block comes from the caching host allocator, which keeps
    it alive until the asynchronous copy has completed. An already pinned source
    is not staged, so it must not be modified until the copy completes.
    """
    if tensor.device == device:
        return tensor
    if device.type != "cuda" or tensor.device.type != "cpu":
        return tensor.to(device)
    return tensor.pin_memory().to(device, non_blocking=True)


@dataclass(eq=False)
class ZingHostLayout:
    """CPU copies of one packed selection's row metadata.

    The sampler builds these once per selected block and passes the same
    object for every forward over that block, together with the per-step CPU
    ``t``. The forward validates and derives row indices from them on the
    host, so repeated denoising forwards read no device metadata back.
    The device tensors passed alongside must hold the same values.
    """

    seq_lens: torch.Tensor
    clean_seq_lens: torch.Tensor
    block_idx: torch.Tensor
    position_ids: torch.Tensor
    row_role: torch.Tensor
    # Device row indices derived from this layout, keyed by SP/shape policy.
    derived: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _RowLayout:
    """One forward's row selections as index tensors on the compute device."""

    sequence_splits: list | None
    row_start: int
    row_stop: int
    # Positions within this rank's hidden rows.
    local_crows: torch.Tensor
    local_vrows: torch.Tensor
    local_arows: torch.Tensor
    # Compact storage rows feeding the local positions above.
    context_source: torch.Tensor
    video_source: torch.Tensor
    audio_source: torch.Tensor
    # Positions within the complete packed sequence.
    vrows: torch.Tensor
    arows: torch.Tensor
    # Per-row AdaLN modality on the metadata device.
    modality: torch.Tensor


class _DeviceFA4Metadata:
    """FA4 lengths built by device kernels, so a graph capture copies nothing in."""

    def lengths(self, q, k, v):
        return (
            torch.arange(2, dtype=torch.int32, device=q.device) * q.shape[0],
            torch.arange(2, dtype=torch.int32, device=k.device) * k.shape[0],
        )


class _BlockOptions(msgspec.Struct, frozen=True, kw_only=True):
    """Host decisions one forward's kernels follow."""

    capture: bool
    use_cache: bool
    raw_attention: bool
    replicated_prefix: bool
    staged_prefix: bool
    ulysses_rank: int
    prefix_heads: int
    history_plan: object
    rotate_batch: object
    projection_cache: object
    projection_key: object
    joint_ends: tuple | None = None
    branch_projection_keys: object = None
    projection_group_sizes: tuple | None = None
    joint_video_ends: tuple | None = None
    fixed_kv: object = None
    fixed_tensors: object = None
    full_rope: object = None
    optimized_fixed_480p: bool = False
    target_fixed_736p: bool = False


class _BodyInputs(msgspec.Struct, frozen=True, kw_only=True):
    """Device tensors one forward's kernels read, plus their row metadata."""

    x: torch.Tensor
    audio_x: torch.Tensor
    context: torch.Tensor
    layout: _RowLayout
    adaln_input: torch.Tensor
    indices: torch.Tensor
    local_t_indices: torch.Tensor
    rope: torch.Tensor
    seq_lens: torch.Tensor
    rows: int


# Inputs a denoise graph reads by address; refreshed before every replay.
_GRAPH_INPUTS = ("x", "audio_x", "context", "indices", "local_t_indices", "rope")
_GRAPH_LAYOUT_INPUTS = (
    "local_crows",
    "local_vrows",
    "local_arows",
    "context_source",
    "video_source",
    "audio_source",
    "vrows",
    "arows",
)
# Kernel routes chosen from the environment at call time, frozen by a capture.
_GRAPH_ROUTE_ENVS = (
    "SGLANG_ZING_SP_DESTINATION_MAJOR",
    "SGLANG_ZING_FUSED_QKNORM_ROPE_KV",
    "SGLANG_ZING_HISTORY_ROPE_LIVE_ROWS",
    "SGLANG_ZING_HISTORY_ROPE_CROSS_GRAPH",
    "SGLANG_ZING_FUSED_CURRENT_ROPE",
    "SGLANG_ZING_FUSED_CURRENT_ROPE_KV",
    "SGLANG_ZING_FA4_FIXED",
    "SGLANG_ZING_FUSED_HISTORY_ROPE",
    "SGLANG_ZING_GROUPED_HISTORY_ROPE",
)


def _projection_block(projection):
    """The dense [rows, parts * width] span its 2-D parts tile in order, or None."""
    first = projection[0]
    if first.dim() != 2:
        return None
    width = first.shape[1]
    stride = (len(projection) * width, 1)
    storage = first.untyped_storage().data_ptr()
    for part, value in enumerate(projection):
        if (
            value.shape != first.shape
            or value.stride() != stride
            or value.dtype != first.dtype
            or value.storage_offset() != first.storage_offset() + part * width
            or value.untyped_storage().data_ptr() != storage
        ):
            return None
    return first.as_strided((first.shape[0], stride[0]), stride)


def _graph_tensors(inputs, projections):
    tensors = {name: getattr(inputs, name) for name in _GRAPH_INPUTS}
    for name in _GRAPH_LAYOUT_INPUTS:
        tensors["layout." + name] = getattr(inputs.layout, name)
    # AdaLN projections are copied by value: every denoise step of a chunk
    # shares one graph although its projections live at other addresses.
    for slot, projection in enumerate(projections):
        block = _projection_block(projection)
        if block is not None:
            tensors[f"projection.{slot}"] = block
            continue
        for part, value in enumerate(projection):
            tensors[f"projection.{slot}.{part}"] = value
    return tensors


def _static_projections(static, projections):
    """Static parts with the strides and offsets of the live projections."""
    result = []
    for slot, projection in enumerate(projections):
        block = static.get(f"projection.{slot}")
        if block is None:
            parts = (static[f"projection.{slot}.{p}"] for p in range(len(projection)))
        else:
            width = block.shape[1] // len(projection)
            parts = (
                block[:, p * width : (p + 1) * width] for p in range(len(projection))
            )
        result.append(tuple(parts))
    return result


def _static_inputs(inputs, static):
    layout = dataclasses.replace(
        inputs.layout,
        **{name: static["layout." + name] for name in _GRAPH_LAYOUT_INPUTS},
    )
    return msgspec.structs.replace(
        inputs, layout=layout, **{name: static[name] for name in _GRAPH_INPUTS}
    )


def _tensor_layout(tensor):
    return tuple(tensor.shape), tensor.stride(), tensor.dtype, tensor.device


def _numerics_mode():
    """Global switches that change which kernels a capture records."""
    matmul = torch.backends.cuda.matmul
    return (
        matmul.allow_tf32,
        matmul.allow_fp16_reduced_precision_reduction,
        matmul.allow_bf16_reduced_precision_reduction,
        torch.are_deterministic_algorithms_enabled(),
    )


def _sequence_parallel_topology_rejection(*, enabled):
    """Why a sequence-parallel topology can never use denoise graphs, or None.

    The answer depends only on process-wide configuration, so every rank agrees.
    """
    if not enabled:
        return "sequence_parallel"
    if get_sp_group().ring_world_size != 1:
        return "sp_topology"
    if get_tp_world_size() != 1:
        return "sp_topology"
    return None


def _has_forward_hooks(model):
    """Hooks run Python a replay skips, e.g. layerwise offload swapping weights."""
    module = torch.nn.modules.module
    if module._global_forward_pre_hooks or module._global_forward_hooks:
        return True
    return any(m._forward_pre_hooks or m._forward_hooks for m in model.modules())


def causal_mask(blocks, clean_count):
    """Single-document [clean history | noisy query] training/reference mask."""
    clean = torch.arange(blocks.numel(), device=blocks.device) < clean_count
    qb, kb = blocks[:, None], blocks[None, :]
    qc, kc = clean[:, None], clean[None, :]
    return (qb >= kb) & ((qc & kc) | (~qc & kc & (qb > kb)) | (~qc & ~kc & (qb == kb)))


def _is_zing_main_block(name, _module):
    parts = name.split(".")
    return len(parts) == 2 and parts[0] == "blocks" and parts[1].isdigit()


def _rms_norm_modulate(hidden, weight, shift, scale, eps):
    return (
        F.rms_norm(hidden, (hidden.shape[-1],), weight, eps=eps) * (1.0 + scale) + shift
    )


@functools.cache
def _compiled_rms_norm_modulate(deterministic):
    kwargs = {"options": {"deterministic": True}} if deterministic else {}
    return torch.compile(_rms_norm_modulate, dynamic=True, **kwargs)


def _native_compiled_adaln(hidden, norm, shift, scale):
    """Native CUDA fusion, including its singleton batch dimension."""
    if not hidden.is_cuda:
        raise ValueError("Zing experimental compiled AdaLN requires CUDA tensors")
    return _compiled_rms_norm_modulate(torch.are_deterministic_algorithms_enabled())(
        hidden.unsqueeze(0), norm.weight, shift, scale, float(norm.eps)
    ).squeeze(0)


# Per-process fused QK-norm route counts: "fused" or the rejection reason.
FUSED_QK_NORM_ROUTES = collections.Counter()


def _qk_norm(module, q, k):
    """Normalize Q/K; the fused route writes in place into the QKV views."""
    reason = _fused_qk_norm_rejection(
        q, k, module.q_norm, module.k_norm, module.head_dim
    )
    FUSED_QK_NORM_ROUTES[reason or "fused"] += 1
    if reason is None:
        fused_inplace_qknorm(
            q,
            k,
            module.q_norm.weight,
            module.k_norm.weight,
            eps=module.q_norm.eps,
            head_dim=module.head_dim,
        )
        return q, k
    return module.q_norm(q), module.k_norm(k)


def _shares_storage(a, b):
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def _exchange_qkv(q, k, v, splits):
    """Use the same input SP layout in eager and fixed Graph attention."""
    if os.environ.get("SGLANG_ZING_SP_DESTINATION_MAJOR", "1") == "1":
        return _usp_input_all_to_all_varlen_qkv(
            q[None], k[None], v[None], splits, destination_major=True
        )
    return _usp_input_all_to_all_varlen_qkv(
        q[None].contiguous(), k[None].contiguous(), v[None].contiguous(), splits
    )


def _fa4_current_stream(device):
    """The compute stream that owns one forward's FA4 length metadata."""
    return torch.get_device_module(device).current_stream(device)


class _FA4ForwardMetadata:
    """Forward-local, read-only FA4 varlen length tensors.

    Same-stream reuse is safe because the pair is created and consumed on one
    stream, so the caching allocator's stream-ordered reclamation covers the
    asynchronous kernel. That is also why a second stream is rejected here
    rather than served: an event plus a wait would not by itself keep the pair
    alive for the consumer stream.
    """

    def __init__(self):
        self._streams = {}
        self._pairs = {}
        self._closed = False
        self.pairs_created = 0
        self.tensors_created = 0
        self.hits = 0

    @property
    def closed(self):
        return self._closed

    def keys(self):
        return tuple(self._pairs)

    def lengths(self, q, k, v):
        if self._closed:
            raise RuntimeError("Zing FA4 forward metadata is already released")
        device = q.device
        if k.device != device or v.device != device:
            raise ValueError("Zing FA4 forward metadata requires Q/K/V on one device")
        stream = _fa4_current_stream(device)
        owner = self._streams.get(device)
        if owner is None:
            self._streams[device] = stream
        elif owner != stream:
            raise RuntimeError(
                "Zing FA4 forward metadata cannot be shared across compute streams"
            )
        # The device is part of the key: the pair's tensors live on it, and the
        # stream guard above only covers stream identity.
        q_rows, k_rows = q.shape[0], k.shape[0]
        key = (q_rows, k_rows, device)
        pair = self._pairs.get(key)
        if pair is not None:
            self.hits += 1
            return pair
        pair = _fa4_cu_seqlens(q_rows, k_rows, q.device, k.device)
        self._pairs[key] = pair
        self.pairs_created += 1
        self.tensors_created += 2
        return pair

    def clear(self):
        self._pairs.clear()
        self._streams.clear()
        self._closed = True


def _fa4_cu_seqlens(q_rows, k_rows, q_device, k_device):
    return (
        host_to_device(torch.tensor([0, q_rows], dtype=torch.int32), q_device),
        host_to_device(torch.tensor([0, k_rows], dtype=torch.int32), k_device),
    )


@torch.library.custom_op(
    "sglang::zing_cpu_varlen_attention", mutates_args=(), device_types="cpu"
)
def _cpu_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """CPU reference for FA4's device-length contract and fixed output shape.

    The recording backend replays this as one operation, reading the current
    length on every invocation, just as a CUDA varlen kernel does internally.
    Masked padding in CPU SDPA changes its reduction/rounding path.
    """
    length = int(cu_seqlens_k[-1])
    return F.scaled_dot_product_attention(
        q.transpose(0, 1)[None],
        k[:length].transpose(0, 1)[None],
        v[:length].transpose(0, 1)[None],
        scale=scale,
    )[0].transpose(0, 1)


@_cpu_varlen_attention.register_fake
def _cpu_varlen_attention_fake(q, k, v, cu_seqlens_k, scale):
    return torch.empty_like(q)


def _fa4_attention(q, k, v, scale, metadata=None):
    """Native FA4 route for one unmasked document."""
    if not q.is_cuda:
        raise ValueError("Zing experimental FA4 attention requires CUDA tensors")
    try:
        from flash_attn.cute import flash_attn_varlen_func
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "Zing experimental FA4 requires flash_attn.cute.flash_attn_varlen_func"
        ) from error
    q_rows, k_rows = q.shape[0], k.shape[0]
    if metadata is None:
        cu_seqlens_q, cu_seqlens_k = _fa4_cu_seqlens(q_rows, k_rows, q.device, k.device)
    else:
        cu_seqlens_q, cu_seqlens_k = metadata.lengths(q, k, v)
    result = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=q_rows,
        max_seqlen_k=k_rows,
        softmax_scale=scale,
        causal=False,
        window_size=(None, None),
        deterministic=False,
        return_lse=False,
    )
    if isinstance(result, tuple):
        if not result:
            raise RuntimeError("Zing experimental FA4 returned an empty result tuple")
        result = result[0]
    if not isinstance(result, torch.Tensor) or result.shape != q.shape:
        raise RuntimeError("Zing experimental FA4 returned an invalid output shape")
    return result


def _fixed_fa4_attention(q, k, v, scale, metadata, *, optimized, target_736p=False):
    """Keep legacy 480 routes; extend fixed Q only for measured 736 targets.

    The slot capacity is not its live length. Pass cu_seqlens_k unchanged so
    source/step0 cannot read future branches or stale slot rows on replay.
    """
    if (
        (
            optimized
            or (
                target_736p
                and q.is_cuda
                and q.shape[1:] == (28, 128)
                and k.shape == v.shape == (44156, 28, 128)
                and _validated_fixed_device(q.device)
            )
        )
        and os.environ.get("SGLANG_ZING_FA4_FIXED", "1") == "1"
        and q.shape[0] in ((390, 780, 920, 1840, 2760) if target_736p else (390, 780))
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and all(x.is_contiguous() for x in (q, k, v))
    ):
        from flash_attn.cute.interface import _flash_attn_fwd

        result = _flash_attn_fwd(
            q[None],
            k,
            v,
            cu_seqlens_k=metadata.lengths(q, k, v)[1],
            max_seqlen_k=k.shape[0],
            softmax_scale=scale,
            causal=False,
            num_splits=1,
        )[0]
        return result.squeeze(0)
    return _fa4_attention(q, k, v, scale, metadata=metadata)


def _joint_attention(q, k, v, ends, scale, backend, metadata):
    """Block-bidirectional branch attention; no token-level causal mask."""
    history_rows = k.shape[0] - ends[-1]
    result, start = [], 0
    for stop in ends:
        qs, ks, vs = q[start:stop], k[: history_rows + stop], v[: history_rows + stop]
        with maybe_record_function("Zing.FA4" if backend == "fa4" else "Zing.SDPA"):
            if backend == "fa4":
                value = _fa4_attention(qs, ks, vs, scale, metadata=metadata)
            else:
                value = F.scaled_dot_product_attention(
                    qs.transpose(0, 1)[None],
                    ks.transpose(0, 1)[None],
                    vs.transpose(0, 1)[None],
                    dropout_p=0.0,
                    is_causal=False,
                    scale=scale,
                )[0].transpose(0, 1)
        result.append(value)
        start = stop
    return torch.cat(result)


def _attention(
    module,
    hidden,
    rope=None,
    history=None,
    history_rope=None,
    mask=None,
    *,
    capture=False,
    sequence_splits=None,
    attention_backend="torch_sdpa",
    append_kv=None,
    history_is_rotated=False,
    capture_rotated=False,
    append_unrotated=False,
    capture_stored_values=False,
    stage_raw_k=None,
    stage_prefix_kv=None,
    fa4_metadata=None,
    joint_ends=None,
):
    if attention_backend not in {"torch_sdpa", "fa4"}:
        raise ValueError(
            f"Unsupported Zing experimental attention: {attention_backend}"
        )
    if attention_backend == "fa4" and mask is not None:
        raise ValueError(
            "Zing experimental FA4 does not support explicit attention masks"
        )
    hidden_tensor = (
        (hidden[0] if isinstance(hidden, tuple) else hidden)
        if hidden is not None
        else None
    )
    total_rows = hidden_tensor.shape[0] if hidden_tensor is not None else 0
    qkv = module.qkv_proj(hidden)[0]
    q, k, v = (
        x.reshape(-1, module.num_heads, module.head_dim) for x in qkv.chunk(3, -1)
    )
    with maybe_record_function("Zing.QK_norm"):
        q, k = _qk_norm(module, q, k)
    if sequence_splits is not None:
        with maybe_record_function("Zing.Ulysses.input_all_to_all"):
            qkv = _exchange_qkv(q, k, v, sequence_splits)
        q, k, v = qkv.chunk(3, dim=-1)
        # Ulysses converts local sequence/all heads into complete sequence/local
        # heads. Cache that layout directly so every rank owns only its head
        # shard and no K/V all-gather is needed after a commit.
        # With a raw-store stager, the unrotated K is written into the store
        # tail after append_kv reserved it, instead of a pool-external clone.
        # all-to-all output never shares QKV storage, so stage unconditionally.
        stage_k = capture and capture_stored_values and stage_raw_k is not None
        current_k = k[0]
        raw = (
            (
                None if stage_k else k[0].clone(),
                None if capture_stored_values else v[0].clone(),
            )
            if capture and not capture_rotated
            else None
        )
        with maybe_record_function("Zing.RoPE_and_KV_append"):
            if rope is not None:
                with maybe_record_function("Zing.current_QK_RoPE"):
                    q = apply_current_rope(q, rope)
                    # The single store rotates K straight into its scratch tail.
                    if not (capture_rotated or append_unrotated):
                        k = apply_current_rope(k, rope)
            if append_kv is not None:
                rows = k.shape[1]
                with maybe_record_function("Zing.cached_KV_append"):
                    k, v = append_kv(k[0], v[0])
                    k, v = k[None], v[None]
                # The store holds what this forward rotated, so the capture
                # borrows those rows and the commit copies nothing.
                if capture and capture_rotated:
                    raw = (k[0, -rows:], v[0, -rows:])
                    if v.shape[1] == rows:
                        raw = (raw[0], raw[1].clone())  # no history: still a QKV view
                elif capture and capture_stored_values:
                    raw = (
                        stage_raw_k(current_k) if stage_k else raw[0],
                        v[0, -rows:],
                    )
            elif history is not None:
                with maybe_record_function("Zing.history_K_RoPE"):
                    history_k = (
                        history[0]
                        if history_is_rotated
                        else _apply_rope(history[0], history_rope)
                    )
                with maybe_record_function("Zing.history_current_KV_concat"):
                    k = torch.cat((history_k[None], k), dim=1)
                    v = torch.cat((history[1][None], v), dim=1)
        if joint_ends is not None:
            result = _joint_attention(
                q[0],
                k[0],
                v[0],
                joint_ends,
                module.softmax_scale,
                attention_backend,
                fa4_metadata,
            )[None]
        elif attention_backend == "fa4":
            with maybe_record_function("Zing.FA4"):
                result = _fa4_attention(
                    q[0], k[0], v[0], module.softmax_scale, metadata=fa4_metadata
                )[None]
        else:
            with maybe_record_function("Zing.SDPA"):
                result = F.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    attn_mask=mask,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=module.softmax_scale,
                ).transpose(1, 2)
        with maybe_record_function("Zing.Ulysses.output_all_to_all"):
            result = _usp_output_all_to_all_varlen(result, sequence_splits, head_dim=2)[
                0
            ].reshape(total_rows, module.num_heads * module.head_dim)
        return module.out_proj(result)[0], raw
    # V is a view into fused QKV. Persist only V, not the entire projection.
    if capture and stage_prefix_kv is not None:
        # Replicated SP prefix: keep only this rank's head shard, written
        # straight into the store before RoPE touches K.
        raw = stage_prefix_kv(k, v)
    else:
        raw = (
            (k, None if capture_stored_values else v.clone())
            if capture and not capture_rotated
            else None
        )
    with maybe_record_function("Zing.RoPE_and_KV_append"):
        if rope is not None:
            with maybe_record_function("Zing.current_QK_RoPE"):
                q = apply_current_rope(q, rope)
                if not (capture_rotated or append_unrotated):
                    k = apply_current_rope(k, rope)
        if append_kv is not None:
            rows = k.shape[0]
            with maybe_record_function("Zing.cached_KV_append"):
                k, v = append_kv(k, v)
            if capture and capture_rotated:
                raw = (k[-rows:], v[-rows:])
                if v.shape[0] == rows:
                    raw = (raw[0], raw[1].clone())  # no history: still a QKV view
            elif capture and capture_stored_values:
                raw = (
                    (
                        stage_raw_k(raw[0])
                        if stage_raw_k is not None and _shares_storage(raw[0], qkv)
                        else raw[0]
                    ),
                    v[-rows:],
                )
        elif history is not None:
            with maybe_record_function("Zing.history_K_RoPE"):
                history_k = (
                    history[0]
                    if history_is_rotated
                    else _apply_rope(history[0], history_rope)
                )
            with maybe_record_function("Zing.history_current_KV_concat"):
                k = torch.cat((history_k, k))
                v = torch.cat((history[1], v))
    if raw is not None and _shares_storage(raw[0], qkv):
        # In-place QK norm leaves K as a QKV view; a capture held until the
        # commit must not keep the whole projection alive.
        raw = (raw[0].clone(), raw[1])
    if joint_ends is not None:
        result = _joint_attention(
            q, k, v, joint_ends, module.softmax_scale, attention_backend, fa4_metadata
        ).reshape(total_rows, -1)
    elif attention_backend == "fa4":
        with maybe_record_function("Zing.FA4"):
            result = _fa4_attention(
                q, k, v, module.softmax_scale, metadata=fa4_metadata
            ).reshape(total_rows, module.num_heads * module.head_dim)
    else:
        with maybe_record_function("Zing.SDPA"):
            result = (
                F.scaled_dot_product_attention(
                    q.transpose(0, 1).unsqueeze(0),
                    k.transpose(0, 1).unsqueeze(0),
                    v.transpose(0, 1).unsqueeze(0),
                    attn_mask=mask,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=module.softmax_scale,
                )[0]
                .transpose(0, 1)
                .reshape(total_rows, -1)
            )
    return module.out_proj(result)[0], raw


@dataclass
class FixedForwardPlan:
    """Host-frozen routes and replaceable device inputs for a complete DiT call."""

    tensors: dict
    signature: tuple
    kv_plan: object
    _execute: object

    @torch.no_grad()
    def execute(self, tensors=None):
        return self._execute(self.tensors if tensors is None else tensors)


class _FixedFA4Metadata:
    def __init__(self, tensors, branch):
        self.tensors, self.branch = tensors, branch

    def lengths(self, q, k, v):
        return self.tensors[f"cu_q.{self.branch}"], self.tensors[f"cu_k.{self.branch}"]


class ZingH3DiTModel(MiniMaxH3DiTModel):
    _aliases = []
    # Shard only the main DiT blocks. The token-refiner blocks are driven by
    # their inner modules below, so they must remain in the root FSDP group.
    _fsdp_shard_conditions = [_is_zing_main_block]
    _fsdp_forward_methods = ()
    _compile_conditions = []
    _supported_attention_backends = {AttentionBackendEnum.TORCH_SDPA}

    def __init__(
        self, config, hf_config, quant_config=None, *, adaln_table=None, **kwargs
    ):
        experimental_attention = os.environ.get(
            "SGLANG_ZING_EXPERIMENTAL_ATTENTION", "fa4"
        )
        if experimental_attention not in {"torch_sdpa", "fa4"}:
            raise ValueError(
                "SGLANG_ZING_EXPERIMENTAL_ATTENTION must be torch_sdpa or fa4"
            )
        stable_shapes = os.environ.get("SGLANG_ZING_EXPERIMENTAL_STABLE_SHAPES", "1")
        if stable_shapes not in {"0", "1"}:
            raise ValueError("SGLANG_ZING_EXPERIMENTAL_STABLE_SHAPES must be 0 or 1")
        compiled_adaln = os.environ.get("SGLANG_ZING_EXPERIMENTAL_COMPILED_ADALN", "1")
        if compiled_adaln not in {"0", "1"}:
            raise ValueError("SGLANG_ZING_EXPERIMENTAL_COMPILED_ADALN must be 0 or 1")
        projection_cache = os.environ.get("SGLANG_ZING_ADALN_PROJECTION_CACHE", "1")
        if projection_cache not in {"0", "1"}:
            raise ValueError("SGLANG_ZING_ADALN_PROJECTION_CACHE must be 0 or 1")
        projection_cache_bytes = int(
            os.environ.get("SGLANG_ZING_ADALN_PROJECTION_CACHE_BYTES", "134217728")
        )
        if projection_cache_bytes < 0:
            raise ValueError("Zing AdaLN projection cache budget must be nonnegative")
        fa4_metadata_reuse = os.environ.get(
            "SGLANG_ZING_EXPERIMENTAL_FA4_METADATA_REUSE", "1"
        )
        if fa4_metadata_reuse not in {"0", "1"}:
            raise ValueError(
                "SGLANG_ZING_EXPERIMENTAL_FA4_METADATA_REUSE must be 0 or 1"
            )
        denoise_graph = os.environ.get("SGLANG_ZING_DENOISE_CUDA_GRAPH", "1")
        if denoise_graph not in {"0", "1"}:
            raise ValueError("SGLANG_ZING_DENOISE_CUDA_GRAPH must be 0 or 1")
        denoise_graph_sp = os.environ.get("SGLANG_ZING_DENOISE_CUDA_GRAPH_SP", "1")
        if denoise_graph_sp not in {"0", "1"}:
            raise ValueError("SGLANG_ZING_DENOISE_CUDA_GRAPH_SP must be 0 or 1")
        denoise_graph_budget = os.environ.get(
            "SGLANG_ZING_DENOISE_CUDA_GRAPH_MAX", "32"
        )
        try:
            denoise_graph_budget = int(denoise_graph_budget)
        except ValueError:
            raise ValueError(
                "SGLANG_ZING_DENOISE_CUDA_GRAPH_MAX must be an integer, "
                f"got {denoise_graph_budget!r}"
            ) from None
        if denoise_graph_budget < 1:
            raise ValueError("SGLANG_ZING_DENOISE_CUDA_GRAPH_MAX must be positive")
        for retired in (
            "SGLANG_ZING_EXPERIMENTAL_CACHED_ROPE",
            "SGLANG_ZING_EXPERIMENTAL_KV_APPEND",
            "SGLANG_ZING_EXPERIMENTAL_KV_SINGLE_STORE",
            "SGLANG_ZING_EXPERIMENTAL_FUSED_ROPE_KV_WRITE",
        ):
            if os.environ.get(retired) is not None:
                warnings.warn(
                    f"{retired} is ignored; Zing always uses the store-backed KV path",
                    DeprecationWarning,
                    stacklevel=2,
                )
        arch = config.arch_config
        self.modality_tags = resolve_zing_modality_tags(
            arch.num_modalities, arch.modality_tags
        )
        if arch.adaln_out_features != 6 * arch.num_modalities * arch.hidden_size:
            raise ValueError("Zing AdaLN width must match the trained modality count")
        if arch.final_adaln_out_features != 2 * arch.hidden_size:
            raise ValueError("Zing final AdaLN width must be 2 * hidden_size")
        for value in (arch.ar_condition_timestep, arch.reference_visual_timestep):
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(
                    "Zing condition timesteps must be finite and within [0, 1]"
                )
        if (
            arch.adaln_curve_grid is not None
            or arch.adaln_affine_input_dim is not None
            or kwargs
        ):
            raise ValueError(
                "Zing inference requires non-curve weights and supported constructor arguments"
            )
        if adaln_table is not None and (
            not isinstance(adaln_table, ZingAdalnTable)
            or adaln_table.manifest["layout"]
            != layout_contract(arch, self.modality_tags)
        ):
            raise ValueError(
                "Zing AdaLN table does not match the worker projection layout"
            )
        if quant_config is not None:
            from sglang.multimodal_gen.runtime.models.dits.zing_h3_quant import (
                ZingLinearQuantConfig,
            )

            if not isinstance(quant_config, ZingLinearQuantConfig):
                quant_config = ZingLinearQuantConfig(
                    quant_config, num_layers=arch.num_layers
                )

        arch.checkpoint_uses_diffusers_layout = True
        super().__init__(
            config,
            hf_config,
            quant_config=quant_config,
            omit_adaln_projections=adaln_table is not None,
        )
        self.quant_config = quant_config
        self.adaln_table = adaln_table
        self.experimental_attention_backend = experimental_attention
        self.experimental_stable_shapes = stable_shapes == "1"
        self.experimental_compiled_adaln = compiled_adaln == "1"
        self.adaln_projection_cache_enabled = projection_cache == "1"
        self.adaln_projection_cache_bytes = projection_cache_bytes
        self.experimental_fa4_metadata_reuse = (
            fa4_metadata_reuse == "1" and experimental_attention == "fa4"
        )
        # The text refiner is small enough to remain resident. Main DiT blocks
        # execute through Module.__call__, so the standard layerwise-offload
        # hooks can stream them on memory-constrained GPUs.
        self.layer_names = ["blocks"]
        self.preprocess_loaded_state_dict = self._preprocess_checkpoint
        self._cache_weight_generation = 0
        self.denoise_graphs = (
            ZingDenoiseGraphs(max_graphs=denoise_graph_budget)
            if denoise_graph == "1"
            else None
        )
        self.denoise_graph_sp = denoise_graph_sp == "1"
        self.inplace_residual = (
            os.environ.get("SGLANG_ZING_INPLACE_RESIDUAL", "1") == "1"
        )

    def _should_fuse_pre_linear_quant(self, value: torch.Tensor) -> bool:
        if os.environ.get("SGLANG_ZING_FUSED_PRE_LINEAR_QUANT", "1") == "0":
            return False
        if not value.is_cuda:
            return False
        if value.dtype != torch.bfloat16:
            return False
        if value.requires_grad:
            return False
        if value.shape[-1] > 8192:
            return False
        return _has_seedleap_kernels()

    def _project_outputs(
        self, hidden, p, layout, *, joint_ends=None, mixed_prefix=False
    ):
        """Project video/audio rows and restore global packed order across SP.

        Stable shapes gather hidden rows before projection to match SP1 GEMM
        shapes. Otherwise, project local rows and gather packed logits once.
        """
        sequence_splits = layout.sequence_splits
        replicated_output = (
            self.experimental_stable_shapes and sequence_splits is not None
        )
        output_vrows, output_arows = layout.local_vrows, layout.local_arows
        if replicated_output:
            if mixed_prefix:
                # One communicator and graph-owned buffers for equal/unequal SP shards.
                maximum = max(sequence_splits)
                if hidden.shape[0] < maximum:
                    hidden = torch.cat(
                        (
                            hidden,
                            hidden.new_zeros(
                                (maximum - hidden.shape[0], hidden.shape[1])
                            ),
                        )
                    )
                parts = [torch.empty_like(hidden) for _ in sequence_splits]
                torch.distributed.all_gather(
                    parts, hidden.contiguous(), group=get_sp_group().ulysses_group
                )
                hidden = torch.cat(
                    [part[:size] for part, size in zip(parts, sequence_splits)]
                )
            else:
                hidden = gather_sequence_varlen(
                    hidden, sequence_splits, get_sp_group().ulysses_group, dim=0
                )
            output_vrows, output_arows = layout.vrows, layout.arows
        video_hidden = hidden.index_select(0, output_vrows).float()
        if joint_ends is None:
            video_logits = self.final_layer.video_out(video_hidden)[0]
        else:
            # FP32 output-head GEMMs can change rounding with M. Tiny step0
            # differences feed subsequent BF16 denoising steps, so preserve
            # each sequential branch's projection shape. Joint transactions
            # contain only video rows; the transformer body remains joint.
            ends = joint_ends
            if sequence_splits is not None and not replicated_output:
                ends = tuple(
                    max(0, min(stop - layout.row_start, hidden.shape[0]))
                    for stop in ends
                )
            starts = (0, *ends[:-1])
            video_logits = torch.cat(
                [
                    self.final_layer.video_out(video_hidden[start:stop])[0]
                    for start, stop in zip(starts, ends)
                ]
            )
        video = video_logits.reshape(-1, self.arch.latents_dim, p).permute(0, 2, 1)
        audio = self.final_layer.audio_out(
            hidden.index_select(0, output_arows).float()
        )[0]
        if sequence_splits is not None and not replicated_output:
            video_width = p * self.arch.latents_dim
            audio_width = self.arch.audio_latents_dim
            local_logits = hidden.new_zeros(
                (hidden.shape[0], video_width + audio_width), dtype=video.dtype
            )
            local_logits[:, :video_width].index_copy_(
                0, layout.local_vrows, video.reshape(-1, video_width)
            )
            local_logits[:, video_width:].index_copy_(
                0, layout.local_arows, audio.to(local_logits.dtype)
            )
            logits = gather_sequence_varlen(
                local_logits,
                sequence_splits,
                get_sp_group().ulysses_group,
                dim=0,
            )
            video = (
                logits[:, :video_width]
                .index_select(0, layout.vrows)
                .reshape(-1, p, self.arch.latents_dim)
            )
            audio = logits[:, video_width:].index_select(0, layout.arows)
        return video, audio

    def _row_layout(
        self,
        row_role,
        host_layout,
        rows,
        ulysses_size,
        ulysses_rank,
        replicated_prefix,
        device,
    ):
        """Row selections for this rank; host layouts cache their device copies."""
        key = (
            ulysses_size,
            ulysses_rank,
            replicated_prefix,
            device,
            tuple(sorted(self.modality_tags.items())),
        )
        if host_layout is not None and key in host_layout.derived:
            return host_layout.derived[key]
        sequence_splits = (
            None
            if ulysses_size == 1 or replicated_prefix
            else compute_sequence_splits(rows, ulysses_size)
        )
        row_start = (
            0 if sequence_splits is None else sum(sequence_splits[:ulysses_rank])
        )
        row_stop = (
            rows
            if sequence_splits is None
            else row_start + sequence_splits[ulysses_rank]
        )
        crows, vrows, arows = storage_masks(row_role)
        local_crows, local_vrows, local_arows = storage_masks(
            row_role[row_start:row_stop]
        )

        def positions(mask):
            return host_to_device(mask.nonzero().squeeze(1), device)

        def compact(global_mask, local_mask):
            compact_indices = global_mask.long().cumsum(0).sub_(1)
            return host_to_device(
                compact_indices[row_start:row_stop][local_mask], device
            )

        tags = self.modality_tags
        modality = torch.full_like(row_role, tags["video"])
        modality[row_role == 0] = tags["text"]
        modality[arows] = tags["audio"]
        modality[row_role == 7] = tags["source_visual"]
        modality[row_role == 1] = tags["condition_visual"]
        layout = _RowLayout(
            sequence_splits=sequence_splits,
            row_start=row_start,
            row_stop=row_stop,
            local_crows=positions(local_crows),
            local_vrows=positions(local_vrows),
            local_arows=positions(local_arows),
            context_source=compact(crows, local_crows),
            video_source=compact(vrows, local_vrows),
            audio_source=compact(arows, local_arows),
            vrows=positions(vrows),
            arows=positions(arows),
            modality=modality,
        )
        if host_layout is not None:
            host_layout.derived[key] = layout
        return layout

    @staticmethod
    def _validate_tp_config(*, arch, tp_size):
        if tp_size != 1:
            raise ValueError("Zing dense inference currently requires TP=1")

    @staticmethod
    def _validate_sequence_parallel_config(*, arch, tp_size, ulysses_size, ring_size):
        if ring_size != 1:
            raise ValueError("Zing causal inference supports Ulysses SP only")
        local_heads = arch.num_attention_heads // tp_size
        if ulysses_size < 1 or local_heads % ulysses_size:
            raise ValueError(
                f"Zing TP-local heads {local_heads} must be divisible by "
                f"Ulysses size {ulysses_size}"
            )

    def initialize_adaln_table(self):
        if self.adaln_table is None:
            return
        table = self.adaln_table
        batches = [
            torch.frombuffer(
                bytearray.fromhex(v["raw_hex"]), dtype=torch.float32
            ).clone()
            for v in table.manifest["batches"]
        ]
        lease = table.prepare(
            batches,
            device=next(self.time_embedder.parameters()).device,
            schedule=table.manifest["schedule"],
            embedder=self.time_embedder,
        )
        lease.close()

    def adaln_deployment_receipt(self):
        parameters = {
            name: value
            for name, value in self.named_parameters()
            if ".adaln_proj." in name
        }
        storages = {
            v.untyped_storage().data_ptr(): v.untyped_storage()
            for v in parameters.values()
        }
        return dict(
            mode="weights_free" if self.adaln_table is not None else "resident",
            projection_parameters=list(parameters),
            projection_storage_bytes=sum(v.nbytes() for v in storages.values()),
            table=None if self.adaln_table is None else self.adaln_table.stats(),
        )

    def validate_weight_update_source(self, *, weights_path):
        if self.adaln_table is not None:
            raise ValueError(
                "Zing weights-free updates require offline rebuild and redeployment"
            )
        return super().validate_weight_update_source(weights_path=weights_path)

    def prepare_lora_adapter(self, adapter):
        if self.adaln_table is not None:
            raise ValueError(
                "Zing weights-free LoRA requires offline merge and redeployment"
            )
        return super().prepare_lora_adapter(adapter)

    def validate_lora_layers(self, layer_names):
        if self.adaln_table is not None:
            raise ValueError(
                "Zing weights-free LoRA requires offline merge and redeployment"
            )
        return super().validate_lora_layers(layer_names)

    def _preprocess_checkpoint(self, weights):
        # Weight loaders may copy through .data without bumping Tensor._version.
        # Invalidate live sessions as soon as a checkpoint load begins.
        if self.adaln_table is not None and self._cache_weight_generation:
            raise ValueError(
                "Zing weights-free checkpoint updates require offline rebuild and redeployment"
            )
        self._cache_weight_generation += 1
        for name, value in _diffusers_h3_checkpoint(
            (name.removeprefix("backbone."), value) for name, value in weights
        ):
            if name == "rope.inv_freq":
                raise ValueError(
                    "Zing checkpoint must use generated non-persistent RoPE frequencies"
                )
            yield name, value
        n = self.arch.rope_inv_freq_len
        yield (
            "rope.inv_freq",
            1.0 / (10000.0 ** (torch.arange(n, dtype=torch.float32) / n)),
        )

    def refresh_weight_derived_caches(self, *, weights_path: str | None) -> None:
        # Tensor hot updates can write through .data without changing _version
        # or invoking the disk checkpoint preprocessor. Invalidate sessions for
        # that public weight-updater path as well.
        if self.adaln_table is not None:
            raise ValueError(
                "Zing weights-free updates require offline rebuild and redeployment"
            )
        self._cache_weight_generation += 1
        super().refresh_weight_derived_caches(weights_path=weights_path)

    def _validate_packed_inputs(
        self,
        x,
        t,
        context,
        seq_lens,
        clean_seq_lens,
        block_idx,
        position_ids,
        row_role,
        audio_x,
        *,
        metadata_device=None,
    ):
        """Validate packed rows; row metadata may be host copies of the inputs."""
        lengths = seq_lens[seq_lens > 0]
        if lengths.numel() != 1:
            raise ValueError("Zing inference accepts one document per request")
        rows = int(lengths.item())
        if (
            rows < 1
            or any(v.shape != (rows,) for v in (t, block_idx, row_role))
            or position_ids.shape != (rows, 3)
        ):
            raise ValueError("Zing row metadata must cover the complete packed input")
        if (
            clean_seq_lens.shape != seq_lens.shape
            or bool((clean_seq_lens < 0).any())
            or bool((clean_seq_lens > seq_lens).any())
        ):
            raise ValueError("Invalid Zing clean sequence lengths")
        metadata_device = x.device if metadata_device is None else metadata_device
        if any(
            v.device != metadata_device for v in (t, block_idx, position_ids, row_role)
        ) or any(v.device != x.device for v in (context, audio_x)):
            raise ValueError("Zing packed tensors must share one device")
        crows, vrows, arows = storage_masks(row_role)
        if not bool((crows | vrows | arows).all()):
            raise ValueError("Unsupported Zing row role")
        p = math.prod(self.arch.patch_size)
        if (
            x.shape != (int(vrows.sum()), p, self.arch.latents_dim)
            or context.shape != (int(crows.sum()), self.arch.text_dim)
            or audio_x.shape != (int(arows.sum()), self.arch.audio_latents_dim)
        ):
            raise ValueError(
                "Zing compact storage does not match row roles or checkpoint dimensions"
            )
        clean = int(clean_seq_lens.sum())
        return rows, clean, p, crows, vrows, arows

    def _validate_cache_inputs(
        self,
        *,
        rows,
        clean,
        t,
        block_idx,
        position_ids,
        row_role,
        cache,
        self_cache_update,
    ):
        if (cache is None) != (self_cache_update is None):
            raise ValueError(
                "Zing cache and transaction phase must be supplied together"
            )
        frames = int(
            position_ids[row_role == (7 if self_cache_update == "source" else 10), 0]
            .unique()
            .numel()
        )
        if self_cache_update == "prefix":
            frames = 0
        if cache is not None:
            if clean != (0 if self_cache_update == "active" else rows):
                raise ValueError("Zing clean row count does not match the KV phase")
            if self_cache_update != "active":
                expected_t = torch.ones_like(t)
                if self_cache_update == "prefix":
                    expected_t[(row_role == 0) | (row_role == 1)] = (
                        self.arch.ar_condition_timestep
                    )
                    expected_t[row_role == 4] = self.arch.reference_visual_timestep
                if not torch.equal(t, expected_t):
                    raise ValueError(
                        "Only clean t=1 states or checkpoint-defined prefix timesteps may enter the Zing KV cache"
                    )
            cache.validate(position_ids, block_idx, self_cache_update, frames)
        return frames

    @torch.no_grad()
    def forward(
        self,
        x,
        t,
        context,
        seq_lens,
        clean_seq_lens,
        block_idx,
        position_ids,
        row_role,
        audio_x,
        cache: ZingH3KVCache | None = None,
        self_cache_update=None,
        host_layout: ZingHostLayout | None = None,
        host_t: torch.Tensor | None = None,
        joint_phases: tuple[str, ...] | None = None,
    ):
        """Run one packed Zing document.

        ``host_layout``/``host_t`` are optional CPU copies of the row metadata.
        With them, validation and row selection run on the host and reach the
        device through asynchronous copies, so the forward reads nothing back.
        """
        if (host_layout is None) != (host_t is None):
            raise ValueError("Zing host row metadata requires both layout and t")
        if host_layout is None:
            meta = (t, seq_lens, clean_seq_lens, block_idx, position_ids, row_role)
            metadata_device = None
        else:
            meta = (
                host_t,
                host_layout.seq_lens,
                host_layout.clean_seq_lens,
                host_layout.block_idx,
                host_layout.position_ids,
                host_layout.row_role,
            )
            device_meta = (
                t,
                seq_lens,
                clean_seq_lens,
                block_idx,
                position_ids,
                row_role,
            )
            if any(h.shape != d.shape for h, d in zip(meta, device_meta)):
                raise ValueError("Zing host row metadata does not match packed rows")
            if any(
                v.device != x.device for v in (t, block_idx, position_ids, row_role)
            ):
                raise ValueError("Zing packed tensors must share one device")
            metadata_device = torch.device("cpu")
        meta_t, meta_seq_lens, meta_clean, meta_block, meta_positions, meta_roles = meta
        rows, clean, p, _, _, _ = self._validate_packed_inputs(
            x,
            meta_t,
            context,
            meta_seq_lens,
            meta_clean,
            meta_block,
            meta_positions,
            meta_roles,
            audio_x,
            metadata_device=metadata_device,
        )
        joint = None
        if joint_phases is not None:
            if cache is None or self_cache_update != "joint":
                raise ValueError("Zing joint phases require a joint cache transaction")
            joint = joint_transaction(
                cache,
                joint_phases,
                meta_positions,
                meta_block,
                meta_roles,
                meta_t,
                clean,
            )
            frames = 0
        else:
            frames = self._validate_cache_inputs(
                rows=rows,
                clean=clean,
                t=meta_t,
                block_idx=meta_block,
                position_ids=meta_positions,
                row_role=meta_roles,
                cache=cache,
                self_cache_update=self_cache_update,
            )

        metadata = None
        projection_cache = None
        try:
            ulysses_size, ulysses_rank = get_ulysses_ctx()
            # Enabled for every SP size; same-SP real-device acceptance covers
            # SP1/2/4 only, SP8 and larger are not correctness-validated.
            if (
                self.experimental_fa4_metadata_reuse
                and self.experimental_attention_backend == "fa4"
            ):
                metadata = _FA4ForwardMetadata()
            replicated_prefix = (
                self.experimental_stable_shapes
                and self_cache_update == "prefix"
                and ulysses_size > 1
            )
            layout = self._row_layout(
                meta_roles,
                host_layout,
                rows,
                ulysses_size,
                ulysses_rank,
                replicated_prefix,
                x.device,
            )
            row_start, row_stop = layout.row_start, layout.row_stop
            group_sizes = video_ends = None
            prefix_joint = isinstance(joint, PrefixJointTransaction)
            if prefix_joint:
                if ulysses_size > 1 and not self.experimental_stable_shapes:
                    raise ValueError("Prompt joint SP requires stable output shapes")
                unique_t, group_sizes, all_indices, t_indices, video_ends = (
                    prefix_modulation(
                        meta_t,
                        meta_roles,
                        joint.ends,
                        self.modality_tags,
                        self.arch.num_modalities,
                    )
                )
                attention_positions = host_to_device(joint.mapped_positions, x.device)
                history_positions = (
                    None
                    if joint.mapped_history is None
                    else host_to_device(joint.mapped_history, x.device)
                )
            else:
                unique_t, t_indices = torch.unique(
                    meta_t, sorted=True, return_inverse=True
                )
                all_indices = t_indices * self.arch.num_modalities + layout.modality
                attention_positions, history_positions = (
                    cache.attention_positions(
                        position_ids,
                        prefix=self_cache_update == "prefix",
                        host_current=None if host_layout is None else meta_positions,
                    )
                    if cache is not None
                    else (position_ids, None)
                )
            # Preserve each serial timestep batch, including mixed-condition
            # batches whose rounding differs from singleton clean/media batches.
            key_unique_t = unique_t
            unique_t = host_to_device(unique_t, x.device)
            adaln_input = (
                F.silu(self.time_embedder(unique_t.float())).bfloat16()
                if joint is None
                else torch.cat(
                    [
                        F.silu(self.time_embedder(t.float())).bfloat16()
                        for t in unique_t.split(group_sizes or 1)
                    ]
                )
            )
            indices = host_to_device(all_indices[row_start:row_stop], x.device)
            local_t_indices = host_to_device(t_indices[row_start:row_stop], x.device)
            rope_positions = (
                attention_positions
                if layout.sequence_splits is not None
                else attention_positions[row_start:row_stop]
            )
            # A replicated SP prefix is captured before head sharding. Its raw store
            # must be allocated by update() after that shard is selected below.
            # Absolute full history still appends during that prefix.
            use_cache = cache is not None
            raw_attention = use_cache and cache.raw_store and not replicated_prefix
            # A pooled cache cannot hold a full-head prefix outside its slot, so a
            # replicated prefix writes each layer's head shard as it is produced.
            staged_prefix = (
                use_cache and replicated_prefix and cache.raw_store and cache.pooled
            )
            weight_addresses = None
            if use_cache:
                weight_addresses = self._bind_cache(cache, history_positions)
            projection_key = None
            if self.adaln_table is not None:
                if audio_x.shape[0]:
                    raise ValueError(
                        "Zing weights-free AdaLN supports video requests only"
                    )
                projection_cache = self.adaln_table
                # Joint forwards preserve each serial timestep batch below,
                # rather than projecting the combined union of all timesteps.
                if joint is None:
                    projection_key = projection_cache.lookup(key_unique_t)
            elif self.adaln_projection_cache_enabled and cache is not None:
                projection_cache = cache.adaln_projection_cache
                if (
                    projection_cache is None
                    or projection_cache.max_bytes != self.adaln_projection_cache_bytes
                ):
                    projection_cache = ZingH3AdalnCache(
                        self.adaln_projection_cache_bytes
                    )
                    cache.adaln_projection_cache = projection_cache
                with maybe_record_function("Zing.AdaLN.cache_key"):
                    projection_key = projection_cache.begin(
                        key_unique_t,
                        adaln_input,
                        self.time_embedder,
                        self._cache_weight_generation,
                    )
            elif cache is not None and cache.adaln_projection_cache is not None:
                cache.adaln_projection_cache.clear()
                cache.adaln_projection_cache = None

            branch_projection_keys = None
            if joint is not None and projection_cache is not None:
                # Clean/source groups share t=1; a six-branch prefix may have
                # mixed condition/reference times. Key complete serial GEMM
                # batches so neither cache mode changes projection rounding.
                branch_projection_keys = [
                    projection_cache.lookup(times)
                    if self.adaln_table is not None
                    else projection_cache.begin(
                        times,
                        embedding,
                        self.time_embedder,
                        self._cache_weight_generation,
                    )
                    for times, embedding in zip(
                        key_unique_t.split(group_sizes or 1),
                        adaln_input.split(group_sizes or 1),
                        strict=True,
                    )
                ]

            with maybe_record_function("Zing.build_RoPE_and_history_positions"):
                rope = _rope_cos_sin_cache(
                    self.rope(host_to_device(rope_positions, x.device)[None]),
                    dtype=torch.bfloat16,
                )
            mask = None
            if cache is None and not (
                clean in (0, rows) and bool((meta_block == meta_block[0]).all())
            ):
                mask = causal_mask(block_idx, clean)
            history_plan = None
            rotate_batch = None
            if raw_attention:
                history_rope_mode = os.environ.get(
                    "SGLANG_ZING_GROUPED_HISTORY_ROPE", "1"
                )
                if history_rope_mode == "contiguous":
                    rotate_batch = apply_contiguous_history_rope_batch
                elif history_rope_mode == "1":
                    rotate_batch = apply_history_rope_batch
            if (
                raw_attention
                and os.environ.get("SGLANG_ZING_HISTORY_ROPE_PLAN", "0") == "1"
            ):
                with maybe_record_function("Zing.history_RoPE_plan"):
                    history_plan = cache.prepare_history_rope_plan()
            options = _BlockOptions(
                capture=self_cache_update in {"prefix", "source", "final", "joint"},
                use_cache=use_cache,
                raw_attention=raw_attention,
                replicated_prefix=replicated_prefix,
                staged_prefix=staged_prefix,
                ulysses_rank=ulysses_rank,
                prefix_heads=self.arch.num_attention_heads // ulysses_size,
                history_plan=history_plan,
                rotate_batch=rotate_batch,
                projection_cache=projection_cache,
                projection_key=projection_key,
                joint_ends=None if joint is None else joint.ends,
                branch_projection_keys=branch_projection_keys,
                projection_group_sizes=group_sizes,
                joint_video_ends=video_ends,
            )
            inputs = _BodyInputs(
                x=x,
                audio_x=audio_x,
                context=context,
                layout=layout,
                adaln_input=adaln_input,
                indices=indices,
                local_t_indices=local_t_indices,
                rope=rope,
                seq_lens=seq_lens,
                rows=rows,
            )
            replayed, projections = None, None
            if self_cache_update == "active" and self.denoise_graphs is not None:
                replayed, projections = self._denoise_graph(
                    inputs,
                    options=options,
                    cache=cache,
                    host_layout=host_layout,
                    ulysses_size=ulysses_size,
                    mask=mask,
                    p=p,
                    weight_addresses=weight_addresses,
                )
            if replayed is None:
                video, audio, layers = self._forward_body(
                    inputs,
                    options=options,
                    cache=cache,
                    mask=mask,
                    p=p,
                    projections=projections,
                    fa4_metadata=metadata,
                )
            else:
                (video, audio), layers = replayed, []
            # Commit after all model work succeeds, so a failed output head cannot
            # leave a partially advanced session.
            if cache is not None:
                if replicated_prefix and not staged_prefix:
                    heads = self.arch.num_attention_heads // ulysses_size
                    start = ulysses_rank * heads
                    layers = [
                        tuple(
                            value[:, start : start + heads].clone() for value in layer
                        )
                        for layer in layers
                    ]
                validation = (
                    {}
                    if host_layout is None
                    else {"validation": (meta_positions, meta_block)}
                )
                if joint is not None:
                    joint.commit(
                        cache,
                        position_ids,
                        block_idx,
                        layers,
                        validation=None
                        if host_layout is None
                        else (meta_positions, meta_block),
                    )
                else:
                    cache.update(
                        position_ids,
                        block_idx,
                        self_cache_update,
                        frames,
                        layers if self_cache_update != "active" else None,
                        **validation,
                    )
            return video, audio
        except BaseException:
            if joint is not None:
                cache.clear()
            if cache is not None and cache.adaln_projection_cache is not None:
                cache.adaln_projection_cache.clear()
            raise
        finally:
            if metadata is not None:
                metadata.clear()

    @torch.no_grad()
    def prepare_fixed_forward(
        self,
        *,
        packed,
        kv_plan,
        phase,
        _model_state=None,
        projection_cache=None,
        target_grid=None,
    ):
        """Validate/pack on the host; execution contains only device arithmetic.

        The cache owner prepares effective RoPE coordinates and each branch's
        validity/length buffers before replay. History length is never a route.
        """
        if phase not in {"active", "source", "final", "prefix", "joint"}:
            raise ValueError("Unsupported fixed Zing transaction phase")
        device = packed["x"].device
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3._sampling_helpers import (
            _host_metadata,
        )

        guarded = _host_metadata(packed) if "_zing_host_guard" in packed else None
        host = None if guarded is None else guarded["host_layout"]
        host_t = None if guarded is None else guarded["host_t"]
        if host is None:
            host = ZingHostLayout(
                **{
                    name: packed[name].detach().cpu()
                    for name in (
                        "seq_lens",
                        "clean_seq_lens",
                        "block_idx",
                        "position_ids",
                        "row_role",
                    )
                }
            )
            host_t = packed["t"].detach().cpu()
        if host_t is None:
            raise ValueError("Fixed Zing forward requires host timestep metadata")
        rows, clean, patch, _, _, _ = self._validate_packed_inputs(
            packed["x"],
            host_t,
            packed["context"],
            host.seq_lens,
            host.clean_seq_lens,
            host.block_idx,
            host.position_ids,
            host.row_role,
            packed["audio_x"],
            metadata_device=torch.device("cpu"),
        )
        if rows != kv_plan.rows or clean != kv_plan.clean_rows:
            raise ValueError("Fixed Zing KV plan does not match clean/current rows")
        ends = tuple(kv_plan.ends)
        if (
            not ends
            or ends[-1] != rows
            or any(a >= b for a, b in zip((0, *ends[:-1]), ends))
        ):
            raise ValueError("Fixed Zing branch boundaries do not cover packed rows")
        joint = phase == "joint"
        if joint != (len(ends) > 1):
            raise ValueError("Fixed Zing joint phase does not match branch boundaries")
        if joint and packed["audio_x"].shape[0]:
            raise ValueError("Fixed Zing joint forward does not support audio")
        if phase == "active" and clean != 0:
            raise ValueError("Fixed Zing active rows cannot enter persistent KV")
        if phase not in {"active", "joint"} and clean != rows:
            raise ValueError("Fixed Zing clean phase requires every row to be clean")
        if phase not in {"active", "joint"}:
            expected_t = torch.ones_like(host_t)
            if phase == "prefix":
                expected_t[(host.row_role == 0) | (host.row_role == 1)] = (
                    self.arch.ar_condition_timestep
                )
                expected_t[host.row_role == 4] = self.arch.reference_visual_timestep
            if not torch.equal(host_t, expected_t):
                raise ValueError(
                    "Only clean t=1 states or checkpoint-defined prefix timesteps may enter the Zing KV cache"
                )
        size, rank = get_ulysses_ctx()
        if size not in {1, 2, 4}:
            raise ValueError(
                "Fixed Zing forward supports sequence parallel sizes 1, 2, 4"
            )
        # Preserve the eager stable-prefix GEMM/attention shapes. Prefix length
        # is already fixed by the request; only its stored head shard is local.
        replicated_prefix = (
            self.experimental_stable_shapes and phase == "prefix" and size > 1
        )
        layout = self._row_layout(
            host.row_role, host, rows, size, rank, replicated_prefix, device
        )
        unique_t, t_indices = torch.unique(host_t, sorted=True, return_inverse=True)
        group_sizes = video_ends = None
        prefix_joint = joint and (
            packed["_zing_joint_phases"][0] == "prefix" or len(ends) in (4, 5)
        )
        if prefix_joint:
            if size > 1 and not self.experimental_stable_shapes:
                raise ValueError("Prompt joint SP requires stable output shapes")
            unique_t, group_sizes, all_indices, t_indices, video_ends = (
                prefix_modulation(
                    host_t,
                    host.row_role,
                    ends,
                    self.modality_tags,
                    self.arch.num_modalities,
                )
            )
        tensors = {
            name: host_to_device(packed[name], device)
            for name in ("x", "audio_x", "context", "seq_lens")
        }
        tensors["unique_t"] = host_to_device(unique_t, device)
        tensors["indices"] = host_to_device(
            (t_indices * self.arch.num_modalities + layout.modality)[
                layout.row_start : layout.row_stop
            ],
            device,
        )
        tensors["local_t_indices"] = host_to_device(
            t_indices[layout.row_start : layout.row_stop], device
        )
        if prefix_joint:
            tensors["indices"] = host_to_device(
                all_indices[layout.row_start : layout.row_stop], device
            )
        for name in _GRAPH_LAYOUT_INPUTS:
            tensors["layout." + name] = getattr(layout, name)
        tensors.update({"kv." + name: value for name, value in kv_plan.tensors.items()})
        # Initialize the embedder's immutable frequency buffer outside capture.
        self.time_embedder(tensors["unique_t"].float())
        # Resolve small modulation outputs before capture. Replay copies these
        # tensors just like latent inputs; it never runs Python cache operations.
        projection_cache = self.adaln_table or projection_cache
        precomputed_projections = projection_cache is not None
        if precomputed_projections:
            if self.adaln_table is not None and packed["audio_x"].shape[0]:
                raise ValueError("Zing weights-free AdaLN supports video requests only")
            time_groups = unique_t.split(group_sizes or 1) if joint else (unique_t,)
            device_groups = (
                tensors["unique_t"].split(group_sizes or 1)
                if joint
                else (tensors["unique_t"],)
            )
            embeddings = [
                F.silu(self.time_embedder(t.float())).bfloat16() for t in device_groups
            ]
            keys = [
                projection_cache.lookup(t)
                if self.adaln_table is not None
                else projection_cache.begin(
                    t, embedding, self.time_embedder, self._cache_weight_generation
                )
                for t, embedding in zip(time_groups, embeddings, strict=True)
            ]
            for slot, module in [
                *((i, b.adaln_proj) for i, b in enumerate(self.blocks)),
                ("final", self.final_layer.adaln_proj),
            ]:
                parts = [
                    projection_cache.project(slot, module, embedding, key)
                    for embedding, key in zip(embeddings, keys, strict=True)
                ]
                for i, values in enumerate(zip(*parts)):
                    tensors[f"projection.{slot}.{i}"] = (
                        values[0] if len(parts) == 1 else torch.cat(values)
                    )
        optimized_fixed_480p = validated_fixed_480p(
            host,
            device,
            sp_size=size,
            heads=self.arch.num_attention_heads,
            head_dim=self.arch.attention_head_dim,
            backend=self.experimental_attention_backend,
            audio=bool(packed["audio_x"].shape[0]),
        )
        # Actual target shape supplied by the graph owner, never inferred from Q
        # or source/reference positions. Missing geometry keeps legacy routes.
        target_fixed_736p = target_grid == (23, 40) and phase in {
            "active",
            "final",
            "joint",
        }
        options = _BlockOptions(
            optimized_fixed_480p=optimized_fixed_480p,
            target_fixed_736p=target_fixed_736p,
            capture=False,
            use_cache=False,
            raw_attention=False,
            replicated_prefix=replicated_prefix,
            staged_prefix=False,
            ulysses_rank=rank,
            prefix_heads=self.arch.num_attention_heads // size,
            history_plan=None,
            rotate_batch=None,
            projection_cache=None,
            projection_key=None,
            joint_ends=ends if joint else None,
            projection_group_sizes=group_sizes,
            joint_video_ends=video_ends,
            fixed_kv=kv_plan,
        )

        def execute(static):
            kv_tensors = {name: static["kv." + name] for name in kv_plan.tensors}
            device_layout = dataclasses.replace(
                layout,
                **{name: static["layout." + name] for name in _GRAPH_LAYOUT_INPUTS},
            )
            # Both current Q and raw pooled K are rotated from effective device
            # coordinates inside the graph, including after window movement.
            rope = _rope_cos_sin_cache(
                self.rope(kv_tensors["current_positions"][None]), dtype=torch.bfloat16
            )
            full_rope = (
                _rope_cos_sin_cache(
                    self.rope(kv_tensors["positions"][None]), dtype=torch.bfloat16
                )
                if kv_plan.rotates_history
                else None
            )
            times = static["unique_t"]
            adaln = (
                torch.cat(
                    [
                        F.silu(self.time_embedder(t.float())).bfloat16()
                        for t in times.split(group_sizes or 1)
                    ]
                )
                if joint
                else F.silu(self.time_embedder(times.float())).bfloat16()
            )
            inputs = _BodyInputs(
                x=static["x"],
                audio_x=static["audio_x"],
                context=static["context"],
                layout=device_layout,
                adaln_input=adaln,
                indices=static["indices"],
                local_t_indices=static["local_t_indices"],
                rope=rope,
                seq_lens=static["seq_lens"],
                rows=rows,
            )
            fixed_options = msgspec.structs.replace(
                options, fixed_tensors=kv_tensors, full_rope=full_rope
            )
            projections = None
            if precomputed_projections:
                projections = [
                    tuple(
                        static[f"projection.{slot}.{i}"]
                        for i in range(2 if slot == "final" else 6)
                    )
                    for slot in (*range(len(self.blocks)), "final")
                ]
            video, audio, _ = self._forward_body(
                inputs,
                options=fixed_options,
                cache=None,
                mask=None,
                p=patch,
                projections=projections,
                fa4_metadata=_DeviceFA4Metadata(),
            )
            return video, audio

        signature = (
            self._cache_weight_generation,
            (model_signatures(self) if _model_state is None else _model_state).graph,
            _numerics_mode(),
            tuple((name, os.environ.get(name)) for name in _GRAPH_ROUTE_ENVS),
            phase,
            group_sizes,
            video_ends,
            ends,
            clean,
            kv_plan.capacity,
            size,
            rank,
            self.experimental_attention_backend,
            self.experimental_stable_shapes,
            self.experimental_compiled_adaln,
            precomputed_projections,
            tuple(
                (name, tuple(value.shape), value.dtype, value.device)
                for name, value in sorted(tensors.items())
            ),
            kv_plan.rotates_history,
            optimized_fixed_480p,
            target_fixed_736p,
            tuple(
                (value.data_ptr(), _tensor_layout(value))
                for value in (
                    *(value for pair in kv_plan.stores for value in pair),
                    *kv_plan.rotated,
                )
            ),
        )
        return FixedForwardPlan(tensors, signature, kv_plan, execute)

    def _bind_cache(self, cache, history_positions):
        # The timestep constant is initialized lazily by the first forward,
        # often under inference_mode. Preserve its values while making its
        # version observable to the same mutation guard as other buffers.
        frequencies = self.time_embedder._frequency_cache
        if frequencies is not None and torch.is_inference(frequencies):
            with torch.inference_mode(False), torch.no_grad():
                self.time_embedder._frequency_cache = frequencies.clone()
        # Parameter identity/version catches load_state_dict, device/dtype
        # moves and normal in-place updates without GPU synchronization.
        signatures = model_signatures(self)
        if not signatures.versioned:
            raise ValueError(
                "Zing cached RoPE requires versioned model parameters and buffers"
            )
        cache.bind_model((self._cache_weight_generation, signatures.cache))
        cache.bind_attention_positions(history_positions)
        if self.denoise_graphs is None:
            return None
        # A .data swap keeps identity and version; a captured graph reads addresses.
        return signatures.addresses

    def _embed_inputs(self, inputs, p, fa4_metadata):
        layout = inputs.layout
        x, context = inputs.x, inputs.context
        hidden = torch.zeros(
            (layout.row_stop - layout.row_start, self.arch.hidden_size),
            dtype=torch.bfloat16,
            device=x.device,
        )
        local_x = x.index_select(0, layout.video_source)
        hidden.index_copy_(
            0,
            layout.local_vrows,
            self.video_patch_proj(
                local_x.permute(0, 2, 1).reshape(-1, p * self.arch.latents_dim).float()
            )[0].bfloat16(),
        )
        hidden.index_copy_(
            0,
            layout.local_arows,
            self.audio_patch_proj(
                inputs.audio_x.index_select(0, layout.audio_source).float()
            )[0].bfloat16(),
        )
        if context.shape[0]:
            refined = self.condition_proj(context.bfloat16())[0]
            for block in self.token_refiner.blocks:
                refined = (
                    refined
                    + _attention(
                        block.attn,
                        block.norm1(refined),
                        attention_backend=self.experimental_attention_backend,
                        fa4_metadata=fa4_metadata,
                    )[0]
                )
                refined = refined + block.mlp(block.norm2(refined))
            hidden.index_copy_(
                0,
                layout.local_crows,
                self.token_refiner.final_norm(refined)
                .index_select(0, layout.context_source)
                .to(hidden.dtype),
            )
        return hidden

    def _projection(self, slot, module, inputs, options, projections):
        """One AdaLN projection: pre-resolved, cached, or computed in place."""
        if projections is not None:
            return projections[-1 if slot == "final" else slot]
        if options.joint_ends is not None:
            # Preserve sequential timestep projection shapes in the joint body.
            parts = [
                module(value)
                if options.projection_cache is None
                else options.projection_cache.project(
                    slot, module, value, options.branch_projection_keys[i]
                )
                for i, value in enumerate(
                    inputs.adaln_input.split(options.projection_group_sizes or 1)
                )
            ]
            return tuple(torch.cat(values) for values in zip(*parts))
        if options.projection_cache is None:
            return module(inputs.adaln_input)
        return options.projection_cache.project(
            slot, module, inputs.adaln_input, options.projection_key
        )

    def _block_attention(
        self,
        current,
        value,
        normed,
        *,
        index,
        inputs,
        options,
        cache,
        mask,
        fa4_metadata,
    ):
        """One block's self-attention; returns the output and its raw capture."""
        rope = inputs.rope
        if options.fixed_kv is not None:
            return self._fixed_block_attention(current, normed, index, inputs, options)

        def make_history_rope(positions):
            return _rope_cos_sin_cache(
                self.rope(host_to_device(positions, value.device)[None]),
                dtype=value.dtype,
            )

        def write_current_kv(k, v, out_k, out_v):
            _apply_rope_kv_into(k, v, rope, out_k, out_v)

        def append_current(k, v):
            if options.raw_attention:
                return cache.raw_attention_view(
                    index,
                    k,
                    v,
                    make_history_rope,
                    apply_history_rope_into,
                    write_current_kv,
                    history_plan=options.history_plan,
                    rotate_batch=options.rotate_batch,
                )
            return cache.attention_view(index, k, v, write_current_kv)

        return _attention(
            current.attn,
            normed,
            rope,
            None,
            None,
            mask,
            capture=options.capture,
            sequence_splits=inputs.layout.sequence_splits,
            attention_backend=self.experimental_attention_backend,
            append_kv=(
                append_current
                if options.use_cache
                and not (options.replicated_prefix and cache.raw_store)
                else None
            ),
            capture_rotated=options.use_cache and cache.single_store,
            append_unrotated=options.raw_attention,
            capture_stored_values=options.raw_attention,
            stage_raw_k=(
                functools.partial(cache.stage_raw_k, index)
                if options.raw_attention
                else None
            ),
            stage_prefix_kv=(
                functools.partial(
                    cache.stage_prefix_kv,
                    index,
                    head_start=options.ulysses_rank * options.prefix_heads,
                    heads=options.prefix_heads,
                )
                if options.staged_prefix
                else None
            ),
            fa4_metadata=fa4_metadata,
            joint_ends=options.joint_ends,
        )

    def _fixed_block_attention(self, current, hidden, index, inputs, options):
        """Use fixed KV storage; choose fixed-Q or varlen FA4 per branch below."""
        hidden_tensor = (
            (hidden[0] if isinstance(hidden, tuple) else hidden)
            if hidden is not None
            else None
        )
        total_rows = hidden_tensor.shape[0] if hidden_tensor is not None else 0
        if options.replicated_prefix:
            start = options.ulysses_rank * options.prefix_heads
            stop = start + options.prefix_heads

            def persist_prefix(k, v):
                options.fixed_kv.persist(
                    index,
                    k[:, start:stop],
                    v[:, start:stop],
                    tensors=options.fixed_tensors,
                )

            return _attention(
                current.attn,
                hidden,
                inputs.rope,
                capture=True,
                attention_backend=self.experimental_attention_backend,
                stage_prefix_kv=persist_prefix,
                fa4_metadata=_DeviceFA4Metadata(),
            )
        module = current.attn
        qkv = module.qkv_proj(hidden)[0]
        q, k, v = (
            value.reshape(-1, module.num_heads, module.head_dim)
            for value in qkv.chunk(3, -1)
        )
        fused_norm = use_current_qknorm_rope_kv(module, q, inputs.rope)
        if not fused_norm:
            q, k = _qk_norm(module, q, k)
        splits = inputs.layout.sequence_splits
        if splits is not None:
            exchanged = _exchange_qkv(q, k, v, splits)
            q, k, v = (value[0] for value in exchanged.chunk(3, dim=-1))
        raw_k, raw_v = k, v
        fixed = options.fixed_kv
        if fixed.rotates_history:
            # One rotation serves every forward sharing these history positions.
            # The optional live length skips unused fixed-capacity rows.
            apply_history_rope_into(
                fixed.stores[index][0],
                options.full_rope,
                fixed.rotated[index],
                live_rows=options.fixed_tensors.get("history_live_rows"),
                row_range=options.fixed_tensors.get("history_rope_range"),
            )
        if fused_norm:
            fused_q = apply_current_qknorm_rope_kv(
                module,
                q,
                raw_k,
                raw_v,
                inputs.rope,
                fixed,
                index,
                options.fixed_tensors,
            )
        else:
            fused_q = apply_current_rope_kv(
                q,
                raw_k,
                raw_v,
                inputs.rope,
                fixed,
                index,
                options.fixed_tensors,
                default=options.optimized_fixed_480p,
            )
        if fused_q is None:
            q = apply_current_rope(q, inputs.rope)
            fixed.write_current(
                index,
                apply_current_rope(raw_k, inputs.rope),
                raw_k,
                raw_v,
                tensors=options.fixed_tensors,
            )
        else:
            q = fused_q
        keys, values = fixed.rotated[index], fixed.stores[index][1]
        results, start = [], 0
        for branch, stop in enumerate(options.fixed_kv.ends):
            query = q[start:stop]
            if self.experimental_attention_backend == "fa4":
                result = _fixed_fa4_attention(
                    query,
                    keys,
                    values,
                    module.softmax_scale,
                    _FixedFA4Metadata(options.fixed_tensors, branch),
                    optimized=options.optimized_fixed_480p,
                    target_736p=options.target_fixed_736p,
                )
            elif query.device.type == "cpu":
                result = _cpu_varlen_attention(
                    query,
                    keys,
                    values,
                    options.fixed_tensors[f"cu_k.{branch}"],
                    module.softmax_scale,
                )
            else:
                valid = (
                    torch.arange(keys.shape[0], device=keys.device)
                    < options.fixed_tensors[f"cu_k.{branch}"][1]
                )
                result = F.scaled_dot_product_attention(
                    query.transpose(0, 1)[None],
                    keys.transpose(0, 1)[None],
                    values.transpose(0, 1)[None],
                    attn_mask=valid[None, None, None],
                    scale=module.softmax_scale,
                )[0].transpose(0, 1)
            results.append(result)
            start = stop
        result = torch.cat(results)
        if splits is not None:
            result = _usp_output_all_to_all_varlen(result[None], splits, head_dim=2)[0]
        return (
            module.out_proj(
                result.reshape(total_rows, module.num_heads * module.head_dim)
            )[0],
            None,
        )

    def _block_forward(
        self,
        current,
        value,
        *,
        index,
        inputs,
        options,
        cache,
        mask,
        projections,
        fa4_metadata,
    ):
        """One main block: modulated attention and MLP; returns output and capture."""
        if not self._should_fuse_pre_linear_quant(value):
            with maybe_record_function("Zing.AdaLN"):
                shift, scale, gate, ff_shift, ff_scale, ff_gate = (
                    v[inputs.indices]
                    for v in self._projection(
                        index, current.adaln_proj, inputs, options, projections
                    )
                )
            with maybe_record_function("Zing.norm_modulation.attention"):
                normed = (
                    _native_compiled_adaln(value, current.norm1, shift, scale)
                    if self.experimental_compiled_adaln
                    else current.norm1(value) * (1 + scale) + shift
                )
            attended, raw = self._block_attention(
                current,
                value,
                normed,
                index=index,
                inputs=inputs,
                options=options,
                cache=cache,
                mask=mask,
                fa4_metadata=fa4_metadata,
            )
            with maybe_record_function("Zing.gated_residual.attention"):
                value = value + gate * attended
            with maybe_record_function("Zing.norm_modulation.mlp"):
                ff_normed = (
                    _native_compiled_adaln(value, current.norm2, ff_shift, ff_scale)
                    if self.experimental_compiled_adaln
                    else current.norm2(value) * (1 + ff_scale) + ff_shift
                )
            ff_output = current.mlp(ff_normed)
            with maybe_record_function("Zing.gated_residual.mlp"):
                return value + ff_gate * ff_output, raw

        with maybe_record_function("Zing.AdaLN"):
            (
                shift_table,
                scale_table,
                gate_table,
                ff_shift_table,
                ff_scale_table,
                ff_gate_table,
            ) = self._projection(
                index, current.adaln_proj, inputs, options, projections
            )

        from seedleap_kernels.ops import (
            fused_gated_residual_rmsnorm_indexed_scale_shift_quant_fp8,
            fused_indexed_gated_residual_add,
            fused_rmsnorm_indexed_scale_shift_quant_fp8,
        )

        eps1 = float(getattr(current.norm1, "eps", 1e-6))
        eps2 = float(getattr(current.norm2, "eps", 1e-6))

        with maybe_record_function("Zing.norm_modulation.attention"):
            quantize_qkv = (
                getattr(
                    current.attn.qkv_proj.quant_method,
                    "should_quantize_activation",
                    lambda n: False,
                )(value.shape[0])
                if hasattr(current.attn.qkv_proj, "quant_method")
                else False
            )
            if quantize_qkv:
                qinput, scale = fused_rmsnorm_indexed_scale_shift_quant_fp8(
                    value,
                    current.norm1.weight,
                    shift_table,
                    scale_table,
                    inputs.indices,
                    eps=eps1,
                    quantize_fp8=True,
                    compiled_norm=self.experimental_compiled_adaln,
                )
                normed = (qinput, scale)
            else:
                normed = fused_rmsnorm_indexed_scale_shift_quant_fp8(
                    value,
                    current.norm1.weight,
                    shift_table,
                    scale_table,
                    inputs.indices,
                    eps=eps1,
                    quantize_fp8=False,
                    compiled_norm=self.experimental_compiled_adaln,
                )

        attended, raw = self._block_attention(
            current,
            value,
            normed,
            index=index,
            inputs=inputs,
            options=options,
            cache=cache,
            mask=mask,
            fa4_metadata=fa4_metadata,
        )
        attn_output = attended

        with maybe_record_function("Zing.norm_modulation.mlp"):
            quantize_fc1 = (
                getattr(
                    current.mlp.fc1.quant_method,
                    "should_quantize_activation",
                    lambda n: False,
                )(value.shape[0])
                if hasattr(current.mlp.fc1, "quant_method")
                else False
            )
            if quantize_fc1:
                value, ff_qinput, ff_scale = (
                    fused_gated_residual_rmsnorm_indexed_scale_shift_quant_fp8(
                        value,
                        attn_output,
                        gate_table,
                        current.norm2.weight,
                        ff_shift_table,
                        ff_scale_table,
                        inputs.indices,
                        eps=eps2,
                        inplace_residual=self.inplace_residual,
                        quantize_fp8=True,
                        compiled_norm=self.experimental_compiled_adaln,
                    )
                )
                ff_normed = (ff_qinput, ff_scale)
            else:
                value, ff_normed = (
                    fused_gated_residual_rmsnorm_indexed_scale_shift_quant_fp8(
                        value,
                        attn_output,
                        gate_table,
                        current.norm2.weight,
                        ff_shift_table,
                        ff_scale_table,
                        inputs.indices,
                        eps=eps2,
                        inplace_residual=self.inplace_residual,
                        quantize_fp8=False,
                        compiled_norm=self.experimental_compiled_adaln,
                    )
                )

        ff_output = current.mlp(ff_normed)
        with maybe_record_function("Zing.gated_residual.mlp"):
            value = fused_indexed_gated_residual_add(
                value,
                ff_output,
                ff_gate_table,
                inputs.indices,
                inplace=self.inplace_residual,
            )
            return value, raw

    def _forward_body(
        self, inputs, *, options, cache, mask, p, projections, fa4_metadata
    ):
        """Every kernel after host preparation; returns video, audio and captures."""
        layout, rope = inputs.layout, inputs.rope
        hidden = self._embed_inputs(inputs, p, fa4_metadata)
        layers = []
        for index, block in enumerate(self.blocks):
            raw = None

            def execute(current, value, index=index):
                nonlocal raw
                value, raw = self._block_forward(
                    current,
                    value,
                    index=index,
                    inputs=inputs,
                    options=options,
                    cache=cache,
                    mask=mask,
                    projections=projections,
                    fa4_metadata=fa4_metadata,
                )
                return value

            hidden = block(
                hidden,
                adaln_input=inputs.adaln_input,
                combined_indices=inputs.indices,
                rope_cache=rope,
                cu_seqlens=inputs.seq_lens,
                max_seqlen=inputs.rows,
                execution_override=execute,
            )
            if options.capture:
                layers.append(raw)
        shift, scale = (
            v[inputs.local_t_indices]
            for v in self._projection(
                "final", self.final_layer.adaln_proj, inputs, options, projections
            )
        )
        hidden = self.final_layer.norm(hidden) * (1 + scale) + shift
        output_options = {"joint_ends": options.joint_ends}
        if options.joint_video_ends is not None:
            output_options.update(
                joint_ends=options.joint_video_ends, mixed_prefix=True
            )
        video, audio = self._project_outputs(hidden, p, layout, **output_options)
        return video, audio, layers

    def _denoise_graph_rejection(
        self, inputs, *, options, cache, host_layout, ulysses_size, mask
    ):
        """Why this denoise forward cannot use a graph, or None."""
        if cache.window_eviction_timing == "after_step0":
            return "window_eviction_timing"
        if self.denoise_graphs.disabled:
            return "graphs_disabled"
        if not self.denoise_graphs.backend.supports(inputs.x.device):
            return "device"
        if ulysses_size != 1:
            reason = self._sequence_parallel_graph_rejection(inputs, ulysses_size)
            if reason is not None:
                return reason
        if host_layout is None:
            return "device_metadata"
        if _has_forward_hooks(self):
            return "module_hooks"
        if not (cache.pooled and options.raw_attention):
            return "unpooled"
        if options.projection_cache is None or options.projection_key is None:
            return "adaln_cache"
        if options.history_plan is not None:
            return "history_rope_plan"
        if not cache.chunk_rope_reuse:
            return "chunk_rope_reuse_off"
        if inputs.context.shape[0] or mask is not None:
            return "context"
        return None

    def _sequence_parallel_graph_rejection(self, inputs, ulysses_size):
        """Why a sequence-parallel denoise forward cannot use a graph, or None.

        Only a pure Ulysses group (no ring, no tensor parallelism) is supported.
        Variable-length all-to-all handles uneven row splits; global rows and
        local tensor layouts distinguish their captured graphs.
        """
        reason = _sequence_parallel_topology_rejection(enabled=self.denoise_graph_sp)
        if reason is not None:
            return reason
        return None

    def _coordinate_sequence_parallel_routes(self, graphs, *, device, ulysses_size):
        """Give a graph set a route coordinator when its topology can use graphs.

        A topology that never uses graphs gets none, so its denoise forwards pay
        no cross-rank vote.
        """
        if ulysses_size == 1 or graphs.coordinator is not None:
            return
        # Devices without graph support never reach the topology check.
        if not graphs.backend.supports(device):
            return
        if _sequence_parallel_topology_rejection(enabled=self.denoise_graph_sp):
            return
        graphs.coordinator = RouteCoordinator(get_sp_group().cpu_group)

    def _denoise_graph(
        self,
        inputs,
        *,
        options,
        cache,
        host_layout,
        ulysses_size,
        mask,
        p,
        weight_addresses,
    ):
        """Replay or capture a denoise forward; returns (outputs or None, projections).

        Projections resolved here are passed to an eager fallback too, so the
        AdaLN cache sees exactly one lookup per slot either way.
        """
        graphs = self.denoise_graphs
        self._coordinate_sequence_parallel_routes(
            graphs, device=inputs.x.device, ulysses_size=ulysses_size
        )
        reason = self._denoise_graph_rejection(
            inputs,
            options=options,
            cache=cache,
            host_layout=host_layout,
            ulysses_size=ulysses_size,
            mask=mask,
        )
        if reason is not None:
            graphs.count(reason)
            graphs.vote_eager()
            return None, None
        projections = [
            self._projection(index, block.adaln_proj, inputs, options, None)
            for index, block in enumerate(self.blocks)
        ]
        projections.append(
            self._projection(
                "final", self.final_layer.adaln_proj, inputs, options, None
            )
        )
        # Global rows: under sequence parallel the cache holds the whole sequence.
        rows = inputs.rows
        history = cache.reused_history_rows(rows)
        if history is None:
            graphs.count("rope_rebuild")
            graphs.vote_eager()
            return None, projections
        tensors = _graph_tensors(inputs, projections)
        key = self._denoise_graph_key(
            tensors=tensors,
            cache=cache,
            history=history,
            rows=rows,
            weight_addresses=weight_addresses,
            ulysses_size=ulysses_size,
        )
        stats = dict(cache.rope_reuse_stats)

        def capture(static):
            # A backend may warm up before capturing; account one forward only.
            cache.rope_reuse_stats.update(stats)
            video, audio, _ = self._forward_body(
                _static_inputs(inputs, static),
                options=options,
                cache=cache,
                mask=None,
                p=p,
                projections=_static_projections(static, projections),
                fa4_metadata=_DeviceFA4Metadata(),
            )
            return video, audio

        outputs = graphs.run(
            key,
            tensors=tensors,
            capture=capture,
            replay_host=cache.replay_reused_attention,
        )
        if outputs is None:
            # A failed warm-up may have run part of the forward's Python.
            cache.rope_reuse_stats.update(stats)
        return outputs, projections

    def _denoise_graph_key(
        self, *, tensors, cache, history, rows, weight_addresses, ulysses_size
    ):
        """Everything a captured forward reads by address or decides on the host."""
        return (
            ulysses_size,
            self._cache_weight_generation,
            weight_addresses,
            tuple((name, _tensor_layout(t)) for name, t in tensors.items()),
            cache.store_layout_signature(),
            (history, rows),
            tuple(os.environ.get(name) for name in _GRAPH_ROUTE_ENVS),
            _numerics_mode(),
            self.experimental_compiled_adaln,
            self.experimental_attention_backend,
        )


EntryClass = ZingH3DiTModel
