# SPDX-License-Identifier: Apache-2.0
"""Zing's four-modality causal execution over the shared H3 parameter modules.

The checkpoint has separate Q/K/V and value|gate SwiGLU matrices. The H3
loader converts these once; execution uses the same projections, RMSNorms,
time embedding, refiner, MLP and output heads as native H3.
"""

import collections
import functools
import math
import os
import warnings
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_group,
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
from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.models.dits.zing_h3_rope import (
    apply_contiguous_history_rope_batch,
    apply_history_rope_batch,
    apply_history_rope_into,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.profiler import maybe_record_function


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
):
    if attention_backend not in {"torch_sdpa", "fa4"}:
        raise ValueError(
            f"Unsupported Zing experimental attention: {attention_backend}"
        )
    if attention_backend == "fa4" and mask is not None:
        raise ValueError(
            "Zing experimental FA4 does not support explicit attention masks"
        )
    qkv = module.qkv_proj(hidden)[0]
    q, k, v = (
        x.reshape(-1, module.num_heads, module.head_dim) for x in qkv.chunk(3, -1)
    )
    with maybe_record_function("Zing.QK_norm"):
        q, k = _qk_norm(module, q, k)
    if sequence_splits is not None:
        with maybe_record_function("Zing.Ulysses.input_all_to_all"):
            qkv = _usp_input_all_to_all_varlen_qkv(
                q[None].contiguous(),
                k[None].contiguous(),
                v[None].contiguous(),
                sequence_splits,
            )
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
                    q = _apply_rope(q, rope)
                    # The single store rotates K straight into its scratch tail.
                    if not (capture_rotated or append_unrotated):
                        k = _apply_rope(k, rope)
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
        if attention_backend == "fa4":
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
            ].reshape(hidden.shape[0], module.num_heads * module.head_dim)
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
                q = _apply_rope(q, rope)
                if not (capture_rotated or append_unrotated):
                    k = _apply_rope(k, rope)
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
    if attention_backend == "fa4":
        with maybe_record_function("Zing.FA4"):
            result = _fa4_attention(
                q, k, v, module.softmax_scale, metadata=fa4_metadata
            ).reshape(hidden.shape[0], module.num_heads * module.head_dim)
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
                .reshape(hidden.shape[0], -1)
            )
    return module.out_proj(result)[0], raw


class ZingH3DiTModel(MiniMaxH3DiTModel):
    _aliases = []
    # Shard only the main DiT blocks. The token-refiner blocks are driven by
    # their inner modules below, so they must remain in the root FSDP group.
    _fsdp_shard_conditions = [_is_zing_main_block]
    _fsdp_forward_methods = ()
    _compile_conditions = []
    _supported_attention_backends = {AttentionBackendEnum.TORCH_SDPA}

    def __init__(self, config, hf_config, quant_config=None, **kwargs):
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
        projection_cache = os.environ.get("SGLANG_ZING_ADALN_PROJECTION_CACHE", "0")
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
        if arch.num_modalities != 4 or arch.adaln_out_features != 24 * arch.hidden_size:
            raise ValueError("Zing requires the trained four-modality AdaLN checkpoint")
        if quant_config is not None or arch.adaln_curve_grid is not None or kwargs:
            raise ValueError(
                "Zing dense inference requires unquantized full AdaLN weights"
            )
        arch.checkpoint_uses_diffusers_layout = True
        super().__init__(config, hf_config, quant_config=None)
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

    def _project_outputs(self, hidden, p, layout):
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
            hidden = gather_sequence_varlen(
                hidden, sequence_splits, get_sp_group().ulysses_group, dim=0
            )
            output_vrows, output_arows = layout.vrows, layout.arows
        video = (
            self.final_layer.video_out(hidden.index_select(0, output_vrows).float())[0]
            .reshape(-1, self.arch.latents_dim, p)
            .permute(0, 2, 1)
        )
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

    @staticmethod
    def _row_layout(
        row_role,
        host_layout,
        rows,
        ulysses_size,
        ulysses_rank,
        replicated_prefix,
        device,
    ):
        """Row selections for this rank; host layouts cache their device copies."""
        key = (ulysses_size, ulysses_rank, replicated_prefix, device)
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

        modality = torch.zeros_like(row_role)
        modality[row_role == 0], modality[arows], modality[row_role == 7] = 1, 2, 3
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

    def _preprocess_checkpoint(self, weights):
        # Weight loaders may copy through .data without bumping Tensor._version.
        # Invalidate live sessions as soon as a checkpoint load begins.
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
            if self_cache_update != "active" and not bool((t == 1).all()):
                raise ValueError("Only clean t=1 states may enter the Zing KV cache")
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
            sequence_splits = layout.sequence_splits
            row_start, row_stop = layout.row_start, layout.row_stop

            hidden = torch.zeros(
                (row_stop - row_start, self.arch.hidden_size),
                dtype=torch.bfloat16,
                device=x.device,
            )
            local_x = x.index_select(0, layout.video_source)
            hidden.index_copy_(
                0,
                layout.local_vrows,
                self.video_patch_proj(
                    local_x.permute(0, 2, 1)
                    .reshape(-1, p * self.arch.latents_dim)
                    .float()
                )[0].bfloat16(),
            )
            hidden.index_copy_(
                0,
                layout.local_arows,
                self.audio_patch_proj(
                    audio_x.index_select(0, layout.audio_source).float()
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
                            fa4_metadata=metadata,
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
            unique_t, t_indices = torch.unique(meta_t, sorted=True, return_inverse=True)
            unique_t = host_to_device(unique_t, x.device)
            adaln_input = F.silu(self.time_embedder(unique_t.float())).bfloat16()
            indices = host_to_device(
                (t_indices * 4 + layout.modality)[row_start:row_stop], x.device
            )
            local_t_indices = host_to_device(t_indices[row_start:row_stop], x.device)
            attention_positions, history_positions = (
                cache.attention_positions(
                    position_ids,
                    prefix=self_cache_update == "prefix",
                    host_current=None if host_layout is None else meta_positions,
                )
                if cache is not None
                else (position_ids, None)
            )
            rope_positions = (
                attention_positions
                if sequence_splits is not None
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
            prefix_heads = self.arch.num_attention_heads // ulysses_size
            if use_cache:
                # The timestep constant is initialized lazily by the first forward,
                # often under inference_mode. Preserve its values while making its
                # version observable to the same mutation guard as other buffers.
                frequencies = self.time_embedder._frequency_cache
                if frequencies is not None and torch.is_inference(frequencies):
                    with torch.inference_mode(False), torch.no_grad():
                        self.time_embedder._frequency_cache = frequencies.clone()
                # Parameter identity/version catches load_state_dict, device/dtype
                # moves and normal in-place updates without GPU synchronization.
                model_signature = tuple(
                    cache._tensor_signature(value)
                    for value in (*self.parameters(), *self.buffers())
                )
                if any(signature[1] is None for signature in model_signature):
                    raise ValueError(
                        "Zing cached RoPE requires versioned model parameters and buffers"
                    )
                cache.bind_model((self._cache_weight_generation, model_signature))
                cache.bind_attention_positions(history_positions)
            projection_key = None
            if self.adaln_projection_cache_enabled and cache is not None:
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
                        unique_t,
                        adaln_input,
                        self.time_embedder,
                        self._cache_weight_generation,
                    )
            elif cache is not None and cache.adaln_projection_cache is not None:
                cache.adaln_projection_cache.clear()
                cache.adaln_projection_cache = None

            with maybe_record_function("Zing.build_RoPE_and_history_positions"):
                rope = _rope_cos_sin_cache(
                    self.rope(host_to_device(rope_positions, x.device)[None]),
                    dtype=hidden.dtype,
                )
            mask = None
            if cache is None and not (
                clean in (0, rows) and bool((meta_block == meta_block[0]).all())
            ):
                mask = causal_mask(block_idx, clean)
            capture = self_cache_update in {"prefix", "source", "final"}
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
            layers = []
            for index, block in enumerate(self.blocks):
                raw = None

                def execute(current, value, index=index):
                    nonlocal raw
                    with maybe_record_function("Zing.AdaLN"):
                        shift, scale, gate, ff_shift, ff_scale, ff_gate = (
                            v[indices]
                            for v in (
                                current.adaln_proj(adaln_input)
                                if projection_cache is None
                                else projection_cache.project(
                                    index,
                                    current.adaln_proj,
                                    adaln_input,
                                    projection_key,
                                )
                            )
                        )
                    with maybe_record_function("Zing.norm_modulation.attention"):
                        normed = (
                            _native_compiled_adaln(value, current.norm1, shift, scale)
                            if self.experimental_compiled_adaln
                            else current.norm1(value) * (1 + scale) + shift
                        )

                    def make_history_rope(positions):
                        return _rope_cos_sin_cache(
                            self.rope(host_to_device(positions, value.device)[None]),
                            dtype=value.dtype,
                        )

                    def write_current_kv(k, v, out_k, out_v):
                        _apply_rope_kv_into(k, v, rope, out_k, out_v)

                    def append_current(k, v):
                        if raw_attention:
                            return cache.raw_attention_view(
                                index,
                                k,
                                v,
                                make_history_rope,
                                apply_history_rope_into,
                                write_current_kv,
                                history_plan=history_plan,
                                rotate_batch=rotate_batch,
                            )
                        return cache.attention_view(index, k, v, write_current_kv)

                    attended, raw = _attention(
                        current.attn,
                        normed,
                        rope,
                        None,
                        None,
                        mask,
                        capture=capture,
                        sequence_splits=sequence_splits,
                        attention_backend=self.experimental_attention_backend,
                        append_kv=(
                            append_current
                            if use_cache and not (replicated_prefix and cache.raw_store)
                            else None
                        ),
                        capture_rotated=use_cache and cache.single_store,
                        append_unrotated=raw_attention,
                        capture_stored_values=raw_attention,
                        stage_raw_k=(
                            functools.partial(cache.stage_raw_k, index)
                            if raw_attention
                            else None
                        ),
                        stage_prefix_kv=(
                            functools.partial(
                                cache.stage_prefix_kv,
                                index,
                                head_start=ulysses_rank * prefix_heads,
                                heads=prefix_heads,
                            )
                            if staged_prefix
                            else None
                        ),
                        fa4_metadata=metadata,
                    )
                    with maybe_record_function("Zing.gated_residual.attention"):
                        value = value + gate * attended
                    with maybe_record_function("Zing.norm_modulation.mlp"):
                        ff_normed = (
                            _native_compiled_adaln(
                                value, current.norm2, ff_shift, ff_scale
                            )
                            if self.experimental_compiled_adaln
                            else current.norm2(value) * (1 + ff_scale) + ff_shift
                        )
                    ff_output = current.mlp(ff_normed)
                    with maybe_record_function("Zing.gated_residual.mlp"):
                        return value + ff_gate * ff_output

                hidden = block(
                    hidden,
                    adaln_input=adaln_input,
                    combined_indices=indices,
                    rope_cache=rope,
                    cu_seqlens=seq_lens,
                    max_seqlen=rows,
                    execution_override=execute,
                )
                if capture:
                    layers.append(raw)
            shift, scale = (
                v[local_t_indices]
                for v in (
                    self.final_layer.adaln_proj(adaln_input)
                    if projection_cache is None
                    else projection_cache.project(
                        "final",
                        self.final_layer.adaln_proj,
                        adaln_input,
                        projection_key,
                    )
                )
            )
            hidden = self.final_layer.norm(hidden) * (1 + scale) + shift
            video, audio = self._project_outputs(hidden, p, layout)
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
            if cache is not None and cache.adaln_projection_cache is not None:
                cache.adaln_projection_cache.clear()
            raise
        finally:
            if metadata is not None:
                metadata.clear()


EntryClass = ZingH3DiTModel
