# SPDX-License-Identifier: Apache-2.0
"""Zing's four-modality causal execution over the shared H3 parameter modules.

The checkpoint has separate Q/K/V and value|gate SwiGLU matrices. The H3
loader converts these once; execution uses the same projections, RMSNorms,
time embedding, refiner, MLP and output heads as native H3.
"""

import functools
import math
import os

import torch
import torch.nn.functional as F
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
    _diffusers_h3_checkpoint,
    _rope_cos_sin_cache,
)
from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.profiler import maybe_record_function


def storage_masks(roles):
    return (
        (roles == 0) | (roles == 1),
        (roles == 4) | (roles == 7) | (roles == 10),
        (roles == 5) | (roles == 11),
    )


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
    """Opt-in native CUDA fusion, including its singleton batch dimension."""
    if not hidden.is_cuda:
        raise ValueError("Zing experimental compiled AdaLN requires CUDA tensors")
    return _compiled_rms_norm_modulate(torch.are_deterministic_algorithms_enabled())(
        hidden.unsqueeze(0), norm.weight, shift, scale, float(norm.eps)
    ).squeeze(0)


def _fa4_attention(q, k, v, scale):
    """Explicit experimental native FA4 route for one unmasked document."""
    if not q.is_cuda:
        raise ValueError("Zing experimental FA4 attention requires CUDA tensors")
    try:
        from flash_attn.cute import flash_attn_varlen_func
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "Zing experimental FA4 requires flash_attn.cute.flash_attn_varlen_func"
        ) from error
    q_rows, k_rows = q.shape[0], k.shape[0]
    result = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=torch.tensor([0, q_rows], dtype=torch.int32, device=q.device),
        cu_seqlens_k=torch.tensor([0, k_rows], dtype=torch.int32, device=k.device),
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
    q, k = module.q_norm(q), module.k_norm(k)
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
        raw = (k[0].clone(), v[0].clone()) if capture else None
        with maybe_record_function("Zing.RoPE_and_KV_append"):
            if rope is not None:
                q, k = _apply_rope(q, rope), _apply_rope(k, rope)
            if history is not None:
                history_k = _apply_rope(history[0], history_rope)
                history_v = history[1]
                k = torch.cat((history_k[None], k), dim=1)
                v = torch.cat((history_v[None], v), dim=1)
        if attention_backend == "fa4":
            with maybe_record_function("Zing.FA4"):
                result = _fa4_attention(q[0], k[0], v[0], module.softmax_scale)[None]
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
    raw = (k, v.clone()) if capture else None
    with maybe_record_function("Zing.RoPE_and_KV_append"):
        if rope is not None:
            q, k = _apply_rope(q, rope), _apply_rope(k, rope)
        if history is not None:
            k = torch.cat((_apply_rope(history[0], history_rope), k))
            v = torch.cat((history[1], v))
    if attention_backend == "fa4":
        with maybe_record_function("Zing.FA4"):
            result = _fa4_attention(q, k, v, module.softmax_scale).reshape(
                hidden.shape[0], module.num_heads * module.head_dim
            )
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
            "SGLANG_ZING_EXPERIMENTAL_ATTENTION", "torch_sdpa"
        )
        if experimental_attention not in {"torch_sdpa", "fa4"}:
            raise ValueError(
                "SGLANG_ZING_EXPERIMENTAL_ATTENTION must be torch_sdpa or fa4"
            )
        stable_shapes = os.environ.get("SGLANG_ZING_EXPERIMENTAL_STABLE_SHAPES", "0")
        if stable_shapes not in {"0", "1"}:
            raise ValueError("SGLANG_ZING_EXPERIMENTAL_STABLE_SHAPES must be 0 or 1")
        compiled_adaln = os.environ.get("SGLANG_ZING_EXPERIMENTAL_COMPILED_ADALN", "0")
        if compiled_adaln not in {"0", "1"}:
            raise ValueError("SGLANG_ZING_EXPERIMENTAL_COMPILED_ADALN must be 0 or 1")
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
        # The text refiner is small enough to remain resident. Main DiT blocks
        # execute through Module.__call__, so the standard layerwise-offload
        # hooks can stream them on memory-constrained GPUs.
        self.layer_names = ["blocks"]
        self.preprocess_loaded_state_dict = self._preprocess_checkpoint

    def _project_outputs(
        self, hidden, p, sequence_splits, local_vrows, local_arows, vrows, arows
    ):
        """Project video/audio rows and restore global packed order across SP.

        Stable shapes gather hidden rows before projection to match SP1 GEMM
        shapes. Otherwise, project local rows and gather packed logits once.
        """
        replicated_output = (
            self.experimental_stable_shapes and sequence_splits is not None
        )
        output_vrows, output_arows = local_vrows, local_arows
        if replicated_output:
            hidden = gather_sequence_varlen(
                hidden, sequence_splits, get_sp_group().ulysses_group, dim=0
            )
            output_vrows, output_arows = vrows, arows
        video = (
            self.final_layer.video_out(hidden[output_vrows].float())[0]
            .reshape(-1, self.arch.latents_dim, p)
            .permute(0, 2, 1)
        )
        audio = self.final_layer.audio_out(hidden[output_arows].float())[0]
        if sequence_splits is not None and not replicated_output:
            video_width = p * self.arch.latents_dim
            audio_width = self.arch.audio_latents_dim
            local_logits = hidden.new_zeros(
                (hidden.shape[0], video_width + audio_width), dtype=video.dtype
            )
            local_logits[local_vrows, :video_width] = video.reshape(-1, video_width)
            local_logits[local_arows, video_width:] = audio
            logits = gather_sequence_varlen(
                local_logits,
                sequence_splits,
                get_sp_group().ulysses_group,
                dim=0,
            )
            video = logits[vrows, :video_width].reshape(-1, p, self.arch.latents_dim)
            audio = logits[arows, video_width:]
        return video, audio

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
    ):
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
        if any(
            v.device != x.device
            for v in (t, context, block_idx, position_ids, row_role, audio_x)
        ):
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
    ):
        rows, clean, p, crows, vrows, arows = self._validate_packed_inputs(
            x,
            t,
            context,
            seq_lens,
            clean_seq_lens,
            block_idx,
            position_ids,
            row_role,
            audio_x,
        )
        frames = self._validate_cache_inputs(
            rows=rows,
            clean=clean,
            t=t,
            block_idx=block_idx,
            position_ids=position_ids,
            row_role=row_role,
            cache=cache,
            self_cache_update=self_cache_update,
        )

        ulysses_size, ulysses_rank = get_ulysses_ctx()
        replicated_prefix = (
            self.experimental_stable_shapes
            and self_cache_update == "prefix"
            and ulysses_size > 1
        )
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
        local_row_role = row_role[row_start:row_stop]
        local_crows, local_vrows, local_arows = storage_masks(local_row_role)

        def local_compact(values, global_mask, local_mask):
            compact_indices = global_mask.long().cumsum(0).sub_(1)
            compact_indices = compact_indices[row_start:row_stop][local_mask]
            return values.index_select(0, compact_indices)

        hidden = torch.zeros(
            (row_stop - row_start, self.arch.hidden_size),
            dtype=torch.bfloat16,
            device=x.device,
        )
        local_x = local_compact(x, vrows, local_vrows)
        hidden[local_vrows] = self.video_patch_proj(
            local_x.permute(0, 2, 1).reshape(-1, p * self.arch.latents_dim).float()
        )[0].bfloat16()
        hidden[local_arows] = self.audio_patch_proj(
            local_compact(audio_x, arows, local_arows).float()
        )[0].bfloat16()
        if context.shape[0]:
            refined = self.condition_proj(context.bfloat16())[0]
            for block in self.token_refiner.blocks:
                refined = (
                    refined
                    + _attention(
                        block.attn,
                        block.norm1(refined),
                        attention_backend=self.experimental_attention_backend,
                    )[0]
                )
                refined = refined + block.mlp(block.norm2(refined))
            hidden[local_crows] = local_compact(
                self.token_refiner.final_norm(refined), crows, local_crows
            )
        unique_t, t_indices = torch.unique(t, sorted=True, return_inverse=True)
        adaln_input = F.silu(self.time_embedder(unique_t.float())).bfloat16()
        modality = torch.zeros_like(row_role)
        modality[row_role == 0], modality[arows], modality[row_role == 7] = 1, 2, 3
        indices = (t_indices * 4 + modality)[row_start:row_stop]
        rope_positions = (
            position_ids
            if sequence_splits is not None
            else position_ids[row_start:row_stop]
        )
        with maybe_record_function("Zing.build_RoPE_and_history_positions"):
            rope = _rope_cos_sin_cache(
                self.rope(rope_positions[None]), dtype=hidden.dtype
            )
            history_positions = cache.positions() if cache is not None else None
            history_rope = (
                None
                if history_positions is None
                else _rope_cos_sin_cache(
                    self.rope(history_positions[None]), dtype=hidden.dtype
                )
            )
        mask = None
        if cache is None and not (
            clean in (0, rows) and bool((block_idx == block_idx[0]).all())
        ):
            mask = causal_mask(block_idx, clean)
        capture = self_cache_update in {"prefix", "source", "final"}
        layers = []
        for index, block in enumerate(self.blocks):
            raw = None

            def execute(current, value):
                nonlocal raw
                with maybe_record_function("Zing.AdaLN"):
                    shift, scale, gate, ff_shift, ff_scale, ff_gate = (
                        v[indices] for v in current.adaln_proj(adaln_input)
                    )
                normed = (
                    _native_compiled_adaln(value, current.norm1, shift, scale)
                    if self.experimental_compiled_adaln
                    else current.norm1(value) * (1 + scale) + shift
                )
                with maybe_record_function("Zing.KV_history_concat"):
                    history = None if cache is None else cache.history(index)
                attended, raw = _attention(
                    current.attn,
                    normed,
                    rope,
                    history,
                    history_rope,
                    mask,
                    capture=capture,
                    sequence_splits=sequence_splits,
                    attention_backend=self.experimental_attention_backend,
                )
                value = value + gate * attended
                ff_normed = (
                    _native_compiled_adaln(value, current.norm2, ff_shift, ff_scale)
                    if self.experimental_compiled_adaln
                    else current.norm2(value) * (1 + ff_scale) + ff_shift
                )
                return value + ff_gate * current.mlp(ff_normed)

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
        local_t_indices = t_indices[row_start:row_stop]
        shift, scale = (
            v[local_t_indices] for v in self.final_layer.adaln_proj(adaln_input)
        )
        hidden = self.final_layer.norm(hidden) * (1 + scale) + shift
        video, audio = self._project_outputs(
            hidden, p, sequence_splits, local_vrows, local_arows, vrows, arows
        )
        # Commit after all model work succeeds, so a failed output head cannot
        # leave a partially advanced session.
        if cache is not None:
            if replicated_prefix:
                heads = self.arch.num_attention_heads // ulysses_size
                start = ulysses_rank * heads
                layers = [
                    tuple(value[:, start : start + heads].clone() for value in layer)
                    for layer in layers
                ]
            cache.update(
                position_ids,
                block_idx,
                self_cache_update,
                frames,
                layers if self_cache_update != "active" else None,
            )
        return video, audio


EntryClass = ZingH3DiTModel
