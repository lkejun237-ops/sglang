# SPDX-License-Identifier: Apache-2.0
"""Zing's four-modality causal execution over the shared H3 parameter modules.

The checkpoint has separate Q/K/V and value|gate SwiGLU matrices. The H3
loader converts these once; execution uses the same projections, RMSNorms,
time embedding, refiner, MLP and output heads as native H3.
"""

import math

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3DiTModel,
    _apply_rope,
    _diffusers_h3_checkpoint,
    _rope_cos_sin_cache,
)
from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


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


def _attention(
    module,
    hidden,
    rope=None,
    history=None,
    history_rope=None,
    mask=None,
    *,
    capture=False,
):
    qkv = module.qkv_proj(hidden)[0]
    q, k, v = (
        x.reshape(-1, module.num_heads, module.head_dim) for x in qkv.chunk(3, -1)
    )
    q, k = module.q_norm(q), module.k_norm(k)
    # V is a view into fused QKV. Persist only V, not the entire projection.
    raw = (k, v.clone()) if capture else None
    if rope is not None:
        q, k = _apply_rope(q, rope), _apply_rope(k, rope)
    if history is not None:
        k = torch.cat((_apply_rope(history[0], history_rope), k))
        v = torch.cat((history[1], v))
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
    # The initial dense implementation runs on one device. Its block loop
    # directly invokes shared primitives, so do not install block FSDP hooks.
    _fsdp_shard_conditions = []
    _fsdp_forward_methods = ()
    _compile_conditions = []
    _supported_attention_backends = {AttentionBackendEnum.TORCH_SDPA}

    def __init__(self, config, hf_config, quant_config=None, **kwargs):
        arch = config.arch_config
        if arch.num_modalities != 4 or arch.adaln_out_features != 24 * arch.hidden_size:
            raise ValueError("Zing requires the trained four-modality AdaLN checkpoint")
        if quant_config is not None or arch.adaln_curve_grid is not None or kwargs:
            raise ValueError(
                "Zing dense inference requires unquantized full AdaLN weights"
            )
        arch.checkpoint_uses_diffusers_layout = True
        super().__init__(config, hf_config, quant_config=None)
        # The text refiner is small enough to remain resident. Main DiT blocks
        # execute through Module.__call__, so the standard layerwise-offload
        # hooks can stream them on memory-constrained GPUs.
        self.layer_names = ["blocks"]
        self.preprocess_loaded_state_dict = self._preprocess_checkpoint

    @staticmethod
    def _validate_tp_config(*, arch, tp_size):
        if tp_size != 1:
            raise ValueError("Zing dense inference currently requires TP=1")

    @staticmethod
    def _validate_sequence_parallel_config(*, arch, tp_size, ulysses_size, ring_size):
        if ulysses_size != 1 or ring_size != 1:
            raise ValueError("Zing dense inference currently requires SP=1")

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

        hidden = torch.zeros(
            (rows, self.arch.hidden_size), dtype=torch.bfloat16, device=x.device
        )
        hidden[vrows] = self.video_patch_proj(
            x.permute(0, 2, 1).reshape(-1, p * self.arch.latents_dim).float()
        )[0].bfloat16()
        hidden[arows] = self.audio_patch_proj(audio_x.float())[0].bfloat16()
        if context.shape[0]:
            refined = self.condition_proj(context.bfloat16())[0]
            for block in self.token_refiner.blocks:
                refined = refined + _attention(block.attn, block.norm1(refined))[0]
                refined = refined + block.mlp(block.norm2(refined))
            hidden[crows] = self.token_refiner.final_norm(refined)
        unique_t, t_indices = torch.unique(t, sorted=True, return_inverse=True)
        adaln_input = F.silu(self.time_embedder(unique_t.float())).bfloat16()
        modality = torch.zeros_like(row_role)
        modality[row_role == 0], modality[arows], modality[row_role == 7] = 1, 2, 3
        indices = t_indices * 4 + modality
        rope = _rope_cos_sin_cache(self.rope(position_ids[None]), dtype=hidden.dtype)
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
                shift, scale, gate, ff_shift, ff_scale, ff_gate = (
                    v[indices] for v in current.adaln_proj(adaln_input)
                )
                normed = current.norm1(value) * (1 + scale) + shift
                attended, raw = _attention(
                    current.attn,
                    normed,
                    rope,
                    None if cache is None else cache.history(index),
                    history_rope,
                    mask,
                    capture=capture,
                )
                value = value + gate * attended
                return value + ff_gate * current.mlp(
                    current.norm2(value) * (1 + ff_scale) + ff_shift
                )

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
        shift, scale = (v[t_indices] for v in self.final_layer.adaln_proj(adaln_input))
        hidden = self.final_layer.norm(hidden) * (1 + scale) + shift
        video = (
            self.final_layer.video_out(hidden[vrows].float())[0]
            .reshape(-1, self.arch.latents_dim, p)
            .permute(0, 2, 1)
        )
        audio = self.final_layer.audio_out(hidden[arows].float())[0]
        # Commit after all model work succeeds, so a failed output head cannot
        # leave a partially advanced session.
        if cache is not None:
            cache.update(
                position_ids,
                block_idx,
                self_cache_update,
                frames,
                layers if self_cache_update != "active" else None,
            )
        return video, audio


EntryClass = ZingH3DiTModel
