# SPDX-License-Identifier: Apache-2.0
"""Zing H3 decoder class selection: nested config schema and legacy mapping."""

from collections.abc import Mapping

import msgspec

DECODER_TYPES = ("official_h3", "taeh3", "lynn_streaming")
DEFAULT_DECODER_TYPE = "lynn_streaming"
LEGACY_DECODER_FIELDS = frozenset(
    {
        "video_decoder_backend",
        "taeh3_checkpoint_path",
        "taeh3_decode_precision",
        "vae_decode_precision",
    }
)
_LEGACY_BACKEND_TYPES = {"native": "official_h3", "taeh3": "taeh3"}
_LEAF_FIELDS = frozenset(
    {"type", "checkpoint_path", "model_dir", "weights", "precision", "options"}
)
_PRECISIONS = ("fp32", "fp16", "bf16")
_LYNN_WEIGHTS = ("ema", "student")
_LYNN_OPTIONS = frozenset({"activation_checkpointing"})


class DecoderSpec(msgspec.Struct, frozen=True):
    """Effective decoder selection after legacy fields are folded in."""

    type: str
    checkpoint_path: str | None = None
    model_dir: str | None = None
    weights: str | None = None
    precision: str = "fp32"
    options: dict = {}

    def redacted(self) -> dict:
        """Loggable form: class and compute choices only, no filesystem paths."""
        return {
            "type": self.type,
            "weights": self.weights,
            "precision": self.precision,
            "options": dict(self.options),
        }


def normalize_decoder_spec(
    decoder_config: Mapping | None,
    *,
    video_decoder_backend: str | None,
    taeh3_checkpoint_path: str | None,
    taeh3_decode_precision: str | None,
    vae_decode_precision: str,
    explicit_legacy: frozenset[str] = frozenset(),
) -> DecoderSpec:
    """Fold the flat legacy fields and the nested config into one validated spec.

    `explicit_legacy` names the legacy fields the caller actually supplied, so a
    dataclass default can never be mistaken for a user choice.
    """
    legacy_type = _LEGACY_BACKEND_TYPES.get(video_decoder_backend)
    if video_decoder_backend is not None and legacy_type is None:
        raise ValueError("Unknown Zing video decoder backend")
    if decoder_config is None:
        # A flat TAEH3 path/precision still selects TAEH3; with no choice at all the
        # default is Lynn, whose missing checkpoint fails instead of falling back.
        if legacy_type is None and (taeh3_checkpoint_path or taeh3_decode_precision):
            legacy_type = "taeh3"
        spec = (
            _legacy_spec(
                legacy_type,
                taeh3_checkpoint_path,
                taeh3_decode_precision,
                vae_decode_precision,
            )
            if legacy_type is not None
            else DecoderSpec(type=DEFAULT_DECODER_TYPE, precision="bf16")
        )
    else:
        spec = _nested_spec(
            decoder_config,
            legacy_type=legacy_type,
            taeh3_checkpoint_path=taeh3_checkpoint_path,
            taeh3_decode_precision=taeh3_decode_precision,
            vae_decode_precision=vae_decode_precision,
            explicit_legacy=explicit_legacy,
        )
    _validate_spec(spec)
    return spec


def _legacy_spec(decoder_type, taeh3_path, taeh3_precision, vae_precision):
    if decoder_type == "taeh3":
        return DecoderSpec(
            type="taeh3",
            checkpoint_path=taeh3_path,
            precision=taeh3_precision or "fp32",
        )
    return DecoderSpec(type="official_h3", precision=vae_precision)


def _nested_spec(
    decoder_config,
    *,
    legacy_type,
    taeh3_checkpoint_path,
    taeh3_decode_precision,
    vae_decode_precision,
    explicit_legacy,
) -> DecoderSpec:
    if not isinstance(decoder_config, Mapping):
        raise ValueError("decoder_config must be a mapping")
    unknown = sorted(set(decoder_config) - _LEAF_FIELDS)
    if unknown:
        raise ValueError(f"Unknown decoder_config field: {unknown[0]}")
    options = decoder_config.get("options") or {}
    if not isinstance(options, Mapping):
        raise ValueError("decoder_config.options must be a mapping")
    decoder_type = decoder_config.get("type", DEFAULT_DECODER_TYPE)
    if decoder_type not in DECODER_TYPES:
        raise ValueError(f"Unknown Zing decoder type: {decoder_type!r}")

    explicit = explicit_legacy
    if (
        legacy_type is not None
        and "video_decoder_backend" in explicit
        and legacy_type != decoder_type
    ):
        raise _conflict("video_decoder_backend", "type")
    legacy_path = legacy_precision = None
    if decoder_type == "taeh3":
        legacy_path, legacy_precision = (
            taeh3_checkpoint_path,
            taeh3_decode_precision or "fp32",
        )
        _reject_conflict(
            decoder_config,
            "checkpoint_path",
            legacy_path,
            "taeh3_checkpoint_path",
            explicit,
        )
        _reject_conflict(
            decoder_config,
            "precision",
            legacy_precision,
            "taeh3_decode_precision",
            explicit,
        )
    else:
        for name in ("taeh3_checkpoint_path", "taeh3_decode_precision"):
            if name in explicit:
                raise _conflict(name, "type")
        if decoder_type == "official_h3":
            legacy_precision = vae_decode_precision
            _reject_conflict(
                decoder_config,
                "precision",
                legacy_precision,
                "vae_decode_precision",
                explicit,
            )
    precision = decoder_config.get("precision")
    if precision is None:
        precision = legacy_precision or (
            "bf16" if decoder_type == "lynn_streaming" else "fp32"
        )
    return DecoderSpec(
        type=decoder_type,
        checkpoint_path=decoder_config.get("checkpoint_path") or legacy_path,
        model_dir=decoder_config.get("model_dir"),
        weights=decoder_config.get("weights"),
        precision=precision,
        options=dict(options),
    )


def _reject_conflict(decoder_config, leaf, legacy_value, legacy_name, explicit):
    if (
        legacy_name in explicit
        and decoder_config.get(leaf) is not None
        and decoder_config[leaf] != legacy_value
    ):
        raise _conflict(legacy_name, leaf)


def _conflict(legacy_name, leaf):
    return ValueError(
        f"{legacy_name} conflicts with decoder_config.{leaf}; set only one of them"
    )


def _validate_spec(spec: DecoderSpec) -> None:
    if spec.type == "taeh3":
        if not spec.checkpoint_path:
            raise ValueError("TAEH3 requires a local checkpoint path")
        if spec.precision not in _PRECISIONS:
            raise ValueError("Unsupported TAEH3 decode precision")
        _require_unset(spec, "model_dir", "weights", "options")
    elif spec.type == "official_h3":
        if spec.precision not in _PRECISIONS:
            raise ValueError("Unsupported official_h3 decode precision")
        if spec.checkpoint_path:
            raise ValueError(
                "official_h3 reuses the exported video_vae; "
                "decoder_config.checkpoint_path is not supported yet"
            )
        _require_unset(spec, "model_dir", "weights", "options")
    else:
        _validate_lynn(spec)


def _validate_lynn(spec: DecoderSpec) -> None:
    if not spec.checkpoint_path or not spec.model_dir:
        raise ValueError(
            "lynn_streaming (the default decoder) requires decoder_config.checkpoint_path and decoder_config.model_dir"
        )
    if spec.weights is not None and spec.weights not in _LYNN_WEIGHTS:
        raise ValueError("Unsupported lynn_streaming weights selection")
    if spec.precision != "bf16":
        raise ValueError("Unsupported lynn_streaming precision")
    unknown = sorted(set(spec.options) - _LYNN_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown lynn_streaming option: {unknown[0]}")
    if spec.options.get("activation_checkpointing", False):
        raise ValueError(
            "lynn_streaming inference requires activation_checkpointing off"
        )


def _require_unset(spec: DecoderSpec, *names: str) -> None:
    for name in names:
        if getattr(spec, name):
            raise ValueError(f"{spec.type} does not support decoder_config.{name}")
