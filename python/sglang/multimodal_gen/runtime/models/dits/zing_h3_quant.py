# SPDX-License-Identifier: Apache-2.0
"""Opt-in Zing Linear quantization; all sensitive modules retain their precision."""

import fnmatch
import logging
import os
import threading

import torch

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)

logger = logging.getLogger(__name__)


def selected_linears(num_layers, scope, excludes=None):
    if excludes is None:
        excludes = ()
    if scope not in {"ffn", "fc1", "interior-qkv-fc1", "ffn-qkv", "ffn-attention"}:
        raise ValueError(
            "SGLANG_ZING_QUANT_SCOPE must be ffn, fc1, interior-qkv-fc1, "
            "ffn-qkv or ffn-attention"
        )
    suffixes = ["mlp.fc1", "mlp.fc2"]
    if scope == "fc1":
        suffixes = ["mlp.fc1"]
    if scope == "interior-qkv-fc1":
        suffixes = ["mlp.fc1", "attn.qkv_proj"]
    if scope == "ffn-qkv":
        suffixes += ["attn.qkv_proj"]
    if scope == "ffn-attention":
        suffixes += ["attn.qkv_proj", "attn.out_proj"]
    candidates = {
        f"blocks.{i}.{suffix}"
        for i in range(num_layers)
        if scope != "interior-qkv-fc1" or 2 <= i < num_layers - 3
        for suffix in suffixes
    }
    for pattern in excludes:
        if not any(fnmatch.fnmatchcase(name, pattern) for name in candidates):
            raise ValueError(
                f"Zing quantization exclusion matches no selected layer: {pattern}"
            )
    selected = {
        name
        for name in candidates
        if not any(fnmatch.fnmatchcase(name, p) for p in excludes)
    }
    if not selected:
        raise ValueError("Zing quantization must select at least one Linear")
    return selected


class ZingFusedQuantWorkspace:
    """Bounded, grow-only scratch for Zing FP8 activations and scales.

    Reuse on one CUDA stream is ordered by that stream; different streams have
    distinct storage. Graph users must preallocate every shape, then freeze
    before capture/replay.
    """

    def __init__(self, max_bytes: int | None = None):
        self.max_bytes = max_bytes if max_bytes is not None else float("inf")
        self.entries = {}
        self.bytes = 0
        self.reallocations = 0
        self.frozen = False
        self.lock = threading.RLock()

    def _get_stream_id(self, device: torch.device, stream=None):
        if stream is not None:
            return getattr(stream, "cuda_stream", stream)
        if torch.cuda.is_available() and device.type == "cuda":
            return torch.cuda.current_stream(device).cuda_stream
        return 0

    def preallocate(self, device, rows: int, width: int, stream=None):
        with self.lock:
            dev = torch.device(device)
            stream_id = self._get_stream_id(dev, stream)
            return self._reserve(dev, stream_id, rows, width)

    def get_activation_buffer(self, device, rows: int, width: int, stream=None):
        with self.lock:
            dev = torch.device(device)
            stream_id = self._get_stream_id(dev, stream)
            capturing = (
                (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
                if dev.type == "cuda"
                else False
            )
            return self._reserve(dev, stream_id, rows, width, capturing=capturing)

    def reserve(
        self, device, stream, rows: int, width: int, *, capturing: bool = False
    ):
        with self.lock:
            return self._reserve(
                torch.device(device), stream, rows, width, capturing=capturing
            )

    def _reserve(
        self,
        dev: torch.device,
        stream,
        rows: int,
        width: int,
        *,
        capturing: bool = False,
    ):
        key = (dev, stream, width)
        previous = self.entries.get(key)
        if previous is not None and previous[0].shape[0] >= rows:
            return previous[0][:rows], previous[1][:rows]
        if self.frozen or capturing:
            raise RuntimeError(
                "Zing FP8 workspace capacity is frozen; preallocate before CUDA Graph capture"
            )
        needed = rows * (width + 4)
        old = 0 if previous is None else previous[0].numel() + previous[1].numel() * 4
        if self.bytes - old + needed > self.max_bytes:
            raise RuntimeError("Zing FP8 workspace memory budget exceeded")
        activation = torch.empty((rows, width), device=dev, dtype=torch.float8_e4m3fn)
        scales = torch.empty((rows, 1), device=dev, dtype=torch.float32)
        self.entries[key] = (activation, scales)
        self.bytes += needed - old
        self.reallocations += 1
        return activation, scales

    def freeze(self):
        with self.lock:
            self.frozen = True

    def unfreeze(self):
        with self.lock:
            self.frozen = False

    def clear(self):
        with self.lock:
            if self.frozen:
                raise RuntimeError(
                    "Release CUDA Graph owners before clearing a frozen workspace"
                )
            self.entries.clear()
            self.bytes = 0


QuantWorkspace = ZingFusedQuantWorkspace


class ZingLinearQuantConfig(QuantizationConfig):
    """Restrict an existing online W8A8 backend to explicitly selected linears.

    Serialized low-precision checkpoints are deliberately rejected: the native
    checkpoint QKV/SwiGLU conversion runs before online weight quantization.
    """

    def __init__(self, backend, num_layers):
        super().__init__()
        for retired in (
            "SGLANG_ZING_QUANT_MIN_ROWS",
            "SGLANG_ZING_QUANT_SHAPES",
            "SGLANG_ZING_QUANT_WORKSPACE_BYTES",
            "SGLANG_ZING_QUANT_CENSUS",
        ):
            if os.environ.get(retired):
                raise ValueError(
                    f"{retired} is not supported; Zing FP8 quantizes all selected "
                    "linears unconditionally without shape fallback or workspace preallocation"
                )
        self.backend = backend
        mode = backend.get_name()
        if mode != "fp8":
            raise ValueError(
                "Zing currently supports online fp8 only; mxfp8/NVFP4 are not validated"
            )
        if getattr(backend, "is_checkpoint_fp8_serialized", False):
            raise ValueError("Zing quantization requires the original BF16 checkpoint")
        if getattr(backend, "activation_scheme", None) != "dynamic":
            raise ValueError("Zing quantization requires dynamic activation scaling")
        if getattr(backend, "weight_block_size", None) is not None:
            raise ValueError("Zing FP8 requires online per-tensor scaling")
        if getattr(backend, "kv_cache_quant_algo", None) is not None:
            raise ValueError("Zing quantization does not support quantized KV")
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in [
            (9, 0),
            (10, 0),
            (10, 3),
        ]:
            raise ValueError(
                "Experimental Zing quantization requires CUDA SM103 / SM90"
            )
        self.scope = os.environ.get("SGLANG_ZING_QUANT_SCOPE", "ffn")
        excludes = [
            p.strip()
            for p in os.environ.get("SGLANG_ZING_QUANT_EXCLUDE", "").split(",")
            if p.strip()
        ]
        excludes += list(getattr(backend, "ignored_layers", []))
        self.selected = selected_linears(num_layers, self.scope, excludes)
        self.coverage = {}

    def get_name(self):
        return self.backend.get_name()

    def get_supported_act_dtypes(self):
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 90

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config):
        raise ValueError("Zing layer policy wraps an online quantization config")

    def get_quant_method(self, layer, prefix):
        if not isinstance(layer, LinearBase):
            return None
        if prefix not in self.selected:
            self.coverage[prefix] = "unquantized: outside selected main-block linears"
            return UnquantizedLinearMethod()
        alignment = 16
        if layer.input_size % alignment or layer.output_size % alignment:
            raise ValueError(
                f"Zing fp8 layer {prefix} dimensions must be divisible by {alignment}"
            )
        method = self.backend.get_quant_method(layer, prefix)
        if (
            method is None
            or isinstance(method, UnquantizedLinearMethod)
            or getattr(method, "use_marlin", False)
        ):
            raise ValueError(
                f"Zing requires W8A8, but {prefix} selected an unsupported fallback"
            )
        kernel = (
            "cutlass_scaled_mm"
            if getattr(method, "cutlass_fp8_supported", False)
            else "torch._scaled_mm"
        )
        self.coverage[prefix] = f"fp8: {kernel}"
        return SelectiveFp8Method(method, self, prefix)

    def diagnostics(self):
        return {
            "kernel": "backend.apply",
            "coverage": dict(self.coverage),
        }

    def reset_diagnostics(self):
        pass

    def validate_loaded(self, model):
        missing = self.selected - self.coverage.keys()
        if missing:
            raise ValueError(
                f"Zing quantization selected missing layers: {sorted(missing)}"
            )
        for name in sorted(self.selected):
            module = model.get_submodule(name)
            if module.weight.dtype != torch.float8_e4m3fn:
                raise ValueError(f"Zing quantized weight {name} is not E4M3")
        for name, module in model.named_modules():
            if isinstance(module, LinearBase) and name not in self.coverage:
                self.coverage[name] = (
                    "unquantized: protected input/output/time projection"
                )
        logger.info(
            "Zing quantization mode=%s scope=%s coverage=%s fallback=disabled",
            self.get_name(),
            self.scope,
            self.coverage,
        )


class SelectiveFp8Method(LinearMethodBase):
    """Selective FP8 linear method delegating to the underlying FP8 backend."""

    def __init__(self, backend, policy, prefix):
        self.backend = backend
        self.policy = policy
        self.prefix = prefix

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def create_weights(self, *args, **kwargs):
        return self.backend.create_weights(*args, **kwargs)

    def should_quantize_activation(self, num_tokens: int = 0) -> bool:
        if getattr(self.policy, "get_name", lambda: None)() == "fp8":
            return True
        backend_config = getattr(self.backend, "quant_config", None)
        return getattr(backend_config, "get_name", lambda: None)() == "fp8"

    def process_weights_after_loading(self, layer):
        return self.backend.process_weights_after_loading(layer)

    def apply(self, layer, x, bias=None):
        return self.backend.apply(layer, x, bias=bias)
