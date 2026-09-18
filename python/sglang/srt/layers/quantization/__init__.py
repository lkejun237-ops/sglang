# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import importlib
import os

from sglang.srt.layers.quantization.base_config import QuantizationConfig

_REGISTRY_MODULE = "sglang.srt.layers.quantization._registry"
__all__ = ["QuantizationConfig"]

# Lightweight diffusion never selects an SRT quantizer. Defining the registry
# containers here lets shared configuration modules inspect supported methods
# without importing every CUDA quantization backend.
BASE_QUANTIZATION_METHODS = {}
CPU_QUANTIZATION_METHODS = {}
QUANTIZATION_METHODS = {}


def _load_registry():
    registry = importlib.import_module(_REGISTRY_MODULE)
    for name in dir(registry):
        if not name.startswith("_"):
            globals()[name] = getattr(registry, name)
    return registry


def __getattr__(name: str):
    registry = _load_registry()
    try:
        return getattr(registry, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc


if os.environ.get("SGLANG_LIGHTWEIGHT_RUNTIME") != "1":
    _load_registry()
