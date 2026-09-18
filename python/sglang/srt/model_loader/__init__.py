# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/model_loader/__init__.py

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig

_LAZY_EXPORTS = {
    "BaseModelLoader": ("sglang.srt.model_loader.loader", "BaseModelLoader"),
    "get_model_loader": ("sglang.srt.model_loader.loader", "get_model_loader"),
    "get_architecture_class_name": (
        "sglang.srt.model_loader.utils",
        "get_architecture_class_name",
    ),
    "get_model_architecture": (
        "sglang.srt.model_loader.utils",
        "get_model_architecture",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(importlib.import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def get_model(
    *,
    model_config: ModelConfig,
    load_config: LoadConfig,
    device_config: DeviceConfig,
) -> nn.Module:
    from sglang.srt.model_loader.loader import get_model_loader

    loader = get_model_loader(load_config, model_config)
    return loader.load_model(
        model_config=model_config,
        device_config=device_config,
    )


__all__ = [
    "get_model",
    "get_model_loader",
    "BaseModelLoader",
    "get_architecture_class_name",
    "get_model_architecture",
]
