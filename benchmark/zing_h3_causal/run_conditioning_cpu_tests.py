# SPDX-License-Identifier: Apache-2.0
"""Run Issue #256 affected CPU tests without eager GPU model registries.

Only unavailable GPU import boundaries are replaced. Real conditioning,
realtime adapters, worker leaf modules, and existing CPU tensor tests run.
This harness does not validate CUDA loading, kernels, or collectives.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
os.environ["SGLANG_LIGHTWEIGHT_RUNTIME"] = "1"
sys.path.insert(0, str(Path.cwd() / "python"))
from unittest.mock import patch

from sglang._platform_stubs import install_platform_stubs

with patch("torch.backends.mps.is_available", return_value=True):
    install_platform_stubs()
# Avoid eager GPU registries during CPU leaf-module tests. Real leaf modules
# (including ServerArgs and pipeline) are still imported and executed.
import types

# CustomTestCase imports the unrelated LLM serving benchmark, whose model
# registry requires accelerator-only grammar libraries. Keep the real test
# framework while forbidding that external benchmark in this CPU harness.
benchmark_serving = types.ModuleType("sglang.benchmark.serving")


def forbidden_llm_benchmark(*args, **kwargs):
    raise RuntimeError("LLM serving benchmarks forbidden in conditioning CPU tests")


benchmark_serving.run_benchmark = forbidden_llm_benchmark
sys.modules[benchmark_serving.__name__] = benchmark_serving

for name, relative in [
    ("sglang.multimodal_gen.configs.pipeline_configs", "configs/pipeline_configs"),
    ("sglang.multimodal_gen.runtime.layers.attention", "runtime/layers/attention"),
    ("sglang.multimodal_gen.runtime.pipelines_core", "runtime/pipelines_core"),
    (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.realtime",
        "runtime/pipelines_core/stages/realtime",
    ),
]:
    module = types.ModuleType(name)
    module.__path__ = [str(Path.cwd() / "python/sglang/multimodal_gen" / relative)]
    sys.modules[name] = module
fsdp = types.ModuleType("sglang.multimodal_gen.runtime.loader.fsdp_load")


def forbidden_gpu_loader(*args, **kwargs):
    raise RuntimeError("GPU weight loading forbidden in CPU test harness")


fsdp.maybe_load_fsdp_model = forbidden_gpu_loader
sys.modules[fsdp.__name__] = fsdp
registry = types.ModuleType("sglang.multimodal_gen.registry")
registry.get_pipeline_config_classes = forbidden_gpu_loader
sys.modules[registry.__name__] = registry
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req

sys.modules["sglang.multimodal_gen.runtime.pipelines_core"].Req = Req
sys.modules["sglang.multimodal_gen.runtime.pipelines_core"].OutputBatch = OutputBatch

stages_name = "sglang.multimodal_gen.runtime.pipelines_core.stages"
stages = types.ModuleType(stages_name)
stages.__path__ = [
    str(Path.cwd() / "python/sglang/multimodal_gen/runtime/pipelines_core/stages")
]
sys.modules[stages_name] = stages
for class_name in [
    "DecodingStage",
    "DenoisingStage",
    "ImageEncodingStage",
    "ImageVAEEncodingStage",
    "InputValidationStage",
    "LatentPreparationStage",
    "PipelineStage",
    "TextEncodingStage",
    "TimestepPreparationStage",
]:
    setattr(stages, class_name, type(class_name, (), {}))
progressive_name = stages_name + ".progressive_resolution.denoising"
progressive = types.ModuleType(progressive_name)
progressive.ProgressiveDenoisingStage = type("ProgressiveDenoisingStage", (), {})
progressive.ProgressiveDenoisingStageRouter = type(
    "ProgressiveDenoisingStageRouter", (), {}
)
sys.modules[progressive_name] = progressive

# Managers evaluate pipeline type annotations at import time. Export the real
# CPU-importable base, and keep accelerator pipeline construction forbidden.
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)


class ForbiddenLoRAPipeline(ComposedPipelineBase):
    def __init__(self, *args, **kwargs):
        forbidden_gpu_loader(*args, **kwargs)


core = sys.modules["sglang.multimodal_gen.runtime.pipelines_core"]
core.ComposedPipelineBase = ComposedPipelineBase
core.LoRAPipeline = ForbiddenLoRAPipeline
core.build_pipeline = forbidden_gpu_loader


def attention_export(name):
    if name not in {"USPAttention", "LocalAttention", "Attention"}:
        raise AttributeError(name)
    import importlib

    return getattr(
        importlib.import_module("sglang.multimodal_gen.runtime.layers.attention.layer"),
        name,
    )


sys.modules[
    "sglang.multimodal_gen.runtime.layers.attention"
].__getattr__ = attention_export
import pytest

DEFAULT_TESTS = [
    "test_zing_conditioning_worker.py",
    "test_zing_conditioning_memory.py",
    "test_zing_conditioning_contract.py",
    "test_zing_conditioning_inputs.py",
    "conditioning",
    "realtime/test_zing_conditioning_lifecycle.py",
    "realtime/test_zing_h3_source_adapter.py",
    "realtime/test_zing_prefill_api.py",
    "realtime/test_zing_h3_realtime_adapter.py",
    "test_zing_h3_media_contract.py",
    "test_zing_h3_shared_primitives.py",
    "test_zing_h3_streaming_codecs.py",
    "realtime/test_zing_progressive_source_session.py",
]
arguments = sys.argv[1:] or [
    *(
        str(ROOT / "python/sglang/multimodal_gen/test/unit" / name)
        for name in DEFAULT_TESTS
    ),
    str(ROOT / "test/registered/unit/multimodal_gen"),
    "-q",
    "--disable-warnings",
]
if __name__ == "__main__":
    raise SystemExit(pytest.main(arguments))
