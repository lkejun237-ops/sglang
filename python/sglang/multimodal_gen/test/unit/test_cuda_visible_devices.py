"""GPU allocation identity must survive startup's NVML-only device queries."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.multimodal_gen.runtime.platforms import cuda


@pytest.fixture
def nvml():
    with patch.object(cuda, "pynvml") as api:
        cuda.NvmlCudaPlatform.get_device_total_memory.cache_clear()
        yield api
        cuda.NvmlCudaPlatform.get_device_total_memory.cache_clear()


def test_uuid_visibility_queries_the_allocated_physical_device(nvml):
    uuids = [
        "GPU-aaaaaaaa-1111-2222-3333-444444444444",
        "GPU-bbbbbbbb-1111-2222-3333-444444444444",
    ]
    handles = {uuids[0]: "physical-6", uuids[1]: "physical-3"}
    nvml.nvmlDeviceGetHandleByUUID.side_effect = handles.__getitem__
    nvml.nvmlDeviceGetIndex.side_effect = {"physical-6": 6, "physical-3": 3}.__getitem__
    nvml.nvmlDeviceGetHandleByIndex.side_effect = lambda index: f"physical-{index}"
    nvml.nvmlDeviceGetMemoryInfo.side_effect = lambda handle: SimpleNamespace(
        total={"physical-6": 288, "physical-3": 144}[handle] * 1024**3
    )
    with (
        patch.dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(uuids)),
        patch.object(
            cuda.torch.cuda,
            "init",
            side_effect=AssertionError("Do not initialize CUDA"),
        ),
    ):
        assert cuda.NvmlCudaPlatform.get_device_total_memory(0) == 288 * 1024**3
        assert cuda.NvmlCudaPlatform.get_device_total_memory(1) == 144 * 1024**3
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ",".join(uuids)
    assert nvml.nvmlInit.call_count == nvml.nvmlShutdown.call_count == 2


def test_unknown_uuid_fails_without_falling_back_to_another_gpu(nvml):
    nvml.nvmlDeviceGetHandleByUUID.side_effect = RuntimeError("Unknown UUID")
    with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="GPU-unavailable"):
        with pytest.raises(RuntimeError, match="Unknown UUID"):
            cuda.NvmlCudaPlatform.get_device_total_memory(0)
    nvml.nvmlDeviceGetHandleByIndex.assert_not_called()
    nvml.nvmlShutdown.assert_called_once()


def test_numeric_unset_and_disabled_visibility_remain_supported(nvml):
    with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="5, 2"):
        assert cuda.device_id_to_physical_device_id(0) == 5
        assert cuda.device_id_to_physical_device_id(1) == 2
    with patch.dict(os.environ):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        assert cuda.device_id_to_physical_device_id(1) == 1
    with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=""):
        with pytest.raises(RuntimeError, match="GPU support is disabled"):
            cuda.device_id_to_physical_device_id(0)
    nvml.nvmlDeviceGetHandleByUUID.assert_not_called()
