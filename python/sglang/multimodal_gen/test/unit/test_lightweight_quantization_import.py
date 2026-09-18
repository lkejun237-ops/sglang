import os
import subprocess
import sys
from pathlib import Path


def test_lightweight_quantization_package_defers_backend_registry():
    python_root = Path(__file__).resolve().parents[4]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(python_root)
    env["SGLANG_LIGHTWEIGHT_RUNTIME"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from sglang.srt.layers.quantization.base_config "
                "import QuantizationConfig; "
                "assert QuantizationConfig is not None; "
                "assert 'sglang.srt.layers.quantization._registry' "
                "not in sys.modules"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_checkpoint_quantization_import_defers_model_loader():
    python_root = Path(__file__).resolve().parents[4]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(python_root)
    env["SGLANG_LIGHTWEIGHT_RUNTIME"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from sglang.srt.model_loader.checkpoint_quantization "
                "import resolve_checkpoint_quant_spec; "
                "assert resolve_checkpoint_quant_spec({}) is None; "
                "assert 'sglang.srt.model_loader.loader' not in sys.modules"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_text_encoder_loader_defers_srt_compiled_kernels():
    python_root = Path(__file__).resolve().parents[4]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(python_root)
    env["SGLANG_LIGHTWEIGHT_RUNTIME"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from sglang.multimodal_gen.runtime.loader.component_loaders."
                "text_encoder_loader import TextEncoderLoader; "
                "assert TextEncoderLoader is not None; "
                "assert 'sglang.multimodal_gen.runtime.layers.quantization.gguf' "
                "not in sys.modules; "
                "assert 'sglang.srt.layers.quantization.gguf' not in sys.modules; "
                "assert 'sglang.srt.layers.quantization.fp8' not in sys.modules; "
                "assert 'sglang.srt.layers.modelopt_utils' not in sys.modules; "
                "assert 'sglang.srt.model_loader.post_load' not in sys.modules"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
