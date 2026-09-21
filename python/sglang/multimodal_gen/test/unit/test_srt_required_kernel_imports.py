# SPDX-License-Identifier: Apache-2.0
"""Isolate GPU import branches so missing/ABI-broken kernels can be tested on CPU."""

import ast
import builtins
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch


@pytest.mark.parametrize("filename", ["activation.py", "layernorm.py"])
@pytest.mark.parametrize("failure", [None, ImportError, OSError])
def test_gpu_kernel_imports_fail_fast(monkeypatch, filename, failure):
    path = Path(__file__).resolve().parents[3] / "srt" / "layers" / filename
    tree = ast.parse(path.read_text())
    # Execute the actual platform import block, isolating unrelated serving
    # dependencies. This tests import failure propagation, not GPU arithmetic.
    block = next(
        n
        for n in tree.body
        if isinstance(n, ast.If)
        and any(
            isinstance(child, ast.ImportFrom) and child.module == "sgl_kernel"
            for child in ast.walk(n)
        )
    )
    kernel = SimpleNamespace(
        **{
            name: object()
            for name in (
                "gelu_and_mul",
                "gelu_tanh_and_mul",
                "silu_and_mul",
                "relu2",
                "fused_add_rmsnorm",
                "gemma_fused_add_rmsnorm",
                "gemma_rmsnorm",
                "rmsnorm",
            )
        }
    )
    original_import = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "sgl_kernel":
            if failure:
                raise failure("required kernel unavailable")
            return kernel
        if name == "sglang.kernels.ops.activation.activation":
            return kernel
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    namespace = dict(
        _is_cuda=True,
        _is_xpu=False,
        _is_musa=False,
        _is_flashinfer_available=False,
        Optional=Optional,
        torch=torch,
    )
    code = compile(ast.Module(body=[block], type_ignores=[]), str(path), "exec")
    if failure:
        with pytest.raises(failure, match="required kernel unavailable"):
            exec(code, namespace)
    else:
        exec(code, namespace)
        name = "_sgl_silu_and_mul" if filename == "activation.py" else "gemma_rmsnorm"
        assert namespace[name] is getattr(kernel, name.removeprefix("_sgl_"))
