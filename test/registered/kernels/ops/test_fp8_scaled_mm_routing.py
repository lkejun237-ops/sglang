"""Unit tests for SM90 FP8 scaled GEMM routing policy (_prefer_torch_rowwise_fp8).

Validates:
1. Decode (M <= 64) strictly stays on CUTLASS Swap-AB.
2. Scalar scales, bias, or non-BF16 output dtypes stay on CUTLASS.
3. Wave-quantized Zing H3 shapes (mlp.fc2, attn.out_proj) at medium-to-large M
   route to Torch nvjet / cublasLt.
"""

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


class MockDevice:
    def __init__(self, type_="cuda", index=0):
        self.type = type_
        self.index = index

    def __eq__(self, other):
        return (
            isinstance(other, MockDevice)
            and self.type == other.type
            and self.index == other.index
        )


class MockTensor:
    def __init__(self, shape, dtype="float8_e4m3fn", stride=None, device=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device or MockDevice("cuda", 0)
        self.ndim = len(self.shape)
        self._contiguous = True
        if stride is not None:
            self._stride = tuple(stride)
        else:
            self._stride = self._compute_contiguous_stride(self.shape)

    @staticmethod
    def _compute_contiguous_stride(shape):
        stride = []
        cur = 1
        for s in reversed(shape):
            stride.append(cur)
            cur *= s
        return tuple(reversed(stride))

    def stride(self, dim):
        return self._stride[dim]

    def is_contiguous(self):
        return self._contiguous

    def numel(self):
        import math

        return math.prod(self.shape) if self.shape else 0


def _find_gemm_file():
    curr = Path(__file__).resolve().parent
    for _ in range(10):
        target = curr / "python/sglang/kernels/ops/gemm/__init__.py"
        if target.exists():
            return target
        curr = curr.parent
    raise FileNotFoundError(
        "Could not locate python/sglang/kernels/ops/gemm/__init__.py"
    )


def _setup_mock_torch():
    if "torch" not in sys.modules:
        torch = ModuleType("torch")
        torch.bfloat16 = "bfloat16"
        torch.float16 = "float16"
        torch.float32 = "float32"
        torch.float8_e4m3fn = "float8_e4m3fn"
        torch._scaled_mm = lambda *args, **kwargs: None
        torch.Tensor = MockTensor
        cuda_mod = ModuleType("torch.cuda")
        cuda_mod.get_device_capability = lambda dev=None: (9, 0)
        torch.cuda = cuda_mod
        sys.modules["torch"] = torch
        sys.modules["torch.cuda"] = cuda_mod
    else:
        torch = sys.modules["torch"]
        if not hasattr(torch, "_scaled_mm"):
            torch._scaled_mm = lambda *args, **kwargs: None
        if not hasattr(torch, "cuda"):
            cuda_mod = ModuleType("torch.cuda")
            cuda_mod.get_device_capability = lambda dev=None: (9, 0)
            torch.cuda = cuda_mod
            sys.modules["torch.cuda"] = cuda_mod


@pytest.fixture(autouse=True)
def _ensure_mock_torch(monkeypatch):
    if "torch" not in sys.modules:
        torch = ModuleType("torch")
        torch.bfloat16 = "bfloat16"
        torch.float16 = "float16"
        torch.float32 = "float32"
        torch.float8_e4m3fn = "float8_e4m3fn"
        torch._scaled_mm = lambda *args, **kwargs: None
        torch.Tensor = MockTensor
        cuda_mod = ModuleType("torch.cuda")
        cuda_mod.get_device_capability = lambda dev=None: (9, 0)
        torch.cuda = cuda_mod
        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.setitem(sys.modules, "torch.cuda", cuda_mod)
    else:
        torch = sys.modules["torch"]
        if not hasattr(torch, "_scaled_mm"):
            monkeypatch.setattr(
                torch, "_scaled_mm", lambda *args, **kwargs: None, raising=False
            )
        if not hasattr(torch, "cuda"):
            cuda_mod = ModuleType("torch.cuda")
            cuda_mod.get_device_capability = lambda dev=None: (9, 0)
            monkeypatch.setattr(torch, "cuda", cuda_mod, raising=False)
            monkeypatch.setitem(sys.modules, "torch.cuda", cuda_mod)


def _load_prefer_torch():
    had_torch = "torch" in sys.modules
    _setup_mock_torch()
    try:
        # Inject mock sglang.kernels scaffolding if not present
        for mod_name in [
            "sglang",
            "sglang.kernels",
            "sglang.kernels.fused_op",
            "sglang.kernels.registry",
            "sglang.kernels.selector",
            "sglang.kernels.spec",
        ]:
            if mod_name not in sys.modules:
                sys.modules[mod_name] = ModuleType(mod_name)

        class HashableReq:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = tuple(sorted(kwargs.items()))

            def __hash__(self):
                return hash((self.args, self.kwargs))

            def __eq__(self, other):
                return isinstance(other, HashableReq) and (self.args, self.kwargs) == (
                    other.args,
                    other.kwargs,
                )

        class AnyBackend:
            def __getattr__(self, name):
                return name.lower()

        sys.modules["sglang.kernels.spec"].CapabilityRequirement = SimpleNamespace(
            CUDA=HashableReq("cuda"),
            cuda=lambda **kw: HashableReq("cuda", **kw),
        )
        sys.modules["sglang.kernels.spec"].FormatSignature = lambda **kw: (
            SimpleNamespace(**kw)
        )
        sys.modules["sglang.kernels.spec"].KernelBackend = AnyBackend()
        sys.modules["sglang.kernels.spec"].KernelSpec = lambda **kw: SimpleNamespace(
            **kw
        )

        sys.modules["sglang.kernels.fused_op"].BaseFusedOp = object
        sys.modules["sglang.kernels.fused_op"].register_fused_op = lambda *a, **kw: (
            lambda cls: cls
        )
        sys.modules["sglang.kernels.registry"].register_kernel = lambda *a, **kw: (
            lambda fn: fn
        )
        sys.modules["sglang.kernels.selector"].get_kernel = lambda *a, **kw: None

        gemm_path = _find_gemm_file()
        spec = importlib.util.spec_from_file_location(
            "sglang.kernels.ops.gemm", str(gemm_path)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return (
            mod._prefer_torch_rowwise_fp8,
            mod.Fp8ScaledMMOp,
            mod._is_seedleap_fp8_sm90_available,
            mod._probe_seedleap_fp8_sm90_backend,
        )
    finally:
        if not had_torch:
            sys.modules.pop("torch", None)
            sys.modules.pop("torch.cuda", None)


(
    prefer_torch_rowwise_fp8,
    Fp8ScaledMMOp,
    _is_seedleap_fp8_sm90_available,
    _probe_seedleap_fp8_sm90_backend,
) = _load_prefer_torch()


def _make_tensors(m, k, n, *, x_scale_scalar=False, bias=None, out_dtype=None):
    torch = sys.modules["torch"]
    mat_a = MockTensor((m, k), dtype=torch.float8_e4m3fn, stride=(k, 1))
    mat_b = MockTensor((k, n), dtype=torch.float8_e4m3fn, stride=(1, k))
    if x_scale_scalar:
        scales_a = MockTensor((1,), dtype=torch.float32, stride=(1,))
    else:
        scales_a = MockTensor((m, 1), dtype=torch.float32, stride=(1, 1))
    scales_b = MockTensor((1, n), dtype=torch.float32, stride=(n, 1))
    out_dtype = out_dtype or torch.bfloat16
    return mat_a, mat_b, scales_a, scales_b, out_dtype, bias


@pytest.mark.parametrize("m", [1, 16, 32, 64])
@pytest.mark.parametrize(
    "linear,k,n",
    [
        ("attn.qkv_proj", 5376, 21504),
        ("attn.out_proj", 7168, 5376),
        ("mlp.fc1", 5376, 28672),
        ("mlp.fc2", 14336, 5376),
    ],
)
def test_sm90_decode_m_le_64_stays_on_cutlass(m, linear, k, n):
    """Small M <= 64 must never route to Torch NVJet.

    Preserves CUTLASS Swap-AB decode performance.
    """
    mat_a, mat_b, scales_a, scales_b, out_dtype, bias = _make_tensors(m, k, n)
    assert (
        prefer_torch_rowwise_fp8(mat_a, mat_b, scales_a, scales_b, out_dtype, bias)
        is False
    )


@pytest.mark.parametrize(
    "case",
    [
        "scalar_scale",
        "with_bias",
        "fp16_out",
        "non_contiguous_scale_a",
        "non_contiguous_scale_b",
    ],
)
def test_sm90_scalar_scale_or_bias_or_fp16_stays_on_cutlass(case):
    """Unsupported Torch rowwise configurations must stay on CUTLASS."""
    torch = sys.modules["torch"]
    m, k, n = 390, 14336, 5376

    if case == "scalar_scale":
        mat_a, mat_b, scales_a, scales_b, out_dtype, bias = _make_tensors(
            m, k, n, x_scale_scalar=True
        )
    elif case == "with_bias":
        bias_tensor = MockTensor((n,), dtype=torch.bfloat16, stride=(1,))
        mat_a, mat_b, scales_a, scales_b, out_dtype, bias = _make_tensors(
            m, k, n, bias=bias_tensor
        )
    elif case == "fp16_out":
        mat_a, mat_b, scales_a, scales_b, out_dtype, bias = _make_tensors(
            m, k, n, out_dtype=torch.float16
        )
    elif case == "non_contiguous_scale_a":
        mat_a, mat_b, scales_a, scales_b, out_dtype, bias = _make_tensors(m, k, n)
        scales_a._contiguous = False
    elif case == "non_contiguous_scale_b":
        mat_a, mat_b, scales_a, scales_b, out_dtype, bias = _make_tensors(m, k, n)
        scales_b._contiguous = False

    assert (
        prefer_torch_rowwise_fp8(mat_a, mat_b, scales_a, scales_b, out_dtype, bias)
        is False
    )


@pytest.mark.parametrize(
    "m", [195, 390, 460, 780, 880, 920, 1560, 1760, 1840, 3136, 3680]
)
@pytest.mark.parametrize(
    "linear,k,n",
    [
        ("mlp.fc2", 14336, 5376),
        ("attn.out_proj", 7168, 5376),
    ],
)
def test_sm90_zing_h3_wave_quantized_shapes_route_to_torch(m, linear, k, n):
    """Zing H3 narrow N deep K wave-quantized shapes at medium M route to Torch."""
    mat_a, mat_b, scales_a, scales_b, out_dtype, bias = _make_tensors(m, k, n)
    # Hardcoded m >= 8192 previously failed this assertion
    assert (
        prefer_torch_rowwise_fp8(mat_a, mat_b, scales_a, scales_b, out_dtype, bias)
        is True
    )


def _setup_mock_backends(
    monkeypatch,
    *,
    seedleap_installed=True,
    cutlass_root=Path("/mock/cutlass"),
    seedleap_side_effect=None,
    seedleap_return="mock_seedleap_tensor",
    sgl_return="mock_sgl_tensor",
):
    _probe_seedleap_fp8_sm90_backend.cache_clear()
    mock_sgl = ModuleType("sgl_kernel")
    mock_sgl.fp8_scaled_mm = MagicMock(return_value=sgl_return)
    monkeypatch.setitem(sys.modules, "sgl_kernel", mock_sgl)

    if not seedleap_installed:
        for mod_name in [
            "seedleap_kernels",
            "seedleap_kernels.backends",
            "seedleap_kernels.backends.cuda",
            "seedleap_kernels.backends.cuda.fp8_gemm_sm90",
            "seedleap_kernels.ops",
        ]:
            monkeypatch.setitem(sys.modules, mod_name, None)
        return None, mock_sgl

    sl_pkg = ModuleType("seedleap_kernels")
    sl_backends = ModuleType("seedleap_kernels.backends")
    sl_cuda = ModuleType("seedleap_kernels.backends.cuda")
    sl_sm90 = ModuleType("seedleap_kernels.backends.cuda.fp8_gemm_sm90")
    if cutlass_root is not None:
        sl_sm90._find_cutlass_root = MagicMock(return_value=cutlass_root)
    else:
        sl_sm90._find_cutlass_root = MagicMock(return_value=None)

    sl_ops = ModuleType("seedleap_kernels.ops")
    if seedleap_side_effect is not None:
        sl_ops.fp8_scaled_mm = MagicMock(side_effect=seedleap_side_effect)
    else:
        sl_ops.fp8_scaled_mm = MagicMock(return_value=seedleap_return)

    sl_pkg.backends = sl_backends
    sl_backends.cuda = sl_cuda
    sl_cuda.fp8_gemm_sm90 = sl_sm90
    sl_pkg.ops = sl_ops

    monkeypatch.setitem(sys.modules, "seedleap_kernels", sl_pkg)
    monkeypatch.setitem(sys.modules, "seedleap_kernels.backends", sl_backends)
    monkeypatch.setitem(sys.modules, "seedleap_kernels.backends.cuda", sl_cuda)
    monkeypatch.setitem(
        sys.modules, "seedleap_kernels.backends.cuda.fp8_gemm_sm90", sl_sm90
    )
    monkeypatch.setitem(sys.modules, "seedleap_kernels.ops", sl_ops)

    return sl_ops, mock_sgl


def test_sm90_forward_aot_bridges_to_seedleap_when_available(monkeypatch):
    """When SM90 and seedleap_kernels is usable, forward_aot routes to
    seedleap_kernels."""
    mock_sl_ops, mock_sgl = _setup_mock_backends(
        monkeypatch,
        seedleap_installed=True,
        cutlass_root=Path("/mock/cutlass"),
        seedleap_return="seedleap_out",
    )
    mat_a, mat_b, scales_a, scales_b, out_dtype, bias = _make_tensors(128, 5376, 5376)
    op = Fp8ScaledMMOp()
    res = op.forward_aot(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=bias)

    assert res == "seedleap_out"
    mock_sl_ops.fp8_scaled_mm.assert_called_once_with(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        bias=bias,
        out_dtype=out_dtype,
        backend="auto",
    )
    mock_sgl.fp8_scaled_mm.assert_not_called()


def test_sm90_forward_aot_falls_back_when_not_installed(monkeypatch):
    """When seedleap_kernels is not installed, forward_aot falls back to sgl_kernel."""
    _mock_sl_ops, mock_sgl = _setup_mock_backends(
        monkeypatch,
        seedleap_installed=False,
        sgl_return="sgl_fallback_out",
    )
    mat_a, mat_b, scales_a, scales_b, out_dtype, bias = _make_tensors(128, 5376, 5376)
    op = Fp8ScaledMMOp()
    res = op.forward_aot(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=bias)

    assert res == "sgl_fallback_out"
    mock_sgl.fp8_scaled_mm.assert_called_once_with(
        mat_a, mat_b, scales_a, scales_b, out_dtype, bias
    )


def test_sm90_forward_aot_falls_back_when_cutlass_missing(monkeypatch):
    """When CUTLASS root cannot be found, forward_aot falls back to sgl_kernel."""
    mock_sl_ops, mock_sgl = _setup_mock_backends(
        monkeypatch,
        seedleap_installed=True,
        cutlass_root=None,
        sgl_return="sgl_fallback_out",
    )
    mat_a, mat_b, scales_a, scales_b, out_dtype, bias = _make_tensors(128, 5376, 5376)
    op = Fp8ScaledMMOp()
    res = op.forward_aot(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=bias)

    assert res == "sgl_fallback_out"
    mock_sl_ops.fp8_scaled_mm.assert_not_called()
    mock_sgl.fp8_scaled_mm.assert_called_once_with(
        mat_a, mat_b, scales_a, scales_b, out_dtype, bias
    )


@pytest.mark.parametrize("val", ["1", "true", "yes", "TRUE", "Yes"])
def test_sm90_forward_aot_falls_back_when_disabled_via_env(monkeypatch, val):
    """When SGLANG_DISABLE_SEEDLEAP_KERNELS is set, forward_aot falls back
    to sgl_kernel."""
    monkeypatch.setenv("SGLANG_DISABLE_SEEDLEAP_KERNELS", val)
    mock_sl_ops, mock_sgl = _setup_mock_backends(
        monkeypatch,
        seedleap_installed=True,
        cutlass_root=Path("/mock/cutlass"),
        sgl_return="sgl_fallback_out",
    )
    mat_a, mat_b, scales_a, scales_b, out_dtype, bias = _make_tensors(128, 5376, 5376)
    op = Fp8ScaledMMOp()
    res = op.forward_aot(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=bias)

    assert res == "sgl_fallback_out"
    mock_sl_ops.fp8_scaled_mm.assert_not_called()
    mock_sgl.fp8_scaled_mm.assert_called_once_with(
        mat_a, mat_b, scales_a, scales_b, out_dtype, bias
    )


@pytest.mark.parametrize(
    "err_msg",
    [
        "CUTLASS 3.x headers could not be found",
        "Ninja is required for JIT compilation",
        "No C++ compiler found (nvcc)",
        "Compiler error: nvcc returned non-zero exit status",
    ],
)
def test_sm90_forward_aot_falls_back_on_runtime_jit_error(monkeypatch, caplog, err_msg):
    """When seedleap JIT compilation fails due to environmental error,
    fallback to sgl_kernel with warning."""
    mock_sl_ops, mock_sgl = _setup_mock_backends(
        monkeypatch,
        seedleap_installed=True,
        cutlass_root=Path("/mock/cutlass"),
        seedleap_side_effect=RuntimeError(err_msg),
        sgl_return="sgl_fallback_out",
    )
    mat_a, mat_b, scales_a, scales_b, out_dtype, bias = _make_tensors(128, 5376, 5376)
    op = Fp8ScaledMMOp()
    with caplog.at_level(logging.WARNING):
        res = op.forward_aot(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=bias)

    assert res == "sgl_fallback_out"
    mock_sl_ops.fp8_scaled_mm.assert_called_once()
    mock_sgl.fp8_scaled_mm.assert_called_once_with(
        mat_a, mat_b, scales_a, scales_b, out_dtype, bias
    )
    assert any(
        "seedleap-kernels SM90 FP8 backend unavailable at runtime" in record.message
        for record in caplog.records
    )


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("Kernel execution failed on SM90"),
        ValueError("Shape mismatch"),
    ],
)
def test_sm90_forward_aot_propagates_non_environmental_error(monkeypatch, exc):
    """Non-environmental runtime errors must NOT be swallowed and must be raised."""
    mock_sl_ops, mock_sgl = _setup_mock_backends(
        monkeypatch,
        seedleap_installed=True,
        cutlass_root=Path("/mock/cutlass"),
        seedleap_side_effect=exc,
    )
    mat_a, mat_b, scales_a, scales_b, out_dtype, bias = _make_tensors(128, 5376, 5376)
    op = Fp8ScaledMMOp()

    with pytest.raises(type(exc), match=str(exc)):
        op.forward_aot(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=bias)

    mock_sl_ops.fp8_scaled_mm.assert_called_once()
    mock_sgl.fp8_scaled_mm.assert_not_called()


def test_is_seedleap_fp8_sm90_available(monkeypatch):
    """Direct tests for _is_seedleap_fp8_sm90_available probe function."""
    _setup_mock_backends(
        monkeypatch,
        seedleap_installed=True,
        cutlass_root=Path("/mock/cutlass"),
    )
    assert _is_seedleap_fp8_sm90_available() is True

    # Missing CUTLASS root
    _setup_mock_backends(
        monkeypatch,
        seedleap_installed=True,
        cutlass_root=None,
    )
    assert _is_seedleap_fp8_sm90_available() is False

    # Disabled via env
    _setup_mock_backends(
        monkeypatch,
        seedleap_installed=True,
        cutlass_root=Path("/mock/cutlass"),
    )
    monkeypatch.setenv("SGLANG_DISABLE_SEEDLEAP_KERNELS", "1")
    assert _is_seedleap_fp8_sm90_available() is False
    monkeypatch.delenv("SGLANG_DISABLE_SEEDLEAP_KERNELS")

    # Not installed
    _setup_mock_backends(
        monkeypatch,
        seedleap_installed=False,
    )
    assert _is_seedleap_fp8_sm90_available() is False

    # Probe method raises unexpected Exception
    _setup_mock_backends(
        monkeypatch,
        seedleap_installed=True,
        cutlass_root=Path("/mock/cutlass"),
    )
    sys.modules[
        "seedleap_kernels.backends.cuda.fp8_gemm_sm90"
    ]._find_cutlass_root = MagicMock(side_effect=RuntimeError("filesystem error"))
    assert _is_seedleap_fp8_sm90_available() is False


def test_seedleap_probe_is_cached_and_only_probes_once(monkeypatch):
    """Static probe is cached with lru_cache and executes CUTLASS check once."""
    _setup_mock_backends(
        monkeypatch,
        seedleap_installed=True,
        cutlass_root=Path("/mock/cutlass"),
    )
    sl_sm90 = sys.modules["seedleap_kernels.backends.cuda.fp8_gemm_sm90"]
    _probe_seedleap_fp8_sm90_backend.cache_clear()
    for _ in range(50):
        assert _is_seedleap_fp8_sm90_available() is True
    assert sl_sm90._find_cutlass_root.call_count == 1


def test_sm90_forward_aot_non_sm90_capability_falls_back_directly(monkeypatch):
    """Device capability != (9, 0) (e.g. SM80) directly falls back to sgl_kernel."""
    mock_sl_ops, mock_sgl = _setup_mock_backends(
        monkeypatch,
        seedleap_installed=True,
        cutlass_root=Path("/mock/cutlass"),
        sgl_return="sgl_fallback_out",
    )
    torch = sys.modules["torch"]
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda dev=None: (8, 0))

    mat_a, mat_b, scales_a, scales_b, out_dtype, bias = _make_tensors(128, 5376, 5376)
    op = Fp8ScaledMMOp()
    res = op.forward_aot(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=bias)

    assert res == "sgl_fallback_out"
    mock_sl_ops.fp8_scaled_mm.assert_not_called()
    mock_sgl.fp8_scaled_mm.assert_called_once_with(
        mat_a, mat_b, scales_a, scales_b, out_dtype, bias
    )
