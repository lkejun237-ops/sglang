# SPDX-License-Identifier: Apache-2.0
"""
Unit and contract tests for Zing H3 Fused Quantization Integration (Task A).

Covers:
1. apply_fp8_linear Per-Token input_prequantized support (fp8_utils.py)
2. Fp8LinearMethod.apply & RowParallelLinear.forward Tuple support (fp8.py, linear.py)
3. ZingLinearQuantConfig, SelectiveFp8Method & ZingFusedQuantWorkspace (zing_h3_quant.py)
4. zing_h3.py & minimax_h3.py Fused Operator Integration (5 hidden.shape[0] fixes,
   _block_forward fused operators, MiniMaxH3MLP fused swiglu dispatch)

Note:
When running on environments with PyTorch (DSW / CI), full functional tests execute.
On local Mac CPU without PyTorch, AST / static source contract inspection tests execute
to verify the exact syntactic and structural contracts without requiring CUDA or PyTorch.
"""

import ast
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(
    not HAS_TORCH, reason="requires torch; runs in Zing CPU CI/DSW"
)

REPO_ROOT = Path(__file__).resolve().parents[5]


# ==============================================================================
# Helper functions for AST / Source inspection
# ==============================================================================


def get_ast_function_source(file_path: Path, func_name: str) -> str:
    """Extract source lines of a top-level or method function from a Python file."""
    assert file_path.exists(), f"File does not exist: {file_path}"
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == func_name:
                start_line = node.lineno - 1
                end_line = node.end_lineno
                return "\n".join(lines[start_line:end_line])
    raise ValueError(f"Function {func_name} not found in {file_path}")


def get_ast_class_source(file_path: Path, class_name: str) -> str:
    """Extract source lines of a class from a Python file."""
    assert file_path.exists(), f"File does not exist: {file_path}"
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if node.name == class_name:
                start_line = node.lineno - 1
                end_line = node.end_lineno
                return "\n".join(lines[start_line:end_line])
    raise ValueError(f"Class {class_name} not found in {file_path}")


# ==============================================================================
# Suite 1: AST / Source Inspection Contract Tests (Runs on Mac CPU & CI)
# ==============================================================================


def test_ast_fp8_utils_accepts_per_token_scale():
    """Verify apply_fp8_linear relaxes input_scale.numel() == 1 to support per-token (M, 1) scale.

    Contract (Section 3.1):
    - is_per_tensor_scale: input_scale.numel() == 1 and input_scale.ndim <= 1
    - is_per_token_scale: input_scale.dtype == torch.float32 and input_scale.numel() == input_2d.shape[0]
    - Must NOT have hardcoded 'assert input_scale.numel() == 1'
    """
    file_path = REPO_ROOT / "python/sglang/srt/layers/quantization/fp8_utils.py"
    func_source = get_ast_function_source(file_path, "apply_fp8_linear")

    assert (
        "assert input_scale is not None and input_scale.numel() == 1" not in func_source
    ), (
        "apply_fp8_linear still has rigid 'assert input_scale is not None and input_scale.numel() == 1' check. "
        "Must allow per-token (M, 1) scale."
    )
    assert "is_per_token_scale" in func_source, (
        "apply_fp8_linear does not define or check is_per_token_scale."
    )
    assert "is_per_tensor_scale" in func_source, (
        "apply_fp8_linear does not distinguish is_per_tensor_scale vs is_per_token_scale."
    )


def test_ast_fp8_linear_method_supports_tuple():
    """Verify Fp8LinearMethod.apply checks isinstance(x, tuple) before is_contiguous and handles prequantized inputs.

    Contract (Section 3.2):
    - def apply(self, layer, x: torch.Tensor | tuple[torch.Tensor, ...], bias=None):
        if isinstance(x, tuple):
            qinput, x_scale = x[0], x[1]
            ...
            if self.block_quant:
                raise ValueError("Pre-quantized tuple input is only supported for non-block-quant FP8 linear")
            return apply_fp8_linear(input=qinput, ..., input_scale=x_scale, ...)
        if not x.is_contiguous():
            x = x.contiguous()
    """
    file_path = (
        REPO_ROOT / "python/sglang/multimodal_gen/runtime/layers/quantization/fp8.py"
    )
    apply_source = get_ast_function_source(file_path, "apply")

    assert (
        "Pre-quantized tuple input is not supported for FP8 Marlin linear"
        in apply_source
    ), (
        "Fp8LinearMethod.apply does not reject Marlin for tuple inputs with expected message."
    )
    assert "self.w8a8_block_fp8_linear" in apply_source, (
        "Fp8LinearMethod.apply does not handle block_quant tuple inputs."
    )
    assert apply_source.find("isinstance(x, tuple)") < apply_source.find(
        "if not x.is_contiguous():"
    ), (
        "isinstance(x, tuple) must be checked before 'if not x.is_contiguous():' to avoid AttributeError on tuple."
    )
    assert (
        "apply_fp8_linear" in apply_source and "input_scale=x_scale" in apply_source
    ), (
        "Fp8LinearMethod.apply does not forward prequantized scale x_scale to apply_fp8_linear."
    )


def test_ast_linear_weight_loader_v2_and_parallel_tuple():
    """Verify RowParallelLinear and WEIGHT_LOADER_V2_SUPPORTED support SelectiveFp8Method.

    Contract (Section 3.2):
    - WEIGHT_LOADER_V2_SUPPORTED contains 'SelectiveFp8Method'
    - RowParallelLinear.forward raises ValueError when not self.input_is_parallel and isinstance(input_, tuple)
    """
    file_path = REPO_ROOT / "python/sglang/multimodal_gen/runtime/layers/linear.py"
    source = file_path.read_text(encoding="utf-8")

    assert '"SelectiveFp8Method"' in source or "'SelectiveFp8Method'" in source, (
        "WEIGHT_LOADER_V2_SUPPORTED in linear.py must include 'SelectiveFp8Method'."
    )
    assert "Pre-quantized tuple input requires input_is_parallel=True" in source, (
        "RowParallelLinear.forward must enforce input_is_parallel=True for tuple inputs."
    )


def test_ast_zing_h3_quant_module():
    """Verify zing_h3_quant.py exists and implements ZingLinearQuantConfig and SelectiveFp8Method.

    Contract (Section 3.3):
    - python/sglang/multimodal_gen/runtime/models/dits/zing_h3_quant.py must exist
    - Exports ZingLinearQuantConfig, SelectiveFp8Method, selected_linears, ZingFusedQuantWorkspace
    - selected_linears for 'interior-qkv-fc1' enforces 2 <= i < num_layers - 3
    - ZingLinearQuantConfig explicitly rejects retired environment variables
    - SelectiveFp8Method delegates directly without zing_bf16_weight shadow
    """
    quant_file = (
        REPO_ROOT / "python/sglang/multimodal_gen/runtime/models/dits/zing_h3_quant.py"
    )
    assert quant_file.exists(), f"zing_h3_quant.py does not exist at {quant_file}."
    source = quant_file.read_text(encoding="utf-8")
    assert "class ZingLinearQuantConfig" in source, (
        "ZingLinearQuantConfig class missing from zing_h3_quant.py."
    )
    assert "class SelectiveFp8Method" in source, (
        "SelectiveFp8Method class missing from zing_h3_quant.py."
    )
    assert "class ZingFusedQuantWorkspace" in source, (
        "ZingFusedQuantWorkspace class missing from zing_h3_quant.py."
    )
    assert "def selected_linears" in source, (
        "selected_linears function missing from zing_h3_quant.py."
    )
    assert "num_layers - 3" in source or "num_layers-3" in source, (
        "selected_linears must exclude sensitive boundaries 2 <= i < num_layers - 3."
    )
    assert "SGLANG_ZING_QUANT_MIN_ROWS" in source and "is not supported" in source, (
        "ZingLinearQuantConfig must explicitly reject retired environment variables."
    )
    assert 'mode != "fp8"' in source or "mode != 'fp8'" in source, (
        "ZingLinearQuantConfig must restrict quantization mode strictly to fp8."
    )
    assert "zing_bf16_weight" not in source, (
        "SelectiveFp8Method must not maintain zing_bf16_weight shadow."
    )
    should_quantize_source = get_ast_function_source(
        quant_file, "should_quantize_activation"
    )
    assert "self.policy" in should_quantize_source, (
        "SelectiveFp8Method.should_quantize_activation must check self.policy."
    )
    assert "quant_config" in should_quantize_source, (
        "SelectiveFp8Method.should_quantize_activation must check self.backend.quant_config."
    )


def test_ast_zing_h3_shape_fixes_and_fused_ops():
    """Verify all 5 hidden.shape[0] sites are fixed in zing_h3.py, fused ops are called,
    and _has_seedleap_kernels guard is used.

    Contract (Section 3.4 & PR #414):
    - In _attention: all 4 occurrences of hidden.shape[0] replaced with total_rows
    - In _fixed_block_attention: 5th occurrence of hidden.shape[0] replaced with total_rows
    - In _block_forward: calls fused_rmsnorm_indexed_scale_shift_quant_fp8,
      fused_gated_residual_rmsnorm_indexed_scale_shift_quant_fp8,
      and fused_indexed_gated_residual_add without v[inputs.indices] expansion.
    - Defines _has_seedleap_kernels with @lru_cache(maxsize=1)
    - ZingH3DiTModel._should_fuse_pre_linear_quant calls _has_seedleap_kernels()
    """
    file_path = (
        REPO_ROOT / "python/sglang/multimodal_gen/runtime/models/dits/zing_h3.py"
    )
    attn_source = get_ast_function_source(file_path, "_attention")
    fixed_attn_source = get_ast_function_source(file_path, "_fixed_block_attention")
    block_fwd_source = get_ast_function_source(file_path, "_block_forward")
    has_kernel_source = get_ast_function_source(file_path, "_has_seedleap_kernels")
    should_fuse_source = get_ast_function_source(
        file_path, "_should_fuse_pre_linear_quant"
    )

    assert "hidden.shape[0]" not in attn_source, (
        "hidden.shape[0] still exists in ZingH3DiTModel._attention! "
        "Must use total_rows to support prequantized tuple inputs."
    )
    assert "hidden.shape[0]" not in fixed_attn_source, (
        "hidden.shape[0] still exists in ZingH3DiTModel._fixed_block_attention! "
        "Must use total_rows to support prequantized tuple inputs."
    )
    assert "fused_rmsnorm_indexed_scale_shift_quant_fp8" in block_fwd_source, (
        "ZingH3DiTModel._block_forward does not call fused_rmsnorm_indexed_scale_shift_quant_fp8."
    )
    assert (
        "fused_gated_residual_rmsnorm_indexed_scale_shift_quant_fp8" in block_fwd_source
    ), (
        "ZingH3DiTModel._block_forward does not call fused_gated_residual_rmsnorm_indexed_scale_shift_quant_fp8."
    )
    assert "fused_indexed_gated_residual_add" in block_fwd_source, (
        "ZingH3DiTModel._block_forward does not call fused_indexed_gated_residual_add."
    )
    assert "_has_seedleap_kernels" in should_fuse_source, (
        "ZingH3DiTModel._should_fuse_pre_linear_quant does not call _has_seedleap_kernels()."
    )
    assert "import seedleap_kernels.ops as ops" in has_kernel_source, (
        "_has_seedleap_kernels does not import seedleap_kernels.ops."
    )
    assert "fused_rmsnorm_indexed_scale_shift_quant_fp8" in has_kernel_source, (
        "_has_seedleap_kernels does not check fused_rmsnorm_indexed_scale_shift_quant_fp8."
    )


def test_ast_minimax_h3_mlp_fused_swiglu():
    """Verify MiniMaxH3MLP implements _should_fuse_swiglu_fp8 and dispatches single SwiGLU + FP8.

    Contract (Section 3.4 & PR #414):
    - MiniMaxH3MLP.forward checks self._should_fuse_swiglu_fp8(hidden)
    - When True: calls fused_swiglu_quant_fp8(hidden) and passes (qinput_fp8, x_scale) to fc2
    - Must NOT call _silu_mul when fused_swiglu_fp8 is active
    - Defines _has_seedleap_kernels with @lru_cache(maxsize=1)
    - _should_fuse_swiglu_fp8 checks hidden.requires_grad and _has_seedleap_kernels()
    """
    file_path = (
        REPO_ROOT / "python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py"
    )
    class_source = get_ast_class_source(file_path, "MiniMaxH3MLP")
    has_kernel_source = get_ast_function_source(file_path, "_has_seedleap_kernels")
    should_fuse_source = get_ast_function_source(file_path, "_should_fuse_swiglu_fp8")

    assert "_should_fuse_swiglu_fp8" in class_source, (
        "MiniMaxH3MLP does not define or check _should_fuse_swiglu_fp8."
    )
    assert "fused_swiglu_quant_fp8" in class_source, (
        "MiniMaxH3MLP does not call fused_swiglu_quant_fp8."
    )
    assert "import seedleap_kernels.ops as ops" in has_kernel_source, (
        "_has_seedleap_kernels does not import seedleap_kernels.ops."
    )
    assert "fused_swiglu_quant_fp8" in has_kernel_source, (
        "_has_seedleap_kernels does not check fused_swiglu_quant_fp8."
    )
    assert "hidden.requires_grad" in should_fuse_source, (
        "_should_fuse_swiglu_fp8 does not check hidden.requires_grad."
    )
    assert "_has_seedleap_kernels" in should_fuse_source, (
        "_should_fuse_swiglu_fp8 does not check _has_seedleap_kernels()."
    )


def test_ast_zing_h3_model_construction_with_fp8():
    """Verify ZingH3DiTModel accepts quant_config without raising ValueError
    and wraps quant_config in ZingLinearQuantConfig and forwards it to super().__init__.
    """
    file_path = (
        REPO_ROOT / "python/sglang/multimodal_gen/runtime/models/dits/zing_h3.py"
    )
    class_source = get_ast_class_source(file_path, "ZingH3DiTModel")

    assert (
        "quant_config is not None" not in class_source.split("raise ValueError")[0]
    ), "ZingH3DiTModel.__init__ must not reject quant_config is not None."
    assert "ZingLinearQuantConfig" in class_source, (
        "ZingH3DiTModel.__init__ must wrap quant_config with ZingLinearQuantConfig."
    )
    assert "quant_config=quant_config" in class_source, (
        "ZingH3DiTModel.__init__ must pass quant_config to super().__init__."
    )


def test_ast_transformer_loader_zing_fp8_support():
    """Verify transformer_loader.py restricts Zing AdaLN table to online FP8 only,
    defines prepare_zing_runtime_quant_config, and calls it to wrap quant_config.
    """
    loader_file = (
        REPO_ROOT
        / "python/sglang/multimodal_gen/runtime/loader/component_loaders/transformer_loader.py"
    )
    source = loader_file.read_text(encoding="utf-8")
    assert '!= "fp8"' in source, (
        "transformer_loader must restrict Zing AdaLN table to online fp8 only."
    )
    assert "def prepare_zing_runtime_quant_config" in source, (
        "transformer_loader must define prepare_zing_runtime_quant_config helper."
    )
    assert "prepare_zing_runtime_quant_config(" in source, (
        "transformer_loader must call prepare_zing_runtime_quant_config in load_customized."
    )
    assert "quant_spec.quant_config = ZingLinearQuantConfig" in source, (
        "prepare_zing_runtime_quant_config must wrap quant_spec.quant_config in ZingLinearQuantConfig."
    )


# ==============================================================================
# Suite 2: Functional Contract Tests (Executed when PyTorch is available)
# ==============================================================================


@requires_torch
def test_apply_fp8_linear_per_token_scale_contract():
    """Test apply_fp8_linear accepts 2D [M, 1] float32 scale with input_prequantized=True."""
    from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

    M, K, N = 4, 32, 64
    qinput = torch.zeros(M, K, dtype=torch.float8_e4m3fn)
    weight = torch.zeros(N, K, dtype=torch.float8_e4m3fn)
    x_scale = torch.ones(M, 1, dtype=torch.float32)
    weight_scale = torch.ones(1, N, dtype=torch.float32)

    with patch(
        "sglang.srt.layers.quantization.fp8_utils._apply_fallback_scaled_mm",
        return_value=torch.zeros(M, N, dtype=torch.bfloat16),
    ):
        out = apply_fp8_linear(
            input=qinput,
            weight=weight,
            weight_scale=weight_scale,
            input_scale=x_scale,
            cutlass_fp8_supported=True,
            use_per_token_if_dynamic=True,
        )
        assert out.shape == (M, N)


@requires_torch
def test_apply_fp8_linear_scale_validation_contract():
    """Test apply_fp8_linear rejects invalid scale shapes or dtypes for input_prequantized."""
    from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

    M, K, N = 4, 32, 64
    qinput = torch.zeros(M, K, dtype=torch.float8_e4m3fn)
    weight = torch.zeros(N, K, dtype=torch.float8_e4m3fn)
    weight_scale = torch.ones(1, N, dtype=torch.float32)

    # Invalid dtype: 1D scale of length M with float16
    invalid_dtype_scale = torch.ones(M, dtype=torch.float16)
    with pytest.raises(AssertionError):
        apply_fp8_linear(
            input=qinput,
            weight=weight,
            weight_scale=weight_scale,
            input_scale=invalid_dtype_scale,
        )

    # Invalid shapes: (1, M), (M, 2), (2, M // 2), (M, 1, 1)
    invalid_shapes = [
        (1, M),
        (M, 2),
        (2, M // 2),
        (M, 1, 1),
    ]
    for shape in invalid_shapes:
        invalid_scale = torch.ones(shape, dtype=torch.float32)
        with pytest.raises(AssertionError, match="Invalid input_scale shape"):
            apply_fp8_linear(
                input=qinput,
                weight=weight,
                weight_scale=weight_scale,
                input_scale=invalid_scale,
            )


@requires_torch
def test_fp8_linear_method_tuple_apply_contract():
    """Test Fp8LinearMethod.apply dispatches tuple (qinput, x_scale) to apply_fp8_linear."""
    from sglang.multimodal_gen.runtime.layers.quantization.fp8 import (
        Fp8LinearMethod,
    )

    method = Fp8LinearMethod(quant_config=MagicMock(weight_block_size=None))
    layer = MagicMock()
    layer.weight = torch.zeros(64, 32, dtype=torch.float8_e4m3fn)
    layer.weight_scale = torch.ones(1, 64, dtype=torch.float32)

    qinput = torch.zeros(4, 32, dtype=torch.float8_e4m3fn)
    x_scale = torch.ones(4, 1, dtype=torch.float32)

    with patch(
        "sglang.multimodal_gen.runtime.layers.quantization.fp8.apply_fp8_linear"
    ) as mock_apply:
        mock_apply.return_value = torch.zeros(4, 64, dtype=torch.bfloat16)
        _out = method.apply(layer, (qinput, x_scale))
        assert mock_apply.called
        call_kwargs = mock_apply.call_args[1]
        assert call_kwargs["input_scale"] is x_scale


@requires_torch
def test_row_parallel_linear_non_parallel_tuple_rejection():
    """Test RowParallelLinear raises ValueError on tuple when input_is_parallel=False."""
    from sglang.multimodal_gen.runtime.layers.linear import RowParallelLinear

    linear = object.__new__(RowParallelLinear)
    linear.input_is_parallel = False
    linear.quant_method = MagicMock()

    qinput = torch.zeros(4, 32, dtype=torch.float8_e4m3fn)
    x_scale = torch.ones(4, 1, dtype=torch.float32)

    with pytest.raises(
        ValueError, match="Pre-quantized tuple input requires input_is_parallel=True"
    ):
        linear.forward((qinput, x_scale))


@requires_torch
def test_zing_h3_quant_selected_linears_scopes():
    """Test selected_linears scope rules and sensitive layer exclusions."""
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_quant import (
        selected_linears,
    )

    # Scope 'ffn': all mlp.fc1 and mlp.fc2 across all layers
    ffn_sel = selected_linears(num_layers=10, scope="ffn", excludes=[])
    assert "blocks.0.mlp.fc1" in ffn_sel
    assert "blocks.0.mlp.fc2" in ffn_sel
    assert "blocks.0.attn.qkv_proj" not in ffn_sel

    # Scope 'interior-qkv-fc1': excludes layers 0, 1 and layers num_layers-3 .. num_layers-1
    interior_sel = selected_linears(
        num_layers=10, scope="interior-qkv-fc1", excludes=[]
    )
    # Layer 0 and 1 must NOT be in interior_sel
    assert "blocks.0.mlp.fc1" not in interior_sel
    assert "blocks.1.mlp.fc1" not in interior_sel
    assert "blocks.0.attn.qkv_proj" not in interior_sel
    # Interior layer 2 .. 6 must be in interior_sel
    assert "blocks.2.mlp.fc1" in interior_sel
    assert "blocks.2.attn.qkv_proj" in interior_sel
    assert "blocks.6.mlp.fc1" in interior_sel
    # Last 3 layers (7, 8, 9) must NOT be in interior_sel
    assert "blocks.7.mlp.fc1" not in interior_sel
    assert "blocks.8.mlp.fc1" not in interior_sel
    assert "blocks.9.mlp.fc1" not in interior_sel


@requires_torch
def test_zing_fused_quant_workspace_contract():
    """Test ZingFusedQuantWorkspace stream keying, preallocation, and freeze protections."""
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_quant import (
        ZingFusedQuantWorkspace,
    )

    ws = ZingFusedQuantWorkspace(max_bytes=1024 * 1024)
    dev = torch.device("cpu")

    # Preallocate capacity
    buf_fp8, scale_fp32 = ws.preallocate(dev, rows=128, width=64)
    assert buf_fp8.shape == (128, 64)
    assert scale_fp32.shape == (128, 1)

    # Freeze workspace
    ws.freeze()
    assert ws.frozen

    # Slicing within preallocated rows succeeds
    slice_fp8, slice_scale = ws.get_activation_buffer(dev, rows=64, width=64)
    assert slice_fp8.shape == (64, 64)
    assert slice_scale.shape == (64, 1)

    # Request exceeding capacity when frozen raises RuntimeError
    with pytest.raises(RuntimeError, match="capacity is frozen"):
        ws.get_activation_buffer(dev, rows=256, width=64)

    # Calling clear() while frozen raises RuntimeError
    with pytest.raises(RuntimeError):
        ws.clear()

    # Unfreeze and clear
    ws.unfreeze()
    ws.clear()


@requires_torch
def test_zing_h3_dit_attention_tuple_contract():
    """Test _attention accepts prequantized tuple without crashing on hidden.shape[0]."""
    from sglang.multimodal_gen.runtime.models.dits.zing_h3 import _attention

    module = MagicMock()
    module.qkv_proj = MagicMock(
        return_value=(torch.zeros(4, 96, dtype=torch.bfloat16), None)
    )
    module.out_proj = MagicMock(
        return_value=(torch.zeros(4, 32, dtype=torch.bfloat16), None)
    )
    module.num_heads = 1
    module.head_dim = 32
    module.softmax_scale = 1.0

    qinput = torch.zeros(4, 32, dtype=torch.float8_e4m3fn)
    x_scale = torch.ones(4, 1, dtype=torch.float32)

    with patch(
        "sglang.multimodal_gen.runtime.models.dits.zing_h3._qk_norm",
        side_effect=lambda m, q, k: (q, k),
    ):
        with patch(
            "torch.nn.functional.scaled_dot_product_attention",
            return_value=torch.zeros(1, 4, 32, dtype=torch.bfloat16),
        ):
            out, raw = _attention(
                module, (qinput, x_scale), attention_backend="torch_sdpa"
            )
            assert out.shape == (4, 32)


@requires_torch
def test_minimax_h3_mlp_fused_swiglu_contract():
    """Test MiniMaxH3MLP.forward calls fused_swiglu_quant_fp8 when _should_fuse_swiglu_fp8 is True."""
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3MLP

    mlp = object.__new__(MiniMaxH3MLP)
    mlp.reuse_fc1_activation = False
    mlp.fc1 = MagicMock(return_value=(torch.zeros(4, 64, dtype=torch.bfloat16), None))
    mlp.fc2 = MagicMock(return_value=(torch.zeros(4, 32, dtype=torch.bfloat16), None))

    # Mock _should_fuse_swiglu_fp8 and _apply_fused_swiglu_fp8
    mlp._should_fuse_swiglu_fp8 = MagicMock(return_value=True)
    fake_fp8 = torch.zeros(4, 32, dtype=torch.float8_e4m3fn)
    fake_scale = torch.ones(4, 1, dtype=torch.float32)
    mlp._apply_fused_swiglu_fp8 = MagicMock(return_value=(fake_fp8, fake_scale))

    x = torch.zeros(4, 32, dtype=torch.bfloat16)
    with patch(
        "sglang.multimodal_gen.runtime.models.dits.minimax_h3._silu_mul"
    ) as mock_silu:
        _out = mlp.forward(x)
        assert mlp._should_fuse_swiglu_fp8.called
        assert mlp._apply_fused_swiglu_fp8.called
        # Verify fc2 received the prequantized tuple
        fc2_arg = mlp.fc2.call_args[0][0]
        assert isinstance(fc2_arg, tuple)
        assert fc2_arg[0] is fake_fp8
        # Verify _silu_mul was NEVER called
        assert not mock_silu.called


@requires_torch
def test_minimax_h3_mlp_should_fuse_swiglu_fallback_guards():
    """Test _should_fuse_swiglu_fp8 fallback guards for requires_grad and missing seedleap-kernels."""
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
        MiniMaxH3MLP,
    )

    mlp = object.__new__(MiniMaxH3MLP)
    mlp.fc2 = MagicMock()
    quant_method = MagicMock()
    quant_method.should_quantize_activation.return_value = True
    mlp.fc2.quant_method = quant_method

    # Guard 1: hidden.requires_grad is True -> returns False
    hidden_grad = MagicMock()
    hidden_grad.is_cuda = True
    hidden_grad.dtype = torch.bfloat16
    hidden_grad.requires_grad = True
    hidden_grad.shape = [16, 5376]
    with patch(
        "sglang.multimodal_gen.runtime.models.dits.minimax_h3._has_seedleap_kernels",
        return_value=True,
    ):
        assert not mlp._should_fuse_swiglu_fp8(hidden_grad)

    # Guard 2: _has_seedleap_kernels is False -> returns False
    hidden_no_grad = MagicMock()
    hidden_no_grad.is_cuda = True
    hidden_no_grad.dtype = torch.bfloat16
    hidden_no_grad.requires_grad = False
    hidden_no_grad.shape = [16, 5376]
    with patch(
        "sglang.multimodal_gen.runtime.models.dits.minimax_h3._has_seedleap_kernels",
        return_value=False,
    ):
        assert not mlp._should_fuse_swiglu_fp8(hidden_no_grad)

    # All pass: returns True
    with patch(
        "sglang.multimodal_gen.runtime.models.dits.minimax_h3._has_seedleap_kernels",
        return_value=True,
    ):
        assert mlp._should_fuse_swiglu_fp8(hidden_no_grad)


@requires_torch
def test_has_seedleap_kernels_helper_resolution():
    """Test _has_seedleap_kernels properly handles module presence, absence, and errors."""
    import sys
    from types import ModuleType

    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
        _has_seedleap_kernels as minimax_has_kernels,
    )
    from sglang.multimodal_gen.runtime.models.dits.zing_h3 import (
        _has_seedleap_kernels as zing_has_kernels,
    )

    required_ops = (
        "fused_rmsnorm_indexed_scale_shift_quant_fp8",
        "fused_gated_residual_rmsnorm_indexed_scale_shift_quant_fp8",
        "fused_indexed_gated_residual_add",
        "fused_swiglu_quant_fp8",
    )

    for fn in (minimax_has_kernels, zing_has_kernels):
        fn.cache_clear()

        # Case 1: all required ops are present and callable
        mock_pkg = ModuleType("seedleap_kernels")
        mock_pkg.__path__ = []
        mock_ops = ModuleType("seedleap_kernels.ops")
        mock_pkg.ops = mock_ops
        for op in required_ops:
            setattr(mock_ops, op, lambda *args, **kwargs: None)
        with patch.dict(
            sys.modules,
            {
                "seedleap_kernels": mock_pkg,
                "seedleap_kernels.ops": mock_ops,
            },
        ):
            fn.cache_clear()
            assert fn() is True

        # Case 2: an operator is missing
        mock_pkg_missing = ModuleType("seedleap_kernels")
        mock_pkg_missing.__path__ = []
        mock_ops_missing = ModuleType("seedleap_kernels.ops")
        mock_pkg_missing.ops = mock_ops_missing
        for op in required_ops[:-1]:
            setattr(mock_ops_missing, op, lambda *args, **kwargs: None)
        with patch.dict(
            sys.modules,
            {
                "seedleap_kernels": mock_pkg_missing,
                "seedleap_kernels.ops": mock_ops_missing,
            },
        ):
            fn.cache_clear()
            assert fn() is False

        # Case 3: an operator is not callable
        mock_pkg_not_callable = ModuleType("seedleap_kernels")
        mock_pkg_not_callable.__path__ = []
        mock_ops_not_callable = ModuleType("seedleap_kernels.ops")
        mock_pkg_not_callable.ops = mock_ops_not_callable
        for op in required_ops[:-1]:
            setattr(mock_ops_not_callable, op, lambda *args, **kwargs: None)
        setattr(mock_ops_not_callable, required_ops[-1], "not_a_callable")
        with patch.dict(
            sys.modules,
            {
                "seedleap_kernels": mock_pkg_not_callable,
                "seedleap_kernels.ops": mock_ops_not_callable,
            },
        ):
            fn.cache_clear()
            assert fn() is False

        # Case 4: import error
        with patch.dict(
            sys.modules,
            {"seedleap_kernels": None, "seedleap_kernels.ops": None},
        ):
            fn.cache_clear()
            assert fn() is False

        fn.cache_clear()


def ensure_distributed_env_defaults() -> None:
    if "WORLD_SIZE" in os.environ:
        return
    os.environ.update(
        {
            "MASTER_ADDR": os.getenv("MASTER_ADDR", "127.0.0.1"),
            "MASTER_PORT": os.getenv("MASTER_PORT", "29505"),
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
        }
    )


@requires_torch
def test_zing_h3_model_construction_with_fp8(monkeypatch):
    """Test ZingH3DiTModel instantiates with ZingLinearQuantConfig and forwards tuple to fc1."""
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_ATTENTION", "torch_sdpa")
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_STABLE_SHAPES", "0")
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_COMPILED_ADALN", "0")
    monkeypatch.setenv("SGLANG_ZING_DENOISE_CUDA_GRAPH", "0")

    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
        ZingH3DiTArchConfig,
        ZingH3DiTConfig,
    )
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8LinearMethod
    from sglang.multimodal_gen.runtime.models.dits.zing_h3 import ZingH3DiTModel
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_quant import (
        SelectiveFp8Method,
        ZingLinearQuantConfig,
    )

    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

    arch = ZingH3DiTArchConfig(
        num_layers=2,
        token_refiner_num_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        attention_head_dim=8,
        ffn_hidden_size=32,
        latents_dim=2,
        audio_latents_dim=2,
        text_dim=12,
        timestep_input_dim=8,
        time_embed_hidden_size=16,
        time_embed_dim=8,
        adaln_out_features=384,
        final_adaln_out_features=32,
        rope_inv_freq_len=1,
    )

    backend = MagicMock()
    backend.get_name.return_value = "fp8"
    backend.checkpoint_uses_comfy_quantization = False
    backend.is_checkpoint_fp8_serialized = False
    backend.activation_scheme = "dynamic"
    backend.weight_block_size = None
    backend.kv_cache_quant_algo = None
    backend.ignored_layers = []

    backend_method = Fp8LinearMethod(quant_config=backend)
    backend_method.cutlass_fp8_supported = False
    backend_method.use_marlin = False
    backend_method.block_quant = False
    backend.get_quant_method.return_value = backend_method

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_capability", return_value=(9, 0)),
    ):
        quant_config = ZingLinearQuantConfig(backend, num_layers=2)

    model = ZingH3DiTModel(
        ZingH3DiTConfig(arch_config=arch), {}, quant_config=quant_config
    ).eval()

    fc1 = model.blocks[0].mlp.fc1
    # 1. Assert quant_method is SelectiveFp8Method
    assert isinstance(fc1.quant_method, SelectiveFp8Method)
    # 2. Assert should_quantize_activation(16) is True
    assert fc1.quant_method.should_quantize_activation(16) is True

    # 3. Call process_weights_after_loading: weight is torch.float8_e4m3fn
    with torch.no_grad():
        fc1.weight.copy_(torch.randn_like(fc1.weight))
    fc1.quant_method.process_weights_after_loading(fc1)
    assert fc1.weight.dtype == torch.float8_e4m3fn

    # 4. Forward pass with tuple passes tuple to fc1 / GEMM
    qinput = torch.zeros(4, 16, dtype=torch.float8_e4m3fn)
    x_scale = torch.ones(4, 1, dtype=torch.float32)

    with patch.object(
        backend_method,
        "apply",
        return_value=(torch.zeros(4, 64, dtype=torch.bfloat16), None),
    ) as mock_apply:
        _out = fc1((qinput, x_scale))
        assert mock_apply.called
        assert mock_apply.call_args[0][1] == (qinput, x_scale)


@requires_torch
def test_selective_fp8_method_should_quantize_activation():
    """Verify SelectiveFp8Method.should_quantize_activation checks policy and backend.quant_config."""
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_quant import (
        SelectiveFp8Method,
    )

    # Case 1: policy has get_name() == "fp8"
    mock_policy = MagicMock()
    mock_policy.get_name.return_value = "fp8"
    mock_backend = MagicMock(spec=[])  # no get_name
    method = SelectiveFp8Method(mock_backend, mock_policy, "blocks.0.mlp.fc1")
    assert method.should_quantize_activation(16) is True

    # Case 2: policy has no get_name, but backend.quant_config has get_name() == "fp8"
    mock_policy_no_name = object()
    mock_backend_with_config = MagicMock()
    mock_backend_with_config.quant_config.get_name.return_value = "fp8"
    method2 = SelectiveFp8Method(
        mock_backend_with_config, mock_policy_no_name, "blocks.0.mlp.fc1"
    )
    assert method2.should_quantize_activation(16) is True

    # Case 3: neither has fp8
    mock_backend_other = MagicMock()
    mock_backend_other.quant_config.get_name.return_value = "other"
    method3 = SelectiveFp8Method(
        mock_backend_other, mock_policy_no_name, "blocks.0.mlp.fc1"
    )
    assert method3.should_quantize_activation(16) is False


@requires_torch
def test_zing_linear_quant_config_rejects_retired_env_vars(monkeypatch):
    """Test ZingLinearQuantConfig explicitly rejects retired environment variables."""
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_quant import (
        ZingLinearQuantConfig,
    )

    retired_vars = (
        "SGLANG_ZING_QUANT_MIN_ROWS",
        "SGLANG_ZING_QUANT_SHAPES",
        "SGLANG_ZING_QUANT_WORKSPACE_BYTES",
        "SGLANG_ZING_QUANT_CENSUS",
    )

    backend = MagicMock()
    backend.get_name.return_value = "fp8"

    for env_var in retired_vars:
        monkeypatch.setenv(env_var, "1")
        with pytest.raises(ValueError, match=f"{env_var} is not supported"):
            ZingLinearQuantConfig(backend, num_layers=4)
        monkeypatch.delenv(env_var, raising=False)


@requires_torch
def test_zing_linear_quant_config_rejects_mxfp8():
    """Test ZingLinearQuantConfig rejects mxfp8 backend and restricts to fp8 only."""
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_quant import (
        ZingLinearQuantConfig,
    )

    backend = MagicMock()
    backend.get_name.return_value = "mxfp8"

    with pytest.raises(ValueError, match="Zing currently supports online fp8 only"):
        ZingLinearQuantConfig(backend, num_layers=4)


@requires_torch
def test_transformer_loader_quant_spec_wrapping():
    """Test prepare_zing_runtime_quant_config helper and TransformerLoader.load_customized wrapping."""
    from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
        TransformerLoader,
        prepare_zing_runtime_quant_config,
    )
    from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
        TransformerQuantLoadSpec,
    )
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_quant import (
        ZingLinearQuantConfig,
    )

    backend = MagicMock()
    backend.get_name.return_value = "fp8"
    backend.checkpoint_uses_comfy_quantization = False
    backend.is_checkpoint_fp8_serialized = False
    backend.activation_scheme = "dynamic"
    backend.weight_block_size = None
    backend.kv_cache_quant_algo = None
    backend.ignored_layers = []

    dit_config = MagicMock()
    dit_config.arch_config.num_layers = 4
    model_cls = type("ZingH3DiTModel", (), {})

    quant_spec = TransformerQuantLoadSpec(
        safetensors_list=[],
        quant_config=backend,
        nunchaku_config=None,
        param_dtype=torch.bfloat16,
    )
    init_params = {"quant_config": quant_spec.runtime_quant_config}

    # 1. Test prepare_zing_runtime_quant_config helper directly
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_capability", return_value=(9, 0)),
    ):
        prepare_zing_runtime_quant_config(
            model_cls=model_cls,
            quant_spec=quant_spec,
            dit_config=dit_config,
            init_params=init_params,
        )

    assert isinstance(quant_spec.quant_config, ZingLinearQuantConfig)
    assert quant_spec.runtime_quant_config is quant_spec.quant_config
    assert init_params["quant_config"] is quant_spec.quant_config

    # 2. Test TransformerLoader.load_customized calls prepare_zing_runtime_quant_config
    loader = TransformerLoader()
    mock_model = MagicMock()
    loader.load_state_dict_model = MagicMock(return_value=mock_model)

    server_args = MagicMock()
    server_args.model_paths = {}
    server_args.pipeline_config.dit_config = dit_config
    server_args.pipeline_config.resolve_adaln_table_path.return_value = None
    server_args.pipeline_config.zing_adaln_table_path = None
    server_args.direct_gpu_weight_loading = False
    server_args.should_use_fsdp_for_component.return_value = False
    server_args.should_start_component_on_cpu.return_value = False
    server_args.minimax_h3_adaln_cache_path = None
    server_args.minimax_h3_adaln_online = False

    fresh_quant_spec = TransformerQuantLoadSpec(
        safetensors_list=[],
        quant_config=backend,
        nunchaku_config=None,
        param_dtype=torch.bfloat16,
    )

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        patch(
            "sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader.get_diffusers_component_config",
            return_value={"_class_name": "ZingH3DiTModel"},
        ),
        patch(
            "sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader.resolve_transformer_gguf_to_load",
            return_value=None,
        ),
        patch(
            "sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader.resolve_transformer_checkpoint_files",
            return_value=MagicMock(safetensors=[], config_path=None),
        ),
        patch(
            "sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader.resolve_transformer_quant_load_spec",
            return_value=fresh_quant_spec,
        ),
        patch(
            "sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader.ModelRegistry.resolve_model_cls",
            return_value=(model_cls, None),
        ),
        patch(
            "sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader._server_args_for_transformer_component",
            return_value=server_args,
        ),
    ):
        result_model = loader.load_customized(
            component_model_path="/dummy/path",
            server_args=server_args,
            component_name="transformer",
        )

    assert result_model is mock_model
    assert isinstance(fresh_quant_spec.quant_config, ZingLinearQuantConfig)
    loader.load_state_dict_model.assert_called_once()
    called_init_params = loader.load_state_dict_model.call_args.kwargs["init_params"]
    assert isinstance(called_init_params["quant_config"], ZingLinearQuantConfig)
    assert called_init_params["quant_config"] is fresh_quant_spec.quant_config
