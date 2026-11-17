# SPDX-License-Identifier: Apache-2.0
"""Boundary gate keeping SGLang model runtime independent of extracted control-plane modules."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[6]
PYTHON_ROOT = REPO_ROOT / "python"
MM_ROOT = PYTHON_ROOT / "sglang/multimodal_gen"

EXTRACTED_MODULE_PREFIXES = (
    "sglang.multimodal_gen.runtime.realtime.whitebox",
    "sglang.multimodal_gen.apps.realtime_webui",
)
EXTRACTED_MODULES = frozenset(
    {
        "sglang.multimodal_gen.runtime.entrypoints.whitebox_room_server",
        "sglang.multimodal_gen.runtime.entrypoints.realtime_coordinator_server",
        "sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server",
        "sglang.multimodal_gen.runtime.entrypoints.realtime_gpu_scaler",
        "sglang.multimodal_gen.runtime.realtime.coordinator",
        "sglang.multimodal_gen.runtime.realtime.gateway",
        "sglang.multimodal_gen.runtime.realtime.world_platform",
        "sglang.multimodal_gen.runtime.realtime.world_directions",
        "sglang.multimodal_gen.runtime.realtime.world_gateway_diagnostics",
        "sglang.multimodal_gen.runtime.realtime.world_gateway_metrics",
        "sglang.multimodal_gen.runtime.realtime.world_recording",
        "sglang.multimodal_gen.runtime.realtime.trace_query",
        "sglang.multimodal_gen.runtime.realtime.player_lifecycle",
        "sglang.multimodal_gen.runtime.realtime.role_pool.admission",
        "sglang.multimodal_gen.runtime.realtime.role_pool.store",
        "sglang.multimodal_gen.runtime.realtime.role_pool.scheduler",
        "sglang.multimodal_gen.runtime.realtime.role_pool.api",
        "sglang.multimodal_gen.runtime.realtime.role_pool.admin",
        "sglang.multimodal_gen.runtime.realtime.role_pool.transport",
    }
)


def _module_name_for_file(py_file: Path) -> str:
    rel = py_file.relative_to(PYTHON_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _is_extracted_module(mod_name: str) -> bool:
    if mod_name in EXTRACTED_MODULES:
        return True
    return any(
        mod_name == prefix or mod_name.startswith(prefix + ".")
        for prefix in EXTRACTED_MODULE_PREFIXES
    )


def _iter_resolved_imports(py_file: Path, pkg_context: str):
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    is_pkg_init = py_file.name == "__init__.py"
    pkg_parts = pkg_context.split(".") if pkg_context else []
    if not is_pkg_init and pkg_parts:
        pkg_parts = pkg_parts[:-1]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                up = node.level - 1
                anchor = pkg_parts[: len(pkg_parts) - up] if up > 0 else pkg_parts
                base = ".".join(anchor)
                if node.module:
                    base = f"{base}.{node.module}" if base else node.module
            if base:
                yield node.lineno, base
            for alias in node.names:
                if alias.name != "*":
                    candidate = f"{base}.{alias.name}" if base else alias.name
                    yield node.lineno, candidate


def test_role_pool_partition_covers_all_modules_and_keeps_worker_agent_isolated():
    from sglang.multimodal_gen.runtime.realtime import role_pool

    role_pool_dir = MM_ROOT / "runtime/realtime/role_pool"
    actual = {
        path.stem for path in role_pool_dir.glob("*.py") if path.name != "__init__.py"
    }
    worker_set = set(role_pool.WORKER_AGENT_MODULES)
    control_set = set(role_pool.CONTROL_PLANE_MODULES)
    assert not (worker_set & control_set)
    assert worker_set | control_set == actual

    for name in sorted(worker_set):
        py_file = role_pool_dir / f"{name}.py"
        mod_name = _module_name_for_file(py_file)
        for lineno, target in _iter_resolved_imports(py_file, mod_name):
            assert not _is_extracted_module(target), (
                f"{py_file.relative_to(REPO_ROOT)}:{lineno} imports control-plane module {target}"
            )


def test_sglang_runtime_has_zero_imports_from_extracted_control_plane():
    violations: list[str] = []

    for py_file in sorted(MM_ROOT.rglob("*.py")):
        rel_mm = py_file.relative_to(MM_ROOT).as_posix()
        if rel_mm.startswith("test/") or "/test/" in rel_mm:
            continue
        mod_name = _module_name_for_file(py_file)
        if _is_extracted_module(mod_name):
            continue
        for lineno, target in _iter_resolved_imports(py_file, mod_name):
            if _is_extracted_module(target):
                violations.append(
                    f"{py_file.relative_to(REPO_ROOT)}:{lineno} -> {target}"
                )

    for extra_dir in (
        PYTHON_ROOT / "sglang/cli",
        REPO_ROOT / "deploy/sglang",
        REPO_ROOT / "benchmark/zing_h3_causal",
    ):
        if not extra_dir.exists():
            continue
        for py_file in sorted(extra_dir.rglob("*.py")):
            pkg_ctx = (
                _module_name_for_file(py_file)
                if py_file.is_relative_to(PYTHON_ROOT)
                else ""
            )
            for lineno, target in _iter_resolved_imports(py_file, pkg_ctx):
                if _is_extracted_module(target):
                    violations.append(
                        f"{py_file.relative_to(REPO_ROOT)}:{lineno} -> {target}"
                    )

    assert not violations, (
        "SGLang runtime/deploy/benchmark must not import Gateway/Coordinator/Room/Whitebox modules:\n"
        + "\n".join(violations)
    )


def test_lightweight_worker_agent_and_wire_modules_load_without_torch_or_control_plane():
    code = f"""
import importlib
import sys
import sglang.multimodal_gen.runtime.realtime.zing_contracts
import sglang.multimodal_gen.runtime.realtime.chunk_telemetry
from sglang.multimodal_gen.runtime.realtime import role_pool

for name in role_pool.WORKER_AGENT_MODULES:
    importlib.import_module(f"sglang.multimodal_gen.runtime.realtime.role_pool.{{name}}")

assert "torch" not in sys.modules, "torch must not be imported in lightweight mode"
extracted_prefixes = {EXTRACTED_MODULE_PREFIXES!r}
extracted_modules = {sorted(EXTRACTED_MODULES)!r}
forbidden = [
    m for m in sys.modules
    if m in extracted_modules
    or any(m == p or m.startswith(p + ".") for p in extracted_prefixes)
    or any(
        m == f"sglang.multimodal_gen.runtime.realtime.role_pool.{{c}}"
        for c in role_pool.CONTROL_PLANE_MODULES
    )
]
assert not forbidden, f"Forbidden control-plane modules loaded: {{forbidden}}"
"""
    env = {
        **os.environ,
        "PYTHONPATH": str(PYTHON_ROOT),
        "SGLANG_LIGHTWEIGHT_RUNTIME": "1",
    }
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_relocated_prompt_cache_and_telemetry_helpers_preserve_contract():
    from sglang.multimodal_gen.runtime.realtime.chunk_telemetry import (
        CHUNK_FIELDS,
        h3_host_durations,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.cache_receipt import (
        public_cache_receipt as wb_cache_receipt,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        MAX_PROMPT_BYTES,
        MAX_PROMPT_CHARS,
        prompt_text,
    )
    from sglang.multimodal_gen.runtime.realtime.zing_contracts import (
        ContractError,
        public_cache_receipt,
        validate_prompt_text,
    )

    assert validate_prompt_text("a snowy mountain") == prompt_text("a snowy mountain")
    for invalid in (
        None,
        123,
        "",
        "   ",
        "a" * (MAX_PROMPT_CHARS + 1),
        "\u96ea" * (MAX_PROMPT_BYTES // 3 + 1),
    ):
        with pytest.raises(ContractError):
            validate_prompt_text(invalid)
        with pytest.raises(ValueError):
            prompt_text(invalid)

    sample_report = {
        "cache_key": "a" * 64,
        "prompt_version": 2,
        "cache_result": "l1_hit",
        "miss_reason": None,
        "lookup": {"internal_only": True},
        "cuda_ms": {"restore": 1.25},
        "elapsed_host_ms": 4.5678,
        "stages": {"restore": {"host_ms": 1.2345, "status": "executed"}},
    }
    sanitized = public_cache_receipt(sample_report)
    assert sanitized == wb_cache_receipt(sample_report)
    assert "lookup" not in sanitized
    assert "cuda_ms" not in sanitized

    durations = h3_host_durations(
        {"zing_source_stage_host_ms": {"prepare_host_ms": 2.3456, "ignored": 9.0}}
    )
    assert durations == {"h3_prepare_host_ms": 2.346}
    assert "h3_prepare_host_ms" in CHUNK_FIELDS


def test_relocated_helpers_preserve_l3_provenance_and_ttff_allowlists():
    from sglang.multimodal_gen.runtime.realtime import chunk_telemetry, zing_contracts
    from sglang.multimodal_gen.runtime.realtime.whitebox import cache_receipt, telemetry

    raw = {
        "cache_key": "a" * 64,
        "prompt_version": 3,
        "cache_result": "l3_hit",
        "lookup": {"artifact_id": "b" * 32, "source_layer": "l3", "secret": "drop"},
        "timings": {"cuda_ms": {"restore": 1.25}},
    }
    receipt = zing_contracts.public_cache_receipt(raw)
    assert cache_receipt.public_cache_receipt is zing_contracts.public_cache_receipt
    assert receipt["cache_result"] == "l3_hit"
    assert receipt["source_layer"] == "l3"
    assert receipt["artifact_id"] == "b" * 32
    assert receipt["timings"]["cuda_ms"] == {"restore": 1.25}
    assert "lookup" not in receipt
    assert zing_contracts.public_cache_receipt(receipt) == receipt
    for field in (
        "TTFF_HOST_STAGES",
        "TTFF_CUDA_STAGES",
        "TTFF_METRIC_STAGES",
        "CLI_STARTUP_STAGES",
    ):
        assert getattr(telemetry, field) is getattr(chunk_telemetry, field)
    assert (
        "whitebox_ttff_prompt_switch_encoder_host_condition_cache_lookup"
        in chunk_telemetry.TTFF_METRIC_STAGES
    )
    assert "whitebox_cli_worldLoadMs" in chunk_telemetry.CLI_STARTUP_STAGES
