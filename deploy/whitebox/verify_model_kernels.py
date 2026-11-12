"""Contract verification for prebuilt binary seedleap-kernels in model images."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import re
import site
import sysconfig
import tempfile
from pathlib import Path

PREBUILT_EXT_MODULE = "seedleap_kernels.backends.cuda._fp8_gemm_sm90_ext"

REQUIRED_OPS: tuple[str, ...] = (
    "fused_rmsnorm_indexed_scale_shift_quant_fp8",
    "fused_gated_residual_rmsnorm_indexed_scale_shift_quant_fp8",
    "fused_indexed_gated_residual_add",
    "fused_swiglu_quant_fp8",
    "block_sparse_attention",
    "fa4_attention",
    "fp8_scaled_mm",
)

REQUIRED_EXT_SYMBOLS: tuple[str, ...] = (
    "fp8_scaled_mm_sm90",
    "fp8_scaled_mm_sm90_by_config",
)


def _parse_torch_major_minor(version: str, label: str) -> str:
    base = version.split("+", 1)[0].strip()
    match = re.match(r"^(\d+)\.(\d+)", base)
    if match is None:
        raise RuntimeError(
            f"Unable to parse major.minor PyTorch version from {label}: {version!r}."
        )
    return f"{match.group(1)}.{match.group(2)}"


def _is_env_package_path(pkg_file: Path) -> bool:
    whitebox_root = Path("/opt/whitebox")
    resolved_pkg = pkg_file.resolve()
    if pkg_file.is_relative_to(whitebox_root) or resolved_pkg.is_relative_to(
        whitebox_root.resolve()
    ):
        return False

    if "site-packages" in pkg_file.parts or "dist-packages" in pkg_file.parts:
        return True
    if "site-packages" in resolved_pkg.parts or "dist-packages" in resolved_pkg.parts:
        return True

    candidate_roots: list[Path] = []
    for key in ("platlib", "purelib"):
        val = sysconfig.get_path(key)
        if val:
            candidate_roots.append(Path(val))
    try:
        for sp in site.getsitepackages():
            candidate_roots.append(Path(sp))
    except Exception:
        pass

    return any(
        resolved_pkg.is_relative_to(root.resolve())
        for root in candidate_roots
        if not root.is_relative_to(whitebox_root)
    )


def _atomic_update_manifest(
    manifest_file: Path, *, kernels_version: str, kernels_git_sha: str
) -> None:
    try:
        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to parse release manifest JSON at {manifest_file}: {exc}"
        ) from exc

    if not isinstance(manifest_data, dict):
        raise RuntimeError(
            f"Release manifest at {manifest_file} must be a JSON object."
        )

    manifest_data["kernels_version"] = kernels_version
    manifest_data["kernels_git_sha"] = kernels_git_sha

    existing_mode = manifest_file.stat().st_mode & 0o777
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=str(manifest_file.parent),
            prefix=f".{manifest_file.name}.",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            json.dump(manifest_data, tmp_file, indent=2)
            tmp_file.write("\n")
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.chmod(tmp_path, existing_mode)
        os.replace(tmp_path, manifest_file)
    except Exception:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def verify_model_kernels(
    manifest_path: Path | None = None,
    *,
    load_ext: bool = True,
) -> dict[str, object]:
    """Verify MODEL_BASE_IMAGE ships an AOT binary seedleap-kernels wheel."""
    jit_flag = os.environ.get("SEEDLEAP_KERNELS_DISABLE_JIT", "").strip().lower()
    if jit_flag not in {"1", "true", "yes"}:
        raise RuntimeError(
            "SEEDLEAP_KERNELS_DISABLE_JIT must be enabled ('1', 'true', or 'yes') "
            "in the production model image to forbid runtime nvcc/JIT compilation."
        )

    try:
        dist = importlib.metadata.distribution("seedleap-kernels")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "Package 'seedleap-kernels' is not installed in MODEL_BASE_IMAGE. "
            "Rebuild MODEL_BASE_IMAGE with the prebuilt binary seedleap-kernels wheel."
        ) from exc

    wheel_text = dist.read_text("WHEEL") or ""
    has_non_purelib = any(
        line.strip().lower() == "root-is-purelib: false"
        for line in wheel_text.splitlines()
    )
    if not has_non_purelib:
        raise RuntimeError(
            "Installed 'seedleap-kernels' distribution is pure-Python "
            "(missing 'Root-Is-Purelib: false' in WHEEL metadata); py3-none-any "
            "source wheels are forbidden in production. Rebuild MODEL_BASE_IMAGE "
            "with an AOT-compiled binary wheel (SEEDLEAP_KERNELS_BUILD_CUDA=1)."
        )

    try:
        seedleap_kernels = importlib.import_module("seedleap_kernels")
    except ImportError as exc:
        raise RuntimeError(
            f"Failed to import 'seedleap_kernels' from MODEL_BASE_IMAGE: {exc}"
        ) from exc

    raw_pkg_file = getattr(seedleap_kernels, "__file__", None)
    if not raw_pkg_file:
        raise RuntimeError(
            "Imported 'seedleap_kernels' has no __file__ attribute; expected an "
            "installed binary wheel under site-packages/dist-packages."
        )
    pkg_file = Path(raw_pkg_file)
    if not _is_env_package_path(pkg_file):
        raise RuntimeError(
            f"'seedleap_kernels' was imported from '{pkg_file}', which is either "
            "inside the '/opt/whitebox' source overlay or outside Python's "
            "site-packages/dist-packages environment."
        )

    try:
        ops = importlib.import_module("seedleap_kernels.ops")
    except ImportError as exc:
        raise RuntimeError(f"Failed to import 'seedleap_kernels.ops': {exc}") from exc

    missing_ops = [
        op_name for op_name in REQUIRED_OPS if not callable(getattr(ops, op_name, None))
    ]
    if missing_ops:
        raise RuntimeError(
            "seedleap_kernels.ops is missing required callable operators: "
            + ", ".join(missing_ops)
        )

    try:
        fp8_gemm_sm90 = importlib.import_module(
            "seedleap_kernels.backends.cuda.fp8_gemm_sm90"
        )
    except ImportError as exc:
        raise RuntimeError(
            f"Failed to import 'seedleap_kernels.backends.cuda.fp8_gemm_sm90': {exc}"
        ) from exc

    has_prebuilt_fn = getattr(fp8_gemm_sm90, "has_prebuilt_cutlass_ext", None)
    if not (callable(has_prebuilt_fn) and has_prebuilt_fn() is True):
        raise RuntimeError(
            f"Prebuilt CUDA extension '{PREBUILT_EXT_MODULE}' is missing. "
            "Rebuild MODEL_BASE_IMAGE with an AOT-compiled binary seedleap-kernels "
            "wheel built via SEEDLEAP_KERNELS_BUILD_CUDA=1."
        )

    try:
        build_info_mod = importlib.import_module("seedleap_kernels._build_info")
    except ImportError as exc:
        raise RuntimeError(
            "Missing 'seedleap_kernels._build_info' in installed seedleap-kernels "
            "wheel. Rebuild MODEL_BASE_IMAGE with an AOT-compiled binary wheel."
        ) from exc

    build_info = getattr(build_info_mod, "BUILD_INFO", None)
    if not isinstance(build_info, dict) or build_info.get("aot_cuda") is not True:
        raise RuntimeError(
            "seedleap_kernels._build_info.BUILD_INFO must declare 'aot_cuda': True."
        )

    wheel_torch = str(build_info.get("torch_version", ""))
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError(
            f"PyTorch ('torch') is not importable in MODEL_BASE_IMAGE: {exc}"
        ) from exc
    runtime_torch = str(getattr(torch, "__version__", ""))

    wheel_mm = _parse_torch_major_minor(wheel_torch, "BUILD_INFO['torch_version']")
    runtime_mm = _parse_torch_major_minor(runtime_torch, "torch.__version__")
    if wheel_mm != runtime_mm:
        raise RuntimeError(
            f"PyTorch ABI mismatch between prebuilt seedleap-kernels wheel "
            f"({wheel_torch!r}, major.minor {wheel_mm}) and runtime torch "
            f"({runtime_torch!r}, major.minor {runtime_mm}). Rebuild the binary "
            "seedleap-kernels wheel and MODEL_BASE_IMAGE against matching PyTorch."
        )

    if load_ext:
        load_fn = getattr(fp8_gemm_sm90, "_load_cutlass_ext", None)
        if not callable(load_fn):
            raise RuntimeError(
                "seedleap_kernels.backends.cuda.fp8_gemm_sm90._load_cutlass_ext "
                "is missing."
            )
        try:
            ext = load_fn()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to dlopen prebuilt extension '{PREBUILT_EXT_MODULE}': {exc}"
            ) from exc

        missing_symbols = [
            sym for sym in REQUIRED_EXT_SYMBOLS if not callable(getattr(ext, sym, None))
        ]
        if missing_symbols:
            raise RuntimeError(
                f"Prebuilt extension '{PREBUILT_EXT_MODULE}' is missing required "
                f"symbols: {', '.join(missing_symbols)}"
            )

    kernels_version = str(dist.version)
    raw_git_sha = build_info.get("git_sha")
    kernels_git_sha = (
        str(raw_git_sha).strip()
        if isinstance(raw_git_sha, str) and raw_git_sha.strip()
        else "unknown"
    )

    if manifest_path is not None:
        manifest_file = Path(manifest_path)
        if not manifest_file.is_file():
            raise RuntimeError(f"Release manifest file does not exist: {manifest_file}")
        _atomic_update_manifest(
            manifest_file,
            kernels_version=kernels_version,
            kernels_git_sha=kernels_git_sha,
        )

    return {
        "aot_cuda": True,
        "ext_loaded": bool(load_ext),
        "kernels_git_sha": kernels_git_sha,
        "kernels_version": kernels_version,
        "runtime_torch_version": runtime_torch,
        "wheel_torch_version": wheel_torch,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional release.json manifest path to enrich with kernels metadata.",
    )
    parser.add_argument(
        "--load-ext",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dlopen and verify exported symbols on the prebuilt SM90 extension.",
    )
    args = parser.parse_args(argv)
    summary = verify_model_kernels(args.manifest, load_ext=args.load_ext)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
