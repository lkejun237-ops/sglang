# SPDX-License-Identifier: Apache-2.0
"""Read frozen first-block norm operands and installed kernel sources on CPU.

This diagnostic describes rounding candidates; it does not accept a replay,
change tolerances, import CUDA kernels, or modify the frozen experiment.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-inputs-dir", type=Path)
    args = parser.parse_args()
    root = args.root
    ready = json.loads((root / "CPU_READY.json").read_text())
    for name, digest in ready["controls"].items():
        if sha(Path(name)) != digest:
            raise ValueError("Frozen CPU control changed")
    sys.path[:0] = [
        str(root / "deps"),
        str(root / "source/python"),
        str(root / "source/benchmark/zing_h3_causal"),
    ]
    from frozen_source import verify_source

    verify_source(root / "source", ready["source_revision"])
    import torch

    if torch.cuda.device_count() or torch.cuda.is_initialized():
        raise ValueError("Norm readback requires zero CUDA devices")
    torch.set_num_threads(4)
    result = dict(source=ready["source_revision"], files={}, cells=[], kernels=[])
    result["reader_cache_environment"] = dict(
        scope="Zero-GPU reader environment, not final native/replay worker overrides",
        values={
            name: os.environ.get(name)
            for name in [
                "TORCHINDUCTOR_CACHE_DIR",
                "TRITON_CACHE_DIR",
                "SGLANG_DIFFUSION_CACHE_ROOT",
                "SGLANG_CACHE_DIR",
            ]
        },
    )
    result["native_cache_log_lines"] = {}
    for path in sorted((root / "work/results").glob("*.log")):
        lines = []
        with path.open(errors="replace") as stream:
            for line in stream:
                if "torch.compile cache: TORCHINDUCTOR_CACHE_DIR=" in line:
                    lines.append(line.strip())
        if lines:
            result["native_cache_log_lines"][str(path)] = lines
    if args.save_inputs_dir is not None:
        if args.save_inputs_dir.resolve().is_relative_to(root.resolve()):
            raise ValueError("Norm extracts must not modify the frozen experiment")
        args.save_inputs_dir.mkdir(parents=True, exist_ok=False)

    def load(path):
        result["files"][str(path)] = dict(bytes=path.stat().st_size, sha256=sha(path))
        return torch.load(path, map_location="cpu", weights_only=True)

    def digest_tensor(value):
        return hashlib.sha256(
            value.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()

    def stats(value):
        return dict(
            shape=list(value.shape),
            dtype=str(value.dtype),
            sha256=digest_tensor(value),
            min=value.min().item(),
            max=value.max().item(),
        )

    def compare(candidate, observed):
        same_sign = torch.signbit(candidate) == torch.signbit(observed)
        nonzero_sign_difference = ~same_sign & ((candidate != 0) | (observed != 0))
        distance = (
            candidate.view(torch.int16).int() - observed.view(torch.int16).int()
        ).abs()
        return dict(
            exact=torch.equal(candidate, observed),
            differing_elements=int((candidate != observed).sum()),
            max_abs=(candidate.float() - observed.float()).abs().max().item(),
            same_sign_max_bf16_ulp=int(distance[same_sign].max())
            if same_sign.any()
            else None,
            opposite_sign_nonzero_elements=int(nonzero_sign_difference.sum()),
        )

    for index, capture in enumerate(sorted((root / "work/captures").iterdir())):
        for rank in range(4):
            native = load(capture / f"rng/rank{rank}/drain-00000.pt")
            replay = load(
                root
                / f"work/results/replay-{index}/rank{rank}/rng/rank{rank}/drain-00000.pt"
            )
            hidden = native["first_body_trace"][0]["output"]["value"]
            actual = native["first_block_linear_trace"][0]["input"]["value"]
            other = replay["first_block_linear_trace"][0]["input"]["value"]
            tensors = native["first_fixed_execute"]["tensors"]
            indices = tensors["indices"].long()
            shift = tensors["projection.0.0"][indices]
            scale = tensors["projection.0.1"][indices]
            weight_info = native["model_parameter_samples"]["blocks.0.norm1.weight"]
            weight_sha = native["model_parameter_sha256"]["blocks.0.norm1.weight"]
            weight = torch.ones(
                weight_info["shape"],
                dtype=getattr(torch, weight_info["dtype"].split(".")[-1]),
            )
            row = dict(
                capture=str(capture),
                rank=rank,
                fused_native=native["first_block_fused_pre_linear"],
                fused_replay=replay["first_block_fused_pre_linear"],
                hidden=stats(hidden),
                hidden_runtime={
                    key: value
                    for key, value in native["first_body_trace"][0]["output"].items()
                    if key not in {"samples", "value"}
                },
                indices=stats(indices),
                shift=stats(shift),
                scale=stats(scale),
                norm_weight_sha256=weight_sha,
                norm_weight_runtime_flags_observed=False,
                unit_weight_verified=digest_tensor(weight) == weight_sha,
                differing_rows=[],
                candidates={},
                native_replay=compare(actual, other),
            )
            weight_verified = row["unit_weight_verified"]
            if not weight_verified:
                from safetensors import safe_open

                model = root / "model/transformer"
                index_file = model / "diffusion_pytorch_model.safetensors.index.json"
                weight_map = json.loads(index_file.read_text())["weight_map"]
                key = "backbone.transformer_blocks.0.norm1.weight"
                # Read only this norm weight, then prove its actual loaded
                # dtype and every byte. Checkpoint names alone are not proof.
                with safe_open(
                    str(model / weight_map[key]), framework="pt", device="cpu"
                ) as checkpoint:
                    weight = checkpoint.get_tensor(key).to(weight.dtype)
                weight_verified = digest_tensor(weight) == weight_sha
                row["checkpoint_weight_verified"] = weight_verified
            if not weight_verified:
                raise ValueError(
                    "Selected checkpoint norm weight differs from loaded parameter"
                )
            replay_tensors = replay["first_fixed_execute"]["tensors"]
            row["operands_equal"] = {
                "hidden": torch.equal(
                    hidden, replay["first_body_trace"][0]["output"]["value"]
                ),
                **{
                    name: torch.equal(tensors[name], replay_tensors[name])
                    for name in ["indices", "projection.0.0", "projection.0.1"]
                },
                "weight_sha256": weight_sha
                == replay["model_parameter_sha256"]["blocks.0.norm1.weight"],
                "eps": native["model_module_settings"]["blocks.0.norm1"]["scalars"][
                    "eps"
                ]
                == replay["model_module_settings"]["blocks.0.norm1"]["scalars"]["eps"],
            }
            if not all(row["operands_equal"].values()):
                raise ValueError("Native and replay norm operands differ")
            for value in indices.unique().tolist():
                mask = indices == value
                different = (actual != other)[mask]
                row["differing_rows"].append(
                    dict(
                        index=value, rows=int(mask.sum()), elements=int(different.sum())
                    )
                )
            # Both paths above require the actual parameter's full SHA.
            if weight_verified:
                eps = native["model_module_settings"]["blocks.0.norm1"]["scalars"][
                    "eps"
                ]
                x = hidden.float()
                norm = (
                    x
                    * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
                    * weight.float()
                )
                x64 = hidden.double()
                norm64 = (
                    x64
                    * torch.rsqrt(x64.square().mean(-1, keepdim=True) + eps)
                    * weight.double()
                )
                candidates = {
                    "fp64_rounded_reference": (
                        norm64 * (1 + scale.double()) + shift.double()
                    ).to(hidden.dtype),
                    "fp32_fused": (norm * (1 + scale.float()) + shift.float()).to(
                        hidden.dtype
                    ),
                    "bf16_norm_scale_add": norm.to(hidden.dtype) * (1 + scale) + shift,
                    "bf16_norm_fp32_modulate": (
                        norm.to(hidden.dtype).float() * (1 + scale.float())
                        + shift.float()
                    ).to(hidden.dtype),
                }
                for name, candidate in candidates.items():
                    row["candidates"][name] = {
                        label: compare(candidate, observed)
                        for label, observed in [("native", actual), ("replay", other)]
                    }
                if args.save_inputs_dir is not None:
                    path = args.save_inputs_dir / f"capture{index}-rank{rank}.pt"
                    torch.save(
                        dict(
                            hidden=hidden,
                            weight=weight,
                            shift=shift,
                            scale=scale,
                            eps=eps,
                            native_output=actual,
                            replay_output=other,
                            provenance=row,
                        ),
                        path,
                    )
                    row["extract"] = dict(
                        path=str(path), sha256=sha(path), bytes=path.stat().st_size
                    )
            result["cells"].append(row)
            del native, replay
    # Locate package files without importing the CUDA extension. Preserve the
    # actual installed Python implementations, not a workstation checkout.
    for entry in dict.fromkeys(sys.path):
        package = Path(entry) / "seedleap_kernels"
        if not package.is_dir():
            continue
        for path in sorted(package.rglob("*.py")):
            text = path.read_text()
            if (
                "compiled_norm" in text
                or "def fused_rmsnorm_indexed_scale_shift_quant_fp8" in text
            ):
                result["kernels"].append(
                    dict(path=str(path), sha256=sha(path), source=text)
                )
        break
    result["scope"] = (
        "CPU descriptive norm-boundary analysis; no numerical acceptance. "
        "CPU reduction, rsqrt and multiply-add need not match GPU rounding; "
        "candidate mismatch cannot rule out a GPU implementation."
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print("NORM_BOUNDARY_READBACK", len(result["cells"]), len(result["kernels"]))


if __name__ == "__main__":
    main()
