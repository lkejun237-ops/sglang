# SPDX-License-Identifier: Apache-2.0
"""CPU-only comparison of trusted native/replay RNG observations and latents."""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-for-collection-seconds", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.wait_for_collection_seconds <= 3600:
        parser.error("Collection wait must be between zero and 3600 seconds")
    root = args.root
    ready = json.loads((root / "CPU_READY.json").read_text())
    for path, digest in ready["controls"].items():
        if sha(path) != digest:
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
        raise ValueError("RNG readback must run with zero CUDA devices")
    job = json.loads((root / "job-config.json").read_text())
    result_path = Path(job["output"]) / "GPU_RESULT.json"
    deadline = time.monotonic() + args.wait_for_collection_seconds
    while not result_path.exists() and time.monotonic() < deadline:
        time.sleep(min(5, max(0, deadline - time.monotonic())))
    result = json.loads(result_path.read_text())
    if not result["collection_completed"]:
        raise ValueError("RNG diagnostic collection failed; inspect original failure")
    output = dict(
        source_revision=ready["source_revision"],
        analyzer_sha256=sha(__file__),
        full_matrix_accepted=False,
        files={},
        cells=[],
    )

    def load(path, *, primitive=False):
        output["files"][str(path)] = dict(bytes=path.stat().st_size, sha256=sha(path))
        return torch.load(path, map_location="cpu", weights_only=primitive)

    def equal(a, b):
        return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)

    def max_abs(a, b):
        # First-block Linear traces can be large. Keep comparison temporaries
        # bounded in the zero-GPU reader, rather than making full FP32 copies.
        maximum = 0.0
        for left, right in zip(
            a.reshape(-1).split(1 << 20), b.reshape(-1).split(1 << 20), strict=True
        ):
            value = (left.float() - right.float()).abs().max().item()
            if math.isnan(value):
                return value
            maximum = max(maximum, value)
        return maximum

    def differences(a, b, path=""):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if equal(a, b):
                return []
            return [
                dict(
                    path=path,
                    native_shape=list(a.shape),
                    replay_shape=list(b.shape),
                    native_dtype=str(a.dtype),
                    replay_dtype=str(b.dtype),
                    max_abs=max_abs(a, b) if a.shape == b.shape and a.numel() else None,
                )
            ]
        if type(a) is not type(b):
            return [
                dict(
                    path=path,
                    native_type=type(a).__name__,
                    replay_type=type(b).__name__,
                )
            ]
        if isinstance(a, dict):
            rows = []
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    rows.append(
                        dict(
                            path=f"{path}.{key}",
                            missing="native" if key not in a else "replay",
                        )
                    )
                else:
                    rows.extend(differences(a[key], b[key], f"{path}.{key}"))
            return rows
        if isinstance(a, (list, tuple)):
            if len(a) != len(b):
                return [dict(path=path, native_length=len(a), replay_length=len(b))]
            return [
                row
                for index, (x, y) in enumerate(zip(a, b))
                for row in differences(x, y, f"{path}[{index}]")
            ]
        return [] if a == b else [dict(path=path, native=a, replay=b)]

    captures = sorted((root / "work/captures").iterdir())
    if len(captures) != 2:
        raise ValueError("Expected two native capture sessions")
    for index, capture in enumerate(captures):
        session = load(capture / "session.pt")
        expected = [
            load(capture / f"chunk-{i:05d}.pt")["expected_visual"] for i in range(3)
        ]
        for rank in range(4):
            replay = root / "work/results" / f"replay-{index}" / f"rank{rank}"
            actual = load(replay / "replay-latents.pt", primitive=True)
            native_files = sorted((capture / "rng" / f"rank{rank}").glob("drain-*.pt"))
            replay_files = sorted((replay / "rng" / f"rank{rank}").glob("drain-*.pt"))
            if len(native_files) != 3 or len(replay_files) != 3 or len(actual) != 3:
                raise ValueError(
                    "Missing complete first-three-drain diagnostic evidence"
                )
            cell = dict(
                capture=str(capture),
                geometry=[session["params"].width, session["params"].height],
                rank=rank,
                drains=[],
                output=[],
            )
            for before_path, after_path in zip(native_files, replay_files):
                native, standalone = (
                    load(before_path, primitive=True),
                    load(after_path, primitive=True),
                )
                row = dict(
                    native_index=native["index"],
                    replay_index=standalone["index"],
                    native_chunks=native["ready_chunks"],
                    replay_chunks=standalone["ready_chunks"],
                    native_noise_count=len(native["noise"]),
                    replay_noise_count=len(standalone["noise"]),
                    native_stream_count=len(native["streams"]),
                    replay_stream_count=len(standalone["streams"]),
                    model_parameter_dtypes_equal=native["model_parameter_dtypes"]
                    == standalone["model_parameter_dtypes"],
                    native_tf32=native["tf32"],
                    replay_tf32=standalone["tf32"],
                    streams=[],
                    noise=[],
                )
                for a, b in zip(native["streams"], standalone["streams"]):
                    row["streams"].append(
                        dict(
                            seed_equal=a["seed"] == b["seed"],
                            **{
                                f"{phase}_{device}_equal": equal(
                                    a[phase][device], b[phase][device]
                                )
                                for phase in ["before", "after"]
                                for device in ["cpu", "cuda"]
                            },
                        )
                    )
                for a, b in zip(native["noise"], standalone["noise"]):
                    same_shape = a["value"].shape == b["value"].shape
                    row["noise"].append(
                        dict(
                            equal=equal(a["value"], b["value"]),
                            native_shape=list(a["value"].shape),
                            replay_shape=list(b["value"].shape),
                            max_abs=(a["value"].float() - b["value"].float())
                            .abs()
                            .max()
                            .item()
                            if same_shape
                            else None,
                        )
                    )
                if native["index"] == 0 or standalone["index"] == 0:
                    # Loading can promote a registered buffer to a Parameter.
                    # Report actual membership instead of inferring it from
                    # the model constructor's register_buffer() declaration.
                    row["rope_parameter_samples"] = {}
                    for label, observation in (
                        ("native", native),
                        ("replay", standalone),
                    ):
                        sample = observation.get("model_parameter_samples", {}).get(
                            "rope.inv_freq"
                        )
                        row["rope_parameter_samples"][label] = (
                            None
                            if sample is None
                            else {
                                **{key: sample[key] for key in ("shape", "dtype")},
                                "head_tail_values": sample["samples"].tolist(),
                            }
                        )
                    row["first_forward_observed"] = dict(
                        native="first_forward" in native,
                        replay="first_forward" in standalone,
                    )
                    row["first_fixed_execute_observed"] = dict(
                        native="first_fixed_execute" in native,
                        replay="first_fixed_execute" in standalone,
                    )
                    row["first_body_trace_summary"] = {}
                    for label, observation in (
                        ("native", native),
                        ("replay", standalone),
                    ):
                        trace = observation.get("first_body_trace")
                        row["first_body_trace_summary"][label] = (
                            None
                            if trace is None
                            else dict(
                                entries=len(trace),
                                embeddings=sum(
                                    t["stage"] == "embedding" for t in trace
                                ),
                                block_indices=[
                                    t["index"] for t in trace if t["stage"] == "block"
                                ],
                                linear_prefixes=[
                                    t["prefix"]
                                    for t in observation.get(
                                        "first_block_linear_trace", []
                                    )
                                ],
                            )
                        )
                    row["boundary_differences"] = {
                        key: differences(native.get(key), standalone.get(key), key)
                        for key in (
                            "prepared",
                            "session_state",
                            "model_buffers_after_prepare",
                            "fixed_plan_preparations",
                            "graph_runs",
                            "first_forward",
                            "first_fixed_execute",
                            "first_body_trace",
                            "first_block_fused_pre_linear",
                            "first_block_linear_trace",
                            "numerics_mode",
                            "execution_settings",
                            "model_arch",
                            "model_parameter_samples",
                            "model_parameter_sha256",
                            "model_module_settings",
                            "sampler_scalars",
                            "model_scalars",
                            "environment",
                        )
                    }
                    row["boundary_observed"] = {
                        key: dict(native=key in native, replay=key in standalone)
                        for key in row["boundary_differences"]
                    }
                cell["drains"].append(row)
            for a, b in zip(expected, actual):
                if (
                    not torch.isfinite(a).all()
                    or not torch.isfinite(b).all()
                    or a.shape != b.shape
                ):
                    raise ValueError("Invalid native/replay latent")
                cell["output"].append(
                    dict(
                        exact=equal(a, b),
                        max_abs=(a.float() - b.float()).abs().max().item(),
                    )
                )
            output["cells"].append(cell)
    output["scope"] = (
        "Two short403-frame sessions, four ranks and first three replay chunks only. Observations are descriptive, not a root-cause assignment or full-case/soak acceptance."
    )
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2)
    print("NATIVE_RNG_DIAGNOSTIC_READBACK", len(output["cells"]))


if __name__ == "__main__":
    main()
