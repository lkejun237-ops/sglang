# SPDX-License-Identifier: Apache-2.0
"""Shared checks for actual Encoder computation and frozen backend policy."""

import hashlib
import json
from pathlib import Path

from prepare_encoder_overlap import digest

BENCH = Path(__file__).resolve().parent


def validate_encode_policy(execution, config):
    policy = execution.get("encode_policy")
    if not isinstance(policy, dict) or "cudnn_benchmark_limit" not in policy:
        raise RuntimeError("Encoder context policy evidence is missing")
    expected = dict(
        encode_mode=config["encode_mode"], encode_dtype=config["encode_dtype"]
    )
    requested_limit = config.get("cudnn_benchmark_limit")
    if requested_limit is not None:
        expected["cudnn_benchmark_limit"] = requested_limit
    for flag, value in dict(
        cudnn_benchmark=config["encode_mode"] == "performance",
        cudnn_deterministic=config["encode_mode"] == "deterministic",
        deterministic_algorithms=config["encode_mode"] == "deterministic",
    ).items():
        if policy.get(flag) is not value:
            raise RuntimeError(
                "Actual Encoder context flags differ from the frozen mode"
            )
    if any(policy.get(key) != value for key, value in expected.items()):
        raise RuntimeError(
            "Actual Encoder context policy differs from the frozen runtime"
        )
    return policy


def validate_execution(execution, world_size):
    """Prove the requested distributed computation actually ran, not a fallback."""
    if not isinstance(execution, dict):
        raise RuntimeError("Encoder execution receipt is missing")
    if (
        execution.get("group") != "dedicated_encoder"
        or execution.get("world_size") != world_size
    ):
        raise RuntimeError("Encoder execution used an unexpected process group")
    if world_size == 1:
        if execution.get("encoder_ranks") != 1:
            raise RuntimeError("Single-rank encoder execution receipt is invalid")
        return
    if (
        execution.get("effective_mode") != "parallel"
        or execution.get("fallback_reason") is not None
        or execution.get("encoder_ranks") != world_size
        or execution.get("participating_group_ranks") != world_size
        or execution.get("replicated_copies") != 1
        or execution.get("gather_calls") != 1
        or execution.get("gather_input_bytes", 0) <= 0
        or not execution.get("local_tile_ids")
    ):
        raise RuntimeError(
            "Multi-rank encoder did not execute tile parallel computation: "
            + json.dumps(execution, sort_keys=True)
        )


def validate_encoder_execution(rows_by_rank, enc):
    for index, baseline in enumerate(rows_by_rank[0]):
        slices = baseline.get("slices")
        if not isinstance(slices, list) or not slices:
            raise RuntimeError("Encoder execution slices are missing")
        cursor = watermark = 0
        for start, end, dependency in slices:
            if (
                start != cursor
                or end <= start
                or not watermark < dependency <= baseline["frames"]
            ):
                raise RuntimeError("Encoder execution slices are invalid")
            cursor, watermark = end, dependency
        if watermark != baseline["frames"]:
            raise RuntimeError("Encoder execution did not drain the source")
        for rank, rows in enumerate(rows_by_rank):
            row = rows[index]
            executions = row.get("encoder_executions")
            if (
                row.get("slices") != slices
                or not isinstance(executions, list)
                or len(executions) != len(slices)
            ):
                raise RuntimeError("Encoder execution evidence is incomplete")
            for step, execution in enumerate(executions):
                if execution.get("rank") != rank or execution.get("world_size") != enc:
                    raise RuntimeError("Encoder execution rank does not match")
                if enc > 1:
                    validate_execution(execution, enc)
                    if (
                        execution.get("group_ranks") != list(range(enc))
                        or execution.get("gather_output_bytes")
                        != enc * execution["gather_input_bytes"]
                    ):
                        raise RuntimeError(
                            "Encoder gather group or byte count is invalid"
                        )
                if execution.get("effective_mode") not in ("parallel", "replicated"):
                    raise RuntimeError("Encoder execution mode is missing")
                shape = execution.get("output_shape", [])
                if (
                    len(shape) != 5
                    or shape[2] < slices[step][1] - slices[step][0]
                    or [
                        execution.get("latent_start"),
                        execution.get("latent_end"),
                        execution.get("source_dependency_end"),
                    ]
                    != slices[step]
                ):
                    raise RuntimeError(
                        "Encoder execution output does not cover the slice"
                    )
        for step in range(len(slices)):
            executions = [
                rows[index]["encoder_executions"][step] for rows in rows_by_rank
            ]
            policies = [x.get("encode_policy") for x in executions]
            if any(policy != policies[0] for policy in policies[1:]):
                raise RuntimeError(
                    "Encoder ranks used different actual context policies"
                )
            count = executions[0].get("tile_count")
            if (
                type(count) is not int
                or count < 1
                or any(x.get("tile_count") != count for x in executions)
            ):
                raise RuntimeError("Encoder tile geometry is missing or inconsistent")
            owners = [tile for x in executions for tile in x.get("local_tile_ids", [])]
            if sorted(owners) != list(range(count)):
                raise RuntimeError(
                    "Encoder tiles are duplicated or missing across ranks"
                )


def validate_encoder_acceptance(
    root,
    source_sha256,
    sizes=(1, 2, 4),
    cudnn_benchmark_limit=None,
    *,
    require_numerical=True,
    width=832,
    height=480,
):
    expected = {(n, tuple(p)) for n in range(1, 35) for p in ([1], [3, 2, 7, 5], [17])}
    frozen_runtime = None
    for enc in sizes:
        rows_by_rank = []
        for rank in range(enc):
            directory = root / f"encoder-{enc}" / f"rank{rank}"
            if require_numerical and not (directory / "COMPLETE").is_file():
                raise RuntimeError("Encoder numerical acceptance is missing")
            manifest = json.loads((directory / "manifest.json").read_text())
            rows = json.loads((directory / "acceptance.json").read_text())
            if (
                manifest.get("execution_schema_version") != 2
                or manifest.get("reference_tile_mode") != "replicated"
                or manifest.get("requested_tile_mode") not in ("auto", "parallel")
                or manifest["source_sha256"] != source_sha256
                or manifest["world_size"] != enc
                or manifest["tolerance"] != dict(atol=0, rtol=0)
                or (manifest["width"], manifest["height"], manifest["frames"])
                != (width, height, 720)
                or not rows
                or len(rows) != len({(r["frames"], tuple(r["pattern"])) for r in rows})
                or not {(r["frames"], tuple(r["pattern"])) for r in rows}.issubset(
                    expected
                )
                or (require_numerical and len(rows) != len(expected))
                or (
                    require_numerical
                    and not all(
                        r[k]["passed"] is True and r[k]["max_abs"] == 0
                        for r in rows
                        for k in ("moments", "latent")
                    )
                )
            ):
                raise RuntimeError("Encoder acceptance does not match this experiment")
            runtime = manifest.get("runtime")
            if not isinstance(runtime, dict):
                raise RuntimeError("Encoder frozen runtime is missing")
            if frozen_runtime is None:
                frozen_runtime = runtime
            elif runtime != frozen_runtime:
                raise RuntimeError("Encoder ranks use different frozen runtimes")
            policy_config = runtime.get("vae_config", {})
            if policy_config.get("cudnn_benchmark_limit") != cudnn_benchmark_limit:
                raise RuntimeError(
                    "Encoder acceptance uses a different benchmark_limit policy"
                )
            if manifest.get("effective_cudnn_benchmark_limit") != cudnn_benchmark_limit:
                raise RuntimeError("Encoder acceptance policy manifest is inconsistent")
            for row in rows:
                actual = row.get("encoder_executions", [])
                reference = row.get("reference_encoder_executions", [])
                if not actual or len(reference) != len(actual):
                    raise RuntimeError("Encoder reference policy evidence is missing")
                for candidate_step, reference_step in zip(actual, reference):
                    policy = candidate_step.get("encode_policy")
                    if (
                        not isinstance(policy, dict)
                        or "cudnn_benchmark_limit" not in policy
                        or policy != reference_step.get("encode_policy")
                    ):
                        raise RuntimeError(
                            "Reference and candidate Encoder context policies differ"
                        )
                    validate_encode_policy(candidate_step, policy_config)
            rows_by_rank.append(sorted(rows, key=lambda r: (r["frames"], r["pattern"])))
        validate_encoder_execution(rows_by_rank, enc)


def input_geometry(config):
    width, height = config.get("width", 832), config.get("height", 480)
    if any(type(n) is not int or n < 1 for n in (width, height)):
        raise ValueError("width and height must be positive integers")
    return width, height


def endpoint_binding(config):
    """Exact frozen configuration and source identity for split collections."""
    source = BENCH.parents[1]
    sha = hashlib.sha256()
    paths = sorted((source / "python" / "sglang").rglob("*.py")) + sorted(
        BENCH.glob("*.py")
    )
    for path in paths:
        sha.update(str(path.relative_to(source)).encode())
        sha.update(bytes.fromhex(digest(path)))
    return dict(
        config=config,
        source_sha256=sha.hexdigest(),
        source_files=len(paths),
        request_sha256=digest(Path(config["request"])),
        source_rgb_sha256=digest(Path(config["source_rgb"])),
        model_config_sha256=digest(Path(config["model"]) / "zing_config.json"),
        realtime_profile_sha256=digest(
            Path(config["model"]) / "zing_realtime_profile.json"
        ),
    )


def validate_profile_binding(root, config, specs, binding):
    path = root / "endpoint-binding.json"
    if not path.is_file() or json.loads(path.read_text()) != binding:
        raise ValueError("Profile inputs or source differ from the formal collection")
    for spec in specs:
        for pace in ("unpaced", "paced"):
            cell = root / f"{spec[0]}-{pace}"
            receipt = json.loads((cell / "FORMAL_COMPLETE.json").read_text())
            expected = ["warmup"] + [f"measured-{i}" for i in range(5)]
            if receipt.get("phases") != expected or not receipt.get(
                "measurement_completed"
            ):
                raise ValueError("Profile requires all five formal runs and warmup")
            for phase in expected:
                row = json.loads((cell / phase / "summary.json").read_text())
                if row.get("ended") is not True or any(
                    row.get(k) != 720
                    for k in ("submitted", "accepted", "consumed", "received")
                ):
                    raise ValueError("Profile cannot bind incomplete formal output")


def validate_serial_overlap(root, spec, config=None, *, require_parity=True):
    """Require five complete paired runs before scheduling a separate soak."""
    name, sp, enc, mode = spec
    if not enc or mode != "overlap":
        raise ValueError(
            "Soak requires an explicitly selected dedicated overlap layout"
        )
    if config is not None:
        input_path = root / "input.json"
        if not input_path.is_file():
            raise RuntimeError("Paired performance input receipt is missing")
        previous = json.loads(input_path.read_text())
        # Paths refer to the immutable prepared assets/model overlay, never a rebuilt request.
        if any(
            previous.get(key) != config.get(key)
            for key in (
                "model",
                "request",
                "source_rgb",
                "soak_rgb",
                "taeh3_weights",
                "cudnn_benchmark_limit",
                "encoder_tile_mode",
                "progressive_decode",
                "source_batch_pattern",
                "width",
                "height",
            )
        ):
            raise RuntimeError("Soak inputs differ from the paired performance receipt")
    rows = []
    for execution in ("serial", "overlap"):
        directory = root / f"enc{enc}-sp{sp}-{execution}-paced"
        for i in range(5):
            path = directory / f"measured-{i}" / "summary.json"
            if not path.is_file():
                raise RuntimeError(
                    "Five paired performance runs are required before soak"
                )
            row = json.loads(path.read_text())
            if not row.get("ended") or any(
                row.get(key) != 720
                for key in ("submitted", "accepted", "consumed", "received")
            ):
                raise RuntimeError("Incomplete paired performance output")
            rows.append(row)
    if any(not row.get("output_sha256") for row in rows):
        raise RuntimeError("Paired performance output hash evidence is missing")
    if require_parity:
        validate_serial_overlap_parity(rows)
    return rows


def validate_serial_overlap_parity(rows):
    """Only numerical equality is an optional exploratory gate."""
    if len({row["output_sha256"] for row in rows}) != 1:
        raise RuntimeError("Serial/overlap output parity failed before soak")


def validate_correctness(root, specs, config):
    receipt = root / "input.json"
    if not receipt.is_file():
        raise RuntimeError("Correctness input receipt is missing")
    previous = json.loads(receipt.read_text())
    if any(
        previous.get(key) != config.get(key)
        for key in (
            "model",
            "request",
            "source_rgb",
            "taeh3_weights",
            "cudnn_benchmark_limit",
            "encoder_tile_mode",
            "progressive_decode",
            "source_batch_pattern",
            "width",
            "height",
        )
    ):
        raise ValueError("Correctness inputs differ from the performance configuration")
    for name, _, enc, _ in specs:
        if not enc:
            continue
        path = root / f"correctness-{name}.json"
        if not path.is_file():
            raise RuntimeError(
                "Shared-versus-dedicated correctness evidence is missing"
            )
        row = json.loads(path.read_text())
        if (
            row.get("status") != "passed"
            or row.get("tolerance") != dict(atol=0, rtol=0)
            or not isinstance(row.get("wire_rgb"), dict)
            or row["wire_rgb"].get("bytes")
            != 720 * input_geometry(config)[0] * input_geometry(config)[1] * 3
        ):
            raise RuntimeError("Shared-versus-dedicated strict correctness gate failed")


def capture_session(cell):
    paths = list((cell / "capture").glob("*/COMPLETE.json"))
    if len(paths) != 1:
        raise RuntimeError("Expected one complete capture session")
    return paths[0].parent
