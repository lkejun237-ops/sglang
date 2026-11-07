# SPDX-License-Identifier: Apache-2.0
"""Controlled single-operator probe; never a full-model or stall acceptance.

Validate frozen extracted operands on a zero-GPU controller first. GPU execution
requires the exact eight-device idle allocation; only its first card is used.
Each compiler history runs in a fresh process and private empty cache.
"""

import argparse
import hashlib
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, default=str)


def tensor_sha(value, torch):
    return hashlib.sha256(
        value.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def history_inputs(tensors, rows, inference, torch):
    # Tensor dispatch attributes are guards too, even inside inference_mode.
    with torch.inference_mode(inference):
        return [
            tensor.repeat((rows + tensor.shape[0] - 1) // tensor.shape[0], 1)[:rows]
            for tensor in tensors
        ]


def import_operator():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.encoder_worker import (
        initialize_runtime_imports,
    )

    initialize_runtime_imports()
    from sglang.multimodal_gen.runtime.models.dits.zing_h3 import _native_compiled_adaln

    return _native_compiled_adaln


def load_payload(case, torch):
    if sha(case["path"]) != case["sha256"]:
        raise ValueError("Frozen norm operands changed")
    value = torch.load(case["path"], map_location="cpu", weights_only=True)
    hidden, weight = value["hidden"], value["weight"]
    if (
        hidden.ndim != 2
        or hidden.shape[1] != 5376
        or weight.shape != (5376,)
        or case["hidden_runtime"]["stride"] != [5376, 1]
        or case["hidden_runtime"]["storage_offset"] != 0
        or case["hidden_runtime"]["requires_grad"] is not False
        or type(case["hidden_runtime"]["inference"]) is not bool
        or case["hidden_runtime"]["shape"] != list(hidden.shape)
        or tensor_sha(hidden, torch) != case["hidden_runtime"]["sha256"]
        or tensor_sha(weight, torch) != value["provenance"]["norm_weight_sha256"]
        or value["provenance"]["fused_native"] is not False
        or value["provenance"]["fused_replay"] is not False
        or not math.isfinite(value["eps"])
        or value["eps"] <= 0
    ):
        raise ValueError("Unsupported or mismatched frozen norm boundary")
    for name in ["shift", "scale", "native_output", "replay_output"]:
        if value[name].shape != hidden.shape or value[name].dtype != torch.bfloat16:
            raise ValueError("Norm operand shape or dtype mismatch")
    if hidden.dtype != torch.bfloat16 or not all(
        bool(torch.isfinite(value[name]).all())
        for name in [
            "hidden",
            "weight",
            "shift",
            "scale",
            "native_output",
            "replay_output",
        ]
    ):
        raise ValueError("Norm operands must be finite BF16 data")
    return value


def idle_allocation(config, *, single_worker=False):
    prior = json.loads(Path(config["allocation"]).read_text())
    if sha(config["allocation"]) != config["allocation_sha256"]:
        raise ValueError("Matched allocation changed")
    devices = prior["allocated_cuda_ordinals"]
    visible = devices[:1] if single_worker else devices
    if len(devices) != 8 or os.environ.get(
        "CUDA_VISIBLE_DEVICES", ",".join(devices)
    ) != ",".join(visible):
        raise ValueError("Norm probe requires the declared eight-device allocation")
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader"],
        text=True,
    )
    rows = {
        parts[0].strip(): parts
        for line in inventory.splitlines()
        if (parts := line.split(","))
    }
    applications = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    )
    expected = {row[0].strip(): row[1].strip() for row in prior["physical_inventory"]}
    if any(
        rows[d][1].strip() != expected[d]
        or expected[d] in applications
        or int(rows[d][2].strip().split()[0]) >= 32
        for d in devices
    ):
        raise RuntimeError(
            "An allocated device is occupied or differs from the frozen UUIDs"
        )
    return devices, expected


def worker(config, case, control, output):
    # CUDA_VISIBLE_DEVICES narrows CUDA only; NVML must still verify every
    # device reserved by this eight-GPU task before torch is imported.
    idle_allocation(config, single_worker=True)
    import torch

    torch.set_num_threads(4)
    if torch.cuda.device_count() != 1:
        raise ValueError("Operator worker must see only its assigned first GPU")
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    allocation = json.loads(Path(config["allocation"]).read_text())
    first = allocation["allocated_cuda_ordinals"][0]
    expected_uuid = next(
        row[1].strip()
        for row in allocation["physical_inventory"]
        if row[0].strip() == first
    )
    if "GPU-" + str(props.uuid).removeprefix("GPU-") != expected_uuid:
        raise ValueError("Operator worker GPU differs from the matched allocation")
    if list(torch.cuda.get_device_capability()) != [10, 3]:
        raise ValueError("Norm reproduction requires the measured CC10.3 hardware")
    native_compiled_adaln = import_operator()

    value = load_payload(case, torch)

    def move(tensor, inference):
        with torch.inference_mode(inference):
            return tensor.to("cuda", copy=True)

    hidden = move(value["hidden"], case["hidden_runtime"]["inference"])
    shift, scale = (move(value[name], True) for name in ["shift", "scale"])
    with torch.inference_mode(control["weight_inference"]):
        weight = torch.nn.Parameter(
            value["weight"].to("cuda"), requires_grad=control["weight_grad"]
        )
    norm = SimpleNamespace(weight=weight, eps=value["eps"])
    warmup_flags = None
    if control["warmup"]:
        rows = control["rows"] or hidden.shape[0]
        # This is an operator shape control, never a concatenated video case.
        tensors = history_inputs(
            [hidden, shift, scale], rows, case["hidden_runtime"]["inference"], torch
        )
        with torch.inference_mode(control["warmup"] == "inference"), torch.no_grad():
            warmup_flags = dict(
                inference_context=torch.is_inference_mode_enabled(),
                tensors=[
                    dict(
                        inference=torch.is_inference(tensor),
                        requires_grad=tensor.requires_grad,
                        shape=list(tensor.shape),
                        stride=list(tensor.stride()),
                    )
                    for tensor in tensors
                ],
            )
            native_compiled_adaln(tensors[0], norm, tensors[1], tensors[2])
        torch.cuda.synchronize()
    outputs = []
    with torch.inference_mode(), torch.no_grad():
        for _ in range(3):
            result = native_compiled_adaln(hidden, norm, shift, scale)
            torch.cuda.synchronize()
            if (
                result.shape != hidden.shape
                or result.dtype != value["native_output"].dtype
            ):
                raise RuntimeError("Compiled norm changed output shape or dtype")
            outputs.append(result.to("cpu", copy=True))
    if not all(bool(torch.isfinite(tensor).all()) for tensor in outputs):
        raise RuntimeError("Compiled norm produced non-finite output")
    torch.save(outputs, output / "outputs.pt")
    import inspect

    from torch._inductor import config as inductor_config

    compiler = {}
    for key in [
        "emulate_precision_casts",
        "epilogue_fusion",
        "max_autotune",
        "coordinate_descent_tuning",
        "triton.cudagraphs",
        "triton.persistent_reductions",
        "triton.multi_kernel",
    ]:
        try:
            value_config = inductor_config
            for part in key.split("."):
                value_config = getattr(value_config, part)
            compiler[key] = value_config
        except AttributeError:
            compiler[key] = "unavailable"
    generated = []
    target = output / "generated-sources"
    target.mkdir()
    for cache in [output / "inductor-cache", output / "triton-cache"]:
        for path in sorted(cache.rglob("*")):
            if path.is_file() and path.suffix in {
                ".py",
                ".json",
                ".ttir",
                ".ttgir",
                ".llir",
                ".ptx",
                ".cubin",
            }:
                digest = sha(path)
                destination = target / (digest + path.suffix)
                if not destination.exists():
                    shutil.copyfile(path, destination)
                generated.append(
                    dict(
                        original=str(path),
                        retained=str(destination),
                        sha256=digest,
                        bytes=path.stat().st_size,
                    )
                )
    write(
        output / "result.json",
        dict(
            control=control,
            geometry=case["geometry"],
            payload_sha256=case["sha256"],
            torch=torch.__version__,
            cuda=torch.version.cuda,
            operator_source_sha256=sha(inspect.getsourcefile(native_compiled_adaln)),
            compiler_config=compiler,
            deterministic_algorithms_enabled=torch.are_deterministic_algorithms_enabled(),
            warmup_actual_flags=warmup_flags,
            cache_environment={
                key: os.environ.get(key)
                for key in ["TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"]
            },
            device=dict(name=props.name, uuid=str(props.uuid), cc=[10, 3]),
            weight_runtime_flags_observed_in_original=False,
            actual_weight_flags=dict(
                inference=torch.is_inference(weight),
                requires_grad=weight.requires_grad,
                stride=list(weight.stride()),
            ),
            actual_hidden_flags=dict(
                inference=torch.is_inference(hidden),
                requires_grad=hidden.requires_grad,
                stride=list(hidden.stride()),
            ),
            repeated_exact=all(
                torch.equal(outputs[0], tensor) for tensor in outputs[1:]
            ),
            outputs=[
                dict(
                    sha256=tensor_sha(tensor, torch),
                    dtype=str(tensor.dtype),
                    native_exact=tensor.dtype == value["native_output"].dtype
                    and torch.equal(tensor, value["native_output"]),
                    replay_exact=tensor.dtype == value["replay_output"].dtype
                    and torch.equal(tensor, value["replay_output"]),
                )
                for tensor in outputs
            ],
            generated=generated,
            scope="Single operator under controlled histories and weight flags; not full-model parity or proof of the historical execution history",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--cell", type=int)
    parser.add_argument("--control", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if len(config["cases"]) != 2 or {
        tuple(case["geometry"]) for case in config["cases"]
    } != {(832, 480), (1280, 736)}:
        raise ValueError("Norm diagnosis requires distinct 480 and 736 operand sets")
    if not config["controls"] or any(
        row["warmup"] not in {None, "no_grad", "inference"}
        or type(row["weight_grad"]) is not bool
        or type(row["weight_inference"]) is not bool
        or (
            row["rows"] is not None
            and (type(row["rows"]) is not int or row["rows"] <= 0)
        )
        for row in config["controls"]
    ):
        raise ValueError("Invalid norm compilation controls")
    sys.path[:0] = [
        config["deps"],
        str(Path(config["runtime_source"]) / "benchmark/zing_h3_causal"),
        str(Path(config["runtime_source"]) / "python"),
    ]
    from frozen_source import verify_source

    verify_source(config["runtime_source"], config["runtime_revision"])
    os.environ["SGLANG_LIGHTWEIGHT_RUNTIME"] = "1"
    if sha(__file__) != config["probe_sha256"]:
        raise ValueError("Probe code differs from frozen configuration")
    from owned_process import owned_popen, verify_owned_cleanup

    supervisor = Path(inspect.getsourcefile(owned_popen)).resolve()
    if args.validate_only:
        import torch

        if torch.cuda.device_count() or torch.cuda.is_initialized():
            raise ValueError("Norm payload validation requires zero GPU devices")
        torch.set_num_threads(4)
        native_compiled_adaln = import_operator()
        if not callable(native_compiled_adaln) or torch.cuda.is_initialized():
            raise ValueError(
                "Native norm entry must import without CUDA initialization"
            )
        for case in config["cases"]:
            value = load_payload(case, torch)
            for control in config["controls"]:
                if control["warmup"]:
                    rows = control["rows"] or value["hidden"].shape[0]
                    tensors = history_inputs(
                        [value[name] for name in ["hidden", "shift", "scale"]],
                        rows,
                        case["hidden_runtime"]["inference"],
                        torch,
                    )
                    if any(
                        tensor.shape != (rows, 5376)
                        or torch.is_inference(tensor)
                        != case["hidden_runtime"]["inference"]
                        or tensor.requires_grad
                        for tensor in tensors
                    ):
                        raise ValueError("History tensor attributes differ from target")
        if sha(config["allocation"]) != config["allocation_sha256"]:
            raise ValueError("Matched allocation changed before CPU validation")
        Path(config["cpu_ready"]).parent.mkdir(parents=True, exist_ok=True)
        write(
            config["cpu_ready"],
            dict(
                passed=True,
                config_sha256=sha(args.config),
                probe_sha256=sha(__file__),
                probe_revision=config["probe_revision"],
                ownership_supervisor_sha256=sha(supervisor),
                runtime_revision=config["runtime_revision"],
                operator_imported_without_cuda=True,
                history_input_attributes_checked=True,
                payloads=[case["sha256"] for case in config["cases"]],
                cuda_count=0,
            ),
        )
        return
    if args.cell is not None:
        output = Path(config["output"]) / f"cell{args.cell}-control{args.control}"
        worker(
            config, config["cases"][args.cell], config["controls"][args.control], output
        )
        return
    ready = json.loads(Path(config["cpu_ready"]).read_text())
    if (
        not ready["passed"]
        or ready["config_sha256"] != sha(args.config)
        or ready["probe_sha256"] != sha(__file__)
        or ready.get("ownership_supervisor_sha256") != sha(supervisor)
    ):
        raise ValueError("Frozen CPU validation differs from this probe")
    if any(sha(case["path"]) != case["sha256"] for case in config["cases"]):
        raise ValueError("Frozen payload changed after CPU validation")
    devices, uuids = idle_allocation(config)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    controls = output / "controls"
    controls.mkdir()
    for source, name in [
        (__file__, "probe.py"),
        (args.config, "config.json"),
        (config["cpu_ready"], "CPU_READY.json"),
        (supervisor, "owned_process.py"),
    ]:
        shutil.copyfile(source, controls / name)
    write(
        output / "ALLOCATION.json",
        dict(allocated=8, active=1, unused=7, devices=devices, uuids=uuids),
    )
    from run_isolated_stall_suite import stop

    results, aborted = [], None
    complete = False
    try:
        for cell in range(len(config["cases"])):
            for index in range(len(config["controls"])):
                # Do not overlap controls or clear any device state.
                deadline = time.monotonic() + 20
                while True:
                    try:
                        idle_allocation(config)
                        break
                    except RuntimeError:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(1)
                path = output / f"cell{cell}-control{index}"
                path.mkdir()
                env = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=devices[0],
                    TORCHINDUCTOR_CACHE_DIR=str(path / "inductor-cache"),
                    TRITON_CACHE_DIR=str(path / "triton-cache"),
                    OMP_NUM_THREADS="4",
                    MKL_NUM_THREADS="4",
                )
                process = None
                result = dict(cell=cell, control=index, returncode=None, result=None)
                try:
                    print("NORM_HISTORY_CONTROL_START", cell, index, flush=True)
                    with (path / "run.log").open("xb") as log:
                        process = owned_popen(
                            [
                                sys.executable,
                                __file__,
                                "--config",
                                str(args.config),
                                "--cell",
                                str(cell),
                                "--control",
                                str(index),
                            ],
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        result["returncode"] = process.wait(timeout=300)
                    if result["returncode"] == 0:
                        result["result"] = json.loads(
                            (path / "result.json").read_text()
                        )
                        observed = result["result"]
                        if (
                            observed["geometry"] != config["cases"][cell]["geometry"]
                            or observed["payload_sha256"]
                            != config["cases"][cell]["sha256"]
                            or observed["control"] != config["controls"][index]
                            or len(observed["outputs"]) != 3
                        ):
                            raise ValueError(
                                "Worker result identity or output count differs from its control"
                            )
                except Exception:
                    result["error"] = traceback.format_exc()
                finally:
                    try:
                        if process is not None:
                            stop(process)
                            verify_owned_cleanup(process)
                    except BaseException:
                        result["cleanup_error"] = traceback.format_exc()
                        raise
                    finally:
                        results.append(result)
                        write(output / f"cell{cell}-control{index}.json", result)
        complete = len(results) == len(config["cases"]) * len(
            config["controls"]
        ) and all(
            row["returncode"] == 0
            and row["result"] is not None
            and not row.get("error")
            and not row.get("cleanup_error")
            for row in results
        )
    except BaseException:
        aborted = traceback.format_exc()
        raise
    finally:
        write(
            output / "COLLECTION.json",
            dict(
                results=results,
                collection_completed=complete,
                aborted=aborted,
                full_acceptance_passed=False,
                scope="Untimed single-operator compilation diagnosis; no serving or full-DiT acceptance",
            ),
        )
    if not complete:
        raise RuntimeError("Norm operator collection contains failed controls")


if __name__ == "__main__":
    main()
