# SPDX-License-Identifier: Apache-2.0
"""Independent CPU checks of trusted capture identities and context probe tensors."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
        raise ValueError("Independent numerical readback requires zero CUDA devices")
    cfg = json.loads((root / "runner-config.json").read_text())
    job = json.loads((root / "job-config.json").read_text())
    expected = {
        row["path"]: row
        for row in json.loads((root / "COMMON_CAPTURE_FILES.json").read_text())
    }

    def trusted_load(path):
        if sha(path) != expected[str(path)]["sha256"]:
            raise ValueError("Captured entity hash mismatch")
        return torch.load(path, map_location="cpu", weights_only=False)

    def fingerprint(tensor):
        raw = (
            tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        )
        return dict(
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
            stride=list(tensor.stride()),
            sha256=hashlib.sha256(raw).hexdigest(),
        )

    result = dict(
        source_revision=ready["source_revision"],
        analyzer_sha256=sha(__file__),
        full_matrix_accepted=False,
        captures=[],
        weights=[],
        controls=[],
        scope="CPU evidence only; matching inputs or weights do not establish replay correctness",
    )
    result["corrected_context_semantics"] = []
    for mode in [False, True]:
        with torch.inference_mode(mode=mode), torch.no_grad():
            actual = dict(
                inference_mode=torch.is_inference_mode_enabled(),
                grad_enabled=torch.is_grad_enabled(),
            )
            if actual != dict(inference_mode=mode, grad_enabled=False):
                raise ValueError(
                    "Corrected replay context does not preserve its contract"
                )
            result["corrected_context_semantics"].append(actual)
    original_model = Path(job["reuse_root"]) / "model"
    result["same_resolved_model_as_original_capture"] = (
        Path(cfg["model"]).resolve() == original_model.resolve()
    )
    if not result["same_resolved_model_as_original_capture"]:
        raise ValueError("Probe and original capture model paths differ")
    assets = json.loads((root / "ASSETS.json").read_text())
    indexed = {str(Path(row["path"]).resolve()): row for row in assets}
    for path in sorted((Path(cfg["model"]) / "transformer").rglob("*")):
        if not path.is_file():
            continue
        identity = indexed.get(str(path.resolve()))
        if identity is None or sha(path) != identity["sha256"]:
            raise ValueError(
                "Transformer file is absent from or differs from the frozen model manifest"
            )
        result["weights"].append(
            dict(
                path=str(path),
                resolved=str(path.resolve()),
                bytes=path.stat().st_size,
                sha256=identity["sha256"],
            )
        )
    if not result["weights"]:
        raise ValueError("Missing Transformer files")
    pipeline = json.loads(Path(cfg["pipeline"]).read_text())
    result["pipeline_numeric_config"] = {
        key: value
        for key, value in pipeline.items()
        if "precision" in key or "quant" in key or "adaln" in key
    }
    for index, capture in enumerate(map(Path, cfg["common_captures"])):
        session = trusted_load(capture / "session.pt")
        chunk = trusted_load(capture / "chunk-00000.pt")
        prompt = trusted_load(capture / chunk["prompt_file"])
        left, right = session["inputs"].positive, prompt.condition
        tensors = []
        for key in sorted(set(left) | set(right)):
            a, b = left.get(key), right.get(key)
            equal = (
                a is not None
                and b is not None
                and a.shape == b.shape
                and a.dtype == b.dtype
                and torch.equal(a, b)
            )
            tensors.append(
                dict(
                    key=key,
                    equal=equal,
                    session=fingerprint(a) if a is not None else None,
                    prompt=fingerprint(b) if b is not None else None,
                )
            )
        result["captures"].append(
            dict(
                index=index,
                path=str(capture),
                seed=session["seed"],
                target_geometry=[session["params"].width, session["params"].height],
                replay_policy=session["replay_policy"],
                initial_positive_equals_first_prompt=all(
                    row["equal"] for row in tensors
                ),
                tensors=tensors,
                source=fingerprint(chunk["chunk"].source),
                initial_noise=fingerprint(chunk["chunk"].noise),
                captured_output=fingerprint(chunk["expected_visual"]),
            )
        )
        for rank in range(4):
            directories = [
                root / "work/results" / f"capture{index}-{context}" / f"rank{rank}"
                for context in ["no_grad", "inference"]
            ]
            values = [
                torch.load(
                    directory / "replay-latents.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                for directory in directories
            ]
            if len(values[0]) != len(values[1]):
                raise ValueError("Context diagnostic output counts differ")
            result["controls"].append(
                dict(
                    capture=index,
                    rank=rank,
                    outputs_equal=all(torch.equal(a, b) for a, b in zip(*values)),
                    latent_files=[
                        dict(
                            path=str(directory / "replay-latents.pt"),
                            sha256=sha(directory / "replay-latents.pt"),
                        )
                        for directory in directories
                    ],
                    active_contexts=[
                        json.loads(
                            (directory / "EXECUTION_CONTEXT_ACTIVE.json").read_text()
                        )
                        for directory in directories
                    ],
                    output_fingerprints=[
                        [fingerprint(value) for value in outputs] for outputs in values
                    ],
                    independently_recomputed_first_output=[
                        dict(
                            exact=torch.equal(outputs[0], chunk["expected_visual"]),
                            max_abs=(
                                outputs[0].float() - chunk["expected_visual"].float()
                            )
                            .abs()
                            .max()
                            .item(),
                        )
                        for outputs in values
                    ],
                )
            )
            row = result["controls"][-1]
            row["active_input_checks"] = [
                dict(
                    source=active["source"]["sha256"]
                    == fingerprint(chunk["chunk"].source)["sha256"],
                    noise=active["noise"]["sha256"]
                    == fingerprint(chunk["chunk"].noise)["sha256"],
                    condition=set(active["prompt"]) == set(right)
                    and all(
                        active["prompt"][key]["sha256"] == fingerprint(tensor)["sha256"]
                        for key, tensor in right.items()
                    ),
                )
                for active in row["active_contexts"]
            ]
    if torch.cuda.is_initialized():
        raise ValueError("CPU analysis unexpectedly initialized CUDA")
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(
        "DIT_CONTEXT_CPU_READBACK",
        len(result["captures"]),
        len(result["controls"]),
        flush=True,
    )


if __name__ == "__main__":
    main()
