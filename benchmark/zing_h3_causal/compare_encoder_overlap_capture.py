#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Strict CPU comparison of trusted, untimed component captures.

Never load captures from untrusted sources: torch pickle contains Python objects.
This gate covers captured decoder tensors. Encoded WebSocket media and lifecycle
acceptance remain separate gates. Missing artifacts cannot pass.
"""

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path

import torch


def exact(left, right, path="value"):
    if type(left) is not type(right):
        raise ValueError(f"{path}: type differs")
    if isinstance(left, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"{path}: tensor shape/dtype differs")
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise ValueError(f"{path}: nonfinite tensor")
        if not torch.equal(left, right):
            delta = (left.double() - right.double()).abs().max().item()
            raise ValueError(f"{path}: tensor differs (max_abs={delta})")
    elif dataclasses.is_dataclass(left):
        for field in dataclasses.fields(left):
            exact(
                getattr(left, field.name),
                getattr(right, field.name),
                f"{path}.{field.name}",
            )
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError(f"{path}: keys differ")
        for key in left:
            exact(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise ValueError(f"{path}: length differs")
        for index, (a, b) in enumerate(zip(left, right)):
            exact(a, b, f"{path}[{index}]")
    elif hasattr(left, "__dict__") and not isinstance(left, type):
        exact(vars(left), vars(right), path)
    elif left != right:
        raise ValueError(f"{path}: value differs")


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def validate(root):
    root = Path(root)
    manifest = json.loads((root / "COMPLETE.json").read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("Capture schema 2 is required")
    for key in ("chunks", "sources", "input_frames", "decoder_frames"):
        if type(manifest.get(key)) is not int or manifest[key] <= 0:
            raise ValueError(f"Missing or invalid capture count: {key}")
    if manifest["input_frames"] != manifest["decoder_frames"]:
        raise ValueError("Decoder did not drain every input frame")
    frames = json.loads((root / "frames.json").read_text())
    for kind, count in (("input", "input_frames"), ("decoder", "decoder_frames")):
        rows = frames[kind]
        if len(rows) != manifest[count]:
            raise ValueError(f"{kind}: frame count mismatch")
        for index, row in enumerate(rows):
            if (
                row["index"] != index
                or len(row["sha256"]) != 64
                or not row["shape"]
                or not row["dtype"]
            ):
                raise ValueError(f"{kind}: invalid frame record")
            int(row["sha256"], 16)
    session = load(root / "session.pt")
    for key in ("inputs", "params", "seed", "runtime", "arch"):
        if key not in session or session[key] is None:
            raise ValueError(f"Missing session field: {key}")
    cursor = 0
    dependency = 0
    for index in range(manifest["sources"]):
        row = load(root / f"source-{index:05d}.pt")
        for name in ("moments", "latent"):
            tensor = row[name]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.ndim != 5
                or tensor.shape[2] != row["end"] - row["start"]
            ):
                raise ValueError(f"source {index}: invalid {name}")
            exact(tensor, tensor, f"source {index}.{name}")
        if row["start"] != cursor or row["end"] <= cursor:
            raise ValueError("Source latent ranges are not contiguous")
        watermark = row["source_dependency_end"]
        if not dependency <= watermark <= manifest["input_frames"]:
            raise ValueError("Source dependency watermark is invalid")
        cursor, dependency = row["end"], watermark
    chunk_end = 0
    for index in range(manifest["chunks"]):
        row = load(root / f"chunk-{index:05d}.pt")
        chunk = row["chunk"]
        for name in ("source", "noise"):
            tensor = getattr(chunk, name)
            if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                raise ValueError(f"chunk {index}: missing {name}")
        if chunk.chunk.start != chunk_end or chunk.chunk.end <= chunk_end:
            raise ValueError("DiT chunk ranges are not contiguous")
        if chunk.provenance.source_dependency_end > manifest["input_frames"]:
            raise ValueError("DiT dependency exceeds input frames")
        chunk_end = chunk.chunk.end
        if (
            not isinstance(row["expected_visual"], torch.Tensor)
            or row["expected_visual"].numel() == 0
        ):
            raise ValueError("DiT output is missing")
        exact(row, row, f"chunk {index}")
    if chunk_end != cursor:
        raise ValueError("DiT did not consume all source latents")
    for prefix, key in (("chunk", "chunks"), ("source", "sources")):
        if len(list(root.glob(f"{prefix}-*.pt"))) != manifest[key]:
            raise ValueError(f"{prefix}: unexpected artifact count")
    return manifest, frames


def comparison_session(value):
    """Exclude only run identity/output locations; preserve all inference knobs."""
    result = dict(value)
    params = value["params"]
    if dataclasses.is_dataclass(params):
        params = {
            field.name: getattr(params, field.name)
            for field in dataclasses.fields(params)
        }
    elif isinstance(params, dict):
        params = dict(params)
    else:
        raise ValueError("Unsupported sampling parameter representation")
    ignored = {}
    for key in ("request_id", "output_path", "output_file_name", "perf_dump_path"):
        if key in params:
            ignored[key] = params.pop(key)
    result["params"] = params
    return result, ignored


def normalize_media_paths(left, right, reference, candidate):
    """Normalize session-local media URIs only using capture-time file hashes.

    No basename or request-manifest fallback: old captures without media hashes
    cannot prove byte identity. List order and every non-URI inference field remain exact.
    """
    evidence = []

    def manifest(root):
        path = Path(root) / "media-assets.json"
        if not path.is_file():
            return {}
        value = json.loads(path.read_text())
        if value.get("schema_version") != 1 or not isinstance(
            value.get("assets"), list
        ):
            raise ValueError("Invalid media asset manifest")
        rows = {}
        for row in value["assets"]:
            if (
                row["field"] in rows
                or type(row["bytes"]) is not int
                or row["bytes"] <= 0
                or len(row["sha256"]) != 64
            ):
                raise ValueError("Invalid media asset identity")
            int(row["sha256"], 16)
            rows[row["field"]] = row
        return rows

    assets_a, assets_b = manifest(reference), manifest(candidate)

    def paired(a, b, field):
        if not isinstance(a, str) or not isinstance(b, str):
            raise ValueError(f"{field}: media paths must be strings")
        aa, bb = assets_a.pop(field, None), assets_b.pop(field, None)
        if aa is None or bb is None:
            raise ValueError(
                f"{field}: missing capture-time media SHA evidence; recapture required"
            )
        if aa["path"] != a or bb["path"] != b:
            raise ValueError(f"{field}: media manifest path binding differs")
        da = {k: aa[k] for k in ("bytes", "sha256")}
        db = {k: bb[k] for k in ("bytes", "sha256")}
        exact(da, db, field + ".content")
        evidence.append(dict(field=field, reference=a, candidate=b, **da))
        identity = f"sha256:{da['sha256']}:bytes:{da['bytes']}"
        return identity, identity

    pa, pb = left["params"], right["params"]
    ia, ib = pa.get("image_path"), pb.get("image_path")
    if isinstance(ia, list) and isinstance(ib, list) and len(ia) == len(ib):
        pairs = [
            paired(a, b, f"params.image_path[{i}]")
            for i, (a, b) in enumerate(zip(ia, ib))
        ]
        pa["image_path"] = [p[0] for p in pairs]
        pb["image_path"] = [p[1] for p in pairs]

    def copy_containers(value):
        if isinstance(value, dict):
            return {k: copy_containers(v) for k, v in value.items()}
        if isinstance(value, list):
            return [copy_containers(v) for v in value]
        return value

    for p in (pa, pb):
        if "zing_request" in p:
            p["zing_request"] = copy_containers(p["zing_request"])
    ra, rb = pa.get("zing_request"), pb.get("zing_request")
    if isinstance(ra, dict) and isinstance(rb, dict):
        ma, mb = ra.get("messages"), rb.get("messages")
        if isinstance(ma, list) and isinstance(mb, list) and len(ma) == len(mb):
            for i, (a, b) in enumerate(zip(ma, mb)):
                ca, cb = a.get("content"), b.get("content")
                if isinstance(ca, list) and isinstance(cb, list) and len(ca) == len(cb):
                    for j, (x, y) in enumerate(zip(ca, cb)):
                        if (
                            x.get("type") in ("image", "video")
                            and x.get("type") == y.get("type")
                            and "uri" in x
                            and "uri" in y
                        ):
                            x["uri"], y["uri"] = paired(
                                x["uri"],
                                y["uri"],
                                f"params.zing_request.messages[{i}].content[{j}].uri",
                            )
    if assets_a or assets_b:
        raise ValueError("Unconsumed media asset manifest fields")
    return evidence


def compare(reference, candidate, reference_rgb=None, candidate_rgb=None):
    result = dict(
        status="failed",
        tolerance=dict(atol=0, rtol=0),
        reference=str(reference),
        candidate=str(candidate),
        scope="untimed source/moments/latent/noise/DiT/decoder capture; lifecycle and wire media require separate acceptance",
    )
    try:
        a, af = validate(reference)
        b, bf = validate(candidate)
        exact(a, b, "manifest")
        exact(af, bf, "frames")
        sa, ignored_a = comparison_session(load(Path(reference) / "session.pt"))
        sb, ignored_b = comparison_session(load(Path(candidate) / "session.pt"))
        result["normalized_media_paths"] = normalize_media_paths(
            sa, sb, reference, candidate
        )
        exact(sa, sb, "session.pt")
        result["excluded_run_metadata"] = dict(reference=ignored_a, candidate=ignored_b)
        for name in [f"source-{i:05d}.pt" for i in range(a["sources"])] + [
            f"chunk-{i:05d}.pt" for i in range(a["chunks"])
        ]:
            exact(load(Path(reference) / name), load(Path(candidate) / name), name)
        result["wire_rgb"] = "not_requested"
        if reference_rgb is not None or candidate_rgb is not None:
            if reference_rgb is None or candidate_rgb is None:
                raise ValueError("Both raw RGB paths are required")

            def digest(path):
                sha = hashlib.sha256()
                size = 0
                with Path(path).open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        size += len(block)
                        sha.update(block)
                if not size:
                    raise ValueError("Raw RGB file is empty")
                return dict(bytes=size, sha256=sha.hexdigest())

            aa, bb = digest(reference_rgb), digest(candidate_rgb)
            expected_bytes = sum(math.prod(row["shape"]) for row in af["decoder"])
            if aa["bytes"] != expected_bytes or bb["bytes"] != expected_bytes:
                raise ValueError("Raw RGB byte count does not cover all decoded frames")
            exact(aa, bb, "raw_rgb")
            result["wire_rgb"] = aa
        result.update(status="passed", counts=a)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference-rgb", type=Path)
    parser.add_argument("--candidate-rgb", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(
        args.reference, args.candidate, args.reference_rgb, args.candidate_rgb
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
