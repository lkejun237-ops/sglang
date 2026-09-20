#!/usr/bin/env python3
"""Compare two Zing diagnostic request/rank directories (events.jsonl + .pt).

Logical tensor hashes exclude storage layout. Cache events default to skipped;
--right-sp-size enables full-left-SP1 to right-rank dim1 head-shard comparison.
Non-cache payloads load only for differing/missing hashes, shapes or dtypes.
"""

import argparse
import json
import math
import re
from pathlib import Path


def read_events(directory):
    records = {}
    for line in (directory / "events.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = (record["index"], record["phase"], record["name"])
        if key in records:
            raise ValueError(f"Duplicate diagnostic event key: {key}")
        records[key] = record
    return records


def event_key(key):
    return dict(zip(("index", "phase", "name"), key))


def flatten_tensors(payload, prefix=""):
    import torch

    if isinstance(payload, torch.Tensor):
        return {prefix: payload}
    result = {}
    if isinstance(payload, dict):
        items = payload.items()
    elif isinstance(payload, (list, tuple)):
        items = enumerate(payload)
    else:
        return result
    for key, value in items:
        result.update(flatten_tensors(value, f"{prefix}.{key}" if prefix else str(key)))
    return result


def load_tensors(directory, record):
    import torch

    payload = torch.load(
        directory / record["file"], map_location="cpu", weights_only=True
    )
    return flatten_tensors(payload)


def number(value):
    # Keep JSON strict; nonfinite diagnostics are explicit strings.
    return value if math.isfinite(value) else str(value)


def compare_values(left, right):
    import torch

    result = {
        "loaded_shape_left": list(left.shape),
        "loaded_shape_right": list(right.shape),
        "loaded_dtype_left": str(left.dtype),
        "loaded_dtype_right": str(right.dtype),
    }
    if left.shape != right.shape:
        return {**result, "status": "shape_mismatch"}
    a, b = left.reshape(-1), right.reshape(-1)
    total = a.numel()
    maximum = absolute_sum = squared_error = squared_reference = 0.0
    first = None
    finite_left = finite_right = True
    for start in range(0, total, 1_000_000):
        raw_a, raw_b = a[start : start + 1_000_000], b[start : start + 1_000_000]
        equal = raw_a == raw_b
        if first is None and not bool(equal.all()):
            first = start + int((~equal).nonzero()[0].item())
        dtype = (
            torch.complex128
            if raw_a.is_complex() or raw_b.is_complex()
            else torch.float64
        )
        ca, cb = raw_a.to(dtype), raw_b.to(dtype)
        finite_left &= bool(torch.isfinite(ca).all())
        finite_right &= bool(torch.isfinite(cb).all())
        error = (ca - cb).abs()
        chunk_max = float(error.max()) if error.numel() else 0.0
        maximum = (
            float("nan")
            if math.isnan(chunk_max) or math.isnan(maximum)
            else max(maximum, chunk_max)
        )
        absolute_sum += float(error.sum())
        squared_error += float(error.square().sum())
        squared_reference += float(ca.abs().square().sum())
    relative = (
        math.sqrt(squared_error / squared_reference)
        if squared_reference
        else (0.0 if squared_error == 0 else float("inf"))
    )
    return {
        **result,
        "status": "numerically_equal" if first is None else "different",
        "numel": total,
        "max_abs": number(maximum),
        "mean_abs": number(absolute_sum / total if total else 0.0),
        "relative_rmse_left_reference": number(relative),
        "first_different_flat_index": first,
        "finite_left": finite_left,
        "finite_right": finite_right,
    }


def compare(
    left_dir,
    right_dir,
    right_sp_size=None,
    right_rank=0,
    right_sequence_size=None,
    right_sequence_rank=0,
    module_phase="prefix",
):
    if right_sp_size is not None and (
        right_sp_size < 1 or not 0 <= right_rank < right_sp_size
    ):
        raise ValueError(
            "right_sp_size must be positive and right_rank must be in range"
        )
    if right_sequence_size is not None and (
        right_sequence_size < 1 or not 0 <= right_sequence_rank < right_sequence_size
    ):
        raise ValueError(
            "right_sequence_size must be positive and right_sequence_rank must be in range"
        )
    left, right = read_events(left_dir), read_events(right_dir)
    if not module_phase or not module_phase.strip():
        raise ValueError("module_phase must not be empty")
    phase_rows = None
    phase_roles = None
    changes, rng_changes, skipped, alignment = [], [], [], []
    checked = hash_equal = loaded = 0
    first_divergence = None
    for key in sorted(left.keys() | right.keys()):
        a, b = left.get(key), right.get(key)
        if a is not None and key[1] == module_phase and key[2] == "forward_input":
            shape = a.get("tensors", {}).get("packed.position_ids", {}).get("shape", [])
            phase_rows = shape[0] if shape else None
            if right_sequence_size is not None:
                phase_roles = load_tensors(left_dir, a).get("packed.row_role")
        if a is None or b is None:
            alignment.append(
                {**event_key(key), "missing_from": "left" if a is None else "right"}
            )
            continue
        if a.get("rng") != b.get("rng"):
            rng_changes.append(
                {**event_key(key), "left": a.get("rng"), "right": b.get("rng")}
            )
        is_cache = key[2] == "cache" or key[2].startswith("cache.")
        if is_cache and right_sp_size is None:
            skipped.append(event_key(key))
            continue
        payloads = None
        ta, tb = a.get("tensors", {}), b.get("tensors", {})
        for name in sorted(ta.keys() | tb.keys()):
            ma, mb = ta.get(name), tb.get(name)
            entry = {
                **event_key(key),
                "tensor": name,
                "metadata_left": ma,
                "metadata_right": mb,
            }
            checked += 1
            if ma is None or mb is None:
                entry["status"] = "missing_tensor"
            else:
                same_hash = bool(ma.get("sha256")) and ma.get("sha256") == mb.get(
                    "sha256"
                )
                same_structure = ma.get("shape") == mb.get("shape") and ma.get(
                    "dtype"
                ) == mb.get("dtype")
                entry["logical_hash_equal"] = same_hash
                entry["stride_changed"] = ma.get("stride") != mb.get("stride")
                entry["dtype_changed"] = ma.get("dtype") != mb.get("dtype")
                align_cache = is_cache and right_sp_size is not None
                if same_hash and same_structure and not align_cache:
                    hash_equal += 1
                    if not entry["stride_changed"]:
                        continue
                    entry["status"] = "layout_only"
                else:
                    if payloads is None:
                        payloads = (
                            load_tensors(left_dir, a),
                            load_tensors(right_dir, b),
                        )
                    loaded += 1
                    left_tensor, right_tensor = payloads[0][name], payloads[1][name]
                    if align_cache and re.fullmatch(
                        r"chunks\.\d+\.(?:layer0|layer_last|layers\.\d+)\.[01]", name
                    ):
                        if left_tensor.ndim < 2 or left_tensor.shape[1] % right_sp_size:
                            raise ValueError(
                                f"Cannot evenly shard cache heads for {name}: {tuple(left_tensor.shape)} across {right_sp_size}"
                            )
                        heads = left_tensor.shape[1] // right_sp_size
                        begin = right_rank * heads
                        left_tensor = left_tensor[:, begin : begin + heads]
                        entry["cache_alignment"] = {
                            "left_is_full_sp1": True,
                            "dimension": 1,
                            "right_sp_size": right_sp_size,
                            "right_rank": right_rank,
                            "head_start": begin,
                            "head_end": begin + heads,
                        }
                    elif align_cache:
                        entry["cache_alignment"] = {
                            "mode": "unsharded_tensor_direct_comparison"
                        }
                    projection = key[2] in {
                        "module.video_patch_proj",
                        "module.audio_patch_proj",
                    }
                    if (
                        projection
                        and right_sequence_size is not None
                        and key[1] == module_phase
                        and left_tensor.shape != right_tensor.shape
                    ):
                        if (
                            phase_roles is None
                            or phase_roles.ndim != 1
                            or len(phase_roles) != phase_rows
                        ):
                            entry["sequence_alignment"] = {
                                "status": "not_aligned",
                                "reason": "No valid selected-phase forward_input packed.row_role",
                            }
                        else:
                            base, remainder = divmod(phase_rows, right_sequence_size)
                            begin = right_sequence_rank * base + min(
                                right_sequence_rank, remainder
                            )
                            end = begin + base + int(right_sequence_rank < remainder)
                            # Match zing_h3.storage_masks; compact order follows global rows.
                            roles = (
                                (4, 7, 10)
                                if key[2] == "module.video_patch_proj"
                                else (5, 11)
                            )
                            mask = phase_roles == roles[0]
                            for role in roles[1:]:
                                mask |= phase_roles == role
                            compact_begin = int(mask[:begin].sum())
                            compact_end = int(mask[:end].sum())
                            if (
                                left_tensor.ndim
                                and right_tensor.ndim
                                and left_tensor.shape[0] == int(mask.sum())
                                and right_tensor.shape[0] == compact_end - compact_begin
                            ):
                                left_tensor = left_tensor[compact_begin:compact_end]
                                entry["sequence_alignment"] = {
                                    "dimension": 0,
                                    "mode": "modality_compact_rows",
                                    "row_start": begin,
                                    "row_end": end,
                                    "compact_start": compact_begin,
                                    "compact_end": compact_end,
                                    "right_sequence_size": right_sequence_size,
                                    "right_sequence_rank": right_sequence_rank,
                                }
                            else:
                                entry["sequence_alignment"] = {
                                    "status": "not_aligned",
                                    "reason": "Leading dimensions do not match modality compact rows",
                                }
                    if (
                        not projection
                        and right_sequence_size is not None
                        and key[1] == module_phase
                        and key[2].startswith("module.")
                        and left_tensor.shape != right_tensor.shape
                    ):
                        if phase_rows is None:
                            entry["sequence_alignment"] = {
                                "status": "not_aligned",
                                "reason": "No preceding selected-phase forward_input packed.position_ids shape",
                            }
                        else:
                            base, remainder = divmod(phase_rows, right_sequence_size)
                            local_rows = base + int(right_sequence_rank < remainder)
                            begin = right_sequence_rank * base + min(
                                right_sequence_rank, remainder
                            )
                            if (
                                left_tensor.ndim
                                and right_tensor.ndim
                                and left_tensor.shape[0] == phase_rows
                                and right_tensor.shape[0] == local_rows
                            ):
                                left_tensor = left_tensor[begin : begin + local_rows]
                                entry["sequence_alignment"] = {
                                    "dimension": 0,
                                    "total_rows": phase_rows,
                                    "right_sequence_size": right_sequence_size,
                                    "right_sequence_rank": right_sequence_rank,
                                    "row_start": begin,
                                    "row_end": begin + local_rows,
                                    "split_rule": "contiguous_remainder_first",
                                }
                            else:
                                entry["sequence_alignment"] = {
                                    "status": "not_aligned",
                                    "reason": "Leading dimensions do not match full and expected local sequence rows",
                                    "total_rows": phase_rows,
                                    "expected_local_rows": local_rows,
                                }
                    entry.update(compare_values(left_tensor, right_tensor))
                    if (
                        align_cache
                        and entry["status"] == "numerically_equal"
                        and not entry["stride_changed"]
                        and not entry["dtype_changed"]
                    ):
                        continue
            changes.append(entry)
            if first_divergence is None and entry["status"] in (
                "different",
                "shape_mismatch",
                "missing_tensor",
            ):
                first_divergence = {
                    **event_key(key),
                    "tensor": name,
                    "status": entry["status"],
                }
    return {
        "left": str(left_dir.resolve()),
        "right": str(right_dir.resolve()),
        "sequence_comparison": {
            "enabled": right_sequence_size is not None,
            "right_sequence_size": right_sequence_size,
            "right_sequence_rank": right_sequence_rank,
            "module_phase": module_phase,
            "scope": f"{module_phase} module.* events only; total rows from preceding same-phase forward_input packed.position_ids",
        },
        "cache_comparison": {
            "enabled": right_sp_size is not None,
            "right_sp_size": right_sp_size,
            "right_rank": right_rank,
            "assumption": "Left cache is full SP1; right cache uses contiguous head shards on dim1.",
        },
        "semantics": [
            "Events align exactly by (index, phase, name); unmatched events are reported without shifting later records.",
            "Cache events default to skipped. With right_sp_size, chunks layer0/layer_last K,V slice full left SP1 heads on dim1 to the right rank; other cache tensors compare directly. All enabled cache comparisons load payloads regardless of hash. RNG differences remain reported.",
            "With right_sequence_size, differing-shape selected-phase module tensors slice left dim0 only when full/local row counts match remainder-first sequence splitting. Other dimensions still require equality; equal-shape tensors compare directly.",
            "Logical hashes come from contiguous tensor bytes; stride/device differences do not imply numerical divergence.",
            "Equal hashes and equal shape/dtype bypass tensor loading. Non-tensor payload values are not compared.",
            "relative RMSE uses the left tensor as reference: sqrt(sum(abs(left-right)^2)/sum(abs(left)^2)); zero reference gives infinity for nonzero error.",
            "First divergence is the earliest aligned tensor value/shape mismatch, then tensor-name order; RNG, dtype-only and layout-only differences are reported separately.",
            "Hash-equal values are not independently checked for finiteness; nonfinite loaded metrics serialize as strings.",
        ],
        "counts": {
            "left_events": len(left),
            "right_events": len(right),
            "checked_tensors": checked,
            "hash_equal_tensors": hash_equal,
            "loaded_tensor_pairs": loaded,
            "changed_tensor_records": len(changes),
            "rng_different_events": len(rng_changes),
            "skipped_cache_events": len(skipped),
        },
        "first_tensor_divergence": first_divergence,
        "first_rng_difference": rng_changes[0] if rng_changes else None,
        "alignment_differences": alignment,
        "tensor_changes": changes,
        "rng_differences": rng_changes,
        "skipped_cache_events": skipped,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--right-sp-size",
        type=int,
        help="Enable cache comparison by slicing full left SP1 heads to this right SP degree",
    )
    parser.add_argument("--right-rank", type=int, default=0)
    parser.add_argument("--right-sequence-size", type=int)
    parser.add_argument("--right-sequence-rank", type=int, default=0)
    parser.add_argument(
        "--module-phase",
        default="prefix",
        help="Phase used for module row alignment; must match capture MODULE_PHASE (default: prefix)",
    )
    args = parser.parse_args()
    if args.right_sp_size is None and args.right_rank != 0:
        parser.error("--right-rank requires --right-sp-size")
    if args.right_sp_size is not None and (
        args.right_sp_size < 1 or not 0 <= args.right_rank < args.right_sp_size
    ):
        parser.error(
            "--right-sp-size must be positive and --right-rank must be within it"
        )
    if args.right_sequence_size is None and args.right_sequence_rank != 0:
        parser.error("--right-sequence-rank requires --right-sequence-size")
    if args.right_sequence_size is not None and (
        args.right_sequence_size < 1
        or not 0 <= args.right_sequence_rank < args.right_sequence_size
    ):
        parser.error(
            "--right-sequence-size must be positive and --right-sequence-rank must be within it"
        )
    result = compare(
        args.left,
        args.right,
        args.right_sp_size,
        args.right_rank,
        args.right_sequence_size,
        args.right_sequence_rank,
        args.module_phase,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "first_tensor_divergence": result["first_tensor_divergence"],
                "counts": result["counts"],
            }
        )
    )


if __name__ == "__main__":
    main()
