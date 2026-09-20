#!/usr/bin/env python3
"""CPU, bounded-memory native/SG diagnostic comparison; no shape coercion.

Forward pairs match (update, per-block unique timesteps, occurrence), never
cross-system global event indices. Inputs/outputs are mmap-loaded one pair at a time.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import torch

PACKED_FIELDS = (
    "x",
    "audio_x",
    "context",
    "t",
    "seq_lens",
    "clean_seq_lens",
    "block_idx",
    "position_ids",
    "row_role",
)
CONDITION_FIELDS = (
    "positive",
    "visual",
    "audio",
    "source_visual",
    "visual_shape",
    "audio_shape",
)


def read(root):
    return [
        json.loads(line)
        for line in (root / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]


def load(root, event):
    return torch.load(
        root / event["file"], map_location="cpu", weights_only=True, mmap=True
    )


def flat(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(flat(item, f"{prefix}.{key}" if prefix else str(key)))
        return result
    # Keep scalar lists (shape metadata) atomic; tensor lists remain addressable.
    if isinstance(value, (list, tuple)) and any(
        isinstance(v, (torch.Tensor, dict, list, tuple)) for v in value
    ):
        result = {}
        for i, item in enumerate(value):
            result.update(flat(item, f"{prefix}.{i}"))
        return result
    return {prefix: list(value) if isinstance(value, (list, tuple)) else value}


def tensor_metrics(a, b):
    out = {
        "native_shape": list(a.shape),
        "sglang_shape": list(b.shape),
        "native_dtype": str(a.dtype),
        "sglang_dtype": str(b.dtype),
    }
    if a.shape != b.shape:
        return {**out, "status": "shape_mismatch_not_compared"}
    aa, bb = a.reshape(-1), b.reshape(-1)
    maximum = total = sq = ref = 0.0
    first = None
    finite = True
    for start in range(0, aa.numel(), 1_000_000):
        x, y = aa[start : start + 1_000_000], bb[start : start + 1_000_000]
        equal = x == y
        if first is None and not bool(equal.all()):
            first = start + int((~equal).nonzero()[0])
        x, y = x.double(), y.double()
        finite &= bool(torch.isfinite(x).all() and torch.isfinite(y).all())
        delta = (x - y).abs()
        peak = float(delta.max())
        maximum = (
            float("nan")
            if math.isnan(peak) or math.isnan(maximum)
            else max(maximum, peak)
        )
        total += float(delta.sum())
        sq += float(delta.square().sum())
        ref += float(x.square().sum())

    def safe(x):
        return x if math.isfinite(x) else str(x)

    return {
        **out,
        "status": "equal" if first is None else "different",
        "max_abs": safe(maximum),
        "mean_abs": safe(total / aa.numel() if aa.numel() else 0),
        "relative_rmse_native": safe(
            math.sqrt(sq / ref) if ref else (0 if sq == 0 else float("inf"))
        ),
        "first_different_flat_index": first,
        "both_finite": finite,
    }


def compare_pair(nroot, sroot, ne, se, label, fields):
    native, sg = load(nroot, ne), load(sroot, se)
    a = flat({k: native[k] for k in fields if k in native})
    b = flat({k: sg[k] for k in fields if k in sg})
    rows = []
    for key in sorted(a.keys() | b.keys()):
        item = {"field": key}
        x, y = a.get(key), b.get(key)
        if key not in a or key not in b:
            item.update(
                status="missing_field",
                missing_from="native" if key not in a else "sglang",
            )
        elif isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            ma, mb = (
                ne.get("tensors", {}).get(key, {}),
                se.get("tensors", {}).get(key, {}),
            )
            item.update(native_stride=ma.get("stride"), sglang_stride=mb.get("stride"))
            if (
                ma.get("sha256")
                and ma.get("sha256") == mb.get("sha256")
                and x.shape == y.shape
                and x.dtype == y.dtype
            ):
                item.update(
                    status="equal_hash",
                    shape=list(x.shape),
                    dtype=str(x.dtype),
                    max_abs=0.0,
                )
            else:
                item.update(tensor_metrics(x, y))
        elif isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            item.update(status="type_mismatch")
        else:
            item.update(
                status="equal" if x == y else "different_metadata", native=x, sglang=y
            )
        rows.append(item)
    return {
        "label": label,
        "native_event": {k: ne.get(k) for k in ("index", "phase", "name", "file")},
        "sglang_event": {k: se.get(k) for k in ("index", "phase", "name", "file")},
        "rng": {
            "native": ne.get("rng"),
            "sglang": se.get("rng"),
            "equal": ne.get("rng") == se.get("rng"),
        },
        "fields": rows,
    }


def forward_pairs(root, events):
    pairs = {}
    occurrences = defaultdict(int)
    for position, event in enumerate(events):
        if event["name"] != "forward_input":
            continue
        payload = load(root, event)
        packed = payload["packed"]
        blocks = packed["block_idx"]
        times = packed["t"]
        if blocks.ndim != 1 or times.shape != blocks.shape:
            raise ValueError("Invalid forward block/t metadata")
        schedule = tuple(
            (
                int(block),
                tuple(
                    float(t).hex()
                    for t in torch.unique(times[blocks == block], sorted=True)
                ),
            )
            for block in torch.unique(blocks, sorted=True)
        )
        base = (str(payload["update"]), schedule)
        occurrence = occurrences[base]
        occurrences[base] += 1
        key = (*base, occurrence)
        following = None
        for other in events[position + 1 :]:
            if other["name"] == "forward_input":
                break
            if other["name"] == "forward_output":
                following = other
                break
        pairs[key] = (event, following)
    return pairs


def signature(key):
    return {"update": key[0], "block_timesteps_hex": key[1], "occurrence": key[2]}


def compare(nroot, sroot):
    ne, se = read(nroot), read(sroot)
    pairs = []
    missing = []

    def single(events, name):
        found = [e for e in events if e["name"] == name]
        if len(found) != 1:
            raise ValueError(f"Expected exactly one {name}; got {len(found)}")
        return found[0]

    # At _sample entry both sides have their actual prepared tensors on execution device.
    ncondition = (
        "conditioning_on_sample_device"
        if any(e["name"] == "conditioning_on_sample_device" for e in ne)
        else "conditioning_ready"
    )
    pairs.append(
        compare_pair(
            nroot,
            sroot,
            single(ne, ncondition),
            single(se, "conditioning"),
            "conditioning",
            CONDITION_FIELDS,
        )
    )
    pairs.append(
        compare_pair(
            nroot,
            sroot,
            single(ne, "initial_noise"),
            single(se, "initial_noise"),
            "initial_noise",
            ("video", "audio"),
        )
    )
    nf, sf = forward_pairs(nroot, ne), forward_pairs(sroot, se)
    # Native execution order defines chronology; matching itself uses semantic signatures.
    for key, (ni, no) in nf.items():
        if key not in sf:
            missing.append({"signature": signature(key), "missing_from": "sglang"})
            continue
        si, so = sf[key]
        label = {"forward": signature(key), "kind": "packed"}
        result = compare_pair(nroot, sroot, ni, si, label, ("update", "packed"))
        result["fields"] = [
            r
            for r in result["fields"]
            if r["field"] == "update"
            or r["field"].removeprefix("packed.") in PACKED_FIELDS
        ]
        pairs.append(result)
        if no is None or so is None:
            missing.append(
                {
                    "signature": signature(key),
                    "missing_forward_output_from": "native" if no is None else "sglang",
                }
            )
        else:
            pairs.append(
                compare_pair(
                    nroot,
                    sroot,
                    no,
                    so,
                    {"forward": signature(key), "kind": "prediction"},
                    ("video", "audio"),
                )
            )
    for key in sf.keys() - nf.keys():
        missing.append({"signature": signature(key), "missing_from": "native"})
    pairs.append(
        compare_pair(
            nroot,
            sroot,
            single(ne, "final_latents"),
            single(se, "final_latents"),
            "final_latents",
            ("video", "audio"),
        )
    )
    first = None
    largest = None
    count = defaultdict(int)
    for pair in pairs:
        for row in pair["fields"]:
            count[row["status"]] += 1
            identity = {
                "label": pair["label"],
                "field": row["field"],
                "status": row["status"],
            }
            if first is None and row["status"] not in ("equal", "equal_hash"):
                first = identity
            if isinstance(row.get("max_abs"), (int, float)) and (
                largest is None or row["max_abs"] > largest["max_abs"]
            ):
                largest = {**identity, "max_abs": row["max_abs"]}
    return {
        "native": str(nroot.resolve()),
        "sglang": str(sroot.resolve()),
        "semantics": [
            "Both directories must be full-SP1 request captures. No cross-rank head/sequence slicing or implicit reshaping.",
            "Forwards match exact update + block IDs and per-block unique timestep float.hex values + occurrence. Different timesteps appear as unmatched signatures, not forced matches.",
            "Only common model-input fields x/audio_x/context/t/seq_lens/clean_seq_lens/block_idx/position_ids/row_role are compared from packed. No patchification/unpatchification conversions are inferred.",
            "Forward predictions are compared only at exact captured shapes. Shape mismatch remains explicit and is not numerical parity. Native and SG current _forward both directly return generator outputs.",
            "First difference follows conditioning, initial noise, native forward execution order, final latents; alignment failures must be considered separately. Field order within one event is lexical.",
            "Only one event pair is mmap-loaded at a time. Metrics use bounded chunks; matching hashes skip arithmetic. CPU snapshots represent logical values, not original strides.",
            "Native minimal observer does not capture scheduler renoise draws. RNG state mismatches are evidence only, not proof of where extra draws occurred.",
            "No GPU executed by this script; exact tensor comparison is a diagnostic, not a claim of DMD quality or full benchmark acceptance.",
        ],
        "first_difference": first,
        "largest_absolute_error": largest,
        "alignment_failures": missing,
        "field_status_counts": dict(count),
        "pairs": pairs,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--sglang", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = compare(args.native, args.sglang)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "first_difference": report["first_difference"],
                "alignment_failures": len(report["alignment_failures"]),
            }
        )
    )


if __name__ == "__main__":
    main()
