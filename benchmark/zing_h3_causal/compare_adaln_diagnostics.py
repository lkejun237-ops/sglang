# SPDX-License-Identifier: Apache-2.0
"""Exact comparison of two independently captured, bounded diagnostic stores.

This is CPU postprocessing, never part of performance measurements. Repeated
references to the same pair of raw blobs reuse its exact comparison result.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch


def compare(left, right):
    left, right = Path(left), Path(right)
    if left.resolve() == right.resolve():
        raise ValueError("Independent diagnostic directories are required")
    events = [
        [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
        for root in (left, right)
    ]
    if not events[0] or len(events[0]) != len(events[1]):
        raise ValueError("Diagnostic event counts differ or are empty")
    verified = set()
    occurrences = 0
    names = Counter()
    # The harness seeds after conditioning. Fresh worker startup RNG states
    # are unrelated to the request's actual draws; conditioning tensors still
    # compare exactly. From `seeded` onward, every RNG snapshot must match.
    rng_seeded = not any(e["name"] == "seeded" for e in events[0])
    rng_compared = rng_before_seed = 0

    def load(root, blob):
        if Path(blob).name != blob or not blob.endswith(".pt"):
            raise ValueError("Invalid diagnostic blob reference")
        tensor = torch.load(
            root / "tensors" / blob, map_location="cpu", weights_only=True
        )
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Diagnostic blob is not a tensor")
        raw_hash = hashlib.sha256(
            tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        identity = json.dumps([str(tensor.dtype), list(tensor.shape), raw_hash])
        if hashlib.sha256(identity.encode()).hexdigest() + ".pt" != blob:
            raise ValueError("Diagnostic blob integrity mismatch")
        return tensor

    def visit(a, b):
        nonlocal occurrences
        if type(a) is not type(b):
            raise ValueError("Diagnostic payload types differ")
        if isinstance(a, dict):
            if a.keys() != b.keys():
                raise ValueError("Diagnostic payload keys differ")
            if set(a) == {"__zing_tensor_blob__"}:
                pair = (a["__zing_tensor_blob__"], b["__zing_tensor_blob__"])
                occurrences += 1
                if pair not in verified:
                    x, y = load(left, pair[0]), load(right, pair[1])
                    if not (
                        x.shape == y.shape
                        and x.dtype == y.dtype
                        and bool(torch.isfinite(x).all())
                        and bool(torch.isfinite(y).all())
                        and torch.equal(x, y)
                    ):
                        raise ValueError("Diagnostic tensors are not finite and exact")
                    verified.add(pair)
                return
            for key in a:
                visit(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            if len(a) != len(b):
                raise ValueError("Diagnostic sequence lengths differ")
            for x, y in zip(a, b):
                visit(x, y)
        elif a != b:
            raise ValueError("Diagnostic scalar values differ")

    for a, b in zip(*events):
        for event in (a, b):
            if event.get("tensor_storage") != "content_addressed_v1":
                raise ValueError("Comparison requires bounded tensor-store captures")
        for field in ("index", "name", "phase", "tensors"):
            if a[field] != b[field]:
                raise ValueError(f"Diagnostic event {a['index']} differs in {field}")
        rng_seeded = rng_seeded or a["name"] == "seeded"
        if rng_seeded:
            if a["rng"] != b["rng"]:
                raise ValueError(
                    f"Diagnostic event {a['index']} differs in request RNG"
                )
            rng_compared += 1
        else:
            rng_before_seed += 1
        names[a["name"]] += 1
        for root, event in ((left, a), (right, b)):
            if Path(event["file"]).name != event["file"]:
                raise ValueError("Invalid diagnostic payload filename")
        visit(
            torch.load(left / a["file"], map_location="cpu", weights_only=True),
            torch.load(right / b["file"], map_location="cpu", weights_only=True),
        )
    return {
        "kind": "independent_diagnostic_tensor_store_exact",
        "events": len(events[0]),
        "names": dict(names),
        "tensor_occurrences": occurrences,
        "unique_blob_pairs_torch_equal": len(verified),
        "request_rng_snapshots_compared": rng_compared,
        "startup_rng_snapshots_before_seed": rng_before_seed,
        "all_exact_finite": True,
        "performance_evidence": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.left, args.right), indent=2))
