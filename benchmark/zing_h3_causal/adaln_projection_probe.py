# SPDX-License-Identifier: Apache-2.0
"""Explicit diagnostic hooks; never install in no-profiler timing workers.

Use ``with ProjectionProbe(model, rank_path): ...`` around a request. Reads
actual embedding bytes and synchronizes every projection: probe wall times
are NOT performance evidence. Output contains hashes/shape metadata, no media.
"""

import hashlib
import inspect
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch


def tensor_key(value):
    raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


class ProjectionProbe:
    def __init__(self, model, path):
        self.model = model
        self.path = Path(path)
        self.handles = []
        self.forward = 0
        self.context = {}

    def __enter__(self):
        self.file = self.path.open("x")
        signature = inspect.signature(self.model.forward)

        def before_model(module, args, kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            self.forward += 1
            t = torch.unique(bound.arguments["t"], sorted=True)
            self.context = {
                "forward": self.forward,
                "phase": bound.arguments.get("self_cache_update"),
                "unique_t": tensor_key(t),
                "unique_t_values": t.cpu().tolist(),
                "generation": module._cache_weight_generation,
            }

        self.handles.append(
            self.model.register_forward_pre_hook(before_model, with_kwargs=True)
        )
        modules = [(str(i), b.adaln_proj) for i, b in enumerate(self.model.blocks)]
        modules.append(("final", self.model.final_layer.adaln_proj))
        for slot, projection in modules:

            def after(module, args, output, slot=slot):
                storages = {
                    v.untyped_storage().data_ptr(): v.untyped_storage() for v in output
                }
                record = {
                    **self.context,
                    "slot": slot,
                    "input": tensor_key(args[0]),
                    "outputs": [tensor_key(v) for v in output],
                    "compact_bytes": sum(s.nbytes() for s in storages.values()),
                }
                self.file.write(json.dumps(record, sort_keys=True) + "\n")

            self.handles.append(projection.register_forward_hook(after))
        return self

    def __exit__(self, *error):
        for handle in self.handles:
            handle.remove()
        self.file.close()


def summarize(path):
    """Potential hits for baseline projection calls, not measured cache hits.

    Insertion keys cross phases within this one session; phase-local uniqueness
    is also reported. A candidate run skips hooked GEMMs and is not suitable for
    deriving baseline call counts. Validate output hashes for each reused key.
    """
    seen = {}
    phases = defaultdict(Counter)
    phase_keys = defaultdict(set)
    conflicts = 0
    with Path(path).open() as source:
        for line in source:
            record = json.loads(line)
            key = json.dumps(
                [
                    record["slot"],
                    record["generation"],
                    record["unique_t"],
                    record["input"],
                ],
                sort_keys=True,
            )
            output = json.dumps(record["outputs"], sort_keys=True)
            phase = record["phase"]
            phases[phase]["calls"] += 1
            phase_keys[phase].add(key)
            if key in seen:
                phases[phase]["potential_hits"] += 1
                conflicts += seen[key][0] != output
            else:
                seen[key] = (output, record["compact_bytes"])
    return {
        "kind": "diagnostic_potential_reuse_not_timing",
        "phases": {
            phase: dict(counts) | {"unique_keys": len(phase_keys[phase])}
            for phase, counts in phases.items()
        },
        "unique_keys": len(seen),
        "compact_bytes_without_eviction": sum(size for _, size in seen.values()),
        "equal_key_output_conflicts": conflicts,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.ledger), indent=2, sort_keys=True))
