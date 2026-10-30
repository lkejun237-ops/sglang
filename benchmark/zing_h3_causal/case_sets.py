# SPDX-License-Identifier: Apache-2.0
"""Resolve named benchmark collections without copying their original requests."""

import hashlib
import json
from pathlib import Path

from prepare_whitebox_video_smoke import load_cases

CHECKOUT = Path(__file__).resolve().parents[2]
DEFAULT_CASE_SET = "product-case-20261029"


def load_case_set(selection=DEFAULT_CASE_SET, checkout=CHECKOUT):
    """Manifest paths are relative to checkout, including for custom collections."""
    checkout = Path(checkout)
    path = Path(selection)
    if path.suffix != ".json":
        path = (
            checkout / "benchmark/zing_h3_causal/case_sets" / (str(selection) + ".json")
        )
    definition = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(definition.get("name"), str) or not definition["name"]:
        raise ValueError("Case collection must have a name")
    if not isinstance(definition.get("cases"), list) or not definition["cases"]:
        raise ValueError("Case collection must have at least one member")
    members, seen, manifests = [], set(), {}
    for entry in definition["cases"]:
        manifest = checkout / entry["manifest"]
        case_id = entry["case_id"]
        identity = case_id
        if identity in seen:
            raise ValueError(f"Duplicate collection member: {case_id}")
        seen.add(identity)
        if manifest not in manifests:
            manifests[manifest] = load_cases(manifest)
        if case_id not in manifests[manifest]:
            raise ValueError(f"Collection case is unavailable: {case_id}")
        members.append(
            {
                "manifest": manifest,
                "case_id": case_id,
                "request": manifests[manifest][case_id],
            }
        )
    return {
        "definition": definition,
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "members": members,
    }
