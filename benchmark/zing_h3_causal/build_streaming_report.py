# SPDX-License-Identifier: Apache-2.0
"""Build a self-contained light report from continuous-source result bundles."""

import argparse
import json
from collections import Counter
from pathlib import Path

from build_benchmark_report import export_results, write_report_page


def load_server_evidence(directory, bundles):
    from run_issue253_split_baseline import validate_session_timing

    rows, health, issues = [], [], []
    counts = Counter()
    for path in sorted((directory / "timing").glob("*.jsonl")):
        current = [json.loads(line) for line in path.read_text().splitlines() if line]
        rows.extend(current)
        counts[path.name.split("-session", 1)[0].removesuffix(".jsonl")] += len(current)
    for path in sorted((directory / "timing").glob("*-health.json")):
        snapshot = json.loads(path.read_text())
        health.append({"file": path.name, **snapshot})
        if snapshot["written"] != counts[path.name.removesuffix("-health.json")] or any(
            snapshot.get(key) for key in ("pending", "queued", "error")
        ):
            issues.append(f"Collector health is incomplete: {path.name}")
    hardware = json.loads((directory / "hardware.json").read_text())
    sp = len(hardware["active_roles"]["dit_decoder"])
    if len(health) != sp + len(hardware["active_roles"]["source_encoder"]):
        issues.append("Missing worker timing files")
    domains = sorted({r["clock_domain"] for r in rows})
    clients = {}
    for path in directory.glob("*/*/*/summary.json"):
        value = json.loads(path.read_text())
        clients[value["session_id"]] = value
    sessions, rank_pair_gaps = {}, []
    for bundle in bundles:
        cases = {case["label"]: case for case in bundle["config"]["cases"]}
        for request in bundle["requests"]:
            if not request["success"]:
                continue
            sid = request["server_session_id"]
            # Compare corresponding DiT/Decoder/Request spans in the SP2 ranks.
            # This localizes a cumulative drop counter to specific requests.
            if sp == 2:
                signatures = []
                for rank in range(sp):
                    signatures.append(
                        Counter(
                            json.dumps(
                                [
                                    row.get(key)
                                    for key in (
                                        "dispatch_id",
                                        "stage",
                                        "component",
                                        "output_latent_range",
                                        "output_frame_range",
                                        "no_output",
                                        "source_dependency_end",
                                    )
                                ]
                            )
                            for row in rows
                            if row["request_id"] == sid
                            and row.get("worker_role") is None
                            and row["rank"] == rank
                            and row["stage"] in ("DiT", "Decoder", "Request")
                        )
                    )
                missing = sum((signatures[1] - signatures[0]).values()) + sum(
                    (signatures[0] - signatures[1]).values()
                )
                if missing:
                    rank_pair_gaps.append(
                        {
                            "session_id": sid,
                            "label": request["label"],
                            "phase": request["phase"],
                            "index": request["index"],
                            "unmatched_spans": missing,
                        }
                    )
            try:
                validate_session_timing(
                    rows, sid, cases[request["label"]]["frames"], sp=sp
                )
                sessions[sid] = {"complete": True}
            except RuntimeError as exc:
                sessions[sid] = {"complete": False, "error": str(exc)}
    return {
        "spans": rows,
        "health": health,
        "hardware": hardware,
        "clients": clients,
        "client_same_node_verified": all(
            bundle["config"].get("client_server_same_node", False) is True
            for bundle in bundles
        ),
        "validation": {
            "collector_complete": not issues and not any(h["dropped"] for h in health),
            "collector_drained": not issues,
            "dropped": sum(h["dropped"] for h in health),
            "rank_pair_gaps": rank_pair_gaps,
            "issues": issues,
            "clock_domains": domains,
            "sessions": sessions,
        },
    }


def build_report(bundles, output, *, server_results=None, json_output=None):
    if server_results is not None:
        server = load_server_evidence(server_results, bundles)
        for bundle in bundles:
            bundle["server"] = server
    reports = [{"kind": "continuous_source", "streaming": bundle} for bundle in bundles]
    write_report_page(
        reports[0] if len(reports) == 1 else {"datasets": reports}, output
    )
    if json_output is not None:
        export_results(reports[0], json_output)
        if len(reports) > 1:
            value = json.loads(json_output.read_text())
            value["results"] = reports
            json_output.write_text(
                json.dumps(value, ensure_ascii=False, allow_nan=False)
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--server-results",
        type=Path,
        help="Historical raw CUDA archive directory (new runs use client profiles)",
    )
    args = parser.parse_args()
    bundles = []
    for path in args.results:
        value = json.loads(path.read_text())
        if value.get("format") == "zing-benchmark-results":
            bundles.extend(
                report["streaming"] for report in value.get("results", [value["data"]])
            )
        else:
            bundles.append(value)
    build_report(
        bundles,
        args.output,
        server_results=args.server_results,
        json_output=args.json_output,
    )


def load_received_profiles(directory, bundle):
    """Build server view from client-owned files; no server filesystem needed."""
    from run_issue253_split_baseline import validate_session_timing

    spans, snapshots, clients, sessions = [], {}, {}, {}
    issues = []
    cases = {c["label"]: c for c in bundle["config"]["cases"]}
    for request in bundle["requests"]:
        cell = directory / request["label"] / f"{request['phase']}-{request['index']}"
        path = cell / "profile.json"
        if not path.exists():
            if request["success"]:
                issues.append(f"Missing stage profile: {request['request_id']}")
            continue
        profile = json.loads(path.read_text())
        sid = profile["session_id"]
        spans.extend(profile["stages"])
        for h in profile["health"]:
            key = h.get("file", str(h.get("pid", request["request_id"])))
            snapshots[key] = {"file": key, **h}
        clients[sid] = json.loads((cell / "summary.json").read_text())
        try:
            if profile["status"] != "complete":
                raise RuntimeError(f"Profile status: {profile['status']}")
            # New completion records describe topology; historical dedicated
            # profiles identify it through their Encoder worker role.
            roles = {
                row.get("source_encoder_role")
                for row in profile["stages"]
                if row.get("component") == "continuous_source_complete"
            }
            dedicated = "dedicated" in roles or any(
                row.get("worker_role") == "dedicated_source_encoder"
                for row in profile["stages"]
            )
            validate_session_timing(
                profile["stages"],
                sid,
                cases[request["label"]]["frames"],
                sp=len(profile["expected_ranks"]),
                require_cuda=False,
                require_dedicated_encoder=dedicated,
            )
            sessions[sid] = {"complete": True}
        except RuntimeError as exc:
            sessions[sid] = {"complete": False, "error": str(exc)}
    if not spans and not issues:
        return None
    health = list(snapshots.values())
    domains = sorted({row["clock_domain"] for row in spans})
    return {
        "spans": spans,
        "health": health,
        "clients": clients,
        "hardware": {"devices": []},
        "client_same_node_verified": False,
        "validation": {
            "sessions": sessions,
            "clock_domains": domains,
            "issues": issues,
            "collector_complete": not issues
            and all(v["complete"] for v in sessions.values()),
            "collector_drained": all(
                not any(h.get(k) for k in ("pending", "queued", "error"))
                for h in health
            ),
            "dropped": sum(h.get("dropped", 0) for h in health),
            "rank_pair_gaps": [],
        },
    }


if __name__ == "__main__":
    main()
