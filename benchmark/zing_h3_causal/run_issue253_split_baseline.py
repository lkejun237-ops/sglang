# SPDX-License-Identifier: Apache-2.0
"""Run exact Issue #253 cases on E1+SP2 serial and overlap, sequentially."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

from prepare_encoder_overlap import digest
from run_encoder_overlap_matrix import (
    endpoint_cell,
    read_stage_timing_health,
    read_stage_timing_rows,
    read_text_retry,
    selected_layouts,
    write_json,
)

LAYOUTS = ("enc1-sp2-serial", "enc1-sp2-overlap")
EXPECTED_CASES = (
    ("zing-whitebox-video-five-triview-0001", 120, 6),
    ("zing-whitebox-video-30s-six-triview-0001", 720, 7),
    ("zing-whitebox-video-30s-six-triview-0002", 720, 7),
)


def expected_latents(frames):
    clips, tail = divmod(frames - 1, 17)
    return clips * 5 + 1 + math.ceil(tail / 4)


def phase_session(cell, phase):
    sessions = {
        event["message"]["session_id"]
        for line in read_text_retry(cell / phase / "events.jsonl").splitlines()
        if line.strip()
        for event in [json.loads(line)]
        if event.get("kind") == "ready"
    }
    if len(sessions) != 1:
        raise RuntimeError(f"Expected one client session in {cell.name}/{phase}")
    return sessions.pop()


def require_cuda_rows(rows, *, session_id, label):
    if not rows:
        raise RuntimeError(f"{label} timing is missing for session {session_id}")
    for row in rows:
        value = row.get("cuda_stream_ms")
        if (
            row.get("request_id") != session_id
            or row.get("dispatch_id_status") != "present"
            or row.get("status") != "ok"
            or row.get("cuda_status") != "complete"
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            raise RuntimeError(
                f"{label} has incomplete CUDA evidence for session {session_id}"
            )


def require_contiguous_ranges(rows, field, expected_end, *, label):
    ranges = [row.get(field) for row in rows if not row.get("no_output")]
    if not ranges or any(
        not isinstance(value, list)
        or len(value) != 2
        or any(type(x) is not int for x in value)
        or value[0] < 0
        or value[1] <= value[0]
        for value in ranges
    ):
        raise RuntimeError(f"{label} ranges are missing or malformed")
    ranges.sort()
    cursor = 0
    for start, end in ranges:
        if start != cursor:
            raise RuntimeError(f"{label} ranges are not contiguous at {cursor}")
        cursor = end
    if cursor != expected_end:
        raise RuntimeError(f"{label} ended at {cursor}, expected {expected_end}")


def validate_session_timing(
    rows,
    session_id,
    expected_frames,
    *,
    sp=2,
    require_cuda=True,
    require_dedicated_encoder=True,
):
    def require_rows(values, **identity):
        if require_cuda:
            return require_cuda_rows(values, **identity)
        if not values or any(
            row.get("status") != "ok"
            or not isinstance(row.get("host_call_ms"), (int, float))
            or not math.isfinite(row["host_call_ms"])
            or row["host_call_ms"] < 0
            for row in values
        ):
            raise RuntimeError(f"Incomplete Host timing: {identity['label']}")

    current = [row for row in rows if row.get("request_id") == session_id]
    encoder = [
        row
        for row in current
        if row.get("stage") == "Encoder"
        and row.get("component") == "source_vae"
        and row.get("worker_role") == "dedicated_source_encoder"
    ]
    if require_dedicated_encoder:
        require_rows(encoder, session_id=session_id, label="dedicated Encoder")
        require_contiguous_ranges(
            encoder,
            "latent_range",
            expected_latents(expected_frames),
            label="dedicated Encoder latent",
        )

    server = [row for row in current if row.get("worker_role") is None]
    expected_ranks = set(range(sp))
    observed = {
        stage: {row.get("rank") for row in server if row.get("stage") == stage}
        for stage in ("DiT", "Decoder")
    }
    if any(ranks != expected_ranks for ranks in observed.values()):
        raise RuntimeError(
            f"Server timing ranks are incomplete for session {session_id}: {observed}"
        )
    for rank in range(sp):
        dit = [
            row
            for row in server
            if row.get("stage") == "DiT" and row.get("rank") == rank
        ]
        decoder = [
            row
            for row in server
            if row.get("stage") == "Decoder" and row.get("rank") == rank
        ]
        complete = [
            row
            for row in server
            if row.get("stage") == "Request"
            and row.get("component") == "continuous_source_complete"
            and row.get("rank") == rank
        ]
        require_rows(dit, session_id=session_id, label=f"DiT rank {rank}")
        require_rows(decoder, session_id=session_id, label=f"Decoder rank {rank}")
        require_rows(complete, session_id=session_id, label=f"completion rank {rank}")
        require_contiguous_ranges(
            dit,
            "output_latent_range",
            expected_latents(expected_frames),
            label=f"DiT rank {rank} latent",
        )
        require_contiguous_ranges(
            decoder,
            "output_frame_range",
            expected_frames,
            label=f"Decoder rank {rank} frame",
        )
    return {
        "session_id": session_id,
        "encoder_rows": len(encoder),
        "dit_rows": sum(row.get("stage") == "DiT" for row in server),
        "decoder_rows": sum(row.get("stage") == "Decoder" for row in server),
    }


def validate_cell(cell, expected_frames, measured_runs):
    phases = ["warmup"] + [f"measured-{i}" for i in range(measured_runs)]
    summaries, session_ids = [], []
    for phase in phases:
        summary = json.loads(read_text_retry(cell / phase / "summary.json"))
        if not summary["ended"] or any(
            summary[key] != expected_frames
            for key in ("submitted", "accepted", "consumed", "received")
        ):
            raise RuntimeError(f"Incomplete output in {cell.name}/{phase}")
        summaries.append(summary)
        session_ids.append(phase_session(cell, phase))
    if len(set(session_ids)) != len(session_ids):
        raise RuntimeError(f"Client sessions were reused in {cell.name}")

    logs = [
        json.loads(line)
        for line in read_text_retry(cell / "encoder.log").splitlines()
        if line.startswith("{") and '"encoder_step"' in line
    ]
    if not set(session_ids) <= {row.get("session") for row in logs}:
        raise RuntimeError("Dedicated Encoder evidence is missing a client session")
    pops = [row for row in logs if row.get("encoder_step") == "pop"]
    if not pops or any(row.get("encoder_execution") is None for row in pops):
        raise RuntimeError("Dedicated Encoder did not provide execution evidence")

    health = read_stage_timing_health(cell / "timing")
    timing_rows = read_stage_timing_rows(cell / "timing")
    rows_by_pid = {
        pid: sum(row.get("pid") == pid for row in timing_rows) for pid in health
    }
    if len(health) != 3 or any(
        row.get("dropped")
        or row.get("pending")
        or row.get("queued")
        or row.get("error")
        for row in health.values()
    ):
        raise RuntimeError("Stage timing evidence is incomplete")
    if any(row.get("written") != rows_by_pid[pid] for pid, row in health.items()):
        raise RuntimeError("Stage timing health snapshots are stale")
    timing_sessions = [
        validate_session_timing(timing_rows, sid, expected_frames, require_cuda=False)
        for sid in session_ids
    ]
    return {
        "phases": phases,
        "summaries": summaries,
        "sessions": timing_sessions,
        "encoder_pop_count": len(pops),
        "timing_rows": len(timing_rows),
    }


def validate_cases(case_paths):
    if len(case_paths) != len(EXPECTED_CASES):
        raise ValueError("Provide the fixed 5s case first, followed by both 30s cases")
    configs = []
    for path, (case_id, frames, references) in zip(case_paths, EXPECTED_CASES):
        config = json.loads(read_text_retry(path / "input.json"))
        prep = json.loads(read_text_retry(path / "PREP_COMPLETE.json"))
        actual = (
            config.get("case_id"),
            config.get("expected_frames"),
            len(config.get("reference_ids", [])),
        )
        if actual != (case_id, frames, references):
            raise ValueError(f"Prepared case contract differs for {case_id}: {actual}")
        if (
            prep.get("case_id") != case_id
            or prep.get("frames") != frames
            or prep.get("reference_ids") != config["reference_ids"]
            or not prep.get("source_rgb", {}).get("sha256")
            or config.get("measured_runs") != 3
            or config.get("stage_timing") != "host"
            or tuple(config.get("layouts", [])) != LAYOUTS
        ):
            raise ValueError(f"Preparation receipt or run policy differs for {case_id}")
        if digest(Path(config["source_rgb"])) != prep["source_rgb"]["sha256"]:
            raise ValueError(f"Prepared source RGB hash differs for {case_id}")
        if digest(Path(config["request"])) != prep.get("request", {}).get("sha256"):
            raise ValueError(f"Prepared request hash differs for {case_id}")
        for asset in prep.get("assets", []):
            local = Path(unquote(urlparse(asset["local_uri"]).path))
            if digest(local) != asset["sha256"]:
                raise ValueError(f"Prepared media hash differs for {case_id}: {local}")
        configs.append(config)
    if len({config["case_id"] for config in configs}) != len(configs):
        raise ValueError("Prepared case IDs must be unique")
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configs = validate_cases(args.case)
    hardware = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                (
                    "import json,torch;print(json.dumps([dict(index=i,"
                    "name=torch.cuda.get_device_name(i),memory_bytes=torch.cuda."
                    "get_device_properties(i).total_memory,cc=torch.cuda."
                    "get_device_capability(i)) for i in range(torch.cuda.device_count())]))"
                ),
            ],
            text=True,
        )
    )
    if len(hardware) < 3:
        raise RuntimeError("E1+SP2 requires three visible GPUs")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_json(
        root / "plan.json",
        {
            "topology": "one dedicated source Encoder GPU; DiT and Decoder share SP2",
            "layouts": LAYOUTS,
            "concurrency": 1,
            "request_order": "serial",
            "warmup_runs": 1,
            "measured_runs": 3,
            "cases": [c["case_id"] for c in configs],
            "hardware": hardware,
        },
    )
    results = []
    for case_index, config in enumerate(configs):
        case_root = root / config["case_id"]
        case_root.mkdir()
        case_results = {}
        for layout in LAYOUTS:
            spec = selected_layouts([layout])[0]
            cell = endpoint_cell(
                case_root,
                config,
                spec,
                False,
                collection_phase="formal",
            )
            case_results[layout] = validate_cell(
                cell, config["expected_frames"], config["measured_runs"]
            )
        serial = case_results[LAYOUTS[0]]["summaries"]
        overlap = case_results[LAYOUTS[1]]["summaries"]
        parity = [
            {
                "phase": case_results[LAYOUTS[0]]["phases"][index],
                "match": left["output_sha256"] == right["output_sha256"],
                "serial_sha256": left["output_sha256"],
                "overlap_sha256": right["output_sha256"],
            }
            for index, (left, right) in enumerate(zip(serial, overlap))
        ]
        receipt = {
            "case_id": config["case_id"],
            "frames": config["expected_frames"],
            "reference_count": len(config["reference_ids"]),
            "serial_overlap_output_match": all(row["match"] for row in parity),
            "serial_overlap_output_parity": parity,
            "output_parity_policy": (
                "Hash equality is strong evidence. Mismatch requires preserved "
                "numeric diagnostics and decoded-video review; it is not by itself "
                "a business-quality failure."
            ),
            "layouts": case_results,
        }
        write_json(case_root / "CASE_COMPLETE.json", receipt)
        results.append(receipt)
        # The first case is the bounded gate. Contract/timing failures above stop 30s.
        if case_index == 0:
            write_json(root / "FIVE_SECOND_GATE.json", {"passed": True})
    write_json(root / "EXPERIMENT_COMPLETE.json", {"passed": True, "cases": results})


if __name__ == "__main__":
    main()
