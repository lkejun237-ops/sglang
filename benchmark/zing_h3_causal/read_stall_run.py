# SPDX-License-Identifier: Apache-2.0
"""Read bounded log tails and request metrics without importing GPU libraries.

Snapshots may describe a running suite. They never constitute acceptance or
replace the original files and terminal independent archive verification.
"""

import argparse
import base64
import hashlib
import json
import time
from pathlib import Path


def snapshot(root, logs, tail_lines):
    result = dict(
        time=time.time(),
        root=str(root),
        regression_accepted=False,
        logs={},
        requests=[],
    )
    for name in logs:
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Log path must remain inside the selected run")
        if not path.exists():
            result["logs"][name] = dict(exists=False)
            continue
        stat = path.stat()
        with path.open("rb") as stream:
            stream.seek(max(0, stat.st_size - (128 << 10)))
            tail = stream.read().decode(errors="replace").splitlines()[-tail_lines:]
        result["logs"][name] = dict(
            exists=True, bytes_at_open=stat.st_size, tail=tail, live_snapshot=True
        )
    results = root / "work/results"
    status = results / "STATUS.json"
    if status.exists():
        result["status"] = json.loads(status.read_text())
    for path in sorted(results.rglob("requests.jsonl")):
        for line in path.read_text().splitlines():
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                result.setdefault("incomplete_lines", []).append(str(path))
                continue
            result["requests"].append(
                dict(
                    receipt=str(path),
                    **{
                        key: row.get(key)
                        for key in [
                            "request_id",
                            "server_session_id",
                            "case_id",
                            "success",
                            "delivery_complete",
                            "output_frames",
                            "expected_frames",
                            "input_sha256",
                            "output_sha256",
                            "metrics",
                        ]
                    },
                )
            )
    return result


def replay_comparisons(root):
    """Describe numerical failures without inventing an acceptance tolerance."""
    rows = []
    for path in sorted(
        (root / "work/results").glob("SOAK-*/rank*/replay-comparison.json")
    ):
        raw = path.read_bytes()
        if len(raw) > 4 << 20:
            raise ValueError("Unexpectedly large replay comparison receipt")
        values = json.loads(raw)
        failures = [
            dict(index=i, **row) for i, row in enumerate(values) if not row["exact"]
        ]
        rows.append(
            dict(
                path=str(path),
                sha256=hashlib.sha256(raw).hexdigest(),
                chunks=len(values),
                nonexact=len(failures),
                max_abs=max((row["max_abs"] for row in values), default=None),
                first_nonexact=failures[:8],
                last_nonexact=failures[-1:],
                complete_marker=(path.parent / "COMPLETE").exists(),
            )
        )
    return rows


def source_step_snapshot(root, logs, sessions, windows):
    """Read an immutable byte prefix, retaining only selected original steps.

    An active log may grow. Its prefix hash and selected raw-line hashes are
    observations, not a final whole-file integrity claim. Output frame windows
    are content coordinates, not wall-clock intervals or causal attribution.
    """
    result = dict(
        logs=[],
        rows=[],
        reference_cache_receipts=[],
        malformed_lines=[],
        malformed_cache_lines=[],
        regression_accepted=False,
    )
    for name in logs:
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Source step log must remain inside the selected run")
        size = path.stat().st_size
        remaining, number, incomplete = size, 0, False
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while remaining:
                raw = stream.readline(remaining)
                if not raw:
                    raise ValueError("Source step log shrank during snapshot")
                remaining -= len(raw)
                digest.update(raw)
                number += 1
                if not raw.endswith(b"\n"):
                    incomplete = True
                    continue
                text = raw.decode(errors="replace")
                cache_marker = "Zing reference cache: "
                if cache_marker in text:
                    try:
                        cache = json.loads(text.split(cache_marker, 1)[1])
                        if not isinstance(cache, dict):
                            raise ValueError(
                                "Reference cache receipt must be an object"
                            )
                    except ValueError:
                        result["malformed_cache_lines"].append(
                            dict(log=name, line=number)
                        )
                    else:
                        result["reference_cache_receipts"].append(
                            dict(
                                log=name,
                                line=number,
                                raw_line_sha256=hashlib.sha256(raw).hexdigest(),
                                logger_prefix=text.split(cache_marker, 1)[0],
                                cache=cache,
                            )
                        )
                marker = "zing_source_step "
                if marker not in text or not any(s in text for s in sessions):
                    continue
                try:
                    row = json.loads(text.split(marker, 1)[1])
                    if not isinstance(row, dict) or not isinstance(
                        row.get("session_id"), str
                    ):
                        raise ValueError(
                            "Source step requires an object and string session"
                        )
                except ValueError:
                    result["malformed_lines"].append(dict(log=name, line=number))
                    continue
                if row.get("session_id") not in sessions:
                    continue
                frame = row.get("output_frames")
                if type(frame) is not int or not any(
                    a <= frame <= b for a, b in windows
                ):
                    continue
                result["rows"].append(
                    dict(
                        log=name,
                        line=number,
                        raw_line_sha256=hashlib.sha256(raw).hexdigest(),
                        step=row,
                    )
                )
        result["logs"].append(
            dict(
                log=name,
                prefix_bytes=size,
                prefix_sha256=digest.hexdigest(),
                incomplete_final_line=incomplete,
            )
        )
    result["sessions"] = sorted(sessions)
    result["output_frame_windows"] = windows
    result["sessions_without_rows_in_selected_windows"] = sorted(
        sessions - {r["step"]["session_id"] for r in result["rows"]}
    )
    return result


def metadata_original(root, name):
    """Export a selected stable metadata file, without touching its writer."""
    path = root / name
    resolved = path.resolve()
    if Path(name).is_absolute() or not resolved.is_relative_to(root.resolve()):
        raise ValueError("Original must remain relative to the selected run")
    parts = (*Path(name).parts, *resolved.relative_to(root.resolve()).parts)
    if (
        path.is_symlink()
        or any("secret" in p.lower() or "private" in p.lower() for p in parts)
        or path.suffix not in {".json", ".jsonl", ".log", ".csv", ".txt"}
        or resolved.suffix not in {".json", ".jsonl", ".log", ".csv", ".txt"}
    ):
        raise ValueError("Original must be nonsecret metadata inside the selected run")
    before = path.stat()
    if before.st_size > 64 << 20:
        raise ValueError("Selected metadata original exceeds 64 MiB")
    with path.open("rb") as stream:
        raw = stream.read((64 << 20) + 1)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or len(raw) != before.st_size:
        raise ValueError("Selected metadata original changed during read")
    return dict(
        bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        data_base64=base64.b64encode(raw).decode("ascii"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", action="append", default=[])
    parser.add_argument("--tail-lines", type=int, default=40)
    parser.add_argument("--immutable-preflight", action="store_true")
    parser.add_argument("--replay-comparisons", action="store_true")
    parser.add_argument(
        "--original",
        action="append",
        default=[],
        help="Selected stable metadata original, relative to root; at most four files, 64 MiB each",
    )
    parser.add_argument("--source-step-session", action="append", default=[])
    parser.add_argument(
        "--source-step-frame-window",
        action="append",
        default=[],
        help="Inclusive output frame range START:END; requires --log and --source-step-session",
    )
    args = parser.parse_args()
    if not 1 <= args.tail_lines <= 1000:
        raise ValueError("Tail lines must be between 1 and 1000")
    if len(args.original) > 4:
        parser.error("At most four selected metadata originals may be exported")
    result = snapshot(args.root, args.log, args.tail_lines)
    if args.original:
        result["selected_metadata_originals"] = {
            name: metadata_original(args.root, name) for name in args.original
        }
    if args.source_step_session or args.source_step_frame_window:
        if (
            not args.source_step_session
            or not args.source_step_frame_window
            or not args.log
        ):
            parser.error(
                "Source step snapshots require sessions, frame windows and logs"
            )
        windows = [
            tuple(map(int, value.split(":"))) for value in args.source_step_frame_window
        ]
        if any(len(w) != 2 or not 0 <= w[0] <= w[1] for w in windows):
            parser.error("Source step frame windows must be nonnegative START:END")
        result["source_steps"] = source_step_snapshot(
            args.root, args.log, set(args.source_step_session), windows
        )
    if args.replay_comparisons:
        result["replay_comparisons"] = replay_comparisons(args.root)
    if args.immutable_preflight:
        ready = json.loads((args.root / "CPU_READY.json").read_text())
        originals = {}
        names = [
            "cpu-imports.log",
            "cpu-contracts-1.log",
            "cpu-contracts-2.log",
            "cpu-contracts-1.xml",
            "cpu-contracts-2.xml",
        ]
        if (args.root / "cpu-probe-imports.log").exists():
            names.append("cpu-probe-imports.log")
        for name in names:
            path = args.root / name
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if len(raw) > 1 << 20 or ready["controls"].get(str(path)) != digest:
                raise ValueError(
                    "Frozen CPU preflight bytes differ from the readiness receipt"
                )
            originals[name] = dict(
                bytes=len(raw),
                sha256=digest,
                data_base64=base64.b64encode(raw).decode("ascii"),
            )
        result["immutable_preflight"] = dict(ready=ready, originals=originals)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(
        json.dumps(
            dict(
                output=str(args.output),
                requests=len(result["requests"]),
                status=result.get("status"),
            )
        )
    )


if __name__ == "__main__":
    main()
