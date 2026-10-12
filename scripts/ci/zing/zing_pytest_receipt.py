"""Child-only pytest plugin preserving exact nodeids and skip reasons."""

import json
import os
from pathlib import Path

import pytest

_reports = {}
_nodeids = []
_deselected = []
_collection_errors = []
_collection_skips = []
_completed = set()


def pytest_collection_finish(session):
    _nodeids.extend(item.nodeid for item in session.items)


def pytest_deselected(items):
    _deselected.extend(item.nodeid for item in items)


def pytest_collectreport(report):
    if report.failed:
        _collection_errors.append(
            {"nodeid": report.nodeid, "error": str(report.longrepr)}
        )
    if report.skipped:
        reason = (
            report.longrepr[2]
            if isinstance(report.longrepr, tuple)
            else str(report.longrepr)
        )
        _collection_skips.append(
            {"nodeid": report.nodeid, "reason": reason.removeprefix("Skipped: ")}
        )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report):
    if report.when == "teardown":
        _completed.add(report.nodeid)
        # JUnit serializes teardown properties on the finalized parent case;
        # subTest reports may add rows without adding collected pytest items.
        report.user_properties.append(("zing_nodeid", report.nodeid))
    state = _reports.setdefault(
        report.nodeid, {"passed": False, "failed": False, "skipped": False}
    )
    if report.failed:
        state["failed"] = True
    if report.when == "call" and report.passed:
        state["passed"] = True
    if report.skipped:
        state["skipped"] = True
        reason = (
            report.longrepr[2]
            if isinstance(report.longrepr, tuple)
            else str(report.longrepr)
        )
        state["reason"] = reason.removeprefix("Skipped: ")


def pytest_sessionfinish(session, exitstatus):
    skips = {
        nodeid: state["reason"]
        for nodeid, state in _reports.items()
        if state["skipped"]
    }
    payload = {
        "collected": len(_nodeids),
        "nodeids": _nodeids,
        "passed": sum(
            s["passed"] and not s["failed"] and not s["skipped"]
            for s in _reports.values()
        ),
        "failed": sum(s["failed"] for s in _reports.values()),
        "skipped": len(skips),
        "skip_nodeids": sorted(skips),
        "skip_reasons": skips,
        "deselected": _deselected,
        "collection_errors": _collection_errors,
        "collection_skips": _collection_skips,
        "completed_nodeids": sorted(_completed),
        "incomplete_nodeids": sorted(set(_nodeids) - _completed),
        "exit_code": int(exitstatus),
    }
    Path(os.environ["ZING_CASE_RECEIPT"]).write_text(
        json.dumps(payload, indent=2) + "\n"
    )
