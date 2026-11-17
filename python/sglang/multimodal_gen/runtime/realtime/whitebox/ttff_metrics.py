# SPDX-License-Identifier: Apache-2.0
"""Render-owned, bounded conversion of timing receipts to Prometheus samples."""

from sglang.multimodal_gen.runtime.realtime.critical_path_metrics import (
    observe_stage_ms,
)

from .inference_timing import public_timing
from .telemetry import TTFF_CUDA_STAGES, TTFF_HOST_STAGES


class TimingMetrics:
    """Observe terminal revisions once, independently of optional recording."""

    VERSION_WINDOW = 8
    REPORTS_PER_VERSION = 256

    def __init__(self):
        self.latest_version = 0
        self.seen = {}

    def observe(self, report, *, trigger):
        report = public_timing(report)
        if report is None or trigger not in {
            "initial",
            "user_control",
            "world_context",
        }:
            return
        # An incomplete snapshot may still contain other pending CUDA events.
        # Wait for a terminal revision rather than losing their eventual samples.
        if report["status"] == "pending" or any(
            row["device_status"] == "pending" for row in report["stages"]
        ):
            return
        version = report["prompt_version"]
        self.latest_version = max(version, self.latest_version)
        floor = self.latest_version - self.VERSION_WINDOW + 1
        self.seen = {v: ids for v, ids in self.seen.items() if v >= floor}
        if version < floor:
            return
        seen = self.seen.setdefault(version, set())
        identity = (report["role"], report["report_id"])
        if identity in seen or len(seen) >= self.REPORTS_PER_VERSION:
            return
        seen.add(identity)
        phase = "initial" if version == 1 else "prompt_switch"
        role = report["role"]
        for row in report["stages"]:
            if row["status"] != "executed":
                continue
            for clock, field, allowed in (
                ("host", "host_ms", TTFF_HOST_STAGES),
                ("cuda", "device_elapsed_ms", TTFF_CUDA_STAGES),
            ):
                if (
                    row["stage"] not in allowed[role]
                    or row[field] is None
                    or (clock == "cuda" and row["device_status"] != "complete")
                ):
                    continue
                observe_stage_ms(
                    f"whitebox_ttff_{phase}_{role}_{clock}_{row['stage']}",
                    row[field],
                    service="room_runtime",
                    model="zing_h3",
                    scope="request",
                )
