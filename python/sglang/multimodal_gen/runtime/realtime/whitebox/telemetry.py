# SPDX-License-Identifier: Apache-2.0
"""Numeric allowlist for model telemetry; never retain prompts or raw usage."""

from ..chunk_telemetry import (
    CHUNK_FIELDS,
    CLI_STARTUP_STAGES,
    HOST_FIELDS,
    STAGE_METRIC_FIELDS,
    STAGE_ROLE_NAMES,
    TTFF_CUDA_STAGES,
    TTFF_HOST_STAGES,
    TTFF_METRIC_STAGES,
    h3_host_durations,
    numeric_durations,
    sanitize_frame_stage_breakdown,
)

__all__ = [
    "CHUNK_FIELDS",
    "TTFF_METRIC_STAGES",
    "TTFF_HOST_STAGES",
    "TTFF_CUDA_STAGES",
    "CLI_STARTUP_STAGES",
    "HOST_FIELDS",
    "STAGE_METRIC_FIELDS",
    "STAGE_ROLE_NAMES",
    "h3_host_durations",
    "numeric_durations",
    "sanitize_frame_stage_breakdown",
]


MAX_RETAINED_STAGE_FRAMES = 4096


class ModelStageTiming:
    """Match optional sidebands to decoded RGB already observed by Room."""

    def __init__(self, manifest, epoch, identity):
        self.manifest = manifest
        self.epoch = epoch
        self.identity = identity
        self.coverage = manifest.setdefault(
            "model_stage_timing",
            dict(
                schema_version=1,
                requested=True,
                output_frames=0,
                retained_frames=0,
                truncated=False,
            ),
        )
        self.observed = {}
        self.archived = {}

    def observe_output(self, start, count, chunk):
        end = start + count
        self.coverage["output_frames"] = end
        if end > MAX_RETAINED_STAGE_FRAMES:
            self.coverage["truncated"] = True
        if type(chunk) is not int or chunk < 0:
            return
        if chunk not in self.observed:
            if start < MAX_RETAINED_STAGE_FRAMES:
                self.observed[chunk] = (start, end)
            return
        previous = self.observed[chunk]
        # Progressive packets can extend one group as RGB becomes available.
        if previous is not None and previous[1] == start and chunk not in self.archived:
            self.observed[chunk] = (previous[0], end)
        else:
            self.observed[chunk] = None
            self.discard(chunk)

    def discard(self, chunk):
        if self.archived.get(chunk) is not None:
            saved = self.manifest["frame_stage_breakdown"]
            saved[:] = [
                row
                for row in saved
                if (row["model_epoch"], row["chunk_index"]) != (self.epoch, chunk)
            ]
            self.coverage["retained_frames"] = len(saved)
        self.archived[chunk] = None

    def receive(self, message):
        if (message.get("session_id"), message.get("generation_id")) != self.identity:
            return []
        start = message.get("output_frame_start")
        frames = message.get("num_frames")
        chunk = message.get("chunk_index")
        if (
            type(start) is not int
            or type(frames) is not int
            or type(chunk) is not int
            or min(start, chunk) < 0
            or not 0 < frames <= 128
            or self.observed.get(chunk) != (start, start + frames)
        ):
            return []
        rows = sanitize_frame_stage_breakdown(
            message.get("frame_stage_breakdown"), start_frame=start, num_frames=frames
        )
        if message.get("decoder_timing_range_verified") is not True:
            for row in rows:
                row.pop("decoder", None)
            rows = [row for row in rows if len(row) > 1]
        if not rows:
            return []
        if chunk in self.archived:
            if rows != self.archived[chunk]:
                self.discard(chunk)
            return []
        self.archived[chunk] = rows
        saved = self.manifest.setdefault("frame_stage_breakdown", [])
        remaining = max(0, MAX_RETAINED_STAGE_FRAMES - len(saved))
        retained = [
            dict(row, model_epoch=self.epoch, chunk_index=chunk)
            for row in rows
            if row["output_frame"] < MAX_RETAINED_STAGE_FRAMES
        ][:remaining]
        saved.extend(retained)
        self.coverage["retained_frames"] = len(saved)
        return retained
