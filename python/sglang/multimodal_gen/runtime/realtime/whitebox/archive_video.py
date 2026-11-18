# SPDX-License-Identifier: Apache-2.0
"""Independent archive codecs and source/model pairing inside one worker."""

from __future__ import annotations

import time

from .archive_encoder import ArchiveVideoEncoder
from .archive_rgb import RGBReader
from .contracts import presentation_pts


class IndependentArchiveVideos:
    def __init__(self, data, config, source_only, *, on_rgb_read=None):
        self.data, self.config, self.source_only = data, config, source_only
        self.reader = RGBReader(data.spec.frame_bytes, on_read=on_rgb_read)
        self.model_spec = getattr(data, "model_spec", data.spec)
        self.model_reader = (
            self.reader
            if self.model_spec == data.spec
            else RGBReader(self.model_spec.frame_bytes, on_read=on_rgb_read)
        )
        names = ("whitebox",) if source_only else ("whitebox", "model")
        self.codecs = {}
        try:
            for name in names:
                bitrate = config[
                    (
                        "source_bitrate"
                        if source_only
                        else (
                            "paired_source_bitrate"
                            if name == "whitebox"
                            else "model_bitrate"
                        )
                    )
                ]
                self.codecs[name] = ArchiveVideoEncoder(
                    self.model_spec if name == "model" else data.spec,
                    bitrate=bitrate,
                    vbv_ms=config["vbv_ms"],
                    keyframe_ms=config["keyframe_ms"],
                )
        except BaseException:
            self.close()
            raise
        self.frames = {}
        self.sources, self.models, self.started = {}, {}, {}
        self.source_indices = {}
        self.pending_bytes = 0
        self.next_index = 0
        self.first_pts_us = self.end_time_us = None
        self.closed = False

    def _limits(self):
        if self.pending_bytes > 32 << 20:
            raise BufferError("Archive pair byte budget exceeded")
        if len(self.started) > self.config["pair_frames"]:
            raise BufferError("Archive pair frame budget exceeded")
        if (
            self.started
            and time.monotonic() - min(self.started.values()) > self.config["pair_age"]
        ):
            raise TimeoutError("Archive source pair expired")

    def _store(self, name, outputs):
        for index, encoded in outputs:
            if self.source_only:
                frame, prompt = self.frames.pop(index)
                self.data.encoders["whitebox"].append(index, encoded, encoded["pts_us"])
                self.data.record_metadata(frame, prompt)
                if self.first_pts_us is None:
                    self.first_pts_us = encoded["pts_us"]
                self.end_time_us = encoded["pts_us"] + encoded["duration_us"]
                self.data.last_frame = index
                self.next_index += 1
            else:
                target = self.sources if name == "whitebox" else self.models
                if index in target:
                    raise ValueError("Duplicate archive pair frame")
                target[index] = encoded
                self.pending_bytes += len(encoded["payload"])
        if not self.source_only:
            self._pairs()
            self._limits()

    def _pairs(self):
        while self.next_index in self.sources and self.next_index in self.models:
            index = self.next_index
            source, model = self.sources.pop(index), self.models.pop(index)
            frame, prompt = self.frames.pop(index)
            pts = presentation_pts(frame)
            if source["pts_us"] != pts or model["pts_us"] != pts:
                raise ValueError("Archive pair source identity changed")
            # The model clock describes successful pairs, including the EOS tail.
            # A selected-but-unreturned source must not lengthen only one movie.
            source["duration_us"] = model["duration_us"]
            self.data.encoders["whitebox"].append(index, source, pts)
            self.data.encoders["model"].append(index, model, pts)
            self.data.record_metadata(frame, prompt)
            if self.first_pts_us is None:
                self.first_pts_us = pts
            self.end_time_us = pts + model["duration_us"]
            self.pending_bytes -= len(source["payload"]) + len(model["payload"])
            self.started.pop(index)
            self.source_indices.pop(index)
            self.data.last_frame = index
            self.next_index += 1

    def operation(self, kind, value):
        self._limits()
        if kind == "source_rgb" and self.source_only:
            frame, descriptor = value
            self.frames[frame.index] = frame, ""
            self._store(
                "whitebox",
                self.codecs["whitebox"].push(
                    frame.index,
                    self.reader.read(descriptor),
                    presentation_pts(frame),
                ),
            )
        elif kind == "selected_source_rgb" and not self.source_only:
            index, pts, source_index, descriptor = value
            self.started[index] = time.monotonic()
            self.source_indices[index] = source_index
            self._limits()
            self._store(
                "whitebox",
                self.codecs["whitebox"].push(index, self.reader.read(descriptor), pts),
            )
        elif kind == "model_rgb" and not self.source_only:
            frame, descriptor, prompt = value
            if frame.index not in self.started or frame.index in self.frames:
                raise ValueError("Archive model has no selected source")
            if (
                frame.provenance.get("source_frame_index")
                != self.source_indices[frame.index]
            ):
                raise ValueError("Archive model source identity changed")
            self.frames[frame.index] = frame, prompt
            self._store(
                "model",
                self.codecs["model"].push(
                    frame.index,
                    self.model_reader.read(descriptor),
                    presentation_pts(frame),
                ),
            )
        else:
            raise ValueError("Invalid independent archive operation")

    def finish(self, outcome):
        for name, codec in self.codecs.items():
            self._store(name, codec.drain())
        if self.models or self.frames:
            raise ValueError("Archive contains unpaired model output")
        if outcome == "completed" and (self.sources or self.started):
            raise ValueError("Archive is missing model output")
        self.data.manifest["archive_encoding_version"] = 2
        self.data.manifest["video_streams"] = {
            name: dict(codec.stream) for name, codec in self.codecs.items()
        }
        self.data.manifest.setdefault("rendering", {})["archive_frames"] = (
            self.next_index
        )
        self.data.manifest["rendering"]["unpaired_source_frames"] = len(self.started)
        if self.end_time_us is not None:
            self.data.inputs.end_us = self.end_time_us
        self.close()

    def close_readers(self):
        self.reader.close()
        if self.model_reader is not self.reader:
            self.model_reader.close()

    def close(self):
        for codec in self.codecs.values():
            codec.close()
        self.close_readers()
        self.closed = True
