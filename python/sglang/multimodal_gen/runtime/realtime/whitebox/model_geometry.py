# SPDX-License-Identifier: Apache-2.0
"""Separate native source, private ingress and generated output geometry.

Without a target, preserve the existing lossless grid padding/crop behavior.
An explicit target leaves grid-aligned source RGB unchanged and retains the full
model output. Source clocks and frame identities are never rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.zing_h3_media_geometry import (
    validate_h3_media_geometry,
)

from .contracts import VideoSpec
from .render_contract import RenderError

MODEL_SPATIAL_GRID = 32
GEOMETRY_PROTOCOL = "whitebox-model-geometry-v1"


def validate_reference_geometry(asset: dict) -> None:
    """Apply this Whitebox H3 connection's grid without changing reference bytes."""
    try:
        validate_h3_media_geometry(
            asset["height"], asset["width"], multiple=MODEL_SPATIAL_GRID
        )
    except ValueError as error:
        raise RenderError("CONDITION_INVALID") from error


def resolution_spec(value, default: VideoSpec) -> VideoSpec:
    """Resolve the public selector without changing the deployment default."""
    if value is None:
        return default
    if value == "480p":
        return VideoSpec(832, 480)
    if value == "720p":
        return VideoSpec(1280, 720)
    from .diagnostics import WhiteboxDiagnostic

    raise WhiteboxDiagnostic("INVALID_RESOLUTION", "validation")


def validate_room_spec(spec: VideoSpec, default: VideoSpec):
    """Only public presets or the existing configured default may start a Room."""
    if spec != default and spec not in (VideoSpec(832, 480), VideoSpec(1280, 720)):
        from .diagnostics import WhiteboxDiagnostic

        raise WhiteboxDiagnostic("RESOLUTION_UNSUPPORTED", "validation")


def session_source_spec(world: dict, default: VideoSpec, rooms: dict) -> VideoSpec:
    """An existing game's immutable canvas wins over omitted request settings."""
    room = rooms.get(world.get("game_id"))
    pinned = room.get("spec", default) if room is not None else default
    selected = resolution_spec(world.get("resolution"), pinned)
    if room is not None and selected != pinned:
        from .diagnostics import WhiteboxDiagnostic

        raise WhiteboxDiagnostic("RESOLUTION_UNSUPPORTED", "validation")
    return selected


def classic_source_spec(value, default: VideoSpec) -> VideoSpec:
    """Legacy admission must reject padded geometry before reserving a model."""
    selected = resolution_spec(value, default)
    if ModelGeometry(selected).model != selected:
        from .diagnostics import WhiteboxDiagnostic

        raise WhiteboxDiagnostic("RESOLUTION_UNSUPPORTED", "validation")
    return selected


@dataclass(frozen=True)
class ModelGeometry:
    source: VideoSpec
    target: VideoSpec | None = None
    ingress: VideoSpec = field(init=False)
    model: VideoSpec = field(init=False)

    def __post_init__(self):
        if any(
            type(value) is not int or value <= 0
            for value in (self.source.width, self.source.height)
        ):
            raise ValueError("Source dimensions must be positive integers")
        if type(self.source.fps) is not int or self.source.fps != 24:
            raise ValueError("Whitebox model geometry requires 24 FPS")
        grid = MODEL_SPATIAL_GRID
        object.__setattr__(
            self,
            "ingress",
            VideoSpec(
                (self.source.width + grid - 1) // grid * grid,
                (self.source.height + grid - 1) // grid * grid,
                self.source.fps,
            ),
        )

        if self.target is not None and not isinstance(self.target, VideoSpec):
            raise ValueError("Model target must be a VideoSpec")
        target = self.target or self.ingress
        if target.fps != 24 or any(
            type(v) is not int or v <= 0 or v % grid
            for v in (target.width, target.height)
        ):
            raise ValueError(
                "Model target requires positive grid-aligned dimensions and 24 FPS"
            )
        object.__setattr__(self, "model", target)

    @property
    def output(self) -> VideoSpec:
        return self.model if self.target is not None else self.source

    def model_init(self, initial: dict) -> dict:
        """Copy the public init and replace only its private model geometry."""
        return {
            **initial,
            "size": f"{self.model.width}x{self.model.height}",
            "source_stream": dict(
                format="rgb24",
                width=self.ingress.width,
                height=self.ingress.height,
                fps=self.ingress.fps,
            ),
        }

    @staticmethod
    def _validate_rgb(rgb: bytes, num_frames: int, spec: VideoSpec):
        if type(num_frames) is not int or num_frames < 1:
            raise ValueError("RGB frame count must be a positive integer")
        if not isinstance(rgb, bytes) or len(rgb) != num_frames * spec.frame_bytes:
            raise ValueError("RGB bytes do not match the declared frame geometry")

    def pad_rgb(self, rgb: bytes, *, num_frames: int = 1) -> bytes:
        """Replicate the final pixel/row; every original pixel stays in place."""
        self._validate_rgb(rgb, num_frames, self.source)
        if self.ingress == self.source:
            return rgb
        row_bytes = self.source.width * 3
        right = self.ingress.width - self.source.width
        bottom = self.ingress.height - self.source.height
        frames = []
        for index in range(num_frames):
            start = index * self.source.frame_bytes
            frame = rgb[start : start + self.source.frame_bytes]
            if right:
                rows = [
                    frame[offset : offset + row_bytes]
                    + frame[offset + row_bytes - 3 : offset + row_bytes] * right
                    for offset in range(0, len(frame), row_bytes)
                ]
                frame = b"".join(rows)
            last_row = frame[-self.ingress.width * 3 :]
            frames.append(frame + last_row * bottom)
        return b"".join(frames)

    def crop_rgb(self, rgb: bytes, *, num_frames: int = 1) -> bytes:
        """Crop each packed model frame independently to the source rectangle."""
        self._validate_rgb(rgb, num_frames, self.model)
        if self.model == self.output:
            return rgb
        row_bytes = self.output.width * 3
        model_row_bytes = self.model.width * 3
        if row_bytes == model_row_bytes:
            return b"".join(
                rgb[start : start + self.output.frame_bytes]
                for start in range(0, len(rgb), self.model.frame_bytes)
            )
        return b"".join(
            rgb[
                start + row * model_row_bytes : start
                + row * model_row_bytes
                + row_bytes
            ]
            for start in range(0, len(rgb), self.model.frame_bytes)
            for row in range(self.output.height)
        )

    def receipt(self) -> dict:
        """Describe both identities so padding cannot be mistaken for 720p."""
        value = dict(
            protocol=GEOMETRY_PROTOCOL,
            source=dict(
                width=self.source.width, height=self.source.height, fps=self.source.fps
            ),
            model=dict(
                width=self.model.width, height=self.model.height, fps=self.model.fps
            ),
            padding=dict(
                mode="replicate",
                left=0,
                top=0,
                right=self.ingress.width - self.source.width,
                bottom=self.ingress.height - self.source.height,
            ),
            output_crop=dict(
                x=0, y=0, width=self.output.width, height=self.output.height
            ),
        )

        if self.target is not None:
            value.update(
                model_input=dict(
                    width=self.ingress.width,
                    height=self.ingress.height,
                    fps=self.ingress.fps,
                ),
                output=dict(
                    width=self.output.width,
                    height=self.output.height,
                    fps=self.output.fps,
                ),
            )
        return value
