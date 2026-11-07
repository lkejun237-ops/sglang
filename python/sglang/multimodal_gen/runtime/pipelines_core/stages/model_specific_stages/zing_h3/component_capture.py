# SPDX-License-Identifier: Apache-2.0
"""Opt-in, untimed component replay inputs; never enable in performance runs."""

import dataclasses
import hashlib
import json
from copy import copy
from pathlib import Path

import torch


def media_asset_manifest(params):
    """Bind each ordered media reference to bytes before session temp cleanup."""
    params = vars(params) if dataclasses.is_dataclass(params) else params
    assets = []

    def record(field, path):
        digest = hashlib.sha256()
        size = 0
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
        if not size:
            raise ValueError("Captured media must not be empty")
        assets.append(
            dict(field=field, path=path, bytes=size, sha256=digest.hexdigest())
        )

    for i, path in enumerate(params.get("image_path") or []):
        record(f"params.image_path[{i}]", path)
    request = params.get("zing_request") or {}
    for i, message in enumerate(request.get("messages", [])):
        for j, item in enumerate(message.get("content", [])):
            if item.get("type") in ("image", "video") and "uri" in item:
                record(
                    f"params.zing_request.messages[{i}].content[{j}].uri", item["uri"]
                )
    return dict(schema_version=1, assets=assets)


def map_tensors(value, transform):
    if isinstance(value, torch.Tensor):
        return transform(value)
    if dataclasses.is_dataclass(value):
        result = copy(value)
        for field in dataclasses.fields(value):
            object.__setattr__(
                result, field.name, map_tensors(getattr(value, field.name), transform)
            )
        return result
    if isinstance(value, dict):
        return {k: map_tensors(v, transform) for k, v in value.items()}
    if isinstance(value, tuple):
        items = tuple(map_tensors(v, transform) for v in value)
        return type(value)(*items) if hasattr(value, "_fields") else items
    if isinstance(value, list):
        return [map_tensors(v, transform) for v in value]
    return value


def pool_metadata(stage):
    pool = getattr(stage, "kv_pool", None)
    if pool is None:
        return None
    plan = pool.plan
    return dict(
        schema_version=2,
        config={
            field.name: getattr(plan.config, field.name)
            for field in dataclasses.fields(plan.config)
            if field.init
        },
        runtime_policy={
            name: getattr(plan, name)
            for name in (
                "runtime_budget_bytes",
                "local_hidden_width",
                "prefix_hidden_width",
            )
        },
        geometry={
            name: getattr(plan.geometry, name)
            for name in (
                "spatial_ratio",
                "patch_size",
                "max_short_edge",
                "time_span_period",
            )
        },
        slot_rows=plan.slot_rows,
        dtype=stage.config.dit_precision,
        world_size=torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1,
    )


class ComponentCapture:
    def __init__(
        self,
        root,
        stage,
        inputs,
        params,
        seed,
        session_id,
        *,
        replay_policy=None,
        scope="full",
    ):
        if scope not in ("full", "dit"):
            raise ValueError("Component capture scope must be full or dit")
        self.scope = scope
        self.root = Path(root) / session_id
        self.root.mkdir(parents=True, exist_ok=False)
        self.count = 0
        self.source_count = 0
        self.prompt_events = {}
        self.frames = {"input": [], "decoder": []}
        inputs = copy(inputs)
        if hasattr(inputs, "condition_media"):
            object.__setattr__(inputs, "condition_media", None)
        (self.root / "media-assets.json").write_text(
            json.dumps(media_asset_manifest(params), indent=2) + "\n"
        )
        self.save(
            "session.pt",
            dict(
                inputs=inputs,
                params=params,
                seed=seed,
                runtime=stage.runtime,
                arch=stage.config.dit_config.arch_config,
                replay_policy=replay_policy,
                capture_scope=scope,
                kv_pool=pool_metadata(stage),
            ),
        )

    def save(self, name, value):
        torch.save(
            map_tensors(value, lambda t: t.detach().cpu().clone()), self.root / name
        )

    def remember_prompt(self, prompt):
        key = (prompt.event_id, prompt.prompt_version)
        if key not in self.prompt_events:
            name = f"prompt-{len(self.prompt_events):05d}.pt"
            self.save(name, prompt)
            self.prompt_events[key] = name
        return self.prompt_events[key]

    def append(self, chunk, result, *, prompt=None):
        if prompt is not None:
            self.remember_prompt(prompt)
        key = (
            getattr(result, "prompt_event_id", None),
            getattr(result, "prompt_version", None),
        )
        prompt_file = self.prompt_events.get(key)
        if hasattr(result, "prompt_event_id") and prompt_file is None:
            raise ValueError("Captured result has no matching prompt snapshot")
        self.save(
            f"chunk-{self.count:05d}.pt",
            dict(chunk=chunk, expected_visual=result.visual, prompt_file=prompt_file),
        )
        self.count += 1

    def append_source(self, encoded, sampled):
        self.save(
            f"source-{self.source_count:05d}.pt",
            dict(
                start=encoded.start,
                end=encoded.end,
                source_dependency_end=encoded.source_dependency_end,
                moments=encoded.values,
                latent=sampled,
            ),
        )
        self.source_count += 1

    def append_frames(self, kind, values, start):
        """Hash individual C,T,H,W frames, independent of dispatch packet sizes.

        Decoder hashes are the pre-wire decoder tensors, not encoded video or
        browser output. This capture intentionally synchronizes and is untimed.
        """
        values = values.detach().cpu()
        records = self.frames[kind]
        if start != len(records):
            raise ValueError("Captured frames must be contiguous")
        for offset in range(values.shape[1]):
            frame = values[:, offset].contiguous()
            if not torch.isfinite(frame).all():
                raise ValueError("Captured frames must be finite")
            raw = frame.view(torch.uint8).numpy().tobytes()
            records.append(
                dict(
                    index=start + offset,
                    shape=list(frame.shape),
                    dtype=str(frame.dtype),
                    sha256=hashlib.sha256(raw).hexdigest(),
                )
            )

    def finish(self):
        (self.root / "frames.json").write_text(json.dumps(self.frames))
        (self.root / "COMPLETE.json").write_text(
            json.dumps(
                dict(
                    schema_version=3,
                    capture_scope=self.scope,
                    chunks=self.count,
                    sources=self.source_count,
                    input_frames=len(self.frames["input"]),
                    decoder_frames=len(self.frames["decoder"]),
                )
            )
        )
