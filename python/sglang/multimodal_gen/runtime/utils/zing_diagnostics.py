# SPDX-License-Identifier: Apache-2.0
"""Opt-in correctness capture. Synchronous copies make these runs non-benchmarks."""

import contextvars
from contextlib import contextmanager
import functools
import hashlib
import json
import os
import re
from pathlib import Path

import torch

_current = contextvars.ContextVar("zing_diagnostics", default=None)


def current_diagnostic():
    return _current.get()


def _hash(tensor):
    return hashlib.sha256(
        tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


class ZingDiagnostic:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.index = 0
        self.phase = "prepare"

    @contextmanager
    def prefix_modules(self, model):
        """Capture the selected phase; retain prefix as the legacy default."""
        handles = []
        selected_phase = os.environ.get("SGLANG_ZING_DIAGNOSTICS_MODULE_PHASE")
        if selected_phase is not None and not selected_phase.strip():
            raise ValueError("Diagnostic module phase must not be empty")
        enabled = (
            selected_phase is not None
            or os.environ.get("SGLANG_ZING_DIAGNOSTICS_PREFIX_MODULES") == "1"
        )
        selected_phase = selected_phase or "prefix"
        try:
            if enabled:

                def hook(name):
                    def record(module, inputs, output):
                        if self.phase == selected_phase:
                            self.record("module." + name, input=inputs, output=output)

                    return record

                for name, module in model.named_modules():
                    if selected_phase != "prefix" and name in {
                        "video_patch_proj",
                        "audio_patch_proj",
                    }:
                        handles.append(module.register_forward_hook(hook(name)))
                        continue
                    parts = name.split(".")
                    if len(parts) < 2 or parts[0] != "blocks" or not parts[1].isdigit():
                        continue
                    if len(parts) == 2 or int(parts[1]) < 2:
                        handles.append(module.register_forward_hook(hook(name)))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def record(self, name, **values):
        metadata = {}

        def copy(value, key):
            if isinstance(value, torch.Tensor):
                cpu = value.detach().to("cpu").contiguous()
                metadata[key] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "stride": list(value.stride()),
                    "contiguous": value.is_contiguous(),
                    "device": str(value.device),
                    "sha256": _hash(cpu),
                }
                return cpu
            if isinstance(value, dict):
                return {k: copy(v, f"{key}.{k}") for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [copy(v, f"{key}.{i}") for i, v in enumerate(value)]
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            raise TypeError(f"Unsupported diagnostic value: {type(value).__name__}")

        payload = {k: copy(v, k) for k, v in values.items()}
        filename = f"{self.index:05d}.pt"
        torch.save(payload, self.directory / filename)
        rng = {"cpu": _hash(torch.random.get_rng_state())}
        if torch.cuda.is_available():
            rng["cuda_local"] = _hash(torch.cuda.get_rng_state())
        entry = {
            "index": self.index,
            "phase": self.phase,
            "name": name,
            "file": filename,
            "tensors": metadata,
            "rng": rng,
        }
        with (self.directory / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        self.index += 1

    def cache(self, cache):
        if cache is None:
            return
        all_layers = os.environ.get("SGLANG_ZING_DIAGNOSTICS_ALL_CACHE_LAYERS") == "1"
        chunks = [
            {
                "block": chunk.block,
                "frames": chunk.frames,
                "positions": chunk.positions,
                "layer0": chunk.layers[0],
                "layer_last": chunk.layers[-1],
                **({"layers": chunk.layers} if all_layers else {}),
            }
            for chunk in cache._history()
        ]
        self.record(
            "cache", chunks=chunks, active=cache.active, last_block=cache.last_block
        )


def diagnostic_request(function):
    @functools.wraps(function)
    def wrapped(self, batch, *args, **kwargs):
        root = os.environ.get("SGLANG_ZING_DIAGNOSTICS_DIR")
        if not root:
            return function(self, batch, *args, **kwargs)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        name = getattr(batch.sampling_params, "output_file_name", None) or "request"
        name = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(name))
        trace = ZingDiagnostic(Path(root) / name / f"rank{rank}")
        token = _current.set(trace)
        try:
            trace.record("request_start", seed=batch.sampling_params.seed)
            return function(self, batch, *args, **kwargs)
        finally:
            _current.reset(token)

    return wrapped
