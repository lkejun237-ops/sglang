# SPDX-License-Identifier: Apache-2.0
"""Diagnostic-only stage events; invoke in a separate round, never formal runs.

Activate before importing worker modules. Decoder source anchors must match the
frozen revision exactly; unsupported revisions fail before GPU allocation.
"""

import functools
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def retain_isolated_stage_files(timing):
    """Keep finite benchmark traces in RAM until the owned runner preserves them.

    The production entry never calls this. It avoids losing incomplete rows on
    a missing profile ACK, and leaves the native model and timing events intact.
    """

    def retain(directory, request_id):
        return None

    @contextmanager
    def persistent_directory():
        # The runner owns this finite tree and copies it after every role stops.
        base = Path(os.environ["SGLANG_ZING_STAGE_TIMING_DIR"]).resolve()
        base.mkdir(parents=True, exist_ok=True)
        previous = os.environ["SGLANG_ZING_STAGE_TIMING_DIR"]
        directory = Path(tempfile.mkdtemp(prefix="zing-stage-timing-", dir=base))
        os.environ["SGLANG_ZING_STAGE_TIMING_DIR"] = str(directory)
        try:
            yield directory
        finally:
            writer = getattr(timing, "_WRITER", None)
            if writer is not None and writer.path.parent == directory:
                writer.close()
                timing._WRITER = None
            os.environ["SGLANG_ZING_STAGE_TIMING_DIR"] = previous

    timing.remove_profile_files = retain
    timing.deployment_timing_directory = persistent_directory


def release_shared_allocator_cache(role, device):
    """Release unused reservations at owned role boundaries in shared tests.

    Dedicated allocations keep their normal allocator policy. This never frees
    live tensors, KV leases or CUDA Graph pools and is not a production policy.
    """
    import os

    if os.environ.get("STALL_SHARED_ALLOCATOR_TRIM") != "1":
        return
    import json
    import time

    import torch

    before = torch.cuda.memory_reserved(device)
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    print(
        "STALL_SHARED_ALLOCATOR_TRIM "
        + json.dumps(
            dict(
                role=role,
                pid=os.getpid(),
                time=time.time(),
                reserved_before=before,
                reserved_after=torch.cuda.memory_reserved(device),
                allocated_after=torch.cuda.memory_allocated(device),
            )
        ),
        flush=True,
    )


def enable_cuda_trace():
    from sglang.multimodal_gen.runtime.utils import zing_stage_timing as timing

    original = timing.StageTrace.__init__

    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        kwargs["measure_cuda"] = True
        kwargs["identity"] = {**(kwargs.get("identity") or {}), "mode": "cuda"}
        original(self, *args, **kwargs)

    timing.StageTrace.__init__ = initialize


def decoder_source(path):
    """Add events around the unchanged decode loop, on its existing stream."""
    text = Path(path).read_text()
    anchors = {
        '                        for latent, ready in job["latents"]:\n': """                        from stall_stage_probe import decoder_span
                        _stall_span, _stall_trace = decoder_span(job)
                        for latent, ready in job["latents"]:
""",
        '                        decoded = self._take(job["frames"])\n': """                        decoded = self._take(job["frames"])
                        if _stall_span is not None:
                            _stall_span.finish()
                            _stall_span.handoff()
                            _stall_trace.finish()
""",
    }
    for needle, replacement in anchors.items():
        if text.count(needle) != 1:
            raise ValueError("Decoder diagnostic anchor differs from frozen source")
        text = text.replace(needle, replacement)
    compile(text, str(path), "exec")
    return text


def decoder_span(job):
    import os

    if not Path(os.environ["STALL_DIAGNOSTIC_SWITCH"]).exists():
        return None, None
    from sglang.multimodal_gen.runtime.utils import zing_stage_timing as timing

    reference = job["reference"] or {}
    trace = timing.trace_for_ids(
        reference.get("session_id"),
        reference.get("request_id"),
        enabled=True,
        identity=dict(
            worker_role="dedicated_decoder",
            generation_id=reference.get("generation_id"),
        ),
    )
    span = timing.Span(
        trace,
        "Decoder",
        True,
        dict(
            component="lynn_decode",
            chunk_index=reference.get("chunk_index"),
            frames=job["frames"],
            scope="decode_stream_wait_latents_push_and_take_excludes_conversion_D2H_and_publication",
        ),
    )
    return span, trace


def memory_snapshot(role, phase, device):
    """Allocator counters for this process, not exclusive physical-card memory."""
    import json
    import os
    import time

    import torch
    import torch.distributed as dist

    root = Path(os.environ["STALL_MEMORY_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    row = dict(
        role=role,
        phase=phase,
        pid=os.getpid(),
        rank=dist.get_rank() if dist.is_initialized() else 0,
        monotonic_ns=time.monotonic_ns(),
        device=device,
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        allocated_bytes=torch.cuda.memory_allocated(device),
        reserved_bytes=torch.cuda.memory_reserved(device),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        scope="whole_process_allocator_on_current_device_including_shared_components",
    )
    with (root / f"memory-{os.getpid()}.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def lifecycle_memory(cls, opening, closing, role):
    original_open, original_close = getattr(cls, opening), getattr(cls, closing)
    if getattr(original_open, "_stall_memory_probe", False):
        return

    @functools.wraps(original_open)
    def begin(self, *args, **kwargs):
        import torch

        if torch.cuda.is_initialized():
            device = torch.cuda.current_device()
            torch.cuda.reset_peak_memory_stats(device)
            self._stall_memory_device = device
            memory_snapshot(role, "begin", device)
        return original_open(self, *args, **kwargs)

    @functools.wraps(original_close)
    def end(self, *args, **kwargs):
        try:
            return original_close(self, *args, **kwargs)
        finally:
            device = getattr(self, "_stall_memory_device", None)
            if device is not None:
                release_shared_allocator_cache(role, device)
                memory_snapshot(role, "end", device)
                self._stall_memory_device = None

    begin._stall_memory_probe = True
    setattr(cls, opening, begin)
    setattr(cls, closing, end)


def install_memory_probes():
    from sglang.multimodal_gen.runtime.conditioning.native import NativeConditionEncoder
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.progressive_encoder import (
        H3ProgressiveEncoderSession,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_session import (
        ZingSourceStreamState,
    )

    lifecycle_memory(
        ZingSourceStreamState,
        "initialize",
        "dispose",
        "DiT_process_including_reference_VAE",
    )
    lifecycle_memory(
        H3ProgressiveEncoderSession, "__init__", "dispose", "source_Encoder_process"
    )
    original = NativeConditionEncoder.encode_device
    if getattr(original, "_stall_memory_probe", False):
        return

    @functools.wraps(original)
    def encode(self, *args, **kwargs):
        import torch

        torch.cuda.set_device(self.gpu_id)
        torch.cuda.reset_peak_memory_stats(self.gpu_id)
        memory_snapshot("Qwen_process", "begin", self.gpu_id)
        import hashlib
        import os
        import time

        from sglang.multimodal_gen.runtime.utils import zing_stage_timing as timing

        condition = args[0] if args else kwargs["condition_input"]
        call = f"qwen-{os.getpid()}-{time.monotonic_ns()}"
        trace = timing.trace_for_ids(
            call,
            call,
            enabled=Path(os.environ["STALL_DIAGNOSTIC_SWITCH"]).exists(),
            identity={"worker_role": "dedicated_qwen", "component_group": "encoder"},
        )
        try:
            with timing.stage_span(
                trace,
                "Encoder",
                cuda=True,
                component="qwen",
                prompt_sha256=hashlib.sha256(condition.prompt.encode()).hexdigest(),
                media_count=len(condition.media),
                scope="native_encode_device_including_preprocess_and_existing_device_drain",
            ) as span:
                result = original(self, *args, **kwargs)
            timing.handoff(span)
            return result
        finally:
            release_shared_allocator_cache("Qwen_process", self.gpu_id)
            memory_snapshot("Qwen_process", "end", self.gpu_id)
            if trace is not None:
                trace.finish()

    encode._stall_memory_probe = True
    NativeConditionEncoder.encode_device = encode
