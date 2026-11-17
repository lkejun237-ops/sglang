# SPDX-License-Identifier: Apache-2.0
"""Metric units, async completion, failed calls and scrape contracts on CPU."""

import asyncio
import os
import queue
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from sglang.multimodal_gen.runtime.realtime import zing_component_metrics as metrics
from sglang.multimodal_gen.runtime.realtime.worker_http import install_health_routes


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    monkeypatch.setattr(metrics, "new_events", lambda: None)
    monkeypatch.delenv("SGLANG_ZING_COMPONENT_METRICS", raising=False)
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)


def value(name, labels):
    return REGISTRY.get_sample_value(name, labels) or 0


def labels(timer, measurement="host"):
    return dict(zip(metrics.LABELS, (*timer.labels, timer.resolution, measurement)))


def test_host_seconds_and_actual_pixel_frames(monkeypatch):
    clock = iter([10, 10.125])
    monkeypatch.setattr(metrics.time, "perf_counter", lambda: next(clock))
    timer = metrics.ComponentTimer("encoder", frames=8, width=832, height=480)
    tag = labels(timer)
    before = value("sglang_zing_component_duration_seconds_sum", tag)
    frames = value("sglang_zing_component_frames_total", tag)
    timer.finish()
    timer.finish()  # A completed call cannot be counted twice.
    assert value("sglang_zing_component_duration_seconds_sum", tag) - before == 0.125
    assert value("sglang_zing_component_frames_total", tag) - frames == 8


class Event:
    def __init__(self, *, ready=True):
        self.ready = ready
        self.recorded = 0

    def record(self):
        self.recorded += 1

    def query(self):
        return self.ready

    def elapsed_time(self, end):
        assert self.recorded == end.recorded == 1
        return 250

    def synchronize(self):
        pytest.fail("Metrics must never synchronize CUDA")


def test_cuda_is_observed_only_after_completion(monkeypatch):
    start, end = Event(), Event(ready=False)
    monkeypatch.setattr(metrics, "new_events", lambda: (start, end))
    pending = []
    monkeypatch.setattr(metrics, "submit_cuda", pending.append)
    with metrics.component_timer("dit", frames=17, width=1280, height=736) as timer:
        pass
    tag = labels(timer, "cuda")
    frames = value("sglang_zing_component_frames_total", tag)
    total = value("sglang_zing_component_duration_seconds_sum", tag)
    collector = object.__new__(metrics.CudaCollector)
    assert not collector.collect(pending[0])
    assert value("sglang_zing_component_frames_total", tag) == frames
    end.ready = True
    assert collector.collect(pending[0])
    assert value("sglang_zing_component_frames_total", tag) - frames == 17
    assert value("sglang_zing_component_duration_seconds_sum", tag) - total == 0.25


def test_failed_or_cancelled_calls_do_not_count_successful_frames():
    for error in (ValueError("failed"), asyncio.CancelledError()):
        with pytest.raises(type(error)):
            with metrics.component_timer("decoder", frames=99) as timer:
                tag = labels(timer)
                before = value("sglang_zing_component_frames_total", tag)
                raise error
        assert value("sglang_zing_component_frames_total", tag) == before


def test_bounded_queue_and_timeout_drop_samples():
    collector = object.__new__(metrics.CudaCollector)
    collector.pending = queue.Queue(maxsize=1)
    timer = metrics.ComponentTimer("qwen_vlm")
    item = (
        (*timer.labels, timer.resolution),
        Event(),
        Event(ready=False),
        1,
        time.monotonic(),
    )
    tag = {"component": "qwen_vlm", "reason": "queue_full"}
    before = value("sglang_zing_component_timing_dropped_total", tag)
    collector.submit(item)
    collector.submit(item)
    assert collector.pending.qsize() == 1
    assert value("sglang_zing_component_timing_dropped_total", tag) - before == 1
    assert collector.collect((*item[:-1], time.monotonic() - 121))


def test_disabled_metrics_do_not_create_events(monkeypatch):
    monkeypatch.setenv("SGLANG_ZING_COMPONENT_METRICS", "0")
    monkeypatch.setattr(metrics, "new_events", lambda: pytest.fail("Disabled timer"))
    with metrics.component_timer("dit", frames=1):
        pass


def test_cuda_record_failure_retains_host_sample(monkeypatch):
    start, end = Event(), Event()

    def fail_record():
        raise RuntimeError("CUDA timing unavailable")

    end.record = fail_record
    monkeypatch.setattr(metrics, "new_events", lambda: (start, end))
    timer = metrics.ComponentTimer("encoder", frames=1)
    tag = labels(timer)
    before = value("sglang_zing_component_duration_seconds_count", tag)
    timer.finish()
    assert value("sglang_zing_component_duration_seconds_count", tag) - before == 1


@pytest.mark.parametrize("reason", ["queue_full", "timeout", "event_error"])
def test_completed_frames_survive_cuda_timing_drop(monkeypatch, reason):
    start, end = Event(), Event(ready=False)
    monkeypatch.setattr(metrics, "new_events", lambda: (start, end))
    collector = object.__new__(metrics.CudaCollector)
    collector.pending = queue.Queue(maxsize=1)
    if reason == "queue_full":
        collector.pending.put_nowait(None)
        monkeypatch.setattr(metrics, "submit_cuda", collector.submit)
    elif reason == "timeout":
        monkeypatch.setattr(
            metrics,
            "submit_cuda",
            lambda item: collector.collect((*item[:-1], time.monotonic() - 121)),
        )
    else:

        def fail_query():
            raise RuntimeError("Event query failed")

        end.query = fail_query
        monkeypatch.setattr(metrics, "submit_cuda", collector.collect)
    timer = metrics.ComponentTimer("dit", frames=17, width=832, height=480)
    completed_tag = labels(timer)
    del completed_tag["measurement"]
    completed_before = value(
        "sglang_zing_component_completed_frames_total", completed_tag
    )
    cuda_before = value("sglang_zing_component_frames_total", labels(timer, "cuda"))
    timer.finish()
    timer.finish()
    assert (
        value("sglang_zing_component_completed_frames_total", completed_tag)
        - completed_before
        == 17
    )
    assert (
        value("sglang_zing_component_frames_total", labels(timer, "cuda"))
        == cuda_before
    )


@pytest.mark.parametrize("mode", ["finite", "local", "local_progressive", "standalone"])
def test_decoder_completed_frames_exclude_tail_padding(monkeypatch, mode):
    import torch

    from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.stream import (
        LynnStream,
    )
    from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.temporal import (
        TemporalProtocol,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_decoder import (
        SourceLynnDecoder,
    )
    from sglang.multimodal_gen.runtime.realtime.zing_contracts import (
        Provenance,
        SourceSpan,
    )

    protocol = TemporalProtocol(
        {"temporal_downsample_factors": [1, 2, 2], "clip_length": 17, "token_drop": 3}
    )

    def decode_step(latent, state, group):
        frames = protocol.new_frames(state.index)
        state.index += 1
        return torch.zeros(1, 3, frames, 32, 32), state

    model = SimpleNamespace(
        protocol=protocol,
        init_stream=lambda: SimpleNamespace(index=0, sm90_lease=None),
        latent_mean=torch.zeros(24),
        compute_dtype=torch.bfloat16,
        spatial_ratio=16,
        decode_step=decode_step,
    )
    model.new_stream = lambda **kwargs: LynnStream(model, **kwargs)
    timer = metrics.ComponentTimer("decoder", width=32, height=32)
    completed_tag = labels(timer)
    del completed_tag["measurement"]
    before = value("sglang_zing_component_completed_frames_total", completed_tag)
    computed_before = value("sglang_zing_component_frames_total", labels(timer))
    latent = torch.zeros(1, 24, 2, 2, 2)
    if mode == "finite":
        result = LynnStream(model, frame_limit=2).push(latent)
        assert result.shape[2] == 2
    elif mode == "standalone":
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.decoder_worker import (
            _DecodeService,
        )

        monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 1)
        monkeypatch.setattr(
            torch.distributed,
            "all_gather_object",
            lambda rows, row, group=None: rows.__setitem__(0, row),
        )
        published = []
        publisher = SimpleNamespace(
            timeout=2,
            publish=lambda ref, frames: published.append(frames),
            fail=lambda *args: pytest.fail("Decoder failed"),
        )
        service = _DecodeService(
            model,
            capacity_latents=2,
            group=None,
            publisher=publisher,
            decode_group=object(),
            device=torch.device("cpu"),
        )
        try:
            service.append(
                latent, latent_start=0, valid_frame_end=2, source_dependency_end=2
            )
            service.submit(2, {"dispatch": 0})
            assert service._last_job["done"].wait(5)
            service.check()
            assert published[0].shape[2] == 2
        finally:
            service.drain(timeout=5)
            service.dispose()
    else:
        codecs = SimpleNamespace(
            modules={"lynn": model},
            decoder=SimpleNamespace(
                module_key="lynn",
                new_stream=lambda module, **kwargs: module.new_stream(**kwargs),
            ),
        )
        decoder = SourceLynnDecoder(
            codecs,
            capacity_latents=2,
            provenance_for=lambda pixels, dep: Provenance(
                pixels, (SourceSpan(1, pixels.start, pixels.start, pixels.count),), dep
            ),
        )
        decoder.append(
            latent, latent_start=0, valid_frame_end=2, source_dependency_end=2
        )
        decoder.finish(total_frames=2)
        if mode == "local_progressive":
            assert decoder.pop(max_latents=1).values.shape[2] == 1
            assert decoder.pop(max_latents=1).values.shape[2] == 1
        else:
            assert decoder.pop().values.shape[2] == 2
        decoder.dispose()
    assert (
        value("sglang_zing_component_completed_frames_total", completed_tag) - before
        == 2
    )
    assert (
        value("sglang_zing_component_frames_total", labels(timer)) - computed_before
        == 5
    )


def test_worker_metrics_exposes_model_thread_samples():
    app = FastAPI()
    install_health_routes(app, lambda: {"ready": True})
    with metrics.component_timer("encoder", frames=8):
        pass
    with TestClient(app) as client:
        response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert 'sglang_zing_component_frames_total{component="encoder"' in response.text


def test_qwen_counts_sampled_media_before_temporal_padding():
    import torch

    from sglang.multimodal_gen.runtime.conditioning.encoder import encode_condition

    class ImageProcessor:
        merge_size = 1

        def __call__(self, **kwargs):
            return {
                "pixel_values": torch.zeros(1),
                "image_grid_thw": torch.tensor([[1, 1, 1]]),
            }

    class VideoProcessor:
        temporal_patch_size = 2

        def __call__(self, *, videos, **kwargs):
            assert (
                videos[0].shape[0] == 4
            )  # Three sampled frames plus one padding frame.
            return {
                "pixel_values_videos": torch.zeros(1),
                "video_grid_thw": torch.tensor([[2, 1, 1]]),
            }

    timer = metrics.ComponentTimer("qwen_vlm")
    tag = labels(timer)
    before = value("sglang_zing_component_frames_total", tag)
    encode_condition(
        {
            "tokenizer": None,
            "processor": SimpleNamespace(
                image_processor=ImageProcessor(), video_processor=VideoProcessor()
            ),
            "text_encoder": SimpleNamespace(
                encode_ids=lambda *args, **kwargs: torch.zeros(1, 8)
            ),
        },
        {"condition_encoder_config": {"video_sample_fps": 1}},
        "test",
        [
            {"type": "image", "frames": torch.zeros(3, 1, 4, 4)},
            {"type": "video", "fps": 1, "frames": torch.zeros(3, 3, 4, 4)},
        ],
        presentation_fn=lambda *args, **kwargs: (torch.tensor([1]), torch.tensor([0])),
    )
    assert value("sglang_zing_component_frames_total", tag) - before == 4


def test_qwen_scrape_keeps_existing_authentication():
    from sglang.multimodal_gen.runtime.conditioning.metrics import ConditioningMetrics
    from sglang.multimodal_gen.runtime.conditioning.service import ConditioningService

    service = object.__new__(ConditioningService)
    service.metrics = ConditioningMetrics()
    service.credentials = {
        "secret": {"tenant": "test"},
        "admin": {"tenant": "test", "admin": True},
    }
    service.limits = SimpleNamespace(max_header_bytes=4096)
    service._closed = False
    service._requests = set()

    async def request(headers):
        messages = []

        async def send(message):
            messages.append(message)

        await service(
            {"type": "http", "method": "GET", "path": "/metrics", "headers": headers},
            None,
            send,
        )
        return messages

    denied = asyncio.run(request([]))
    assert denied[0]["status"] == 401
    forbidden = asyncio.run(request([(b"authorization", b"Bearer secret")]))
    assert forbidden[0]["status"] == 403
    accepted = asyncio.run(request([(b"authorization", b"Bearer admin")]))
    assert accepted[0]["status"] == 200
    assert b"sglang_zing_component_duration_seconds" in accepted[1]["body"]


def test_multiprocess_export_includes_child_samples(tmp_path):
    # Prometheus chooses its value implementation at import time. The setting
    # must exist before starting Python, just as in a deployed model worker.
    code = """
import os
from prometheus_client import Counter
frames = Counter('zing_child_frames_total', 'Child frames')
if os.fork() == 0:
    frames.inc(17)
    os._exit(0)
os.wait()
from sglang.multimodal_gen.runtime.realtime.critical_path_metrics import prometheus_latest
assert b'zing_child_frames_total 17.0' in prometheus_latest()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PROMETHEUS_MULTIPROC_DIR": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_single_latent_fast_path_keeps_decoder_metrics():
    import torch

    from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.stream import (
        LynnStream,
    )

    class Model:
        latent_mean = torch.zeros(24)
        compute_dtype = torch.bfloat16
        protocol = None

        def init_stream(self):
            return SimpleNamespace(sm90_lease=None)

        def _decode_step_impl(self, latent, state, group, *, clamp_output):
            assert clamp_output
            return torch.zeros(1, 3, 4, 32, 32), state, True

    timer = metrics.ComponentTimer("decoder", width=32, height=32)
    tag = labels(timer)
    before = value("sglang_zing_component_frames_total", tag)
    completed_tag = dict(tag)
    del completed_tag["measurement"]
    completed_before = value(
        "sglang_zing_component_completed_frames_total", completed_tag
    )
    stream = LynnStream(Model(), frame_limit=2)
    try:
        assert stream.push(torch.zeros(1, 24, 1, 2, 2)).shape[2] == 2
        assert value("sglang_zing_component_frames_total", tag) - before == 4
        assert (
            value("sglang_zing_component_completed_frames_total", completed_tag)
            - completed_before
            == 2
        )
    finally:
        stream.close()
