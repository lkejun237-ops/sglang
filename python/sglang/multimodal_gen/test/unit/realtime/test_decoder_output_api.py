# SPDX-License-Identifier: Apache-2.0
"""Decoder HTTP and WS integration using the existing RGB publisher/adapter."""

import http.client
import json

import msgspec.msgpack
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang.multimodal_gen.runtime.disaggregation.transport.shm_publisher import (
    SharedMemoryFramePublisher,
)
from sglang.multimodal_gen.runtime.realtime.decoder_output import DecoderOutputBroker
from sglang.multimodal_gen.runtime.realtime.worker_http import (
    WorkerHTTPServer,
    publish_role_warmup,
)
from sglang.multimodal_gen.runtime.utils.realtime_video import (
    cancel_async_raw_rgb_frame_reference,
)


def test_http_health_stays_responsive_and_port_conflict_is_fatal():
    server = WorkerHTTPServer("dit", "127.0.0.1", 0)
    port = server.socket.getsockname()[1]
    try:
        for ready, status in [(False, 503), (True, 200), (False, 503)]:
            server.ready = ready
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            connection.request("GET", "/health")
            response = connection.getresponse()
            assert response.status == status
            assert json.loads(response.read()) == {"role": "dit", "ready": ready}
            connection.close()
        warmup_summary = {
            "role": "dit",
            "warmed_up": True,
            "elapsed_s": 1.25,
            "width": 832,
            "height": 480,
            "latent_frames": 1,
        }
        # Warmup pipelines call publish_role_warmup(role, summary) automatically so
        # worker entrypoints do not need to remember an explicit server.record_warmup().
        publish_role_warmup("dit", warmup_summary)
        server.ready = True
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        connection.request("GET", "/health")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read()) == {
            "role": "dit",
            "ready": True,
            "warmed_up": True,
            "warmup": warmup_summary,
        }
        connection.close()
        with pytest.raises(OSError):
            WorkerHTTPServer("decoder", "127.0.0.1", port)
    finally:
        server.close()
    assert not server.thread.is_alive()


@pytest.mark.parametrize("output_format", ["raw", "webp", "jpeg"])
def test_direct_decoder_reuses_frame_protocol_and_releases_shm(
    tmp_path, monkeypatch, output_format
):
    monkeypatch.setenv("SGLANG_REALTIME_VAE_SHM_DIR", str(tmp_path / "shm"))
    broker = DecoderOutputBroker(cancel_async_raw_rgb_frame_reference)
    app = FastAPI()
    broker.install(app)
    publisher = SharedMemoryFramePublisher(timeout_seconds=2, max_in_flight=10)
    identity = dict(session_id="s", generation_id="g")
    session = broker.open(
        dict(identity, width=4, height=4, fps=24, realtime_output_format=output_format)
    )
    try:
        with TestClient(app) as client:
            # A stale attachment must not cancel the valid owner.
            with client.websocket_connect("/v1/realtime_video/output") as ws:
                ws.send_bytes(
                    msgspec.msgpack.encode(
                        dict(type="output.attach", token="wrong", **identity)
                    )
                )
                assert msgspec.msgpack.decode(ws.receive_bytes())["type"] == "error"
            assert not session.stopped.is_set()
            with client.websocket_connect("/v1/realtime_video/output") as ws:
                ws.send_bytes(
                    msgspec.msgpack.encode(
                        dict(type="output.attach", token=session.token, **identity)
                    )
                )
                assert msgspec.msgpack.decode(ws.receive_bytes()) == dict(
                    type="output.ready", **identity
                )
                reference = publisher.reserve(
                    frames=1,
                    width=4,
                    height=4,
                    identity=dict(identity, request_id="r", chunk_index=0),
                )
                from sglang.multimodal_gen.runtime.utils.zing_ttff_timing import (
                    InferenceTiming,
                )

                timing = InferenceTiming("decoder", event_id=7, prompt_version=2)
                timing.record("vae_decode", 12.5)
                timing.finished = True
                session.put(
                    dict(
                        type="decoder.frames",
                        timing=timing,
                        reference=reference,
                        metadata=dict(
                            format="rgb24",
                            width=4,
                            height=4,
                            channels=3,
                            bytes_per_frame=48,
                        ),
                        output_metadata=dict(
                            output_frame_start=0,
                            num_frames=1,
                            source_dependency_end=1,
                            source_alignment=[
                                dict(
                                    source_id=1,
                                    source_frame_start=0,
                                    output_frame_start=0,
                                    num_frames=1,
                                )
                            ],
                            block_idx=0,
                            event_id=7,
                            prompt_version=2,
                        ),
                    ),
                    1,
                )
                publisher.publish(reference, torch.ones(1, 3, 1, 4, 4))
                session.source_progress(1, [[1, 0, 1]], 1)
                session.put(dict(type="session.completed", output_frames=1), 1)
                frame = msgspec.msgpack.decode(ws.receive_bytes())
                assert frame["type"] == "frame_batch"
                assert frame["event_id"] == 7 and frame["prompt_version"] == 2
                assert (
                    frame["output_frame_start"] == 0
                    and frame["source_dependency_end"] == 1
                )
                assert frame["encoding"] == output_format
                if output_format == "raw":
                    assert frame["payload"] == bytes([255]) * 48
                report = msgspec.msgpack.decode(ws.receive_bytes())
                assert report["type"] == "model.timing"
                assert (
                    report["model_session_id"] == "s" and report["generation_id"] == "g"
                )
                assert (
                    report["timing"]["event_id"] == 7
                    and report["timing"]["prompt_version"] == 2
                )
                assert report["timing"]["stages"][0]["host_ms"] == 12.5
                consumed = msgspec.msgpack.decode(ws.receive_bytes())
                assert consumed == dict(
                    type="source.consumed", source_frame_end=1, source_bytes_end=48
                )
                completion = msgspec.msgpack.decode(ws.receive_bytes())
                assert completion == dict(
                    type="session.completed", output_frames=1, **identity
                )
                assert session.done.wait(2) and session.completed
        broker.close(session)
        next_session = broker.open(dict(identity, generation_id="next"))
        broker.close(next_session)
    finally:
        broker.close(broker.session)
        publisher.close()
    assert not list((tmp_path / "shm").glob("*"))
