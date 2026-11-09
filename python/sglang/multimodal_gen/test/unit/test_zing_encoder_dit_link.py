# SPDX-License-Identifier: Apache-2.0
"""Loopback transport and queued DiT lifecycle; real-checkpoint tests are separate."""

import asyncio
import json
import multiprocessing
import socket
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_io import (
    DiTIOChannel,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_link import (
    DiTInputService,
    DiTLinkClient,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_transport import (
    EncoderDiTChannel,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_worker import (
    argument_parser,
    create_link_session,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.packing import (
    H3InferenceInputs,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (  # noqa: F401
    cpu_execution_options,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_queued_dit import (
    Harness,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_queued_dit import (  # noqa: F401
    model as model_fixture,
)

model = model_fixture


def test_dit_cache_storage_is_explicit_and_preserves_mount_validation(
    tmp_path, monkeypatch
):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        block0_identity,
    )

    monkeypatch.setattr(block0_identity, "storage_mounted", lambda _: False)
    argv = [
        "--model",
        str(tmp_path),
        "--zing-block0-cache-mode",
        "on",
        "--zing-block0-cache-host-gb",
        "416",
        "--zing-block0-cache-disk-gb",
        "700",
        "--zing-block0-cache-dir",
        str(tmp_path),
    ]
    mounted = argument_parser().parse_args(argv)
    assert mounted.zing_block0_cache_storage == "mounted"
    with pytest.raises(ValueError, match="mounted before"):
        block0_identity.startup_config(mounted)
    local = argument_parser().parse_args(
        argv + ["--zing-block0-cache-storage", "local"]
    )
    assert block0_identity.startup_config(local) == (
        416 * 10**9,
        700 * 10**9,
        tmp_path.resolve(),
    )
    local.zing_block0_cache_dir = str(tmp_path / "missing")
    with pytest.raises(ValueError, match="existing mounted directory"):
        block0_identity.startup_config(local)


def _echo_peer(port, backend):
    device = "cuda:1" if backend == "nccl" else "cpu"
    if backend == "nccl":
        torch.cuda.set_device(1)

    def connect():
        if backend == "nccl":
            torch.cuda.set_device(1)
        return EncoderDiTChannel.connect(
            port, rank=1, device=device, backend=backend, timeout=30
        )

    channel = DiTIOChannel(connect).start()
    try:
        # A distinct consumer stream exercises receive completion, allocator
        # ownership and result completion before the I/O thread sends it.
        context = (
            torch.cuda.stream(torch.cuda.Stream(device=device))
            if backend == "nccl"
            else nullcontext()
        )
        with context:
            for _ in range(3):
                deadline = time.monotonic() + 30
                while not channel.poll():
                    assert time.monotonic() < deadline
                    time.sleep(0.005)
                value = channel.receive()
                if value["type"] == "conditioning":
                    value["tensors"] = {
                        k: tensor + 7 for k, tensor in value["tensors"].items()
                    }
                channel.send(value)
    finally:
        channel.close()


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_two_process_loopback_control_and_native_tensors(backend):
    if backend == "nccl" and torch.cuda.device_count() < 2:
        pytest.skip("NCCL bridge requires two distinct GPUs")
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
    process = multiprocessing.get_context("spawn").Process(
        target=_echo_peer, args=(port, backend)
    )
    process.start()
    channel = None
    try:
        device = "cuda:0" if backend == "nccl" else "cpu"
        if backend == "nccl":
            torch.cuda.set_device(0)
        channel = EncoderDiTChannel.connect(
            port, rank=0, device=device, backend=backend, timeout=30
        )
        # No tensor bridge calls at all for init/EOS, including no size tensors.
        import sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_transport as transport

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                transport,
                "send_bridge_tensor",
                lambda *a, **k: pytest.fail("Control sent over NCCL"),
            )
            patch.setattr(
                transport,
                "receive_bridge_tensor",
                lambda *a, **k: pytest.fail("Control received over NCCL"),
            )
            message = dict(
                type="session.init",
                session_id="s",
                width=832,
                height=480,
                fps=24,
                seed=7,
            )
            channel.send(message)
            assert channel.receive() == message
        values = {
            "context": torch.arange(12, device=device, dtype=torch.float32)
            .reshape(4, 3)
            .T,
            "row_role": torch.arange(4, device=device, dtype=torch.int64),
        }
        reference = torch.arange(16, device=device, dtype=torch.float32).reshape(
            1, 2, 2, 2, 2
        )
        channel.send(
            dict(
                type="conditioning",
                tensors=values,
                event_id=3,
                inputs=dict(visual={"reference": reference}, audio={}),
            )
        )
        result = channel.receive()
        assert result["event_id"] == 3
        torch.testing.assert_close(
            result["inputs"]["visual"]["reference"], reference, rtol=0, atol=0
        )
        for key, value in values.items():
            torch.testing.assert_close(
                result["tensors"][key], value + 7, rtol=0, atol=0
            )
        channel.send(dict(type="source.end", latent_frames=36))
        assert channel.receive() == dict(type="source.end", latent_frames=36)
        # NCCL shutdown requires both peers to close before joining either one.
        channel.close()
        channel = None
        process.join(30)
        assert process.exitcode == 0
    finally:
        if channel is not None:
            channel.close()
        if process.is_alive():
            process.terminate()
            process.join(10)


class LocalChannel:
    """Drive the actual service/queue on receipt without hiding wire ordering."""

    def __init__(self):
        self.requests, self.replies = [], []
        self.service = None
        self.closed = False

    def send(self, message):
        self.requests.append(message)
        self.service.queue.append(message)
        asyncio.run(self.service.queue.run_once())

    def receive(self):
        return self.replies.pop(0)

    def close(self):
        self.closed = True


def test_async_sender_continues_without_processed_ack_and_drains_at_eos():
    from collections import deque

    class Channel:
        def __init__(self):
            self.sent, self.replies = [], deque()

        def send(self, message, **kwargs):
            self.sent.append(message)
            if message["type"] == "source.end":
                self.replies.extend(
                    [
                        dict(
                            type="decoder.input_sent",
                            transfer_id=0,
                            session_id="s",
                            generation_id="g",
                        ),
                        dict(type="input.processed", transfer_id=0),
                        dict(type="input.processed", transfer_id=1),
                        dict(type="input.processed", transfer_id=2),
                    ]
                )

        def poll(self):
            return bool(self.replies)

        def receive(self):
            assert self.replies, "Data send must not wait for a reply"
            return self.replies.popleft()

    channel = Channel()
    client = DiTLinkClient(channel)
    client.owner, client.async_inputs = ("s", "g"), True
    assert client.send("conditioning") == []
    assert client.send("source.moments") == []
    assert [m["type"] for m in channel.sent] == ["conditioning", "source.moments"]
    results = client.send("source.end")
    assert [r["type"] for r in results] == ["decoder.input_sent"]
    assert not channel.replies


def test_async_sender_rejects_late_foreign_owner_notice():
    class Channel:
        def poll(self):
            return True

        def receive(self):
            return dict(
                type="decoder.input_sent",
                transfer_id=0,
                session_id="other",
                generation_id="g",
            )

    client = DiTLinkClient(Channel())
    client.owner, client.async_inputs, client.sequence = ("s", "g"), True, 1
    with pytest.raises(ValueError, match="another session"):
        client.collect()
    assert client.failed


def input_metadata(h):
    return dict(
        output_type="video",
        conditions=h.inputs.conditions,
        media=h.inputs.media,
        visual={},
        audio={},
        visual_shape=h.inputs.visual_shape,
        audio_shape=None,
        pixel_frame_count=43,
        sample_count=None,
        sample_rate=24000,
    )


def pooled_stage(h, codecs):
    """Use the original admission builder and real fixed storage on this device."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.stage import (
        ZingH3GenerationStage,
    )
    from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
        ZingKVGeometry,
        ZingKVPool,
        ZingKVPoolConfig,
        ZingKVPoolPlan,
        peak_rows,
    )

    sampler = h.sampler()
    config = ZingKVPoolConfig(
        max_target_area=4,
        max_target_long_edge=2,
        window_chunks=sampler.window_policy.window_chunks,
        sink_chunks=sampler.window_policy.sink_chunks,
        window_eviction_timing=sampler.window_eviction_timing,
    )
    geometry = ZingKVGeometry(
        1,
        tuple(h.model.arch.patch_size),
        2,
        h.packer.chunk_plan,
        video_t_span=h.packer._video_t_span,
        time_span_period=len(h.packer.visual_token_frame_spans),
    )
    stage = object.__new__(ZingH3GenerationStage)
    stage.codecs = codecs
    codecs.validate_prepared_compute_inputs = lambda inputs: None
    stage.packer = h.packer
    stage.kv_pool = ZingKVPool.allocate(
        ZingKVPoolPlan.build(
            config,
            geometry,
            token_capacity=peak_rows(
                geometry,
                config.window_policy,
                prefix_rows=64,
                rows_per_frame=geometry.rows_per_frame(2, 2),
                has_source=True,
                window_eviction_timing=config.window_eviction_timing,
            ),
        ),
        num_layers=h.model.arch.num_layers,
        local_heads=h.model.arch.num_attention_heads,
        head_dim=h.model.arch.attention_head_dim,
        dtype=torch.bfloat16,
        device=next(h.model.parameters()).device,
    )
    return stage


@torch.no_grad()
@pytest.mark.parametrize("pooled", [False, True])
def test_session_init_prompt_boundary_eos_cancel_and_reuse(model, pooled):  # noqa: F811
    h = Harness(model)
    link = LocalChannel()
    created = []
    stage = None

    def create(init, output):
        nonlocal stage
        codecs = SimpleNamespace(
            target_fps=24,
            spatial_ratio=1,
            temporal_ratio=4,
            vae_frames_per_chunk=17,
            vae_latents_per_chunk=5,
        )
        codecs.sample_streaming_visual_moments = lambda v, **k: (
            v
        )  # transport fixture uses sampled source
        if pooled and stage is None:
            stage = pooled_stage(h, codecs)
        worker = create_link_session(h.sampler(), codecs, init, output, stage=stage)
        created.append(worker)
        return worker

    link.service = DiTInputService(SimpleNamespace(send=link.replies.append), create)
    client = DiTLinkClient(link)
    init = dict(session_id="s", generation_id="g", seed=7, width=2, height=2, fps=24)
    client.open(init)
    assert created[-1].state is None  # init never waits for Qwen or creates model KV
    # Reject a second session and a stale generation without touching this owner.
    for bad in (
        dict(init, type="session.init", session_id="other"),
        dict(init, type="source.end", generation_id="stale"),
    ):
        link.send(dict(bad, transfer_id=client.sequence))
        client.sequence += 1
        assert link.receive()["type"] == "input.rejected"
        assert link.service.owner == ("s", "g")
    client.release("s")  # before first S0/Qwen
    assert created[-1].closed
    outputs = []
    client.open(dict(init, generation_id="next"))
    client.send(
        "conditioning",
        tensors=h.condition(10),
        event_id=10,
        prompt_version=1,
        inputs=input_metadata(h),
    )
    current = created[-1]
    for start, end in ((0, 1), (1, 3), (3, 5), (5, 8), (8, 10), (10, 13)):
        if start == 1:
            client.send(
                "conditioning", tensors=h.condition(11), event_id=11, prompt_version=2
            )
        outputs.extend(
            client.send(
                "source.moments",
                tensor=h.inputs.source_visual[:, :, start:end],
                latent_start=start,
                latent_end=end,
                source_dependency_end=43,
                source_records=[[1, 0, 43]],
                final_source_frames=43,
            )
        )
    if pooled:
        assert stage.kv_pool.available == 0
        assert current.state.cache.pooled
        assert current.state.cache.stats["store_allocations"] == 0
    outputs.extend(client.send("source.end", latent_frames=13))
    if pooled:
        assert stage.kv_pool.available == 1
    generated = [v for v in outputs if v["type"] == "dit.latents"]
    assert [v["event_id"] for v in generated] == [10, 10, 10, 11, 11, 11]
    assert outputs[-1]["type"] == "session.completed"
    assert outputs[-1]["source_frames"] == 43 and current.closed
    assert len(generated) == 6
    # Original direct SourceReadySession with the same model, source and noise.
    inputs = H3InferenceInputs(
        **input_metadata(h), positive=h.condition(10), source_visual=None
    )
    direct = h.session(h.sampler(), inputs)
    try:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.streaming_noise import (
            indexed_video_noise,
        )
        from sglang.multimodal_gen.test.unit.test_zing_h3_chunks import (
            fixture_provenance,
        )

        for item in generated:
            start, end = item["latent_start"], item["latent_end"]
            if start == 1:
                direct.update_prompt(h.condition(11), 11, 2)
            value = h.inputs.source_visual[:, :, start:end]
            noise = indexed_video_noise(
                value.shape,
                seed=7,
                latent_start=start,
                purpose="target_initial",
                device=value.device,
                dtype=value.dtype,
            )
            direct.append(
                value,
                noise,
                provenance=fixture_provenance(
                    1,
                    43,
                    item["output_frame_start"],
                    item["output_frame_start"] + item["num_frames"],
                ),
                final=end == 13,
            )
            actual = direct.step()
            torch.testing.assert_close(item["tensor"], actual.visual, rtol=0, atol=0)
    finally:
        direct.dispose()
    client.release("s")
    client.open(dict(init, session_id="third"))
    client.shutdown()
    assert link.closed and link.service.stopped and created[-1].closed


def test_service_polls_five_ms_and_drains_queued_controls_without_new_arrival(
    monkeypatch,
):
    inbox, replies, ticks, closed = [], [], [], []
    channel = SimpleNamespace(
        poll=lambda: bool(inbox), receive=lambda: inbox.pop(0), send=replies.append
    )
    worker = SimpleNamespace(close=lambda: closed.append(True))
    service = DiTInputService(channel, lambda init, output: worker)

    async def tick(delay):
        ticks.append(delay)
        if len(ticks) == 2:
            # A backlog exists independently of any transport notification.
            service.queue.append(
                dict(
                    type="session.init",
                    transfer_id=0,
                    session_id="s",
                    generation_id="g",
                )
            )
            service.queue.append(dict(type="worker.shutdown", transfer_id=1))
        assert len(ticks) <= 2

    monkeypatch.setattr(asyncio, "sleep", tick)
    asyncio.run(service.run())
    assert ticks == [0.005, 0.005] and closed
    assert [r["transfer_id"] for r in replies] == [0, 1]
    assert not service.queue and service.stopped


def test_decoder_seam_routes_latents_and_drains_before_session_completion():
    trace, replies = [], []

    class Decoder:
        def source_progress(self, end, records):
            assert end == 1 and records == [[1, 0, 1]]

        def output(self, result, message, item):
            trace.append(("decode", message["request_id"], item["event_id"]))
            return dict(
                {k: v for k, v in item.items() if k != "tensor"},
                type="decoder.frames",
                reference=None,
            )

        def finish(self, frames):
            assert frames == 1
            trace.append("decoder_eos")

        def close(self):
            trace.append("decoder_close")

    def create(init, output):
        worker = SimpleNamespace(
            queue=[],
            closed=False,
            initial_received=True,
            complete=False,
            bridge=SimpleNamespace(
                clip_latents=5,
                clip_frames=17,
                ratio=4,
                accepted_latents=1,
                accepted_pixels=1,
            ),
            close=lambda: None,
        )

        async def put(message):
            if message["type"] == "source.moments":
                await output(
                    SimpleNamespace(
                        visual=torch.ones(1),
                        chunk=SimpleNamespace(start=0, end=1),
                        prompt_event_id=7,
                        prompt_version=2,
                        provenance=SimpleNamespace(metadata=lambda: {}),
                    )
                )
            elif message["type"] == "source.end":
                worker.complete = True

        worker.put = put
        return worker

    def send(reply):
        trace.append(reply["type"])
        replies.append(reply)

    service = DiTInputService(
        SimpleNamespace(send=send), create, lambda init: Decoder()
    )

    async def run():
        for i, kind in enumerate(
            ("session.init", "conditioning", "source.moments", "source.end")
        ):
            service.queue.append(
                dict(
                    type=kind,
                    session_id="s",
                    generation_id="g",
                    transfer_id=i,
                    request_id="request",
                    chunk_index=0,
                    latent_end=1,
                    source_records=[[1, 0, 1]],
                )
            )
            await service.queue.run_once()

    asyncio.run(run())
    assert replies[0]["output_kind"] == "rgb"
    assert ("decode", "request", 7) in trace
    assert any(r["type"] == "decoder.frames" and "tensor" not in r for r in replies)
    assert (
        trace.index("decoder_eos")
        < trace.index("decoder_close")
        < trace.index("session.completed")
    )
    assert trace.count("decoder_close") == 1


@torch.no_grad()
@pytest.mark.parametrize("reference_ids", [["new-a", "new-b"], []])
@pytest.mark.parametrize("pooled", [False, True])
def test_reference_update_and_cancel_match_original_session(
    model, reference_ids, pooled
):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.streaming_noise import (
        indexed_video_noise,
    )

    h = Harness(model)
    device = next(model.parameters()).device
    metadata = input_metadata(h)
    metadata.update(
        conditions=[
            dict(content_id="old", role="reference", latent_modalities=["visual"])
        ],
        media=[dict(id="old", type="image")],
        visual={"old": torch.zeros(1, 2, 1, 2, 2, device=device)},
    )
    outputs = []
    codecs = SimpleNamespace(
        target_fps=24,
        spatial_ratio=1,
        temporal_ratio=4,
        vae_frames_per_chunk=17,
        vae_latents_per_chunk=5,
    )
    codecs.sample_streaming_visual_moments = lambda value, **kw: value

    async def output(result):
        outputs.append(result)

    stage = pooled_stage(h, codecs) if pooled else None
    worker = create_link_session(
        h.sampler(),
        codecs,
        dict(
            session_id="s",
            generation_id="g",
            seed=7,
            width=2,
            height=2,
            fps=24,
        ),
        output,
        stage=stage,
    )
    worker.source_records = [[1, 0, 43]]
    updated = dict(
        metadata,
        conditions=[
            dict(content_id=k, role="reference", latent_modalities=["visual"])
            for k in reference_ids
        ],
        media=[dict(id=k, type="image") for k in reference_ids],
        visual={
            k: torch.full((1, 2, 1, 2, 2), 0.75, device=device) for k in reference_ids
        },
    )

    async def put(kind, **data):
        await worker.put(dict(type=kind, session_id="s", generation_id="g", **data))
        while worker.queue:
            await worker.queue.run_once()

    async def run():
        await put("conditioning", tensors=h.condition(10), event_id=10, inputs=metadata)
        for start, end in [(0, 1), (1, 3), (3, 5), (5, 8), (8, 10), (10, 13)]:
            if start == 1:
                await put(
                    "conditioning", tensors=h.condition(11), event_id=11, inputs=updated
                )
                await put("conditioning.cancel", event_id=12)
            if start == 5:
                await put(
                    "conditioning", tensors=h.condition(13), event_id=13, inputs=updated
                )
            if start == 8:
                await put("conditioning", tensors=h.condition(14), event_id=14)
            clip, offset = divmod(end - 1, 5)
            pixels = clip * 17 + 1 + offset * 4
            await put(
                "source.latents",
                tensor=h.inputs.source_visual[:, :, start:end],
                latent_start=start,
                latent_end=end,
                pixel_end=pixels,
                source_dependency_end=pixels,
            )
            if start == 5:
                assert set(worker.state.inputs.visual) == set(reference_ids)
        if pooled:
            assert stage.kv_pool.available == 0
            assert worker.state.cache.pooled
            assert worker.state.cache.stats["store_allocations"] == 0
        await put("source.end", latent_frames=13)

    asyncio.run(run())
    assert [r.prompt_event_id for r in outputs] == [10, 10, 10, 13, 13, 14]
    direct_inputs = H3InferenceInputs(
        **metadata, positive=h.condition(10), source_visual=None
    )
    direct = h.session(h.sampler(), direct_inputs)
    try:
        for item in outputs:
            start, end = item.chunk.start, item.chunk.end
            if start == 1:
                direct.update_prompt(
                    h.condition(11),
                    11,
                    inputs=H3InferenceInputs(
                        **updated, positive=h.condition(11), source_visual=None
                    ),
                )
                direct.discard_pending_prompt()
            if start == 5:
                direct.update_prompt(
                    h.condition(13),
                    13,
                    inputs=H3InferenceInputs(
                        **updated, positive=h.condition(13), source_visual=None
                    ),
                )
            if start == 8:
                direct.update_prompt(h.condition(14), 14)
            value = h.inputs.source_visual[:, :, start:end]
            noise = indexed_video_noise(
                value.shape,
                seed=7,
                latent_start=start,
                purpose="target_initial",
                device=device,
                dtype=value.dtype,
            )
            direct.append(value, noise, provenance=item.provenance, final=end == 13)
            torch.testing.assert_close(
                item.visual, direct.step().visual, rtol=0, atol=0
            )
    finally:
        direct.dispose()
        worker.close()
        worker.close()
        if pooled:
            assert stage.kv_pool.available == 1


@torch.no_grad()
@pytest.mark.parametrize("fail_output", [False, True])
def test_pooled_session_releases_after_cancel_or_output_failure(model, fail_output):
    h = Harness(model)
    codecs = SimpleNamespace(
        target_fps=24,
        spatial_ratio=1,
        temporal_ratio=4,
        vae_frames_per_chunk=17,
        vae_latents_per_chunk=5,
        sample_streaming_visual_moments=lambda value, **kw: value,
    )
    stage = pooled_stage(h, codecs)
    pointers = []

    async def run(fail):
        async def output(result):
            cache = worker.state.cache
            assert cache.pooled and stage.kv_pool.available == 0
            pointers.append([store.k.data_ptr() for store in cache.storage])
            if fail:
                raise RuntimeError("Injected output failure")

        worker = create_link_session(
            h.sampler(),
            codecs,
            dict(session_id="s", generation_id="g", seed=7, width=2, height=2, fps=24),
            output,
            stage=stage,
        )
        worker.source_records = [[1, 0, 1]]

        async def put(kind, **data):
            await worker.put(dict(type=kind, session_id="s", generation_id="g", **data))
            while worker.queue:
                await worker.queue.run_once()

        await put(
            "conditioning",
            tensors=h.condition(10),
            event_id=10,
            inputs=input_metadata(h),
        )
        assert stage.kv_pool.available == 1  # no lease before the first source step
        try:
            if fail:
                with pytest.raises(RuntimeError, match="Injected output failure"):
                    await put(
                        "source.latents",
                        tensor=h.inputs.source_visual[:, :, :1],
                        latent_start=0,
                        latent_end=1,
                        pixel_end=1,
                        source_dependency_end=1,
                    )
                assert worker.closed and stage.kv_pool.available == 1
            else:
                await put(
                    "source.latents",
                    tensor=h.inputs.source_visual[:, :, :1],
                    latent_start=0,
                    latent_end=1,
                    pixel_end=1,
                    source_dependency_end=1,
                )
                assert stage.kv_pool.available == 0
        finally:
            worker.close()
            worker.close()
        assert stage.kv_pool.available == 1

    asyncio.run(run(fail_output))
    asyncio.run(run(False))
    assert pointers[0] == pointers[1]  # subsequent sessions reuse the same storage


@pytest.mark.parametrize("external_inputs", [False, True])
def test_stage_initialization_preserves_encoder_startup_unless_inputs_are_external(
    model, monkeypatch, external_inputs
):
    from sglang.multimodal_gen.runtime.pipelines_core.stages import base
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        stage as module,
    )

    calls = []
    codecs = SimpleNamespace(
        prepare_visual_encoder_weights=lambda: calls.append("prepare"),
        warmup_encoder=lambda: calls.append("warmup"),
    )
    monkeypatch.setattr(base, "get_global_server_args", lambda: None)
    monkeypatch.setattr(module, "ZingH3Codecs", lambda *a, **kw: codecs)
    config = SimpleNamespace(
        dit_config=SimpleNamespace(arch_config=model.arch),
        prompt_switch_prefix_latent_frames=5,
    )
    runtime = dict(
        num_frame_per_block=[3, 2],
        split_first_latent=True,
        rope_config=dict(
            spatial_extent=1.0, frame_scale=1.0, visual_token_frame_spans=[1]
        ),
    )
    stage = module.ZingH3GenerationStage(
        {"transformer": model},
        config,
        runtime,
        external_inputs=external_inputs,
    )
    assert calls == ([] if external_inputs else ["prepare", "warmup"])
    assert hasattr(stage, "local_conditioner") == (not external_inputs)
    assert stage.packer.visual_channels == model.arch.latents_dim
    assert stage.packer.patch_size == tuple(model.arch.patch_size)
    assert stage.prompt_switch_prefix_latent_frames == 5


@torch.no_grad()
@pytest.mark.parametrize("cached", [False, True])
def test_pool_rejection_is_nonfatal_and_discards_pending_prompt(model, cached):
    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError

    h = Harness(model)
    codecs = SimpleNamespace(
        target_fps=24,
        spatial_ratio=1,
        temporal_ratio=4,
        vae_frames_per_chunk=17,
        vae_latents_per_chunk=5,
        sample_streaming_visual_moments=lambda value, **kw: value,
    )
    stage = pooled_stage(h, codecs)
    channel = LocalChannel()
    created = []

    def create(init, output):
        worker = create_link_session(h.sampler(), codecs, init, output, stage=stage)
        created.append(worker)
        return worker

    def reject_cached(message, *, initial):
        raise ConditionError("CONDITION_INVALID", "This update is too large")

    create.cached_condition = reject_cached
    channel.service = DiTInputService(
        SimpleNamespace(send=channel.replies.append), create
    )
    client = DiTLinkClient(channel)
    client.open(
        dict(session_id="s", generation_id="g", seed=7, width=2, height=2, fps=24)
    )
    try:
        client.send(
            "conditioning",
            event_id=10,
            tensors=h.condition(10),
            inputs=input_metadata(h),
        )
        first = client.send(
            "source.moments",
            tensor=h.inputs.source_visual[:, :, :1],
            latent_start=0,
            latent_end=1,
            source_dependency_end=43,
            source_records=[[1, 0, 43]],
            final_source_frames=43,
        )
        assert first[0]["event_id"] == 10 and stage.kv_pool.available == 0
        client.send("conditioning", event_id=11, tensors=h.condition(11))
        condition = h.condition(12)
        oversized = dict(
            context=condition["context"][:1].expand(65, -1).clone(),
            row_role=torch.zeros(
                65, dtype=torch.int64, device=condition["context"].device
            ),
        )
        rejected = client.send(
            "conditioning",
            event_id=12,
            **(
                {"cache_hit": True, "cache_key": "a" * 64}
                if cached
                else {"tensors": oversized}
            ),
        )
        assert len(rejected) == 1 and rejected[0]["type"] == "conditioning.rejected"
        assert (
            rejected[0]["event_id"] == 12 and rejected[0]["code"] == "CONDITION_INVALID"
        )
        assert not client.failed and not created[0].closed
        output = []
        for start, end in [(1, 3), (3, 5), (5, 8)]:
            output += client.send(
                "source.moments",
                tensor=h.inputs.source_visual[:, :, start:end],
                latent_start=start,
                latent_end=end,
                source_dependency_end=43,
                source_records=[[1, 0, 43]],
                final_source_frames=43,
            )
        assert output and all(item["event_id"] == 10 for item in output)
        client.send("conditioning", event_id=13, tensors=h.condition(13))
        assert not client.failed
    finally:
        client.release("s")
    assert stage.kv_pool.available == 1


from sglang.multimodal_gen.test.unit.realtime.zing_cache_fixtures import (  # noqa: E402,F401
    cpu_dma as cpu_dma_fixture,
)

cpu_dma = cpu_dma_fixture


@torch.no_grad()
@pytest.mark.parametrize("joint_forward", ["off", "cross_chunk"])
def test_cached_prefix_matches_uncached_queued_dit(model, cpu_dma, joint_forward):
    if next(model.parameters()).is_cuda:
        pytest.skip("CPU DMA fixture; real CUDA restore is validated separately")

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
        fill_prepared_prefix,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_bridge import (
        source_pixel_end,
    )
    from sglang.multimodal_gen.test.unit.realtime.zing_cache_fixtures import cpu_cache

    h = Harness(model, joint_forward=joint_forward)
    codecs = SimpleNamespace(
        target_fps=24,
        spatial_ratio=1,
        temporal_ratio=4,
        vae_frames_per_chunk=17,
        vae_latents_per_chunk=5,
    )
    codecs.sample_streaming_visual_moments = lambda value, **kw: value
    stage = pooled_stage(h, codecs)
    stage.modules = {"transformer": model}
    stage._use = lambda *names: nullcontext(model)
    prepared = H3InferenceInputs(
        **input_metadata(h), positive=h.condition(10), source_visual=None
    )
    init = dict(session_id="s", generation_id="g", seed=7, width=2, height=2, fps=24)

    stage.block0_cache = cpu_cache(
        stage.kv_pool._pool._slots[0].kv_cache, rows=64, page_rows=2
    )

    async def run(cached):
        results = []

        async def output(result):
            results.append(result.visual.clone())

        worker = create_link_session(h.sampler(), codecs, init, output, stage=stage)
        worker.source_records = [[1, 0, 43]]
        await worker.put(
            dict(
                type="conditioning",
                session_id="s",
                generation_id="g",
                tensors=h.condition(10),
                event_id=10,
                inputs=input_metadata(h),
                cache_key="a" * 64,
                cache_hit=cached,
                cache_write=not cached,
            )
        )
        while worker.queue:
            await worker.queue.run_once()
        for start, end in ((0, 1), (1, 3), (3, 5), (5, 8), (8, 10), (10, 13)):
            if start == 1:
                await worker.put(
                    dict(
                        type="conditioning",
                        session_id="s",
                        generation_id="g",
                        tensors=h.condition(11),
                        event_id=11,
                        cache_key="b" * 64,
                        cache_hit=cached,
                        cache_write=not cached,
                    )
                )
                while worker.queue:
                    await worker.queue.run_once()
            await worker.put(
                dict(
                    type="source.latents",
                    session_id="s",
                    generation_id="g",
                    tensor=h.inputs.source_visual[:, :, start:end],
                    latent_start=start,
                    latent_end=end,
                    pixel_end=source_pixel_end(end, worker.bridge, frame_end=43),
                    source_dependency_end=43,
                )
            )
            while worker.queue:
                await worker.queue.run_once()
        await worker.put(
            dict(type="source.end", session_id="s", generation_id="g", latent_frames=13)
        )
        while worker.queue:
            await worker.queue.run_once()
        worker.close()
        return results

    baseline = asyncio.run(run(False))
    assert set(stage.block0_cache.entries) == {"a" * 64, "b" * 64}
    assert torch.equal(
        stage.block0_cache.entries["b" * 64].inputs.positive["context"],
        h.condition(11)["context"],
    )
    stage.block0_cache = cpu_cache(
        stage.kv_pool._pool._slots[0].kv_cache, rows=64, page_rows=2
    )
    fill_prepared_prefix(
        stage,
        h.sampler(),
        prepared,
        owner=SimpleNamespace(),
        key="a" * 64,
        identity=None,
        request={},
        source_shape=(2, 2),
    )
    updated = H3InferenceInputs(
        **input_metadata(h), positive=h.condition(11), source_visual=None
    )
    fill_prepared_prefix(
        stage,
        h.sampler(),
        updated,
        owner=SimpleNamespace(),
        key="b" * 64,
        identity=None,
        request={},
        source_shape=(2, 2),
    )
    assert len(stage.block0_cache.entries["a" * 64].page_ids) > 1
    actual = asyncio.run(run(True))
    assert len(actual) == len(baseline) > 0
    assert all(torch.equal(a, b) for a, b in zip(actual, baseline))


@torch.no_grad()
@pytest.mark.parametrize("model", ["cpu"], indirect=True)
@pytest.mark.parametrize("same_key", [True, False])
def test_link_reuses_only_current_complete_condition_key(model, same_key):
    h = Harness(model, split_first_latent=False)
    codecs = SimpleNamespace(
        target_fps=24,
        spatial_ratio=1,
        temporal_ratio=4,
        vae_frames_per_chunk=17,
        vae_latents_per_chunk=5,
    )
    codecs.sample_streaming_visual_moments = lambda v, **k: v
    stage = pooled_stage(h, codecs)
    stage.block0_cache = None
    init = dict(session_id="s", generation_id="g", seed=7, width=2, height=2, fps=24)

    async def run():
        results = []

        async def output(r):
            results.append(r)

        worker = create_link_session(h.sampler(), codecs, init, output, stage=stage)
        worker.source_records = [[1, 0, 43]]

        async def put(kind, **payload):
            await worker.put(
                dict(type=kind, session_id="s", generation_id="g", **payload)
            )
            while worker.queue:
                await worker.queue.run_once()

        async def source(a, b):
            await put(
                "source.moments",
                tensor=h.inputs.source_visual[:, :, a:b],
                latent_start=a,
                latent_end=b,
                pixel_end=(b // 5) * 17 + (0 if b % 5 == 0 else 1 + (b % 5 - 1) * 4),
                source_dependency_end=43,
                source_records=[[1, 0, 43]],
                final_source_frames=43,
            )

        try:
            await put(
                "conditioning",
                tensors=h.condition(10),
                event_id=10,
                prompt_version=1,
                inputs=input_metadata(h),
                cache_key="a" * 64,
            )
            await source(0, 3)
            await source(3, 5)
            prefix = worker.state.cache.prefix
            before = [
                (k.data_ptr(), v.data_ptr(), k.clone(), v.clone())
                for k, v in prefix.layers
            ]
            await put(
                "conditioning",
                tensors=h.condition(10),
                event_id=11,
                prompt_version=2,
                inputs=input_metadata(h),
                cache_key=("a" if same_key else "b") * 64,
                _reuse_gpu_prefix=True,
            )  # Wire flag alone is not trusted.
            assert worker.state.pending_prompt.reuse_prefix is same_key
            await source(5, 8)
            assert results[-1].prompt_event_id == 11 and results[-1].prompt_version == 2
            assert worker.state.metrics.get("gpu_prefix_reuses", 0) == int(same_key)
            if same_key:
                assert worker.state.cache.prefix is prefix
                for (k, v), (kp, vp, k0, v0) in zip(prefix.layers, before):
                    assert k.data_ptr() == kp and v.data_ptr() == vp
                    assert torch.equal(k, k0) and torch.equal(v, v0)
        finally:
            worker.close()
        assert stage.kv_pool.available == 1

    asyncio.run(run())


def test_managed_dit_normalizes_platform_nccl_before_startup(tmp_path, monkeypatch):
    import json
    import os
    import sys

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        dit_worker,
        encoder_worker,
    )

    path = tmp_path / "worker.json"
    path.write_text(
        json.dumps(
            dict(
                role="dit",
                service_count=1,
                hardware_ids=["gpu-test"],
                profile_limits={"480": {}},
            )
        )
    )
    monkeypatch.setenv("SGLANG_ROLE_POOL_WORKER_CONFIG", str(path))
    monkeypatch.setenv("NCCL_MIN_NCHANNELS", "4")
    monkeypatch.setenv("NCCL_MAX_NCHANNELS", "24")
    monkeypatch.setenv("NCCL_MIN_CTAS", "4")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(
        sys, "argv", ["dit_worker", "--model", str(tmp_path), "--sp-degree", "2"]
    )

    class RuntimeBoundary(Exception):
        pass

    def initialize():
        assert os.environ["NCCL_MIN_NCHANNELS"] == "2"
        assert "NCCL_MAX_NCHANNELS" not in os.environ
        assert "NCCL_MIN_CTAS" not in os.environ
        raise RuntimeBoundary

    monkeypatch.setattr(encoder_worker, "initialize_runtime_imports", initialize)
    with pytest.raises(RuntimeBoundary):
        dit_worker.main()


@pytest.mark.parametrize("static", [False, True])
def test_decoder_startup_normalizes_nccl_for_each_launch_mode(
    static, tmp_path, monkeypatch
):
    import os
    import sys

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        decoder_worker,
    )
    from sglang.multimodal_gen.runtime.realtime.role_pool import reporting

    monkeypatch.setenv("NCCL_MIN_NCHANNELS", "4")
    monkeypatch.setenv("NCCL_MAX_NCHANNELS", "24")
    monkeypatch.setenv("NCCL_MIN_CTAS", "4")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "decoder_worker",
            "--model",
            str(tmp_path),
            "--decoder-config",
            "{}",
            *(["--static-connection"] if static else []),
        ],
    )

    class ConfigBoundary(Exception):
        pass

    def validate(role, count, *, static_connection=False):
        assert role == "decoder" and count == 1 and static_connection == static
        assert os.environ["NCCL_MIN_NCHANNELS"] == "2"
        assert os.environ.get("NCCL_MAX_NCHANNELS") == None
        assert os.environ.get("NCCL_MIN_CTAS") == None
        raise ConfigBoundary

    monkeypatch.setattr(reporting, "validate_worker_configuration", validate)
    with pytest.raises(ConfigBoundary):
        decoder_worker.main()


def _startup_rank(rank, rendezvous, output, fail):
    from datetime import timedelta

    import torch.distributed as dist

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_worker import (
        _initialize_service_descriptor,
    )

    class Parallel:
        def broadcast_object(self, value):
            values = [value]
            dist.broadcast_object_list(values, src=0)
            return values[0]

    class Ports:
        services = [{"worker_id": "startup-fixture"}]

        def initialize_stores(self, *, timeout):
            assert timeout == 3
            if fail:
                raise OSError("Service store initialization failed")

    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        try:
            descriptor = _initialize_service_descriptor(
                Ports() if rank == 0 else None, Parallel(), timeout=3
            )
            result = {"descriptor": descriptor}
        except Exception as exc:
            result = {"error_type": type(exc).__name__, "message": str(exc)}
        Path(output, str(rank) + ".json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("fail", [False, True])
def test_both_ranks_finish_service_startup_or_fail_together(tmp_path, fail):
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_startup_rank,
            args=(rank, str(tmp_path / "rendezvous"), str(tmp_path), fail),
        )
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)
            assert not process.is_alive(), "A rank remained blocked during startup"
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    results = [json.loads((tmp_path / f"{rank}.json").read_text()) for rank in range(2)]
    if fail:
        assert results[0]["error_type"] == "OSError"
        assert results[0]["message"] == "Service store initialization failed"
        assert results[1]["error_type"] == "RuntimeError"
        assert (
            results[1]["message"] == "DiT service initialization failed on the leader"
        )
    else:
        assert results == [{"descriptor": {"worker_id": "startup-fixture"}}] * 2


def _prebind_startup_rank(rank, rendezvous, output, occupied, http_port, failure):
    """Run main's real pre-bind and Gloo path; replace only GPU import boundaries."""
    import os
    import sys
    import types
    from datetime import timedelta

    import torch.distributed as dist

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        dit_worker,
        encoder_worker,
    )

    initialized = []
    loaded = []

    class Parallel:
        def broadcast_object(self, value):
            values = [value]
            dist.broadcast_object_list(values, src=0)
            return values[0]

    def initialize(**kwargs):
        dist.init_process_group(
            "gloo",
            init_method="file://" + rendezvous,
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=6),
        )
        initialized.append(True)

    def cleanup():
        if dist.is_initialized():
            dist.destroy_process_group()

    class Loader:
        def load(self, *args, **kwargs):
            loaded.append(True)
            raise AssertionError("Model loading must not run after pre-bind failure")

    def forbidden(*args, **kwargs):
        raise AssertionError("GPU-only setup must not run after pre-bind failure")

    modules = {
        "sglang.multimodal_gen.runtime.distributed.parallel_state": dict(
            cleanup_dist_env_and_memory=cleanup,
            get_world_group=lambda: Parallel(),
            maybe_init_distributed_environment_and_model_parallel=initialize,
        ),
        "sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader": dict(
            TransformerLoader=Loader
        ),
        "sglang.multimodal_gen.runtime.managers.forward_context": dict(
            set_forward_context=forbidden
        ),
        "sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.zing_h3_source_adapter": dict(
            _source_sampling_schedule=forbidden, _with_source_pipeline_policy=forbidden
        ),
        dit_worker.__package__ + ".stage": dict(ZingH3GenerationStage=forbidden),
    }
    for name, fields in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(fields)
        sys.modules[name] = module
    encoder_worker.initialize_runtime_imports = lambda: (
        SimpleNamespace(from_kwargs=lambda **kwargs: SimpleNamespace()),
        lambda args: None,
    )
    torch.cuda.set_device = lambda device: None
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
    sys.argv = [
        "dit_worker",
        "--model",
        output,
        "--sp-degree",
        "2",
        "--http-host",
        "127.0.0.1",
        "--http-port",
        str(occupied if failure == "http" else http_port),
        "--rendezvous-port",
        str(occupied),
    ]
    try:
        dit_worker.main()
    except Exception as exc:
        result = dict(
            error_type=type(exc).__name__,
            message=str(exc),
            errno=getattr(exc, "errno", None),
        )
    else:
        result = dict(error_type=None)
    result.update(
        initialized=bool(initialized),
        model_loaded=bool(loaded),
        group_released=not dist.is_initialized(),
    )
    Path(output, f"prebind-{rank}.json").write_text(json.dumps(result))


@pytest.mark.parametrize("failure", ["http", "rendezvous"])
def test_pre_distributed_bind_failure_is_published_to_both_ranks(tmp_path, failure):
    import errno

    with socket.socket() as occupied, socket.socket() as free:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        free.bind(("127.0.0.1", 0))
        http_port = free.getsockname()[1]
        free.close()
        context = multiprocessing.get_context("spawn")
        processes = [
            context.Process(
                target=_prebind_startup_rank,
                args=(
                    rank,
                    str(tmp_path / "group"),
                    str(tmp_path),
                    occupied.getsockname()[1],
                    http_port,
                    failure,
                ),
            )
            for rank in range(2)
        ]
        try:
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=20)
                assert not process.is_alive(), (
                    "A peer remained blocked after pre-bind failure"
                )
                assert process.exitcode == 0
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
    leader, peer = [
        json.loads((tmp_path / f"prebind-{rank}.json").read_text()) for rank in range(2)
    ]
    assert leader["errno"] == errno.EADDRINUSE
    assert peer["error_type"] == "RuntimeError"
    assert peer["message"] == "DiT service initialization failed on the leader"
    assert all(
        row["initialized"] and not row["model_loaded"] and row["group_released"]
        for row in [leader, peer]
    )


@torch.no_grad()
@pytest.mark.parametrize("cached", [False, True])
def test_initial_pool_rejection_allows_inflight_ack_cancel_and_reuse(
    model, monkeypatch, cached
):
    h = Harness(model)
    codecs = SimpleNamespace(
        target_fps=24,
        spatial_ratio=1,
        temporal_ratio=4,
        vae_frames_per_chunk=17,
        vae_latents_per_chunk=5,
        sample_streaming_visual_moments=lambda value, **kw: value,
    )
    stage = pooled_stage(h, codecs)
    channel = LocalChannel()
    created = []

    def create(init, output):
        worker = create_link_session(h.sampler(), codecs, init, output, stage=stage)
        created.append(worker)
        return worker

    channel.service = DiTInputService(
        SimpleNamespace(send=channel.replies.append), create
    )
    client = DiTLinkClient(channel)
    init = dict(session_id="bad", generation_id="g", seed=7, width=2, height=2, fps=24)
    client.open(init)
    condition = h.condition(10)
    rows = stage.kv_pool.plan.slot_rows + 1
    oversized = dict(
        context=condition["context"][:1].expand(rows, -1).clone(),
        row_role=torch.zeros(
            rows, dtype=torch.int64, device=condition["context"].device
        ),
    )
    if cached:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
            dit_preparation,
        )
        from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import ZingKVRequest

        config = stage.kv_pool.plan.config
        entry = SimpleNamespace(
            inputs=h.inputs,
            request=ZingKVRequest(
                2,
                2,
                (2, 2),
                rows,
                config.window_chunks,
                config.sink_chunks,
                config.rope_position_mode,
                config.rope_max_frame_gap,
                window_eviction_timing=config.window_eviction_timing,
            ),
        )
        finished = []
        stage.block0_cache = SimpleNamespace(
            lookup=lambda key: entry, finish_fill=lambda: finished.append(True)
        )
        create.cached_condition = dit_preparation.DiTPreparation(
            stage, lambda: None
        ).cached_condition

        def forbidden(*args, **kwargs):
            raise AssertionError("Rejected initial cache reached tensor restoration")

        monkeypatch.setattr(dit_preparation, "inputs_on", forbidden)
    rejected = client.send(
        "conditioning",
        event_id=10,
        **(
            {"cache_hit": True, "cache_key": "a" * 64}
            if cached
            else {"tensors": oversized, "inputs": input_metadata(h)}
        ),
    )
    assert len(rejected) == 1
    assert rejected[0]["type"] == "conditioning.rejected" and rejected[0]["initial"]
    assert rejected[0]["code"] == "CONDITION_INVALID" and not client.failed
    assert created[-1].closed and stage.kv_pool.available == 1
    if cached:
        assert finished
    assert (
        client.send(
            "source.moments",
            tensor=h.inputs.source_visual[:, :, :1],
            latent_start=0,
            latent_end=1,
            source_dependency_end=43,
            source_records=[[1, 0, 43]],
        )
        == []
    )
    client.release("bad")
    assert channel.service.owner is None

    client.open(dict(init, session_id="good", generation_id="next"))
    try:
        client.send(
            "conditioning", event_id=10, tensors=condition, inputs=input_metadata(h)
        )
        outputs = []
        for start, end in [(0, 1), (1, 3), (3, 5), (5, 8), (8, 10), (10, 13)]:
            outputs += client.send(
                "source.moments",
                tensor=h.inputs.source_visual[:, :, start:end],
                latent_start=start,
                latent_end=end,
                source_dependency_end=43,
                source_records=[[1, 0, 43]],
                final_source_frames=43,
            )
        outputs += client.send("source.end", latent_frames=13)
        assert any(row["type"] == "dit.latents" for row in outputs)
        assert any(row["type"] == "session.completed" for row in outputs)
        assert stage.kv_pool.available == 1 and not client.failed
    finally:
        client.release("good")
