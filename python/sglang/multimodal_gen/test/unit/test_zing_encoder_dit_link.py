# SPDX-License-Identifier: Apache-2.0
"""Loopback transport and queued DiT lifecycle; real-checkpoint tests are separate."""

import asyncio
import multiprocessing
import socket
import time
from contextlib import nullcontext
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
    create_link_session,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.packing import (
    H3InferenceInputs,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (  # noqa: F401
    cpu_execution_options,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_queued_dit import (  # noqa: F401
    Harness,
    model,
)


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
    )

    sampler = h.sampler()
    config = ZingKVPoolConfig(
        pool_size=1,
        max_target_area=4,
        max_target_long_edge=2,
        max_prefix_rows=64,
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
    stage.packer = h.packer
    stage.kv_pool = ZingKVPool.allocate(
        ZingKVPoolPlan.build(config, geometry),
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
def test_pool_rejection_is_nonfatal_and_discards_pending_prompt(model):
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
        rejected = client.send("conditioning", event_id=12, tensors=oversized)
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
    cpu_dma,
)


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
