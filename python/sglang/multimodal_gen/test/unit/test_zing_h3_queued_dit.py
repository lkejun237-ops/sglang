# SPDX-License-Identifier: Apache-2.0
"""Queue integration using the original tiny DiT, sampler and KV implementation."""

import asyncio
import gc
import weakref
from copy import copy

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.incremental import (
    SourceReadySession,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.queued_dit import (
    QueuedDiTSession,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
    ZingH3Sampler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_bridge import (
    SourceChunkBridge,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.streaming_noise import (
    indexed_video_noise,
)
from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy
from sglang.multimodal_gen.test.unit.test_zing_h3_chunks import (
    _long_request,
    fixture_provenance,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (  # noqa: F401
    cpu_execution_options,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import model as model_fixture


@pytest.fixture(params=["cpu", "cuda"])
def model(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for the GPU queue regression")
    return model_fixture.__wrapped__().to(request.param)


@torch.no_grad()
def test_reference_payload_is_atomic_bounded_and_snapshotted(model):
    h = Harness(model)
    condition = h.condition(10)
    reference = condition["context"].new_ones(1, 2, 1, 2, 2)
    condition_bytes = sum(t.numel() * t.element_size() for t in condition.values())
    reference_bytes = reference.numel() * reference.element_size()

    async def run():
        # Reference-only wire payloads are invalid: Qwen also depends on images.
        with pytest.raises(ValueError, match="context and row_role"):
            await h.put(
                "conditioning", event_id=10, inputs={"visual": {"r": reference}}
            )
        h.worker.max_condition_bytes = condition_bytes
        with pytest.raises(ValueError, match="exceeds queue capacity"):
            await h.put(
                "conditioning",
                event_id=10,
                tensors=condition,
                inputs={"visual": {"r": reference}},
            )
        assert not h.worker.initial_received and not h.worker.queue
        h.worker.max_condition_bytes += reference_bytes
        await h.put(
            "conditioning",
            event_id=10,
            tensors=condition,
            inputs={"visual": {"r": reference}},
        )
        assert len(h.worker.queue) == 1
        assert h.worker.condition_bytes == condition_bytes + reference_bytes
        reference.zero_()
        task = h.worker.queue.popleft()
        assert torch.all(task["inputs"]["visual"]["r"] == 1)
        h.worker.close()

    asyncio.run(run())


@torch.no_grad()
@pytest.mark.parametrize("model", ["cpu"], indirect=True)
def test_external_sink_control_and_prompt_share_the_original_session(model):
    h = Harness(model, frames=15, joint_forward="cross_chunk")
    h.packer.split_first_latent = False
    h.bridge.plan = h.packer.chunk_plan
    original = h.sampler

    def sampler():
        value = original()
        value.window_policy = ZingWindowPolicy(6, 2)
        return value

    h.sampler = sampler

    async def run():
        await h.prompt(10)
        await h.drain_ready()
        for start, end in [(0, 3), (3, 5)]:
            await h.source(start, end)
            await h.drain_ready()
        await h.prompt(12)
        await h.put("sink.update", operation="off", event_id=11)
        await h.drain_ready()
        await h.source(5, 8)
        await h.drain_ready()
        result = h.results[-1]
        assert (result.prompt_event_id, result.sink_event_id, result.sink_chunks) == (
            12,
            11,
            0,
        )
        await h.put("sink.update", operation="refresh", event_id=13)
        await h.drain_ready()
        await h.source(8, 10)
        await h.drain_ready()
        result = h.results[-1]
        assert (result.sink_event_id, result.sink_chunk_ids) == (13, (1, 2))
        assert h.worker.control_tasks == h.worker.condition_bytes == 0
        h.worker.close()
        assert h.worker.state.closed

    asyncio.run(run())


class Harness:
    def __init__(
        self,
        model,
        frames=13,
        *,
        joint_forward="off",
        defer_final=False,
        pooled_kv=False,
        **options,
    ):
        self.packer, self.inputs, _, _ = _long_request(model, [3, 2], frames)
        device = next(model.parameters()).device
        self.inputs.positive = {
            k: v.to(device) for k, v in self.inputs.positive.items()
        }
        self.inputs.source_visual = self.inputs.source_visual.to(device)
        self.model = model
        self.joint_forward = joint_forward
        self.defer_final = defer_final
        self.pooled_kv = pooled_kv
        self.results = []
        self.rebuilds = []
        self.bridge = SourceChunkBridge(
            self.packer.chunk_plan,
            clip_frames=17,
            clip_latents=5,
            temporal_ratio=4,
            capacity_latents=8,
            capacity_bytes=1 << 20,
            seed=7,
            provenance_for=lambda p, d: fixture_provenance(1, d, p.start, p.end),
        )

        def create(condition, event, version):
            inputs = copy(self.inputs)
            inputs.positive = condition
            sampler = self.sampler()
            original = sampler.rebuild_prompt_prefix

            def rebuild(inputs, condition, recent, cache):
                self.rebuilds.append((condition["context"].clone(), recent.clone()))
                return original(inputs, condition, recent, cache)

            sampler.rebuild_prompt_prefix = rebuild
            return self.session(
                sampler,
                inputs,
                event,
                version,
                defer_final=self.defer_final,
                pooled_kv=self.pooled_kv,
            )

        async def output(result):
            self.results.append(result)

        self.worker = QueuedDiTSession(
            session_id="s",
            generation_id="g",
            create_session=create,
            bridge=self.bridge,
            sample_moments=lambda *a, **kw: pytest.fail("Unexpected posterior call"),
            on_output=output,
            **options,
        )

    def sampler(self):
        return ZingH3Sampler(
            self.model,
            self.packer,
            solver="few_step",
            denoising_step_list=[1000, 500],
            window_chunks=21,
            sink_chunks=3,
            joint_forward=self.joint_forward,
        )

    @staticmethod
    def session(
        sampler, inputs, event=10, version=1, *, defer_final=False, pooled_kv=False
    ):
        acquire_storage = None
        if pooled_kv:
            from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
                allocate_layer_stores,
            )

            device = next(sampler.generator.parameters()).device
            arch = sampler.generator.arch

            def acquire_storage(positive, ready):
                return allocate_layer_stores(
                    2048,
                    num_layers=len(sampler.generator.blocks),
                    local_heads=arch.num_attention_heads,
                    head_dim=arch.attention_head_dim,
                    dtype=torch.bfloat16,
                    device=device,
                )

        return SourceReadySession(
            sampler,
            inputs,
            seed=7,
            max_pending_chunks=1,
            max_pending_bytes=1 << 20,
            initial_prompt_event_id=event,
            initial_prompt_version=version,
            defer_final=defer_final,
            acquire_storage=acquire_storage,
        )

    async def put(self, kind, **fields):
        await self.worker.put(
            dict(session_id="s", generation_id="g", type=kind, **fields)
        )

    def condition(self, event):
        return dict(
            context=torch.full_like(self.inputs.positive["context"], event / 10),
            row_role=self.inputs.positive["row_role"],
        )

    async def prompt(self, event):
        await self.put(
            "conditioning",
            tensors=self.condition(event),
            event_id=event,
            prompt_version=event - 9,
        )

    async def source(self, start, end):
        clip, offset = divmod(end - 1, 5)
        pixels = clip * 17 + 1 + offset * 4
        await self.put(
            "source.latents",
            tensor=self.inputs.source_visual[:, :, start:end],
            latent_start=start,
            latent_end=end,
            pixel_end=pixels,
            source_dependency_end=pixels,
        )

    async def drain_ready(self):
        while self.worker.queue:
            await self.worker.queue.run_once()


@torch.no_grad()
@pytest.mark.parametrize("joint_forward", ["off", "cross_chunk"])
def test_prompt_bypasses_incomplete_group_and_reuses_original_kv_rebuild(
    model, joint_forward
):
    h = Harness(model, joint_forward=joint_forward)

    async def run():
        await h.prompt(10)
        await h.source(0, 1)
        await h.drain_ready()
        assert len(h.results) == 1
        await h.source(1, 2)  # Next model chunk needs two: no runnable frame task.
        assert not h.worker.queue
        for event in (11, 12, 13):
            await h.prompt(event)
        assert [t["type"] for t in h.worker.queue] == ["conditioning"] * 3
        await h.drain_ready()
        assert h.worker.state.pending_prompt.event_id == 13
        assert not h.rebuilds  # Keep the live KV until a complete five-latent cycle.
        await h.source(2, 5)
        await h.drain_ready()
        await h.source(5, 8)
        await h.drain_ready()
        assert len(h.results) == 4
        await h.prompt(14)
        await h.drain_ready()
        await h.source(8, 10)
        await h.source(10, 13)
        await h.put("source.end", latent_frames=13)
        # No further ingress: polling must drain the backlog and then EOS.
        await asyncio.wait_for(h.worker.run(), 10)

    asyncio.run(run())
    assert h.worker.complete and h.worker.closed
    assert [r.prompt_event_id for r in h.results] == [10, 10, 10, 13, 13, 14]
    assert [r.prompt_version for r in h.results] == [1, 1, 1, 4, 4, 5]
    assert [(r.chunk.start, r.chunk.end) for r in h.results] == [
        (0, 1),
        (1, 3),
        (3, 5),
        (5, 8),
        (8, 10),
        (10, 13),
    ]
    assert len(h.rebuilds) == 2
    for index, (_, prefix) in zip((3, 5), h.rebuilds):
        torch.testing.assert_close(
            prefix, torch.cat([r.visual for r in h.results[:index]], 2)[:, :, -5:]
        )

    # Numerical regression against the original direct session, with identical
    # real model calls, noise and prompt boundaries (no mock cache or sampler).
    inputs = copy(h.inputs)
    inputs.positive = h.condition(10)
    direct = h.session(h.sampler(), inputs)
    try:
        for index, actual in enumerate(h.results):
            if index == 1:
                for event in (11, 12, 13):
                    direct.update_prompt(h.condition(event), event, event - 9)
            if index == 4:
                direct.update_prompt(h.condition(14), 14, 5)
            start, end = actual.chunk.start, actual.chunk.end
            source = inputs.source_visual[:, :, start:end]
            noise = indexed_video_noise(
                tuple(source.shape),
                seed=7,
                latent_start=start,
                purpose="target_initial",
                device=source.device,
                dtype=torch.float32,
            )
            direct.append(source, noise, provenance=actual.provenance, final=index == 5)
            expected = direct.step()
            torch.testing.assert_close(actual.visual, expected.visual, rtol=0, atol=0)
        assert direct.complete
    finally:
        direct.dispose()
    assert not h.bridge.buffered_latents and h.worker.state.inputs is None


@torch.no_grad()
@pytest.mark.parametrize("frames", [1, 2, 4, 5, 6, 9, 13])
def test_backlog_eos_and_short_tail_without_further_arrivals(model, frames):
    h = Harness(model, frames)

    async def run():
        await h.prompt(10)
        await h.drain_ready()
        for start in range(0, frames, 5):
            await h.source(start, min(start + 5, frames))
            await h.drain_ready()
        await h.put("source.end", latent_frames=frames)
        await asyncio.wait_for(h.worker.run(), 10)

    asyncio.run(run())
    assert h.worker.complete
    assert [(r.chunk.start, r.chunk.end) for r in h.results] == [
        (c.start, c.end) for c in h.packer.chunk_plan.chunks(frames)
    ]
    assert h.results[-1].chunk.end == frames


@torch.no_grad()
def test_identity_validation_cancel_and_reuse(model):
    h = Harness(model)

    async def run():
        with pytest.raises(ValueError, match="Initial conditioning"):
            await h.source(0, 1)
        with pytest.raises(ValueError, match="does not match"):
            await h.worker.put(
                dict(type="conditioning", session_id="s", generation_id="stale")
            )
        await h.prompt(10)
        await h.drain_ready()
        with pytest.raises(ValueError, match="increasing"):
            await h.prompt(10)
        with pytest.raises(ValueError, match="out-of-order"):
            await h.source(1, 2)
        await h.source(0, 1)
        await h.drain_ready()
        task = asyncio.create_task(h.worker.run())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert h.worker.closed and h.worker.state.inputs is None
        assert not h.bridge.buffered_latents
        with pytest.raises(ValueError, match="closed"):
            await h.prompt(11)
        # Same model is reusable after releasing the cancelled session's KV.
        again = Harness(model, 1)
        await again.prompt(10)
        await again.source(0, 1)
        await again.put("source.end", latent_frames=1)
        await asyncio.wait_for(again.worker.run(), 10)
        assert again.worker.complete

    asyncio.run(run())


@torch.no_grad()
def test_control_backpressure_and_output_failure_release_state(model):
    h = Harness(model, max_control_tasks=1)

    async def run():
        await h.prompt(10)
        waiting = asyncio.create_task(h.prompt(11))
        await asyncio.sleep(0.01)
        assert not waiting.done()
        await h.worker.queue.run_once()
        await asyncio.wait_for(waiting, 1)
        await h.drain_ready()
        assert h.worker.control_tasks == h.worker.condition_bytes == 0
        await h.source(0, 1)

        async def failed(_):
            raise RuntimeError("Output disconnected")

        h.worker.on_output = failed
        with pytest.raises(RuntimeError, match="Output disconnected"):
            await h.worker.run()
        assert h.worker.closed and not h.worker.queue
        assert h.worker.state.inputs is None and not h.bridge.buffered_latents

    asyncio.run(run())


@torch.no_grad()
@pytest.mark.parametrize("defer_final", [False, True])
def test_queued_session_uses_chunk_graphs_and_defer_final(model, defer_final):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
        ZingChunkGraphs,
    )

    model.chunk_graphs = ZingChunkGraphs(capture_enabled=False)
    h = Harness(
        model,
        frames=8,
        joint_forward="cross_chunk",
        defer_final=defer_final,
        pooled_kv=True,
    )

    async def run():
        await h.prompt(10)
        await h.drain_ready()
        for start, end in ((0, 5), (5, 8)):
            await h.source(start, end)
            await h.drain_ready()
        await h.put("source.end", latent_frames=8)
        await asyncio.wait_for(h.worker.run(), 10)

    asyncio.run(run())
    assert h.worker.complete
    routes = dict(model.chunk_graphs.routes)
    assert routes.get(("init", "eager"), 0) == 1
    if defer_final:
        assert routes.get(("deferred", "eager"), 0) == 4
        assert routes.get(("resume", "eager"), 0) == 3
        assert routes.get(("recovery", "eager"), 0) == 0
    else:
        assert routes.get(("finish", "eager"), 0) == 4
        assert routes.get(("recovery", "eager"), 0) == 3


def test_dit_worker_defer_final_policy(monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        dit_worker,
    )

    parser = dit_worker.argument_parser()
    assert parser.parse_args(["--model", "/tmp/m"]).zing_source_defer_final is None
    assert (
        parser.parse_args(
            ["--model", "/tmp/m", "--zing-source-defer-final"]
        ).zing_source_defer_final
        is True
    )
    assert (
        parser.parse_args(
            ["--model", "/tmp/m", "--no-zing-source-defer-final"]
        ).zing_source_defer_final
        is False
    )
    with pytest.raises(ValueError, match="Invalid DiT session geometry or seed"):
        dit_worker.create_link_session(
            None,
            None,
            {"seed": 1, "width": 832, "height": 480, "fps": 16},
            None,
            defer_final="invalid",
        )


@torch.no_grad()
@pytest.mark.parametrize("exit_kind", ["eos", "cancel", "output_failure"])
def test_session_exit_releases_shared_packer_prefix_memo(model, exit_kind):
    h = Harness(model, frames=5)

    async def run():
        await h.prompt(10)
        await h.drain_ready()
        await h.source(0, 1)
        await h.drain_ready()
        assert h.packer._prefix_memo is not None
        context = weakref.ref(h.packer._prefix_memo[1][0])

        if exit_kind == "eos":
            await h.source(1, 5)
            await h.put("source.end", latent_frames=5)
            await asyncio.wait_for(h.worker.run(), 10)
            assert h.worker.complete
        elif exit_kind == "cancel":
            task = asyncio.create_task(h.worker.run())
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:

            async def failed(_):
                assert not h.worker.state.ended
                assert h.packer._prefix_memo is not None
                raise RuntimeError("Output disconnected")

            h.worker.on_output = failed
            await h.source(1, 3)
            with pytest.raises(RuntimeError, match="Output disconnected"):
                await h.worker.run()
        assert h.worker.closed and h.worker.state.inputs is None
        assert h.packer._prefix_memo is None
        h.worker.close()  # Repeated cleanup must be harmless.
        return context

    context = asyncio.run(run())
    gc.collect()
    assert context() is None

    # The deployment retains this same packer for the next session.
    inputs = copy(h.inputs)
    inputs.positive = h.condition(20)
    session = h.session(h.sampler(), inputs, event=20)
    try:
        source = inputs.source_visual[:, :, :1]
        session.append(
            source,
            torch.zeros_like(source),
            provenance=fixture_provenance(1, 1, 0, 1),
            final=True,
        )
        assert session.step().prompt_event_id == 20
        assert session.complete and session.closed
        assert h.packer._prefix_memo is None
    finally:
        session.dispose()
