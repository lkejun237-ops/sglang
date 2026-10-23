# SPDX-License-Identifier: Apache-2.0
"""Worker wiring tests only: synthetic encoder/DiT/decoder, no GPU evidence."""

from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_bridge import (
    SourceChunkBridge,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_session import (
    ZingSourceStreamState,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.streaming_codecs import (
    SourcePixelBuffer,
)
from sglang.multimodal_gen.runtime.realtime.zing_chunk_plan import ChunkPlan
from sglang.multimodal_gen.runtime.realtime.zing_contracts import PixelRange
from sglang.multimodal_gen.runtime.realtime.zing_source_stream import (
    SourceLedger,
    SourceLimits,
)


class SyntheticEncoder:
    """Exercises service ownership, not any actual encoder computation."""

    def __init__(self):
        self.accepted_frames = self.emitted_latents = 0
        self.buffered_frames = 0
        self.ended = self.disposed = False

    def append(self, pixels, *, frame_start):
        assert frame_start == self.accepted_frames
        self.accepted_frames += pixels.shape[1]

    def next_slice(self):
        clip, offset = divmod(self.emitted_latents, 5)
        if clip * 17 >= self.accepted_frames:
            return None
        stop = 1 if offset == 0 else 3 if offset == 1 else 5
        dependency = clip * 17 + 1 + (stop - 1) * 4
        if dependency > self.accepted_frames:
            if not self.ended:
                return None
            count = self.accepted_frames - clip * 17
            stop = min(stop, 1 + (count - 1 + 3) // 4)
            dependency = self.accepted_frames
        if stop <= offset:
            return None
        return NS(
            values=torch.ones(1, 1, stop - offset, 1, 1),
            start=self.emitted_latents,
            end=clip * 5 + stop,
            source_dependency_end=dependency,
        )

    @property
    def ready(self):
        return not self.disposed and self.next_slice() is not None

    def pop(self):
        value = self.next_slice()
        self.emitted_latents = value.end
        return value

    def finish(self):
        self.ended = True

    def dispose(self):
        self.disposed = True

    def state_usage(self):
        return NS(retained_bytes=0, temporal_positions=0)


class SyntheticSampler:
    def __init__(self, plan):
        self.plan, self.index, self.calls = plan, 0, []
        self.sampler = NS(generator=None)
        self.ended = self.complete = self.closed = False

    def append(self, source, noise, *, provenance, final):
        self.value, self.provenance = source, provenance
        self.ended = final

    def step(self):
        chunk = self.plan.chunk(self.index)
        result = NS(
            chunk=NS(
                chunk_id=self.index,
                start=chunk.start,
                end=chunk.start + self.value.shape[2],
            ),
            visual=self.value,
            provenance=self.provenance,
            prompt_event_id=None,
            prompt_version=None,
        )
        self.calls.append(result)
        self.index += 1
        self.complete = self.ended
        return result

    def memory_snapshot(self):
        return {}

    def finish(self):
        self.ended = self.complete = True

    def dispose(self):
        self.closed = True


class SyntheticDecoder:
    def __init__(self, ledger):
        self.ledger, self.buffered_latents = ledger, 0
        self.appends = []
        self.ready = self.closed = False

    def append(self, latent, **kwargs):
        self.appends.append(kwargs)

    def finish(self, *, total_frames):
        self.total = total_frames
        self.ready = True

    def pop(self):
        self.ready = False
        return NS(
            start=0,
            end=self.total,
            values=torch.zeros(1, 3, self.total, 1, 1),
            provenance=self.ledger.provenance_for(
                PixelRange(0, self.total), self.total
            ),
        )

    def dispose(self):
        self.closed = True


def session(plan, mode="progressive"):
    state = ZingSourceStreamState()
    state.ledger = SourceLedger(
        width=1, height=1, fps=24, limits=SourceLimits(17, 4096, 68, 4096, 136)
    )
    state.pixels = SyntheticEncoder()
    state.sampler = SyntheticSampler(plan)
    state.decoder = SyntheticDecoder(state.ledger)
    state.policy = {"source_encoder": {"mode": mode}}
    state.prompt = "test"
    state.bridge = SourceChunkBridge(
        plan,
        clip_frames=17,
        clip_latents=5,
        temporal_ratio=4,
        capacity_latents=10,
        capacity_bytes=1024,
        seed=42,
        provenance_for=state.ledger.provenance_for,
    )
    stage = NS(
        codecs=NS(
            vae_frames_per_chunk=17,
            vae_latents_per_chunk=5,
            temporal_ratio=4,
            encode_visual_moments=lambda pixels: torch.ones(
                1, 1, 1 if pixels.shape[1] == 1 else 5, 1, 1
            ),
            video_decoder_name="synthetic",
            sample_streaming_visual_moments=lambda value, **kwargs: value,
        ),
        _use=lambda *args: nullcontext(None),
    )
    if mode == "clip17":
        state.pixels = SourcePixelBuffer(stage.codecs, capacity_frames=51)
    return state, stage


def batch(state, start, count, sequence, *, final=False):
    payload = dict(
        client_seq=sequence,
        source_frame_start=start,
        num_frames=count,
        pts_us=start * 1_000_000 // 24,
        data=bytes(count * 3),
    )
    appends = (
        [dict(payload=payload, event_id=sequence + 1, message_bytes=count * 3 + 100)]
        if count
        else []
    )
    end = dict(frame_end=start + count, event_id=sequence + 2) if final else None
    return NS(
        block_idx=state.command_index,
        extra={"zing_source_command": dict(appends=appends, end=end)},
        sampling_params=NS(prompt="test"),
        realtime_event_id=None,
        metrics=None,
        realtime_session_id="synthetic",
        request_id="synthetic",
    )


@pytest.mark.parametrize("mode", ["clip17", "progressive"])
@pytest.mark.parametrize("frames", range(1, 35))
@pytest.mark.parametrize("partition", [[1], [2, 3, 1], [17]])
@pytest.mark.parametrize(
    "plan", [ChunkPlan(5), ChunkPlan([1, 2, 2]), ChunkPlan(5, True)]
)
def test_worker_drains_partial_groups_and_releases_ownership(
    frames, partition, plan, mode
):
    state, stage = session(plan, mode)
    encoder, sampler, decoder = state.pixels, state.sampler, state.decoder
    cursor = index = 0
    while cursor < frames:
        count = min(partition[index % len(partition)], frames - cursor)
        result = state.run(stage, batch(state, cursor, count, index))
        cursor += count
        index += 1
        assert result.usage["zing_source_consumed_end"] == state.bridge.accepted_pixels
        assert state.ledger.consumed_frames <= state.bridge.accepted_pixels <= cursor
        assert not result.usage["zing_source_complete"]
        if cursor == 1 and mode == "progressive":
            assert encoder.emitted_latents == 1
            assert len(sampler.calls) == int(
                plan.lengths[0] == 1 or plan.split_first_latent
            )
    result = state.run(stage, batch(state, cursor, 0, index, final=True))
    assert result.usage["zing_source_complete"]
    assert sum(x["num_frames"] for x in result.usage["zing_realtime_blocks"]) == frames
    assert sampler.calls[-1].provenance.pixels.end == frames
    assert [item.get("valid_frame_end") for item in decoder.appends] == [
        item.provenance.pixels.end for item in sampler.calls
    ]
    assert state.closed and encoder.disposed and sampler.closed and decoder.closed
    assert state.pixels is state.bridge is state.ledger is None


def test_worker_rejects_invalid_encoder_output_and_releases_all_state():
    state, stage = session(ChunkPlan([1, 2, 2]))
    encoder, sampler, decoder = state.pixels, state.sampler, state.decoder
    encoder.pop = lambda: NS(
        values=torch.ones(1, 1, 1, 1, 1), start=0, end=2, source_dependency_end=1
    )
    with pytest.raises(ValueError, match="invalid stable slice"):
        state.run(stage, batch(state, 0, 1, 0))
    assert state.closed and encoder.disposed and sampler.closed and decoder.closed
    assert not sampler.calls


def test_continuous_source_path_emits_dit_and_decoder_stage_spans(monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        source_session,
    )
    from sglang.multimodal_gen.runtime.utils.zing_stage_timing import StageTrace

    rows = []
    monkeypatch.setattr(
        source_session,
        "trace_for_batch",
        lambda request: StageTrace(
            request.realtime_session_id,
            request.request_id,
            lambda item: rows.append(item[0]),
        ),
    )
    state, stage = session(ChunkPlan(1))
    result = state.run(stage, batch(state, 0, 1, 0, final=True))
    assert result.usage["zing_source_complete"]
    assert [row["stage"] for row in rows] == ["DiT", "Decoder", "Request"]
    assert rows[0]["output_latent_range"] == [0, 1]
    assert rows[1]["output_frame_range"] == [0, 1]
    assert rows[2]["component"] == "continuous_source_complete"
    assert all(row["request_id"] == "synthetic" for row in rows)


def test_explicit_factory_selection_and_version_guard(monkeypatch):
    import sys
    from types import ModuleType

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.progressive_encoder import (
        PROGRESSIVE_ENCODER_VERSION,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_encoder import (
        create_source_buffer,
    )

    codecs = NS(vae_frames_per_chunk=17, vae_latents_per_chunk=5, temporal_ratio=4)
    created = []
    module = ModuleType("_synthetic_progressive_test")

    def factory(codecs, *, capacity_frames):
        assert capacity_frames == 51
        session = SyntheticEncoder()
        session.contract_version = PROGRESSIVE_ENCODER_VERSION
        session.minimum_credit_frames = 8
        created.append(session)
        return session

    module.create_session = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    profile = dict(
        mode="progressive",
        contract_version=PROGRESSIVE_ENCODER_VERSION,
        factory=module.__name__ + ":create_session",
    )
    first = create_source_buffer(codecs, profile, capacity_frames=51)
    second = create_source_buffer(codecs, profile, capacity_frames=51)
    assert first is not second
    first.append(torch.zeros(3, 1, 1, 1), frame_start=0)
    assert second.accepted_frames == 0
    with pytest.raises(ValueError, match="profile"):
        create_source_buffer(
            None, {**profile, "contract_version": "invalid"}, capacity_frames=51
        )
    assert len(created) == 2
    with pytest.raises(ModuleNotFoundError):
        create_source_buffer(
            codecs,
            {**profile, "factory": "_missing_real_encoder:create_session"},
            capacity_frames=51,
        )

    module.create_session = lambda *args, **kwargs: first
    with pytest.raises(ValueError, match="empty session"):
        create_source_buffer(codecs, profile, capacity_frames=51)
    assert first.disposed


def test_lookahead_does_not_grant_pixel_alignment_or_input_credit():
    state, stage = session(ChunkPlan([1, 2, 2]))
    original = state.pixels.pop

    def with_lookahead():
        item = original()
        item.source_dependency_end = 3
        return item

    state.pixels.pop = with_lookahead
    result = state.run(stage, batch(state, 0, 3, 0))
    assert result.usage["zing_source_consumed_end"] == 1
    # The three-frame append is indivisible for credit. Dependency 3 must not
    # grant credit for pixels 1 and 2 whose aligned latents are still missing.
    assert state.ledger.consumed_frames == 0
    assert state.sampler.calls[0].provenance.pixels == PixelRange(0, 1)
    assert state.sampler.calls[0].provenance.source_dependency_end == 3
    # Pixel validity is 1 even though the encoder depends on source pixel 3.
    assert state.decoder.appends[0].get("valid_frame_end") == 1
    assert state.decoder.appends[0]["source_dependency_end"] == 3
    state.dispose()


def test_progressive_append_and_finish_hold_declared_vae_component():
    from contextlib import contextmanager

    state, stage = session(ChunkPlan([1, 2, 2]))
    active, observed = [], []

    @contextmanager
    def use(name, *args):
        active.append(name)
        try:
            yield None
        finally:
            active.pop()

    stage._use = use
    append, finish = state.pixels.append, state.pixels.finish

    def append_checked(*args, **kwargs):
        assert active[-1] == "video_vae"
        observed.append("append")
        return append(*args, **kwargs)

    def finish_checked():
        assert active[-1] == "video_vae"
        observed.append("finish")
        return finish()

    state.pixels.append, state.pixels.finish = append_checked, finish_checked
    result = state.run(stage, batch(state, 0, 1, 0, final=True))
    assert result.usage["zing_source_complete"]
    assert observed == ["append", "finish"] and not active


@pytest.mark.parametrize("explicit_video", [False, True])
def test_source_s0_uses_memory_unless_video_is_explicit(explicit_video):
    request = {
        "messages": [
            {
                "role": "user",
                "content": (
                    [{"type": "video", "id": "<Video 1>"}] if explicit_video else []
                ),
            }
        ]
    }
    seen = {}

    def prepare(actual_request, use, **kwargs):
        seen.update(request=actual_request, **kwargs)
        raise RuntimeError("stop after prompt preparation")

    stage = NS(
        _validate_sampling_params=lambda params: None,
        _build_realtime_sampler=lambda params: NS(
            window_policy=NS(window_chunks=21, sink_chunks=3),
            rope_position_mode="block_relative",
            rope_max_frame_gap=12,
        ),
        _build_request=lambda batch: request,
        _conditioning_consumer=lambda batch: None,
        _use=lambda *args: nullcontext(),
        codecs=NS(video_decoder_name="lynn_decoder", prepare=prepare),
    )
    batch = NS(
        sampling_params=NS(),
        extra={
            "zing_source_command": {
                "source_preview": not explicit_video,
                "appends": [
                    {
                        "payload": {
                            "source_frame_start": 0,
                            "data": bytes([17] * (32 * 32 * 3)),
                        }
                    }
                ],
            }
        },
    )
    with pytest.raises(RuntimeError, match="stop after prompt preparation"):
        ZingSourceStreamState().initialize(
            stage, batch, {"width": 32, "height": 32, "fps": 24}
        )
    assert seen["request"] is request
    assert "prompt_source_frame" not in seen
    media = seen["source_preview"]
    if explicit_video:
        assert media is None
    else:
        assert media["type"] == "video" and media["id"] == "<Video 1>"
        assert media["clip"] == {"start": 0, "end": 0} and media["fps"] == 24
        assert "uri" not in media
        assert media["frames"].shape == (3, 1, 32, 32)
        assert media["frames"].dtype == torch.uint8
        assert torch.all(media["frames"] == 17)
        assert batch.extra["zing_source_command"]["appends"][0]["payload"][
            "data"
        ] == bytes([17] * 3072)


@pytest.mark.parametrize("rank", [0, 1])
def test_dispose_receipt_identifies_worker_and_is_emitted_once(monkeypatch, rank):
    import json
    import os

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        source_session,
    )

    state, _ = session(ChunkPlan([1, 2, 2]))
    state.session_id = "cancelled-session"
    records = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(
        source_session.logger,
        "info",
        lambda message, payload, **kwargs: records.append(json.loads(payload)),
    )
    state.dispose()
    state.dispose()
    assert len(records) == 1
    receipt = records[0]
    assert (receipt["session_id"], receipt["rank"], receipt["pid"]) == (
        "cancelled-session",
        rank,
        os.getpid(),
    )
    assert receipt["memory"]["encoder_state"]["retained_bytes"] == 0
    assert receipt["memory"]["codec_pixel_frames"] == 0
    assert state.closed and state.sampler is None


@pytest.mark.parametrize("frames", range(1, 35))
def test_quantum_one_drains_eos_by_continuation_without_new_input(monkeypatch, frames):
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_SOURCE_WORK_QUANTUM", "1")
    state, stage = session(ChunkPlan([1, 2, 2]))
    sampler = state.sampler
    # The existing synthetic Decoder intentionally holds all RGB until EOS:
    # zero-output continuations must still advance rather than waiting for input.
    cursor = sequence = 0
    results = []
    while cursor < frames:
        count = min(17, frames - cursor)
        result = state.run(stage, batch(state, cursor, count, sequence))
        cursor += count
        sequence += 1
        results.append(result)
        while result.usage["zing_source_more_work"]:
            before = len(sampler.calls)
            result = state.run(stage, batch(state, cursor, 0, sequence))
            results.append(result)
            assert len(sampler.calls) - before <= 1
    result = state.run(stage, batch(state, cursor, 0, sequence, final=True))
    results.append(result)
    for _ in range(20):
        if result.usage["zing_source_complete"]:
            break
        assert result.usage["zing_source_more_work"]
        result = state.run(stage, batch(state, cursor, 0, sequence))
        results.append(result)
    assert result.usage["zing_source_complete"] and state.closed
    assert all(r.usage["zing_source_dispatch_steps"] <= 1 for r in results)
    assert (
        sum(b["num_frames"] for r in results for b in r.usage["zing_realtime_blocks"])
        == frames
    )


def test_quantum_yields_ready_rgb_before_next_dit_step(monkeypatch):
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_SOURCE_WORK_QUANTUM", "1")
    state, stage = session(ChunkPlan([1, 2, 2]))

    class StepDecoder:
        buffered_latents = 0
        ready = False
        emitted = latents = 0
        closed = False

        def append(self, latent, **kwargs):
            self.latents += latent.shape[2]
            self.end = 1 + (self.latents - 1) * 4
            self.dependency = kwargs["source_dependency_end"]
            self.ready = True

        def pop(self):
            self.ready = False
            start = self.emitted
            self.emitted = self.end
            return NS(
                start=start,
                end=self.end,
                values=torch.zeros(1, 3, self.end - start, 1, 1),
                provenance=state.ledger.provenance_for(
                    PixelRange(start, self.end), self.dependency
                ),
            )

        def finish(self, **kwargs):
            pass

        def dispose(self):
            self.closed = True

    state.decoder = StepDecoder()
    sampler = state.sampler
    result = state.run(stage, batch(state, 0, 17, 0))
    assert (
        len(sampler.calls) == 1
        and result.usage["zing_realtime_blocks"][0]["num_frames"] == 1
    )
    assert result.usage["zing_source_more_work"]
    stale = batch(state, 17, 0, 1)
    result = state.run(stage, stale)
    assert (
        len(sampler.calls) == 2
        and result.usage["zing_realtime_blocks"][0]["num_frames"] == 8
    )
    with pytest.raises(ValueError, match="consecutive"):
        state.run(stage, stale)
    state.dispose()
    with pytest.raises(ValueError, match="consecutive"):
        state.run(stage, batch(state, 17, 0, 1))


@pytest.mark.parametrize("quantum", [0, 1])
@pytest.mark.parametrize("queue_bytes", [1024, 16])
@pytest.mark.parametrize("capture", [False, True])
def test_joint_ready_worker_batches_without_crossing_command_boundary(
    monkeypatch, queue_bytes, capture, quantum
):
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_SOURCE_WORK_QUANTUM", str(quantum))
    from collections import deque

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        source_session,
    )
    from sglang.multimodal_gen.runtime.utils.zing_stage_timing import StageTrace

    timing_rows = []
    monkeypatch.setattr(
        source_session,
        "trace_for_batch",
        lambda request: StageTrace(
            request.realtime_session_id,
            request.request_id,
            lambda item: timing_rows.append(item[0]),
        ),
    )

    class JointSampler(SyntheticSampler):
        def __init__(self, plan):
            super().__init__(plan)
            self.sampler.joint_forward = "cross_chunk"
            self.sampler.last_run_metrics = {"joint_fallbacks": {}}
            self.pending = deque()
            self.max_pending_chunks = 2
            self.max_pending_bytes = queue_bytes
            self.pending_bytes = 0
            self.batches = []

        def append(self, source, noise, *, provenance, final):
            chunk = self.plan.chunk(self.index + len(self.pending))
            self.pending.append(
                NS(
                    source=source,
                    noise=noise,
                    provenance=provenance,
                    final=final,
                    chunk=chunk,
                )
            )
            self.pending_bytes += source.numel() * 8

        def drain_ready(self):
            self.batches.append(len(self.pending))
            while self.pending:
                value = self.pending.popleft()
                self.pending_bytes -= value.source.numel() * 8
                super().append(
                    value.source,
                    value.noise,
                    provenance=value.provenance,
                    final=value.final,
                )
                yield self.step()

    state, stage = session(ChunkPlan([1, 2, 2]))
    state.sampler = sampler = JointSampler(state.bridge.plan)
    decoder = state.decoder
    captured = []
    if capture:
        state.component_capture = NS(
            append=lambda chunk, result: captured.append(
                (chunk.chunk, result.chunk, chunk.final)
            ),
            append_source=lambda *args: None,
            append_frames=lambda *args: None,
            finish=lambda: None,
        )
    # Multiple codec slices are ready in this command; a small byte budget
    # must fall back before popping the next bridge chunk.
    result = state.run(stage, batch(state, 0, 17, 0, final=True))
    results = [result]
    for _ in range(20):
        if result.usage["zing_source_complete"]:
            break
        assert result.usage["zing_source_more_work"]
        result = state.run(stage, batch(state, 17, 0, 1))
        results.append(result)
    assert sum(x.shape[1] for r in results for x in (r.output or [])) == 17
    if quantum:
        assert all(r.usage["zing_source_dispatch_steps"] <= quantum for r in results)
    assert not sampler.pending
    assert len(decoder.appends) == len(sampler.calls)
    assert [item.get("valid_frame_end") for item in decoder.appends] == [
        item.provenance.pixels.end for item in sampler.calls
    ]
    assert state.closed
    assert any(row["stage"] == "DiT" for row in timing_rows)
    assert any(row["stage"] == "DiT" and row.get("no_output") for row in timing_rows)
    assert timing_rows[-1]["component"] == "continuous_source_complete"
    assert max(sampler.batches) == (2 if queue_bytes == 1024 and not quantum else 1)
    if capture:
        assert len(captured) == 3
        assert all(
            (a.chunk_id, a.start, a.end) == (b.chunk_id, b.start, b.end)
            for a, b, _ in captured
        )
        # Full chunks drain before bridge.finish(); EOS is handled separately.
        assert [final for _, _, final in captured] == [False, False, False]
    else:
        assert not captured
    if queue_bytes == 16 and not quantum:
        assert (
            sampler.sampler.last_run_metrics["joint_fallbacks"]["ready_queue_capacity"]
            > 0
        )


class SyntheticRemoteDecoder(SyntheticDecoder):
    """Publishes pixels elsewhere; the DiT only receives one reference."""

    def __init__(self, ledger):
        super().__init__(ledger)
        self.accepted_frames = self.emitted_frames = 0
        self.pops = 0
        self.dispatch = None

    def bind_dispatch(self, **identity):
        self.dispatch = identity

    def append(self, latent, **kwargs):
        super().append(latent, **kwargs)
        self.accepted_frames = kwargs["valid_frame_end"]
        self.ready = True

    def finish(self, *, total_frames):
        self.total = total_frames

    def pop(self):
        self.pops += 1
        self.ready = False
        start, end = self.emitted_frames, self.accepted_frames
        self.emitted_frames = end
        return NS(
            start=start,
            end=end,
            values=None,
            frames=end - start,
            reference={"frames": end - start} if end > start else None,
            transport_metadata={"format": "rgb24"},
            provenance=self.ledger.provenance_for(PixelRange(start, end), end),
        )


@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("first_dispatch", [False, True])
def test_remote_decoder_returns_one_reference_per_dispatch(
    final, first_dispatch, monkeypatch
):
    """The first and later dispatches bind identity after decoder initialization."""
    state, stage = session(ChunkPlan(1))
    ledger = state.ledger
    decoder = state.decoder = SyntheticRemoteDecoder(ledger)
    state.remote_decoder = True
    request = batch(state, 0, 17, 0, final=final)
    request.realtime_generation_id = "generation"
    if first_dispatch:
        state.ledger = state.decoder = None
        state.remote_decoder = False
        request.extra["zing_source_command"]["config"] = dict(
            limits=dict(
                batch_frames=17,
                message_bytes=4096,
                pending_frames=68,
                pending_bytes=4096,
                metadata_batches=136,
            )
        )

        def initialize(*args):
            state.ledger, state.decoder = ledger, decoder
            state.remote_decoder = True
            state.qwen_init_host_ms = None

        monkeypatch.setattr(state, "initialize", initialize)
    result = state.run(stage, request)
    assert result.usage["zing_source_complete"] is final

    assert len(decoder.appends) > 1
    assert decoder.pops == 1
    assert decoder.dispatch == dict(
        request_id="synthetic", chunk_index=0, generation_id="generation"
    )
    assert result.output is None
    assert result.raw_frame_shared_memory_ref == {"frames": decoder.emitted_frames}
    assert sum(x["num_frames"] for x in result.usage["zing_realtime_blocks"]) == (
        decoder.emitted_frames
    )
    if not final:
        state.dispose()
