# SPDX-License-Identifier: Apache-2.0
"""Worker wiring tests only: synthetic encoder/DiT/decoder, no GPU evidence."""

from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.media_geometry import (
    SourceGeometry,
)
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
        self.committed_pixel = 0

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
        self.committed_pixel = result.provenance.pixels.end
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
    state.source_geometry = SourceGeometry(1, 1, 1, 1, 1, 1)
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
            spatial_ratio=1,
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


@pytest.mark.parametrize("operation", ["off", "refresh"])
@pytest.mark.parametrize("capture_enabled", [False, True])
def test_sink_updates_require_a_replayable_capture_contract(
    monkeypatch, operation, capture_enabled
):
    state, stage = session(ChunkPlan([3, 2]))
    sampler = state.sampler
    updates, completed = [], []
    monkeypatch.setattr(
        sampler,
        "update_sink",
        lambda operation, event_id: updates.append((operation, event_id)),
        raising=False,
    )
    if capture_enabled:
        state.component_capture = NS(finish=lambda: completed.append(True))
    request = batch(state, 0, 0, 0)
    request.extra["zing_sink_update"] = dict(operation=operation, event_id=7)
    try:
        if capture_enabled:
            with pytest.raises(
                ValueError,
                match="Component capture does not support dynamic sink updates",
            ):
                state.run(stage, request)
            assert updates == [] and completed == []
            assert sampler.closed
        else:
            state.run(stage, request)
            assert updates == [(operation, 7)]
    finally:
        state.dispose()


@pytest.mark.parametrize("finish", ["dispose", "eos"])
def test_dispose_logs_paged_block0_capacity_and_releases_state(monkeypatch, finish):
    import json

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        source_session,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
        Block0PreparationCache,
    )

    state, stage = session(ChunkPlan([3, 2]))
    # Exercise the real paged-cache layout without allocating pinned CUDA pages.
    cache = Block0PreparationCache.__new__(Block0PreparationCache)
    cache.mode, cache.pending = "on", None
    cache.pages = torch.empty(2, 2, 3, 4, dtype=torch.bfloat16)
    cache.page_rows, cache.total_pages = 2048, 0
    cache.free_pages = []
    cache.entries = {}
    cache.evictions = cache.hits = cache.misses = 0
    state.block0_cache = cache
    records = []
    monkeypatch.setattr(
        source_session.logger,
        "info",
        lambda message, payload, **kwargs: records.append(
            (message, json.loads(payload))
        ),
    )

    if finish == "eos":
        state.run(stage, batch(state, 0, 3, 0))
        result = state.run(stage, batch(state, 3, 0, 1, final=True))
        assert result.usage["zing_source_complete"]
        assert sum(value.shape[1] for value in result.output) == 3
    else:
        state.dispose()

    receipt = next(
        value for message, value in records if message == "zing_block0_cache %s"
    )
    assert (
        receipt["cpu_capacity_bytes"]
        == cache.pages.numel() * cache.pages.element_size()
    )
    assert state.closed and state.sampler is None and state.block0_cache is None


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


@pytest.mark.parametrize("remote", [False, True])
def test_worker_resizes_only_local_pixels_and_preserves_target_noise(remote):
    state, stage = session(ChunkPlan([1, 2, 2]))
    state.source_geometry = SourceGeometry(64, 64, 32, 32, 64, 64)
    stage.codecs.spatial_ratio = 32
    state.ledger = SourceLedger(
        width=64,
        height=64,
        fps=24,
        limits=SourceLimits(17, 1 << 20, 68, 4 << 20, 136),
    )
    state.bridge.provenance_for = state.ledger.provenance_for
    state.bridge.target_shape = (1, 2, 2)
    state.decoder.ledger = state.ledger
    if remote:
        state.policy["source_encoder"]["remote"] = {}
        state.pixels.append_capacity_frames = 17
    shapes, noises = [], []
    append_pixels, append_latent = state.pixels.append, state.sampler.append

    def encode(pixels, **kwargs):
        shapes.append(tuple(pixels.shape))
        append_pixels(pixels, **kwargs)

    def sample(source, noise, **kwargs):
        assert source.shape == (1, 1, 1, 1, 1)
        noises.append(noise)
        append_latent(source, noise, **kwargs)

    state.pixels.append, state.sampler.append = encode, sample
    command = batch(state, 0, 1, 0)
    item = command.extra["zing_source_command"]["appends"][0]
    item["payload"]["data"] = bytes(64 * 64 * 3)
    item["message_bytes"] = 64 * 64 * 3 + 100
    state.run(stage, command)
    assert shapes == [(3, 1, 64, 64) if remote else (3, 1, 32, 32)]
    assert noises[0].shape == (1, 1, 1, 2, 2)
    result = state.run(stage, batch(state, 1, 0, 1, final=True))
    assert result.usage["zing_source_complete"]
    assert state.closed


def test_prefetch_close_failure_retains_pool_until_original_owner_drains():
    state, _ = session(ChunkPlan([1, 2, 2]))
    encoder, sampler, decoder = state.pixels, state.sampler, state.decoder
    releases = []

    def fail_close():
        raise RuntimeError("Conditioning prefetch did not drain")

    owner = NS(close=fail_close)
    state.conditioning_prefetch = owner
    state.kv_lease = NS(release=lambda: releases.append("lease"))
    with pytest.raises(RuntimeError, match="unconfirmed: conditioning"):
        state.dispose()
    assert state.closed and encoder.disposed and sampler.closed and decoder.closed
    assert state.pixels is state.bridge is state.ledger is state.inputs is None
    assert state.conditioning_prefetch is owner
    assert state.kv_lease is not None and releases == []
    owner.close = lambda: None
    state.dispose()
    assert state.kv_lease is None and releases == ["lease"]


@pytest.mark.parametrize("prefetch_fails", [False, True])
def test_block0_fill_failure_attempts_other_cleanup_but_retains_pool(
    prefetch_fails,
):
    state, _ = session(ChunkPlan([1, 2, 2]))
    encoder, sampler, decoder = state.pixels, state.sampler, state.decoder
    events = []

    def fail_fill():
        events.append("fill")
        raise RuntimeError("Block0 fill failed")

    def close_prefetch():
        events.append("prefetch")
        if prefetch_fails:
            raise RuntimeError("Conditioning prefetch did not drain")

    owner = NS(close=close_prefetch)
    state.block0_cache = NS(finish_fill=fail_fill)
    state.conditioning_prefetch = owner
    state.kv_lease = NS(release=lambda: events.append("lease"))
    error = (
        "unconfirmed: block0, conditioning" if prefetch_fails else "unconfirmed: block0"
    )
    with pytest.raises(RuntimeError, match=error):
        state.dispose()
    assert events == ["fill", "prefetch"]
    assert state.closed and encoder.disposed and sampler.closed and decoder.closed
    assert state.pixels is state.bridge is state.ledger is state.inputs is None
    assert state.conditioning_prefetch is (owner if prefetch_fails else None)
    assert state.block0_cache is not None and state.kv_lease is not None


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


def test_continuation_with_new_input_consumes_one_plus_eight_in_one_dispatch(
    monkeypatch,
):
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_SOURCE_WORK_QUANTUM", "1")
    state, stage = session(ChunkPlan([3, 2]))
    first = batch(state, 0, 1, 0).extra["zing_source_command"]["appends"][0]
    state.ledger.accept(**first)
    state.pixels.append(torch.zeros(3, 1, 1, 1, dtype=torch.uint8), frame_start=0)
    assert state.pixels.ready and state.bridge.accepted_latents == 0
    # The old API forced an empty append here, producing a pure progress turn.
    result = state.run(stage, batch(state, 1, 8, 1))
    assert state.pixels.emitted_latents == 3
    assert result.usage["zing_source_consumed_end"] == 9
    assert result.usage["zing_source_dispatch_steps"] == 1
    assert len(state.sampler.calls) == 1


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
            # This queue-only sampler has no conditioning snapshot. The capture
            # double still accepts the optional snapshot used by the real sampler.
            self.prompt = None

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
            append=lambda chunk, result, *, prompt=None: captured.append(
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


def test_control_only_switch_rejects_a_peer_boundary_before_conditioning(monkeypatch):
    from sglang.multimodal_gen.runtime.conditioning import worker

    state, stage = session(ChunkPlan([3, 2], True))
    state.run(stage, batch(state, 0, 17, 0))
    control = batch(state, 17, 0, 1)
    control.extra["zing_prompt_update"] = dict(
        source_boundary=17, prompt="new", event_id=2
    )

    def forbidden(*_):
        raise AssertionError("A disagreeing boundary must not prepare a new condition")

    monkeypatch.setattr(state, "_update_prompt", forbidden)
    monkeypatch.setattr(worker, "coordinate_validation", lambda value: [value, False])
    sampler = state.sampler
    with pytest.raises(ValueError, match="barrier"):
        state.run(stage, control)
    assert state.closed and state.sampler is None and sampler.closed


@pytest.mark.parametrize("peer_matches", [True, False])
def test_control_only_dispatch_returns_ready_without_future_source_and_checks_identity(
    monkeypatch, peer_matches
):
    from sglang.multimodal_gen.runtime.conditioning import worker

    state, stage = session(ChunkPlan([3, 2], True))
    state.run(stage, batch(state, 0, 17, 0))
    control = batch(state, 17, 0, 1)
    control.extra["zing_prompt_update"] = dict(
        source_boundary=17, prompt="new", event_id=2
    )
    monkeypatch.setattr(state, "_update_prompt", lambda *_: None)
    state.sampler.device = torch.device("cpu")
    state.sampler.activate_prompt_at_source_boundary = lambda end: dict(
        event_id=2, prompt_version=2, source_frame_index=end
    )

    def agree(value):
        return [
            value,
            (
                value
                if isinstance(value, bool) or peer_matches
                else ("other-session", 2, 2, 17)
            ),
        ]

    monkeypatch.setattr(worker, "coordinate_validation", agree)
    if not peer_matches:
        with pytest.raises(ValueError, match="confirmed"):
            state.run(stage, control)
        assert state.closed
        return
    result = state.run(stage, control)
    assert result.raw_frame_batches == []
    assert result.usage["zing_condition_ready"] == dict(
        type="control_ack",
        kind="prompt.update",
        state="kv_ready",
        event_id=2,
        prompt_version=2,
        source_frame_index=17,
    )
    assert (
        result.usage["zing_source_committed_end"] == 17
        and state.ledger.accepted_frames == 17
    )
    state.dispose()


class IncrementalSyntheticDecoder:
    """Dispatch/lifecycle fixture; actual Lynn equality is tested separately."""

    def __init__(self, ledger):
        self.ledger = ledger
        self.accepted_latents = self.seen = self.end = self.valid = 0
        self.ended = self.closed = self.drained = False
        self.finishes = 0

    @property
    def buffered_latents(self):
        return self.accepted_latents - self.seen

    @property
    def ready(self):
        return bool(self.buffered_latents or (self.ended and not self.drained))

    def append(self, latent, *, latent_start, valid_frame_end, **_):
        assert latent_start == self.accepted_latents
        self.accepted_latents += latent.shape[2]
        self.valid = valid_frame_end

    def finish(self, *, total_frames):
        assert not self.ended and total_frames == self.valid
        self.ended = True
        self.finishes += 1

    def pop(self, *, max_latents):
        assert max_latents == 1
        self.seen += min(1, self.buffered_latents)
        clips, phase = divmod(self.seen, 5)
        end = min(self.valid, clips * 17 + (1 + (phase - 1) * 4 if phase else 0))
        start, self.end = self.end, end
        self.drained = self.ended and not self.buffered_latents
        return NS(
            start=start,
            end=end,
            values=torch.zeros(1, 3, end - start, 1, 1),
            provenance=(
                self.ledger.provenance_for(PixelRange(start, end), self.valid)
                if end > start
                else None
            ),
        )

    def dispose(self):
        self.closed = True


@pytest.mark.parametrize("frames", [1, 14, 17, 32, 34])
def test_decode_quantum_returns_each_slice_before_more_compute_and_drains_eos(frames):
    state, stage = session(ChunkPlan([3, 2], True))
    state.work_quantum = state.decode_quantum = 1
    decoder = state.decoder = IncrementalSyntheticDecoder(state.ledger)
    encoder, sampler = state.pixels, state.sampler
    outputs = []
    first = True
    for dispatch in range(40):
        pending_before = decoder.buffered_latents
        calls_before = len(sampler.calls)
        command = batch(
            state,
            0 if first else frames,
            min(frames, 17) if first else 0,
            dispatch,
            final=first and frames <= 17,
        )
        if first and frames > 17:
            tail = batch(state, 17, frames - 17, 1, final=True)
            command.extra["zing_source_command"]["appends"].extend(
                tail.extra["zing_source_command"]["appends"]
            )
            command.extra["zing_source_command"]["end"] = tail.extra[
                "zing_source_command"
            ]["end"]
        result = state.run(stage, command)
        first = False
        blocks = result.usage["zing_realtime_blocks"]
        outputs.extend(blocks)
        assert sum(x["num_frames"] for x in blocks) <= 4
        assert result.usage["zing_source_decode_steps"] <= 1
        if pending_before:
            assert len(sampler.calls) == calls_before
        if result.usage["zing_source_complete"]:
            break
        assert result.usage["zing_source_more_work"]
    else:
        pytest.fail("Decoder did not drain in bounded dispatches")
    assert sum(x["num_frames"] for x in outputs) == frames
    assert decoder.finishes == 1
    assert state.closed and decoder.closed and encoder.disposed and sampler.closed
    if frames == 17:
        assert [x["num_frames"] for x in outputs] == [1, 4, 4, 4, 4]


def test_decode_quantum_cancel_disposes_pending_latents():
    state, stage = session(ChunkPlan([3, 2], True))
    state.work_quantum = state.decode_quantum = 1
    decoder = state.decoder = IncrementalSyntheticDecoder(state.ledger)
    state.run(stage, batch(state, 0, 17, 0))
    state.run(stage, batch(state, 17, 0, 1))
    assert decoder.buffered_latents > 0
    encoder, sampler = state.pixels, state.sampler
    state.dispose()
    assert state.closed and decoder.closed and encoder.disposed and sampler.closed


@pytest.mark.parametrize("drain", [False, True])
def test_decode_quantum_condition_switch_waits_for_previous_rgb(monkeypatch, drain):
    from sglang.multimodal_gen.runtime.conditioning import worker

    state, stage = session(ChunkPlan([3, 2], True))
    state.work_quantum = state.decode_quantum = 1
    state.decoder = IncrementalSyntheticDecoder(state.ledger)
    result = state.run(stage, batch(state, 0, 17, 0))
    state.run(stage, batch(state, 17, 0, 1))
    if drain:
        for index in range(2, 12):
            result = state.run(stage, batch(state, 17, 0, index))
            if not result.usage["zing_source_more_work"]:
                break
        else:
            pytest.fail("Previous RGB did not drain")
    control = batch(state, 17, 0, 20)
    control.extra["zing_prompt_update"] = dict(
        source_boundary=17, prompt="new", event_id=2
    )
    monkeypatch.setattr(worker, "coordinate_validation", lambda value: [value, value])
    updates = []
    monkeypatch.setattr(state, "_update_prompt", lambda *_: updates.append("updated"))
    state.sampler.device = torch.device("cpu")
    state.sampler.activate_prompt_at_source_boundary = lambda end: dict(
        event_id=2, prompt_version=2, source_frame_index=end
    )
    if drain:
        result = state.run(stage, control)
        assert result.usage["zing_condition_ready"]["source_frame_index"] == 17
        assert updates == ["updated"]
        state.dispose()
    else:
        with pytest.raises(ValueError, match="barrier"):
            state.run(stage, control)
        assert updates == [] and state.closed


@pytest.mark.parametrize(
    "work,decode,valid",
    [
        ("1", "1", True),
        ("0", "0", True),
        ("1", "0", True),
        ("0", "1", False),
        ("1", "2", False),
    ],
)
def test_decode_quantum_requires_trusted_bounded_dispatch(
    monkeypatch, work, decode, valid
):
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_SOURCE_WORK_QUANTUM", work)
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_DECODE_WORK_QUANTUM", decode)
    # Worker policy is now resolved from the first command, so API-provided
    # settings can override this process's legacy environment fallback.
    state, stage = session(ChunkPlan([1, 2, 2]))
    request = batch(state, 0, 17, 0)
    if valid:
        state.run(stage, request)
        assert state.decode_quantum == int(decode)
    else:
        with pytest.raises(ValueError, match="decode quantum"):
            state.run(stage, request)
    state.dispose()


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
        # The real remote stream also needs a final receipt after its last RGB.
        self.ready = True

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
            provenance=(
                self.ledger.provenance_for(PixelRange(start, end), end)
                if end > start
                else None
            ),
        )


@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("first_dispatch", [False, True])
@pytest.mark.parametrize("local_decode_quantum", [0, 1])
def test_remote_decoder_returns_one_reference_per_dispatch(
    final, first_dispatch, local_decode_quantum, monkeypatch
):
    """The first and later dispatches bind identity after decoder initialization."""
    state, stage = session(ChunkPlan(1))
    state.decode_quantum = local_decode_quantum
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


def test_progressive_output_drains_accepted_encoder_prefix_without_empty_dispatch(
    monkeypatch,
):
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_SOURCE_WORK_QUANTUM", "0")
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_DECODE_WORK_QUANTUM", "0")
    state, stage = session(ChunkPlan([3, 2], split_first_latent=True))
    state.remote_progressive = True
    state.run(stage, batch(state, 0, 1, 0))
    result = state.run(stage, batch(state, 1, 17, 1))
    assert result.usage["zing_source_dispatch_steps"] == 1
    # Publish the current result first. Collect the next clip's prefix after
    # the already-ready second AR block, without introducing an empty RPC.
    assert state.pixels.ready
    assert result.usage["zing_source_consumed_end"] == 17
    assert result.usage["zing_source_more_work"]
    result = state.run(stage, batch(state, 18, 0, 2))
    assert result.usage["zing_source_dispatch_steps"] == 1
    assert not result.usage["zing_source_more_work"]
    assert not state.pixels.ready
    assert result.usage["zing_source_consumed_end"] == 18
    state.dispose()


@pytest.mark.parametrize(
    "protocol", ["whitebox-startup-prepare-v1", "whitebox-startup-prepare-v2"]
)
def test_startup_control_preserves_rng_and_source_positions_until_attach(
    monkeypatch, protocol
):
    import random
    from dataclasses import asdict

    state = ZingSourceStreamState()
    prepared, stage = session(ChunkPlan([1, 2, 2]))
    initializations = []

    def initialize(stage, batch, config, trace=None):
        assert batch.extra["zing_source_command"]["appends"] == []
        initializations.append(config)
        state.__dict__.update(prepared.__dict__)

    monkeypatch.setattr(state, "initialize", initialize)
    identity = dict(
        preparation_id="p",
        source_session_id="s",
        source_epoch="se",
        worker_epoch="we",
        condition_fingerprint="a" * 64,
    )
    config = dict(width=1, height=1, fps=24, limits=asdict(prepared.ledger.limits))

    def control(op, **fields):
        return NS(
            realtime_session_id="synthetic",
            block_idx=0,
            extra={
                "zing_source_command": dict(
                    config=config,
                    appends=[],
                    end=None,
                    startup=dict(op=op, identity=identity, protocol=protocol, **fields),
                )
            },
            metrics=None,
        )

    preview = (
        dict(rgb=bytes(3))
        if protocol.endswith("v1")
        else dict(rgb=bytes([17]) * 6, width=2, height=1)
    )
    cpu_rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    result = state.run(stage, control("prepare", **preview))
    assert result.usage["zing_startup_ready"]["level"] == "condition"
    assert (
        state.command_index
        == state.ledger.accepted_frames
        == state.bridge.accepted_pixels
        == state.sampler.index
        == 0
    )
    assert state.pixels.accepted_frames == 0 and state.decoder.appends == []
    assert (
        torch.equal(torch.get_rng_state(), cpu_rng) and random.getstate() == python_rng
    )
    state.run(stage, control("prepare", **preview))
    assert len(initializations) == 1
    state.run(stage, control("attach"))
    assert state.command_index == 0
    sampler = state.sampler
    first = state.run(stage, batch(state, 0, 1, 0, final=True))
    assert first.usage["zing_source_complete"] and state.command_index == 1
    assert len(sampler.calls) == 1


def test_prepare_s0_has_identical_condition_media_to_original_first_append():
    seen = []

    def prepare(request, use, **kwargs):
        seen.append(kwargs["source_preview"])
        raise RuntimeError("condition boundary")

    stage = NS(
        _validate_sampling_params=lambda p: None,
        _build_realtime_sampler=lambda p: NS(
            window_policy=NS(window_chunks=21, sink_chunks=3),
            rope_position_mode="block_relative",
            rope_max_frame_gap=12,
        ),
        _build_request=lambda b: {"messages": []},
        _conditioning_consumer=lambda b: None,
        _use=lambda *a: nullcontext(),
        codecs=NS(video_decoder_name="lynn_decoder", prepare=prepare),
    )
    rgb = bytes([17] * 3072)
    for command in [
        dict(source_preview=True, appends=[dict(payload=dict(data=rgb))]),
        dict(
            source_preview=True,
            appends=[],
            startup=dict(op="prepare", rgb=rgb, width=32, height=32),
        ),
    ]:
        with pytest.raises(RuntimeError, match="condition boundary"):
            ZingSourceStreamState().initialize(
                stage,
                NS(sampling_params=NS(), extra={"zing_source_command": command}),
                dict(width=32, height=32, fps=24),
            )
    assert torch.equal(seen[0].pop("frames"), seen[1].pop("frames"))
    assert seen[0] == seen[1] and seen[0]["origin"] == "source_s0"


def test_v2_prepare_condition_tensor_keeps_native_dimensions_without_source_advance():
    from sglang.multimodal_gen.runtime.realtime.source_prepare import V2_PROTOCOL

    seen = []

    def prepare(request, use, **kwargs):
        seen.append(kwargs["source_preview"])
        raise RuntimeError("condition boundary")

    stage = NS(
        _validate_sampling_params=lambda p: None,
        _build_realtime_sampler=lambda p: NS(
            window_policy=NS(window_chunks=21, sink_chunks=3),
            rope_position_mode="block_relative",
            rope_max_frame_gap=12,
        ),
        _build_request=lambda b: {"messages": []},
        _conditioning_consumer=lambda b: None,
        _use=lambda *a: nullcontext(),
        codecs=NS(video_decoder_name="lynn_decoder", prepare=prepare),
    )
    rgb = bytes([17]) * 16 * 48 * 3
    command = dict(
        source_preview=True,
        appends=[],
        startup=dict(op="prepare", protocol=V2_PROTOCOL, rgb=rgb, width=16, height=48),
    )
    with pytest.raises(RuntimeError, match="condition boundary"):
        ZingSourceStreamState().initialize(
            stage,
            NS(sampling_params=NS(), extra={"zing_source_command": command}),
            dict(width=32, height=32, fps=24),
        )
    assert tuple(seen[0]["frames"].shape) == (3, 1, 48, 16)
    assert torch.equal(
        seen[0]["frames"], torch.full((3, 1, 48, 16), 17, dtype=torch.uint8)
    )
    assert seen[0]["id"] == "<Video 1>" and seen[0]["clip"] == dict(start=0, end=0)


@pytest.mark.parametrize("change", ["protocol", "dimensions", "pixels", "oversize"])
def test_v2_worker_rejects_changed_preparation_before_source_advance(
    monkeypatch, change
):
    from dataclasses import asdict

    from sglang.multimodal_gen.runtime.realtime.source_prepare import V2_PROTOCOL

    state = ZingSourceStreamState()
    prepared, stage = session(ChunkPlan([1, 2, 2]))
    initialized = []

    def initialize(stage, batch, config, trace=None):
        initialized.append(config)
        state.__dict__.update(prepared.__dict__)

    monkeypatch.setattr(state, "initialize", initialize)
    identity = dict(
        preparation_id="p",
        source_session_id="s",
        source_epoch="se",
        worker_epoch="we",
        condition_fingerprint="a" * 64,
    )
    startup = dict(
        op="prepare",
        protocol=V2_PROTOCOL,
        identity=identity,
        rgb=bytes(6),
        width=2,
        height=1,
    )
    command = dict(
        config=dict(width=1, height=1, fps=24, limits=asdict(prepared.ledger.limits)),
        appends=[],
        end=None,
        startup=startup,
    )
    request = NS(
        realtime_session_id="synthetic",
        block_idx=0,
        extra={"zing_source_command": command},
        metrics=None,
    )
    if change != "oversize":
        state.run(stage, request)
    if change == "protocol":
        startup["protocol"] = "whitebox-startup-prepare-v1"
    elif change == "dimensions":
        startup.update(width=1, height=2)
    elif change == "pixels":
        startup["rgb"] = bytes([1]) * 6
    else:
        startup.update(width=16_777_217, height=1)
    with pytest.raises(ValueError):
        state.run(stage, request)
    assert state.command_index == 0 and state.closed
    assert len(initialized) == (0 if change == "oversize" else 1)


class ProgressiveJointSampler(SyntheticSampler):
    """Bounded synthetic ready queue; exercises ordering, not model performance."""

    def __init__(self, plan, events):
        super().__init__(plan)
        self.sampler.joint_forward = "cross_chunk"
        self.sampler.last_run_metrics = {"joint_fallbacks": {}}
        self.pending = []
        self.max_pending_chunks = 2
        self.max_pending_bytes = 1024
        self.pending_bytes = 0
        self.events = events

    def append(self, source, noise, *, provenance, final):
        self.pending.append(
            NS(
                source=source,
                noise=noise,
                provenance=provenance,
                final=final,
                chunk=self.plan.chunk(self.index + len(self.pending)),
            )
        )
        self.pending_bytes += source.numel() * 8

    def drain_ready(self, *, max_outputs=None):
        emitted = 0
        while self.pending and (max_outputs is None or emitted < max_outputs):
            value = self.pending.pop(0)
            self.pending_bytes -= value.source.numel() * 8
            super().append(
                value.source,
                value.noise,
                provenance=value.provenance,
                final=value.final,
            )
            self.events.append(("dit", value.chunk.end))
            emitted += 1
            yield self.step()


def progressive_remote_session(monkeypatch):
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_SOURCE_WORK_QUANTUM", "0")
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_DECODE_WORK_QUANTUM", "0")
    state, stage = session(ChunkPlan([3, 2], split_first_latent=True))
    events = []
    state.sampler = ProgressiveJointSampler(state.bridge.plan, events)
    state.decoder = SyntheticRemoteDecoder(state.ledger)
    state.remote_decoder = state.remote_progressive = True
    encoder_pop, decoder_pop = state.pixels.pop, state.decoder.pop

    def pop_source():
        result = encoder_pop()
        events.append(("encoder", result.end))
        return result

    def pop_output():
        events.append(("decode", state.decoder.accepted_frames))
        return decoder_pop()

    monkeypatch.setattr(state.pixels, "pop", pop_source)
    monkeypatch.setattr(state.decoder, "pop", pop_output)
    return state, stage, events


def run_remote(state, stage, start, count, *, final=False):
    request = batch(state, start, count, state.ledger.sequence, final=final)
    request.realtime_generation_id = "generation"
    return state.run(stage, request)


@pytest.mark.parametrize("prepared_start", [False, True])
def test_progressive_joint_starts_current_dit_and_decoder_before_future_encoder(
    monkeypatch,
    prepared_start,
):
    state, stage, events = progressive_remote_session(monkeypatch)
    if prepared_start:
        from dataclasses import asdict

        prepared = state
        state = ZingSourceStreamState()
        monkeypatch.setattr(
            state, "initialize", lambda *args: state.__dict__.update(prepared.__dict__)
        )
        identity = dict(
            preparation_id="p",
            source_session_id="s",
            source_epoch="se",
            worker_epoch="we",
            condition_fingerprint="a" * 64,
        )
        control = batch(prepared, 0, 0, 0)
        control.realtime_generation_id = "generation"
        command = control.extra["zing_source_command"]
        command["appends"] = []
        command["config"] = dict(
            width=1, height=1, fps=24, limits=asdict(prepared.ledger.limits)
        )
        command["startup"] = dict(
            op="prepare",
            protocol="whitebox-startup-prepare-v2",
            identity=identity,
            rgb=bytes([17]) * 6,
            width=2,
            height=1,
        )
        state.run(stage, control)
        assert state.ledger.accepted_frames == 0 and events == []
        command["startup"] = dict(
            op="attach", protocol="whitebox-startup-prepare-v2", identity=identity
        )
        state.run(stage, control)
    run_remote(state, stage, 0, 1)
    run_remote(state, stage, 1, 8)
    events.clear()
    # The current eight-frame block needs latent 3:5. Latent 5:6 is the
    # following clip's prefix and must not delay this block's DiT or D1 submit.
    result = run_remote(state, stage, 9, 9)
    assert events.index(("dit", 5)) < events.index(("encoder", 6))
    assert events.index(("decode", 17)) < events.index(("encoder", 6))
    assert result.raw_frame_shared_memory_ref == {"frames": 8}
    assert result.usage["zing_source_dispatch_steps"] == 1
    assert result.usage["zing_source_consumed_end"] == 18
    assert not result.usage["zing_source_more_work"]
    assert not state.pixels.ready
    if prepared_start:
        tail = run_remote(state, stage, 18, 0, final=True)
        assert tail.usage["zing_source_complete"]
        assert tail.raw_frame_shared_memory_ref == {"frames": 1}
        assert state.closed
    else:
        state.dispose()


@pytest.mark.parametrize("frames", [1, 8, 9, 17, 18, 26, 34, 35])
@pytest.mark.parametrize("packet_frames", [1, 17])
@pytest.mark.parametrize("end_with_last_packet", [False, True])
def test_progressive_joint_preserves_all_output_boundaries_and_eos(
    monkeypatch, frames, packet_frames, end_with_last_packet
):
    state, stage, _ = progressive_remote_session(monkeypatch)
    encoder, sampler, decoder = state.pixels, state.sampler, state.decoder
    cursor = 0
    results = []
    while cursor < frames:
        count = min(packet_frames, frames - cursor)
        results.append(
            run_remote(
                state,
                stage,
                cursor,
                count,
                final=end_with_last_packet and cursor + count == frames,
            )
        )
        cursor += count
        while results[-1].usage["zing_source_more_work"]:
            results.append(run_remote(state, stage, cursor, 0))
    if not end_with_last_packet:
        results.append(run_remote(state, stage, cursor, 0, final=True))
    for _ in range(20):
        if results[-1].usage["zing_source_complete"]:
            break
        assert results[-1].usage["zing_source_more_work"]
        results.append(run_remote(state, stage, cursor, 0))
    assert results[-1].usage["zing_source_complete"]
    assert (
        sum((r.raw_frame_shared_memory_ref or {}).get("frames", 0) for r in results)
        == frames
    )
    assert all(r.usage["zing_source_dispatch_steps"] <= 1 for r in results)
    expected = [call.provenance.pixels.end for call in sampler.calls]
    assert [item["valid_frame_end"] for item in decoder.appends] == expected
    assert expected == sorted(set(expected)) and expected[-1] == frames
    assert state.closed and encoder.disposed and sampler.closed and decoder.closed


def test_progressive_joint_failure_after_decoder_submission_releases_state(monkeypatch):
    state, stage, events = progressive_remote_session(monkeypatch)
    encoder, sampler, decoder = state.pixels, state.sampler, state.decoder
    run_remote(state, stage, 0, 1)
    run_remote(state, stage, 1, 8)
    pop_source = encoder.pop

    def fail_future_prefix():
        if encoder.emitted_latents == 5:
            assert ("decode", 17) in events
            raise RuntimeError("Future source transfer failed")
        return pop_source()

    monkeypatch.setattr(encoder, "pop", fail_future_prefix)
    with pytest.raises(RuntimeError, match="Future source transfer failed"):
        run_remote(state, stage, 9, 9)
    assert state.closed and encoder.disposed and sampler.closed and decoder.closed


def test_prepare_native_preview_geometry_is_independent_of_source_stream():
    seen = []

    def prepare(request, use, **kwargs):
        seen.append(kwargs["source_preview"])
        raise RuntimeError("condition boundary")

    stage = NS(
        _validate_sampling_params=lambda p: None,
        _build_realtime_sampler=lambda p: NS(
            window_policy=NS(window_chunks=21, sink_chunks=3),
            rope_position_mode="block_relative",
            rope_max_frame_gap=12,
        ),
        _build_request=lambda b: {"messages": []},
        _conditioning_consumer=lambda b: None,
        _use=lambda *a: nullcontext(),
        codecs=NS(video_decoder_name="lynn_decoder", prepare=prepare),
    )
    width, height = 960, 540
    rgb = bytes([17, 23, 42]) * width * height
    command = dict(
        source_preview=True,
        appends=[],
        startup=dict(
            op="prepare",
            protocol="whitebox-startup-prepare-v2",
            rgb=rgb,
            width=width,
            height=height,
        ),
    )
    with pytest.raises(RuntimeError, match="condition boundary"):
        ZingSourceStreamState().initialize(
            stage,
            NS(sampling_params=NS(), extra={"zing_source_command": command}),
            dict(width=832, height=480, fps=24),
        )
    preview = seen[0]
    assert preview["origin"] == "source_s0"
    assert tuple(preview["frames"].shape) == (3, 1, 540, 960)
    assert preview["frames"][:, 0, -1, -1].tolist() == [17, 23, 42]


@pytest.mark.parametrize("quantum", [0, 1])
def test_worker_command_quantum_overrides_worker_environment(monkeypatch, quantum):
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_SOURCE_WORK_QUANTUM", "invalid")
    state, stage = session(ChunkPlan([1, 2, 2]))
    request = batch(state, 0, 17, 0)
    request.extra["zing_source_command"]["work_quantum"] = quantum
    result = state.run(stage, request)
    assert state.work_quantum == quantum
    if quantum == 1:
        assert result.usage["zing_source_dispatch_steps"] == 1
        assert result.usage["zing_source_more_work"]
    else:
        assert result.usage["zing_source_dispatch_steps"] > 1
    state.dispose()


@pytest.mark.parametrize("defer_final,prompt_joint", [(False, False), (True, True)])
def test_worker_initialization_uses_command_policy(
    monkeypatch, defer_final, prompt_joint
):
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_SOURCE_WORK_QUANTUM", "invalid")
    monkeypatch.setenv("SGLANG_ZING_SOURCE_DEFER_FINAL", "0" if defer_final else "1")
    monkeypatch.setenv("SGLANG_ZING_SOURCE_PROMPT_JOINT", "0" if prompt_joint else "1")

    class MediaBoundary(Exception):
        pass

    def prepare(*args, **kwargs):
        raise MediaBoundary()

    stage = NS(
        _validate_sampling_params=lambda params: None,
        _build_realtime_sampler=lambda params: NS(
            window_policy=NS(window_chunks=21, sink_chunks=3),
            rope_position_mode="block_relative",
            rope_max_frame_gap=12,
        ),
        _build_request=lambda batch: {},
        _conditioning_consumer=lambda batch: None,
        codecs=NS(video_decoder_name="test", prepare=prepare),
    )
    request = NS(
        sampling_params=NS(),
        extra={
            "zing_source_command": {
                "work_quantum": 1,
                "defer_final": defer_final,
                "prompt_joint": prompt_joint,
            }
        },
    )
    state = ZingSourceStreamState()
    with pytest.raises(MediaBoundary):
        state.initialize(stage, request, {})
    assert state.policy["source_work_quantum"] == 1
    assert state.policy["source_defer_final"] == defer_final
    assert state.policy["source_prompt_joint"] is prompt_joint


@pytest.mark.parametrize("enabled", [False, True])
def test_block0_prefill_and_live_share_key_without_scheduling_policy(
    monkeypatch, enabled
):
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_SOURCE_WORK_QUANTUM", "invalid")
    for name in ("SGLANG_ZING_SOURCE_DEFER_FINAL", "SGLANG_ZING_SOURCE_PROMPT_JOINT"):
        monkeypatch.setenv(name, "0" if enabled else "1")
    keys, disk_reads = [], []

    class LookupBoundary(Exception):
        pass

    def lookup(key, *, allow_disk=False):
        disk_reads.append(allow_disk)
        keys.append(key)
        raise LookupBoundary()

    stage = NS(
        block0_cache=NS(
            lookup=lookup, identity={"model": {"dit": "fixture"}, "layout": {"sp": 1}}
        ),
        _validate_sampling_params=lambda params: None,
        _build_realtime_sampler=lambda params: NS(
            joint_forward="cross_chunk" if enabled else "off",
            window_policy=NS(window_chunks=6, sink_chunks=3),
            rope_position_mode="block_relative",
            rope_max_frame_gap=12,
        ),
        _build_request=lambda batch: {},
        codecs=NS(video_decoder_name="test"),
    )
    config = dict(width=832, height=480, fps=24)
    command = dict(work_quantum=1, defer_final=enabled, prompt_joint=enabled)
    request = NS(sampling_params=NS(), extra={"zing_source_command": command})
    live, warmup = ZingSourceStreamState(), ZingSourceStreamState()
    with pytest.raises(LookupBoundary):
        live.initialize(stage, request, config)
    with pytest.raises(LookupBoundary):
        warmup.prefill({}, stage=stage, batch=request, config=config)
    assert keys[0] == keys[1]
    assert live.policy == warmup.policy
    assert warmup.policy["source_defer_final"] is enabled
    assert warmup.policy["source_prompt_joint"] is enabled
    command["prompt_joint"] = not enabled
    with pytest.raises(LookupBoundary):
        ZingSourceStreamState().prefill({}, stage=stage, batch=request, config=config)
    assert keys[2] == keys[0]  # Scheduling flags do not change cached KV identity.
    command["defer_final"] = not enabled
    with pytest.raises(LookupBoundary):
        ZingSourceStreamState().prefill({}, stage=stage, batch=request, config=config)
    assert keys[3] == keys[0]
    assert disk_reads == [False, True, True, True]


@pytest.mark.parametrize("progressive", [False, True])
def test_s0_v2_and_prefill_share_native_preview_and_resolved_policy(
    monkeypatch, progressive
):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_session import (
        source_preview,
    )

    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_SOURCE_WORK_QUANTUM", "invalid")
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_DECODE_WORK_QUANTUM", "invalid")
    keys = []

    class LookupBoundary(Exception):
        pass

    def lookup(key, *, allow_disk=False):
        keys.append(key)
        raise LookupBoundary()

    stage = NS(
        block0_cache=NS(lookup=lookup, identity={"model": {}, "layout": {}}),
        _validate_sampling_params=lambda params: None,
        _build_realtime_sampler=lambda params: NS(
            joint_forward="cross_chunk",
            window_policy=NS(window_chunks=6, sink_chunks=3),
            rope_position_mode="block_relative",
            rope_max_frame_gap=12,
        ),
        _build_request=lambda batch: {},
        codecs=NS(video_decoder_name="lynn_decoder"),
    )
    config = dict(width=32, height=32, fps=30)
    rgb = bytes([17, 23, 42]) * 48 * 24
    command = dict(
        work_quantum=0 if progressive else 1,
        decode_quantum=0,
        defer_final=False,
        prompt_joint=False,
        remote_progressive=progressive,
        source_preview=True,
        appends=[],
        startup=dict(
            op="prepare",
            protocol="whitebox-startup-prepare-v2",
            rgb=rgb,
            width=48,
            height=24,
        ),
    )
    batch = NS(sampling_params=NS(), extra={"zing_source_command": command})
    live, warmup = ZingSourceStreamState(), ZingSourceStreamState()
    with pytest.raises(LookupBoundary):
        live.initialize(stage, batch, config)
    with pytest.raises(LookupBoundary):
        warmup.prefill(
            {},
            stage=stage,
            batch=batch,
            config=config,
            source_preview=source_preview({**config, "width": 48, "height": 24}, rgb),
        )
    assert keys[0] == keys[1]
    assert live.policy == warmup.policy
    assert live.policy["source_defer_final"] is progressive


def test_same_prompt_omitted_references_skips_conditioning_and_cpu_cache(monkeypatch):
    from sglang.multimodal_gen.runtime.conditioning import worker

    monkeypatch.setattr(worker, "coordinate_validation", lambda v: [v])
    state = ZingSourceStreamState()
    condition = {
        "context": torch.ones(3, 4),
        "row_role": torch.zeros(3, dtype=torch.long),
    }
    live = NS(condition=condition, event_id=4)
    updates = []
    state.prompt, state.prompt_event = "same", 4
    state.sampler = NS(
        prompt=live,
        applied_prompt=live,
        cache=NS(prefix=object()),
        discard_pending_prompt=lambda: None,
        update_prompt=lambda *a, **kw: updates.append((a, kw)),
    )

    def forbidden(*_, **__):
        raise AssertionError("Must not load or encode identical conditions")

    state.block0_cache = NS(lookup=forbidden)
    stage = NS(encode_condition_for_batch=forbidden)
    batch = NS(
        extra={"zing_prompt_update": dict(prompt="same", event_id=5)},
        sampling_params=NS(prompt="same"),
        realtime_event_id=5,
        realtime_prompt_version=2,
    )
    assert state._update_prompt(stage, batch) is None
    assert updates[0][0][0] is condition and updates[0][1]["reuse_prefix"] is True
    assert state.prompt_event == 5


@pytest.mark.parametrize("changed", ["text", "references", "pending", "rank"])
def test_resident_reuse_never_hides_real_condition_change(monkeypatch, changed):
    from sglang.multimodal_gen.runtime.conditioning import worker

    def vote(v):
        return [v, (False, v[1])] if changed == "rank" and isinstance(v, tuple) else [v]

    monkeypatch.setattr(worker, "coordinate_validation", vote)
    state = ZingSourceStreamState()
    live = NS(condition={}, event_id=3 if changed == "pending" else 4)
    state.prompt, state.prompt_event = "same", 4
    state.sampler = NS(
        prompt=live,
        applied_prompt=live,
        cache=NS(prefix=object()),
        discard_pending_prompt=lambda: None,
    )

    class NormalPath(Exception):
        pass

    def normal(*a, **kw):
        raise NormalPath()

    state.block0_cache = NS(lookup=normal)
    state._block0_update_key = lambda *a: ("key", {})
    cmd = dict(prompt="different" if changed == "text" else "same", event_id=5)
    if changed == "references":
        cmd["references"] = []
    batch = NS(
        extra={"zing_prompt_update": cmd},
        sampling_params=NS(prompt="same"),
        realtime_event_id=5,
    )
    with pytest.raises(NormalPath):
        state._update_prompt(NS(), batch)
