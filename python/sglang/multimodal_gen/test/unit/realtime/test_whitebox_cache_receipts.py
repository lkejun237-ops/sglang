import asyncio
from types import SimpleNamespace as NS

import msgspec
import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.cache_receipt import (
    public_cache_receipt,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import SourceSession


def test_cache_receipt_alias_version_and_private_fields():
    x = {
        "key": "a" * 64,
        "prompt_version": 2,
        "cache_result": "l2_hit",
        "prompt": "PRIVATE",
        "request": {"secret": "PRIVATE"},
        "stages": {"cache_lookup": {"host_ms": 12.5, "status": "executed"}},
    }
    result = public_cache_receipt(x)
    assert result["cache_key"] == "a" * 64 and result["prompt_version"] == 2
    assert "PRIVATE" not in str(result)
    assert public_cache_receipt(result) == result
    assert public_cache_receipt({**x, "prompt_version": None}) is None
    assert public_cache_receipt({**x, "stages": 42}) is not None


def test_prompt_activation_reports_its_version_and_preserves_prior_l2_lookup():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_session import (
        ZingSourceStreamState,
    )

    state = ZingSourceStreamState()
    state.pending_block0 = ("b" * 64, {}, NS(), None, None)
    state.pending_block0_prompt_version = 4
    state.pending_block0_lookup = {
        "cache_result": "l2_hit",
        "lookup_host_ms": 20,
        "miss_reason": None,
    }
    state.block0_cache = NS(
        finish_fill=lambda: None,
        lookup=lambda key: NS(),
        last_lookup={"cache_result": "l1_hit", "lookup_host_ms": 1},
    )
    state.sampler = NS(capture_initial_prefix=None)
    candidate = NS(reuse_prefix=False)
    restore = state._prepare_block0_switch(candidate)
    assert (
        callable(restore)
        and state.block0_prompt_version == 4
        and not state.block0_reported
    )
    assert state.block0_telemetry.cache_result == "l2_hit"
    state.pending_block0 = ("c" * 64, {}, NS(), None, None)
    state.pending_block0_prompt_version = 5
    state.pending_block0_lookup = {"cache_result": "l1_hit", "lookup_host_ms": 1}
    state.block0_cache.lookup = lambda key: None
    state.block0_cache.last_lookup = {
        "cache_result": "miss",
        "miss_reason": "not_found",
        "lookup_host_ms": 2,
    }
    assert state._prepare_block0_switch(candidate) is None
    assert (
        state.block0_prompt_version == 5
        and state.block0_telemetry.cache_result == "miss"
    )


def test_room_cache_receipts_reject_other_model_generations_and_remain_bounded():
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_segment import (
        RenderSegment,
    )

    async def run():
        source = object.__new__(SourceSession)
        source.outgoing = asyncio.Queue()
        source.source_identity = lambda: {
            "source_session_id": "source",
            "source_epoch": "epoch",
        }
        segment = object.__new__(RenderSegment)
        from sglang.multimodal_gen.runtime.realtime.whitebox.sink_control import (
            SinkControl,
        )

        segment.sink = SinkControl(segment)
        segment.owner = source
        segment.segment_id = "seg"
        segment.segment_epoch = "segepoch"
        segment.assignment = NS(session_id="model", generation_id="generation")
        segment.recording = NS(
            manifest={
                "scene_prefill": {
                    "transition_id": "cutscene",
                    "request_id": "previous-prefill",
                    "cache_key": "b" * 64,
                }
            }
        )
        source.current_segment = segment
        source.current_model_stream = "segepoch"
        observed = []

        async def observe(transition_id, cache):
            observed.append((transition_id, cache))

        source.scene_transition = NS(record_generation_cache=observe)
        for version in range(1, 68):
            receipt = {
                "key": "a" * 64,
                "prompt_version": version,
                "cache_result": "l1_hit",
            }
            m = {
                "type": "model.cache",
                "model_session_id": "model",
                "generation_id": "generation",
                "usage": {"zing_block0_cache": receipt},
            }
            assert await segment.model_control({**m, "generation_id": "other"})
            assert source.outgoing.empty()
            await segment.model_control(m)
            sent = msgspec.msgpack.decode(await source.outgoing.get())
            assert (
                sent["prompt_version"] == version
                and sent["segment_epoch"] == "segepoch"
            )
        scene_prefill = segment.recording.manifest["scene_prefill"]
        assert scene_prefill["request_id"] == "previous-prefill"
        assert scene_prefill["cache_key"] == "b" * 64
        assert scene_prefill["generation_cache"]["cache_key"] == "a" * 64
        assert scene_prefill["generation_cache"]["prompt_version"] == 1
        assert observed == [("cutscene", scene_prefill["generation_cache"])]
        assert len(segment.recording.manifest["model_cache_receipts"]) == 64
        assert (
            segment.recording.manifest["model_cache_receipts"][0]["prompt_version"] == 4
        )

    asyncio.run(run())


@pytest.mark.parametrize("tier", ["l1_hit", "l2_hit"])
def test_native_prefix_restore_report_does_not_require_consumer_events(tier):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.block0_telemetry import (
        Block0Telemetry,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
        PrefixRestore,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_session import (
        ZingSourceStreamState,
    )

    class Event:
        def query(self):
            return True

        def elapsed_time(self, end):
            assert isinstance(end, Event)
            return 148.5

        def synchronize(self):
            raise AssertionError("Reporting must not add a GPU synchronization")

    state = ZingSourceStreamState()
    state.block0_key = "a" * 64
    state.block0_prompt_version = 2
    state.block0_telemetry = Block0Telemetry()
    state.block0_telemetry.cache_result = tier
    # Use the native PR499 ticket, not a permissive mock hiding missing fields.
    state.block0_restore = PrefixRestore(
        None, Event(), Event(), None, None, [], 1.0, 0.0
    )
    assert not hasattr(state.block0_restore, "consumer_end")
    assert state._take_block0_report([]) is None
    assert state._take_block0_report([{"prompt_version": 1}]) is None
    report = state._take_block0_report([{"prompt_version": 2}])
    assert report["cache_key"] == "a" * 64
    assert report["prompt_version"] == 2 and report["cache_result"] == tier
    assert report["cuda_ms"] == {"restore": 148.5}
    assert state._take_block0_report([{"prompt_version": 2}]) is None


def test_pending_restore_report_never_waits_or_fabricates_cuda_time():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.block0_telemetry import (
        Block0Telemetry,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
        PrefixRestore,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_session import (
        ZingSourceStreamState,
    )

    class Pending:
        def query(self):
            return False

        def elapsed_time(self, end):
            raise AssertionError("An incomplete transfer has no elapsed CUDA time")

        def synchronize(self):
            raise AssertionError("Reporting must not wait for an incomplete transfer")

    state = ZingSourceStreamState()
    state.block0_key = "b" * 64
    state.block0_prompt_version = 1
    state.block0_telemetry = Block0Telemetry()
    state.block0_telemetry.cache_result = "l2_hit"
    state.block0_restore = PrefixRestore(
        None, Pending(), Pending(), None, None, [], 0, 0
    )
    report = state._take_block0_report([{}])
    assert report["cache_result"] == "l2_hit" and report["cuda_ms"] == {}
