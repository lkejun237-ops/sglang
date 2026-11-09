# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.runtime.realtime.whitebox.prewarm_observation import (
    PrewarmPlayerObservations,
    is_prewarm_observation_request,
)


def observer():
    value = PrewarmPlayerObservations("source", "epoch", "consumer")
    assert value.open(
        "window-1", 1, segment_id="segment", segment_epoch="segment-epoch"
    )
    return value


def candidate(**details):
    return dict(
        intent_id="intent",
        candidate_revision=1,
        event_id="choose-S06-2",
        request_id="prefill",
        **details,
    )


def test_observation_requires_explicit_v2_player_opt_in_and_replays_small_wire():
    request = dict(version=2, type="model.context.prewarm.observe")
    assert is_prewarm_observation_request(request)
    for invalid in [
        None,
        [],
        {},
        dict(request, version=2.0),
        dict(request, source="other"),
    ]:
        assert not is_prewarm_observation_request(invalid)
    value = observer()
    assert value.capability == dict(
        model_prewarm_observation="worldkit-model-prewarm-v1",
        model_prewarm_consumer_id="consumer",
    )
    assert value.record("window-1", 1, "accepted", **candidate())
    assert value.record("window-1", 1, "memory-ready", **candidate(cache_key="1" * 64))
    assert value.record(
        "window-1", 2, "invalidated", **candidate(reason="query-changed")
    )
    assert value.record(
        "window-1", 1, "persistence-failed", **candidate(reason="storage")
    )
    assert value.record("window-1", 1, "computing", **candidate()) is None
    assert (
        value.record("window-1", 1, "memory-ready", **candidate(cache_key="1" * 64))
        is None
    )
    assert (
        value.record("window-1", 2, "memory-ready", **candidate(prompt="secret"))
        is None
    )
    replay = value.replay(3)
    assert [event["status"] for event in replay] == [
        "window",
        "memory-ready",
        "persistence-failed",
        "invalidated",
    ]
    assert all(event["connection_generation"] == 3 for event in replay)
    assert replay[0] == dict(
        version=2,
        type="model.context.prewarm.observation",
        source_session_id="source",
        source_epoch="epoch",
        connection_generation=3,
        consumer_id="consumer",
        window_id="window-1",
        inference_revision=1,
        window_generation=1,
        decision_revision=0,
        segment_id="segment",
        segment_epoch="segment-epoch",
        status="window",
    )
    replay[1]["intent_id"] = "mutated"
    assert value.replay(4)[1]["intent_id"] == "intent"
    assert value.open("window-2", 2)
    assert value.open("window-1", 1) is None
    assert value.record("window-1", 3, "persisted", **candidate()) is None
    assert value.replay(5)[0]["window_id"] == "window-2"
    assert len(value.replay(5)) == 1


def test_formal_cache_requires_native_result_and_actual_formal_scope():
    value = observer()
    value.record("window-1", 1, "memory-ready", **candidate(cache_key="1" * 64))
    cache = dict(cache_key="2" * 64, cache_result="l1_hit", prompt_version=3)
    formal = dict(
        formal_request_id="formal-1",
        prompt_version=3,
        segment_id="segment",
        segment_epoch="segment-epoch",
    )
    assert value.record("window-1", 1, "hit", **candidate()) is None
    assert (
        value.record_cache("window-1", cache=dict(cache, prompt_version=1), **formal)
        is None
    )
    assert (
        value.record_cache(
            "window-1", cache=dict(cache, cache_result="bypass"), **formal
        )
        is None
    )
    assert (
        value.record_cache("window-1", cache=cache, **dict(formal, segment_epoch="old"))
        is None
    )
    result = value.record_cache("window-1", cache=cache, **formal)
    assert result["status"] == "hit"
    assert result["formal_request_id"] == "formal-1"
    assert result["cache_key"] == "2" * 64
    assert value.replay(1)[1]["cache_key"] == "1" * 64
    assert value.record_cache("window-1", cache=cache, **formal) is None


def test_formal_epoch_can_bind_only_to_the_original_logical_segment():
    value = PrewarmPlayerObservations("source", "epoch", "consumer")
    window = value.open("window-1", 1, segment_id="segment")
    assert "segment_id" not in window
    assert value.record("window-1", 1, "memory-ready", **candidate(cache_key="1" * 64))
    cache = dict(cache_key="1" * 64, cache_result="mixed_hit", prompt_version=2)
    formal = dict(
        formal_request_id="formal-2",
        prompt_version=2,
        segment_id="segment",
        segment_epoch="active-epoch",
    )
    assert (
        value.record_cache(
            "window-1", cache=cache, **dict(formal, segment_id="foreign")
        )
        is None
    )
    observed = value.record_cache("window-1", cache=cache, **formal)
    assert observed["status"] == "hit"
    assert observed["cache_key"] == "1" * 64
    assert observed["segment_id"] == "segment"
    assert observed["segment_epoch"] == "active-epoch"


def test_a_new_reserved_segment_gets_a_new_window_at_the_same_inference_revision():
    value = observer()
    assert value.open("duplicate", 1, segment_id="segment") is None
    newer = value.open("new-window", 1, segment_id="next-segment")
    assert newer["window_generation"] == 2
    assert value.record("window-1", 1, "accepted", **candidate()) is None
    assert value.replay(7)[0]["window_id"] == "new-window"
