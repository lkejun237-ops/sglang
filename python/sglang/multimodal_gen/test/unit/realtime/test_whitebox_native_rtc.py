"""Native RTC admission, handoff, and per-session encoder feedback."""

import base64
import hashlib
import hmac
import json

import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_auth import RTCCapabilities
from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_transport import (
    AUQueue,
    encoder_budget,
)


def test_rtp_identity_does_not_duplicate_reliable_ui_events_or_mutate_archive_frame():
    import asyncio

    from sglang.multimodal_gen.runtime.realtime.whitebox.native_rtc import NativeRTC

    async def run():
        peer = NativeRTC(lambda _: None)
        calls = []

        async def exchange(kind, **values):
            calls.append((kind, values))
            return {"rtp_timestamp": 123}

        peer.exchange = exchange
        message = dict(
            stream="model",
            media_stream_id="epoch",
            pts_us=42,
            source_frame_index=7,
            payload=b"encoded-h264",
            states={"ui_events": [{"revision": 9}]},
            source={"cli_frame_header": {"source": {"uiRevision": 9}}},
        )
        message["area_offer"] = {"challenge": "attachment", "output": {"pts_us": 42}}
        await peer.send_frame(message)
        values = calls[0][1]
        assert values["payload"] == message["payload"]
        assert (
            "states" not in values["metadata"] and "payload" not in values["metadata"]
        )
        assert values["metadata"]["source"] == message["source"]
        assert values["metadata"]["area_offer"] == message["area_offer"]
        assert message["states"]["ui_events"] == [{"revision": 9}]
        assert values["pts_us"] == 42
        # Optional RTP retiming must not change the source identity or archives.
        await peer.send_frame(dict(message, rtc_pts_us=1234))
        retimed = calls[-1][1]
        assert retimed["pts_us"] == 1234
        assert retimed["metadata"]["pts_us"] == message["pts_us"] == 42
        assert "rtc_pts_us" not in retimed["metadata"]
        assert "rtc_pts_us" not in message

    asyncio.run(run())


def test_source_only_does_not_reserve_bandwidth_for_an_absent_model():
    assert encoder_budget(1_000_000, 1_000_000, 2_500_000, model_active=False) == {
        "whitebox": 900_000
    }
    assert encoder_budget(4_200_000, 1_000_000, 2_500_000, model_active=False) == {
        "whitebox": 1_000_000
    }


def test_dual_stream_feedback_reserves_model_quality_before_source_bandwidth():
    assert encoder_budget(4_200_000, 1_000_000, 2_500_000, model_active=True) == {
        "whitebox": 1_000_000,
        "model": 2_500_000,
    }
    constrained = encoder_budget(1_000_000, 1_000_000, 2_500_000, model_active=True)
    assert constrained == {"whitebox": 300_000, "model": 1_000_000}
    assert encoder_budget(1_500_000, 1_000_000, 2_500_000, model_active=True) == {
        "whitebox": 350_000,
        "model": 1_000_000,
    }


@pytest.mark.parametrize(
    "rate", [1, 100_000, 250_000, 300_854, 500_000, 1_000_000, 2_000_000, 10_000_000]
)
def test_model_encoder_floor_survives_feedback_collapse_and_recovery(rate):
    targets = encoder_budget(rate, 1_000_000, 2_500_000, model_active=True)
    assert 1_000_000 <= targets["model"] <= 2_500_000
    assert 300_000 <= targets["whitebox"] <= 1_000_000


def test_explicit_lower_deployment_ceiling_is_never_exceeded():
    assert encoder_budget(300_000, 100_000, 500_000, model_active=True) == {
        "whitebox": 100_000,
        "model": 500_000,
    }


def test_shanghai_model_range_and_whitebox_budget(monkeypatch):
    from sglang.multimodal_gen.runtime.realtime.whitebox import rtc_transport

    monkeypatch.setattr(rtc_transport, "MODEL_ENCODER_MIN_BITRATE", 2_500_000)
    for rate in [1, 279_744, 1_000_000, 3_200_000, 4_200_000, 5_600_000, 10_000_000]:
        target = encoder_budget(rate, 1_000_000, 4_000_000, model_active=True)
        assert 2_500_000 <= target["model"] <= 4_000_000
        assert 300_000 <= target["whitebox"] <= 1_000_000
    assert encoder_budget(5_600_000, 1_000_000, 4_000_000, model_active=True) == {
        "whitebox": 1_000_000,
        "model": 4_000_000,
    }


def test_native_feedback_keeps_measured_bandwidth_separate_from_send_floor():
    from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_transport import (
        RTCSubscription,
    )

    sub = object.__new__(RTCSubscription)
    sub.closed = False
    sub.native_event(
        {"type": "bitrate", "bitrate": 1_500_000, "receiver_bitrate": 279_744}
    )
    assert sub.target_bitrate == 1_500_000
    assert sub.receiver_bitrate == 279_744
    assert encoder_budget(
        sub.target_bitrate, 1_000_000, 2_500_000, model_active=True
    ) == {"whitebox": 350_000, "model": 1_000_000}


def test_feedback_lifts_subfloor_target_even_inside_hysteresis_and_retries_unready(
    monkeypatch,
):
    import asyncio
    from types import SimpleNamespace as NS

    from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_transport import (
        RTCSubscription,
    )

    async def run():
        calls = []
        source = NS(max_bitrate=1_000_000, set_target_bitrate=lambda rate: True)
        model = NS(max_bitrate=2_500_000, context=object(), drained=False)

        def update(rate):
            calls.append(rate)
            return len(calls) > 1

        model.set_target_bitrate = update
        sub = object.__new__(RTCSubscription)
        sub.closed = False
        sub.generation = 1
        sub.peer = NS(delivery_stats_version=0)
        sub._record_delivery = lambda: None
        sub.snapshot = lambda: {}
        sub.offer_control = lambda _: None
        sub.target_bitrate = 300_854
        sub.owner = NS(
            session=NS(
                source_playback=source,
                current_segment=NS(playback=model),
                config=NS(source_max_bitrate=1_000_000, playback_max_bitrate=2_500_000),
            ),
            host=NS(
                config=NS(source_max_bitrate=1_000_000, playback_max_bitrate=2_500_000)
            ),
        )
        sub.sent = {"model": 1}
        sub.applied_bitrates = {("model", id(model)): 950_000}
        ticks = 0

        async def tick(_):
            nonlocal ticks
            ticks += 1
            if ticks == 2:
                assert sub.applied_bitrates[("model", id(model))] == 950_000
            if ticks == 3:
                sub.closed = True

        monkeypatch.setattr(asyncio, "sleep", tick)
        await sub._feedback()
        assert calls == [1_000_000, 1_000_000]
        assert sub.applied_bitrates[("model", id(model))] == 1_000_000
        assert sub.target_bitrate == 300_854  # Never raise the receiver/pacer budget.

    asyncio.run(run())


def au(index, *, key=False, size=10):
    return dict(
        output_frame_start=index,
        keyframe=key,
        payload=b"x" * size,
        stream="model",
        media_stream_id="epoch",
        pts_us=index * 33333,
    )


def test_rtc_handoff_never_waits_for_application_ack_and_preserves_order():
    now = [0.0]
    requests = []
    queue = AUQueue(lambda: requests.append(now[0]), clock=lambda: now[0])
    for i in range(4):
        queue.offer(au(i, key=i == 0))
    assert [queue.take()["output_frame_start"] for _ in range(4)] == list(range(4))
    assert queue.bytes == 0 and sum(queue.drops.values()) == 0
    assert not requests


def test_handoff_expiry_is_from_au_enqueue_not_source_capture_and_keeps_new_idr():
    now = [1000.0]
    queue = AUQueue(lambda: None, clock=lambda: now[0])
    for i in range(4):
        queue.offer(au(i, key=i == 0))
    now[0] += 0.15
    for i in range(4, 7):
        queue.offer(au(i, key=i == 4))
    now[0] += 0.06
    assert [queue.take()["output_frame_start"] for _ in range(3)] == [4, 5, 6]
    assert queue.drops["expired"] == 4


def test_expired_p_frames_are_not_replayed_and_idr_requests_are_coalesced():
    now, requests = [0.0], []
    queue = AUQueue(lambda: requests.append(now[0]), clock=lambda: now[0])
    queue.offer(au(0, key=True))
    assert queue.take()
    for i in range(1, 4):
        queue.offer(au(i))
    now[0] += 0.3
    assert queue.take() is None
    for i in range(4, 20):
        queue.offer(au(i))
        assert queue.take() is None
    assert len(requests) == 1
    assert not queue.items and queue.bytes == 0
    queue.offer(au(20, key=True))
    assert queue.take()["output_frame_start"] == 20


def test_rtc_queue_byte_budget_and_epoch_reset_are_bounded():
    queue = AUQueue(lambda: None, byte_limit=512)
    queue.offer(au(0, key=True, size=1024))
    assert not queue.items
    queue.offer(au(1, key=True))
    queue.clear("replaced")
    queue.offer(au(2))
    assert queue.take() is None
    assert queue.drops["replaced"] == 1
    assert queue.drops["awaiting_keyframe"] == 1


def test_model_clip_capacity_does_not_extend_the_network_expiry_budget():
    now = [0.0]
    queue = AUQueue(lambda: None, clock=lambda: now[0], frames=17)
    for i in range(17):
        queue.offer(au(i, key=i == 0))
    assert len(queue.items) == 17 and not any(queue.drops.values())
    now[0] = 0.201
    assert queue.take() is None
    assert queue.drops["expired"] == 17
    queue.offer(au(17))
    assert queue.take() is None
    queue.offer(au(18, key=True))
    assert queue.take()["output_frame_start"] == 18


def signed(secret, claims):
    def b64(value):
        return base64.urlsafe_b64encode(value).decode().rstrip("=")

    payload = b64(json.dumps(claims).encode())
    return (
        payload
        + "."
        + b64(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest())
    )


def test_rtc_capability_is_bound_to_login_world_origin_expiry_and_one_create():
    secret = "test-key-" * 8
    claims = dict(
        v=1,
        aud="whitebox-rtc",
        sub="logged-in-user",
        origin="https://platform.test",
        world_id="world",
        source_id="source",
        iat=100,
        exp=220,
        jti="1" * 32,
    )
    guard = RTCCapabilities(secret, now=lambda: 110)
    world = dict(world_id="world", source_id="source")
    token = signed(secret, claims)
    for origin, supplied in [
        ("https://other.test", world),
        (claims["origin"], {**world, "world_id": "other"}),
    ]:
        with pytest.raises(ValueError, match="authorization"):
            guard.verify(token, origin=origin, world=supplied)
    assert (
        guard.verify(token, origin=claims["origin"], world=world)["sub"]
        == "logged-in-user"
    )
    with pytest.raises(ValueError):
        guard.verify(token, origin=claims["origin"], world=world)
    for changed in [
        {**claims, "exp": 109},
        {**claims, "exp": 400},
        {**claims, "iat": 200},
        [],
    ]:
        with pytest.raises(ValueError):
            RTCCapabilities(secret, now=lambda: 110).verify(
                signed(secret, changed), origin=claims["origin"], world=world
            )
    with pytest.raises(ValueError):
        guard.verify(
            signed("wrong-secret-" * 5, claims), origin=claims["origin"], world=world
        )


def test_prefill_observation_revalidates_scope_without_consuming_create_tickets():
    secret = "test-key-" * 8
    claims = dict(
        v=1,
        aud="whitebox-prefill",
        sub="logged-in-user",
        origin="https://platform.test",
        world_id="world",
        source_id="source",
        scope="world",
        operation="status",
        iat=100,
        exp=220,
        jti="3" * 32,
    )
    world = dict(world_id="world", source_id="source")
    guard = RTCCapabilities(secret, audience="whitebox-prefill", now=lambda: 110)
    token = signed(secret, claims)
    for _ in range(2):
        assert (
            guard.verify(
                token,
                origin=claims["origin"],
                world=world,
                operation="status",
                scope="world",
                consume=False,
            )["sub"]
            == "logged-in-user"
        )
    for changes in (
        {"world": {**world, "world_id": "other"}},
        {"origin": "https://other.test"},
        {"operation": "submit"},
        {"scope": "other"},
    ):
        with pytest.raises(ValueError):
            guard.verify(
                token,
                **{
                    "origin": claims["origin"],
                    "world": world,
                    "operation": "status",
                    "scope": "world",
                    "consume": False,
                    **changes,
                },
            )
    rtc = RTCCapabilities(secret, now=lambda: 110)
    with pytest.raises(ValueError):
        rtc.verify(
            signed(secret, {**claims, "aud": "whitebox-rtc"}),
            origin=claims["origin"],
            world=world,
            consume=False,
        )


def test_api_token_rtc_ticket_accepts_any_client_but_retains_signed_scope():
    secret = "test-key-" * 8
    claims = dict(
        v=2,
        auth_method="api-token",
        aud="whitebox-rtc",
        sub="token-owner",
        world_id="world",
        source_id="source",
        iat=100,
        exp=220,
        jti="2" * 32,
    )
    world = dict(world_id="world", source_id="source")
    for origin in (None, "null", "http://localhost:4326", "https://native-client.test"):
        guard = RTCCapabilities(secret, now=lambda: 110)
        assert (
            guard.verify(signed(secret, claims), origin=origin, world=world)["sub"]
            == "token-owner"
        )
        with pytest.raises(ValueError):
            guard.verify(signed(secret, claims), origin=origin, world=world)
    for changed in (
        {**claims, "auth_method": "cookie"},
        {**claims, "origin": "https://platform.test"},
        {**claims, "v": 3},
        {**claims, "v": True},
        {**claims, "exp": 110},
        {**claims, "exp": 221},
        {**claims, "sub": ""},
    ):
        with pytest.raises(ValueError):
            RTCCapabilities(secret, now=lambda: 110).verify(
                signed(secret, changed), origin=None, world=world
            )
    for field in ("world_id", "source_id"):
        with pytest.raises(ValueError):
            RTCCapabilities(secret, now=lambda: 110).verify(
                signed(secret, claims), origin=None, world={**world, field: "other"}
            )
    with pytest.raises(ValueError):
        RTCCapabilities(secret, now=lambda: 110).verify(
            signed("wrong-secret-" * 5, claims), origin=None, world=world
        )


@pytest.mark.parametrize("immediate", [False, True])
def test_congestion_feedback_keeps_frame_identity_and_gop_across_rate_changes(
    immediate,
):
    import av
    import numpy as np

    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.playback import (
        H264PlaybackEncoder,
    )

    if immediate:
        from sglang.multimodal_gen.runtime.realtime.whitebox.native_capture import (
            library,
        )

        try:
            library()
        except RuntimeError:
            pytest.skip("Build the native encoder before running this integration case")
    encoder = H264PlaybackEncoder(
        VideoSpec(832, 480),
        capture_clock=True,
        immediate_capture=immediate,
        max_bitrate=2_500_000,
        gop=6,
    )
    decoder = av.CodecContext.create("h264", "r")
    rows = []
    try:
        for i in range(50):
            if i in (10, 30):
                encoder.set_target_bitrate(500_000 if i == 10 else 2_500_000)
            pixels = np.full((480, 832, 3), i * 4, np.uint8).tobytes()
            rows.extend(encoder.push(i, pixels, i * 33333))
        rows.extend(encoder.drain())
        assert [i for i, _ in rows] == list(range(50))
        assert (
            sum(len(decoder.decode(av.Packet(value["payload"]))) for _, value in rows)
            == 50
        )
        assert encoder.stream["max_bitrate"] == 2_500_000
        assert 1 < sum(value["keyframe"] for _, value in rows) < 15
        with pytest.raises(ValueError):
            encoder.set_target_bitrate(2_500_001)
        assert encoder.set_target_bitrate(1_000_000) is False
    finally:
        encoder.close()
