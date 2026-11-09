"""Bounded native overlap, stream epochs and final-frame completion."""

import asyncio
from types import SimpleNamespace as NS

import msgspec
import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_transport import (
    RTCSubscription,
)


class Peer:
    def __init__(self):
        self.calls = []
        self.closed = False

    async def send_frame(self, message):
        done = asyncio.get_running_loop().create_future()
        self.calls.append((message, done))
        self.on_event(dict(type="accepted", track=message["stream"]))
        return await done

    def finish(self, i):
        message, done = self.calls[i]
        done.set_result({"rtp_timestamp": 100 + message["output_frame_start"]})

    async def close(self):
        self.closed = True


def make_sub(*, model_cadence=False):
    peer = Peer()
    config = NS(
        source_max_bitrate=1_000_000,
        playback_max_bitrate=2_500_000,
        rtc_model_cadence=model_cadence,
    )
    owner = NS(
        session=NS(
            config=config,
            source_epoch="source",
            current_model_stream="model",
            source_playback=None,
            current_segment=None,
        ),
        report=NS(playback_delivery={}),
        terminal=None,
        user_id="user",
        request_keyframe=lambda _: None,
        host=NS(config=config),
    )
    return RTCSubscription(owner, 1, peer), peer, owner


def test_cadence_is_opt_in_attachment_local_and_model_only():
    async def run():
        for enabled in (False, True):
            sub, peer, _ = make_sub(model_cadence=enabled)
            try:
                original = media("model", 0)
                sub.offer_media(original)
                sub.offer_media(media("whitebox", 0))
                assert "rtc_pts_us" not in original
                assert sub.media["whitebox"].frame_limit == 8
                assert sub.media["whitebox"].age == 0.2
                assert sub.media["model"].age == (0.5 if enabled else 0.2)
                assert sub.media["model"].frame_limit == (17 if enabled else 8)
                await until(lambda: len(peer.calls) == 2)
                for message, _ in peer.calls:
                    assert ("rtc_pts_us" in message) == (
                        enabled and message["stream"] == "model"
                    )
                sub.reset_model("next")
                if enabled:
                    assert sub.model_clock.count == 0
            finally:
                sub.stop("done")
                await sub.wait_closed()

    asyncio.run(run())


def media(stream, index, *, epoch=None, size=100):
    return dict(
        stream=stream,
        media_stream_id=epoch or ("source" if stream == "whitebox" else "model"),
        output_frame_start=index,
        source_frame_index=index,
        pts_us=index * 33333,
        keyframe=index == 0,
        payload=b"x" * size,
    )


async def until(predicate):
    async with asyncio.timeout(1):
        while not predicate():
            await asyncio.sleep(0.001)


def test_both_tracks_overlap_but_no_more_than_two_per_track():
    async def run():
        sub, peer, owner = make_sub()
        try:
            for name in ("whitebox", "model"):
                for i in range(3):
                    sub.offer_media(media(name, i))
            await until(lambda: len(peer.calls) == 4)
            assert sub._inflight("model") == sub._inflight("whitebox") == 2
            assert [
                m["output_frame_start"] for m, _ in peer.calls if m["stream"] == "model"
            ] == [0, 1]
            # The second AU need not wait for the first AU's departure.
            for i in range(4):
                peer.finish(i)
            await until(lambda: len(peer.calls) == 6)
            for i in range(4, 6):
                peer.finish(i)
            await until(lambda: not sub.handoffs)
            for name in ("whitebox", "model"):
                row = sub.snapshot()[name]
                assert (
                    row["offered_frames"]
                    == row["submitted_frames"]
                    == row["departed_frames"]
                    == 3
                )
                assert not sum(row["dropped"].values())
        finally:
            sub.stop("done")
            await sub.wait_closed()
        saved = owner.report.playback_delivery["rtc"]["recent_attachments"][0]
        assert saved["closed"] and saved["streams"]["model"]["departed_frames"] == 3

    asyncio.run(run())


def test_large_idr_runs_alone_when_feedback_is_below_quality_floor():
    async def run():
        sub, peer, _ = make_sub()
        sub.target_bitrate = 100_000
        try:
            sub.offer_media(media("model", 0, size=2000))
            sub.offer_media(media("whitebox", 0, size=2000))
            await until(lambda: len(peer.calls) == 1)
            await asyncio.sleep(0.015)
            assert len(peer.calls) == 1 and sub._inflight() == 1
            peer.finish(0)
            await until(lambda: len(peer.calls) == 2)
            assert sub._inflight() == 1
        finally:
            sub.stop("done")
            await sub.wait_closed()
        assert peer.calls[-1][1].cancelled() and peer.closed

    asyncio.run(run())


def test_eos_waits_for_all_model_departures_and_uses_highest_frame_not_completion_order():
    async def run():
        sub, peer, _ = make_sub()
        try:
            for i in range(2):
                sub.offer_media(media("model", i))
            await until(lambda: len(peer.calls) == 2)
            sub.offer_control(
                dict(
                    type="render.status",
                    segment={"state": "ended", "segment_epoch": "model"},
                )
            )
            assert msgspec.msgpack.decode(await sub.recv())["type"] == "render.status"
            end = asyncio.create_task(sub.recv())
            peer.finish(1)
            await asyncio.sleep(0.015)
            assert not end.done()
            peer.finish(0)
            result = msgspec.msgpack.decode(await asyncio.wait_for(end, 1))
            assert result["type"] == "rtc.eos" and result["final_rtp_timestamp"] == 101
            assert result["delivery"]["handoff_frames"] == 2
        finally:
            sub.stop("done")
            await sub.wait_closed()

    asyncio.run(run())


def test_epoch_change_does_not_assign_old_completion_to_new_model():
    async def run():
        sub, peer, _ = make_sub()
        try:
            sub.offer_media(media("model", 0))
            await until(lambda: len(peer.calls) == 1)
            sub.reset_model("next")
            sub.offer_media(media("model", 0, epoch="next"))
            await until(lambda: len(peer.calls) == 2)
            peer.finish(0)
            await asyncio.sleep(0.015)
            assert sub.sent["model"] == 0 and sub.last_rtp["model"] is None
            peer.finish(1)
            await until(lambda: not sub.handoffs)
            assert (
                sub.sent["model"] == 1
                and sub.snapshot()["model"]["departed_frames"] == 2
            )
        finally:
            sub.stop("done")
            await sub.wait_closed()

    asyncio.run(run())


def test_failed_handoff_releases_credits_without_emitting_successful_eos():
    async def run():
        sub, peer, owner = make_sub()
        try:
            for i in range(2):
                sub.offer_media(media("model", i))
            await until(lambda: len(peer.calls) == 2)
            owner.terminal = {"type": "session.completed"}
            peer.calls[0][1].set_exception(RuntimeError("test send failure"))
            await until(lambda: sub.media_task.done())
            assert not sub.handoffs and peer.calls[1][1].cancelled()
            message = msgspec.msgpack.decode(await sub.recv())
            assert message["type"] == "error" and message["code"] == "RTC_MEDIA_STALLED"
            assert sub.failed["model"] == sub.cancelled["model"] == 1
            assert sub.departed["model"] == 0
        finally:
            sub.stop("done")
            await sub.wait_closed()

    asyncio.run(run())


def test_source_and_raw_model_counts_are_separate_from_encoded_handoff():
    async def run():
        sub, peer, owner = make_sub()
        owner.session.submitted = 30
        owner.session.current_segment = NS(raw_model_frames=23)
        try:
            snapshot = sub.snapshot()
            assert snapshot["whitebox"]["produced_frames"] == 30
            assert snapshot["model"]["produced_frames"] == 23
            assert snapshot["model"]["offered_frames"] == 0
            assert (
                snapshot["whitebox"]["sample_mono_ms"]
                == snapshot["model"]["sample_mono_ms"]
            )
            owner.session.current_segment = None
            sub.reset_model(None)
            assert sub.snapshot()["model"]["produced_frames"] is None
        finally:
            sub.stop("done")
            await sub.wait_closed()

    asyncio.run(run())


@pytest.mark.parametrize("version", [0, 1])
def test_periodic_delivery_does_not_break_cached_players_without_opt_in(version):
    async def run():
        sub, peer, owner = make_sub()
        peer.delivery_stats_version = version
        try:
            await asyncio.sleep(0.55)
            types = [kind for _, kind, _, _ in sub.control]
            assert ("rtc.delivery" in types) == (version == 1)
            # Reports remain available even when an older player cannot consume
            # the additional signaling event.
            assert owner.report.playback_delivery["rtc"]["recent_attachments"][0][
                "streams"
            ]
        finally:
            sub.stop("done")
            await sub.wait_closed()

    asyncio.run(run())


def test_replaced_rtc_attachment_gets_a_fresh_area_challenge():
    async def run():
        first, _, _ = make_sub()
        second, _, _ = make_sub()
        try:
            assert len(first.area_challenge) == 32
            assert first.area_challenge != second.area_challenge
        finally:
            first.stop("done")
            second.stop("done")
            await first.wait_closed()
            await second.wait_closed()

    asyncio.run(run())


def test_model_burst_survives_vbv_drain_but_still_expires_when_stalled():
    """A healthy 250 ms keyframe drain must not break the following IDR chain."""

    async def run():
        sub, _, _ = make_sub(model_cadence=True)
        try:
            now = [0.0]
            model = sub.media["model"]
            source = sub.media["whitebox"]
            model.clock = source.clock = lambda: now[0]
            for queue, name in ((model, "model"), (source, "whitebox")):
                queue.offer(media(name, 0, size=100_000))
                queue.offer(media(name, 1, size=20_000))
            now[0] = 0.25
            assert model.peek_size() is not None
            assert source.peek_size() is None
            assert model.take()["output_frame_start"] == 0
            assert model.take()["output_frame_start"] == 1
            model.offer(media("model", 2, size=20_000))
            now[0] = 0.751
            assert model.peek_size() is None
            assert model.drops["expired"] == 1
            assert model.needs_key
            assert model.byte_limit == 1 << 20
        finally:
            sub.stop("done")
            await sub.wait_closed()

    asyncio.run(run())


@pytest.mark.parametrize("budget", ["messages", "bytes"])
def test_optional_prewarm_control_cannot_close_the_reliable_attachment(budget):
    async def run():
        sub, _, _ = make_sub()
        try:
            if budget == "messages":
                for _ in range(512):
                    sub.offer_control(dict(type="fixture"))
            else:
                sub.offer_control(dict(type="fixture", payload=b"x" * ((8 << 20) - 64)))
            count = len(sub.control)
            size = sub.control_bytes
            observation = dict(
                type="model.context.prewarm.observation",
                status="memory-ready",
                window_id="a" * 32,
            )
            sub.offer_control(observation, optional=True)
            assert not sub.closed
            assert len(sub.control) == count
            assert sub.control_bytes == size
            assert msgspec.msgpack.decode(await sub.recv())["type"] == "fixture"
            sub.offer_control(observation, optional=True)
            assert sub.control[-1][1] == "model.context.prewarm.observation"

            # A queued optional receipt must make room for a later required event.
            required = dict(type="required")
            if budget == "bytes":
                required["payload"] = b"x" * ((8 << 20) - 64)
            sub.offer_control(required)
            assert not sub.closed
            assert sub.control[-1][1] == "required"
            assert all(
                entry[1] != "model.context.prewarm.observation" for entry in sub.control
            )
        finally:
            sub.stop("done")
            await sub.wait_closed()

    asyncio.run(run())
