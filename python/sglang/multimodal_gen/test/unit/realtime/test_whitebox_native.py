# SPDX-License-Identifier: Apache-2.0
import asyncio
import copy
import json
from dataclasses import replace
from fractions import Fraction

import av
import msgspec
import pytest
from test_whitebox import setup_pipeline

from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
from sglang.multimodal_gen.runtime.realtime.whitebox.native_pixels import (
    decode_native_packet,
)


def native_frame():
    spec = VideoSpec(32, 32)
    descriptor = dict(
        sessionId="test", epoch=0, worldBuildHash="a" * 64, uiBundleHash="b" * 64
    )
    header = dict(
        kind="worldkit-pixel-frame",
        schemaVersion=2,
        pixelFormat="rgb24",
        colorSpace="srgb",
        widthPixels=32,
        heightPixels=32,
        rowStrideBytes=96,
        payloadBytes=spec.frame_bytes,
        worldBuildHash="a" * 64,
        uiBundleHash="b" * 64,
        input=dict(
            protocolVersion=1,
            sessionId="test",
            epoch=0,
            mediaGeneration=1,
            outputFrameId=1,
            outputPtsUs=1000,
            uiRevision=1,
            uiCompleteThroughUs=1000,
            widthPixels=32,
            heightPixels=32,
            source=dict(
                presentationId="p",
                sdkEpoch=0,
                sourceFrameId=10,
                sourceTimeUs=1000,
                simulationTick=5,
                worldRevision=0,
            ),
        ),
    )
    return spec, descriptor, header


def wire(header, spec):
    data = json.dumps(header).encode()
    return len(data).to_bytes(4, "big") + data + bytes(spec.frame_bytes)


def test_native_contract_rejects_corrupt_identity_layout_and_clock():
    spec, descriptor, header = native_frame()
    decoded, pixels = decode_native_packet(wire(header, spec), descriptor, spec)
    assert decoded == header["input"] and len(pixels) == spec.frame_bytes
    for key, value in (
        ("worldBuildHash", "c" * 64),
        ("pixelFormat", "bgr24"),
        ("rowStrideBytes", 97),
        ("schemaVersion", True),
        ("payloadBytes", 1),
    ):
        bad = copy.deepcopy(header)
        bad[key] = value
        with pytest.raises(ValueError):
            decode_native_packet(wire(bad, spec), descriptor, spec)
    for key, value in (
        ("epoch", True),
        ("sourceTimeUs", float("nan")),
        ("simulationTick", -1),
    ):
        bad = copy.deepcopy(header)
        target = bad["input"] if key == "epoch" else bad["input"]["source"]
        target[key] = value
        with pytest.raises(ValueError):
            decode_native_packet(wire(bad, spec), descriptor, spec)


@pytest.mark.parametrize("processes", [False, True])
def test_dual_stream_uses_capture_timeline_and_never_calls_server_ui(
    tmp_path, processes
):
    async def run():
        pipeline, h3, engine, archives = setup_pipeline(tmp_path)
        archives.processes = processes

        class Browser:
            metadata = {"mode": "browser", "files": {}}

            async def render(self, *args):
                raise AssertionError("Server UI must not run")

            async def close(self):
                pass

        engine.renderer = Browser()
        times = [0, 170000, 330000, 600000, 830000, 1000000]
        original = engine.frames

        async def frames():
            async for frame in original():
                yield replace(frame, provenance={"capture_pts_us": times[frame.index]})

        engine.frames = frames
        await pipeline.send(
            msgspec.msgpack.encode(
                dict(type="init", whitebox={"world_id": "a"}, prompt="test")
            )
        )
        ready = msgspec.msgpack.decode(await pipeline.recv())
        assert ready["playback_flow_control"] == "whitebox-dual-v2"
        assert ready["play_seconds"] == 180
        messages = {"whitebox": [], "model": []}
        while True:
            message = msgspec.msgpack.decode(await asyncio.wait_for(pipeline.recv(), 5))
            if message["type"] == "session.completed":
                break
            messages[message["stream"]].append(message)
            pipeline.media_delivered()
        for stream in messages.values():
            assert [m["pts_us"] for m in stream] == times
        await pipeline.close()
        await archives.close()
        directory = pipeline.recording.directory
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["recording_status"] == "complete"
        assert set(manifest["videos"]) == {"model", "whitebox"}
        assert manifest["client_ui"]["status"] == "awaiting_upload"
        for name, luminance in (("whitebox", 20), ("model", 100)):
            with av.open(str(directory / (name + ".mp4"))) as video:
                decoded = list(video.decode(video=0))
                assert [f.pts * f.time_base for f in decoded] == [
                    Fraction(t, 1_000_000) for t in times
                ]
                assert (
                    abs(float(decoded[0].to_ndarray(format="rgb24").mean()) - luminance)
                    < 4
                )
        inputs = [
            json.loads(line)
            for line in (directory / "inputs.jsonl").read_text().splitlines()
        ]
        assert inputs[0]["end_time_us"] == times[2]
        assert inputs[1]["start_time_us"] == times[2]
        # The unchanged model still consumes its own regular 24fps clock.
        assert [
            e["payload"]["pts_us"] for e in h3.events if e["kind"] == "source.append"
        ] == [pipeline.config.spec.pts(i) for i in range(6)]

    asyncio.run(run())


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("backlog", [513, 967, 4096])
def test_cli_ui_backlog_is_bounded_without_dropping_or_reordering_events(
    monkeypatch, native, backlog
):
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox import cli_room
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        canonical,
    )

    async def run():
        spec, descriptor, header = native_frame()
        view_type = cli_room.NativeCLIView if native else cli_room.CLIView
        view = view_type(
            SimpleNamespace(
                descriptor=descriptor,
                spec=spec,
                prepared_delivery=None,
                trace_id="ui-batch",
            )
        )
        packets = asyncio.Queue()
        count = (backlog + 511) // 512
        headers = []
        for index in range(count):
            item = copy.deepcopy(header)
            pts = (index + 1) * 1000
            item["input"].update(
                outputFrameId=index + 1, outputPtsUs=pts, uiCompleteThroughUs=pts
            )
            item["input"]["source"].update(sourceFrameId=index + 1, sourceTimeUs=pts)
            frame_header = item["input"]
            if native:
                packets.put_nowait(wire(item, spec))
            else:
                frame_header = dict(frame_header, type="key")
                packets.put_nowait((frame_header, b"encoded-vp8"))
            headers.append(frame_header)

        if not native:
            decoded = SimpleNamespace(
                to_ndarray=lambda **_: SimpleNamespace(
                    tobytes=lambda: bytes(spec.frame_bytes)
                )
            )
            decoder = SimpleNamespace(decode=lambda _: [decoded])
            monkeypatch.setattr(
                av, "CodecContext", SimpleNamespace(create=lambda *_: decoder)
            )
            monkeypatch.setattr(cli_room, "decode_packet", lambda value, *_: value)

        async def send(_):
            pass

        view.control = SimpleNamespace(send=send)
        view.control_task = asyncio.get_running_loop().create_future()
        view.media = SimpleNamespace(recv=packets.get)
        view.config = {"mediaGeneration": 1}
        view.configured.set()
        view.ui_revision, view.ui_through = backlog, count * 1000
        events = [
            dict(
                type="ui.clock",
                epoch=0,
                clock={"uiRevision": index, "completeThroughUs": index * 1000},
            )
            for index in range(backlog)
        ]
        view.ui_events = copy.deepcopy(events)
        view.ui_bytes = sum(len(canonical(event)) for event in events)
        iterator = view.frames()
        delivered = []
        try:
            for index in range(count):
                frame = await asyncio.wait_for(anext(iterator), 1)
                batch = frame.states["ui_events"]
                start = index * 512
                assert len(batch) <= 512
                assert batch == events[start : start + 512]
                delivered.extend(batch)
                remaining = events[start + 512 :]
                assert view.ui_events == remaining
                assert view.ui_bytes == sum(
                    len(canonical(event)) for event in remaining
                )
                assert frame.provenance["cli_frame_header"] == headers[index]
            assert delivered == events
            assert view.ui_events == [] and view.ui_bytes == 0
        finally:
            await iterator.aclose()
            view.control_task.cancel()

    asyncio.run(run())


def test_native_bootstrap_pixels_preceding_ui_are_preserved():
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import NativeCLIView

    async def run():
        spec, descriptor, header = native_frame()
        packets = asyncio.Queue()
        for index, pts in enumerate((1000, 2000, 3000), 1):
            item = copy.deepcopy(header)
            item["input"].update(
                outputFrameId=index, outputPtsUs=pts, uiCompleteThroughUs=pts
            )
            item["input"]["source"].update(sourceFrameId=index, sourceTimeUs=pts)
            packets.put_nowait(wire(item, spec))
        grants = []

        async def send(value):
            grants.append(json.loads(value))

        view = NativeCLIView(
            SimpleNamespace(descriptor=descriptor, spec=spec, prepared_delivery=None)
        )
        view.control = SimpleNamespace(send=send)
        view.control_task = asyncio.get_running_loop().create_future()
        view.media = SimpleNamespace(recv=packets.get)
        view.config = {"mediaGeneration": 1}
        view.configured.set()
        view.ui_bootstrap_time = 1500
        view.ui_revision, view.ui_through = 1, 3000
        snapshot = {"type": "ui.snapshot", "snapshot": {"effectiveSourceTimeUs": 1500}}
        view.ui_events = [snapshot]
        iterator = view.frames()
        first = await anext(iterator)
        second = await anext(iterator)
        assert first.index == 0 and second.index == 1
        assert first.provenance["cli_frame_header"]["outputFrameId"] == 1
        assert first.provenance["capture_pts_us"] == 0
        assert second.provenance["capture_pts_us"] == 1000
        assert first.states == {"ui_events": [snapshot]}
        assert second.states == {"ui_events": []}
        assert len(grants) == 2  # Initial window, then one actually consumed frame.
        assert grants[0]["frames"] == 2
        await iterator.aclose()
        view.control_task.cancel()

    asyncio.run(run())


@pytest.mark.parametrize("byte_budget_frames", [1, 2])
def test_native_pixels_do_not_wait_for_ui_and_keep_prefetch_bounded(
    monkeypatch, byte_budget_frames
):
    """Missing UI never stalls pixels; held plus prefetched frames remain bounded."""
    from types import SimpleNamespace

    import sglang.multimodal_gen.runtime.realtime.whitebox.cli_room as cli

    async def run():
        spec, descriptor, template = native_frame()
        monkeypatch.setattr(cli, "MAX_PIXELS", byte_budget_frames * spec.frame_bytes)
        packets, grants, produced = asyncio.Queue(), [], []
        granted = asyncio.Event()

        async def send(value):
            credit = json.loads(value)
            grants.append(credit)
            assert credit["bytes"] == credit["frames"] * spec.frame_bytes
            for _ in range(credit["frames"]):
                index = len(produced) + 1
                item = copy.deepcopy(template)
                item["input"].update(
                    outputFrameId=index,
                    outputPtsUs=index * 1000,
                    uiCompleteThroughUs=index * 1000,
                )
                item["input"]["source"].update(
                    sourceFrameId=index,
                    sourceTimeUs=index * 1000,
                    simulationTick=index,
                )
                produced.append(index)
                packets.put_nowait(wire(item, spec))
            granted.set()

        view = cli.NativeCLIView(
            SimpleNamespace(descriptor=descriptor, spec=spec, prepared_delivery=None)
        )
        view.control = SimpleNamespace(send=send)
        view.control_task = asyncio.get_running_loop().create_future()
        view.media = SimpleNamespace(recv=packets.get)
        view.config = {"mediaGeneration": 1}
        view.configured.set()
        view.ui_bootstrap_time = None
        view.ui_revision, view.ui_through = -1, -1
        action_receipts = [
            {
                "type": "action.receipt",
                "epoch": 0,
                "actionId": "needs",
                "status": "completed",
            },
            {
                "type": "action.receipt",
                "epoch": 0,
                "actionId": "pet",
                "status": "failed",
                "error": "Unavailable",
            },
        ]
        view.receipts = [
            *action_receipts,
            {"simulationTick": 1, "sequence": 10},
            {"simulationTick": 2, "sequence": 11},
        ]
        iterator = view.frames()
        pending = asyncio.create_task(anext(iterator))
        try:
            await asyncio.wait_for(granted.wait(), 1)
            # No UI snapshot/clock has arrived, but pixels and prefetch can proceed.
            first = await asyncio.wait_for(pending, 1)
            assert produced == list(range(1, byte_budget_frames + 1))
            assert len(grants) == 1
            assert first.states == {"ui_events": []}
            assert first.provenance["cli_frame_header"]["outputFrameId"] == 1
            assert first.observed_input["receipts"] == [
                *action_receipts,
                {"simulationTick": 1, "sequence": 10},
            ]
            # Returning a frame does not return its credit: downstream still owns it.
            assert len(produced) == byte_budget_frames
            assert packets.qsize() == byte_budget_frames - 1
            assert len(grants) == 1
            snapshot = {
                "type": "ui.snapshot",
                "snapshot": {"effectiveSourceTimeUs": 1500},
            }
            view.ui_events = [snapshot]
            # This event does not make the current UI watermark complete; no wait is allowed.
            granted.clear()
            pending = asyncio.create_task(anext(iterator))
            await asyncio.wait_for(granted.wait(), 1)
            assert len(produced) == byte_budget_frames + 1
            assert grants[-1]["frames"] == 1
            second = await asyncio.wait_for(pending, 1)
            assert second.states == {"ui_events": [snapshot]}
            assert view.ui_events == []
            assert second.index == 1
            assert second.provenance["capture_pts_us"] == 1000
            assert second.observed_input["receipts"] == [
                {"simulationTick": 2, "sequence": 11}
            ]
            assert packets.qsize() == byte_budget_frames - 1
            view.stopped = True
            with pytest.raises(StopAsyncIteration):
                await anext(iterator)
            # Stopping with a held frame must not issue another credit.
            assert len(grants) == 2
        finally:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            await iterator.aclose()
            view.control_task.cancel()

    asyncio.run(run())


def test_native_credit_cancellation_does_not_replenish_while_pixels_are_pending():
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import NativeCLIView

    async def run():
        spec, descriptor, _ = native_frame()
        grants, receiving = [], asyncio.Event()

        async def send(value):
            grants.append(json.loads(value))

        async def recv():
            receiving.set()
            await asyncio.Future()

        view = NativeCLIView(
            SimpleNamespace(descriptor=descriptor, spec=spec, prepared_delivery=None)
        )
        view.control = SimpleNamespace(send=send)
        view.control_task = asyncio.get_running_loop().create_future()
        view.media = SimpleNamespace(recv=recv)
        view.config = {"mediaGeneration": 1}
        view.configured.set()
        iterator = view.frames()
        pending = asyncio.create_task(anext(iterator))
        await asyncio.wait_for(receiving.wait(), 1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await iterator.aclose()
        assert len(grants) == 1 and grants[0]["frames"] == 2
        view.control_task.cancel()

    asyncio.run(run())


def test_vfr_remux_preserves_long_stall_and_three_minute_timestamps(tmp_path):
    from sglang.multimodal_gen.runtime.realtime.whitebox.playback import (
        H264PlaybackEncoder,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.recording import H264MuxFile

    spec = VideoSpec(32, 32)
    encoder = H264PlaybackEncoder(spec, max_bitrate=100000)
    path = tmp_path / "long-stall.mp4"
    output = H264MuxFile(path, spec)
    times = [0, 1000000, 179000123, 180000456]
    try:
        for index, pts in enumerate(times):
            encoded = encoder.encode(index, bytes([40 + index * 20]) * spec.frame_bytes)
            output.append(index, encoded, pts_us=pts)
    finally:
        output.close()
        encoder.close()
    with av.open(path) as container:
        frames = list(container.decode(video=0))
        assert [frame.pts * frame.time_base for frame in frames] == [
            Fraction(pts, 1000000) for pts in times
        ]
        assert abs(container.duration / av.time_base - 180.042122) < 0.000002


def test_native_world_reset_retires_queued_pixels_without_restarting_source():
    """The public reset event must fence old pixels/UI and refill only new credit."""
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import NativeCLIView
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        canonical,
    )

    async def run():
        spec, descriptor, template = native_frame()
        descriptor["media"] = {"kind": "pixels"}
        control, packets, grants = asyncio.Queue(), asyncio.Queue(), []
        renewed = asyncio.Event()
        config = dict(
            kind="pixels",
            pixelFormat="rgb24",
            widthPixels=32,
            heightPixels=32,
            rowStrideBytes=96,
            colorSpace="srgb",
            fps=24,
        )

        async def messages():
            while True:
                yield await control.get()

        async def send(wire_value):
            value = json.loads(wire_value)
            grants.append(value)
            if value.get("mediaGeneration") == 2:
                renewed.set()

        async def consume_ui(_):
            pass

        def packet(index, generation=1):
            value = copy.deepcopy(template)
            value["input"].update(
                outputFrameId=index,
                outputPtsUs=index * 1000,
                mediaGeneration=generation,
                uiCompleteThroughUs=index * 1000,
            )
            value["input"]["source"].update(
                sourceFrameId=index, sourceTimeUs=index * 1000
            )
            if generation == 2:
                value["input"]["source"].update(
                    presentationId="reset-p", sdkEpoch=1, simulationTick=0
                )
            return wire(value, spec)

        view = NativeCLIView(
            SimpleNamespace(descriptor=descriptor, spec=spec, prepared_delivery=None)
        )

        # Async iteration belongs to the real control owner; sending remains observable.
        class Control:
            def __aiter__(self):
                return messages()

            async def send(self, value):
                await send(value)

        view.control = Control()
        view.renderer = SimpleNamespace(consume_ui=consume_ui)
        view.media = SimpleNamespace(recv=packets.get)
        view.control_task = asyncio.create_task(view._control())
        control.put_nowait(
            json.dumps(
                dict(type="media.config", epoch=0, mediaGeneration=1, config=config)
            )
        )
        packets.put_nowait(packet(1))
        iterator = view.frames()
        try:
            first = await asyncio.wait_for(anext(iterator), 1)
            assert first.index == 0
            view.receipts = [{"simulationTick": 99, "sequence": 7}]
            old = {
                "type": "ui.snapshot",
                "epoch": 0,
                "snapshot": {"effectiveSourceTimeUs": 1000},
            }
            view.ui_events = [copy.deepcopy(old) for _ in range(700)]
            view.ui_bytes = sum(len(canonical(event)) for event in view.ui_events)
            pending = asyncio.get_running_loop().create_future()
            view.area_pending["old"] = ("old-operation", pending)
            view.area_transition = {"revision": 1}
            view.area_frames[1] = {"old": True}
            packets.put_nowait(packet(2))  # Already granted before the reset arrives.
            reset = dict(
                type="source.reset", epoch=0, presentationId="reset-p", sdkEpoch=1
            )
            control.put_nowait(json.dumps(reset))
            control.put_nowait(
                json.dumps(
                    dict(type="media.config", epoch=0, mediaGeneration=2, config=config)
                )
            )
            await asyncio.wait_for(view._wait_or_error(renewed.wait()), 1)
            assert view.ui_events == [reset]
            assert view.ui_bytes == len(canonical(reset))
            assert view.area_transition is None and not view.area_frames
            assert pending.done() and isinstance(pending.exception(), ValueError)
            packets.put_nowait(packet(3, 2))
            second = await asyncio.wait_for(anext(iterator), 1)
            assert second.index == 1
            assert second.provenance["cli_frame_header"]["source"]["sdkEpoch"] == 1
            assert second.provenance["capture_pts_us"] == 2000
            assert second.states == {"ui_events": [reset]}
            assert second.observed_input["receipts"] == []
            assert [v for v in grants if v["type"] == "ui.resync"] == [
                {"type": "ui.resync", "epoch": 0}
            ]
            assert [
                (v["mediaGeneration"], v["frames"])
                for v in grants
                if v["type"] == "media.credit"
            ] == [(1, 2), (2, 2)]
        finally:
            await iterator.aclose()
            view.control_task.cancel()
            await asyncio.gather(view.control_task, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize(
    "fault",
    [
        "no-reset",
        "skipped-generation",
        "pixel-config",
        "boolean-generation",
        "stale-sdk",
        "same-sdk",
        "boolean-sdk",
        "extra-field",
        "wrong-epoch",
    ],
)
def test_native_reset_keeps_source_and_configuration_fences(fault):
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import NativeCLIView

    async def run():
        spec, descriptor, _ = native_frame()
        descriptor["media"] = {"kind": "pixels"}
        view = NativeCLIView(
            SimpleNamespace(descriptor=descriptor, spec=spec, prepared_delivery=None)
        )
        config = dict(
            kind="pixels",
            pixelFormat="rgb24",
            widthPixels=32,
            heightPixels=32,
            rowStrideBytes=96,
            colorSpace="srgb",
            fps=24,
        )
        view.config = dict(
            type="media.config", epoch=0, mediaGeneration=1, config=config
        )
        view.source_identity = ("p", 3)
        reset = dict(type="source.reset", epoch=0, presentationId="next", sdkEpoch=4)
        next_config = dict(
            type="media.config", epoch=0, mediaGeneration=2, config=dict(config)
        )
        if fault == "skipped-generation":
            next_config["mediaGeneration"] = 3
        elif fault == "boolean-generation":
            next_config["mediaGeneration"] = True
        elif fault == "pixel-config":
            next_config["config"]["fps"] = 25
        elif fault == "stale-sdk":
            reset["sdkEpoch"] = 2
        elif fault == "same-sdk":
            reset["sdkEpoch"] = 3
        elif fault == "boolean-sdk":
            reset["sdkEpoch"] = True
        elif fault == "extra-field":
            reset["extra"] = True
        elif fault == "wrong-epoch":
            reset["epoch"] = 1

        async def messages():
            if fault != "no-reset":
                yield json.dumps(reset)
            yield json.dumps(next_config)

        async def consume_ui(_):
            pass

        view.renderer = SimpleNamespace(consume_ui=consume_ui)

        class Control:
            def __aiter__(self):
                return messages()

            async def send(self, _):
                pass

        view.control = Control()
        with pytest.raises(ValueError):
            await view._control()

    asyncio.run(run())


@pytest.mark.parametrize(
    "variant",
    [
        "exact",
        "same-transport",
        "ui-changed",
        "tick-changed",
        "source-changed",
        "backwards",
        "twice",
    ],
)
def test_native_reconnect_accepts_only_one_exact_retained_bootstrap(variant):
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import NativeCLIView

    async def run():
        spec, descriptor, first_packet = native_frame()
        packets, credits, mapped = asyncio.Queue(), [], []
        bound = asyncio.Event()
        bound.set()

        async def send(raw):
            credits.append(json.loads(raw))

        async def frame(header):
            mapped.append(header["outputFrameId"])
            return {"mapped": True}

        view = NativeCLIView(
            SimpleNamespace(descriptor=descriptor, spec=spec, prepared_delivery=None)
        )
        view.control = SimpleNamespace(send=send)
        view.control_task = asyncio.get_running_loop().create_future()
        view.media = SimpleNamespace(recv=packets.get)
        view.model_context = SimpleNamespace(
            suspended=False, bound=bound, check=lambda: None, frame=frame
        )
        view.config = {"mediaGeneration": 1}
        view.configured.set()
        packets.put_nowait(wire(first_packet, spec))
        iterator = view.frames()
        try:
            first = await anext(iterator)
            assert first.index == 0 and len(credits) == 1
            # Downstream may retain/mutate its DTO; replay comparison owns a copy.
            first.provenance["cli_frame_header"]["uiRevision"] = 999
            if variant != "same-transport":
                view.transport_generation += 1
            replay = copy.deepcopy(first_packet)
            if variant == "ui-changed":
                replay["input"]["uiRevision"] += 1
            elif variant == "tick-changed":
                replay["input"]["source"]["simulationTick"] += 1
            elif variant == "source-changed":
                replay["input"]["source"]["presentationId"] = "different"
            elif variant == "backwards":
                replay["input"]["outputFrameId"] = 0
                replay["input"]["source"]["sourceTimeUs"] = 500
            packets.put_nowait(wire(replay, spec))
            if variant == "twice":
                packets.put_nowait(wire(replay, spec))
            fresh = copy.deepcopy(first_packet)
            fresh["input"].update(
                outputFrameId=2, outputPtsUs=2000, uiCompleteThroughUs=2000
            )
            fresh["input"]["source"].update(sourceFrameId=11, sourceTimeUs=2000)
            packets.put_nowait(wire(fresh, spec))
            snapshot = {"type": "ui.snapshot", "snapshot": {"fresh": True}}
            view.ui_events = [snapshot]
            if variant == "exact":
                second = await asyncio.wait_for(anext(iterator), 1)
                assert second.index == 1
                assert second.provenance["cli_frame_header"] == fresh["input"]
                assert second.provenance["capture_pts_us"] == 1000
                assert second.states == {"ui_events": [snapshot]}
                assert mapped == [1, 2]
                # Old held-frame return grants nothing; new window and consumed
                # retained bootstrap each replenish only their actual ownership.
                assert [value["frames"] for value in credits] == [2, 2, 1]
            else:
                with pytest.raises(
                    ValueError, match="source generation changed|clock moved backwards"
                ):
                    await asyncio.wait_for(anext(iterator), 1)
        finally:
            await iterator.aclose()
            view.control_task.cancel()

    asyncio.run(run())


def test_native_held_frame_does_not_return_credit_to_replacement_transport():
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import NativeCLIView

    async def run():
        spec, descriptor, packet = native_frame()
        packets, credits = asyncio.Queue(), []

        async def send(raw):
            credits.append(json.loads(raw))

        view = NativeCLIView(
            SimpleNamespace(descriptor=descriptor, spec=spec, prepared_delivery=None)
        )
        view.control = SimpleNamespace(send=send)
        view.control_task = asyncio.get_running_loop().create_future()
        view.media = SimpleNamespace(recv=packets.get)
        view.config = {"mediaGeneration": 1}
        view.configured.set()
        packets.put_nowait(wire(packet, spec))
        iterator = view.frames()
        try:
            await anext(iterator)
            view.transport_generation += 1
            fresh = copy.deepcopy(packet)
            fresh["input"].update(
                outputFrameId=2, outputPtsUs=2000, uiCompleteThroughUs=2000
            )
            fresh["input"]["source"].update(sourceFrameId=11, sourceTimeUs=2000)
            packets.put_nowait(wire(fresh, spec))
            await anext(iterator)
            assert [value["frames"] for value in credits] == [2, 2]
        finally:
            await iterator.aclose()
            view.control_task.cancel()

    asyncio.run(run())
