"""Own a real Worldkit CLI and view for one source benchmark session."""

import asyncio
import time
from uuid import uuid4

import msgspec
from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import cli_room
from sglang.multimodal_gen.runtime.realtime.whitebox.room_startup import (
    FactoryCleanupReceipt,
    startup_cleanup_receipt,
)


class LiveSourceEngine:
    def __init__(self, room, view, startup_ms, capture_fps, retirement):
        self.room, self.view = room, view
        self.startup_ms, self.capture_fps = startup_ms, capture_fps
        self.emitted_frames = 0
        self.control_log = []
        self.closed = False
        self.input_receipts = []
        self.retirement = retirement
        # SourceSession uses this identity for optional RoomRuntime bookkeeping.
        # The standalone CLI view has no RoomRuntime member of its own.
        self.stream_id = retirement["view_id"]

    def __getattr__(self, name):
        return getattr(self.view, name)

    async def ui_asset(self, path):
        return await self.view.renderer.asset(path)

    async def frames(self):
        async for frame in self.view.frames():
            self.emitted_frames += 1
            self.input_receipts.extend(
                (getattr(frame, "observed_input", None) or {}).get("receipts", [])
            )
            yield frame

    async def close(self):
        if not self.closed:
            await self.room.close()
            self.closed = True

    async def retire(self):
        await self.close()
        return dict(self.retirement, state="retired")


def live_source_factory(*, world_id, capture_fps, on_created, identity=None):
    owner_epoch = uuid4().hex
    failed_cleanup = None

    async def create(**kwargs):
        nonlocal failed_cleanup
        started = time.perf_counter()
        retirement = dict(
            source_id=world_id, owner_epoch=owner_epoch, view_id=kwargs["trace_id"]
        )
        try:
            room = await cli_room(
                world_id=world_id,
                spec=kwargs["spec"],
                trace_id=kwargs["trace_id"],
                cli_fps=capture_fps,
            )
        except BaseException as error:
            receipt = startup_cleanup_receipt(error)
            if (
                isinstance(receipt, FactoryCleanupReceipt)
                and receipt.operation_id == kwargs["trace_id"]
            ):
                failed_cleanup = dict(retirement, state="not_created")
            raise
        try:
            if identity is not None and room.descriptor.get("artifact") != identity:
                raise ValueError("Live game publication identity changed")
            view = await room.join(player_id=kwargs["trace_id"], spec=kwargs["spec"])
        except BaseException:
            await room.close()
            failed_cleanup = dict(retirement, state="not_created")
            raise
        engine = LiveSourceEngine(
            room,
            view,
            round((time.perf_counter() - started) * 1000, 3),
            capture_fps or kwargs["spec"].fps,
            retirement,
        )
        on_created(engine)
        return engine

    async def retire_failed_start():
        if failed_cleanup is None:
            raise RuntimeError("Live engine startup cleanup is unconfirmed")
        return dict(failed_cleanup)

    create.retire_failed_start = retire_failed_start
    return create


async def replay_player_inputs(source, events, started):
    """Preserve recorded relative wall times through the production input entry."""
    sent = []
    for sequence, event in enumerate(events):
        await asyncio.sleep(
            max(0, started + event["offset_ms"] / 1000 - time.perf_counter())
        )
        await source.send(
            msgspec.msgpack.encode(
                {
                    "type": "event",
                    "kind": "input",
                    "sequence": sequence,
                    "payload": event["input"],
                }
            )
        )
        sent.append(
            {
                "sequence": sequence,
                "scheduled_ms": event["offset_ms"],
                "sent_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        )
    return sent


async def observe_world_run(
    source,
    engine,
    case,
    segment,
    batches,
    input_task,
    wait_for,
    applied_times,
    init_started,
    target_frames,
    detailed_timing,
):
    """Observe native gameplay-driven conditions through the first three changes."""
    first_rgb = batches[0][0]
    versions = [int(s["prompt_version"]) for s in case["condition_switches"]]

    def completed():
        changes = segment.recording.manifest.get("condition_switches", [])
        seen = {r["prompt_version"] for r in changes}
        return (
            all(
                v in seen and (segment.segment_id, v) in applied_times for v in versions
            )
            and sum(m.get("num_frames", 1) for _, m in batches) >= target_frames
        )

    await wait_for(completed)
    inputs = await input_task
    changes = [
        dict(row)
        for row in segment.recording.manifest["condition_switches"]
        if row["prompt_version"] in versions
    ]
    await source.unbind("SUITE_COMPLETED")
    await source.close()
    if getattr(source.model_binding.lease, "resource_state", None) != "released":
        raise RuntimeError("World session release was not verified")
    if not engine.input_receipts:
        raise RuntimeError("World session has no SDK input receipts")
    return {
        "case_id": case["case_id"],
        "world_id": case["world_id"],
        "recording_id": case["recording_id"],
        "source_kind": "worldkit-live",
        "engine_init_to_first_rgb_ms": round((first_rgb - init_started) * 1000, 3),
        "whitebox_session_ttff_ms": None,
        "engine_startup_ms": engine.startup_ms,
        "condition_switches": changes,
        "detailed_timing_requested": detailed_timing,
        "detailed_timing_enabled": segment.detailed_timing,
        "lease_release_verified": True,
        "player_inputs_sent": inputs,
        "player_input_receipts": engine.input_receipts,
        "received_model_frames": sum(m.get("num_frames", 1) for _, m in batches),
    }
