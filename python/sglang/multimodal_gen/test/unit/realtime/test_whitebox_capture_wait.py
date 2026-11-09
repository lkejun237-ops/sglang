import asyncio

import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.capture_wait import CaptureWait


def test_hold_interrupts_pending_media_deadline_and_resume_rearms_it():
    async def run():
        state = CaptureWait(frame_seconds=0.04, hold_seconds=0.3)
        frame = asyncio.Queue()
        pending = asyncio.create_task(state.receive(frame.get()))
        try:
            await asyncio.sleep(0.01)
            state.pending("ctx")
            state.held("ctx")
            await asyncio.sleep(0.08)
            assert not pending.done()  # More than two ordinary frame deadlines.
            state.resumed("ctx")
            await asyncio.sleep(0.01)
            await frame.put(b"fresh")
            assert await pending == b"fresh"
            with pytest.raises(ValueError, match="did not become ready"):
                await state.receive(frame.get())
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)

    asyncio.run(run())


def test_unconfirmed_hold_and_expired_hold_remain_bounded():
    async def run():
        for confirm in (False, True):
            state = CaptureWait(frame_seconds=0.01, hold_seconds=0.03)
            state.pending("ctx")
            if confirm:
                state.held("ctx")
            with pytest.raises(ValueError, match="did not become ready"):
                await asyncio.wait_for(state.receive(asyncio.Event().wait()), 0.2)
            if not confirm:
                with pytest.raises(ValueError, match="expired"):
                    state.held("ctx")

    asyncio.run(run())


def test_foreign_receipts_and_repeated_holds_cannot_extend_silence():
    state = CaptureWait(clock=lambda: 100)
    state.pending("ctx")
    original = state.until
    with pytest.raises(ValueError):
        state.pending("ctx")
    with pytest.raises(ValueError):
        state.held("other")
    with pytest.raises(ValueError):
        state.resumed("other")
    assert state.until == original


def test_control_failure_still_ends_a_held_media_read():
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import CLIView

    async def run():
        state = CaptureWait()
        state.pending("ctx")
        state.held("ctx")

        async def failed():
            await asyncio.sleep(0)
            raise ConnectionError("control lost")

        owner = SimpleNamespace(
            control_task=asyncio.create_task(failed()), finish_requested=asyncio.Event()
        )
        with pytest.raises(ConnectionError, match="control lost"):
            await asyncio.wait_for(
                CLIView._wait_or_error(
                    owner, state.receive(asyncio.Event().wait()), timeout=None
                ),
                0.5,
            )

    asyncio.run(run())
