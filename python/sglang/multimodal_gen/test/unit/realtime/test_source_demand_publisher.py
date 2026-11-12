# SPDX-License-Identifier: Apache-2.0
"""Bounded demand publication under decoder/network backpressure, without GPUs."""

import asyncio
from types import SimpleNamespace

import msgspec
import pytest

from sglang.multimodal_gen.runtime.realtime.source_demand import SourceDemandPublisher
from sglang.multimodal_gen.runtime.realtime.websocket_transport import (
    _LockedRealtimeWebSocket,
)


def test_one_pending_snapshot_is_built_after_lock_acquisition():
    async def run():
        wire, snapshots, observations = [], [], []
        state = SimpleNamespace(accepted=0, dispatched=0, sequence=0)
        sent, finish, ready = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class Socket:
            async def send_bytes(self, payload):
                wire.append(msgspec.msgpack.decode(payload))
                sent.set()

        class Adapter:
            def observe_source_input_wait(self, *, waiting_for_input):
                observations.append(waiting_for_input)

            def source_demand_message(self, session, *, waiting_for_input):
                state.sequence += 1
                value = dict(
                    sequence=state.sequence,
                    accepted=state.accepted,
                    dispatched=state.dispatched,
                    queued=state.accepted - state.dispatched,
                )
                snapshots.append(value)
                return value

        ws = _LockedRealtimeWebSocket(Socket())
        publisher = SourceDemandPublisher(Adapter(), object(), ws)

        async def generation():
            publisher.publish(waiting=True)
            ready.set()
            await finish.wait()

        async with ws.send_group():
            owner = asyncio.create_task(publisher.run(generation()))
            await ready.wait()
            await asyncio.sleep(0)
            sender = publisher.task
            for i in range(1, 1001):
                state.accepted, state.dispatched = i, i - 1
                publisher.publish(waiting=False)
            assert publisher.task is sender
            assert not snapshots and not wire
            assert observations[0] and not observations[-1]
        await asyncio.wait_for(sent.wait(), 1)
        finish.set()
        await owner
        assert wire == [dict(sequence=1, accepted=1000, dispatched=999, queued=1)]
        assert publisher.closed and publisher.task.done() and not publisher.pending

    asyncio.run(run())


def test_writer_failure_cancels_generation_blocked_on_input():
    async def run():
        entered, stopped = asyncio.Event(), asyncio.Event()

        class Socket:
            async def send_bytes(self, payload):
                await entered.wait()
                raise ConnectionError("writer failed")

        adapter = SimpleNamespace(source_demand_message=lambda *a, **k: {"sequence": 1})
        publisher = SourceDemandPublisher(
            adapter, object(), _LockedRealtimeWebSocket(Socket())
        )

        async def generation():
            try:
                publisher.publish(waiting=True)
                entered.set()
                await asyncio.Event().wait()
            finally:
                stopped.set()

        with pytest.raises(ConnectionError, match="writer failed"):
            await asyncio.wait_for(publisher.run(generation()), 1)
        assert stopped.is_set() and publisher.closed and publisher.task.done()

    asyncio.run(run())


def test_cancellation_discards_blocked_demand_without_late_send():
    async def run():
        messages, entered = [], asyncio.Event()

        class Socket:
            async def send_bytes(self, payload):
                messages.append(payload)

        ws = _LockedRealtimeWebSocket(Socket())
        publisher = SourceDemandPublisher(
            SimpleNamespace(source_demand_message=lambda *a, **k: {"sequence": 1}),
            object(),
            ws,
        )

        async def generation():
            publisher.publish()
            entered.set()
            await asyncio.Event().wait()

        async with ws.send_group():
            task = asyncio.create_task(publisher.run(generation()))
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        await asyncio.sleep(0)
        assert not messages and publisher.closed and publisher.task.done()
        with pytest.raises(RuntimeError, match="closed"):
            publisher.publish()

    asyncio.run(run())


def test_updates_during_a_slow_write_are_coalesced_and_sent_in_order():
    async def run():
        values, state = [], [0]
        entered, release, done, finish = (asyncio.Event() for _ in range(4))

        class Socket:
            async def send_bytes(self, payload):
                values.append(msgspec.msgpack.decode(payload)["value"])
                if len(values) == 1:
                    entered.set()
                    await release.wait()
                else:
                    done.set()

        adapter = SimpleNamespace(
            source_demand_message=lambda *a, **k: {"value": state[0]}
        )
        publisher = SourceDemandPublisher(
            adapter, object(), _LockedRealtimeWebSocket(Socket())
        )

        async def generation():
            publisher.publish()
            await finish.wait()

        owner = asyncio.create_task(publisher.run(generation()))
        try:
            await entered.wait()
            sender = publisher.task
            for i in range(1, 101):
                state[0] = i
                publisher.publish()
            assert publisher.task is sender and values == [0]
            release.set()
            await asyncio.wait_for(done.wait(), 1)
            finish.set()
            await owner
        finally:
            owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)
        assert values == [0, 100]

    asyncio.run(run())
