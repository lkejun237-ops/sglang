# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from sglang.multimodal_gen.runtime.realtime.polling_task_queue import PollingTaskQueue


def test_default_fifo_drains_backlog_without_new_input():
    async def run():
        processed = []

        async def process(queue):
            item = queue.popleft()
            processed.append(item)
            return item

        queue = PollingTaskQueue(process)
        for item in range(4):
            queue.append(item)
        assert queue.poll_interval_s == 0.005
        for item in range(4):
            assert await queue.run_once() == item
        assert processed == list(range(4))
        assert not queue

    asyncio.run(run())


def test_arranger_can_buffer_split_and_combine_input():
    staged = []

    def arrange(items):
        staged.extend(items)
        tasks = []
        while len(staged) >= 2:
            tasks.append(tuple(staged[:2]))
            del staged[:2]
        return tasks

    queue = PollingTaskQueue(lambda q: None, arrange=arrange)
    queue.append([0])
    assert not queue
    queue.extend([[1, 2, 3, 4], [5]])
    assert list(queue) == [(0, 1), (2, 3), (4, 5)]
    assert not staged


def test_custom_predicate_interval_and_batched_processing(monkeypatch):
    async def run():
        delays, calls = [], []

        async def process(queue):
            calls.append([queue.popleft(), queue.popleft()])
            return calls[-1]

        queue = PollingTaskQueue(
            process, ready=lambda q: len(q) >= 2, poll_interval_s=0.012
        )
        queue.append("first")

        async def tick(delay):
            delays.append(delay)
            assert not calls
            queue.append("second")

        monkeypatch.setattr(asyncio, "sleep", tick)
        assert await queue.run_once() == ["first", "second"]
        assert delays == [0.012]
        assert len(calls) == 1

    asyncio.run(run())


def test_split_preparation_preserves_callback_arguments():
    async def run():
        async def process(queue, batch, *, client):
            assert not queue
            return batch, client

        queue = PollingTaskQueue(process)
        queue.append("input")
        await queue.wait_ready()
        batch = queue.popleft()
        assert await queue.process(batch, client="encoder") == ("input", "encoder")

    asyncio.run(run())


def test_ready_can_drain_internal_model_work_with_empty_queue():
    async def run():
        remaining = [1, 2, 3]

        async def process(queue):
            return remaining.pop(0)

        queue = PollingTaskQueue(process, ready=lambda q: bool(q) or bool(remaining))
        assert [await queue.run_once() for _ in range(3)] == [1, 2, 3]

    asyncio.run(run())


def test_idle_cancellation_does_not_call_processor():
    async def run():
        async def process(queue):
            pytest.fail("Empty queue must not call the model")

        queue = PollingTaskQueue(process)
        task = asyncio.create_task(queue.run_once())
        await asyncio.sleep(0.012)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())


def test_processor_failure_propagates_without_consuming_later_tasks():
    async def run():
        async def process(queue):
            assert queue.popleft() == "fail"
            raise RuntimeError("Model failed")

        queue = PollingTaskQueue(process)
        queue.extend(["fail", "later"])
        with pytest.raises(RuntimeError, match="Model failed"):
            await queue.run_once()
        assert list(queue) == ["later"]

    asyncio.run(run())


@pytest.mark.parametrize("interval", [0, -1, float("inf"), float("nan"), True, "5"])
def test_invalid_interval(interval):
    with pytest.raises(ValueError, match="Polling interval"):
        PollingTaskQueue(lambda q: None, poll_interval_s=interval)
