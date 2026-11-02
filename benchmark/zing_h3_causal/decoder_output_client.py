# SPDX-License-Identifier: Apache-2.0
"""Independent Decoder receive thread for the existing streaming test client."""

import asyncio
import threading
from concurrent.futures import Future


class ThreadCredit:
    """Wake the sender loop from either WS reader without blocking that reader."""

    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.changed = asyncio.Event()
        self.lock = threading.Lock()

    async def wait_for(self, predicate):
        while True:
            with self.lock:
                self.changed.clear()
                if predicate():
                    return
            await self.changed.wait()

    def notify(self):
        self.loop.call_soon_threadsafe(self.changed.set)


def retire_source_boundaries(state, boundaries):
    # Decoder credit can arrive before the independent Encoder ACK. Keep the
    # original packet identity until both channels have confirmed it.
    end = min(state["accepted"], state["consumed"])
    while boundaries and boundaries[0][0] <= end:
        boundaries.popleft()


def consume_source_credit(state, boundaries, message, *, from_decoder):
    end, nbytes = message["source_frame_end"], message["source_bytes_end"]
    if (
        type(end) is not int
        or type(nbytes) is not int
        or not state["consumed"]
        < end
        <= state["submitted" if from_decoder else "accepted"]
        or (end, nbytes) not in [b[:2] for b in boundaries]
    ):
        raise ValueError("Invalid source credit watermark")
    state["consumed"], state["consumed_bytes"] = end, nbytes
    retire_source_boundaries(state, boundaries)


class OutputReceiverThread:
    """Own an output event loop; cancellation always joins its network cleanup."""

    def __init__(self, receive):
        self.receive = receive
        self.stop = threading.Event()
        self.future = Future()
        self.thread = threading.Thread(target=self._run, name="decoder-receiver")
        self.thread.start()

    async def _receive_until_stopped(self):
        async def stop_requested():
            while not self.stop.is_set():
                await asyncio.sleep(0.01)

        work = asyncio.create_task(self.receive())
        stopping = asyncio.create_task(stop_requested())
        try:
            done, _ = await asyncio.wait(
                [work, stopping], return_when=asyncio.FIRST_COMPLETED
            )
            if work in done:
                return await work
            raise asyncio.CancelledError
        finally:
            work.cancel()
            stopping.cancel()
            await asyncio.gather(work, stopping, return_exceptions=True)

    def _run(self):
        try:
            self.future.set_result(asyncio.run(self._receive_until_stopped()))
        except BaseException as exc:
            self.future.set_exception(exc)

    async def wait(self):
        future = asyncio.wrap_future(self.future)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            future.add_done_callback(
                lambda f: f.exception() if not f.cancelled() else None
            )
            raise

    async def close(self):
        self.stop.set()
        await asyncio.to_thread(self.thread.join)
        # Retrieve an exception even if the sender failed first.
        if self.future.done():
            self.future.exception()


def validate_split_completion(state):
    if (
        not state.get("input_completed")
        or not state["completed"]
        or any(
            state[key] != state["submitted"]
            for key in ("accepted", "consumed", "received")
        )
    ):
        raise ValueError(
            "The completed session is missing input receipts or output frames"
        )


async def receive_decoder_output(ready, url, options, receive):
    import msgpack
    import websockets

    async with websockets.connect(url, **options) as ws:
        identity = {k: ready[k] for k in ("session_id", "generation_id")}
        await ws.send(
            msgpack.packb(
                dict(
                    type="output.attach",
                    token=ready["output_attachment"]["token"],
                    **identity,
                ),
                use_bin_type=True,
            )
        )
        response = msgpack.unpackb(await ws.recv(), raw=False)
        if response != dict(type="output.ready", **identity):
            raise ValueError("Decoder output attachment failed")
        await receive(ws, channel="output")
