# SPDX-License-Identifier: Apache-2.0
"""CPU fault injection for cleanup ACK handling; no model or CUDA required."""

import queue
import threading

import pytest
from probe_encoder_transport_gpu import dispose_checked, reset_checked
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.remote_encoder import (
    RemoteEncoderSession,
)


@pytest.mark.parametrize("close_fails", [False, True])
def test_disposal_requires_producer_close_ack_even_after_local_state_is_released(
    close_fails,
):
    # Exercise the real dispose/join/drain implementation with a producer that
    # exits exactly as _produce's finally does after a successful/failed close.
    session = object.__new__(RemoteEncoderSession)
    session.stop = threading.Event()
    session.permits = threading.Semaphore(0)
    session.timeout = 1
    session.disposed = False
    session.failure = None
    session.accepted_frames = 34
    session._consumed = 0
    session.signature = (480, 832)
    session.remote_usage = {"retained_bytes": 4096, "temporal_positions": 2}
    session.commands = queue.Queue()
    session.results = queue.Queue()
    session.commands.put(("finish", None, None))
    failure = TimeoutError("Injected missing worker close acknowledgement")

    def producer():
        if not session.stop.wait(2):
            session.failure = RuntimeError("Test producer was not stopped")
        elif close_fails:
            session.failure = failure

    session.thread = threading.Thread(target=producer)
    session.thread.start()
    if close_fails:
        with pytest.raises(
            RuntimeError, match="Remote encoder cleanup failed"
        ) as caught:
            dispose_checked(session)
        assert caught.value.__cause__ is failure
    else:
        dispose_checked(session)
    # Both scenarios free local storage; only the acknowledged scenario passes.
    assert session.disposed and not session.thread.is_alive()
    assert session.state_usage().retained_bytes == 0
    assert session.commands.empty()


@pytest.mark.parametrize("close_fails", [False, True])
def test_reset_cannot_erase_the_old_producer_close_failure(close_fails):
    class ResetSession(RemoteEncoderSession):
        def __init__(self, codecs, *, capacity_frames, config):
            self.codecs, self.capacity_frames, self.config = (
                codecs,
                capacity_frames,
                config,
            )
            self.disposed = False
            self.failure = None
            self.generation = getattr(self, "generation", 0) + 1

        def dispose(self):
            self.disposed = True
            if close_fails:
                self.failure = TimeoutError("Old worker close did not ACK")

        def state_usage(self):
            from types import SimpleNamespace

            return SimpleNamespace(retained_bytes=0)

    session = ResetSession(None, capacity_frames=51, config={})
    session.thread = None
    if close_fails:
        with pytest.raises(RuntimeError, match="remote close failed"):
            reset_checked(session)
        assert session.generation == 1 and session.failure is not None
    else:
        reset_checked(session)
        assert session.generation == 2 and session.failure is None
    assert "dispose" not in vars(session)
