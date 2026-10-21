# SPDX-License-Identifier: Apache-2.0
"""Applicable pre-RPC regressions retained under the approved v6 redesign."""

import importlib

import pytest

from sglang.multimodal_gen.runtime.realtime.source_transport import (
    PROTOCOL_OVERHEAD_BYTES,
)

PREFIX = "sglang.multimodal_gen.runtime.entrypoints.openai.realtime."


@pytest.mark.asyncio
async def test_ws_reserves_before_first_byte_and_releases_on_receive_failure():
    module = importlib.import_module(PREFIX + "conditioning")
    receive = getattr(module, "receive_conditioning_message", None)
    assert callable(receive), "External API must reserve transport bytes before receive"
    from types import SimpleNamespace

    reserved = []

    class Reservation:
        def release(self):
            reserved.clear()

    class Resources:
        def reserve_api_message(self, size):
            reserved.append(size)
            return Reservation()

    class Socket:
        async def receive_bytes(self):
            assert reserved == [PROTOCOL_OVERHEAD_BYTES + 1], (
                "Body was read before admission"
            )
            raise OSError("socket ended")

    session = SimpleNamespace(conditioning_resources=Resources())
    args = SimpleNamespace(
        zing_conditioning_mode="remote",
        realtime_max_message_bytes=PROTOCOL_OVERHEAD_BYTES + 1,
    )
    with pytest.raises(OSError):
        await receive(Socket(), session, args)
    assert not reserved


@pytest.mark.asyncio
async def test_ws_busy_rejects_without_reading_body():
    module = importlib.import_module(PREFIX + "conditioning")
    receive = getattr(module, "receive_conditioning_message", None)
    assert callable(receive), "External API must reject admission before reading body"
    from types import SimpleNamespace

    class Resources:
        def reserve_api_message(self, size):
            raise RuntimeError("busy")

    class Socket:
        async def receive_bytes(self):
            pytest.fail("Rejected message was read")

    session = SimpleNamespace(conditioning_resources=Resources())
    args = SimpleNamespace(
        zing_conditioning_mode="remote",
        realtime_max_message_bytes=PROTOCOL_OVERHEAD_BYTES + 1,
    )
    with pytest.raises(RuntimeError, match="busy"):
        await receive(Socket(), session, args)


@pytest.mark.asyncio
async def test_ws_admission_failure_has_friendly_code_and_no_private_detail():
    from types import SimpleNamespace

    import msgspec.msgpack

    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError

    module = importlib.import_module(PREFIX + "conditioning")

    class Resources:
        def reserve_api_message(self, size):
            raise ConditionError("CONDITION_BUSY", "secret internal server path")

    class Socket:
        def __init__(self):
            self.messages = []

        async def receive_bytes(self):
            pytest.fail("busy must reject before body")

        async def send_bytes(self, raw):
            self.messages.append(msgspec.msgpack.decode(raw))

    socket = Socket()
    session = SimpleNamespace(conditioning_resources=Resources())
    args = SimpleNamespace(
        zing_conditioning_mode="remote",
        realtime_max_message_bytes=PROTOCOL_OVERHEAD_BYTES + 1,
    )
    try:
        messages = [
            message
            async for message in module.iter_conditioning_messages(
                socket, session, args
            )
        ]
    except ConditionError:
        pytest.fail("Admission error must be returned through the wire contract")
    assert messages == []
    assert socket.messages == [
        dict(
            type="error",
            code="CONDITION_BUSY",
            content="The service is busy. Please try again shortly.",
        )
    ]


def test_media_ram_reservation_is_retained_until_dispatch_completion():
    from types import SimpleNamespace

    module = importlib.import_module(PREFIX + "conditioning")
    reserve = getattr(module, "reserve_media_memory", None)
    assert callable(reserve), "Encoded ingress bytes need a persistent RAM reservation"

    class Ledger:
        def reserve(self, tenant, **charges):
            assert (
                tenant == "trusted"
                and charges["ram"] == 2048
                and charges["inflight"] == 2048
            )
            return object()

    session = SimpleNamespace(
        conditioning_resources=SimpleNamespace(tenant="trusted", ledger=Ledger())
    )
    reservation = reserve(session, 2048)
    assert session._api_media_reservations == [reservation]


def test_release_media_reservations_preserves_other_uploads():
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.conditioning.resources import AdmissionLedger

    module = importlib.import_module(PREFIX + "conditioning")
    ledger = AdmissionLedger({"ram": (4096, 4096), "inflight": (4096, 4096)})
    session = SimpleNamespace(
        conditioning_resources=SimpleNamespace(tenant="trusted", ledger=ledger)
    )
    initial = module.reserve_media_memory(session, 1024)
    update = module.reserve_media_memory(session, 2048)

    module.release_media_reservations(session, [update])
    assert update.released and not initial.released
    assert session._api_media_reservations == [initial]
    # Released update credit is reusable while initial media remains charged.
    replacement = module.reserve_media_memory(session, 3072)
    module.release_api_media(session)
    assert initial.released and replacement.released
    assert session._api_media_reservations == []
