# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from websockets.exceptions import ConnectionClosedOK
from websockets.frames import Close

from sglang.multimodal_gen.runtime.realtime.startup_warmup import (
    build_warmup_request,
    create_reference_frame,
    parse_size,
    wait_for_first_frame,
)


class _GenerationCompleteWebSocket:
    async def recv(self):
        raise ConnectionClosedOK(Close(1000, "generation complete"), None)


def test_startup_warmup_rejects_empty_generation_complete_by_default():
    with pytest.raises(ConnectionClosedOK):
        asyncio.run(wait_for_first_frame(_GenerationCompleteWebSocket(), timeout_s=1.0))


def test_startup_warmup_allows_empty_generation_complete_for_async_denoiser():
    result = asyncio.run(
        wait_for_first_frame(
            _GenerationCompleteWebSocket(),
            timeout_s=1.0,
            allow_empty_complete=True,
        )
    )

    assert result == {"chunk_index": 0, "empty_complete": True}


def test_startup_warmup_uses_requested_resolution():
    frame = create_reference_frame("1280x704")
    request = build_warmup_request(
        model="minwm",
        first_frame=frame,
        trace_id="startup-warmup-test",
        size="1280x704",
    )

    assert request["size"] == "1280x704"
    assert request["first_frame"] == frame


def test_startup_warmup_rejects_invalid_resolution():
    with pytest.raises(ValueError, match="WIDTHxHEIGHT"):
        parse_size("704*1280")
