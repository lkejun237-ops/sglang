# SPDX-License-Identifier: Apache-2.0

"""Opt-in per-connection write bounds for the realtime Gateway."""

import logging
import socket


def bounded_websocket_protocol(send_buffer_bytes: int):
    from uvicorn.protocols.websockets.websockets_impl import WebSocketProtocol

    if send_buffer_bytes < 16384:
        raise ValueError("Gateway socket send buffer must be at least 16384 bytes")

    class BoundedWebSocketProtocol(WebSocketProtocol):
        def connection_made(self, transport):
            sock = transport.get_extra_info("socket")
            if sock is not None:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, send_buffer_bytes)
                option = getattr(socket, "TCP_NOTSENT_LOWAT", None)
                if option is not None:
                    try:
                        sock.setsockopt(socket.IPPROTO_TCP, option, 32768)
                    except OSError:
                        logging.getLogger(__name__).warning(
                            "TCP_NOTSENT_LOWAT is unavailable; SO_SNDBUF remains bounded"
                        )
            super().connection_made(transport)
            transport.set_write_buffer_limits(high=65536, low=16384)

    return BoundedWebSocketProtocol
