"""Negotiate compression only for the RTC metadata/control endpoint."""

from uvicorn.protocols.websockets.websockets_impl import WebSocketProtocol
from websockets.extensions.permessage_deflate import ServerPerMessageDeflateFactory


class WhiteboxWebSocketProtocol(WebSocketProtocol):
    async def process_request(self, path, request_headers):
        # H.264 WS media is already compressed. RTC carries no video on WSS,
        # but repeated UI and frame identities otherwise consume Mbps here.
        # handshake() already holds this connection's list by reference.
        self.available_extensions[:] = (
            [
                ServerPerMessageDeflateFactory(
                    # Keep only this connection's bounded 4 KiB dictionary;
                    # repeated frame identities benefit from cross-message reuse.
                    server_no_context_takeover=False,
                    client_no_context_takeover=True,
                    server_max_window_bits=12,
                    compress_settings={"memLevel": 5, "level": 3},
                )
            ]
            if path.partition("?")[0] == "/v1/webrtc/play"
            else []
        )
        return await super().process_request(path, request_headers)
