# SPDX-License-Identifier: Apache-2.0
"""Bounded direct-WebSocket RGB transport, shared by deployment and clients.

This reserves transport buffers, not model/codec/KV memory or total process RSS.
All byte limits apply to the complete encoded MessagePack message.
"""

from dataclasses import dataclass
from importlib.metadata import version

from packaging.specifiers import SpecifierSet

from sglang.multimodal_gen.runtime.realtime.zing_contracts import ContractError

DEFAULT_MESSAGE_BYTES = 64 << 20
DEFAULT_HOST_BYTES = 8 << 30
PROTOCOL_OVERHEAD_BYTES = 4096
REFERENCE_BYTES = 8 << 20
RECEIVE_QUEUE_MESSAGES = 1
RECEIVE_ASSEMBLY_MESSAGES = 2
PENDING_BATCHES = 4


class SourceTransportError(ContractError):
    """Only these curated messages and stable codes may cross the wire."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def check(condition, code, message):
    if not condition:
        raise SourceTransportError(code, message)


@dataclass(frozen=True)
class SourceTransportBudget:
    message_bytes: int = DEFAULT_MESSAGE_BYTES
    host_bytes: int = DEFAULT_HOST_BYTES

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.message_bytes <= PROTOCOL_OVERHEAD_BYTES:
            raise ValueError("message_bytes must exceed the protocol overhead")

    def websocket_options(self):
        # The bounded legacy backend depends on the declared dependency range.
        # In particular, newer websockets versions removed its drain lock and
        # can fail when a pong overlaps a large backpressured video write.
        for package, supported in (
            ("uvicorn", ">=0.35.0,<0.40"),
            ("websockets", ">=12.0,<16"),
        ):
            installed = version(package)
            if installed not in SpecifierSet(supported):
                raise RuntimeError(
                    f"Realtime transport requires {package}{supported}; "
                    f"found {installed}. Install the repository dependencies."
                )
        return dict(
            ws="websockets",
            ws_max_size=self.message_bytes,
            ws_max_queue=RECEIVE_QUEUE_MESSAGES,
            ws_per_message_deflate=False,
        )

    @property
    def session_bytes(self):
        # Four pending RGB batches; one queued WS message; two receive buffers
        # for frame assembly/unmasking and the completed prefetch waiting for
        # queue credit; current encoded/decoded input; dispatch/IPC copies (two);
        # output staging/encoding/send
        # (three). Reference bytes are retained by the init request. Object and
        # bounded ledger metadata allowance is one additional MiB per session.
        return (
            (PENDING_BATCHES + RECEIVE_QUEUE_MESSAGES + RECEIVE_ASSEMBLY_MESSAGES + 7)
            * self.message_bytes
            + REFERENCE_BYTES
            + (1 << 20)
        )

    def validate_sessions(self, sessions):
        if type(sessions) is not int or sessions < 1:
            raise ValueError("Transport session count must be a positive integer")
        if sessions * self.session_bytes > self.host_bytes:
            raise ValueError(
                "Realtime transport host budget is too small: "
                f"{sessions} sessions require {sessions * self.session_bytes} bytes; "
                f"configured {self.host_bytes}. Reduce session/message limits or "
                "increase --realtime-transport-host-bytes."
            )

    def source_message_bytes(self, width, height, clip):
        size = clip * width * height * 3 + PROTOCOL_OVERHEAD_BYTES
        check(
            size <= self.message_bytes,
            "source_message_too_large",
            "The video dimensions exceed this service's streaming size limit.",
        )
        return size

    @classmethod
    def from_args(cls, args):
        return cls(
            getattr(args, "realtime_max_message_bytes", DEFAULT_MESSAGE_BYTES),
            getattr(args, "realtime_transport_host_bytes", DEFAULT_HOST_BYTES),
        )


def validate_source_dimensions(width, height, pipeline_config):
    vae = pipeline_config.vae_config.arch_config
    patch = pipeline_config.dit_config.arch_config.patch_size
    grid_h = vae.spatial_compression_ratio * patch[1]
    grid_w = vae.spatial_compression_ratio * patch[2]
    check(
        min(width, height) > 0 and width % grid_w == height % grid_h == 0,
        "source_dimensions_unaligned",
        f"Please use a video width divisible by {grid_w} and height divisible by {grid_h}. No automatic resizing is applied.",
    )
