#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Test Master: connect workers and reuse the production source/prompt sender.

Worker-group mode sends to Encoder and receives RGB from Decoder independently.
--direct-ws preserves component-only testing. No GPU is needed on the client. output.moments contains contiguous tensor bytes;
summary.json records byte offsets, dtype, shape, source coverage and SHA256 for
each slice. Qwen tensors are saved separately with their event identities.
This is not RGB and no video metrics are inferred from it.
"""

import asyncio
import hashlib
import json
import math
import subprocess
from pathlib import Path


class MomentsSink:
    filename = "output.moments"
    message_type = "source.moments"
    output_kind = "source_moments"
    tensor_format = "vae_posterior_moments"

    def __init__(self):
        self.chunks = []
        self.latents = self.frames = self.offset = 0
        self.signature = None
        self.conditions = []

    def negotiate(self, ready):
        if ready.get("output_kind") != self.output_kind:
            raise ValueError(
                "The endpoint did not negotiate the requested tensor output"
            )
        self.session_id = ready["session_id"]
        self.generation_id = ready["generation_id"]
        self.expects_qwen = (
            self.output_kind == "source_moments"
            and "qwen" in ready.get("encoded_components", [])
        )

    async def receive_conditioning(self, message, file):
        tensors = message.get("tensors", {})
        event = message.get("event_id")
        previous = self.conditions[-1]["event_id"] if self.conditions else None
        if (
            not self.expects_qwen
            or message.get("session_id") != self.session_id
            or message.get("generation_id") != self.generation_id
            or set(tensors) != {"context", "row_role"}
            or (event is not None and (type(event) is not int or event < 1))
            or (self.conditions and (event is None or event <= (previous or 0)))
        ):
            raise ValueError("Invalid conditioning identity or tensor manifest")
        receipt = dict(event_id=event, prompt_version=message.get("prompt_version"))
        receipt["tensors"] = {}
        for name, tensor in tensors.items():
            shape, dtype, payload = tensor["shape"], tensor["dtype"], tensor["payload"]
            sizes = {"bfloat16": 2, "float16": 2, "float32": 4, "int64": 8}
            if (
                len(shape) != (2 if name == "context" else 1)
                or any(type(v) is not int or v < 1 for v in shape)
                or dtype not in sizes
                or (name == "row_role" and dtype != "int64")
                or (name == "context" and dtype == "int64")
                or len(payload) != math.prod(shape) * sizes[dtype]
                or shape[0] != tensors["context"]["shape"][0]
            ):
                raise ValueError("Invalid conditioning tensor geometry")
        # Validate the whole result before persisting either tensor.
        for name, tensor in tensors.items():
            payload = tensor["payload"]
            path = (
                Path(file.name).parent / f"condition-{len(self.conditions)}-{name}.bin"
            )
            written = await asyncio.to_thread(path.write_bytes, payload)
            if written != len(payload):
                raise OSError("Incomplete conditioning write")
            receipt["tensors"][name] = dict(
                file=path.name,
                dtype=tensor["dtype"],
                shape=tensor["shape"],
                bytes=written,
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        self.conditions.append(receipt)
        return receipt

    def frame_end(self, message):
        return message["source_dependency_end"]

    async def receive(self, message, file, *, accepted_frames):
        shape, payload = message["shape"], message["payload"]
        item_size = {"bfloat16": 2, "float16": 2, "float32": 4}.get(message["dtype"])
        if (
            message.get("session_id") != self.session_id
            or message.get("generation_id") != self.generation_id
            or message.get("format") != self.tensor_format
            or len(shape) != 5
            or any(type(v) is not int or v < 1 for v in shape)
            or shape[0] != 1
            or item_size is None
            or len(payload) != math.prod(shape) * item_size
            or message["latent_start"] != self.latents
            or message["latent_end"] != self.latents + shape[2]
            or not self.frames
            < self.frame_end(message)
            <= message["source_dependency_end"]
            <= accepted_frames
        ):
            raise ValueError("Invalid source moments payload or sequence")
        frame_end = self.frame_end(message)
        signature = (message["dtype"], *shape[:2], *shape[3:])
        if self.signature is not None and signature != self.signature:
            raise ValueError("Source moments geometry changed")
        self.signature = signature
        written = await asyncio.to_thread(file.write, payload)
        if written != len(payload):
            raise OSError("Incomplete source moments write")
        self.chunks.append(
            dict(
                **{k: v for k, v in message.items() if k not in ("payload", "type")},
                byte_offset=self.offset,
                bytes=written,
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
        self.offset += written
        self.latents, self.frames = (
            message["latent_end"],
            frame_end,
        )
        return self.frames

    def complete(self, message, state):
        if (
            not state["ended"]
            or (self.expects_qwen and not self.conditions)
            or state["completed"]
            or message.get("output_kind") != self.output_kind
            or message.get("session_id") != self.session_id
            or message.get("generation_id") != self.generation_id
            or message.get("source_frames") != self.frames
            or message.get("latent_frames") != self.latents
            or any(
                state[k] != self.frames
                for k in ("submitted", "accepted", "consumed", "received")
            )
        ):
            raise ValueError("The completed source encoding session is incomplete")

    def summary(self):
        return dict(
            output_kind=self.output_kind,
            latent_frames=self.latents,
            chunks=self.chunks,
            conditions=self.conditions,
        )


class DiTLatentSink(MomentsSink):
    """Persist generated latents with the prompt identity actually used by DiT."""

    filename = "output.latents"
    message_type = "dit.latents"
    output_kind = tensor_format = "dit_latents"

    def frame_end(self, message):
        event = message.get("event_id")
        previous = self.chunks[-1].get("event_id") if self.chunks else None
        if (
            message.get("output_frame_start") != self.frames
            or type(message.get("num_frames")) is not int
            or message["num_frames"] < 1
            or (event is not None and (type(event) is not int or event < 0))
            or (previous is not None and (event is None or event < previous))
        ):
            raise ValueError("Invalid DiT output coverage or prompt identity")
        return self.frames + message["num_frames"]


async def run_test(args, run_client):
    """Keep all media I/O and two-thread streaming in the existing client."""
    sink = (
        None
        if args.output_kind == "rgb"
        else DiTLatentSink()
        if args.output_kind == "dit_latents"
        else MomentsSink()
    )
    if args.direct_ws:
        return await run_client(args, output_sink=sink)
    from worker_group_master import from_args

    async with from_args(args) as group:
        args.url, args.decoder_url = group.input_ws, group.output_ws
        result = await run_client(args, output_sink=sink)
        result["worker_group"] = group.summary()
    (Path(args.output) / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    from continuous_source_client import parser, run
    from worker_group_master import add_arguments

    cli = parser()
    cli.description = __doc__
    add_arguments(cli)
    cli.add_argument(
        "--output-kind",
        choices=["source_moments", "dit_latents", "rgb"],
        default="rgb",
    )
    args = cli.parse_args()
    if args.profile:
        cli.error("Profiling is not implemented by this endpoint yet")
    if min(args.width, args.height, *args.batch_pattern) < 1:
        cli.error("Dimensions and batch lengths must be positive")
    if not args.direct_ws and args.output_kind != "rgb":
        cli.error(
            "Worker group testing requires --output-kind rgb; use --direct-ws for component-only tests"
        )
    if args.direct_ws and (
        args.worker_launch_config
        or any(getattr(args, role + "_http") for role in ("encoder", "dit", "decoder"))
    ):
        cli.error("--direct-ws cannot be combined with worker lifecycle arguments")
    if min(args.worker_startup_timeout, args.worker_control_timeout) <= 0:
        cli.error("Worker timeouts must be positive")
    result = asyncio.run(run_test(args, run))
    if args.output_kind == "rgb":
        output = Path(args.output)
        video = output / "output.mp4"
        subprocess.run(
            [
                args.ffmpeg,
                "-v",
                "error",
                "-nostdin",
                "-n",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{args.width}x{args.height}",
                "-framerate",
                "24",
                "-i",
                str(output / "output.rgb"),
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(video),
            ],
            check=True,
        )
        result["video"] = str(video)
        (output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
