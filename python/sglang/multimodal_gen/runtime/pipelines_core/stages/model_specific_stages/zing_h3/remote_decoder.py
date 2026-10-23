# SPDX-License-Identifier: Apache-2.0
"""DiT-side session for a dedicated Lynn decoder service on another GPU group.

The DiT leader hands normalized latents to the decoder host and receives a
shared-memory reference per dispatch instead of pixels. Only identity, frame
boundaries and that reference travel back. D2 append waits for the previous
tile job before bridge transfer; publisher capacity can also apply backpressure.
"""

import uuid
from datetime import timedelta
from typing import Any

import msgspec
import torch
import torch.distributed as dist
import zmq

from sglang.multimodal_gen.runtime.realtime.zing_contracts import PixelRange

from .decoder_transport import bridge_group, send_latent
from .source_decoder import check_latent_append


class RemoteDecodedSlice(msgspec.Struct, frozen=True):
    """Decoded boundary whose pixels the decoder host published itself."""

    start: int
    end: int
    source_dependency_end: int
    provenance: Any
    reference: dict | None
    frames: int
    transport_metadata: dict | None = None
    values: None = None


class RemoteLynnDecoder:
    """Bounded source provenance around a decoder service on its own GPU group.

    Latents are sent as they are appended; one shared-memory reference is
    reserved per dispatch. D2 transfer ordering and publisher capacity can
    backpressure the scheduler. EOS finishes the stream; dispose releases it.
    """

    def __init__(self, codecs, *, capacity_latents, provenance_for, config):
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            get_world_group,
        )

        module = codecs.modules[codecs.decoder.module_key]
        self.protocol = module.protocol
        self.group = get_world_group()
        self.leader = self.group.rank_in_group == 0
        self.capacity_latents = capacity_latents
        self.provenance_for = provenance_for
        self.port = config["port"]
        if type(self.port) is not int or not 1024 <= self.port <= 65535:
            raise ValueError("Invalid decoder service port")
        self.timeout = config.get("timeout_seconds", 120)
        if not 1 <= self.timeout <= 600:
            raise ValueError("Invalid decoder service timeout")
        self.token = config.get("session_id") or uuid.uuid4().hex
        if not isinstance(self.token, str) or not self.token:
            raise ValueError(
                "Remote decoder session identity must be a nonempty string"
            )
        self.accepted_latents = self.emitted_latents = self.emitted_frames = 0
        self.dependency = self.valid_frame_end = 0
        self.total_frames = None
        self.closed = False
        self.drained = False
        self.failure = None
        self.socket = self.context = self.bridge = self.store = None
        self.channel = None
        self.request_id = self.chunk_index = self.generation_id = None
        self._connect()

    @property
    def buffered_latents(self):
        return self.accepted_latents - self.emitted_latents

    @property
    def valid_frames(self):
        return self.valid_frame_end

    @property
    def ready(self):
        # A finished stream stays ready until one last pop, so an EOS dispatch
        # without new latents still surfaces a rejected finish.
        return not self.closed and (
            self.buffered_latents > 0
            or self.valid_frame_end > self.emitted_frames
            or (self.total_frames is not None and not self.drained)
        )

    def bind_dispatch(self, *, request_id, chunk_index, generation_id):
        """Bind the frame identity the shared-memory reference must carry."""
        self.request_id = request_id
        self.chunk_index = chunk_index
        self.generation_id = generation_id

    def _failure_message(self):
        return "Remote decoder failed"

    def _connect(self):
        if not self.leader:
            return
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout * 1000)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout * 1000)
        socket.connect(f"tcp://127.0.0.1:{self.port}")
        self.context, self.socket = context, socket
        try:
            store = dist.TCPStore(
                "127.0.0.1",
                0,
                2,
                True,
                timeout=timedelta(seconds=self.timeout),
                wait_for_workers=False,
            )
            self.store = store
            self.channel = self._request(
                "open", bridge_port=store.port, capacity_latents=self.capacity_latents
            )["channel"]
            self.bridge = bridge_group(store, 0, timeout_seconds=self.timeout)
        except BaseException as error:
            # Reported by the first pop on every DiT rank, like a later failure.
            self.failure = error

    def _request(self, op, **fields):
        if self.failure is not None:
            raise RuntimeError(self._failure_message()) from self.failure
        # open proves the session once; every later command carries the channel
        # that open returned, and abort keeps session so it can release an owner
        # whose channel this client no longer holds.
        if op in ("open", "abort"):
            identity = dict(session=self.token)
        else:
            identity = dict(channel=self.channel)
        try:
            self.socket.send_json(dict(op=op, **identity, **fields))
            reply = self.socket.recv_json()
        except BaseException as error:
            self.failure = error
            raise RuntimeError(self._failure_message()) from error
        if not isinstance(reply, dict) or not reply.get("ok"):
            raise RuntimeError(self._failure_message())
        return reply

    def append(self, latent, *, latent_start, source_dependency_end, valid_frame_end):
        if self.closed or self.total_frames is not None:
            raise ValueError("Decoder stream is closed")
        count = check_latent_append(
            latent=latent,
            latent_start=latent_start,
            source_dependency_end=source_dependency_end,
            valid_frame_end=valid_frame_end,
            accepted_latents=self.accepted_latents,
            valid_frame_end_so_far=self.valid_frame_end,
            protocol=self.protocol,
            capacity_latents=self.capacity_latents,
            buffered_latents=self.buffered_latents,
        )
        self.accepted_latents = count
        self.valid_frame_end = valid_frame_end
        self.dependency = max(self.dependency, source_dependency_end)
        if not self.leader or self.failure is not None:
            return
        # A leader-only failure is reported by the next pop on every DiT rank;
        # raising here alone would strand the peer in its next SP collective.
        try:
            self.socket.send_json(
                dict(
                    op="append",
                    channel=self.channel,
                    shape=[int(n) for n in latent.shape],
                    dtype=str(latent.dtype).split(".")[-1],
                    latent_start=latent_start,
                    valid_frame_end=valid_frame_end,
                    source_dependency_end=source_dependency_end,
                )
            )
            # The decoder allocates from this metadata before posting its receive.
            send_latent(self.bridge, latent, peer=1)
            reply = self.socket.recv_json()
            if not isinstance(reply, dict) or not reply.get("ok"):
                raise RuntimeError("Decoder service rejected the latent")
        except BaseException as error:
            self.failure = error

    @torch.no_grad()
    def pop(self):
        if not self.ready:
            return None
        end = self.valid_frame_end
        pixels = (
            PixelRange(self.emitted_frames, end) if end > self.emitted_frames else None
        )
        provenance = self.provenance_for(pixels, self.dependency) if pixels else None
        if pixels and (
            provenance.pixels != pixels
            or provenance.source_dependency_end != self.dependency
        ):
            raise ValueError("Provenance must match decoded pixels and dependency")
        frames = end - self.emitted_frames
        reference = metadata = None
        if self.leader and self.failure is None:
            try:
                reply = self._request(
                    "output",
                    frames=frames,
                    request_id=self.request_id,
                    chunk_index=self.chunk_index,
                    generation_id=self.generation_id,
                )
                reference, metadata = reply.get("reference"), reply.get("metadata")
                if bool(frames) != isinstance(reference, dict):
                    raise ValueError("Remote decoder output reference is invalid")
            except BaseException as error:
                self.failure = error
        # Only the failure flag crosses ranks: a reference reaching a non-output
        # rank would be cancelled there as an unsent result.
        status = [self.failure is not None]
        dist.broadcast_object_list(
            status, src=self.group.ranks[0], group=self.group.cpu_group
        )
        if status[0]:
            self.closed = True
            raise RuntimeError(self._failure_message()) from self.failure
        self.emitted_latents = self.accepted_latents
        self.emitted_frames = end
        if self.total_frames is not None:
            self.drained = True
        return RemoteDecodedSlice(
            start=end - frames,
            end=end,
            source_dependency_end=self.dependency,
            provenance=provenance,
            reference=reference,
            frames=frames,
            transport_metadata=metadata,
        )

    def finish(self, *, total_frames):
        if self.closed or self.total_frames is not None:
            raise ValueError("Decoder stream is already closed")
        if type(total_frames) is not int or total_frames < max(1, self.emitted_frames):
            raise ValueError("Invalid final video frame count")
        if not self.accepted_latents or total_frames != self.valid_frame_end:
            raise ValueError("Final video frame count does not match latent count")
        self.total_frames = total_frames
        if self.leader and self.failure is None:
            try:
                self._request("finish", total_frames=total_frames)
            except BaseException as error:
                self.failure = error

    def dispose(self):
        if self.leader and self.context is not None:
            # A receive timeout leaves REQ waiting for its old reply. A fresh
            # channel can still release this owner after that reply is sent.
            self._close_socket()
            socket = self.context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, min(self.timeout, 1) * 1000)
            socket.setsockopt(zmq.SNDTIMEO, min(self.timeout, 1) * 1000)
            socket.connect(f"tcp://127.0.0.1:{self.port}")
            try:
                socket.send_json(dict(op="abort", session=self.token))
                socket.recv_json()
            except zmq.ZMQError:
                pass
            finally:
                socket.close()
        self.closed = True
        self._close_channel()
        self.provenance_for = None

    def _close_socket(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def _close_channel(self):
        self._close_socket()
        if self.context is not None:
            self.context.term()
        self.context = self.bridge = self.store = None


def create_decoder(
    codecs, *, policy, capacity_latents, provenance_for, session_id=None
):
    """Select the local stream or the dedicated service from trusted profile."""
    if not policy or "remote" not in policy:
        return codecs.decoder.source_decoder(
            codecs,
            capacity_latents=capacity_latents,
            provenance_for=provenance_for,
        )
    if codecs.decoder.module_key != "lynn_decoder":
        raise ValueError("Remote source decoding requires the Lynn decoder")
    config = dict(policy["remote"])
    if session_id is not None:
        config["session_id"] = session_id
    return RemoteLynnDecoder(
        codecs,
        capacity_latents=capacity_latents,
        provenance_for=provenance_for,
        config=config,
    )
