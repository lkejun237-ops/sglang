# SPDX-License-Identifier: Apache-2.0
"""DiT-side session for a dedicated Lynn decoder service on another GPU group.

The DiT leader hands normalized latents to the decoder host and receives a
shared-memory reference per dispatch instead of pixels. Only identity, frame
boundaries and that reference travel back. D2 append waits for the previous
tile job before bridge transfer; publisher capacity can also apply backpressure.
"""

import atexit
import json
import os
import time
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

import msgspec
import torch
import torch.distributed as dist
import zmq

from sglang.multimodal_gen.runtime.realtime.zing_contracts import PixelRange

from .decoder_transport import bridge_group, send_latent
from .source_decoder import check_latent_append

_PERSISTENT_DECODER_BRIDGES: dict[int, tuple[str, Any, Any]] = {}
_DYNAMIC_DECODER_PORTS: list[int] | None = None


def _decoder_control_rpc(port: int, request: dict, *, timeout: float = 5.0) -> dict:
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    timeout_ms = max(1, int(timeout * 1000))
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.connect(f"tcp://127.0.0.1:{port}")
    try:
        socket.send_json(request)
        return socket.recv_json()
    finally:
        socket.close()
        context.term()


def set_decoder_max_sessions(
    port: int, max_sessions: int, *, timeout: float = 5.0
) -> dict:
    """Dynamically adjust max_sessions on a running decoder_worker."""
    return _decoder_control_rpc(
        port,
        {"op": "set_max_sessions", "max_sessions": max_sessions},
        timeout=timeout,
    )


def set_decoder_draining(
    port: int, draining: bool = True, *, timeout: float = 5.0
) -> dict:
    """Toggle graceful draining mode on a running decoder_worker."""
    return _decoder_control_rpc(
        port,
        {"op": "set_draining", "draining": bool(draining)},
        timeout=timeout,
    )


def set_dynamic_decoder_ports(ports: list[int] | tuple[int, ...] | None) -> None:
    """Dynamically override the decoder port pool for new sessions in this process."""
    global _DYNAMIC_DECODER_PORTS
    if ports is None:
        _DYNAMIC_DECODER_PORTS = None
        return
    validated = [int(p) for p in ports if type(p) is int and 1024 <= p <= 65535]
    if not validated or len(validated) != len(ports):
        raise ValueError("Invalid dynamic decoder service ports")
    _DYNAMIC_DECODER_PORTS = validated


def _resolve_dynamic_decoder_ports(config: dict) -> list[int] | None:
    if _DYNAMIC_DECODER_PORTS:
        return list(_DYNAMIC_DECODER_PORTS)
    reg_path = config.get("pool_registry_path") or os.environ.get(
        "SGLANG_DISAGG_POOL_REGISTRY"
    )
    if reg_path:
        try:
            p = Path(reg_path)
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                ports = data.get("decoder_ports")
                if isinstance(ports, list) and ports:
                    return [int(x) for x in ports]
        except Exception:
            pass
    return None


def close_decoder_bridge(
    port: int, bridge_id: str, *, timeout: float = 5.0, force: bool = False
) -> dict:
    """Explicitly close a persistent bridge on a running decoder_worker."""
    return _decoder_control_rpc(
        port,
        {"op": "bridge_close", "bridge_id": bridge_id, "force": bool(force)},
        timeout=timeout,
    )


def close_persistent_decoder_bridges(
    ports: list[int] | tuple[int, ...] | None = None,
    *,
    timeout: float = 5.0,
    force: bool = False,
) -> dict[int, dict]:
    """Close and destroy cached persistent decoder bridges for this DiT worker."""
    target_ports = (
        list(_PERSISTENT_DECODER_BRIDGES.keys())
        if ports is None
        else [p for p in ports if p in _PERSISTENT_DECODER_BRIDGES]
    )
    results: dict[int, dict] = {}
    for port in target_ports:
        entry = _PERSISTENT_DECODER_BRIDGES.pop(port, None)
        if entry is None:
            continue
        bridge_id, _store, bridge = entry
        try:
            results[port] = close_decoder_bridge(
                port, bridge_id, timeout=timeout, force=force
            )
        except Exception as exc:
            results[port] = {"ok": False, "bridge_id": bridge_id, "error": str(exc)}
        if bridge is not None:
            try:
                dist.destroy_process_group(bridge)
            except Exception:
                pass
    return results


atexit.register(close_persistent_decoder_bridges)


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

        module = codecs.modules.get(codecs.decoder.module_key)
        if module is not None:
            self.protocol = module.protocol
        else:
            # A dedicated DiT has no local decoder weights. Read the same
            # architecture metadata used by the decoder loader, without
            # constructing a second model or a synthetic codecs object.
            from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.loader import (
                read_architecture,
            )
            from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.temporal import (
                TemporalProtocol,
            )

            spec = codecs.pipeline_config.resolve_decoder_spec()
            if spec.type != "lynn_streaming":
                raise ValueError("Remote source decoding requires the Lynn decoder")
            self.protocol = TemporalProtocol(read_architecture(spec.model_dir))
        self.connection = config.get("connection")
        self.host = config.get("host", "127.0.0.1")
        self.auto_output = config.get("auto_output", False)
        self.output_session = config.get("output_session")
        self.timing_updates = False
        self.oneway = config.get("oneway", False)
        if type(self.oneway) is not bool or (
            self.oneway and not (self.auto_output and self.output_session)
        ):
            raise ValueError("One-way decoding requires direct output")
        self.output_attachment = None
        self.output_capabilities = {}
        self.output_metadata = None
        if type(self.auto_output) is not bool:
            raise ValueError("Invalid decoder auto-output mode")
        self._pending_output = None
        self.group = get_world_group()
        self.leader = self.group.rank_in_group == 0
        self.capacity_latents = capacity_latents
        self.provenance_for = provenance_for
        dyn_ports = (
            None
            if self.connection is not None
            else _resolve_dynamic_decoder_ports(config)
        )
        raw_ports = dyn_ports if dyn_ports is not None else config.get("ports")
        if raw_ports is not None:
            if not isinstance(raw_ports, (list, tuple)) or not raw_ports:
                raise ValueError("Invalid decoder service ports")
            ports = [int(p) for p in raw_ports if type(p) is int]
            if len(ports) != len(raw_ports) or any(
                not 1024 <= p <= 65535 for p in ports
            ):
                raise ValueError("Invalid decoder service port")
            self.ports = ports
            self.port = config.get("port", ports[0])
            self._pool_mode = True
        else:
            self.port = config["port"]
            if type(self.port) is not int or not 1024 <= self.port <= 65535:
                raise ValueError("Invalid decoder service port")
            self.ports = [self.port]
            self._pool_mode = self.port in _PERSISTENT_DECODER_BRIDGES
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
        if self.connection is not None:
            self._connect_group()
        else:
            self._connect()

    def _connect_group(self):
        """Reuse the group built by HTTP; opening a session loads no weights."""
        if not self.leader:
            return
        context = self.context = zmq.Context()
        self.socket = self._command_socket(context)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout * 1000)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout * 1000)
        self.socket.connect(f"tcp://{self.host}:{self.port}")
        self.bridge = self.connection["bridge"]
        try:
            opened = self._request(
                "open",
                bridge_id=self.connection["group_id"],
                capacity_latents=self.capacity_latents,
                auto_output=self.auto_output,
                output_session=self.output_session,
            )
            self.channel = opened["channel"]
            self._opened(opened)
        except BaseException as error:
            self.failure = error

    def _opened(self, reply):
        self.timing_updates = reply.get("timing_updates") is True
        self.output_attachment = reply.get("output_attachment")
        self.oneway = self.oneway and bool(
            self.output_attachment and reply.get("oneway_supported")
        )
        # An old or unrecognized peer cannot acquire a newer release contract
        # merely because the local launcher enables progressive decoding.
        self.output_capabilities = {}
        if reply.get("source_output_release_protocol") == "valid-source-frames-v1":
            self.output_capabilities = dict(
                source_output_release_protocol="valid-source-frames-v1",
                progressive_output=reply.get("progressive_output") is True,
            )

    def _command_socket(self, context):
        if self.oneway:
            from sglang.multimodal_gen.runtime.realtime.worker_services import (
                OneWayCommandClient,
            )

            return OneWayCommandClient(context)
        return context.socket(zmq.REQ)

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

    def _rank_candidate_ports(self, context) -> list[int]:
        if len(self.ports) == 1 and not self._pool_mode:
            return list(self.ports)
        scored = []
        for idx, cand_port in enumerate(self.ports):
            sock = context.socket(zmq.REQ)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.RCVTIMEO, min(self.timeout, 2) * 1000)
            sock.setsockopt(zmq.SNDTIMEO, min(self.timeout, 2) * 1000)
            sock.connect(f"tcp://127.0.0.1:{cand_port}")
            try:
                sock.send_json({"op": "ping"})
                reply = sock.recv_json()
                if isinstance(reply, dict) and reply.get("ok"):
                    busy = bool(reply.get("busy", False))
                    max_s = int(reply.get("max_sessions", 1))
                    act_s = int(reply.get("active_sessions", 1 if busy else 0))
                    free_slots = max(0, max_s - act_s)
                    scored.append(((1 if busy else 0, -free_slots, idx), cand_port))
                else:
                    scored.append(((2, 0, idx), cand_port))
            except Exception:
                scored.append(((2, 0, idx), cand_port))
            finally:
                sock.close()
        scored.sort(key=lambda item: item[0])
        return [p for _, p in scored]

    def _connect(self):
        if not self.leader:
            return
        context = zmq.Context()
        if not self._pool_mode:
            socket = self._command_socket(context)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, self.timeout * 1000)
            socket.setsockopt(zmq.SNDTIMEO, self.timeout * 1000)
            socket.connect(f"tcp://{self.host}:{self.port}")
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
                opened = self._request(
                    "open",
                    bridge_port=store.port,
                    capacity_latents=self.capacity_latents,
                    **({"auto_output": True} if self.auto_output else {}),
                    **(
                        {"output_session": self.output_session}
                        if self.output_session
                        else {}
                    ),
                )
                self.channel = opened["channel"]
                self._opened(opened)
                self.bridge = bridge_group(store, 0, timeout_seconds=self.timeout)
            except BaseException as error:
                # Reported by the first pop on every DiT rank, like a later failure.
                self.failure = error
            return

        self.context = context
        try:
            candidates = self._rank_candidate_ports(context)
            last_error = None
            for cand_port in candidates:
                socket = self._command_socket(context)
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.RCVTIMEO, self.timeout * 1000)
                socket.setsockopt(zmq.SNDTIMEO, self.timeout * 1000)
                socket.connect(f"tcp://127.0.0.1:{cand_port}")
                try:
                    if cand_port in _PERSISTENT_DECODER_BRIDGES:
                        bridge_id, store, bridge = _PERSISTENT_DECODER_BRIDGES[
                            cand_port
                        ]
                    else:
                        store = dist.TCPStore(
                            "127.0.0.1",
                            0,
                            2,
                            True,
                            timeout=timedelta(seconds=self.timeout),
                            wait_for_workers=False,
                        )
                        bridge_id = uuid.uuid4().hex
                        socket.send_json(
                            dict(
                                op="bridge_open",
                                bridge_id=bridge_id,
                                bridge_port=store.port,
                            )
                        )
                        b_reply = socket.recv_json()
                        if not isinstance(b_reply, dict) or not b_reply.get("ok"):
                            raise RuntimeError(self._failure_message())
                        bridge = bridge_group(store, 0, timeout_seconds=self.timeout)
                        _PERSISTENT_DECODER_BRIDGES[cand_port] = (
                            bridge_id,
                            store,
                            bridge,
                        )
                    socket.send_json(
                        dict(
                            op="open",
                            session=self.token,
                            bridge_id=bridge_id,
                            capacity_latents=self.capacity_latents,
                            **({"auto_output": True} if self.auto_output else {}),
                            **(
                                {"output_session": self.output_session}
                                if self.output_session
                                else {}
                            ),
                        )
                    )
                    reply = socket.recv_json()
                    if isinstance(reply, dict) and reply.get("ok"):
                        self._opened(reply)
                        self.port = cand_port
                        self.socket = socket
                        self.store = store
                        self.bridge = bridge
                        self.channel = reply["channel"]
                        return
                    if isinstance(reply, dict) and reply.get("code") == "DECODER_BUSY":
                        socket.close()
                        last_error = RuntimeError("Decoder is busy")
                        continue
                    _PERSISTENT_DECODER_BRIDGES.pop(cand_port, None)
                    raise RuntimeError(self._failure_message())
                except BaseException as exc:
                    _PERSISTENT_DECODER_BRIDGES.pop(cand_port, None)
                    socket.close()
                    last_error = exc
            raise last_error or RuntimeError(self._failure_message())
        except BaseException as error:
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
            self.socket.send_json(
                dict(
                    op=op,
                    **identity,
                    **fields,
                    **(
                        {"worker_group_id": self.connection["group_id"]}
                        if self.connection is not None
                        else {}
                    ),
                )
            )
            reply = self.socket.recv_json()
        except BaseException as error:
            self.failure = error
            raise RuntimeError(self._failure_message()) from error
        if not isinstance(reply, dict) or not reply.get("ok"):
            raise RuntimeError(self._failure_message())
        return reply

    def configure_timing(self, message):
        if not self.timing_updates or self.closed or self.total_frames is not None:
            raise ValueError("Decoder timing control is unavailable")
        return self._request("timing.configure", **message)

    def _build_chunk_frame_timings(
        self,
        timing_ctx: dict,
        *,
        start_frame: int,
        end_frame: int,
        send_t0_ns: int,
        send_done_ns: int,
    ) -> list[dict]:
        dit_start_ns = timing_ctx.get("dit_start_ns")
        if type(dit_start_ns) is not int:
            dit_start_ns = send_t0_ns
        ev_dit_start = timing_ctx.get("ev_dit_start")
        ev_dit_end = timing_ctx.get("ev_dit_end")
        total_dit_plus_ret_ms = max(0.0, (send_done_ns - dit_start_ns) / 1e6)
        if ev_dit_start is not None and ev_dit_end is not None:
            ev_dit_end.synchronize()
            raw_dit_proc_ms = min(
                total_dit_plus_ret_ms,
                max(0.0, float(ev_dit_start.elapsed_time(ev_dit_end))),
            )
            dit_return_ms = max(0.0, total_dit_plus_ret_ms - raw_dit_proc_ms)
        else:
            raw_dit_proc_ms = max(0.0, (send_t0_ns - dit_start_ns) / 1e6)
            dit_return_ms = max(0.0, (send_done_ns - send_t0_ns) / 1e6)
        by_frame: dict[int, dict] = {}
        for lt in timing_ctx.get("input_timings") or ():
            if not isinstance(lt, dict):
                continue
            p0 = lt.get("pixel_start")
            p1 = lt.get("pixel_end")
            if type(p0) is not int or type(p1) is not int:
                continue
            sample_ms = max(0.0, float(lt.get("sample_ms", 0.0) or 0.0))
            arrived_ns = lt.get("arrived_ns")
            dit_wait_ms = (
                max(0.0, (dit_start_ns - arrived_ns) / 1e6 - sample_ms)
                if type(arrived_ns) is int
                else 0.0
            )
            # Events are local to this rank and precede the already synchronized
            # DiT/send boundary. Reading them here adds no sampling-path wait.
            # Keep Host sampling time for Host queue accounting; CUDA elapsed
            # time belongs only in compute, not in a subtraction across clocks.
            sample_events = lt.get("sample_events")
            sample_compute_ms = (
                max(0.0, float(sample_events[0].elapsed_time(sample_events[1])))
                if sample_events is not None
                else sample_ms
            )
            dit_proc_ms = raw_dit_proc_ms + sample_compute_ms
            enc_frames = lt.get("encoder_frames") or {}
            for frame_idx in range(max(start_frame, p0), min(end_frame, p1)):
                row: dict[str, Any] = {
                    "output_frame": frame_idx,
                    "dit": {
                        "wait_ms": round(dit_wait_ms, 3),
                        "process_ms": round(dit_proc_ms, 3),
                        "return_ms": round(dit_return_ms, 3),
                    },
                }
                enc_raw = enc_frames.get(frame_idx)
                if isinstance(enc_raw, dict):
                    enc_stage = {
                        k: round(max(0.0, float(enc_raw[k])), 3)
                        for k in ("wait_ms", "process_ms", "return_ms")
                        if type(enc_raw.get(k)) in (int, float)
                    }
                    if enc_stage:
                        row["encoder"] = enc_stage
                by_frame[frame_idx] = row
        return [
            by_frame.get(frame_idx)
            or {
                "output_frame": frame_idx,
                "dit": {
                    "wait_ms": 0.0,
                    "process_ms": round(raw_dit_proc_ms, 3),
                    "return_ms": round(dit_return_ms, 3),
                },
            }
            for frame_idx in range(start_frame, end_frame)
        ]

    def _emit_chunk_timing(
        self,
        timing_ctx: dict,
        *,
        start_frame: int,
        end_frame: int,
        send_t0_ns: int,
        send_done_ns: int,
    ) -> None:
        if not self.output_attachment or end_frame <= start_frame:
            return
        frame_timings = self._build_chunk_frame_timings(
            timing_ctx,
            start_frame=start_frame,
            end_frame=end_frame,
            send_t0_ns=send_t0_ns,
            send_done_ns=send_done_ns,
        )
        if not frame_timings:
            return
        fields = dict(
            generation_id=self.output_session["generation_id"],
            request_id=self.request_id,
            chunk_index=self.chunk_index,
            output_frame_start=start_frame,
            num_frames=end_frame - start_frame,
            frame_stage_breakdown=frame_timings,
        )
        if self.oneway:
            self.socket.send_json(
                dict(
                    op="chunk_timing",
                    channel=self.channel,
                    oneway=True,
                    **fields,
                    **(
                        {"worker_group_id": self.connection["group_id"]}
                        if self.connection is not None
                        else {}
                    ),
                )
            )
        else:
            self._request("chunk_timing", **fields)

    def append(
        self,
        latent,
        *,
        latent_start,
        source_dependency_end,
        valid_frame_end,
        timing_ctx=None,
    ):
        if self.closed or self.total_frames is not None:
            raise ValueError("Decoder stream is closed")
        if self.auto_output and self._pending_output is not None:
            raise ValueError("Consume the preceding decoder output before appending")
        start_frame = self.emitted_frames
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
            send_t0_ns = time.perf_counter_ns()
            self.socket.send_json(
                dict(
                    op="append",
                    **(
                        {"worker_group_id": self.connection["group_id"]}
                        if self.connection is not None
                        else {}
                    ),
                    **({"oneway": True} if self.oneway else {}),
                    channel=self.channel,
                    shape=[int(n) for n in latent.shape],
                    dtype=str(latent.dtype).split(".")[-1],
                    latent_start=latent_start,
                    valid_frame_end=valid_frame_end,
                    source_dependency_end=source_dependency_end,
                    **(
                        {"output_metadata": self.output_metadata}
                        if self.output_attachment
                        else {}
                    ),
                    **(
                        dict(
                            request_id=self.request_id,
                            chunk_index=self.chunk_index,
                            generation_id=self.generation_id,
                        )
                        if self.auto_output
                        else {}
                    ),
                )
            )
            # The decoder allocates from this metadata before posting its receive.
            send_latent(self.bridge, latent, peer=1)
            if self.oneway:
                # Preserve ownership until NCCL has finished reading this tensor.
                # This is transport completion, not Decoder model completion.
                if latent.is_cuda:
                    torch.cuda.current_stream(latent.device).synchronize()
                if isinstance(timing_ctx, dict):
                    self._emit_chunk_timing(
                        timing_ctx,
                        start_frame=start_frame,
                        end_frame=valid_frame_end,
                        send_t0_ns=send_t0_ns,
                        send_done_ns=time.perf_counter_ns(),
                    )
                return
            reply = self.socket.recv_json()
            if not isinstance(reply, dict) or not reply.get("ok"):
                raise RuntimeError("Decoder service rejected the latent")
            if self.auto_output:
                self._pending_output = reply
            if isinstance(timing_ctx, dict):
                ev_dit_end = timing_ctx.get("ev_dit_end")
                if ev_dit_end is not None:
                    ev_dit_end.synchronize()
                self._emit_chunk_timing(
                    timing_ctx,
                    start_frame=start_frame,
                    end_frame=valid_frame_end,
                    send_t0_ns=send_t0_ns,
                    send_done_ns=time.perf_counter_ns(),
                )
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
        if self.leader and self.failure is None and not self.oneway:
            try:
                reply = (
                    (self._pending_output or {})
                    if self.auto_output
                    else self._request(
                        "output",
                        frames=frames,
                        request_id=self.request_id,
                        chunk_index=self.chunk_index,
                        generation_id=self.generation_id,
                    )
                )
                reference, metadata = reply.get("reference"), reply.get("metadata")
                self._pending_output = None
                if self.output_attachment:
                    if frames and (
                        reference is not None or reply.get("output_queued") is not True
                    ):
                        raise ValueError("Decoder did not accept direct output")
                elif bool(frames) != isinstance(reference, dict):
                    raise ValueError("Remote decoder output reference is invalid")
            except BaseException as error:
                self.failure = error
        # Only the failure flag crosses ranks: a reference reaching a non-output
        # rank would be cancelled there as an unsent result.
        status = [self.failure is not None]
        if getattr(self.group, "world_size", 2) > 1 or (
            getattr(self.group, "cpu_group", None) is not None
            and not hasattr(self.group, "unique_name")
        ):
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

    def source_progress(self, *, owned_frame_end, source_records):
        if self.output_attachment and self.leader:
            fields = dict(
                generation_id=self.output_session["generation_id"],
                owned_frame_end=owned_frame_end,
                source_records=source_records,
            )
            if self.oneway:
                self.socket.send_json(
                    dict(
                        op="source_progress",
                        channel=self.channel,
                        oneway=True,
                        **fields,
                        **(
                            {"worker_group_id": self.connection["group_id"]}
                            if self.connection is not None
                            else {}
                        ),
                    )
                )
            else:
                self._request("source_progress", **fields)

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
        if self._pending_output is not None:
            reference = self._pending_output.get("reference")
            if reference is not None:
                from sglang.multimodal_gen.runtime.utils.realtime_video import (
                    cancel_async_raw_rgb_frame_reference,
                )

                cancel_async_raw_rgb_frame_reference(reference)
            self._pending_output = None
        if self.leader and self.context is not None:
            if self.oneway and self.failure is None:
                # Keep abort behind all previously sent append/progress commands.
                # A fresh REQ peer could overtake queued DEALER data, leaving
                # stale commands to hit the next session after its owner closes.
                try:
                    self._request("abort")
                except RuntimeError:
                    self.failure = RuntimeError("Decoder teardown failed")
                else:
                    self.closed = True
                    self._close_channel()
                    self.provenance_for = None
                    return
            # A receive timeout leaves REQ waiting for its old reply. A fresh
            # channel can still release this owner after that reply is sent.
            self._close_socket()
            socket = self.context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, min(self.timeout, 1) * 1000)
            socket.setsockopt(zmq.SNDTIMEO, min(self.timeout, 1) * 1000)
            socket.connect(f"tcp://{self.host}:{self.port}")
            try:
                socket.send_json(
                    dict(
                        op="abort",
                        session=self.token,
                        **(
                            {"worker_group_id": self.connection["group_id"]}
                            if self.connection is not None
                            else {}
                        ),
                    )
                )
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
