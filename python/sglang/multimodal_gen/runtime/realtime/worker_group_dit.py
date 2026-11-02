# SPDX-License-Identifier: Apache-2.0
"""Lifecycle commands on the original DiT model owner, with existing I/O queues."""

import asyncio
from concurrent.futures import Future
from queue import Empty, Queue

from .worker_connections import ServiceConnections, endpoint


class DiTGroups:
    def __init__(
        self,
        descriptor,
        *,
        store,
        device,
        timeout,
        create_session,
        codecs,
        parallel=None,
    ):
        self.descriptor, self.store = descriptor, store
        self.device, self.timeout = device, timeout
        self.create_session, self.codecs = create_session, codecs
        self.parallel = parallel
        self.leader = parallel is None or parallel.rank_in_group == 0
        self.commands = Queue()
        self.link = self.service = self.decoder_bridge = self.decoder_store = None
        self.running = True
        self.manager = ServiceConnections(
            descriptor,
            ready=lambda: self.running,
            build=lambda group, topology: self.submit("build", group, topology),
            destroy=lambda: self.submit("destroy"),
            rollback=lambda: self.submit("rollback"),
        )

    def submit(self, kind, *args):
        if kind == "destroy" and self.link is not None:
            self.link.pause_receiving_local()
        future = Future()
        self.commands.put((kind, args, future))
        # Transport and output operations have their own finite timeouts. Do not
        # return a failure to HTTP while a queued operation is still executing.
        return future.result()

    def build(self, group, topology):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_decoder import (
            DiTDecoderOutput,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_link import (
            DiTInputService,
        )

        from .worker_parallel import ParallelChannel

        raw_link = None

        def connect():
            nonlocal raw_link
            self._build_transport(group, topology)
            raw_link = self.link

        channel = ParallelChannel(self.parallel)
        try:
            channel.leader_call(connect)
        finally:
            # Keep partial transport resources reachable for rollback.
            channel.local = raw_link if raw_link is not None else self.link
            self.link = channel
        decoder_host, decoder_port = endpoint(topology["decoder"]["endpoints"]["zmq"])

        def decoder(init):
            result = None
            error = None
            try:
                result = DiTDecoderOutput(
                    self.codecs,
                    init,
                    {
                        "host": decoder_host,
                        "port": decoder_port,
                        "timeout_seconds": self.timeout,
                        "connection": {
                            "bridge": self.decoder_bridge,
                            "group_id": group,
                        },
                    },
                    parallel=channel if self.parallel is not None else None,
                )
            except Exception as exc:  # noqa: BLE001 - report worker failure to owner
                error = exc
            # RemoteDecoder already makes external connections only on rank 0.
            # Publish open failure before any rank enters model collectives.
            try:
                channel.check_errors(error)
            except Exception:
                if result is not None:
                    result.close()
                raise
            return result

        self.service = DiTInputService(self.link, self.create_session, decoder)

    def _build_transport(self, group, topology):
        import torch

        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.decoder_transport import (
            bridge_group,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_io import (
            DiTIOChannel,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_transport import (
            EncoderDiTChannel,
        )

        _host, port = endpoint(topology["dit"]["endpoints"]["zmq"])
        self.link = DiTIOChannel(
            lambda: EncoderDiTChannel.connect(
                port,
                rank=1,
                device=self.device,
                timeout=self.timeout,
                bind_host="0.0.0.0",
                worker_group_id=group,
                identities=[
                    topology[role]["service_id"] for role in ("encoder", "dit")
                ],
            )
        )
        self.link.start()
        self.decoder_store = self.store
        from .worker_connection_ports import bridge_epoch

        epoch = bridge_epoch(
            self.store,
            group
            + ":"
            + topology["dit"]["service_id"]
            + ":"
            + topology["decoder"]["service_id"],
            0,
            timeout=self.timeout,
        )
        self.decoder_bridge = bridge_group(
            self.store,
            0,
            timeout_seconds=self.timeout,
            prefix="zing-decoder-group:" + epoch,
        )
        from sglang.multimodal_gen.runtime.disaggregation.transport.nccl_bridge import (
            send_bridge_tensor,
        )

        send_bridge_tensor(
            self.decoder_bridge, torch.zeros(1, device=self.device), peer=1
        )

    async def destroy(self, *, force=False):
        service = self.service
        if service is not None:
            if not force:
                self.link.pause_receiving()
                # Stop admitting at the model queue boundary. Finish the finite
                # inbox already accepted by the I/O owner before synthesizing EOS.
                while service.queue or self.link.poll():
                    if service.queue:
                        await service.queue.run_once()
                    else:
                        service.queue.append(self.link.receive())
                worker = service.worker
                if (
                    worker is not None
                    and worker.initial_received
                    and worker.bridge.accepted_latents > 0
                    and not worker.complete
                    and not worker.eos_received
                ):
                    # Use normal session processing, preserving source tail and
                    # decoder completion behavior, but without requiring Encoder.
                    service.queue.append(
                        {
                            "type": "source.end",
                            "transfer_id": service.sequence,
                            "session_id": service.owner[0],
                            "generation_id": service.owner[1],
                            "latent_frames": worker.bridge.accepted_latents,
                            "request_id": getattr(service, "current_message", {}).get(
                                "request_id", "group-drain"
                            ),
                            "chunk_index": getattr(service, "current_message", {}).get(
                                "chunk_index", 0
                            ),
                        }
                    )
                    await service.queue.run_once()
            service.queue.clear()
            service._close_decoder()
            if service.worker is not None:
                service.worker.close()
            self.service = None
        if self.link is not None:
            self.link.close()
            self.link = None
        if self.decoder_bridge is not None:
            shutdown = getattr(self.decoder_bridge, "shutdown", None)
            if shutdown is not None:
                shutdown()
            self.decoder_bridge = self.decoder_store = None

    async def run(self):
        try:
            while self.running:
                command = None
                future = None
                if self.leader:
                    try:
                        kind, args, future = self.commands.get_nowait()
                        command = (kind, args)
                    except Empty:
                        pass
                if self.parallel is not None:
                    command = self.parallel.broadcast_object(command)
                if command is not None:
                    kind, args = command
                    try:
                        if kind == "build":
                            self.build(*args)
                        else:
                            await self.destroy(force=kind == "rollback")
                    except Exception as exc:  # noqa: BLE001 - report worker failure to owner
                        if future is not None:
                            future.set_exception(exc)
                    else:
                        if future is not None:
                            future.set_result(None)
                    continue
                if self.service is not None:
                    try:
                        if self.service.queue:
                            await self.service.queue.run_once()
                        elif self.link.poll():
                            self.service.queue.append(self.link.receive())
                    except Exception:  # noqa: BLE001 - worker failure cleanup
                        # A failed model invocation cannot be treated as a clean
                        # transport disconnect. Keep the worker unavailable even
                        # when host-side resource release succeeds.
                        transport_failed = (
                            self.link.error is not None or self.service.transport_failed
                        )
                        with self.manager.lock:
                            self.manager.state = "draining"
                        try:
                            await self.destroy(force=True)
                        except Exception:  # noqa: BLE001 - worker failure cleanup
                            with self.manager.lock:
                                self.manager.restart_required = True
                                self.manager.state = "failed"
                        else:
                            with self.manager.lock:
                                self.manager.restart_required = not transport_failed
                                self.manager.state = (
                                    "idle" if transport_failed else "failed"
                                )
                                self.manager.token = self.manager.group = (
                                    self.manager.topology
                                ) = None
                await asyncio.sleep(0.005)
        finally:
            self.running = False
            await self.destroy(force=True)
