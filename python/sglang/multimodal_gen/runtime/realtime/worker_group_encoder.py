# SPDX-License-Identifier: Apache-2.0
"""Connect the existing Encoder executor/adapter to per-service HTTP control."""

import asyncio

from .worker_connections import ServiceConnections, endpoint


def install_encoder_group(app, executor, pool, descriptor, store):
    async def owned(callback):
        return await asyncio.get_running_loop().run_in_executor(pool, callback)

    async def build(group, topology):
        def connect():
            from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_io import (
                DiTIOChannel,
            )
            from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_link import (
                DiTLinkClient,
            )
            from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_transport import (
                EncoderDiTChannel,
            )

            host, port = endpoint(topology["dit"]["endpoints"]["zmq"])
            store_host, _ = endpoint(descriptor["endpoints"]["rendezvous"])
            executor.dit_link = DiTLinkClient(
                DiTIOChannel(
                    lambda: EncoderDiTChannel.connect(
                        port,
                        rank=0,
                        device="cuda:0",
                        host=host,
                        store_host=store_host,
                        store=store,
                        worker_group_id=group,
                        identities=[
                            topology[role]["service_id"] for role in ("encoder", "dit")
                        ],
                    ),
                    on_failure=controller.failed,
                    cleanup_timeout=120,
                ).start()
            )
            executor.conditioning.encode_tensor = lambda value: value.detach()

        await owned(connect)

    async def close_link():
        def close():
            try:
                if executor.dit_link is not None:
                    executor.dit_link.disconnect()
            finally:
                executor.dit_link = None
                if executor.encoder is not None:
                    executor.encoder.dispose()
                executor.owner = executor.encoder = None
                executor.conditioning.dispose()

        await owned(close)

    async def rollback():
        def fail_link():
            if executor.dit_link is not None:
                executor.dit_link.failed = True

        await owned(fail_link)
        adapter = getattr(app.state, "connection_adapter", None)
        if adapter is not None and not adapter.connection_loop.is_closed():

            def clear():
                adapter.connection_draining = True
                if hasattr(adapter, "ledger"):
                    adapter.ledger.dispose()
                    adapter.pending.clear()
                    adapter.input_arranger.clear()

            adapter.connection_loop.call_soon_threadsafe(clear)
        await close_link()

    async def destroy():
        adapter = getattr(app.state, "connection_adapter", None)
        if adapter is not None:
            # Adapter queues belong to the input WS loop, not the HTTP loop.
            future = asyncio.run_coroutine_threadsafe(
                adapter.drain_connection(), adapter.connection_loop
            )
            await asyncio.wrap_future(future)
        await close_link()

    controller = ServiceConnections(
        descriptor,
        ready=lambda: hasattr(app.state, "realtime_execution_client"),
        build=build,
        destroy=destroy,
        rollback=rollback,
        cleanup_timeout=120,
    )
    app.state.worker_connections = controller
    executor.connection_controller = controller
    return controller
