"""Disposable Room control process for SIGKILL/persistent-journal regression.

The production RenderHost owns the SQLite store and HTTP recovery route. No GPU
is allocated: the reserved/apply records below are deterministic model evidence.
"""

import asyncio
import hashlib
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import uvicorn
from fastapi import FastAPI

from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
    RenderError,
    RenderJournal,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import RenderHost
from sglang.multimodal_gen.runtime.realtime.whitebox.source_conditions import (
    payload_hash,
)


async def main():
    spool, phase, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    owner = "original-owner" if mode == "seed" else "replacement-owner"
    runtime = SimpleNamespace(
        spool=spool, epoch=owner, owner_id="room", rooms={}, players={}
    )
    host = RenderHost(runtime, SimpleNamespace(), None, None, None, None)
    args = dict(
        user_id="user",
        source_session_id="a" * 32,
        source_epoch="e" * 32,
        owner_epoch=owner,
    )
    journal = RenderJournal(host.store, **args)
    capability = "disposable-process-test-capability"
    fence = resume = None
    if mode == "seed":
        await journal.create()
        await journal.update(
            lambda state: state.update(
                source_state="ready",
                capability_sha256=hashlib.sha256(capability.encode()).hexdigest(),
            )
        )
        value = json.loads(
            Path(__file__)
            .with_name("fixtures")
            .joinpath("model-context-v1.json")
            .read_text()
        )
        condition = next(m for m in value["messages"] if m["type"] == "model.condition")
        condition["segmentId"] = (await journal.reserve_model_context_segment())[
            "segment_id"
        ]
        condition["references"] = []
        condition["payloadHash"] = payload_hash(condition)
        operation = await journal.model_condition(condition, [])
        await journal.update(
            lambda state: state["segment"].update(
                resource_state="reserved", state="streaming"
            )
        )
        if phase == "applied":
            await journal.prompt_applied(
                operation["segment_id"],
                operation["prompt_version"],
                dict(first_output_frame=0, num_frames=1),
            )
        state = await journal.read()
    else:
        state = await host.store.find_source("a" * 32)
        # A replacement owner cannot mutate, adopt, or replay the lost source.
        try:
            await journal.update(lambda state: state.update(source_state="ready"))
        except RenderError as error:
            fence = error.code
        try:
            await host.attach(
                dict(
                    user_id="user",
                    message=dict(
                        type="resume",
                        control_protocol="whitebox-render-v2",
                        source_session_id="a" * 32,
                        source_epoch="e" * 32,
                        owner_epoch="original-owner",
                        whitebox=dict(game_id="game"),
                    ),
                )
            )
        except RenderError as error:
            resume = error.code
    operation = next(iter(state["operations"].values()))
    app = FastAPI()
    host.install(app, "disposable-internal-token")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    while not server.started:
        if task.done():
            await task
        await asyncio.sleep(0.01)
    print(
        json.dumps(
            dict(
                port=listener.getsockname()[1],
                operation=operation,
                fence=fence,
                resume=resume,
                active_sources=len(host.sessions),
            )
        ),
        flush=True,
    )
    try:
        await task
    finally:
        await host.close()


if __name__ == "__main__":
    asyncio.run(main())
