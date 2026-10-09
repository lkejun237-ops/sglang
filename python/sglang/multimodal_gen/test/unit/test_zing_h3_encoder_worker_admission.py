# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual service loop with CPU control/bridge fault injection."""

import ast
import json
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace, ModuleType

import pytest


@pytest.mark.parametrize("close_op", ["close", "abort", "quarantine"])
@pytest.mark.parametrize("rank", [0, 1])
def test_busy_open_preserves_owner_and_all_ranks_recover(monkeypatch, rank, close_op):
    # Extract the actual main loop, avoiding unrelated CUDA platform imports on
    # CPU-only hosts. No alternate implementation of its admission logic.
    path = (
        Path(__file__).resolve().parents[2]
        / "runtime/pipelines_core/stages/model_specific_stages/zing_h3/encoder_worker.py"
    )
    tree = ast.parse(path.read_text())
    main = next(
        x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "main"
    )
    commands = iter(
        [
            dict(op="open", session="owner", bridge_port=1001, capacity_frames=51),
            dict(op="open", session="intruder", bridge_port=1002, capacity_frames=51),
            dict(op="ping"),
            dict(op=close_op, session="owner"),
            dict(op="open", session="intruder", bridge_port=1002, capacity_frames=51),
            dict(op=close_op, session="intruder"),
            dict(op="ping"),
        ]
    )

    if close_op == "quarantine":
        commands = iter(
            [
                dict(op="open", session="owner", bridge_port=1001, capacity_frames=51),
                dict(op="abort", session="owner", require_restart=True),
                *[
                    dict(op=op, session="owner")
                    for op in ("append", "finish", "pop", "close")
                ],
                dict(
                    op="open", session="intruder", bridge_port=1002, capacity_frames=51
                ),
                dict(op="ping"),
            ]
        )

    class StopLoop(BaseException):
        pass

    def take():
        try:
            return next(commands)
        except StopIteration:
            raise StopLoop

    replies, ports, sessions, barriers = [], [], [], []
    socket = SimpleNamespace(
        setsockopt=lambda *a: None,
        bind=lambda *a: None,
        recv_multipart=lambda: [json.dumps(take()).encode()],
        send_json=lambda value: replies.append(value),
        close=lambda: None,
    )
    context = SimpleNamespace(socket=lambda *a: socket, term=lambda: None)

    class Session:
        ready = False
        buffered_frames = 0

        def __init__(self, codecs, capacity_frames, cache):
            self.cache = cache
            self.closed = False
            sessions.append(self)

        def dispose(self):
            assert not self.closed
            self.closed = True

        def state_usage(self):
            return SimpleNamespace(retained_bytes=0, temporal_positions=0)

    module_name = "sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.parallel_encoder"
    fake = ModuleType(module_name)
    fake.create_encoder_cache = lambda *a, **kw: SimpleNamespace(last_execution={})
    monkeypatch.setitem(sys.modules, module_name, fake)

    def broadcast(values, src):
        if rank:
            values[0] = take()

    class Parser:
        def __init__(self, **kw):
            pass

        def add_argument(self, *a, **kw):
            pass

        def parse_args(self):
            return SimpleNamespace(model=Path("model"), port=31001, timeout_seconds=120)

    dist = SimpleNamespace(
        init_process_group=lambda *a, **kw: None,
        get_rank=lambda: rank,
        new_group=lambda **kw: object(),
        barrier=lambda: barriers.append(True),
        broadcast_object_list=broadcast,
        get_world_size=lambda: 2,
        all_gather_object=lambda output, value, **kw: output.__setitem__(
            slice(None), [value, value]
        ),
        group=SimpleNamespace(WORLD=object()),
        destroy_process_group=lambda: None,
        TCPStore=lambda host, port, *a, **kw: ports.append(port),
    )
    from test_zing_h3_encoder_abort_recovery import recovery

    namespace = dict(
        abort_owner=recovery.abort_owner,
        __doc__="test",
        argparse=SimpleNamespace(ArgumentParser=Parser),
        Path=Path,
        initialize_runtime_imports=lambda: (
            SimpleNamespace(from_kwargs=lambda **kw: None),
            lambda a: None,
        ),
        torch=SimpleNamespace(
            cuda=SimpleNamespace(
                set_device=lambda n: None,
                memory_allocated=lambda: 0,
                memory_reserved=lambda: 0,
            )
        ),
        os=SimpleNamespace(environ={"LOCAL_RANK": str(rank)}),
        dist=dist,
        timedelta=timedelta,
        load_encoder=lambda root: SimpleNamespace(
            modules={"video_vae": object()},
            runtime={
                "vae_config": {"encode_mode": "performance", "encode_dtype": "bfloat16"}
            },
        ),
        zmq=SimpleNamespace(Context=lambda: context, REP=1, LINGER=2),
        json=json,
        time=SimpleNamespace(time_ns=lambda: 0),
        H3ProgressiveEncoderSession=Session,
        bridge_group=lambda *a, **kw: object(),
    )
    exec(
        compile(ast.Module(body=[main], type_ignores=[]), str(path), "exec"), namespace
    )
    with pytest.raises(StopLoop):
        namespace["main"]()
    if close_op == "quarantine":
        assert len(sessions) == 1 and sessions[0].closed
        assert len(barriers) == 1
        if rank == 0:
            assert ports == [1001]
            assert all(r["code"] == "ENCODER_RESTART_REQUIRED" for r in replies[1:-1])
            assert replies[-1]["restart_required"] and replies[-1]["busy"]
        else:
            assert not replies
        return
    assert len(sessions) == 2 and all(x.closed for x in sessions)
    assert len(barriers) == (3 if close_op == "close" else 1)
    if rank == 0:
        assert ports == [1001, 1002]  # Rejected open never enters its bridge.
        assert replies[1] == dict(
            ok=False,
            code="ENCODER_BUSY",
            error="The encoder is busy. Please try again later.",
        )
        assert replies[2]["busy"] is True and replies[-1]["busy"] is False
        assert replies[3]["ok"] and replies[4]["ok"] and replies[5]["ok"]
    else:
        assert not ports and not replies
