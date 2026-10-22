# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual service loop with CPU control/bridge fault injection."""

import ast
import importlib.util
import json
import logging
import sys
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("close_op", ["close", "abort", "quarantine", "warmup_failure"])
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
    command_rows = [
        dict(op="open", session="owner", bridge_port=1001, capacity_frames=51),
        dict(op="open", session="intruder", bridge_port=1002, capacity_frames=51),
        dict(op="abort", session="intruder"),
        dict(op="ping"),
        dict(op="close", session="owner"),
        dict(op="open", session="intruder", bridge_port=1002, capacity_frames=51),
        dict(op="close", session="intruder"),
        dict(op="ping"),
    ]
    if close_op == "abort":
        command_rows = [
            dict(row, op="abort") if row["op"] == "close" else row
            for row in command_rows
        ]
    commands = iter(command_rows)
    sequence = 0

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
    startup = []

    def warmup():
        startup.append("warmup")
        if close_op == "warmup_failure":
            raise RuntimeError("Injected encoder warmup failure")

    def bind(address):
        assert startup == ["warmup"]

    def emit(value, **kwargs):
        if json.loads(value).get("encoder_service_ready"):
            startup.append("ready")

    socket = SimpleNamespace(
        setsockopt=lambda *a: None,
        bind=bind,
        poll=lambda *_: True,
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

    def broadcast(values, src, **kwargs):
        nonlocal sequence
        if rank:
            sequence += 1
            values[0] = ("COMMAND", sequence, take())

    def gather(output, value, **kwargs):
        # Count lifecycle gates after actual cleanup, matching the old close
        # barrier. A quarantined rejection also completes a control command,
        # but must not count as session cleanup or alter the active owner.
        if (
            isinstance(value, tuple)
            and len(value) == 3
            and (
                value[0] == "startup"
                or (value[0] == "close_complete" and sessions and sessions[-1].closed)
            )
        ):
            barriers.append(True)
        output[:] = [value, value]

    class Parser:
        def __init__(self, **kw):
            pass

        def add_argument(self, *a, **kw):
            pass

        def parse_args(self):
            return SimpleNamespace(
                model=Path("model"),
                port=31001,
                timeout_seconds=120,
                startup_timeout_seconds=600,
            )

    dist = SimpleNamespace(
        init_process_group=lambda *a, **kw: None,
        get_rank=lambda: rank,
        new_group=lambda **kw: object(),
        barrier=lambda: barriers.append(True),
        broadcast_object_list=broadcast,
        get_world_size=lambda *a: 2,
        all_gather_object=gather,
        group=SimpleNamespace(WORLD=object()),
        destroy_process_group=lambda: None,
        TCPStore=lambda host, port, *a, **kw: ports.append(port),
    )
    from test_zing_h3_encoder_abort_recovery import recovery

    control_spec = importlib.util.spec_from_file_location(
        "zing_encoder_control_cpu", path.with_name("encoder_control.py")
    )
    control = importlib.util.module_from_spec(control_spec)
    control_spec.loader.exec_module(control)
    namespace = dict(
        EncoderCommandControl=control.EncoderCommandControl,
        EncoderControlError=control.EncoderControlError,
        logging=logging,
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
                max_memory_allocated=lambda: 0,
                max_memory_reserved=lambda: 0,
            )
        ),
        os=SimpleNamespace(environ={"LOCAL_RANK": str(rank)}),
        dist=dist,
        timedelta=timedelta,
        load_encoder=lambda root: SimpleNamespace(
            modules={"video_vae": object()},
            warmup_encoder=warmup,
            runtime={
                "vae_config": {"encode_mode": "performance", "encode_dtype": "bfloat16"}
            },
        ),
        zmq=SimpleNamespace(Context=lambda: context, REP=1, LINGER=2, SNDTIMEO=3),
        json=json,
        time=SimpleNamespace(time_ns=lambda: 0),
        H3ProgressiveEncoderSession=Session,
        bridge_group=lambda *a, **kw: object(),
        print=emit,
    )
    exec(
        compile(ast.Module(body=[main], type_ignores=[]), str(path), "exec"), namespace
    )
    if close_op == "warmup_failure":
        with pytest.raises(
            control.EncoderControlError, match="participating rank"
        ) as err:
            namespace["main"]()
        assert str(err.value.__cause__) == "Injected encoder warmup failure"
        assert startup == ["warmup"]
        assert not replies and not ports and not sessions
        return
    with pytest.raises(StopLoop):
        namespace["main"]()
    assert startup == ["warmup", "ready"]
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
        assert replies[2]["current_owner_unchanged"] and not replies[2]["released"]
        assert replies[3]["busy"] is True and replies[-1]["busy"] is False
        assert replies[4]["ok"] and replies[5]["ok"] and replies[6]["ok"]
    else:
        assert not ports and not replies


@pytest.mark.parametrize("zing", [True, False])
def test_deployment_admission_capacity_is_single_session_for_zing(monkeypatch, zing):
    path = Path(__file__).resolve().parents[2] / "runtime/server_args/server_args.py"
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "ServerArgs"
    )
    method = next(
        n for n in cls.body if getattr(n, "name", None) == "_validate_batching"
    )
    module_name = "sglang.multimodal_gen.configs.pipeline_configs.zing_h3"
    module = ModuleType(module_name)

    class ZingConfig:
        pass

    module.ZingH3CausalPipelineConfig = ZingConfig
    monkeypatch.setitem(sys.modules, module_name, module)
    from sglang.multimodal_gen.runtime.realtime.source_transport import (
        SourceTransportBudget,
    )

    budget = SourceTransportBudget()
    capacities = []
    namespace = dict(
        SourceTransportBudget=SimpleNamespace(
            from_args=lambda args: SimpleNamespace(
                validate_sessions=capacities.append,
                message_bytes=budget.message_bytes,
                host_bytes=budget.host_bytes,
            )
        ),
        REALTIME_VAE_BACKENDS=("local",),
        REALTIME_VAE_TRANSPORTS=("auto",),
        uses_remote_vae=lambda backend: False,
    )
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    args = SimpleNamespace(
        pipeline_config=ZingConfig() if zing else object(),
        batching_mode="dynamic",
        batching_max_size=1,
        batching_delay_ms=0,
        realtime_max_sessions=8,
        realtime_max_sessions_per_worker=8,
        realtime_session_lease_ttl_s=120,
        realtime_session_idle_timeout_s=0,
        realtime_session_max_lifetime_s=0,
        realtime_worker_max_consumed_age_s=0,
        realtime_admission_wait_s=0,
        realtime_vae_backend="local",
        realtime_vae_transport="auto",
        realtime_vae_worker_url=None,
        realtime_vae_shared_memory_dir=None,
        realtime_vae_timeout_s=120,
        realtime_vae_max_message_mb=128,
    )
    namespace[method.name](args)
    expected = 1 if zing else 8
    assert (
        args.realtime_max_sessions == args.realtime_max_sessions_per_worker == expected
    )
    assert capacities == [expected]
