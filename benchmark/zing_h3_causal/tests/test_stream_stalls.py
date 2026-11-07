# SPDX-License-Identifier: Apache-2.0
import pytest
from analyze_stream_stalls import analyze_events


@pytest.mark.parametrize("shared", [False, True])
def test_job_allocation_and_runner_agree_on_dedicated_or_shared_roles(shared):
    from run_isolated_stall_suite import role_layout
    from stall_job import instance_devices

    config = {}
    if shared:
        config = dict(
            allocated_gpu_count=4,
            role_devices=dict(dit=[0, 1, 2, 3], encoder=[1], decoder=[2]),
            allow_shared_role_devices=True,
        )
    allocated, active, roles = instance_devices(config)
    assert (allocated, active) == ((4, 4) if shared else (8, 6))
    assert role_layout(dict(config, sp=4, role_devices=roles), active) == roles


def test_six_allocated_devices_cover_the_complete_dedicated_instance():
    from run_isolated_stall_suite import role_layout
    from stall_job import instance_devices

    config = dict(allocated_gpu_count=6)
    allocated, active, roles = instance_devices(config)
    assert (allocated, active) == (6, 6)
    assert roles == dict(dit=[0, 1, 2, 3], encoder=[4], decoder=[5])
    assert role_layout(dict(config, sp=4, role_devices=roles), allocated) == roles


@pytest.mark.parametrize(
    "config",
    [
        dict(allocated_gpu_count=3),
        dict(allocated_gpu_count=4),
        dict(allocated_gpu_count=6, decoder_world=2),
        dict(role_devices=dict(dit=[0, 1, 2, 3], encoder=[5], decoder=[6])),
        dict(
            decoder_world=2,
            role_devices=dict(dit=[0, 1, 2, 3], encoder=[4], decoder=[5]),
        ),
    ],
)
def test_job_allocation_rejects_capacity_gaps_and_decoder_count_mismatch(config):
    from stall_job import instance_devices

    with pytest.raises(ValueError, match="allocated GPUs"):
        instance_devices(config)


def test_shared_allocation_still_requires_explicit_runner_admission():
    from run_isolated_stall_suite import role_layout
    from stall_job import instance_devices

    config = dict(
        allocated_gpu_count=4,
        role_devices=dict(dit=[0, 1, 2, 3], encoder=[1], decoder=[2]),
    )
    _, active, roles = instance_devices(config)
    with pytest.raises(ValueError, match="Shared model devices"):
        role_layout(dict(config, sp=4, role_devices=roles), active)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("transpose", [False, True])
def test_parameter_digest_detects_interior_changes_omitted_by_head_tail_samples(
    dtype, transpose
):
    import hashlib

    import torch
    from diagnostics.dit_replay.dit_rng_probe import parameter_sha256

    values = torch.arange(64, dtype=getattr(torch, dtype)).reshape(8, 8)
    if transpose:
        values = values.t()
    expected = hashlib.sha256(
        values.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()
    assert parameter_sha256(values) == expected
    changed = values.clone()
    changed[3, 3] += 1
    assert torch.equal(values.reshape(-1)[:8], changed.reshape(-1)[:8])
    assert torch.equal(values.reshape(-1)[-8:], changed.reshape(-1)[-8:])
    assert parameter_sha256(changed) != expected


def test_parameter_digest_includes_bytes_after_the_copy_chunk_boundary():
    import hashlib

    import torch
    from diagnostics.dit_replay.dit_rng_probe import parameter_sha256

    values = torch.zeros((16 << 20) + 1, dtype=torch.uint8)
    values[-1] = 1
    expected = hashlib.sha256(values.numpy().tobytes()).hexdigest()
    assert parameter_sha256(values) == expected
    values[-1] = 2
    assert parameter_sha256(values) != expected


def test_decoder_pilot_layouts_have_disjoint_ports_and_correct_active_devices(tmp_path):
    import json

    from diagnostics.decoder.run_decoder_layout_pilot import layout_config

    cfg = dict(ports=dict(http=24000, decoder=24103))
    (tmp_path / "runner-config.json").write_text(json.dumps(cfg))
    job = dict(
        root=str(tmp_path),
        output=str(tmp_path / "output"),
        diagnostic_switch="switch",
        decoder_world=2,
        parity_master_ports=[28011, 28021],
    )
    first, second = [layout_config(job, world) for world in [1, 2]]
    assert first["role_devices"]["decoder"] == [5]
    assert second["role_devices"]["decoder"] == [5, 6]
    assert first["root"] != second["root"]
    assert set(first["ports"].values()).isdisjoint(second["ports"].values())
    with pytest.raises(ValueError, match="ports"):
        layout_config(dict(job, parity_master_ports=[24000, 28021]), 2)


def test_timing_retention_copies_raw_files_after_owned_roles_stop(
    tmp_path, monkeypatch
):
    import json

    import run_isolated_stall_suite as suite
    from stall_stage_probe import retain_isolated_stage_files

    runner = suite.Runner.__new__(suite.Runner)
    runner.c = dict(stage_timing_tmpfs=str(tmp_path / "ram"))
    runner.out = tmp_path / "results"
    runner.out.mkdir()
    runner.instance = [object()]
    source = runner.stage_timing_directory()
    source.mkdir(parents=True)
    raw = source / "raw-session.jsonl"
    raw.write_text("before stop\n")
    (source / "writer-health.json").write_text(
        json.dumps(dict(queued=0, pending=0, dropped=0, error=None))
    )
    monkeypatch.setattr(suite, "stop", lambda _: raw.write_text("last drained row\n"))
    runner.shutdown_instance()
    assert (
        raw.read_text()
        == (runner.out / "timing/raw-session.jsonl").read_text()
        == "last drained row\n"
    )
    assert (
        json.loads((runner.out / "TIMING_RETENTION.json").read_text())["source_removed"]
        is False
    )
    from sglang.multimodal_gen.runtime.utils import zing_stage_timing as timing

    monkeypatch.setenv("SGLANG_ZING_STAGE_TIMING_DIR", str(source))
    monkeypatch.setattr(timing, "remove_profile_files", timing.remove_profile_files)
    monkeypatch.setattr(
        timing, "deployment_timing_directory", timing.deployment_timing_directory
    )
    retain_isolated_stage_files(timing)
    timing.remove_profile_files(source, "session")
    assert raw.exists()
    with timing.deployment_timing_directory() as deployment:
        nested = deployment / "rank0.jsonl"
        nested.write_text("raw rows")
    assert nested.read_text() == "raw rows"


def test_dit_context_probe_freezes_four_independent_numerical_controls(tmp_path):
    import json

    from diagnostics.dit_replay.run_dit_context_probe import cells

    captures = [tmp_path / f"capture{i}" for i in range(2)]
    for capture in captures:
        capture.mkdir()
        (capture / "COMPLETE.json").write_text(
            json.dumps(dict(input_frames=15921, chunks=1873))
        )
    cfg = dict(
        common_captures=list(map(str, captures)),
        ports=dict(http=29000),
        model="model",
        pipeline="pipeline",
    )
    (tmp_path / "runner-config.json").write_text(json.dumps(cfg))
    job = dict(
        root=str(tmp_path),
        probe_blocks=1,
        probe_master_ports=[28011, 28021, 28031, 28041],
    )
    _, controls = cells(job)
    assert [(row["capture"], row["context"]) for row in controls] == [
        (str(capture), context)
        for capture in captures
        for context in ["no_grad", "inference"]
    ]
    assert all(
        "--diagnostic-blocks" in row["command"]
        and "--require-captured-kv-pool" in row["command"]
        for row in controls
    )
    with pytest.raises(ValueError, match="ports"):
        cells(dict(job, probe_master_ports=[29000, 28021, 28031, 28041]))


def test_encoder_gc_controls_have_disjoint_valid_ports():
    from diagnostics.encoder.run_encoder_gc_comparison import policy_ports

    base = dict(http=29000, encoder=29101, decoder=29103)
    first, second = [policy_ports(base, policy) for policy in ["0", "1"]]
    assert first == base
    assert set(first.values()).isdisjoint(second.values())
    assert second == {key: port + 1000 for key, port in base.items()}
    assert base == dict(http=29000, encoder=29101, decoder=29103)
    from diagnostics.encoder.run_encoder_gc_comparison import validate_control_ports

    frozen = dict(ports=base, encoder_gc_ports={"0": first, "1": second})
    assert validate_control_ports(frozen) == frozen["encoder_gc_ports"]
    with pytest.raises(ValueError, match="frozen"):
        validate_control_ports(dict(ports=base))
    for invalid in [dict(http=32000), dict(http=29000, encoder=30000), dict(http=80)]:
        with pytest.raises(ValueError, match="port blocks"):
            policy_ports(invalid, "1")


def test_finished_process_cleanup_never_resolves_a_reused_pid(monkeypatch):
    from types import SimpleNamespace

    import psutil
    from run_isolated_stall_suite import stop

    def forbidden(pid):
        raise AssertionError("A finished process PID must not be resolved")

    monkeypatch.setattr(psutil, "Process", forbidden)
    stop(SimpleNamespace(pid=12345, poll=lambda: 0))


def test_owned_session_cleanup_does_not_require_kernel_pidfd(monkeypatch, tmp_path):
    import contextlib
    import errno
    import os
    import signal
    import subprocess
    import sys
    import time

    import psutil
    from run_isolated_stall_suite import stop

    def unsupported_wait(*args, **kwargs):
        raise OSError(errno.EINVAL, "pidfd_open is unavailable in this container")

    monkeypatch.setattr(psutil, "wait_procs", unsupported_wait)
    ready = tmp_path / "child.pid"
    child_code = (
        "import os,pathlib,sys,time;"
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(60)"
    )
    parent_code = (
        "import subprocess,sys,time;"
        "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]],start_new_session=True);"
        "time.sleep(60)"
    )
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(60)"], start_new_session=True
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code, child_code, str(ready)],
        start_new_session=True,
    )
    child = None
    try:
        deadline = time.monotonic() + 10
        while not ready.exists():
            if parent.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("Owned child fixture did not start")
            time.sleep(0.01)
        child = psutil.Process(int(ready.read_text()))
        stop(parent)
        assert parent.poll() is not None
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
        assert unrelated.poll() is None
    finally:
        if child is not None:
            with contextlib.suppress(psutil.NoSuchProcess):
                child.kill()
        for process in [parent, unrelated]:
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


def test_passive_gen2_switch_records_real_gc_without_changing_policy(
    monkeypatch, tmp_path, capsys
):
    import gc
    import json
    import os

    from gc_pause_observer import install_passive_gc_diagnostic

    flag = tmp_path / "gc-enabled"
    monkeypatch.setenv("STALL_GC_DIAGNOSTIC_SWITCH", str(flag))
    monkeypatch.setenv("STALL_ROLE_GC_LABEL", "owned-test-encoder")
    policy = (gc.isenabled(), gc.get_threshold(), gc.get_freeze_count())
    uninstall = install_passive_gc_diagnostic()
    try:
        gc.collect(2)
        assert capsys.readouterr().out == ""
        flag.touch()
        gc.collect(2)
        raw = capsys.readouterr().out.strip()
        assert raw.startswith("STALL_ROLE_GC_JSON ")
        event = json.loads(raw.split(" ", 1)[1])
        assert event["generation"] == 2 and event["seconds"] >= 0
        assert event["start_unix_ns"] <= event["end_unix_ns"]
        assert (
            event["role_label"] == "owned-test-encoder" and event["pid"] == os.getpid()
        )
        assert (gc.isenabled(), gc.get_threshold(), gc.get_freeze_count()) == policy
    finally:
        uninstall()
    gc.collect(2)
    assert capsys.readouterr().out == ""


def test_encoder_gc_join_rejects_dit_with_encoder_control_prefix(tmp_path):
    import json

    from analyze_stall_gc import analyze

    results = tmp_path / "work/results"
    results.mkdir(parents=True)
    (results / "REQUEST_BOUNDARIES.json").write_text(
        json.dumps([dict(phase="passive-gc-diagnostic", session_id="s")])
    )
    rows = [
        dict(
            encoder_step="pop",
            session="s",
            rank=0,
            host_start_ns=start,
            host_end_ns=end,
            shape=[1, 48, 2, 30, 52],
        )
        for start, end in [
            (1000000000, 1100000000),
            (1200000000, 1300000000),
            (1400000000, 1900000000),
        ]
    ]
    events = [
        dict(
            role_label="encoder-gc-off-1280x736-" + role,
            pid=pid,
            rank=0,
            generation=2,
            seconds=0.4,
            start_unix_ns=1450000000,
            end_unix_ns=1850000000,
        )
        for role, pid in [("encoder", 10), ("dit", 11)]
    ]
    (results / "instance.log").write_text(
        "\n".join(
            [json.dumps(row) for row in rows]
            + ["STALL_ROLE_GC_JSON " + json.dumps(e) for e in events]
        )
    )
    report = analyze(tmp_path)
    assert len(report["observed_gc_events"]) == 2
    assert len(report["long_encoder_operations"]) == 1
    matches = report["long_encoder_operations"][0]["overlapping_encoder_gc"]
    assert len(matches) == 1 and matches[0]["event"]["pid"] == 10
    assert matches[0]["overlap_ms"] == 400
    assert report["regression_accepted"] is False


def test_complete_long_direct_diagnostic_has_explicit_raw_budget(monkeypatch, tmp_path):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    import run_isolated_stall_suite as suite

    commands = []
    out = tmp_path / "results"
    out.mkdir()

    def client(command, **kwargs):
        commands.append(command)
        target = Path(command[command.index("--output-dir") + 1])
        target.mkdir()
        (target / "requests.jsonl").write_text(
            json.dumps(dict(success=True, delivery_complete=True)) + "\n"
        )

    monkeypatch.setattr(suite.subprocess, "run", client)
    owner = SimpleNamespace(
        root=tmp_path,
        switch=tmp_path / "diagnostic-switch",
        out=out,
        bench=tmp_path,
        ports=dict(http=29000),
        c=dict(
            media=str(tmp_path),
            frozen_inputs=str(tmp_path / "frozen.json"),
            archive_dir=str(tmp_path / "archive"),
            raw_retention_budget_bytes=64 << 30,
        ),
        phase=lambda label: None,
        environment=lambda *args: {},
    )
    suite.Runner.client(
        owner,
        "diagnostic",
        width=1280,
        height=736,
        diagnostic=True,
        case_set=tmp_path / "long-only.json",
    )
    command = commands[0]
    budget = int(command[command.index("--retention-budget-bytes") + 1])
    assert budget - (64 << 20) > 15921 * 1280 * 736 * 3
    assert command[command.index("--media-mode") + 1] == "raw"
    assert "--no-stage-timing" not in command


@pytest.mark.parametrize("before_init", [False, True])
@pytest.mark.parametrize(
    "reason,reply_type,code,accepted",
    [
        ("USER_SESSION_LIMIT", "error", 1008, True),
        ("CAPACITY_EXHAUSTED", "error", 1008, True),
        ("UNRELATED_ERROR", "error", 1008, False),
        ("USER_SESSION_LIMIT", "ready", 1008, False),
        ("USER_SESSION_LIMIT", "error", 1011, False),
    ],
)
def test_lifecycle_admission_close_before_or_after_init(
    before_init, reason, reply_type, code, accepted
):
    import asyncio

    import msgpack
    import websockets
    from run_stall_lifecycle import rejected_session

    async def run():
        async def server(ws):
            if not before_init:
                await ws.recv()
            await ws.send(msgpack.packb(dict(type=reply_type, content=reason)))
            await ws.close(code, reason)

        async with websockets.serve(server, "127.0.0.1", 0) as listener:
            port = listener.sockets[0].getsockname()[1]
            async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                if before_init:
                    await asyncio.wait_for(ws.wait_closed(), 10)
                if accepted:
                    assert await rejected_session(ws, b"init") == dict(
                        type="error", content=reason
                    )
                else:
                    with pytest.raises(RuntimeError, match="explicit admission"):
                        await rejected_session(ws, b"init")

    asyncio.run(run())


def test_lifecycle_close_without_error_frame_is_not_a_passing_rejection():
    import asyncio

    import websockets
    from run_stall_lifecycle import rejected_session
    from websockets.exceptions import ConnectionClosed

    async def run():
        async def server(ws):
            await ws.close(1008, "USER_SESSION_LIMIT")

        async with websockets.serve(server, "127.0.0.1", 0) as listener:
            port = listener.sockets[0].getsockname()[1]
            async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                await asyncio.wait_for(ws.wait_closed(), 10)
                with pytest.raises(ConnectionClosed):
                    await rejected_session(ws, b"init")

    asyncio.run(run())


def batch(start, count, arrival):
    return dict(
        kind="frames",
        received_seconds=arrival,
        header=dict(output_frame_start=start, num_frames=count),
    )


def complete(frames):
    return [
        dict(kind="source_end", frames=frames),
        dict(kind="session.completed", message=dict(output_frames=frames)),
    ]


def test_long_batch_gap_is_not_a_stall_when_playback_is_buffered():
    # A 1.2 s packet gap is covered by the first batch's 2 s of video.
    report = analyze_events([batch(0, 48, 10), batch(48, 24, 11.2), *complete(72)])
    assert report["status"] == "no_stall_observed"
    assert report["ttff_seconds"] == 10  # Cold TTFF is retained, not hidden.
    assert report["dit_verdict"] == "not_evaluated"


def test_single_hitch_is_detected_despite_fast_catchup():
    report = analyze_events(
        [batch(0, 12, 1), batch(12, 12, 2.5), batch(24, 240, 2.51), *complete(264)]
    )
    assert report["status"] == "stall_observed"
    assert report["max_underflow_seconds"] == pytest.approx(0.75)


@pytest.mark.parametrize(
    "deficit, stalls, severe", [(0.25, 0, 0), (0.75, 1, 0), (1.0, 1, 0), (1.25, 1, 1)]
)
def test_delivery_stall_and_one_second_severity_have_distinct_boundaries(
    deficit, stalls, severe
):
    report = analyze_events(
        [batch(0, 12, 1), batch(12, 12, 1.75 + deficit), *complete(24)]
    )
    assert report["schema_version"] == 2
    assert report["underflow_count"] == 1
    assert report["stall_count"] == stalls
    assert report["severe_underflow_count"] == severe
    assert report["severe_underflow_threshold_seconds"] == 1.0
    assert report["status"] == ("stall_observed" if stalls else "no_stall_observed")


def test_credit_wait_is_reported_without_erasing_output_stall():
    source = dict(
        kind="source_send", credit_wait_seconds=2, source_schedule_lag_seconds=1.9
    )
    report = analyze_events([source, batch(0, 12, 1), batch(12, 12, 3), *complete(24)])
    assert report["status"] == "stall_observed"
    assert report["max_credit_wait_seconds"] == 2
    assert report["attribution"].startswith("unresolved")


def test_no_output_or_incomplete_stream_never_passes():
    assert analyze_events([])["status"] == "insufficient_evidence"
    assert analyze_events([batch(0, 12, 1)])["status"] == "insufficient_evidence"
    assert (
        analyze_events([batch(0, 12, 1), *complete(24)])["status"]
        == "insufficient_evidence"
    )


@pytest.mark.parametrize(
    "bad",
    [batch(13, 12, 2), batch(12, 0, 2), batch(12, 12, float("nan")), batch(12, 12, 0)],
)
def test_corrupted_receipts_are_rejected(bad):
    with pytest.raises(ValueError):
        analyze_events([batch(0, 12, 1), bad, *complete(24)])


def test_explicit_zero_buffer_changes_the_measured_policy():
    events = [batch(0, 12, 1), batch(12, 12, 1.7), *complete(24)]
    assert analyze_events(events)["underflow_count"] == 0
    report = analyze_events(
        events, startup_buffer_seconds=0, stall_threshold_seconds=0.1
    )
    assert report["status"] == "stall_observed"


def test_default_roles_require_four_distinct_allocated_devices():
    from run_isolated_stall_suite import role_layout

    assert role_layout({"sp": 2}, 4) == {
        "dit": [0, 1],
        "encoder": [2],
        "decoder": [3],
    }
    with pytest.raises(ValueError):
        role_layout({"sp": 2}, 2)


def test_shared_roles_require_an_explicit_allocation_contract():
    from run_isolated_stall_suite import role_layout

    config = {
        "sp": 2,
        "role_devices": {"dit": [0, 1], "encoder": [1], "decoder": [1]},
    }
    for flag in (False, "false"):
        with pytest.raises(ValueError):
            role_layout({**config, "allow_shared_role_devices": flag}, 2)
    assert (
        role_layout({**config, "allow_shared_role_devices": True}, 2)
        == config["role_devices"]
    )


@pytest.mark.parametrize("decoder_count", [2, 3])
def test_parallel_decoder_uses_dedicated_allocated_devices(decoder_count):
    from run_isolated_stall_suite import role_layout, shared_role_devices

    roles = {
        "dit": [0, 1, 2, 3],
        "encoder": [4],
        "decoder": list(range(5, 5 + decoder_count)),
    }
    assert role_layout({"sp": 4, "role_devices": roles}, 5 + decoder_count) == roles
    assert not shared_role_devices(roles)


@pytest.mark.parametrize("decoder_devices", [[], [3, 3], [1, 3]])
def test_parallel_decoder_rejects_empty_duplicate_or_shared_devices(decoder_devices):
    from run_isolated_stall_suite import role_layout

    with pytest.raises(ValueError):
        role_layout(
            {
                "sp": 2,
                "role_devices": {
                    "dit": [0, 1],
                    "encoder": [2],
                    "decoder": decoder_devices,
                },
                "allow_shared_role_devices": True,
            },
            4,
        )


@pytest.mark.parametrize(
    "roles",
    [
        {"dit": [0, 0], "encoder": [0], "decoder": [1]},
        {"dit": [0, 1], "encoder": [2], "decoder": [1]},
        {"dit": [0, 1], "encoder": [True], "decoder": [1]},
        {"dit": [0, 1], "encoder": [0], "decoder": [1]},
    ],
)
def test_role_mapping_cannot_escape_or_duplicate_the_dit_allocation(roles):
    from run_isolated_stall_suite import role_layout

    with pytest.raises(ValueError):
        role_layout(
            {"sp": 2, "role_devices": roles, "allow_shared_role_devices": True}, 2
        )


def test_worker_ordinals_preserve_the_original_allocation():
    from run_isolated_stall_suite import numeric_cuda_devices

    assert numeric_cuda_devices(None, 2) == ["0", "1"]
    assert numeric_cuda_devices("2,5", 2) == ["2", "5"]


@pytest.mark.parametrize("original", ["GPU-unallocated", "0,0", "1", "-1,0"])
def test_worker_ordinals_cannot_expand_or_alias_devices(original):
    from run_isolated_stall_suite import numeric_cuda_devices

    with pytest.raises(ValueError):
        numeric_cuda_devices(original, 2)


def test_shared_cache_trim_is_opt_in_and_preserves_live_allocations(
    monkeypatch, capsys
):
    import contextlib
    import sys
    from types import SimpleNamespace

    from stall_stage_probe import release_shared_allocator_cache

    state = dict(reserved=80, allocated=30, calls=0)

    def empty_cache():
        state["calls"] += 1
        state["reserved"] = state["allocated"]

    fake = SimpleNamespace(
        cuda=SimpleNamespace(
            memory_reserved=lambda device: state["reserved"],
            memory_allocated=lambda device: state["allocated"],
            device=lambda device: contextlib.nullcontext(),
            empty_cache=empty_cache,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.delenv("STALL_SHARED_ALLOCATOR_TRIM", raising=False)
    release_shared_allocator_cache("Decoder_process", 0)
    assert state == dict(reserved=80, allocated=30, calls=0)
    monkeypatch.setenv("STALL_SHARED_ALLOCATOR_TRIM", "1")
    release_shared_allocator_cache("Decoder_process", 0)
    assert state == dict(reserved=30, allocated=30, calls=1)
    assert '"reserved_before": 80' in capsys.readouterr().out


def test_worker_environment_binds_configured_role_ports_and_layout_policy(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from run_isolated_stall_suite import Runner

    instance = SimpleNamespace(
        c=dict(deps="/owned/deps", source="/owned/source"),
        bench=tmp_path,
        out=tmp_path,
        switch=tmp_path / "diagnostic",
        cuda_devices=["0", "1", "2", "3"],
        hardware=[None] * 4,
        roles=dict(dit=[0, 1], encoder=[2], decoder=[3]),
        sp=2,
        ports=dict(encoder=25101, decoder=25103),
    )
    monkeypatch.setenv("MASTER_PORT", "44106")
    monkeypatch.setenv("STALL_SHARED_ALLOCATOR_TRIM", "1")
    env = Runner.environment(instance, [0, 1], "test")
    assert env["STALL_ENCODER_PORT"] == "25101"
    assert env["STALL_DECODER_PORT"] == "25103"
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert "MASTER_PORT" not in env
    assert env["STALL_SHARED_ALLOCATOR_TRIM"] == "0"
    instance.hardware = [None] * 2
    instance.roles = dict(dit=[0, 1], encoder=[1], decoder=[1])
    assert (
        Runner.environment(instance, [1], "test")["STALL_SHARED_ALLOCATOR_TRIM"] == "1"
    )
    instance.hardware = [None] * 5
    instance.cuda_devices = list(map(str, range(5)))
    instance.roles = dict(dit=[0, 1], encoder=[2], decoder=[3, 4])
    assert (
        Runner.environment(instance, [3, 4], "test")["STALL_SHARED_ALLOCATOR_TRIM"]
        == "0"
    )


def test_cold_requests_precede_capture_which_would_prime_disk_compiler_caches():
    from types import SimpleNamespace

    from run_isolated_stall_suite import Runner

    actions = []

    class ReachedFormalMatrix(Exception):
        pass

    def client(label, **kwargs):
        actions.append(("client", label))
        if label == "instance-matrix":
            raise ReachedFormalMatrix

    instance = SimpleNamespace(
        launch_instance=lambda label, **kwargs: actions.append(("launch", label)),
        shutdown_instance=lambda: actions.append(("stop", None)),
        client=client,
        check_capture_gc=lambda: actions.append(("capture_gc", None)),
        cold_component_diagnostic=lambda w, h: actions.append(
            ("cold_components", (w, h))
        ),
    )
    with pytest.raises(ReachedFormalMatrix):
        Runner.run(instance)
    assert actions[0] == ("launch", "S3-832x480-restart0")
    capture_index = actions.index(("launch", "capture"))
    cold = [
        label
        for action, label in actions[:capture_index]
        if action == "client" and label.endswith("-cold")
    ]
    assert len(cold) == 6
    assert actions.index(("capture_gc", None)) < capture_index
    assert actions.index(("cold_components", (832, 480))) < capture_index
    assert actions.index(("cold_components", (1280, 736))) < capture_index
    assert actions[capture_index + 1] == ("client", "capture-inputs")


def test_gc_observer_preserves_policy_callbacks_and_cleans_up_on_error(monkeypatch):
    import gc_pause_observer
    from gc_pause_observer import GCPauseObserver

    def existing(phase, info):
        pass

    callbacks = [existing]
    monkeypatch.setattr(gc_pause_observer.gc, "callbacks", callbacks)
    observer = GCPauseObserver(clock=iter([100, 400000100]).__next__)
    with pytest.raises(RuntimeError):
        with observer:
            assert callbacks[0] is existing
            observer.callback("start", dict(generation=2))
            observer.callback("stop", dict(generation=2, collected=3, uncollectable=0))
            raise RuntimeError("owned replay failed")
    assert callbacks == [existing]
    assert observer.events == [
        dict(
            generation=2,
            start_ns=100,
            end_ns=400000100,
            seconds=0.4,
            collected=3,
            uncollectable=0,
        )
    ]


def test_gc_observer_does_not_invent_a_start_for_an_incomplete_collection():
    from gc_pause_observer import GCPauseObserver

    observer = GCPauseObserver(clock=lambda: 100)
    observer.callback("stop", dict(generation=2, collected=0, uncollectable=0))
    assert observer.events == []


def load_static_gc_helper():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/multimodal_gen/runtime/utils/zing_static_gc.py"
    )
    spec = importlib.util.spec_from_file_location("owned_static_gc_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_static_gc_preparation_keeps_automatic_gc_and_unfreezes_once(monkeypatch):
    from types import SimpleNamespace

    module = load_static_gc_helper()
    calls = []
    state = dict(frozen=375)

    def freeze():
        calls.append("freeze")
        state["frozen"] = 1000

    monkeypatch.setattr(
        module,
        "gc",
        SimpleNamespace(
            collect=lambda: calls.append("collect"),
            freeze=freeze,
            unfreeze=lambda: calls.append("unfreeze"),
            get_freeze_count=lambda: state["frozen"],
            isenabled=lambda: True,
        ),
    )
    monkeypatch.delenv("SGLANG_ZING_FREEZE_STATIC_GC", raising=False)
    handle = module.prepare_static_gc("owned startup")
    assert calls == ["collect", "freeze"]
    assert handle.receipt["automatic_gc_enabled"]
    assert handle.receipt["frozen_before"] == 375
    assert handle.receipt["frozen_after"] == 1000
    handle.close()
    handle.close()
    assert calls == ["collect", "freeze", "unfreeze"]


def test_static_gc_opt_out_and_invalid_policy_never_collect(monkeypatch):
    module = load_static_gc_helper()
    monkeypatch.setattr(
        module.gc, "collect", lambda: pytest.fail("Unexpected GC policy change")
    )
    monkeypatch.setenv("SGLANG_ZING_FREEZE_STATIC_GC", "0")
    assert module.prepare_static_gc("disabled") is None
    monkeypatch.setenv("SGLANG_ZING_FREEZE_STATIC_GC", "false")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        module.prepare_static_gc("invalid")


def test_static_gc_keeps_new_request_cycles_collectible_in_owned_process(monkeypatch):
    import subprocess
    import sys
    from pathlib import Path

    monkeypatch.setenv("SGLANG_ZING_FREEZE_STATIC_GC", "1")
    path = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/multimodal_gen/runtime/utils/zing_static_gc.py"
    )
    program = """import gc,importlib.util,sys
spec=importlib.util.spec_from_file_location('owned_static_gc',sys.argv[1])
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
static_model={'weights':object()}
handle=m.prepare_static_gc('isolated lifecycle test')
assert gc.isenabled() and gc.get_freeze_count()>0
released=[]
class Request:
 def __del__(self):released.append(True)
r=Request();r.cycle=r;del r
gc.collect();assert released==[True]
assert static_model['weights'] is not None
handle.close();assert gc.get_freeze_count()==0 and gc.isenabled()
"""
    subprocess.run([sys.executable, "-c", program, str(path)], check=True, timeout=30)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_capture_gc_deferral_restores_original_policy(enabled, fail):
    import gc

    module = load_static_gc_helper()
    original = gc.isenabled()
    try:
        (gc.enable if enabled else gc.disable)()
        try:
            with module.defer_cyclic_gc():
                assert not gc.isenabled()
                with module.defer_cyclic_gc():
                    assert not gc.isenabled()
                assert not gc.isenabled()
                if fail:
                    raise RuntimeError("Injected capture failure")
        except RuntimeError:
            assert fail
        assert gc.isenabled() is enabled
    finally:
        (gc.enable if original else gc.disable)()


def test_capture_gc_deferral_leaves_new_cycles_collectible():
    import gc
    import weakref

    module = load_static_gc_helper()

    class Cycle:
        pass

    with module.defer_cyclic_gc():
        obj = Cycle()
        obj.self = obj
        reference = weakref.ref(obj)
        del obj
        assert reference() is not None
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize(
    "mode,bad_hash,exit_code",
    [
        ("full", False, 0),
        ("full", True, 0),
        ("diagnostic", False, 0),
        ("full", False, 1),
    ],
)
def test_full_matrix_preserves_metrics_evidence_and_separates_raw_diagnostics(
    tmp_path, monkeypatch, mode, bad_hash, exit_code
):
    import json
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    import run_stall_regression as suite

    cid = "fixture-source-832x480-output-832x480"
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({cid: {"sha256": "input-sha"}}))
    monkeypatch.setattr(
        suite,
        "load_case_set",
        lambda: {"members": [{"case_id": "fixture"}], "definition": {"name": "test"}},
    )
    commands = []

    def request(command, **kwargs):
        commands.append(command)
        out = Path(command[command.index("--output-dir") + 1])
        out.mkdir(parents=True)
        raw = out / "output.rgb"
        if mode == "diagnostic":
            raw.write_bytes(bytes(12))
        row = dict(
            case_id=cid,
            success=True,
            delivery_complete=True,
            input_sha256="input-sha",
            output_sha256="output-sha",
            output_rgb_bytes=12,
            media_path=str(raw) if mode == "diagnostic" else None,
            media=dict(
                output_rgb_bytes=12, output_sha256="wrong" if bad_hash else "output-sha"
            ),
        )
        (out / "requests.jsonl").write_text(json.dumps(row) + "\n")
        return SimpleNamespace(returncode=exit_code)

    monkeypatch.setattr(suite.subprocess, "run", request)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "suite",
            "--media-root",
            str(tmp_path),
            "--frozen-inputs",
            str(frozen),
            "--output-dir",
            str(tmp_path / "result"),
            "--resolutions",
            "480",
            "--measured-runs",
            "1",
            "--mode",
            mode,
        ],
    )
    if bad_hash:
        with pytest.raises(ValueError, match="Media hash receipt"):
            suite.main()
        assert not (tmp_path / "result/COMPLETE.json").exists()
        return
    if exit_code:
        with pytest.raises(RuntimeError, match="Instance regression failed"):
            suite.main()
        assert not (tmp_path / "result/COMPLETE.json").exists()
        matrix = json.loads((tmp_path / "result/matrix.json").read_text())
        assert matrix[0]["returncode"] == 1
        assert matrix[0]["requests"][0][
            "success"
        ]  # Preserve the valid earlier receipt.
        assert len(commands) == 1  # No retry or continuation hides the failed teardown.
        return
    suite.main()
    assert len(commands) == (1 if mode == "diagnostic" else 2)
    for command in commands:
        assert command[command.index("--media-mode") + 1] == (
            "raw" if mode == "diagnostic" else "metrics"
        )
        assert int(command[command.index("--retention-budget-bytes") + 1]) == (
            (64 if mode == "diagnostic" else 2) << 30
        )
    receipt = json.loads((tmp_path / "result/COMPLETE.json").read_text())
    assert receipt["transport_completed"]
    assert receipt["verdict"] == "requires_stage_and_capacity_analysis"


@pytest.mark.parametrize("geometry", [(832, 480), (1280, 736)])
@pytest.mark.parametrize("fail", [False, True])
def test_cold_component_probe_restores_switch_and_owned_services(
    tmp_path, geometry, fail
):
    from types import SimpleNamespace

    from run_isolated_stall_suite import Runner

    switch = tmp_path / "diagnostic"
    calls = []

    def launch(label):
        assert switch.exists()  # Includes the Qwen before-ready warmup boundary.
        calls.append("launch")

    def client(label, **kwargs):
        assert (kwargs["width"], kwargs["height"]) == geometry
        assert kwargs["diagnostic"] is True
        calls.append("client")
        if fail:
            raise RuntimeError("Injected diagnostic failure")

    owned = SimpleNamespace(
        switch=switch,
        launch_instance=launch,
        client=client,
        shutdown_instance=lambda: calls.append("shutdown"),
    )
    if fail:
        with pytest.raises(RuntimeError, match="Injected diagnostic failure"):
            Runner.cold_component_diagnostic(owned, *geometry)
    else:
        Runner.cold_component_diagnostic(owned, *geometry)
    assert calls == ["launch", "client", "shutdown"]
    assert not switch.exists()


def test_resident_soak_excludes_sync_and_receipt_delays_from_work_and_gc_windows():
    from resident_soak import run_resident_round

    now, calls, receipts = [0], [], []

    def session():
        calls.append("complete")
        now[0] += 400
        return {"input_frames": 15921, "fully_replayed": True}

    def receipt(value):
        receipts.append(value["completed_sessions"])
        now[0] += 1000  # Slow storage must not satisfy the work-time gate.

    def prepare():
        now[0] += 1000  # A slow peer's starting barrier is outside session work.

    result = run_resident_round(
        session,
        600,
        clock=lambda: now[0],
        prepare_session=prepare,
        on_progress=receipt,
    )
    assert calls == ["complete", "complete"]
    assert receipts == [1, 2]
    assert result["seconds"] == 800
    assert result["wall_seconds_including_coordination_and_receipts"] == 4800
    assert [r["input_frames"] for r in result["sessions"]] == [15921, 15921]
    assert [
        (r["resident_session_entered_ns"], r["resident_session_returned_ns"])
        for r in result["sessions"]
    ] == [(1000 * 10**9, 1400 * 10**9), (3400 * 10**9, 3800 * 10**9)]


def test_resident_soak_preserves_completed_progress_and_does_not_retry_failure():
    from resident_soak import run_resident_round

    now, calls, receipts = [0], [], []

    def session():
        calls.append("attempt")
        if len(calls) == 2:
            raise RuntimeError("Full case failed")
        now[0] += 400
        return {"fully_replayed": True}

    with pytest.raises(RuntimeError, match="Full case failed"):
        run_resident_round(
            session,
            600,
            clock=lambda: now[0],
            on_progress=lambda value: receipts.append(value["completed_sessions"]),
        )
    assert calls == ["attempt", "attempt"]
    assert receipts == [1]


def test_resident_soak_rejects_invalid_duration_before_running_a_case():
    from resident_soak import run_resident_round

    for duration in [0, -1, float("inf"), float("nan")]:
        with pytest.raises(ValueError, match="finite and positive"):
            run_resident_round(lambda: pytest.fail("Unexpected session"), duration)


def test_resident_soak_waits_for_collective_completion_at_session_boundaries():
    from resident_soak import run_resident_round

    now, decisions = [0], []

    def session():
        now[0] += 400
        return {"fully_replayed": True}

    def all_ranks_complete(seconds):
        decisions.append(seconds)
        return seconds >= 1000  # A peer has not reached its gate after session 2.

    result = run_resident_round(
        session,
        600,
        clock=lambda: now[0],
        all_ranks_complete=all_ranks_complete,
    )
    assert decisions == [400, 800, 1200]
    assert result["completed_sessions"] == 3


@pytest.mark.parametrize("fail_soak", [False, True])
def test_default_dit_matrix_runs_both_full_context_and_resident_soak(
    tmp_path, fail_soak
):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from run_isolated_stall_suite import Runner

    captures = tmp_path / "captures"
    for geometry in ["832x480", "1280x736"]:
        directory = captures / geometry
        directory.mkdir(parents=True)
        (directory / "COMPLETE.json").write_text(
            json.dumps(dict(input_frames=15921, chunks=1873, capture_scope="dit"))
        )
    launched = []

    def start(label, command, devices, *, group):
        assert devices == [0, 1, 2, 3] and group is False
        launched.append((label, command))
        return SimpleNamespace(
            wait=lambda timeout: int(fail_soak and label == "SOAK-0")
        )

    owned = SimpleNamespace(
        root=tmp_path,
        out=tmp_path / "results",
        model=tmp_path / "model",
        bench=tmp_path / "benchmark",
        c={"pipeline": str(tmp_path / "pipeline.json")},
        sp=4,
        ports={"replay_master": 26011},
        roles={"dit": [0, 1, 2, 3]},
        phase=lambda name: None,
        start=start,
    )
    if fail_soak:
        with pytest.raises(RuntimeError, match="Resident complete-case"):
            Runner.replay_dit(owned)
        assert [label for label, _ in launched] == ["D1-0", "SOAK-0"]
    else:
        Runner.replay_dit(owned)
        assert [label for label, _ in launched] == ["D1-0", "SOAK-0", "D1-1", "SOAK-1"]
    for label, command in launched:
        assert "--require-captured-kv-pool" in command
        assert "--nproc-per-node=4" in command
        assert command[command.index("--measured-runs") + 1] == "3"
        assert Path(command[command.index("--capture") + 1]).parent == captures
        if label.startswith("SOAK"):
            assert command[command.index("--minimum-round-seconds") + 1] == "600"
        else:
            assert "--minimum-round-seconds" not in command
