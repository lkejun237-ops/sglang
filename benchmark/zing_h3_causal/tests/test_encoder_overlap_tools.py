# SPDX-License-Identifier: Apache-2.0

import errno
import json

import pytest
from analyze_encoder_overlap import (
    analyze,
    component_rows,
    decoder_evidence,
    intersection_us,
    soak_evidence,
    strict_dit_evidence,
    trace_intervals,
)
from run_encoder_overlap_matrix import (
    client_command,
    formal_phases,
    layouts,
    read_text_retry,
    required_gpus,
    selected_layouts,
    session_timing_ready,
    validate_encoder_acceptance,
    validate_serial_overlap,
)
from run_issue253_split_baseline import validate_session_timing


def test_issue253_formal_collection_is_one_warmup_and_three_measured():
    assert formal_phases({"measured_runs": 3}) == [
        "warmup",
        "measured-0",
        "measured-1",
        "measured-2",
    ]


def test_shared_filesystem_read_retries_only_transient_errors(monkeypatch, tmp_path):
    path = tmp_path / "health.json"
    path.write_text('{"written": 1}')
    original = type(path).read_text
    attempts = []

    def transient(self, *args, **kwargs):
        attempts.append(self)
        if len(attempts) == 1:
            raise OSError(errno.EIO, "transient shared filesystem read")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(path), "read_text", transient)
    assert json.loads(read_text_retry(path, delay=0))["written"] == 1
    assert len(attempts) == 2

    def permanent(self, *args, **kwargs):
        raise OSError(errno.EACCES, "permanent")

    monkeypatch.setattr(type(path), "read_text", permanent)
    with pytest.raises(OSError, match="permanent"):
        read_text_retry(path, delay=0)


def timing_row(session, stage, rank, **fields):
    return dict(
        request_id=session,
        dispatch_id=f"{session}:dispatch",
        dispatch_id_status="present",
        stage=stage,
        rank=rank,
        status="ok",
        cuda_status="complete",
        cuda_stream_ms=1.25,
        **fields,
    )


@pytest.mark.parametrize("enc", [1, 2, 4])
def test_split_timing_is_validated_per_session_rank_and_range(enc):
    session = "client-session"
    rows = [
        timing_row(
            session,
            "Encoder",
            0,
            component="source_vae",
            worker_role="dedicated_source_encoder",
            latent_range=[0, 1],
        ),
        timing_row(
            session,
            "Encoder",
            0,
            component="source_vae",
            worker_role="dedicated_source_encoder",
            latent_range=[1, 2],
        ),
    ]
    encoder_rows = list(rows)
    for rank in range(1, enc):
        rows.extend({**row, "rank": rank} for row in encoder_rows)
    for rank in range(2):
        rows.extend(
            [
                timing_row(session, "DiT", rank, output_latent_range=[0, 1]),
                timing_row(session, "DiT", rank, output_latent_range=[1, 2]),
                timing_row(session, "Decoder", rank, output_frame_range=[0, 3]),
                timing_row(session, "Decoder", rank, output_frame_range=[3, 5]),
                timing_row(
                    session,
                    "Request",
                    rank,
                    component="continuous_source_complete",
                ),
            ]
        )
    assert session_timing_ready(rows, session, 2, 5, enc=enc)
    if enc == 1:
        assert validate_session_timing(rows, session, 5)["encoder_rows"] == 2
    missing_encoder_rank = [
        row
        for row in rows
        if not (row["stage"] == "Encoder" and row["rank"] == enc - 1)
    ]
    assert not session_timing_ready(missing_encoder_rank, session, 2, 5, enc=enc)

    stale_only = [{**row, "request_id": "warmup"} for row in rows]
    assert not session_timing_ready(stale_only, session, 2, 5, enc=enc)
    with pytest.raises(RuntimeError, match="missing"):
        validate_session_timing(stale_only, session, 5)

    incomplete = [dict(row) for row in rows]
    incomplete[-1]["cuda_status"] = "timeout"
    incomplete[-1]["cuda_stream_ms"] = None
    assert not session_timing_ready(incomplete, session, 2, 5, enc=enc)
    if enc == 1:
        with pytest.raises(RuntimeError, match="incomplete CUDA"):
            validate_session_timing(incomplete, session, 5)


def test_host_timing_without_dedicated_encoder_uses_server_completion():
    session = "local-source"
    rows = []
    for rank in range(2):
        rows.extend(
            [
                {
                    **timing_row(session, "DiT", rank, output_latent_range=[0, 2]),
                    "cuda_status": "not_requested",
                    "cuda_stream_ms": None,
                },
                {
                    **timing_row(session, "Decoder", rank, output_frame_range=[0, 5]),
                    "cuda_status": "not_requested",
                    "cuda_stream_ms": None,
                },
                {
                    **timing_row(
                        session,
                        "Request",
                        rank,
                        component="continuous_source_complete",
                    ),
                    "cuda_status": "not_requested",
                    "cuda_stream_ms": None,
                },
            ]
        )
    assert session_timing_ready(rows, session, 2, 5, enc=0, timing_mode="host")


def test_matrix_never_exceeds_eight_cards_and_does_not_invent_sp6():
    assert all(sp + enc <= 8 and sp in (1, 2, 4, 8) for _, sp, enc, _ in layouts())
    assert not any(sp == 8 and enc for _, sp, enc, _ in layouts())


def test_missing_matrix_is_unknown_not_success(tmp_path):
    result = analyze(tmp_path)
    assert result["sp8_speedup_ge_6"] is None
    assert result["encoder_4_vs_2"] is None
    assert all(row["status"] == "missing" for row in result["endpoints"])


def test_rank_times_are_maximized_per_repetition_before_median(tmp_path):
    for rank, times in enumerate(([1, 9, 1, 9, 1], [8, 1, 8, 1, 8])):
        path = tmp_path / "enc2" / f"rank{rank}"
        path.mkdir(parents=True)
        (path / "measured.json").write_text(
            json.dumps([dict(seconds=x) for x in times])
        )
    assert component_rows(tmp_path, "enc", (2,))[0]["median_seconds"] == 8


def test_overlap_is_interval_union_not_sum_of_nested_or_rank_kernels():
    assert intersection_us([(0, 10), (2, 8), (8, 12)], [(5, 15), (6, 9)]) == 7
    assert intersection_us([(0, 1)], [(1, 2)]) == 0


def test_trace_alignment_requires_verifiable_wall_anchor(tmp_path):
    p = tmp_path / "trace.json"
    p.write_text(json.dumps(dict(traceEvents=[])))
    assert trace_intervals(p, "Zing.encoder_tiles")[0] is None
    p.write_text(
        json.dumps(
            dict(
                baseTimeNanoseconds=1_000_000_000,
                traceEvents=[
                    dict(
                        ph="X",
                        ts=1,
                        dur=1,
                        cat="user_annotation",
                        name="Zing.clock_anchor.unix_ns2000000000",
                    )
                ],
            )
        )
    )
    assert "alignment" in trace_intervals(p, "Zing.encoder_tiles")[1]


def test_trace_attributes_by_launch_ids_and_aligns_epoch(tmp_path):
    p = tmp_path / "trace.json"
    p.write_text(
        json.dumps(
            dict(
                baseTimeNanoseconds=1_000_000_000,
                traceEvents=[
                    dict(
                        ph="X",
                        ts=0,
                        dur=1,
                        pid=1,
                        tid=1,
                        cat="user_annotation",
                        name="Zing.clock_anchor.unix_ns1000000000",
                    ),
                    dict(
                        ph="X",
                        ts=10,
                        dur=10,
                        pid=1,
                        tid=1,
                        cat="user_annotation",
                        name="Zing.encoder_tiles",
                    ),
                    dict(
                        ph="X",
                        ts=11,
                        dur=2,
                        pid=1,
                        tid=1,
                        cat="cuda_runtime",
                        name="cudaLaunchKernel",
                        args={"correlation": 3},
                    ),
                    dict(
                        ph="X",
                        ts=30,
                        dur=4,
                        pid=2,
                        tid=2,
                        cat="kernel",
                        name="conv",
                        args={"correlation": 3},
                    ),
                ],
            )
        )
    )
    intervals, evidence = trace_intervals(p, "Zing.encoder_tiles")
    assert intervals == [(1_000_030, 1_000_034)]
    assert evidence["clock_max_error_ns"] == 0


def test_acceptance_rejects_stale_source_or_missing_tail_case(tmp_path):
    for enc in (1, 2, 4):
        for rank in range(enc):
            directory = tmp_path / f"encoder-{enc}" / f"rank{rank}"
            directory.mkdir(parents=True)
            (directory / "COMPLETE").touch()
            (directory / "manifest.json").write_text(
                json.dumps(
                    dict(
                        execution_schema_version=2,
                        runtime={
                            "vae_config": {
                                "encode_mode": "performance",
                                "encode_dtype": "bfloat16",
                            }
                        },
                        effective_cudnn_benchmark_limit=None,
                        requested_tile_mode="parallel",
                        reference_tile_mode="replicated",
                        source_sha256="source",
                        world_size=enc,
                        tolerance=dict(atol=0, rtol=0),
                        width=832,
                        height=480,
                        frames=720,
                    )
                )
            )
            rows = [
                dict(
                    frames=n,
                    pattern=p,
                    moments=dict(passed=True, max_abs=0),
                    latent=dict(passed=True, max_abs=0),
                )
                for n in range(1, 35)
                for p in ([1], [3, 2, 7, 5], [17])
            ]
            for row in rows:
                row["slices"] = [[0, 1, row["frames"]]]
                row["encoder_executions"] = [
                    dict(
                        effective_mode="parallel" if enc > 1 else "replicated",
                        fallback_reason=None,
                        group="dedicated_encoder",
                        group_ranks=list(range(enc)),
                        rank=rank,
                        world_size=enc,
                        encoder_ranks=enc,
                        participating_group_ranks=enc,
                        replicated_copies=1,
                        gather_calls=1,
                        gather_input_bytes=100,
                        gather_output_bytes=100 * enc,
                        tile_count=enc,
                        local_tile_ids=[rank],
                        output_shape=[1, 48, 1, 60, 104],
                        latent_start=0,
                        latent_end=1,
                        source_dependency_end=row["frames"],
                    )
                ]
            for row in rows:
                row["encoder_executions"][0]["encode_policy"] = dict(
                    encode_mode="performance",
                    encode_dtype="bfloat16",
                    cudnn_benchmark_limit=10,
                    cudnn_benchmark=True,
                    cudnn_deterministic=False,
                    deterministic_algorithms=False,
                )
                row["reference_encoder_executions"] = row["encoder_executions"]
            (directory / "acceptance.json").write_text(json.dumps(rows))
    validate_encoder_acceptance(tmp_path, "source")
    with pytest.raises(RuntimeError, match="different benchmark_limit"):
        validate_encoder_acceptance(tmp_path, "source", cudnn_benchmark_limit=1)
    with pytest.raises(RuntimeError, match="does not match"):
        validate_encoder_acceptance(tmp_path, "different-source")
    altered = json.loads(json.dumps(rows))
    altered[0]["encoder_executions"][0]["encode_policy"]["cudnn_benchmark_limit"] = 1
    (directory / "acceptance.json").write_text(json.dumps(altered))
    with pytest.raises(RuntimeError, match="context policies differ"):
        validate_encoder_acceptance(tmp_path, "source")
    partial = json.loads(json.dumps(rows[:1]))
    partial[0]["moments"]["passed"] = False
    for rank in range(4):
        target = tmp_path / "encoder-4" / f"rank{rank}"
        rank_rows = json.loads((target / "acceptance.json").read_text())[:1]
        rank_rows[0]["moments"]["passed"] = False
        (target / "acceptance.json").write_text(json.dumps(rank_rows))
    # Parallel/policy/source evidence remains required even when the numeric
    # matrix is incomplete and must not be labelled accepted.
    # Restore the deliberately altered rank policy before the structural check.
    (directory / "acceptance.json").write_text(json.dumps(partial))
    validate_encoder_acceptance(tmp_path, "source", sizes=(4,), require_numerical=False)
    partial[0]["encoder_executions"][0]["gather_calls"] = 0
    (directory / "acceptance.json").write_text(json.dumps(partial))
    with pytest.raises(RuntimeError, match="tile parallel"):
        validate_encoder_acceptance(
            tmp_path, "source", sizes=(4,), require_numerical=False
        )
    (directory / "acceptance.json").write_text(json.dumps(rows[:-1]))
    with pytest.raises(RuntimeError, match="does not match"):
        validate_encoder_acceptance(tmp_path, "source")


def test_client_keeps_six_references_and_original_prompt_video(tmp_path):
    frame = tmp_path / "frame.png"
    frame.touch()
    refs = []
    for i in range(1, 7):
        path = tmp_path / f"ref{i}.png"
        path.touch()
        refs.append(dict(type="image", id=f"<Picture {i}>", uri=path.as_uri()))
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            dict(
                messages=[
                    dict(content=[dict(type="text", text="Frozen prompt"), *refs])
                ]
            )
        )
    )
    config = dict(
        request=str(request),
        prompt_video_frame=str(frame),
        prompt_video_id="<Video 1>",
        source_rgb="source.rgb",
    )
    cmd = client_command(config, tmp_path / "out", paced=True)
    assert cmd[cmd.index("--batch-pattern") + 1 : cmd.index("--timeout")] == [
        "1",
        "3",
        "17",
        "2",
        "5",
    ]
    single = client_command(
        dict(config, source_batch_pattern=[1]), tmp_path / "single", paced=True
    )
    assert single[single.index("--batch-pattern") + 1 : single.index("--timeout")] == [
        "1"
    ]
    for invalid in ([], [0], [True], "1", [1.5]):
        with pytest.raises(ValueError, match="source_batch_pattern"):
            client_command(
                dict(config, source_batch_pattern=invalid), tmp_path / "bad", paced=True
            )
    for width, height in ((832, 480), (896, 512), (1280, 736)):
        sized = client_command(
            dict(config, width=width, height=height), tmp_path / "sized", paced=True
        )
        assert sized[sized.index("--width") + 1] == str(width)
        assert sized[sized.index("--height") + 1] == str(height)
    for width in (0, True, 1.5):
        with pytest.raises(ValueError, match="positive integers"):
            client_command(dict(config, width=width), tmp_path / "bad", paced=True)
    assert cmd.count("--reference") == 6
    assert "--prompt-video-frame" not in cmd
    assert "--prompt-video-id" not in cmd
    frame.unlink()
    with pytest.raises(ValueError, match="original source prompt frame"):
        client_command(config, tmp_path / "out", paced=True)


def test_client_accepts_frozen_seven_reference_case(tmp_path):
    frame = tmp_path / "frame.png"
    frame.touch()
    refs = []
    for index in range(1, 8):
        path = tmp_path / f"ref{index}.png"
        path.touch()
        refs.append(dict(type="image", id=f"<Picture {index}>", uri=path.as_uri()))
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(dict(messages=[dict(content=[dict(type="text", text="p"), *refs])]))
    )
    config = dict(
        request=str(request),
        prompt_video_frame=str(frame),
        prompt_video_id="<Video 1>",
        source_rgb="source.rgb",
        reference_ids=[f"<Picture {index}>" for index in range(1, 8)],
    )
    command = client_command(config, tmp_path / "out", paced=False)
    assert command.count("--reference") == 7


def test_soak_reports_growth_without_inventing_a_pass(tmp_path):
    (tmp_path / "provisional-selection.json").write_text(
        json.dumps(dict(layout=["enc2-sp4-overlap", 4, 2, "overlap"]))
    )
    cell = tmp_path / "enc2-sp4-overlap-paced-soak"
    measured = cell / "measured-0"
    measured.mkdir(parents=True)
    (measured / "summary.json").write_text(json.dumps(dict(received=2880)))
    events = [dict(kind="ready", message=dict(session_id="measured"))]
    events += [
        dict(
            kind="frames",
            header=dict(output_frame_start=n * 720),
            block_capture_to_output_seconds=0.1 * (n + 1),
        )
        for n in range(4)
    ]
    (measured / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    rows = [
        dict(
            session_id="measured",
            request_id=f"dispatch-{n}",
            rank=0,
            source_frames=n * 720 + 1,
            memory=dict(
                gpu_allocated_bytes=100 + n * 10,
                gpu_peak_reserved_bytes=200,
                encoder_queues=dict(pending_results=2),
            ),
        )
        for n in range(4)
    ]
    (cell / "server.log").write_text(
        "\n".join(
            "zing_source_step "
            + json.dumps(r)
            + "\nzing_source_output "
            + json.dumps(
                dict(
                    session_id="measured",
                    request_id=r["request_id"],
                    memory=dict(
                        gpu_peak_reserved_bytes=240 + n * 20,
                        pending_results=2,
                    ),
                )
            )
            for n, r in enumerate(rows)
        )
    )
    result = soak_evidence(tmp_path)
    assert result["status"] == "collected_requires_growth_review"
    assert result["windows"][3]["p95_seconds"] == 0.4
    assert result["windows"][3]["ranks"][0]["gpu_allocated_bytes"]["last"] == 130
    assert result["windows"][3]["ranks"][0]["gpu_peak_reserved_bytes"]["maximum"] == 300


@pytest.mark.parametrize("enc,sp", [(1, 1), (2, 2), (2, 4), (4, 4)])
def test_gpu_budget_tracks_selected_layout_without_expanding_it(enc, sp):
    specs = selected_layouts([f"enc{enc}-sp{sp}-overlap"])
    assert required_gpus("overlap", specs, [sp], [enc]) == enc + sp
    assert required_gpus("encoder", specs, [sp], [enc]) == enc
    assert required_gpus("transport", specs, [sp], [enc]) == enc + 1
    assert required_gpus("correctness", specs, [sp], [enc]) == enc + sp
    assert required_gpus("components", specs, [sp], [enc]) == max(enc, sp, 1)
    with pytest.raises(ValueError):
        selected_layouts([specs[0][0], specs[0][0]])


def test_dit_timing_does_not_override_numerical_failure(tmp_path):
    assert strict_dit_evidence(tmp_path, 2)["status"] == "unknown"
    for rank, passed in enumerate((True, False)):
        directory = tmp_path / "dit-sp2" / f"rank{rank}"
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text(
            json.dumps(dict(strict_replay_passed=passed))
        )
    assert strict_dit_evidence(tmp_path, 2)["status"] == "failed"
    assert analyze(tmp_path)["dit"][1]["numerical_gate"]["status"] == "failed"


def test_soak_requires_five_complete_matching_runs(tmp_path):
    spec = selected_layouts(["enc2-sp4-overlap"])[0]
    with pytest.raises(RuntimeError, match="Five paired"):
        validate_serial_overlap(tmp_path, spec)
    for mode in ("serial", "overlap"):
        for i in range(5):
            directory = tmp_path / f"enc2-sp4-{mode}-paced" / f"measured-{i}"
            directory.mkdir(parents=True)
            path = directory / "summary.json"
            row = dict(
                ended=True,
                submitted=720,
                accepted=720,
                consumed=720,
                received=720,
                output_sha256="same",
            )
            path.write_text(json.dumps(row))
    validate_serial_overlap(tmp_path, spec)
    row["output_sha256"] = "different"
    path.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match="parity failed"):
        validate_serial_overlap(tmp_path, spec)
    from run_encoder_overlap_matrix import (
        prerequisite_gate,
        validate_serial_overlap_parity,
    )

    paired = validate_serial_overlap(tmp_path, spec, require_parity=False)
    records = {}
    prerequisite_gate(
        "serial_overlap_rgb_parity",
        lambda: validate_serial_overlap_parity(paired),
        exploratory=True,
        records=records,
        path=tmp_path / "gates.json",
    )
    assert records["serial_overlap_rgb_parity"]["passed"] is False
    row["received"] = 719
    path.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match="Incomplete"):
        validate_serial_overlap(tmp_path, spec, require_parity=False)
    row["received"] = 720
    row.pop("output_sha256")
    path.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match="hash evidence is missing"):
        validate_serial_overlap(tmp_path, spec, require_parity=False)


def test_shared_tile_scope_and_decoder_cuda_attribution(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    path = traces / "sample.global-rank0.trace.json"
    common = dict(ph="X", pid=1, tid=1)
    events = [
        dict(
            common,
            ts=0,
            dur=1,
            cat="user_annotation",
            name="Zing.clock_anchor.unix_ns1000000000",
        ),
        dict(
            common,
            ts=10,
            dur=10,
            cat="user_annotation",
            name="H3ProgressiveTiles/(0, 1)",
        ),
        dict(
            common,
            ts=11,
            dur=2,
            cat="cuda_runtime",
            name="cudaLaunchKernel",
            args={"correlation": 3},
        ),
        dict(common, ts=40, dur=10, cat="user_annotation", name="Zing.source_decode"),
        dict(
            common,
            ts=41,
            dur=2,
            cat="cuda_runtime",
            name="cudaLaunchKernel",
            args={"correlation": 4},
        ),
        dict(
            common,
            ts=43,
            dur=2,
            cat="cuda_runtime",
            name="cudaLaunchKernel",
            args={"correlation": 5},
        ),
        dict(
            ph="X",
            pid=2,
            tid=2,
            ts=30,
            dur=4,
            cat="kernel",
            name="conv",
            args={"correlation": 3},
        ),
        dict(
            ph="X",
            pid=2,
            tid=2,
            ts=60,
            dur=10,
            cat="kernel",
            name="conv",
            args={"correlation": 4},
        ),
        dict(
            ph="X",
            pid=2,
            tid=2,
            ts=70,
            dur=50,
            cat="kernel",
            name="ncclBroadcast",
            args={"correlation": 5},
        ),
    ]
    path.write_text(
        json.dumps(dict(baseTimeNanoseconds=1_000_000_000, traceEvents=events))
    )
    intervals, _ = trace_intervals(path, "Zing.encoder_tiles")
    assert intervals == [(1_000_030, 1_000_034)]
    result = decoder_evidence(tmp_path)
    assert result["status"] == "collected"
    assert result["cuda_active_ms"] == 0.01
    assert result["block_p95_cuda_active_ms"] == 0.01
    assert result["eos_cost_status"].startswith("unknown")


def test_correctness_capture_defaults_to_same_sp_baseline_and_two_modes(
    tmp_path, monkeypatch
):
    import run_encoder_overlap_matrix as runner

    calls = []

    def fake_cell(root, config, spec, paced, *, capture=False):
        calls.append((spec, paced, capture))
        directory = root / spec[0]
        session = directory / "capture" / "session"
        session.mkdir(parents=True)
        (session / "COMPLETE.json").write_text("{}")
        return directory

    monkeypatch.setattr(runner, "endpoint_cell", fake_cell)
    runner.correctness_cells(
        tmp_path, {}, selected_layouts(["enc2-sp4-serial", "enc2-sp4-overlap"])
    )
    assert [row[0][0] for row in calls] == [
        "shared-sp4",
        "enc2-sp4-serial",
        "enc2-sp4-overlap",
    ]
    assert all(not paced and capture for _, paced, capture in calls)
    commands = json.loads((tmp_path / "capture-comparisons.json").read_text())[
        "commands"
    ]
    assert len(commands) == 2
    assert all(
        "--reference-rgb" in command and "--candidate-rgb" in command
        for command in commands
    )


def test_correctness_gate_requires_exact_capture_and_full_wire_rgb(tmp_path):
    from run_encoder_overlap_matrix import validate_correctness

    specs = selected_layouts(["enc2-sp4-serial", "enc2-sp4-overlap"])
    config = dict(
        model="frozen/model",
        request="frozen/request",
        source_rgb="frozen/source",
        taeh3_weights="frozen/tae",
    )
    (tmp_path / "input.json").write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="evidence is missing"):
        validate_correctness(tmp_path, specs, config)
    for name, *_ in specs:
        path = tmp_path / f"correctness-{name}.json"
        row = dict(
            status="passed",
            tolerance=dict(atol=0, rtol=0),
            wire_rgb=dict(bytes=720 * 832 * 480 * 3),
        )
        path.write_text(json.dumps(row))
    validate_correctness(tmp_path, specs, config)
    with pytest.raises(ValueError, match="inputs differ"):
        validate_correctness(tmp_path, specs, dict(config, cudnn_benchmark_limit=1))
    with pytest.raises(ValueError, match="inputs differ"):
        validate_correctness(
            tmp_path,
            specs,
            dict(
                config, source_attention_window={"window_chunks": 6, "sink_chunks": 3}
            ),
        )
    row["wire_rgb"] = "not_requested"
    path.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match="gate failed"):
        validate_correctness(tmp_path, specs, config)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_execution",
        "fallback",
        "duplicated_tiles",
        "missing_slice",
        "wrong_rank",
        "no_gather",
    ],
)
def test_encoder_execution_gate_rejects_fake_parallel(mutation):
    from encoder_execution_evidence import validate_encoder_execution

    rows = []
    for rank in range(2):
        execution = dict(
            effective_mode="parallel",
            fallback_reason=None,
            group="dedicated_encoder",
            group_ranks=[0, 1],
            rank=rank,
            world_size=2,
            encoder_ranks=2,
            participating_group_ranks=2,
            replicated_copies=1,
            gather_calls=1,
            gather_input_bytes=100,
            gather_output_bytes=200,
            tile_count=2,
            local_tile_ids=[rank],
            output_shape=[1, 48, 1, 60, 104],
            latent_start=0,
            latent_end=1,
            source_dependency_end=1,
        )
        rows.append(
            [dict(frames=1, slices=[[0, 1, 1]], encoder_executions=[execution])]
        )
    validate_encoder_execution(rows, 2)
    row = rows[1][0]
    execution = row["encoder_executions"][0]
    if mutation == "missing_execution":
        row.pop("encoder_executions")
    elif mutation == "fallback":
        execution.update(
            effective_mode="replicated",
            fallback_reason="unvalidated_encode_mode",
            encoder_ranks=1,
            replicated_copies=2,
        )
    elif mutation == "duplicated_tiles":
        execution["local_tile_ids"] = [0]
    elif mutation == "missing_slice":
        row["slices"] = []
    elif mutation == "wrong_rank":
        execution["rank"] = 0
    elif mutation == "no_gather":
        execution["gather_calls"] = 0
    with pytest.raises(RuntimeError):
        validate_encoder_execution(rows, 2)


def test_role_tile_override_does_not_change_shared_baseline():
    from run_encoder_overlap_matrix import role_environment

    key = "MINIMAX_H3_PROGRESSIVE_ENCODER_TILING"
    base = {key: "parallel"}
    assert (
        role_environment({"encoder_tile_mode": "parallel"}, "encoder", "4,5", base)[key]
        == "parallel"
    )
    assert (
        role_environment({"encoder_tile_mode": "parallel"}, "server", "0,1,2,3", base)[
            key
        ]
        == "auto"
    )
    assert (
        role_environment({"encoder_tile_mode": "parallel"}, "consumer", "0", base)[key]
        == "auto"
    )
    assert role_environment({}, "encoder", "4,5", base)[key] == "parallel"
    with pytest.raises(ValueError, match="encoder_tile_mode"):
        role_environment({"encoder_tile_mode": "invalid"}, "encoder", "0", base)


def test_limit_overlay_freezes_shared_and_remote_policy_without_mutating_base(tmp_path):
    from run_encoder_overlap_matrix import overlay

    base = tmp_path / "base"
    base.mkdir()
    original = dict(
        vae_config=dict(encode_mode="performance", encode_dtype="bfloat16"),
        dit_config={"unchanged": True},
    )
    (base / "zing_config.json").write_text(json.dumps(original))
    from prepare_encoder_overlap import digest

    frozen_profile = dict(
        model_config_sha256=digest(base / "zing_config.json"),
        solver="few_step",
        denoising_step_list=[1000, 750, 500, 250],
        noise_version="indexed-cpu-v1",
        guidance_scale=1,
        source_encoder=dict(
            mode="progressive",
            contract_version="h3-clip17-progressive-v1",
            factory="unchanged",
        ),
    )
    (base / "zing_realtime_profile.json").write_text(json.dumps(frozen_profile))
    runtimes = []
    for name, remote in (("shared", None), ("dedicated", {"port": 1})):
        out = tmp_path / name
        out.mkdir()
        model = overlay(base, out, remote, 1)
        runtimes.append(json.loads((model / "zing_config.json").read_text()))
        receipt = json.loads((out / "encoder-policy.json").read_text())
        assert receipt["requested_cudnn_benchmark_limit"] == 1
        assert receipt["original_cudnn_benchmark_limit"] is None
        profile = json.loads((model / "zing_realtime_profile.json").read_text())
        assert profile["model_config_sha256"] == digest(model / "zing_config.json")
        assert profile["model_config_sha256"] != frozen_profile["model_config_sha256"]
        for key in ("solver", "denoising_step_list", "noise_version", "guidance_scale"):
            assert profile[key] == frozen_profile[key]
        for key in ("mode", "contract_version", "factory"):
            assert (
                profile["source_encoder"][key] == frozen_profile["source_encoder"][key]
            )
    assert runtimes[0] == runtimes[1]
    assert runtimes[0]["vae_config"] == dict(
        original["vae_config"], cudnn_benchmark_limit=1
    )
    assert runtimes[0]["dit_config"] == original["dit_config"]
    assert json.loads((base / "zing_config.json").read_text()) == original


def test_exploratory_gate_records_failure_without_converting_it_to_pass(tmp_path):
    from run_encoder_overlap_matrix import prerequisite_gate

    records = {}

    def failed():
        raise RuntimeError("exact parity failed")

    prerequisite_gate(
        "numeric",
        failed,
        exploratory=True,
        records=records,
        path=tmp_path / "gates.json",
    )
    assert records["numeric"]["passed"] is False
    assert "exact parity failed" in records["numeric"]["error"]
    with pytest.raises(RuntimeError, match="exact parity"):
        prerequisite_gate(
            "numeric",
            failed,
            exploratory=False,
            records=records,
            path=tmp_path / "gates.json",
        )

    def unexpected():
        raise ValueError("invalid execution")

    with pytest.raises(ValueError, match="invalid execution"):
        prerequisite_gate(
            "numeric",
            unexpected,
            exploratory=True,
            records=records,
            path=tmp_path / "gates.json",
        )


@pytest.mark.parametrize(
    "field", ["source_rgb", "model", "cudnn_benchmark_limit", "encoder_tile_mode"]
)
def test_exploratory_never_bypasses_frozen_correctness_input_mismatch(tmp_path, field):
    from run_encoder_overlap_matrix import prerequisite_gate, validate_correctness

    config = dict(
        model="model",
        source_rgb="source",
        cudnn_benchmark_limit=1,
        encoder_tile_mode="parallel",
    )
    previous = dict(config, **{field: "different"})
    (tmp_path / "input.json").write_text(json.dumps(previous))
    records = {}
    with pytest.raises(ValueError, match="inputs differ"):
        prerequisite_gate(
            "capture_correctness",
            lambda: validate_correctness(tmp_path, [], config),
            exploratory=True,
            records=records,
            path=tmp_path / "gates.json",
        )
    assert "capture_correctness" not in records


def test_small_overlap_layouts_preserve_independent_roles():
    for enc, sp in ((1, 1), (1, 2), (2, 1), (2, 2), (2, 4)):
        for mode in ("serial", "overlap"):
            name = f"enc{enc}-sp{sp}-{mode}"
            assert selected_layouts([name]) == [(name, sp, enc, mode)]


def test_endpoint_matrix_finishes_all_formals_before_any_profile(monkeypatch, tmp_path):
    import run_encoder_overlap_matrix as runner

    calls = []

    def cell(root, config, spec, paced, *, defer_profile):
        assert defer_profile
        calls.append(("formal", spec[0], paced))
        return lambda: calls.append(("profile", spec[0], paced))

    monkeypatch.setattr(runner, "endpoint_cell", cell)
    runner.endpoint_matrix(
        tmp_path, {}, [(name, 1, 0, "shared") for name in ("a", "b", "c")]
    )
    assert [x[0] for x in calls] == ["formal"] * 6 + ["profile"] * 6
    assert [x[1:] for x in calls[:6]] == [x[1:] for x in calls[6:]]


@pytest.mark.parametrize("defer", [False, True])
def test_endpoint_completion_requires_full_profile_after_five_runs(
    monkeypatch, tmp_path, defer
):
    import contextlib
    from types import SimpleNamespace

    import run_encoder_overlap_matrix as runner

    calls = []

    @contextlib.contextmanager
    def process(command, log, env):
        log.write_text("server.session_cleanup_done sid\n")
        yield SimpleNamespace(poll=lambda: None)

    monkeypatch.setattr(runner, "process", process)
    monkeypatch.setattr(runner, "wait_server", lambda server: None)
    monkeypatch.setattr(runner, "server_command", lambda *a, **kw: [])
    monkeypatch.setattr(runner, "overlay", lambda model, out, *a, **kw: model)
    monkeypatch.setattr(runner, "client_command", lambda config, path, **kw: [path])

    def run(command, log, env):
        path = command[0]
        calls.append(path.name)
        path.mkdir()
        (path / "summary.json").write_text(
            json.dumps(
                dict(
                    ended=True, submitted=720, accepted=720, consumed=720, received=720
                )
            )
        )
        (path / "events.jsonl").write_text(
            json.dumps(dict(kind="ready", message=dict(session_id="sid")))
        )

    monkeypatch.setattr(runner, "command_run", run)
    config = dict(model=str(tmp_path), taeh3_weights="weights")
    result = runner.endpoint_cell(
        tmp_path, config, ("shared-sp1", 1, 0, "shared"), False, defer_profile=defer
    )
    out = tmp_path / "shared-sp1-unpaced"
    if defer:
        assert calls == ["warmup"] + [f"measured-{i}" for i in range(5)]
        assert (out / "FORMAL_COMPLETE.json").is_file()
        assert not (out / "PROFILE_COMPLETE.json").exists()
        assert not (out / "COMPLETE.json").exists()
        result()
        with pytest.raises(RuntimeError, match="already finished"):
            result()
    assert calls == ["warmup"] + [f"measured-{i}" for i in range(5)] + [
        "profile-warmup",
        "profile",
    ]
    assert (out / "PROFILE_COMPLETE.json").is_file()
    assert (out / "COMPLETE.json").is_file()
    with pytest.raises(FileExistsError):
        runner.endpoint_cell(tmp_path, config, ("shared-sp1", 1, 0, "shared"), False)


def test_formal_failure_does_not_start_deferred_profiles(monkeypatch, tmp_path):
    import run_encoder_overlap_matrix as runner

    profiles = []
    calls = []

    def cell(root, config, spec, paced, *, defer_profile):
        calls.append(paced)
        if paced:
            raise RuntimeError("incomplete formal output")
        return lambda: profiles.append(spec[0])

    monkeypatch.setattr(runner, "endpoint_cell", cell)
    with pytest.raises(RuntimeError, match="incomplete formal output"):
        runner.endpoint_matrix(tmp_path, {}, [("a", 1, 0, "shared")])
    assert calls == [False, True]
    assert profiles == []


@pytest.mark.parametrize("exploratory", [False, True])
def test_lifecycle_failure_isolated_from_formal_measurements(
    monkeypatch, tmp_path, exploratory
):
    import contextlib
    import subprocess
    from types import SimpleNamespace

    import run_encoder_overlap_matrix as runner

    active = []
    stopped = []
    starts = []

    @contextlib.contextmanager
    def process(command, log, env):
        if log.name.endswith("encoder.log"):
            assert not active
        child = SimpleNamespace(poll=lambda: None, name=str(log))
        active.append(child)
        starts.append(str(log))
        log.write_text("server.session_cleanup_done sid\n")
        try:
            yield child
        finally:
            active.remove(child)

    monkeypatch.setattr(runner, "process", process)
    monkeypatch.setattr(runner, "wait_server", lambda child: None)
    monkeypatch.setattr(runner, "wait_encoder", lambda child, port: None)
    monkeypatch.setattr(runner, "server_command", lambda *a, **kw: [])
    monkeypatch.setattr(runner, "overlay", lambda model, out, *a, **kw: model)
    monkeypatch.setattr(runner, "client_command", lambda config, path, **kw: [path])
    monkeypatch.setattr(
        runner,
        "stop_failed_process_groups",
        lambda children: stopped.extend(child.name for child in children),
    )

    def run(command, log, env):
        if log.name == "lifecycle.log":
            log.write_text("real probe failure evidence")
            raise subprocess.CalledProcessError(1, command)
        path = command[0]
        path.mkdir()
        (path / "summary.json").write_text(
            json.dumps(
                dict(
                    ended=True, submitted=720, accepted=720, consumed=720, received=720
                )
            )
        )
        (path / "events.jsonl").write_text(
            json.dumps(dict(kind="ready", message=dict(session_id="sid")))
        )

    monkeypatch.setattr(runner, "command_run", run)
    monkeypatch.setattr(runner, "snapshot_process_groups", lambda children: children)
    config = dict(
        model=str(tmp_path),
        taeh3_weights="weights",
        source_rgb="rgb",
        request="request",
        exploratory=exploratory,
    )
    if exploratory:
        runner.endpoint_matrix(tmp_path, config, [("enc1-sp1-serial", 1, 1, "serial")])
        formal = json.loads(
            (tmp_path / "enc1-sp1-serial-unpaced/FORMAL_COMPLETE.json").read_text()
        )
        assert formal["measurement_completed"] is True
        assert formal["lifecycle"]["passed"] is False
        assert formal["collection_status"] == "measurements_complete_lifecycle_failed"
        receipt = json.loads(
            (tmp_path / "enc1-sp1-serial-unpaced/COMPLETE.json").read_text()
        )
        assert receipt["lifecycle"]["passed"] is False
        assert receipt["performance_validated"] is False
        assert (tmp_path / "enc1-sp1-serial-paced/COMPLETE.json").is_file()
        assert len(stopped) == 2
        assert len(starts) == 8  # two roles, fresh formal/profile processes, both paces
    else:
        with pytest.raises(subprocess.CalledProcessError):
            runner.endpoint_matrix(
                tmp_path, config, [("enc1-sp1-serial", 1, 1, "serial")]
            )
        assert not (tmp_path / "enc1-sp1-serial-paced").exists()
        assert not (tmp_path / "enc1-sp1-serial-unpaced/FORMAL_COMPLETE.json").exists()
    assert not active
    gate = json.loads(
        (tmp_path / "enc1-sp1-serial-unpaced/lifecycle-gate.json").read_text()
    )
    assert gate["passed"] is False and gate["exit_code"] == 1


def test_lifecycle_gate_does_not_swallow_programming_errors(monkeypatch, tmp_path):
    import run_encoder_overlap_matrix as runner

    def fail(*args):
        raise ValueError("invalid harness configuration")

    monkeypatch.setattr(runner, "command_run", fail)
    with pytest.raises(ValueError):
        runner.lifecycle_gate([], tmp_path, {}, exploratory=True)


def test_failed_process_groups_use_cached_identities(monkeypatch):
    from types import SimpleNamespace

    import psutil
    import run_encoder_overlap_matrix as runner

    calls = []
    proc = SimpleNamespace(
        is_running=lambda: True,
        status=lambda: "running",
        terminate=lambda: calls.append("terminate"),
        kill=lambda: calls.append("kill"),
    )
    waits = iter([([], [proc]), ([proc], [])])
    monkeypatch.setattr(psutil, "wait_procs", lambda members, timeout: next(waits))
    monkeypatch.setattr(psutil, "process_iter", lambda: [])
    monkeypatch.setattr(
        runner.os, "killpg", lambda *args: pytest.fail("Must not signal reused PGID")
    )
    runner.stop_failed_process_groups(({123}, [proc]))
    assert calls == ["terminate", "kill"]


def test_unverified_group_cleanup_blocks_continuation(monkeypatch):
    from types import SimpleNamespace

    import psutil
    import run_encoder_overlap_matrix as runner

    monkeypatch.setattr(psutil, "wait_procs", lambda *a, **kw: ([], []))
    monkeypatch.setattr(
        psutil,
        "process_iter",
        lambda: [SimpleNamespace(pid=456, create_time=lambda: 99)],
    )
    monkeypatch.setattr(runner.os, "getpgid", lambda pid: 123)
    with pytest.raises(RuntimeError, match="cleanup is unverified"):
        runner.stop_failed_process_groups(({123}, []))


def test_confirmed_zombie_is_stopped_not_running(monkeypatch):
    from types import SimpleNamespace

    import psutil
    import run_encoder_overlap_matrix as runner

    zombie = SimpleNamespace(
        pid=123,
        create_time=lambda: 10,
        is_running=lambda: True,
        status=lambda: psutil.STATUS_ZOMBIE,
        terminate=lambda: pytest.fail("Zombie must not be signaled"),
    )
    monkeypatch.setattr(psutil, "wait_procs", lambda *a, **kw: ([], []))
    monkeypatch.setattr(psutil, "process_iter", lambda: [zombie])
    monkeypatch.setattr(runner.os, "getpgid", lambda pid: 123)
    receipt = runner.stop_failed_process_groups(({123}, [zombie]))
    assert receipt["confirmed_stopped"] is True
    assert receipt["zombies"][0]["status"] == "zombie"


@pytest.mark.parametrize(
    "phase,expected",
    [
        ("formal", ["warmup"] + [f"measured-{i}" for i in range(5)]),
        ("profile", ["profile-warmup", "profile"]),
    ],
)
def test_split_collection_preserves_complete_phase_and_no_combined_receipt(
    monkeypatch, tmp_path, phase, expected
):
    import contextlib
    from types import SimpleNamespace

    import run_encoder_overlap_matrix as runner

    calls = []

    @contextlib.contextmanager
    def process(cmd, log, env):
        log.write_text("server.session_cleanup_done sid\n")
        yield SimpleNamespace(poll=lambda: None)

    monkeypatch.setattr(runner, "process", process)
    monkeypatch.setattr(runner, "wait_server", lambda child: None)
    monkeypatch.setattr(runner, "server_command", lambda *a, **kw: [])
    monkeypatch.setattr(runner, "overlay", lambda model, *a, **kw: model)
    monkeypatch.setattr(runner, "client_command", lambda config, path, **kw: [path])

    def run(cmd, log, env):
        path = cmd[0]
        path.mkdir()
        calls.append(path.name)
        (path / "summary.json").write_text(
            json.dumps(
                dict(
                    ended=True, submitted=720, accepted=720, consumed=720, received=720
                )
            )
        )
        (path / "events.jsonl").write_text(
            json.dumps(dict(kind="ready", message=dict(session_id="sid")))
        )

    monkeypatch.setattr(runner, "command_run", run)
    config = dict(model=str(tmp_path), taeh3_weights="weights")
    runner.endpoint_matrix(
        tmp_path, config, [("shared-sp1", 1, 0, "shared")], collection_phase=phase
    )
    assert calls == expected * 2
    for pace in ("unpaced", "paced"):
        cell = tmp_path / f"shared-sp1-{pace}"
        assert (cell / f"{phase.upper()}_COMPLETE.json").exists()
        assert not (cell / "COMPLETE.json").exists()


def test_profile_binding_requires_same_snapshot_and_six_complete_runs(tmp_path):
    import run_encoder_overlap_matrix as runner

    binding = dict(config=dict(width=832), source_sha256="sourceA")
    (tmp_path / "endpoint-binding.json").write_text(json.dumps(binding))
    phases = ["warmup"] + [f"measured-{i}" for i in range(5)]
    for pace in ("paced", "unpaced"):
        cell = tmp_path / f"shared-sp1-{pace}"
        cell.mkdir()
        (cell / "FORMAL_COMPLETE.json").write_text(
            json.dumps(dict(phases=phases, measurement_completed=True))
        )
        for phase in phases:
            folder = cell / phase
            folder.mkdir()
            (folder / "summary.json").write_text(
                json.dumps(
                    dict(
                        ended=True,
                        submitted=720,
                        accepted=720,
                        consumed=720,
                        received=720,
                    )
                )
            )
    specs = [("shared-sp1", 1, 0, "shared")]
    runner.validate_profile_binding(tmp_path, {}, specs, binding)
    with pytest.raises(ValueError, match="source differ"):
        runner.validate_profile_binding(
            tmp_path, {}, specs, dict(binding, source_sha256="sourceB")
        )
    (tmp_path / "shared-sp1-paced/measured-4/summary.json").write_text(
        json.dumps(dict(ended=False))
    )
    with pytest.raises(ValueError, match="incomplete"):
        runner.validate_profile_binding(tmp_path, {}, specs, binding)


def test_stage_gate_reads_client_profiles_after_server_files_are_removed(tmp_path):
    from run_encoder_overlap_matrix import (
        read_stage_timing_health,
        read_stage_timing_rows,
    )

    for index in range(2):
        cell = tmp_path / f"measured-{index}"
        cell.mkdir()
        (cell / "profile.json").write_text(
            json.dumps(
                {
                    "stages": [
                        {"pid": 5, "request_id": f"session-{index}", "host_call_ms": 1}
                    ],
                    "health": [
                        {
                            "profile": {"pid": 5},
                            "written": index + 1,
                            "pending": 0,
                            "queued": 0,
                        }
                    ],
                }
            )
        )
    rows = read_stage_timing_rows(tmp_path / "timing")
    assert {row["request_id"] for row in rows} == {"session-0", "session-1"}
    assert read_stage_timing_health(tmp_path / "timing")[5]["written"] == 2
