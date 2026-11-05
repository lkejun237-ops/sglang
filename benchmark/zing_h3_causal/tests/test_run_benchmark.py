# SPDX-License-Identifier: Apache-2.0
"""Checkpoint entry contracts; no model startup or GPU allocation."""

import hashlib
import io
import json
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import run_benchmark as runner
from benchmark_artifacts import ExperimentRetention, MediaCapture, check_retention


@pytest.fixture
def model(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "zing_config.json").write_text(
        json.dumps({"causal": True, "num_train_timestep": 1000})
    )
    (root / "zing_realtime_profile.json").write_text(
        json.dumps(
            {
                "model_config_sha256": runner.bench_client.sha256_file(
                    root / "zing_config.json"
                ),
                "solver": "few_step",
                "guidance_scale": 1,
                "denoising_step_list": [1000, 750, 500, 250],
            }
        )
    )
    return root


def test_checkpoint_settings_and_mismatched_profile(model):
    sampling, deployment = runner.checkpoint_settings(model)
    assert sampling["zing_denoising_step_list"] == [1000, 750, 500, 250]
    assert sampling["num_inference_steps"] == 4
    assert deployment["model_path"] == str(model)
    (model / "zing_config.json").write_text('{"causal":false}')
    with pytest.raises(ValueError, match="inconsistent"):
        runner.checkpoint_settings(model)


def test_single_checkpoint_preflight_creates_configs_without_launch(
    model, tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "prepare_media", lambda *a: None)
    monkeypatch.setattr(runner.bench_client, "prepare_case", lambda *a: {})
    calls = []
    monkeypatch.setattr(runner.bench_client, "main", calls.append)
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("preflight launched a server"),
    )
    out = tmp_path / "run"
    runner.main(
        [
            "--case-set",
            "whitebox-basic",
            "--model-path",
            str(model),
            "--source-revision",
            "a" * 40,
            "--output-dir",
            str(out),
            "--preflight-only",
        ]
    )
    assert "--preflight-only" in calls[0]
    assert calls[0][-3:-1] == ["--source-revision", "a" * 40]
    deployment = json.loads((out / "deployment.json").read_text())
    assert deployment["sampling_schedule"] == {
        "solver": "few_step",
        "initial_block_count": 3,
        "initial_denoising_step_list": [1000, 750, 500, 250],
        "denoising_step_list": [1000, 500],
    }
    command = deployment["server_command"]
    for flag, value in (
        ("--zing-solver", "few_step"),
        ("--zing-initial-block-count", "3"),
        ("--zing-denoising-step-list", "1000,500"),
        ("--zing-initial-denoising-step-list", "1000,750,500,250"),
    ):
        assert command[command.index(flag) + 1] == value
    assert deployment["sp_degree"] == 2
    assert deployment["pipeline_config"] == {"video_decoder_backend": "native"}
    assert "--case-set" in calls[0]
    assert calls[0][calls[0].index("--case-set") + 1].endswith("whitebox-basic.json")


def test_bundled_decoder_config_resolves_checkpoint_relative_path(
    model, tmp_path, monkeypatch
):
    decoder = model / "lynn.pt"
    decoder.write_bytes(b"checkpoint")
    (model / "pipeline_config.json").write_text(
        json.dumps(
            {
                "decoder_config": {
                    "type": "lynn_streaming",
                    "checkpoint_path": "lynn.pt",
                    "model_dir": ".",
                },
            }
        )
    )
    monkeypatch.setattr(runner, "prepare_media", lambda *a: None)
    monkeypatch.setattr(runner.bench_client, "prepare_case", lambda *a: {})
    monkeypatch.setattr(runner.bench_client, "main", lambda *a: None)
    out = tmp_path / "run"
    runner.main(
        [
            "--case-set",
            "whitebox-basic",
            "--model-path",
            str(model),
            "--output-dir",
            str(out),
            "--preflight-only",
        ]
    )
    pipeline = json.loads((out / "pipeline.json").read_text())
    assert pipeline["decoder_config"]["checkpoint_path"] == str(decoder)
    assert pipeline["decoder_config"]["model_dir"] == str(model)


def test_verified_media_is_downloaded_once_and_reused(tmp_path, monkeypatch):
    import hashlib

    uri = "s3://bucket/asset"
    payload = b"verified media"
    monkeypatch.setattr(
        runner.bench_client,
        "load_cases",
        lambda *a: {runner.bench_client.CASES["5s"][0]: {}},
    )
    monkeypatch.setattr(
        runner.bench_client,
        "expected_assets",
        lambda *a: {
            uri: {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        },
    )
    calls = []

    def download(*a):
        calls.append(a)
        return io.BytesIO(payload)

    monkeypatch.setattr(runner, "open_media", download)
    first = runner.prepare_media(
        [
            (
                "5s",
                runner.bench_client.default_manifest("5s"),
                runner.bench_client.CASES["5s"][0],
            )
        ],
        tmp_path,
    )
    second = runner.prepare_media(
        [
            (
                "5s",
                runner.bench_client.default_manifest("5s"),
                runner.bench_client.CASES["5s"][0],
            )
        ],
        tmp_path,
    )
    assert (tmp_path / "asset").read_bytes() == payload
    assert len(calls) == 1
    assert first["assets"][0]["cache_state"] == "miss"
    assert second["assets"][0]["cache_state"] == "hit"
    assert second["assets"][0]["status"] == "verified"
    assert second["duration_seconds"] >= second["assets"][0]["duration_seconds"] >= 0


def test_bad_download_is_not_published_to_cache(tmp_path, monkeypatch):
    case = {"sample_id": "case"}
    monkeypatch.setattr(
        runner.bench_client,
        "load_cases",
        lambda *a: {runner.bench_client.CASES["5s"][0]: case},
    )
    monkeypatch.setattr(
        runner.bench_client,
        "expected_assets",
        lambda *a: {"s3://bucket/asset": {"bytes": 5, "sha256": "incorrect"}},
    )
    monkeypatch.setattr(runner, "open_media", lambda *a: io.BytesIO(b"wrong"))
    with pytest.raises(ValueError, match="does not match"):
        runner.prepare_media(
            [
                (
                    "5s",
                    runner.bench_client.default_manifest("5s"),
                    runner.bench_client.CASES["5s"][0],
                )
            ],
            tmp_path,
        )
    assert not (tmp_path / "asset").exists()


def test_failed_client_still_releases_owned_server(model, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "prepare_media", lambda *a: None)
    monkeypatch.setattr(runner.bench_client, "prepare_case", lambda *a: {})
    monkeypatch.setattr(runner, "wait_ready", lambda *a: None)
    process = type("Process", (), {"pid": 12345, "wait": lambda self, **k: None})()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: process)
    signals = []
    monkeypatch.setattr(
        runner.os, "killpg", lambda pid, sig: signals.append((pid, sig))
    )

    def fail(_):
        raise SystemExit(1)

    monkeypatch.setattr(runner.bench_client, "main", fail)
    with pytest.raises(SystemExit):
        runner.main(
            [
                "--case-set",
                "whitebox-basic",
                "--model-path",
                str(model),
                "--output-dir",
                str(tmp_path / "run"),
                "--port",
                "54321",
            ]
        )
    assert signals == [(12345, signal.SIGTERM)]


def test_startup_exit_and_timeout_cannot_pass(monkeypatch):
    process = type("Process", (), {"poll": lambda self: 1})()
    with pytest.raises(RuntimeError, match="exited"):
        runner.wait_ready(process, 30000, 1)
    process.poll = lambda: None
    ticks = iter([0, 2])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError):
        runner.wait_ready(process, 30000, 1)


def test_checkpoint_entry_forwards_custom_collection(model, tmp_path, monkeypatch):
    member = {
        "manifest": str(runner.bench_client.default_manifest("30s-town")),
        "case_id": runner.bench_client.CASES["30s-town"][0],
    }
    collection = tmp_path / "town.json"
    collection.write_text(json.dumps({"name": "town-only", "cases": [member]}))
    prepared, calls = [], []
    monkeypatch.setattr(
        runner, "prepare_media", lambda selected, root, *a: prepared.extend(selected)
    )
    monkeypatch.setattr(runner.bench_client, "prepare_case", lambda *a: {})
    monkeypatch.setattr(runner.bench_client, "main", calls.append)
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("preflight started a server"),
    )
    runner.main(
        [
            "--model-path",
            str(model),
            "--case-set",
            str(collection),
            "--output-dir",
            str(tmp_path / "run"),
            "--preflight-only",
        ]
    )
    assert prepared == [
        (
            "30s-town",
            runner.bench_client.default_manifest("30s-town"),
            member["case_id"],
        )
    ]
    assert calls[0][calls[0].index("--case-set") + 1] == str(collection)
    assert "--case" not in calls[0]


@pytest.mark.parametrize("mode", [None, "host", "off"])
def test_checkpoint_stage_timing_preflight_enables_client_receipt(
    model, tmp_path, monkeypatch, mode
):
    monkeypatch.setattr(runner.bench_client, "expected_assets", lambda case: {})
    output = tmp_path / "run"
    runner.main(
        [
            "--case-set",
            "whitebox-basic",
            "--model-path",
            str(model),
            *([] if mode is None else ["--stage-timing", mode]),
            "--output-dir",
            str(output),
            "--preflight-only",
        ]
    )
    config = json.loads((output / "client/config.json").read_text())
    assert config["receive_stage_timing"] is (mode != "off")
    assert [case["frames"] for case in config["cases"]] == [120, 720, 720, 720]
    receipt = json.loads((output / "media_preparation.json").read_text())
    assert config["deployment"]["media_preparation"] == receipt
    assert receipt["status"] == "verified"


@pytest.mark.parametrize("outcome", ["success", "checksum", "interrupted"])
def test_refresh_verifies_before_replacing_and_preserves_old_cache(
    tmp_path, monkeypatch, outcome
):
    import hashlib

    payload = b"last verified media"
    cached = tmp_path / "asset"
    cached.write_bytes(payload)
    inode = cached.stat().st_ino
    uri = "s3://bucket/asset"
    monkeypatch.setattr(runner.bench_client, "load_cases", lambda *a: {"case": {}})
    monkeypatch.setattr(
        runner.bench_client,
        "expected_assets",
        lambda *a: {
            uri: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        },
    )
    calls = []

    class Download(io.BytesIO):
        def read(self, size=-1):
            # Readers continue seeing the last verified file during the download.
            assert cached.read_bytes() == payload
            assert cached.stat().st_ino == inode
            if outcome == "interrupted" and self.tell():
                raise OSError("Interrupted download")
            return super().read(4 if outcome == "interrupted" else size)

    def download(*a):
        calls.append(a)
        return Download(payload if outcome != "checksum" else b"x" * len(payload))

    monkeypatch.setattr(runner, "open_media", download)
    receipt = {}
    selected = [("one", None, "case"), ("two", None, "case")]
    if outcome == "success":
        runner.prepare_media(selected, tmp_path, True, receipt)
        assert cached.stat().st_ino != inode
        assert receipt["status"] == "verified"
    else:
        with pytest.raises((ValueError, OSError)):
            runner.prepare_media(selected, tmp_path, True, receipt)
        assert cached.stat().st_ino == inode
        assert receipt["status"] == "failed"
        assert receipt["assets"][0]["status"] == "failed"
    assert cached.read_bytes() == payload
    assert list(tmp_path.iterdir()) == [cached]
    assert len(calls) == len(receipt["assets"]) == 1
    assert receipt["assets"][0]["cache_state"] == "refresh"
    assert receipt["duration_seconds"] >= receipt["assets"][0]["duration_seconds"] >= 0


def test_refresh_entry_records_failed_preparation_without_launch(
    model, tmp_path, monkeypatch
):
    def fail(selected, root, refresh, receipt):
        assert refresh is True
        receipt.update(status="failed", assets=[], duration_seconds=0.25)
        raise OSError("Interrupted download")

    monkeypatch.setattr(runner, "prepare_media", fail)
    monkeypatch.setattr(
        runner.subprocess, "Popen", lambda *a, **k: pytest.fail("launched server")
    )
    output = tmp_path / "run"
    with pytest.raises(OSError, match="Interrupted"):
        runner.main(
            [
                "--case-set",
                "whitebox-basic",
                "--model-path",
                str(model),
                "--refresh-cache",
                "--output-dir",
                str(output),
            ]
        )
    assert (
        json.loads((output / "media_preparation.json").read_text())["status"]
        == "failed"
    )


def test_comprehensive_entry_forwards_existing_tool_arguments():
    import benchmark

    value = benchmark.command(["dit", "--layers", "2", "--profile-cycle"])
    assert value[-3:] == ["--layers", "2", "--profile-cycle"]
    assert value[1].endswith("/bench_dit_phases.py")
    assert benchmark.command(["report", "--help"])[-1] == "--help"


def test_memory_checkpoint_waits_for_the_exact_worker_count(tmp_path):
    import profile_worker

    memory = tmp_path / "memory"
    memory.mkdir()
    for pid in [2, 3]:
        (memory / f"process-{pid}.json").write_text(json.dumps(dict(pid=pid)))
        (memory / f"ack-1-{pid}.json").write_text(
            json.dumps(dict(pid=pid, passed=True, label="after-request"))
        )
    values = profile_worker.checkpoint(tmp_path, "after-request", "snapshot", 2, 0.1)
    assert {v["pid"] for v in values} == {2, 3}
    assert (
        json.loads((tmp_path / "memory-control.json").read_text())["operation"]
        == "snapshot"
    )
    with pytest.raises(ValueError, match="alphanumeric"):
        profile_worker.checkpoint(tmp_path, "../unsafe", "snapshot", 2, 0.1)


@pytest.mark.parametrize("workers_remain", [False, True])
def test_block0_cache_teardown_requires_verified_worker_exit(
    model, tmp_path, monkeypatch, workers_remain
):
    from pathlib import Path

    monkeypatch.setattr(runner, "prepare_media", lambda *a: None)
    monkeypatch.setattr(runner.bench_client, "prepare_case", lambda *a: {})
    monkeypatch.setattr(runner, "wait_ready", lambda *a: None)
    monkeypatch.setattr(runner.bench_client, "main", lambda *a: None)
    process = type("Process", (), {"pid": 12345, "wait": lambda self, **k: None})()
    cache_path = None

    def launch(command, **kwargs):
        nonlocal cache_path
        cache_path = Path(command[command.index("--zing-block0-cache-dir") + 1])
        (cache_path / "kv.bin").write_bytes(bytes(4096))
        (cache_path / "manifest.json").write_text('{"dtype":"bfloat16"}')
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", launch)

    def signal_group(pid, sig):
        assert pid == process.pid
        if sig == 0 and not workers_remain:
            raise ProcessLookupError()

    monkeypatch.setattr(runner.os, "killpg", signal_group)
    command = [
        "--model-path",
        str(model),
        "--output-dir",
        str(tmp_path / "run"),
        "--case-set",
        "whitebox-basic",
        "--port",
        "54321",
        "--block0-cache-disk-gb",
        "50",
    ]
    if workers_remain:
        with pytest.raises(RuntimeError, match="Workers must be stopped"):
            runner.main(command)
        assert cache_path.is_dir() and (cache_path / "kv.bin").exists()
    else:
        runner.main(command)
        assert not cache_path.exists()
        receipt = json.loads((tmp_path / "run/cache-cleanup.json").read_text())
        assert receipt["status"] == "complete"
        assert receipt["metadata"][0]["content"] == '{"dtype":"bfloat16"}'
        assert json.loads((tmp_path / "run/retention.json").read_text())[
            "within_budget"
        ]


def test_metrics_hash_every_byte_without_raw_or_video(tmp_path):
    capture = MediaCapture(tmp_path, 2, 2, 5, "metrics")
    chunks = [bytes(range(24)), bytes(range(36))]
    for chunk in chunks:
        assert capture.write(chunk) == len(chunk)
    receipt = capture.finish()
    assert receipt["output_sha256"] == hashlib.sha256(b"".join(chunks)).hexdigest()
    assert receipt["output_frames"] == 5
    assert receipt["media_path"] is None
    assert not list(tmp_path.glob("*.rgb")) and not list(tmp_path.glob("*.mp4"))


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
def test_representative_samples_across_entire_delivery_and_encodes_afterwards(tmp_path):
    capture = MediaCapture(tmp_path, 8, 8, 240)
    raw = bytes(range(192)) * 240
    for i in range(0, len(raw), 192 * 17):
        capture.write(raw[i : i + 192 * 17])
    assert not capture.path.exists()
    receipt = capture.finish()
    assert receipt["sampled_frame_indices"][0] == 0
    assert receipt["sampled_frame_indices"][-1] == 239
    assert len(receipt["sampled_frame_indices"]) == 120
    assert receipt["output_sha256"] == hashlib.sha256(raw).hexdigest()
    assert capture.path.stat().st_size < 64 << 20
    # The pinned CPU recipe ships FFmpeg through imageio-ffmpeg, not ffprobe.
    # Fully decode the preview to validate its frame count and geometry as well.
    decoded = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(capture.path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )
    assert len(decoded) == 120 * 8 * 8 * 3
    assert not list(tmp_path.glob("*.rgb"))


def test_encoder_failure_keeps_hash_receipt_and_removes_partial_video(tmp_path):
    capture = MediaCapture(tmp_path, 2, 2, 1, ffmpeg="missing-benchmark-encoder")
    capture.write(bytes(12))
    with pytest.raises(FileNotFoundError):
        capture.finish()
    receipt = json.loads((tmp_path / "media.json").read_text())
    assert receipt["output_sha256"] == hashlib.sha256(bytes(12)).hexdigest()
    assert receipt["error"] and receipt["media_path"] is None


def test_owner_only_removes_registered_cache_after_worker_teardown(tmp_path):
    owner = ExperimentRetention(tmp_path)
    cache = owner.create_cache("scratch/block0")
    (cache / "kv.bin").write_bytes(bytes(4096))
    unrelated = tmp_path / "other-owner"
    unrelated.mkdir()
    (unrelated / "kv.bin").write_bytes(b"keep")
    try:
        with pytest.raises(RuntimeError, match="Workers must"):
            owner.finish()
        assert cache.exists()
        receipt = owner.finish(workers_stopped=True)
        assert receipt["within_budget"] and not cache.exists()
        assert (unrelated / "kv.bin").read_bytes() == b"keep"
    finally:
        owner.close()


def test_active_owner_blocks_second_process(tmp_path):
    owner = ExperimentRetention(tmp_path)
    cache = owner.create_cache("block0")
    try:
        code = "from benchmark_artifacts import ExperimentRetention; import sys; ExperimentRetention(sys.argv[1])"
        env = {
            **__import__("os").environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        }
        result = subprocess.run(
            [sys.executable, "-c", code, str(tmp_path)],
            env=env,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0 and b"BlockingIOError" in result.stderr
        assert cache.exists()
    finally:
        owner.close()


@pytest.mark.parametrize("relative", ["../foreign", "/tmp/foreign"])
def test_rejects_cache_path_escape(tmp_path, relative):
    owner = ExperimentRetention(tmp_path)
    try:
        with pytest.raises(ValueError):
            owner.create_cache(relative)
    finally:
        owner.close()


def test_refuses_existing_cache_and_symlink_replacement(tmp_path):
    owner = ExperimentRetention(tmp_path)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    try:
        with pytest.raises(FileExistsError):
            owner.create_cache("foreign")
        cache = owner.create_cache("block0")
        cache.rmdir()
        cache.symlink_to(foreign, target_is_directory=True)
        with pytest.raises(RuntimeError, match="identity changed"):
            owner.finish(workers_stopped=True)
        assert foreign.exists()
    finally:
        owner.close()


def test_refuses_link_inside_owned_cache_and_closed_lease(tmp_path):
    owner = ExperimentRetention(tmp_path)
    cache = owner.create_cache("block0")
    (cache / "external").symlink_to(tmp_path.parent)
    with pytest.raises(RuntimeError, match="link or mount"):
        owner.finish(workers_stopped=True)
    assert cache.exists()
    owner.close()
    with pytest.raises(RuntimeError, match="lease is closed"):
        owner.finish(workers_stopped=True)


def test_budget_failure_preserves_evidence_and_reports_failure(tmp_path):
    (tmp_path / "requests.jsonl").write_bytes(bytes(65536))
    with pytest.raises(RuntimeError, match="retention budget"):
        check_retention(tmp_path, budget=8192)
    assert (tmp_path / "requests.jsonl").stat().st_size == 65536
    assert not json.loads((tmp_path / "retention.json").read_text())["within_budget"]


def test_cache_metadata_is_retained_before_payload_removal(tmp_path):
    owner = ExperimentRetention(tmp_path)
    cache = owner.create_cache("block0")
    manifest = b'{"shape":[50,2,16,14,128],"dtype":"bfloat16"}'
    (cache / "manifest.json").write_bytes(manifest)
    (cache / "kv.bin").write_bytes(bytes(4096))
    try:
        owner.finish(workers_stopped=True)
    finally:
        owner.close()
    receipt = json.loads((tmp_path / "cache-cleanup.json").read_text())
    assert receipt["status"] == "complete"
    assert receipt["metadata"][0]["content"] == manifest.decode()
    assert receipt["metadata"][0]["sha256"] == hashlib.sha256(manifest).hexdigest()
    assert not cache.exists()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
def test_finite_client_default_keeps_only_first_measured_preview(tmp_path):
    import asyncio
    from types import SimpleNamespace

    import msgpack
    from bench_client import run_request
    from websockets.asyncio.server import serve

    async def exercise():
        async def handler(ws):
            await ws.recv()
            await ws.send(
                msgpack.packb(
                    {
                        "type": "frame_batch",
                        "format": "rgb24",
                        "channels": 3,
                        "width": 2,
                        "height": 2,
                        "num_frames": 2,
                        "total_size": 24,
                        "chunk_index": 0,
                        "frame_batch_index": 0,
                        "num_frame_batches": 1,
                        "is_final_frame_batch": True,
                        "request_id": "one",
                        "payload": bytes(24),
                    }
                )
            )
            await ws.close(code=1000, reason="generation complete")

        async with serve(handler, "127.0.0.1", 0) as server:
            args = SimpleNamespace(
                url=f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
                output_dir=tmp_path,
                media_mode="representative",
                timeout=5,
            )
            case = {
                "label": "case",
                "case_id": "fixture",
                "init": {"type": "init"},
                "frames": 2,
                "width": 2,
                "height": 2,
            }
            return [
                await run_request(case, args, phase, index)
                for phase, index in [("warmup", 0), ("measured", 0), ("measured", 1)]
            ]

    rows = asyncio.run(exercise())
    assert all(row["success"] for row in rows)
    assert [row["media_path"] is not None for row in rows] == [False, True, False]
    assert all(
        row["output_sha256"] == hashlib.sha256(bytes(24)).hexdigest() for row in rows
    )
    assert len(list(tmp_path.rglob("*.mp4"))) == 1
    assert not list(tmp_path.rglob("*.rgb"))
