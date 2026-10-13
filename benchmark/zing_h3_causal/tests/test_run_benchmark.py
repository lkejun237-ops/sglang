# SPDX-License-Identifier: Apache-2.0
"""Checkpoint entry contracts; no model startup or GPU allocation."""

import io
import json
import signal

import pytest
import run_benchmark as runner


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
    assert deployment["sp_degree"] == 2
    assert deployment["pipeline_config"] == {"video_decoder_backend": "native"}
    assert "all" in calls[0]


def test_bundled_decoder_config_resolves_checkpoint_relative_path(
    model, tmp_path, monkeypatch
):
    decoder = model / "taeh3.pth"
    decoder.write_bytes(b"checkpoint")
    (model / "pipeline_config.json").write_text(
        json.dumps(
            {
                "video_decoder_backend": "taeh3",
                "taeh3_checkpoint_path": "taeh3.pth",
            }
        )
    )
    monkeypatch.setattr(runner, "prepare_media", lambda *a: None)
    monkeypatch.setattr(runner.bench_client, "prepare_case", lambda *a: {})
    monkeypatch.setattr(runner.bench_client, "main", lambda *a: None)
    out = tmp_path / "run"
    runner.main(
        ["--model-path", str(model), "--output-dir", str(out), "--preflight-only"]
    )
    pipeline = json.loads((out / "pipeline.json").read_text())
    assert pipeline["taeh3_checkpoint_path"] == str(decoder)


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


def test_checkpoint_stage_timing_preflight_enables_client_receipt(
    model, tmp_path, monkeypatch
):
    monkeypatch.setattr(runner.bench_client, "expected_assets", lambda case: {})
    output = tmp_path / "run"
    runner.main(
        [
            "--model-path",
            str(model),
            "--stage-timing",
            "cuda",
            "--output-dir",
            str(output),
            "--preflight-only",
        ]
    )
    config = json.loads((output / "client/config.json").read_text())
    assert config["receive_stage_timing"] is True
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
            ["--model-path", str(model), "--refresh-cache", "--output-dir", str(output)]
        )
    assert (
        json.loads((output / "media_preparation.json").read_text())["status"]
        == "failed"
    )
