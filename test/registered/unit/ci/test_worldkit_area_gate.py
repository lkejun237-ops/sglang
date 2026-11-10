"""The maintained AREA gate executes real child tests without historical approvals."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, stage="base-a", runner_config="cpu")
ROOT = Path(__file__).resolve().parents[4]


def test_area_gate_runs_real_child_receipts_and_requires_aggregate_success(
    tmp_path, monkeypatch
):
    runner = ROOT / "scripts/ci/zing/run_area_tests.py"
    monkeypatch.syspath_prepend(str(runner.parent))
    spec = importlib.util.spec_from_file_location("area_runner", runner)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    roots = module.AREA_FILES
    assert roots[0].endswith("test_whitebox_area_handoff.py")
    assert {
        "test_source_prepare_contract.py",
        "test_whitebox_startup.py",
        "test_whitebox_room_cleanup.py",
        "test_whitebox_room_control.py",
        "test_whitebox_player_transport.py",
        "test_whitebox_player_media.py",
        "test_whitebox_native.py",
        "test_control_prefill.py",
        "test_whitebox_model_inputs.py",
        "test_whitebox_model_binding.py",
        "test_bound_video.py",
        "test_whitebox_source_encoding.py",
        "test_whitebox_source_sampling.py",
        "test_whitebox_live_input_archive.py",
        "test_whitebox_session_encoding.py",
        "test_whitebox_report_context.py",
        "test_whitebox_report_timeline.py",
        "test_whitebox_stream_archive.py",
        "test_whitebox_recording_process.py",
    } <= {Path(name).name for name in roots}
    assert not any(name.endswith("test_whitebox_cli_integration.py") for name in roots)
    (tmp_path / "test").mkdir()
    (tmp_path / "test/pytest.ini").write_text("[pytest]\n")
    for name in roots:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("def test_native_child():\n    assert 2 + 3 == 5\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    args = [
        sys.executable,
        str(runner),
        "--repo",
        str(tmp_path),
        "--output-dir",
        str(tmp_path / "receipts"),
    ]
    result = subprocess.run(args, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((tmp_path / "receipts/receipt.json").read_text())
    assert len(receipt["results"]) == 2 * len(roots)
    assert all(
        row["passed"] == 1 and row["collected"] == 1 and row["skipped"] == 0
        for row in receipt["results"]
    )
    (tmp_path / roots[0]).write_text("def test_native_child():\n    assert False\n")
    result = subprocess.run(args, capture_output=True, text=True, env=env)
    assert result.returncode == 1
    assert (
        json.loads((tmp_path / "receipts/receipt.json").read_text())["status"]
        == "failed"
    )
    import yaml

    jobs = yaml.safe_load((ROOT / ".github/workflows/zing-ci.yml").read_text())["jobs"]
    assert "area-contracts" in jobs["zing-ci"]["needs"]
    area_steps = jobs["area-contracts"]["steps"]
    build_index = next(
        i
        for i, step in enumerate(area_steps)
        if "docker build --target rtc-builder" in step.get("run", "")
    )
    execute_index = next(
        i
        for i, step in enumerate(area_steps)
        if "python scripts/ci/zing/run_area_tests.py" in step.get("run", "")
    )
    assert build_index < execute_index
    assert "WHITEBOX_TEST_RTC_BINARY=" in area_steps[build_index]["run"]
    assert "$GITHUB_ENV" in area_steps[build_index]["run"]
    assert any(
        "needs.area-contracts.result" in step.get("env", {}).get("AREA_RESULT", "")
        for step in jobs["zing-ci"]["steps"]
    )
