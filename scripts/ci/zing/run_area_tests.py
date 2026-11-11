"""Execute maintained AREA owners; historical CPU approval records stay immutable."""

import argparse
from pathlib import Path

from common import execute_pytest, provenance, write_json

AREA_ROOT = "python/sglang/multimodal_gen/test/unit/realtime"
AREA_FILES = [
    f"{AREA_ROOT}/{name}"
    for name in (
        "test_whitebox_area_handoff.py",
        "test_whitebox_rtc_pipeline.py",
        "test_whitebox_source_encoding.py",
        "test_whitebox_session_encoding.py",
        "test_whitebox_cli_lifetime.py",
        "test_whitebox_render_contract.py",
        "test_whitebox_inference_control.py",
        "test_whitebox_sink_control.py",
        "test_condition_refresh.py",
        "test_whitebox_sdk_control_relay.py",
        "test_whitebox_render_lifecycle.py",
        "test_whitebox_source_sampling.py",
        "test_whitebox_source_flow.py",
        "test_whitebox_native_source_cadence.py",
        "test_whitebox_live_input_archive.py",
        "test_whitebox_room_pipeline.py",
        "test_whitebox_rtc_lifecycle.py",
        "test_whitebox_render_reports.py",
        "test_whitebox_report_context.py",
        "test_whitebox_report_timeline.py",
        "test_whitebox_stream_archive.py",
        "test_whitebox_recording_process.py",
        "test_source_prepare_contract.py",
        "test_whitebox_startup.py",
        "test_whitebox_source_conditions.py",
        "test_whitebox_prewarm_observation.py",
        "test_whitebox_startup_prewarm.py",
        "test_whitebox_opening_presentation.py",
        "test_whitebox_context_intervention.py",
        "test_whitebox_receipt_delta.py",
        "test_whitebox_ui_history.py",
        "test_whitebox_error_origins.py",
        "test_whitebox_model_geometry.py",
        "test_whitebox_resolution.py",
        "test_realtime_coordinator.py",
        "test_realtime_coordinator_server.py",
        "test_realtime_gateway_server.py",
        "test_whitebox_room_cleanup.py",
        "test_whitebox_room_control.py",
        "test_whitebox_player_transport.py",
        "test_whitebox_player_media.py",
        "test_whitebox_native.py",
        "test_whitebox_player_control.py",
        "test_whitebox_owner_retirement.py",
        "test_control_prefill.py",
        "test_source_pause.py",
        "test_whitebox_model_inputs.py",
        "test_whitebox_capture_wait.py",
        "test_whitebox_player_idle.py",
        "test_whitebox_model_binding.py",
        "test_whitebox_cache_receipts.py",
        "test_bound_video.py",
        "test_player_lifecycle.py",
        "test_player_room_lifecycle.py",
        "test_player_cleanup_retention.py",
    )
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/area-contracts")
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    receipt = {"status": "failed", "results": []}
    try:
        receipt.update(provenance(repo, paths=AREA_FILES))
        for index, path in enumerate(AREA_FILES):
            if not (repo / path).is_file():
                raise ValueError("Missing AREA test owner: " + path)
            for trial in (1, 2):
                receipt["results"].append(
                    execute_pytest(
                        repo,
                        path,
                        [f"--confcutdir={repo / AREA_ROOT}", repo / path],
                        args.output_dir,
                        f"area-{index}",
                        timeout=500,
                        trial=trial,
                        required_files=[path],
                    )
                )
        if all(row["status"] == "CPU_PASS" for row in receipt["results"]):
            receipt["status"] = "success"
    except (ValueError, OSError) as error:
        receipt["error"] = str(error)
    write_json(args.output_dir / "receipt.json", receipt)
    return 0 if receipt["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
