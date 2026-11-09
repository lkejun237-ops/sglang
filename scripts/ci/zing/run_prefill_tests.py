"""Collect fresh CPU evidence for newly integrated Prefill contract owners."""

import argparse
from pathlib import Path

from common import execute_pytest, provenance, write_json
from run_cpu_tests import ordinary_recipe

ROOTS = [
    "python/sglang/multimodal_gen/test/unit/realtime/test_zing_paged_cache.py",
    "python/sglang/multimodal_gen/test/unit/realtime/test_player_cleanup_sp.py",
    "python/sglang/multimodal_gen/test/unit/realtime/test_output_materialization.py",
    "python/sglang/multimodal_gen/test/unit/conditioning/test_prompt_token_budget.py",
    "python/sglang/multimodal_gen/test/unit/realtime/test_zing_disk_cache_io.py",
    "python/sglang/multimodal_gen/test/unit/realtime/test_zing_prefill_persistence.py",
    "python/sglang/multimodal_gen/test/unit/realtime/test_zing_h3_source_adapter.py",
    "python/sglang/multimodal_gen/test/unit/realtime/test_zing_progressive_source_session.py",
    "python/sglang/multimodal_gen/test/unit/realtime/test_zing_prefill_api.py",
    "python/sglang/multimodal_gen/test/unit/realtime/test_bound_video.py",
]


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = Path.cwd().resolve()
    receipt = {"status": "failed", "results": []}
    try:
        manifest = repo / "scripts/ci/zing/cpu_test_manifest.json"
        payload = json.loads(manifest.read_text())
        recipe = payload["recipes"]["dit-pipeline-contracts"]
        receipt["environment_recipe"] = ordinary_recipe(repo, recipe)
        receipt.update(provenance(repo, manifest=manifest, paths=ROOTS))
        for index, path in enumerate(ROOTS):
            for trial in (1, 2):
                receipt["results"].append(
                    execute_pytest(
                        repo,
                        path,
                        [repo / path],
                        args.output_dir,
                        f"prefill-{index}",
                        trial=trial,
                        timeout=500,
                        validated_cpu_recipe=True,
                        required_files=[path],
                    )
                )
        if all(row["status"] == "CPU_PASS" for row in receipt["results"]):
            receipt["status"] = "success"
    except Exception as error:
        receipt["error"] = str(error)
    write_json(args.output_dir / "receipt.json", receipt)
    return 0 if receipt["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
