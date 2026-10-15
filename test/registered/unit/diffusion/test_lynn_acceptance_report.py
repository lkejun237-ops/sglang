# SPDX-License-Identifier: Apache-2.0
"""Acceptance needs the planned workers and gates, including absent outputs."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "lynn_acceptance_manifest",
    ROOT / "benchmark/zing_h3_causal/lynn_acceptance_manifest.py",
)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


class TestLynnAcceptanceReport(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        (self.inputs / "inputs.json").write_text(
            json.dumps({"latents": [{"id": "lifecycle"}, {"id": "image"}]})
        )
        self.manifest = report.prepare(self.root, self.inputs, [1, 2], "lifecycle")
        self.rows = []
        # Mirror the producer contract independently of the report's matrix.
        for sp in (1, 2):
            for rank in range(sp):
                for sample in ("lifecycle", "image"):
                    for variant in (
                        "offline-vs-preclamp",
                        "finite-vs-offline",
                        "source-vs-offline",
                    ):
                        self.add_row(sp, rank, "paths", sample, variant)
                    for variant in ("preclamp", "offline", "finite"):
                        self.add_row(sp, rank, "ranks-equal", sample, variant)
                    for variant in ("decode_step", "offline"):
                        self.add_row(sp, rank, "timing", sample, variant)
                    if sp == 2 and rank == 0:
                        self.add_row(sp, rank, "vs-SP1", sample, "preclamp")
                if sp == 2:
                    self.add_row(sp, rank, "failure", None, "single-rank-failure")

    def add_row(self, sp, rank, gate, sample, variant):
        self.rows.append(
            dict(
                sp=sp,
                rank=rank,
                gate=gate,
                sample=sample,
                variant=variant,
                run_id=self.manifest["run_id"],
                ok=True,
                seconds=1,
                owned_tiles=1,
                state_bytes=1,
                peak_allocated=1,
            )
        )

    def summarize(self, rows):
        for path in self.root.glob("sp*/metrics-rank*.jsonl"):
            path.unlink()
        for row in rows:
            directory = self.root / f"sp{row['sp']}"
            directory.mkdir(exist_ok=True)
            with (directory / f"metrics-rank{row['rank']}.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
        with contextlib.redirect_stdout(io.StringIO()):
            rc = report.run_report(SimpleNamespace(output=str(self.root)))
        result = json.loads((self.root / "sp-summary.json").read_text())
        self.assertEqual(rc, 0 if result["result"] == "passed" else 1)
        return result

    def test_complete_without_optional_reference(self):
        self.assertEqual(self.summarize(self.rows)["result"], "passed")

    def test_prepare_invalidates_previous_summary(self):
        self.assertEqual(self.summarize(self.rows)["result"], "passed")
        new = report.prepare(self.root, self.inputs, [1, 2], "lifecycle")
        summary = json.loads((self.root / "sp-summary.json").read_text())
        self.assertEqual(summary["result"], "incomplete")
        self.assertEqual(summary["run_id"], new["run_id"])
        self.assertNotEqual(new["run_id"], self.manifest["run_id"])

    def test_missing_rank(self):
        result = self.summarize([r for r in self.rows if r["rank"] == 0])
        self.assertEqual(result["result"], "incomplete")
        self.assertIn([2, 1, "failure", None, "single-rank-failure"], result["missing"])

    def test_missing_ranks_equal(self):
        result = self.summarize([r for r in self.rows if r["gate"] != "ranks-equal"])
        self.assertEqual(result["result"], "incomplete")

    def test_missing_planned_degree_or_sample(self):
        for key, value in (("sp", 2), ("sample", "image")):
            with self.subTest(key=key):
                result = self.summarize([r for r in self.rows if r[key] != value])
                self.assertEqual(result["result"], "incomplete")

    def test_failed_gate(self):
        self.rows[0]["ok"] = False
        self.assertEqual(self.summarize(self.rows)["result"], "failed")

    def test_old_rows_cannot_fill_missing_rank(self):
        for row in self.rows:
            if row["rank"] == 1:
                row["run_id"] = "previous-run"
        result = self.summarize(self.rows)
        self.assertEqual(result["result"], "incomplete")
        self.assertGreater(result["stale_rows_ignored"], 0)

    def test_missing_manifest(self):
        (self.root / "acceptance-manifest.json").unlink()
        self.assertEqual(self.summarize(self.rows)["result"], "incomplete")

    def test_reference_is_required_when_requested(self):
        self.manifest["reference"] = True
        (self.root / "acceptance-manifest.json").write_text(json.dumps(self.manifest))
        self.assertEqual(self.summarize(self.rows)["result"], "incomplete")
        for sp in self.manifest["sp"]:
            for sample in self.manifest["samples"]:
                self.rows.append(
                    dict(
                        sp=sp,
                        rank=0,
                        gate="vs-R",
                        sample=sample,
                        variant="preclamp",
                        ok=True,
                        run_id=self.manifest["run_id"],
                    )
                )
        self.assertEqual(self.summarize(self.rows)["result"], "passed")


if __name__ == "__main__":
    unittest.main()
