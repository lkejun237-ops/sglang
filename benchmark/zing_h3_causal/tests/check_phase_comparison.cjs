// SPDX-License-Identifier: Apache-2.0
// Browser contract for paired statistics, compatible targets and kernel conservation.
const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const { pathToFileURL } = require("node:url");
(async () => {
  const browser = await chromium.launch({
    headless: true,
    args: ["--no-sandbox"],
  });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(pathToFileURL(process.argv[2]).href);
    const result = await page.evaluate(() => {
      const original = structuredClone(D.phase_runs[0]);
      const targets = [
        ["H20", 1],
        ["H100 目标 · L20Z", 0.5],
        ["B300 目标 · L20D", 0.25],
      ].map(([name, scale]) => {
        const r = structuredClone(original);
        r.name = name;
        Object.assign(r.config, {
          source_archive_sha256: "test-source",
          selected_weights: { test: { sha256: "test" } },
          kernel_route_envs: { test: "1" },
          device: name,
        });
        for (const sample of r.samples) {
          sample.wall_ms *= scale;
          sample.cuda_ms *= scale;
        }
        PhaseReport.validate({ phase_runs: [r] });
        r.profile = {
          schema_version: 1,
          chunks: r.summary.cycles.at(-1).chunks,
          categories: Object.fromEntries(
            ["GEMM", "Attention", "KV/RoPE", "Other"].map((k, i) => [
              k,
              { cuda_ms: (i + 1) * scale, kernels: i + 1 },
            ]),
          ),
          kernel_count: 10,
          total_kernel_ms: 10 * scale,
          output_sha256: r.runs[0].output_sha256,
          attributed_time_fraction: 1,
          attributed_kernel_fraction: 1,
        };
        return r;
      });
      const data = { phase_runs: targets };
      PhaseReport.render(data);
      window.comparisonTestData = data;
      const expected = targets.map(
        (r) => r.summary.cycles.at(-1).wall_ms.median,
      );
      const values = [
        ...document.querySelectorAll(
          "#comparison-latency-cycle .comparison-bar-row",
        ),
      ].map((e) => Number(e.dataset.value));
      const text = document.querySelector(
        "#comparison-latency-cycle",
      ).textContent;
      const pies = document.querySelectorAll(".comparison-pie").length;
      const bad = structuredClone(targets[0]);
      bad.profile.categories.GEMM.cuda_ms += 1;
      let rejected = false;
      try {
        PhaseReport.validate({ phase_runs: [bad] });
      } catch {
        rejected = true;
      }
      // A source or weights mismatch must not create a cross-target ratio.
      const mismatch = structuredClone(targets.slice(0, 2));
      mismatch[1].config.selected_weights.test.sha256 = "changed";
      PhaseReport.render({ phase_runs: mismatch });
      const disabled =
        document.querySelectorAll("#phase-comparison").length === 0;
      PhaseReport.render(data);
      return { expected, values, text, pies, rejected, disabled };
    });
    assert.deepEqual(result.values, result.expected);
    assert(
      result.text.includes("1.00×") &&
        result.text.includes("2.00×") &&
        result.text.includes("4.00×"),
    );
    assert.equal(result.pies, 3);
    assert(result.rejected);
    assert(result.disabled);
    await page.setViewportSize({ width: 390, height: 844 });
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth > window.innerWidth,
    );
    assert.equal(overflow, false, "mobile horizontal overflow");
    assert.deepEqual(errors, []);
    console.log(
      "COMPARISON_BROWSER_PASS: paired medians, ratios, pies, incompatible weights, invalid profile, mobile",
    );
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
