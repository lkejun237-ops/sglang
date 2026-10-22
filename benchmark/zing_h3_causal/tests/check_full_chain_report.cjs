// SPDX-License-Identifier: Apache-2.0
// Run with Playwright and a report built from results/issue334-full-chain-20261019.json.
const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const { pathToFileURL } = require("node:url");
const path = require("node:path");

(async () => {
  const browser = await chromium.launch({
    headless: true,
    args: ["--no-sandbox"],
  });
  try {
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1100 },
    });
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(pathToFileURL(path.resolve(process.argv[2])).href);
    assert(await page.locator("#full-content").isVisible());
    const options = await page
      .locator("#full-case option")
      .evaluateAll((es) => es.map((e) => e.value));
    for (const value of options) {
      await page.selectOption("#full-case", value);
      for (const fig of await page.locator(".component-pie").all()) {
        const percentages = await fig
          .locator("figcaption strong")
          .allTextContents();
        assert(
          Math.abs(percentages.reduce((s, v) => s + parseFloat(v), 0) - 100) <
            0.3,
        );
      }
    }
    await page.selectOption("#full-case", "5s");
    assert.equal(await page.locator(".component-pie").count(), 3);
    assert.equal(await page.locator("#full-content figure img").count(), 3);
    const download = page.waitForEvent("download");
    await page.click("#export-results");
    const data = JSON.parse(
      fs.readFileSync(await (await download).path(), "utf8"),
    ).data;
    for (const target of data.targets)
      for (const c of target.cases)
        for (const r of c.requests) {
          const m = r.metrics;
          assert.equal(
            m.post_ttff_fps,
            (r.frames - m.first_batch_frames) /
              (m.e2e_seconds - m.ttff_seconds),
          );
        }
    await page.evaluate((data) => {
      const original = data.targets[0].cases[0].client.e2e_seconds.median;
      data.targets[0].cases[0].client.e2e_seconds.median = -100;
      FullChainReport.validate(data);
      if (data.targets[0].cases[0].client.e2e_seconds.median !== original)
        throw Error("Cached client statistics were trusted");
      const expected =
        data.targets[0].cases[0].requests[0].metrics.post_ttff_fps;
      data.targets[0].cases[0].requests[0].metrics.post_ttff_fps = 1e9;
      FullChainReport.validate(data);
      if (
        data.targets[0].cases[0].requests[0].metrics.post_ttff_fps !== expected
      )
        throw Error("Cached post-TTFF FPS was trusted");
      const bad = structuredClone(data);
      bad.targets[0].operator_profile.total_kernel_ms += 1e6;
      let rejected = false;
      try {
        FullChainReport.validate(bad);
      } catch {
        rejected = true;
      }
      if (!rejected) throw Error("Invalid operator totals were accepted");
      const legacy = structuredClone(data);
      for (const t of legacy.targets)
        for (const c of t.cases)
          for (const r of c.requests) {
            delete r.metrics.first_batch_frames;
            delete r.metrics.post_ttff_fps;
          }
      FullChainReport.validate(legacy);
      if (legacy.targets[0].cases[0].client.post_ttff_fps !== null)
        throw Error("Missing first batch was fabricated");
    }, data);
    await page.setViewportSize({ width: 390, height: 844 });
    assert(
      await page
        .locator(".component-pie")
        .first()
        .evaluate((e) => e.scrollWidth <= e.clientWidth + 1),
    );
    assert.deepEqual(errors, []);
    console.log(
      "PASS: full-chain scenes, pies, export, post-TTFF FPS, recomputation, legacy and mobile",
    );
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
