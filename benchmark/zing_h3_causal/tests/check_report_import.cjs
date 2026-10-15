// SPDX-License-Identifier: Apache-2.0
// Development browser regression. Run with Playwright installed and a generated HTML/JSON pair.
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {pathToFileURL} = require('node:url');

(async () => {
  const report = path.resolve(process.argv[2]);
  const bundle = JSON.parse(fs.readFileSync(report.replace(/\.html$/, '.json'), 'utf8'));
  const data = bundle.data;
  const original = Object.keys(data.modes)[0];
  const attack = '<img src=x onerror="window.__reportImportInjected=1">';
  data.modes[attack] = data.modes[original];
  delete data.modes[original];
  data.runs.find(run => run.name === original).name = attack;
  data.modes[attack].summary.warmup_failed = attack;
  for (const chunk of data.modes[attack].chunks) {
    for (const field of ['delivery_index', 'chunk_index', 'frame_batch_index', 'payload_bytes']) {
      chunk[field] = attack;
    }
  }
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'zing-report-import-'));
  const file = path.join(temporary, 'hostile.json');
  fs.writeFileSync(file, JSON.stringify({format: bundle.format, version: bundle.version, data}));
  const browser = await chromium.launch({headless: true, args: ['--no-sandbox']});
  try {
    const page = await browser.newPage();
    const errors = [], unexpectedRequests = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => {
      if (request.url() !== pathToFileURL(report).href && !request.url().startsWith('blob:')) {
        unexpectedRequests.push(request.url());
      }
    });
    await page.goto(pathToFileURL(report).href);
    // Legacy CUDA fields do not hide Host measurements, regardless of request order.
    const mixed = JSON.parse(JSON.stringify(bundle.data));
    const mixedName = Object.keys(mixed.modes)[0], mode = mixed.modes[mixedName];
    assert(mode.requests.length >= 2 && mode.spans.some(s => s.stage === 'DiT'));
    const missing = mode.requests[1].request_id;
    const missingDispatches = new Set(mode.chunks.filter(c => c.request_id === missing).map(c => c.server_request_id));
    for (const span of mode.spans.filter(s => s.stage === 'DiT')) {
      span.cuda_status = missingDispatches.has(span.dispatch_id) ? 'not_requested' : 'complete';
      span.cuda_stream_ms = missingDispatches.has(span.dispatch_id) ? null : 12.5;
    }
    const mixedFile = path.join(temporary, 'mixed-cuda.json');
    for (const missingFirst of [false, true]) {
      mode.requests.sort((a, b) => (a.request_id === missing ? 1 : 0) - (b.request_id === missing ? 1 : 0));
      if (missingFirst) mode.requests.reverse();
      fs.writeFileSync(mixedFile, JSON.stringify({format: bundle.format, version: bundle.version, data: mixed}));
      const count = await page.locator('#result-file option').count();
      await page.locator('#open-results').setInputFiles(mixedFile);
      await page.waitForFunction(n => document.querySelectorAll('#result-file option').length === n + 1, count);
      await page.selectOption('#result-file', String(count));
      const value = await page.evaluate(() => [...document.querySelectorAll('#stage-table tr')]
        .find(row => row.children[1].textContent === '0' && row.children[2].textContent === 'DiT').children[3].textContent);
      assert(Number.isFinite(Number(value)) && Number(value) >= 0);
    }
    const before = await page.locator('#result-file option').count();
    await page.locator('#open-results').setInputFiles(file);
    await page.waitForFunction(count => document.querySelectorAll('#result-file option').length === count + 1, before);
    for (const selector of ['#stage-table', '#raw-calculation', '#raw-chunks', '#client-table']) {
      assert((await page.locator(selector).textContent()).includes(attack), selector);
      assert.equal(await page.locator(selector + ' img').count(), 0, selector);
    }
    assert.equal(await page.evaluate(() => window.__reportImportInjected), undefined);
    assert.deepEqual(errors, []);
    assert.deepEqual(unexpectedRequests, []);
    console.log('PASS: legacy CUDA availability does not affect Host results in either request order; imported markup stays literal with no resource requests');
  } finally {
    await browser.close();
    fs.rmSync(temporary, {recursive: true});
  }
})().catch(error => {console.error(error);process.exitCode = 1;});
