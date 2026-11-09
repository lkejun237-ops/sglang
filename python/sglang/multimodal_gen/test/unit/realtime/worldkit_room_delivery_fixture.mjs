// SPDX-License-Identifier: Apache-2.0
// Real SDK compilation, playtest and sealing; owns its local fixture until stdin closes.
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import {readFile,writeFile} from 'node:fs/promises';
import {appendFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import path from 'node:path';

const require = createRequire(path.join(process.env.WHITEBOX_TEST_CLI_SDK, 'packages/cli/bin/worldkit.mjs'));
const sdk = async name => import(pathToFileURL(require.resolve(name)).href);

// Test-only observations are installed before SDK imports can launch a page.
// Keep diagnostics outside the world directory so fixture cleanup cannot erase
// the evidence. Never record descriptors, arguments, headers, query or DOM text.
const diagnosticFile = process.env.WHITEBOX_TEST_BROWSER_DIAGNOSTICS;
const diagnosticCounts = {};
const diagnosticLimits = {events:256, urlBytes:512, entryBytes:4096, totalBytes:65536};
let diagnosticEvents = 0, diagnosticPages = 0, diagnosticBytes = 0;
const safeUrl = value => {
  try {
    const url = new URL(value);
    if (url.protocol !== 'http:' && url.protocol !== 'https:') return '<unsupported-url>';
    // HTTP URL serialization is ASCII (punycode host and percent-encoded path).
    return `${url.protocol}//${url.host}${url.pathname}`.slice(0, diagnosticLimits.urlBytes);
  }
  catch { return '<invalid-url>'; }
};
const safeError = value => String(value)
  .replace(/https?:\/\/[^\s"'<>]+/g, safeUrl)
  .replace(/(?:Bearer\s+|worldkit\.cap\.)[A-Za-z0-9_.~+\/-]+/gi, '<credential>')
  .replace(/(["']?(?:accessToken|token|credential|secret|authorization)["']?\s*[:=]\s*)["']?[^\s,"'}]+/gi, '$1<credential>')
  .replace(/[A-Za-z0-9_-]{32,}/g, '<opaque>')
  .slice(0, 1600);
const diagnostic = (event, fields = {}) => {
  if (!diagnosticFile || diagnosticEvents >= diagnosticLimits.events) return;
  try {
    const line = JSON.stringify({schemaVersion:1, event, ...fields}) + '\n';
    const bytes = Buffer.byteLength(line);
    if (bytes > diagnosticLimits.entryBytes || diagnosticBytes + bytes > diagnosticLimits.totalBytes) return;
    appendFileSync(diagnosticFile, line, {mode:0o600});
    diagnosticBytes += bytes;
    diagnosticEvents++;
    diagnosticCounts[event] = (diagnosticCounts[event] ?? 0) + 1;
  } catch { /* An unavailable diagnostic file must not change the tested behavior. */ }
};
const observedError = error => ({
  message: safeError(error?.message ?? error),
  messageHash: createHash('sha256').update(String(error?.message ?? error)).digest('hex'),
});
const rpcMethods = new Set(['describeContent', 'acquire', 'progress', 'close']);
let rpcCallId = 0;
const observeAuthorityRpc = page => {
  const evaluate = page.evaluate;
  page.evaluate = (...args) => {
    let method;
    try {
      const candidate = args[1]?.method;
      if (rpcMethods.has(candidate)) method = candidate;
    }
    catch { /* Diagnostic inspection cannot prevent the original evaluate. */ }
    if (!method) return Reflect.apply(evaluate, page, args);
    const callId = ++rpcCallId, started = performance.now();
    const trace = stage => diagnostic('authority-rpc', {
      method, stage, callId, durationMilliseconds:Math.max(0, performance.now() - started),
    });
    trace('start');
    try {
      const operation = Reflect.apply(evaluate, page, args);
      // Observe settlement without retaining arguments or replacing the native promise.
      void operation.then(() => trace('finish'), () => trace('error')).catch(() => {});
      return operation;
    } catch (error) {
      trace('error');
      throw error;
    }
  };
};
if (diagnosticFile) {
  if (!path.isAbsolute(diagnosticFile)) throw Error('LOCAL_ROOM_DIAGNOSTIC_PATH_INVALID');
  const {chromium} = require('playwright');
  const launch = chromium.launch.bind(chromium);
  chromium.launch = async (...args) => {
    const browser = await launch(...args);
    diagnostic('browser-launched');
    const newContext = browser.newContext.bind(browser);
    browser.newContext = async (...args) => {
      const context = await newContext(...args);
      diagnostic('context-created');
      context.on('page', page => {
        const pageId = ++diagnosticPages;
        diagnostic('page-created', {pageId});
        observeAuthorityRpc(page);
        page.on('pageerror', error => diagnostic('pageerror', {pageId, ...observedError(error)}));
        page.on('console', message => {
          if (message.type() === 'error') diagnostic('console-error', {pageId, ...observedError(message.text())});
        });
        page.on('requestfailed', request => diagnostic('requestfailed', {
          pageId, url:safeUrl(request.url()), method:request.method(), ...observedError(request.failure()?.errorText ?? 'unknown'),
        }));
        page.on('response', response => {
          if (response.status() >= 400) diagnostic('http-error', {pageId, url:safeUrl(response.url()), status:response.status()});
        });
        page.on('framenavigated', frame => {
          if (frame === page.mainFrame()) diagnostic('navigation', {pageId, url:safeUrl(frame.url())});
        });
        page.on('domcontentloaded', () => diagnostic('dom-ready', {pageId, url:safeUrl(page.url())}));
        const waitForFunction = page.waitForFunction.bind(page);
        page.waitForFunction = async (...args) => {
          try { return await waitForFunction(...args); }
          catch (error) {
            diagnostic('function-wait-failed', {pageId, url:safeUrl(page.url()), ...observedError(error)});
            throw error;
          }
        };
      });
      return context;
    };
    return browser;
  };
  diagnostic('hook-installed');
}
const {compiledRoomFixture} = await sdk('@worldkit/stream-host/testing/creator-room');
const {ThreeCreatorTools} = await sdk('@worldkit/creator-host/tools');
const {createCreatorMultiplayerRuntime} = await sdk('@worldkit/stream-host/creator-runtime');
let fixture, tools;
try {
  fixture = await compiledRoomFixture('courtyard', {modelContext: process.env.WHITEBOX_TEST_MODEL_CONTEXT === '1'});
  if (process.env.WHITEBOX_TEST_MODEL_CONTEXT === '1') {
    // Use a real sealed PNG large enough for the model non-upscaling buckets.
    const image = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAIAAABMXPacAAABLUlEQVR4nO3TMREAMAwDsST8ObcwtLwIePjzvol0dD0F0HoAVgCsAFgBsAJgBcAKgBUAKwBWAKwAWAGwAmAFwAqAFQArAFYArABYAbACYAXACoAVACsAVgCsAFgBsAJgBcAKgBUAKwBWAKwAWAGwAmAFwAqAFQArAFYArABYAbACYAXACoAVACsAVgCsAFgBsAJgBcAKgBUAKwBWAKwAWAGwAmAFwAqAFQArAFYArABYAbACYAXACoAVACsAVgCsAFgBsAJgBcAKgBUAKwBWAKwAWAGwAmAFwAqAFQArAFYArABYAbACYAXACoAVACsAVgCsAFgBsAJgBcAKgBUAKwBWAKwAWAGwAmAFwAqAFQArAFYArABYAbACYAXACoAVACsAVgCsAFgBsAKM9QHP4QH/xCXf5QAAAABJRU5ErkJggg==', 'base64');
    const manifestFile = path.join(fixture.directory, 'model/context.json');
    const manifest = JSON.parse(await readFile(manifestFile, 'utf8'));
    manifest.assets.landmark = {path:'landmark.png', format:'png', sha256:createHash('sha256').update(image).digest('hex')};
    manifest.baseReferences = ['landmark'];
    manifest.slots.feedback = {scope:'world',initial:'pending',choices:{pending:{text:'Waiting for model feedback.'},confirmed:{text:'SDK confirmed later model application.'}}};
    manifest.composition.order.push('feedback');
    manifest.events.later_open = {...manifest.events.open, transition:{...manifest.events.open.transition,onset:'A later opening after initial model application.'}};
    manifest.events.confirm = {kind:'state',scope:'world',audience:'all',changes:[{slot:'feedback',select:'confirmed'}]};
    const usageFile = path.join(fixture.directory, 'model/usage.json');
    const usage = JSON.parse(await readFile(usageFile, 'utf8'));
    usage.composition.order.push('feedback');
    usage.slots.feedback = {scope:'world',initial:'pending',resetOn:'lifecycle',choices:{pending:{parameters:{}},confirmed:{parameters:{}}}};
    usage.events.later_open = {...usage.events.open};
    usage.events.confirm = {...manifest.events.confirm,endBehavior:'keep-state'};
    await writeFile(usageFile, JSON.stringify(usage));
    await writeFile(path.join(fixture.directory, 'model/landmark.png'), image);
    await writeFile(path.join(fixture.directory, 'model/base.txt'), 'Two player courtyard. Landmark reference {{ref:landmark}}.');
    await writeFile(manifestFile, JSON.stringify(manifest));
  }
  // The SDK's c/d sample layout is only a fixed test roster. This consumer
  // admits random trusted actor IDs, so authored members allocate free spawn
  // positions instead of sending every unfamiliar ID to the same location.
  const contentFile = path.join(fixture.directory, 'content.ts');
  const original = await readFile(contentFile, 'utf8');
  const marker = "character.root.position.set(['c','player-1'].includes(admission.actorId)?-4:4,.03,0);";
  const replacements = [marker, 'return {world,', 'dispose:()=>character.dispose()', 'catch(error){character.dispose();throw error;}'];
  if (replacements.some(value => original.split(value).length !== 2)) throw Error('LOCAL_ROOM_SPAWN_FIXTURE_CHANGED');
  const content = original
    .replace('return {world,', 'const occupied = new Map(); return {world,')
    .replace(marker, `const spawn = [-4,4].find(value => ![...occupied.values()].includes(value));
      if (spawn === undefined) throw Error('LOCAL_ROOM_SPAWN_FULL');
      occupied.set(admission.actorId, spawn); character.root.position.set(spawn,.03,0);`)
    .replace('dispose:()=>character.dispose()', 'dispose:()=>{occupied.delete(admission.actorId);character.dispose();}')
    .replace('catch(error){character.dispose();throw error;}', 'catch(error){occupied.delete(admission.actorId);character.dispose();throw error;}');
  let verifiedContent = content;
  if (process.env.WHITEBOX_TEST_MODEL_CONTEXT === '1') {
    const marker = 'let receivers=0;';
    if (verifiedContent.split(marker).length !== 2) throw Error('LOCAL_ROOM_FEEDBACK_FIXTURE_CHANGED');
    // The real SDK applied observer drives the next intent outside its read-only callback.
    verifiedContent = verifiedContent.replace(marker, `let receivers=0,laterId,started=false,confirmed=false;
      conditions.subscribe(receipt=>{
        if(receipt.phase!=='applied')return;
        if(!started){started=true;queueMicrotask(()=>{const result=conditions.trigger('later_open');if(!result.accepted)throw Error(result.error.code);laterId=result.occurrenceId;});}
        else if(!confirmed&&laterId&&receipt.occurrenceIds?.includes(laterId)){confirmed=true;queueMicrotask(()=>{const result=conditions.trigger('confirm');if(!result.accepted)throw Error(result.error.code);});}
      });`);
  }
  await writeFile(contentFile, verifiedContent);
  tools = new ThreeCreatorTools(fixture.directory, 'three-sdk', {
    taskParameters: {schemaVersion: 1, playMode: 'multiplayer', multiplayer: {maxPlayers: 2}},
    multiplayerRuntime: createCreatorMultiplayerRuntime({players: ['c', 'd'], width: 160, height: 90, fps: 12, browser: {rendering: 'software'}}),
  });
  await writeFile(path.join(fixture.directory, 'episode.json'), JSON.stringify({
    schemaVersion: 3, playMode: 'multiplayer', durationSeconds: 1,
    tracks: ['c', 'd'].map(playerId => ({playerId,
      steps: [{atSeconds: 0, durationSeconds: .2, keys: ['KeyW']}],
      assertions: [{kind: 'state', scope: 'ui', key: 'secret', equals: playerId}],
    })),
  }));
  const operation = tools.start('world.playtest', id => tools.playtest(id, undefined, 2));
  let report;
  do { report = await tools.getOperation(operation.operationId, 25); }
  while (report.status === 'running');
  if (report.result?.status !== 'passed') {
    throw Error('LOCAL_ROOM_PLAYTEST_FAILED: ' + JSON.stringify({error:report.error,failure:report.result?.failure}));
  }
  const receipt = await tools.submit();
  await tools.close();
  const identity = Object.fromEntries(['archiveSha256', 'worldBuildHash', 'runtimeHash', 'sourceHash'].map(key => [key, receipt[key]]));
  process.stdout.write(JSON.stringify({
    archive: receipt.archivePath, receipt, identity, directory: fixture.directory,
    reference: {platformUrl:'http://127.0.0.1',sourceId:'local-fixture',worldId:'room-fixture',attempt:1},
  }) + '\n');
  // Python sends EOF after closing its Room and verifying all descendants exited.
  for await (const _ of process.stdin) { /* lifetime owned by the test */ }
} finally {
  diagnostic('fixture-finishing', {counts:diagnosticCounts});
  await tools?.close();
  await fixture?.dispose();
}
