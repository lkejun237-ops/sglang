// SPDX-License-Identifier: Apache-2.0
// Compile frozen candidate inputs with the pinned SDK, without launching games.
import {readFile, writeFile, mkdir} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';

const [planPath, outputDirectory] = process.argv.slice(2);
if (!planPath || !outputDirectory) throw Error('Expected plan path and new output directory');
const plan = JSON.parse(await readFile(planPath, 'utf8'));
const release = JSON.parse(await readFile('/app/release.json', 'utf8'));
if (release.sdk_sha !== plan.sdk_sha) throw Error('Frozen SDK identity differs from the image');
const require = createRequire('/opt/worldkit/packages/stream-host/package.json');
const {register} = await import(pathToFileURL(require.resolve('tsx/esm/api')).href);
register();
const sdk = await import(pathToFileURL(require.resolve('@worldkit/model-context')).href);
await mkdir(outputDirectory); // Never overwrite an earlier experiment.
const summary = [];
for (const item of plan.cases) {
  const raw = await readFile(resolve(item.sealed_path));
  if (createHash('sha256').update(raw).digest('hex') !== item.sealed_sha256) {
    throw Error('Frozen model-context bytes changed');
  }
  const sealed = JSON.parse(raw);
  const prepared = sdk.prepareModelContext(sealed.prepared.manifest, {texts: sealed.prepared.texts});
  if (prepared.configHash !== sealed.prepared.configHash) throw Error('SDK configuration digest differs');
  const source = {worldId: item.world_id, sourceEpoch: 'frozen-cache-validation'};
  const state = sdk.createModelContextState(prepared, source);
  const projection = {consumer: {consumerId: 'probe', consumerEpoch: 'probe-epoch', playerId: 'probe-player', viewId: 'main'}, visibleSubjects: []};
  const compiled = [{event_id: null, ...sdk.compileModelContext(prepared, sdk.projectModelContextState(prepared, state, projection))}];
  const unsupported = [];
  for (const [eventId, definition] of Object.entries(prepared.manifest.events)) {
    if (!['world', 'view'].includes(definition.scope) || Object.keys(definition.parameters).length) {
      unsupported.push({event_id: eventId, reason: 'Requires explicit subject or parameter fixture'});
      continue;
    }
    const event = {eventId, target: definition.scope === 'world' ? {scope: 'world'} : {scope: 'view', viewId: 'main'}, params: {}, lifecycle: state.lifecycle};
    if (definition.audience === 'private-player') event.receiverPlayerId = 'probe-player';
    compiled.push({event_id: eventId, ...sdk.compileModelContextCandidate(prepared, state, event, projection)});
  }
  const result = {
    ...item, sdk_sha: release.sdk_sha,
    scope: 'Offline SDK compilation only; manifest events do not prove live prewarm scheduling, gameplay or model execution',
    compiled, unsupported, images: sealed.images,
  };
  await writeFile(resolve(outputDirectory, item.world_id + '.json'), JSON.stringify(result, null, 2));
  summary.push({world_id: item.world_id, candidates: compiled.length, unsupported, sdk_sha: release.sdk_sha});
}
await writeFile(resolve(outputDirectory, 'summary.json'), JSON.stringify(summary, null, 2));
console.log(JSON.stringify(summary));
