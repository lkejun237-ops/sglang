// SPDX-License-Identifier: Apache-2.0
// Test-owned local WTP delivery. Only the compiler is imported from the SDK's
// public exports; SDK-private test paths are deliberately not dependencies.
import {cp, mkdir, mkdtemp, readFile, readdir, rm, writeFile} from 'node:fs/promises';
import {createServer} from 'node:http';
import {createServer as createHttpsServer} from 'node:https';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {pathToFileURL} from 'node:url';
import os from 'node:os';
import path from 'node:path';

const sdk = path.resolve(process.env.WHITEBOX_TEST_CLI_SDK);
const require = createRequire(path.join(sdk, 'packages/stream-host/package.json'));
const {ThreeCompiler} = await import(pathToFileURL(require.resolve('@worldkit/creator-host/compiler')).href);
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const root = await mkdtemp(path.join(os.tmpdir(), 'whitebox-real-cli-'));
let server;
async function close() {
  server?.closeAllConnections();
  if (server?.listening) await new Promise(resolve => server.close(resolve));
  await rm(root, {recursive: true, force: true});
}
try {
  const author = path.join(root, 'author');
  await cp(path.join(sdk, 'examples/three-creator/streaming-ui'), author,
    {recursive: true, filter: file => !file.includes('.three-creator')});
  const compilerOptions=process.env.WHITEBOX_TEST_PROJECT_AUDIO==='1'
    ? await (await import('./whitebox_project_audio_fixture.mjs')).prepareProjectAudio(author,require,root)
    : undefined;
  const candidate = await new ThreeCompiler(author, 'three-sdk',compilerOptions).prepare();
  const payload = path.join(root, 'payload');
  await mkdir(payload);
  await cp(candidate.playableRoot, path.join(payload, 'playable'), {recursive: true});
  const files = {};
  async function inventory(directory, prefix) {
    for (const entry of await readdir(directory, {withFileTypes: true})) {
      const file = path.join(directory, entry.name), name = prefix + entry.name;
      if (entry.isDirectory()) await inventory(file, name + '/');
      else if (entry.isFile()) files[name] = digest(await readFile(file));
      else throw Error('Fixture must contain regular files only');
    }
  }
  await inventory(path.join(payload, 'playable'), 'playable/');
  const manifest = {kind: 'three-creator-delivery', schemaVersion: 1, profile: 'three-sdk',
    worldBuildHash: candidate.worldBuildHash, runtimeHash: candidate.runtimeHash,
    sourceHash: candidate.sourceHash, files: {...files}};
  const manifestBytes = JSON.stringify(manifest);
  await writeFile(path.join(payload, 'delivery.json'), manifestBytes);
  files['delivery.json'] = digest(manifestBytes);
  await writeFile(path.join(payload, 'artifact-hashes.json'), JSON.stringify({schemaVersion: 1, files}));
  const archiveFile = path.join(root, 'creator-delivery.tar.gz');
  // Python's archive writer avoids platform tar extensions such as macOS
  // AppleDouble entries; the SDK correctly rejects anything outside payload.
  await promisify(execFile)('python3', ['-c',
    "import sys,tarfile\nwith tarfile.open(sys.argv[1], 'w:gz', format=tarfile.GNU_FORMAT) as archive: archive.add(sys.argv[2], arcname='payload')",
    archiveFile, payload]);
  const archive = await readFile(archiveFile);
  const receipt = {...manifest, archiveSha256: digest(archive), archiveByteLength: archive.length,
    deliveryManifestSha256: files['delivery.json']};
  const playableFiles=new Map();
  async function playableWalk(dir,prefix=''){
    for(const entry of await readdir(dir,{withFileTypes:true})){
      const name=prefix+entry.name,target=path.join(dir,entry.name);
      if(entry.isDirectory())await playableWalk(target,name+'/');
      else playableFiles.set(name,await readFile(target));
    }
  }
  await playableWalk(candidate.playableRoot);
  const inputSha='b'.repeat(64),prefix='media/v1/fixture/playable/v1/s/w/local/'+inputSha+'/';
  const playableManifest=Buffer.from(JSON.stringify({schemaVersion:1,sourceId:'s',worldId:'w',publicationId:'local',
    sourceManifestSha256:'a'.repeat(64),renditionInputSha256:inputSha,entryPath:'index.html',
    files:[...playableFiles].map(([path,bytes])=>({path,bytes:bytes.length,sha256:digest(bytes)}))}));
  playableFiles.set('playable-manifest.json',playableManifest);
  let published=process.env.WHITEBOX_TEST_PLAYABLE_READY!=='false';
  const counts={archives:0,publicFiles:0,publicBytes:0,reconciles:0};
  const handler=(req, res) => {
    const publicPath=req.url?.startsWith('/'+prefix)?decodeURIComponent(req.url.slice(prefix.length+1)):null;
    if(publicPath!==null){
      const bytes=playableFiles.get(publicPath);
      if(!bytes){res.writeHead(404);res.end();return;}
      counts.publicFiles++;counts.publicBytes+=bytes.length;res.end(bytes);return;
    }
    if (req.headers.authorization !== 'Bearer local-fixture-token') {
      res.writeHead(403); res.end(); return;
    }
    res.setHeader('content-type', 'application/json');
    if(req.url==='/__metrics'){res.end(JSON.stringify(counts));return;}
    if(req.url==='/api/cdn/playables/reconcile' && req.method==='POST'){counts.reconciles++;published=true;res.end('{}');return;}
    if(req.url?.startsWith('/api/worlds/s/w/playable-location?')){
      if(!published){res.end(JSON.stringify({ready:false,publicationId:'local'}));return;}
      const origin='https://127.0.0.1:'+server.address().port;
      res.end(JSON.stringify({ready:true,transport:'cdn',location:{schemaVersion:1,bucket:'fixture-assets',region:'oss-cn-beijing',
        sourceId:'s',worldId:'w',publicationId:'local',sourceManifestSha256:'a'.repeat(64),renditionInputSha256:inputSha,
        manifestSha256:digest(playableManifest),entryPath:'index.html',keyPrefix:prefix},
        entryUrl:origin+'/'+prefix+'index.html',manifestUrl:origin+'/'+prefix+'playable-manifest.json',
        publication:{sourceId:'s',worldId:'w',publicationId:'local',renditionInputSha256:inputSha}}));return;
    }
    if (req.url?.startsWith('/api/worlds/s/w?'))
      res.end(JSON.stringify({world: {id: 'w', sourceId: 's', attempt: 1}}));
    else if (req.url === '/api/worlds/s/w/artifacts')
      res.end(JSON.stringify({attempt: 1, artifacts: {files: [
        {fileName: 'attempt-1/creator-result.json', platformAttempt: 1,
          url: '/media/s/worlds/w/sdk-artifacts/attempt-1/creator-result.json'},
        {fileName: 'attempt-1/creator-delivery.tar.gz', platformAttempt: 1,
          url: '/media/s/worlds/w/sdk-artifacts/attempt-1/creator-delivery.tar.gz', bytes: archive.length},
      ]}}));
    else if (req.url?.endsWith('/creator-result.json')) res.end(JSON.stringify(receipt));
    else if (req.url?.endsWith('/creator-delivery.tar.gz')) {
      counts.archives++;res.setHeader('content-type', 'application/gzip'); res.end(archive);
    } else {res.writeHead(404); res.end('{}');}
  };
  const tls=process.env.WHITEBOX_TEST_TLS_CERT;
  server=tls?createHttpsServer({cert:await readFile(tls),key:await readFile(process.env.WHITEBOX_TEST_TLS_KEY)},handler):createServer(handler);
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  console.log(JSON.stringify({url: `${tls?'https':'http'}://127.0.0.1:${server.address().port}`,candidate:{worldBuildHash:candidate.worldBuildHash,runtimeHash:candidate.runtimeHash,runtimeSourceHash:candidate.runtimeSourceHash,runtimeCacheHit:candidate.runtimeCacheHit,runtimeFiles:candidate.runtimeFiles,audioSourceSha256:process.env.WHITEBOX_TEST_PROJECT_AUDIO==='1'?digest(await readFile(path.join(sdk,'packages/audio/src/index.ts'))):undefined}}));
  for (const signal of ['SIGINT', 'SIGTERM'])
    process.once(signal, () => {void close().then(() => process.exit(0));});
} catch (error) {
  await close();
  throw error;
}
