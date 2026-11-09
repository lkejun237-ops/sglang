// SPDX-License-Identifier: Apache-2.0
// The official SDK owns cache integrity/leases and resident state. Never start a world.
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import {spawn} from 'node:child_process';
import {readFile} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

const entry=process.env.WHITEBOX_CLI_ENTRYPOINT;
const require=createRequire(entry);
const {openGameCache}=await import(pathToFileURL(require.resolve('@worldkit/stream-host/game-cache/store')).href);
const {gameCacheDirectory}=await import(pathToFileURL(require.resolve('@worldkit/stream-host/game-cache/config')).href);
const {DEFAULT_PLATFORM_URL,platformOrigin}=await import(pathToFileURL(require.resolve('@worldkit/stream-host/wtp-options')).href);
const limit=2*1024*1024,observationTimeoutMs=10_000;
function serviceStatus(){return new Promise(resolve=>{
 const child=spawn(process.execPath,[entry,'serve','status','--service-name',process.env.WHITEBOX_CLI_SERVICE_NAME||'whitebox','--json'],{stdio:['ignore','pipe','ignore']});
 let bytes=0,chunks=[],failed=false;
 const unavailable=()=>({status:'unknown',runningServices:null,active:null,queued:null,workers:null,error:'CLI_STATUS_UNAVAILABLE'});
 const timer=setTimeout(()=>{failed=true;child.kill('SIGKILL');},observationTimeoutMs);
 child.stdout.on('data',chunk=>{bytes+=chunk.length;if(bytes>limit){failed=true;child.kill('SIGKILL');}else chunks.push(chunk);});
 child.once('error',()=>{failed=true;});
 child.once('close',code=>{clearTimeout(timer);try{
  if(failed||code!==0)throw Error();const value=JSON.parse(Buffer.concat(chunks));
  if(value.kind!=='worldkit-service'||!['stopped','starting','ready','draining','failed'].includes(value.status))throw Error();
  resolve({status:value.status,runningServices:value.status==='stopped'?0:1,
   active:value.status==='stopped'?0:value.active,queued:value.status==='stopped'?0:value.queued,
   name:value.name,instanceId:value.instanceId,pid:value.pid,readyAt:value.readyAt,
   idleDeadline:value.idleDeadline,lifetimeDeadline:value.lifetimeDeadline,
   workers:(value.workers??[]).map(w=>({id:w.id,busy:w.busy,healthy:w.healthy,restarts:w.restarts}))});
 }catch{resolve(unavailable());}});
});}
const game=value=>value?Object.fromEntries(['platformUrl','sourceId','worldId','publicationId','manifestSha256','pinned','preparedAt','lastUsed','status','referencedBytes','bytesByCategory'].map(key=>[key,value[key]])):null;
async function currentSelection(worldId,platformUrl,directory){
 try{
  const {readWtpGameCacheSelection}=await import(pathToFileURL(require.resolve('@worldkit/stream-host/game-cache/status')).href);
  const result=await readWtpGameCacheSelection(worldId,{platformUrl,token:process.env.WORLD_TEST_API_TOKEN||'',gameCacheDirectory:directory,resourceTransport:process.env.WHITEBOX_CLI_RESOURCE_TRANSPORT==='auto'?'cdn':process.env.WHITEBOX_CLI_RESOURCE_TRANSPORT||'cdn',signal:AbortSignal.timeout(observationTimeoutMs)});
  return {...result,state:result.matches?'current':result.cachedManifestSha256===null?'missing':'stale'};
 }catch{return {state:'unavailable',code:'CURRENT_EXECUTION_UNAVAILABLE'};}
}
try{
 let text='';for await(const chunk of process.stdin){text+=chunk;if(Buffer.byteLength(text)>2048)throw Error('RUNTIME_REQUEST_INVALID');}
 const {operation,world_id,page=0}=JSON.parse(text);
 if(!['status','inspect','inventory','remove-preview','remove'].includes(operation)||!Number.isSafeInteger(page)||page<0||page>10_000)throw Error('RUNTIME_REQUEST_INVALID');
 const parts=typeof world_id==='string'?world_id.split('/'):[];
 if(!(world_id==null&&['status','inventory'].includes(operation))&&(parts.length!==2||parts.some(p=>!p||p==='.'||p==='..'||p.length>256||!/^[a-zA-Z0-9_.-]+$/.test(p))))throw Error('RUNTIME_WORLD_INVALID');
 const platformUrl=platformOrigin(process.env.WHITEBOX_CLI_PLATFORM_URL||DEFAULT_PLATFORM_URL);
 const directory=process.env.WHITEBOX_CLI_GAME_CACHE_DIRECTORY||process.env.WORLDKIT_GAME_CACHE_DIR||
  (process.env.WHITEBOX_CLI_CACHE_DIRECTORY?path.join(process.env.WHITEBOX_CLI_CACHE_DIRECTORY,'games'):gameCacheDirectory());
 if(!path.isAbsolute(directory))throw Error('RUNTIME_CACHE_UNAVAILABLE');
 const cache=await openGameCache(directory);
 // One selected world only: no history/inventory backscan, resource prefetch or
 // World/model startup. A complete local record alone cannot prove freshness.
 const [service,selection]=await Promise.all([serviceStatus(),world_id&&['status','inspect'].includes(operation)?currentSelection(world_id,platformUrl,directory):null]);
 let action;
 if(operation==='remove'||operation==='remove-preview'){
  if(!['stopped','ready'].includes(service.status)||service.active!==0||service.queued!==0)throw Error('RUNTIME_SERVICE_BUSY');
  action=await cache.remove(world_id,platformUrl,{dryRun:operation==='remove-preview'});
 }
 const inventory=await cache.list();
 const checked=operation==='inspect'?await cache.inspect(world_id,platformUrl):inventory;
 const games=inventory.games.filter(g=>g.platformUrl===platformUrl).sort((a,b)=>b.lastUsed-a.lastUsed||a.worldId.localeCompare(b.worldId));
 const current=checked.games.find(g=>g.platformUrl===platformUrl&&g.sourceId===parts[0]&&g.worldId===parts[1]);
 if(selection?.state==='current'&&selection.selected.manifestSha256!==current?.manifestSha256){selection.matches=false;selection.state=current?'stale':'missing';}
 let rssBytes=null;
 if(Number.isSafeInteger(service.pid)&&service.pid>0&&process.platform==='linux'){
  try{const status=await readFile(`/proc/${service.pid}/status`,'utf8');const match=/^VmRSS:\s+(\d+) kB$/m.exec(status);if(match)rssBytes=Number(match[1])*1024;}catch{}
 }
 const result={schemaVersion:1,type:'world.runtime',state:'ready',observedAt:new Date().toISOString(),
  machine:{host:os.hostname(),scope:'configured-room-owner',node:process.version},
  service:{...service,memory:{rssBytes,scope:'resident-daemon-only',includesWorkers:false}},
  cache:{objectBytes:inventory.objectBytes,reservedBytes:inventory.reservedBytes,maxBytes:inventory.maxBytes,
   verification:checked.verification,current:game(current),selection,worlds:games.length,
   ...(operation==='inventory'?{page,pageSize:20,entries:games.slice(page*20,page*20+20).map(game),hasMore:(page+1)*20<games.length}:{}),
   ...(action?{action:{...action,dryRun:operation==='remove-preview'}}:{})}};
 process.stdout.write(JSON.stringify(result)+'\n');
}catch(error){
 const code=error instanceof Error&&/^[A-Z][A-Z0-9_]+$/.test(error.message)?error.message:'RUNTIME_INSPECTION_FAILED';
 process.stdout.write(JSON.stringify({type:'world.runtime',state:'unavailable',error:{code}})+'\n');process.exitCode=1;
}
