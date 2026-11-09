// SPDX-License-Identifier: Apache-2.0
// Platform readiness only. The official CLI remains the sole content/session owner.
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import {setTimeout as delay} from 'node:timers/promises';
const require=createRequire(process.env.WHITEBOX_CLI_ENTRYPOINT);
const {resolveWtpPlayableWorld}=await import(pathToFileURL(require.resolve('@worldkit/stream-host/wtp-reference')).href);
const {platformOrigin,DEFAULT_PLATFORM_URL}=await import(pathToFileURL(require.resolve('@worldkit/stream-host/wtp-options')).href);
const controller=new AbortController();let timer;
for(const name of ['SIGTERM','SIGINT'])process.once(name,()=>controller.abort());
const started=performance.now();
try{
 let input='';for await(const chunk of process.stdin){input+=chunk;if(input.length>8192)throw Error('WTP_PLAYABLE_PREPARATION_INVALID');}
 const {worldId,timeoutSeconds,expectedPublication,transport="cdn"}=JSON.parse(input);
 if(typeof worldId!=='string'||worldId.length>2048||!Number.isFinite(timeoutSeconds)||timeoutSeconds<=0||timeoutSeconds>120)throw Error('WTP_PLAYABLE_PREPARATION_INVALID');
 timer=setTimeout(()=>controller.abort(),timeoutSeconds*1000);
 const signal=controller.signal,origin=platformOrigin(process.env.WHITEBOX_CLI_PLATFORM_URL||DEFAULT_PLATFORM_URL),token=process.env.WORLD_TEST_API_TOKEN;
 if(!token)throw Error('WTP_PLAYABLE_AUTHENTICATION_REQUIRED');
 const resolved=await resolveWtpPlayableWorld(worldId,{platformUrl:origin,token,signal});
 const query=new URLSearchParams({transport});
 const fields=['sourceId','worldId','publicationId','sourceManifestSha256','renditionInputSha256','manifestSha256'];
 if(expectedPublication!==undefined){
  if(!expectedPublication||expectedPublication.sourceId!==resolved.sourceId||expectedPublication.worldId!==resolved.worldId
   ||!['cdn','oss-public','oss-internal'].includes(transport)||fields.some(key=>typeof expectedPublication[key]!=='string'))throw Error('WTP_PLAYABLE_PUBLICATION_CHANGED');
  for(const key of fields.slice(2))query.set(key,expectedPublication[key]);
 }
 const url=new URL(`/api/worlds/${encodeURIComponent(resolved.sourceId)}/${encodeURIComponent(resolved.worldId)}/playable-location?${query}`,origin);
 const headers={authorization:'Bearer '+token};
 const read=async()=>{const response=await fetch(url,{headers,redirect:'error',signal});if(!response.ok)throw Error('WTP_PLAYABLE_PREPARATION_FAILED');return response.json();};
 const initial=await read();
 if(expectedPublication!==undefined){
  // Frozen single launch is read-only. Never reconcile or authorize whichever
  // replacement is current after the original WTP operation was frozen.
  if(initial.ready!==true||fields.some(key=>initial.location?.[key]!==expectedPublication[key]))throw Error('WTP_PLAYABLE_PUBLICATION_CHANGED');
 }else if(initial.ready!==true){
  if(initial.ready!==false||typeof initial.publicationId!=='string')throw Error('WTP_PLAYABLE_PREPARATION_FAILED');
  // Exactly one attempt. An unknown POST result is never retried here. The
  // platform's existing source/world publication lease owns idempotency.
  const response=await fetch(new URL('/api/cdn/playables/reconcile',origin),{method:'POST',headers:{...headers,'content-type':'application/json'},
   body:JSON.stringify({sourceId:resolved.sourceId,worldId:resolved.worldId}),redirect:'error',signal});
  if(!response.ok)throw Error('WTP_PLAYABLE_PREPARATION_FAILED');await response.arrayBuffer();
  for(;;){
   const value=await read();
   if(value.ready===true){if(value.location?.publicationId!==initial.publicationId)throw Error('WTP_PLAYABLE_PUBLICATION_CHANGED');break;}
   if(value.ready!==false||value.publicationId!==initial.publicationId)throw Error('WTP_PLAYABLE_PUBLICATION_CHANGED');
   await delay(1000,undefined,{signal});
  }
 }
 process.stdout.write(JSON.stringify({ready:true,elapsedMs:performance.now()-started,...(expectedPublication===undefined?{}:{publicationIdentity:Object.fromEntries(fields.map(key=>[key,initial.location[key]]))})})+'\n');
}catch(error){
 const code=/^[A-Z][A-Z0-9_]{1,100}$/.test(error?.message||'')?error.message:'WTP_PLAYABLE_PREPARATION_FAILED';
 process.stderr.write(code+'\n');process.exitCode=1;
}finally{clearTimeout(timer);controller.abort();}
