// SPDX-License-Identifier: Apache-2.0
// Resource preparation only. Never import/start StreamHost or reserve a model.
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
const require=createRequire(process.env.WHITEBOX_CLI_ENTRYPOINT);
const {prefetchWtpGame,gameCacheDirectory}=await import(pathToFileURL(require.resolve('@worldkit/stream-host/game-cache')).href);
const {DEFAULT_PLATFORM_URL}=await import(pathToFileURL(require.resolve('@worldkit/stream-host/wtp-options')).href);
const controller=new AbortController();
for(const signal of ['SIGTERM','SIGINT'])process.once(signal,()=>controller.abort());
const timer=setTimeout(()=>controller.abort(),300_000);
try {
 let input='';for await(const chunk of process.stdin){input+=chunk;if(input.length>1024)throw Error('INVALID_PREPARATION');}
 const {worldId}=JSON.parse(input),cacheDirectory=process.env.WHITEBOX_CLI_GAME_CACHE_DIRECTORY || process.env.WORLDKIT_GAME_CACHE_DIR || (process.env.WHITEBOX_CLI_CACHE_DIRECTORY ? path.join(process.env.WHITEBOX_CLI_CACHE_DIRECTORY,'games') : gameCacheDirectory());
 if(typeof worldId!=='string'||worldId.length>256||!cacheDirectory||!path.isAbsolute(cacheDirectory))throw Error('INVALID_PREPARATION');
 let last=0,phase='';
 const prepared=await prefetchWtpGame(worldId,{
  platformUrl:process.env.WHITEBOX_CLI_PLATFORM_URL||DEFAULT_PLATFORM_URL,
  token:process.env.WORLD_TEST_API_TOKEN||'',resourceTransport:process.env.WHITEBOX_CLI_RESOURCE_TRANSPORT==='auto'?'cdn':process.env.WHITEBOX_CLI_RESOURCE_TRANSPORT || 'cdn',gameCacheDirectory:cacheDirectory,
  signal:controller.signal,
  downloadConcurrency:Number(process.env.WHITEBOX_CLI_DOWNLOAD_CONCURRENCY||6),
  onProgress:value=>{if(value.phase==='ready')return;const now=Date.now();if(phase!==value.phase||now-last>=200){last=now;phase=value.phase;process.stdout.write(JSON.stringify({type:'world.preparation',state:'preparing',...value})+'\n');}},
 });
 if(!prepared.cacheComplete)throw Error('CACHE_INCOMPLETE');
 process.stdout.write(JSON.stringify({type:'world.preparation',state:'ready',identity:prepared.identity,...prepared.preparation})+'\n');
}catch{
 process.stdout.write(JSON.stringify({type:'world.preparation',state:'unavailable',code:'PREPARATION_UNAVAILABLE',content:'World preparation is unavailable. You can still enter normally.'})+'\n');
 process.exitCode=1;
}finally{clearTimeout(timer);controller.abort();}
