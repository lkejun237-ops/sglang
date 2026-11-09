// SPDX-License-Identifier: Apache-2.0
// Real authored WAV + loop/positional one-shot through public compiler admission.
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
export async function prepareProjectAudio(author,require,root){
 const load=async name=>import(pathToFileURL(require.resolve(name)).href);
 const {createFrozenEligibility,createAssetProposal,sealAssetPlan}=await load('@worldkit/creator-host/task-selection');
 const {createTaskAssetPolicySnapshot,assetPolicyHash,taskProjectSelection}=await load('@worldkit/creator-host/asset-policy');
 const {materializeTaskAssetLibrary}=await load('@worldkit/creator-host/task-selected-library');
 const hash=v=>createHash('sha256').update(v).digest('hex'),hex='1'.repeat(64);
 const budget={maximumResourceBytes:100000,maximumTotalBytes:200000,maximumDurationSeconds:2,maximumChannels:2,maximumSampleRate:48000,maximumDecodedBytes:200000};
 const eligibility=createFrozenEligibility({taskId:'local-audio',caseId:'no-inference',promptSha256:hex,registryId:'local-fixture',namespace:'local-fixture',baseRuntimeHash:hex,capabilityCatalogDigest:hex,policy:{id:'local-fixture',version:'1.0.0',digest:hex},resolver:{version:'1.0.0',implementationDigest:hex},canonicalFormatVersion:'worldkit-canonical-json/1',mediaPolicy:{modelAndMotion:'sealed-only',taskUiFormats:['png'],maxTaskUiBytes:10000,projectAudioBudget:budget},limits:{maxRoots:8,maxClosure:8,maxResourceBytes:67108864,maxCandidates:8},exceptions:{allow:[],deny:[]},taskParameters:{schemaVersion:1,includeUnreleasedAssetsInRetrieval:true}});
 const plan=createAssetProposal(eligibility,{revision:1,needs:[{slotId:'floor',role:'ground',subject:'object',control:'passive'}],choices:[{slotId:'floor'}],motionCompositions:[]});
 const {lock}=sealAssetPlan(eligibility,plan,{admissionContextDigest:hex,admissionDecisionId:'local-audio-admitted',roots:[],closure:[],resources:[],catalogSha256:hash('[]')});
 const snapshot=createTaskAssetPolicySnapshot(eligibility,plan,lock,[]);
 const library=await materializeTaskAssetLibrary({eligibility,plan,seal:{lock,catalog:[]},hostCacheRoot:root,readResource:async()=>{throw Error('Unexpected external audio fixture resource');}});
 process.env.ASSET_LIBRARY_ROOT=library.root;
 const policy=path.join(root,'audio-policy.json');await writeFile(policy,JSON.stringify(snapshot));
 const project=JSON.parse(await readFile(path.join(author,'project.json'),'utf8'));
 delete project.assetIds;
 await writeFile(path.join(author,'project.json'),JSON.stringify({...project,...taskProjectSelection(plan,lock),audio:{manifest:'assets/audio/manifest.json'}}));
 await mkdir(path.join(author,'assets/audio'),{recursive:true});
 const declaration={contract:'worldkit.project-audio',version:1,resources:[{id:'tone',path:'assets/audio/tone.wav'}],bindings:{contract:'worldkit.audio-bindings',version:1,sounds:[{id:'loop',resources:['tone'],gain:.15,loop:true},{id:'hit',resources:['tone'],gain:.8,loop:false}],events:[{event:'game.loop',soundId:'loop',command:'play'},{event:'game.hit',soundId:'hit',command:'play'},{event:'game.stop',soundId:'loop',command:'stop'}]}};
 await writeFile(path.join(author,'assets/audio/manifest.json'),JSON.stringify(declaration));
 const b=Buffer.alloc(44+1600);b.write('RIFF');b.writeUInt32LE(b.length-8,4);b.write('WAVEfmt ',8);b.writeUInt32LE(16,16);b.writeUInt16LE(1,20);b.writeUInt16LE(1,22);b.writeUInt32LE(8000,24);b.writeUInt32LE(16000,28);b.writeUInt16LE(2,32);b.writeUInt16LE(16,34);b.write('data',36);b.writeUInt32LE(1600,40);
 for(let i=0;i<800;i++)b.writeInt16LE(Math.round(Math.sin(i/8000*2*Math.PI*440)*8000),44+i*2);
 await writeFile(path.join(author,'assets/audio/tone.wav'),b);
 const startTick=Number(process.env.WHITEBOX_TEST_AUDIO_START_TICK??'1');if(![1,120].includes(startTick))throw Error('Invalid fixture audio start tick');
 const stopTick=Number(process.env.WHITEBOX_TEST_AUDIO_STOP_TICK??'0');if(!Number.isSafeInteger(stopTick)||stopTick<0)throw Error('Invalid fixture audio stop tick');
 const scene=path.join(author,'scene.ts');let code=await readFile(scene,'utf8');
 code="import audioDeclaration from './assets/audio/manifest.json';\n"+code;
 code=code.replace('await world.start();',`const audio=world.audio.createPlayer('game',audioDeclaration.bindings);await audio.load(id=>new URL(audioDeclaration.resources.find(r=>r.id===id)!.path,document.baseURI).href);let audioSequence=0;world.onFixedUpdate(({simulationTick})=>{if(simulationTick===${startTick})audio.handle({event:'game.loop',sequence:audioSequence++});if(${stopTick}>0&&simulationTick===${stopTick})audio.handle({event:'game.stop',sequence:audioSequence++});if(${stopTick}>0&&simulationTick===${stopTick}+120)audio.handle({event:'game.loop',sequence:audioSequence++});if(simulationTick===150)audio.handle({event:'game.hit',sequence:audioSequence++,positionMetersXYZ:[3,1,0]});});\nawait world.start();`);
 await writeFile(scene,code);
 return {assetPolicySnapshotPath:policy,assetPolicySha256:assetPolicyHash(snapshot)};
}
