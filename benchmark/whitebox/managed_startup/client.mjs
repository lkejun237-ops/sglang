// Server-side protocol observer. It never labels receipt arrival as browser presentation.
import fs from 'node:fs';
import {createRequire} from 'node:module';
import {createHash,createHmac,randomBytes} from 'node:crypto';
import {pathToFileURL} from 'node:url';
const require=createRequire('/opt/worldkit/apps/game-player/package.json');
const {encode,decode}=require('@msgpack/msgpack');
const sdk=file=>import(pathToFileURL('/opt/worldkit/packages/stream-player/src/'+file).href);
const {WhiteboxClient}=await sdk('whitebox-client.mjs');
const {buildManagedWhiteboxInit}=await sdk('whitebox-init.mjs');
const {assembleWhiteboxExecution}=await sdk('whitebox-execution.mjs');

const settings = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const root = settings.output, began = performance.now();
const events = fs.openSync(`${root}/events.jsonl`, 'a');
const frameLog = fs.openSync(`${root}/frames.jsonl`, 'a');
const statesLog = fs.openSync(`${root}/states.jsonl`, 'a');
const counts = {whitebox: 0, model: 0}, bytes = {whitebox: 0, model: 0};
const seen = {whitebox: new Set(), model: new Set()}, inputs = new Map(), receipts = new Map();
const matched = new Set(), latencies = [], frames = [];
let client, ready, firstModel = null, finished = false, failure = null, currentInput = null;
let held = [], inputTimer, renewalTimer, releaseTimer, activeUntil = null;
function sanitize(value) {
  return JSON.parse(JSON.stringify(value, (key, item) =>
    /token|secret|password|authorization|capability|ticket|cookie/i.test(key) ? '[redacted]' :
    item instanceof Uint8Array ? {bytes: item.byteLength} : typeof item === 'bigint' ? String(item) : item));
}
function record(kind, data = {}) {
  fs.writeSync(events, JSON.stringify({kind, mono_ms: performance.now(), epoch_ms: Date.now(), ...sanitize(data)}) + '\n');
}
function remember(message) {
  for (const receipt of message.observed_input?.receipts || []) {
    const sequence = receipt.clientSequence ?? receipt.sequence;
    if (receipt.type === 'input.receipt' && inputs.has(sequence)) receipts.set(sequence, receipt);
  }
}
function acknowledge(stream, frame) {
  client.send({type: 'event', kind: 'playback_ack', payload: {protocol: 'whitebox-dual-v2', stream, frame, window: 4}});
}
function frame(stream, message) {
  const now = performance.now();
  remember(message);
  const id = `${message.stream_id ?? ''}:${message.rendering?.model_epoch ?? ''}:${message.output_frame_start}`;
  if (!seen[stream].has(id)) {
    seen[stream].add(id); counts[stream]++; bytes[stream] += message.payload.byteLength;
    if (stream === 'model' && firstModel === null) {
      firstModel = now;
      activeUntil = now + settings.observation_seconds * 1000;
      record('first_model_packet', {width: message.width, height: message.height});
    }
    const header = {...message}; delete header.payload;
    fs.writeSync(frameLog, JSON.stringify({stream, mono_ms: now, epoch_ms: Date.now(), ...sanitize(header)}) + '\n');
    frames.push({stream, at_ms: now, index: message.output_frame_start});
    const identity = message.source?.cli_frame_header;
    if (identity && Number.isSafeInteger(identity.source?.simulationTick)) {
      for (const [sequence, receipt] of receipts) {
        const key = `${stream}:${sequence}`, input = inputs.get(sequence);
        if (!matched.has(key) && receipt.epoch === identity.epoch && receipt.simulationTick <= identity.source.simulationTick) {
          matched.add(key);
          const sample = {stream, sequence, kind: input.kind, delay_ms: now - input.at_ms,
            applied_tick: receipt.simulationTick, received_tick: identity.source.simulationTick,
            output_frame_start: message.output_frame_start};
          latencies.push(sample); record('input_mapped_frame_received', sample);
        }
      }
    }
  }
  acknowledge(stream, message.output_frame_start);
}
client = new WhiteboxClient({url: settings.url, transport: 'websocket', maxQueuedFrames: 64,
  codec: {encode, decode},
  onReady: async message => {
    ready = message;
    fs.writeFileSync(`${root}/ready.private.json`, JSON.stringify(message, (_key, value) => typeof value === 'bigint' ? String(value) : value));
    record('ready', message); acknowledge('whitebox', -1);
    if (client.streamIds.model) acknowledge('model', -1);
  },
  onSegment: async message => {record('segment', message); if (client.streamIds.model) acknowledge('model', -1);},
  onSourceFrame: async message => frame('whitebox', message),
  onFrame: async message => frame('model', message),
  onStates: async message => {remember(message); fs.writeSync(statesLog, JSON.stringify({mono_ms: performance.now(), ...sanitize(message)}) + '\n');},
  onInputSent: (sequence, kind, at_ms) => {
    if (kind !== 'renewal') {inputs.set(sequence, {at_ms, kind, action: currentInput}); record('input_sent', {sequence, kind, at_ms, action: currentInput});}
  },
  onControl: message => record('control', message),
  onRenderStatus: message => record('render', message),
  onDiagnostic: message => record('diagnostic', message),
  onRecover: message => record('recovery', message),
  onError: message => {failure = String(message); record('error', {message: failure});},
  onClose: () => {finished = true; record('closed');},
});
const references = (settings.references || []).map(ref => ({id: ref.id, format: ref.format, payload: new Uint8Array(fs.readFileSync(ref.file))}));
let managedOperation;
const digest=value=>createHash('sha256').update(value).digest('hex');
async function control(action){
  const operation={...managedOperation,...(action==='getAttachment'?{controlSequence:client.controlSequence??0}:{})};
  const raw=JSON.stringify({schemaVersion:1,action,singlePlayerOperation:operation});
  const now=Math.floor(Date.now()/1000);
  const claims={v:3,aud:'whitebox-single-control',scope:'single-player-control',service_id:settings.service_id,
    sub:settings.subject,origin:null,action,request_sha256:digest(raw),iat:now,exp:now+90,jti:randomBytes(16).toString('hex')};
  const payload=Buffer.from(JSON.stringify(claims)).toString('base64url');
  const signature=createHmac('sha256',settings.api_secret).update(payload).digest('base64url');
  const response=await fetch(settings.api_base+'/v1/whitebox/single-player-control',
    {method:'POST',headers:{'content-type':'application/json',authorization:'Bearer '+payload+'.'+signature},body:raw,signal:AbortSignal.timeout(300000)});
  if(!response.ok)throw Error('MANAGED_CONTROL_HTTP_'+response.status);
  return response.json();
}
async function startManaged(){
  if(settings.environment!=='test-shanghai-copy'||settings.api_base!=='http://127.0.0.1:18080'
    ||settings.url!=='ws://127.0.0.1:18080/v1/realtime_video/generate'||settings.observation_seconds<180)throw Error('Invalid test scope');
  const identity=await (await fetch(settings.api_base+'/v1/whitebox/version')).json();
  if(identity.source_sha!==settings.expected_runtime_source||identity.sdk_sha!==settings.expected_sdk
    ||identity.release_scope!=='isolated_development_test')throw Error('Unexpected gateway identity');
  const initialFrame=settings.initial_frame?{...settings.initial_frame,payload:new Uint8Array(fs.readFileSync(settings.initial_frame.file))}:undefined;
  if(initialFrame)delete initialFrame.file;
  const init=await buildManagedWhiteboxInit({sourceId:settings.source_id,worldId:settings.world_id,prompt:settings.prompt,
    references,seed:42,cliFps:settings.cli_fps,inferenceControl:'manual',initialFrame});
  const frozenLaunchParameters={transport:'websocket',publicationIdentity:settings.publication,initMsgpackBase64:Buffer.from(encode(init)).toString('base64')};
  managedOperation={operationId:settings.trace_id,parametersSha256:digest(JSON.stringify(frozenLaunchParameters)),
    caller:{serviceId:settings.service_id,subject:settings.subject},frozenPublication:settings.frozen_publication,frozenLaunchParameters};
  fs.writeFileSync(root+'/STARTED.private.json',JSON.stringify(managedOperation),{flag:'wx',mode:0o600});
  record('managed_start_sent',{with_preview:!!initialFrame});
  const start=await control('start');
  record('managed_start_returned',{state:start.state,execution:start.execution});
  if(!['starting','ready'].includes(start.state))throw Error(start.failureCode??'MANAGED_START_FAILED');
  const deadline=performance.now()+120000;
  let current;
  do{
    current=await control('getAttachment');
    if(!['starting','ready'].includes(current.state))throw Error(current.failureCode??'MANAGED_SOURCE_UNAVAILABLE');
    if(current.attachment)break;
    if(performance.now()>deadline)throw Error('MANAGED_ATTACHMENT_TIMEOUT');
    await new Promise(resolve=>setTimeout(resolve,100));
  }while(true);
  const descriptor=assembleWhiteboxExecution({signallingUrl:settings.url,attachment:current.attachment,
    execution:current.execution,publication:settings.publication,subject:settings.subject});
  client.attach(descriptor,{refreshAttachment:async()=>{
    const refreshed=await control('getAttachment');
    if(!refreshed.attachment)throw Error('MANAGED_ATTACHMENT_UNAVAILABLE');
    return refreshed.attachment;
  }});
}

async function waitUntil(test, timeout) {
  const deadline = performance.now() + timeout;
  while (!test()) {
    if (failure || client.closed) throw Error(failure || 'Connection closed early');
    if (performance.now() > deadline) throw Error('Stage deadline exceeded');
    await new Promise(resolve => setTimeout(resolve, 50));
  }
}
try {
  await startManaged();
  await waitUntil(() => client.ready && counts.whitebox > 0, 120000);
  let action = 0;
  function press() {
    if (client.closed || activeUntil !== null && performance.now() >= activeUntil) return;
    const code = ['KeyW', 'KeyD', 'KeyS', 'KeyA'][action++ % 4];
    held = [code]; currentInput = {code, pressed: true, action};
    client.input({held_keys: held, held_buttons: [], events: [{type: 'keydown', code}]});
    releaseTimer = setTimeout(() => {
      held = []; currentInput = {code, pressed: false, action};
      if (!client.closed) client.input({held_keys: [], held_buttons: [], events: [{type: 'keyup', code}]});
    }, 1200);
  }
  press(); inputTimer = setInterval(press, 6000);
  renewalTimer = setInterval(() => {
    if (held.length && !client.closed) client.input({held_keys: held, held_buttons: [], events: []});
  }, 200);
  if (settings.inference_control === 'manual') {
    await waitUntil(() => client.inference?.read().availability === 'ready', 30000);
    const current = client.inference.read();
    if (!current.scope || !current.value) throw Error('Manual inference scope is unavailable');
    if(counts.model!==0)throw Error('MODEL_PIXELS_BEFORE_MANUAL_ENABLE');
    record('manual_model_enable_requested',{model_frames_before_enable:counts.model});
    await client.inference.configure({scope: current.scope, expectedRevision: current.value.revision, enabled: true});
    record('manual_model_enable_confirmed');
  }
  await waitUntil(() => firstModel !== null, 240000);
  // Optional explicit prompt-only probes retain the references of this session.
  for (const update of settings.prompt_updates || []) {
    if (!Number.isFinite(update.after_seconds) || update.after_seconds < 0 || update.after_seconds >= settings.observation_seconds - 20 || typeof update.text !== 'string') throw Error('Invalid prompt probe');
    await waitUntil(() => performance.now() >= firstModel + update.after_seconds * 1000, settings.observation_seconds * 1000);
    let request, text = update.text;
    if (client.worldDriven) {
      client.ensureModelInstruction();
      await waitUntil(() => client.modelInstruction && client.modelInstructionAvailable, 30000);
      const revision = client.modelInstruction.revision;
      text = update.instruction;
      if (typeof text !== 'string' || !text.trim()) throw Error('A world-driven probe requires a game instruction');
      request = client.setModelInstruction(text, revision);
      await waitUntil(() => client.modelInstruction?.request_id === request, 30000);
      if (!['accepted', 'duplicate'].includes(client.modelInstruction.status) || client.modelInstruction.revision <= revision) throw Error('Game instruction was not accepted');
    } else request = client.updatePrompt(text);
    record('prompt_only_probe', {request, text, interface: client.worldDriven ? 'game-instruction' : 'prompt.replace', references_omitted: true});
  }
  await waitUntil(() => performance.now() >= activeUntil, settings.observation_seconds * 1000 + 5000);
  record('observation_complete', {counts, observation_seconds: (performance.now() - firstModel) / 1000});
} catch (error) {
  failure = error.message; record('failed', {message: failure});
} finally {
  clearInterval(inputTimer); clearInterval(renewalTimer); clearTimeout(releaseTimer);
  if (client.ready && !client.closed) {
    client.input({held_keys: [], held_buttons: [], events: []}); client.finish();
  }
  const deadline = performance.now() + 90000;
  while (!finished && performance.now() < deadline) await new Promise(resolve => setTimeout(resolve, 100));
  if (!finished) {failure ??= 'Client close not confirmed'; client.close();}
  if(managedOperation){
    try{let closed=await control('close');const until=performance.now()+90000;
      while(closed.state==='cancelling'&&performance.now()<until){
        await new Promise(resolve=>setTimeout(resolve,500));closed=await control('getOperation');
      }
      if(closed.state==='failed')closed=await control('close'); // Verify a completed original teardown.
      record('managed_closed',{state:closed.state});
      if(closed.state!=='cancelled')failure??='MANAGED_CLOSE_UNCONFIRMED';
    }catch(error){failure??=error.message;record('managed_close_failed',{message:error.message});}
  }
  const summary = {trace_id: settings.trace_id, cli_fps_requested: settings.cli_fps,
    engine_capture_fps: ready?.engine?.capture_fps, source_session_id: ready?.source_session_id,
    source_epoch: ready?.source_epoch, counts, bytes, first_model_mono_ms: firstModel,
    observation_end_mono_ms: activeUntil, duration_ms: performance.now() - began,
    input_samples: latencies, input_edges: [...inputs].map(([sequence, v]) => ({sequence, ...v})),
    frames, error: failure, client_closed: finished,
    scope: 'Server-side real CLI/model and mapped-input receipt arrival; no browser display or WAN latency claim'};
  fs.writeFileSync(`${root}/RESULT.json`, JSON.stringify(summary));
  for (const descriptor of [events, frameLog, statesLog]) fs.closeSync(descriptor);
  console.log(JSON.stringify({trace_id: summary.trace_id, counts, error: failure, client_closed: finished}));
  process.exitCode = failure ? 1 : 0;
}
