// Copy alongside the platform-owned whitebox-client.mjs and whitebox-playback.mjs.
import { encode, decode } from '@msgpack/msgpack';
import { WhiteboxClient } from './whitebox-client.mjs';
import { createPlaybackFeedback } from './whitebox-playback.mjs';
import { WhiteboxLiveDecoder } from './whitebox-live-decoder.mjs';

export function connectWorld({gatewayUrl, worldId, sourceId, prompt, references = [],
  leftCanvas, rightCanvas, transport = 'websocket', onStates = async () => {},
  onStatus = () => {}, onError = () => {}}) {
  let selected = 'whitebox';
  const cache = {whitebox: document.createElement('canvas'), model: document.createElement('canvas')};
  const available = {whitebox:false, model:false};
  function draw(target, source, width = source.width, height = source.height) {
    if (target.width !== width) target.width = width;
    if (target.height !== height) target.height = height;
    target.getContext('2d').drawImage(source,0,0);
  }
  function decoder(stream) {
    return new WhiteboxLiveDecoder({
      maxPendingFrames:32, onError: message => client.fail(message),
      onFrame(frame, metadata) {
        draw(cache[stream],frame,metadata.width,metadata.height); available[stream] = true;
        if (stream === 'whitebox') draw(leftCanvas,cache[stream]);
        if (selected === stream) { draw(rightCanvas,cache[stream]); client.presented(metadata); }
      },
    });
  }
  const left = decoder('whitebox'), model = decoder('model');
  const feedback = createPlaybackFeedback(message => {
    if (transport === 'websocket') client.send(message);
  },{window:2});
  const url = new URL(gatewayUrl); url.searchParams.set('whitebox_control','v1');
  const client = new WhiteboxClient({url:url.href,codec:{encode,decode},transport,
    onReady: async ready => {
      if (ready.control_protocol !== 'whitebox-display-v1') throw Error('Update the gateway to support continuous inference.');
      await left.configure(ready.source_video_stream); await model.configure(ready.video_stream);
      feedback.ready(ready);
    },
    onSourceFrame: async frame => { left.decode({...frame,type:'frame_batch'}); feedback.received(frame); },
    // Model decoding continues in A. Do not filter on the display revision.
    onFrame: async frame => { model.decode(frame); feedback.received(frame); },
    onStates, onStatus, onError,
    onDrain: async () => {await Promise.all([left.flush(),model.flush()]);},
    onClose: () => {left.close();model.close();},
  });
  client.connect({worldId,sourceId,prompt,references,renderMode:selected});
  function selectOutput(mode) {
    client.selectOutput(mode); selected = mode;
    if (available[mode]) draw(rightCanvas,cache[mode]);
    else onStatus('waiting_for_selected_stream');
  }
  // Once ready, these can be called repeatedly without reconnecting or resetting:
  // selectOutput('model'); selectOutput('whitebox'); selectOutput('model'); ...
  return {client,selectOutput,input:payload=>client.input(payload),finish:()=>client.finish()};
}
