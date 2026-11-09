// Browser integration helpers. Rendering and UI video capture belong to the caller.
// Use your MessagePack decoder + WebCodecs player for the dual stream documented
// in CLIENT-INTEGRATION.zh.md. Never store the capability in a public URL.
export function sourceContext(modelFrame) {
  if (modelFrame.stream !== 'model' || modelFrame.type !== 'frame_batch') throw Error('A model frame is required');
  const identity = modelFrame.source?.cli_frame_header;
  if (!identity) throw Error('The model frame is missing its source identity');
  return {frame: modelFrame.output_frame_start, ptsUs: modelFrame.pts_us,
    identity, states: modelFrame.states, promptEventId: modelFrame.event_id};
}

export async function uploadUiRecording({reportUrl, reportToken, mp4, contentMd5, signal}) {
  const headers = {Authorization: `Bearer ${reportToken}`, 'Content-Type': 'application/json'};
  const sha256 = [...new Uint8Array(await crypto.subtle.digest('SHA-256', await mp4.arrayBuffer()))]
    .map(byte => byte.toString(16).padStart(2, '0')).join('');
  // contentMd5 comes from the caller's incremental MD5 implementation. WebCrypto
  // does not implement MD5. For large files hash in a worker, outside playback.
  const intentResponse = await fetch(`${reportUrl}/ui-recording/intent`, {method:'POST', headers, signal,
    body:JSON.stringify({bytes:mp4.size, sha256, content_md5:contentMd5})});
  if (!intentResponse.ok) throw Error('The upload could not be prepared');
  const intent = await intentResponse.json();
  if (intent.status === 'complete' || intent.status === 'verifying') return intent;
  const uploaded = await fetch(intent.url, {method:intent.method, headers:intent.headers, body:mp4, signal});
  if (!uploaded.ok) throw Error('The video could not be uploaded');
  const complete = await fetch(`${reportUrl}/ui-recording/${intent.upload_id}/complete`, {method:'POST', headers, signal});
  if (!complete.ok) throw Error('The recording could not be submitted for verification');
  return complete.json(); // Poll reportUrl until client_ui.status === 'complete'.
}

export function replayInput(intervals, videoTimeSeconds) {
  const time = Math.round(videoTimeSeconds * 1e6);
  return intervals.find(row => row.start_time_us <= time && time < row.end_time_us) ?? null;
}
