// SPDX-License-Identifier: Apache-2.0
// Private file and public SDK boundary only; no copied Room or session internals.
import { createRequire } from 'node:module';
import { open, constants } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const cancellation = new AbortController();
const deadline = setTimeout(() => cancellation.abort(), 25000);
for (const signal of ['SIGTERM', 'SIGINT']) {
  process.once(signal, () => cancellation.abort());
}
let handle;
try {
  const require = createRequire(process.env.WHITEBOX_CLI_ENTRYPOINT);
  const { parseSessionConnection, parseSessionDescriptor, parsePlayerSessionBinding, playerStreamProtocols, MAX_CONTROL_BYTES } =
    await import(pathToFileURL(require.resolve('@worldkit/stream-protocol')).href);
  const { readSession, streamEndpoints } =
    await import(pathToFileURL(require.resolve('@worldkit/stream-host/session')).href);

  let input = '';
  for await (const chunk of process.stdin) {
    input += chunk;
    if (Buffer.byteLength(input) > 8192) throw Error('ROOM_SESSION_REQUEST_INVALID');
  }
  const { memberFile, uiRoot, publicOrigin, purpose = 'viewer' } = JSON.parse(input);
  if (!['viewer', 'model-source'].includes(purpose) ||
      (purpose === 'model-source' && publicOrigin !== undefined)) {
    throw Error('ROOM_SESSION_REQUEST_INVALID');
  }
  if (typeof memberFile !== 'string' || !path.isAbsolute(memberFile) ||
      typeof uiRoot !== 'string' || !path.isAbsolute(uiRoot)) {
    throw Error('ROOM_SESSION_REQUEST_INVALID');
  }
  handle = await open(memberFile, constants.O_RDONLY | constants.O_NOFOLLOW);
  const stat = await handle.stat();
  if (!stat.isFile() || (stat.mode & 0o777) !== 0o600 ||
      (process.getuid && stat.uid !== process.getuid()) || stat.size > MAX_CONTROL_BYTES) {
    throw Error('ROOM_MEMBER_FILE_INVALID');
  }
  // A private member record carries a minimal SessionConnection, not a cached
  // SessionDescriptor. Its current epoch is obtained from authenticated live truth.
  const bytes = Buffer.alloc(MAX_CONTROL_BYTES + 1);
  const { bytesRead } = await handle.read(bytes, 0, bytes.length, 0);
  if (bytesRead > MAX_CONTROL_BYTES) throw Error('ROOM_MEMBER_FILE_INVALID');
  const record = JSON.parse(bytes.subarray(0, bytesRead).toString('utf8'));
  if (record.kind !== 'worldkit-room-member' || record.schemaVersion !== 1) {
    throw Error('ROOM_MEMBER_FILE_INVALID');
  }
  const ordinary = parseSessionConnection(record.session);
  const useModel = purpose === 'model-source' && record.modelSession !== undefined;
  const connection = useModel ? parseSessionConnection(record.modelSession) : ordinary;
  if (connection.sessionId !== ordinary.sessionId || connection.baseUrl !== ordinary.baseUrl) {
    throw Error('ROOM_MEMBER_IDENTITY_INVALID');
  }
  const binding = parsePlayerSessionBinding(record.binding);
  if (connection.accessMode !== 'capability' || record.roomId !== binding.room.roomId ||
      binding.render?.sessionId !== connection.sessionId) {
    throw Error('ROOM_MEMBER_FILE_INVALID');
  }
  playerStreamProtocols(connection.accessToken);
  const { session } = await readSession(connection, { requireRunning: true, signal: cancellation.signal });
  // Undeclared legacy sources may retain their ordinary transport. A declared
  // model source must never silently downgrade to the player's input capability.
  if (purpose === 'model-source' && session.modelContext && !useModel) {
    throw Error('ROOM_MODEL_SESSION_REQUIRED');
  }
  const audience = session.audience;
  if (session.accessMode !== 'capability' || session.epoch !== binding.render.epoch ||
      JSON.stringify(session.playerBinding) !== JSON.stringify(binding) ||
      audience?.kind !== 'player' || audience.roomId !== binding.room.roomId ||
      audience.roomEpoch !== binding.room.roomEpoch || audience.playerSessionId !== binding.playerSessionId) {
    throw Error('ROOM_MEMBER_IDENTITY_INVALID');
  }
  if (publicOrigin !== undefined) {
    const url = new URL(publicOrigin);
    const local = ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname);
    if (url.origin !== publicOrigin || url.username || url.password
      || (url.protocol !== 'https:' && !(url.protocol === 'http:' && local)))
      throw Error('ROOM_PUBLIC_ORIGIN_INVALID');
    const prefix = `${publicOrigin}/v1/sessions/${encodeURIComponent(session.sessionId)}`;
    const rewrite = value => {
      const asset = new URL(value);
      if (asset.origin !== new URL(connection.baseUrl).origin
        || !asset.pathname.startsWith(`/ui/${session.uiBundleHash}/`)
        || asset.search || asset.hash) throw Error('ROOM_PUBLIC_UI_INVALID');
      return prefix + asset.pathname;
    };
    const published = parseSessionDescriptor({...session, baseUrl: publicOrigin,
      ui: {...session.ui, moduleUrl: rewrite(session.ui.moduleUrl),
        styles: session.ui.styles.map(style => ({...style, url: rewrite(style.url)}))}});
    process.stdout.write(JSON.stringify(published) + '\n');
  } else {
  process.stdout.write(JSON.stringify({
    ...session, kind: 'worldkit-stream-session', schemaVersion: 1, status: 'ready',
    streams: streamEndpoints(session), localResources: { uiRoot },
  }) + '\n');
  }
} catch (error) {
  const code = /^[A-Z][A-Z0-9_]{1,100}$/.test(error?.message ?? '')
    ? error.message : 'ROOM_SESSION_UNAVAILABLE';
  process.stderr.write(code + '\n');
  process.exitCode = 1;
} finally {
  await handle?.close();
  clearTimeout(deadline);
}
