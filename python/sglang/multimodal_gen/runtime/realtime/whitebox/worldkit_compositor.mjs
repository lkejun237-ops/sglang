// SPDX-License-Identifier: Apache-2.0
// Lifecycle bridge only. The colleague SDK owns capture, UI and source history.
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import {createInterface} from 'node:readline';

const require = createRequire(process.env.WHITEBOX_CLI_ENTRYPOINT);
const {startCompositionHost} = await import(pathToFileURL(require.resolve('@worldkit/stream-compositor')).href);
const lines = createInterface({input: process.stdin, crlfDelay: Infinity});
let host;
const close = async () => { lines.close(); await host?.close(); };
for (const signal of ['SIGTERM', 'SIGINT']) {
  process.once(signal, () => { void close().finally(() => process.exit(0)); });
}
try {
  for await (const line of lines) {
    if (line.length > 1024 * 1024 || host) throw Error('COMPOSE_BRIDGE_REQUEST_INVALID');
    const {source, videoInput, video, browser} = JSON.parse(line);
    host = await startCompositionHost({browser});
    const descriptor = await host.createComposition({source, videoInput, video}, {idempotencyKey: videoInput.streamId});
    // Return starting, before the first model frame, so Python can subscribe
    // before output begins. CLI compose waits for decoding and is unsuitable here.
    process.stdout.write(JSON.stringify(descriptor) + '\n');
  }
} catch {
  process.stderr.write('WHITEBOX_COMPOSITION_FAILED\n');
  process.exitCode = 1;
} finally {
  await close();
}
