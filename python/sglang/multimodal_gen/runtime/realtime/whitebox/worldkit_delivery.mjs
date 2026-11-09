// SPDX-License-Identifier: Apache-2.0
// Deployment transport policy around public SDK APIs; no SDK source overrides.
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
import {createHmac, createHash, randomBytes} from 'node:crypto';

const require = createRequire(process.env.WHITEBOX_CLI_ENTRYPOINT);
const {downloadWtpDelivery, downloadWtpOperationDelivery, DEFAULT_PLATFORM_URL} = await import(pathToFileURL(require.resolve('@worldkit/stream-host/wtp')).href);
const controller = new AbortController();
for (const signal of ['SIGTERM', 'SIGINT']) process.once(signal, () => controller.abort());
let deadline;
try {
  let input = '';
  for await (const chunk of process.stdin) {
    input += chunk.toString();
    if (input.length > 256 * 1024) throw Error('DELIVERY_REQUEST_INVALID');
  }
  const {worldId, directory, timeoutSeconds, createContext, verifiedCaller} = JSON.parse(input);
  if (!Number.isFinite(timeoutSeconds) || timeoutSeconds <= 0 || timeoutSeconds > 900) throw Error('DELIVERY_DEADLINE_INVALID');
  deadline = setTimeout(() => controller.abort(), timeoutSeconds * 1000);
  const options = {
    platformUrl: process.env.WHITEBOX_CLI_PLATFORM_URL || DEFAULT_PLATFORM_URL,
    signal: controller.signal,
    archiveTimeoutMs: timeoutSeconds * 1000,
    fetch: (url, options) => fetch(url, {...options, signal: controller.signal}),
  };
  let result;
  if (createContext !== undefined) {
    // Controlled creation has exactly one content selector: the original WTP
    // operation. Missing SDK/config support must never fall back to inventory.
    const secret = process.env.WHITEBOX_FROZEN_READ_SECRET;
    const serviceId = process.env.WHITEBOX_FROZEN_READ_SERVICE_ID;
    if (typeof downloadWtpOperationDelivery !== 'function'
      || typeof secret !== 'string' || Buffer.byteLength(secret) < 32
      || !serviceId || typeof verifiedCaller?.subject !== 'string'
      || !verifiedCaller.subject || verifiedCaller.subject.length > 256)
      throw Error('FROZEN_READ_CONFIGURATION_REQUIRED');
    const frozen = createContext.frozenPublication;
    const world = frozen?.world;
    if (`${world?.sourceId}/${world?.worldId}` !== worldId)
      throw Error('FROZEN_WORLD_IDENTITY_CHANGED');
    result = await downloadWtpOperationDelivery({
      operation: createContext.operation,
      expected: {receiptSha256: frozen.evidence.receiptSha256,
        archiveSha256: frozen.evidence.archiveSha256, runtimeHash: frozen.runtime.runtimeHash},
    }, path.join(directory, 'download'), {
      ...options,
      authorizeRequest: async bytes => {
        const now = Math.floor(Date.now() / 1000);
        const payload = Buffer.from(JSON.stringify({v: 3, aud: 'worldkit-frozen-read',
          service_id: serviceId, sub: verifiedCaller.subject, action: 'readFrozen',
          request_sha256: createHash('sha256').update(bytes).digest('hex'),
          iat: now, exp: now + 120, jti: randomBytes(16).toString('hex')})).toString('base64url');
        return payload + '.' + createHmac('sha256', secret).update(payload).digest('base64url');
      },
    });
  } else {
    result = await downloadWtpDelivery(worldId, path.join(directory, 'download'), {
      ...options, token: process.env.WORLD_TEST_API_TOKEN || '',
    });
  }
  // The official stream CLI performs the single sealed extraction. Compare its
  // actual archive identity against this downloaded receipt before using it.
  const identity=Object.fromEntries(['archiveSha256','worldBuildHash','runtimeHash','sourceHash'].map(key=>[key,result.receipt[key]]));
  process.stdout.write(JSON.stringify({...result, identity}) + '\n');
} catch (error) {
  // Do not expose URLs, bearer tokens, response bodies or archive paths in logs.
  const code = controller.signal.aborted ? 'DELIVERY_DEADLINE_EXCEEDED'
    : /^[A-Z][A-Z0-9_]{1,100}$/.test(error?.message || '') ? error.message : 'DELIVERY_PREPARATION_FAILED';
  process.stderr.write(code + '\n');
  process.exitCode = 1;
} finally {
  clearTimeout(deadline);
}
