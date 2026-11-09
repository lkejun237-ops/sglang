// Exercise the same public browser policy as worldkit stream --rendering gpu.
// No world, input, model session, recording or platform request is created here.
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';

const require = createRequire(process.env.WHITEBOX_CLI_ENTRYPOINT);
const load = async name => import(pathToFileURL(require.resolve(name)).href);
const {streamBrowserLaunchOptions, inspectStreamEnvironment} = await load('@worldkit/stream-media/browser-environment');
const {launchChromiumWithSystemFallback} = await load('@worldkit/browser-capture/browser');
const policy = streamBrowserLaunchOptions({rendering: 'gpu', launchTimeoutMs: 30000});
const browser = await launchChromiumWithSystemFallback({...policy.launch, handleSIGINT: false});
try {
  const page = await browser.newPage();
  const pixels = await page.evaluate(() => {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 2;
    const gl = canvas.getContext('webgl2');
    if (!gl) throw Error('CLI_WEBGL2_REQUIRED');
    gl.clearColor(1, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    const pixel = new Uint8Array(4);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
    return Array.from(pixel);
  });
  if (pixels.join(',') !== '255,0,0,255') throw Error('CLI_GPU_READBACK_FAILED');
  const runtime = await inspectStreamEnvironment(browser, 'gpu');
  if (!/NVIDIA/i.test(runtime.rendering.renderer ?? '')) throw Error('CLI_NVIDIA_RENDERER_REQUIRED');
  console.log(JSON.stringify({kind: 'whitebox-cli-gpu-preflight', ...runtime, readback: 'passed'}));
} finally {
  await browser.close();
}
