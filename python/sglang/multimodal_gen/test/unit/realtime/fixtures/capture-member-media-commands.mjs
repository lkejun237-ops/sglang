// Actual shared SDK client envelopes; this helper starts no browser or transport.
import { pathToFileURL } from 'node:url';
import path from 'node:path';
const { WhiteboxClient } = await import(pathToFileURL(path.join(
  process.env.WHITEBOX_TEST_CLI_SDK, 'packages/stream-player/src/whitebox-client.mjs',
)).href);
let input = '';
for await (const chunk of process.stdin) input += chunk;
const { ticket, revision, mode, prefix } = JSON.parse(input);
const messages=[];
const codec={encode:message=>{messages.push(structuredClone(message));return new Uint8Array(1);}};
const client = new WhiteboxClient({url:'wss://gateway.test/v1/webrtc/play',codec,transport:'webrtc-rtp'});
Object.assign(client,{ready:true,controlProtocol:'whitebox-render-v2',resumeTicket:ticket,renderRevision:revision});
client.socket={readyState:1,bufferedAmount:0,send(){}};
client.control('status.get',{},`${prefix}-status`);
client.selectOutput(mode,`${prefix}-display`);
client.control('operation.get',{operation_request_id:`${prefix}-display`},`${prefix}-operation`);
process.stdout.write(JSON.stringify(messages));
