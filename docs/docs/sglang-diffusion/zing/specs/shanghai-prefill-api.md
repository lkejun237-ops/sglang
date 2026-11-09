---
title: 上海与泰国共享缓存 Prefill API
description: 预热条件与 Block0，查询 CPFS 持久化结果和分阶段耗时。
---

# 上海与泰国共享缓存 Prefill API

**2026-10-30：上海 H200、泰国 B300 均已部署并通过 Prefill 冷/L1/L2、完整落盘、视频输出与公网 API 验收。**
不需要修改试玩页面；本地前端或服务脚本使用现有 WTP API Token。不要在 URL 中传 Token。

## 两地入口

| 地域 | WTP 票据接口 | Gateway WebSocket |
| --- | --- | --- |
| 上海 H200 | `https://world-test-platform.loopit.com.cn/whitebox-shanghai/api/whitebox/prefill-ticket` | `wss://whitebox-sh.loopit.me/whitebox/v1/realtime_video/prefill` |
| 泰国 B300 | `https://world-test-platform.loopit.com.cn/whitebox-test/api/whitebox/prefill-ticket` | `wss://whitebox-th.loopit.me/whitebox/v1/realtime_video/prefill` |

状态查询使用对应 Gateway 的 HTTPS origin，保留 `/whitebox` 前缀。
同一请求从签发票据到上传、查询均使用同一地域；两地的票据签名和 CPFS 独立。

## 调用过程

1. 将完整请求编码为二进制 MessagePack，计算原字节 SHA-256 和字节数。
2. 从 WTP 获取 `submit` 票据。
3. 连接 Gateway WebSocket，发送 `prefill.begin`，收到 `prefill.upload_ready` 后按顺序分片上传。
4. 观察 `prefill.status`；只有 `state=persisted` 表示 CPFS 完整缓存已经确认。
5. 断开后以相同 request_id 查询原任务，不自动换 request_id 重复计算。

预热只申请模型计算，不启动 CLI、Room 或录像。已有试玩时返回忙，不抢占试玩。
计算完成并确认 worker 清理后即释放模型名额；等待 CPFS 写入不占 GPU 会话名额。

## WTP 票据

`POST https://world-test-platform.loopit.com.cn/whitebox-shanghai/api/whitebox/prefill-ticket`

请求头：`Authorization: Bearer <WTP_API_TOKEN>`、`Content-Type: application/json`。

```json
{
  "sourceId": "interactive-production",
  "worldId": "world-…",
  "requestId": "prefill-<UUID>",
  "operation": "submit",
  "payloadSha256": "<MessagePack 原字节的 64 位小写 SHA256>",
  "payloadBytes": 123456
}
```

响应为 `{"token":"<短期票据>","expiresAt":1790000000000}`。有效期 120 秒。
`requestId` 为 1–128 位 ASCII 字母、数字、`.`、`_` 或 `-`。世界必须属于调用方可访问范围。
查询票据使用相同 sourceId/worldId/requestId、`operation=status`，省略 payloadSha256/payloadBytes。
浏览器 Cookie 调用需要同源；本地前端使用 API Token，并按 CORS 规则发起请求。

## 请求负载

这是 MessagePack 对象，`source_frame` 和 `references[].payload` 必须是二进制字节，不能填 Base64 字符串。

```javascript
const payload = {
  type: 'prefill',
  request_id: requestId,
  zing_realtime_api_version: 1,
  prompt: '与正式试玩完全相同的 prompt',
  seed: 42,
  size: '832x480',
  source_stream: { format: 'rgb24', width: 832, height: 480, fps: 24 },
  source_frame: rgbBytes, // 单帧、HWC/RGB 顺序，长度 width × height × 3
  source_frame_size: { width: 832, height: 480 },
  references: [
    { id: '<Picture 1>', format: 'png', payload: pngBytes }
  ],
  persist_timeout_s: 180
};
```

- source_frame 是首帧白模 RGB24；可以使用独立原生尺寸，由 source_frame_size 明确指定。
  不传 source_frame_size 时使用 source_stream 宽高。RGBA 必须先去掉 alpha，不重新缩放来凑字节数。
- 首帧必传。此接口不会为了预热启动 CLI 抓帧；普通实时试玩不传首帧的路径保持原行为。
- references 为 PNG/JPEG/WebP 原始编码字节、唯一 id 和正确 format。原生模型继续校验图片、数量和解码预算。
- size、source_stream FPS/几何、prompt、参考图 id/顺序/内容、首帧像素/尺寸，应与实际试玩入模请求一致。
  当前 Room 的模型输入 `VideoSpec.fps` 为 24，CLI 采集默认 30 FPS；不要把采集 FPS 直接填入模型输入时钟。应使用正式模型 init 的 source_stream 字段；尺寸与 Seed 同样必须一致。
- 不接受客户端指定本地路径、模型地址、CPFS 路径、GPU 数或缓存预算；模型采样配置仍由部署控制。
- Gateway 强制 persist=true；persist_timeout_s 为 1–300 秒，默认 180 秒，从内存就绪后开始等待。

## WebSocket 上传和结果

连接上表对应地域的 Gateway WebSocket；公网地址必须保留 `/whitebox` 前缀。
第一条二进制 MessagePack 消息：

```javascript
{
  type: 'prefill.begin', authorization: ticket.token,
  world: { source_id: sourceId, world_id: worldId }, request_id: requestId
}
```

收到 `{"type":"prefill.upload_ready","request_id":"…","max_chunk_bytes":1048576}` 后，
将最初编码好的请求字节切成最多 1 MiB 的连续二进制 WebSocket 消息。分片本身没有 MessagePack 包装，
不能逐片重新 encode。总长度和 SHA 必须与签发票据一致。
总请求上限 `64 MiB + 4096 B`；上传期限 60 秒；授权首消息期限 10 秒。
相同调用方/世界/request_id 已存在时直接返回已有状态，**不会**再次发 upload_ready。

收到的二进制消息解码为：

```json
{
  "type": "prefill.status",
  "request_id": "prefill-…",
  "trace_id": "…",
  "state": "persisted",
  "upload_ms": 25.4,
  "admission_ms": 12.1,
  "gateway_total_ms": 12345.6,
  "receipt": {
    "type": "prefill.persisted",
    "cache_key": "…",
    "hit": false,
    "prefill_to_memory_ms": 1234.5,
    "prefill_to_persisted_ms": 12340.1,
    "timings": { "cache_result": "miss", "stages": {}, "cuda_ms": {} },
    "rank_timings": [],
    "storage": { "state": "persisted", "generation": "…", "kv_bytes": 123, "conditions_bytes": 456, "ranks": [] }
  }
}
```

数值仅展示结构，**不是实测**。

| state | 含义 |
| --- | --- |
| accepted / computing | 已接收 / 已准入，正在预热 |
| memory_ready | 本实例 L1 可用，计算资源清理已确认；不能据此认定 CPFS 完成 |
| persisting | 等待完整 CPFS 条目 |
| persisted | 所有 SP 分片、conditions、manifest、完整标记校验通过 |
| failed | 请求失败；检查 code 和 receipt.code，保留 trace_id |
| cancelled | 服务关闭导致执行取消 |
| unknown | 原执行状态长期未更新，结果不可确认；不得当成功或盲目重提 |

已接受的执行与观察连接分离。关闭浏览器不会取消已接受的请求；后端执行上限 540 秒。
结果保留 1 小时。相同 request_id、相同上传摘要重连复用状态；内容不同会拒绝。
同用户每分钟最多 10 个新请求、Gateway 共享范围每分钟最多 30 个；重连查询不计新请求。
持久化超时返回 `PREFILL_PERSISTENCE_TIMEOUT`，表示未确认，后台写入可能仍在进行。
磁盘故障不会被转换成 persisted。状态存储不可用时不能假定任务没有执行。

## 查询状态

先签发 operation=status 的票据，然后：

```http
GET /whitebox/v1/realtime_video/prefill/requests/<requestId>?source_id=<sourceId>&world_id=<worldId>
Authorization: Bearer <status-ticket>
```

成功返回与 WebSocket status 相同的结果字段（没有 type 包装）。403 为授权错误，404 为无记录或已过期。
结果不包含绝对磁盘路径或原始图片。WTP 长期 Token 只发给 WTP，不能作为 Gateway 的状态票据。

## 缓存与时间口径

缓存目录沿用 SGLang v2，由模型指纹/执行布局指纹/内容 key 决定，不含 slot。
预热实例 A 完成后，兼容实例 B 可从同一 CPFS 读取；这属于 B 的 L2 命中，不是 B 的 L1 已预热。
配置或输入改变可以正常 miss。锁机制、LRU 与并发写入协调沿用主线，由负责同事维护。

2026-10-30 后续候选恢复 `main@b65640dd4b` 的原生 KV 读写：去掉先前新增的全量
KV SHA-256 扫描和数据文件强制 fsync；保留文件大小、shape/dtype、身份、generation
及完整标记检查。已有带 sha256 字段的缓存可直接读取，不需要删除或重建。
这是磁盘大文件 IO 调整，不改变接口上传摘要、鉴权或内容 key；`persisted` 表示
主线完整条目已发布且文件可读，不额外承诺每份大文件已执行强制刷盘。
候选的本地回归和磁盘读盘测试不等于模型进程已部署；运行版本需独立核验。

显式选择实例本地盘时，部署参数使用 `--zing-block0-cache-storage local`，并将
`--zing-block0-cache-dir` 指向事先创建的本地专用目录。默认 `mounted` 仍要求独立挂载，
不会在 CPFS 丢失时静默写到容器根盘。`--zing-block0-cache-disk-gb` 必须按本地实际容量
设置，不能照搬共享盘预算。API 契约不变，但 `persisted` 此时表示该实例本地完整条目，
不代表 CPFS 备份或其他实例可复用；容器重建、迁机或本地盘丢失后可能需要重新 Prefill。

- cache_result：l1_hit、l2_hit、mixed_hit、miss 或 bypass；miss_reason 解释未命中/回退。
- timings.stages.*.host_ms：CPU 调用墙钟时间；没有执行的步骤为 null/skipped，不填 0 假装更快。
- rank_timings：各 SP rank 的独立 prefill 阶段；不能将 rank 耗时相加。
- cuda_ms 仅返回已实际记录并完成的 CUDA event；缺失意味着未测，不等于零。
- lookup 含条件读取、KV 读取、现有锁等待等细分；这些与外层 cache_lookup 有包含关系。
- prefill_to_memory_ms 从模型 API 开始到计算/清理完成；prefill_to_persisted_ms 是到确认落盘的累计时间。
  两者不能相加，写盘可与其他阶段重叠。
- 实时试玩首个输出的 usage.zing_block0_cache 及对应服务日志报告该次启动的命中和准备阶段。
  它不等同于按键到浏览器显示延迟，也不包含浏览器网络/播放等待。


## 部署目录与复用边界

| 地域 | 新独立 CPFS | 模型容器缓存根目录 |
| --- | --- | --- |
| 上海 | `bmcpfs-01000o08rvganvryk6eu6`，上海 B / B3，10 TiB | `/mnt/prefill-cpfs/zing-cache/shanghai` |
| 泰国 | `bmcpfs-38001xr47mjulfoxcqdfu`，泰国 A / A1，10 TiB | `/mnt/prefill-cpfs/zing-cache/thailand` |

完整条目仍使用原生目录：

```text
<cache-root>/v2/<model-fingerprint>/<execution-fingerprint>/<key[:2]>/<key>/
  manifest.json
  .complete
  rank-0/kv.bin
  rank-0/conditions.pt
  rank-1/kv.bin
  rank-1/conditions.pt
```

目录不含 slot、实例 ID 或用户 ID。相同游戏不自动等于相同缓存：prompt、参考图、首帧、目标几何或执行配置改变，都可能产生新 key。
跨实例复用要求模型和执行指纹兼容；H200 与 B300 的数值执行环境不同，不承诺跨这两种卡直接复用。
传原图作为首帧可以测试 API，但不能据此预热另一个真实白模首帧；正式接入请传该次试玩实际使用的白模初始帧。

当前部署配置为 `--zing-block0-cache-host-gb 176`（所有 SP rank 合计预算）、
`--zing-block0-cache-disk-gb 9000`（该 cache root 的十进制 GB 上限），E1 + SP2 + D1，单试玩 Slot。
配置预算不是每次预热的实际占用量；以 receipt 中两 rank 的实际字节数为准。
正常试玩 miss 仍走原生后台写回，显式 Prefill 等待并返回持久化回执；页面无需新增按钮。

缓存锁、并发写入协调和 LRU 机制沿用主线，未在本轮改造。重启验收应完整重启相互关联的模型角色，不能用只重启 DiT、保留旧 NCCL 条件编码通道来代替跨实例验收。


## 本地前端调用示例

项目安装 `@msgpack/msgpack`，从已有登录流程取得 WTP API Token。下面的 `payload` 使用前文定义的字段，图片和 RGB 数据均为 `Uint8Array`；`requestId` 由调用方生成并保存，重连时继续使用它。

```javascript
import { encode, decode } from '@msgpack/msgpack';

export async function prefill({ region, token, sourceId, worldId, requestId, payload, onProgress }) {
  const sh = region === 'shanghai';
  const lane = sh ? 'whitebox-shanghai' : 'whitebox-test';
  const host = sh ? 'whitebox-sh.loopit.me' : 'whitebox-th.loopit.me';
  const bytes = encode({ ...payload, type: 'prefill', request_id: requestId });
  if (bytes.length > 64 * 1024 ** 2 + 4096) throw new Error('Prefill payload too large');
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  const payloadSha256 = Array.from(digest, b => b.toString(16).padStart(2, '0')).join('');
  const auth = await fetch(`https://world-test-platform.loopit.com.cn/${lane}/api/whitebox/prefill-ticket`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ sourceId, worldId, requestId, operation: 'submit',
      payloadSha256, payloadBytes: bytes.length }),
  });
  if (!auth.ok) throw new Error(`Prefill ticket HTTP ${auth.status}`);
  const ticket = await auth.json();
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(`wss://${host}/whitebox/v1/realtime_video/prefill`);
    ws.binaryType = 'arraybuffer';
    let finished = false;
    const finish = (error, result) => {
      if (finished) return;
      finished = true;
      ws.close();
      error ? reject(error) : resolve(result);
    };
    ws.onopen = () => ws.send(encode({ type: 'prefill.begin', authorization: ticket.token,
      world: { source_id: sourceId, world_id: worldId }, request_id: requestId }));
    ws.onmessage = async ({ data }) => {
      try {
        const event = decode(new Uint8Array(data));
        onProgress?.(event);
        if (event.type === 'prefill.upload_ready') {
          const chunk = Math.min(event.max_chunk_bytes, 1024 ** 2);
          for (let offset = 0; offset < bytes.length; offset += chunk) {
            while (ws.bufferedAmount > 2 * 1024 ** 2 && ws.readyState === WebSocket.OPEN)
              await new Promise(r => setTimeout(r, 10));
            if (ws.readyState !== WebSocket.OPEN) throw new Error('Upload disconnected');
            ws.send(bytes.subarray(offset, offset + chunk));
          }
        } else if (event.state === 'persisted') {
          finish(null, event);
        } else if (['failed', 'cancelled', 'unknown'].includes(event.state) || event.type === 'prefill.failed') {
          finish(Object.assign(new Error(event.code ?? event.state ?? 'Prefill failed'), { requestId, detail: event }));
        }
      } catch (error) { finish(error); }
    };
    ws.onerror = () => finish(Object.assign(new Error('Prefill connection error; query this requestId'), { requestId }));
    ws.onclose = () => {
      if (!finished) finish(Object.assign(new Error('Prefill observer disconnected; query this requestId'), { requestId }));
    };
  });
}
```

断开后重新签发 `operation=status` 票据并 GET 原 `requestId`。`memory_ready` 和 `persisting` 都不应在调用方显示为“已保存”；成功判断必须使用 `persisted`。


## 接入时的时间和幂等口径

- L1 指本实例进程内的 pinned CPU KV/conditions 缓存。实际试玩仍需恢复到 GPU，并执行后续 DiT 与 Decoder；`memory_ready` 不代表视频已经生成。
- `gateway_total_ms` 从上传完成后的 Gateway 执行开始计时，包含准入及原生执行/持久化等待。`upload_ms` 单列；`admission_ms` 与原生累计耗时已包含在执行时间中，不重复相加。
- 每次新的业务预热动作生成新的 requestId。网络重连、查询、同一次调用重试复用原 requestId；已完成请求不会因重连再次计算。
- 历史 `persisted` 回执证明该次已落盘，不保证未来 L1 永不被淘汰。需要重新加热当前实例时发起新的业务请求。
- 临时诊断输出目录不参与执行指纹；模型算术、采样和会影响计算的环境配置仍参与。不同实例的随机日志目录不会再导致缓存 miss。
