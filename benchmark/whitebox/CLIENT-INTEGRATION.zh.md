# 白模试玩前端接入（双路 H.264）

此版本：服务端不运行 UI 合成器，试玩页只显示白模与模型视频。UI 叠加、第三路视频捕获由接入方实现。本接口保留 UI 数据、不可变资源和第三路归档入口。源 CLI 为原生 RGB24；公网播放为 H.264 Annex B。模型维持当前 session 独占 GPU slot。

## 连接与生命周期

连接既有 `wss://codex-th.loopit.com.cn/whitebox/v1/realtime_video/generate?user_id=<tab-scoped-id>&trace_id=<uuid>`。所有 WS 数据为二进制 MessagePack，不是 JSON 文本；`version:2`。业务鉴权仍由外层负责，现有 authorize 路径保持。

```js
socket.send(encode({version: 2, type: 'init',
  whitebox: {world_id: 'world-...', source_id: 'br-...'},
  prompt: '保持几何与相机运动', seed: 42,
  references: [{id: 'reference-0', format: 'png', payload: imageBytes}]}));
```

`session.ready` 返回 `session_id`、`recording_id`、`report_token`、`play_seconds:180`、`source_stream`、`video_stream`、`source_video_stream`、`ui`、`multiplayer` 和 `timeline:source_capture_vfr`。report_token 仅在该会话 WS 返回，不写入数据集或 URL。每个客户端单独使用自己的 token。

初始化期间可以直接关闭 WS 取消；ready 后发送 `{type:'event',kind:'finish',payload:{}}` 停止源并接收尾帧。收到 `session.completed` 后 flush 两个 VideoDecoder，再关闭 WS，服务端才确认播放尾帧已接收并释放槽位；断连同样会回收。每 10 秒发送 heartbeat。180 秒由服务端在首个模型输出交付后计时，页面在首帧接收后显示倒计时；启动阶段不消耗试玩时长。录制归档异步完成。

## 双路画面与时间轴

每条消息含一帧：白模 `type:whitebox.frame, stream:whitebox`，模型 `type:frame_batch, stream:model`。

```js
{
  type: 'frame_batch', stream: 'model', output_frame_start: 12, num_frames: 1,
  pts_us: 2030000, model_pts_us: 500000,
  width: 832, height: 480, encoding: 'h264', bitstream_format: 'annexb',
  codec: 'avc1.42c01f', keyframe: false, duration_us: 41667,
  payload: Uint8Array /* Annex B */, event_id: 1,
  source: {capture_pts_us:2030000,capture_origin_us:100000,
    cli_frame_header:{sessionId:'...',epoch:0,mediaGeneration:1,
      outputFrameId:27,outputPtsUs:2030000,uiRevision:4,uiCompleteThroughUs:2130000,
      source:{presentationId:'...',sdkEpoch:0,sourceFrameId:27,
        sourceTimeUs:2130000,simulationTick:108,worldRevision:0}}},
  states: {ui_events: [/* CLI 原始 UI 消息，保持原字段 */]},
  observed_input: {latest:{},events:[],receipts:[],mapping:'receipt_evidence_only'},
  pipeline_ms: 320, server_epoch_ms: 1790000000000
}
```

示例省略其他 FrameIdentity 字段，实际按返回值保留。**codec 以 ready 为准**，不要硬编码示例值。两路各用一个 VideoDecoder、各自有序解码；绘制模型帧时可以查询对应 states。实时播放立即绘制，不能根据 `model_pts_us` 等待播放。

`output_frame_start` 是本次流连续帧号；两路同编号来自同一个白模帧。接入层忽略早于初始 UI 快照的 CLI 缓存首帧，并保留快照与增量事件直到首个可映射帧，因此本次输出从帧号 0 / PTS 0 开始。`pts_us` 是真实源采集相对时间，保留不均匀间隔，MP4 与 inputs.jsonl 使用同一时钟。`model_pts_us = floor(index×1e6/24)` 仅用于未改动的 H3 输入时间轴，不能作为录像时间。`duration_us` 为实时解码的名义帧时长提示；实际录像当前帧持续至下一帧 PTS，末帧暂按 1/24 秒补齐。不能用帧数/24重建总播放时间，否则低于目标 FPS 时会加速。

收到 `playback_flow_control:whitebox-dual-v2` 后，为每路发送一次初始 credit，然后每次将帧成功提交 decoder 后回执：

```js
send({type:'event', kind:'playback_ack',
  payload:{protocol:'whitebox-dual-v2',stream:'model',frame:-1,window:8}});
// whitebox 也发送 -1；随后 frame 使用该路已收到的 output_frame_start。
```

不能丢弃 H.264 delta 帧；必须限制解码和应用队列。传输断代或身份不匹配应结束会话，不能混用另一个 session 的历史状态。

## 输入、states 和 UI 资源

输入沿用 `{type:'event',kind:'input',sequence:0,payload:{held_keys:['KeyW'],held_buttons:[],events:[...]}}`；sequence 在该连接内递增。失焦发送空 held_keys/held_buttons，避免卡键。Prompt 更新为 `kind:prompt,payload:'...'`。

states 不限制业务字段。目前适配器透传 `states.ui_events`，包含 CLI snapshot、patch、clock 等消息；它是增量历史，**不保证每帧为完整 UI 快照**。同事需要使用历史 UI runtime，按 `sourceTimeUs`、`uiRevision`、`uiCompleteThroughUs` 选择状态，不能把最新状态直接覆盖较旧模型帧。CLI 输入回执的 simulationTick 只证明输入已交给仿真，不等于模型一定产生了预期视觉变化。

ready.ui 包含 UI bundle hash、world build hash、source session/epoch、runtime ABI、document、catalog、module/styles/assets 和 `files:{relativePath:sha256}`。资源入口：

`GET /whitebox/v1/whitebox/recordings/{recording_id}/ui/{relativePath}`

附 `Authorization: Bearer <report_token>`。只能访问该会话 manifest 声明的已核验文件；响应包含 X-Content-SHA256。接入方校验字节 hash 后在隔离 UI runtime 中执行，不能使用携带令牌的第三方脚本 URL。当前试玩页面不执行这些 UI 代码。

## 录像与第三路上传

`GET /whitebox/v1/whitebox/recordings/{id}`（Bearer）返回后台状态、签名播放链接、timings、录制队列、`client_ui`。两路成功不代表第三路已上传。`GET .../{id}/inputs` 返回 `{time_unit:'us',interval:'[start,end)',intervals:[...]}`；回放用 video.currentTime×1e6 匹配 start_time_us ≤ t < end_time_us，显示完整 input 和 prompt。参考素材继续独立保存，通过 inputs 中的引用定位。

同事生成 UI MP4 后：单轨 H.264、同尺寸、同帧数、每帧 PTS 与模型相同（允许 1μs 容器舍入），不附音轨，上限 256MiB。不能直接把不受控帧率的 MediaRecorder 文件当作严格逐帧对齐录像。

1. 等两路后台归档完成，`POST .../{id}/ui-recording/intent`，Bearer，JSON `{bytes,sha256,content_md5}`。content_md5 为完整文件 MD5 的 base64；sha256 为小写 hex。
2. 按响应 method/headers/url 直接 PUT 到泰国 OSS，签名有效 900 秒。需要 bucket CORS 允许指定平台 origin 的 PUT 与 Content-Type/Content-MD5；发布时读回验证。签名只允许该录制下随机 upload_id 的 incoming 对象，无长期 OSS 凭据。
3. `POST .../{id}/ui-recording/{upload_id}/complete`，Bearer。接口立即返回 verifying；独立上传工作进程校验完整字节、H.264 解码、帧数/尺寸/PTS，成功发布第三路和 ui-recording.json。
4. 轮询原录制状态。验证成功才出现 ui_rendered 播放链接，失败保留证据，可对同内容重试 intent/complete。进程重启导致 verifying 悬挂时重发 complete 可重试。一个 recording 只接收一份内容身份，防止替换已有录像。

OSS 前缀仍为 `white_box_game_record/test-whitebox-th/<world-key>/<game_id>/streams/<recording_id>/`：

- `manifest.json`：参数、身份、模型版本、两路媒体、时间轴、参考图索引、第三路 receipt_path。
- `whitebox.mp4` / `model.mp4`：复用实时 H.264 结果的 VFR MP4。
- `inputs.jsonl` / `states.jsonl` / `references/*` / `ui_bundle/*`。
- 可选 `client-ui/<upload_id>/incoming.mp4`：客户端原始上传证据。
- 验证成功后 `client-ui/<upload_id>/ui_rendered.mp4`、`ui-recording.json`：第三路及其帧数、hash、字节数、身份。原 manifest 不回写，避免与后台上传竞争。

第三路缺失不影响模型播放与前两路归档；训练消费方应按自己的用途检查 client_ui receipt，不能推断三路齐全。

## 指标含义与发布边界

接收/绘制 FPS 来自浏览器实际回调；源/模型 FPS 各算各的，不把目标24写成实测。启动耗时从点击到第一次模型 drawImage；CLI 启动包含跨地域交付下载校验与浏览器启动。网络上/下行用 ping 中点估计时钟偏移，显示估算与 ±RTT/2，不当作精确单向延迟。输入→画面根据同 epoch 的 CLI receipt tick 与已展示源 tick 关联，不能当作模型视觉响应正确性的证明。

当前常驻 H3 Job 不包含本次外围代码更新。Gateway、Room、试玩泳道独立发布；禁止为页面/录制/心跳配置变更删除并重新提交 GPU Job。当前模型发布入口见 [统一发布手册](../../deploy/whitebox/README.md)，硬故障隔离后由人工恢复，不自动重启整组。
