# Whitebox WebRTC 游戏 UI actions 接入

适用：`whitebox-render-v2` / `native-h264-rtp-v1`。媒体通过 WebRTC RTP，控制消息仍通过 `/whitebox/v1/webrtc/play` 的可靠 WebSocket，以 MessagePack 二进制发送。WS 媒体入口与 RTC 媒体入口共用 Room 的游戏控制实现。

## 游玩投诉证据读取

投诉主表、reportId、处理权限与管理页面由 WTP 负责。SGLang 不写投诉数据库，不把
reportId 当作 recordingId、traceId 或新的会话，也不改变录制、编码和模型默认值。
`reports.py` 的原录制 owner 安装以下只读接口；这与原 recording status 使用同一份
`Authorization: Bearer <report_token>` 能力，错误能力返回 404。

| 方法与路径 | 行为 |
| --- | --- |
| `GET /v1/whitebox/recordings/{recording_id}/context` | 返回无签名 URL、无 token 的有界证据快照 |
| `GET /v1/whitebox/recordings/{recording_id}/context/artifact-access?artifact_id=states` | 鉴权后签发指定证据的 15 分钟访问 URL |
| `GET /v1/whitebox/recordings/{recording_id}/context/timeline` | 精确模型段的有界实际输出 Prompt/参考图时间线 |

Gateway 对应入口为 `/v1/whitebox/owners/{owner_id}/recordings/{recording_id}/...`，
只转发到配置的 owner，保留查询参数和原 Bearer。仅有一个 owner 时原兼容入口仍可用。
两个响应均使用 `Cache-Control: private, no-store`，不得把 artifact-access 响应持久化为
稳定证据位置。接口不接收 URL、bucket、object key 或磁盘路径作为抓取目标。

`context` 可传 `segment_id=<精确模型段 recordingId>`；也支持 `segment_epoch`，只在父
manifest 中唯一匹配该 epoch 的 entry；同时传两者时必须一致。必须在父 source manifest
的 render_segments 中，并同时匹配 source_session_id、source_epoch 和对象前缀。
选择器不受旧 status 接口最后 64 段展示上限限制。未传时仅返回 source，不猜测最新段；
指定他人或不存在的段返回 404，不回退到其他段。已结束的录制沿原能力索引读取本地保留
manifest 或 diskless 对象归档；没有保留的索引/manifest 不会重建、恢复或续租会话。

响应 `schema_version=1`，包含 `recording_id`、`observed_at_ms`、`identity`、`source`、
精确选中的 `segments`、`mapping` 和有界 `timings`。每个 source/segment 包含：

- `identity`：manifest 已记录的 world/trace/source/epoch/segment 身份，以及已存在的
  multiplayer game/player/stream/user/owner 身份；缺少字段不伪造。
- `identity_provenance=archive_manifest`；`deployment` 只返回白名单元数据，
  `deployment_provenance=recorded_manifest_metadata`。配置与历史 manifest 不是当前
  实例运行身份的独立证明，也不是 WTP 用户/operation 关联的授权凭据。
- 原 `recording_status`、`upload_status`、`dataset_status`、timeline 和客户端 UI 状态。
- `artifacts[]`：`artifact_id`、role、recording_id、availability、`locator`、可用时的
  bytes/sha256、`access={artifact_id,segment_id}`。locator 保存 provider/bucket/region/
  object_key；不返回存储端点、凭据或签名 URL。availability 依据归档 manifest/receipt，
  不代表本次已对全部对象执行 HEAD 验证。输入和状态未完整归档时保留 pending/unavailable。
  需要客户端 UI 录像但 receipt 尚未完成时，也返回独立 pending artifact，其 locator 为 null，
  不能因为该文件未出现就把整场证据标记为完整。
- artifact ID 支持 manifest、inputs、states、`video:whitebox`、`video:model`、
  `video:ui_rendered`、已验证独立上传的 `client_ui`，以及至多 64 个 reference。
  artifact-access 也可传相同 segment_id，未归档返回 409，无签名器返回 503。

可选 `frame_index` 明确表示**归档帧号**，配合 `stream=whitebox|model`；它不是 CLI
源帧、模拟 tick 或提交时间。或者单独传 `source_frame_index`，从 states 的 source provenance
精确匹配；存在多个对应归档帧时标记 ambiguous，不擅自选择。两个 frame selector 不能同时传。
新录制由原 archive writer 同步生成小型 `report-frames.jsonl`，manifest 的
`report_frames={path,format:report_frame_index_v1}` 声明它，并由原对象清单记录大小/SHA256。
每行只有 schema_version、归档 frame、可用的 source_frame_index 与逐流 archive_pts_us；
不包含 opaque states、输入、图像或第二份播放队列。五分钟会话无需扫描完整游戏状态。
reader 优先读取该有界索引，旧包才回退 states。仅当对应视频有实际 `first_pts_us`、
明确的 `playback_timeline=mp4_preserved_pts_v1` 与 `playback_time_origin_us=0`，
且有不超过 8 MiB、大小和 SHA256 均匹配的索引/旧 states 时，按实际逐流 archive_pts_us 返回
`mapping={status:mapped,recording_id,artifact_id,frame_index,pts_us,seek_seconds,method,
precision,index_path,index_sha256}`。新 streaming/disk worker 在实际 mux 边界记录逐流 archive_pts_us，
区分 source_frame 的 capture PTS 与 pair/独立录制的 presentation PTS；视频 first_pts_us
来自 mux 首个实际 packet。playback_time_origin_us 表示 HTMLVideo currentTime=0 对应的
归档 PTS；当前 H264MuxFile 保留非零 PTS/edit list，故该值为 0，不能填首帧 PTS。
seek_seconds=(实际归档 PTS-playback_time_origin_us)/1e6，当前等于实际绝对 PTS 秒数。
例如首帧 PTS=12.0s、投诉帧 PTS=12.49s，浏览器必须跳到 12.49s；减首帧得到 0.49s
会被浏览器夹到 12.0s，错误地显示第一帧。不使用帧号/24，也不假设每个 MP4 从首帧归零。
旧包缺少明确 playback marker 时返回 archive_playback_timeline_unknown；即使有 first_pts_us
也不生成看似精确的跳转。旧独立编码 v2 只有同时具备明确 playback 合同与视频起点时，
才可用其原 presentation_pts_us。帧不存在、文件过大或不完整同样返回明确 reason。
本地 CPU 回归覆盖实际 mux/解码 PTS 与 HTTP 响应；Google Chrome 对当前普通 MP4 和
streaming fragmented MP4 的隔离非零起点素材均已确认上述 currentTime 语义，使用支持
HTTP 206 Range 的本地读取，并等待 canplaythrough 与呈现后检查对应帧颜色。不同浏览器、
真实 WTP/CDN 与用户整场点击仍需联合验收。
输出快照上限 256 KiB；磁盘与对象读取沿原有有界 IO，不扫描整场日志或读取原始状态到响应中。

### 同一模型段的白模与模型画面对比

一个 `render_segment` 已有两条录像：whitebox 保存该段每个模型输出实际配对的源帧，model
保存对应模型画面。对比应选择**同一个段**的 `video:whitebox` 与 `video:model`，不能拿父
source 的完整预览录像按相同 currentTime 冒充一一对应。原配对 owner 在相同归档 frame 和
实际 PTS 下写入两条录像；独立编码的 pair owner 也在写入前核对两边实际 PTS。

`context.source.comparison` 与每个 `context.segments[].comparison` 提供：

```text
schema_version: 1
recording_id, source_session_id, source_epoch, segment_id, segment_epoch
source_artifact_id: video:whitebox
model_artifact_id: video:model
status: aligned | pending | unavailable
method: shared_recording_frame_index_and_pts    # 仅 aligned 时
frames, first_pts_us, last_pts_us
first_playback_seconds, last_playback_seconds
playback_timeline: mp4_preserved_pts_v1
playback_time_origin_us: 0
index_sha256
reason                                             # 未证明对应关系时
```

aligned 必须同时具备精确父/子身份、完整归档、两条视频对象的 receipt、相同有效帧数及明确
播放时间合同；读取并验证整个有界小索引的大小/SHA256 后，逐帧核对连续归档 frame 与两条
实际 PTS 相等且严格递增，并匹配 manifest 首帧 PTS。共同的非零起点和共同时间空隙允许存在；
PTS 不同、缺轨、数量不符、部分归档或缺少证明材料时明确不可同步，不用“帧数相同”替代证据。
这项验证读取 manifest/小索引，不重新下载或解码整条视频，也不独立证明 CDN 对象未被外部修改。
消费者据此鉴权取得这两个 artifact 的临时 URL，同步播放、暂停和跳转；浏览器调度与实际
解码仍需播放器纠偏和验收。不同模型段不能只凭相同 currentTime 自动合并为一条轨道。

### 实际输出的 Prompt 与参考图时间线

`timeline` 复用父录制 Bearer、精确 segment_id 或唯一 segment_epoch；不默认选最新模型段。
查询字段为 `from_seconds`（默认 0）、`to_seconds`（默认 from+300）、`limit`（默认64、最大128）。
时间必须有限、非负，且 `0 < to-from <= 300`。`recording_id` 是实际选中段，
`source_recording_id` 是原父录制；响应带完整 source/segment identity 供消费者复核。

响应 `schema_version=1`、`semantic=model_output_applied`，包含 status、window、coverage、
events、includes_prior_context、truncated、next_from_seconds 与可用时的 provenance/index_sha256。
每个 event 保存 frame_index、source_frame_index、实际 playback_seconds，
`prompt={text,sha256,version?}` 和 `references=[{id,sha256,artifact_id?}]`。
参考图 artifact 只根据本段已归档对象的实际哈希匹配，不返回任意 URL、对象端点或能力凭据。
参考图对象有有效 SHA256/正大小 receipt 时，png/jpeg/jpg/webp 扩展名可提供白名单 content_type，
provenance 为 archive_reference_extension；未知格式和 SVG 不提供可内嵌图片类型。
原接入 owner 负责图像内容与格式校验，本只读投影不重新下载图像或把后缀当作独立解码验收。
Prompt 是用户原文，响应不复制包含服务凭据的完整 init、输入或日志。

新归档在原 metadata writer 中生成 `output-contexts.jsonl`，只在实际输出 Prompt/有序参考图/
版本改变时记录，不因键盘输入变化重复长 Prompt。manifest 用
`output_contexts={path,format:model_output_context_v1,frames}` 声明其格式和已处理输出帧数。
counter 必须等于实际 model 视频帧数，且每个事件必须精确匹配已核验帧索引的 frame/源帧/PTS。
实际语义来自 `RenderSegment.model_output_condition` 按模型输出 event_id 选中的条件；
“请求已接受”不生成已应用事件。缺失或无效输出上下文、范围外帧、错误 PTS、部分归档、
不完整覆盖都返回 unavailable，不能向后沿用上一条 Prompt 冒充已知状态。

旧包只允许回退读取不超过8 MiB且大小/哈希均匹配的 inputs，并要求它的区间从0连续覆盖
实际全部模型帧，再按 start_frame 关联小帧索引的真实模型 PTS。旧 inputs 的 start_time_us 是
capture 时间，不能直接作为模型录像时间；只有 accepted metadata 或 opaque states 的旧包
不能推出应用时间线。每个索引/上下文对象读取上限8 MiB，响应预算128 KiB，不读整场大 states。

窗口返回 from 时刻已经应用的最后一条上下文，以及 `from < playback_seconds <= to` 的
后续变化；包含 to 时刻和共同末帧发生的上下文切换，避免末帧误显示上一条 Prompt。没有已知先前输出时
不填充 Prompt。coverage 表示真实归档帧 PTS 的首/末边界，窗口超过末帧不继续传播上一条
上下文。达到事件/字节预算时返回 truncated 和第一条未返回事件的 next_from_seconds，
消费者可按窗口重读并按 frame_index 去重。时间线按需读取，不在普通列表或每次状态轮询中
下载。真实多段切换、参考图预览和双视频同步仍需 WTP/播放器联合验收。

集成边界：WTP 必须从原受信 launch/control owner 取得并安全保留原录制 capability，
核对原 operation/subject 与返回身份；不能将浏览器任填的 scope 升格为服务端核验。
本接口没有新增通用服务身份读取或 direct-play scope receipt。无法取得既有 capability
的旧 direct-play 路径仍可保存用户投诉，但应标注证据关联不完整，不得开放跨用户读取。
能力不得进入普通 report JSON、日志或前端管理详情。保留期限仍遵循原归档，引用不自动 pin。

发布时分别记录构建工具与运行源码的精确 SHA。使用现行 `main` 正式构建工具，验证运行
源码可达新鲜 `origin/main`、内嵌 SDK 可达 `origin/dev`，再导出完整归档；不运行历史
`white_box` 构建器或绕过镜像原生检查与层历史门禁。仅更新 CPU 组件时，还须核对 Gateway、
Room 实际加载的模块和插件，不能将未启用的模型代码一并声明为已发布。保留经核对的实际
bindings、Secret、SDK pin 与挂载；配置沿用前逐项核对原 bundle 指纹、地域与 release-owned
环境，不以新 plan 隐式推进模型策略。源码身份和镜像通过不代表线上会话已排空。

本地合同回归：`test_whitebox_report_context.py` 与 `test_whitebox_render_reports.py`
覆盖错能力、他人段、超 64 段精确选择、保留索引、稳定定位、VFR 与 hash 校验、pending
归档、任意路径拒绝和 owner 查询转发。真实 WTP 提交、对象播放与点击定位仍需配套端到端
验收；源码实现和 CPU 回归不代表线上已部署或完成产品验收。
`test_whitebox_report_timeline.py` 另用实际模型输出条件选择、真实双轨编码/归档，覆盖非零
起点、共同空隙、PTS/轨道不匹配、延迟应用的 Prompt、已知覆盖、旧包缺失、分页及权限边界。
context、timeline、原 `test_whitebox_stream_archive.py` 与 `test_whitebox_recording_process.py`
均登记在维护中的 `scripts/ci/zing/run_area_tests.py`，通过原 `execute_pytest` 隔离入口执行两轮；
后者严格检查包含两个新 sidecar 的八对象上传集合及 manifest 最后发布。入口断言由
`test/registered/unit/ci/test_worldkit_area_gate.py` 维护，不改写历史 CPU 批准 manifest。

## 外部 Player 按需推理

2026-11-02 本地候选实现。外部单人和成员 Player 由受信 WTP 冻结以下 init 字段：

```json
{
  "source_operation": "create.manual",
  "inference_control": "manual"
}
```

两字段必须同时出现。旧 Room 严格要求 `source_operation: "create"`，因此会在创建
SourceSession 前拒绝新 discriminator；不能只加一个旧服务会忽略的字段，也不能自动
降级为旧协议。内部自动流程仍发送 `source_operation: "create"` 并省略
`inference_control`。Gateway、RTC reservation、单人和成员冻结启动验证共用同一配对规则。

手动来源先验证并保存受信启动 prompt、有序参考图与 seed，启动原游戏、CLI、白模和 UI。
此时不构造 PlayStartup，不启动 ModelBinding，也不申请模型租约。CLI 声明 Harness 时，
原条件通道以 `start_suspended` 打开 preview，不等待模型订阅再出首张白模。

CLI 升级不能证明旧封存 World 的 binding 支持订阅暂停。玩家确认开启后，手动 Harness
先在原连接发送 `model.context.subscribe`，携带 `requireSubscriptionPause: true` 和
`paused: true`，原 segment/audience 身份不变。服务端必须从实际 binding 回执
`subscriptionPause: true, paused: true`；此时仍没有条件、样本或模型计算准入，白模继续。
SG 只有验证该精确回执后才创建 ModelBinding/申请模型租约；准备完成后沿原 active
subscribe/switch_segment 激活同一 binding，仍携带 requireSubscriptionPause，回执
paused 必须为 false。重连保留该要求。初始默认关闭阶段不发探测、不创建 consumer。

缺少能力、旧 producer 或旧 World binding 返回
`MODEL_CONTEXT_SUBSCRIPTION_PAUSE_UNSUPPORTED`；provisional binding 由原 SDK owner
释放，SG 不申请模型资源，来源保持可预览。不能用 CLI descriptor、源码 pin 或手写
capability 代替实际回执。内部 auto 路径不发送新增字段，保留旧 subscribed 形状；
auto 来源不声明或接受手动推理控制，以免把旧 World 误报为可以暂停。

玩家确认后，客户端通过原 source 控制连接发送：

```json
{
  "type": "command",
  "op": "model.inference.configure",
  "request_id": "original-request-id",
  "source_session_id": "current-source-session",
  "source_epoch": "current-source-epoch",
  "enabled": true,
  "expected_inference_revision": 0
}
```

该操作仅接受开关和期望版本；浏览器不能在这里提供 prompt、seed 或参考图。
手动来源也不能通过 `render.start`、`render.replace`、`prompt.replace`、`render.prepare`
或 `asset.upload` 绕过受信冻结输入。游戏输入仍走原 owner；成员只改变自己的模型通路。

RenderJournal 在原 request/source/epoch 下冻结意图并执行 CAS，受理后立即返回，不等待
模型准备或清理。独立 SourceInference 控制任务调用原模型与 segment owner。
新的开启意图在同一次 journal CAS 中选择 model 显示，关闭选择 whitebox，并在模型
任务启动前发送原 source.status 的 display_mode/display_revision。无需再依赖一条可能
被限流或 CAS 拒绝的 display.configure 才能显示模型画面；客户端仍须等真实模型像素。
相同期望的 no-op 和历史请求重放不覆盖后来显式选择的调试画面；旧关闭重放也不会
围栏后来已开启的新段。
普通来源每次开启使用 journal.start / RenderSegment、冻结 prompt/图序/seed、当前真实源帧
和新模型身份；不复用已经转交或关闭的 PlayStartup。Harness 首次恢复原订阅，后续使用
start_model_context_segment / switch_segment，仍由游戏条件 owner 提交完整上下文。
子操作 ID 从原 inference operation ID 和操作类型确定，查询恢复不会另造一轮子请求。

手动来源 `session.ready` 的 `capabilities.model_inference_control` 为 true；ready、可靠
`render.status`、`source.status`、操作结果、HTTP 状态和重新连接快照携带：

```json
{"inference":{"enabled":false,"revision":0,"state":"disabled"}}
```

state 为 disabled、starting、enabled、stopping 或 failed，可附稳定英文 code。
enabled 是已受理的期望值；state enabled 表示原模型连接就绪，不代表首帧或可见画面。
新来源默认 disabled；恢复原来源读取已有状态。相同期望在准备中重复发送为无额外准入的
no-op。启用期间模型自行结束或失败时，当前手动推理状态改为 failed，并恢复白模。
关闭取代尚在准备的开启意图时，原 configure 结果为 completed，附
`superseded_by_inference_revision`。新 revision 的 CAS 同时终结旧 accepted configure 意图，
即使旧任务还没开始调度也不会留下永久 accepted。相同期望且未改变 revision 的 no-op
保留原准备任务。该终态只说明控制意图被取代；stopping 仍不表示资源已释放，也不修改
任何模型条件的 unknown 回执。

false/failed 的可重试关闭返回可选 `inference.retry`：

```json
{"request_id":"original-stop-id","expected_inference_revision":1}
```

这里的 expected 值来自原关闭请求；权威 revision 为该值加一。只有玩家明确点击重试，
客户端才使用原 request_id、原 expected 值和相同 source/type/op/version 继续同一条冻结关闭。
服务端原子恢复 stopping，复用原 operation、revision 和清理 owner，不再追加记录或申请模型。
后续开启已经改变来源状态时，旧关闭重试会被拒绝。ModelLeases 只对保留的原 assignment
重查 verified release；缺少原 assignment 时仍报告失败，不重新调用 admit 猜测结果。

新手动来源的 journal 根状态另存策略标记。普通写入在原 2 MiB 硬上限内留出 16 KiB：
4 KiB 用于最终关闭/重试记录和 source/lease 终态，至多 16 个待处理 prompt/event 各留
768 bytes 供原回执结算。正常 1024 条历史之外只保留一条最终关闭逻辑记录，失败重试
继续同记录；无资源的 disabled no-op 和 enable 均不能使用关闭保留空间。
内部自动来源保持原 2 MiB 接受边界；全局字节、模型条件和原超时预算不增加。

关闭先围栏条件、段切换和新模型帧，再停止实际模型子段并等待 ModelBinding 父租约的
原生清理。子资源与父租约确认 released，或从未申请资源，才报告 disabled。
清理不确定时保留 stopping/failed；原已派发但丢回执的模型条件继续保持 unknown/reconciling
及新段恢复屏障，不伪造 applied、cancelled 或 GPU 释放。

配置丢回执时查询同一原 request ID，不能自动重发写请求。可靠控制连接的
`operation.get` 与 configure 共用单串行队列；旧连接在入口和队列消费处均受 generation
围栏。对应 `source.status` 回显 query 的 request_id、原 operation_request_id、
connection_generation 和 operation。只有同 source/epoch/generation、精确匹配原查询的
operation:null 可证明该 configure 尚未准入；HTTP null 或普通状态快照没有这个保证。
已准入操作继续通过原 journal 查询，模型条件 unknown 的恢复规则保持不变。

CPU 验证入口为 `test_whitebox_inference_control.py`，注册在原 AREA gate。覆盖默认零准入、
双成员独立开关与输入、重复启停、真实模型子段与父租约清理、入场取消、失败状态、CAS、
丢回执查询和 Harness 订阅/新段恢复。模型 peer 使用明确的 CPU 夹具；GPU 模型、真实线上
席位、共享镜像和外部部署仍须分别验收。

## 认证和参考图

沿用 [API Token、prompt 与 reference 接入指南](https://github.com/seedleap/world-test-platform/blob/white_box/docs/whitebox-token-reference-client-guide.zh.md)：有效 WTP API Token 换取一次性短期连接票据，服务端校验 Token 身份与世界权限；任意字符串不能换票。API Token 票据不绑定客户端 Origin。票据有效期 120 秒用于首次入场，不是 5 分钟游戏的时长限制。

`asset.upload` 必须等待上一张 `complete`，并确保浏览器发送缓冲有空间，再上传下一张。单图最多 8 MiB；当前部署每次 reference 集合最多 21 张、总共 64 MiB。按 `session.ready` 能力字段确认实际限制。参考图尺寸不限定为 384×384；输出分辨率与参考图尺寸是两个独立概念。

prompt 和 reference 使用现有 `prompt.replace` 更新接口：同一次命令携带新 prompt 和 reference 描述；先完成新图上传。省略 references 表示保留、`[]` 表示清空、`null` 无效。`accepted` 是接收，`applied` 才是模型条件生效；还需对应输出帧才能确认视觉结果。完整消息、恢复和切换示例见上面的指南。

## named UI action 的正确接口

不需要单独的 RTC action 端点。沿用 input 扩展：

```javascript
// 使用 WhiteboxClient 时，由实例统一维护 sequence，不要另外手写序号。
client.input({
  held_keys: [],
  held_buttons: [],
  events: [],
  actions: [{
    id: crypto.randomUUID(),
    name: "renderGateConfigure",
    params: { epoch: currentGameUiEpoch }
  }],
  ui_source_frame: currentSourceFrame.cli_frame_header
});
```

自实现客户端发送的线格式为：

```javascript
{
  type: "event",
  kind: "input",
  sequence: 0,
  payload: {
    held_keys: [], held_buttons: [], events: [],
    actions: [{id: "unique-action-id", name: "renderGateConfigure",
               params: {epoch: "epoch-from-current-game-ui"}}],
    ui_source_frame: correspondingCliFrameHeader
  }
}
```

空键盘状态和 actions 在同一条 input 内是合法的。这里的空键盘状态会释放当前按键；如果动作触发时仍应持续移动，发送当前真实的 held_keys，不能无条件清空。服务端先转发 `input.state`，再按列表顺序转发 `input.action`。

| 字段 | 约束与来源 |
| --- | --- |
| sequence | 每个新 source 从 0 开始，严格连续递增。键盘、鼠标和自动 UI 动作共用同一序列。恢复原 source 不能重置为 0。 |
| actions | 数组，最多 32 项；每项为对象。动作名必须出现在该世界下发的 UI catalog 中。 |
| actions[].id | 建议显式提供唯一字符串，最多 128 字符；同一动作重传保留原 ID。不要把同一操作在每帧生成新 ID 重发。 |
| actions[].params | 对象；按当前世界 catalog 的 paramsSchema 构造。不同世界可能不同。 |
| params.epoch | 游戏 UI 状态中的业务 epoch；不能拿 Room source_epoch、segment_epoch 或 CLI 传输 epoch 替代。 |
| ui_source_frame | 对应白膜源帧的完整 cli_frame_header 对象，不是 JSON 字符串，不是模型输出帧头。 |
| 整条 input | MessagePack 编码后最多 64 KiB，包含 actions、params 和帧头。 |

`ui_source_frame` 当前用于携带来源上下文；Python action adapter 不依赖它调度执行，不承诺在指定模拟帧执行，也不把它当成动作成功证明。服务端生成 CLI action 的顶层 epoch，客户端只提供世界要求的 `params.epoch`。

该湖畔世界当前 catalog 中的三个动作如下；接入时以实际下发 catalog 为准：

| 动作 | params |
| --- | --- |
| renderGateConfigure | `{epoch: string}` |
| npcSelect | `{epoch: string, profileIds: string[4]}`，4 个 ID 必须唯一且来自该游戏配置。 |
| npcConfigure | `{epoch: string, configured: boolean, model: string}` |

这些动作支持经 RTC 控制链路发送。Room 不提供客户端本地 LLM 服务；`npcConfigure` 成功也不证明客户端 LLM 桥接已工作。需要对话能力时，仍由已有游戏客户端处理其桥接逻辑。

## 回执、自动动作与断线恢复

- 等 `session.ready` 后发送 input。自动配置按游戏 UI epoch / 配置变化触发，避免每帧重复配置。
- `whitebox.input_committed` 确认对应 input 已由 Room 转发处理，可释放该 sequence 的待确认记录；它不等于某个 UI 动作已在游戏中生效。
- 查游戏 `action.receipt` 的 actionId 与结果，并结合后续源 UI 状态确认业务变化。收到 action 拒绝时检查 catalog、业务 epoch 和 params，不要仅因媒体仍播放就视为成功。
- 恢复同一个 source 时遵循 `session.attached` / `session.ready` 返回的输入进度，重发待确认 input 时保持原 sequence、完整 payload 和 action ID。已确认序号的不同 payload 会被判定为 replay mismatch。
- 控制队列有限：最多 64 条待处理消息，render-v2 的共享总量最多 8 MiB + 16 KiB；资产、render 命令、input 共用这一预算。单条 input 的 64 KiB 限制仍保留。不要把 64 条队列容量当作固定发送频率；按回执控制在途消息，避免瞬时无限突发。

## 错误定位

| code / stage | 含义和处理 |
| --- | --- |
| CONTROL_QUEUE_FULL / control | 待处理队列达到真实消息数或字节上限。停止继续堆积，按恢复协议重连，核对发送突发与回执。 |
| INVALID_CONTROL_MESSAGE / control | 控制处理失败。检查连续序号、重放 payload、catalog 动作名和参数类型；用 trace_id 查 Room 原始异常。 |
| RTC_CONNECTION_FAILED / transport | 初始连接尚未完成 attachment 时的建立失败。 |
| RTC_DISCONNECTED / transport | 已建立 attachment 后连接中断；不能按初次 ICE 建连失败处理。 |

浏览器只收到稳定错误码和安全文案；原始异常类型、message、stack 保留在 Room 日志。反馈 trace_id、时间窗口、source_session_id 即可让服务端调查，不需要发送 API Token 或图片内容。

## 2026-10-22 故障的服务端修复

trace `bcde0779-18a9-4da5-b29e-8cdcb8b49595` 的首个 Room 异常为 `ValueError: CONTROL_QUEUE_FULL`，发生在北京时间 10:55:40.406，调用链为 `rtc_routes.receive → RenderPipeline.accept → RoomPipeline.accept`。旧实现允许资产进入约 8 MiB 的共享队列，但 input 路径又拿同一个队列字节计数与 512 KiB 比较；因此一张合法的 729,992 字节图片和一条正常 UI input 同时到达即可错误拒绝 input。异常又被统一包装成 `RTC_CONNECTION_FAILED`。

修复统一共享队列的容量检查，保留单消息及总队列上限；控制处理失败记录原始异常，并区分控制错误和连接中断。这个缺陷也存在于共用该 Room 实现的 WS 媒体入口，单纯切换媒体协议不构成可靠规避。该证据不能用来指认某一个 NPC 参数错误，也不能把历史后续每次断线都推断为同一个原因。

代码回归入口：`test_whitebox_control_queue.py`、`test_whitebox_rooms.py`、`test_whitebox_rtc_lifecycle.py`。线上部署版本和实际验收结果随修复 PR 记录；未部署前不能仅凭文档认定线上已修复。
