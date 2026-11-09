# Whitebox WebRTC 游戏 UI actions 接入

适用：`whitebox-render-v2` / `native-h264-rtp-v1`。媒体通过 WebRTC RTP，控制消息仍通过 `/whitebox/v1/webrtc/play` 的可靠 WebSocket，以 MessagePack 二进制发送。WS 媒体入口与 RTC 媒体入口共用 Room 的游戏控制实现。

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
