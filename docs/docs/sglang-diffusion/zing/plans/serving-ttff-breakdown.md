---
title: "生产 TTFF 分时观测与 Prompt 联合执行"
description: "按 render 开启详细分时，保留基础 TTFF，复用既有采集和归档"
---

# 生产 TTFF 分时观测与 Prompt 联合执行

本 PR 区分 Room 控制/供帧等待与 Encoder、DiT、Decoder 执行，避免把线上端到端 TTFF
直接与离线模型耗时比较。2026-11-12 上海副本已部署候选；客户端发送起点和播放器呈现仍待接入。
单 service 单活跃会话、模型生命周期与既有 Host/CUDA 计时边界不变。

## 使用方式与默认行为

基础首帧 `segment_setup_to_first_rgb`、切换 `room_accept_to_first_rgb` 始终进入 Room
现有 Histogram。详细计时初始默认关闭；Room 启动环境 `WHITEBOX_DETAILED_TIMING`
提供新 render 的默认值，`render.start` / `render.replace` 的显式 bool `detailed_timing`
优先。分离模型直接请求在 WS init 使用同名字段，沿 Encoder config / DiT open →
Decoder output_session 传递。每个新 render 重新取初始值，不继承上一段的动态覆盖。

2026-11-13 新增 `model.timing.configure` 控制已有 streaming segment，三角色协商能力并
回执生效边界，详见文末操作示例。无需为当前会话切换再重启 Room；Worldkit/WTP 的界面
按钮仍待客户端接入。中途开启只能采未来阶段，完整首帧须在创建 render 时开启。

全程关闭时不创建本 PR 的角色详细计时/CUDA events，不发送详细回执或写详细 JSONL；
Room 仍保留基础总窗口。动态关闭后不创建新的详细计时对象，关闭前已开始的对象和在途
回执继续正常结束。原有 cache、CLI、运行所需 CUDA fence 不受此开关影响。
开启时旧版本角色缺项保持缺失；动态控制遇未声明能力的角色直接拒绝。
配置开启不是覆盖完整或优化命中的证明。manifest 保存初始 `detailed_timing` 和实际
`timing_changes`；CSV 区分 disabled、not_recorded、incomplete、complete，缺项不补零。

```bash
python benchmark/whitebox/summarize_ttff.py /path/to/archives \
  --environment prod-shanghai --output /path/to/report
```

输出 `first_frame.csv`、`prompt_switch.csv`、`inference_stages.csv` 和 `summary.json`。
每项包含有效样本数、缺失数、P50、nearest-rank P95 和最大值，并按 trigger 分组。
三个环境分别导出；一个 render 一条首帧，重开 render 不等于新试玩。关闭详细计时时，
总窗口从 metrics 查看，CSV 仍可导出既有 manifest 的切换/cache 信息，不能补造逐 case 分解。

## 分时口径

所有归档/CSV 时长单位为 ms。Room 首 RGB 是输出消费入口，不是播放器呈现。
`segment_setup_to_first_rgb` 不含此前世界创建/排队；`room_accept_to_first_rgb` 不含客户端、
Gateway 和 Room 接纳前校验。`source_session_to_first_rgb` 从 SourceSession 建立计时，
重开 render 会包含此前会话时长，不应当作该次新世界的启动时间。

| 范围 | 新字段/阶段 | 起止及限制 |
| --- | --- | --- |
| 首帧 | `source_session_to_first_rgb` | SourceSession 本地建立至首次模型 RGB；不是浏览器点击或 Gateway 世界创建入口 |
| 首帧 | `segment_setup_to_first_rgb` | RenderSegment 建立至首 RGB；不含此前 render 排队 |
| 首帧 | `segment_setup_to_source_dispatch`、`source_dispatch_to_first_rgb` | 同一个 Room 单调时钟；dispatch 在发送首 source.append 前记录，包含发送/对端等待 |
| 切换 | `condition_dequeue_to_first_rgb` | Room 取出条件到实际使用该版本的首 RGB；不冒称入口接纳到 RGB |
| 切换 | `room_accept_to_first_rgb` | Room journal 接纳条件至首 RGB；不包含客户端发送、Gateway 转发或接纳前校验 |
| Room 排队 | `room_accept_to_condition_dequeue` | journal 将条件加入待执行队列至 RenderSegment 取出；包含持久化、其他控制及 sink 等待 |
| Room 准备 | `condition_dequeue_to_model_send`、`room_accept_to_model_send` | 取出/接纳至持有模型发送锁后开始编码和发送 prompt.update；前者嵌套于后者 |
| 模型回执 | `prompt_send_to_accepted`、`accepted_to_condition_ready` | Room 开始发送至收到 accepted，以及收到 accepted 至收到 ready；不是对端纯计算或纯网络时间 |
| 恢复输入 | `condition_ready_to_resume`、`resume_to_fresh_source_dispatch`、`resume_to_first_rgb` | Room 收到 ready 至允许恢复、允许恢复至首份新 source dispatch，以及允许恢复至首 RGB |
| 其他链路 | `sink_idle_wait`、`effective_boundary_wait`、`reference_resolve_store`、`send_lock_wait` | 分别记录 sink、源有效边界、参考图解析存储和发送锁 |
| 其他链路 | `prompt_send_to_accepted`、`old_clip_input_wait`、`old_clip_dispatch_to_condition_ready`、`resume_to_fresh_source_dispatch` | 明确“旧 clip 补帧慢”和“模型 ready 慢”；后者还含服务调度/网络，不能称纯 GPU |
| Encoder | `encoder_queue_wait`、`condition_cache_lookup`、`media_prepare`、Qwen 调用、`reference_vae` | 复用 codecs 准备入口；cache 复用标 skipped；Qwen 调用窗口不是远端服务内部排队/纯算时 |
| Encoder | `source_prepare`、`source_vae_append`、`source_vae_pop` | 记录到首个该版本 DiT 输出通知为止的 source dispatch，带 source 范围；异步缓冲不能只测 append |
| DiT | `condition_queue_wait`、`condition_admission`、`condition_activate`、`history_kv_rebuild`、`condition_ready_wait` | admission、KV 操作和既有 readiness 同步分别计时 |
| DiT | `condition_to_source_ready`、`first_output_compute` | 前者是条件处理结束到首块可执行的窗口，包含输入与调度；后者包围实际迭代器执行，不含 await 输出发送 |
| DiT | `initial_joint_step0`、`prompt_joint_step0`、`initial_joint_graph`、`resume_joint_graph`、`source_step0_graph`、`denoise_graph`、`denoise_steps`、`condition_prefix_apply` | 沿实际分支记录；Graph 内部不拆 kernel；joint 不拆成三份相加 |
| Decoder | `decoder_queue_wait`、`vae_decode`、`rgb_copy_publish` | 在实际解码线程计时，覆盖队列/执行 gate、分片 decode、RGB 转换与 D2H 发布；不是远程代理提交耗时 |
| Cache | 原生回执及 `cuda_ms` | 保留 cache key、结果、miss reason、host stages 和设备恢复时间；所有逐版本回执另进入追加归档 |


只有以下六段能闭合为 `room_accept_to_first_rgb`：
`room_accept_to_model_send` + `prompt_send_to_accepted` + `accepted_to_condition_ready` +
`condition_ready_to_resume` + `resume_to_fresh_source_dispatch` + `source_dispatch_to_first_rgb`。
CSV 仅在全部齐全时计算 `room_path_closure_error_ms`。其他阶段嵌套或重叠，不能相加，
各阶段 P95 也不能相加。`sink_idle_wait` 是消费循环等待，不是请求独占耗时。

Room 接纳标记带进程时钟身份，恢复旧 journal 后不可跨进程相减；accepted/ready 在回执
处理入口计时。角色只报告本进程时长，Host 区间与 CUDA stream 区间分别呈现。
首块完整 Decoder 报告可能晚于首张 RGB；报告数量不是试玩数。

## 数据归属与归档优先级

`InferenceTiming` 和 `TimingBuffer` 由角色持有，按原 dispatch/输出边界非阻塞查询 events。
报告使用稳定 report_id、递增 revision、role、prompt_version 和 clock_domain；pending
不进直方图，同一终态只计一次。Room 校验 session/generation，再清洗、采集和归档。
去重保留最近 8 个版本、每版本最多 256 个 report，阶段名由固定白名单限定。
30 秒未完成或退出时仍未完成的 CUDA 区间标 unresolved，不填零，计时热路径不新增同步/collective；显式切换沿原控制通道确认所有 rank。

AsyncRecording 的计时使用独立缓冲：最多 32 条待发送，连同一条在途写入总计不超过
256 KiB。沿用同一个 worker，必需队列优先；视频、参考图和配置的队列/字节预算与确认
序号不受计时占用影响。计时超限或过期只丢计时并保持 incomplete；不得清除先前缺失标记。
必需数据本身超限仍按原策略停止归档，不掩盖真实归档失败。
已经送入 worker 的一次写入不能抢占，仍使用既有超时机制；该设计不意味着零 I/O 开销。
其他 Recording/ProcessRecording/StreamingRecording 消费方保留同一报告格式。

## 接入现有 Grafana

复用 Room `GET /metrics` 和 `world_model_critical_path_stage_duration_seconds`，不新建
看板或角色 HTTP 服务。2026-11-11 只读核对的现有入口：

| 环境 | 看板 UID | region | pool | 附加过滤 |
| --- | --- | --- | --- | --- |
| 上海正式 | sglang-whitebox-shanghai | cn-shanghai | zing-demo-shanghai-01 | — |
| 泰国正式 | sglang-whitebox-thailand | ap-southeast-7 | zing-demo-thailand-01 | — |
| 上海测试副本 | sglang-whitebox-test-shanghai-copy | cn-shanghai | test-shanghai-copy | deployment_id="test-shanghai-copy" |

[环境导航](https://grafana-cn-2104moyj101.grafana.aliyuncs.com/d/sglang-whitebox-production)
已有 `business_line_id="world-model-data"`、`service="room"` 等采集标签；不要将本地 runtime
的 service 参数或旧 environment/target 模板直接套成线上查询。由已有 collector 保持环境
映射，部署后核对真实标签。下面是上海正式切换 TTFF 的 P95 查询（单位秒）：

```promql
histogram_quantile(0.95,
  sum by (le) (
    rate(world_model_critical_path_stage_duration_seconds_bucket{
      business_line_id="world-model-data", region="cn-shanghai",
      pool="zing-demo-shanghai-01", service="room",
      stage="whitebox_ttff_prompt_switch_room_host_room_accept_to_first_rgb"
    }[5m])
  )
)
```

首帧替换 stage 为 `whitebox_ttff_initial_room_host_segment_setup_to_first_rgb`。
详细项使用 `whitebox_ttff_{initial|prompt_switch}_{role}_{host|cuda}_{stage}`，例如
`whitebox_ttff_prompt_switch_dit_cuda_history_kv_rebuild`。同窗以 `_count` 的 increase
显示有效样本数；没有新样本显示 No data，不用零或固定历史值兜底。不跨环境汇总分位数。
只有显式开启的 render 才进入详细分时分布，不能当成所有试玩的无偏分布；基础总窗口仍全量。

上线依次更新 Room/模型角色、核验 /metrics、现有 collector、Prometheus 数据源和现有
看板分组。代码合并不会自动部署或修改线上看板。本轮 Token 只能读看板，面板更新需由
有编辑权限的维护者完成。只核对源代码/JSON 不等于线上 scrape 验收。采集器原有每 target
4000 样本限制仍需测整个 target，多标签组合不能用单组合白名单预算代替。

## Prompt 联合执行与观测分开验收

同一 #662 保留生产接线。独立 DiT 的 `--zing-source-prompt-joint` /
`--no-zing-source-prompt-joint` 优先于环境变量，未配置 false；集成式 ServerArgs 的
默认值不同。Room/Encoder 协商 source_condition_deferred_kv 后才发送 allow_deferred_kv。
2026-11-16 同步 main `6247343cc2fd` 后，新路径遵循 `clip-barrier-v2` 的 source_ready
与自动 sink 协商。旧 v1 显式协商 early release 或 deferred KV 时接纳 source_ready，
v1 同步完成时仍可返回 kv_ready；v2 只用 source_ready 放行。Room 同时校验切换事件与边界。
DiT 在旧 clip 安全边界准备历史，返回 kv_deferred；Encoder 转为 source_ready，Room
恢复新 source 后再执行已有联合 forward。只有新版本实际 RGB 才结束 TTFF/确认 applied。
旧端、未协商、关闭和不支持布局仍走串行 kv_ready；KV admission、cache、取消、释放和
下一会话隔离沿用既有所有者。实际命中看 forward_counts.prompt_joint 及 fallback，
不是仅看开关。INITIAL_PREFIX_JOINT 独立控制。

联合执行计入新 source 后的 prompt_joint_step0 / first_output_compute，source_ready
不等于 KV 已建好。manifest/CSV 的 readiness_state 区分两种回执。
验收分两组：固定 joint 配置，仅切详细计时测观测开销；固定计时配置，仅切 joint 测优化
收益。两组都须同硬件/checkpoint/输入/布局，覆盖 480、736，报告三角色耗时与常驻/峰值
显存、TTFF、FPS、TPOC/ICL、样本数及 P50/P95。新数据尚未执行，不能声称开销为零或加速。

## 验证与 review 入口

本轮同步远端 main `df057b0ec3cab9759c2650f8d55d8891b75392d0`，消除部署说明和
Encoder 测试的追加冲突，保留双方内容。CPU 回归使用 WSL/Python 3.11/torch 2.8.0+cpu，
轻量 runtime；h100 仅为 fixture 选择，不是硬件实测。同步后 `79338cb3aa` 的 11 个文件
为 **394 passed、46 CUDA skipped**，覆盖计时/导出、异步归档、render 合同/生命周期、
cache、AREA、推理控制、Encoder API、Encoder–DiT、QueuedDiT 和远程 Decoder。
本 PR 的 47 个 Python 文件通过固定 Ruff/isort/format，diff whitespace 检查通过。
原合并进程被系统终止，未作成功计数；以上为按文件独立进程重跑的完整 JUnit 结果。
历史测试流水保留 Git 历史与实验记录，不与当前样本数累加。

自审核对方案必要性、实际参数/回执链路和上述测试覆盖；修正了可选计时污染必需
归档队列的反例，并将 Encoder buffer 初始化移到生命周期入口，去除热路径重复检查。
已检查范围未发现新的阻塞缺陷。GitHub CI 另曾在安装 Redis 时因 Ubuntu 源 DNS
解析失败、退出 100，未进入该 job 的合同测试；远端最终状态以 PR Checks 为准，
不能把依赖安装失败或旧提交的绿灯当作当前全量 CI 通过。

重点：关闭时模型不创建详细 events/回执，基础 TTFF 仍可 HTTP 读取；连续两个 render
开关隔离；计时先入队仍能写视频；字节溢出/过期、参考图 ACK 和最终缺失标记；两种
readiness 的真实接收→source→RGB→metrics/CSV 六段闭合；既有取消与资源释放。

阅读顺序：render_contract/RenderSegment → Encoder adapter/executor →
QueuedDiTSession/SourceReadySession → Decoder worker → AsyncRecording/TimingMetrics → CSV。
模型计算不依赖 Room，复用现有 Histogram 与异步 worker，未引入 profiler 服务。
参考 [SGLang profiling](https://docs.sglang.io/docs/sglang-diffusion/profiling) 和
[PyTorch CUDA Event](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html)：
stream 区间不等于纯 kernel busy time；GPU、SP、真实效果、两档性能及线上 Grafana
仍未验收，CPU 结果不能替代。

### #662 测试部署的恢复边界

全 PAI H200 SP1 测试组在 DiT 首次端口冲突后以独立作业重试。为保留已加载的 E/D，
使用显式原部署/重试部署记录核验身份，再互相复用角色，所有权保留在创建它的 run。
不引入所有权迁移或通用 compose；不伪造自测通过。恢复工具仅改变私有部署记录，
标准 inspect/selftest/attach 仍逐项执行。DiT reuse 限定为已固定身份的 PAI 单 GPU
单 service；共享角色始终不由借用方重新注册或停止。该适配不进入推理热路径。

阅读入口：`benchmark/whitebox/reconcile_role_retry.py` → orchestrate 的 reuse 校验与
owned_roles → inspect/selftest/attach/down。验证覆盖终止失败状态、错源码、错复用
Worker 身份的拒绝，以及成功后两份记录的所有权与历史保留；真实平台出帧仍待验收。
这些是私有部署工具适配，不作为 SGLang 上游通用功能或模型性能证据。

### 网页试玩的部署默认值（2026-11-12）

网页未传 `detailed_timing` 时，Room 启动配置 `WHITEBOX_DETAILED_TIMING=0`
（默认）或 `1` 决定新 render 的详细计时。RenderHost 启动时严格解析该值，
通过 RenderJournal 默认参数传递并持久化到 segment；自动 model-context、startup
及显式 render.start 均沿 journal.start 消费。请求显式 bool 优先，非法请求仍拒绝。
仅修改启动默认值仍需更换/重启 Room；当前会话的动态覆盖见下节，不读取动态配置文件。
恢复用的 journal 仅读已有状态，不据当前默认值改写已创建 segment。

两轮手动对照使用同一份代码、SDK、模型、联合 forward 和缓存策略，只改变该环境变量。
应先退出游戏并确认 Room 当前 owner 没有活跃 source，再切换 Room；模型池空闲本身不足以判断。
关闭时仍保留 `segment_setup_to_first_rgb` 和 `room_accept_to_first_rgb` 两项基础指标，
详细模型 Host/CUDA 分时和 Room 子阶段仅在开启时采集；两轮比较基础 TTFF，不以缺项代替零。
CPU 验证覆盖默认值/显式覆盖、持久化、幂等重试及非法启动值；GPU A/B 与 Grafana 入库仍待验收。

## 2026-11-12 测试采集链路修复

Shanghai copy 的原生 FC 采集器固定了旧容器名，并在主机和 relay 两层限制 stage。
因此 Room /metrics 有新值不等于 Grafana 有值。当前基线包 SHA256
`d2970807683ab037bc6327814b038b57b6d5de4002f749cb1cf9157ee6184afb`。
`deploy/whitebox/patch_test_timing_collector.py` 是这个已部署测试包的有界更新工具：
校验完整原包及两个修改模块的 SHA256，保留其余依赖，从本 PR telemetry 合同导出固定 stage 集合，
在两个过滤点同时添加，增加受限文本/series预算，并按显式参数绑定当前测试容器。
只更新 controls 读取命令及测试 FC 的两个过滤点；不更改推理服务、池采集命令或正式 FC，也不发通知。
新包和主机命令均从已推送的脚本生成；后续 Room 改名须重新核实并更新绑定。
这是内部测试部署兼容工具，不作为社区通用 FC 集成接口。

本地实测：1548 条真实分时 bucket/sum/count 经两层过滤及 remote_write 编码完整保留；
压缩主机包 10007 字节，解压往返一致；未知 stage 被丢弃，过期身份凭据被拒绝。
云端采集/存储验收尚未完成。2026-11-12 后续核验 Grafana 已可编辑并保存测试看板；
2026-11-13 只读复核发现测试 FC 的 relay 仍过滤新指标，完整包发布缺少 fc:UpdateFunction 权限。
看板保存成功不代表详细分时已入库。本次动态开关改动不改变该线上状态。

2026-11-12 19:39 快照中最新八个 render 的分时归档均为空，其中两个标记 incomplete。
Room 聚合 metrics 仍存在，不能把聚合值反推为逐 case 分项。
该归档丢失与“游戏开始按钮无响应”分别跟踪，均未宣称修复。


## 当前会话动态计时控制（2026-11-13，代码候选，未部署）

用途：在线试玩过程中开启或关闭后续详细分时，无需重新创建 render 或重启服务。
`WHITEBOX_DETAILED_TIMING` / `render.start.detailed_timing` 继续只决定新 segment 的初始值。
动态覆盖只属于当前 segment/model session/generation，结束后不继承到下一玩家。

客户端在原 Room 控制连接发送 `model.timing.configure`，携带原有 source/segment 身份、
唯一 `request_id`、`enabled: bool` 和 `expected_timing_revision`（初始为 0）。
Room 使用既有 operation journal 幂等记录，并把 `timing.configure` 事件传入原 Encoder
连接；仅在模型三角色全部返回实际采用回执后报告 completed。旧角色未声明能力时拒绝。
超时不能冒称失败未生效，operation 保持 reconciling，原 request_id 可查询；并行切换被拒绝。

实现边界：Encoder 在原单线程执行池内处理，DiT 在原有序输入服务内处理，Decoder 在原
channel 命令队列内处理。各自生效边界分别为下一 Encoder command、DiT 已接收 source
水位、Decoder 下一次提交的 output frame，不宣称全链在同一墙钟瞬间切换。
关闭阻止后续新详细计时对象；已有对象仍在原回执通道自然完成，不新增 CUDA synchronize。
回执携带 timing_revision，Room 在 OFF 后仍接收当前 generation 的在途详细回执。
不重置已观察的 prompt/output 标记，不把中途开启伪装成完整首帧；Room 已开始的基础
TTFF 继续保留，详细阶段只用于新开始的条件操作。

复用现有有序控制通道而非新增 HTTP 管理服务器或全局变量，避免跨玩家状态、额外线程
和生产管理端口。多 rank 继续通过现有 leader_call/输入广播路径传播控制。

验证计划：CPU 覆盖真实控制入口、身份/旧 revision 拒绝、ON→OFF→ON、旧 peer、重复
request、取消和下一会话隔离、OFF 后在途计时；测试不替代 CUDA/NCCL 或线上验收。
480/736 同条件性能与三角色显存仍须后续实测；本次不自动部署或申请 GPU。

### 操作方式及回执

先从原控制连接的 `status.get` / `render.status` 取得 source、segment 身份和
`segment.timing.revision`。在 **同一玩家原有 Room 控制连接**发送：

```json
{
  "version": 2, "type": "command", "op": "model.timing.configure",
  "request_id": "timing-on-1",
  "source_session_id": "<current-source>", "source_epoch": "<current-source-epoch>",
  "segment_id": "<current-segment>", "segment_epoch": "<current-segment-epoch>",
  "expected_timing_revision": 0, "enabled": true
}
```

先收到 accepted，最终 `operation` 为 completed 才表示已采用；最终回执包含
`timing_revision` 与 `boundaries.encoder.command_index`、`boundaries.dit.source_frame_end`、
`boundaries.decoder.output_frame_start`。关闭发送新的 request_id、最新 revision、enabled=false。
重试原 request_id 只查询/重发原操作回执，不再次切换；缺失或错误身份不更改运行中的玩家。
超时为 reconciling，禁止继续堆叠开关，以原 request_id 查询或等旧回执；会话取消后不再发送。

仅 streaming 的当前 segment 支持该操作。要记录新世界的完整首帧，仍须在该 render
创建前开启初始 detailed_timing；中途开启只能得到未来阶段，不能补录已经结束的 Qwen、
KV 或首 RGB。新界面按钮/Worldkit SDK 封装不在 #662；本 PR 提供 Room 控制协议。

归档 `manifest.timing_changes` 保留实际成功的切换 revision/边界，数量受现有 operation
journal 总预算约束。JSONL/明细 CSV 携带 `timing_revision`，不把它加入 Prometheus 标签。
同一条件出现跨 revision 的片段时 CSV 标为 incomplete，不拼成一轮完整计时；动态开启有
实际回执时不因 manifest 的初始 false 误标 disabled。OFF 后的旧在途报告可继续到达。

### 结构与社区惯例

SGLang SRT 已有 `http_server.py` 的 `/start_profile` / `/stop_profile` 显式管理命令；此处
沿用显式控制、实际完成后回执的做法，但按实时模型的 session/generation 隔离，不引入
进程级 profiler。`TimingControl` 只负责 Room 的 journal 与 pending 生命周期；EncoderClient
负责原执行池串行化；DiTInputService 和 Decoder worker 仍是各自状态的唯一写入者。
审查重点是有序通道/ACK、取消后不能发送、OFF 后异步回执以及跨采样窗口报表。
2026-11-13 合并 SHA `8ccdabddb26cf1a9bcd99cc99db6a09008dbcc60` 在 Linux CPU
环境验证：动态控制 22 项、Room 控制路由 14 项、首帧标记 1 项、prefix 真实 HTTP
metrics 1 项、双进程 Gloo 控制/张量 1 项通过；首帧标记的 CUDA 参数 1 项跳过。
统一采集器 33 项通过；本轮变更的 28 个 Python 文件通过 isort 7.0.0、Ruff 0.15.1
检查。合并前另通过 render contract 48、Encoder API 64、render lifecycle 69 项。
Windows 挂载路径上的广泛 transport 回归未完成（进程中止，部分输出含失败），不列为
通过；同一合并 SHA 的双进程 Gloo 定向测试在 Linux 原生路径复验通过，不能据此
宣称整个 transport 套件通过。480/736 CUDA/NCCL 与线上使用仍未验收。
原始日志、JUnit、源码包和采集包回执保存在交付目录
`D:/CodexArtifacts/pr662-dynamic-timing-20261113/`；这是本地证据路径，不是线上部署凭据。


## 合并 #687 的 Qwen prefix 观测（2026-11-13）

按用户要求，将 #687 `aae1496b9bba7553539f9eeb21ece18a5e39cea4` 合入 #662，
同一交付分支包含 TTFF/动态计时与 Qwen prefix 观测。#687 是对已存在前缀复用的指标
扩展，不是新的 Qwen 缓存算法；完整合同见
[Qwen 前缀复用监控](../specs/conditioning-prefix-api.zh-CN.md)。

两个入口职责不同：Room `/metrics` 提供首帧/切换分时；remote conditioning 服务的
管理员认证 `/metrics` 提供前缀 hit/miss/unknown、复用 token、编码/排队和交付来源。
独立 Encoder 的本地 Qwen 路径通过 SessionPrefixConditioner 接入同一组 prefix 指标；以独立认证的 Encoder /metrics 采集。部署时仍须区分本地与 remote conditioning 目标。
conditioning 聚合指标常开，不由 render 的 detailed_timing 控制。后续计时 A/B 应使用
同一个合并 SHA，两组保留相同 prefix 配置与采集，只切 detailed_timing，不能混入算法差异。

采集包可一次准备：`prepare_conditioning_collector.py` 以已有 #662 r3 包（SHA256
`a80254563127d1bc76448a390b96ae4984ca7b7100e68371c88cf6056682f402`）为基线，
保留其 TTFF stage 支持与 Room host，仅扩展 relay 合同并添加认证 reader。reader 仍须
绑定真实、可达的 conditioning 目标/当前身份 proof/Secret/既有采集周期；合并代码或构建
ZIP 不代表已经绑定或上线。测试 FC 的发布权限限制也不会因合并自动消失。

统一采集包已本地构建，SHA256
`87fad76a03525b62d5574cfbd9a09203bafaa4aa67c6abbec00102e407e5c539`。
逐条比较确认：4034 个原有 ZIP entry 内容不变，仅修改 `relay.py`，新增两个
conditioning 指标模块，未删除 entry。此包尚未发布、未注册 conditioning 目标。


## 共享测试分支（2026-11-13）

用户选择以 `main@f27559be499487b9374cc325289e49b81261e4e1` 新建
`codex/shared-test-ttff-prefix`，合入 #662
`b53707917d973807d58833a90583d956d79f66fa`（已含 #687 与独立 Encoder prefix 接线）。
该分支用于多人共用的功能集成验收；未合入 main，未更新在线服务。
它包含 main 的后续优化，不能冒称与先前 GPU 基线仅差计时开关的 A/B。

### 使用与开关

- Room 部署设置 `WHITEBOX_DETAILED_TIMING=1`；显式 render 请求仍可覆盖，当前会话可走
  `model.timing.configure` 协议。仅环境变量不够，必须部署本分支的 Room 和三角色。
- 独立 Encoder 设置 `SGLANG_ZING_QWEN_PREFIX_BYTES=4294967296` 作为本次测试预算，
  `SGLANG_ZING_CONDITIONING_METRICS_TOKEN_FILE` 指向部署的只读密钥文件；密钥不入库。
  Prefix 指标常开，和 detailed_timing 独立。预算仍需目标卡实测，不能保证所有档位都可用。
- main 的 `return_stage_timing` 继续控制逐帧时间线；不因 TTFF 计时开启而自动开启它。
  集成测试覆盖两个 boolean 的四种组合，不能把两套指标重复相加当作端到端耗时。
- Grafana 需要采集实际目标及认证绑定，提交分支不等于采集已上线。

### 合并边界与阅读重点

保留 main 的 Encoder 数值路径、逐帧时间线、source early-release、provisional boundary、
Decoder 渐进输出与集中 open-response 解析，再接入 #662 的阶段计时与动态控制。
Encoder 的 source 准备和 cache 查找保留原语义并加计时；Decoder job 同时持有逐帧
`timing_state` 与 TTFF `timing`，生命周期各自沿原通道完成。Room 对 source_ready 的接纳
要求 peer 声明 early-release 或 deferred-KV 能力。Remote conditioning 的 /metrics
保留管理员鉴权，同时暴露 main 原有全局指标与 prefix 专用 registry。

Review 路线：Encoder adapter/init → SourceEncoderExecutor → DiTInputService →
Decoder remote/_opened/submit → DecoderOutputBroker → Room TimingControl/RenderSegment。
重点检查四种开关组合、延迟回执、缓存命中、取消释放、两套计时没有覆盖彼此。
复用现有 SGLang explicit control 与异步 event 观测结构，不增加新的调度层或全局采样开关。

### 验证

本次以新 main SHA 冻结配置，读取 PERFORMANCE_DEFAULTS、PIPELINE_DEFAULTS。
合并前旧候选的 CPU/GPU 结果只保留原 SHA；共享分支必须重新跑冲突相关 CPU 回归。
静态语法、Ruff（Python 3.12）与 diff whitespace 检查通过。

- `17c358c189b169a4d72fa7e03ee02fd1ab1487f5`：三角色/conditioning CPU 回归
  491 passed、49 skipped、1 deselected，另有 3 subtests passed；部署与采集工具 169 passed。
- `49363d35cf4be197c5366569cb8a41cc39eab2a1`：Room 生命周期/计时 153 passed、
  2 deselected（两项组件测试已归入上面的三角色套件）。后续运行时代码仅改 Encoder 回执校验，
  Room 源码没有变化。
- 修复合并引入的 deferred-KV 回执读取了 source 请求而非 prompt 请求的问题；
  延迟到后续 source dispatch 的回执使用冻结的原请求授权，新增该组合回归通过。
- 49 项 skip 保留实际测试条件；历史 fixture 测试首次因源码归档无 `.git` 失败，最终排除，
  未完成其固定旧提交对照。CPU harness 替换不可用的 GPU import 边界，不验证 CUDA/NCCL。

480/736 真 GPU、NCCL、显存、真实首帧/切换与 Grafana 入库均未验证，不能根据 CPU 结果
宣称零性能影响或已经可在线试玩。详细命令、初次失败和复测边界见
[共享分支 CPU 验证记录](../../../../../benchmark/whitebox/SHARED_TTFF_PREFIX_20261113.md)。

## OSS 流式归档补齐（2026-11-16）

上海副本 `016_volcanic_island_navigator` 的实际 Room 为 `bf5371c0d26c`。
15:29:30 关闭、15:30:13 开启详细计时均取得三角色 completed/applied 回执，
但结束后 `inference-timing.jsonl` 为 0 字节。动态开关生效与明细成功归档是两个独立验证点。
此次修复不改变开关协议、阶段字段或 Prometheus/Grafana 接口。

真实调用链为 `write_inference_timing` → 独立 `timing_items` 队列 →
`StreamingRecording._run` → spawned worker → OSS。旧流式消费循环只处理必需队列
`items`，会在计时队列未清空时完成归档；已有磁盘归档路径没有此遗漏。
修复在同一循环中优先处理视频/参考图，再处理计时，完成前排空两个队列。
计时仍使用独立字节预算，不推进必需项 ACK 序号；超龄计时丢弃并标记 incomplete，
异常收尾释放计时缓冲。保留单 worker 与原超时策略，不增加上传进程或推理同步点。

验证使用真实 spawned worker、boto3、回环 HTTP 对象存储与 H264 视频，覆盖正常计时
写入、过期计时不影响视频、必需 ACK 隔离和失败释放。新增 3 项在旧源码全部失败，
修复后通过；stream/async archive 两个测试文件共 28 项通过（Linux CPU）。
这是归档端到端回归，不是 GPU 推理性能证明；480/736 GPU 开销本轮未测。
阅读入口：`whitebox/stream_archive.py` 的 `_run` 和
`test_whitebox_stream_archive.py` 的 `test_streamed_timing_*`。

本节为源码修复与本地验证记录，尚未表示测试 Room 已更新。已丢失的历史 JSONL 无法
事后补录；部署后须用新会话核验实际 OSS JSONL 和 manifest，历史开关回执仍可保留。
