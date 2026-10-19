# Zing 单活跃会话流式运行时设计

本文帮助维护者审查 Decoder、分次调度与输入预取的职责边界，说明当前实现、目标和未验证范围。它不是性能验收报告或用户部署承诺；实验结果与命令保留在 benchmark 目录。#268 的取消清理验收见 [SLOW_CONSUMER_LIFECYCLE.md](../../../../../../../../benchmark/zing_h3_causal/SLOW_CONSUMER_LIFECYCLE.md)。

产品目标是已就绪的每个 latent 独立推进 Decoder，后续自研 Decoder 尽可能在一个 latent 内逐 frame 交付。DiT 的分块计划与 Decoder 的解码粒度独立：当前只采用启动 `1,2,2`、随后循环 `3,2`（`num_frame_per_block=[3,2]`、`split_first_latent=true`），不把五 latent 一组作为后续 Decoder 的目标默认。

当前使用 Lynn streaming decoder。`SourceLynnDecoder` 独立管理每个会话的解码状态、
有界 latent 缓冲和 source provenance。每次 append 通过 `valid_frame_end` 声明实际
有效像素边界，与计算依赖水位分开；解码器逐 latent 执行，pop 立即输出全部有效帧，
只裁剪短尾 padding，不等待下一 chunk 或 EOF。短尾后禁止继续追加 latent；
`finish(total_frames)` 校验已声明的实际帧数，随后 pop 关闭流，不重复输出已发送帧。
TAEH3 专用的五 latent 分组、十二帧暂留和 progressive decode 环境开关已移除。

Decoder 粒度、分次调度和独立 Encoder 部署是不同配置。CPU 测试不代替真实权重、
EOS、画面边界、分布式清理和端到端性能验收。Lynn 的 SP tile 验证入口为
`benchmark/zing_h3_causal/validate_lynn_sp_tiles.py`。

## 分次调度

`SGLANG_ZING_EXPERIMENTAL_SOURCE_WORK_QUANTUM=1` 将每次 scheduler dispatch 限制为最多一个已就绪的 DiT chunk。启动分块为一、二、二个 latent，后续循环三、二个 latent；每次生成后先解码并返回可释放的 RGB，再处理下一 chunk。chunk 不等于一帧 source 或一个 latent。

API 和 worker 必须使用相同的可信部署环境。客户端请求不增加字段，WebSocket 不增加消息类型。内部 continuation 消费已有工作，不重复 append/end；EOS 经 continuation 排空后才完成会话。没有 RGB 的 continuation 也必须继续推进，不能等待不存在的新输入。

该机制本身不允许正在执行 DiT 的 dispatch 接收新 source；跨 dispatch admission/prefetch 是独立功能。取消仍需等待在途 GPU 工作结束后清理。输出背压应发生在下一 dispatch 前，不得累积无界输出队列。

跨 chunk 联合执行也遵守上述工作量上限：一次联合处理的 chunk 数不能超过剩余 dispatch 预算；已有工作待排空时，API 不再等待收集下一批 source。

部署只允许一个活跃会话。第二个会话应被拒绝且不影响当前会话；session/generation 校验及断连、取消、故障、EOS 后释放与下一会话可用性仍需验证。

## 验证与证据

更改任一选项都必须重新冻结源码、配置和输入。对照保持输入和后端策略一致，保留严格的 source、水位、RNG 和 RGB 检查，再采集独立预热、五轮正式耗时、单独 profile 与120秒稳定性结果。默认关闭的实验不能证明默认开启后的性能，历史耗时只属于原冻结版本。

`test_encoder_overlap_tools.py` 验证验收工具的容量、完整输出、统计和资源证据契约；它已接入统一 `zing-ci.yml`，不是 GPU 推理验收。

`zing_source_output` 记录输出缓冲和显存阶段观测，仅在 DEBUG 日志开启时打印。`OutputBatch.usage` 内的资源字段用于内部结果与日志分析，不直接成为客户端帧 metadata；异步 materializer 管理两槽 RGB 转换、D2H、共享内存、背压和取消。

Decoder GPU exact 与完整 profile 证据仍需单独验收。将代码提取或合入 PR 不代表这些门槛已经通过。历史实验和一次性诊断可在各自冻结的 Git 快照中查阅。

## 清理层次与框架复用

#268 不另建 Zing 清理框架。LingBot World、MinWM 和 Zing 共用
`realtime_video_api.py` 的会话清理、`ReleaseRealtimeSessionReq`、
`GPUWorker.release_realtime_session` 与 `RealtimeSessionCache`；模型状态
通过已有 `dispose()` 释放。三层职责如下：

| 层次 | 现有机制与复用方式 | #268 / #274 的边界 |
| --- | --- | --- |
| API 连接生命周期 | 共用最终清理和准入归还 | #268 保护整个清理 task，防止外层重复取消截断清理；不在模型 adapter 各写一份 |
| worker 会话生命周期 | 共用会话缓存释放与输出 materializer 取消 | 保留现有入口和 session 身份，不新增模型专属 scheduler 释放协议 |
| Encoder 角色与控制恢复 | 模型适配层停止 producer、排空队列并释放缓存/bridge | #274 复用 disaggregation 的角色、调度与请求状态，并提供 token 校验和跨 rank abort 共识；#268 不复制这些模块 |

核对 [#274](https://github.com/seedleap/sglang/pull/274) 的冻结版本
`0b2786179d7590f6bba2453c0828204afb3f3b3c`：
`RemoteEncoderSession` 的普通控制 RPC 改用 `request_role_operation`，
恢复走 `request_owner_abort` / `abort_role_owner`，角色占用使用
`StatefulRoleOrchestrator`。这些收敛了 Encoder 的控制和 owner 状态；
角色只有在适配层证明各 rank 清理完成后才能释放，故障组保持隔离。

该版本仍保留 producer 线程、停止与队列排空，以及
`dispose()` 中的 `6 * self.timeout + 2` 等待上限；没有修改共用 API 的
最终清理。因此不能因 #274 就删除 #268 的取消隔离或把 scheduler 等待
恢复为两秒。两者可组合，但不代表等待上限或进程故障自动恢复已简化。

#274 尚未合并，本 PR 不导入其新增模块，也不另行复制实现。合并后应通过
上述已有清理入口自然复用，并在组合版本复验重复取消、Encoder 恢复、
故障隔离及下一会话可用性；历史单 PR 结果不能替代组合验收。

## 私有预取通道的校验

`require` 是 `zing_contracts.py` 定义的显式条件校验，失败抛出 `ContractError(ValueError)`，不受 `python -O` 影响。四步是当前持续 source 接口冻结的部署推理契约，不是 Encoder overlap 数学上只能支持四步。窗口只能来自服务端部署 profile，客户端不得覆盖默认21/3。

会话临时目录用于存放 worker 要读取的参考图，也用于随机命名的 Unix IPC socket；目录跟随会话清理，不是长期结果目录。`_initialize_source_admission` 集中建立预取通道，绑定失败时关闭 socket 和 context。

这里的 PUSH/PULL 是本实现显式创建的 ZeroMQ socket 类型，不是 SGLang disaggregation 的调度、KV 传输或部署语义：

- API 侧 `ZingH3SourceAdapter._initialize_source_admission`（`runtime/entrypoints/openai/realtime/adapters/zing_h3_source_adapter.py`）创建 `zmq.PUSH` 并 bind 会话内 `ipc://` socket。
- worker 侧 `RemoteEncoderSession._produce`（本目录 `remote_encoder.py`）创建 `zmq.PULL` 并 connect 相同 endpoint，再通过独立 REQ 控制 RPC 驱动 Encoder 服务。PULL 位于 RemoteEncoder producer，不能笼统写成 Encoder 服务端。

该同机私有通道单向发送 API 已接受的 source RGB 输入；`LINGER=0` 使关闭时不等待未发送消息。`SNDHWM` 限制消息数量，不代替 ledger 和 Encoder 的帧数/字节预算。`append` 头包含帧起点、数量、宽高等元数据，RGB 位于独立消息段；`finish` 只提供结束水位。它们用于检查顺序、容量和 EOS，与 Encoder 的 `open/pop/close/abort` 控制 RPC 分离。
