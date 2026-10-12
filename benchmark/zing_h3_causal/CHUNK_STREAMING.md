# TAEH3 增量解码与分次调度

这两个可选功能与独立 Encoder 部署分别配置。默认保留五 latent 一组的 Decoder 解码策略，且不限制每次 dispatch 的工作量。这里的增量解码指保留时间状态、分小块解码，不代表一种名为 progressive 的标准流式协议。

## 增量 Decoder

`SGLANG_ZING_EXPERIMENTAL_PROGRESSIVE_DECODE=1` 让 TAEH3 处理已就绪的一到两个 latent，不必等待完整五 latent 组。环境变量的历史名称保留以兼容实验。时间层缓存、每二十个原始帧的裁剪、十二帧暂留和 EOS 填充仍遵守原有输出契约。

完成解码计算不等于可以交付 RGB：累计一个或三个 latent 时可能已完成计算，但没有可释放的画面；累计五个 latent 后才能释放五帧。因此不能声称首个 latent 立即产生首帧 RGB。

benchmark 输入 JSON 需冻结 `"progressive_decode": true`。它作用于 server/consumer，不作用于 Encoder。`--stage decoder` 使用真实权重重放实际捕获的 latent，以零容差比较原路径，包含不规则尾部和 EOS；回执保存为 `progressive-acceptance.json`。CPU 分块卷积测试不代表真实权重的 GPU exact 或画质验收通过。

## 分次调度

`SGLANG_ZING_EXPERIMENTAL_SOURCE_WORK_QUANTUM=1` 将每次 scheduler dispatch 限制为最多一个已就绪的 DiT chunk。启动分块为一、二、二个 latent，后续分块以冻结模型配置为准；每次生成后先解码并返回可释放的 RGB，再处理下一 chunk。chunk 不等于一帧 source 或一个 latent。

API 和 worker 必须使用相同的可信部署环境。客户端请求不增加字段，WebSocket 不增加消息类型。内部 continuation 消费已有工作，不重复 append/end；EOS 经 continuation 排空后才完成会话。没有 RGB 的 continuation 也必须继续推进，不能等待不存在的新输入。

该机制本身不允许正在执行 DiT 的 dispatch 接收新 source；跨 dispatch admission/prefetch 是独立功能。取消仍需等待在途 GPU 工作结束后清理。输出背压应发生在下一 dispatch 前，不得累积无界输出队列。

跨 chunk 联合执行也遵守上述工作量上限：一次联合处理的 chunk 数不能超过剩余 dispatch 预算；已有工作待排空时，API 不再等待收集下一批 source。

部署只允许一个活跃会话。第二个会话应被拒绝且不影响当前会话；session/generation 校验及断连、取消、故障、EOS 后释放与下一会话可用性仍需验证。

## 验证与证据

更改任一选项都必须重新冻结源码、配置和输入。对照保持输入和后端策略一致，保留严格的 source、水位、RNG 和 RGB 检查，再采集独立预热、五轮正式耗时、单独 profile 与120秒稳定性结果。默认关闭的实验不能证明默认开启后的性能，历史耗时只属于原冻结版本。

`test_encoder_overlap_tools.py` 验证验收工具的容量、完整输出、统计和资源证据契约；它已接入 `zing-taeh3-cpu.yml`，不是 GPU 推理验收。

`zing_source_output` 记录输出缓冲和显存阶段观测，仅在 DEBUG 日志开启时打印。`OutputBatch.usage` 内的资源字段用于内部结果与日志分析，不直接成为客户端帧 metadata；异步 materializer 管理两槽 RGB 转换、D2H、共享内存、背压和取消。

Decoder GPU exact 与完整 profile 证据仍需单独验收。将代码提取或合入 PR 不代表这些门槛已经通过。历史实验和一次性诊断可在各自冻结的 Git 快照中查阅。

## 部署校验与私有预取通道

`require` 是 `zing_contracts.py` 定义的显式条件校验，失败抛出 `ContractError(ValueError)`，不受 `python -O` 影响。四步是当前持续 source 接口冻结的部署推理契约，不是 Encoder overlap 数学上只能支持四步。窗口只能来自服务端部署 profile，客户端不得覆盖默认21/3。

会话临时目录用于存放 worker 要读取的参考图，也用于随机命名的 Unix IPC socket；目录跟随会话清理，不是长期结果目录。`_initialize_source_admission` 集中建立预取通道，绑定失败时关闭 socket 和 context。

PUSH 与 Encoder 端 PULL 配对，单向发送已接受输入；`LINGER=0` 使关闭时不等待未发送消息。`SNDHWM` 限制消息数量，不代替 ledger 和 Encoder 的帧数/字节预算。`append` 头包含帧起点、数量、宽高等元数据，RGB 位于独立消息段；`finish` 只提供结束水位。它们用于检查顺序、容量和 EOS，与 Encoder 的 `open/pop/close/abort` 控制 RPC 分离。
