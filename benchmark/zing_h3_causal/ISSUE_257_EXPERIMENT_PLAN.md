# Issue #257 主实验准备

> **Update 2026-10-10:** the SP1 real-checkpoint GPU comparison has since been run (exact parity, about 41 GiB less memory); results and current scope are in [ADALN_TABLE.md](ADALN_TABLE.md#status-and-support-scope). The "not yet run" statements below describe the state at handoff.

日期：2026-10-09。分支：`codex/zing-adaln-persistent-table`；worktree：`/Users/yiwencai/.codex/worktrees/issue-257-adaln-table/sglang`。main 基线：`f43a7f97e0a68a09dcb6956f08b4e275b5120df8`，已包含 #264 与 #258。candidate 由本分支实际 Git HEAD 标识，不能用 main 基线 SHA 代替；执行前必须冻结该 HEAD 的完整 patch/bundle 及 SHA256。交接时远端 main 已推进到 `c5764eb01cb629088bd9a43f161d032c7c6279f8`（#176），本分支尚未集成该提交。

当前阶段：本地实现与 CPU 验证。尚未提交 #257 云作业，没有本功能的真实 GPU 数值、显存或性能结果。本文是预先约定的测量与停止条件，不是验收报告。

## 可证伪问题

1. worker 在初始化与加载期间是否完全不构造/读取/materialize 主层和 final AdaLN 参数，仍能完成实时 video 请求？
2. 同一有效 checkpoint、完整 unique-t 批次与算术策略下，表模式的紧凑参数、每步预测/latent、完整 KV 与最终输出是否严格相等且 finite？
3. 实测 startup/steady allocated 是否降低，表与 KV/graph 生命周期是否正常？相同 SP 的冷/热 E2E 和 chunk p50/p95 是否无回归？

## 已准备的实现

- 离线逐层 builder 与 `zing-adaln-table-v1` 工件，绑定完整 checkpoint/shard、TimeMLP/频率、tag、完整批次与算术身份。
- loader 预读取过滤，meta 构造跳过主层/final projection；模型版本内跨会话共享固定小表，每 rank 一份。
- KV 申请前完整计划校验，显式 lease 与异常/close 清理；未知调度、audio、在线权重/LoRA更新拒绝。
- eager 与原 graph provider 共用取表路径；启动 storage 回执、请求 metrics 与现有真实 checkpoint harness 参数。
- CPU 小模型覆盖四/五支、condition 共享/独立、重排 tag、Euler/few-step、source+六张合成 reference、CFG、尾块、prompt 更新和双 session graph replay。

用法与 schema 见 [ADALN_TABLE.md](ADALN_TABLE.md)。CPU 的数值一致不替代真实 CUDA；真实五分支 asset 未核验前不能宣称完整部署通过。

## 冻结资产与对照

优先复用 #242 的既有 PAI/CPFS 资产，待用户确认环境后现场只读核验实际目录、分片、config、checkpoint hash、依赖与硬件。历史报告是定位线索：不能由作业曾经成功推断当前路径可用，也不能把 #242 的 cache-only 性能当本功能收益。

历史候选资产为 DMD generator step300、checkpoint SHA256 `7aa362bcce2971022c5193aa5ebf392aadaf9110cd123fffad213f2e394ec191`，condition t=0、reference t=0.999、source+默认六图、四步 `[1000,750,500,250]`。必须以现场 effective config/原始请求为准确认 shift、窗口、codec、分辨率、噪声及 schedule；禁止从本文补齐未核实配置。原报告：[#242 CACHE_ONLY_COMMITS.md](https://github.com/seedleap/sglang/blob/9dcfaab10f812c68e44c5989df370bd2b31fc91f/benchmark/zing_h3_causal/CACHE_ONLY_COMMITS.md)。

resident 与 table 使用同一 candidate 源码、有效权重、FA4/compiled AdaLN、KV pool、窗口、输入/噪声/codec，只改变表模式。resident AdaLN LRU 按当前默认开启。SP1、SP2 分别比较；当前 main 的 denoise graph 支持边界原样保留，SP2 eager 与 SP1 graph 不能混成同一因果对照。

表预算按所有完整批次而非单 timestep 核算：`sum(M_batch) * (layers * 6 * R + 2) * H * 2` 字节，另记录实际 storage。H=5376、E=2688、L=50、R=5 时，投影权重含 bias 结构估算约 40.44 GiB，每 unique-t 行紧凑输出约 15.40 MiB。覆盖所有 prefix 子集的四步示例可能有 15 行，约 231 MiB，故默认完整表预算为 512 MiB。这是结构估算，不是显存实测。

## 最小执行矩阵与停止条件

| 阶段 | 最小工作 | 通过条件 / 停止条件 |
| --- | --- | --- |
| CPU 预检 | 复用 serving export，核验全部资产/索引、请求、源码 bundle、镜像 digest；渲染唯一 run 目录与 JobSpec | 任一 hash/provenance/资源配置缺失即停止，不申请补救 GPU |
| GPU builder | 单 rank TP1/SP1，在目标 worker 同软件/硬件算术条件下逐层生成表 | 每批保留原形状；compact finite、checksum 完整；记录同步构建时间及 GPU peak |
| 启动与数值 | SP1、SP2 独立进程，每个 SP 一次 resident→table 严格对照 | 参数清单/storage 为零、实际 reader 不读 projection、无 miss/fallback；compact/预测/每步 latent/KV/终态严格相等；不相等立即停止计时 |
| 生命周期 | 少量 prompt 更新、缺计划、audio 拒绝、异常/close、并行双 session、跨 stream、graph capture/replay | KV/版本不串、引用最终归零、没有额外热路径 collective；保留失败证据，不自动改配置 |
| 时间 | 正确性通过后，同 SP 成对交替 A/B，完整预热后无 profiler 测量冷/热请求与 chunk p50/p95 | 记录全部原始样本、配对顺序与置信区间；超过稳定性门槛保留并报告，不挑最快样本 |

若同 SP 的 resident 因容量 OOM，不自动降尺寸/关模块/改变 offload；只能报告容量边界。候选能运行不等于该 SP 数值对照通过，也不能与另一 SP 的 baseline 直接计算速度收益。历史 SP1 resident OOM 是特别需要现场确认的风险。

成本限制：先完成 CPU 与单卡 builder；数值/启动门槛通过后才做最小 SP1/SP2 请求，最后才进入性能重复。当前没有获批 GPU 型号、GPU 数量、时长上限或 JobSpec；不能声称 READY_TO_SUBMIT。提交前按 AGENTS.md 要求以 `gpt-6-astra` 只读审核具体脚本、patch 和 JobSpec，必须取得 SAFE_TO_SUBMIT，并获得用户明确提交授权；PAI 监控交给 `gpt-6-luna`。

## 必须保存的证据

- source/bundle、镜像、checkpoint、export config/shards、请求/条件/噪声、表文件与 manifest 的 hashes。
- GPU UUID/实测 MiB/compute capability、driver、PyTorch/CUDA/NCCL/FA4、matmul 与 deterministic flags，每 rank 原始日志、退出码及完成标记。
- builder 同步时间与 peak；worker load/startup/steady allocated/reserved；实际 table storage 字节、projection 参数与 loader 读取审计。reserved 不降不能解释为权重仍驻留。
- strict parity 的每步 receipt 与比较结果；运行成功、frame/媒体通过、数值通过、性能通过分别列出。
- 冷/热 E2E 与 chunk 延迟样本；profile 单列，不作为正式计时；统计、打包、媒体导出在 CPU/DSW 完成。

## 本地结果记录

- 完整 Zing/export/table/采样/SP CPU/graph 回归：**320 passed，2 CUDA skipped**，25.24 秒。
- 最后 builder 回执/逐层参数释放等受影响合同重跑：**35 passed，72 deselected**，6.57 秒。
- 独立 CI 固定依赖环境 exporter+table：**63 passed**（38+25）。
- 共享 H3 cache 在显式 CPU pageable fallback 边界下：**22 passed**，5.64 秒。Mac 的 torch 2.8 pinned allocation 返回 MPS tensor，原测试触发既有平台错误；此项没有验证 CUDA pinned 行为。
- 适用 pre-commit hooks 通过。全仓 CI 注册 hook 被未修改的 `test/registered/unit/realtime/test_zing_realtime_contracts.py` 缺注册阻塞；新增 table 文件单独通过同一注册校验，其余 hooks 在明确跳过该基线问题后通过。

上述集合有重叠，不累计为独立测试数。CPU graph 使用记录后回放 aten 的测试后端，不代表 CUDA Graph 验收。真实 GPU 数值、显存与性能仍待执行。

原始日志、harness、退出码与 patch/hash 仅本机保存于被忽略的 `outputs/issue-257-r0-cpu/`，不随 push 上传；完整接续步骤与复现入口见 [ISSUE_257_HANDOFF.md](ISSUE_257_HANDOFF.md)。
