# Issue #257 HANDOFF：离线 AdaLN 表与 weights-free worker

> **Update 2026-10-10:** the SP1 real-checkpoint GPU comparison has since been run (exact parity, about 41 GiB less memory); results and current scope are in [ADALN_TABLE.md](ADALN_TABLE.md#status-and-support-scope). The "not yet run" statements below describe the state at handoff.

交接日期：2026-10-09。Issue：[seedleap/sglang#257](https://github.com/seedleap/sglang/issues/257)。

## 当前断点

首版实现已完成本地 CPU 验证，本次将实现、测试和交接文档一起提交并推送。真实 checkpoint 的 CUDA 数值、SP1/SP2、显存和端到端性能验收尚未执行；没有提交 #257 GPU/云作业，没有新建主实验 PR。

- 本地与远端目标分支：`codex/zing-adaln-persistent-table`。
- Worktree：`/Users/yiwencai/.codex/worktrees/issue-257-adaln-table/sglang`。
- 已验证的代码基线：`f43a7f97e0a68a09dcb6956f08b4e275b5120df8`，包含已合入的 #264 split/five-branch foundation 与 #258 denoise graph。
- 2026-10-09 推送前 fetch 读到 main 为 `c5764eb01cb629088bd9a43f161d032c7c6279f8`，新增 #176。当前分支尚未集成 #176；不能把其 Encoder overlap 能力算入本分支验证。
- Candidate 以本分支实际 Git HEAD 为准。接续时先 `git fetch --prune origin`，核对 HEAD、工作区及远端；部署前生成该 HEAD 的 bundle/patch 并记录 SHA256，不能用基线 SHA 代替 candidate。
- 旧 `codex/zing-split-adaln-foundation` 与 `codex/issue-257-adaln-table` 分支保留；主工作目录 `dev/yiwen-cai` 的已有修改没有纳入本次提交。

## 已实现内容与代码入口

| 入口 | 最终行为 |
| --- | --- |
| [build_adaln_table.py](build_adaln_table.py) | 从规范 serving export 逐层流式加载真实投影，按原完整 unique-t GEMM 形状离线生成 BF16 紧凑值；显式 generator/EMA、native/component/merged-LoRA provenance；记录同步构建时间及 CUDA 构建显存回执 |
| [zing_h3_adaln_table.py](../../python/sglang/multimodal_gen/runtime/models/dits/zing_h3_adaln_table.py) | `zing-adaln-table-v1` 工件、checksum、checkpoint/TimeMLP/频率/tag/完整批次/精确 timestep 与 sigma 网格/软件与设备算术身份；完整表预算、不可变共享、lease、跨 stream event fence |
| [transformer_loader.py](../../python/sglang/multimodal_gen/runtime/loader/component_loaders/transformer_loader.py) | 读取 tensor 前过滤主层与 final AdaLN 参数，先校验所有 config/shard hash，再初始化每 rank 的小表并输出启动回执 |
| [minimax_h3.py](../../python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py)、[zing_h3.py](../../python/sglang/multimodal_gen/runtime/models/dits/zing_h3.py) | meta 构造阶段就不创建主层/final 投影；统一 eager/graph provider；TimeMLP/refiner 与其余权重保留；在线权重/LoRA 更新在写入前拒绝 |
| [sampling.py](../../python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/zing_h3/sampling.py) | 在 KV 申请或 prefix 写入前，枚举实际 prefix/source/active/final、尾块及 CFG 的完整计划；异常、prompt 更新、close 释放 lease；未知计划/audio 拒绝，无 miss fallback |
| [validate_real_checkpoint.py](validate_real_checkpoint.py) | `sglang` 子命令增加 `--adaln-table` 和预算参数，沿用现有真实 checkpoint 请求、结果与 metrics 路径 |
| [.github/workflows/zing-adaln-cpu.yml](../../.github/workflows/zing-adaln-cpu.yml) | 最小 CPU 依赖执行 exporter 与持久表完整性/接入合同测试 |

部署参数、schema、builder 用法见 [ADALN_TABLE.md](ADALN_TABLE.md)。首版限定固定有效权重、video、未量化、TP1/SP1 或 SP2、无 DiT FSDP/CPU 或逐层 offload。SP2 沿用基线 eager，SP1 可进入原有 graph 路径。默认完整表预算 512 MiB；每个完整批次都保留独立形状，没有在线构建、补表或 eviction。

## 验证结果与证据边界

| 验证 | 结果 | 范围 |
| --- | --- | --- |
| Zing/export/table/采样/SP CPU/graph 回归 | **320 passed，2 skipped** | 25.24 秒；两项需要 CUDA；最后仅 builder 回执/逐层释放等合同补充修改，随后重跑下行受影响测试 |
| 最后受影响合同重跑 | **35 passed，72 deselected** | 6.57 秒；工件、完整批次、loader、真实流式 builder、每层参数 weakref 释放、weights-free 模型与 graph session 合同 |
| 独立 CI 依赖环境 | **63 passed** | exporter 38 + table 25；Python 3.11.12、torch 2.8 CPU、pytest 8.4.2、pyyaml 6.0.3、safetensors 0.8.0、numpy 2.2.6 |
| 共享 H3 cache 的 CPU pageable fallback | **22 passed** | 5.64 秒；仅将 CUDA pinned 分配入口设为不可用，验证生产 pageable fallback；没有验证 CUDA pinned 行为 |
| 格式与静态检查 | 适用 hooks 通过；一个基线全局 hook 单列 | 完整 pre-commit 曾被未修改的 `test/registered/unit/realtime/test_zing_realtime_contracts.py` 缺 CI 注册阻塞；新增 table 测试单独注册校验通过；跳过该全局 hook 后其余适用检查通过 |

这些集合有重叠，不能相加为独立测试总数。CPU 使用合成小模型和六张合成 reference，覆盖四支/五支、condition 共享/独立、合法 tag 重排、Euler/few-step、CFG、尾块、prompt 更新、双 session capture/replay、参数与实际 reader 审计。CPU graph 为记录后回放真实 aten 算子的测试后端，不是 CUDA Graph 验收。

Mac 的 torch 2.8 `pin_memory=True` 返回 MPS tensor，既有 H3 host cache 测试触发 `as_strided_tensorimpl does not work with MPS`；没有为此修改共享生产缓存。GPU 表与 CPU 表有不同算术身份，CPU 构建表不能用于 CUDA worker。

原始日志、Mac CPU bootstrap、pageable 边界 harness、带退出码的 receipt 及源码 patch/hash 本机保存在 `benchmark/zing_h3_causal/outputs/issue-257-r0-cpu/`。该目录已被仓库忽略，**不随 Git push 上传**；本文件的结果记录与复现入口随分支保存。文件名：

- `zing-257-cpu-validation-final.log`
- `zing-257-final-contracts.log`
- `zing-257-ci-contracts-final.log`
- `zing-257-h3-pageable-cpu-validation.log`
- `zing-257-pre-commit.log`、`zing-257-pre-commit-final.log`
- `receipt.json`、`candidate.patch`、`candidate.patch.sha256`

## 本地复现入口

最小 CI 合同可以使用 workflow 中固定版本的干净 CPU 环境复现：

```bash
SGLANG_LIGHTWEIGHT_RUNTIME=1 PYTHONPATH=python python -m pytest -q \
  test/registered/unit/diffusion/test_zing_adaln_export.py \
  test/registered/unit/diffusion/test_zing_adaln_table.py
```

本机全模型 CPU 环境位于 `/tmp/zing-foundation-venv`，使用保存到 outputs 的 `zing-foundation-cpu-tests.py`。它启用仓库 Apple import compatibility stubs、固定 CPU platform 和单线程，并先导入 pipeline configs 避免旧 conftest 的循环导入；CUDA 计算没有在 Mac 执行。

```bash
PYTHONPATH=python /tmp/zing-foundation-venv/bin/python \
  benchmark/zing_h3_causal/outputs/issue-257-r0-cpu/zing-foundation-cpu-tests.py -q \
  test/registered/unit/diffusion/test_zing_adaln_export.py \
  test/registered/unit/diffusion/test_zing_adaln_table.py \
  python/sglang/multimodal_gen/test/unit/test_zing_h3_model.py \
  python/sglang/multimodal_gen/test/unit/test_zing_h3_registration.py \
  python/sglang/multimodal_gen/test/unit/test_zing_h3_sp_cpu.py \
  python/sglang/multimodal_gen/test/unit/test_zing_h3_denoise_graph.py \
  python/sglang/multimodal_gen/test/unit/test_zing_h3_chunks.py \
  python/sglang/multimodal_gen/test/unit/test_zing_h3_adaln_projection.py
```

`/tmp` 环境不是持久依赖。其他机器重建环境时以 workflow 与项目依赖为准；Mac harness 不作为 GPU 执行脚本。

## 主实验接续顺序

具体可证伪问题、对照、停止条件与证据清单见 [ISSUE_257_EXPERIMENT_PLAN.md](ISSUE_257_EXPERIMENT_PLAN.md)。

1. 确认使用 #242 的 PAI/CPFS 资产还是实验室 H100；此前环境选择问题尚未收到回复。只读核验 serving export/config/shards、真实 source+六图、噪声、codec、四步调度、依赖与实测硬件。历史 #242 资产/作业只作定位线索。
2. 检查并决定是否集成最新 main 的 #176；集成后对受影响路径重跑验证并重新冻结 candidate。不要以旧 CPU 结果证明新集成代码通过。
3. 补齐真实 GPU 对照脚本与唯一 run manifest：当前 harness 已能选择表模式，但不等于已完成本实验的 compact/每步预测与 latent/完整 KV 严格 dump 和审计。分别冻结 resident 与 table 的同 SP、同请求、同源码条件；默认 compare 容差不能代替严格相等门槛。
4. 先单 rank GPU builder，再 SP1/SP2 各自同 SP 的 startup/storage/reader、逐步严格相等与 finite、生命周期/stream/graph 门槛。若 resident 同 SP OOM，记录容量边界，不与其他 SP 的 baseline 计算收益。
5. 通过正确性后再测 allocated/reserved、冷/热 E2E 和 chunk p50/p95，采用成对交替 A/B；构建成本单列，原始样本与退出码保存，CPU 完成汇总。

目前未获批具体 GPU 型号、数量、时长上限或 JobSpec，不是 READY_TO_SUBMIT。GPU/云作业提交仍需用户明确授权，并依项目规则由干净上下文 `gpt-6-astra` 子 agent 对实际脚本、补丁和 JSON 只读审核，取得 SAFE_TO_SUBMIT；阻塞修复由同一审核者复审。PAI 监控交给 `gpt-6-luna`，禁止递归派生。

## 接续约束

- 当前根会话 `01a11f8a-9589-74e1-bb45-7cd0ddefce19` 的 Dev Mode 为 disabled；不自动开启。接续时按 AGENTS 恢复检查当前会话状态。
- 项目全部 7 个 Infra Skills 固定来源 `5e5879e4480a9f76be02e68b3d308b57f05bdf51`；本轮 33 个文件 Git blob hash 一致，安装目录仅本地 exclude，不提交。
- 用户本次授权生成 HANDOFF 和推送本地分支。该授权不包含 GPU 申请、云提交、报告发布、主实验 PR 创建或合并。
- [ISSUE_257_ANALYSIS.md](ISSUE_257_ANALYSIS.md) 是 b1899c0 上的初始历史分析；其“未实现”缺口不是当前状态。
