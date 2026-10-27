# Qwen remote 六项优化的执行契约

## 目标与使用

原 remote 服务只允许六项显式全关，因为 profile 不能区分优化后的数值路径。本 PR 把六项布尔值纳入已有 `numerical_environment`，使独立 TP1 Qwen 服务能够使用本地 H3 的同一组选项。默认值仍为全开，通用 Qwen 不变。

构建 profile 与启动服务使用相同的 `SGLANG_H3_QWEN_` 环境变量：`HF_RMSNORM`、`FUSED_MROPE`、`FUSED_GATE_UP`、`TEXT_FA4`、`VISION_FA4`、`FP8`。每项只接受 `0`/`1`；未设置等价 `1`。以下命令在服务实际 GPU/软件环境中运行；`ENV.json` 的 gpu、cc、software 必须来自实际部署环境，不能照抄其他机器。

```bash
python -m sglang.multimodal_gen.runtime.conditioning.cli build-profile \
  --model MODEL --numerical-environment ENV.json --output PROFILE.json
python -m sglang.multimodal_gen.runtime.conditioning.cli serve \
  --config SERVICE.json --profile PROFILE.json
```

服务配置、worker 接入和凭据沿用 [conditioning 使用说明](../specs/ISSUE_256_CONDITIONING_USAGE.md)。ENV 保留 `encoder_tp=1`、`gpu`、`cc`、`precision=bf16`、`software`；`backend` 按下表填写。`execution_options` 可省略，由 builder 根据环境填入；显式提供时必须包含六个布尔字段且与环境一致。`precision=bf16` 描述模型计算/输出精度，FP8 权重由 `execution_options.fp8` 单独记录。

|文本 FA4|视觉 FA4|backend|
|---|---|---|
|0|0|`torch_sdpa`|
|1|0 或相反|`fa4+torch_sdpa`|
|1|1|`fa4`|

每次更改任一开关都须重新构建并分发 profile，服务和 worker 使用同一份文件。加载前环境与 profile 不一致报 `CONDITION_PROFILE_MISMATCH`，在 CUDA/权重加载前拒绝。旧 profile 缺少选项只代表全关；本次代码身份已改变，即使全关也须重建，旧 profile 不会被静默升级。

## 边界与决策

真实调用链为 CLI build-profile → `build_profile` → profile ID；serve → `NativeConditionEncoder` → encoder-only pipeline → H3 loader → `_measure_execution` → `validate_runtime_receipt`。服务加载时校验实际数值回执，worker 握手校验 profile ID 与 result transport；现有握手不传递完整数值回执。缓存/输入 digest 已包含 profile ID，因此不同执行选项不能共用身份。请求编码、CPU/NCCL 结果传输、SP 广播和会话调度均沿用 main；没有新增缓存、异步队列或前缀复用。

- 配置解析归现有 H3 config 模块，profile 与模型构造复用它，避免默认值漂移。直接删除 remote 禁令会让不同数值路径共用身份，不能满足契约。
- profile、数值环境和模型回执归 `conditioning/native.py`。启动后检查真实文本/视觉 attention、每层 RMSNorm、fused RoPE 和 gate/up，及 resident FP8 权重；允许原生 FP32 scale，拒绝其他非 BF16 浮点参数。FP8 是否实际存在由权重读取，不凭环境变量推断。
- 通用 ServerArgs attention fallback 使用 SDPA；H3 逐模块选项选择文本/视觉 FA4。混合配置分别验证两部分，不能用整个模型只有一种 backend 的旧约束。
- 所有新增检查仅在构建/加载时运行；不进入逐请求或逐层 forward 热路径。回执证明加载结构与精度，不能代替 profiler 对实际 kernel 执行的证明。
- 旧实验曾兼容 `fused_mrope` 布尔字段；当前 main 使用 `fused_rope` 对象，本 PR 只验证当前实现，不保留实验版本兼容分支。GPU UUID/NVML 修复是独立平台问题，不混入本 PR。

## 验证与证据边界

需要保持：六项组合身份互异；保存后环境变化拒绝；混合后端合法；实际模块与声明不一致拒绝；BF16/FP8+FP32 scale 可观测；异常 dtype 拒绝；local 和现有传输契约不退化。

CPU 回归覆盖 64 种 profile 身份、32 种真实小模型算子构造与回执、模拟 resident FP8 参数/scale 的精度检查、开关不一致与非法 profile，以及既有 conditioning/本地编码测试。FP8 参数检查不运行 GPU 量化加载或内核。2026-10-27 在基线 `290d08c797` 加本 PR 工作区运行 conditioning 全目录、H3 execution options、Qwen3VL text/vision：226 passed、0 failed、0 skipped（2 warnings）；isort、Ruff format/lint 和 diff --check 通过。命令为 `pytest python/sglang/multimodal_gen/test/unit/conditioning python/sglang/multimodal_gen/test/unit/test_h3_qwen_execution_options.py python/sglang/multimodal_gen/test/unit/test_qwen3vl_text.py python/sglang/multimodal_gen/test/unit/test_qwen3vl_vision.py -q`。本地 WSL CPU-only runtime 使用仓库已有 lightweight/CPU 平台开关，测试 runner 令 RoPE 使用 forward_native，并修正 WSL Git 路径与 multiprocessing main guard；首轮 runner 环境失败后重跑全部通过。日志 `D:/qwen-remote-contract-final-tests.log`，不将该适配当作标准 GPU runtime 证据。

历史证据：2026-10-24 r5 在同一次四卡 L20X/SM90 分配测试本地接入补丁，三例 832×480、360 帧，每组每例各 1 次预热/正式/诊断，BF16 与 FP8 全尺寸加载及推理成功。底座 `b4ad75d696138c8466f70fc8558c1c58d8472b91`，Qwen 覆盖 `f60867b6269457ba39ab30db8ce165ec71c85855` 并叠加本地文件；它同时改变六分支联合路径，性能不能归因为本 PR 单独收益。详情和原始入口见 [历史记录](https://icnimsatr0zz.feishu.cn/docx/NBKJdyRZpoTb1cxmA99cRxZDnhs)。

当前整理版本未新增 GPU 测试，480/736 同条件整体 benchmark、Encoder（Qwen/参考 VAE/source VAE）/DiT/Decoder 的 P50/P95、样本量及常驻/峰值显存均待补。历史 480 不能外推到当前 head 或 736。质量有可见差异，未建立画质等价；未来按 [统一性能基线](../specs/PERFORMANCE_DEFAULTS.md) 分档复测。PR 不承诺加速比例或上线验收。

## 社区实践与 review 导读

基线为 seedleap/sglang `290d08c7975ffdb0050dab1cb43aa149239c17c5`。已核对该版本 diffusion contributing、邻近 H3 config 和 conditioning 契约，复用 `DeploymentProfile`、每实例 execution options 和启动回执机制，无新依赖。这是派生仓库 Zing 扩展，未把私有协议描述成官方上游共识；贡献指南要求实际测试、避免多余异常包装，GPU 验证缺口如上。

建议阅读：本设计 → `configs/models/encoders/minimax_h3_qwen3vl.py::h3_qwen_execution_options` → `runtime/conditioning/native.py::build_profile/NativeConditionEncoder/_measure_execution` → `test/unit/conditioning/test_service_resources.py` 与 `test/unit/test_h3_qwen_execution_options.py`（代码均在 `python/sglang/multimodal_gen/`）。重点核实：身份是否覆盖每一项、回执是否独立观察权重/算子、混合后端的 SDPA fallback 是否正确、升级是否要求重建 profile、现有 CPU/NCCL 传输是否保持原契约。
