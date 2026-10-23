# H3 Qwen 编码器独立优化开关

## 目标与用法

将此前临时 A/B adapter 的五项优化，以及 T030 测过的原生 online FP8 接入 H3 编码器，共六个独立开关。算子选项在实例创建时读取，FP8 在组件加载配置阶段读取，修改后重启 worker。按用户 2026-10-23 的要求，MiniMax H3 的六项优化默认全部开启；环境变量未设置等价于 `1`，独立设置 `0` 关闭对应项。通用 Qwen3-VL 的默认 options 保持不变。

接入范围是正式组件 loader 和 encoder 执行路径。本改动不更新部署镜像、线上启动配置或运行中的 worker；合入代码也不等于线上已经启用。启用前需在目标 GPU 上完成当前实现的组合数值和完整推理验证。

| 环境变量 | 未设置 | `0` | `1` |
| --- | --- | --- | --- |
| `SGLANG_H3_QWEN_HF_RMSNORM` | 开启 | 原生 RMSNorm | HF 舍入顺序 fused RMSNorm |
| `SGLANG_H3_QWEN_FUSED_MROPE` | 开启 | 原生 MRoPE | SRT fused interleaved MRoPE |
| `SGLANG_H3_QWEN_FUSED_GATE_UP` | 开启 | 分开投影 | 合并 gate/up + rounded SwiGLU |
| `SGLANG_H3_QWEN_TEXT_FA4` | 强制 FA4 | 强制 SDPA | 强制 FA4 |
| `SGLANG_H3_QWEN_VISION_FA4` | 强制 FA4 | 强制 SDPA | 强制 FA4 |
| `SGLANG_H3_QWEN_FP8` | 原生 online FP8 W8A8 | 关闭在线 FP8；与其他 FP8 配置或 FP8 checkpoint 冲突时报错 | 原生 online FP8 W8A8 |

只接受 `0` / `1`，不提供优先级不清晰的总开关。六项可组合，受下述硬件/TP/量化格式限制。未量化 BF16 checkpoint 的全关基线应显式设置六项为 `0`，并取消已有 CLI online FP8 配置。下列显式全开设置与六项都未设置等效；需原生 online FP8 支持、FA4 可用及编码器 TP=1：

```bash
export SGLANG_H3_QWEN_HF_RMSNORM=1
export SGLANG_H3_QWEN_FUSED_MROPE=1
export SGLANG_H3_QWEN_FUSED_GATE_UP=1
export SGLANG_H3_QWEN_TEXT_FA4=1
export SGLANG_H3_QWEN_VISION_FA4=1
export SGLANG_H3_QWEN_FP8=1
# 在原有启动命令中使用 --encoder-parallel replicate（编码器 TP=1）。
```

FP8 未设置或 `1` 等价于 H3 的 `component_quantizations={"text_encoder":"fp8"}`，沿用原 loader 的权重转换、精度/硬件检查和后处理；没有额外 weight-only 转换。与其他显式量化方法冲突时拒绝；使用其他量化格式须先设 `FP8=0`，并遵守 gate/up 支持范围。默认 online FP8 要求未量化 checkpoint，已有 checkpoint 量化仍按 loader 冲突规则拒绝，不重复量化。`0` 与显式 CLI `fp8` 或 FP8 checkpoint 冲突也拒绝，避免优先级含糊。该开关不负责把已序列化的量化 checkpoint 还原成 BF16，也不覆盖其他量化格式。

H3 始终要求保留原生加载路径，包括默认全开和显式全关；加载失败报错，不静默换成 Transformers。默认或显式开启 FP8 均拒绝 weight-only Marlin 路径，确保运行的是 T030 所测的 W8A8 类别。不修改 DiT、通用 Qwen encoder 或其他实例的全局 backend。

Attention 开关参与现有组件 backend 登记和校验：FA4 登记为 `FA` 家族，显式视觉 SDPA 登记为 `TORCH_SDPA`。若同时指定组件 backend，两端实际选择必须与其相容，否则加载失败；例如组件指定 SDPA 时不能另开视觉 FA4。语言/视觉混合消融请省略组件级显式 backend，让两个开关分别决定并登记实际选择。

**远程 conditioning 限制**：默认全开仅支持本地 H3 编码路径。远程 native RPC 的 `build-profile` 和服务启动要求六项变量都显式设为 `0`，同时 profile 的 backend 必须为 `torch_sdpa`；任一项未设置（默认开启）或为 `1` 均在 CUDA/权重加载前报 `CONDITION_PROFILE_MISMATCH`。服务端与 profile 构建环境都须使用这个全关基线，加载阶段继续核对实际 backend 和 BF16 回执。此前“远程取消六项变量”的说明不再适用。

现有远程 profile/receipt 未冻结这些 options，且只支持 BF16、单一 backend。沿用 reviewer 提出的临时准入限制，避免默认全开后不同数值路径共用身份；相比本次扩展远程量化与混合 attention 协议，要求明确全关 SDPA 的范围更小。`native.py` 已属于代码身份哈希，升级后须重建并分发 profile。后续支持远程优化时，仍需冻结实际 options、attention、量化与代码身份，并验证变更后的身份和回执；本次没有完成该协议扩展。

## 边界与关键决策

- H3 入口拥有配置解析；不可变 options 经构造参数传入语言与视觉模块。算子 helper 不读取环境变量。
- gate/up 支持编码器 TP=1、SiLU，以及未量化或原生 online dynamic FP8（无 block quant、无 ignored layers）；其他配置明确拒绝。与此前实测 `--encoder-parallel replicate` 对齐。权重在加载时按 HF gate/up 名称写入一个 packed 参数，保留完整性检查，不长期保留两份权重，不在 forward 拼接。FP8 仍由原 loader 对这个参数执行一次后处理。
- HF RMSNorm 保留权重名称和 HF cast-before-multiply 顺序；fused MRoPE 使用独立输入副本，保护原生路径不受原地写入影响。每个语言编码器持有一个 `QwenVLFusedMRoPE`，供其各层共享；首用时将原生 cache 转为目标 device/dtype，后续同配置直接复用。变化时从原生 cache 重建，避免从已舍入的副本再转精度。独立的非持久 buffer 随模型迁移，不写入 checkpoint，也不修改 `get_rope` 共享的原生缓存。相比原实现增加一份目标 dtype 的常驻缓存，大小为原生 cache 元素数乘目标元素字节数，换取避免每层全表转换；峰值显存与 GPU 耗时仍须实测。
- FA4 直接复用仓库已有 kernel wrapper，固定 `num_splits=1`，语言 causal / GQA，视觉 non-causal / packed varlen；包含视觉 head_dim=72。显式启用时依赖或硬件不支持必须报错，不静默换 kernel。
- 保留第 50 层未归一化特征、全部 token、DeepStack 注入顺序与 H3 输入输出契约。不新增 QKV 合并或 prefix cache。
- 社区 SRT 的 MLP 使用合并 gate/up 和 SiLU+multiply，attention 与缓存元数据耦合；此处只复用底层算子，不把 H3 特征提取器替换成聊天服务。已核实社区固定版本 [Qwen2MLP](https://github.com/sgl-project/sglang/blob/79eb10e182c62af6564e5d8aa2b62e19810b6cc2/python/sglang/srt/models/qwen2.py#L74) 与 Qwen3 的复用关系。上游审核重点：默认行为、checkpoint 兼容、配置归属、原地写入、量化/TP 边界，以及实际 GPU 数值证据。

## 验证与证据

检查覆盖：64 种构造组合（含真实 Fp8Config/Fp8LinearMethod，但不执行 GPU GEMM）、非法值与量化冲突、加载 gate/up 顺序与缺失半片、默认权重与前向兼容、实例隔离、FA4 dense/varlen 元数据、MRoPE 原地写入和 cache 保护，以及已有 Qwen text/vision/rope 和 loader 单测。FA4 调用和 fused MRoPE 所有权检查使用 CPU stand-in，不能当作 CUDA 数值测试。

此前临时 adapter 在源码 `82fd85da3eeb8d586c85c1a8c75f870b59f6035b`、L20D SM103、832×480、3 个真实输入上的五项全开实验（不含 FP8）：Qwen CUDA 阶段约减少 32%，但更新 prompt 后条件特征相对 L2 差异约 10.2%–62.0%，画面有变化。不能称数值等价。此次默认开启是用户指定的策略调整，并不补足当前组合的 GPU 验收。

该历史实验不能代替正式实现验证。现已补测 `77f40e18c1` 六项全开与全关的 480p 三例完整推理，结果见下节；736 和人眼画质仍未验收，不外推该档或其他硬件收益。

T030 的 `qwen-encoder-only-20261023-r4` 使用原生 online FP8，实际量化 350 个语言线性层，视觉未量化；三个更新 prompt 的相对 L2 为 5.31% / 6.09% / 28.09%。本次复用相同配置入口。与 gate/up 合并一起开时，语言线性模块由 350 减为 300，合并权重的量化尺度可能改变；此组合不能直接引用 FP8 单开误差作为实测结论。

### 本次实际验证（2026-10-23）

- 默认全开调整后：**206 passed、2 CUDA skipped、4 T5 deselected**，60.03 秒；覆盖未设置时 H3 五项实例选项全开、默认 FP8 进入原生 loader、通用 Qwen 默认不变、默认/显式 FP8 拒绝 Marlin、全关原生输出/权重兼容，以及远程任一项未设置/开启均拒绝、全关必须匹配 SDPA profile。其余环境、测试入口和 CPU 限制同下文；没有新增 GPU 实测。
- 历史提交 `e2b848d` 在修复 backend 登记和 fused cache 复用后回归：155 项 CPU 测试通过，2 项 CUDA 测试跳过，4 项 T5 restoration 测试排除。最新 [PR #404](https://github.com/seedleap/sglang/pull/404) 以本节上方默认全开回归为准。
- 覆盖 `test_h3_qwen_execution_options.py`、`test_qwen3vl_text.py`、`test_qwen3vl_vision.py`、`test_qwen_vl_rope.py`、`test_text_encoder_loader.py`、`test_minimax_h3_encoder_device.py`。64 种组合包含真实 FP8 线性模块构造；使用有限、已初始化的 CPU 小模型权重检查全关语言输出、视觉 pooler 和 DeepStack 逐位一致。
- 额外覆盖四种 attention 开关组合与三种组件请求（无显式请求、FA、SDPA）的真实登记/冲突校验；CPU kernel stand-in 验证连续 50 次 fused RoPE 调用仅一次缓存转换、dtype 变化从原生缓存重建、同编码器跨层共享、不同编码器实例隔离、异常时保护输入及原生结果保持不变。不据此声称 GPU kernel 数值或性能通过。
- T5 restoration 的扩大回归曾因 world group 未初始化而失败，未修改该无关路径；随后用 `-k 'not test_native_encoder_restoration_checks_checkpoint_before_fallback'` 排除四项，不能称完整 loader 套件通过。image loader 扩大回归受缺少 `compressed_tensors` 依赖限制，未执行。
- 环境：WSL Python 3.11 / torch 2.8.0+cpu / Transformers 5.19.0；`SGLANG_LIGHTWEIGHT_RUNTIME=1`、`SGLANG_DIFFUSION_PLATFORM_OVERRIDE=cpu`。外部测试入口仅令 RoPE 构造识别 CPU，避免导入缺失的 vLLM，前向仍调用真实 `forward_native`。未加载 SGL CPU serving extension，不验证该 extension。
- Ruff 的 F401/F821/UP037/import 检查、格式检查和 `git diff --check` 通过。PR 描述提供测试入口、CPU 环境适配和对应提交信息。
- 当前六项组合 GPU 实测（2026-10-23，`77f40e18c1`）：两卡 L20D CC10.3，SP2/Qwen TP1，832×480 三例360帧；两组仅六项开关0/1不同，无encoder overlap、六分支联合、Qwen结果/前缀缓存。W6/S3、gap12、Graph关、前三块4步后续`[1000,750]`，每例正式3次及独立诊断1次。
- Prompt响应中位数：2.052→1.916、2.260→2.072、2.271→2.110秒；Qwen切换编码CUDA区间（每例每组单次，取较慢rank）：419.5→236.0、510.3→280.9、512.8→285.0ms。每次诊断命中200次HF RMSNorm、50次MRoPE、50次gate/up、50次语言FA4、54次视觉FA4、300次原生FP8，无Marlin。
- 切换特征相对L2为30.93%/10.35%/67.68%，同次GPU分配的原始RGB相对L2为10.25%/25.60%/10.67%；两rank特征一致、实际输入哈希一致，但不属于数值等价优化。原版前两轮正式样本复用前一作业，第三轮及全部诊断/全开数据来自新作业；原版跨作业RGB不同，视频误差不混用跨作业输出。每组新作业正式与诊断RGB逐位一致。
- 共享进程allocated峰值最大159.61→136.95GiB/卡；不能分离MRoPE cache或Qwen独占显存。GPU及CPU导出作业成功，18正式和6诊断样本齐全；详细结果、计时边界与两次harness失败恢复见[PR #404](https://github.com/seedleap/sglang/pull/404)。736、人眼质量、DiT/Decoder分项CUDA与组件独占显存仍未测，未部署线上。
- Review 跟进验证：原六文件回归，加 `conditioning/test_service_resources.py`、`test_zing_conditioning_contract.py`、`test_zing_conditioning_worker.py`，合计 **200 passed、2 CUDA skipped、4 T5 deselected**，65.31 秒。该历史修复按当时默认关闭策略测试显式开关拒绝；本轮改为仅允许六项显式全关且 SDPA 的远程基线，未设置/开启的任一项均拒绝，须重新回归。冻结旧 `e2b848d` 的 profile builder 做 CPU 最小复现，证实 RMSNorm=0/1 的 profile_id 相同；不把 CPU 身份/准入测试视为远程 GPU 数值验证。部署决策入口同步历史三例 prompt 响应及未验证范围。

## 阅读指引与回退

从 `MiniMaxH3Qwen3VLEncoder.__init__ / load_weights` 查看配置及 checkpoint 入口；沿 `Qwen3VLModel` → text / vision 构造参数查看实例边界；算子适配位于 `qwen3vl_ops.py`，共享 RoPE 适配位于 `qwen_vl_rope.py`。

回退：使用 BF16 checkpoint、取消 CLI FP8 量化并将六项都显式设为 `0`，重启后使用原生算子/SDPA 基线。取消变量会重新开启优化，不能作为回退。FP8 只转换进程内权重，不修改源 checkpoint；无额外 KV 池需求。
