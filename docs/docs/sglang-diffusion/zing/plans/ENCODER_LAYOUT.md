# H3 progressive source Encoder 布局策略（#349）

## 使用与边界

布局优化默认开启；设置 `MINIMAX_H3_PROGRESSIVE_CHANNELS_LAST=0` 可显式关闭。仅对 CUDA、BF16、performance、tiled H3 source progressive Encoder 生效。逻辑 shape 始终是 NCTHW；deterministic、CPU、reference whole-clip 和原始参数的 storage/stride 保持原行为。启用后，模型准备阶段创建共享、只读的卷积权重布局副本，调用方在 ready 前显式预热默认 tile 的 1/8 帧形状；这部分启动时间与额外显存必须单独报告。

## 为什么采用局部布局策略

初始原型只转换卷积权重，布局转换成本转移到了 norm/pad 的 copy，整体反而更慢。保留原 GroupNorm 归约顺序、直接生成 channels-last 输出后，native reflection padding 又成为主要回退。因此只为已测得的 padding 瓶颈加入索引搬运 Triton kernel；不重写卷积或归一化计算。

多帧 source tile 在入口转为 channels_last_3d。归一化仍把时间维独立并入 batch，按每个 `(N,T,group)`（N 为 batch 中的样本数，T 为时间帧，group 为 GroupNorm 的通道组）使用原始 `F.group_norm` 和激活，只改变输出布局。缓存初始零历史、cat、时间切片与 clone 保留相应布局。reflection kernel 接受 batch 间有间隔的时间切片，使用 64 位 batch 地址计算；有梯度或不支持的设备/dtype 使用 native 路径。

空间反射填充是在每帧的 H/W 边界外镜像取值，不扩展时间维，也不重复边界元素。例如一行 `a b c d` 左右各填充两格，得到 `c b | a b c d | c b`。本 PR 实际使用的 `reflect_ndhwc` 只搬运这些值，直接生成 channels-last 输出，避免 padding 前后的整块布局转换。

完全传播 channels-last 会改变部分 cuDNN BF16 计算计划，不能仅因权重和逻辑 shape 相同就宣称逐值一致。逐算子对比已定位第三个 down stage 的 5 个卷积存在相同输入下的舍入差异。因此该 stage 和 1×1 卷积继续使用原 NCDHW 计算路径，之后恢复 channels-last。quant_conv 和公开 moments 输出也恢复原布局。单帧 startup block 继续使用原路径，避免小输入转换成本增加首 latent 延迟。

`progressive_layout.py` 用策略对象集中表达这些选择。模型准备时建立原布局和多帧混合布局两个策略，为每个卷积绑定权重、输入转换、历史分配／转换和输出转换。会话构造时固定准入；forward 只按单帧／多帧选一次策略，conv 执行已绑定的操作，不再判断第三个 stage、1×1 或临时查找权重回退。策略由模型共享，历史状态仍由各 session 持有。

channels-last 通常有利于支持它的 cuDNN Conv3d，特别是低精度卷积；逐元素激活、残差加法可以保持该布局，减少算子之间的转换。但它不保证 GroupNorm、padding 或所有 shape 都更快。本实现保留原 GroupNorm 归约，单独优化 reflection 搬运，并保留已定位的卷积数值例外，不能把整个模型无条件转换。

布局切换时只创建当前 forward 的历史视图/副本，不原地修改旧 cache。成功后按既有 commit 协议替换状态；失败仍可重试。clip reset、EOS、取消与下一会话复用沿用现有生命周期。

## 预热触发

普通 diffusion 服务使用 `--warmup-mode server` 在启动时执行合成请求，`--warmup-mode request` 在首个真实请求前插入预热，`off` 关闭通用请求预热；`--enable-torch-compile` 在未指定模式时默认选择 server。入口是 `server_warmup.py` 与 `warmup_request_builder.py`。

当前 realtime adapter（包含 Zing）被 `should_run_synthetic_server_warmup` 排除，通用合成请求不覆盖 source 会话。本 PR 将权重准备与执行预热拆开：`prepare_visual_encoder_weights()` 只准备精度、权重副本和布局策略；`warmup_encoder()` 使用临时 replicated cache 预热默认 tile 的 1／8 帧，在 finally 中清理缓存，成功后同步 CUDA。

独立 Encoder worker 的顺序是 `load_encoder → warmup_encoder → 绑定控制端口 → startup rank 协商 → encoder_service_ready`。预热异常沿已有启动失败路径传播，各 rank 不宣告 ready；完整模型 stage 也显式调用同一方法，保留原有预热行为。这段组件预热不受通用 `--warmup-mode` 控制，也不保证覆盖所有运行时形状；它尚未接入 SGLang disaggregation role 的通用生命周期。

## 验证标准与当前限制

组件验收工具为 `benchmark/zing_h3_causal/benchmark_encoder_layout.py`，复现命令见 [benchmark 说明](../../../../../benchmark/zing_h3_causal/README.md#encoder-layout-组件对照)。它使用真实 trained Encoder、严格加载权重、实际 progressive session 与 tile assembly，支持单 GPU 和 SP2。每种布局独立预热，再做三次无 profiler 计时，最后单独 profile；输出严格逐值数值门槛、逐调用延迟、cache 字节数、峰值显存、准备时间和卷积 stride。失败数据保留，不放宽 BF16 门槛。

性能验收分别覆盖 832×480 与 1312×736，按 [客户端 benchmark 规范](../../../../../benchmark/zing_h3_causal/CLIENT_BENCHMARK.md)执行相同配置的 baseline/candidate 完整服务对照，包含真实 source、六参考图与 Lynn Decoder。按 [阶段计时规范](../../../../../benchmark/zing_h3_causal/STAGE_TIMING.md)另行采集 Encoder 的逐 rank Host/CUDA 耗时与采集开销。组件结果不能当作完整模型端到端收益，10 层 DiT 仅作诊断；具体实验数据与未验证项维护在 [PR #355](https://github.com/seedleap/sglang/pull/355)。

bias 融合独立评估。已有候选 profile 仍存在独立 bias add，当前实现保留原 bias 语义，不声称完成融合；改变 FP32 accumulator 的 epilogue 可能改变 BF16 舍入，应另行验证。

## Bias 后续约束

[cuDNN frontend graph API](https://docs.nvidia.com/cudnn/latest/developer/overview/) 能把 Conv 与 Bias 描述为一个 graph，是否融合仍须由实际 trace 证明。本次单 shape 原型确实得到一个 kernel，但 FP32/BF16 intermediate 两种设置均未满足原 Torch 的逐值门槛；历史测量见[冻结验收记录](https://github.com/seedleap/sglang/blob/ce079d7de8acc54ee5a2976492995fd9f74028b0/benchmark/zing_h3_causal/ENCODER_LAYOUT_VALIDATION.md)。

这个 Conv+Bias 融合原型未用于当前运行时；其用途是让卷积结果直接加 bias、减少独立 bias add 的遍历／启动。它与本 PR 已使用的 reflection padding kernel 是两件事，不能把原型的单 kernel 测量算作当前实现的收益。

后续可评估保留 BF16 中间舍入后，再将 bias 与下一次 norm 遍历融合；需要同时保留归约顺序、原 bias 与缓存事务语义，并重新验证全部实际 source shapes 的净耗时、plan 构建和 workspace 成本。单 shape 的速度不能作为完整 Encoder 收益。
