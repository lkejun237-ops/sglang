---
title: "Zing H3 Visual Encoder H100 单卡融合加速方案"
description: "H3 Video VAE 视觉编码器在 NVIDIA H100 (SM90) 单卡上的全 Tile 批处理与 Triton 融合引擎设计、契约与实测结果。"
---

本文档记录 Zing H3 视觉编码器（`MiniMaxH3VideoVAE` / `EncoderFCN3D` / `CausalEncoderCache`）在 NVIDIA H100（SM90）单卡上的执行引擎优化设计、契约不变量与真机实测结果。

## 1. 优化目标与核心结论

针对单卡 H100 上 Source VAE 流式编码（渐进会话 `H3ProgressiveEncoderSession.pop()`）与参考图/片段编码（`ZingH3Codecs.encode_visual` / `encode_visual_moments`）的延迟瓶颈，目标为：

- `480p (832×480)` 单卡单帧耗时优化至 **`< 20 ms`**；
- `736p (1280×736 / 1312×736)` 单卡单帧耗时优化至 **`< 60 ms`**。

在上海灵骏 H100（`NVIDIA L20Z`，SM90 / CC 9.0 / 132 SMs / 81,559 MiB，隔离卡 `GPU-6d5005a8-3721-3368-3cb1-893d25094c1c`）上基于真实服务模型权重（1027 `global_step_010000` 配套 H3 Video VAE，BF16）同条件 A/B 实测：

- **`832×480`（12 tiles）**：
  - 首帧 `pop(T=1)` 端到端耗时中位数从 `66.984 ms` 降至 **`18.901 ms`**（加速 **`3.54×`**，纯 GPU **`18.606 ms`**）；
  - 参考图 `encode_visual(T=1)` 中位数从 `36.683 ms` 降至 **`17.567 ms`**（加速 **`2.09×`**）；
  - 34 帧流式序列（`1+8+8+8+8+1`）均摊每帧从 `28.836 ms`（34.68 FPS）降至 **`12.306 ms`**（**81.26 FPS**，加速 **`2.34×`**）；
  - 峰值显存从 `21,355.23 MiB`（reserved `32,192.0 MiB`）下降至 **`10,937.41 MiB`**（reserved **`13,620.0 MiB`**，节省 **48.8%** 分配峰值）。
- **`1280×736`（28 tiles）**：
  - 首帧 `pop(T=1)` 端到端耗时中位数从 `155.700 ms` 降至 **`38.957 ms`**（加速 **`4.00×`**，纯 GPU **`38.302 ms`**）；
  - 参考图 `encode_visual(T=1)` 中位数从 `82.887 ms` 降至 **`36.799 ms`**（加速 **`2.25×`**）；
  - 34 帧流式均摊每帧从 `67.456 ms`（14.82 FPS）降至 **`29.963 ms`**（**33.37 FPS**，加速 **`2.25×`**）。
- **`1312×736`（28 tiles）**：
  - 首帧 `pop(T=1)` 端到端耗时中位数从 `155.828 ms` 降至 **`38.930 ms`**（加速 **`4.00×`**，纯 GPU **`38.763 ms`**）；
  - 参考图 `encode_visual(T=1)` 中位数从 `82.967 ms` 降至 **`36.792 ms`**（加速 **`2.26×`**）；
  - 34 帧流式均摊每帧从 `67.481 ms`（14.82 FPS）降至 **`29.941 ms`**（**33.40 FPS**，加速 **`2.25×`**）。
- **数值精度与状态安全**：真实模型 34 帧流式 `moments` 相对原版 BF16 基线的 RRMSE 仅 **`0.0042`（~0.42%）**、余弦相似度 `>= 0.9996`；会话 `reset()` / `dispose()` 后 `retained_bytes_after_dispose == 0`；SM90 CUDA 单测与独立 `zing_source_encoder_server` 启动预热契约检查全部通过。

## 2. 开关配置与自动安全回退

快速路径默认开启，无需修改既有启动命令；做 A/B 对比或故障排查时可通过环境变量无损切回原路径：

| 环境变量 | 默认值 | 作用说明 |
| --- | --- | --- |
| `MINIMAX_H3_ENCODER_NUMERICAL_OPTIMIZATIONS` | `1` | 数值重排许可；设为 `0` 关闭本 PR 快速引擎（包括全 tile 合批、全栈 NTHWC、延迟 bias、融合 GN 与扩展 cuDNN 搜索），并关闭渐进路径旧的融合 GN；保留原布局策略和历史/padding 搬运 |
| `MINIMAX_H3_ENCODER_SINGLE_GPU_FAST` | `1` | 控制 `CausalEncoderCache` 单卡 BF16 Triton+cuDNN 全融合加速引擎开关；设为 `0` 回退原版逐 tile 路径 |
| `MINIMAX_H3_FAST_VISUAL_ENCODE` | `1` | 控制 `ZingH3Codecs.encode_visual` / `encode_visual_moments` 在单卡条件下复用快速引擎；设为 `0` 回退 `vae.encode` |
| `MINIMAX_H3_FAST_ENCODER_WARMUP` | `1` | 控制 `warmup_source_vae()` 启动阶段在保留原有 256×256 1/8 帧预热后，预编译默认分辨率（`MINIMAX_H3_FAST_ENCODER_WARMUP_SHAPES`）的 Triton/cuDNN 算子 |
| `MINIMAX_H3_FAST_PINNED_H2D` | `1` | 控制 CPU `uint8` 帧经双槽锁页内存环形缓冲执行异步 H2D 传输 |
| `MINIMAX_H3_FAST_ENCODER_BENCHMARK_LIMIT` | `8` | 快速引擎内部的线程安全 cuDNN 算法搜索上限（允许 SM90 命中 WGMMA implicit GEMM 核，退出上下文自动恢复原值） |
| `MINIMAX_H3_FAST_CONV_MAX_ELEMS` | `2147483647` | 3D 卷积输入/输出张量单批次元素上限（超过时沿 tile batch 维自动安全分片，防止 32-bit 索引溢出） |

**自动回退守卫（无需人工干预）**：当检测到以下任一情况时，自动透明回退至原有实现：
1. 多卡分布式并行（SP `world_size > 1` 或 VAE tile parallel `world_size > 1`）；
2. 非 CUDA 设备、Triton 不可用或计算精度非 `torch.bfloat16`（含运行期动态切换 `.float()`）；
3. 开启了 `torch.are_deterministic_algorithms_enabled()`、`cudnn.deterministic=True` 或 `encode_mode="deterministic"`；
4. 显式配置了 `MINIMAX_H3_PROGRESSIVE_ENCODER_TILE_BATCH_SIZE`、`stack_tiling=True` 或非均匀/极小不支持 `reflect` 填充的 tile 尺寸；
5. 模型结构非标准 H3 `EncoderFCN3D`，或测试对 `model`、`encoder.down` 子模块、`quant_conv`、`CausalEncoderCache` 注册了自定义实例级 `forward` / hook 覆盖。

## 3. 瓶颈根因与五大核心优化方法

### 数值重排回退（2026-11-13）

新增 `MINIMAX_H3_ENCODER_NUMERICAL_OPTIMIZATIONS=0`，在 Encoder worker 启动前设置并重启。
默认仍为 `1`，不改变已有部署行为；仅接受 `0/1`。source 会话构造时冻结该值。
关闭后不准备/预热新引擎，source 和 reference/clip 编码均回退原计算路径；source 同时关闭
`MINIMAX_H3_PROGRESSIVE_FUSED_NORM_INPUT`，但保留独立的历史/padding 搬运和原有
`MINIMAX_H3_PROGRESSIVE_CHANNELS_LAST` 策略。无需同时关闭两个 FAST 开关。

不能只关闭 bias 就称为数值等价：tile 合批可改变卷积算法，GN 归约与全栈布局也相互依赖，
因此这些部分成组回退。新开关不覆盖显式设置的旧 tile batch、SP、checkpoint、后验噪声或采样策略，
不承诺跨设备/跨进程生成视频逐位一致；它控制的是 Encoder 计算重排，而非全局 deterministic 模式。
关闭后的性能、真实权重 CUDA 数值及完整视频尚未实测；既有 H100/B300 报告属于开启模式。
已增加原路径逐值比较的 CUDA 回归，未执行时不得据 CPU 准入测试标记数值验收通过。

### 3.1 全 Tile 单次批处理与全栈物理 `[N, T, H, W, C]` 零拷贝布局

- **原路径瓶颈**：`832×480` 切分为 `3×4 = 12` 个 `256×256` 空间重叠 tile，`1280×736` / `1312×736` 切分为 `4×7 = 28` 个 tile。原版 `CausalEncoderCache.encode()` 默认逐 tile 串行循环（或 `encode_visual` 按 `encode_tile_batch_size=2` 分批），每个流式步在 34 个 3D 卷积层与数十个 GroupNorm/SiLU/Pad 之间产生上千次微小 CUDA kernel 启动，无法填满 H100 的 132 个 SM；同时 PyTorch `channels_last_3d` 在 `[B, C, T, H, W]` 逻辑形状与 `[B, T, H, W, C]` 物理步幅之间频繁触发非合并跨步读写。
- **优化设计**：将整帧全部 `N = 12`（480p）或 `N = 28`（736p）个空间 tile 一次性打包为单一张量，并在整个编码管线内部统一使用物理连续的 `[N, T, H, W, C]` 张量存储。向 `F.conv3d` 传参时，仅通过零拷贝元数据视图 `x_nthwc.permute(0, 4, 1, 2, 3)` 暴露为 `channels_last_3d` 兼容的 `[N, C, T, H, W]`，权重复用 `_progressive_encoder_weights` 已转换好的 `channels_last_3d` 副本（零重复权重显存开销）。

### 3.2 五类 100% HBM3 合并访存的 Triton 融合核

在 `fast_encoder_kernels.py` 中实现针对物理 `[N, T, H, W, C]` 连续通道维（`C in (128, 256, 512)`，`stride(-1) == 1`）向量化读写的 5 类 Triton 核，消除所有中间逐元素、拼接与填充微核：

1. **`tile_pack_and_pad_uint8` / `tile_pack_and_pad_bf16`**：单核融合全画布 `reflect2d` 边界映射、全部 `N` 个重叠 tile 坐标提取、ImageNet 归一化（`mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)`，对全部 256 个 `uint8` 像素值精确复刻 PyTorch BF16 三步舍入 `(u8.bf16 / 255 - mean) / std`，实现 **0-ulp 逐位一致**）、因果时间前导补零或 2 帧历史缓存拼接，以及四边空间 `reflect`（或 `constant`）`pad=1`，直接输出 `[N, T_pad, 258, 258, 3]`。
2. **`fused_gn_silu_history_pad_nthwc`**：单算子融合可选延迟残差/偏置相加（FP32 寄存器累加）、逐帧 `GroupNorm(32, eps=1e-6)`、SiLU 激活、因果卷积历史缓存读写，以及下游 `3×3×3` 卷积所需的四边空间 `reflect`（或 `constant`）`pad=1`。
   - 小空间网格（`H * W <= 4096`，即 `64×64`、`32×32`、`16×16` 阶段）采用单核 L1-Resident 单遍归约（`_gn_silu_pad_single_pass_kernel`）；
   - 大空间网格（`256×256` 与 `128×128` 阶段）采用双核合并归约（`_gn_partials_coalesced_kernel` + `_gn_apply_silu_pad_coalesced_kernel`），中间统计量由当前流缓存分配器按需分配，天然支持多流与跨线程并发。
3. **`fused_residual_bias_downsample_pad_nthwc`**：针对 Stage 0–3 残差块末端接 `Downsample3D` 的路径，单核完成 `y2 + conv2_bias + residual` 的 FP32 求和、因果时间历史更新，以及右下角非对称 `(0, 1, 0, 1)` `reflect`（或 `constant`）填充，直接送入下采样卷积。
4. **`fused_add_bias_inplace_nthwc`**：针对后续需要直接消费残差和的分支点，原地向量化完成残差与延迟偏置相加。
5. **`fused_tile_blend_nthwc_to_ncdhw`**：预计算每输出位置重叠 tile 索引与线性混合权重查找表（`TileBlendLUT`），在输出端单核将 `[N, T_lat, 16, 16, 48]` 线性混合并转置回标准 `[1, 48, T_lat, H_lat, W_lat]` NCDHW 布局，与原版 `_assemble_tiles` 顺序一致。

### 3.3 全网 34 层 3D 卷积的 FP32 代数偏置延迟吸收（Bias Absorption）

cuDNN `F.conv3d(..., bias=b)` 在部分 SM90 implicit GEMM 内核下会退化为独立偏置广播写回核。由于 `EncoderFCN3D` 中绝大多数卷积紧接逐帧 `GroupNorm(32)` 或残差相加，对于卷积输出 `y = Conv(x) + b`：
- 无残差直连 GroupNorm 的层（如 `conv1 -> norm2`）：卷积以 `bias=None` 调用，偏置向量直接传入下一个融合 GroupNorm 核的输入寄存器相加，省去一次全张量 HBM 读写；
- 残差汇合点：在 `prepare_fast_single_gpu_encoder()` 阶段预先以 FP32 合并偏置向量（例如 `conv_out` 偏置经 `quant_conv` 线性变换后与 `quant_conv.bias` 合并），在下游融合核或 `F.linear` 中一次完成累加。

这不仅消除了近 30 次独立偏置广播读写，还减少了一次中间 BF16 截断（在合成小模型相对精确 FP32 真值的对比单测中，Fast BF16 的 RRMSE 为 `1.27%`，比原生 BF16 的 `1.65%` 更接近 FP32 真值）。

### 3.4 双槽锁页内存（Pinned Memory）异步 `uint8` H2D DMA

在 `H3ProgressiveEncoderSession.pop()` 中，当 `CausalEncoderCache.can_encode_uint8_fast` 为真时，跳过 CPU 侧逐帧列表复制与 Python 层逐帧归一化拼接，直接将 `[3, T, H, W]` `uint8` 张量通过 `stage_uint8_to_gpu()` 拷入按设备维护的 2-slot Pinned Memory 环形缓冲（每槽按 CUDA event 同步复用，扩容重分配前同步全部在途 event），执行 `non_blocking=True` 的紧凑 `uint8` PCIe DMA（相比 BF16 传输字节量减少 **50%**，832×480 单帧仅 `1.19 MB`，H2D 实测仅 `0.27 ms`）。

### 3.5 SM90 cuDNN WGMMA 搜索策略与 32-bit 索引溢出切片保护

- 原版全局编码策略将 `cudnn_benchmark_limit` 限制为 `1`，导致 cuDNN 在部分大 batch tile 形状上回退至次优核。快速引擎在可重入锁保护的 `_fast_cudnn_plan_policy()` 上下文中将 `cudnn_benchmark_limit` 临时提升至 `8`（退出立即还原），使 H100 稳定选出 `sm90_xmma_fprop_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize256x128x64_warpgroupsize2x1x1` WGMMA 核。
- 同时在 `_conv3d_nthwc()` 中同时检查输入与输出张量元素总数是否超过 `_MAX_32BIT_ELEMS = 2,147,483,647`（涵盖 `conv_in` 通道 `3 -> 128` 扩张），超限时自动沿 batch 维安全分片，杜绝 cuDNN 32-bit 索引溢出。

## 4. 契约不变量与正确性保障

快速引擎严格保持原版渐进式编码器的全部外部语义与状态契约，并由 `python/sglang/multimodal_gen/test/unit/test_zing_h3_fast_single_gpu_encoder.py` 强制守护：

1. **流式水位契约（`1 + 8 + 8 + ...`）**：首帧单独输出 1 个 latent 时间步，后续每满 8 帧输出 2 个 latent 时间步，`finish()` 处理尾部不足 8 帧余量并在 17 帧 clip 边界自动重置。
2. **任意分包 0-ulp 一致性**：同一视频序列按 `[1, 8, 8]`、`[3, 6, 8]`、`[9, 8]` 或逐帧 `append` + `pop` 编码，快速引擎产生的流式输出之间 **0-ulp 逐位完全一致**。
3. **严格因果性（0-ulp）**：修改 `t > t_0` 的任何未来像素帧，对 `t <= t_0` 已产出的所有 latent 张量保持 **0-ulp 零影响**。
4. **异常事务回滚原子性**：所有层间因果历史缓存在本步全部计算中只写入局部 `pending` 字典而不原地修改 `CausalEncoderCache.states`，仅在 `pop()` 末尾 `cache.commit()` 时原子提交并推进会话 `_pixels` / `_consumed` 游标；当注入 `quant_conv.forward` 抛错时，会话缓存帧与历史状态分毫不损，恢复后重试与无故障执行逐位一致。
5. **会话显存零残留**：`session.reset()` 与 `session.dispose()` 调用 `cache.clear()` 清空 `CausalEncoderCache.states` 字典，实测 `retained_bytes_after_dispose == 0`。

## 5. 上海 H100 单卡真机实测结果

### 5.1 测试环境与资产身份

- **机器与隔离卡**：上海灵骏节点 `e01-cn-2zd4xi8m505`（`world-model-b27mys7b-0000`），物理隔离单卡 GPU 0 `GPU-6d5005a8-3721-3368-3cb1-893d25094c1c`，设备名 `NVIDIA L20Z`（H100 SM90 / CC 9.0 / 132 SMs / 81,559 MiB）。
- **软件栈**：PyTorch `2.13.0+cu130`、CUDA `13.0`、cuDNN `92000`、Triton `3.7.1`。
- **模型权重**：`/mnt/zing-nvme/copy-whitebox-release-20261107/runtime/model/vae`（1027 `global_step_010000` 配套 H3 Video VAE，`MiniMaxH3VideoVAEConfig(load_decoder=False)`，BF16）。
- **复现工具与原始回执**：基准脚本 `benchmark/zing_h3_causal/benchmark_h100_fast_encoder.py`（预热后正式采样 `repeats=9`，34 帧流式序列 `1+8+8+8+8+1`）；原始报告归档于 `oss://leap-world-shanghai/world-model/benchmarks/encoder-fast-h100/c30453558261b3474f5ee666a924a262285cf01a/r6/results.tar.gz`。

### 5.2 同条件 Baseline vs Candidate 耗时对比（`n = 9`，17f 为 `n = 4`）

| 分辨率 (Tiles) | 指标口径 | Baseline 中位数 / P95 (ms) | Candidate 中位数 / P95 (ms) | 加速比 (中位数) | 目标门限 |
| --- | --- | --- | --- | --- | --- |
| **`832×480`** (12) | 渐进首帧 `pop(T=1)` 端到端 (含 CPU `uint8` H2D) | `66.984` / `67.109` | **`18.901`** / `19.289` | **`3.54×`** | `< 20 ms` (PASS) |
| **`832×480`** (12) | 渐进首帧 `pop(T=1)` 纯 GPU 输入 [注1] | `66.595` / `66.684` | **`18.606`** / `131.362` | **`3.58×`** | `< 20 ms` (PASS) |
| **`832×480`** (12) | 参考图单帧 `encode_visual(T=1)` | `36.683` / `37.374` | **`17.567`** / `19.704` | **`2.09×`** | `< 20 ms` (PASS) |
| **`832×480`** (12) | 渐进稳态块 `pop(T=8)` 折合每帧 | `26.827` / `26.909` | **`11.900`** / `12.105` | **`2.25×`** | `< 20 ms` (PASS) |
| **`832×480`** (12) | 34 帧流式序列总计 / 均摊每帧 | `980.415` (`28.836`/f, 34.68 FPS) | **`418.387`** (**`12.306`**/f, **81.26 FPS**) | **`2.34×`** | `< 20 ms` (PASS) |
| **`832×480`** (12) | 17 帧片段 `encode_visual` 折合每帧 | `29.999` / `63.745` | **`12.247`** / `12.542` | **`2.45×`** | `< 20 ms` (PASS) |
| **`1280×736`** (28) | 渐进首帧 `pop(T=1)` 端到端 (含 CPU `uint8` H2D) | `155.700` / `165.328` | **`38.957`** / `40.497` | **`4.00×`** | `< 60 ms` (PASS) |
| **`1280×736`** (28) | 渐进首帧 `pop(T=1)` 纯 GPU 输入 [注1] | `154.942` / `155.139` | **`38.302`** / `156.216` | **`4.05×`** | `< 60 ms` (PASS) |
| **`1280×736`** (28) | 参考图单帧 `encode_visual(T=1)` | `82.887` / `84.282` | **`36.799`** / `39.180` | **`2.25×`** | `< 60 ms` (PASS) |
| **`1280×736`** (28) | 渐进稳态块 `pop(T=8)` 折合每帧 | `62.778` / `62.822` | **`29.394`** / `29.621` | **`2.14×`** | `< 60 ms` (PASS) |
| **`1280×736`** (28) | 34 帧流式序列总计 / 均摊每帧 | `2293.501` (`67.456`/f, 14.82 FPS) | **`1018.734`** (**`29.963`**/f, **33.37 FPS**) | **`2.25×`** | `< 60 ms` (PASS) |
| **`1280×736`** (28) | 17 帧片段 `encode_visual` 折合每帧 | `70.597` / `143.159` | **`29.730`** / `30.047` | **`2.37×`** | `< 60 ms` (PASS) |
| **`1312×736`** (28) | 渐进首帧 `pop(T=1)` 端到端 (含 CPU `uint8` H2D) | `155.828` / `156.092` | **`38.930`** / `40.212` | **`4.00×`** | `< 60 ms` (PASS) |
| **`1312×736`** (28) | 渐进首帧 `pop(T=1)` 纯 GPU 输入 [注1] | `155.033` / `155.290` | **`38.763`** / `158.775` | **`4.00×`** | `< 60 ms` (PASS) |
| **`1312×736`** (28) | 参考图单帧 `encode_visual(T=1)` | `82.967` / `84.080` | **`36.792`** / `39.869` | **`2.26×`** | `< 60 ms` (PASS) |
| **`1312×736`** (28) | 渐进稳态块 `pop(T=8)` 折合每帧 | `62.853` / `62.992` | **`29.154`** / `29.455` | **`2.16×`** | `< 60 ms` (PASS) |
| **`1312×736`** (28) | 34 帧流式序列总计 / 均摊每帧 | `2294.344` (`67.481`/f, 14.82 FPS) | **`1018.003`** (**`29.941`**/f, **33.40 FPS**) | **`2.25×`** | `< 60 ms` (PASS) |
| **`1312×736`** (28) | 17 帧片段 `encode_visual` 折合每帧 | `70.519` / `70.613` | **`29.919`** / `30.025` | **`2.36×`** | `< 60 ms` (PASS) |

> **注 1**：基准脚本预热阶段使用 CPU `uint8` 输入走锁页缓冲 H2D；随后测试纯 GPU `uint8` 直传 `pop_t1_gpu` 时，第 1 个样本首次触发设备侧指针对齐特化的 `_tile_pack_pad_uint8_kernel` Triton JIT 编译（约 110–120 ms），第 2–9 个样本分布在 `18.23–20.42 ms`（480p，中位数 `18.606 ms`）与 `37.17–40.25 ms`（736p，中位数 `38.302` / `38.763 ms`）。包含完整 H2D 传输的 `pop_t1_e2e` 9 个样本 P95 分别为 `19.289 ms`（832×480）、`40.497 ms`（1280×736）、`40.212 ms`（1312×736）。

### 5.3 显存占用与数值误差对比（仅 Source/Reference VAE 组件单卡进程口径）

| 分辨率 | 模式 | 常驻 Allocated (MiB) | 请求峰值 Allocated (MiB) | 请求峰值 Reserved (MiB) | 流式会话缓存 `retained` (GiB) | `dispose()` 后 `retained` | 34f `moments` RRMSE / Cosine (相对 Baseline) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `832×480` | Baseline | `769.95` | `21,355.23` | `32,192.0` | `3.16` | `0 B` | — |
| `832×480` | Candidate | `769.95` | **`10,937.41`** (-48.8%) | **`13,620.0`** (-57.7%) | `3.15` | `0 B` | **`0.00425`** / `0.99997` |
| `1280×736` | Baseline | `822.75` | `26,598.60` | `37,728.0` | `7.38` | `0 B` | — |
| `1280×736` | Candidate | `823.11` | `27,270.17` (+2.5%) | `39,876.0` (+5.7%) | `7.36` | `0 B` | **`0.00424`** / `0.99963` |
| `1312×736` | Baseline | `825.04` | `25,441.75` | `55,600.0` | `7.38` | `0 B` | — |
| `1312×736` | Candidate | `825.04` | `27,273.34` (+7.2%) | **`39,880.0`** (-28.3%) | `7.36` | `0 B` | **`0.00422`** / `0.99963` |

### 5.4 CUDA Profiler 算子耗时归因（`pop(T=1)` + `pop(T=8)` 共 9 帧联合采样）

- **`832×480`（总 GPU 时间 `106.66 ms`）**：
  - `aten::cudnn_convolution`（60 次调用，SM90 WGMMA implicit GEMM）：**`80.58 ms`（占 `75.6%`）**；
  - 全部融合 GroupNorm + SiLU + History + SpatialPad Triton 核合计：**`18.06 ms`（占 `16.9%`）**；
  - 融合 Residual + Bias + DownsamplePad Triton 核：**`3.99 ms`（占 `3.7%`）**；
  - 原地偏置注入与其它辅助算子：**`3.76 ms`（占 `3.5%`）**；
  - 锁页内存异步 H2D DMA（`Memcpy HtoD`）：**`0.27 ms`（占 `0.25%`）**。
- **`1280×736`（总 GPU 时间 `259.28 ms`）**：
  - `aten::cudnn_convolution`（65 次调用，SM90 WGMMA）：**`190.83 ms`（占 `73.6%`）**，已逼近 H100 Tensor Core + HBM3 硬件带宽上界。

### 5.5 独立 `zing_source_encoder_server` 启动预热验证

使用官方启动预热验证脚本执行独立 Encoder 服务进程（加载 H3 Source VAE + Qwen3-VL FP8 文本编码器）：
- 保留既有 `256×256` 1 帧与 8 帧 `CausalEncoderCache.forward` 契约调用，并新增 `832×480` 与 `1280×736` 快速路径 Triton/cuDNN 启动预热；
- `/health` 在全部预热完成后由 `503` 切换为 `200`，总预热耗时 **`30.19 s`**（其中 Source VAE 预热 **`6.09 s`**，Qwen 无参考图文本编码预热 **`24.10 s`**，输出 `[6, 5120]` BF16），输出 `ENCODER_STARTUP_WARMUP_PASS`，进程干净退出且 GPU 驱动显存归零。

## 6. 验证边界说明

- 本轮实测覆盖上海 H100 单卡上的 H3 Visual Encoder（Source VAE 渐进流式编码与 Reference/Clip `encode_visual` / `encode_visual_moments`）及独立 `zing_source_encoder_server`（含 Qwen FP8 加载与启动预热）。
- 尚未在本分支复测三角色（Encoder + DiT + Decoder）联跑 `product-case-20261029` / 736p 全链路的客户端 TTFF、E2E、FPS、TPOC、ICL 及 DiT/Decoder 分项耗时与显存，亦未在泰国 B300（CC 10.3）上实测该快速引擎；跨角色与跨硬件指标将在完整建组回归中单独采集，不从单卡组件加速比直接外推。
