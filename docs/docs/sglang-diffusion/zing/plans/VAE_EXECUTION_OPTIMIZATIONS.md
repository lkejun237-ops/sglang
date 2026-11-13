# H3 Encoder 与 Lynn Decoder 融合优化

本方案涵盖 H3 EncoderFCN3D/CausalEncoderCache 融合优化、交付 EMA Lynn StreamingDecoder 块内优化，以及 2026-11-12 新增的 SM90 单卡快速解码引擎（`SGLANG_LYNN_SM90_FAST_DECODE`），保留原权重与逐 latent 输出语义。
2026-11-05 根据 Jack47 的 review 收缩早期范围：移除早期低效 Lynn workspace、整块 compile 及每流重复捕获的旧版 Graph；保留 Encoder 两项融合和 Lynn 三项块内优化。2026-11-12 新增面向 Hopper SM90 单卡的 `LynnSm90FastEngine`（模型级常驻引擎 + 单活跃会话槽位租约 + 6 模式预捕获池化 CUDA Graph + WGMMA FP8 / FA3 FP8 + 浅层 4 层 Full-Tile / 深层 22 层双线性唯一网格混合），2026-11-13 按用户确认默认开启，可用 `SGLANG_LYNN_SM90_FAST_DECODE=0` 关闭；完整 product-case 冻结 latent Decoder A/B 已完成，三角色 E2E 与人工画质仍待验收。在非 SM90 / 多卡 / 小尺寸输入时自动无感回退。

## 使用与回退

启动相应 worker 前设置环境变量，初始化时读取并固定。显式 `0` 优先于默认值。

| 完整变量 | 默认 | 计算 |
|---|---|---|
| `MINIMAX_H3_PROGRESSIVE_FUSED_NORM_INPUT` | `1` | 逐帧 GN/SiLU、布局、padding、下一步历史写出 |
| `MINIMAX_H3_PROGRESSIVE_FUSED_CONV_INPUT` | `1` | 历史拼接、padding、next-cache 搬运 |
| `SGLANG_LYNN_PACKED_QKV` | `1` | 原 Q/K/V 参数构成非持久 packed 投影 |
| `SGLANG_LYNN_FUSED_QK_NORM` | `1` | 当前 Q/K 的 FP32 RMSNorm 融合 |
| `SGLANG_LYNN_COMPILED_FFN` | `1` | 局部编译 norm2、FFN、scale2 与残差 |
| `SGLANG_LYNN_SM90_FAST_DECODE` | `1` | SM90 单卡 WGMMA FP8 + FA3 FP8 + 唯一网格混合 + 6 模式池化 CUDA Graph |

以下六项默认开启，显式环境变量仍优先：

```bash
export MINIMAX_H3_PROGRESSIVE_FUSED_NORM_INPUT=1
export MINIMAX_H3_PROGRESSIVE_FUSED_CONV_INPUT=1
export SGLANG_LYNN_PACKED_QKV=1
export SGLANG_LYNN_FUSED_QK_NORM=1
export SGLANG_LYNN_COMPILED_FFN=1
export SGLANG_LYNN_SM90_FAST_DECODE=1
# 然后使用实际部署命令启动 worker。
```

仅做原版 A/B 或故障回退时使用以下关闭示例；这不是默认启动配置：

```bash
export MINIMAX_H3_PROGRESSIVE_FUSED_NORM_INPUT=0
export MINIMAX_H3_PROGRESSIVE_FUSED_CONV_INPUT=0
export SGLANG_LYNN_PACKED_QKV=0
export SGLANG_LYNN_FUSED_QK_NORM=0
export SGLANG_LYNN_COMPILED_FFN=0
export SGLANG_LYNN_SM90_FAST_DECODE=0
# 然后使用原部署命令启动对应 worker。
```

已删除旧版 `SGLANG_LYNN_WORKSPACE` 和 `SGLANG_LYNN_BLOCK_EXECUTION` 的解析与执行分支，旧部署应删除这两项配置。局部 compiled FFN 不等于整块 compile；当 `SGLANG_LYNN_SM90_FAST_DECODE=1` 在 SM90 单卡命中时直接走整步快速引擎，未命中或显式设为 `0` 时回退 `SGLANG_LYNN_PACKED_QKV` / `FUSED_QK_NORM` / `COMPILED_FFN` 块内融合或参考实现。

## 决策与边界

历史 34 帧实测 workspace/Graph 组合耗时 351.36→746.53ms，peak reserved 7.41→20.68GiB；无捕获稳态仍无收益。两者绑定同一消融轴，不能判断每项单独均慢，但不足以支持这套组合进入生产并长期维护。整块 compile 缺少全权重验收，也一并移除。未来重新引入需独立 PR，分别消融并提供真实服务收益。

Encoder 保留逐帧统计、causal history、stride phase 和失败重试事务；新历史独立分配。ConvLayout 显式携带 input_memory_format，不能用 1×1×1 权重连续性推断输入格式；shortcut/通道变换层和第三阶段继续保留 NCDHW。

Lynn packed 权重与 compiled FFN 缓存归模型；参数/device/dtype 改变时失效。活跃 stream 仅保存参数签名，检测变化后关闭流。pre-RoPE KV 归 tile/session；输入仍走原来的 cat，移除 Graph/static input/history workspace；不影响已有像素输出缓冲和 SP 协议。

GN 融合、packed GEMM 与编译可能改变舍入。默认开启是部署策略，不是严格无损或整体加速承诺。历史 GN 完整生成 PSNR 28.116dB，人工画质尚未验收。

## 启动链与配置传递

仓库通用入口 `runtime/disaggregation/role_launcher.py::start_role_workers` 为 H3/Lynn 的 Encoder 和 Decoder 分别组装环境，以 setdefault 写入对应优化默认值，再把该环境传给 Popen；显式 0 保留。池化入口 `runtime/launch_server.py` 用 multiprocessing spawn，继承父进程环境，worker 仍按运行时默认值解析。

H200/B300 指推理运行的硬件环境，launcher 指这些环境使用的通用服务启动入口，不要求按显卡型号提供两套独立脚本。代码默认值及 Encoder/Decoder 子进程环境传递已修改并完成 CPU 回归；剩余验证是 H200/B300 环境中真实 worker 的启用情况、精度、性能与显存，而非查找按型号命名的文件。本次未修改或重启线上实例。

## 代码职责与 review 阅读路线

| 文件/入口 | 职责与检查重点 |
|---|---|
| `minimax_h3_video_vae/progressive.py`、`progressive_layout.py` | 默认值、显式布局与 cache 事务 |
| `progressive_fusion.py`、`progressive_kernels.py` | dtype/device/layout 准入与 Triton 张量计算 |
| `lynn_streaming/sm90_fast_decode.py` | SM90 准入校验、唯一网格映射、WGMMA FP8 + FA3 FP8、融合 Triton 核、单会话槽位租约与 6 模式池化 CUDA Graph |
| `lynn_streaming/execution.py` | 四项选项、packed 权重、compiled FFN 缓存及 `LynnSm90FastEngine` 委托 |
| `lynn_streaming/model.py`、`stream.py` | tile KV、活跃流参数签名（含稳态 CPU–GPU 重叠校验）、SM90 租约释放与单步融合截断输出 |
| `disaggregation/role_launcher.py`、`deploy/sglang/launch.py` | 角色子进程环境默认注入与显式覆盖白名单传递 |
| `benchmark/zing_h3_causal/vae_execution_matrix.py`、`benchmark_vae_execution.py` | `decoder-sm90` 评测格定义与组件基准评测 |
| `test_zing_lynn_execution.py`、`test_zing_h3_progressive_fusion.py`、`test_zing_source_decoder_launcher.py` | 数值、跨流确定性、并发租约回退、稳态权重篡改拒绝、默认/覆盖和启动传参回归 |

社区依据沿用同源 [ComfyUI H3 Encoder](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/vae.py) 的 GN/SiLU/布局融合和 [Diffusers 局部编译实践](https://huggingface.co/docs/diffusers/main/en/optimization/fp16)，并适配本仓库增量 cache。它们不证明本 PR 性能；上游 review 应重点检查精度、回退、派生权重失效及真实收益，不以默认关闭替代方案必要性论证。

## 当前验证与历史证据

当前工具保留七个组合：`baseline`、`encoder-norm`、`encoder-copy`、`encoder-all`、`decoder-block`、`decoder-sm90`、`all`。Encoder 为 2×2，Decoder 支持全关、块内融合与 SM90 单卡快速解码；所有六项变量均显式设置，避免继承部署默认。已删除的旧版配置不能继续作为可运行矩阵格，旧八格原始记录保留在下方。

### SM90 单卡快速解码验收（2026-11-12）

- **代码与硬件身份**：源码提交 `1c0246cfecb2b65e84b0d40f2c8f751ac7139a34`，上海 H100 节点 `e01-cn-2zd4xi8m505` 单卡 `NVIDIA L20Z`（真实 CUDA SM90 / CC 9.0，132 SM，81,079 MiB），Torch `2.13.0+cu130`、CUDA `13.0`；模型为交付 26 层 EMA BF16 `StreamingDecoder`（`width=2048`, `heads=32`, `dim_head=64`, `tile_size=256`, `tile_overlap=64`, `temporal_window=3`）。
- **基准工具与样本口径**：官方 `benchmark/zing_h3_causal/benchmark_vae_execution.py decoder`，输入来自 `hiker-clay-01-yellow-raincoat` 原版 Encoder 冻结 latent（34 帧 = 10 latent steps），每格 1 次不计时数值采集 + 1 次完整预热 + 5 次正式重复；`steady_push_ms` / `steady_cuda_ms` 统计每流 `index >= 3` 的完整稳态 push（`n = 35`），包含输入搬运、图重放、输出拷贝截断与状态更新。

| 分辨率 | 评测格 | `steady_push_ms` 中位 / P95 (最大) | `steady_cuda_ms` 中位 / P95 | `step0_ttff` (ms) | 34 帧总耗时中位 (s) | 相对 `baseline` 加速 | 常驻 alloc / reserved (MiB) | 峰值 alloc / reserved (MiB) | `[0,1]` RGB PSNR / RMSE / MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `832×480` (12 tiles) | `baseline` | 35.441 / 35.490 (35.495) | 35.432 / 35.481 | 32.391 | 0.3504 | 1.00× | 3393.4 / 7590.0 | 7436.2 / 7590.0 | exact |
| `832×480` (12 tiles) | `decoder-block` | 30.183 / 30.905 (31.314) | 30.174 / 30.894 | 27.465 | 0.2994 | 1.17× | 4017.7 / 8150.0 | 8011.6 / 8150.0 | 65.653 dB / 0.000522 / 0.000184 |
| `832×480` (12 tiles) | **`decoder-sm90`** | **9.759 / 9.824 (9.872)** | **9.748 / 9.819** | **9.612** | **0.0971** | **3.63×** (较 `block` **3.09×**) | 6756.2 / 7232.0 | **6774.4 / 7232.0** | **41.809 dB** / 0.00812 / 0.00499 |
| `1280×736` (28 tiles) | `baseline` | 81.347 / 81.993 (82.146) | 81.337 / 81.984 | 73.851 | 0.8043 | 1.00× | 3419.2 / 13232.0 | 12852.4 / 13232.0 | exact |
| `1280×736` (28 tiles) | `decoder-block` | 68.345 / 70.828 (71.325) | 68.336 / 70.818 | 60.437 | 0.6734 | 1.19× | 4043.5 / 13792.0 | 13362.6 / 13792.0 | 66.688 dB / 0.000463 / 0.000151 |
| `1280×736` (28 tiles) | **`decoder-sm90`** | **22.361 / 23.099 (23.314)** | **22.352 / 23.090** | **19.804** | **0.2180** | **3.64×** (较 `block` **3.06×**) | 9013.1 / 9990.0 | **9056.2 / 9990.0** | **43.849 dB** / 0.00642 / 0.00440 |

- **附加数值、生命周期与生产契约核验**：
  - `wire_uint8_truncation`（0–255）：`832×480` MAE = `1.269`、RMSE = `2.106`；`1280×736` MAE = `1.122`、RMSE = `1.679`。
  - Tile 接缝边界梯度误差比（相对 `baseline` 内部梯度误差）：`832×480` `seam_to_int_ratio_x = 1.379`、`y = 1.543`；`1280×736` `seam_to_int_ratio_x = 1.457`、`y = 1.673`，无分块接缝跳变。
  - 启动预热与零捕获首会话：模型权重常驻 `3344.8 / 6766.0 MiB`（alloc / reserved），`warmup_decoder(model, width=832, height=480)` 耗时 `6.696 s` 预捕获全部 6 种时序环模式，预热后常驻 `6707.6 / 7172.0 MiB`；随后首个真实 BF16 会话在独立私有 CUDA Stream 上 10 步 CUDA 耗时 `[10.780, 9.102, 9.638, 9.640, 9.647, 9.652, 9.577, 9.559, 9.587, 9.689] ms`，零懒加载捕获停顿。
  - 确定性与安全边界：跨流重复逐位一致（`exact_repeat_480 = True`、`exact_repeat_736 = True`）；遵循单活跃会话部署约束，并发第二个流自动回退块内融合/参考路径且不窃取首流 `_SlotLease`；流内中途修改权重/设备/hook 立即同步抛出 `ValueError` 并释放租约。
  - 单测回归：CPU launcher 传参单测 `27 passed`，H100 GPU Lynn execution / stream / startup warmup 单测 `195 passed`（合计 `222 passed`）。
- **未验证边界**：本轮仅修改并实测上海 H100（SM90）单卡 Lynn Decoder 组件；Encoder 与 DiT 代码未修改，H200 / B300、Decoder SP 多卡、完整三角色 `product-case-20261029` 在线流 E2E 及人工主观画质尚未在本 PR 重测。

### 2026-11-05 历史快照回归记录

已执行：Linux Python 3.11 / Torch 2.8 CPU，Encoder 6 passed / 70 CUDA skipped，Lynn 5 passed / 3 CUDA skipped，launcher 27 passed；共 38 passed / 73 CUDA skipped。覆盖默认值、显式关闭、CPU 数值/流生命周期、packed 权重失效及 Popen 环境传参，launcher 使用受控进程替身，不是线上 GPU 启用证据。矩阵六格/全开五个 1/基线五个 0 检查、Ruff/isort/diff 检查通过；manifest 按本次 receipt 更新。CUDA 单测仍覆盖真实 Triton/小型 Lynn；本轮没有重跑这些专项用例。H200/B300、736、完整组件性能与 allocator 显存 A/B、人工效果仍未验收。

2026-11-05，当前 VAE runtime `a44609ba9893` 在上海 L20Z/SM90 四卡完成默认组合完整推理：832×480、W6/S3、FA4/SP2、DiT Graph 关闭，同一黄色雨衣 case，1 次完整预热 + 1 次正式均 360/360，prompt 更新与 EOS 通过。测试 runner 省略五项 VAE 开关；Encoder 实际命中 norm 19500 / conv 6240 / fallback 0，Lynn worker 从通用 launcher 收到三个 `1`。正式轮 18.209 FPS，仅为单次功能验收记录，不用于宣称整体加速。RGB 与历史 r21 五项开启且 workspace/Graph 关闭的输出逐字节一致，仍不等于与原版无损或人工画质通过。源码为此前冻结成功快照覆盖当前 PR 全部 Python/benchmark 改动，保留已合入 main #520 的两行 EOS 修复；不是纯分支 HEAD 或最新 main 的端到端验收。原始回执、配置差异和指标统一见 [PR #524](https://github.com/seedleap/sglang/pull/524)。

以下 GPU 数据来自 P2 输入布局修复及本次范围收缩之前的冻结快照，仅解释历史消融和移除原因，不能充当当前 head 的效果或性能验收。

### 当前实测身份与范围（2026-11-03）

生产代码`5bb2967485c8bb9d9904b46f83e0c52f304056ed`加main分页缓存收尾两行修复及回归。
基线和优化格使用同一源码，只切换本页7个VAE选项。实际GPU为上海NVIDIA L20Z、SM90/132SM，
Torch2.13.0+cu130、CUDA13.0、Triton3.7.1、FA4 4.0.0b19。
DiT为默认1027 step10000 checkpoint，SHA256
`89c3a67d77214e57d810e6a4e5554c7c919ae929db0174e6439dfadb53a1d731`，配置SHA256
`9a8acb3331a82f1e6d3c61b98d5e75f12faaabbc5b808040e0fcfda7925721ce`；
保留`[3,2]`、unsplit及对应原生weights-free AdaLN表。

四卡固定布局：GPU0 Encoder/Qwen，GPU1–2 DiT SP2，GPU3 Lynn；832×480、W6/S3、FA4，无FSDP、无offload。用户指定DiT Graph总开关及SP开关均0，Lynn Graph仍为矩阵轴。单一黄色雨衣case完整360帧、原5图与两版prompt；source120发送更新、实际生效边界逐轮核对。每格1完整预热、3正式、1独立诊断，共40个成功360帧请求；正式FPS仅使用24次无stage timing的请求。Qwen配置各格相同，跨格复用同一服务，每格相同hot-cache策略。

除7个VAE开关外逐格环境diff均为空；运行源码逐文件hash、模型身份、实际两版prefix9521/9650、
池上限9728/23970行/25.773GB均冻结。Encoder tiling=auto在独立group1实际回退replicated，
原因为`unvalidated_group_size`，没有据开关名宣称tile并行命中。输入融合的实际计数另列。

### 同一case前34帧的组件四格

已完成；2026-11-03，上海 SM90/L20Z，同一物理 GPU0。实际 H3 EncoderFCN3D 与 26层 EMA BF16 Lynn；使用本地 `hiker-clay-01-yellow-raincoat` source 的前34帧、832×480。Decoder 输入为同一次原版 Encoder 的 latent，所有 decoder 格共用。每格预热后3次，计时覆盖完整34帧重建；不是完整360帧DiT生成或整体FPS。

| 组件 | 开关 | 中位 ms | P95 ms | 相对速度 | 波动范围/中位 | 峰值 reserved GiB |
|---|---|---:|---:|---:|---:|---:|
| encoder | 全关 | 1208.04 | 1208.19 | 1.000× | 0.02% | 8.17 |
| encoder | GN/SiLU/布局融合 | 988.12 | 988.37 | 1.223× | 0.03% | 8.17 |
| encoder | 历史/cache/padding复制融合 | 1198.34 | 1198.48 | 1.008× | 0.15% | 8.01 |
| encoder | Encoder全开 | 969.41 | 970.71 | 1.246× | 0.15% | 8.01 |
| decoder | 全关 | 351.36 | 351.81 | 1.000× | 0.29% | 7.41 |
| decoder | QKV/QK norm/FFN融合 | 300.41 | 301.91 | 1.170× | 0.65% | 7.96 |
| decoder | workspace/CUDA Graph | 746.53 | 1069.10 | 0.471× | 53.40% | 20.68 |
| decoder | Decoder全开 | 792.77 | 1610.77 | 0.443× | 123.14% | 20.78 |

数值：Encoder复制融合的 moments、normalized latent 逐位一致。GN融合（单开/全开相同）moments max_abs=0.59375、RMSE=0.0132477、RRMSE=0.2644%；normalized latent max_abs=0.421875、RMSE=0.00543541、RRMSE=0.5084%，不能标为严格无损。

Decoder workspace/Graph 单开，preclamp RGB、[0,1] RGB、实际截断uint8均逐位一致。块内融合（单开/全开相同）的[0,1] RGB max_abs=0.00715636、RMSE=0.000308007、PSNR=70.2288 dB；uint8最大差2、相同像素分量比例96.9647%。Graph格已实际命中：每次流26 captures、182 replays、234 workspace，52次冷历史路径eager；不是因开关未命中导致的负收益。

结论边界：本case支持Encoder复制融合的数值无损判断；单开收益不足1%，不夸大。Encoder GN与Decoder块内融合有舍入差异，完整生成数值另列，不能用34帧重建误差替代完整生成比较。历史 Decoder Graph 测试显著变慢且波动大；当前已移除 workspace/Graph，保留五项融合默认开启；不从组件速度推算整体FPS。

#### 组件Graph捕获与稳态

每格先执行不计时数值轮，再执行一次完整预热，然后3次正式。`_GraphCall`捕获前也会在独立CUDA stream执行两次function。正式轮每次创建新stream；Graph归stream所有并随close释放，所以每次新请求仍捕获26张图。34帧重建793ms含这些真实启动成本，不能称为纯replay耗时。

已有记录中的无捕获稳态调用：排除每流前3次push及graph_capture调用，相同phase分布，3流合计21次。只比较完整latent push，包含正常输入/输出复制和状态更新，不冒称裸kernel耗时。

| 配置 | n | 中位ms/push | P95ms/push |
|---|---:|---:|---:|
| baseline | 21 | 35.527 | 35.628 |
| decoder-block | 21 | 30.370 | 30.706 |
| decoder-execution | 21 | 42.558 | 43.508 |
| decoder-all | 21 | 36.155 | 36.308 |

稳态全开相对原版慢约1.8%，与整流0.443×是两种口径；这组34帧组件调用即使排除捕获，Graph仍未显示收益；完整服务首批后指标另见下表。实际完整请求需要计入stream初始化/捕获成本，完整360帧FPS继续使用真实服务原生语义，不能从计时中删掉请求本身承担的捕获。

### 完整360帧整体矩阵

| 配置 | 整体FPS中位 | 3轮范围 | 相对全关 | 首批后FPS | 完整生成RGB数值 |
|---|---:|---|---:|---:|---|
| 全关 | 18.250 | 18.241–18.264 | 1.0000× | 21.400 | 逐字节一致 |
| Encoder GN/SiLU | 18.301 | 18.275–18.344 | 1.0028× | 21.416 | PSNR 28.116dB |
| Encoder历史搬运 | 18.060 | 16.533–18.149 | 0.9896× | 21.253 | 逐字节一致 |
| Encoder全开 | 18.270 | 18.213–18.306 | 1.0011× | 21.317 | PSNR 28.116dB |
| Decoder QKV/QK/FFN | 18.256 | 18.231–18.311 | 1.0003× | 21.405 | PSNR 62.307dB |
| Decoder workspace/Graph | 18.129 | 17.404–18.197 | 0.9934× | 21.707 | 逐字节一致 |
| Decoder全开 | 18.029 | 17.954–18.099 | 0.9879× | 21.626 | PSNR 62.307dB |
| Encoder+Decoder全开 | 18.217 | 18.052–18.236 | 0.9982× | 21.734 | PSNR 28.116dB |

完整RGB相对全关第一正式轮比较，报告PSNR是数值差异，不是相对真值的画质分数。
完整逐轮MAE、max、RMSE、相同比例、SHA及prompt生效回执保存在`FPS_SUMMARY.json`。
不删除慢轮、不从组件速度推算整体速度；同节点另有其它GPU作业，小幅中位变化不证明稳定收益。

Encoder原生末次完整诊断会话的命中：GN单开norm19500；搬运单开conv25740；全开norm19500/conv6240，
三者fallback均0。Lynn原生完整worker未导出执行计数；组件矩阵有直接capture/replay证据，
不把完整服务环境变量为1当成命中计数。

整体显存采用正式请求时间段内0.5秒NVML采样峰值及请求前占用，GPU0为Encoder/Qwen共享；
GPU1首轮请求前占用随格顺序74407→76451MiB，本实验未独立归因该趋势，不把它直接归因于VAE。
不能称作组件独占或精确allocator峰值。各格原生诊断为Host spans，CUDA状态not_requested；
enqueue、等待和重叠阶段不能相加当作E2E/GPU计算。阶段明细见`NATIVE_OBSERVATIONS.json`，
直接组件Host/CUDA中位/P95见组件原始JSON；完整服务各组件CUDA归因及人工画质未验收。

### 前置修复与失败证据

main#499改为分页`Block0PreparationCache.pages`后，`source_session.dispose()`仍用旧`.data`
字段写容量日志，EOS阻断最后3帧。测试快照包含两行pages修复及真实分页布局收尾回归，
Linux完整source-session模块681 passed、1 warning/8.63s，完整矩阵均正常收尾。
该修复已由main的[#520](https://github.com/seedleap/sglang/pull/520)合入，且增加dispose/EOS参数化回归；
本PR不重复提交这两处改动。上述GPU证据对应冻结测试快照，不冒称最新main的其它改动已重测。

早期r5–r14使用错误旧DiT/混合配置，全部无有效正式FPS，不列入本次对照。r16虽然预热360/360，
但首正式DiT init Graph额外快照约8.08GB/卡导致OOM；按用户要求关闭DiT Graph后不再扩展修改该逻辑。
r17只读缓存链接、r18首次遗漏JSONL为测试准备错误，均无正式样本；失败分别保留。
r18b采用独立可写缓存、模型加载前真实benchmark输入预检，执行期间源码和runner未修改。

原始证据位于仓库外`D:/vae-matrix-h100-20261103-r18/`及同名NVMe目录：
`FPS_REPORT.md`、`FPS_SUMMARY.json`、`NATIVE_OBSERVATIONS.json`、`fps-readback.tar.gz`。
组件原始值为r15的`SELECTED_COMPONENTS_RAW.json`及`SELECTED_COMPONENT_REPORT.md`。
完整归档SHA/字节数另记`FINAL_ARTIFACT_RECEIPT.json`；原始日志、媒体和跟踪文件不提交仓库。

归档：`oss://leap-world-shanghai/world-model/sglang/evals/vae-matrix-h100-20261103-r18/fps-readback.tar.gz`，
5757392字节，SHA256 `61ee9737e1d09e2433eb85af3d5ce4406af83b7f10e7dbe300ed455f0f05d0b0`。
[原版/联合全开视频对比](https://review-cdn.loopit.com.cn/world-model/sglang/evals/vae-video-20261103-r20/index.html)
需公司网络/VPN，媒体签名有效至2026-11-10；视频仅在云端处理，未下载到本地。
联合全开视频重新生成后的原始RGB SHA与r18正式轮一致；尚未完成人工画质验收。

当前 review 重点：五项默认开启与显式关闭是否正确；Encoder cache/layout/stride 合同是否保持；packed 权重失效及流参数签名是否正确；launcher 是否向实际子进程传入配置。历史 Graph 已移除，不再验收其资源生命周期。当前 L20Z 完整推理证据与 H200/B300 等未验证范围见前文。
