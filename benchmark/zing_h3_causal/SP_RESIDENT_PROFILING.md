# Zing H3 resident SP profiling

工作负载使用 `request_10s.json`：896×512、240 帧、固定 24 FPS、
video_audio、seed 42、guidance 1、四步 `[1000,750,500,250]`。
AR step 3500 权重用于验证执行速度，不代表 DMD 成片质量。
数值验收和性能测量分开进行；CPU 回归不能替代实卡 latent 对齐。

在同一硬件、软件环境和 checkpoint 上串行执行 SP1/SP2/SP4：

```bash
export SGLANG_DIFFUSION_PROFILE_ALL_RANKS=1
export SGLANG_DIFFUSION_STAGE_LOGGING=1
# Runtime 和验证 CLI 默认启用 FA4 + stable shapes + compiled AdaLN。
# SP>1 拒绝显式关闭该组合；环境需提供 flash_attn.cute。
export SGLANG_DIFFUSION_TORCH_PROFILER_DIR="$OUTPUT/traces"
python benchmark/zing_h3_causal/validate_real_checkpoint.py sglang \
  --sglang-root "$SGLANG_ROOT" --model "$EXPORTED_MODEL" \
  --request benchmark/zing_h3_causal/request_10s.json --output "$OUTPUT" \
  --sp-size "$SP" --dit-placement resident --encoder-parallel replicate \
  --no-vae-cpu-offload --no-text-encoder-offload \
  --solver few_step --denoising-step-list 1000 750 500 250 \
  --sigma-points 4 --guidance-scale 1 --seed 42 \
  --warmup-runs 1 --measured-runs 3 --save-all-raw --collect-rank-memory --profile-run
python benchmark/zing_h3_causal/analyze_sp_profile.py "$OUTPUT/traces" \
  --output "$OUTPUT/profile-summary.json"
```

每种 SP 的 `$OUTPUT` 应分别为 `$RESULT_ROOT/sp1`、`sp2`、`sp4`。
离线 trace 分析可加 `--workers 4` 并行读取各 rank；默认串行，每个 worker
会将一份完整 trace 加载进内存，需根据可用 RAM 控制并行数。
显式设置 `encoder_parallel=replicate`，防止默认 auto 在多卡时切分 encoder 权重。所有配置关闭 FSDP、DiT 与 codec
CPU offload；遇到 OOM 必须保留错误，不自动改变 placement。
正式数据先写出，profiler 单独运行后更新 `profile_seconds_separate`。

## 计时口径

- `generation_seconds_all` 是调用方 `generator.generate` 的完整 wall time，
  包括实际输出路径中的转换、编码/写盘与验证；不含模型加载和 raw tensor
  归档、有限性检查、ffprobe 或 profiler。
- torch.profiler CPU/CUDA trace 包含各 rank、causal block/history、step、
  GEMM/SDPA、Ulysses、AdaLN、KV/RoPE 和 decode。
  CPU annotation 是 host 范围；CUDA kernels 的重叠累计值不等于 wall time。
- `maybe_record_function` 在 profiler 活跃时自动记录，无 profiler 时不创建事件。
  不需要每次采集临时修改源码；profiler 扰动独立报告，不参与 speedup。
- `SP × API wall` 为参与推理的 GPU-seconds；若预留 4 卡串行跑 SP1/SP2，
  作业总分配成本另外按 4 卡乘实际占用时间计算。
- 显存需区分 allocated/reserved 以及模型加载、warmup、runtime peak。

原始 `.pt`、request、成片、每 rank trace 与环境清单需要随运行归档。
媒体检查应验证 240 帧、24 FPS、896×512、10 秒以及音轨存在；有限性检查
不能替代跨实现 latent 数值对齐，也不能证明 AR 四步输出的视觉质量。

## 并行范围

本配置的 VAE decode group 使用 `tp-sp-pp-cfg` 轴，因此 SP=1/2/4 时，
video VAE 的 tile decode group 也随之扩大。这里的 API/full-pipeline
加速同时包含 DiT Ulysses 和 VAE tile 并行；单独用 `ZingH3Denoise`
比较 DiT scaling。各配置仍保持完整权重驻留，不使用 FSDP。

Trace 可能同时提供 CPU host 与 GPU stream annotation。分析器分别报告，
不将二者累计；kernel 通过 correlation / External id 归属到最内层指定
host region。Named-region 与 causal-block 是同一批 kernel 的两个视角，
不可相加。无法归属的 kernel 仍明确列出。

## 工具保留范围

- `validate_real_checkpoint.py`：可复用的 reference / SGLang / compare 命令，
  支持外部 checkpoint、request、SP 配置、warmup、正式计时和独立 profiler。
- `capture_native_pipeline.py`、`compare_native_pipeline.py`：独立原生 Zing 捕获和
  跨实现数值比较。`compare_sp_diagnostics.py` 比较不同 SP 的诊断捕获。
- `analyze_sp_profile.py`：通过 CPU scope、External id 和 CUDA correlation
  归属 kernel，避免把嵌套区间或不同 rank 的累计时长当成端到端耗时。
- `request_10s.json`：可复用输入示例，240 帧在本例 24 FPS 下为 10 秒；
  实际 FPS 需与模型导出的 `media.fps` 一致，不代表所有 checkpoint 固定 24 FPS。

逐次实验报告、视频 HTML、冻结 prefix/refiner/tail 的一次性 replay 脚本放在
实验产物中，不作为运行时依赖或仓库长期维护入口。验收结果在 PR 中关联到
被测提交及原始产物，不能将历史结果当作修改后提交已完成的实卡测试。

`SGLANG_ZING_DIAGNOSTICS_DIR` 是可选的正确性诊断入口；启用后保存实际
scheduler 随机噪声、中间张量和 KV，以区分 RNG、计算和缓存首差。它不重新
抽样，也不改变调度算法，但同步拷贝会干扰计时，性能运行中必须关闭。

## 在 trace 中区分 attention 实现

先定位 DiT 的 `Zing.forward.*`，再看 `Zing.SDPA` 或 `Zing.FA4`。
`Zing.SDPA` 内的 `aten::_scaled_dot_product_flash_attention` 说明 PyTorch
SDPA 选择了内置 Flash Attention，不能据此称为 FA4；efficient/math 子算子
分别表示其它 SDPA 实现。`Zing.FA4` 对应显式调用
`flash_attn.cute.flash_attn_varlen_func`。最后用调用栈和 CUDA correlation
核对 GPU kernel，不凭 kernel 名中有 flash 就断定版本。VAE 与 DiT 分别查看。

## 默认数值路径

SP1/SP2/SP4 默认启用 FA4、stable shapes 和 compiled AdaLN，无需设置三个
`SGLANG_ZING_EXPERIMENTAL_*` 环境变量。旧名称保留兼容；SP1 可以显式覆盖
以做消融实验，SP>1 仍拒绝关闭已验证组合。缺少 FA4 依赖时明确报错，不静默
回落为 SDPA。验证 CLI 默认值与运行时一致，提供 `--no-sp-preserve-projection-shapes`
和 `--no-compile-adaln` 供 SP1 对照。

`--sp-preserve-projection-shapes` 保持的是关键投影的计算形状，不是视频分辨率、长度或 chunk 数：
每个 rank 完整计算小 prefix，再只保存本 rank 的 KV heads；输出端先 gather
hidden rows，再用完整行数执行 video/audio projection。这样避免 SP1/SP2/SP4
的 BF16 GEMM 因行数变化而走不同的舍入路径。主干 target attention 仍使用
Ulysses，历史 KV 仍按 heads 分片。代价是 prefix/output 的重复计算及输出
hidden gather；这是已验证的数值一致性措施，不保证独立带来加速。

## 启动参数与 checkpoint 的边界

这三个选项都是启动时的执行策略，不从 checkpoint 权重或导出配置中读取：

| CLI 参数 | 默认 | 用途 |
| --- | --- | --- |
| `--dit-attention-backend {fa4,torch_sdpa}` | `fa4` | 选择 Zing DiT attention 实现 |
| `--sp-preserve-projection-shapes` / `--no-sp-preserve-projection-shapes` | 开启 | SP 下保留关键投影的单卡行数，避免形状导致的数值分歧 |
| `--compile-adaln` / `--no-compile-adaln` | 开启 | 启用 native compiled RMSNorm + modulation 路径 |

CLI 在创建 worker 前传入运行时，模型初始化时保存选择。FA4 是当前已验证
组合的默认值，不是 checkpoint 对 attention 实现的规定；SP>1 的组合门禁
是当前支持范围约束。这个 CLI 参数仅控制 Zing DiT，不是 VAE 的全局后端开关。

旧的 `--experimental-attention`、`--experimental-stable-shapes`、
`--experimental-compiled-adaln`（含两个布尔选项的 `--no-` 形式）仍作为兼容
别名。worker 环境变量沿用旧名以兼容已有部署；新的结果 JSON 使用上述
三个规范参数名对应的下划线字段。

## Per-rank allocator evidence

`--collect-rank-memory` is opt-in. It preserves load and request snapshots as
`rank_N_load_peak` / `rank_N_request_peak` in each result's
`memory_snapshots`, with both `peak_allocated_mb` and `peak_reserved_mb`.
The load snapshot covers worker initialization; request peaks use the worker's
existing allocator reset after warmup-pool release and include output transport.
These values are allocator peaks in MiB, not physical device utilization.

The harness retains `warmup_metrics`, `measured_metrics`, and `profile_metrics`
separately. Harness warmups are ordinary requests, distinct from the server's
calibration warmups. A missing/non-finite rank peak fails the harness.
Snapshots are gathered through the replica CPU group, adding host collective
overhead to API latency; use the same collection setting across configurations.
The default runtime path does not add this collective.

## Region-by-block analysis

`profile-summary.json` now includes `region_by_causal_block` per trace. Each row
is one CPU region occurrence, with enclosing block/history rows, latent span,
denoising step and forward phase when available. Kernel attribution uses launch
correlation / External IDs even when GPU execution occurs after the host scope;
unknown or ambiguous attribution stays explicit. Do not add this view to global
region or block totals: they describe the same kernels.

`Zing.AdaLN` covers projection/indexing, not all normalization/modulation work.
`Zing.build_RoPE_and_history_positions` combines current and historical RoPE;
`Zing.RoPE_and_KV_append` is a joint scope. Kernel sums are work totals, not
end-to-end wall-time savings.

## Validate all measured raw outputs

Freeze a JSON shape specification with `visual_latents` and `audio` shape arrays,
derived from the verified model/request before examining the outputs. Then run:

```bash
python benchmark/zing_h3_causal/check_profile_outputs.py \
  --run "1=$RESULT_ROOT/sp1" --run "2=$RESULT_ROOT/sp2" \
  --run "4=$RESULT_ROOT/sp4" --expected-shapes "$EXPECTED_SHAPES" \
  --measured-runs 3 --output "$RESULT_ROOT/raw-validation.json"
```

Every `sglang-N-raw.pt` must contain nonempty finite latent/audio tensors of the
expected shapes. Invalid/missing files produce a JSON report and nonzero exit.
The report compares each repetition against run 0 and each larger SP against SP1.
`valid` checks supplied files; `matrix_complete` separately requires SP1/SP2/SP4.
A valid partial matrix still exits zero. Numerical differences remain descriptive:
no tolerance is inferred and strict parity is not certified. Media decoding,
frame rate/resolution/audio-track checks remain separate. The profiler request
has no raw archive and is not included in these raw comparisons.

CPU regressions (PyTorch, NumPy and pytest; no checkpoint or CUDA required):

```bash
python -m pytest -q benchmark/zing_h3_causal/test_profile_tools.py
```

## Current RoPE and fixed-KV writer A/B

Run the existing writer's off/on comparison on a CUDA worker:

```bash
SGLANG_LIGHTWEIGHT_RUNTIME=1 python benchmark/zing_h3_causal/benchmark_current_rope_kv.py \
  --source-revision "$SOURCE_SHA" --output-dir "$RESULT_ROOT/current-rope-kv" --trace
```

The default cases cover two-, three- and five-latent current inputs at 832×480
and 1280×736, with 28 local heads (SP2), head dimension 128 and rotary dimension
96. Both variants consume the same strided Ulysses output. The tool checks Q,
rotated K, V and clean raw K exactly, then records independent eager and warmed
Graph timings. Traces are separate from timing samples. `results.json` retains
all samples, hardware/software identity, kernel names and the fused route count.

For the full client benchmark, use separate server processes with
`SGLANG_ZING_FUSED_CURRENT_ROPE_KV=0` and `=1`, keeping the fixed-KV route and other
settings equal. Capture and warm each Graph configuration before measuring;
compare DiT stage timing and uninstrumented E2E as well as the component result.

## Fixed Zing input layout A/B

`SGLANG_ZING_SP_DESTINATION_MAJOR=1` enables destination-major packing for the
Zing batch-one input all-to-all in eager and fixed Graph attention. The pack
kernel reads projection Q/K/V views directly; the collective receive buffer is
the final attention view. This switch defaults to `1`.
`0` retains the original layout for an A/B comparison. Use the same setting on
every SP rank. The switch participates in the CUDA Graph cache key.

## Fixed Zing Norm + RoPE + KV fusion

`SGLANG_ZING_FUSED_QKNORM_ROPE_KV=1` defers QK RMSNorm until after input SP
and runs it with current Q/K RoPE and fixed KV writes in one CUDA kernel.
It supports H3 BF16 heads of width 128 with 96 rotary dimensions. The kernel
uses the existing QK norm warp reduction and preserves the BF16 norm and
RoPE product rounding boundaries. It persists normalized raw K only for the
clean prefix. The switch defaults to `0` and is part of the Graph cache key.
Combine it with `SGLANG_ZING_SP_DESTINATION_MAJOR=1` to measure the cumulative
change; compare to the existing writer enabled via
`SGLANG_ZING_FUSED_CURRENT_ROPE_KV=1` for the incremental fusion benefit.

## Fixed history RoPE live-row A/B

`SGLANG_ZING_HISTORY_ROPE_LIVE_ROWS=1` limits history RoPE reads and writes to
`[0, history_live_rows)`. The captured grid still covers the pool capacity;
blocks outside the live range return before loading or storing history. The
scalar is refreshed with the other Graph device metadata, so changing history
length does not change captured tensor shapes. Block scheduling overhead
remains. This does not introduce cross-Graph history reuse or change cache
eviction. It defaults to `1`; set it to `0` to measure the full-capacity
baseline. Disabling fused history RoPE retains the original full-capacity
fallback.

## Attention preparation ablation

Run the cumulative comparison with real NCCL communication on two CUDA devices:

```bash
SGLANG_LIGHTWEIGHT_RUNTIME=1 torchrun --standalone --nproc-per-node=2 \
  benchmark/zing_h3_causal/benchmark_attention_neighborhood.py \
  --source-revision "$SOURCE_SHA" --output-dir "$RESULT_ROOT/neighborhood" --trace
```

The five variants are the original path, existing current writer, destination
layout, post-SP Norm fusion, and live history rows. Each checks valid Q/K/V
against the original path before measuring eager and warmed Graph execution.
The defaults cover 832×480 and 1280×736, windows 6/12/21, and uneven or empty
sequence shards. Four-device runs use `--nproc-per-node=4`.

Timings include a common projection-buffer clone to refresh in-place Norm
inputs; QKV projection GEMM and Attention are outside this scope. Synthetic
history uses a 4096-row prefix and records the separate pool capacity. Traces
and per-rank samples are retained. Use the client benchmark for actual prefix
occupancy, model-stage latency, memory and E2E changes.
