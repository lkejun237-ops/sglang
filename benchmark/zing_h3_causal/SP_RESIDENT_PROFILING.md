# Zing H3 resident SP profiling

工作负载使用 `request_10s.json`：896×512、240 帧、固定 24 FPS、
video_audio、seed 42、guidance 1、四步 `[1000,750,500,250]`。
AR step 3500 权重用于验证执行速度，不代表 DMD 成片质量。
数值验收和性能测量分开进行；CPU 回归不能替代实卡 latent 对齐。

在同一硬件、软件环境和 checkpoint 上串行执行 SP1/SP2/SP4：

```bash
export SGLANG_DIFFUSION_PROFILE_ALL_RANKS=1
export SGLANG_DIFFUSION_STAGE_LOGGING=1
# SP>1 启动前强制检查该已验证组合；SP1 对照也使用相同设置。
export SGLANG_ZING_EXPERIMENTAL_ATTENTION=fa4
export SGLANG_ZING_EXPERIMENTAL_STABLE_SHAPES=1
export SGLANG_ZING_EXPERIMENTAL_COMPILED_ADALN=1
export SGLANG_DIFFUSION_TORCH_PROFILER_DIR="$OUTPUT/traces"
python benchmark/zing_h3_causal/validate_real_checkpoint.py sglang \
  --sglang-root "$SGLANG_ROOT" --model "$EXPORTED_MODEL" \
  --request benchmark/zing_h3_causal/request_10s.json --output "$OUTPUT" \
  --sp-size "$SP" --dit-placement resident --encoder-parallel replicate \
  --no-vae-cpu-offload --no-text-encoder-offload \
  --solver few_step --denoising-step-list 1000 750 500 250 \
  --sigma-points 4 --guidance-scale 1 --seed 42 \
  --experimental-attention fa4 --experimental-stable-shapes \
  --experimental-compiled-adaln \
  --warmup-runs 1 --measured-runs 3 --profile-run
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
