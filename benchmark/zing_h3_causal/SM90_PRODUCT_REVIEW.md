# Lynn SM90 产品视频画质对照

先运行原生 Encoder 准备，在有足够显存的资源上单独加载 DiT 生成一次，
最后在同一张 SM90 GPU 上分别加载 `decoder-block` 和 `decoder-sm90`，解码同一份冻结 latent。
已实测的 80 GiB H100 即使移出 Encoder，仍不能容纳完整产品 case 的 DiT 历史。
大显存生成可使用 `stage_sm90_frozen_dit.py` 的 CPU 准备、单卡 B300 生成、CPU 导出流程；
B300 只生成输入，不用于计算 H100 Decoder 的性能收益。
这是离线组件验证，不能替代生产三角色的 E2E 性能测试。
完整生产指标及采集口径见 [CLIENT_BENCHMARK.md](CLIENT_BENCHMARK.md)。

## 输入与版本

- 固定 `product-case-20261103` 全部 8 例，包含 7 个 `832×480` 和 1 个 `1280×736`。
- 保留完整 prompt、参考图顺序、每次完整参考图替换及原始 source 帧边界。
- 素材按 manifest 的 SHA256 与字节数校验，不裁剪、不重采样、不循环输入。
- 标准原生 Qwen/reference VAE 准备、渐进 source VAE、queued DiT 和 Lynn stream。
- seed 123、BF16 DiT、W6/sink3、after-step0、block-relative RoPE、weights-free AdaLN，
  首 3 块 4 步，后续 2 步。模型、AdaLN table 必须成对匹配。
- 实际 GPU UUID、镜像 digest、模型身份、每个阶段源码完整 SHA、命令和日志需要随结果归档。
  所有实验源码应先提交并推送，再从该 SHA 打包运行。

## 运行

80 GiB H100 分阶段执行：先加 `--prepare-only` 在 CPU 核验素材并建立 8 例描述，
再用同一参数加 `--phase encode` 运行 Encoder。将完整编码结果交给大显存 DiT worker；
收回全部 `latents/*/COMPLETE.json` 及对应张量后，在 H100 上加 `--resume` 执行两种 Decoder。
不要在 80 GiB H100 上直接运行完整 `all` 流程；入口会在已知不够的容量上提前拒绝。

在有完整 GPU 依赖的现有实验镜像中，先核实指定 GPU 空闲且属于本次实验。
`model`、`lynn`、`adaln-table` 使用只读挂载；`media-root` 保存 manifest 对应的素材树。

```bash
SGLANG_LIGHTWEIGHT_RUNTIME=1 \
SGLANG_ZING_EXPERIMENTAL_ATTENTION=fa4 \
PYTHONPATH=/path/to/checkout/python \
python3 benchmark/zing_h3_causal/run_sm90_frozen_review.py \
  --model /path/to/model \
  --lynn /path/to/lynn \
  --adaln-table /path/to/paired-adaln-table \
  --media-root /path/to/media \
  --output /path/to/new-owned-results \
  --source-sha FULL_PUSHED_COMMIT_SHA
```

`render_sm90_review_pod.py --frozen-latents` 可生成指定 UUID 的隔离 Pod 配置，
只生成配置，不自行提交。素材和源码分别放在该 owned root 的 `media/`、`source/`。
只有已确认完整的阶段才可加 `--resume` 复用；部分生成结果不能作为续写 RNG/KV 状态，
应先按精确路径归档，不能覆盖或混入完整结果。

Encoder 结果在 `encoded/`，生成 latent 在 `latents/`，两种解码结果分别在
`decoder-block/` 和 `decoder-sm90/`。每例生成回执记录逐文件 SHA256；
两次解码前均重新校验。SM90 正式结果要求真实 26 层模型、`fast_start=4` 且实际执行计数大于零。

## 指标与视频

每种 Decoder 每例完整预热 1 次、完整正式重复 3 次，第一轮正式重复保存全部帧。
每次预热和正式回放都保存完整 uint8 RGB 的 SHA256，可检查同组重复是否一致；
这不证明浮点输出逐位一致，也不能替代人工 A/B 画质验收。
`summary.json` 中逐 push 的 Host/CUDA 时间排除磁盘读取、输入 H2D 和视频编码；
一个 push 可能包含不同数量的 latent，比较时必须保持相同 chunk 序列。
中位数/P95 与样本量是 Decoder push 口径，不能解释为浏览器 TTFF 或单帧延迟。

显存是该独立解码进程的 allocated/reserved，包含输入 latent、输出及可视化临时张量；
常驻与峰值分开记录，不等于生产角色的 driver 显存。
本工具不测 Encoder/DiT 的正式性能，也不提供生产 E2E、source-to-output、播放停顿指标。

在有 FFmpeg 的 CPU 环境生成对比页：

```bash
python3 benchmark/zing_h3_causal/build_sm90_video_review.py \
  --results /path/to/owned-results \
  --output /path/to/new-report
```

页面提供三栏同步播放、滑动对比、A/B 瞬时切换、逐帧、慢速和更新边界跳转。
原录屏可能是 VFR，白膜预览按帧序定时为 24 FPS，不增删帧。
原尺寸全帧 H.264 CRF10 视频是视觉副本，不是无损像素差异数据。
构建时检查全部 8 例、两组冻结 latent 身份、完整解码、尺寸、帧数及白膜预览时长。
人工画质结论由 reviewer 给出。

RGB 临时 spool 和编码消息是运行期 scratch。完成并停止所有自有 worker 后，
按实验的精确路径清单清理 scratch，并检查所有尝试合计的最终保留预算（默认 2 GiB）。
保留视频、原始指标、身份与哈希回执；不要把模型、视频或缓存提交到 Git。


## 默认开启与显式关闭对照

`decoder-block` 显式设置 `SGLANG_LYNN_SM90_FAST_DECODE=0`；`decoder-sm90`
移除该环境变量，验证 serving 默认开启。其他执行选项不变。
逐 case `summary.json` 同时保存 `switch_environment`（默认组为 null）、
解析后的 `execution_options` 和实际路径计数，避免把显式设为 1 冒充默认值验收。
