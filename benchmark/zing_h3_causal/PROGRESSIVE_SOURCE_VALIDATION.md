# 渐进 source：部署与验证

## 默认行为与兼容路径

省略 `zing_realtime_profile.json` 的 `source_encoder` 时，frontend 与 worker
均选择内置 `progressive_encoder:create_session`（契约 `h3-clip17-progressive-v1`）。
其余 checkpoint hash、solver、四步 schedule、guidance、`indexed-cpu-v1` 保持不变。
显式配置 `"source_encoder": {"mode": "clip17"}` 可运行旧整 clip 基线。
自定义渐进 profile 仍需完整的 mode、contract_version、factory；不兼容时失败，
不静默回退。factory 只来自可信部署配置，不接受客户端指定。

渐进路径按 1/8/8 个新像素帧输出 1/2/2 个 latent，首帧即可提供首段稳定结果。
每层保留有限卷积历史及 stride 相位，每 17 帧重置，保持 checkpoint clip 语义。
它减少等待输入的时间，不保证提高总吞吐；BF16 小 shape 卷积与旧路径可存在数值差异。
详细接口及算法约束见 [PROGRESSIVE_ENCODER.md](../../docs/docs/sglang-diffusion/zing/specs/PROGRESSIVE_ENCODER.md)。

## 哪些文件需要保留

这里保留可复现的工具、固定输入契约和必要的证据摘要，不把逐次作业调度日志当实现文档。

| 文件 | 用途 |
| --- | --- |
| `validate_progressive_encoder.py`、`test_progressive_encoder.py` | 真实权重的分包/EOS、未来扰动、隔离/reset/所有权、数值与计时验证，以及验证器自身回归 |
| `validate_incremental_codecs.py` | 公共 codec 对照验证；复用同一数值比较与失败证据保存逻辑 |
| `continuous_source_client.py` | 可复用 MessagePack 客户端、paced 分包及输出指标采集 |
| `continuous_source_lifecycle_probe.py` | 真实服务取消、重连、暂停、慢消费者等探针 |
| `continuous_source_media.py`、对应测试 | 按真实图片字节识别 MIME，保持原始素材与 reference ID |
| [SOURCE_REFERENCE_INPUTS.md](../../docs/docs/sglang-diffusion/zing/specs/SOURCE_REFERENCE_INPUTS.md)、`WHITEBOX_VIDEO_SMOKE.md` | 固定输入顺序、校验和与准备方法 |
| `PROGRESSIVE_SOURCE_SP_RESULTS.md` | 原冻结版本的简明性能结果与适用边界 |

这些 Python 工具是复现实验和防止回归所需，不由生产服务导入。
逐次作业日志、失败张量和视频留在对应工件目录，不加入源码。
本文件原有完整调度历史保留于
[修改前的版本](https://github.com/seedleap/sglang/blob/de3629d2ce182c69b41abf2f316d0e0002e9be62/benchmark/zing_h3_causal/PROGRESSIVE_SOURCE_VALIDATION.md)，
以及 [PR #148 讨论](https://github.com/seedleap/sglang/pull/148)。历史失败没有被改判为通过。

## 输入与证据边界

默认 case 为 `zing-whitebox-video-five-triview-0001`：Picture 1 全局图、Picture 2–6
五张配对图，保持原 prompt 与顺序。参考图最多九张，不能将 source 首 RGB 自动注入
prompt 视频。独立首帧变体必须通过 `--case` 显式选择，见输入文档。

已记录的 encoder-only 证据冻结于
`63b4972ce656c0d1ebe6e9c1044b1dd914ce4dd2`：140 个真实用例、12 组计时，
跨分包 hash 一致，无 NaN/Inf，未来扰动、candidate 隔离/reset/所有权/dispose 通过。
旧 clip17 对照仍有 136/140 非 exact，有限数值差异仅报告，不能称数值验收通过。

| encoder-only 指标 | clip17 | 渐进 |
| --- | --- | --- |
| paced 首稳定 moments 中位 | 1.12997 s | 0.08312 s |
| 计算范围吞吐 | 32.152 FPS | 28.540 FPS |
| allocated 峰值 | 11.932 GB | 17.768 GB |

该结果只支持提前产出；吞吐约下降 11%，显存约增加 49%，不是端到端速度或画质证明。
证据位于 `world-model/sglang/evals/zing148-final-20260922/sp1-r1/` 的
`encoder-reviewed-and-service-stopped.json` 及原始报告。旧输入服务作业已停止，
无完整矩阵标记；旧五图视频和历史 FPS 不代表当前六图输入结果。

## 重跑要求

- 冻结源码 SHA、镜像 digest、模型/config/profile、source RGB、prompt 与有序 refs 的 hash，
  以及 seed、RNG、分包、SP、实测硬件、缓存预算和误差策略。
- 同版 clip17/渐进 A/B：5/10/60 秒各一预热三次正式；SP1 审核后 SP2。
  EOS 1–34、逐帧/不规则包 hash，暂停、慢消费者、取消/end 竞态、重连及不同 source/seed 隔离。
- 非有限值、隔离失败、漏帧和资源释放仍是硬门槛；有限 encoder 数值差异保留 report-only。
- `measure_source_cuda=true` 才生成同步 `latent_ready`；`latent_submitted` 和 `dit_started`
  是 host 事件。两侧测量模式一致，独立 profiling 不混入正式吞吐。
- 正常 EOS 与异常释放分别审计，逐 rank 检查长流缓存恒定与增量计算线性。
  10/60 秒拼接仅证明技术连续性，24 FPS 和独立画质另行验收。

完整六图矩阵冻结 `386543ac257d7fd190707a1a1796ad6006a4873d`，SP1/SP2 原始产物已独立复核。
默认入口、独立 target-video、双 rank 异常释放另冻结
`5d20e78e64e2d5f4e432dbb94edf663fb89c2b20` 完成增量真机与 87 件产物逐 SHA 复核；
不把旧矩阵改标为最新源码。速度、首画面、显存及仍独立跟进的目标见上述结果文档。
