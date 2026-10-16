# Zing H3 初始三块混合去噪：实现设计与实测

本文对应 [Issue #297](https://github.com/seedleap/sglang/issues/297) 和 [PR #301](https://github.com/seedleap/sglang/pull/301)，描述分支 `codex/zing-initial-block-schedule` 的实时混合去噪实现，基线为 `3842143159c40e2397de94d85f27cfc4a806aa9f`。当前已开放 review；实现、验证证据和阅读路线如下。数据 PR #299 仅提供其中一个测试 case 的输入，不包含此实现。

目标：在每次 generation 的前三个 target 生成块增加去噪计算，然后恢复两步，比较启动画质与延迟。首版只验收 OFF 和 ON 两种策略，不增加任意分段调度、训练修改或多活跃会话调度。

本 PR 仅覆盖实时推理。`sampling.py`、原离线测试 `test_zing_h3_model.py` 与 PR 基线整文件无差异，stage 的 `_build_sampler()` 与 `forward()` 保持原文。混合策略与逐块指标由实时专用 `realtime_sampling.py` 承担，仅两个实时入口使用 `_build_realtime_sampler()`；离线仍按请求原有 solver/grid 构造原 `ZingH3Sampler`，不读取新增部署覆盖。离线 AdaLN 双表支持不在本轮范围。

## 1. 使用与外部行为

已有可用的 Zing 模型部署、checkpoint 和业务请求是前提。在原服务启动命令中加入以下配置；GPU 数量、并行、文本编码器及解码器沿用部署原有配置。

ON：

```bash
sglang serve --model-path /path/to/model \
  --zing-solver few_step \
  --zing-denoising-step-list 1000,750 \
  --zing-initial-denoising-step-list 1000,750,500,250 \
  --zing-initial-block-count 3
```

OFF：

```bash
sglang serve --model-path /path/to/model \
  --zing-solver few_step \
  --zing-denoising-step-list 1000,750 \
  --zing-initial-block-count 0
```

客户端不用新增请求字段。实时入口在部署指定 solver 时，部署的 solver/grid 优先于请求；未指定部署覆盖时保留请求级路径。初始块数默认 0，初始表默认 None。原 Euler 的 `num_inference_steps` 仍是 sigma 点数；few-step 的实际迭代数来自显式列表长度。

| target 生成块（从 0 计） | 本轮 latent 数 | ON 使用的完整列表 | OFF |
|---|---:|---|---|
| 0 | 1 | 1000,750,500,250 | 1000,750 |
| 1 | 2 | 1000,750,500,250 | 1000,750 |
| 2 | 2 | 1000,750,500,250 | 1000,750 |
| 3 及以后 | 3,2 循环 | 1000,750 | 1000,750 |

`num_frame_per_block=[3,2]`、`split_first_latent=true` 使初始块为 `[1,2,2]`。本模型下前五个 latent 对应前17帧，所以第18帧起属于常规两步区间。代码按生成块编号切换，没有硬编码第17帧。**当前初始步数校验不强制核对 chunk plan；如果改变分块，前三块未必仍等于五个 latent。上述帧数解释只适用于本轮固定分块。**

## 2. 模块边界和调用链

```text
ServerArgs：解析/校验四个部署参数
  → ZingH3CausalPipeline：把四个值交给 stage
  → ZingH3GenerationStage._build_realtime_sampler：确定实时部署与请求优先级
  → ZingH3RealtimeSampler：持有常规/初始 schedule，按 state 选用
  → _sample_block：完整执行当前块的 schedule 并提交 clean KV
```

新增 `configs/zing_sampling.py` 只负责共享的列表解析与实验策略约束；不依赖模型、服务器或云部署。ServerArgs 和实时 sampler 共同复用校验，避免服务入口和直接构造实时 sampler 的调用者有不同策略。

Pipeline 传入的 `deployment_sampling` 只包含 solver、regular grid、initial grid、initial count，未把整个 ServerArgs 传入计算层。当前使用普通字典，这是首版窄接口；尚未抽象成通用策略框架。

实时 sampler 继承原 sampler，仅扩展初始化、实时逐块选择和指标记录。有限实时生成在 `step()` 内临时绑定当前块的 scheduler，调用原 `step()` 后通过 `finally` 恢复；该操作属于单活跃 generation 的同步执行。source-ready 的局部状态使用实时专用 `ZingH3RealtimeSamplingState`，无需修改原状态类型。stage 保留独立实时 factory 的少量参数组装重复，以确保离线 factory 无任何改动；不为本功能重构离线代码。

实时 sampler 持有：常规 `video_scheduler/audio_scheduler`，以及可选的初始视频/音频 scheduler 对。两者都复用已有 `ZingH3CausalRenoiseSchedule`，没有新增求解器或修改模型权重。每次构造 sampler 时创建这些对象；pipeline 启动时额外构造一次用于校验，那个校验对象不会成为后续请求共享的 sampler。

## 3. 每块怎样执行四步或两步

`prepare_schedules(device)` 将两组 schedule 的小型时间张量放到设备，并设置模态诊断信息。普通生成在 initialize 时准备；SourceReadySession 的局部状态准备路径也调用它，不在每次去噪迭代里构造 schedule。

选择逻辑等价于：

```python
block_index = (
    state.index
    if state.generation_block_index is None
    else state.generation_block_index
)
video_schedule, audio_schedule = (
    initial_schedulers if block_index < initial_block_count
    else regular_schedulers
)
for step in range(len(video_schedule.timesteps)):
    prediction = predict_current_target_at_this_timestep(...)
    video = video_schedule.step(prediction, video, step_index=step)[0]
commit_clean_target_to_history(...)
```

伪代码省略已有 CFG、source、joint 和音频分支。关键是每个块都从其列表第一个时间点开始；列表不是分摊给几个块，也不是在全程四步循环中简单跳过最后两次。

列表元素是未 shift 的整数噪声级别。已有 scheduler 按模态配置计算：

```text
base_sigma = noise_level / num_train_timesteps
sigma = shift * base_sigma / (1 + (shift - 1) * base_sigma)
t = 1 - sigma
x0 = sample + sigma * model_output
```

中间步将 x0 舍入到模型 dtype 后按下一 sigma re-noise；最后一项直接输出 clean latent，不再抽噪声，不追加 0。本轮视频 shift=8、音频 shift=3、N=1000（音频关闭）。时间表带来的差异发生在模型实际 forward 和 re-noise 中。

四步每块消耗3次 re-noise 抽样，两步每块1次。前三块 ON 合计比 OFF 多6次目标去噪迭代和6次视频 re-noise 抽样；这不是整个服务总 forward 次数，因为还有 source、clean commit、条件编码等。沿用连续 RNG，同 seed 不保证后续块使用相同噪声。

## 4. generation 状态与切换生命周期

普通有限生成使用 `ZingH3SamplingState.index`，初始值0，每成功完成一个 target 块加1。新 initialize 返回新的 state，因此同一个 sampler 再做一次 generation 仍从初始策略开始。

SourceReadySession 每次只准备当前一个局部 span，局部 `state.index` 总是0。若直接使用它，每个流式块都会错误执行四步。因此该路径显式设置：

```python
generation_block_index = int(ready.chunk.chunk_id)
```

它是这次 generation 内的全局 chunk 游标。`schedules_for` 优先使用它，确保第四块起切到两步。source 编码、参考条件预填充、clean KV commit 都不增加这个生成块计数。

同一 generation 的 `update_condition` 保留 state/index 和已有 target 历史，只按既有规则替换 prefix conditioning。因此 prompt 更新不会重新触发前三块四步。结束后创建新 generation 才重置。执行异常仍通过既有 state.close 清理缓存和 lease；本功能不增加全局可变计数器。

仓库部署模型是单活跃会话。本次未扩展并发会话能力；sampler 上既有的 last_run_metrics/device 等可变字段不应被解释为已经支持共享 sampler 的并发生成。

## 5. KV、AdaLN 与 joint 路径

**Causal KV：** 切换只改变当前块使用的时间表，不创建新 cache、不清除历史，不改变已有 window/sink/RoPE 策略。每块完成后沿用 `_commit` 把 clean target 提交到历史。两个阶段共用同一个 generation state/cache，避免第四块失去开头内容。

**AdaLN：** 时间表变化与 cache 的时间点覆盖有关，不能把“保留 causal KV”扩展解释为所有缓存都天然兼容。首版在启动参数层检查 `zing_adaln_table_path`，并在 sampler 检查 transformer.adaln_table，混合模式遇到离线单表即拒绝。本次实际运行设置 `SGLANG_ZING_ADALN_PROJECTION_CACHE=0`，使用 resident projections；没有实现或验证同时覆盖两套表的离线 AdaLN 缓存。普通启动校验不会自动证明 checkpoint 的两步画质；本次另行核对原生模型与真实推理。

**Joint：** 实时普通块和 source-ready 分别通过实时 `step()` 绑定或显式传入选定 schedule。原 `_sample_cross_chunk` 完全还原，无 PR diff。原 `_ready_cross_prediction` 也保持不变：策略校验强制两表都以1000起始，相同模态 shift 下首个 timestep 相同，因此 source-ready lookahead 可复用原实现。前一块 clean commit 的 final timestep 不随下一块 schedule 改变。此前 CPU 回归包含 joint 路径；本次代码隔离后，新增文件中的真实小模型与 source-ready cross-chunk 回归已完成两轮，每轮10项通过。

## 6. 校验与可观测性

启动参数接受逗号分隔整数；拒绝空表、非整数、不从1000起始、非严格递减、含0或越界的列表。初始块数只允许0或3；count=0不能带初始表；count=3必须为 few_step 和既定四步/两步组合。未启用混合时保留原Euler和统一schedule路径，不把所有旧请求强制成两步。

仅实时 sampler 的实际采样块记录 target_block、latent_frames、denoise_steps、noise_grid；有限实时生成保存在 `last_run_metrics.block_schedules`，最近完成块另存 `last_block_schedule`。source-ready 实验记录器从后者采集，并加上generation块游标和时间戳。记录的是实际选用的表，不仅是启动参数回显。

仅实时 metrics 字典增加字段；原离线 metrics 和对应旧测试整文件均还原。有限实时生成的逐块列表随块数增长，尚未专门测量长程记录开销。

## 7. 验证方案、实际结果与缺口

**已有 GPU 实测：** 四个 case 的 ON/OFF 共8条视频已完成，128个正式 target block 的分块、实际步数与时间表均已核对。此后将混合逻辑隔离到实时专用模块并恢复离线原文；隔离后的实现通过下述 CPU 一致性检查和 CI，但未重新运行 GPU。已有 GPU 结果对应的源码归档与运行条件在下文单独列出，画质收益仍由视频对比判断。

**当前代码的 CI：** 提交 `103ffad92ab0cb1a8ae7f1114e07f9f62b687f3d` 的 [Zing CPU CI](https://github.com/seedleap/sglang/actions/runs/38398289455) 共13项检查全部通过，覆盖 lint、CI tooling、全部选中的业务分片及汇总检查。新增测试文件已按固定 Linux Python 3.11 CPU 配方运行两轮，每轮10项通过、零跳过；完整子进程耗时63.96秒和19.62秒，pytest自身耗时49.88秒和10.67秒。核对源码哈希、JUnit和回执后登记为 `CPU_PASS`，证据见 `scripts/ci/zing/cpu_test_manifest.json`，任务 `dlc1o7a32j4xhv5q`。这些耗时不是GPU性能指标；文档后续更新不改变上述测试所对应的代码提交。

**隔离改动与审查时的补充验证：** 对比固定 PR 基线，确认 `sampling.py`、`test_zing_h3_model.py` 全文一致，stage `_build_sampler` / `forward` 源码一致。Windows 本地 CPU smoke 使用真实 sampler/scheduler 源码及合成预测、cache，检查 OFF/ON、有限实时块和 source-ready 全局游标共四组：实际步数与 `e57085899` 一致，固定 seed 的张量输出逐位相等；另测异常时 scheduler 恢复、离线/实时 factory 参数隔离，共7项，审查时重新执行通过。脚本为 `outputs/check_realtime_isolation.py`；通过 AST 隔离模型依赖，不属于真实 DiT 或 source-ready 队列集成验收。审查时另执行18项配置边界检查，通过 AST 提取生产校验函数，覆盖有效配置、非法组合和混合模式拒绝静态 AdaLN 单表；全部通过，不代表完整服务启动验证。真实小模型步数、source-ready cross-chunk、factory 隔离和异常恢复由上述两轮 Linux CPU 测试覆盖。

下表前五项为当前代码的 CPU 测试，后三项为已有 GPU 实验及其输入核验。原实验的203项 CPU、8条 GPU 视频及性能数据均对应隔离前归档源码。

| 要保护的行为 | 方法 / 通过标准 | 已有证据 |
|---|---|---|
| 实际去噪迭代正确 | 小模型真实 scheduler.step调用应为OFF全2、ON 4/4/4/2/2 | 新增test_actual_target_steps_and_new_generation，通过 |
| 新generation重置 | 同一sampler initialize两次，均重新执行相同序列 | 同上，通过 |
| prompt更新不重置计数/更换cache | 中途调用update_condition后继续，cache身份相同 | 同上，通过；不是不同文本的真实GPU切换实验 |
| source-ready全局计数正确 | 多个局部span、cross_chunk，实际输出步数为预期序列 | test_ready_chunks_use_generation_index，通过 |
| 两种CLI配置有效 | 实际ServerArgs CLI解析与校验 | test_deployment_cli，通过 |
| 真实服务分块/步数正确 | 4case×2组，实际128个target block检查 | 全部符合；每条120帧 |
| 输入与checkpoint可追溯 | 原素材bytes/SHA、checkpoint全文件SHA | 27份素材核验通过；checkpoint step1600 |
| 启动延迟与画质代价 | 同设备同模型固定配置、预热后记录首帧/前17帧/全片 | 数据已交付；质量由用户肉眼判断 |

CPU五个文件共203项通过（62.76秒）：initial_blocks、causal_renoise、chunks、h3_model、joint。实际源码归档SHA256为 `9df552eba42e1a50d1328f660806a3e52f563e054ddfc1bdecdc9e876258da28`；本文不在当时测试归档内。仓库选定 Ruff F401/F821/UP037 与 diff whitespace检查通过，不声称完整pre-commit/全仓CI通过。

GPU条件：同一双卡作业，硬件报告NVIDIA L20D、每卡275040MiB、compute capability10.3；BF16 DiT、SP2/Ulysses2、H3 FA4、TAEH3 FP16、文本编码器layerwise CPU offload、VAE CPU offload、audio off。两组各预热一次，按相同case顺序测量；seed42，每case每组一次。没有规定统计显著性阈值。

| Case | 前17帧OFF / ON（秒） | 额外等待（秒） |
|---|---:|---:|
| 海岸巡逻 | 11.513 / 12.021 | 0.508 |
| 废土拾荒 | 16.012 / 16.599 | 0.587 |
| 地中海小镇 | 14.857 / 15.452 | 0.594 |
| 潇潇／林姐 | 17.889 / 18.249 | 0.360 |

计时包含条件/source编码、生成与解码。后17帧以外的短程交付fps不等同于纯DiT或长期稳态吞吐。数据足以确认执行策略和这轮延迟代价，尚不足以证明普遍画质提升、长期稳定性能或不同seed的收益。

明确排除本轮范围：离线推理与离线 AdaLN 双表，不为其追加实现或验收。

尚未覆盖的关键路径：混合模式取消后创建新 generation、持续 source 中途更新 prompt 的专项回归，以及隔离后代码的 GPU 重跑。配置边界检查目前属于本地审查验证，尚未全部转成仓库回归用例。

本轮不承诺其他 chunk plan 的五 latent 语义、混合模式真实音视频及其他硬件兼容性、全部现有模型兼容 CI、重复性能统计或长程稳定性。已有8条视频可用于人工检查开头和切换效果，尚未据此确认普遍画质提升。

[正常、汇总慢放及各case独立慢放报告](https://review-cdn.loopit.com.cn/world-model/sglang/evals/issue297-20261013/review-web-r1/index-v2.html#cases)。本地原始证据见 `C:/sglang297/outputs/export-r1`，复现入口为 `outputs/cpu_retest.py`、`outputs/gpu_run.py`；部署JSON、逐块trace和指标均已保存。

## 8. 社区实践与 reviewer 导读

2026-10-13已核实[SGLang官方贡献指南](https://docs.sglang.io/docs/developer_guide/contribution_guide)：要求围绕真实不变量增加回归，按改动验证模型输出与性能，提交前运行配置的pre-commit。这是一般贡献要求，不代表上游已采用本Zing策略。此次复用的是seedleap fork固定基线的multimodal pipeline/stage/sampler架构。

优先review以下问题：部署覆盖优先级是否符合现有API；生成块游标是否在新generation重置、prompt更新时保留；切换是否保留clean KV；改变chunk plan时如何避免“前三块=五latent”的错误理解；AdaLN拒绝范围与后续双表需求；metrics常驻记录和schema兼容；启动时临时构造sampler是否值得保留为校验入口。

推荐阅读路线（代码路径相对仓库根目录）：

1. `python/sglang/multimodal_gen/configs/zing_sampling.py`：parse_noise_grid / validate_initial_blocks。
2. `python/sglang/multimodal_gen/runtime/server_args/server_args.py`：ServerArgs._validate_zing_sampling；再看 pipelines/zing_h3_causal_pipeline.py 如何传递四个字段。
3. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/zing_h3/stage.py`：_build_realtime_sampler 的实时部署覆盖优先级，原 _build_sampler 未修改。
4. 同目录 `realtime_sampling.py`：prepare_schedules → schedules_for → step/_sample_block → ZingH3RealtimeSamplingState；原 `sampling.py` 无差异。
5. 同目录 `incremental.py`：SourceReadySession._prepare 传递绝对chunk游标，drain_ready选用对应schedule。
6. `python/sglang/multimodal_gen/test/unit/test_zing_initial_blocks.py`：核对实际调用次数及生命周期；结合真实trace和视频判断验收程度。

当前实现已进入 review，适用的 Zing CPU CI 已通过；请结合第7节的实测范围与剩余缺口审阅。离线推理与离线 AdaLN 支持不作为本 PR 的验收项。
