# Zing H3 初始三块混合去噪：实现设计与实测

当前 PR308 默认开启前三块四步、后续两步；默认值与覆盖规则见[默认混合采样](default-mixed-sampling.md)。静态预计算 AdaLN 表支持完整混合计划，见[静态表设计与迁移](initial-block-static-adaln.md)。下文的历史实验保留原提交和范围，旧单计划表仍不能直接用于混合采样。原离线推理路径仍保持不变。

本文对应 [Issue #297](https://github.com/seedleap/sglang/issues/297) 和 [PR #301](https://github.com/seedleap/sglang/pull/301)，保留初始混合去噪设计与历史实测。PR308 直接面向 main，承接文档迁移并将当前用法统一为默认开启，不再依赖 PR305 的独立配置方案。数据 PR #299 仅提供其中一个测试 case 的输入，不包含此实现。

目标：在每次 generation 的前三个 target 生成块增加去噪计算，然后恢复两步，比较启动画质与延迟。首版只验收 OFF 和 ON 两种策略，不增加任意分段调度、训练修改或多活跃会话调度。

本功能仅覆盖实时推理。`sampling.py`、原离线测试 `test_zing_h3_model.py` 与 PR 基线整文件无差异，stage 的 `_build_sampler()` 与 `forward()` 保持原文。混合策略与逐块指标由实时专用 `realtime_sampling.py` 承担，仅两个实时入口使用 `_build_realtime_sampler()`；离线仍按请求原有 solver/grid 构造原 `ZingH3Sampler`，不读取新增部署覆盖。离线 AdaLN 双表支持不在本轮范围。

## 1. 使用与外部行为

已有可用的 Zing 模型部署、checkpoint 和业务请求是前提。在原服务启动命令中加入以下配置；GPU 数量、并行、文本编码器及解码器沿用部署原有配置。

ON：默认开启前三块策略，无需附加采样参数。

```bash
sglang serve --model-path /path/to/model
```

OFF（有限实时入口保持请求 solver/grid；连续 source 保留 profile 的常规 grid）：

```bash
sglang serve --model-path /path/to/model --zing-initial-block-count 0
```

需要与 ON 做全程两步对照时，显式指定常规两步即可：

```bash
sglang serve --model-path /path/to/model \
  --zing-initial-block-count 0 \
  --zing-solver few_step \
  --zing-denoising-step-list 1000,500
```

客户端不用新增请求字段。初始块数默认 3，解析时补齐未指定的 `few_step`、`[1000,500]`、`[1000,750,500,250]`；显式冲突 solver/grid、空表仍报错。连续 source 的显式 CLI 优先于 profile 明确声明的计划，profile 未声明 initial 阶段时采用默认混合计划；profile 显式 count=0 保留单阶段。静态表须声明完整匹配计划。原 Euler 的 `num_inference_steps` 仍是 sigma 点数；few-step 的实际迭代数来自显式列表长度。

| target 生成块（从 0 计） | 本轮 latent 数 | ON 使用的完整列表 | OFF（显式常规两步） |
|---|---:|---|---|
| 0 | 1 | 1000,750,500,250 | 1000,500 |
| 1 | 2 | 1000,750,500,250 | 1000,500 |
| 2 | 2 | 1000,750,500,250 | 1000,500 |
| 3 及以后 | 3,2 循环 | 1000,500 | 1000,500 |

`num_frame_per_block=[3,2]`、`split_first_latent=true` 使初始块为 `[1,2,2]`。本模型下前五个 latent 对应前17帧，所以第18帧起属于常规两步区间。代码按生成块编号切换，没有硬编码第17帧。**当前初始步数校验不强制核对 chunk plan；如果改变分块，前三块未必仍等于五个 latent。上述帧数解释只适用于本轮固定分块。**

## 2. 模块边界和调用链

```text
ServerArgs：补齐 ON 缺省值，再解析/校验四个部署参数
  → ZingH3CausalPipeline：把四个值交给 stage
  → ZingH3GenerationStage._build_realtime_sampler：确定实时部署与请求优先级
  → ZingH3RealtimeSampler：持有常规/初始 schedule，按 state 选用
  → _sample_block：完整执行当前块的 schedule 并提交 clean KV
```

`ServerArgs._validate_zing_sampling()` 是 CLI 和 Python 构造共用的部署边界。dataclass 和 argparse 的 count 默认均为3，再按 ON 条件补齐三项缺省配置；显式 count=0 关闭初始阶段。每次创建新的列表，避免部署实例间共享可变配置。连续 source 按参数来源解析完整计划，不将默认补齐误判为显式覆盖。直接构造实时 sampler 的调用方仍必须给齐策略，保持计算层的显式接口。

`configs/zing_sampling.py` 只负责共享的列表解析与实验策略约束；不依赖模型、服务器或云部署。ServerArgs 和实时 sampler 共同复用校验，避免服务入口和直接构造实时 sampler 的调用者有不同策略。

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

同一 generation 的 prompt 更新统一按最近 5 个已生成 target latent 重建 KV，清除旧条件和旧历史 KV，但保留全局 generation block index。因此 clean KV 重建不会重新触发前三块四步；后续去噪仍按全局块号选择调度。结束后创建新 generation 才重置。执行异常沿用会话 dispose 清理缓存和 lease，不增加全局可变计数器。

仓库部署模型是单活跃会话。本次未扩展并发会话能力；sampler 上既有的 last_run_metrics/device 等可变字段不应被解释为已经支持共享 sampler 的并发生成。

## 5. KV、AdaLN 与 joint 路径

**Causal KV：** 切换只改变当前块使用的时间表，不创建新 cache、不清除历史，不改变已有 window/sink/RoPE 策略。每块完成后沿用 `_commit` 把 clean target 提交到历史。两个阶段共用同一个 generation state/cache，避免第四块失去开头内容。

**AdaLN：** 时间表变化与 cache 的时间点覆盖有关，不能把“保留 causal KV”扩展解释为所有缓存都天然兼容。首版在启动参数层检查 `zing_adaln_table_path`，并在 sampler 检查 transformer.adaln_table，混合模式遇到离线单表即拒绝。最新 30 秒实测设置 `SGLANG_ZING_ADALN_PROJECTION_CACHE=1`，三组 ON/OFF 均成功，动态投影缓存有命中且没有驱逐或旁路；这是动态 cache 兼容证据，不是静态 AdaLN 双表支持或 cache 开关数值 A/B。没有实现离线 AdaLN 双表。普通启动校验不会自动证明 checkpoint 的两步画质；本次另行核对原生模型与真实推理。

**Joint：** 实时普通块和 source-ready 分别通过实时 `step()` 绑定或显式传入选定 schedule。原 `_sample_cross_chunk` 完全还原，无 PR diff。原 `_ready_cross_prediction` 也保持不变：策略校验强制两表都以1000起始，相同模态 shift 下首个 timestep 相同，因此 source-ready lookahead 可复用原实现。前一块 clean commit 的 final timestep 不随下一块 schedule 改变。此前 CPU 回归包含 joint 路径；原 #301 代码隔离后，真实小模型与 source-ready cross-chunk 回归已完成两轮，每轮10项通过；本次默认配置修正的回归另见第7节。

## 6. 校验与可观测性

启动参数接受逗号分隔整数；拒绝空表、非整数、不从1000起始、非严格递减、含0或越界的列表。初始块数只允许0或3；count=0不能带初始表；count=3必须为 few_step 和既定四步/两步组合。未启用混合时保留原Euler和统一schedule路径，不把所有旧请求强制成两步。

仅实时 sampler 的实际采样块记录 target_block、latent_frames、denoise_steps、noise_grid；有限实时生成保存在 `last_run_metrics.block_schedules`，最近完成块另存 `last_block_schedule`。source-ready 实验记录器从后者采集，并加上generation块游标和时间戳。记录的是实际选用的表，不仅是启动参数回显。

仅实时 metrics 字典增加字段；原离线 metrics 和对应旧测试整文件均还原。有限实时生成的逐块列表随块数增长，尚未专门测量长程记录开销。

## 7. 验证方案、实际结果与缺口

### 当前配置修正的验证

需要保护的行为是：最小 ON 命令解析为既有四步/两步组合；部分显式配置只补缺项；冲突配置和静态 AdaLN 单表仍拒绝；OFF 继续允许请求级默认、Euler 或显式的其他 few-step 表；不同实例不共享默认列表。回归集中在 `test_zing_initial_blocks.py`，同时保留真实小模型、generation 重置、source-ready cross-chunk、离线 factory 隔离和异常恢复测试。

验证入口为固定 Linux Python 3.11 CPU 配方，两轮运行该完整测试文件，再核对 receipt、JUnit 和测试源码哈希更新 CI 清单。已完成两轮，每轮 22 项通过、零失败、零跳过（作业 `dlc9xyf4bveogqwz`）；实际结果在本次 PR 描述与 `scripts/ci/zing/cpu_test_manifest.json` 的对应项记录。这里只修改启动参数归一化，没有在逐块热路径增加计算；不重跑 GPU 性能 benchmark，也不将下述已有实验写成本次新代码的 GPU 结果。

### 已有 30 秒 GPU 实测（#301）

[完整报告：正常播放、独立慢放、逐块记录和显存数据](https://review-cdn.loopit.com.cn/world-model/sglang/evals/issue297-20261013/long30-r1/review-web-r1/index.html)。该 CDN 有 IP 白名单；当前本机访问返回 403，需要可访问该 CDN 的办公网络或 VPN。云端发布文件 SHA 回读与所有视频完整解码已通过；不将此表述为当前本机浏览器播放通过。

三份原始 30 秒 source 完整输入，无 target prefix，三例各 ON/OFF，共 6 条 720 帧 / 24fps / 864×480 输出。每条 86 个正式 target block，共 516 块；实际初始 latent 分块为 `[1,2,2]`，OFF 全两步，ON 前三块四步随后两步，已逐块核对。相比历史 5 秒实验，本次源码已经包含实时/离线隔离。

- 源码：`b7014fab0be976eb6e0099b9608663069a318cff`；归档 SHA256 `f408845e6c6d391985810d01c4629e28b001812b6d9afcd5028ff35cd739e486`；GPU 作业 `dlcgez2mjqiwobyl`。
- checkpoint step1600，seed42，BF16 DiT，SP2/Ulysses2，H3 FA4，TAEH3 FP16，audio off。两组各做一次完整 720 帧预热，随后各 case 一次正式测量，先 OFF 后 ON。
- DiT、文本编码器及 VAE 的 CPU/layerwise offload 均关闭；`SGLANG_ZING_ADALN_PROJECTION_CACHE=1`。历史 5 秒实验的 offload 来自当时沿用的配置，不是 OOM 后临时开启。
- 实际设备报告 NVIDIA L20D，compute capability 10.3，每卡 275040 MiB。2 秒间隔显存采样峰值 273898 MiB（267.48 GiB），余量仅 1142 MiB（1.12 GiB）；无 OOM，但不能称为显存余量充足，更不能据此声称 H200 已验证。
- 每例动态 cache：OFF hits=17391、misses=204、bytes=80747520；ON hits=17595、misses=306、bytes=113046528；没有 evictions/bypasses。

| Case | 前17帧 OFF / ON（秒） | 前三块采样墙钟 OFF / ON（秒） | 720帧端到端 OFF / ON（秒） |
|---|---:|---:|---:|
| 废土拾荒 | 35.662 / 36.418 | 1.831 / 2.300 | 65.789 / 66.017 |
| 地中海小镇 | 37.857 / 37.698 | 1.885 / 2.317 | 69.238 / 67.313 |
| 潇潇／林姐 | 51.182 / 53.032 | 1.368 / 2.597 | 91.502 / 92.287 |

前三块采样墙钟记录没有额外 CUDA synchronize，不等同于纯 kernel 耗时。前17帧和端到端还包含 source/条件编码、解码、传输等；ON 多六次目标去噪迭代不能推导端到端翻倍。单次测量有波动，例如小镇的端到端 ON 更短，不应解释为 ON 更快。两组连续 RNG 的消耗不同，同 seed 也不保证后续噪声相同。画质收益由正常播放及独立慢放人工判断。

### 历史证据与仍未覆盖的范围

[四例 5 秒报告](https://review-cdn.loopit.com.cn/world-model/sglang/evals/issue297-20261013/review-web-r1/index-v2.html#cases) 保留海岸、废土、小镇、潇潇各 120 帧结果；该轮存在文本编码器/VAE offload 且投影 cache 关闭，不与上表混作同配置重复测量。

实现提交 `103ffad92ab0cb1a8ae7f1114e07f9f62b687f3d` 的 [Zing CPU CI](https://github.com/seedleap/sglang/actions/runs/38398289455) 13 项全部通过；当时 initial_blocks 文件两轮各 10 项通过，无跳过。该证据对应原实现，本次默认值补充的批准证据单独更新。

本轮未验收 H200、cache 开/关数值 A/B、混合模式取消后重新生成的 GPU 专项、持续 source 中途更新真实 prompt、其他 chunk plan 的五 latent 语义或真实音视频。30 秒单次生成也不证明长时间运行稳定性和统计性能收益。离线推理和离线 AdaLN 双表不在本功能范围。

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

原实现已合入；当前 review 重点与最新验证见[静态表方案](initial-block-static-adaln.md)和[默认配置方案](default-mixed-sampling.md)。第7节保留原实现的实验记录，不能当作最新 main 整合提交的 GPU 验收。离线推理不属于本次改动范围。
