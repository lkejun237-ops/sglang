---
title: "Zing 实时默认混合采样"
description: "默认前三块四步、后续两步，显式关闭与部署迁移。"
---

# Zing 实时默认混合采样

本文说明 PR308 中的默认采样行为：前三个 target block 四步、后续两步。该配置最初由 PR323 实现，现已纳入直接面向 main 的 PR308，不依赖 PR305 合入。默认值部分不修改模型、attention、采样算法或离线推理工厂；静态表适配与 GPU 验证见[静态表设计](/docs/sglang-diffusion/zing/specs/initial-block-static-adaln)。

## 使用与迁移

已有可运行的 Zing 模型、设备与外围组件配置保持原样；默认离线 AdaLN 模式从本地模型目录的 `adaln-table/` 加载匹配表，省略表路径即可使用。可在 pipeline JSON 中设置 `zing_adaln_table_path` 覆盖路径，或显式 `zing_adaln_mode="resident"` 选择常驻投影。表须在部署前生成，缺失或不匹配时仍会报错。采样默认值无需附加采样参数：

```bash
sglang serve --model-path /path/to/model --pipeline-config-path /path/to/pipeline.json
```

实时有效计划是 `few_step`，initial grid `[1000,750,500,250]`，initial block count `3`，regular grid `[1000,750]`。块的边界与 generation 重置规则沿用原实现。

关闭初始阶段：

```bash
sglang serve --model-path /path/to/model --pipeline-config-path /path/to/pipeline.json --zing-initial-block-count 0
```

有限实时入口此时恢复请求的 solver/grid；连续 source 此时保留 profile 原有 regular grid。要明确全程两步，关闭时再传 `--zing-solver few_step --zing-denoising-step-list 1000,750`；全程四步则传 `1000,750,500,250`。显式 Euler 必须配合 count=0，且连续 source 仍要求 few_step。两套 grid 均可显式覆盖默认值，见[自定义去噪列表](/docs/sglang-diffusion/zing/specs/configurable-step-lists)；未传入的 grid 才补默认值。

连续 source 仍需有效的 checkpoint profile，但不需要为开启混合采样添加字段：

| 条件 | 有效计划 |
| --- | --- |
| 无采样 CLI，profile 没有 initial_block_count | 默认混合计划，替代旧 profile 的单阶段 regular grid |
| 无采样 CLI，profile 显式 initial_block_count=0 | profile 单阶段计划 |
| 无采样 CLI，profile 声明完整 initial_block_count=3 计划 | profile 混合计划 |
| 显式采样 CLI | CLI 整套计划优先；仅 count=0 时沿用 profile regular grid |

两种入口都复用原有 ServerArgs 显式参数标记；Python 调用者通过 `ServerArgs.from_kwargs` / `from_dict` 保留显式配置来源。profile 自身先校验，默认值或 CLI 不掩盖坏配置。客户端省略 `num_inference_steps` 即采用有效计划；混合模式 ready 返回常规步数2和完整 sampling_schedule。写死4的旧客户端需要升级，或部署者显式关闭混合策略。

静态 AdaLN 仍须预先提供匹配两套 schedule 的表。旧四步单计划表会在准入时拒绝，不能因投影条目看似齐全就继续使用；构建命令和 shift/checkpoint 约束见 PR308 文档。本 PR 不自动重建表，不改变离线表工具的显式计划参数。动态 cache 不需要预计算资产。

## 边界与决策

- `ServerArgs` 的 dataclass 和 argparse 默认 count 同为3，复用 `_validate_zing_sampling` 补齐 solver/grid。显式0仍保留原 OFF 分支，避免新增开关。
- source adapter 区分“默认补齐后的 CLI 值”和“显式 CLI”。否则默认 solver=few_step 会覆盖 profile 的显式关闭。缺少 initial 字段的旧 profile 采用新默认，显式0则作为迁移出口。
- adapter 冻结计划后通过内部命令传给 worker。默认值变更复用本 PR 的 stage 与 realtime sampler 计划接口，不再增加一套执行路径，也不增加每一步的计算、同步和内存分配。
- 有意改变旧部署的默认画质/计算预算：前三块四步，后续两步。初始三块相较全两步多6次去噪；长视频相较全四步减少后续计算。这里是步数推导，不是新性能实测。

## 验证

验收不变量：默认 CLI 与显式 ON 解析相同；source 缺省 profile 的 ready、内部命令和小模型实际步数一致；显式关闭和 profile/CLI 优先级可预测；错误配置继续拒绝；离线入口不读实时部署覆盖。

测试在现有 `test_zing_initial_blocks.py` 和 `realtime/test_zing_h3_source_adapter.py` 中扩展：默认 CLI、九种 profile/CLI 开关组合、缺省 profile 经实际内部命令和 worker 执行 `4/4/4/2/2`，并保留原静态表、joint、生命周期及离线隔离回归。既有四步夹具明确写 count=0，使原用例继续测试其原有行为。

本地通过16项配置探针（从源码提取实际 argparse 语句、校验和 source 解析函数，覆盖九种优先级组合、关闭及非法配置）、Ruff/isort/格式检查、inventory 和 diff 检查。探针不等同于完整 ServerArgs 或模型测试；本机无 Torch。运行时代码提交 `ae5937d0e19d132b38ae282b63020fe60ccaf011` 的完整 Linux CPU CI [38759128467](https://github.com/seedleap/sglang/actions/runs/38759128467) 已通过。

2026-10-16 在同一提交上完成一次真实连续 source WebSocket 对照，作业 `dlcd5j1u30td8mcm` 为 Succeeded。两张 GPU 各跑一组：A 使用缺少 initial 字段的旧四步 profile，不传采样 CLI；B 使用同一 profile，显式传入混合采样 CLI。固定模型、静态混合 AdaLN 表、source/reference/prompt、seed=42、确定性视觉编码、shift/window 和 offload 设置；每组输出64帧、864×480、24 FPS，joint 模式为 `cross_chunk`。两组 DiT/VAE 常驻 GPU，文本编码器 CPU offload。

两组 ready/worker 计划一致，实际 target 步数均为 `4,4,4,2,2,2,2,2,2`；最终 latent 逐位相同（max_abs=0），捕获的输入/输出张量与输出 RGB 字节均相同。每组实际命中 cross_chunk 3次。设备报告 NVIDIA L20D（SM 10.3），Torch 2.13.0+cu130；未发生 OOM，两组单卡峰值 allocated 均为150.94 GiB。会话耗时默认93.915秒、显式93.930秒，包含首次编译和诊断捕获，仅作本次正确性验证的附带数据，不代表稳态性能；没有追加性能专测。

原始收据位于 CPFS `/cpfs/world-model/sglang/static-adaln-mixed/20261016-default-cli-r1/RESULT.json`，同目录保留配置、捕获张量和比较结果。最终 latent SHA256 为 `f6fec54e918474545d76357cddf43bdb5d2972288549317dce1ac981495e48b9`，RGB SHA256 为 `4e78eaf89a09232d943442239710172ae3eb06e9382453e38c7d8f9f9fff7377`。服务会话清理确认释放 scheduler、删除输入临时文件；进程退出各有一条 multiprocessing semaphore 清理警告，作业仍成功结束。本次覆盖默认值与显式 CLI 的一个真实输入、一个 joint 模式，未新增其他模式、长视频或显式关闭的 GPU 矩阵。

## 社区实践与 review 导读

已核对基线 `4e25af693fa7f705d11af92daf79e56c4d0b74f6` 的贡献指南、ServerArgs 参数来源机制及 Zing CPU CI。沿用当前 fork 的参数解析、完整计划校验和现有测试目录，不新增通用调度抽象。Zing 混合策略是 seedleap fork 的功能，不宣称上游社区已采用这种默认值；贡献规范见[官方贡献指南](https://github.com/sgl-project/sglang/blob/main/docs/docs/developer_guide/contribution_guide.mdx)。

推荐依次阅读：`ServerArgs` 字段/CLI help及 `_validate_zing_sampling` → `_source_sampling_schedule` → 两份回归测试。review 重点是默认值是否误判为显式覆盖、profile 显式0是否有效、旧客户端/单计划表迁移是否清楚，以及累计 diff 是否保持离线和模型计算代码不变。
