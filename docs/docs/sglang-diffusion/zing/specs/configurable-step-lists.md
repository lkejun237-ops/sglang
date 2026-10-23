---
title: "Zing 实时自定义去噪列表"
description: "覆盖 initial/regular step list，保持默认混合策略与静态表精确匹配。"
---

# Zing 实时自定义去噪列表

此前 CLI 已暴露两套 step list，但校验强制 initial 为 `[1000,750,500,250]`、regular 为 `[1000,750]`，连续 source 还限制 regular 只能有2或4项。本改动允许部署者配置合法列表，当前默认为前三块四步、后续两步 `[1000,500]`。`initial_block_count` 仍只支持0/3，不扩展 Euler 混合计划、客户端采样控制或离线推理路径。

## 使用方式

在已有可运行的 Linux/CUDA Zing 部署上，保留其模型、并行和组件配置，追加以下采样参数即可选择前三块3步、后续2步：

```bash
--zing-initial-denoising-step-list 1000,800,600 \
--zing-denoising-step-list 1000,500
```

这是启动命令的参数片段；`--model-path`、设备/并行等参数沿用现有部署。默认 count=3 时，solver 缺省补为 few_step，两套列表只补未提供的字段。合法列表非空、由整数构成、首项1000、严格递减且不含0；支持单步 `[1000]`。不要求 initial 比 regular 长，也不要求一个列表包含另一个。这里验证执行契约，不保证每个列表都适合 checkpoint 的生成质量。

连续 source 的 profile 也可声明完整计划：

```json
{
  "solver": "few_step",
  "initial_block_count": 3,
  "initial_denoising_step_list": [1000, 800, 600],
  "denoising_step_list": [1000, 500]
}
```

这些字段补入原有有效 profile，保留模型哈希、noise_version 等必需字段。优先级沿用原实现：显式采样 CLI 的整套计划优先；否则使用明确声明 initial_block_count 的 profile；旧 profile 未声明该字段时采用默认混合计划。不是逐字段从 profile 和 CLI 拼接。profile 本身仍先做合法性校验；显式 CLI 不会掩盖损坏的 profile。单阶段 profile 显式设置 count=0，且不提供 initial 列表。

客户端继续省略 num_inference_steps；若显式提供，须等于 regular 列表长度。ready 的 sampling_schedule 和内部 worker 命令保持完整实际计划，num_inference_steps 回显 regular 长度。仓库的连续 source 测试客户端也接受任意正整数步数，不再仅接受2/4；零、负数、布尔值和其他非整数仍在发送 source 帧前拒绝。

## 设计与兼容性

按配置、服务入口和计算分工，保留现有模块，不新增计划类、注册机制或全局可变默认值：

| 层次 | 职责与输入/输出 | 状态归属 |
| --- | --- | --- |
| `configs/zing_sampling.py` | `parse_noise_grid` 校验并复制列表；`validate_initial_blocks` 检查两阶段关系；`normalize_source_sampling_schedule` 产出完整计划 | 无共享可变状态 |
| `ServerArgs._validate_zing_sampling` | 解析CLI/配置，只补未提供的默认字段，再调用共用校验 | 部署参数 |
| source adapter `_source_sampling_schedule/on_init` | 按CLI/profile优先级选完整计划；生成ready和内部worker命令 | 单个会话持有计划副本 |
| worker `_build_realtime_sampler` 与静态表准入 | 再验证进程边界传入的完整计划，构建scheduler并精确匹配表 | sampler/会话持有scheduler和表租约 |
| 连续 source 测试客户端 `exchange` | 检查ready的步数是正整数；按协议发送source和接收输出 | 不参与选择采样计划 |

列表合法性集中到配置层，入口不再维护2/4步白名单；客户端只检查协议字段类型和值域，不复制服务端的采样策略。保留入口和worker两次验证，是为了分别保护外部配置和进程边界，校验不进入逐步去噪热路径。

静态表继续匹配完整计划，而非只检查“表中是否有这些timestep”：相同投影集合也可能对应不同顺序、sigma或切换位置。放宽这层检查会混淆“可查到投影”与“执行了指定计划”；本PR只移除固定列表限制，不改变该兼容契约。

底层 `ZingH3CausalRenoiseSchedule` 原本按实际列表构建 sigma，采样循环按实际长度执行；joint forward 原本按实际 timestep 查 singleton，lookahead 绑定下一块的计划。本改动不修改这些计算路径，也不改变给定计划的噪声生成顺序。

静态表构建器复用同一个校验，因此同时接受新列表。表格式和 `initial_blocks_plan` 不变：继续记录两套完整 order/sigma/count，并按实际 batch 去重。部署时必须提供匹配新计划的表；即使旧表含有所需投影，计划不同仍在 KV 写入前拒绝。构建工具的 `--denoising-step-list` / `--initial-denoising-step-list` 使用空格分隔整数，服务 CLI 使用逗号。构建流程及算术/硬件限制沿用[静态表设计](/docs/sglang-diffusion/zing/specs/initial-block-static-adaln)，静态表仍为 SP1/TP1。

## 验证方法与通过标准

回归扩展现有采样、source adapter 和连续 source 客户端测试，不新增测试框架或模型夹具：

- CLI 只覆盖任一列表时，另一个仍补默认值；非法值继续拒绝。
- profile 单阶段1/3/5步、CLI混合后段5步经过ready和内部命令传递，实际步数取列表长度；显式request步数冲突仍拒绝。
- profile 与 CLI 两种入口、source_first/cross_chunk 两种 joint 模式，比较动态 cache 和静态表的五块小模型 latent、KV、会话 RNG；覆盖默认列表、同长度不同数值、initial3步/regular1步、initial1步/regular3步，核对第3到4块切换和实际 cross_chunk 次数。
- 同一表对应的不同合法计划仍拒绝；静态表构建 CLI 覆盖自定义列表。既有默认、取消释放和离线隔离回归保留。
- 连续 source 客户端通过真实 WebSocket 协议对端验证1/2/3/4/5步下的帧交付、背压和EOS，并拒绝非法步数。真实4→1验证发现旧客户端在收到ready后拒绝1步，已补齐这处客户端准入；修复后已完成下述真实 GPU 对照。

实测：实现提交 `8e4a3dedd19e` 的 [Linux CPU CI](https://github.com/seedleap/sglang/actions/runs/38883971701) 中，混合采样文件112项、source adapter文件116项，均连续两轮零失败、零跳过；上述16组动态/静态小模型比较全部通过。静态表导出、实时契约及 lint 分片也通过。本地28项配置探针、inventory 和 diff 检查通过。首次 CI 另一个分片的既有 SP4 并发 KV 测试第一轮触发 `Zing model changed within a cached session`，同次第二轮通过，未改该测试或放宽断言；后续受测提交的完整CI已全部通过，见下文。

## 已执行的真实 GPU：前三块四步、后续单步

2026-10-16 验证运行时代码 `7b8e36fc59ebad62df8c5c0c0bbd8dec36eb2f6f`。同一两卡作业内各用一张 GPU，动态 projection cache 与匹配计划的静态表各完成一次完整 WebSocket 连续 source 请求：64帧、864×480、seed=42、相同6张参考图和 source、TP1/SP1、cross_chunk、window=21、sink=3。硬件实际枚举报为两张 NVIDIA L20D（每卡约268 GiB，SM10.3），模型 timestep shift=8；DiT/VAE常驻，text encoder CPU offload。显式参数为：

```bash
--zing-solver few_step --zing-initial-block-count 3 \
--zing-initial-denoising-step-list 1000,750,500,250 \
--zing-denoising-step-list 1000
```

- 两边实际9块的去噪步数均为 `[4,4,4,1,1,1,1,1,1]`，ready回显1步及完整两阶段计划，与worker一致。
- 最终latent按原始字节逐位相同，最大绝对误差0；capture内条件、输入噪声、各块输入/输出及最终会话RNG一致，64帧输出RGB哈希相同。
- 两边cross_chunk实际执行3次、source_first执行6次；静态表走weights_free模式，投影权重存储为0，无miss/fallback；动态cache有真实命中。两边收到/消费64帧并完成会话释放。
- 单次验证请求耗时：动态97.34秒、静态67.05秒；PyTorch峰值allocated分别191.27/150.94 GiB，reserved分别197.04/248.87 GiB。这是带只读capture的首次请求，含条件编码、编译、输出与记录开销，仅作本次运行参考，不作为稳态吞吐或加速比结论；没有增加性能专测。

最终latent SHA256为 `076d98d11de336bdce076cf6c068ef6c0371a6bd5ec8025e5a47902c6d3f9ba4`，输出RGB SHA256为 `cc80ed04208bf0511697ddce379e7fdad8428851cfef449a32116e3b714df482`。成功GPU作业为 `dlc1vdterviptpqi`，冻结结果位于 `/cpfs/world-model/sglang/static-adaln-mixed/20261016-four-one-r6/RESULT.json`，各lane保留配置、capture、最终latent和释放记录。镜像digest为 `sha256:4b9bca7dd61784cd84e43ac760db4c458c08a2d6bf22887e3740455fffe9e015`。

### 历史失败与证据边界

尝试记录：首次GPU请求被旧客户端的2/4步白名单拒绝，已修复并新增22项WebSocket用例；另一轮在去噪前被测试观测器的旧initialize签名阻断，已修复trace透传，并在CPU逐一执行三个观测包装器后重跑。CPU准备阶段的旧模型路径、CRLF补丁和脚本模板错误均保留失败记录，未作为通过样本；没有发生OOM。成功轮每种cache各完成一次推理。此验证只覆盖该真实checkpoint的4→1、cross_chunk、SP1/TP1执行正确性，不代表任意列表的画质验收或真实GPU上source_first模式的新增覆盖。

受测提交的[完整Linux CPU CI](https://github.com/seedleap/sglang/actions/runs/38889667160)13个job全部通过，包含此前失败的SP4分片及客户端回归；本地22项客户端针对性测试也通过。本轮整理只修改测试和设计文档，生产运行时代码仍与该受测提交一致，因此没有重跑GPU。新增5步客户端用例后的本地针对性测试26项通过；新增的profile/CLI五步回归及静态表去重断言由本次PR CI执行，不能用旧CI结果代表新测试已通过。

## Reviewer 阅读入口

先核对上述用法和职责，再沿以下路径阅读：

| 文件与入口 | 需要确认的问题 |
| --- | --- |
| `configs/zing_sampling.py`：`parse_noise_grid/validate_initial_blocks/normalize_source_sampling_schedule` | 任意合法长度是否都能通过，非法类型、顺序和阶段组合是否仍拒绝？ |
| `runtime/server_args/server_args.py`：`_validate_zing_sampling` | 缺省补齐是否保留显式列表？ |
| `runtime/entrypoints/openai/realtime/adapters/zing_h3_source_adapter.py`：`_source_sampling_schedule/on_init/prepare_next_request` | CLI是否整套覆盖profile，ready和发送到worker的计划是否一致且互不被后续修改？ |
| `test/unit/test_zing_initial_blocks.py`：`test_source_custom_plan_matches_dynamic_cache` | 命名用例`default-4-to-2/custom-4-to-2/3-to-1/1-to-3`是否实际跨过第三块，并检查latent/KV/RNG、joint调用和错误计划拒绝？ |
| `test/unit/realtime/test_zing_h3_source_adapter.py`、`benchmark/zing_h3_causal/tests/test_bench_client.py` | 5步能否通过入口及真实WebSocket客户端，非法步数是否在发送source前拒绝？ |

前三个路径相对于 `python/sglang/multimodal_gen/`，两条`test/unit/`路径也在该目录内。静态表夹具同时检查batch集合与数量：集合正确但重复存储的情况也必须失败。默认策略、离线隔离、取消释放和完整计划匹配的既有回归保留。

## 社区惯例与合入边界

本方案基于 seedleap main `0dafc687b02cc6e63c1078d91116fc89e04fad7d`。已核对仓库贡献指南和邻近参数解析/测试约定，并查看[上游 SamplingParams](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/configs/sample/sampling_params.py)的配置层位置。Zing 的两阶段计划是本 fork 扩展，不声称上游具有同一接口。社区审阅重点是默认行为兼容、合法性与生成质量的边界、配置是否集中校验，以及测试是否证明自定义计划而非只检查参数解析。
