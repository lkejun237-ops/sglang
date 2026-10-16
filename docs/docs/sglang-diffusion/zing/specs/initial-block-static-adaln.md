# 实时初始三块混合采样的静态 AdaLN 表

PR308 直接相对 main 提供默认开启的实时混合采样及静态 AdaLN 适配，不依赖 PR305。前三个 target block 使用四步、之后使用两步；静态表省去常驻 AdaLN projection 权重。动态 projection cache 会在线计算未命中项；静态表来自离线预计算，在线没有权重和 miss fallback。原有离线推理 sampler、stage 的离线入口及其测试文件保持不变。

## 使用与迁移

适用范围沿用静态表已有约束：SP1/TP1、纯视频、未量化且无 FSDP/component offload、固定 checkpoint 和算术环境。实时入口支持 `joint_forward=off/source_first/cross_chunk`；SP2 仍不在静态表支持范围内。

先生成包含两套计划的配置；以下 shift、condition、reference 必须与部署和 checkpoint 对齐，不能机械套用示例值：

```bash
PYTHONPATH=python python benchmark/zing_h3_causal/build_adaln_table.py plan \
  --solver few_step --denoising-step-list 1000 750 \
  --initial-block-count 3 --initial-denoising-step-list 1000 750 500 250 \
  --num-train-timesteps 1000 --video-shift 5 \
  --condition 0 --reference 0.999 --output mixed-plan.json

PYTHONPATH=python torchrun --standalone --nproc-per-node=1 \
  benchmark/zing_h3_causal/build_adaln_table.py ensure \
  --model-path "$EXPORT" --plan mixed-plan.json --device cuda
```

`ensure` 返回表目录，把已有 `--pipeline-config-path` JSON 中的 `zing_adaln_table_path` 指向该目录。实时采样默认 initial_block_count=3，自动补齐 few_step、常规 1000,750 和初始 1000,750,500,250，无需附加采样 CLI。显式 `--zing-initial-block-count 0` 关闭初始阶段；完整覆盖规则见[默认混合采样](default-mixed-sampling.md)。表路径的原有配置入口不变，构建静态表仍须显式提供上面的完整计划。

有限实时入口沿用内部 `cross_chunk` 默认值；本PR不增加请求方对 `zing_joint_forward` 的控制。模式选择保留在内部sampler和source-ready部署profile的 `joint_forward` 中。有限请求的 `step()` 在 streaming boundary 回退为 source_first，真实跨块合并需 source-ready `drain_ready()` 中下一块已经就绪。未修改 joint 默认值；本节讨论的 joint 选择保持原有部署方式，混合 schedule 的 profile 扩展见下节。原有CFG、source就绪、窗口淘汰及KV scratch容量等joint条件仍然生效；不满足时按原逻辑回退。SP1的joint attention分别调用各分支attention，不表示命中限定SP2形状的融合kernel。

旧单计划表不能直接用于混合采样：即使四步表恰好包含全部投影，也没有记录两步计划的 sigma 和切换位置。本改动要求重新生成匹配的表，保留旧目录；`ensure` 不覆盖已有不匹配目录。关闭 initial blocks 后仍使用原来的单计划表。没有重训要求。

## 连续 source 入口的采样契约

profile 可以直接声明完整混合计划：`solver="few_step"`、`denoising_step_list=[1000,750]`、`initial_block_count=3`、`initial_denoising_step_list=[1000,750,500,250]`。保留原有 checkpoint hash、guidance 和 noise 字段；不提供采样 CLI 覆盖时，入口和 worker 都使用这份计划。旧 profile 未声明 initial_block_count 时采用默认混合计划；显式写 initial_block_count=0 则保留 profile 的单阶段 grid。

显式启动 solver/grid 或指定 initial 时，CLI 的整套计划优先，不把两处的初始/常规 grid 拼成第三种计划；显式 `--zing-initial-block-count 0` 且未指定 solver/grid 时，保留 profile 的常规 grid 并取消 initial 阶段。用 ServerArgs 的显式参数标记区分自动补齐的默认计划和用户配置，默认值不会覆盖 profile 的显式关闭。profile 自身和最终计划都必须合法，覆盖不掩盖坏配置。

入口在会话初始化时冻结有效计划，经内部 `zing_source_command.sampling_schedule` 传给 worker；source state 将其显式交给实时 sampler 工厂，工厂复用同一配置校验后构造 sampler，不修改共享 stage 的部署状态。只增加内部 source 命令字段，不增加客户端 request 控制。stage 文件仅改 `_build_realtime_sampler`；离线工厂、离线 forward 与 sampler 不改。验证计划补充 profile-only 混合、旧 profile、CLI 覆盖与显式关闭、非法 initial 配置、命令序列化到 worker，以及 profile 混合静态表对照。

`num_inference_steps` 定义为常规阶段步数。请求可省略，显式填写时必须与有效计划一致。`session.ready` 返回该步数和 `sampling_schedule`（solver、常规 grid、initial block 数量、initial grid），混合计划返回 2 与前三块四步的完整描述。内部 sampling params 使用同一有效常规 grid；worker 从内部 source 命令接收完整有效计划。profile 的 checkpoint hash、noise、guidance、window 等约束保持有效，CLI 不绕过这些检查。

客户端迁移：混合部署不再接受写死 `num_inference_steps=4` 的请求，请省略此字段或填写2；全程四步部署仍接受4。仓库的 `continuous_source_client.py` 与 lifecycle probe 已改为省略该字段，客户端接受两步/四步 ready，并在原有 `request.json` / 事件记录中保留完整协商结果。混合采样的 ready 片段为：

```json
{"num_inference_steps":2,"sampling_schedule":{"solver":"few_step","denoising_step_list":[1000,750],"initial_block_count":3,"initial_denoising_step_list":[1000,750,500,250]}}
```

改动前，入口要求四步并固定返回4，即使命令行已让 worker 执行混合计划；这是入口契约不一致，并非此前静态表推理错误。此次让 profile 也能表达完整计划，入口解析 CLI 优先级后传给实时 worker，模型和 sampler 数值路径不变。验证先完成CPU入口到worker回归，随后补齐下文64帧完整WebSocket GPU对照；无额外性能专测。

本轮 profile 完整计划 CPU 实测（2026-10-15）：两次新 CPU pytest 进程各 **912项通过，零失败/跳过**（source adapter 103、initial blocks 91、progressive source session 663、原模型回归 55）。profile-only 混合计划和 CLI 覆盖经真实 adapter、MessagePack 内部命令、source state 与实时 worker 工厂，实际小模型逐块执行与 ready 一致；profile 混合静态表在 source_first/cross_chunk 下与常驻投影逐块 torch.equal，实际步数4/4/4/2/2，错配静态表拒绝且无残留 lease。通用请求工厂与媒体编码边界被隔离，不代表完整 WebSocket 媒体链路验收。Python3.11.12/Torch2.8.0+cpu/PyAV18.1.0；作业 `dlc1pewikr6a9f9j`，[原始回执](https://review-cdn.loopit.com.cn/world-model/sglang/evals/issue297-20261013/pr308-profile-mixed-r2/result.json)、JUnit 与日志同目录；清单绑定源码哈希。该 CPU 阶段未运行 GPU；后续完整服务对照见下文。

本轮首次运行 `dlc1mmz66jsgsxdy` 为906通过/6失败：新增贯通测试未提交 source 首帧就初始化 worker，触发预览读取错误。测试补齐真实 `ingest_wire_event` 输入后完成上述两轮通过，未更改运行时代码或放宽断言。

前一轮仅 CLI 计划适配结果（2026-10-15，`d432599cd9`）：Python 3.11.12 / Torch 2.8.0+cpu，独立补充 PyAV 18.1.0，两次新 pytest 进程各 **174项通过，零失败/跳过**（source adapter 88项、initial blocks 86项）。三组贯通测试实际调用未修改的 worker factory 和真实小模型，逐块执行全四步、全两步及 `4/4/4/2/2`，与 ready 一致；只替换通用 API 参数工厂以隔离服务资源，source adapter 的组装与 sampler 均为真实实现。作业 `dlc1jkwm4rm10dcy`，[原始回执](https://review-cdn.loopit.com.cn/world-model/sglang/evals/issue297-20261013/pr308-source-profile-r4/result.json)、JUnit 与日志同目录，源码哈希已更新 CPU 清单。r1在源码准备时因CRLF补丁不匹配失败；r2实际173通过/1失败，失败为既有媒体测试缺少PyAV，补依赖后未修改断言完成r4（r3因PyAV19要求Python3.12而在安装阶段失败）。本轮不重跑GPU或完整WebSocket媒体链路；客户端改动仅去掉固定步数和放宽ready协商，做了语法与lint检查。

## 决策和资源归属

1. **一份表，两套完整计划。** `initial_blocks_plan` 保留常规 schedule 字段，增加 `initial_blocks: {count, schedule}`。两个 schedule 都包含有序 timestep、完整 sigma、solver、condition、reference。既有 schema 和单计划路径不变，新旧程序的精确比较会拒绝不匹配计划。
2. **保留实际投影的 batch 形状。** 两阶段的 prefix/source/active/final 批次取并集。普通前向按完整 unique-t batch 查表；joint 原实现逐个 unique timestep 做 embedding 和 projection，所以按对应 singleton 查表，再按原有索引映射回 token。cross_chunk 的干净 target 和 source 共用 `[1]`，下一 target 使用 `[t]`；不查询未执行的 `[1,t]` GEMM，也不从完整 batch 切片伪造 singleton。现有构建器已包含这些条目，不增加 schema 或投影副本，不在线补算。
3. **先准入再写 KV。** 实时专属准入检查实际 packed roles 和两阶段的所有目标 timestep；短于三个 block 的请求仍须持有完整计划。调用原表的 `prepare`，保留 checkpoint、算术环境、embedder、完整 batch 覆盖校验。
4. **采样绑定与配置身份分开。** 实时 sampler 保存常规 scheduler 的引用。即使 `step(inputs=...)` 暂时绑定四步 scheduler，prompt 更新校验的仍是固定的两阶段部署计划。
5. **逐块 source 会话负责 lease。** 每次准备 chunk 和 prompt 时，在申请池化 KV、prefix forward 之前准入；取得新 lease 后关闭旧 lease。正常结束、取消、池申请失败或 forward 失败均由 session 清理。表的设备存储仍由 worker 持有，不随 lease 关闭释放。

实时准入直接放在 `ZingH3RealtimeSampler._admit_adaln()`，因为它只有这个调用方，且依赖 sampler 的计划、角色映射和内部方法；单独拆文件会让读者在两个文件之间追踪同一职责。它沿用原 sampler 的角色语义；没有为了抽取公共函数而修改用户要求保持原样的离线 sampler。这带来少量有边界的枚举重复，测试需要同时保护两条路径。

## Joint 适配验证（2026-10-15）

模型仅在 joint + 静态表时用 `lookup(singleton)` 代替动态 cache 的 `begin()`；普通非 joint 路径的完整 batch 契约保持不变。实时 sampler 同时覆盖单计划与混合计划的 joint 准入，离线 sampler 仍保留原限制。实时 lookahead 临时绑定下一块的 scheduler，退出或异常时恢复，不依赖两个 grid 恰好有相同第一步。

实际结果：固定 Linux Python 3.11.12 / Torch 2.8.0+cpu 环境，`test_zing_initial_blocks.py` 两个新进程各 **83 项通过，零失败/跳过**（58.81 / 22.82 秒，作业 `dlc103wf7vrrv6jg` Succeeded）。同模式的常驻 projection 与静态表在真实小模型上 latent/KV 均 `torch.equal`。覆盖单计划/混合计划、off/source_first/cross_chunk、有限 step/source-ready step/drain、prompt 更新、缺 clean/active singleton、取消释放及 generation 重置。drain 实际产生 4 次 cross_chunk、1 次 source_first，五块步数 `4/4/4/2/2`，确实跨过第 3→4 块；测试移除未使用的组合 `[0,1]` 条目，仍能运行，证明不依赖不存在的组合 GEMM。

另有既有 joint 59、模型 55、chunks 64、静态表 29、导出 38 项，共245项通过。首轮作业 `dlcyzwn4bcado098` 总计316通过/12失败，失败均发生在新增前置断言把未排序 `[1,0]` 当成合法key；生产代码未改变，仅修正测试断言后完成上述两轮。运行时模型 timestep 在 `[0,1]`，noise grid 的1000对应第一步0，clean为1。

源码 LF 哈希、JUnit 和日志位于 `/cpfs/world-model/sglang/static-adaln-mixed/20261015-joint-r2/`，清单记录实际直接 pytest 入口，不冒称 canonical triage runner 通过。项目固定 Ruff0.15.1、isort7.0.0及inventory通过。该轮 stage 整文件未修改；后续 profile 适配只调整其实时工厂，离线部分、原离线 sampler 和模型测试仍与 PR 基线完全一致。

随后在实测提交 `f8be788c1c18cf6a44c7ded0300edded049dd017` 上完成真实 checkpoint GPU 验证：四个真实媒体 case 各64帧，分别运行 source_first/cross_chunk 的动态 projection cache 与静态表，共16次推理、8组对照。GPU作业 `dlcvltl7zcbx7u3e` Succeeded，**8/8组完整最终 latent 原始字节及SHA256一致，最大绝对误差0、差异元素0**；逐tensor输入哈希、采样计划和会话结束CPU/CUDA RNG也全部一致，无OOM。[报告、原始记录及复现脚本](https://review-cdn.loopit.com.cn/world-model/sglang/evals/issue297-20261013/pr308-joint-gpu-r1/index.html)。

本轮直接加载真实checkpoint，通过 `SourceReadySession.drain_ready()` 运行，复用r8实际媒体编码的conditioning/source latent、初始噪声和入口CPU/CUDA RNG，未重新编码、解码或启动WebSocket服务；验收对象是最终latent，不声称本轮验证了RGB。每例19 latent、9块，实际步数 `4/4/4/2/2/2/2/2/2`；source_first合并9次，cross_chunk合并8次加source_first1次，真实跨过第3→4块。结束RNG直接取session.rng后再dispose，避免历史全局RNG观察问题。

两卡各TP1/SP1，FA4，shift8、seed42、guidance1、window21/sink3、RoPE block_relative/max_gap12；无DiT offload、torch.compile或CUDA graph。设备报告NVIDIA L20D、SM10.3、267.69GiB/card，Torch2.13.0+cu130。静态表checkpoint校验通过，projection参数存储为0，miss/fallback=0，结束后active_plans=0；动态cache确有命中。

| 模式 / Case | 动态/静态峰值 allocated（GiB） | 动态/静态采样耗时（秒） |
| --- | --- | --- |
| source_first / 海岸 | 177.05 / 136.76 | 48.961 / 35.553 |
| source_first / 荒原 | 144.08 / 103.79 | 3.873 / 4.154 |
| source_first / 地中海小镇 | 145.40 / 105.10 | 3.765 / 3.730 |
| source_first / 潇潇/林姐 | 199.72 / 159.43 | 4.464 / 4.479 |
| cross_chunk / 海岸 | 177.29 / 137.00 | 4.597 / 4.608 |
| cross_chunk / 荒原 | 144.33 / 104.03 | 4.163 / 4.130 |
| cross_chunk / 地中海小镇 | 145.64 / 105.35 | 3.491 / 3.473 |
| cross_chunk / 潇潇/林姐 | 199.95 / 159.66 | 4.315 / 4.295 |

静态组每卡峰值约减少40.29GiB。指标仅来自上述正确性运行，没有额外预热、重复或性能专测；首例含kernel首次编译，两进程共享主机资源，不能据此判断稳定态速度收益。未加载VAE和文本编码器，因此不能直接与下方历史整服务峰值比较。完整源码归档、输入哈希、部署配置、两组日志及结果在 `/cpfs/world-model/sglang/static-adaln-mixed/20261015-joint-gpu-r1/`；后续文档补充提交不改变实测代码身份。

## profile 与显式 CLI 的完整服务 GPU 对照（2026-10-15）

实测提交 `3bd25683a4066fda433af4a0fb74ac2f09164b9c`：A 仅使用完整混合 profile，不传采样 CLI；B 使用原全程四步 profile，并显式传入 `--zing-solver few_step --zing-denoising-step-list 1000,750 --zing-initial-block-count 3 --zing-initial-denoising-step-list 1000,750,500,250`。两组均使用同一混合静态 AdaLN 表，分别测试 source_first/cross_chunk，共4次有效推理。

输入为同一真实海岸 case 的64帧 RGB（864×480、24 FPS）、相同参考图与 prompt。完整链路覆盖客户端 WebSocket、source adapter、内部命令、worker、条件/source编码、DiT、TAEH3和客户端RGB；采用17帧批次、不按实时帧率限速追加。每组TP1/SP1、FA4、shift8、seed42、guidance1、window21/sink3；DiT/VAE常驻GPU，仅文本编码器使用后卸载CPU。设备报告 NVIDIA L20D、SM10.3、267.69GiB；Torch2.13.0+cu130。

**两种模式均通过：同模式 A/B 的全部 conditioning/source/noise tensor、最终完整 latent 原始字节、会话结束 CPU/CUDA RNG 及输出 RGB SHA256 完全一致，latent 最大误差0。** 四次均正常完成并收到64帧，无OOM；ready 常规步数2、完整计划与worker一致，实际每块步数 `4/4/4/2/2/2/2/2/2`。静态projection存储0、miss/fallback0，结束active_plans0。实际joint计数两组相同：source_first模式合并9次；cross_chunk模式跨块合并3次、source_first合并6次。在线渐进编码的source就绪时机影响合并机会，不强行套用冻结队列的合并次数。

| 模式 | latent 误差 / RGB | 会话耗时 A / B（秒） | 进程峰值 allocated A / B（GiB） |
| --- | --- | --- | --- |
| source_first | 0，原始字节一致 | 94.11 / 93.99 | 150.94 / 150.94 |
| cross_chunk | 0，原始字节一致 | 29.59 / 29.78 | 150.94 / 150.94 |

上述通过结果来自两组都显式设置已有 `SGLANG_ZING_VISUAL_ENCODE_MODE=deterministic` 的两卡作业 `dlc1ltpo97pw028d`（Succeeded）。只使用这4次正确性运行的指标；启用了现有component capture和只读诊断包装，包含编码、首次编译、快照I/O及客户端接收，不能据此比较稳定态性能或推断默认performance编码的速度。未修改产品数值路径、离线推理或线上profile，未部署/合并。

首个两卡作业 `dlcfd8ruzqg8cvkk` 整体Failed：测试脚本把两服务HTTP端口设为32100/32101，与默认HTTP+1内部broker冲突，CLI服务在推理前启动失败。profile两模式结果保留；原作业结束后，单卡作业 `dlc1wn5mh7wlx0wx` 改CLI端口32110补齐两模式，四次均输出64帧且无OOM，但最终一致性断言Failed，latent最大误差2.667696/2.425148且RGB不同。CPU作业 `dlc1t1dpmzxx51yg` 定位source_first的差异已出现在DiT之前的参考图/source VAE编码：实际噪声、text embedding和结束RNG一致。默认performance编码的cuDNN benchmark不保证跨进程可复现，不能视为固定了实际模型输入。因此两组同设deterministic编码重新完成4次正确性对照，未放宽判据。首轮失败JSON和逐tensor诊断均保留在报告。此前两次CPU预检分别修正测试启动器解析入口和探针sys.argv，不是产品代码失败。

[报告、逐项结果、日志及复现脚本](https://review-cdn.loopit.com.cn/world-model/sglang/evals/issue297-20261013/pr308-profile-cli-gpu-r4/index.html)。原始tensor/RGB在 `/cpfs/world-model/sglang/static-adaln-mixed/20261015-profile-cli-gpu-r2/gpu-deterministic`，同级gpu目录为失败对照；报告保留源码/模型/媒体锁与确定性配置。本轮验证的是profile与CLI配置等价，动态/静态对照仍由前文原提交的8组GPU结果证明。仍未覆盖SP2、长视频、prompt更新/取消与稳定态性能。

## 验证方案与当前状态

需要保证：真实小模型在相同 RNG 下，静态混合计划与常驻 projection 的输出和逐层 KV 零容差一致，实际步数为 `4/4/4/2/2`，新 generation 重置，prompt 更新不改变计划身份；有限请求和 source-ready 的 step/drain 两入口都覆盖。

失败测试必须证明：旧单计划、切换 count/sigma 不匹配、缺完整批次、音频、缺失 clean/active singleton 在池申请及 prefix 写入前拒绝；取消及申请失败后 active leases 归零。构建器测试核对 CLI 生成与 runtime 合约一致，旧单计划构建和离线推理回归继续通过。

已在仓库固定 Linux Python 3.11.12 / Torch 2.8.0+cpu 环境完成：

| 测试文件 | 每轮通过数 | 两轮状态 |
| --- | ---: | --- |
| `test_zing_initial_blocks.py` | 41 | 通过，零失败/跳过 |
| `test_zing_h3_model.py`（原文件未改） | 55 | 通过，零失败/跳过 |
| `test_zing_h3_chunks.py` | 64 | 通过，零失败/跳过 |
| `test_zing_adaln_table.py` | 29 | 通过，零失败/跳过 |
| `test_zing_adaln_export.py` | 38 | 通过，零失败/跳过 |

合计 227 项，各执行两轮。新增真实模型比较使用 FP32 时间嵌入、BF16 projection、固定 RNG、CFG=2、参考图和 5 个 target blocks；有限请求及 source-ready step/drain 的 latent/KV 均 `torch.equal`，实际步数 `4/4/4/2/2`。projection 参数在 meta 构造时即不存在，公共批次不重复保存。CLI 的单计划与混合计划都和运行时合约一致。

原有四文件证据来自作业 `dlciapki30zbyqy8`，该作业整体失败：首次新增夹具的 condition/reference 未与模型配置对齐，且缺失 prefix 的测试误删了未使用的三值组合，新增文件 35 通过/6 失败。修正夹具后，作业 `dlcgwueq3fgddlk1` 两轮各 41 项通过，耗时 64.61/24.67 秒（测试进程耗时，不是推理性能）。生产代码未因这些夹具修正改变。

可复跑入口为 `scripts/ci/zing/run_cpu_tests.py --suite dit-pipeline-contracts --triage --file python/sglang/multimodal_gen/test/unit/test_zing_initial_blocks.py --trials 2 --output-dir <目录>`。最终源码哈希、固定环境配方、作业及 receipt/JUnit 路径已登记到 `scripts/ci/zing/cpu_test_manifest.json`。源码由 #305 的固定归档加本次 overlay 组成，LF 规范化后核验哈希；不是声称已跑过尚未生成的提交。

准入逻辑合并到 sampler 后，在同一固定环境重新执行该文件两轮，均 41 项通过、零失败/跳过（作业 `dlcxucnwrigobq9w`，69.33/27.31 秒）。源码归档与工作树、receipt 与 JUnit 均核验一致，证据目录 `/cpfs/world-model/sglang/static-adaln-mixed/20261014-inline-r1/dit/`。另外，语法树比对确认迁移仅改变接收者名称和导入位置；旧离线 sampler、stage 与原测试整文件无差异。未新增测试或改写既有断言。

2026-10-14 真实 checkpoint 验证：四个 case 各生成 64 帧（864×480、24fps），source 同步无损裁切。两卡各运行一个 TP1/SP1 实例，复用实际 conditioning、初始噪声及入口 CPU/CUDA RNG；前三块四步、后续两步，实际 shift=8、seed=42、guidance=1、window=21、sink=3、joint_forward=off。FA4、TAEH3 FP16，两组均仅将文本编码器卸载到 CPU。设备实际报告 NVIDIA L20D、SM10.3、每卡267.69 GiB，Torch 2.13.0+cu130。

**四例完整最终 latent 均 `torch.equal`，原始 RGB SHA256 均一致，8次均输出64帧，无OOM。** 每例实际9块，步数为 `4/4/4/2/2/2/2/2/2`；静态表 projection 参数不存在、存储为0，miss/fallback为0，结束后active plans为0。GPU作业 `dlc1q6zv9nexnpa3` 已Succeeded。测试源码为 `83705fca` 加实时adapter转交joint字段的修复（文件SHA256 `124585745b1a5e4cbe898018de57671b942a57a2a5f76a2b99ca896c9d50c9f3`）。[视频与原始数据报告](https://review-cdn.loopit.com.cn/world-model/sglang/evals/issue297-20261013/pr308-gpu-r9/review/index.html)。

| Case | 动态/静态峰值 allocated（GiB） | 动态/静态 sampler 会话耗时（秒） |
| --- | --- | --- |
| 海岸 | 187.40 / 147.07 | 41.12 / 8.59 |
| 荒原 | 154.49 / 114.16 | 4.50 / 3.70 |
| 地中海小镇 | 155.77 / 115.45 | 3.95 / 3.64 |
| 潇潇/林姐 | 209.75 / 169.43 | 4.77 / 4.21 |

上述指标来自同次正确性验证，没有额外性能运行、预热或重复。峰值在条件准备后重置，每卡约减少40.32 GiB；移除的projection权重为40.44 GiB，静态表为0.226 GiB。首例包含首次编译，会话计时还含解码间隙及采集开销，不能作为稳定态速度收益。

原始CPU验收脚本总标记仍为失败，唯一失败条件是结束时全局RNG不同：观察hook位于 `sampling.close()`，此时 `_sample_realtime_block` 已退出 `state.random_stream()` 的 `fork_rng`，恢复了各worker自己的全局RNG。它不是会话末尾RNG，不能用该比较判断采样随机序列不同。原始JSON保留失败标记，本轮仅确认入口固定与最终输出一致；没有采集会话末尾RNG，也不声称其验证通过。CPU验收作业 `dlc1qqz217z711fk` 因此为Failed，不能与GPU作业成功混称。

此前120帧SP1基线因KV翻倍扩容OOM；缩为64帧后，为运行当时仅支持off的静态表测试，曾临时增加实时adapter对请求中joint字段的透传。对应历史回归在添加前off/source_first两项失败，添加后GPU镜像中adapter与initial-block共70项通过；固定Python3.11.12/Torch2.8.0+cpu环境下adapter两轮各29项通过（`dlc7jao0bjvp16l0`）。第一次固定runner因归档缺少history_base提交在执行测试前失败，随后直接用固定解释器执行并保留JUnit，不冒称该runner已通过。

根据review意见，现已移除这项请求端控制及其专属测试，adapter和对应测试整文件恢复到PR基线。内部已有字段不等于请求方应控制执行优化；静态表已兼容joint，无需为了off测试扩展业务请求。上方最新8组GPU对照直接在sampler设置模式，不依赖adapter透传，相关模型/采样实现未改变，因此没有重跑GPU。历史r9 off端到端报告仍只对应其明确记录的测试提交，不能当作移除透传后的HTTP/WebSocket off请求验收。

动态/静态joint验证覆盖真实GPU的source-ready有限就绪队列与跨块合并；上文profile/CLI完整服务对照另覆盖64帧在线批次追加、条件/source编码与RGB。仍未覆盖SP2、prompt更新/取消、长视频与稳定态性能，不以以前的动态cache/SP2三段30秒结果替代。历史off视频画质由用户通过报告肉眼判断。

## 社区与 reviewer 导读

2026-10-16 整合基线为 main `18f5c1b29058820c3dffa449e2f233e97406f8b5`。保留 main 的窗口淘汰配置、source-ready 候选准备及多卡投票流程；静态准入复用已经准备好的 layout，使用 `model_chunk` 的局部 block 编号，混合阶段选择仍使用 generation 的全局 `ready.chunk.chunk_id`。准入发生在池申请和 prefix/KV 写入之前，失败沿既有会话清理路径释放 lease。静态表仍限定 SP1/TP1，本次不扩展为 SP2。

文档迁移保留在本 PR；旧的 `initial-block-schedule-defaults.md` 已移除，当前默认策略统一由 `default-mixed-sampling.md` 说明，不再依赖 PR305。上方 GPU 和固定 CPU 结果均绑定各自原提交，不能视作本次整合版本已重新运行 GPU。整合版本的检查结果与 CI 状态在 PR 描述中单独列出。

2026-10-16 对整合提交 `6d581b4e5d22c3d2174b43776f7859b37c042d19` 相对 main `18f5c1b29058820c3dffa449e2f233e97406f8b5` 完成代码审查，未发现已证实的阻塞缺陷。重点检查了显式 CLI/profile 优先级、两阶段静态计划、joint singleton key、第三到第四块的 lookahead、main 的窗口淘汰与候选准备流程，以及异常/取消时 lease 清理。离线 `sampling.py`、stage 离线工厂/forward、原模型测试相对 main 无差异。

该整合版本本地通过16项配置探针、7项 source work quantum 测试、固定 Ruff/isort/格式检查、inventory 和 diff 检查。Windows 环境直接运行模型回归时在测试收集前因缺少 `addict` 失败，未执行模型用例，不能算通过。最新整合版本的完整 Linux CPU CI 尚待结果；已通过的 CI `38847036113` 对应整合前 `350ca78523`，不覆盖本次 main 冲突解决。此次未重跑 GPU，因此这份审查不替代最新整合版本的 CPU/GPU 验收。

目标为 seedleap/sglang 的 Zing 扩展；沿用该分支已有 `ZingAdalnTable.prepare` 的 fail-closed 合约、streaming builder、模型专属 realtime sampler 和固定 CPU CI 注册机制。这些静态表与混合采样接口是项目扩展，不宣称为 SGLang 社区通用 API。已有设计与贡献规范的核实基线见 [初始 block 设计](initial-block-schedule.md)。

建议依次阅读：

| 文件 / 入口 | 核查问题 |
| --- | --- |
| `zing_h3_source_adapter.py:_source_sampling_schedule`、`on_init`、`build_sampling_params` | 是否完整解析 profile 并保留 CLI 优先级？请求、ready、内部命令是否一致？旧 profile 是否继续可用？ |
| `source_session.py:initialize`、`stage.py:_build_realtime_sampler` | 完整计划是否传入 worker 并重新校验？是否避免修改共享部署状态及离线路径？ |
| `build_adaln_table.py:make_plan`、`zing_h3_adaln_table.py:initial_blocks_plan` | 新旧计划是否明确区分？是否保留顺序、sigma 和完整批次？ |
| `realtime_sampling.py:ZingH3RealtimeSampler._admit_adaln`、`step`、`_ready_cross_prediction` | 是否对实际 packed 输入验证两阶段覆盖？prompt 更新及 lookahead 是否绑定正确的计划？ |
| `zing_h3.py:forward` 的 projection key 选择、`_projection` | 普通批次与 joint singleton 是否保持各自 GEMM 形状？同 timestep 的 clean 分支是否复用？ |
| `incremental.py:_prepare`、`dispose` | KV 写入前准入、单个 session lease 的成功/失败生命周期是否完整？ |
| `test_zing_initial_blocks.py` | 有无实际模型输出/KV、步数、构建入口与失败清理证据？ |

上游 reviewer 还会关注公开配置增长、CPU 准入开销、不同 GPU 算术环境及 SP2 的实测证据；本次不以 CPU 相等推断跨硬件可用性。
