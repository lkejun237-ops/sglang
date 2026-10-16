# Issue #257：AdaLN weights-free 首版代码分析

本文保留 2026-10-09 在 b1899c0 基线上的初始分析，以下缺口描述是历史快照。#264 已合入，当前实现、CPU 验证与 GPU 待验收状态见 [ADALN_TABLE.md](../specs/ADALN_TABLE.md) 和 [ISSUE_257_EXPERIMENT_PLAN.md](../../../../../benchmark/zing_h3_causal/ISSUE_257_EXPERIMENT_PLAN.md)。

- Issue：[seedleap/sglang#257](https://github.com/seedleap/sglang/issues/257)，读取到的最后更新时间为 2026-10-09 04:59:36 UTC。
- Worktree：`/Users/yiwencai/.codex/worktrees/issue-257-adaln-table/sglang`。
- 分支：`codex/issue-257-adaln-table`。
- main 基线：`b1899c0bd538a342056a39893e8ce048b17afa32`，已经包含默认开启的 denoise CUDA Graph（#258）。
- native 核查提交：`4c01c82d17db366bd82c81541b42bbada16d0e43`，以 `git show` 阅读，没有切换 native 工作目录。
- #242 核查提交：`9dcfaab10f812c68e44c5989df370bd2b31fc91f`，远端仍为 OPEN，未合入本 worktree。

## 结论与首版范围

这是模型构造、checkpoint 加载及部署工件的变化，不能仅靠扩大当前缓存完成。首版按 issue 已确认范围采用离线 builder、版本化表工件和固定模型/合并 LoRA/调度的 DiT worker。worker 启动时不构造、不读取或 materialize 主层与 final AdaLN 投影参数；表或完整请求计划不匹配时，在生成前拒绝。

显式 resident 部署继续作为独立对照和兼容模式。动态 LoRA、在线 builder、任意 schedule 在线补表、活跃会话热换模型、video_audio 和 GEMM 粒度优化均不纳入首版。

## 已核实的实现缺口

### 1. main 仍限制四分支及固定角色路由

- [zing_h3.py](../../../../../python/sglang/multimodal_gen/runtime/models/dits/zing_h3.py) 构造函数要求 `num_modalities == 4` 与 `adaln_out_features == 24 * hidden_size`；拒绝额外构造参数，并要求完整 resident AdaLN 权重。
- `_row_layout` 把 text/audio/source 固定到 tag 1/2/3，其余默认 tag 0。forward 使用 `t_indices * 4 + modality`。
- [export_checkpoint.py](../../../../../benchmark/zing_h3_causal/export_checkpoint.py) 只接受四分支和固定 `modality_tags`；输出配置也写死四分支。
- issue 指定 native 的 `src/h3/modules/backbone.py` 按配置决定四/五支，按实际 tag 排序 `projection_names`；`src/h3/modeling.py` 明确把 CONDITION_VISUAL 路由到 condition tag，把 VAE reference/target 路由到 video tag。
- native final AdaLN 是独立的 `2 * hidden_size` 投影，不按主层分支数展开。

需要先建立从实际 projection 集合、角色/tag 与 final 独立布局派生的明确契约。已有训练后的 source/condition 参数必须保留；不得借用 native 初始化复制逻辑去补齐损坏的推理工件。

### 2. 现有 resident 缓存不能在删除投影参数后继续工作

[zing_h3_adaln_cache.py](../../../../../python/sglang/multimodal_gen/runtime/models/dits/zing_h3_adaln_cache.py) 的 `_signature` 包含参数对象身份、版本、storage 指针、shape/stride/dtype/device。`project` 每次先取 module signature，miss/bypass 直接执行原 module。缓存由每个 `ZingH3KVCache` 持有，会话清理时释放。

所以仅预热后设 `adaln_proj = None` 或删除参数会破坏该接口，缓存身份也不能作为持久工件身份。新表应使用内容与版本身份；resident LRU 保留原语义。weights-free 消费路径不能依赖参数签名、请求 KV 是否存在，或调用已经不存在的投影模块。尤其要覆盖 CFG/unconditional 的 `cache=None` 路径。

### 3. loader 有可复用的过滤机制，但现有 H3 sidecar 不能直接接到 Zing

- [transformer_loader.py](../../../../../python/sglang/multimodal_gen/runtime/loader/component_loaders/transformer_loader.py) 的 `checkpoint_key_filter` 会在权重 iterator 读取 tensor 前过滤；已有 H3 路径可跳过 `.adaln_proj.linear.`。
- [fsdp_load.py](../../../../../python/sglang/multimodal_gen/runtime/loader/fsdp_load.py) 先在 meta device 构造模型，再传递该 filter 加载权重。
- [minimax_h3.py](../../../../../python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py) 的 block/final 已支持 `use_adaln_cache=True` 时不构造投影模块，但模型级开关同时绑定普通 `MiniMaxH3AdalnCache`。
- 当前 server 参数入口仅接受类名 `MiniMaxH3DiTModel`；Zing 子类还显式拒绝相关 kwargs。
- 普通 [minimax_h3_adaln_cache.py](../../../../../python/sglang/multimodal_gen/runtime/models/dits/minimax_h3_adaln_cache.py) 使用 H3 专属布局、model variant、计划格式和加载/更新契约，不能据此宣称 Zing 四/五支精确性已经通过。

可以复用“构造时跳过投影”和“读取前过滤”的底层机制，单独接入 Zing 工件与模型校验。filter 必须按实际加载格式生成，并覆盖主层和 final；仅在加载后删除模块不能满足启动峰值要求。其他模型行为不能随之改变。

### 4. #242 提供部分五分支基础，还缺合法 tag 重排

[#242](https://github.com/seedleap/sglang/pull/242) 的核查提交已经接入 four/five split 导出、严格完整键/shape 检查、trained branch 保留、export v2、condition/reference timestep 和默认第五支路由。

但 `adaln_modalities` 只接受默认四支顺序，以及 condition 共享 video tag 0 或独立 tag 4；运行时 `_row_layout` 仍使用固定角色 tag。因此其支持范围不能直接覆盖 #257 要求的合法 tag 重排。

建议衔接或提取对应的导出/布局部分，并补齐 tag 派生契约。不要为本 issue 无关地合入整套 cache-only clean-commit 优化。#240 的 full-batch timestep reuse 也是 OPEN，可作为数值/生命周期经验，不能作为已经存在于 main 的功能。

### 5. 完整 unique-t 批次必须从实际请求计划枚举

当前 [sampling.py](../../../../../python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/zing_h3/sampling.py) 在 initialize 中执行 prefix；每个 chunk 包含 source、若干 active 和 clean final。[_sampling_helpers.py](../../../../../python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/zing_h3/_sampling_helpers.py) 生成逐 row 的 FP32 timestep，模型再做原有 `torch.unique(sorted=True)`。

#242 又增加 `ar_condition_timestep` 和 `reference_visual_timestep`，实际批次可能混合这些值、当前 denoise t 与 clean t。不能只构建“4 个 denoise timestep + 1 个 clean”或沿用 #207 的 5 批统计。尾块、条件更新、新 prefix、无 source、CFG 和可进入的 overlap 路径均需由同一计划枚举逻辑覆盖；不支持的请求在任何 forward/KV 写入之前拒绝。

builder 与接入检查复用同一 CPU timestep 计划计算，保留每个完整批次的原始值、dtype、shape 和顺序。启动时核对工件，接入时为完整计划绑定并锁定表引用，热路径只使用已解析索引。

### 6. main 的 CUDA Graph 路径也要接入表 provider

`_projection` 已是统一取紧凑参数的消费点；norm、token 索引、gate 和输出 head 可保持原数学操作。主层消费六组 `[M * modalities, hidden]`，final 消费两组 `[M, hidden]`。

然而 `_denoise_graph_rejection` 当前要求 `options.projection_cache` 与 `projection_key`；`_denoise_graph` 预先调用所有投影，并将紧凑参数拷入 graph 静态输入。新 weights-free 模式若只替换普通 forward，将使当前默认 CUDA Graph 路径回退。

表 provider 应同时服务 eager 与 graph 的 projection 解析，保持正确布局/视图和 graph 静态地址生命周期。验收应记录 graph capture/replay/fallback，并覆盖关闭会话、计划锁定和跨 stream 的 event/fence。保留 TimeMLP 的首版也要改 `_bind_cache` 及模型/table generation 绑定；不要仅因它的输出进入表便删除仍被运行时读取的 embedder 状态。

## 数值契约与显存量级

结构计算取 `hidden=5376`、`time_embed_dim=2688`、50 层、BF16，主层每支六组，final 两组独立布局，包含 projection bias：

| 主层分支数 | 投影权重及 bias | 每个 unique-t 行的全部紧凑输出 |
| --- | ---: | ---: |
| 4 | 32.3657 GiB | 12.3252 MiB |
| 5 | 40.4436 GiB | 15.4014 MiB |

公式为 `((50 * 6 * branches + 2) * hidden * (time_embed_dim + 1) * 2)` 字节；输出去掉 `(time_embed_dim + 1)` 因子。它们是结构估算，不是模型实测或实际显存节省。表大小应按所有完整批次行数和实际 storage 累计，还要计独立表布局、对齐及可能保留的 timestep embedding。

同一次部署的 resident 基线与 builder 必须使用相同有效权重、完整有序 unique-t、投影形状、bias、dtype 和算术策略。#242 把 split 参数拼接为 single runtime 权重；若采用该 serving 格式，应冻结该表示与策略作 resident 对照。native 分支 GEMM、serving 拼接 GEMM及逐 timestep GEMM不能仅以代数等价宣称逐位一致。CPU 构建表不能默认与目标 CUDA resident 前向逐位一致。

manifest 至少绑定 schema、有效 generator/EMA 与合并 LoRA 权重内容身份、time embedder/频率常量身份、角色/tag 与输出布局、完整 timestep 批次、构建设备/torch/CUDA/算术设置、tensor checksum 和验证范围。工件内容身份与加载到哪个设备、哪个 stream 已就绪应分别管理。

## 建议实现与验收顺序

1. **布局及工件契约**：实际 branch/tag 派生、single/split/final 描述、有效权重身份、完整批次表示、版本匹配；以四/五支、condition 共享/独立、合法重排和损坏键/shape 的小模型 CPU 测试先锁住契约。
2. **离线 builder**：逐层流式加载，在每层按原完整批次和选定 serving 数值策略生成紧凑参数，单独处理 final；导出 manifest、tensor checksum 和构建资源记录。不为了“批量构建”合并不同批次或改变 GEMM 形状。
3. **weights-free loader 与 provider**：构造时跳过投影、权重读取前过滤、启动严格验证工件；完整接入计划预检查与引用锁定；模型版本内跨会话共享小表，SP1/SP2 每 rank 复制且热路径无新增 collective。对 loader 的读取与模型/storage 引用做直接检查。
4. **eager/graph 与生命周期**：统一 `_projection` 取表，保留其后操作；异常、取消、close、prompt 更新和多 stream 正确释放引用。首版固定版本更新通过重新部署执行；明确拒绝不兼容在线更新。
5. **真实 GPU 精确性**：另行授权后，冻结 checkpoint、真实 source+默认六图、噪声、codec、窗口与 schedule；在同 SP 下对 resident/weights-free 比较紧凑参数、每步 latent、完整 KV 与最终输出，要求严格相等、finite，并验证实际命中且无隐藏 fallback。分别报告 SP1/SP2 与四/五支覆盖，结构通过不替代真实资产验收。
6. **显存与性能**：分列 builder 峰值、worker 启动峰值、稳态、allocated/reserved、表 storage 字节及 projection 引用；不能用 `empty_cache` 或 reserved 下降替代 storage 释放证据。冷/热 E2E、chunk p50/p95 用成对 A/B 测量，构建成本单列；现有 resident 缓存已默认开启，不能把历史缓存加速再次算为本功能收益。

## 本次完成及未完成

已完成新 worktree/分支、最新 main 核验、issue/#207/#240/#242 远端阅读、指定 native 与 #242 源码核对及本分析文档。项目级 7 个 infra skills 来自固定提交 `5e5879e4480a9f76be02e68b3d308b57f05bdf51`；33 个文件的 Git blob 内容及 Markdown 相对引用核验通过，安装目录使用 Git 本地 exclude。原主工作目录的已有文件未覆盖。

本次没有修改运行时代码、提交/push、申请 GPU 或提交云作业。没有执行数值/性能测试；本文的实现拆分与验收顺序是工程建议，不是已完成实现或性能结论。

相关链接：[既有 resident 缓存 #207](https://github.com/seedleap/sglang/pull/207)、[timestep reuse #240](https://github.com/seedleap/sglang/pull/240)、[split/five-branch 与 clean-commit #242](https://github.com/seedleap/sglang/pull/242)。
