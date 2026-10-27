# 六分支 prompt joint 与静态、动态 AdaLN 缓存兼容

## 使用与范围

在已配置 weights-free AdaLN table 的连续 source 服务中设置
`SGLANG_ZING_SOURCE_PROMPT_JOINT=1`，并使用 `joint_forward=cross_chunk`、
few-step、after_step0、纯视频和 raw KV。prompt 切换后的 prefix、三段 history、
source 和 active 首次求值可进入同一次六分支 forward。SP 仍要求 stable shapes，
部署层既有的 AdaLN/SP 准入范围不变；固定执行仍要求固定 raw KV 池，
CUDA Graph 保留已有设备和拓扑检查。
默认开关不变；这是执行兼容性改动，不改变 prompt 的生效边界、求解步数或权重。

此前 session 因 `model.adaln_table is not None` 回退串行；直接调用 eager/fixed
六分支会报错。完成后按原条件选择六分支，表项不全仍拒绝执行，不恢复已卸载的投影权重。
本功能独立于 Qwen remote 契约 PR #437。

驻留权重部署也支持 eager 六分支的动态投影缓存：沿用
`SGLANG_ZING_ADALN_PROJECTION_CACHE=1`（默认开启）与
`SGLANG_ZING_ADALN_PROJECTION_CACHE_BYTES`（默认 134217728 字节）。
以前六分支显式绕过并清空该缓存；现在按完整投影组填充、复用现有 LRU。
静态 table 优先；关闭动态缓存或容量不足时仍按组计算驻留权重。

## 设计

`prefix_modulation` 已保留每次串行投影的完整 timestep batch 并去重完整组。
例如 prefix `[0, 0.999]`、clean `[1]`、active `[0]`，拼接时间为
`[0, 0.999, 1, 0]`，组大小为 `(2, 1, 1)`。两处的 `0` 不能合并：
BF16 GEMM 的批次形状是 table 的数值契约。简单删除 guard、逐行 singleton 查表
或对全部时间重新 unique 都不能保持这个契约。

因此 eager 按相同 group sizes 生成 table key 和投影；fixed 在 capture 外解析同样的
完整组，并逐 component 拼接结果，作为固定输入交给已有 `_forward_body`。
普通两/三分支仍用 singleton 组，非 joint 仍用完整 unique batch。
复用现有 table 文件格式、`video_plan` 和 realtime admission；无需增加表项类型。
session 持有 admission lease，模型持有 table，graph 持有静态投影 tensor。
所有新 lookup 均在 Python 准备阶段完成，不进入 CUDA capture/replay。

动态路径复用 `ZingH3AdalnCache.begin/project`，key 保留完整、有序的 timestep
batch、embedding 布局、权重版本和执行环境。混合 prefix `[0, 0.999]`
与 active `[0]` 是不同条目，不能拼接两个 singleton 来代替混合 batch。
缓存由会话 KV cache 持有，继续遵循原来的字节上限、失效和失败清理规则；
不增加模型级共享缓存，不改变 `cache.clear()` 生命周期。
因此在线 prompt 切换先清空缓存，六分支当次是冷填充；后续相同投影组才会命中，
不能把接通动态路径解释成 prompt 切换首块必然获得暖缓存收益。
本次动态支持限于 eager；fixed/Graph 继续使用已解析投影输入，不在 replay 中操作 LRU。

动态扩展的验证计划：真实六分支和后续 active forward 对照不缓存的输出与逐层 KV，
检查混合 prefix 的实际 GEMM 批次、冷 miss、后续 hit、零容量回退和清空后重建；
复测已有权重/embedding/generation 失效、LoRA bypass、逐出和在线 prompt 测试。
GPU 性能尚待独立实测，既有静态 table 实验不能替代动态缓存证据。

## 验证

计划：真实小模型 resident/table 对照，检查六分支输出与每层 KV；混合 prefix、
active 与 prefix 共享 timestep、无参考图、普通 joint 回归；fixed warmup/capture/replay
及在线 prompt 更新路由，缺表失败清理。CPU 记录图只能验证捕获算子和输入刷新，
不能证明 CUDA/NCCL 性能或跨卡数值正确性。

2026-10-27 实测（初始实现 `bd5c9d9c40`、WSL CPU）：

| 范围 | 结果 | 能证明的内容 |
| --- | --- | --- |
| `test_zing_h3_joint.py`、`test_zing_h3_denoise_graph.py`、`test_zing_adaln_table.py` | 237 passed、4 skipped | 普通 joint 回归、表完整性、连续 prompt 更新、weights-free 固定执行和录制回放 |
| 两个 `benchmark/zing_h3_causal/tests/test_prefix_joint*.py` 的既有及新增组件测试 | 19 passed、1 skipped | 混合/单值 prefix、输出及逐层 KV、缺表项失败清理、输入刷新 |
| 新增 `test_weights_free_prefix_sp2_matches_same_sp_resident` | 1 passed | CPU 双 rank collectives 模拟，实际模型分片计算的 table/resident 输出及 KV 对照 |

共 257 项通过、5 项跳过；Ruff、isort 和 `git diff --check` 通过。
移植到 main 后，功能提交 `b8bef7931b` 将上述五个文件合并复测，结果仍为
**257 passed、5 skipped**（209.45 秒，1 条 pytest 插件重写 warning）。
独立只读 review 在七文件范围内未发现阻塞问题；后续提交仅补充本验证记录。
输出/KV 使用已有 BF16 容差 `atol=rtol=0.008`；在线完整推理使用已有 `0.02`，
fixed 与 recording replay 输出和 KV 另作逐位相等检查。

测试入口为 `pytest -q` 加上述路径，CPU 环境沿用现有轻量运行适配器
（platform override=cpu、RoPE native CPU 实现、OMP_NUM_THREADS=1）。
原始日志保存在本机任务输出目录 `outputs/zing-prefix-adaln-20261027/`。
未进行本次代码的 480/736 GPU 完整推理、SP2/SP4 CUDA Graph 或峰值显存测量；
此前测试不能用作本功能性能证据。尤其此前单数据测试的 `source_prompt_joint=false`。

后续 r5 已在历史冻结快照上叠加 `0f1e31222c` 的三个生产文件，完成 832×480、
360 帧预热和正式推理；两个 DiT rank 实际命中六分支。该实验使用静态 table，
切换首块 DiT 为 958.5ms，总耗时含尚未细分的 EOS 停顿；它不是完整 PR head
或本次动态缓存扩展的 GPU 验收。原始材料见
`outputs/qwen-six-h100-all-on-20261027-r5/`，
[PR440 飞书记录](https://icnimsatr0zz.feishu.cn/docx/Kdfjd4vQ4o2G5GxAHtJcALqinNd)。

2026-10-27 动态扩展实测（`0f1e31222c` 加本次四文件增量）：上述五个文件加
`benchmark/zing_h3_causal/test_adaln_projection_cache.py` 合并执行，
**273 passed、5 skipped、1 warning，210.46 秒**。WSL Python 3.12.3 /
PyTorch 2.14.1+cpu，沿用同一 CPU 适配器和线程配置。
新增 4 项覆盖混合/单值 prefix、实际 GEMM batch、后续 active 的动态命中、
零容量回退与清空后重建，输出及逐层 history KV 均通过已有 0.008 容差。
原有在线 prompt、两/三分支、静态表和缓存失效测试同时通过。
Ruff 指定 lint/format、isort、diff 检查通过，未执行全仓锁版本 pre-commit。
日志、runner 和作者自审记录：`outputs/pr440-dynamic-adaln-20261027/`。
本轮未进行动态扩展的 GPU 测试，不据 CPU 结果宣称性能或显存收益。

## 版本、实践与 review 导读

PR 直接基于 seedleap/sglang 的 main `290d08c7975ffdb0050dab1cb43aa149239c17c5`，
当前包含本兼容功能的八个文件（初始静态表增量为七个）；原实现从 PR #437 的分支
移植，未引入其 Qwen 改动。
已核对本版本 `docs/docs/developer_guide/contribution_guide.mdx`：使用独立分支、
聚焦回归测试和现有格式检查；测试沿用已有 diffusion unit 与 benchmark contract 位置。
核对仓库现有 `zing_h3_adaln_table.py::video_plan/project` 的完整 batch 契约、
`zing_h3.py::_projection/prepare_fixed_forward` 和 `PrefixJointGraphs` 的固定输入模式，
沿用这些仓库实现，不将其宣称为社区上游已验收的能力。
上游合入仍需真实 GPU/SP/Graph 证据；review 重点是分组与索引一致性、普通 joint
兼容性、缺表失败边界和捕获期是否存在 host lookup。

阅读路线：本说明 → `incremental.py::_prefix_prediction`（路由）→
`zing_h3_joint.py::prefix_modulation`（分组语义）→
`zing_h3.py::forward/prepare_fixed_forward/_projection`（查表与消费）→
`zing_h3_prefix_graph.py::PrefixJointGraphs`（固定执行）→ 对照测试。
