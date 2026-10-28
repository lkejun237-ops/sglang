# Prompt 切换六分支组件与在线接入

六分支将 conditions、五个 latent 的 1/2/2 三块历史、source、target 首步合为一次 DiT forward。当前 #439 接入 #445：配置 KV pool 并设置 `SGLANG_ZING_BLOCK0_CACHE_MODE=overlap` 后自动启用，默认 off；旧 `SOURCE_PROMPT_JOINT` 不再控制在线路径。命中静态 KV 时省去 conditions 分支，进入五分支，见[缓存消费设计](../specs/CACHED_PREFIX_JOINT.md)。续推由 `SOURCE_DEFER_FINAL` 单独控制，无融合 attention kernel。

在线入口为 `ZingSourceStreamState` → `SourceReadySession.drain_ready`。会话在 prompt 边界准备四块 prefix，并在 source/target 就绪后调用原生模型事务；Graph 使用模型持有的 `PrefixJointGraphs`。runtime 不导入 benchmark、不安装 Python monkeypatch。模型负责分组 AdaLN、紧凑 visual 输出边界及一致的 RoPE，cache 事务负责逐块提交。Graph 仍采用独立 pool 和整代 shape 回收。

仅 video-only、raw chunk KV、cross_chunk、after_step0 支持；SP 要求 stable shapes，AdaLN 支持驻留权重／动态缓存和静态表。不满足时使用同一批 prefix 张量串行提交并记录 fallback；外部 KV 命中但不支持联合执行时明确失败。Graph 未启用时支持 eager 五／六分支；安装缓存消费者后的 `step` 和 `drain_ready` 均支持。取消/异常清空会话持有的 prefix；event 标签仍由成功输出的 ChunkResult 携带，打包结束不代表 GPU 已采用。

VAE 单独实验已复现 performance 模式跨进程编码差异，确定性模式相同；固定输入的串行/joint KV 抽样相同。这解释旧 RGB A/B 差异的已知来源，不替代本次原生实现的验证。本次原生接入的 CPU 验证见下文；480/736 当前提交端到端数据待补。

## 调用方式

设置 `PYTHONPATH=python:benchmark/zing_h3_causal`，传入真实 sampler、inputs、condition、五 latent history 和按重建 origin 准备好的 current state：

```python
import prefix_joint
from prefix_joint_graph import PrefixJointGraphs

packs = prefix_joint.packs_for(sampler, inputs, condition, history, current)
# cache 必须是已清空的逻辑 cache，Graph 还要求固定 raw KV pool。
prediction, audio = prefix_joint.joint(sampler, packs, cache)
cache.clear()
graphs = PrefixJointGraphs(max_graphs=2)
prediction, audio = graphs.run(sampler, packs, cache)
```

仅保留六次分段 attention，不再提供 fused kernel 或 attention 模式参数。Graph 要求 resident AdaLN 和 stable-shape SP，不支持音频；服务采用单活跃会话。

## 联合的语义

分支内部双向，后分支看之前 KV，前分支不能看未来。QKV/MLP 和通信合批提供主要收益，不能由 attention kernel 数量推算速度。

`prefix_modulation` 保留串行 timestep 分组和 GEMM batch 形状；conditions 可能有多个 timestep，target 的单 timestep 即使值相同，也不能直接复用多 timestep batch 的 AdaLN 行。输出投影按压缩 visual 行数切分，不能用含 text 的 attention token 边界。

`PrefixJointTransaction` 先核对各分支 RoPE 与串行视图一致，再按 prefix→三段 history final→source→active 提交。失败清空逻辑 cache，live cache 在执行前拒绝。事务持有本次坐标映射，不修改 cache 的方法；benchmark 也调用同一原生模型和 Graph 实现。

## Graph 生命周期

复用 `ZingChunkGraphs` 静态输入刷新、rank 协调及回滚。每个 prefix shape 使用独立 allocator pool；shape 预算满时先同步、清掉整代 Graph，再捕获新代。历史第三种形状曾非法访问，改成此生命周期后通过三形状及回访检查；没有 sanitizer 定位，不能把相关性写成确定根因。

SP 输出对等长/不等长 shards 都用同一 Ulysses communicator 和 Graph 自有 receive buffer，再还原全局 visual 索引。首见 shape warmup、第二次 capture、之后 replay；淘汰后再次付出冷启动成本。调用方须在销毁进程组前释放 Graph。

## 历史组件实测

以下融合 kernel 数据为删除前历史结果；当前只保留分段路径。三例真实冻结输入、50层、SP2/BF16，每路径3次 warmup、12次轮转。表中为整体组件 CUDA ms：

| 场景 | 串行 | joint 分段 | joint fused | 分段 Graph | fused Graph |
| --- | ---: | ---: | ---: | ---: | ---: |
| 雨衣 | 779.57 | 484.27 | 472.50 | 449.38 | 472.83 |
| 玻璃房 | 880.95 | 560.05 | 550.67 | 525.70 | 551.53 |
| 发光树 | 884.65 | 561.68 | 551.28 | 526.40 | 552.92 |

prediction、50层KV、位置和 active 状态 max_abs/RMSE=0，包括 payload 变化和淘汰回访。组件中分段 Graph 最快；在线实验只比较串行与 fused 六分支 Graph，不能推断分段 Graph 的端到端排名。旧 GPU SHA/归档见主文证据 JSON，当前主线移植没有新 GPU 数据。

## Review 与验证入口

先读 `source_session.initialize` 的开关，再读 `SourceReadySession._activate_pending_prompt/_prefix_prediction/drain_ready`，随后是 `zing_h3_joint.py` 的六分支事务与 modulation 分组、`zing_h3.py` 原生 eager/fixed forward、`zing_h3_prefix_graph.py` 的形状生命周期。Graph 独立成文件是因为其 pool/形状回收策略不同于续推，不是新增 attention kernel。

测试入口：`test_zing_h3_denoise_graph.py` 在线会话用例，以及 `benchmark/zing_h3_causal/tests/test_prefix_joint.py`、`test_prefix_joint_graph.py`。覆盖 mixed timestep/reference、串行/联合 prediction/KV、changed-input replay、shape 回收、等长/不等长 gather；在线另覆盖四次 prompt 更新、与 deferred final 共存、RNG/event、EOS、失败清理及下一会话复用同一槽。明确区分原生 eager 路径与 fixed-forward 不捕获执行、recording replay。

本轮本地验证结果记录到 [PR 主设计](SOURCE_DEFERRED_FINAL.md)。CPU recording backend 只能检查计划、输入刷新和生命周期，不替代 CUDA/NCCL。当前实现 480/736、真实 SP2/SP4、Encoder/DiT/Decoder 分项耗时与峰值仍待 GPU 验证；首次新形状冷启动代价仍存在。开关默认关闭，代码集成不表示线上进程已切换。
