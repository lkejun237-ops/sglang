# Zing 续推联合与 prompt 切换优化

本 PR 整理固定单块 dispatch 的续推联合，以及 prompt 重建的六分支在线路径。两者均可在 source 服务中显式启用；六分支组件与在线流程共享原生模型和 Graph 实现。保留 joint forward 与分段 attention；一次 DiT forward 不等于一次 attention kernel。融合 joint kernel 已删除。

最初基于主线 `b913437ad1a237a1af9dbf1fcfd5cce11e22ec5b`，现已合入 `1dbe50bf4d`（#399 的两步默认值 `[1000, 500]`），保留固定五 latent 重建、source 输入规范和有界 Graph 回滚，不恢复 #389 删除的旧 eager kernel 自动路由。历史 GPU 实验基于 `82fd85da3e` 及记录中的 overlay，不是更新后主线版本的 GPU 验收。

## 使用方式

已有可运行的 Zing 连续 source 部署设置：

```bash
export SGLANG_ZING_EXPERIMENTAL_SOURCE_WORK_QUANTUM=1
export SGLANG_ZING_SOURCE_DEFER_FINAL=1
export SGLANG_ZING_SOURCE_PROMPT_JOINT=1
```

两个开关默认均为 0；`SOURCE_PROMPT_JOINT` 的准入与回退见[六分支设计](PROMPT_PREFIX_JOINT_PROBE.md)，不开 Graph 时仍支持 eager 六分支。`SOURCE_DEFER_FINAL` 启用要求 cross_chunk、五帧 history、video-only、无 CFG、after_step0 淘汰、每次 dispatch 一个块，不满足会在入口报错。客户端协议、source credit、事件归属和采样契约不变。

续推与六分支都使用原来的分段 attention。融合 kernel、专用路由及开关已移除；历史脚本应删除该开关。

## 状态与执行顺序

| 边界 | 行为 |
| --- | --- |
| 首块 | 原 prefix → source+target 首步 → 余下去噪 → 保存已有 recent_visual 并输出 |
| 续推 | history 末块 final 与本次 source、target 首步联合，再完成新块 |
| prompt 实际激活 | 原 rebuild 清掉旧 active/source，用五 latent 重建，不提交旧条件 final |
| EOS / 取消 | 已知末块立即 final；晚到 EOS、取消和失败释放会话，不补无用 forward |
| 联合资格或 scratch 不足 | 单独 final，再走原生成路径 |

`cache.active` / `_active_reference` 持有末块身份和 host 几何，`recent_visual` 已持有五 latent。不新增 pending-final 标记、块号水位或独立 latent 副本。`_deferred_final` 临时打包 clean rows；metadata device 跟随 layout，不能由 CUDA latent 推断 metadata 也在 CUDA。

`ZingChunkGraphs.resume` 沿用 joint transaction 与 planning copy：先提交旧 final，再 begin 新块，成功后发布逻辑状态。`chunk(..., defer_final=True)` 在本块 final 前结束，仍执行相同去噪和采样。Graph 静态输入仍要复制，不声称零拷贝；捕获回滚沿用主线实际写入区间。

## 删除融合 kernel 的依据

三分支完整 case 无速度收益；六分支组件开启 Graph 后，分段 attention 为 449.38/525.70/526.40 ms，融合 kernel 为 472.83/551.53/552.92 ms。按用户决定移除融合 kernel、选择器、动态 mask/稀疏列表及专用测试，不再恢复 #389 删除的两个模块。原有 FA4 分段 attention、fixed KV 和 Graph 生命周期继续使用主线实现。六分支组件也移除 attention 模式参数，避免留下不可用入口。

历史融合实验保留在下表和证据 JSON 中，仅用于解释决策，不代表当前代码仍支持该路径。六分支额外契约见[组件设计](PROMPT_PREFIX_JOINT_PROBE.md)。

## 历史实测与限制

实际设备报告 NVIDIA L20D，CC10.3/148SM，每卡 287428771840 bytes。SP2、BF16、w6s3、q1、832×480、seed42、前三块四步/后续两步；w6s3 是用户指定的 PERFORMANCE_DEFAULTS window21/sink3 例外。

| 实验 | 三例结果 | 结论 |
| --- | --- | --- |
| 续推 joint 关→开 | FPS 15.877→16.131、15.610→15.847、15.579→15.820 | 同 allocation 顺序 A/B 约 +1.5%–1.6%，不作显著性结论 |
| 续推 kernel 关→开，组件 Graph | 108.645→107.806、110.888→110.680、110.313→109.869 ms | 收益不到 1%；冻结输入 prediction 及 50 层 KV max_abs/RMSE=0 |
| 续推 kernel 关→开，完整 case | FPS 16.126→15.947、15.665→15.615、15.776→15.593 | 无端到端收益，已删除 kernel |
| 六分支在线实验桥接 | prompt→完整新 event RGB：2.128→1.792、2.332→1.996、2.391→2.000 s | 热态约省 0.34–0.39 s，桥接不属于生产入口 |

完整 case 每组每例正式三次，另有独立诊断；视频 360 帧。source 按 24 FPS 上限发送并遵守反压，收到 128 帧后发 prompt，新 event 首帧零基 136。FPS 是首批后完整批次到达速率，包含切换停顿；响应是客户端发送至完整 RGB 到达，不是人工看见变化，也不含真实 WAN。

六分支在线两组 9 对原始 RGB SHA256 都不同，各组内三次重复一致；后续固定噪声诊断找到 VAE 编码先于 DiT 出现差异；VAE 独立复测 performance 跨进程不同、deterministic 相同。固定相同条件/history 后串行与 joint 的全层 KV 采样一致；尚未定位具体卷积算法，组件抽样不等于完整张量证明。在线联合组在 prompt 前也更快，不能把全部 FPS 差异归因六分支。首次新形状响应约 10.256 s、第二遍捕获 2.773 s，正式表取预热后回放。六分支峰值 allocated 约 173.272 GiB/card，对照 172.462 GiB/card。

736、SP4、更新主线后的真实 GPU 回归、完整当前版本 latent 对照和人工视觉验收未完成。无 Graph 复测已由用户取消：仅 CPU 准备任务停止，没有提交 GPU。本 PR 保持实验开关默认关闭，当前 PR 已转为可审阅；代码接入不表示已经部署。Encoder（Qwen/reference/source VAE）、DiT、Decoder 的分阶段中位数/P95与独立常驻/峰值显存未完整采集；当前仅有整进程 allocated 峰值及历史组件计时，后续须同主线同条件补测。TPOC/ICL、source-to-output、播放停顿未按统一 benchmark 完整验收；已有 TTFF、响应、批次间隔及 P95 见证据摘要。

[证据 JSON](../../../../../benchmark/zing_h3_causal/results/joint_forward_20261023.json) 保留来源、统计和哈希。原始 trace、RGB、视频、脚本与失败实验留在原 D 盘/CPFS 目录，不提交媒体，也不删除原始数据。

## 验证与 reviewer 阅读路线

本次原生在线接入：模型/KV/会话/Graph 九文件本地回归 **501 passed / 5 skipped**（287.51 s）；CUDA 依赖项跳过。串行与 eager/fixed 六分支输出按既有 CPU 容差比较，事件和 RNG 相同；fixed eager 与 recording replay 的输出/KV 逐位相同。四次 prompt 更新、deferred final 联动、unsupported 回退、故障清理与同槽下一会话通过。首次扩大回归发现旧二/三分支 SP 测试包装器不接受 mixed-prefix 新参数，已限定只在六分支传入；保留原测试并重跑通过。source worker 接线、默认采样调度三文件补充回归 **788 passed**（66.60 s）。CI inventory、isort/Ruff/diff 检查通过。没有新增 GPU 作业、生产部署或当前版本性能结论。

PR：[#403](https://github.com/seedleap/sglang/pull/403)。合入 #399 后的九文件回归461通过/5跳过，另3项preflight因Windows worktree在WSL的Git路径失败，使用native Git适配后定向3项全部通过；无GPU复测。CI清单已登记两份六分支测试及新增CUDA专用用例的精确跳过原因。删除融合 kernel 后的本地回归335通过/5跳过（109.97秒），覆盖续推、fixed forward、六分支模型/KV、changed-input Graph replay、source输入与生命周期；isort/Ruff/diff检查通过。本轮未新增GPU测试。此前更新主线后的 CPU 九文件回归353通过/14跳过；最终组件及服务输入/生命周期合跑348通过/8跳过；独立审查发现的单步空snapshot、SP本地准备失败投票已修复，定向3项通过。isort/Ruff和diff检查通过，完整pre-commit未安装。各轮有重叠，不相加为独立测试数量。CPU recording backend 不能证明 CUDA/NCCL 正确。重点断言：跨 dispatch 输出/RNG/event/KV，prompt 清理旧 final，晚到 EOS/取消/回退，CPU/meta 打包，changed-input Graph replay。

依次阅读 `source_session.initialize` → `SourceReadySession.drain_ready` → `_sampling_helpers._deferred_final` / `sampling._resume_cross_packs` → `ZingChunkGraphs.resume/chunk` → `prepare_fixed_forward`，再看 `_prefix_prediction` → `PrefixJointGraphs` → 原生六分支事务与对应测试。

已核对当前仓库 contribution guide、diffusion contributing、PR 模板和相邻 Graph/cache 实现。复用现有 transaction、plan 及主线分段 attention，不修改通用 backend。上游 reviewer 仍需评估开关维护成本、跨硬件回退、完整数值差异、736 验证和收益；不声称已获社区认可。

本地复现（Python3.11、依赖齐备）：

```bash
export PYTHONPATH=python:benchmark/zing_h3_causal
export SGLANG_LIGHTWEIGHT_RUNTIME=1 SGLANG_DIFFUSION_PLATFORM_OVERRIDE=cpu
export SGLANG_DIFFUSION_PERF_BASELINE_PLATFORM=h100 OMP_NUM_THREADS=2
python -m pytest python/sglang/multimodal_gen/test/unit/test_zing_h3_denoise_graph.py python/sglang/multimodal_gen/test/unit/test_zing_h3_fixed_forward.py benchmark/zing_h3_causal/tests/test_prefix_joint.py benchmark/zing_h3_causal/tests/test_prefix_joint_graph.py -q
```
