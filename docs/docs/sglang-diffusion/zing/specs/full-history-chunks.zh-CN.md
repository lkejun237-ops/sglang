# Zing temporal chunk 与 full-history 状态接口

> 当前默认已改为 21-chunk window（3 sink）与 block_relative RoPE，详见 [窗口配置](temporal-window.zh-CN.md)。本文 full-history 数值参考需显式设置 `zing_window_chunks=null, zing_sink_chunks=0, zing_rope_position_mode="absolute"`；下述历史验收不代表相对 RoPE 已通过。

这是 #113 的 worker 接口与后续集成约束，不是已上线的 v1 source-stream 能力。
保留 #103 的有限视频生产路径；新的 `SourceReadySession` 只接受冻结 latent/noise，
未接入 WebSocket，也不代表未知时长会话内存有界。
依赖 #111 / PR #118 的独立 `runtime/realtime/zing_contracts.py` [像素 provenance 契约](realtime-contracts.zh-CN.md)，
与 #112 / PR #120 的增量 codec 通过显式 bridge 衔接。

## Checkpoint 分块与时间轴

原生实现核对点为 seedleap/zing `6454246a5d248fb5fd895ffa4314409803c36644`：

- [`h3_native_varlen_ar.yaml`](https://github.com/seedleap/zing/blob/6454246a5d248fb5fd895ffa4314409803c36644/src/configs/exp/h3_native_varlen_ar.yaml)：`num_frame_per_block: [1, 2, 2]`、`split_first_latent: false`。
- [`h3_native_varlen_dmd.yaml`](https://github.com/seedleap/zing/blob/6454246a5d248fb5fd895ffa4314409803c36644/src/configs/exp/h3_native_varlen_dmd.yaml)：覆盖 `num_frame_per_block: 5`。
- [`h3_packed.py`](https://github.com/seedleap/zing/blob/6454246a5d248fb5fd895ffa4314409803c36644/src/dataloader/processors/h3_packed.py)：从数组第 0 项开始循环，尾块截短；`split_first_latent` 仅插入首个边界 1，不移动后续原始边界。
- 本仓库 `export_checkpoint.py` 原样写入上述字段到 `zing_config.json`，没有将数组转换成标量。2026-09-20 已通过零 GPU 挂载预检，核对现有 `zing-h3-ar-step3500-631fe72/zing_config.json` 为 `num_frame_per_block: 5`、`split_first_latent: false`。这份 step 3500 权重不使用新 AR `[1,2,2]` 周期；分块方式以具体 checkpoint 为准。

共享入口：`runtime/realtime/zing_chunk_plan.py::ChunkPlan`。
`chunk(i, total_frames=None)` 返回绝对 latent 半开区间和 temporal chunk ID；
`source_block=2*i+1`、`target_block=2*i+2` 仅用于 packed role routing。
source、target 和 target audio 的同一时间范围只算一个 temporal chunk。
首个 AR 周期边界为 `[0,1), [1,3), [3,5)`，之后从 `[5,6)` 重复。
DMD=5 是 latent 分块长度，不是 solver 步数；也不是 VAE clip、网络 append 帧数或历史 window。

adapter、packer、sampler 几何校验共享计划。legacy adapter 仍为有限请求计算实际结束块，
仍保留原协议行为；不会因本改动接受 v1 未知时长 init 或对外宣告 four-step profile。
所有全局位置和 provenance 仍是绝对时间轴；prompt 变长时只移动 prefix 的原点，媒体位置不动。

## Source-ready 调用与提交

每个会话使用独立 sampler，所有操作由单写者顺序执行，不支持同一实例并发调用。
静态 references 由调用方提供并保持不可变；session 持有引用直到清理。

```python
session = SourceReadySession(
    sampler, static_inputs, seed=seed,
    max_pending_chunks=2, max_pending_bytes=byte_budget,
    total_latent_frames=None,
    initial_rng_state=(cpu_rng_after_initial_noise, cuda_rng_after_initial_noise),
)
# source/noise 已按 ChunkPlan 切块，不能直接把 codec clip 当成 model chunk。
session.append(
    source_latents, frozen_target_noise,
    provenance=pixel_provenance,
    audio=None, final=False,
)
result = session.step()  # 没有就绪输入时返回 None，不消耗 RNG，不推进游标。
# 消费 result 后由下游释放；这里没有隐式 source.consumed ACK。
# EOS：最后一个截短块用 append(..., final=True)，完整边界则 finish()。
# 调用方应在取消或终止性异常时执行 session.dispose()。
```

- `append` 在修改游标之前验证 chunk、连续 pixel provenance、空间几何、设备、音频区间及双预算；克隆输入到独立 storage，避免小 slice 持有完整视频。
- `step` 使用最新编码 prompt 与独立 `event_id`；后续更新用于下一块。source ID 不会覆盖 prompt 版本，也不重置 chunk 周期。
- `step` 执行异常后关闭会话，释放全部 KV、ready 队列、prompt 和会话 RNG，禁止重试或继续该会话；包括打包、prefix、source、denoise、final commit 后异常。没有为原样重试保存 latent/layout/KV/RNG 快照。正常执行仍隔离各会话随机流，避免影响其他请求。外部协议 ACK/decoder append 仍由 #114 协调。
- 输出依赖水位取已提交历史与当前 source 实际依赖的最大值，不能使用 latest accepted。像素区间完全由 #111/#112 bridge 提供，worker 不做 `latent_index * 4` 推算。
- 正常 EOS 排空后自动释放 session 引用；取消/终止性错误调用幂等 `dispose()`。调用方持有的结果和原始 fixture 不属于 session，不会被它释放。
- legacy `sampler.step(state, inputs=updated_inputs)` 执行 prefix 更新与下一块采样，执行失败即关闭 sampling state。原来单独的 `update_condition` 保留供边界操作使用。

增量路径尚需明确两项 RNG 合约：原生 source VAE 在完整 `[B,C,T,H,W]` moments 上分配 posterior noise；
原生 target 初始噪声同样一次按全 T 张量生成。改成每 clip/chunk 随机生成会改变 channel-major 对应关系。
因此本接口要求已冻结 source latent / target noise，以及初始 noise 生成后的 CPU/CUDA RNG 状态；
`initial_rng_state=None` 仅表示显式从 request seed 开始新的采样随机流，不能用于复现已消费 initial noise 的原生流程。
CUDA snapshot 必须携带本 rank 的 CUDA RNG，不能只给 CPU state。
#120 encoder 的输出是 **moments**，不能直接传入 `append`。
CPU fixture 对照证明给定同一 noise 的执行一致性，不证明新的在线噪声方案等价。

## 内存所有权与预算

| 状态 | owner / 生命周期 | 当前约束 |
| --- | --- | --- |
| ready source/noise/audio | session 队列；提交前保留，提交后释放 | `max_pending_chunks` 与 `max_pending_bytes`；独立 storage |
| 当前 FP32 visual/audio、layout | 当前 step，失败丢弃或成功转出 | 单 chunk 工作集，另含固定 prefix；不能把 queue byte budget 当作总显存预算 |
| 静态 reference 和 prompt 快照 | session，dispose 释放 | references 受 #111 输入限制；编码后大小与 prefix KV 需部署单独测量；最多 latest/applied 两个 prompt 快照 |
| clean pre-RoPE KV 与 positions | session cache；full-history，dispose 释放 | **线性增长，无 eviction**；不受 ready queue 的 byte budget 限制 |
| 绝对 RoPE 网格、history 拼接、attention workspace | 每次 forward 的临时 owner | 当前网格构造和 attention 临时数据仍可随历史增长；未声明平台期 |
| provenance / spans | ready 队列及当前 result；session 只留绝对游标与最大依赖水位 | 队列条目数受限，具体 metadata 字节需 #111 admission 限制；不在 session 存全量结果历史 |
| codec 像素、moments、overlap | #112 codec owner | 与 AR 分别预算；17 pixel 最小 source credit 与 decoder 12 latent 容量见 #120 文档 |
| 输入消息、输出 result/decoded pixels、待发送包 | #111/#112/#114 | 下游必须有界接管后才能 consumed；本接口没有无限结果队列 |
| legacy 全量 visual/audio/layout/spans | 有限 #103 session | 保留可运行 full-history 对照；仍随请求总长增长 |

`memory_snapshot()` 按底层 storage 返回 ready、KV、静态引用的保留字节和条目计数；
不含 allocator reserved bytes、kernel workspace、外部输出和进程共享模型，不能用它替代 GPU/host 峰值测量。
32-chunk CPU 测试明确验证 KV 持续增长，以及正常结束/取消后的引用释放，没有把该测试解释为有界会话验收。

## #90 / #110 后续边界和验收

已查询关联讨论与 open PR：#90 为独立 SP PR；未发现公开的独立 chunk eviction / relative RoPE PR。
具体后续负责人仍需在 #90 指定，本任务不重复实现其算法。

`sink=3`、`window=21` 单位是 temporal chunk；21 是否包含 sink/current 未确认。
仅可用“3 sink + 17 recent + 1 current”作讨论示例，不能转成默认配置或 `num_frame_per_block=21`。
未来 eviction 必须以完整 source+target/audio chunk 为单位，旧历史在成功提交前不得丢弃。
全局 chunk ID、pixel/latent 范围和 provenance 不重编号；block-relative RoPE 是另一层 attention 位置映射。
若映射改变，#110 的旋转后 K 必须重新旋转、失效重建或保留 pre-RoPE 权威数据，不能直接复用。

合入生产前仍需：在线 source/target noise 决策、codec→chunk 重分批与 provenance/dependency bridge、
下游消费事务，以及持续会话的内存 admission/释放实测。有限 10 秒请求的 checkpoint/native/SP1/SP2 验证见下节。
语义明确后再以 30+ 实际 temporal chunks 验证 sink 保留、淘汰 ID、异常清理和长时 GPU/host 内存；
窗口策略相对 full-history 的画质验收，与同策略 native/SP 的数值对齐分别设门槛。


## 2026-09-20 AR step 3500 四步有限请求验收

执行候选 `fbfee6c92460d4adbfec1fb22825b22d93ae0d07`，原生基线
`97d60848db3a4364c2e816d1362ff3330d046391`。按用户指定使用现有 AR step 3500，
`few_step` schedule `[1000,750,500,250]`、seed 42、CFG 1、FA4；实际 source 视频
10 秒、240 帧，输出 896×512、24 fps。该 checkpoint 分块为 scalar 5，72 个 latent 帧
共 15 个 temporal chunks（最后一块 2 帧）。这不是 DMD checkpoint 质量证明。

正式编码修复保持原生 `encode_tile_batch_size=15` 的 tile-major、同形状跨 clip 批次，
并在编码作用域内将空间 reflection padding 等价降维，绕过大张量 3D padding 索引限制。
先前逐 clip 编码造成 BF16 source latent 差异；只替换 source 的诊断以及独立批次编码实验
分别隔离和确认了根因。最终候选独立编码输入，不替换成 native 输入；外部采样 harness
只用于同请求 full-history 与 source-ready 对照和结果记录。

| 验收 | 实际证据 | 结果 |
| --- | --- | --- |
| SP1 | `sp1-r3` / `dlc1gvo0eiue0gcz` | Succeeded，exit 0；15 chunks 视频/音频 latent exact |
| SP2 | `sp2-r3` / `dlc1edvvnuy52q8w` | Succeeded，exit 0；rank 0/1 均 15 chunks exact |
| SP1 → SP2 | `sp1-vs-sp2.json` | 视频 latent、解码音频 max_abs=0、relative RMSE=0 |
| native → SP1、SP2 | 两份 `native-vs-sp*/comparison.json` | 视频 latent 和全部 240 帧画面 exact；解码音频 max_abs=0.0073115602、relative RMSE=0.0032173190 |
| 媒体与结束释放 | `media-check.json`、chunk memory snapshots | 240 帧完整解码；EOS 后 KV/static storage 为 0 |

原门槛未放宽：视频 latent max_abs ≤ 0.03、relative RMSE ≤ 0.01；音频 max_abs ≤ 0.1、
relative RMSE ≤ 0.05。native 媒体完整解码复用已成功 `input-diag-r1` 的证据；基线复用核验
请求相等、FA4 实际调用完成及 reference SHA256
`661b4e5737a60e54937915b7ae33c42bab41aa7413d1c4bfedb70061ccf84b4f`。

报告已从 OSS 回读核验：桶 `leap-worldmodel-thailand`，前缀
`world-model/sglang/evals/zing-h3-dmd/zing113-fourstep-sp1-r3/` 和
`world-model/sglang/evals/zing-h3-dmd/zing113-fourstep-sp2-r3/`。
候选 archive SHA256 为 `24d5c92aa3f44151252573794079f4c617c279e1f24a9003d06f64d30cff90f0`。
新增 12 项 CPU 编码顺序/padding 回归通过；本机通过隔离模块加载运行，普通测试收集受无关 GPU
导入限制。此前 199 项 CPU 检查属于先前候选，不将其写成最终候选全量重跑或 CI 通过。

本轮有限请求技术验收通过。source-ready 的 pixel provenance 仍为 synthetic fixture，
不代表 codec bridge 已验证；posterior/initial noise 仍按完整有限请求生成，未定义在线未知 T
的等价 RNG 合约；full-history KV 仍线性增长，未开启窗口淘汰，也未证明分钟级有界内存。
旧 `sp1-r1` 因 source 字段层级错误没有实际 source，不能计入通过证据。


### 合并上游后的验证

已同步 main 的 FA4 默认配置/输出投影重构与 #118 的共享类型模块迁移，移除旧 entrypoint
协议原型。上文 GPU 证据对应明确记录的候选，不冒称合并后新提交已重跑 GPU。
已同步合入 main 的 #103，实现 diff 不再包含历史 benchmark/计时改动。
本 PR 直接包含并使用 #118 的共享类型，目标改为 main；合并后的 305 项相关 CPU 测试通过，最终 GPU 回归也已完成，见下节。


### 最终 main 候选 r5 回归（2026-09-21 核验）

执行候选 `f06c0ebb659906065d9333ac2dafcf594db9a5ad`，archive SHA256
`dc85dc628e48f188da6da802d33443afa997d98b60c10a179af5b1383e2799c5`。
本记录之后的文档提交不改变执行代码。SP1 `dlc16omc5mknj4f0` 与 SP2
`dlct3ag3uz13wkwx` 均 Succeeded / exit 0，全部 OSS 报告已下载回读：

- SP1、SP2 两个 rank 均完成 15 chunks，full-vs-chunk 视频/音频 latent exact；EOS KV/static 清零。
- SP1→SP2 视频 latent 与解码音频 max_abs=0、relative RMSE=0。
- native→SP1 和 SP2 视频 latent、全部 240 帧画面 exact；音频 max_abs=0.0073115602、relative RMSE=0.0032173190，满足原门槛。
- 两份生成视频均完整解码、240 帧；native 基线请求/hash/backend 与既有完整解码证据均已核验。

报告前缀为同一 OSS 桶下 `world-model/sglang/evals/zing-h3-dmd/zing113-fourstep-sp1-r5/`
及 `zing113-fourstep-sp2-r5/`。跨 SP 检查显式断言音频 shape、finite 与两项误差，
避免旧检查器的可选字段缺失绕过阈值。所有阈值未放宽。

本轮有限请求合并验收完成；PR 指向 main、非 Draft、无冲突。GitHub 未上报 CI checks，
不把本地/云端测试写成 CI 通过。上述在线 noise、codec bridge、窗口/有界内存及 DMD 质量边界不变。

## 失败语义简化后的 r7 验收（2026-09-21）

执行候选 `666b1441d21f77813b74879694da8d22e76468bc`，archive SHA256
`8a86195e769f6bd2f6981690cf568ce1beb10bda60b0239e4efa6ed8506450ab`。
移除整块事务回滚和原样重试快照；失败即关闭状态并释放 KV、ready 队列、prompt、会话 RNG。
保留正常会话随机流隔离；EOS 清理不能将未提交的块误报为完成。302 项 Zing 相关 CPU 测试
通过，包括打包/prefix/source/active/final/提交后失败清理、拒绝重试、正常分块数值一致与交错
会话随机流隔离。Ruff、diff 检查通过。

- SP1 `sp1-r7` / `dlcf3vh8qroitsn0`：Succeeded / exit 0，15 chunks 视频/音频 latent exact。
- SP2 `sp2-r7` / `dlc90jv2icuprbdf`：Succeeded / exit 0，rank 0/1 均 15 chunks exact。
- SP1→SP2 视频 latent 与解码音频 max_abs=0、relative RMSE=0。
- native→SP1/SP2 视频 latent、全部 240 帧画面 exact；音频 max_abs=0.0073115602、
  relative RMSE=0.0032173190，满足原门槛（视频 .03/.01；音频 .1/.05）。
- 两份媒体均完整解码、240 帧，EOS KV/static storage 清零。原生基线请求、FA4 完成证明及
  reference SHA256 与前述基线一致，native 完整媒体解码复用已成功的 input-diag-r1。

全部报告从 OSS 桶 `leap-worldmodel-thailand` 回读核验，前缀为
`world-model/sglang/evals/zing-h3-dmd/zing113-fourstep-sp1-r7/` 与
`world-model/sglang/evals/zing-h3-dmd/zing113-fourstep-sp2-r7/`。
固定 AR3500 四步、seed42、CFG1、FA4、真实10秒240帧 source，未放宽阈值。
本记录是文档更新，不改变已测试执行代码；旧 r5 仅作修改前历史证据。
本轮有限请求验收通过，不自动合并。上述 DMD 质量、codec bridge、在线未知 T 噪声、
窗口淘汰/有界内存限制不变；GitHub 未上报 CI checks，不声称 CI 通过。
