# Zing 联合 forward（#229）

评审建议从 [七项材料与代码/复现入口](REVIEW_GUIDE.md) 开始。

用户入口 `zing_joint_forward` 和在线 profile 缺省 `joint_forward` 现默认
`cross_chunk`（2026-10-09 按评审意见调整），仍可显式设 `off` 或 `source_first`。
低层 sampler 构造器保留 `off`，由入口显式传入策略，便于原路径对照。
动态优先三分支，不满足时尝试两分支；两分支也不适用就回退原路径。
在线 `joint_wait_ms` 缺省为 `"auto"`，不再接受显式 0。

后续性能与优化验收默认使用 WebSocket 连续像素输入、渐进 source encoder 和
真实 source + 六图；离线 generate 与算子测试仅作明确标注的补充诊断。
在线 Q/窗口、编码开关及补测说明见 [ONLINE_SP2_RETEST.md](ONLINE_SP2_RETEST.md)。

历史验收：真实 SP1/SP2 的完整 latent 与媒体对照通过。**只有 SP1 window3/sink1 那轮没有端到端净收益；SP2 在线有正收益**，旧确定性配置全程 -5.79%，新 performance 配置 -6.10%（有数值限制）。
真实 SP2 流式五档、逐层 KV/step0 与媒体对照通过；独立双 rank profiler 归因已完成。
默认策略已按用户要求调整；历史 GPU 结果未覆盖新的自动等待策略，不能当作新默认的 GPU 验收。SP4 和音视频仍不声明联合支持。
下方失败记录保留当时的源码和结果，不能覆盖后续通过结果，也不能从离线通过推断流式通过。

- `source_first` 将同块 source 和 target step0 的 QKV、output projection、
  AdaLN 后的逐行运算及 MLP 放到同一层遍历。两个无 mask attention 分别读取
  H+S、H+S+T；块内双向，不使用 token causal mask。
- `cross_chunk` 在离线 sampler 中联合上一块 clean final、下一块 source 和
  step0。各层三次 attention 分别读取 H+F、H+F+S、H+F+S+T。
  末块仍执行 final。初始 noise 已由原入口准备；re-noise 调用顺序不变。
- 流式 `step()` 不保留未提交 final，不等未来 source；`cross_chunk` 在此
  回退到 `source_first`，记录 `streaming_boundary`，提交后才交付当前块。
  prompt 更新仍发生在已提交边界。连续 source 服务使用下述有界批次入口。

联合事务先验证全部 block、role、timestep、active geometry 和 clean 边界，
仅允许 clean 分支持久提交。raw cache 的既有 scratch 存放本次 K/V，noisy
rows 位于逻辑持久长度之外。所有模型计算（包括输出 head）成功后按序提交；
执行或提交异常清空会话，不复用半提交状态。每个分支保留独立 modality 和
实际 timestep，因此不会覆盖 source AdaLN 权重。

每个分支的顺序执行 RoPE map 必须与联合 map 相同。跨块遇到窗口淘汰（包括
sink 策略）保留独立 final，再 `prepare_chunk`，再尝试 source_first；不构造
错误的共用历史。不同 RoPE 视图回退，首版不实现多视图缓存。CFG、音频、legacy
frame window、post-RoPE single store 和不足的 pooled scratch 回退到原路径。
未验证带梯度训练、reference/target overlap；本入口本身是 no-grad 推理。
实际分支由 `forward_counts`、`joint_fallbacks` 记录，`joint_row_shapes` 按
成功联合调用记录各分支 token 行数（例如 `880+390`）及次数；默认 off 的原 metrics
结构不变。

当前仍执行 clean 分支输出 head，最终 sampler 仅消费最后 target 的预测；
不将省略 head 计入收益。当前 KV payload/attention 工作量不因联合而减少，
各 attention 仍重复读历史。SP 对联合 token rows 均分，输入/输出各一次交换；
不按 frame 分配 rank。

## 验收状态

CPU 小模型覆盖 [1,2,2]、尾块、窗口首次/后续淘汰、sink、逐层 raw KV、分支
可见性、最终 latent、失败清理、re-noise RNG 顺序及 SP2 布局。
CPU 测试不代表 FA4、真实权重、媒体质量、GPU 性能或 SP4 验收通过。
GPU 与真实业务验收结果须单独记录，未完成前 PR 保持 draft。

```sh
SGLANG_LIGHTWEIGHT_RUNTIME=1 PYTHONPATH=python OMP_NUM_THREADS=1 python - <<'PY'
import sys
if sys.platform == 'darwin':
    from sglang._platform_stubs import install_platform_stubs
    install_platform_stubs()
import sglang.multimodal_gen.runtime.models.dits.zing_h3
import pytest
raise SystemExit(pytest.main(['-q',
    'python/sglang/multimodal_gen/test/unit/test_zing_h3_joint.py']))
PY
```

### 真实 SP2 诊断（2026-09-29，未通过）

冻结计算源码 `4d552bbd1d24abafc5ca3c73f0b1a3e84725ebdb`，作业
`dlc1d5r9qizndik1`，两张实际报告 274113 MiB、CC 10.3 的 GPU。
使用原始 source + 六图，120 帧 832×480、四步 few-step、CFG=1、seed=42、
window=21/sink=3、block-relative RoPE；权重 63 个文件已在零 GPU 作业中记录 SHA256。

原路径三次正式 latent 完全一致。source_first 相对原路径最大绝对误差为
4.5173，相对 RMSE 为 0.5041，未通过预先冻结的 0.03 / 0.01 门槛。
作业因此停止，没有执行 cross_chunk 或 SP1 后续组；不能据此声称质量或准入通过。

| SP2 模式 | API 中位数/秒 | DiT 中位数/秒 | 主干次数（含 prefix） | 峰值显存/MiB |
| --- | ---: | ---: | ---: | ---: |
| off | 32.3549 | 11.4249 | 133 | 236370 |
| source_first | 31.2994 | 10.2373 | 111 | 236604 |

以上时间来自一轮完整预热后的三次无 profiler 运行。单独的 profiler 不在计时中。
数值不通过使这些时间只具有诊断意义，不能作为优化收益验收。
前一轮 SP1 原窗口基线（`dlc149neoa1gks8z`）在 source KV 扩容时 OOM，
也没有有效联合结果。所有失败结果保留，不放宽数值阈值。

后续受控诊断让联合分支的时间嵌入及紧凑 AdaLN 投影保持原路径的单 timestep
批形状，主干 token 投影仍联合执行。CPU 回归通过不证明该假设已在真实模型上成立。
该次实验时，配置化的流式有界等待、真实逐层数值与完整媒体/性能验收尚未完成。

受控诊断 `dlc1ifz096hb0o9z` 中，新进程 off 与前述基线仍完全一致；保持单
timestep 投影后的 source_first 仍有 max_abs=4.3985、rrmse=0.5010。
该假设没有解决问题，相关投影改动已撤回。零 GPU 分析还确认两组各自的三次
输出均完全一致，原联合路径首个 latent 帧已出现 rrmse=0.0795，后续继续增大。
下一步须定位首块逐层差异，不能把此问题解释成允许的 BF16 舍入或放宽门槛。

首块逐层诊断 `dlc14krqiyc7rvau`（两 GPU，计算源码 `2fb94b8ad9` 加仅用于
抓取张量的临时覆盖）已完成；零 GPU 作业 `dlc19kkfeyp76k8b` 完成比对。
该诊断不运行完整生成，不计性能：输入 latent、timestep、位置、RoPE 系数和
patch projection 的 hidden 在两路中完全相同。第 0 层 AdaLN gate 和调制后的
hidden 已出现差异，在 QKV 投影与通信之前。第 49 层 target hidden 的相对
RMSE 为 0.0250，首步输出预测为 0.01068（max_abs=0.08835）。这只定位了
最早分歧，不证明已找到全部误差来源；之前单独保持 scalar timestep 投影的
完整生成仍失败。

实际首块 source/target 分别为 880/390 个 token，因此原路径 SP2 本地矩阵
分别有 440/195 行，联合路径为 635 行。不能直接套用两路均为 390 行的估算。
后续单算子复算使用保存的真实输入区分紧凑投影与主干 GEMM 形状的影响。
逐层张量保留在私有 CPFS，分析 JSON 位于私有 OSS
`world-model/results/zing-joint-229-r4/layer-analysis.json`，未发布匿名访问链接。

首层单算子复算 `dlcplja4qnsuaec0`（单 GPU）保持原路径保存的调制输入不变，
分别按 440/440/195/195 行和联合 635/635 行运行真实 QKV 权重：两者与保存
结果均逐位一致，BF16 reduced-precision reduction 开关两种设置均相同。
这排除了该层 QKV 行数变化本身；不能据此排除其他层或 MLP 的形状影响。

进一步诊断 `dlc1igegi2509yr1` 保持 scalar timestep 投影后，首块所有 50 层
hidden 均与原路径逐位一致（CPU 比对作业 `dlc1mw7ymiot2b7i`）。最终 target
预测仍有 max_abs=2.8610e-6，差异发生在 FP32 输出 head 的合批形状上。
当前修复同时保留单 timestep 的紧凑投影和每个分支的输出 head 形状；主干
QKV/output projection/MLP 与 SP 交换仍逐层联合。此前仅修复 timestep 的
失败记录仍有效。该组合修复随后通过下述 r6/r7 完整生成复验；首块诊断本身不作为完整验收。

## SP2 离线完整生成验收 r6（2026-09-29）

冻结计算源码 `205866937bea304b83cacfef9b4186b53e4eddfe`；GPU
`dlc10fagwa4wxqia` 与 CPU 导出作业 `dlc15z2cjunw12u9` 均成功。
固定原始 source + 六图、window=21/sink=3、四步、CFG=1、seed=42。
每组一次预热、三次无 profiler 正式运行。off 的全部重复与 r2 off 参考逐位一致；
两种联合模式各三份完整 latent 也与 off 逐位一致，max_abs=0、relative RMSE=0。
三组首个 MP4 的 SHA256 均为
`bc7200abdb12024b63c6fa6b01b0ac3985cd11d73f54af497ddb1e3fd6897bd4`。

| 模式 | API 中位数/秒 | API 范围/秒 | DiT 中位数/秒 | 峰值/MiB | forward 次数 |
|---|---:|---:|---:|---:|---:|
| off | 32.2676 | 32.1382-32.3442 | 11.3994 | 236370 | 133 |
| source_first | 31.5124 | 31.1094-34.7639 | 10.2889 | 236592 | 111 |
| cross_chunk | 30.1868 | 30.0180-30.4163 | 9.1126 | 236072 | 91 |

source_first 有一个慢样本，不能仅据三次运行宣称收益稳定。cross_chunk 包含
20 次三分支、2 次 source_first、2 次 final，以及一次 window_eviction 回退；
含 prefix 共 91 次 forward，不是无回退公式得到的 90 次。
私有证据位于 `world-model/results/zing-joint-229-r6/`，原始张量保留在 CPFS
同名实验目录。这是离线纯视频结果，不代表在线或音频验收。

## 在线有界等待与跨块合并

连续 source 部署 profile 的 `joint_forward` 缺省为 `cross_chunk`，
`joint_wait_ms` 缺省为 `"auto"`；无需用户填写毫秒数。兼容显式正数上限
`0 < joint_wait_ms <= 1000`，但不再接受 0；旧部署请删除该字段或改为 `"auto"`。
显式 `off` 可关闭联合执行，不需要填写等待参数。

自动等待读取 checkpoint 的 chunk plan（包括首 latent 拆分），寻找下一对
chunk 所需的 source 水位；按 progressive clip17 的 1/9/17 像素→1/3/5 latent
映射计算，并受单次 dispatch 的一个 clip 上限约束。只为尚缺的 source 帧按
输入 FPS 估算等待预算，最高 1000 ms；不是再睡一整段 DiT 推理耗时。
例如 `[1,2,2]` 首次已有 1 帧时，等到第 9 帧即可准备前两个 chunk，24 FPS
下预算约 333 ms；已经到齐则立即执行。跨 clip 或容量不允许时仍可能无法三合一。
clip17 整段编码基线保留首像素即时 presentation，避免阻塞条件准备。

prompt 更新、EOS 或取消立即中断等待，超时也继续推进。`zing_joint_wait`
记录 policy、目标帧数、预算、实际 host 等待和原因；它不是 GPU 完成时间。
自动等待只是提高合并机会，不保证每块三合一，也不代表实时 24 FPS 达标。

worker 在既有字节预算内至多合并两个已就绪的 AR chunk。它可继续编码已缓冲的
切片来获得第二块，但不会在 worker 内等待网络输入。`drain_ready()` 只在这次同步
批次内保存下一块的 step0 预测。上一块 final 先提交，再交给 decoder；每次解码前
释放 transformer 的使用权。解码失败或取消后关闭迭代器，会清空包括预取状态在内
的整段会话。执行 drain 时拒绝外部 append/prompt/step，预取不跨命令或 prompt
边界保留。队列容量、pooled scratch、窗口淘汰和不兼容视图均有回退。末块始终提交；
初始 noise 与 re-noise 顺序不变。跨 decoder yield 保留的执行张量单独计入
`execution_storage_bytes`。

CPU 测试覆盖 [1,2,2] 已就绪批次的数值/RNG 一致性、短 EOS、窗口淘汰、迭代器关闭、
worker 字节容量回退及异步等待中断。真实在线验收见下方“最终流式验收”；实现或 CPU
测试本身不构成在线性能证据。

## SP1 离线完整生成验收 r7

GPU `dlc1n6k74a84gpkx` 成功，冻结源码 `428e69edbe`。所有组均采用
window=3/sink=1、text encoder offload、同一 source + 六图、四步和 seed。
这是独立的小显存配置，不能用来比较 SP1/SP2 扩展效率。两种联合模式各三份 latent
与各自 off 基线逐位一致，各组首个 MP4 哈希也一致。

| 模式 | API 中位数/秒 | API 范围/秒 | DiT 中位数/秒 | 峰值/MiB | forward 次数 |
|---|---:|---:|---:|---:|---:|
| off | 41.5850 | 41.4942-41.6257 | 7.7142 | 136462 | 133 |
| source_first | 41.6381 | 41.4838-41.6744 | 7.5007 | 136464 | 111 |
| cross_chunk | 41.7769 | 41.6815-41.7915 | 7.5235 | 136464 | 109 |

私有证据：`world-model/results/zing-joint-229-r7/`。

## SP2 独立 profiler：r8

计算源码 `a2f953c8bb`，GPU 作业 `dlc3518nevome4vg` 已成功。每组先完整
预热和三次无 profiler 测量，再单独采集两个 rank；不能用以下 kernel 累计或
CPU 发射间隙替代正常时延。详细分析作业 `dlc1dr6fiicj0bfp` 已完成六份 trace。

两个 rank 的调用次数和显式 copy 字节统计一致：

| 每 rank 指标 | off | source_first | cross_chunk |
| --- | ---: | ---: | ---: |
| forward（含 prefix） | 133 | 111 | 91 |
| Ulysses input / output（各自） | 6600 | 5500 | 4500 |
| 主干 attention | 6650 | 6650 | 6650 |
| prefix token-refiner attention | 2 | 2 | 2 |
| history K RoPE rebuild scope | 2200 | 2200 | 1300 |
| input exchange 内显式 copy 字节 | 219082752000 | 219082752000 | 219082752000 |
| output exchange 内显式 copy 字节 | 36513792000 | 36513792000 | 36513792000 |
| cached_KV_append 内显式 copy 字节 | 106024211552 | 104762643552 | 103644435552 |
| 外层 RoPE_and_KV_append 内显式 copy 字节 | 16386048000 | 21417984000 | 21417984000 |

copy 统计来自 `aten::copy_` 已知 dtype 的目标 shape；按最近的 CPU scope
互斥归类，包含 scope 内元数据，不含 fused kernel 写入，不是全部 HBM 流量。
外层 raw K staging 的增加也计入，不能只选下降的 scope 声称 KV 拷贝整体收益。
按实际行数推导的每 rank 非 prefix 逻辑 QKV payload 为 109541376000 字节，
output 为 36513792000 字节，三组相同；包含逻辑交换中的本 rank 部分，不是
网卡实测字节。三组总处理行数均为 101880，attention 有效工作量不因合批减少。

主干四个 BF16 GEMM 分别为 `[M,5376]×[5376,21504]`、
`[M,7168]×[7168,5376]`、`[M,5376]×[5376,28672]` 和
`[M,14336]×[14336,5376]`，每个 forward 各 50 次。trace 的本地 M：

| 分支或转换 | SP2 本地 M | 成功 forward 数 |
| --- | ---: | ---: |
| off source：1 / 2 latent | 440 / 880 | 8 / 14 |
| off active：1 / 2 latent | 195 / 390 | 32 / 56 |
| source_first：1 / 2 latent | 635 / 1270 | 8 / 14 |
| cross 1→2：390+1760+780 全局行 | 1465 | 7 |
| cross 2→2：780+1760+780 全局行 | 1660 | 7 |
| cross 2→1：780+880+390 全局行 | 1025 | 6 |

cross 另有两次 M=635 的 source_first，以及两个独立 final。prefix 主干
M=12246，复制到每个 rank；两层 token-refiner 不混入上述主干 GEMM 计数。
保留单 timestep 的紧凑 AdaLN 和逐分支 FP32 输出 head，未把这些操作合批。

由冻结 window=21/sink=3 策略和实际行数推导：第 21 个 chunk 前淘汰
chunk 3 的 1270 行，一次 compact 移动后续 36830 行。此行数是策略推导，
不是 GPU 计数器。不同 RoPE 视图会回退，首版没有额外多视图缓存分配。
原始与基础归因位于私有 `world-model/results/zing-joint-229-r8/`，逐 forward
shape、CPU 发射间隙及同 CUDA stream 间隙位于
`world-model/results/zing-joint-229-r8-detail/`。后两项受 profiler 扰动，
包含 host 工作或同步；同 stream 间隙也不等于整个设备空闲。

零 GPU 完整性复核 `dlcggylu2f5dgtyy` 覆盖 r6/r7/r8 全部 27 个正式 raw：
latent shape 均为 `[1,24,36,30,52]` 且全部 finite。30 个正式及 profile MP4
逐帧完整解码通过，均为 120 帧、832×480、24 FPS、5 秒；每个文件记录 SHA256。
证据为私有 `world-model/results/zing-joint-229-offline-audit/audit.json`。
这些 MP4 是 video-only，不能据此验收音频。

r8 三次正常计时（在单独 profiler 之前）及逐次 latent 对照也通过：

| 模式 | API 中位数 / 范围（秒） | DiT 中位数（秒） | 峰值 reserved MiB |
| --- | --- | ---: | ---: |
| off | 32.3960 / 32.3917–32.4485 | 11.5516 | 236370 |
| source_first | 31.2939 / 31.1205–31.3958 | 10.4019 | 236592 |
| cross_chunk | 30.5800 / 30.3172–30.7275 | 9.3963 | 236072 |

cross_chunk 的 API 中位数在 r6/r8 分别下降 6.45% / 5.61%。source_first
在 r8 下降 3.40%，但 r6 有慢样本，保留该离散度记录。独立 profile 的总耗时
包含 trace 导出，不能作为正常生成性能。逐 rank kernel 关联各保留一个未归因
随机数 kernel；不隐藏未归因项。cache compact 在外层 `Zing.denoise`，单列
成本，不混入 `Zing.forward.*` 的主干累计分母。

额外 CPU shape 审计 `dlcq6iznuc49zzns` 将完整 KV 形状
`[rows,28,128]`、BF16 的显式 copy 与元数据、局部 RoPE 写入分开：

| 每 rank 完整 KV copy 目标字节 | off | source_first | cross_chunk |
| --- | ---: | ---: | ---: |
| cached_KV_append | 96881254400 | 95619686400 | 94501478400 |
| 外层 raw K staging | 16386048000 | 21417984000 | 21417984000 |
| 两项合计 | 113267302400 | 117037670400 | 115919462400 |

所以这两项在 cross_chunk 中净增加 2652160000 字节；不能将更少的 collective
或重建次数解释为 KV 拷贝字节同比减少。prefix 的 17555865600 字节三组相同。
这些仍只统计显式 copy，不包括 fused 写入或读流量。
六份 trace 各实际观察到一次 `_zing_kv_compact_kernel`，每 rank 约
13.48–13.50 ms；结合上述独立策略推导，对应搬移 36830 行、26399744000
字节 K+V。原始 shape/dtype 直方图、kernel 时间及 trace SHA256 位于私有
`world-model/results/zing-joint-229-r8-copy/`。

按 trace 中实际重建 occurrence 与冻结缓存策略重放可见行数，off/source_first
每层累计重建历史 1528084 行，cross_chunk 为 916206 行；50 层分别为
76404200 / 45810300 行。最大单次历史均为 56306 行，最终 clean store 均为
56696 行。这里 occurrence 是实测，行数是几何与策略推导；不是设备行计数器。

| profiler 主干 kernel 累计 ms | rank | GEMM | attention | 通信（含 IPC 等待） | 全部非通信 |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | 0 | 1965.46 | 3188.01 | 6724.17 | 6450.47 |
| off | 1 | 1958.59 | 3186.21 | 17200.59 | 6529.57 |
| source_first | 0 | 1785.35 | 3186.42 | 19301.72 | 6318.86 |
| source_first | 1 | 1799.25 | 3183.99 | 904.24 | 6206.89 |
| cross_chunk | 0 | 1718.97 | 3186.33 | 16368.95 | 6159.56 |
| cross_chunk | 1 | 1733.10 | 3186.34 | 872.18 | 6075.03 |

这张表按 CPU nesting 与唯一 correlation ID 归属，不按 CPU scope 时间裁切
异步 kernel。非通信包含 GEMM 和 attention，不能把这些列相加。通信 kernel
中的等待方向随运行变化，且受 profiler 的 CPU 开销影响；它不是网络带宽统计，
也不能用这些累计时间预测无 profiler 的 wall time。

按本地 M 分组的每 forward 平均 kernel 累计与 CPU 发射间隙（各列 rank0 / rank1）：

| 分组 | 非通信 ms | attention ms（包含于前列） | CPU gap p95 的组内中位数 ms |
| --- | ---: | ---: | ---: |
| off active:M195 | 36.72 / 37.29 | 20.72 / 20.71 | 0.334 / 0.335 |
| off active:M390 | 42.30 / 42.91 | 21.18 / 21.14 | 0.335 / 0.337 |
| source_first joint:M635 | 72.52 / 71.62 | 41.19 / 41.17 | 0.352 / 0.354 |
| source_first joint:M1270 | 111.78 / 111.44 | 61.58 / 61.56 | 0.347 / 0.350 |
| cross_chunk joint:M1465 | 136.16 / 136.08 | 78.54 / 78.59 | 0.349 / 0.350 |
| cross_chunk joint:M1660 | 150.42 / 150.59 | 84.44 / 84.46 | 0.358 / 0.356 |
| cross_chunk joint:M1025 | 103.15 / 102.27 | 61.45 / 61.40 | 0.352 / 0.352 |

原始逐 forward 分布与同 stream gap 分布保留在 detail JSON，不把 p95 的中位数称为全局 p95。


## 最终流式验收（2026-09-29）

GPU `dlc1fpj1fuctc1lw` 与零 GPU 后处理 `dlc1j1e616nahc2x` 均 Succeeded。
冻结源码 `cdd9cb915fb609f2247c791bd8d607086ffd16f9`，两张上述实测设备，SP2、
FA4、stable compiled AdaLN、resident、原生 video VAE。每档独立服务进程，
完整一次预热与三次正式运行；同一个原始 source + 六图，120 帧 832×480，
客户端按 24 FPS 发送，四步 `[1000,750,500,250]`、CFG=1、seed=42、
window=21/sink=3、block-relative gap=12、indexed-cpu-v1 noise。
流式 source/target 每 latent 各 390 rows，与离线 880/390 是不同入口基线，
不声称离线与流式逐像素相同。音频关闭。

固定数值门槛：重复 RGB max-abs/relative-RMSE 为 0.006/0.002，联合 RGB
及逐层 KV/step0 为 0.03/0.01；均在实验前固定。五档各三次完整 RGB 均相对
原路径误差为零。预热诊断覆盖 block 2/4/6、双 rank 全部 50 层的 clean
source+final normalized pre-RoPE K/V（每档 300 项），以及六项 step0 预测；
联合档全部 bitwise/零误差。诊断只在预热取样，不混入正式计时。
这不是全 22 块逐层保存；完整序列由离线 latent 与流式 RGB 验收补足。

| 流式档位 | 全程秒中位数 [min,max] | 交付 FPS | TTFF 秒 | input→交付 p95 秒 | 间隔 p95 / max 秒 |
| --- | --- | ---: | ---: | ---: | ---: |
| off | 32.9279 [32.9205,32.9498] | 3.6443 | 7.0797 | 29.0778 | 4.4578 / 8.1065 |
| source_first | 31.4421 [31.4298,31.5171] | 3.8165 | 6.6273 | 27.5957 | 4.2434 / 7.9134 |
| cross 0 ms | 31.0218 [31.0076,31.2019] | 3.8683 | 6.4913 | 27.1749 | 4.1947 / 7.8537 |
| cross 50 ms | 31.3578 [31.2531,31.3720] | 3.8268 | 6.5934 | 27.5117 | 4.1995 / 7.9190 |
| cross 350 ms | 31.3463 [31.3225,31.4938] | 3.8282 | 5.8138 | 26.7909 | 4.1990 / 4.8385 |

后四列也是各次统计的中位数；p95 基于每次 120 个 source frame，间隔基于
22 个 AR 对齐交付批次，包含 codec 突发。原始逐帧和逐批次记录均保留。
这是 localhost WebSocket 客户端收到帧的时间，不包含浏览器呈现或 WAN；
没有测得真正 input-to-visible，也不据此宣称达到 24 FPS 或外部延迟 SLA。
零等待 cross 相对 off 吞吐提高约 6.15%，全程时间下降约 5.79%。

各 cross 档正式运行均 104 次主干（7 次三段、15 次 source_first、15 次 final、
66 次独立 denoise 和 prefix）。0/50 ms 的三段均为 `780+780+780`；
350 ms 为 `390+780+780` 一次和 `780+390+390` 六次，并有一次窗口淘汰回退。
因此较长等待改变输入缓冲与配对边界：350 ms 改善本例 TTFF 和最大交付间隔，
没有增加三段次数或获得额外吞吐。每次 50/350 ms 均一次 timeout、四次 ready、
三次 EOS；实际最长等待的三次范围分别 49.9478–50.0909 / 349.6104–350.2402 ms。
这些上限是调度预算而非硬实时保证。其余等待接近零。默认等待仍为 0 ms。

| 档位 | 双 rank / 三次最大 allocated MiB | 最大 reserved MiB |
| --- | ---: | ---: |
| off | 183644.03 | 184174 |
| source_first | 183665.36 | 184174 |
| cross 0 ms | 183620.26 | 184192 |
| cross 50 ms | 183620.26 | 184192 |
| cross 350 ms | 183614.29 | 184292 |

未 OOM；reserved 包含预热分配池，单会话通过不保证任意并发容量。
server stage_host_ms 是 host scope，不是同步 GPU 耗时。冷启动不在上述 TTFF 内。
首次启动发现镜像 uvicorn/websockets 超出仓库支持范围，未运行模型；实验环境固定
uvicorn 0.35.0、websockets 15.0.1 后重跑，保留失败日志，没有绕过 transport 检查。

### 可复查材料与限制

全部结果仅保留私有 OSS/CPFS，未发布公网报告。桶为
`leap-worldmodel-thailand-single-az`，完整流式日志、配置、逐帧事件、逐层诊断和
五个首轮 MP4 位于 `world-model/results/zing-joint-229-stream-r2/`。
离线 r6/r7/r8 的 27 个正式 latent 全部 finite、shape `[1,24,36,30,52]`；
30 个导出 MP4 均完整解码为 120 帧，审计位于
`world-model/results/zing-joint-229-offline-audit/audit.json`。
流式五个 MP4 也完整解码为 120 帧，媒体及曲线审计在下述复查包中。

复查包 `world-model/results/zing-joint-229-review/` 包含等待曲线 PNG/SVG、
完整汇总 JSON、分析/复现脚本、六个原始 trace 的压缩/解压 SHA256 manifest、
Perfetto 截图和 overview。原始六份 trace 仍在 r8 前缀。
使用 [Perfetto 原生大文件模式](https://perfetto.dev/docs/visualization/large-traces)
在本机加载 cross rank0 并实际查看 CPU joint 与 CUDA 时间线；文件未上传 Perfetto。
原始深层 Python trace 导入有 8588 个 depth-limit 丢弃与 121346 个 overlap spill，
所以不能把 UI 当完整计数来源。另提供仅保留全部 CUDA kernel/memcpy/memset、
CPU Zing annotation 与 metadata 的 overview；没有 depth-limit 丢弃，但仍有
16908 个 overlap spill（溢出轨道，不删除事件）。所有量化直接解析原始 JSON。

最终核心源码的 50 个 joint CPU 测试一次执行全部通过；此前 170 个联合/增量/
host-row/adapter 回归，以及 620 个既有 worker 测试和两个新增队列容量测试通过。
这些计数有重叠，不累加。仓库 `taeh3-contracts` CI 是独立 CPU 检查，不充当 GPU CI。
SP4、CFG>1、音视频由显式 eligibility 检查回退旧路径。训练及
reference/target overlap 未取得联合验收结论，不应配置启用本开关；本入口为 no-grad。原生 Zing 仓库的实现/验收不由本 PR 代替。当前准入范围是默认关闭的
推理优化，生产默认变更需要单独决策与目标延迟标准。


## 2026-10-09 attention 默认选择与评审问答

`SGLANG_ZING_JOINT_ATTENTION=auto|separate`，缺省 `auto`。`auto` 在 SM103、
BF16、D128、28 heads、历史 12246–55426 和已验证离线分支长度家族内，使用
FA4/CuTe 分段 mask + full/partial/empty 跳块；其余保留逐分支 FA4。未知架构、
在线形状、短历史或可选 sparse 接口缺失均不强行融合。`separate` 用于 A/B 或回滚。
新 runtime 直接接入已测 kernel，不再需要实验 sitecustomize；最多缓存 64 组元数据，
按 CUDA stream 隔离，捕获期间不创建特化。kernel 执行失败直接上报，不吞掉 CUDA 错误。
当前在线 `(390,390)/(780,780)/(780,780,780)` 尚无专用融合 GPU 证据，不能称其最优。

保留分支 timestep 并不意味着三者始终不同：clean final/source 为 1，step0 使用
checkpoint 的采样时间，四步网格首项归一化后也可能为 1。必须按各自语义传值，
不能把更一般的网格硬编码成同一 timestep；modality/AdaLN 分支也不能混用。
原实现还保留单 timestep 投影与逐分支 FP32 head 的 GEMM 形状，以避免已经观察到
的舍入差异沿 denoise 放大。合并主干不是合并采样状态或减少四步。

进一步优化应分别预算：主干 projection/MLP 可检查小 GEMM 利用率、布局转换和
残差/归一化融合；Ulysses 可检查通信与计算重叠及布局拷贝，但当前有效 token
payload 未随 launch 次数同比减少。低精度或省略 clean head 属于独立正确性变更，
不能捆绑为本次评审修改的已测收益。现有在线 decoder 约 17.2 秒，大幅端到端
改善应同时评估这个最大单项；不从“三合一”推算三倍速度。

本次修改的 CPU/选择策略回归单列；2026-10-08 的性能表和数值差异仍对应原冻结
源码，未将自动等待或新 runtime dispatch 冒充已跑过的 GPU 结果。
