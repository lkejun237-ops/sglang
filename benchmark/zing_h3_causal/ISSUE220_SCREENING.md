# #220：复用 PR #184 的低成本复验

本轮保留 #158 的历史失败结论，复用同一 strided writer；不会将旧 kernel
microbenchmark 的 2.1–2.2× 当作 denoise 或 API 收益。实现默认关闭。

## 冻结与分阶段预算

- 基线：2026-09-28 的 main `4da9d907d76b2a9627e18db7cd05d1d84bcecba0`。
  PR #184 合入该 main 后，A 使用 main 原始 writer/guard 两个文件，B 使用
  开启 strided writer 的候选文件；其余运行时代码和测量 hook 相同。避免把
  候选关闭路径新增的检查成本纳入基线。
- 使用原真实 source + 六图、ODE step1000、seed42、四步采样、video-only、
  indexed noise、window21/sink3/block_relative/gap12、resident FA4。
  history RoPE/chunk reuse 开关在 A/B 中保持相同，记录解析值。
- 输入 source 原始为 1280×720、24fps、120 帧（5 秒）；240 帧测试循环两次。
  这不是独立 10 秒源片，也不代表连续 60 秒场景质量通过。
- 512 档：896×512。720p 档须先确定网格对齐：当前输出要求 32 倍数，
  1280×720 不可直接作为该 source-stream 的输出几何；候选 1280×736，
  必须在运行回执和报告标注实际值，不称为精确 720p。
- 第一步：零 GPU 准备、现有资源/资产核验、CPU 回归；不重新复制权重。
- 第二步：最多 2 GPU 的 SP2 初筛，同设备串行独立 A/B 进程。每进程每尺寸
  一次完整 240 帧预热、三次无 profiler 240 帧计时、一次独立正确性请求。
  两尺寸共 20 次请求；不重复已有 10/15/60 秒全矩阵，不先采集完整 profiler。
  2 GPU 作业最多 90 分钟，实际执行结束即释放。
- correctness 请求逐 writer 检查 raw K 不变、旋转 K 和 V 精确一致，记录
  每 rank/phase/shape/stride 的路由；计时请求不做 per-write 日志/计数/拷贝。
  各 rank 记录 denoise、DiT、API wall、block p50/p95、峰值显存及释放回执。
  基线/候选的真实 DiT 输入输出与 RGB 比较单列，不靠相同 seed 宣称一致。
- 先完成小尺寸再大尺寸；OOM、数值错误或非法 shape 保留现场，停止依赖步骤。
  不自动降分辨率、改后端、启用 offload 或改窗口掩盖失败。
- SP1 回退和 SP2 collective 使用已有轻量 CUDA harness。SP4 的最新树验证
  单独调度最短充分 4 GPU 作业；仅在 SP2 显存不足或有扩展价值时扩大高分辨率
  性能矩阵，禁止把 SP2 初筛写成 SP4 已验收。

## 初筛与正式门槛

本轮初筛不构成合并验收。所有重复完整保留；若任一变体
`(max-min)/median > 10%`，标记不稳定，不能挑最快样本。
writer-only denoise 中位数改善 ≥2% 且正确性/显存通过，才追加独立反向顺序
与充分重复，沿用 #158 的 ≥2% 及置信区间门槛。没有信号即停止扩大该格子的
重复样本，默认继续关闭；不能据三次样本证明收益为零。

#219 metadata-only、writer-only 和 combined 必须独立消融；在 metadata 候选
尚未冻结前先做 writer-only，不归因尚未实测的组合收益。#221 负责 CPU 调度
与 rank 因果诊断，本任务不重复实施调度干预。

GPU 作业只挂 CPFS。输入准备、媒体编码/完整解码、SHA 验收和汇总在本机或
零 GPU 作业完成。完整源码、实验 hook、配置及脚本另存不可变运行清单；
hook 不进入生产实现。云平台成功、测试完整、媒体有效、数值一致、肉眼画质
和性能门槛分别报告。

## 2026-09-28 首轮结果

896×512 的 SP2 初筛完成，writer-only denoise 改善 **1.84%**，未达到预设
≥2% 的扩展门槛，停止追加该尺寸重复样本。三次样本不足以证明收益为零或
确认稳定收益；本轮没有反向顺序验证，也没有新的 SP4 性能结果。默认继续关闭，
PR #184 保持 Draft，#158 的历史 no-go 结论不改写。

| 指标（240 帧，中位数） | main 基线 | writer 开启 | 耗时减少 |
| --- | ---: | ---: | ---: |
| denoise CUDA event 区间 | 24.943611 s | 24.484821 s | 1.839% |
| DiT CUDA event 区间 | 24.060601 s | 23.619975 s | 1.831% |
| API wall | 34.027092 s | 33.450473 s | 1.695% |
| denoise 主机调用区间 | 24.775962 s | 24.258114 s | 2.090% |

CUDA event 区间包含计算、通信及主机提交间隙，不是纯 kernel 时间；跨 rank
取各 rank 总时间最大值，不相加。正式初筛判定采用事先约定的 denoise CUDA
指标，不能因 host 指标跨过 2% 就换指标判通过。基线与候选 denoise 样本跨度
分别为中位数的 1.53%、0.54%；无 profiler，每组独立完整预热后三次计时。

正确性与资源核验：

- 每 rank 265 次实际 DiT forward，共 530 对输入/输出，完整原始 tensor
  `torch.equal` 且有限值通过；forward 数与保存配对数一致。
- 两组正确性请求原始 RGB 相同，进一步核对全部 10 次请求（含预热和正式
  计时）的输入哈希各只有一个、输出哈希也只有一个；240 帧完整接收。
- 每 rank 的 source/active/final writer 调用为 2200/8800/2200，基线均因
  head stride 回退、候选均 fused；实际 H=28、D=128、head stride=384。
- 各组峰值 allocated 完全相同：rank0 180.546 GiB、rank1 180.064 GiB；
  reserved 为 180.992/180.564 GiB。请求结束时 session 的 KV/bridge/decoder
  计数归零，仍驻留模型内存，不将 allocated 非零误报成泄漏。
- CPU 回归 823 passed / 133 skipped；CUDA writer 单测 51 passed；SP1/SP2
  collective harness 通过。新树 SP4 与 metadata combined 尚未验收。
- 两个独立视频和 A/B 拼接视频均通过完整解码；独立视频为 896×512、24fps、
  240 帧。抽看 0–9 秒的十个画面未见 A/B 差异，但这不代替连续 60 秒人工
  画质验收；本轮原始 RGB 的相等证据仅说明 writer 没有引入画质变化。

1280×736 在 baseline 请求协商时被现有服务限制拒绝：
`The source dimensions exceed the supported streaming size.` 入口要求
`17 * width * height * 3 + 4096 <= 32 MiB`，该尺寸需要 48,050,176 字节。
这次尚未执行该尺寸的 denoise，不是 OOM，也没有高分辨率 A/B 结果。
仅满足 32 倍数几何不足以通过入口，准备阶段漏检了此协议约束。保留失败证据，
没有自动改变分辨率、窗口或后端，也没有重跑已经完成的 512 档。

运行候选冻结 `65b7bbaeee2b01fe3039e32318a200b30fad44c6`，基线见上；
文档修订不改变本轮运行代码。GPU 作业 `dlce7na4q9yfmooc` 使用 2×L20D，
实际运行约 19 分 16 秒（约 0.64 GPU·小时，另有少量调度/退出时间），终态
Failed 对应上述高分辨率入口错误，不能因 512 完成就称整个矩阵通过。
零 GPU 后处理 `dlc3e7bwqee1er6e` Succeeded；源码、资产、脚本和失败回执保留。

机器可读指标及完整归档 SHA 见 [ISSUE220_SCREENING_RESULT.json](ISSUE220_SCREENING_RESULT.json)。
归档位于 `oss://leap-worldmodel-thailand-single-az/world-model/sglang/evals/zing220-20260928/r2/post-r1/`，
下载后核验归档 SHA 与内部 115 个文件的大小/SHA。本地媒体和完整证据位于
`results/stride-20260928/sp2-screen-r1/`（未将视频和大量日志纳入 Git）。


## 2026-09-29 高分辨率补测授权与冻结

用户确认仅在实验环境将 baseline/candidate 的 source admission 上限同时由
32 MiB 放宽为 64 MiB。此改动不进入生产代码；实际客户端仍逐帧发送，单帧
1280×736 RGB 为 2,826,240 字节。保留 clip17、window21/sink3、四步采样、
同源同 seed、同 writer 实现；不将入口限制修改作为 writer 收益的一部分。

补测仅覆盖 1280×736/SP2，独立正确性 A/B 各一次，独立性能 A/B 各一次
预热和三次计时，共 10 次240帧请求。512 档和已经通过的 primitive 测试不重复。
GPU 上限 2 卡60分钟，准备与后处理合用零 GPU controller；没有结果前不宣称
高分辨率支持或性能通过。继续采用原有 ≥2% denoise 扩展门槛。

新运行目录 `/cpfs/world-model/sglang/evals/zing220-20260929-r1`；冻结输入
`oss://leap-worldmodel-thailand-single-az/world-model/code/sglang/zing220-20260929/r1/`。
源码树从上一轮冻结树建立独立目录，修改文件先断开硬链接，避免改变历史证据。
