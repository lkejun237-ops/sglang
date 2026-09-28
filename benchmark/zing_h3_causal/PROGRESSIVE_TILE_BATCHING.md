# 渐进 encoder 有界 tile 合批（#197）

状态：实验实现；SP1/SP2 batch2/3/6真实权重严格数值均失败，默认行为不变，
不宣称无损。完整encoder、六图服务、媒体技术检查与视觉抽查已完成；严格数值等价验收失败；画质可接受性由用户评审，当前待确认。

`MINIMAX_H3_PROGRESSIVE_ENCODER_TILE_BATCH_SIZE=1|2|3|6` 在 session 创建时
冻结。设置正整数上限后，覆盖旧 stack_tiling 的无限合批选择；不设置则保留既有
行为。上限计量单位是 spatial tile，实际 batch 为 tile 数乘输入 batch。
同 shape 的本 rank tile 分组，不跨时间块、不修改空间布局、反射 padding、
1/8/8 输入和 1/2/2 latent 边界。缓存 key 是稳定的全局 tile ID 元组，batch
slot 按元组顺序固定；最后不足上限的组独立执行，clip 边界、EOS、reset、dispose
沿用 session 契约。

与 `MINIMAX_H3_PROGRESSIVE_ENCODER_TILING=auto` 独立组合，deterministic BF16
的 SP2 保留真实 round-robin 分片（默认12 tiles，每 rank 6）。异形 tile
按 shape 合批，但因当前 gather 要求同 shape，跨 rank 遇异形时明确回退
`heterogeneous_tile_shapes`，不冒充真实分片。输出恢复全局 row-major 顺序再拼接。
`last_execution` 增加 `tile_batch_size`、`tile_batches` 和逐 tile 输入 shape。
SP4 合批的 CUDA 验收尚未完成，不继承 #177 的 batch1 验收结论。

## 运行前冻结的门槛

- 基线 origin/main `b235905dd7f16a387ac9c801340c3368713c46e8`。
- infra-skills `f74c315adb4b71fced9c22ae79bd4e58f83141cd`，全部6目录逐 Git blob 核验。
- 原生只读参考 `e50c476d1b256c8672bafb091dd93a8c6d949bf2`、
  `9044817f48f309fc937f9cd63e83775cb508fad2` 和 H3VisualVAE._encode_clip_group。
  借鉴同 shape 分组及有限上限，不复用其 batch15 或跨未来 clip 的计算方式。
- 沿用项目 #177 数值门槛：moments、indexed posterior 和精确搬运严格 bitwise；
  额外报告 max abs、relative L2 和逐位结果。没有已获认可的 BF16 batch 变化
  容差，故不预设宽松阈值。差异存在即数值不等价，不称无损，不事后放宽。
- 完整 encoder 中位耗时改善至少5%；首 latent、端到端延迟、输出间隔p95
  不退化超过5%；无 OOM、死锁、泄漏。未达标保留实验模式。
- SP1/SP2，batch1/2/3/6，同设备独立进程 A/B、B/A；每配置完整预热覆盖
  1帧和两种8帧历史，再至少3次无 profiler 测量。独立逐 rank 原始 trace。
- 默认真实 source + 全局图 + 五张三视图，832×480、四步、seed42、video-only。
  素材SHA、模型SHA、镜像digest、代码及环境在 CPU 准备回执冻结。
- 已有默认 source 仅约5秒；主任务已明确授权确定性循环2次/12次用于
  10/60秒技术压力测试，记录原始/派生SHA和每5秒循环边界。连续session
  不在循环边界reset；不能称自然长视频或自然运动画质验收。
- 与 #195 performance、#196 padding 独立；组合收益须另列，不混入单项数据。

## 当前验收表

| 项目 | 状态 |
|---|---|
| CPU CNN、尾帧、跨clip、有限batch映射 | 81项通过，非GPU证明 |
| CPU Gloo多rank、异常、不等任务数与异形回退 | 22项通过，非GPU证明 |
| NVIDIA SP1/SP2 batch2/3/6相对batch1数值 | 每组140/140严格逐位失败，见下表 |
| 同设备独立进程A/B、B/A及10/60秒长流 | SP1/SP2全矩阵完成，性能子门槛通过 |
| 逐rank trace与Perfetto打开 | 12份原始文件全部打开、计数一致；有重叠事件溢出提示 |
| GPU未来隔离、多session、reset/dispose/ownership | SP1/SP2 batch1/2/3/6全部通过 |
| 六图端到端媒体及延迟/FPS | 192次完整请求；24组服务时延门槛通过，严格数值等价失败；画质由用户评审 |
| SP4 | 未实测，不宣称支持 |

## 复现命令

预先冻结真实RGB、模型目录和完整源码清单。每次调用新建进程和输出目录，
在同一物理GPU集合上按 `1, candidate, candidate, 1` 顺序调用（candidate为2/3/6）；
每次timing调用内部有完整预热与3次正式测量。显式记录CUDA_VISIBLE_DEVICES及UUID。

```bash
PYTHONPATH=python torchrun --standalone --nproc_per_node=2 \
  benchmark/zing_h3_causal/validate_progressive_tile_batches.py \
  --model "$MODEL" --rgb "$RGB" --rgb-sha256 "$RGB_SHA256" \
  --source-sha "$SOURCE_SHA" --source-manifest "$SOURCE_MANIFEST" \
  --batch-size 2 --phase timing --frames 240 1440 --output "$NEW_OUTPUT"
```

分别运行 `--phase correctness`（EOS 1..34及120帧，多种分包，moments及indexed
posterior）、`--phase lifecycle`（未来帧隔离、交错session、非连续输入拒绝、
reset/dispose/输出所有权）、`--phase profile --frames 34`（每rank原始trace）。
这些命令仅验证encoder；完整服务用当前 `continuous_source_client.py` 的真实
source与六图reference接口，不使用已移除的 `--prompt-video-frame`。
`encoder_probe_complete_not_service_acceptance` 不表示数值或完整验收通过；必须
单独检查 `numerical_passed`、真实分片receipt及性能表。

## 真实数值与完整encoder结果

PAI作业 `dlc1yewowaddahff` 已Succeeded，38/38配置完成：6个数值配置、24个
独立计时进程、8个profile配置（12份逐rank trace）。运行源码
`a1911145c1840dbfb17958ebf8cb1f2008fec53a`；后续提交只更新测试、探针和文档，
运行时实现未变。两卡均实测CC10.3、287428771840 bytes，设备原始名称NVIDIA L20D。

每个候选配置140项EOS/分包用例，所有rank的moments及indexed posterior均
140/140逐位不等，但全部finite。以下为每配置所有用例的最大差异；SP2两rank相同。

| SP | batch | moments max abs | moments relative L2 | posterior max abs | posterior relative L2 |
|---|---:|---:|---:|---:|---:|
| 1 | 2 | 0.453125 | 0.002504797 | 0.4604187 | 0.004186197 |
| 1 | 3 | 0.578125 | 0.002528434 | 0.5342255 | 0.004705253 |
| 1 | 6 | 0.5898438 | 0.00253159 | 0.5298843 | 0.004694285 |
| 2 | 2 | 0.5214844 | 0.00248358 | 0.4604187 | 0.004182704 |
| 2 | 3 | 0.578125 | 0.002534892 | 0.5298843 | 0.004690778 |
| 2 | 6 | 0.5820312 | 0.002533033 | 0.5298843 | 0.004693001 |

每个进程、每个长度均完整预热后3次无profiler测量。表中时间是各独立进程/
rank中位数的范围，收益按匹配的A/B、B/A分别计算，包含输入搬运、gather/拼接；
不将不同baseline混合后挑最优结果。

| SP | batch | 10秒encoder (s) | 60秒encoder (s) | 匹配baseline改善 | allocated峰值GiB | reserved峰值GiB |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 9.4412–9.4818 | 55.1756–55.3324 | baseline | 16.543 | 16.877 |
| 1 | 2 | 8.7130–8.7251 | 50.8038–50.9740 | 7.71–7.92% | 17.385 | 17.947 |
| 1 | 3 | 8.3514–8.3577 | 48.7958–48.8086 | 11.65–11.85% | 18.225 | 19.311 |
| 1 | 6 | 8.0333–8.0371 | 46.9110–46.9162 | 15.08–15.25% | 20.744 | 22.746 |
| 2 | 1 | 4.9523–5.0203 | 28.7944–29.2003 | baseline | 13.390 | 13.885 |
| 2 | 2 | 4.6027–4.6117 | 26.9337–26.9624 | 7.02–8.15% | 14.233 | 14.812 |
| 2 | 3 | 4.4014–4.4293 | 25.6635–25.7991 | 11.12–11.53% | 15.072 | 16.166 |
| 2 | 6 | 4.2283–4.2454 | 24.7516–24.7997 | 14.02–14.94% | 17.591 | 19.838 |

全部36项逐rank/长度/顺序性能比较通过encoder子门槛；首latent与输出间隔p95
均无超过5%的退化。三次完整encoder样本最大相对spread为1.505%。72项
进程/rank/长度的缓存阶段平台期全部通过，dispose后retained bytes均为0；
SP1缓存峰值3.152GiB，SP2每rank1.576GiB，四种batch相同。

以下为60秒流的分组pop p50/p95范围（ms）；含同步和该次gather/拼接。

| SP | batch | 1-latent p50 | 1-latent p95 | 2-latent p50 | 2-latent p95 |
|---|---:|---:|---:|---:|---:|
| 1 | 1 | 94.20–94.30 | 94.48–94.56 | 275.04–275.24 | 275.36–275.69 |
| 1 | 2 | 73.12–73.24 | 73.20–73.42 | 259.65–260.04 | 259.89–260.64 |
| 1 | 3 | 67.91–67.98 | 68.17–68.19 | 250.17–250.21 | 250.75–250.82 |
| 1 | 6 | 60.36–60.38 | 60.50–60.56 | 242.74–242.82 | 243.06–243.14 |
| 2 | 1 | 50.12–50.78 | 51.05–51.75 | 141.79–142.87 | 142.88–144.01 |
| 2 | 2 | 40.20–40.36 | 40.94–41.22 | 135.04–135.21 | 136.17–136.41 |
| 2 | 3 | 37.28–37.59 | 37.93–38.17 | 129.38–129.70 | 130.48–130.89 |
| 2 | 6 | 33.30–33.44 | 34.02–34.33 | 126.05–126.50 | 127.32–127.83 |

10秒EOS的1-latent尾组内部仍需8帧padding，不能将该分组p95称为首帧延迟。
全部原始三次样本、首latent、输出间隔、显存及cache阶段数据保留在导出JSON。

## 首帧诊断与生命周期

后续服务作业 `dlc1cixi4qvwncdi` 使用源码
`290ddbb100b0a2a048537c60d5bbed75fe19f4ce`，分配到同一对GPU；运行时实现
与首轮完全一致。独立首帧诊断不计入任何正式耗时：tile0最早差异位于
`encoder.down[1].block[0].conv1`。输入逐位一致，候选输入shape为
`[2,128,1,128,128]`，该层输出max abs=0.00390625、relative L2=1.84507e-5；
用候选完全相同的tile0输入单独重算，也得到这个差异。首帧完整moments
max abs=0.1640625、relative L2=0.00243238。这证明至少存在batch形状相关的
卷积数值差异，尚未定位具体cuDNN算法，不能由此放宽预先冻结的标准。

SP1/SP2 batch1/2/3/6的未来像素隔离、非连续输入拒绝、交错session、输出所有权、
reset和重复dispose均通过（12份rank回执）；这些检查比较同一候选自身，
与跨batch的严格数值失败分别记录。

## 输入与证据

当前默认六图案例按冻结URI重建后的完整请求SHA256为
`3e0859367573fac6136678d9ebf5a5e3decc91d50013ab2d9aeffbf7801329f2`，
与运行输入逐字节一致、provenance一致。原始source用
`fps=24,scale=832:480:flags=bilinear`、120帧、RGB24独立重解后的SHA256为
`d917780f9c0bb002fbd646b77f23d442d73b5a42d840c4c911cef74f5af7adbb`，
与冻结输入一致（独立核验回执来自#196的零GPU审计，非重跑本实验）。

GPU UUID为 `9568bfd5-eb84-deb7-c5ed-0cf1d670af62` 和
`ca38d8cc-c4d8-7236-eb9c-f4215d746f20`。完整encoder证据包SHA256为
`6cc21e86e1450d501568e6da0a6f6f31f866130c3366a3ae0c3c226732d0dc37`，
172文件及12份原始trace的压缩/解压哈希均已回读核验。媒体与大文件仅保留在
本地或私有存储。完整六图服务结果见下文；encoder性能子门槛通过不表示整体验收通过。

## 原始 trace 的 Perfetto 核验

12份逐rank原始JSON均已在Perfetto `v58.3-11fbaed83`实际打开，检查CPU线程、
CUDA stream及H3事件，并用UI SQL核对kernel与H3事件计数。所有计数与原始文件一致。
全部trace均有`slice_spill_overlapping_complete_event`导入提示：UI说明重叠的
complete事件移至溢出轨道而非丢弃，轨道布局可能有歧义。原文件未修改时间戳。

| SP | batch | 每rank CUDA kernels | 每rank CPU tile scopes | 每rank CPU/GPU H3 annotations | 溢出提示数rank0/rank1 |
|---|---:|---:|---:|---:|---:|
| 1 | 1 | 50384 | 72 | 72 | 242 / — |
| 1 | 2 | 25568 | 36 | 36 | 146 / — |
| 1 | 3 | 17272 | 24 | 24 | 97 / — |
| 1 | 6 | 8976 | 12 | 12 | 48 / — |
| 2 | 1 | 25532 | 36 | 42 | 121 / 157 |
| 2 | 2 | 13124 | 18 | 24 | 72 / 108 |
| 2 | 3 | 8976 | 12 | 18 | 48 / 84 |
| 2 | 6 | 4828 | 6 | 12 | 24 / 60 |

SP2每rank额外包含6个gather标注及6个NCCL kernel，SP1无NCCL kernel。所有CUDA
kernel均找到launch correlation；CPU归属用同线程scope嵌套和launch correlation
连接，不用GPU时间戳裁剪推断CPU归属。UI已实际选中CPU tile事件及GPU kernel，
逐文件回执保存在`perfetto/ui-checks.json`。PyTorch对部分kernel报告的估计
occupancy为0，不作为硬件GPU利用率结论。


## 完整服务最终结果（2026-09-25）

GPU作业 `dlc1cixi4qvwncdi` Succeeded，24个独立服务进程、192次请求全部结束；
每配置10/60秒各一次完整预热和三次正式测量。当前六图案例、四步、seed42、
video-only、连续source协议保持冻结。以下收益来自匹配A/B与B/A中位数。

| SP | batch | 10秒端到端耗时改善 | 60秒端到端耗时改善 | 10秒FPS范围 | 60秒FPS范围 |
|---|---:|---:|---:|---:|---:|
| 1 | 2 | 1.61–2.18% | 2.41–2.46% | 5.407–5.423 | 5.296–5.303 |
| 1 | 3 | 3.67–4.83% | 2.93–3.55% | 5.492–5.503 | 5.318–5.370 |
| 1 | 6 | 3.10–3.43% | 3.12–3.43% | 5.467–5.492 | 5.338–5.364 |
| 2 | 2 | 0.37–0.90% | 0.76–1.24% | 6.043–6.051 | 6.078–6.101 |
| 2 | 3 | 1.19–2.13% | 2.41–2.48% | 6.178–6.181 | 6.217–6.243 |
| 2 | 6 | 1.59–1.99% | 1.51–2.48% | 6.162–6.186 | 6.187–6.217 |

24/24服务比较满足不超过5%退化门槛：最差耗时ratio=0.996344、首输出
ratio=1.006113、WebSocket帧批次间隔p95 ratio=0.993133。性能门槛的至少5%
改善针对完整encoder；不将其与较小的完整服务收益混淆。没有观察到OOM或死锁。
服务期间每秒nvidia-smi设备占用峰值范围190788–244002MiB，包含加载与预热；
它不是PyTorch allocated/reserved峰值或精确瞬时峰值，也不能单独证明无泄漏。
逐配置、逐rank四分段统计保存在service-memory-analysis.json；encoder缓存平台期
和dispose检查是独立证据。

192个原始RGB输出的完整字节数、输入/输出SHA、收发帧数及ended均通过；
去重14个MP4均完成ffprobe帧数/24fps/832×480检查及完整解码。14个视频下载后
再按导出manifest验证哈希。12个候选/基线原始RGB对全部不等：max abs uint8
242–255、relative L2 0.197273–0.330414、RMSE 36.884–52.833，不能称无损。

视觉检查覆盖14个视频的抽帧联系表：10秒取0/16/17/118/119/120/121/238/239帧，
60秒另取720/1438/1439帧。不是连续播放或逐帧人工观看。早期人物/场景可辨认；
60秒基线与候选中后段均出现纹理重复、场景漂移及人物变形，候选轨迹与基线有
可见差异。画质可接受性由用户判断，当前待确认；这些抽帧观察不构成画质不合格结论。5秒源循环压力测试
仍只用于技术验证；严格数值失败不由吞吐提升抵消，画质评审独立进行。

最终保留实验开关和原默认策略，不推荐自动启用batch2/3/6；SP4及与其他优化的
组合仍未实测。本PR交付实现与正负验收证据，不表示候选已满足整体采用标准。

私有证据根为 `oss://leap-worldmodel-thailand-single-az/world-model/sglang/evals/zing197-20260924/`：
- `encoder-evidence-r1.tar.gz`：上述encoder证据包SHA；原始逐rank trace在包内。
- `evidence-addendum-r1.tar.gz`：SHA256 `c53705dad43087b102fece0fdd132a61243dcd91118596a000f133ecaab25d74`，含Perfetto UI回执和控制脚本。
- `service-evidence-r2/`：972文件，862012517 bytes；manifest SHA256 `643f576f5a3865ed8354e6a9f019a51a68195466cf31673701577f410e6d9b2f`。

首次零GPU导出 `dlcucv4lm52x0ags` 完成媒体分析后，在OSS挂载立即回读时失败。
保留失败目录；`dlc1copedm1ilblc` 仅重新复制既有CPFS产物，逐文件SHA回读后
Succeeded，没有重跑GPU或覆盖失败证据。媒体保持私有，无公开预览链接。

## 2026-09-28 独立5秒画质对照补测

按用户要求重新生成原始120帧（24fps），不循环，不从旧10/60秒输出裁剪。
固定SP2、原实验源码 `290ddbb100b0a2a048537c60d5bbed75fe19f4ce`、原六图请求、
seed42，分别运行batch1/2/3/6，每配置两次独立session，共8次。未叠加PR #200；
本次不同GPU UUID且没有性能测量预热矩阵，不用于更新原性能结论。
旧CPFS临时目录已清理，零GPU恢复后源RGB SHA256仍为
`d917780f9c0bb002fbd646b77f23d442d73b5a42d840c4c911cef74f5af7adbb`；
模型配置、VAE权重和TAEH3权重均按旧回执核验。仅素材本地URI随恢复目录变化。

8次请求均完整收发120帧并正常ended；四个配置各自两次原始RGB SHA均相同。
每rank执行回执确认6个本地tile及指定batch参数。5秒服务输出诊断如下，
不是新增encoder moments实验，也不以像素误差设定人工画质门槛：

| 相对batch1 | 原始RGB bitwise | relative L2 | RMSE（uint8） | max abs（uint8） |
|---|---|---:|---:|---:|
| batch2 | 不等 | 0.189468 | 35.768896 | 238 |
| batch3 | 不等 | 0.185904 | 35.096055 | 241 |
| batch6 | 不等 | 0.191328 | 36.119957 | 245 |

三组从第0帧即有像素差异；画质可接受性仍由用户评审，当前待确认。
四个单路MP4及三个直接从原始RGB并排编码的A/B MP4，均通过120帧、24fps、
5秒及完整解码检查；下载后逐文件SHA再次验证。A/B为同一视频的左右画面，
避免两个播放器不同步。旧原始RGB已不在CPFS，未完成新5秒与旧10秒前缀的
逐字节比较；不以有损MP4代替原始RGB宣称等价。

GPU作业 `dlc1rpyx3em3e7k5`、零GPU导出 `dlc1g2mzuyfi3ptp` 均Succeeded。
私有证据位于上述OSS根的 `service-5s-20260928-r1/`，16文件、46933541 bytes；
manifest SHA256 `2472eea7c5027c039f05872e3df16906ed57131682ba54ff9b2a4e789e528369`，
report SHA256 `fdb877b63b98e37d46e382c55defda5907a3c959fec71d2225803309b1864ee0`。
原始输出保存在CPFS `world-model/sglang/evals/zing197-5s-20260928-r1/gpu-service-5s-20260928-r1/`。
