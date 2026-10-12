# 在线 SP2 编码一致性诊断（#269）

此目录保留不计时的组件诊断工具；生产修复位于 Zing codecs。
根因在启用 benchmark 的 visual VAE 卷积策略，而不是必须由 joint 才触发：
off 也跨进程变化，且复制编码可以跨 rank 不同。修复统一实际送入 SP 的
visual latent；需要跨进程逐位复现时仍应选择 deterministic 编码。
业务输入仍是 `zing-whitebox-video-five-triview-0001` 的真实 source + 六图，
120 帧、832×480、24 FPS paced WebSocket、seed=42、四步、纯视频。
回放替换编码结果，只用于隔离 joint，不代表 native 效果或性能验收。

## 工具与复现边界

`audit_historical.py` 读取 2026-10-08 在线补测 summary，并可核验结果包回执：

```sh
python benchmark/zing_h3_causal/issue269/audit_historical.py SUMMARY.json \
  --archive results.tar.gz --receipt receipt.json --output audit.json
```

`frozen_codecs.install()` 必须在诊断 worker 导入模型后显式调用。
它读取 `ZING269_ROOT`、`ZING269_ARM`、`ZING269_FREEZE`，模式为
`raw`、`capture` 或 `replay`。capture 将首个 session 的 rank0 Qwen、
reference、采样 source latent 保存到可信 CPFS，并让所有 rank 使用这些值；
replay 重新使用同一组值。reference/Qwen 输入哈希及 source 采样位置参数必须
匹配；不同 session 重置事件计数。它同时保存 target latent，并记录初始 noise。
诊断会拷贝张量、同步 rank，不能混入正式计时。只加载可信本地张量文件。

`conv_probe.install()` 在 codecs hook 后安装，观察前六张 reference 内的
`torch.nn.functional.conv3d`。首图记录全部卷积哈希，后续记录新增几何的首个调用；
首图前五个卷积另存实际输入、权重及输出，支持离线数值检查。原始输入图像也
按 rank 保存。采集的 CPU 拷贝可能影响 benchmark 选择，因此只有观察到的
相同输入/权重而输出不同才可定位卷积差异；不能把未观察到差异当作其不存在。

可选 `ZING269_ENCODE_MODE` 显式选择编码策略；
`ZING269_BENCHMARK_OFF=1` 仅关闭编码范围内 cuDNN benchmark，保留其余
performance 策略，用于单变量对照。历史 context manager 没有返回 policy dict
时也可运行。工具不由生产入口自动加载。

## 已确认的历史隔离结果

历史计算源码 `1224489aea6923b47a5ab01a74b87f65f3fad04b` 保持不变，
joint wait=0、FA4，与最新默认配置分开记录。
`dlclttxjftbf0tus` 的八档共 16 个完整 session 已成功生成；该作业随后因
诊断工具不兼容历史 context manager 的 None 返回值而失败，不能标为成功作业。
失败的 raw-nobench-a 排除；修正工具后的续跑使用独立目录与作业。

冻结全部条件/source latent 后，off 的两个独立服务、source_first、cross_chunk
共八个 session 的全部 22 个 target latent chunk 与原始 RGB 均逐位一致。
这排除了该固定输入和历史配置下 joint 引入额外差异；不推广到其他配置。

raw performance + tile + off 的两个独立服务各自重复稳定，但跨进程不同。
Qwen 和全部初始 noise 相同；第二张 reference 与首个 source moments/latent
已经不同，早于 target DiT。raw deterministic + tile + off 的两个独立服务
共四个 session 的 RGB 完全相同。不能将 performance 的跨进程差异解释为
允许的 BF16 误差，也不能用旧速度数据替代无损验收。

历史续跑 `dlc1l2plor3c17ky` 保留相同作业内的 performance + tile + off
两个独立服务作为控制组：各自重复一致、跨进程 RGB 不同。仅关闭编码范围内
benchmark 的两个独立服务共四次请求，RGB 全部相同，且与原 deterministic
四次请求完全相同。此对照不改 encode_dtype、tile 分工或 joint，也不把不同
作业机器的变化当作单变量证据。剩余逐张量与媒体核验由零 GPU 后处理完成。
这与 [PyTorch reproducibility 文档](https://docs.pytorch.org/docs/2.14/notes/randomness.html)
对 cuDNN benchmark 算法选择的说明一致；仍需卷积输入/权重证据才能定位具体算子。

## 首个已观察到的卷积分歧

修复后的新默认配置诊断保留了编码器内部观测，广播没有掩盖原始算子差异。
performance + tile + cross 的首张 reference 中，`conv3d` 的 `call=41`
（从 0 计数）首次跨 rank 不同；此前采集的 0–40 次卷积记录完全相同。
该调用的输入、weight、bias、stride、padding、dilation 完全相同，只有输出
哈希不同。当时 `benchmark=True`、`deterministic=False`。

| 张量 | shape | dtype |
|---|---|---|
| input | `[13,256,3,130,130]` | BF16 |
| weight | `[256,256,3,3,3]` | BF16 |
| output | `[13,256,1,128,128]` | BF16 |

input SHA256 `488ff9ff50ee843c8009bcbe8d3477535f37bf676d2e6670bf8d32508e803e35`，
weight SHA256 `750b7ca291a627e7ee90372d63c01c1f6277938b5426dff379cb9c7862a8097d`；
两 rank 的 output 分别为
`7e017e5c473b83c4456469290a923bb9fc20525f68f901af5b0d46dfe5377102`、
`ef6aa82552b02b6031f8886d0cdc28407389bbe9b22ac7bbb67ee7f78c4ee178`。
此前 deterministic 三档的全部 87 条卷积观测跨 rank 相同。
结合仅关闭 benchmark 的单变量控制，可将本 workload 的差异归因到启用
benchmark 的 visual VAE 卷积策略；实际 cuDNN kernel/algorithm ID 没有采集，
不能声称已证明具体算法编号。采集开销可改变算法选择，因此不同轮次最早出现
差异的 reference/call 可以变化；本次实际首点与历史 warmup 首点分别保留。

这里记录的是完整张量哈希。保存的前五次卷积张量并不覆盖 call=41；不为该
算子编造 max_abs。完整 target/RGB 的误差量级由历史数值验收给出。

## 历史数值与媒体验收

历史续跑 GPU 与合并后处理 `dlc1lwwt8rdm8xh1` 均 Succeeded / exit=0。
合并集的 28 次请求均 submitted/accepted/consumed/received=120、ended=true；
RGB 字节数和 SHA256、22 个有限 target chunk、MP4 完整解码核验通过。
这是八档已完成的旧作业结果与六档成功续跑结果的组合，旧失败作业仍记为 Failed。

| 对照（独立服务，首个完整 session） | RGB8 MAE | RGB relative L2 | RGB max abs | 平均 SSIM | 首个 target chunk relative L2 |
|---|---:|---:|---:|---:|---:|
| 冻结 off → 独立 off / source / cross | 0 | 0 | 0 | 1 | 0 |
| deterministic off，跨进程 | 0 | 0 | 0 | 1 | 0 |
| benchmark off，跨进程 | 0 | 0 | 0 | 1 | 0 |
| performance tile off，r3 跨进程 | 19.03542 | 0.176624 | 225 | 0.46579 | 0.096886 |
| performance replicated off，r3 跨进程 | 20.72470 | 0.192857 | 235 | 0.45654 | 0.080056 |

SSIM 从未压缩 RGB 的 gbrp 对照计算，120 帧逐帧统计；不是两个 MP4 的压缩误差。
performance tile 跨进程逐帧 PSNR 为 16.382–28.006 dB，22 个 target chunk 的
最大 relative L2 为 0.602683。变化已明显放大到完整画面，不能按允许舍入误差放行。
本地同时间戳抽帧可见后段平台边缘、栏杆、远处房屋/灯塔位置及角色姿态不同；
这是输出间画面差异检查，不是对 native 语义正确性的独立验收。
所有诊断有张量拷贝，客户端耗时不作为性能结论。纯视频不声明音视频验收。

私有 bucket `leap-worldmodel-thailand-single-az` 的
`world-model/results/zing269-historical-merged-r1/` 保存 summary、结果包和回执；
结果包 SHA256 为 `39a48cf668daaa5f136f6d84bb31e7d83395059a4479719aa12627792698b778`。
原始 RGB、target 张量及 capture canonical 保留在各自 CPFS 目录。

## 修复范围

performance 编码下，复制 reference/offline source 的两 rank 曾产生不同 latent，
破坏 SP 对同一输入的假设。`encode_visual` 及
`sample_streaming_visual_moments` 现在在标准化后通过 SP group 从 leader
广播完整 latent；不重采样、不改变 RNG 顺序。deterministic 和非分布式路径
保留原行为。此修复保证该输入的 rank 一致性，不保证 performance 跨进程
逐位复现。渐进 tile source 的 moments 已由 gather 统一；replicated source
对照的全部 22 个 moments 与采样 latent 跨 rank 不同，因此在线采样也必须同步。
两路原先的 target latent 跨 rank 相同并不能证明其复制输入正确：SP 通信仍会
组合两 rank 的不同输入。source 与 target 的验收分别记录。

关联 #202 的实际部署 encode_mode 尚无回执，不能仅凭相似现象判为同源或关闭它。

## 修复后最新默认配置验收

GPU `dlc8szhmzep7j5yg`、零 GPU 后处理 `dlc1e5i0i3z75xid` 均
Succeeded / exit=0；冻结计算源码
`9c19d3b4f6a69c509894a35edfb12093eed9ec8b`，源码包 SHA256
`9e0442afe069f1119af62cc7edc72b34b684fcdfe6502244bcf02deb851508cd`，
诊断工具包 SHA256
`7e940a8a83e2ca1a2f9587c14e39de31fa58d5e39bafc84deb19835ffd9abbc9`。
#230 已外部合并到 main；修复 rebase 后对应 runtime commit `de1510a484`，
两者源码 tree 的 diff 为空，后续提交仅补报告/诊断材料。

五档独立服务各两次完整在线 source + 六图请求：det-off、det-source、det-cross、
perf-cross、perf-replicate。两种 cross 使用最新 `cross_chunk + auto wait +
attention auto`；off/source 保留显式策略对照。编码模式是单独控制变量，
diagnostic hook 的 raw 模式没有回放编码输出；哈希/拷贝开销仍使其不适合计时。
实际两 GPU 均为 CC 10.3、274113 MiB；素材、六图与模型准备回执保留。

- 十次均 submitted/accepted/consumed/received=120、ended=true，原始 RGB 大小/
  SHA256、有限 target latent、MP4 完整解码核验通过，下载后再次核验结果包回执。
- 两个 session 的六图、22 个采样 source latent、22 个初始 noise 和 22 个
  target chunk 均跨 rank 相同，各服务两次也完全相同。
- performance replicated 的两次各 22 个原始 source moments 仍跨 rank 不同，
  但广播后的采样 source latent 与 DiT 输入全部相同，验证了在线 source 修复。
- deterministic 三档共六次请求的完整 target latent/RGB 逐位相同，max_abs=0、
  relative L2=0。RGB SHA256 均为
  `b01f9c0a76709ac71bf9de3fe44a22859a661a489f7343b43ea7dcce3e7d0f30`，
  也与历史 deterministic / benchmark-off 结果相同。

| 修复后对照 | RGB8 MAE | RGB relative L2 | RGB max abs | 最大 target chunk relative L2 |
|---|---:|---:|---:|---:|
| det-off → det-source | 0 | 0 | 0 | 0 |
| det-off → det-cross | 0 | 0 | 0 | 0 |
| det-off → perf-cross | 16.97986 | 0.159043 | 221 | 0.564284 |
| perf-cross → perf-replicate | 20.33328 | 0.188690 | 231 | 0.680416 |

因此本修复通过 **rank 输入一致性**，不把 performance 编码宣称为跨进程稳定或
相对 deterministic 无损。生产需要复现时使用
`SGLANG_ZING_VISUAL_ENCODE_MODE=deterministic`，保持部署 profile 与实际模式
一致；不自动改变 performance 的数值策略，也不放宽旧 latent 门槛。

实际 source_first 为 111 次 forward（含 prefix），22 次两分支。
cross 三档每次均 104 次 forward：7 次三分支、15 次两分支、15 次独立 final，
source_not_ready=14、window_eviction=1。自动等待确实执行，最大预算
333.33 ms，最大实际等待 330.65 ms。两 rank 的 attention auto 观测全部
`fused=False`、backend=FA4，Q=780/1560/1950；不声明专用 kernel 已通过。

589 项 codec/media/streaming CPU 回归、5 项诊断工具回归通过。
现有 `taeh3-contracts` CI 检查通过；不代替真实模型/速度验收。
此轮纯视频，不声明音视频、SP4、native 语义正确性或性能收益。
所有业务媒体仅在本地和私有 OSS/CPFS，未发布公网，也未上传 GitHub。

私有 bucket 下 `world-model/results/zing269-current-r2/` 保留 summary、日志、
十份 MP4 和结果包；结果包 SHA256
`eaccf7ac88fe7389aa0e336770a56a34e4315678261dc321dac71cf4e19279a1`。
结构化证据见 [evidence.json](evidence.json)，修复 PR 为
[#271](https://github.com/seedleap/sglang/pull/271)。#269 保留待评审，#202 不因
本次 SP2 结果自动关闭。
