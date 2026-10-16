# 增量 H3 codec 状态（#112，尚未接入生产协议）

模型基线为 #103 `2fd837aa2d80908e2238c1ad6e59d68bccd4546d`；PR 依赖
#118 `1856d33d18` 的纯数据 provenance 契约。
本实现复用该版本的范围 decode，不改变已有有限 session 的默认行为。

## 接口与所有权

实现位于 `model_specific_stages/zing_h3/streaming_codecs.py`：

- `SourcePixelBuffer(codecs, capacity_frames=...)`：`append(pixels, frame_start=...)`
  接收连续的 `[3,T,H,W]` RGB uint8；`ready` / `pop()` 每次返回一个原生
  VAE clip 的 **posterior moments**，不是采样后的 source latent。
- `StableVideoDecoder(codecs, capacity_latents=...)`：
  `append(latent, latent_start=..., source_dependency_end=...)` 接收已经提交的
  `[B,C,T,H,W]` latent；`pop()` 返回下一段定稿像素。
  一个 append 内所有 latent 必须具有相同的实际依赖水位；不同水位应分别 append，
  不要把最新 accepted 帧水位当作计算依赖。
- `CodecSlice.values/start/end/source_dependency_end` 的区间是绝对半开区间。
  source 输出区间单位是 latent，decoder 输出单位是 pixel；dependency 始终是
  exclusive source pixel 上界。source ID/alignment 由 #111 的 provenance 账本按
  输出 pixel 区间切片；本模块不分配 source ID，也不触发 consumed ACK。
- `finish()`（decoder 为 `finish(total_frames=...)`）声明 EOS；
  `drain()` 惰性逐项返回尾部；取消、异常与会话销毁调用 `dispose()`。
  pop kernel 失败时保留原缓冲与游标，可重试。调用者不得并发操作同一实例。
  append 拷贝并接管输入，pop 输出独立 tensor；下游需先预留有界队列容量再 pop。
  消费者收到结果后自身失败的跨组件回滚由 #113/#114 的事务负责。

## 时间窗口与容量

17 pixel / 5 latent / ratio=4 / token_drop=3 时，encoder 必须等到完整
17 pixel clip；只有 EOS 才复制最后一帧补尾。整个会话仅一帧时走原生单图路径。
网络 append 与 AR `[1,2,2]` 或 DMD=5 分块都不改变这个边界。

encoder 的最小 credit 为完整 17 帧（`minimum_credit_frames`）。
codec 只校验本地像素容量；协商后的帧/字节双 credit 由服务 owner 校验，
必须允许至少一个完整窗口，避免死锁。按固定 RGB 尺寸配置字节 credit 至少
`17 * 3 * H * W`，同时限制网络单批与队列预算。容量错误在状态改变前抛出；
超大 append 由上游先拆分，并在有输出时 pop。

decoder 就绪需要当前 5 latent 加 2 latent 前视；容量至少 12 latent，
包含前一 5 latent clip 的 blending 上下文。按一次 pop 一个 clip 工作，
保留的 tensor、依赖水位数量有固定上限。切片后 clone 防止底层 storage
继续持有已丢弃历史；输出结束清空 latent。保留整数绝对游标而不保留历史列表。
EOS 只填充必要尾部，不改已发送输出。

复用范围 decode 仍会重新计算前一 clip（每次至多两个 learned windows），
尚未缓存已经解码的 overlap。该优化留待与 #107 的 decoder 优化一起验证；
当前不修改共享 VAE kernel、decode 精度或全局缓存。

## 明确阻塞与后续验收

1. 现有 `encode_visual` 的完整张量 posterior 采样保持不变。用户已授权先确定
   可调整的流式规则，新增显式 `indexed-cpu-v1`；未知 T 不再等待算法输入才能
   生成 source posterior / target initial noise，但新规则与 legacy native 完整张量
   RNG 不逐值等价，质量对照仍需实测。生产接线尚未启用此版本。
2. #111 只提供纯数据类型。decoder 接受可选 `provenance_for(pixels, dependency)`
   查询回调；必须返回匹配输出区间与实际依赖的 `Provenance`。查找或校验失败时
   不推进 codec 游标。回调由服务 owner 提供，须为无副作用查询，不得发 ACK 或提交输出；
   codec 不持有 ledger。#113/#114 的有界交接和事务接线仍需完成。
   codec 的依赖上界是实际参与 decode 的 latent 窗口依赖的最大值；EOS
   复制不产生新的真实 source 依赖。完整 source_alignment 应由 shared ledger
   构造，不能把 latent index 乘 4 当作 pixel 边界。
3. CPU 测试使用真实 H3 时间 assembly/blending，替代 learned per-window kernel；
   encoder 测试验证分批、EOS padding、normalization 与 moments 拼接。
   另已完成真实权重单卡 VAE 数值验收：35 个用例（243 帧 896×512 与
   1..34 帧 64×64），三种 source 分批的 moments、三种 latent 分批的
   stable decode 均与已有完整 SG 路径逐值一致。作业 `dlcpxe8eivtrdadk`
   成功退出并回读 `PASSED`；[原始证据与哈希](https://github.com/seedleap/sglang/pull/120#issuecomment-5748823844)。
   双卡 tile 每 rank 35 项 exact 对照与单/双卡 1,440 帧长测也已通过；
   平台期 GPU allocated / host RSS 观测跨度均为 0 MiB，三份长测输出哈希一致。
   [真机证据](https://github.com/seedleap/sglang/pull/120#issuecomment-5749118219)属于接口迁移前版本；
   本次仅迁移元数据查询接口，未修改数值 kernel。未完成独立 native、DiT SP 或成片质量验收。
4. 本实现没有 attention 淘汰、block-relative RoPE 或 sink/window 默认值。
   codec 缓冲有界不意味着 AR KV、full-history 或完整 session 内存有界。

CPU 回归：

```bash
PYTHONPATH=python python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_zing_h3_streaming_codecs.py \
  python/sglang/multimodal_gen/test/unit/test_zing_h3_media_contract.py
```


真实权重数值验收入口（不含 DiT 或独立 native 质量验收）：

```bash
PYTHONPATH=python python benchmark/zing_h3_causal/validate_incremental_codecs.py \
  --model /path/to/exported-zing-checkpoint --output /new/run/sp1
PYTHONPATH=python torchrun --standalone --nproc-per-node 2 \
  benchmark/zing_h3_causal/validate_incremental_codecs.py \
  --model /path/to/exported-zing-checkpoint --output /new/run/tile2
```

验收保持 FP32 decoder 和 checkpoint 指定 encoder dtype，固定 Torch SDPA；
读取真实 `video_vae/source/model.safetensors`，严格加载并记录哈希。
双卡运行的参考关闭 tile parallel，候选开启双卡完整 tile 分配，每个 rank
独立比对。测试用合成 RGB 图样检查数值、分批和尾部，不用于人工画质结论。
输出目录必须全新；失败不覆盖先前证据。仅生成 `PASSED` 且作业成功退出后
才算该运行通过。


可加 `--soak-frames 1440` 验证 64×64、60 秒像素时间轴的 codec 缓冲。
此阶段逐 clip 产生不同 RGB 输入，通过真实 encoder 与显式 indexed-cpu-v1 posterior 采样
驱动真实 decoder；不保留整段输入、latent 或输出，只保存帧数、哈希和每 clip
内存记录。预先固定平台期门槛：前 20 clip 之后 GPU allocated 波动 <16 MiB，
当前 host RSS 波动 <64 MiB。该测试不替代生产 RNG、真实场景或完整会话内存验收。


## 验收范围映射

| 要求 | 可重复验证 | 限制 |
| --- | --- | --- |
| 网络分批独立、仅 EOS padding | CPU 全尾长/不规则分批；真实 encoder moments 与整段路径 exact | sampled posterior RNG 契约待定 |
| stable 跨调用、首帧/clip/drop/尾裁剪 | CPU token_drop 0–4、1..34/243/527 帧；真实权重默认 drop=3 | zero-drop 需 EOS 或下一真实 token 以避免过发 |
| 真实依赖与 source ID | 服务提供 typed Provenance，跨 ID、拒绝用 accepted 51 替换 actual dependency 34 | 生产所有者接线归 #113/#114 |
| 容量、暂停、错误与取消释放 | 500 append storage/autograd 检查、kernel 失败重试、dispose、元数据失败重试 | 跨 owner 已提交输出回滚不由 codec 代办 |
| 真实 VAE / 多卡 | `validate_incremental_codecs.py` 严格 exact 数值门槛 | 是完整 SG 路径参考，不是独立 native / DiT SP |
| codec 长时内存 | `--soak-frames 1440`，GPU allocated 与当前 host RSS 门槛 | 64×64、显式版本化采样；不宣称完整会话有界 |

逐次作业、权重/输入哈希与原始结果保留在 PR #120 的验收评论和外部工件目录，
不把一次性地区脚本、大 JSON 或成片文件提交到仓库。未通过/未执行项与数值通过
分别报告；生产功能接线、noise 契约或独立 native 质量验收未完成前，不将 #112/#114 标为生产完成。

## 最新接口集成回归

#118 删除旧 `SourceLedger` 后，codec 已切换到
`runtime.realtime.zing_contracts`，不复制旧协议状态机。532 项 codec、媒体时间轴与
共享数据测试通过；119 项有限会话适配器、输出传输与 runtime 回归通过。
这些测试验证正式 CPU 控制代码，learned kernels / worker 使用 fixture，不能算真机 WebSocket 验收。

完整持续输入验收目前无法执行：正式 Zing adapter 仍为有限请求路径，只接受 prompt event；
#121 的 SourceReadySession 仍要求外部冻结 source/noise，尚无 codec→chunk 重分批 bridge；
未知 T 的 source/target noise 已有显式可选版本（见下节）；生产 caller 尚未接入。
需要在正式 #114 caller 完成这些能力后，运行真实 MessagePack WebSocket 的
append/accepted/consumed、暂停、慢消费者、EOS、取消竞态、断线及多会话隔离测试。
不会用临时 mock 协议或有限请求的成功代替上述验收。

## 已确认的窗口与 RoPE 方向

用户明确：`window=21` 包含 sink 和 current；沿用 `sink=3`，窗口满时为
3 sink + 17 recent + 1 current，单位为 temporal chunk。RoPE 使用与 MinWM
一致的 relative RoPE。此语义不再列为待确认，但尚未在本 codec PR 实现淘汰。

MinWM 的 `block_relative` 参考实现保存 RMSNorm 后、RoPE 前的 K；选择可见
窗口后重建相对位置并应用旋转，不能直接复用旧位置下旋转后的 K。
实际位置压缩受 `rope_max_frame_gap` 控制（类默认 1，部分 MinWM profile 为 12）；
移植时需显式记录所用 profile，不能将不同 profile 当作同一数值基线。
全局 chunk ID 与像素 provenance 仍保持绝对坐标。

## 显式流式随机数版本

`streaming_noise.indexed_video_noise` 提供 `indexed-cpu-v1`：ASCII
`indexed-cpu-v1:{purpose}:{seed}:{absolute_latent_frame}` 经 SHA256 后取前 8 字节
little-endian 作为每帧 CPU generator seed；按 `[B,C,H,W]` 生成 float32 高斯噪声，
再转调用者 dtype/device。`purpose` 区分 `source_posterior` 和 `target_initial`。
同一版本下固定 batch/channel/空间尺寸，按绝对 latent 位置取样，不依赖网络批次、
VAE clip 或 DiT chunk 划分；重试不推进全局 RNG，也不分配完整历史噪声。

VAE 调用 `sample_streaming_visual_moments(..., latent_start=..., noise_version=...)`，
沿用 checkpoint posterior_seed、标准 posterior logvar clamp、FP16 roundtrip 和
latent mean/std normalization。DiT bridge 使用 request seed 和 `target_initial`
取得其块的初始噪声；后续 re-noise 继续由 #113 每会话 RNG 事务管理。新版本初始化
不消耗该事务的 RNG；legacy 有限请求仍使用原来的 post-initial-noise snapshot。

该版本目前是显式工具接口，尚未接入 #114 服务。它保证固定 geometry/runtime 下的
分批和重试复现，不承诺跨 PyTorch 版本、不同 batch shape 或旧 native seed 的 exact。
运行记录应包含版本、两种 seed、geometry、PyTorch 版本和 noise hash；若后续调整
映射，必须增加版本，不得静默改变 v1。需在相同新规则的 native 对照下验收生成质量。

## 新随机数验收进展

执行候选 `7abdbff1c3`：660 项当前分支 CPU 回归通过；静态检查通过。
另在 #121 `12b9d6897c` 隔离源码快照中，以 indexed target noise 替换初始噪声，
8 项小型真实 CPU DiT 的 full-history / source-ready 对照 exact，覆盖固定 5 / AR
[1,2,2]、Euler / few-step、CFG 1 / 2。该附加实验使用 source/audio fixture，
不代表 learned VAE→DiT bridge 或生产服务已经接通。

新增真机脚本覆盖 indexed posterior 分批、target noise 分批/重试、全局 RNG 不变、
真实 decoder typed provenance、legacy/indexed 解码及 indexed 1,440 帧长测。
单卡 `dlckh8zka70d7z2b` 与双卡 `dlc1ke8qvewik9ew` 均 Succeeded / exit 0，
输出已从 OSS 回读并核对每个 rank 的 `PASSED`、results 和 soak：

- 单卡及双卡每 rank 均通过 35 项用例（243 帧 896×512 与 1..34 帧 64×64）。
  三种 source 分批的 moments / indexed posterior、target initial noise 的分批与重试、
  legacy 三种 latent 分批与 indexed decode 对完整路径均 exact。
- 单/双卡各完成 1,440 帧 64×64 indexed posterior 长测；前 20 clip 后
  GPU allocated 与当前 host RSS 观测跨度均为 0 MiB，满足 <16 / <64 MiB 门槛。
- 单/双卡每个用例的输入、sampled latent、target noise 哈希一致；三份长测输出
  SHA256 均为 `33baf5412918af9a7b064f67dea940b8febf0438104d1a154d1f4b9e76d5bd57`。
- 实际 NVIDIA L20D，274113 MiB，CC 10.3；源码包 SHA256
  `e7f1d398b5726a8497dead3cefcc2a390d2c05873be592093e2e473c050980a9`。
  已逐文件确认当前 runtime/harness 与执行候选一致，后续仅更新验收文档。

原始证据位于 OSS 桶 `leap-worldmodel-thailand` 的
`world-model/sglang/evals/codec112-indexed-sp1-r1/` 和
`world-model/sglang/evals/codec112-indexed-sp2-r1/`，CPFS 同名目录。

本 PR 基础 codec 与显式 RNG 工具的预定测试已通过。新 RNG 尚未接入生产服务，
上述合成输入/完整 SG 参考与小型 CPU DiT 测试不证明真实场景生成质量、
独立 native 新 RNG 一致性、完整 WebSocket 协议或完整会话内存有界；这些属于
后续 #113/#114 集成验收，不以本 PR 测试结果关闭完整生产需求。
