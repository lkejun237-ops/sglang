# Progressive encoder tile 并行设计

本文解释 PR #178 的行为、与既有 MiniMax H3 encoder SP 的关系，以及数值与故障处理边界。
配置、测量和验收入口见 [运行与验收说明](../../../../../benchmark/zing_h3_causal/PROGRESSIVE_TILE_PARALLEL.md)。

## 三个容易混淆的开关

| 名称 | 控制对象 | 不控制什么 |
|---|---|---|
| 模型 `parallel_tiling` | 旧 `tiled_encode` / `tiled_decode` 的空间 tile 分卡 | 不会让 `CausalEncoderCache` 自动经过旧的完整 clip encode 路径 |
| `MINIMAX_H3_PROGRESSIVE_ENCODER_TILING=auto/replicated/parallel` | 本 PR 有状态 progressive source encoder；`parallel` 是枚举值 | 不改 decoder，不表示开启全部模型的 SP |
| 模型 `stack_tiling` | 将本 rank 的多个 tile 拼成一个 batch 执行 | 只是本卡 batch，不是跨卡分工；改变 batch 形状可能影响浮点结果 |

新开关独立于旧 `parallel_tiling`，是为了不让 source encoder 的启用/回退同时改变
reference 编码或 decoder。`auto` 默认仅启用已验收的 deterministic/BF16/stack-off/
2或4卡 encoder 组；显式 `replicated` 可回退。`performance` 不被自动改成 deterministic。

## 与既有 MiniMax H3 encoder SP 的逐项对照

仓库中的旧实现是 `minimax_h3_video_vae/klvae.py` 的 `tiled_encode`。

| 项目 | 旧 H3 `tiled_encode` | 本 PR progressive encoder |
|---|---|---|
| 输入时间范围 | 一次处理调用方提供的完整 clip | 首次1帧，随后8帧、8帧增量；EOS 可复制末帧补齐后裁剪有效输出 |
| tile 几何 | `split_tiles` | 直接复用同一函数与参数 |
| tile 归属 | `_local_tile_indices`，`i % P` | 直接复用，clip 内归属固定 |
| 本地前向 | `encode` / `_run_tile_tasks`；可带 cls token 收集 | 支持的 causal CNN 增量前向，按全局 tile ID 保存逐层历史/stride 相位 |
| gather | 等长任务维 padding，all-gather，再恢复全局 ID | 同样的 padding/排序规则；将分配与 gather 拆开，以便分配失败先做错误汇合 |
| 拼接 | `_assemble_tiles` 和原 overlap blending | 直接复用，保持 row-major 拼接与浮点运算顺序 |
| 状态 | 不跨调用保存逐层增量历史 | 每 tile、每层的 session 私有历史，17帧 clip 边界释放 |
| tiles 少于 ranks | replicated fallback | 同样回退，所有 rank 安全参与，无空任务 gather |
| 生命周期错误 | 普通同步调用 | 额外核对 session、水位、EOS、准备/计算/拼接/切片阶段，避免其他 rank 卡住 |

没有直接调用旧 `_all_gather_tiled_results`，因为它把本地内存分配和 collective 包在
一起；如果某 rank 在分配时失败，其他 rank 可能已进入 gather。本实现先分配、汇合
错误，再执行设备 collective。没有复制另一套 tile 几何或 blending 算法。

progressive 仅支持构造时检查过的 causal `EncoderFCN3D`、零时间 padding、时间隔离
normalization。旧 encoder 的其他架构/cls-token 分支不因此获得 progressive 支持。

## 数学上与单卡的关系

这里的单卡基准是**相同 tiling 几何的单卡 progressive encoder**，不是整幅图像不切
块的 encoder。空间切块的边界 padding 和 overlap blending 已由原实现定义；本 PR
不声称空间切块与整图不切块严格等价，也不把旧 clip17 与 progressive 的误差自动放行。

令 `T_i` 为第 i 个空间 tile，`H_i(t)` 为该 tile 在第 t 次 pop 前的逐层历史，`X_t`
为本次新增 RGB，`E` 为增量 encoder（含 quant_conv）：

```text
(M_i(t), H_i(t+1)) = E(T_i(X_t), H_i(t))
M(t) = A(M_0(t), M_1(t), ..., M_(N-1)(t))
```

`A` 是原始顺序下的 `_assemble_tiles`。replicated 模式每个 rank 都求出全部 N 项；
P 卡并行只将任务集合分为 `I_r = {i | i mod P = r}`。rank r 只计算、保存 `I_r`
的状态，gather 取回各项并按全局 ID 恢复序列，随后各 rank 执行相同的 `A`。

这是任务分配，不是将一个 tile 的卷积或 normalization 拆到不同卡上。设备通信是
all-gather，不是改变求和结合顺序的 all-reduce。stack-off 下每个 tile 的输入形状、
历史、参数和计算顺序保持一致。历史只沿时间依赖本 tile，不跨 tile 传播。

对于支持的 causal 层，保留卷积所需历史及 stride 相位后，新输出与同一 tile 的
相应因果前缀有关，不依赖未来帧。时间隔离 normalization 是这个条件的重要组成。
第一个输入帧即可产出1个 latent，水位9/17分别再产出2个 latent；到17帧重置历史，
但保留全局 latent 索引。不能为了并行等满17帧再一次输出。

posterior 在完整 moments 拼接后按绝对 latent 索引和同一 seed 生成噪声；rank 或
网络分包不参与噪声索引。因此相同 moments、同一索引/噪声策略应得到相同 posterior。

### 数学等价不自动等于 bitwise 一致

bitwise 还要求相同精度、后端、算子实现、batch 形状和浮点操作顺序。确定性算法约束
并不保证跨所有 GPU 型号或库版本的普遍一致；验收冻结这些条件并实际比较完整字节。
本轮 deterministic 的同 SP replicated/parallel 已通过；performance 允许 cuDNN
自动选择算法，replicated 自身跨卡已出现差异，故不能无条件宣称 SP1/SP2 bitwise。

新增的跨 SP 验证先由 SP1 保存完整 moments 和 indexed posterior，SP2/SP4 在相同
最终提交、权重、输入、精度和后端下逐字节对照这个共同基准。它与“每种 SP 内部
比较 replicated/parallel”是两项独立检查；旧的同 SP 通过不冒充跨 SP 已通过。

## 缓存和 `agree` 分别是什么

每个进程有独立的 Python session、`CausalEncoderCache` 和本地 tile tensor。
`cache` 不是跨卡共享对象，也不广播全部历史。`agree(value, error=None)` 只在
`vae_decode` 的 CPU/Gloo group 上交换小的控制元数据；moments 才走设备 group。

它先通过有超时的 monitored barrier 确认所有 rank 到达，再 all-gather
`(error is not None, value)`：任一 rank 失败则共同抛错；否则要求所有 value 相同。
异常对象/内部文本不作为 payload 传给其他 rank。未启用并行时不增加这些通信，
本地 error 直接抛出。超时不代表可以继续使用损坏的进程组，调用方须结束请求。

`pop` 开始的 tuple 各字段如下：

| 字段 | 检查目的 |
|---|---|
| `"pop"` | 阶段标记，避免 rank 对上不同阶段的 collective |
| `session_sequence` | 各 rank 同顺序创建的 session 序号，防止交错时把 A/B 两个会话混在一起 |
| `disposed` | 防止某 rank 已释放而其他 rank 继续使用 |
| `ended` | 明确是否收到 EOS；相同剩余帧数下，EOS 决定能否补齐尾组和输出最终 slice |
| `accepted_frames` | 已接收的绝对 RGB 水位 |
| `emitted_latents` | 已输出的绝对 latent 水位 |
| `buffered_frames` | 当前未消耗帧数，决定 ready 与本次消费量 |
| `_clip_frames` | 当前17帧 clip 进度，决定本次需要1帧还是8帧、是否重置历史 |

`ended` 不能仅由缓冲区长度推导。例如同样剩4帧，一卡收到 EOS 后会补齐并进入编码，
另一卡未收到 EOS 则不 ready；不提前核对就可能让一边返回、另一边等待 collective。
如果某 rank 根本没调用 `pop`，则依靠 monitored barrier 超时检测，不能保证无等待。

其他 `agree` 阶段：

| 阶段 | 内容 / 目的 |
|---|---|
| `prepared` | 输入切片、尾组补齐、权重 dtype、设备搬运和归一化参数准备是否成功 |
| 几何检查 | 输入 shape/dtype、回退原因、tile 几何、tiling/stack 配置一致 |
| `computed` | 本地前向、输出 shape/dtype 与 gather 缓冲分配是否成功；确认后才设备 gather |
| `assembled` | gather 后按固定顺序拼接是否成功 |
| `slice_ready` | 有效 latent 裁剪/克隆、剩余输入克隆及返回对象构造成功后才提交状态 |

## 为什么增加 `try`

单卡可直接抛异常，多卡不能让某个 rank 在其他 rank 进入 collective 前自行退出。
`try` 捕获本地异常，随后由所有 rank 调用同一 `agree` 阶段共同失败；它不忽略错误，
也不把失败结果当作成功结果继续执行。逐层历史先写入 pending，`slice_ready` 成功
后才 commit，避免失败导致部分 rank 提前推进历史或水位。

这覆盖可捕获的 Python/分配错误；进程被杀、异步 CUDA 错误、collective 内故障仍
依赖 launcher、进程组超时和整次请求清理，不承诺任意设备故障的事务恢复。

## 为什么需要 `last_execution`

`last_execution` 是最近一次 pop 的执行记录，不是下一次前向的输入，也不是共享
缓存。它只保留标量、字符串和列表：请求/实际模式、回退原因、tile IDs、shape、
逐层计算量、真实状态 bytes、gather 次数/payload，以及 session 补充的绝对水位。
不持有 tensor，因此不延长显存生命周期。

它让测试和报告能够证明“真正分卡”“无遗漏/重复”“首帧即可输出”“缓存有界”，
避免只看到 SP 参数就宣称 encoder 已并行。`state_usage` 汇总实际保留状态；CUDA
allocated/reserved 另行测量；这些都不能代替 GPU trace。首次记录写日志用于排查，
逐 pop 记录进入原始验收 JSON。

## 回归与 GPU 验证入口

- `test_zing_h3_progressive_tiles.py`：真实小 CNN、多进程 Gloo、完整 moments 字节
  对照；默认 BF16 SP1/SP2/SP4 共用一个 SP1 基准；另覆盖不可整除、fallback、
  stack 和异常一致退出。CPU 结果不代替真实权重 CUDA 结果。
- `validate_progressive_tiles.py --parallel-mode auto --sp1-reference <私有目录>`：
  SP1 先生成完整基准，SP2/SP4 再读取；同时检查同 SP 和跨 SP 的 moments/posterior。
  基准绑定完整源码 SHA、输入/权重/config 哈希、精度策略、库版本，逐文件校验哈希，
  缺项/改动/非 SP1 基准失败。目录包含 source 衍生 tensor，不公开发布。
- EOS 1..34/120 帧、4种分包、水位、未来帧、生命周期及10/60秒计时继续保留。
  默认路径测试必须确认 `requested_mode=auto`，并检查 SP2/4 实际为 `parallel`，
  不能把误回退导致的 exact 成功当作分卡成功。
