# #195 performance progressive encoder 验收（r1）

状态：实验实现，默认关闭。2026-09-28 的 SP1/SP2 正确性矩阵、encoder 与服务 ABBA、全矩阵媒体和逐 rank trace 技术验收已完成。数值门槛仍失败，人工画质待用户判断；不得据此声明无损或默认采用。

## 历史依据与范围

#178 合并提交 `5b1b93db8cb7cf6bd3008adc3de1830d8e524bc4` 的
`PROGRESSIVE_TILE_PARALLEL_DESIGN.md` 明确记载 performance replicated 本身跨卡
存在差异。原验收零容差失败保留：SP2 moments max abs 0.125，relative L2
0.002116739（`PROGRESSIVE_SOURCE_SP_RESULTS.md`）。这说明原 auto gate 限制已验证
数值范围，不是 tile 算法不可分片。

原生 Zing `docs/h3/vae-cudnn-ab.md`（#307，4308b7a2）记录 H200 6.21% encoder
收益，同时报告 FP32 视频 latent max abs 0.1238954、单帧 relative L2 0.0026373。
该结果不是本次硬件收益或生成质量证据。minWM 的经验仅用于固定输入、完整预热、
独立进程和分开统计首轮成本，不套用其硬件配置。

实验开关 `MINIMAX_H3_PROGRESSIVE_ENCODER_PERFORMANCE_AUTO=1` 在 session 构造时
冻结，使 performance + BF16 + stack-off + SP2/4 的 auto 真实分片；默认关闭。
原 deterministic gate、显式 replicated/parallel、少 tile 回退均保留。
SP4 在实际 GPU 验收前仅为实验可执行配置，不宣称支持。此开关与 #196 padding、
#197 batch 优化独立；本次两者关闭。后端策略复用 `_visual_encode_policy`：
benchmark=True、cudnn.deterministic=False、deterministic_algorithms=False，正常与
异常恢复包含 warn_only 的全局状态。不同策略不可跨线程并发执行。

## 运行前冻结的数值标准

这是新的非 exact 工程误差门槛，不改变 #177/#178 历史判定，也不证明训练质量：

- 搬运、tile ID 排序、同一已计算 tensor 的 gather、posterior 固定输入与 RNG、
  状态释放及时间线：bitwise / exact；不允许遗漏、重复、错序、未来帧污染。
- 卷积策略产生的完整 moments 与 indexed posterior 对比：shape/dtype 相同、
  全部有限，max abs <= 0.25 且 relative L2 <= 0.005。定义 relative L2 为
  `norm(actual-reference)/max(norm(reference),1e-15)`，FP64 归约。
  依据为上述历史同类 BF16 差异约 0.125 / 0.00264，约两倍包络作为预先固定的
  保守工程上限；它不是通用 BF16 误差界或感知质量界。保留 bitwise 结果、完整
  max abs、relative L2、tensor hash 及失败张量。超限停止依赖验收，不能事后放宽。
- deterministic 同策略完整 moments/posterior 继续零容差。网络分包、生命周期
  在同进程相同策略下继续 exact；跨进程 performance 重复不宣称 bitwise。
- 完整输出视频技术验收和人工可视检查独立记录，未完成不得声称质量通过。

## 冻结矩阵与采用门槛

同设备 SP1/SP2 起步，SP4 可选实测。冻结最终源码、权重与镜像 digest、GPU UUID、
库版本、原始 source 及六图 SHA256。默认 case
`zing-whitebox-video-five-triview-0001`，832×480、24FPS、四步、seed42、video-only。
使用当前 source 协议，不传旧 prompt_video_frame。共享 CPFS 资产只读，任务输出
按 revision 写入独立目录；本轮为 `zing195-20260928`，`zing195-20260924`
保留为历史实验标识。复用素材前重新核验字节哈希。

主对照为 deterministic auto 与 performance auto（实验开关开启），相同 SP 与
同一物理卡组，独立进程按 A/B、B/A 执行。另做 performance replicated/auto
数值与分工消融，避免混算策略和分片收益。每进程完整预热至少两个17帧 clip，
覆盖首1帧与两种8帧历史状态；保留冷启动样本。10秒/60秒每组至少3次无 profiler
正式测量，另采逐rank原始 Perfetto trace，不把 profile 样本纳入收益。

逐rank保存 tile IDs、真实输入 shape、encoder ranks、fallback、显存、缓存平台期；
SP2 默认12 tiles，每卡6个。完整 encoder 包含 gather/拼接，报告1/2 latent分组
p50/p95、首latent、端到端FPS、输出间隔p95。采用门槛：encoder中位耗时改善至少
5%，首latent及端到端延迟/输出间隔p95不退化超过5%，无OOM、死锁或泄漏。
未通过或未实测保持实验开关，不能将CPU测试冒充GPU验收。

## 当前验收表（2026-09-28）

| 项目 | 状态 |
| --- | --- |
| 独立 auto 实验开关、状态冻结和执行回执 | 已实现，默认关闭 |
| CPU gate / 生命周期 / 后端恢复 | 冻结版本原 656 项及验收器 13 项通过；同步 main 后 1066 项通过 |
| SP1/SP2 deterministic 真实权重数值/生命周期 | 每 rank 140 例 exact，生命周期和异常恢复通过 |
| performance 数值 | 140 例已收齐；同模式/跨模式超限仍记失败，详见结果表 |
| SP2 实际分工及 gathered 输出跨卡一致性 | 每卡 6 tiles，140 例 moments/posterior 跨卡 bitwise |
| 10/60 秒独立进程 A/B B/A encoder 计时 | 已完成；SP1 耗时减少约 8.2%，SP2 约 7.8% |
| encoder 显存、缓存释放 | 两长度显存一致；各正式流 dispose 后 retained bytes = 0 |
| 逐 rank 原始 trace、Perfetto | 12 份 encoder 与 96 份服务原始文件已分析；encoder 12 视图、服务六组原始/视图 UI 抽检完成，完整大文件 UI 限制保留 |
| 当前协议六图端到端及完整媒体 | SP1/SP2 ABBA、80 个服务请求、56 份正式/profile/画质视频技术验收完成；收益见结果表 |
| 原始 120 帧 5 秒画质 | SP1/SP2 两臂及并排输出均技术验收通过，人工判断待用户 |
| SP4 | 未实测，不宣称支持 |

详见 [真机结果与历史失败证据](PERFORMANCE_ENCODER_RESULT.md)。数值门槛保持不变；
正式 encoder 收益不等于完整服务收益或质量采用通过。

## 复现入口

下列变量指向同一冻结 revision 的源码、私有模型/素材与全新结果目录。模型目录
只允许 `encode_mode` 和对应 realtime profile 的 config hash 随策略改变；其他配置
逐项相同。不要在原目录覆盖结果，也不要在 GPU 作业中下载/准备权重。

```bash
export PYTHONPATH="$SOURCE_ROOT/python"
export SGLANG_LIGHTWEIGHT_RUNTIME=1 SGLANG_DIFFUSION_PLATFORM_OVERRIDE=cuda
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MINIMAX_H3_PROGRESSIVE_ENCODER_PERFORMANCE_AUTO=1
unset MINIMAX_H3_PROGRESSIVE_ENCODER_TILING
# SP1 先写 oracle；SP2 读取同一 oracle。两模式分别保留独立基准目录。
# 每条 torchrun 均为新进程；固定同一物理 GPU 子集。
for sp in 1 2; do
  for mode in deterministic performance; do
    extra=()
    if [ "$mode" = performance ]; then
      extra=(--deterministic-reference "$RESULT_ROOT/sp1-deterministic"
             --deterministic-config-sha256 "$DETERMINISTIC_CONFIG_SHA256")
    fi
    CUDA_VISIBLE_DEVICES="$GPU_IDS" torchrun --standalone --nproc_per_node="$sp" \
      "$SOURCE_ROOT/benchmark/zing_h3_causal/validate_performance_encoder.py" \
      --model "$ASSET_ROOT/model-$mode" --mode "$mode" --phase correctness \
      --rgb "$ASSET_ROOT/input/source-1440.rgb" --rgb-sha256 "$RGB_SHA256" \
      --source-sha "$SOURCE_SHA" --source-manifest "$SOURCE_ROOT/source-manifest.json" \
      --sp1-reference "$RESULT_ROOT/sp1-$mode" "${extra[@]}" \
      --output "$RESULT_ROOT/correctness-sp$sp-$mode"
  done
 done
```

计时使用相同命令的 `--phase timing`，移除两个 oracle 参数及 deterministic config
hash 参数，按 deterministic/performance/performance/deterministic 顺序分别启动独立
进程，输出到不同目录。每进程内记录冷启动、完整预热和各流长度3次正式样本。
`--phase profile` 单独执行，每个 rank 输出自己的 `encoder-240.json` 与
`encoder-1440.json`，可导入 Perfetto；不使用 profile wall time 计算改善。

这些命令只验收 encoder。完整服务必须另用当前 `continuous_source_client.py`，真实
六图通过 `--reference` / `--reference-id` 提交，原始 source 通过 `--source-rgb` 提交，
不能传已删除的 `--prompt-video-frame`。独立进程、预热、10/60秒三次正式测量及原始
server/client 命令均保留在本轮受控工件，未完成的服务/媒体行继续标 pending。

## 2026-09-28 补充采集授权

用户授权继续完成 SP1/SP2、正式性能、长流与逐 rank trace，并要求画质只使用原始
120 帧 / 24 FPS / 5 秒素材。本轮不修改既有数值门槛，也不默认开启功能。
显式 `--report-numerical-only` 将 finite、同格式的数值差异保留为失败并继续收集；
NaN/Inf、格式错误、分包不一致、跨 rank 数据搬运不一致及生命周期错误仍停止该阶段。
正确性、计时、profile、服务是独立进程；任何阶段失败均保留原日志和非零退出码，
其余独立采集可继续，最终分别汇总，不能将采集完成写成整体通过。

性能仍用冻结的 10/60 秒重放输入，每进程预热后各三次、独立 A/B/B/A；画质新增
5 秒请求，不重放、不截取长流代替独立生成。A 为 deterministic auto，B 为 performance
auto；两者均保持六图、四步、seed42 和 SP2 每卡六 tile。SP4不在本轮范围。
