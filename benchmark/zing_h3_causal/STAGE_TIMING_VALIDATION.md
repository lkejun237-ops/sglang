# 固定 5s 模型阶段计时：2026-10-10 r2

使用客户端基线指定的 source + 六参考 case，120 帧 / 24 FPS；有限接口在两张
SP2 GPU 内完成全部编码、DiT 与解码。三个模式各自启动常驻服务，完整预热一次，
再串行测三次，并发为 1。30s 等准确 payload，本轮没有替代用例。

| 模式 | 正式成功 | TTFF 中位数 (s) | E2E 中位数 (s) | FPS 中位数 | TPOC 中位数 (s) |
|---|---:|---:|---:|---:|---:|
| off | 3/3 | 11.7298 | 20.9434 | 5.7297 | 0.4450 |
| host | 3/3 | 11.6913 | 21.2673 | 5.6425 | 0.4560 |
| cuda | 3/3 | 11.6264 | 21.2319 | 5.6519 | 0.4572 |

所有 12 次请求（包含 3 次预热）均输出 120 帧、22 个客户端 frame batch。
host/cuda 分别保留 312 条阶段记录：每请求、每 rank 各有 8 个 Encoder、15 个
DiT、15 个 Decoder 和 1 个整体 Conditioning 区间。货号、连续 latent/frame
范围、每个 Decoder 输出 block 与客户端帧范围均核验通过；CUDA 区间完整。
九个正式输出的原始 RGB SHA 相同，转换后的九个 MP4 均为 832×480 / 24 FPS，
通过 120 帧计数及完整解码检查。这是技术完整性证据，人工画质验收待观看。

CUDA 模式下，rank 0 的每请求累计 stream 区间中位数：Encoder 7.7291s、
DiT 11.0702s、Decoder 0.0394s。Encoder 中 Qwen 0.6260s，六参考 VAE 合计
0.3003s，source VAE 6.8028s。两 rank 分别统计，不相加；Conditioning 包含
Encoder；current-stream elapsed 含等待和 CPU 提交间隙，不是纯 kernel 时间。
三样本及固定模式顺序只支持本次观察，不证明稳定采集开销或长期性能分位数。

## 作业与版本

- GPU：`dlc94vx1ng4fyr15`，Succeeded，10:47:42–11:03:44 UTC。
- CPU 预检：`dlczrnbij5hkn2db`，Succeeded；39 项合同检查，off/host 模型回归各 7 项。
- 第一次 CPU 导出：`dlcwq1ma3svhlq0l`，Failed；host rank 0 的 health JSON 在服务退出时被截断。
- CPU 恢复导出：`dlcg2u70r1q6wwwd`，Succeeded；重新执行全部 ledger 合同，结果与运行中严格 health + ledger 验收快照完全一致；原 health 文件保持原状，并单独报告损坏。
- 更早两次 CPU 预检失败分别为生产初始化导入顺序、采集器对缺失 request_id 的兼容问题；修复后冻结 r2。失败日志及作业回执保留。
- 区域 `ap-southeast-7`，GPU workspace 384 / quota `quota1betfyi1ilu`。
- 实测 NVIDIA L20D，274113 MiB/卡，CC 10.3；分配四卡、模型使用两卡，其余两卡闲置。
- Torch `2.13.0+cu130`、uvicorn `0.39.0`、websockets `15.0.1`、msgpack `1.2.2`、numpy `2.3.5`。
- 基础提交 `c945761b1c7e0dad0de3abfc4f7f4f0a864f3285` + `freeze-r2.json` 六项源码覆盖；全部 9578 源文件在 GPU 执行前校验。

本次实测 utility SHA：`355ebe7ecbe0bed03def901008d56b9ecf38218efb963b0825a96a08b3c42aa8`。
运行后补的原子 health 发布 SHA：`8c7b0b228493aa4d2ac74e78075875d47ac1440b87d6807144cf313b508bd620`。
后者只有 CPU 中断回归证据（本地 39 passed / 1 skipped、适用格式检查通过）；
当前工作树不能称为已重新做过 GPU 验证。阶段边界、CUDA event 与货号逻辑未改，
函数 AST 对比保存在 `post-run-fix.json`。恢复导出工具属于此次故障恢复，不改变
benchmark 主线的严格 health 验收，也不补造 health 数据。

## 工件与边界

本地证据根目录：`/home/ss80194/work/sglang-zing/exports/issue253-stage-5s-20261010/`。
`r2/REPORT.md` 为逐 rank 结果；`r2/results.json` 为结构化汇总；
`r2/gpu/{host,cuda}/timing/` 为原始事件；`r2/gpu/*/client/` 为原始接收记录；
`r2/host-timeline.png` 为实际 host 区间图；九个正式 MP4 在 `r2/`。
`export-receipt.json` 保留 OSS 上传回读及本地下载 SHA。

私有传输：`oss://leap-worldmodel-thailand-single-az/world-model/sglang/evals/issue253-stage-5s-20261010-r2/`。
CPFS 原始 RGB 保留于相同 run 名下的 `gpu/*/client/`。本轮重验七项素材 SHA，
沿用既有权重目录与部署元数据，没有重新逐字节哈希全部大权重。

没有覆盖 30s、source.append、独立 progressive Encoder、enqueue/queue wait、
D2H/send/browser 时间、音频或 kernel profiler。IFL 缺少逐帧接收时间，未测量。
本次不新增未经校准的数值或性能门槛。完整语义见 [STAGE_TIMING.md](STAGE_TIMING.md)。
