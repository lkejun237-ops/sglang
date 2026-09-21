# Zing H3 benchmark 工具范围

本目录是开发和验收工具，不由推理运行时导入。TAEH3 新增文件的保留理由是让正确性、
持续吞吐和画质结果可复现；启动服务只需要运行时代码、pipeline 配置与模型权重。
现有非 TAEH3 工具不因本 PR 删除。

| 本 PR 文件 | 用途与保留理由 |
| --- | --- |
| `validate_taeh3_decoder.py` | 同一真实 latent 对固定上游做三精度和分块正确性检查，独立测量 decoder/chunk 耗时 |
| `validate_real_checkpoint.py`（修改） | 复用已有完整请求入口，增加 decoder 选择、预热、重复测量和独立 profile |
| `validate_taeh3_repeats.py` | 检查每次正式 raw 的 shape、finite、latent/audio 一致性和重复性 |
| `benchmark_taeh3_stream.py` | 实际 scheduler 的冷请求、60 秒 warm 请求、首帧、chunk 与播放卡顿统计 |
| `benchmark_taeh3_websocket.py` | 上述工具的 WebSocket 模块；校验完整 RGB payload、取消重连和 prompt 更新 |
| `generate_taeh3_quality.py` | 每个 decoder 只加载一次模型，批量生成四场景，减少重复加载开销 |
| `compare_taeh3_quality.py` | 同 latent 数值对照：PSNR、SSIM、时间差分与可用时的 LPIPS |
| `taeh3_cases/*.json` | 人物、细节文字、快速运动和镜头切换的固定输入，避免比较时更换 workload |
| `request_10s.json`（修改） | 现有默认请求改为 832×480，和用户指定默认值一致 |
| `build_taeh3_report.py` | 将上述证据生成可同步播放、逐帧比较的静态审阅页；属于可选展示工具 |
| `TAEH3_VALIDATION.md` | 本 PR 的实测证据与未通过项目；属于验收记录，不是运行时依赖 |

没有提交权重、raw tensor、视频、trace 或一次性云作业 manifest。展示工具和验收记录
不影响运行时正确性，保留它们是为了让审阅者复核实验，而不是要求部署时执行。

## CPU CI 的范围

`.github/workflows/zing-taeh3-cpu.yml` 为模型级测试提供无需 GPU/权重的执行环境。
它覆盖分块一致性、尾帧、reset 和会话隔离；不覆盖 pipeline 参数或完整服务链路。
触发范围只包含该 workflow、TAEH3 模块和对应测试，benchmark 文档修改不再触发它。
测试临时移出 CUDA runtime conftest 的目录后执行，因此无需安装完整 GPU 服务依赖。
它不能替代真实权重的 GPU gate 和长会话测试。

权重位置、精度选择与复现命令见
[TAEH3 README](../../python/sglang/multimodal_gen/runtime/models/vaes/taeh3/README.md)。
