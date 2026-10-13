# Benchmark 文件报告：2026-10-12 验收

本补充承接 PR #275 / issue #253。沿用已确认的 `5s-results.html` 布局，
从本次 benchmark 输出生成离线 HTML 和版本化结果 JSON；网页支持打开多个
JSON、逐文件切换、重新导出以及原有 chunk CSV 下载。没有服务器或多端同步。

## 场景与观察

| 场景 | 预期与实际观察 | 证据 |
|---|---|---|
| 新测量目录生成报告 | 从到达记录复算指标，排除预热和失败；失败与未执行数保留 | Python 合同检查通过 |
| 保存结果 | 默认与 HTML 同名的 JSON，或指定 `--json-output`；包含配置、请求、chunk 和可选阶段记录 | 导出合同及浏览器下载回读通过 |
| 一次打开多个文件 | 有效文件各自加入选择器，切换时更新元数据、指标、曲线、阶段数据，不合并不同文件 | Chromium 检查通过 |
| 错误文件与有效文件混选 | 不符合格式或请求/帧数对应关系的文件明确拒绝，其余有效文件仍可查看 | 浏览器负例通过 |
| 导出后再次打开 | 保留正式请求与双 rank 阶段记录，重新复算后显示一致 | 浏览器 round trip 通过 |
| 页面与可移植性 | 保留参考页面的 CSS 和主要区块，390px 下没有页面横向溢出；无外部请求 | Chromium 检查通过 |

本地受影响测试：**63 passed / 1 skipped**；跳过项依赖本机缺少的 ffmpeg。
适用仓库格式检查通过。短报告合同进入现有 CPU CI；浏览器检查是本次开发验收，
不新增浏览器依赖或 CI 服务。

导入文件的模式名、chunk 显示字段与预热失败数均以文本渲染。开发浏览器回归
`tests/check_report_import.cjs` 对原始报告可复现 HTML 注入，修复后的模板通过检查：
恶意字符串保留为文本，不执行脚本、不发起资源请求。该检查使用开发环境的 Playwright，
不增加 CPU CI 依赖。

```bash
python3 -m pytest -q \
  benchmark/zing_h3_causal/tests/test_benchmark_report.py \
  benchmark/zing_h3_causal/tests/test_bench_client.py \
  benchmark/zing_h3_causal/tests/test_run_benchmark.py \
  benchmark/zing_h3_causal/tests/test_stage_timing.py \
  benchmark/zing_h3_causal/tests/test_analyze_stage_timing.py \
  benchmark/zing_h3_causal/test_whitebox_video_smoke.py \
  benchmark/zing_h3_causal/test_whitebox_six_reference.py \
  benchmark/zing_h3_causal/test_whitebox_video_long.py
```

## 版本、工件与证据边界

首轮文件报告验证使用 2026-10-10 GPU 作业 `dlc94vx1ng4fyr15` 的 off/host/cuda
5s 数据；运行身份见 [阶段计时验收](STAGE_TIMING_VALIDATION.md)。后续补齐三个
case 与 checkpoint 简化入口，并完成新云端实测，见
[三 case 验收记录](THREE_CASE_BENCHMARK_VALIDATION.md)。模型与采集器未修改。

网页 FPS 沿用参考页的首包后口径，区别于原始 benchmark 的全程 FPS；两种公式
在页面中分别说明，原始指标文件不改写。缺少 CUDA 或逐帧时间戳不补造数值；
不同主机时钟域不绘制共用 host 时间线。阶段展示沿用现有 SP1/SP2 范围。
格式与计数检查不等于严格 ledger、媒体完整解码、人工画质或性能结论通过。

首轮报告验收的 720 帧输入仅用于 CPU/浏览器合成合同，未做真实 30s 业务生成。
2026-10-12 后续按用户确认补齐海岸 5s（六图）、废土 30s 和地中海小镇 30s
（七图）的默认矩阵与阶段验收。新输入保留 PR #261 的准确七图身份，不冒充旧六图 ID。
补充实测已经完成，作业状态、三组结果、Qwen 未独立测量的边界、媒体与最终工件记于
THREE_CASE_BENCHMARK_VALIDATION.md。

本地工件根：`/home/ss80194/work/sglang-zing/exports/issue253-stage-5s-20261010/r2/`。
`benchmark-report.html` / `benchmark-report.json` 为页面与便携结果；
`benchmark-report-validation.json` 保存 SHA 和验证摘要；
`benchmark-report-files-desktop.png` 为浏览器截图。运行数据与截图留在工件目录，
不提交媒体或冻结历史数值到源码。使用方法见
[客户端 benchmark 说明](CLIENT_BENCHMARK.md)。

三 case 实测工件根：
`/home/ss80194/work/sglang-zing/exports/issue253-three-cases-20261012-r1/`；最终网页位于
`view/benchmark-report.html`，同目录 JSON 可再次导入网页。该目录同样不提交到仓库。

另对照原始 JSONL 和当时的 stage-validation.json，更正文档中误写的调用数：
每请求每 rank 为 8 Encoder、22 DiT、8 Decoder、1 Conditioning；总计仍为 39，
每模式含预热仍为 312。计时数值不变。
