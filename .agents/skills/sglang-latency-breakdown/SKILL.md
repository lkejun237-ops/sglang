---
name: sglang-latency-breakdown
description: 串联 SGLang Zing H3 三角色 source 流式路径的延迟分解实测、数据核验与 HTML/PNG 可视化；支持只重绘已有 JSON。用于测已部署实例或解释逐帧等待、计算和返回耗时；不替代通用模型 benchmark、部署或 GPU 申请。
---

# SGLang 延迟分解实测与可视化

目前适配 Zing H3 Encoder / DiT / Lynn Decoder，经进程内 Room 回放白膜和按键。
不把这个适配范围描述为所有 SGLang 模型均可使用。
若目标是常规集成 pipeline / 统一 Diffusion Server 输出路径，先核对对应适配；
不能用当前 Decoder 直出路径的结果替代那条链路的验收。

在目标 SGLang checkout 的仓库根目录执行。先读
[延迟分解使用说明](../../../benchmark/whitebox/LATENCY_BREAKDOWN.md)，复用现有采集器和渲染器，
不在 skill 中复制供帧、建组、埋点、指标计算或上传脚本。

## 选择执行路径

- **实测并出报告**：用户要求测试实例、重新采集或比较延迟时，使用 `run_latency_breakdown_replay.py --mode real-sglang`。
  它已串联素材回放、采集、JSON 和 HTML/PNG 输出，不需要再写一个总控 runner。
- **只重绘/解释已有数据**：用户已提供 `latency-breakdown-data.json` 且只要求调整图表、缩放或解释结果时，
  使用 `latency_breakdown_visualizer.py --data-json <JSON>`，按需指定 `--output-html`、`--output-png`、
  `--zoom-window START_S END_S`。不重新请求服务或占用 GPU。默认输出到 JSON 同目录；保留旧版本时显式指定新输出路径。
- **mock 演示**：仅在用户明确要求模拟、UI 试用或 CPU 冒烟时用 `--mode mock`。
  模型服务失败、缺凭据、无空闲会话时不能以 mock 替代实测。

## 实测与数据核验

1. 核对客户端与三角色运行版本、模型/checkpoint、配置和服务身份。普通请求 `return_stage_timing` 默认关闭，
   工具用 `--stage-timing on` 为当前会话开启；只升级客户端不能补齐旧服务没有记录的指标。
   `--server-revision` 只是部署方提供的记录，不是自动验证结果。
2. 使用已获授权、空闲的测试实例：现有 Coordinator 和隔离三个 service HTTP 地址二选一，
   按使用说明配置路由和凭据。保持单活跃会话，测试结束核验释放；失败或释放不明时先报告，
   不自动重启 Worker、接管路由或重试占用会话。申请资源和部署沿对应工作流及当前授权处理。
3. 核验素材身份、实际宽×高、输入事件及案例窗口。480/736 独立运行、独立输出；
   自定义缩放素材标为派生诊断输入，不改称冻结产品 case。沿用用户指定的采样、SP 和分组配置。
4. 每次使用独立输出目录，保留命令（无密钥）、完整源码 SHA、运行配置和原始 JSON。
   核对 `summary.mode`、`tool_revision`、`server_revision_supplied`、输入哈希、请求尺寸、
   供给/输出帧数、注入事件数、timing chunk 数和阶段覆盖率；缺失为未测，不以 0 补齐。
   保存失败、未执行及服务释放结果，不能只凭脚本退出码宣称验收通过。
5. 运行结束才生成的 HTML/PNG 不属于服务端推理耗时。服务端计时、Room 录制/编码和离线渲染的
   成本分别解释；若比较收益或采集开销，遵循
   [统一 benchmark](../../../benchmark/zing_h3_causal/CLIENT_BENCHMARK.md) 与
   [计时规范](../../../benchmark/zing_h3_causal/STAGE_TIMING.md)，固定条件、预热与重复次数。
   同版本用 `--stage-timing off/on` 对照，只改变该项并交错运行，记录该字段也启用的既有 stage timing；没有可靠对照时标为开销未测。

## 重绘、解释与交付

- 渲染器要求 schema v2。旧数据缺少修正后的测量边界，不能只改 schema 编号来升级。
- 重绘沿用原始 `summary.mode` 和测量身份；新 renderer 版本不能把旧数据变成新 runtime 实测。
  在交付说明中明确标注 real-sglang 或 mock，并保留数据来源。分享页要让读者能看到该区别。
- 逐帧阶段条是耗时示意，不是实际 GPU 起止时间。同组计算耗时由多个帧共享，不累加为工作量，
  不用阶段和与 E2E 的差值证明流水线重叠。`dit.return_ms` 也不能称为纯 DiT→Decoder 网络耗时。
  具体测量边界以使用说明为准。
- 交付原始 JSON、HTML、PNG，概述有效样本数/覆盖率、主要延迟、失败与未测项。
  正式性能验收还需统一 benchmark 的 E2E 和三角色性能/显存证据，本工具的短片诊断不能替代它。
- 用户要求可分享链接时，复用已安装的 `oss-report-preview` 发布流程及已有授权；
  没有该 skill 时交付本地产物并说明未发布，不自行复制一套上传逻辑。
  发布只选择获准分享的报告与附件，核对页面上的数据模式和版本、对象完整性及浏览器可访问性。
