---
name: room-sglang-source-replay
description: 使用录制白膜源和真实 Room 供帧代码，对 SGLang 三角色运行回放集成测试，覆盖自然供帧、确定性切词、prefill/cache 及会话释放；不负责部署、GPU 申请或浏览器/WebRTC 验收。
---

# Room–SGLang 白膜供帧回放集成测试

先阅读仓库根目录下 `benchmark/whitebox/SOURCE_REPLAY_INTEGRATION.md`，用唯一入口
`benchmark/whitebox/run_room_sglang_source_replay.py`；不另写供帧、建组或结果判断实现。

- 确认使用当前已推送源码、匹配依赖、真实素材及其哈希。只接受用户指定且获授权的三个
  **service HTTP 地址**，或指定的测试 Coordinator；两者互斥。多个 service 不等于多个 Worker 进程。
- 三地址模式自动发现非秘密配置；测试机需预置共享签名密钥。缺少凭据时报清缺项，不能
  打印密钥、从 Worker 下载密钥、关闭鉴权或替换成 mock。真实模式不能使用合成素材。
- 默认 `--flow-mode natural`：复用真实 Room admission，观察供帧行为。
  验证精确切词位置时选择 `--flow-mode deterministic`，明确额外存在测试闸门。
  测试时长按输出帧数目标解释，full 不等于源 MP4 单次完整播放；详见工具文档。
- 给每次运行单独的 `--output-dir`。已有匹配素材可复用，否则使用 prepare-media 下载并核验。
  mock-worker 只用于显式请求的 CPU 回归，不用它代替真实 Worker 失败。
- 对用户授权的测试执行 run 并跟进退出结果。检查 summary.json、REPORT.md 中的模式、输入身份、
  实际边界、cache 回执、持久化计时来源、目标/实际帧数及释放证据。失败或 release unknown
  时停止追加测试并报告，不自动重启、排空、停止 Worker 或修改路由。
- 汇报完整源码 SHA、三个 service 身份或测试池路由、模式、长度目标、运行命令（无密钥）、
  结果目录、执行/失败/未测项。区分进程内 Room 集成、真实模型、已部署 Room 和浏览器验收。
  不从一次集成回归推断端到端性能收益；需要发布报告或部署时另按用户授权处理。
