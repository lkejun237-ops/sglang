# 视觉编码器完整产品用例 A/B

`stage.py` 从已推送、已记录 SHA-256 的 runtime/harness 归档做 CPU 准备，校验 checkpoint、素材和依赖后输出可恢复的准备包；GPU 阶段只还原该准备包，不再修改源码。三角色运行方式沿用已验证的 main product / latency matrix harness。

`runner.py` 对同一冻结 `product-case-20261103` 八个 case 依次执行 baseline 和 candidate。两组只有 Encoder 的 `MINIMAX_H3_ENCODER_SINGLE_GPU_FAST` / `MINIMAX_H3_FAST_VISUAL_ENCODE` 从 0 变为 1；其余硬件、源码、checkpoint、采样、并行布局、seed=42、输入与动态边界均一致。每个 case 一次完整预热、三次正式请求；736p case 保留原 480p source → 1280×736 output 身份，不覆盖尺寸。使用独立且本次拥有的 Encoder/DiT/Decoder，finally 拆组、停止进程并清理本次 KV 缓存。

性能轮统一使用 metrics，随后独立执行一次完整视觉请求，不将视频落盘开销混入性能统计。视觉轮使用 `--media-mode video`：保存全部原尺寸 RGB 为本次临时 spool，计时结束后以 24 FPS / CRF18 编码完整 MP4，成功后删除 spool。其他请求只留指标与全量 RGB 哈希。运行时临时盘峰值至少约 7 GiB，结束留存所有组合计仍按 2 GiB 验收。视频是有损压缩的肉眼对比工件，不作为像素 parity；其计时口径包含 RGB 落盘背压，应只与同模式 baseline 比较。

CPU 预检检查 FFmpeg/libx264、模型与全量素材身份。`STATUS.json` 和每组 `requests.jsonl` 保留进度与失败；不能将平台任务成功等同于生成验收。此 harness 关闭请求 profiler，因为独立 Encoder 的该协议尚不支持；未启用下述 diagnostics 时，三组件耗时及 allocated/reserved 分项未测，不能据此推断组件归因。

完整视频测试：`python3 -m pytest -q benchmark/zing_h3_causal/encoder_fast_validation/test_video_capture.py`。

数值优化对照：`RUN_CONFIG.json` 设置 `comparison="numerical"`，两组 FAST 开关固定为1，
只切换 `MINIMAX_H3_ENCODER_NUMERICAL_OPTIMIZATIONS=0/1`；这不等同于此前 FAST 总开关 A/B。
设置 `diagnostics=true` 时，各 case 先做1次预热及1次显存诊断请求，记录各实际 GPU 进程
whole-role allocated/reserved 常驻与请求区间峰值、0.2秒整卡driver采样、角色 `/metrics` 前后快照。
driver 首个采样在请求内，只称请求内首样；allocator 常驻使用请求前 checkpoint。
Encoder 的进程值包含 Qwen、参考图VAE及sourceVAE，不冒称各子组件独占；DiT逐rank记录。
复用 `memory_probe` 的summary-only模式，不记录大型allocator历史/快照；诊断结束所有进程确认
`disable-for-timing` 后再做原有1预热+3正式metrics和独立完整视频轮。诊断计时不进入正式性能。
组件Prometheus P50/P95由bucket区间估计，和客户端逐请求精确统计区分；丢样计数必须报告。
