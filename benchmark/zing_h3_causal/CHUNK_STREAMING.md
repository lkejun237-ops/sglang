# Zing 流式运行时验收入口

运行时设计说明已移到代码目录：[STREAMING_DESIGN.md](../../python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/zing_h3/STREAMING_DESIGN.md)。其中区分逐 latent / 逐 frame 的目标、当前 TAEH3 输出限制、分次调度和私有 ZeroMQ 输入通道。

本目录保留可重放实验和验收材料：

- [SLOW_CONSUMER_LIFECYCLE.md](SLOW_CONSUMER_LIFECYCLE.md)：超时取消、会话释放与资源证据及其生产限制。
- `test_encoder_overlap_tools.py`：验收工具契约的 CPU 回归，不能替代 GPU 验收。
- `--stage decoder` / `progressive-acceptance.json`：真实权重 latent 重放与原路径零容差对比；修改解码默认后必须重新冻结并验收。
