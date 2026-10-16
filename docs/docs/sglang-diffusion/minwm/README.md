# MinWM 设计文档

架构、接口契约与设计说明保存在 `specs/`，实现计划与设计分析保存在 `plans/`。
已有的历史方案保留原有状态和适用范围；当前部署遵守仓库的单活跃会话约定。

| 主题 | 文档 |
| --- | --- |
| Realtime API 实现与阅读路线 | [变更指南](specs/CHANGE_GUIDE.zh-CN.md) |
| 后端接口 | [接口说明](specs/BACKEND_API_ZH.md) |
| 天鹏模型直连协议 | [WebSocket 接口](specs/TIANPENG_DIRECT_WEBSOCKET_API_ZH.md) |
| 启动期显存与会话资源 | [KV cache pool 设计](specs/MINWM_STARTUP_KV_CACHE_POOL_DESIGN.zh-CN.md) |
| 性能瓶颈与优化方案 | [性能分析](plans/PERFORMANCE_ANALYSIS.zh-CN.md) |

其他历史设计和计划见 [specs/](specs/) 与 [plans/](plans/)。
测试和复现入口保留在 [parity harness](../../../../benchmark/minwm_realtime_parity/README.md)
和 [async VAE benchmark](../../../../benchmark/minwm_realtime_async_vae/README.md)。
