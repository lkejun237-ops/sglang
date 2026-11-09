# CLI 的 GPU 部署

> 历史配方记录：参数、路径和分支描述只适用于原记录版本，不是当前部署输入。
> 当前操作统一从 [deploy/whitebox](../../../deploy/whitebox/README.md) 进入；本次没有重新进行 GPU 验收。

CLI/Room 与 H3 分别持有自己的 GPU 分配。Room 使用一张图形 API 可用的普通 NVIDIA GPU，不使用 B300；H3 使用两张 B300 做 SP2、共享 encoder。Room 单独发布，不重建 H3 Job。用户 session slot 仍由 Coordinator 管理；CLI 的 GPU 不新增模型 session 容量。

SDK 的已测基线是 V100-SXM2-16GB、8 vCPU、32 GiB（`ecs.gn6v-c8g1.2xlarge`）。泰国若无该规格，可用独立 A10 24GB、8 vCPU、30 GiB（`ecs.gn7i-c8g1.2xlarge`）验证；同规格名称或更大显存不等于性能验收。CLI 部署在 ACK 的专用 GPU 节点池，节点规格白名单排除 B300，限一张 GPU，并设置专用 taint。原有训练节点与驱动不变。

SDK 必须包含 `dev` 的 `2a445015`（PR #518），当时镜像读取固定 SDK 提交；当前 Room 使用 `sdk-revision.txt` 固定的 dev 提交。合并保留已有旧世界 `inspectLoading?.()` 兼容处理和捕获初始化修复；上游为 control、encoded、pixels 各自限制消息大小。

Room 镜像预装 Node 24、锁定依赖对应的 Chromium/Headless Shell、Vulkan/EGL/GLX 加载器。NVIDIA 设备和匹配的驱动图形库由 GPU 容器运行时注入，不能在共享训练节点安装或替换驱动。GPU 配置：

```text
WHITEBOX_CLI_RENDERING=gpu
NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display
WHITEBOX_WIDTH=832
WHITEBOX_HEIGHT=480
WHITEBOX_MAX_ROOMS=1
WHITEBOX_MAX_PLAYERS=1
```

不要设置 `NVIDIA_VISIBLE_DEVICES=all`；Room 通过 Kubernetes 的 `nvidia.com/gpu: 1` 请求独占分配。原生 RGB24 白模不需要 NVENC。Gateway 继续编码两路 H.264，通过 WebSocket 返回；A/B 只切显示，H3 和原有录制持续运行。

Room 保持单副本、稳定 Service 与 owner ID，替换时不允许新旧两副本同时接收请求。切换前结束旧 Room 的活跃会话，旧录制保留。GPU 节点可用区与旧录制盘不同时，创建新 PVC，不重建或覆盖旧盘。复用相同 Service 时无需修改 Coordinator 的 owner 路由。

在 GPU Pod 中先运行：

```sh
cd /opt/worldkit
node --import tsx /app/cli-gpu-preflight.mjs
```

预检使用 SDK 公共浏览器启动策略，创建真实 WebGL2 上下文并读回像素，检查 `implementation=hardware` 且 renderer 为 NVIDIA。没有可用图形驱动时应失败，不能静默使用 SwiftShader，也不能把 `nvidia-smi` 成功当成图形渲染成功。之后真实 CLI session 的 `runtime.rendering` 仍须为 hardware，该信息会随已有 `engine_metadata` 保存。

真实验收采用同一 WTP 世界、832×480 / RGB24 / 24fps，先测 CLI 直出，再测完整模型链路的持续 FPS 与输入至出帧 p50/p95。记录 GPU/驱动/浏览器身份、CPU 限额、采集等待和排队；简单世界在 V100 上的约 24fps 不能代替本次泰国实测。

本次继续使用已有封存交付的下载路线，不启用新 `--resource-transport` 选项，不改 SDK 的北京 OSS/资产 URL。模型资产使用 CPFS；Room 日志与新录制仍写泰国限定前缀。
