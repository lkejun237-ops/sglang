# ABAB 与按键延迟验收记录（2026-10-13）

## 结论

**尚未证明真实 H3 按键端到端延迟低于 1 秒。** 本地新协议验证通过，真实泰国测试验证的是仍在运行的旧版本基线。两者不能混为新版本性能成绩。

## 泰国真实基线

基线 ID `baseline-631340316cc6`，recording_id `373a9157351f4fc79189995eaf54bd59`。现有 GPU Job `dlc1594cgbank0ns`，step 2500 / 两张卡 SP2 / shared encoder / W6 / `[1000,500]`。未重启或重新申请 GPU。

真实 CLI、真实 H3、两路 H.264 共 55 帧均可解码；主动结束成功，slot 约 4.46 秒释放；两路录制与 OSS 上传 complete。报告和 manifest 已回读。首次重测曾因平台交付物 API 503 失败，随后两个读取接口恢复 200，再测成功，不能将第一次失败误判为 GPU 故障。

| 观测项 | 结果 | 测量边界 |
|---|---:|---|
| 唯一有对应输出的按键样本 | **11461 ms** | 测试客户端发输入 → 收到带对应 CLI receipt 的模型帧；不含真实浏览器绘制，样本数 1，不能称 P95 |
| 点击对应 init → session.ready | 39.71 s | 含远程链路和初始化 |
| init → 首个白模帧 / 模型帧 | 39.85 s / 50.97 s | 接收端时间 |
| CLI 交付物准备 | 21.04 s | 读取、下载、核验交付物 |
| CLI Room 启动 | 24.67 s | **包含**上项，不能相加 |
| Engine start | 25.09 s | 也包含 CLI 启动 |
| CLI 帧等待 | 均值 561 ms / P95 846 ms | Room 等待下一白模帧，56 次 |
| 白模 → 模型输出 | 均值 9.96 s / P95 17.09 s | 55 帧；包含源等待、推理和下游反压等嵌套成本 |
| Encoder host | 均值 26 ms / P95 120 ms | 38 chunk；host span，非 CUDA event 时间 |
| DiT host | 均值 320 ms / P95 464 ms | 7 chunk；不可和不同 chunk 的 P95 直接相加 |
| Decoder host | 均值 3.5 ms / P95 10.1 ms | 7 chunk |
| WS 接收/发送相关播放等待 | 均值 111 ms / P95 739 ms | `playback_delivery_wait`，含下游等待，不能完全归因网络 |
| 录制队列反压 | 均值 0.0093 ms / P95 0.0161 ms | 166 次；本次不是主要延迟来源 |
| 应用 RTT | 413–757 ms | 本地测试机到泰国；包括服务调度，不等于纯链路 RTT |

manifest 的实际运行信息：Chromium 151、`rendering.requested=software`、ANGLE Vulkan SwiftShader、源传输 RGB24。55 个白模帧的接收首尾相隔约 31.88 秒，即约 **1.69fps**。配置中的 24fps 是目标/模型时间轴，不是白模实测吞吐。

## 为什么换 WebRTC 不能直接解决 1 秒问题

1. CPU SwiftShader 在这个真实世界中产帧很慢。H3 progressive encoder 的源帧水位是 1/9/17；源帧少，后续 chunk 就需要等待。
2. 按 1.69fps 粗算，连续新增 8 帧需要约 4.7 秒，新增 16 帧约 9.5 秒。该数量级与本次模型出帧等待一致。它是基于吞吐和代码协议的解释，尚不能替代逐 chunk 源等待 profiler。
3. 即使提高到 24fps，新增 16 帧也需约 667ms，还要容纳 Encoder、DiT、Decoder 和中国→泰国网络耗时。W6 是注意力历史窗口，不能当成“只等 6 个 RGB 帧”。模型 chunk 大小与 source watermarks 需要联合验证。
4. WebRTC 的收益是避免可靠视频传输在拥塞时坚持重传旧像素，并限制播放积压。它不能加快 Three 世界渲染、补出不存在的源帧或缩短 DiT 运算。

## 本次代码中的减延迟措施

- CLI 采集、左侧预览与右侧 generation 解耦；慢模型不再通过同一循环停止源采集。
- 模型只取最新的未提交源帧；H3 已接收的有状态时序不丢不乱序。pending 硬上限 34 帧；该上限保护内存和协议活性，**不承诺 34 帧以内即低于 1 秒**。
- WebSocket 播放 credit 从 8 调到 2，浏览器解码待处理预算 4；慢客户端有界，避免越看越落后。
- WebRTC unordered / maxRetransmits=0 数据通道，过期分片和积压整帧可丢；WS 控制与 states 保持可靠。
- 配置 revision + generation epoch 避免切换后显示旧帧；重新 conditioning 不重启世界/GPU。
- 沿用独立录制/上传进程；增加 source age、源帧替换、RTC 丢帧计数，以及媒体编码耗时。
- 不透明 states 单独可靠下发并归档；视频允许丢帧不等于可以丢 UI 增量状态。

## 下一步达到 1 秒的验证顺序

1. **先测 A 模式。** 固定世界、分辨率、按键节奏，对比 WS / RTC 的输入回执 → 白模显示。如果 A 就超过 1 秒，先处理渲染吞吐或网络，不调 DiT。
2. CLI 世界采集需接近 24fps。优先用具有硬件 WebGL 的运行环境；也可由世界作者减少场景绘制开销，再用同一世界实测。不能仅增加 CPU 核数就承诺 SwiftShader 达到目标。当前不擅自占训练 GPU，也不修改同事 SDK。
3. H3 增加/核对每个 chunk 的源帧水位等待、实际 CUDA 执行、输入/输出队列年龄。在当前 checkpoint 上验证较小 AR chunk / progressive dispatch 的兼容性与画质，再决定配置。不要未经验证更改训练相关语义。
4. 单独比较 shared encoder 与独立 encoder overlap。后者需要额外的已确认空闲 GPU；当前两卡保持原分配。
5. 同一路线收集足够多的稳定期按键样本，分别报告 A/B 与 WS/RTC 的 P50/P95/最大值、显示 FPS、丢帧率和画质。切换初始化与稳定期分开报告。
6. 持续验收目标设为 **稳定期按键到浏览器对应帧绘制 P95 < 1000ms**；同时报告最坏值。当前没有此达标证据，不能靠丢掉慢样本宣称通过。

## 已完成的验证与未完成项

- 本地 Chrome + 真实 Gateway/Coordinator 控制代码、真实 H.264 编解码和 WebRTC 通道，通过 A→B→A→Reset→B、Prompt 变更、主动结束。CLI/H3 为合成 fixture；约 23fps、按键样本约 22ms **只证明协议链路工作，不是 H3 性能**。
- Python 测试包含两个用户隔离、真实 WS Gateway、模型租约释放重用、取消 admission 不漏租约、参考图内容寻址、冲突拒绝、失败后结束、真实 aiortc 双端收发解码、进程录制和配对 VFR 归档。
- 前端类型检查与构建、分片丢失/乱序/超限、ready 与媒体乱序、协议回归通过。
- Linux AMD64 CPU 镜像测试覆盖真实依赖；aiortc 1.15.0 要求 PyAV <18，故白模 CPU 镜像固定 av 17.1.0。H3 GPU 运行依赖未改。
- H3 adapter 的完整单元测试在本地 Mac 因缺 Triton 无法收集；新增 `[1000,750]` profile 的 CPU 渲染配置测试通过，真实 GPU 还未验证。
- 泰国未发布新协议；未配置公网 TURN；运行中的 H3 仍 `[1000,500]`。现有 r5 启动器在模型子进程退出后会结束 Job，不能直接 kill 重启后宣称保留 GPU。需要先确定受支持的保留分配升级路径。
