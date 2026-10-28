# Prompt Switch Block0 CPU 缓存 H100 性能与正确性报告

## 摘要

本报告记录 Zing H3 在连续流式视频生成场景下，通过统一的 `Block0PreparationCache` 实现冷启动 TTFF 与运行时 Prompt Switch 动态切换加速的实测数据。

- **测试环境**：阿里云灵骏 Eflo H100 节点（`e01-cn-2zd4xi8m505`，NVIDIA L20Z / CC9.0）；
- **拓扑规格**：E1 + SP2 + D1，832×480，120 帧（24 FPS），6 张参考图，BF16，FA4，Graph off；
- **测试 Case**：A→B→A 往返动态切换（分别在 source frame 40、80 触发切换指令，实际输出生效帧在 17、34 帧）；
- **对比变量**：Baseline (`SGLANG_ZING_BLOCK0_CACHE_MODE=off`) vs Prompt Switch Cache (`SGLANG_ZING_BLOCK0_CACHE_MODE=overlap`, `entries=6`)。

---

## 核心基准测试结果 (Formal Benchmark)

各组均经 2 次预热后进行 3 次正式请求（中位数 / P95）：

| 指标 | Baseline (重算) | Prompt Switch Cache (命中) | 优化幅度 | 说明 |
|---|---:|---:|---:|---|
| **Block0 算子时间 (DiT 50 层)** | 1052.5 ms | **150.3 ms** | **-85.7%** | 重算前向转为纯 PCIe DMA 异步恢复 |
| **完整 Prefix 重建耗时** | 1269.1 ms | **369.8 ms** | **-70.9%** | 包含 150ms DMA + 215ms 原生 clean history |
| **切换停顿 (Playback Stall)** | 1659.1 ms | **486.1 ms** | **-70.7%** | 客户端播放卡顿由 1.66s 骤降至 0.48s |
| **切换事件到首帧新画面** | 4612.2 ms | **2568.6 ms** | **-44.3%** | 新指令响应时间缩短 2.04 秒 |
| **端到端总时间 (E2E)** | 13.738 s | **9.864 s** | **-28.2%** | 全程耗时缩短 3.87 秒 |
| **首帧交付延迟 (TTFF)** | 2.832 s | **1.479 s** | **-47.8%** | 首帧耗时直接腰斩 |
| **全流程生成 FPS** | 8.735 FPS | **12.165 FPS** | **+39.3%** | 稳态生成速率突破 12 FPS |
| **首包后稳定 FPS** | 11.027 FPS | **14.193 FPS** | **+28.7%** | 首包后接近 15 FPS |

---

## 缓存行为与容量

1. **统一缓存命中轨迹**：
   - 启动时 Prompt A 恢复并驻留在 Slot 0；
   - 第 40 帧切 Prompt B：首次计算后由首输出 D2H 回写填入 Slot 1；
   - 第 80 帧切回 Prompt A：直接命中 Slot 0，零 Block0 前向开销；
   - 随后的正式切换中，切 B 与切 A 均 100% 命中，Qwen Conditioning RPC 增量由 Baseline 的 9 次降至仅 1 次。
2. **内存开销**：
   - 预分配 6 个 Pinned CPU Slots（每 rank 49.22 GiB，SP2 98.44 GiB），启动时一次性预分配锁页，运行时零系统调用开销；
   - 本测试驻留 A、B 两个条目，槽位复用正常，Block0 缓存无显存或锁页内存泄漏（Decoder 端的显存增长为既有基线共有表现）。

---

## 数值精度与一致性

- **RGB 逐位一致**：所有 3 对正式请求生成的完整 120 帧视频，Baseline 与 Cache 组的 SHA256 完全对齐（`57a915fc46aad1de29b744264741f8082e039dbb4d93e41db1ad07f877c6a562`）；
- **KV 逐层比对一致**：26 次 50 层全部 K/V 清空显存后对比，`torch.equal` 逐元素 100% 一致；
- **单元测试**：21 项专用测试（包含 A→B→A 往返切换、LRU 槽位复用、参数化容量与异常隔离）全部通过。

---

## 产物归档

- 实验目录：`exports/prompt-switch-cache-h100-20261027-r1/`
- 详细报告：`exports/prompt-switch-cache-h100-20261027-r1/REPORT.md`
- 完整打包证据包：`exports/prompt-switch-cache-h100-20261027-r1/final-evidence.tgz`（包含 752 个回执、日志、指标与抽帧检验）。
