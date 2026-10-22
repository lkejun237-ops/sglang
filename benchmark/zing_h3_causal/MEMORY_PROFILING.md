# 运行时显存采集与重放

使用场景：请求可以完成，但想知道加载后的常驻、预热/capture 尖峰和正式轮新增占用分别来自哪里。
采集器与重放算法来自 issue334 的 B300/H200 实验，保留相同的事件语义；不做事前容量模拟。
综合入口见 [README](README.md#综合-benchmark-入口)。

## 采集

在已部署的模型、条件服务及 Source Encoder 参数上，显式使用 benchmark worker launcher。
它只包住现有服务入口，在 CUDA device 初始化后启动 allocator observer；不注册模型 forward hook。
每个 GPU worker 只记录自己的当前 device。SP worker 通过 spawn 继承同一输出目录和角色。
模型、checkpoint、输入、Graph 和 KV pool 配置仍使用原服务参数，不由本工具改写。

```bash
python3 benchmark/zing_h3_causal/benchmark.py capture run \
  --role server --output /path/to/memory-run -- --model-path /path/to/checkpoint [原服务参数]
```

独立条件服务和 Source Encoder 分别使用 `--role qwen|encoder`，共享同一配置的输出目录。
不要在同一个输出目录中混用 eager/Graph 或重复启动的服务；每个配置单独启动常驻进程。
观察目录 `memory/` 保存进程身份、PyTorch/硬件版本、allocated/reserved 采样和检查点回执。

在服务已就绪、第一次请求之前记录 `ready`，再分别执行预热、capture 和正式诊断请求。
检查点会同步当前 GPU、保存快照并重置历史/区间 peak；这些同步不属于正式延迟测量。
例中 `4` 指实际初始化的两个 SP worker、Qwen 和 Source Encoder，其他部署按实际进程数填写。
身份数或回执不齐即失败，不拿部分 worker 当完整结果。

```bash
python3 benchmark/zing_h3_causal/benchmark.py capture checkpoint \
  --output /path/to/memory-run --label ready --workers 4
# 执行一次独立诊断请求；每个请求后保留独立快照
python3 benchmark/zing_h3_causal/benchmark.py capture checkpoint \
  --output /path/to/memory-run --label after-warmup --workers 4
```

整卡 NVML 另行采样，按物理 UUID/index 与各 worker 对应，不将 Qwen/Encoder 各自的峰值相加。
采样时长显式给定，输出文件不得已存在；采样只记录整卡读数，不执行模型。

```bash
python3 benchmark/zing_h3_causal/benchmark.py capture sample \
  --output /path/to/memory-run/gpu-memory.jsonl --duration 600 --interval 0.5
```

独立测速前，等待所有 worker 确认关闭 allocator history，并结束上述 NVML 采样：

```bash
python3 benchmark/zing_h3_causal/benchmark.py capture checkpoint \
  --output /path/to/memory-run --label disable-for-timing --operation disable --workers 4
```

关闭后 observer 线程退出，peak reset 包装直接调用原实现。随后使用原客户端的预热与正式请求，
TTFF/E2E/FPS 不从显存诊断轮取数。开启标准阶段记录与否应写入该次运行配置。

## 重放可信快照

`analyze_memory.py` 只在明确指定 `--trust-local-snapshots` 时加载 pickle；只处理自己的可信实验产物。
网页导入只接受 JSON，不反序列化上传的快照。分析输入是一个 JSON 数组，每项包含：

- `snapshot`：相对清单的 `.pickle` 或 `.pickle.gz` 路径。
- `device`、`pid`、`role`、`case`：真实进程身份和轮次标签；不根据文件名推断卡型或场景。
- `start_us`：同一设备 trace 时钟中的请求开始时间，单位微秒。
- `history_may_be_truncated`：由本轮历史容量和采集范围明确判定，不能省略或隐式当完整。
- 可选 `stage_timeline`：`stage/start/end`，start/end 是相对请求开始的 Host 秒数。

`memory/process-*.json` 保留相邻采样的 `clock_monotonic` 与 `clock_unix_ns`，用于核对
客户端/阶段与快照 trace 的时钟域。已有实验使用采样中的 offset 中位数映射请求边界；
不能将不同主机时钟直接相减，也不能把 CUDA event 耗时伪造成此轴上的 GPU 起止点。

```bash
python3 benchmark/zing_h3_causal/benchmark.py memory \
  --manifest /path/to/snapshot-requests.json --trust-local-snapshots \
  --output /path/to/replayed-memory
python3 benchmark/zing_h3_causal/benchmark.py report \
  --memory-run '配置A=/path/to/replayed-memory' \
  --include-results '旧结果=/path/to/report.json' --output /path/to/report.html
```

重放先从最终快照反推区间起点，再正向核对地址、请求字节和释放状态。相同地址释放后再次分配
算新分配；`free_requested` 和 `free_completed` 分开，避免漏掉等待 GPU 释放的占用。
历史截断明确显示为部分区间。曲线按时间桶保留极值，调用栈按新增存活峰值的实际地址集合汇总。

`--memory-run` 读取已分析的 `MEMORY_REPLAY.json`，不加载 pickle，兼容本次发布的两份实验目录。
原始采集目录的 `memory/`、`gpu-memory.jsonl` 可以复制或链接到重放输出目录，报告会汇总检查点。
未单独确认的 NVML 请求前基线显示为未测，峰值只对应实际采样窗口。
已有 `COUNTER_OVERVIEW.json` 时保留其完整采样范围和基线，不重复计算不同口径。

可选 `INDEPENDENT_TIMING.json` 保存独立请求的 `cell/case/count/partial/requests`；
每个 request 保留 `client` 的 TTFF/E2E/post-TTFF FPS 和 `stage_cuda_ms`。
报告从这些逐请求值重新计算 median/min/max，显示真实有效次数；阶段饼图排除 Conditioning RPC
父等待，只归一化 Qwen、Source/Reference Encoder、DiT、Decoder 的 CUDA 阶段耗时，不能当作 E2E 分摊。
历史导出器的 `results/H200_TIMING.json` 是同一格式，保留读取兼容。

## 已有实测与本轮验证边界

[团队分享报告](https://review-cdn.loopit.com.cn/world-model/results/issue334-pr373-profile/20261021T110447Z/index.html)
需要办公网或公司 VPN。它对应冻结 PR373 `67c2b323df3954e603ee3ba33650b8c8695cc882`，
全50层、BF16、SP2、window6、KV pool、cross_chunk，eager/Graph 分列，5s 与两组30s。

| 测试目标 / 实际名称 | DiT NVML eager峰值 / GiB | Graph峰值 / GiB | 快照 / 闭合重放 |
|---|---:|---:|---:|
| B300目标 / NVIDIA L20D | 80.838 | 104.109 | 80 / 72 |
| H200目标 / NVIDIA L20X | 80.097 | 103.405 | 80 / 72 |

两配置各 rank 的 capture 前 K/V clone 新分配为18.862915 GiB、100个分配，正式 replay 不含这组备份。
H200目标另有18次关闭显存采集的计时，Graph两组30s E2E中位数较eager增加2.14%/2.73%（每组3次）。
B300采样轮耗时含采集开销，不作为独立跨设备性能比较。L20X 的 CUDA/torch capability9.0、
NVML capability8.9 原始差异及目标设备 AdaLN 表重建均保留在报告。

本轮是工具整理，没有补跑 GPU。旧冻结实验的数值不作为新 launcher 或最新 main 的 GPU 验收；
CPU CI 验证地址复用、延迟释放、重放闭合、统计重算及结果导入，真实设备行为仍以各次冻结记录为准。
权重、完整快照、视频、云提交器和运行回执留在私有存储/实验目录，不进入 Git。
