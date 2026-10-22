# Lynn Decoder 准备路径的验证工具

本目录保留冻结归档的校验、汇总与 trace 归因工具。设计与数值契约见
[Lynn Decoder 准备路径](../../../docs/docs/sglang-diffusion/zing/specs/LYNN_DECODER_PREPARATION.md)，
Graph 写入与恢复约束见
[capture 恢复边界](../../../docs/docs/sglang-diffusion/zing/specs/CHUNK_GRAPH_CAPTURE_MEMORY.md)。
逐次实验结果、源码身份、失败记录与验收状态统一维护在
[PR #356](https://github.com/seedleap/sglang/pull/356)。

## 当前版本的性能验证

新实验使用英伦维护的 [客户端 benchmark](../CLIENT_BENCHMARK.md)，有限输入用
`run_benchmark.py` / `bench_client.py`，连续 source 用 `bench_streaming_source.py`
及配套报告生成器。分别冻结 480 和 736 的 case manifest、实际宽×高、输入及
checkpoint 身份；两臂固定硬件、采样、并行布局、精度、offload、Graph 和 allocator。
每个服务先完整请求预热，再正式重复，保留失败与未执行数。

测量 TTFF、E2E、全程/首 batch 后 FPS、TPOC 和 ICL 中位数/P95 与样本量；
连续 source 另报告 source-to-output、tail drain、credit wait 和播放停顿。
缺测项明确标为未测。受影响的 Decoder 阶段遵循 [阶段计时说明](../STAGE_TIMING.md)，
记录逐 rank、Host/CUDA 边界与采集开销，不用阶段求和替代 E2E。

本目录的历史汇总器读取旧连续 source 客户端和独立 Decoder replay 的冻结格式，
不能代替当前 benchmark 或证明新 main 组合已经通过 GPU 验证。只重建归档不提交
GPU 作业，也不加载 checkpoint。

## 正确性测试入口

在具备项目依赖的环境执行：

```sh
PYTHONPATH=python SGLANG_LIGHTWEIGHT_RUNTIME=1 python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_zing_lynn_rope_reuse.py \
  python/sglang/multimodal_gen/test/unit/test_zing_lynn_startup_warmup.py \
  python/sglang/multimodal_gen/test/unit/test_zing_lynn_fused_preparation.py \
  python/sglang/multimodal_gen/test/unit/test_zing_h3_denoise_graph.py
```

RoPE/cache 测试检查 aging、特殊 token、紧凑存储及新 stream；startup 测试检查
预编译失败时不能进入 KV 分配或就绪；Graph 测试检查多次执行、历史保持与失败回滚。
融合核与真实启动预编译测试在 SM90（H100/H200）或 SM103（B300）GPU 执行，
不限制 SM 数；其余环境跳过。CPU 测试另覆盖设备准入和不支持设备的预编译跳过。
生产源码 `c3cc5714eb` 已分别在 H100/H200（SM90）和实测 CC10.3/148 SM 设备上
完成准备核、启动预编译及 480/736 完整 Decoder 回放；明确标识的 B300 产品设备
及其他 SM 数配置仍待验证。具体设备、源码、输入边界与
逐轮数据见 [PR #356](https://github.com/seedleap/sglang/pull/356)。CPU 或组件回放通过
不能替代完整服务准确性、CUDA Graph 和单活跃会话生命周期验收。

## 重建冻结归档

先核对归档 SHA256、`READY.json` 文件清单及实际源码，再运行对应汇总器。
输出放在仓库外；依赖仅 Python 标准库。

```sh
python benchmark/zing_h3_causal/lynn_decoder_attention/summarize.py \
  /path/to/FINAL_METRICS.json.gz --output /path/to/output/decoder-summary.json
python benchmark/zing_h3_causal/lynn_decoder_attention/summarize_e2e.py \
  /path/to/WARMUP_E2E_METRICS.json.gz --output /path/to/output/e2e-480.json
python benchmark/zing_h3_causal/lynn_decoder_attention/summarize_736p.py \
  /path/to/KV_MEMORY_E2E_METRICS.json.gz /path/to/WARMUP_E2E_METRICS.json.gz \
  --output /path/to/output/e2e-736.json
```

- `summarize.py`：检查三档、三组 SP2 进程、两 rank、真实 latent、完成标记与
  严格像素相等；延迟取每步较慢 rank，RGB FPS 使用完整输出周期。
- `summarize_e2e.py`：检查完整模型、输入 hash、720 帧完整请求及启动预编译证明。
- `summarize_736p.py`：进一步检查两臂共同 offload、容量、capture 备份与源码清单。
  归档文件名可不同，参数按内容识别；实际对象名称与 SHA 见 PR 的证据索引。

历史格式的 `720p` 标签在本组实验中对应 **1312×736**，`1080p` 对应
**1920×1088**；不能据标签推断其他实验的尺寸。源码以 base 加覆盖文件冻结时，
同时记录完整 base SHA 与 `READY.json` 的覆盖文件 SHA256，不能只标一个候选 commit。

## 独立 trace 归因

```sh
python benchmark/zing_h3_causal/lynn_decoder_attention/profile.py \
  /path/to/trace.json --output /path/to/output/kernel-categories.json
```

工具按 CPU scope 和 CUDA launch correlation 将每个 kernel 唯一归类，
H32/D64 的 cat/cache 与 NCCL 分列，并报告未匹配的 launch 数。
输出是 kernel duration 求和，不是 Host span、墙钟占比或端到端 FPS，且不含 memcpy。
profile 应独立采集，不把受采集影响的样本混入正式性能统计。

冻结 harness、命令、原始 JSON、trace、成功与失败回执留在原产物存储；位置及
历史版本链接见 PR 的证据索引。复现 GPU 实验需要相同私有 checkpoint、依赖及真实
输入，不能用合成 latent 或减少模型层数冒充完整服务验收。
