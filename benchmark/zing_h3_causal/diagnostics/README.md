# Zing 专项诊断入口

默认完整回归继续使用上一级的 `stall_job.py`、`run_isolated_stall_suite.py`、
`run_stall_regression.py`、`run_stall_lifecycle.py` 和 `benchmark_dit_replay.py`。
播放欠载、GC 观察、证据读回及归档工具也继续由核心回归共享。

本目录保存需要显式选择的专项实验，不自动加入默认性能轮。短前缀、算子实验、
纯组件或布局 pilot 的成功，不能替代完整实例的数值、性能及生命周期验收。

| 目录 | 执行入口 | 读回／辅助工具 |
| --- | --- | --- |
| `encoder/` | `run_encoder_gc_comparison.py`：同卡 Encoder GC off/on | 复用上一级 `analyze_stall_gc.py` |
| `decoder/` | `benchmark_decoder_parallel.py`、`benchmark_decoder_resident.py`、`run_decoder_layout_pilot.py` | `analyze_decoder_parallel.py` |
| `dit_replay/` | `run_dit_context_probe.py`、`run_native_rng_probe.py`、`probe_compiled_norm_history.py` | `analyze_dit_context_probe.py`、`analyze_native_rng_probe.py`、`inspect_dit_norm_boundary.py`、`dit_rng_probe.py` |

从仓库根目录可直接运行脚本，例如：

```bash
python3 benchmark/zing_h3_causal/diagnostics/encoder/run_encoder_gc_comparison.py --help
python3 benchmark/zing_h3_causal/diagnostics/decoder/benchmark_decoder_resident.py --help
```

正式执行仍须按 [完整复现说明](../STALL_REGRESSION.md) 冻结源码、资产、配置及
controls，先完成 CPU 预检，再使用已授权的隔离 GPU 分配。`stall_job.py` 的既有
专项配置继续选择这些新路径；不新增旧文件名转发脚本。

目录调整不修改已经冻结或正在执行的实验 root。旧命令、归档及测试回执继续绑定
原提交和原路径；新一轮必须从已推送的新提交重新准备，不能修改旧 controls。
本次整理本身不提供新的 GPU 性能或精度结论。

## 旧版本适配的退役条件

`run_main_stall_suite.py` 和 `run_main_dit_replay.py` 暂留上一级，仅服务于本轮冻结
main 的 baseline 复现，不作为未来版本的长期兼容层。`run_dit_resident_collection.py`
暂留作独立采集入口；其中 nonexact 数据只能用于诊断，不能改标为数值通过。

完成以下事项后，在后续提交中退役旧版本适配，并复核独立采集入口是否仍有调用者：

1. 当前候选和匹配主线的两档完整矩阵、逐 rank D1、耐久及生命周期取得终态结果；
   失败和未执行项单列，尚未完成的验收不算结束。
2. 原始数据、完整源码 SHA、命令和配置已归档，并由独立读取核对完整性。
3. 最终报告保留精确历史提交的复现链接；未来 baseline/candidate 使用统一入口，
   不再依赖旧 main 的反序列化／GC shim。

现在不删除这三个入口，也不为目录整理扩大旧版本兼容范围。
