# 2026-11-13 共享 TTFF / Qwen prefix 分支 CPU 回归

分支 `codex/shared-test-ttff-prefix` 从 main `f27559be499487b9374cc325289e49b81261e4e1`
合入 #662 `b53707917d973807d58833a90583d956d79f66fa`（含 #687 和独立 Encoder 接线）。
本记录是功能整合验证，不是原部署仅差 #662 的性能 A/B。main、PR head 与在线服务未切换。
设计和开关见 [TTFF 设计](../../docs/docs/sglang-diffusion/zing/plans/serving-ttff-breakdown.md)。

## 受测环境与范围

现有 H100 主机的隔离 CPU 容器，`--net none --cpus 2 --memory 8g`，
`NVIDIA_VISIBLE_DEVICES=void`；未分配 GPU。镜像
`sha256:e132494ca7a1fb21b0655aeca65a13b1e3bffa2fb8696330678f59348b610ea1`，Python 3.12。
源码来自已推送提交的完整 git archive，下载后核验 SHA256，源码只读挂载。
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、
`OPENBLAS_NUM_THREADS=1`。条件模块使用仓库 CPU import harness，不能证明实际 GPU 模型执行。

| 套件 | 受测 SHA | 结果 |
|---|---|---|
| 三角色与 conditioning | `17c358c189b169a4d72fa7e03ee02fd1ab1487f5` | 491 passed、49 skipped、1 deselected；3 subtests passed；97.95s |
| Room 生命周期/计时 | `49363d35cf4be197c5366569cb8a41cc39eab2a1` | 153 passed、2 deselected；190.81s |
| 部署与采集工具 | `17c358c189b169a4d72fa7e03ee02fd1ab1487f5` | 169 passed；5.48s |

Room 的两项 deselected 是组件测试，已在模型套件执行；两个套件包含少量重复用例，不相加作为唯一 case 数。
最终模型套件的 1 项 deselected 是 `test_local_mode_preserves_native_token_range`：
它执行 `git show b1899c0...:.../codecs.py`，完整源码归档没有 Git 历史，首次运行失败；
本轮未验证该历史对照，不改成虚假通过。49 项跳过的实际原因保留在 JUnit XML。

## 可复现入口

模型套件使用 `benchmark/zing_h3_causal/run_conditioning_cpu_tests.py`，测试目标：

- `python/sglang/multimodal_gen/test/unit/conditioning/`
- 同一 unit 目录的 `test_zing_encoder_dit_link.py`、`test_zing_h3_queued_dit.py`、`test_zing_h3_remote_decoder.py`
- 同一 unit/realtime 目录的 `test_zing_source_encoder_api.py`、`test_decoder_output.py`、
  `test_decoder_output_api.py`、`test_frame_latency_breakdown.py`、`test_inference_timing.py`
- `test/registered/unit/multimodal_gen/test_conditioning_rpc_core.py`

追加 `-k 'not test_local_mode_preserves_native_token_range' --confcutdir=python/sglang/multimodal_gen/test/unit -q --tb=short`。
Room 使用 `SGLANG_LIGHTWEIGHT_RUNTIME=1 PYTHONPATH=/src/python` 下的 `python -m pytest`，
目标为 realtime 下的 `test_inference_timing.py`、`test_whitebox_render_contract.py`、
`test_whitebox_render_lifecycle.py`、`test_whitebox_sink_control.py`，
`--confcutdir=python/sglang/multimodal_gen/test/unit/realtime`，用 `-k` 排除
`dynamic_timing_real_encoder_dit_control_chain_and_decoder_boundary` 和
`decoder_timing_rejects_stale_owner_without_mutation` 两个已在模型套件运行的组件测试。

部署工具使用 `/opt/sglang/bin/python3 -m pytest deploy/whitebox/test_conditioning_collector.py deploy/sglang/test_orchestrate.py -q --tb=short`。
其中已有测试假定控制器 `sys.executable` 与固定的目标 `/opt/sglang/bin/python` 路径不同；
最初使用该相同路径启动时 1 项假设失败，改用同一解释器的 `python3` 入口后通过，未修改产品代码。
Windows 试跑还遇到 POSIX 绝对路径与本机缺依赖/GBK 问题，以 Linux 结果为本轮证据。

## 合并修复与自审

保留 main 的逐帧计时、source early-release、渐进 Decoder 和接口 capability；
#662 的 detailed_timing 与 return_stage_timing 各自传递。
沿 Adapter → Encoder → DiT → Decoder → Room 回执检查字段和生命周期。
/metrics 复用现有两个 registry，不增加 exporter；HTTP 测试确认鉴权、全局及 prefix
指标均存在，且 metric family 不重复。

首次模型回归 489 passed、2 failed、49 skipped：除历史 fixture 外，另一个失败是合并
后 deferred-KV 校验误读 source command。修复为使用 prompt 请求，并在异步 early-release
时冻结该请求的 deferred 授权；新增延迟回执回归。复测覆盖新组合和四种计时开关组合。
最终静态语法、Ruff E9/F63/F7/F82（py312）与 git diff --check 通过。
没有新增调度抽象；在已检查的合并和 CPU 协议范围内未发现剩余阻塞问题。

## 证据与未验证

原始 log/JUnit、源码 archive 校验和及云调用回执保存在本次部署 artifacts 的
`shared-cpu-evidence/` 与 `shared-cpu-identity.json`；模型首次提交命令调用
`t-sh072yt1f0ltgxs`，最终复测 `t-sh072ytiinhwef4`。
最终代码之后的提交只更新本文及设计/部署说明，不更改受测运行时。

未运行 480/736 GPU、NCCL、多 rank、数值一致性、显存与实际延迟 A/B；未切在线服务，
未验证 Grafana 入库。共享分支包含新 main 改动，旧基线性能结果不能直接归因到它。
