# grouped + chunk reuse 组件验收（#211）

范围仅为默认分支接线与连续失效区间调度，不改 kernel、位置数学或失效规则。
使用合成小张量，不构成 source 视频 + 六参考图业务基线、音视频或画质验收。

## 可重复 CPU 命令

轻量 cache oracle 只需 Python >=3.10、PyTorch 和 pytest：

```sh
SGLANG_LIGHTWEIGHT_RUNTIME=1 PYTHONPATH=python OMP_NUM_THREADS=1 \
  python -m pytest -q benchmark/zing_h3_causal/test_chunk_rope_reuse.py
```

真实两层小模型接线测试需要项目模型依赖。先导入模型以避免现有 unit conftest
的 server_args 循环导入；Mac 用仓库自带平台兼容层加载 CUDA-only 依赖定义，
CPU arithmetic、模型 forward、cache 和 eager fallback 均执行真实实现：

```sh
SGLANG_LIGHTWEIGHT_RUNTIME=1 PYTHONPATH=python OMP_NUM_THREADS=1 python - <<'PY'
import sys
if sys.platform == "darwin":
    from sglang._platform_stubs import install_platform_stubs
    install_platform_stubs()
import sglang.multimodal_gen.runtime.models.dits.zing_h3
import pytest
raise SystemExit(pytest.main([
    "-q", "python/sglang/multimodal_gen/test/unit/test_zing_grouped_chunk_reuse.py",
    "python/sglang/multimodal_gen/test/unit/test_zing_history_rope.py",
    "-k", "not cuda and not history_matches_eager_with_offset_and_sentinels and not grouped_dynamic_chunks",
]))
PY
```

第二条命令可能保留以 skipif 标记的 CUDA-only 测试；跳过项不计为通过。
新模型测试默认移除三个环境变量，观察真实 forward 调用的 batch 片段数，
并逐位比较同一模型、同一 cache 的 eager 输出。没有以默认值断言替代接线验证。

## 最小 CUDA 命令

在真实 CUDA 环境和项目依赖中运行；无 GPU 时第一条会失败，不会 skip：

```sh
SGLANG_LIGHTWEIGHT_RUNTIME=1 PYTHONPATH=python OMP_NUM_THREADS=1 \
  python benchmark/zing_h3_causal/validate_grouped_chunk_reuse.py
ZING_TEST_DEVICE=cuda SGLANG_LIGHTWEIGHT_RUNTIME=1 PYTHONPATH=python \
  python -m pytest -q benchmark/zing_h3_causal/test_chunk_rope_reuse.py
```

另在同一 CUDA 环境执行现有回退回归（同样先导入模型）：

```sh
SGLANG_LIGHTWEIGHT_RUNTIME=1 PYTHONPATH=python python - <<'PYTEST'
import sglang.multimodal_gen.runtime.models.dits.zing_h3
import pytest
raise SystemExit(pytest.main([
    "-q", "python/sglang/multimodal_gen/test/unit/test_zing_history_rope.py",
    "-k", "rejected_alias_dtype_and_layout or grouped_alias_and_dtype_fallback or contiguous_batch_views_raw_k or history_plan_is_session_owned",
]))
PYTEST
```

组件使用 BF16 H14/D128，四片段 `[17,33,65,129]` 与 `[390,390,390,390]`。
验证全失效、全命中及中间命中两侧失效的 K/V 逐位 oracle 和实际 grouped 计数。
每种路径预热 10 次、测量 20 次，报告 cache rebuild 的 host wall 和 CUDA-event
中位数。计时前绑定坐标并预热系数，每次显式清除旋转 publication 强制四片段
失效；不包括位置绑定成本。CUDA-event 包含 enqueue gaps；这里不声称纯 kernel
时间或端到端收益。

## 实际结果

基线 `04369a33f90b3d04673df426c0fe276cb58dc7fa`，2026-09-28。

- CPU cache oracle：17 passed（Python 3.12.12 / PyTorch 2.14.0）。覆盖 source→target
  合并、窗口淘汰/搬移、prefix 替换、store growth、raw K 原地写、clear、全命中、
  非连续失效、首批/后批 grouped 写入失败和 current-tail 失败。
- CPU 真实模型接线与已有局部回归：38 passed，12 CUDA-only skipped，143 deselected
  （Python 3.11.13 / PyTorch 2.13.0）。12 个新增组合用真实 batch dispatcher 和 eager
  fallback；另 26 项为既有 plan、cache 生命周期、alias/dtype 等组件回归。
- 修复前新回归因 grouped 没被调用而失败；修复后通过。只改缓存调度，不改算子。
- CUDA 首轮 `dlcg2n30q5wzmmn7` 在创建 `/workspace/issue211` 时因父目录缺失而
  失败，未执行测试。修正启动脚本后以相同单卡配置提交 `dlc1c9bzeyua911y`，
  平台终态 Succeeded，worker 退出码 0。BF16 组件 PASS；CUDA cache oracle 17 passed；
  CUDA 环境中的选定 alias/dtype/layout/contiguous/plan 回归 13 passed、168 deselected，
  无跳过。后者包括其 CPU 参数分支，不把全部 13 项称作 GPU kernel 测试。

| 四个片段的行数 | 逐片段 wall 中位数 | grouped wall 中位数 | 逐片段 CUDA-event | grouped CUDA-event |
| --- | --- | --- | --- | --- |
| 17 / 33 / 65 / 129 | 0.4303 ms | 0.3414 ms | 0.4141 ms | 0.3260 ms |
| 390 / 390 / 390 / 390 | 0.4068 ms | 0.3431 ms | 0.3885 ms | 0.3275 ms |

每组 20 次测量，历史旋转实际路由计数为逐片段 fused 80 次、grouped 20 次，
即每次 4→1。未采集端到端吞吐、长视频、SP 矩阵或画质，不据此推断业务收益。

运行时 PyTorch `2.13.0+cu130` / CUDA `13.0`，torch 报告 capability `[10,3]`。
平台设备名为 `NVIDIA L20D`；nvidia-smi 显示 275040 MiB 和 capability `8.9`，
与 torch 不一致，按原始结果分别保留，不按设备名称推断硬件规格。
源码哈希、镜像 digest、作业 ID 与完整组件测量见
[grouped_chunk_reuse_validation.json](grouped_chunk_reuse_validation.json)。


## 单一 BF16 生命周期补测

```sh
SGLANG_LIGHTWEIGHT_RUNTIME=1 PYTHONPATH=python OMP_NUM_THREADS=1 \
  python benchmark/zing_h3_causal/validate_grouped_chunk_lifecycle.py
```

复用 `test_cache_transactions_eviction_and_failure`，只选 CUDA、grouped 开启、
plan 关闭、window3/sink1/block_relative 一个组合。H14/D128 BF16，每次 source
7 行、target 9 行，共 28 次事务；prefix 从 5 行替换为 6 行。既有 oracle 逐步
比较 K/V 和缓存内容，包含异常后不发布与 clear。包装器仅记录 clear 前统计，
并断言真实 grouped 调用、合批、无 grouped fallback、多次 store allocation
和 compaction 均实际发生，不替换计算算子。无 CUDA 时失败而非跳过。

实际结果：2026-09-28 单卡作业 `dlc6d9q16zbwc210` Succeeded，worker exit 0。
28 次事务 K/V 和缓存内容逐位一致，真实 grouped 56 次、处理 92 个片段、
无 grouped fallback；两侧 cache 各发生 25 次 compaction、5 次 store allocation
和 5 次 rotated allocation。异常后不发布及 clear 的既有断言同样通过。
这项补测不替代业务输入、视频质量或多卡验收。原始计数见上述 JSON 的
`cuda_lifecycle` 字段。
