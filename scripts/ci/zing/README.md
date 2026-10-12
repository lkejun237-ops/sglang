# Zing CPU CI

入口为 [Zing CPU CI workflow](../../../.github/workflows/zing-ci.yml)。所有 PR 和手动 dispatch 都执行独立工具测试与只检查、不改写文件的 lint；涉及 Zing、公共运行时、测试或 CI 配置的变更触发全部六个业务套件。最终 `zing-ci` 始终执行，要求路径选择、lint、工具测试和所有选中业务任务成功。

普通 CPU 与 conditioning 的执行步骤分别在 `zing-cpu-suite.yml` 和 `zing-conditioning-suite.yml`，由统一入口调用。新增、删除套件及检查名迁移见 [CI 维护说明](MAINTENANCE.md)。

## 环境与本地复跑

以下命令从仓库根目录执行。工具测试使用 Python 3.11，仅需 `requirements-tooling.txt`，不需要 Torch：

```sh
python -m pip install -r scripts/ci/zing/requirements-tooling.txt
export SGLANG_LIGHTWEIGHT_RUNTIME=1 PYTHONPATH="$PWD/python"
python scripts/ci/zing/run_tooling_tests.py --output-dir /tmp/zing-tooling
python scripts/ci/zing/run_cpu_tests.py --check-inventory --output-dir /tmp/zing-inventory
python scripts/ci/zing/select_changes.py --base origin/main --head HEAD --output /tmp/zing-selection.json
python scripts/ci/zing/check_python.py --base origin/main --head HEAD --output /tmp/zing-lint.json
```

普通业务测试需要独立 Linux Python 3.11 环境、固定 CPU 轮子和 ffmpeg：

```sh
python -m pip install -r scripts/ci/zing/requirements-torch.txt
python -m pip install -r scripts/ci/zing/requirements-cpu.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 ZING_FULL_TILE_TESTS=1
python scripts/ci/zing/run_cpu_tests.py --suite benchmark-contracts --trials 2 --output-dir /tmp/zing-benchmark
```

普通套件还有 `taeh3-contracts`、`adaln-export-contracts`、`dit-pipeline-contracts`、`realtime-contracts`。DiT 正式 CI 按完整文件分四片；分别传入 `--shard-index 0` 至 `3`、`--shard-count 4` 和不同输出目录。执行前检查整个套件的批准状态，不能借分片隐藏未批准文件。

分布式 progressive tiles 测试保留完整 34 项矩阵、真实 BF16/float64 运算和零容差比较；其 CNN 仅在最低空间分辨率保留一个残差块，降低无 BF16 加速 CPU 的执行成本。独立 progressive kernel 测试继续使用各层一个残差块的完整夹具。普通 workflow 记录 `lscpu`，排查超时时需区分慢计算和进程停滞。

`conditioning-contracts` 使用独立 Linux Python 3.12 环境。按 workflow 的安装步骤读取 [conditioning 环境锁](../../../benchmark/zing_h3_causal/ISSUE_256_CPU_ENVIRONMENT.json)，安装 CPU Torch/torchvision 和附加固定依赖，再运行 `--suite conditioning-contracts --trials 2`。保留完整默认测试根和专用子进程 runner，不与普通套件混用环境或模块 stubs。checkout 必须包含清单配方引用的历史 Git 对象；GitHub 使用完整历史 checkout。

## 新增测试和更新清单

1. 在既有测试目录添加实际行为测试，并更新 [cpu_test_manifest.json](cpu_test_manifest.json)。新候选先设 `status: UNRUN`、`approved: false`，填写 suite、sources 和依赖；修改已有测试或执行配方时重新审视批准状态及证据。
2. 在对应固定 Linux CPU 环境试跑，例如：

   ```sh
   python scripts/ci/zing/run_cpu_tests.py --suite benchmark-contracts --triage --file benchmark/zing_h3_causal/test_example.py --trials 2 --output-dir /tmp/zing-candidate
   ```

   路径需已登记。`--file` 可重复；triage 不支持分片参数。普通文件使用独立子进程；conditioning 变更还要复跑完整默认套件。
3. 审查两轮 `receipt.json`、日志和 JUnit，核对源码、环境配方、用例完成情况及跳过原因。通过后更新 `trial_evidence`，明确批准为 `CPU_PASS` 或带精确 CUDA-only skip 清单的 `MIXED`。批准证据由维护者更新，不由 triage 自动写回。
4. 重新执行 inventory 和固定工具测试，再检查 PR 最新 head 的 GitHub run。新增 CI 工具测试时同时维护固定入口 `TOOLING_FILES`；不要用路径选择跳过工具自测。

## 结果与排错

每个任务上传带 run/SHA 标识的产物，包括执行回执、日志和 JUnit。优先查看失败步骤及回执的 `error`、`reason_code`，再定位相应子进程日志。

- 未登记文件、未批准候选、未知 skip、collection skip、非预期 deselection、零用例、超时或缺少必测文件都不能通过。
- 用例日志保留名称与执行状态；单个用例长时间停滞时，在外层进程超时前输出 Python 调用栈。默认在 120 秒时诊断，短超时任务使用其上限的四分之一；此诊断不延长超时、不重试失败、不改变通过条件。
- 退出码为 0 不充分：已收集用例必须完整执行，JSON 与 JUnit 的用例身份、失败和跳过数量必须一致。
- lint 配置变更触发全量检查，包含嵌套配置、Ruff `extend` 依赖及符号链接目标；不支持的 hook 参数明确失败。lint 不自动改文件。
- `--check-inventory` 只核对清单完整性，不执行业务，也不构成 CPU 或 GPU 验收。

GPU smoke、CUDA 算子、多卡通信和性能不在本 workflow 的验收范围。合入 main 的 GPU 必过门禁属于后续实施范围。过程性方案、审查记录和历史验收报告放在 PR/issue 或仓库外归档；依赖锁、测试清单及脚本实际读取的数据文件必须保留。
