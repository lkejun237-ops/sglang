# 添加和删除 Zing CI

本文从仓库根目录操作。环境安装、单文件试跑和批准规则见 [README](README.md)。过程报告保存在 PR/issue；这里维护长期操作规则。

## 三层结构

| 文件 | 职责 |
| --- | --- |
| `.github/workflows/zing-ci.yml` | 唯一 PR/手动入口；路径选择、独立工具测试与 lint、套件调用、最终 `zing-ci` 汇总 |
| `.github/workflows/zing-cpu-suite.yml` | `workflow_call` 普通 CPU 执行配方；接收 `suite`、`shard-index`、`shard-count` |
| `.github/workflows/zing-conditioning-suite.yml` | `workflow_call` conditioning 专用 Python 3.12 环境和完整套件 |

主入口保留各套件的逻辑 job ID，汇总通过这些 ID 读取调用结果。执行步骤集中在可复用文件。新增普通套件不必复制依赖安装、ffmpeg、两轮执行和产物上传步骤。

GitHub 不会把调用方的 workflow `env` 传给被调用方，因此两个执行文件都必须显式设置轻量运行时、完整 tile、离线 Hub、线程数与 `PYTHONPATH`。不要给被调用方添加与主入口相同的 concurrency group，否则可能互相取消。参见 [GitHub reusable workflows](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows)。

## 给已有套件添加或删除测试

- 添加：在 `cpu_test_manifest.json` 登记文件、suite、sources 和依赖，初始设为 `UNRUN`、`approved: false`。按 README 在对应固定 Linux CPU 环境完成两轮 triage，核对回执、日志、JUnit 和精确 skip，再更新批准证据。单文件通过不替代 conditioning 完整默认套件验证。
- 删除：确认测试职责确实取消或已有替代，再同步删除源测试及 manifest 条目。文件仍在自动发现范围内时，不能仅删除登记来跳过测试；inventory 会拒绝遗漏。移动文件则同步更新路径、sources 和证据。
- 这类变更通常不需要修改 workflow。套件归属、依赖配方或源码变化后须重新核对批准依据。

同步运行时契约时，也要检查跨套件保留的回归：绕过构造器的夹具须显式补齐新增状态和调用参数；边界与诊断断言应核对当前实现及其专项测试。冻结 native AST 指纹仍保持原始常量，仅归一化有独立行为覆盖、且默认不改变 native 路径的可选扩展，不以重算指纹替代兼容性检查。

## 添加普通 CPU 套件

1. 在 `scripts/ci/zing/common.py` 的 `SUITES` 登记唯一套件 ID，并在 manifest 中登记其测试文件及执行配方。检查 `run_cpu_tests.py` 的配方选择逻辑是否适用；仅添加名称并不会自动生成新的隔离方式或依赖安装流程。
2. 在主入口添加调用 job，job ID 与 suite 必须一致。以下 `example-contracts` 是待登记的新名称，不可直接作为已批准套件运行：

   ```yaml
   example-contracts:
     needs: [zing-changes]
     if: needs.zing-changes.outputs.run_cpu == 'true'
     uses: ./.github/workflows/zing-cpu-suite.yml
     with:
       suite: example-contracts
   ```

3. 将该 job ID 加入最终 `zing-ci.needs`。`select_changes.py` 与 `summarize_results.py` 从 `SUITES` 获取套件集合；保持手动 dispatch 全选、相关路径全选。若新源码目录不在现有选择范围内，扩展路径规则及其行为测试。
4. 有分片需求时参考 DiT：调用方声明 matrix，传入 `shard-index` 与 `shard-count`，关闭 fail-fast。分片覆盖必须是 `0..count-1`，按完整文件划分，先验证整个套件已批准再切片。
5. 更新 README、相关工具测试及 PR 中的覆盖范围；测试夹具 `test/registered/unit/ci/zing_ci_fixtures/__init__.py` 独立声明了预期 `SUITES`，也须同步维护。按下文验证。新增 CI 工具测试文件还需更新 `common.py` 中的固定 `TOOLING_FILES`。

普通可复用配方采用 Python 3.11、`requirements-torch.txt` 与 `requirements-cpu.txt`。需要不同 Python/依赖或特殊隔离协议的套件，应新建仅含 `workflow_call` 的执行 workflow，再由同一主入口调用；同时为新配方补环境与调用关系检查。不要在执行文件额外添加 `pull_request` 触发，避免重复运行。

conditioning 保留独立配方、固定历史对象、完整默认测试根及专用子进程 runner，不接入普通配方或拆成若干替代完整验收的子集。

## 删除整个套件

1. 在 PR 描述说明删除原因、覆盖去向，以及哪些测试仍然保留。
2. 将保留的测试迁移到合适套件并重新验证，或删除已废弃测试；同步调整 manifest。检查发现规则，不可用删清单掩盖仍然存在的候选文件。
3. 同时删除 `SUITES` 项、主入口调用 job 和 `zing-ci.needs` 项；清理仅供该套件使用的执行配方、依赖与测试假设。共用 CPU 执行 workflow 仍有调用者时保留它。
4. 删除专用执行 workflow 前搜索所有调用者，确认不再被引用；同步更新 README 和工具回归测试。
5. 核对仓库 rulesets/branch protection 的 required checks。如果配置仍要求被删除的检查名，会一直等待。由有权限的维护者同步迁移配置，不能通过让汇总忽略失败来规避门禁。

## 检查名与产物

可复用 workflow 的 GitHub 检查显示名包含调用方和内部 job，例如 `decoder-contracts / cpu-suite`；DiT 还包含 matrix 值。逻辑 ID 保留不等于旧的 required check 名称自动保持。新增、删除或重命名后以实际 PR Checks/API 返回的名称为准，核对保护规则；统一最终检查 `zing-ci` 保持稳定。

普通套件的产物名包含 suite、shard、run ID、attempt 与 SHA；输出目录在各自独立 runner 内固定为 `artifacts/cpu-suite`。conditioning 保留单独产物。不要删除 `if: always()` 上传步骤或用 `continue-on-error` 掩盖失败。

## 提交前和 GitHub 验证

在 README 指定的工具环境执行：

```sh
python scripts/ci/zing/run_tooling_tests.py --output-dir /tmp/zing-tooling
python scripts/ci/zing/run_cpu_tests.py --check-inventory --output-dir /tmp/zing-inventory
python scripts/ci/zing/check_python.py --base origin/main --head HEAD --output /tmp/zing-lint
python scripts/lint/check_workflow_job_names.py
```

`test_zing_ci_summary.py` 检查三个 workflow 的显式 CPU 环境、独立工具/lint、套件调用与汇总依赖以及分片覆盖；原有行为测试继续验证失败、取消、遗漏和不应跳过的任务无法获得成功结果。新增专用配方时扩展这些检查，不固定总 job 数。

推送后核对最新 PR head 的实际 run：相关变更必须选中全部套件；各分片、conditioning、工具、lint 及最终汇总都应成功。下载产物确认文件覆盖、两轮完成、JUnit 一致和精确允许的 skip，并核对新显示名。失败应修复后重跑；静态 YAML 解析和 inventory 通过不能替代实际执行。重跑时优先重跑全部 jobs，确保选择产物与 summary 使用相同 run attempt。

上述只验收 CPU CI。合入 main 所需 GPU 门禁仍是后续工作，本次拆分不增加 GPU smoke。
