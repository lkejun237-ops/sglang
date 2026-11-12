# SGLang 独立角色池编排说明（`deploy/sglang/orchestrate.py`）

## 1. 职责与启动链边界

`deploy/sglang/orchestrate.py` 负责已授权目标环境的三角色（Encoder / DiT / Decoder）只读探测、离线计划、创建或复用、隔离建组自测、Coordinator 角色池原地挂载、清理回收，以及持久化部署历史与受控回滚。

- **独立三角色唯一启动链**：`orchestrate.py` → `launch.py` → `zing_h3_profile_launcher.py`（`--mode independent`）→ `zing_h3_role_launcher.py` → 原生角色 worker。
- **单机集成式启动链**：`deploy/whitebox/release.py` → `deploy/whitebox/model_service.py` → `zing_h3_profile_launcher.py`（`--mode integrated`）。两条路径互不反调，硬件策略与进程监管统一由 `zing_h3_profile_launcher.py` 维护。
- **操作边界**：支持测试与生产环境，按明确授权的目标、模型池和 owned 资源执行；保留源码/实例身份、占用、活跃会话及 create/reuse 校验。生产环境不因名称、IP 或端口被拒绝；这不代表已完成生产实机验收。支持 `H100 Encoder/Decoder + H200 DiT` 与 `B300 三角色` 拓扑。
- 详细架构设计见 [`docs/docs/sglang-diffusion/zing/plans/sglang-role-pool-orchestration.md`](../../docs/docs/sglang-diffusion/zing/plans/sglang-role-pool-orchestration.md)；实机运行记录与基准指标见 [`benchmark/whitebox/sglang_shanghai_copy_20261110.md`](../../benchmark/whitebox/sglang_shanghai_copy_20261110.md)。

## 2. 输入配置与产物

### 输入配置（`spec.json`）

- `target`：环境标识（如 `test-shanghai-copy`、`prod-shanghai` 或 `prod-thailand`）、`region`、`workspace_id`、`quota_id`、`coordinator_url`、`pool_id`、`room_url`、`wtp_url`、`oss_prefix`，以及可选的 `node_id`、`h100_ip`、`topology_profile`、`history_dir`。
- `roles` / `existing_services`：
  - Encoder 与 Decoder 支持按角色配置 `mode: "create" | "reuse"`（或通过顶层 `existing_services` 声明复用），并支持 `service_count >= 1`（单进程多 service、每 service 单活跃会话，容量以实测为准）。
  - `profile` 支持 `auto`、`h100`、`h200`、`b300`；用户侧无 `role_launcher` 开关，统一走 profile launcher。
  - 通用编排器不强制注入 `NCCL_MIN_NCHANNELS=2` / `NCCL_MIN_CTAS=2`；跨机 SM90 H100+H200 拓扑可通过 `target.topology_profile = "cross_host_sm90_h100_h200"` 启用版本化 Workaround。
  - `secret_files` 仅保存容器内平台挂载路径引用（`ROLE_POOL_SIGNING_KEY`、`ROLE_POOL_REGISTRATION_TOKEN`），严禁写入密钥明文。

### 输出产物

- `probe.json` / `spec.json`：并发采集的节点空闲 GPU、端口、镜像 digest、模型路径与 PAI 作业快照（满足条件时 `blockers` 为 `[]`），以及从最小环境输入生成的完整冻结 `spec.json`。
- `state.json`：包含独立 `deployment_id`、`run_id`、源码归档 SHA256 与逐角色资源状态。
- `deployment.json` 与 `DEPLOYMENT.md`：自动汇总逐角色真实源码 SHA、镜像 digest、资产 SHA256、`create`/`reuse` 模式、资源/process/service/boot/GPU/端口身份、`service_count`、生效 profile/argv 及各阶段验证/清理状态。
- `history/<deployment_id>.json` 与 `history/index.json`：持久化部署历史记录与完整 `rollback_bundle`（不含密钥明文）。

## 3. 命令说明

所有变更操作默认仅展示离线计划；带 `--execute` 执行变更时默认打印计划并要求确认，显式传 `--yes` 可跳过交互确认（仍强制核验目标身份、资源占用与计划完整性）。

```bash
# 1. 只读探测（支持最小环境输入或完整 spec，并发查询节点与 PAI 列表）
python3 deploy/sglang/orchestrate.py probe --aliyun-profile <your-aliyun-profile> \
  --spec /private/spec.json --output /private/probe

# 2. 生成部署计划并执行（自动生成 deployment.json、DEPLOYMENT.md 与历史记录）
python3 deploy/sglang/orchestrate.py up --aliyun-profile <your-aliyun-profile> \
  --spec /private/probe/spec.json \
  --commit "$(git rev-parse HEAD)" --output /private/run --history-dir /private/history --execute --yes

# 3. 并发核验三角色运行回执与 /health
python3 deploy/sglang/orchestrate.py inspect --aliyun-profile <your-aliyun-profile> \
  --state /private/run/state.json

# 4. 隔离自测（并发 reserve/build_group → quiesce_group → destroy_group）
python3 deploy/sglang/orchestrate.py selftest --aliyun-profile <your-aliyun-profile> \
  --state /private/run/state.json --execute --yes

# 5. 原地热挂载到 Coordinator 角色池（默认 --in-place，支持部分成功幂等重试）
python3 deploy/sglang/orchestrate.py attach --aliyun-profile <your-aliyun-profile> \
  --state /private/run/state.json --execute --yes

# 6. 回收本轮新建资源（reuse 角色仅释放本轮租约，不 drain/retire/重启共享服务）
python3 deploy/sglang/orchestrate.py down --aliyun-profile <your-aliyun-profile> \
  --state /private/run/state.json --execute --yes

# 7. 部署历史查询、详情、对比与受控回滚
python3 deploy/sglang/orchestrate.py history --history-dir /private/history
python3 deploy/sglang/orchestrate.py show --history-dir /private/history --deployment-id <id>
python3 deploy/sglang/orchestrate.py diff --history-dir /private/history --from-id <id1> --to-id <id2>
python3 deploy/sglang/orchestrate.py rollback --aliyun-profile <your-aliyun-profile> \
  --history-dir /private/history \
  --deployment-id <target_id> --current-state /private/run/state.json \
  --output /private/rollback-run --execute --yes
```

## 4. 失败恢复与重入保证

- **未知创建保护**：`up` 提交前先落盘 `submitting` 状态与 `InvokeId`；若云 API 超时，状态保留以供人工对账，禁止自动重复创建。
- **原地 `attach` 幂等重入**：部分角色完成 `/managed/attach` 后若后续角色超时，已挂载角色立即持久化 `boot_epoch` 与 `service_id`；重试时通过签名 `/managed/observe` 校验空闲状态并幂等重放，最后在 `verify_pool` 对账三角色身份。
- **`down` 删后重试恢复**：容器或作业停止后写入 `stopped-unretired` 状态；若 Coordinator `confirm-stopped` 或 `retire-process` 瞬时失败，再次执行 `down` 会跳过已删除的 Pod/Job 预检并直接完成 Coordinator 退役闭环。
- **受控回滚恢复**：`rollback` 校验完整 `rollback_bundle`（拒绝仅含摘要或回执的记录），保护 `reuse` 共享服务，不要求重获旧 GPU UUID / Pod UID / Job ID，也不删除共享缓存；中断后重跑可从 `rollback_stopped_current` 检查点安全续跑。

## 5. 验收边界

- `inspect` 与 `/health` 仅证明进程就绪与配置身份匹配，不代表通信组可建或业务出帧。
- `selftest` 仅证明隔离通信组 `build → quiesce → destroy` 生命周期正常，不代表模型生成质量或帧率。
- `attach` 仅证明角色已注册到 Coordinator 且路由容量可见，不代表 Room/WTP 真实浏览器 WebRTC 呈现通过；完整端到端验收需结合 4-Case 真实模型回归与浏览器出帧证据。
