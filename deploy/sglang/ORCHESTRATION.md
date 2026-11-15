# SGLang 独立角色池编排说明（`deploy/sglang/orchestrate.py`）

## 1. 职责与启动链边界

`deploy/sglang/orchestrate.py` 负责已授权目标环境的三角色（Encoder / DiT / Decoder）只读探测、离线计划、创建或复用、隔离建组自测、Coordinator 角色池原地挂载、清理回收，以及持久化部署历史与受控回滚。

- **独立三角色唯一启动链**：`orchestrate.py` → `launch.py` → `zing_h3_profile_launcher.py`（`--mode independent`）→ `zing_h3_role_launcher.py` → 原生角色 worker。
- **单机集成式启动链**：`deploy/whitebox/release.py` → `deploy/whitebox/model_service.py` → `zing_h3_profile_launcher.py`（`--mode integrated`）。两条路径互不反调，硬件策略与进程监管统一由 `zing_h3_profile_launcher.py` 维护。
- **操作边界**：支持测试与生产环境，按明确授权的目标、模型池和 owned 资源执行；保留源码/实例身份、占用、活跃会话及 create/reuse 校验。生产环境不因名称、IP 或端口被拒绝；这不代表已完成生产实机验收。支持 **`H200 全三角色`（`topology: "h200"`：新建 `1 Encoder + 1 DiT + 1 Decoder`）**、**`复用已有 Encoder/Decoder + 新建 H200 DiT`**、`H100 Encoder/Decoder + H200 DiT` 与 `B300 三角色` 拓扑。
- 详细架构设计见 [`docs/docs/sglang-diffusion/zing/plans/sglang-role-pool-orchestration.md`](../../docs/docs/sglang-diffusion/zing/plans/sglang-role-pool-orchestration.md)；实机运行记录与基准指标见 [`benchmark/whitebox/sglang_h200_deploy_verify_20261114.md`](../../benchmark/whitebox/sglang_h200_deploy_verify_20261114.md) 与 [`benchmark/whitebox/sglang_shanghai_copy_20261110.md`](../../benchmark/whitebox/sglang_shanghai_copy_20261110.md)。

## 2. 输入配置与产物

### 输入配置（`spec.json`）

- `target`：环境标识（如 `test-shanghai-copy`、`prod-shanghai` 或 `prod-thailand`）、`region`、`workspace_id`、`quota_id`、`coordinator_url`、`pool_id`、`room_url`、`wtp_url`、`oss_prefix`，以及可选的 `node_id`、`h100_ip`、`topology_profile`、`history_dir`、`sync_history_oss`。
- `roles` / `existing_services`：
  - 支持最小化拓扑 `topology: "h200"` 自动展开全 H200 三角色（默认单角色 `ResourceConfig` 为 `GPU: "1", CPU: "16", Memory: "128Gi", SharedMemory: "64Gi"`）。
  - Encoder 与 Decoder 支持按角色配置 `mode: "create" | "reuse"`（或通过顶层 `existing_services` 声明复用，PAI H200 复用实例支持固定 `pai: true, job_id, job_name` 进行运行态身份核验），并支持 `service_count >= 1`（单进程多 service、每 service 单活跃会话，容量以实测为准）。
  - `profile` 支持 `auto`、`h100`、`h200`、`b300`；用户侧无 `role_launcher` 开关，统一走 profile launcher。
  - 通用编排器不强制注入 `NCCL_MIN_NCHANNELS=2` / `NCCL_MIN_CTAS=2`；跨机 SM90 H100+H200 拓扑可通过 `target.topology_profile = "cross_host_sm90_h100_h200"` 启用版本化 Workaround。
  - `secret_files` 仅保存容器内平台挂载路径引用（`ROLE_POOL_SIGNING_KEY`、`ROLE_POOL_REGISTRATION_TOKEN`）；PAI 站点支持 `image_auth_file`（或环境变量 `SGLANG_PAI_IMAGE_AUTH_FILE`）在本地渲染 `pai.json` 时注入 ACR 凭据，严禁在 `spec.json` 或 `history` 持久化密钥明文。

### 输出产物（部署凭据）

- `probe.json` / `spec.json`：并发采集的节点空闲 GPU、端口、镜像 digest、模型路径与 PAI H200 GPU/CPU 配额占用快照（满足条件时 `blockers` 为 `[]`），以及从最小环境输入生成的完整冻结 `spec.json`。
- `state.json`：包含独立 `deployment_id`、`run_id`、源码归档 SHA256、逐阶段耗时 `timings_sec`（含 `within_10min_sla`）与逐角色资源状态。
- `deployment.json` 与 `DEPLOYMENT.md`：自动汇总部署凭据，含固定环境 OSS 归档前缀 `oss_history_prefix`、逐角色真实源码 SHA、镜像 digest、资产 SHA256、`create`/`reuse` 模式、资源/process/service/boot/GPU/端口身份、`warmup` 预热回执（由运行时 `publish_role_warmup` 自动绑定到 `/health`）、分阶段墙钟耗时 `timings_sec` 以及四大验收门禁 `acceptance_gates`（`within_10min_sla`、`all_roles_warmed_up`、`traffic_ready`、`quality_unchanged`）。
- **默认归档至固定环境 OSS 前缀（可选额外保存到本地 `--local-history-dir`）**：`up`、`selftest`、`attach`、`down`、`rollback` 与 `verify_traffic_quality.py` 默认自动将不可变快照 `records/<deployment_id>/<record_sha256>.json`、`<deployment_id>.json`、不可变逐条索引分片 `entries/<deployment_id>/<entry_sha256>.json`（读取端枚举并与遗留 `index.json` 确定性合并，非 `NotFound` 错误 fail-closed 报错且绝不覆写共享 `index.json`）以及 `<deployment_id>/{deployment.json,DEPLOYMENT.md,traffic_verification.json}` 归档至按环境严格隔离的固定 OSS 路径前缀（如需关闭传 `--no-sync-oss`；如需额外镜像一份到本地目录传 `--local-history-dir <dir>`）：
  - 自测环境（`selftest`，上海 / 泰国）：`oss://leap-world-shanghai/sglang/deployments/selftest` / `oss://leap-worldmodel-thailand-single-az/sglang/deployments/selftest`（自定义自测名自动归入 `.../selftest/<slug>`）
  - 测试上海（`test-shanghai-copy`）：`oss://leap-world-shanghai/sglang/deployments/test-shanghai-copy`
  - 线上上海（`prod-shanghai`）：`oss://leap-world-shanghai/sglang/deployments/prod-shanghai`
  - 线上泰国（`prod-thailand`）：`oss://leap-worldmodel-thailand-single-az/sglang/deployments/prod-thailand`
- `traffic_verification.json`：由 `deploy/sglang/verify_traffic_quality.py` 产出的 Coordinator 真实接流会话、预热核验与逐帧 PSNR/SSIM 画质门禁回执（纯标准库 PNG 编解码，默认同步归档至上述固定 OSS 前缀）。

## 3. 命令说明

所有变更操作默认仅展示离线计划；带 `--execute` 执行变更时默认打印计划并要求确认，显式传 `--yes` 可跳过交互确认（仍强制核验目标身份、资源占用与计划完整性）。

```bash
# 1. 只读探测（支持最小环境输入或完整 spec，并发查询节点与 PAI 列表）
python3 deploy/sglang/orchestrate.py probe --aliyun-profile <your-aliyun-profile> \
  --spec /private/spec.json --output /private/probe

# 2. 生成部署计划并执行（默认自动归档至固定环境 OSS 前缀；可选 --local-history-dir 额外镜像到本地）
python3 deploy/sglang/orchestrate.py up --aliyun-profile <your-aliyun-profile> \
  --spec /private/probe/spec.json \
  --commit "$(git rev-parse HEAD)" --output /private/run --execute --yes

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

# 7. 部署历史查询、详情、对比与受控回滚（默认按 --environment 直接读写固定 OSS 前缀；可选 --local-history-dir）
python3 deploy/sglang/orchestrate.py history --aliyun-profile <your-aliyun-profile> --environment test-shanghai-copy
python3 deploy/sglang/orchestrate.py show --aliyun-profile <your-aliyun-profile> --environment test-shanghai-copy --deployment-id <id>
python3 deploy/sglang/orchestrate.py diff --aliyun-profile <your-aliyun-profile> --environment test-shanghai-copy --from-id <id1> --to-id <id2>
python3 deploy/sglang/orchestrate.py rollback --aliyun-profile <your-aliyun-profile> \
  --previous --current-state /private/run/state.json \
  --output /private/rollback-prev --execute --yes
python3 deploy/sglang/orchestrate.py rollback --aliyun-profile <your-aliyun-profile> \
  --deployment-id <target_id> --current-state /private/run/state.json \
  --output /private/rollback-run --execute --yes

# 8. 真实 Coordinator 接流、三角色预热与相对基线画质核验（严格分离录制基线与候选验收，Fail-Closed）
# 8a. 独立录制基线（仅产出 frames/ + baseline_manifest.json，不绑定候选部署状态）
python3 deploy/sglang/verify_traffic_quality.py record-baseline \
  --source-state /private/baseline-run/state.json \
  --output-dir /private/baseline/quality

# 8b. 候选部署验收（必填已绑定身份的 --baseline-dir，全三角色显式 warmed_up=true + traffic_accepted + PSNR/SSIM/luma 均通过才放行）
python3 deploy/sglang/verify_traffic_quality.py run \
  --state /private/run/state.json \
  --output-dir /private/run/quality \
  --baseline-dir /private/baseline/quality \
  --aliyun-profile <your-aliyun-profile>
```

## 4. H200 `<10 分钟` 极速部署与踩坑沉淀

以下实战踩坑均已固化到 `launch.py`、`orchestrate.py` 与 `verify_traffic_quality.py` 中，无需人工手工规避：

1. **PAI H200 CPU 配额饥饿排队防护**：8 卡 H200 节点总 vCPU 有限（约 192~224），若单角色声明 `CPU: "32"`，在已有 5 卡作业运行时新建 3 角色会因 CPU 超限卡在 `Queuing`。`topology: "h200"` 默认采用 `CPU: "16", Memory: "128Gi", SharedMemory: "64Gi"`，`probe` 自动汇总配额在途 GPU/CPU 并提前告警。
2. **按角色按需 Stage 本地热权重**：`launch.py` 的 `stage_assets()` 按角色仅预拷热权重（`dit` 拷 `transformer` + `adaln-table-unsplit`；`encoder` 拷 `video_vae` + `processor`；`decoder` 拷 `lynn`），其余外围目录建立符号链接，避免全 H200 部署时 `encoder`/`decoder` 白拷 56 GB `transformer` 抢占 OSS 带宽，同时保持 `verify_assets` 完整路径校验 `<1s` 完成。
3. **快速资产清单 `--sized-weights`**：大体积 `.safetensors/.pt` 权重采用 `{bytes, sha256}` 快照校验文件大小，配置文件仍强制计算全文 SHA256，消除每次启动重读数十 GB OSS 挂载文件的耗时。
4. **PAI 多角色并发提交**：`submit()` 先完成全部新建角色的 write-ahead 状态落盘，再并发调用 PAI `CreateJob` 同时拉起三角色，将启动耗时从串行累加缩短为单角色最慢路径（实测端到端 3~5 分钟，稳定 `< 10 分钟`）。
5. **独立 Encoder 源预填开关**：`launch.py` 为独立 `encoder` 默认注入 `SGLANG_ZING_SOURCE_PREPARE=1`，防止 Coordinator 下发 `ControlPrefill` 时因缺少预填路由失败。
6. **ACR 凭据零明文持久化**：`save_history_record()` 自动剥离内联 `ImageConfig.Auth`；渲染时支持 `sites.h200.image_auth_file` 或 `SGLANG_PAI_IMAGE_AUTH_FILE` 动态注入，确保 `history` 与 OSS 归档零密钥泄漏且跨会话一键回滚可用。
7. **Eflo HTTP 状态码透传与全 PAI 0-GPU 容器自检中继**：修复经 Eflo 转发时 `403 MANAGED_CONTROL_REQUIRED` 被吞为通用异常导致托管复用/重入识别失败的问题；全 H200 PAI 部署无 K3s Pod 时，`remote_selftest()` 自动优先复用 Eflo 节点本地缓存的 0-GPU SGLang 运行时容器（`ctr ... run --rm --net-host`）运行 `worker_group_master.py`（规避宿主机 Python 缺失 `numpy`/`torch` 的问题，并使用 `__SHA__` 等唯一占位符避免替换子串冲突）。
8. **多租户 Coordinator 精确兼容性组容量核验**：共享路由下其他租户组满载会导致公开 `/v1/capacity` 的 `free_slots=0`；当配置 `WHITEBOX_ROOM_TOKEN` 时，`verify_pool()` 与 `verify_traffic_quality.py` 调用 `POST /v1/role-pool/capacity` 精确读取 `roles.model_group.free_slots >= 1`，并在复用模式下强制校验被复用服务的 `compatibility` 一致性。
9. **阿里云 OSS CLI Endpoint 规范**：`aliyun oss cp` 统一使用顶层 `--endpoint oss-<region>.aliyuncs.com`，避免 `--region` 参数报错或默认 Endpoint `403 AccessDenied`。
10. **PAI-DLC 2048 字节单行日志截断与 `runtime.gpu` 冲突防护**：阿里云 PAI-DLC `GetPodLogs` 严格在 2048 字节处截断单行容器日志，而完整 `receipt` 嵌套的 `runtime` 字典自带 `"gpu": "NVIDIA L20X"`，排在顶层 `"gpu": "GPU-<uuid>, ..."` 之前。`launch.py` 在输出完整行外额外输出 `<1800` 字节的紧凑回执（含 `worker_config` 时自动裁剪 `runtime` 等大字段），且 `orchestrate.py:last_receipt()` 优先提取含 `GPU-` UUID 的字段并在 `attach()` 自动刷新缺失 UUID 的历史回执。
11. **`rollback --previous` 对优雅下线（`stopped`）上一版本的精确选择**：升版替换时先执行 `down` 会将上一稳定版本的历史 `phase` 更新为 `stopped`；`resolve_previous_deployment_id()` 将具备完整 `rollback_bundle` 的 `stopped` 版本与 `ready`/`attached` 按时间倒序同级选择，防止越过紧邻的上一版本错误回退到更早的旧部署。
12. **DMD Few-Step 步数对齐与 Causal AR+DMD 多帧流式画质门限**：`verify_traffic_quality.py` 自动从 `state.json` 的 `dit_options.zing_denoising_step_list` 推导推理步数注入 `WHITEBOX_INFERENCE_STEPS`（规避默认 4 步与 H200 DMD 2 步冲突触发 `prefill_invalid`），优先解码 `SourceSession` 归档的 `spool/*/model.mp4` 全量输出帧，并联合校验首帧高精度门禁（`PSNR >= 28 dB`、`SSIM >= 0.90`）、多帧自回归流式均值门禁（`PSNR >= 25 dB`、`SSIM >= 0.85`）与全局亮度一致性门禁（`|Δluma| <= 5.0`）。

## 5. 失败恢复与重入保证

- **未知创建保护**：`up` 提交前先落盘 `submitting` 状态与 `InvokeId`；若云 API 超时，状态保留以供人工对账，禁止自动重复创建。
- **原地 `attach` 幂等重入**：部分角色完成 `/managed/attach` 后若后续角色超时，已挂载角色立即持久化 `boot_epoch` 与 `service_id`；重试时通过签名 `/managed/observe` 校验空闲状态并幂等重放，最后在 `verify_pool` 对账三角色身份。
- **`down` 删后重试恢复**：容器或作业停止后写入 `stopped-unretired` 状态；若 Coordinator `confirm-stopped` 或 `retire-process` 瞬时失败，再次执行 `down` 会跳过已删除的 Pod/Job 预检并直接完成 Coordinator 退役闭环。
- **受控回滚恢复**：`rollback` 校验完整 `rollback_bundle`（拒绝仅含摘要或回执的记录），保护 `reuse` 共享服务，不要求重获旧 GPU UUID / Pod UID / Job ID，也不删除共享缓存；中断后重跑可从 `rollback_stopped_current` 检查点安全续跑。

## 6. 验收边界

- `inspect` 与 `/health` 仅证明进程就绪与配置身份匹配，不代表通信组可建或业务出帧。
- `selftest` 仅证明隔离通信组 `build → quiesce → destroy` 生命周期正常，不代表模型生成质量或帧率。
- `attach` 仅证明角色已注册到 Coordinator 且路由容量可见，不代表 Room/WTP 真实浏览器 WebRTC 呈现通过；完整端到端验收需结合 4-Case 真实模型回归与浏览器出帧证据。

## 预热、验收工具、镜像和回滚的边界

- `verify_traffic_quality.py` 是部署后由操作人员或部署流程显式运行的验收 CLI，调用示例见上文；`orchestrate.py` 不自动调用它，模型 worker 也不调用它。它通过 Coordinator 发起真实 `SourceSession`，核验出帧、租约释放和基线画质，再把结果写回部署凭据。仅传 `--state` 时，Coordinator 地址取自 `state.target.coordinator_url`；显式 `--coordinator-url` 优先。
- `warmed_up: true` 仅表示该 worker 的启动预热请求已完成。DiT 默认运行 `832×480`、一个 latent 的合成输入请求；Decoder 默认解码 `832×480`、一个 latent。Encoder 的 source VAE 预热使用 tile 大小的输入（仅在启用对应 layout 时运行），Qwen 使用短文本且无参考图。它们不证明 736p、长 prompt、多参考图、动态换图、长会话或所有形状的编译缓存已经预热；736p 必须另跑匹配实际宽×高的请求和验收。当前接流画质工具的实时回放固定为 `832×480`，不用于宣称 736p 已通过。
- 本编排器消费 `sites.*.image` 中固定的 `@sha256:` 镜像，不构建镜像，也不自动安装 seedleap-kernels。仓库另有 `docker/whitebox-model.Dockerfile`，通过 `deploy/whitebox/kernels-wheel.lock.txt` 安装预编译 wheel 并调用 `verify_model_kernels.py` 校验。指定给本编排器的任意其他镜像不能据此视为已包含该 wheel，仍须核验镜像来源和内部依赖。
- `rollback --previous` 仅选择当前部署之前、环境/region/workspace/quota/Coordinator/pool、模型和路由均匹配的 `ready` / `attached` / `stopped` 历史；不回退到其他环境或仅规划的版本。显式指定历史版本也在停止当前资源前核验相同身份边界。源码 SHA、镜像 digest 和配置允许随目标版本变化。
- 回滚执行会校验历史源码包和渲染工具、停止当前 owned 资源、按历史配置新建 owned 资源并等待角色就绪，保护 reused 服务。它不是热切换，也不自动完成后续 `selftest`、`attach` 和接流画质验收；这些步骤完成前不能宣称流量已恢复。镜像/资产/凭据不可用、历史复用的 E/D 已停机或配额不足仍会阻止恢复。回滚完整 E2E 统计包含停止旧版本的 `stop_previous_sec`。
