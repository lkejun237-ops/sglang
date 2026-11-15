# SGLang 多角色编排、唯一硬件策略启动链与受控回滚设计

本文档记录 SGLang Zing-H3 独立角色池（Encoder / DiT / Decoder）与单机集成式部署的唯一启动链设计、硬件策略与拓扑 Workaround 归位、多 service 创建/复用模型，以及持久化部署历史与受控回滚机制。

## 1. 两条互不反调的唯一启动链

所有模型角色统一由 `python/sglang/multimodal_gen/tools/zing_h3_profile_launcher.py` 维护硬件策略（H100 / H200 / B300）、参数合成与进程监管，不在 `deploy/sglang/` 或 `deploy/whitebox/` 中维护第二套硬件加速默认值或进程监管循环：

1. **单机集成式路径（Integrated）**：
   `deploy/whitebox/release.py`（`plan` / `build` / `verify`）→ `deploy/whitebox/model_service.py`（bundle / 资产 / 布局校验薄适配）→ `zing_h3_profile_launcher.py`（`--mode integrated`，统一维护四卡完整组 argv 生成与进程监管）。
   - `deploy/whitebox/fixed_roles.py` 与 `zing_b300_launcher.py` 仅作为向后兼容薄转发层委托给 `zing_h3_profile_launcher.py`。
2. **独立三角色编排路径（Independent）**：
   `deploy/sglang/orchestrate.py` → `deploy/sglang/launch.py` → `zing_h3_profile_launcher.py`（`--mode independent`）→ `zing_h3_role_launcher.py` → 原生角色入口（`zing_source_encoder_server` / `dit_worker` / `decoder_worker`）。
   - 用户侧移除 `role_launcher` 选择项，不静默回退无策略的 native 默认值。

## 2. 硬件策略、版本化拓扑 Workaround 与条件化显存策略

硬件与拓扑相关策略统一收敛在 `python/sglang/multimodal_gen/tools/zing_h3_role_launcher.py`：

- **版本化拓扑 Workaround（`TOPOLOGY_WORKAROUNDS`）**：
  - `NCCL_MIN_NCHANNELS` 与 `NCCL_MIN_CTAS` 不是主机配额，通用编排器默认不强制注入 `2`。
  - 跨主机 SM90（K3s H100 ↔ PAI-DLC H200）建组需要对齐通道数时，通过版本化拓扑 profile `cross_host_sm90_h100_h200` 显式声明，并附带 `nccl_version: "2.27.3+cuda12.9"` 与拓扑背景记录。
- **条件化 DiT 显存比例策略（`SCOPED_MEMORY_POLICIES`）**：
  - `mem_fraction_static=0.875` 仅在 `h200_model1027_480_bf16_sp1_block0_cache` 条件（`model_id="model1027"`、`global_step_010000`、`num_frame_per_block=[3,2]`、`split_first_latent=False`、`profile="h200"`、`resolution="480"`、`precision="bf16"`、`sp_degree=1`、`zing_block0_cache_mode="on"`）下生效，不作为 H200 通用默认值。

## 3. 多拓扑环境配置与 Encoder/Decoder Create/Reuse

- **环境拓扑覆盖**：
  - `H200 Encoder + H200 Decoder + H200 DiT`（全 PAI-DLC H200 三角色并行拓扑，`topology: "h200"`，默认每角色 `CPU: "16"` 防配额饥饿）
  - `复用在线 H100/H200 Encoder + Decoder + 新建 H200 DiT`（支持 PAI 复用实例固定 `pai: true, job_id, job_name` 校验在线身份与 `compatibility` 域一致）
  - `H100 Encoder + H100 Decoder + H200 DiT`（K3s + PAI-DLC 跨机拓扑）
  - `B300 Encoder + B300 Decoder + B300 DiT`（全 PAI-DLC 三角色拓扑）
- **逐角色 `create` / `reuse` 与原生多 Service**：
  - Encoder 与 Decoder 支持按角色独立配置 `mode: "create" | "reuse"`（或通过 `existing_services` 声明复用），并接入原生 `--service-count >= 1` 多 service 能力（单进程共享模型、每个 service 同时最多一个活跃会话，容量以实测 `pool.capacity` 为准）。
  - `reuse` 模式下绝不重启、不重复注册、不 `drain`、不 `retire` 共享服务；`down` 与 `rollback` 仅释放本轮持有的活跃租约或临时通信组，只停止本轮新建（`owned`）的资源。

## 4. 部署汇总、10 分钟 SLA、预热/接流/画质核验与受控回滚

- **自动汇总产物与 10 分钟 SLA 门禁**：每次部署生成独立 `deployment_id`，并在运行目录自动产出结构化 `deployment.json` 与可读 `DEPLOYMENT.md`，逐角色记录真实源码 SHA、镜像 digest、资产 SHA256、`create`/`reuse` 模式、资源/process/service/boot/GPU/端口身份、`service_count`、生效 profile/argv、各阶段墙钟耗时 `timings_sec`（含 `within_10min_sla`）、`/health` 启动预热回执（`warmed_up` / `warmup.elapsed_s`）及接流与画质门禁 `acceptance_gates`。
- **预热、真实接流与画质核验（`deploy/sglang/verify_traffic_quality.py`，全门禁 Fail-Closed）**：
  - `warmup_encoder()`、`warmup_dit()`、`warmup_decoder()` 内部默认调用 `publish_role_warmup()` 自动向本进程 `WorkerHTTPServer` 发布预热回执；编排器与核验工具强制要求新建与复用全部三角色 `/health` 显式返回 `warmed_up: true`（缺失字段或 `false` 均 fail-closed 记为 `false`），并在 Coordinator `attach` 后核验精确 `(profile, compatibility, network_domain)` 角色池 `roles.model_group.free_slots >= 1`。
  - 严格分离基线录制（`record-baseline`，产出 `baseline_manifest.json` 锁定 `case_id`/`seed`/`inference_steps`/`frames_sha256`，绝不标记候选部署通过）与候选验收（`run --baseline-dir`）；`acceptance_gates.traffic_ready` 显式依赖 `pool_attached` 与 `traffic_accepted is True`，`acceptance_gates.quality_unchanged` 显式依赖已绑定身份的 `baseline_identity` 与 `baseline_comparison.passed is True`。
- **固定环境 OSS 默认归档与回滚材料校验**：
  - 部署历史与凭据（不可变记录 `records/<deployment_id>/<record_sha256>.json`、`<deployment_id>.json`、不可变逐条索引分片 `entries/<deployment_id>/<entry_sha256>.json`、`deployment.json`、`DEPLOYMENT.md`、`traffic_verification.json`）默认归档至按四类环境隔离的固定 OSS 路径前缀（`selftest`、`test-shanghai-copy`、`prod-shanghai` 使用 `oss://leap-world-shanghai/sglang/deployments/<env>`；`prod-thailand` 与泰国自测使用 `oss://leap-worldmodel-thailand-single-az/sglang/deployments/<env>`），读取端枚举不可变分片并与遗留 `index.json` 确定性合并（严格区分 `NotFound` 与网络/鉴权/限流错误，非 `NotFound` 失败 fail-closed 中止），支持可选 `--local-history-dir` 额外保存到本地。历史记录包含完整 `rollback_bundle`（固定源码包 commit/sha256/uri、带 `@sha256:` 的镜像 digest、完整验证后的 spec、逐角色资产身份与可获取挂载位置、路由/容量前后状态、Secret 文件路径引用及角色所有权）。
  - 严禁在 `state.json`、`spec.json` 或历史记录持久化密钥明文；仅凭摘要或运行回执（缺少完整 `rollback_bundle`）的记录在 `validate_rollback_record` 中直接拒绝回滚。
  - 支持 `rollback --previous` 自动定位最近一次完成就绪（含升级时经 `down` 正常停机的 `stopped` 版本）的历史稳定版，先用目标历史版本的 `launch.py` 离线校验源码与配置包，再优雅停止当前 `owned` 资源并拉起目标版本；实机验收记录见 `benchmark/whitebox/sglang_h200_deploy_verify_20261114.md` 与 `benchmark/whitebox/sglang_shanghai_copy_20261110.md`。
