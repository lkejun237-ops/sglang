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
  - `H100 Encoder + H100 Decoder + H200 DiT`（K3s + PAI-DLC 跨机拓扑）
  - `B300 Encoder + B300 Decoder + B300 DiT`（全 PAI-DLC 三角色拓扑）
- **逐角色 `create` / `reuse` 与原生多 Service**：
  - Encoder 与 Decoder 支持按角色独立配置 `mode: "create" | "reuse"`（或通过 `existing_services` 声明复用），并接入原生 `--service-count >= 1` 多 service 能力（单进程共享模型、每个 service 同时最多一个活跃会话，容量以实测 `pool.capacity` 为准）。
  - `reuse` 模式下绝不重启、不重复注册、不 `drain`、不 `retire` 共享服务；`down` 仅释放本轮持有的活跃租约或临时通信组，只停止本轮新建（`owned`）的资源。

## 4. 部署汇总、持久化历史与受控回滚

- **自动汇总产物**：每次部署生成独立 `deployment_id`，并在运行目录自动产出结构化 `deployment.json` 与可读 `DEPLOYMENT.md`，逐角色记录真实源码 SHA、镜像 digest、资产 SHA256、`create`/`reuse` 模式、资源/process/service/boot/GPU/端口身份、`service_count`、生效 profile/argv 及各阶段验证与清理状态。
- **持久化历史与回滚材料校验**：
  - 历史记录包含完整 `rollback_bundle`（固定源码包 commit/sha256/uri、带 `@sha256:` 的镜像 digest、完整验证后的 spec、逐角色资产身份与可获取挂载位置、路由/容量前后状态、Secret 文件路径引用及角色所有权）。
  - 严禁将密钥明文写入历史记录；仅凭摘要或运行回执（缺少完整 `rollback_bundle`）的记录在 `validate_rollback_record` 中直接拒绝回滚。
  - `rollback` 生成新的 `deployment_id` 并通过 `rollback_of` 关联目标历史记录，保护复用的共享服务，不要求重获旧 GPU UUID / Pod UID / Job ID，也不删除共享源码或模型缓存。
