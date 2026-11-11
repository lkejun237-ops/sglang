---
name: sglang-env-deploy
description: 使用通用环境配置完成 SGLang Zing H3 三角色（H100 Encoder/Decoder + H200 DiT，或 B300 三角色）的探测、计划确认、新建或复用部署、隔离建组自测、4-Case 回归、Coordinator/Room/WTP 原地挂载、事实回执汇总及历史查询与受控回滚。
---

# SGLang 通用环境部署、历史与受控回滚（`sglang-env-deploy`）

通过 `deploy/sglang/orchestrate.py` 与 `deploy/sglang/launch.py` → `zing_h3_profile_launcher.py`（`--mode independent`） → `zing_h3_role_launcher.py` 唯一启动链，驱动非生产环境下的 SGLang Zing H3 三角色部署、验收与受控回滚。

## 支持拓扑与安全边界

1. **支持拓扑**：
   - **`H100 Encoder/Decoder + H200 DiT`**：Encoder/Decoder 运行在 K3s H100 节点（或复用已有服务），DiT 运行在 PAI-DLC H200 单卡（SP1）。
   - **`B300 三角色`**：Encoder、DiT、Decoder 均使用 `profile="b300"` 与 PAI-DLC B300 站点规格。
   - **角色独立 `create` / `reuse` 与多 Service**：
     - Encoder / Decoder 各自独立支持 `create`（新建）或 `reuse`（通过 `existing_services` 或 `roles.<role>.mode="reuse"` 复用已有服务），并接入原生多 service（`service_count >= 1`，逐 service 单活跃会话，容量以实测为据）。
     - `reuse` **绝不**重启、重复注册、`drain` 或 `retire` 共享服务，仅在测试/会话结束时释放本次租约或通信组；`down` 与 `rollback` **只停止本次新建的 owned 资源**。
2. **严格安全边界**：
   - 严禁操作 `prod-shanghai`、`prod-thailand` 及正式端口（`28110`/`28210`）或停止他人作业。
   - 所有执行变更的子命令（`up --execute`、`attach --execute`、`down --execute`、`rollback --execute`）默认展示冻结计划并要求确认；显式传 `--yes` 可跳过交互确认，但仍强制核验身份、占用与计划，绝不自动扩大操作范围。

## 标准工作流

### 1. 只读探测与规格生成（`probe`）

接受最小环境输入或完整 `spec.json`，输出可追溯实机观测 `probe.json`（空闲 GPU、端口占用、镜像 repoDigest、模型目录、PAI 作业列表，无阻塞时 `blockers` 为 `[]`）及冻结 `spec.json`：

```bash
python3 deploy/sglang/orchestrate.py probe \
  --aliyun-profile <your-aliyun-profile> \
  --spec /private/env-or-spec.json \
  --output /private/probe-run
```

### 2. 隔离部署与健康对账（`up` + `inspect`）

从已提交的 Git commit 打包源码，按冻结计划创建新建角色（复用角色直接核验就绪状态），自动生成 `deployment.json`、`DEPLOYMENT.md` 并写入持久化部署历史：

```bash
COMMIT_SHA="$(git rev-parse HEAD)"
python3 deploy/sglang/orchestrate.py up \
  --aliyun-profile <your-aliyun-profile> \
  --spec /private/probe-run/spec.json \
  --commit "$COMMIT_SHA" \
  --output /private/live-run \
  --execute --yes

python3 deploy/sglang/orchestrate.py inspect \
  --aliyun-profile <your-aliyun-profile> \
  --state /private/live-run/state.json
```

### 3. 隔离并行建组自测（`selftest`）与 4-Case 真实生成

```bash
python3 deploy/sglang/orchestrate.py selftest \
  --aliyun-profile <your-aliyun-profile> \
  --state /private/live-run/state.json \
  --execute --yes

python3 benchmark/whitebox/run_room_sglang_source_replay.py run \
  --mode real-sglang \
  --encoder-url <ENCODER_HTTP_URL> \
  --dit-url <DIT_HTTP_URL> \
  --decoder-url <DECODER_HTTP_URL> \
  --output-dir /private/live-run/four-case
```

### 4. 原地挂载到 Coordinator / Room / WTP（`attach`）

```bash
export ROLE_POOL_SIGNING_KEY="<signing_key>"
export ROLE_POOL_OPERATOR_TOKEN="<operator_token>"

python3 deploy/sglang/orchestrate.py attach \
  --aliyun-profile <your-aliyun-profile> \
  --state /private/live-run/state.json \
  --execute --yes
```

- 默认 `--in-place` 仅对本次新建的 owned 角色签名调用 `/managed/attach`（支持 `service_count >= 1` 逐 endpoint 幂等激活），复用角色保持原注册状态不重入。
- 自动更新 `deployment.json`、`DEPLOYMENT.md` 与历史记录。

### 5. 历史查询、对比与受控回滚（`history` / `show` / `diff` / `rollback` / `down`）

```bash
# 查看持久化的部署历史列表
python3 deploy/sglang/orchestrate.py history --history-dir /private/history

# 查看单次部署完整记录
python3 deploy/sglang/orchestrate.py show --deployment-id <DEPLOYMENT_ID> --history-dir /private/history

# 对比两次部署的源码、镜像、资产、配置与角色所有权差异
python3 deploy/sglang/orchestrate.py diff --from-id <OLD_ID> --to-id <NEW_ID> --history-dir /private/history

# 受控回滚到历史部署（校验完整回滚材料，保护复用服务，生成关联旧 ID 的新 deployment_id）
python3 deploy/sglang/orchestrate.py rollback \
  --aliyun-profile <your-aliyun-profile> \
  --deployment-id <OLD_ID> \
  --current-state /private/live-run/state.json \
  --output /private/rollback-run \
  --history-dir /private/history \
  --execute --yes

# 仅回收本次新建的 owned 资源
python3 deploy/sglang/orchestrate.py down \
  --aliyun-profile <your-aliyun-profile> \
  --state /private/live-run/state.json \
  --execute --yes
```
