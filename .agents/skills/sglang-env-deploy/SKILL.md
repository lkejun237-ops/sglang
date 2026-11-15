---
name: sglang-env-deploy
description: 在已授权的目标环境部署、复用、检查或回滚 SGLang Zing H3 Encoder/DiT/Decoder，使用仓库原生编排器处理源码、资产、角色池接线和部署回执；不负责外围服务发布、生产切流或性能基准。
---

# SGLang 模型服务部署

这是 SGLang 仓库自有 skill，与 `deploy/sglang` 同版本维护。复用原生工具，不另写源码打包器、选卡器、启动器或平台提交器。
根目录下的 `deploy/sglang/ORCHESTRATION.md` 是编排接口入口，`deploy/sglang/README.md` 是底层资产/启动合同；
按当前源码读取，不能沿用旧 #655 的 schema 2、固定 1102 或“无编排命令”的历史说明。
原始需求与现状的对应、凭据/回执缺口见 [接口与证据边界](references/contracts.md)。

## 先确定输入与范围

- 明确目标环境、角色 create/reuse、模型/checkpoint、分辨率、硬件/profile、源码 SHA、已有服务及本次资源授权。
  缺失信息先从已指定部署的 spec/state/history 和仓库配方取得；仍无法确定才询问，不能猜测现场 IP、UUID 或资产目录。
- 支持测试与生产环境；使用真实环境名和目标地址，执行范围须覆盖用户已授权的精确环境、模型池和 owned 资源。
  Gateway/Coordinator/Room/WTP/SDK 的发布不在本 skill 范围；已有 Coordinator 仅作为模型侧注册/租约接线依赖。
- 新建三角色（含全 H200 `topology: "h200"` 新建 `1 Encoder + 1 DiT + 1 Decoder`，默认 `CPU: "16"` 防配额饥饿）与仅新建 H200 DiT、复用现有 H100/H200 Encoder/Decoder（PAI 实例支持固定 `pai: true, job_id, job_name` 核验身份与 `compatibility` 域一致）都可用。每 service 单活跃会话；E/D 多 service 须有实际进程预算，
  不把多 service 当作已验证容量。reuse 仅借用/释放本次组或租约，不重启、重复注册、drain 或 retire 共享服务。
- 使用前读 `benchmark/AI_INFRA_TEST_MACHINES.md`、`AI_INFRA_TASK_OWNERS.md` 和
  `docs/docs/sglang-diffusion/zing/deployment_zing_h3.mdx`，按数字 UserId 分清提交账号与业务负责人，重新核验占用。
  H100 K3s 通过 Eflo 进入节点，不在本机未知 kube context 创建资源；PAI GPU 由平台分配。

## 固定源码、模型与配置

1. 读取 `deploy/sglang/examples/orchestration.json`，由部署准备端根据现场观测生成 Git 外的 spec；
   不直接执行示例、不把密钥值写进 spec/history。模型、Lynn/Qwen/VAE、pipeline/profile、AdaLN 必须匹配，三角色不能混用新旧模型。
   指定 1102 时缺少上海 serving/table provenance 就停止，不能回退到1027；没有指定模型时遵循当前仓库部署决策入口。
2. 准备端固定已推送的完整 SHA、镜像 digest、资产回执 hash、输入和分辨率身份。
   正式交付使用 `origin/main` 可达 SHA；开发验证可用明确标注的已推送 PR SHA，不把它报告为正式发布版本。
   `up` 会校验归档工具与当前 checkout 一致；若不一致，使用指定 SHA 的独立 checkout，不从镜像或脏目录覆盖代码。
3. 按目标环境选择原生 profile/pipeline。共享策略由 `zing_h3_profile_launcher` 维护，
   显式覆盖由 spec 中的角色配置提供并进入事实回执；不在 skill 复制优化默认值，也不为了通过旧 runtime 静默改采样、chunk/split 或模型。
4. 优先复用已验证资产。需要生成/核验清单时使用 `launch.py assets`（详见参考），外部组件也要列入。
   仅资产缺失才按需使用项目已安装的 `zing-serving-checkpoint`、`aliyun-mgw-transfer`；
   仅目标缺少镜像才使用 `aliyun-acr-sglang-mirror`，不将迁移/导出/镜像同步设为每次部署必做。

## 探测、计划与执行

以下命令从选定 SGLang checkout 根目录运行。路径来自本次准备结果；本机 Aliyun 使用 `shengdong` profile，其他执行机使用其已有授权身份。
`probe` 只读查询远端、写本地观测；`up` 不带 `--execute` 只归档、渲染并保存计划，不创建 GPU 资源。

```bash
python3 deploy/sglang/orchestrate.py probe --aliyun-profile shengdong \
  --spec /private/deploy/input.json --output /private/deploy/probe
python3 deploy/sglang/orchestrate.py up --aliyun-profile shengdong \
  --spec /private/deploy/probe/spec.json --commit "$SOURCE_SHA" \
  --output /private/deploy/plan --history-dir /private/deploy/history
```

检查 probe blockers、完整模型/镜像/源码身份、拓扑与端口、owned/reuse 列表、Secret 文件路径和挂载。
执行前确保本次部署授权覆盖精确目标与资源；已有明确授权时无需重复询问，保留工具身份/占用 gate。
执行使用**新的输出目录**（`up` 不是已有 plan/state 的续提命令）：

```bash
# 仅在本次目标与资源已获部署授权后；--yes 只跳过工具交互，不跳过核验。
python3 deploy/sglang/orchestrate.py up --aliyun-profile shengdong \
  --spec /private/deploy/probe/spec.json --commit "$SOURCE_SHA" \
  --output /private/deploy/live --history-dir /private/deploy/history --execute --yes
python3 deploy/sglang/orchestrate.py inspect --aliyun-profile shengdong \
  --state /private/deploy/live/state.json
```

提交超时或 UNKNOWN：保留原 state/InvokeId/JobId，按唯一身份与规格回读，先排除实际已创建；
不重新运行 up 创建另一批资源、不删除意图记录。不要把 Pending/Running 或 API 受理写成部署成功。

## 分层验收与模型接线

- `inspect`：核对实际 imageID、源码与导入、模型/config/table、GPU UUID、Job/Pod/process/service/boot 身份及 readiness。
  对缺失字段保留 unknown/null，不能用期望值补成实测值。
- `selftest --state ... --execute --yes`：会预留、建组、排空并释放；只证明通信生命周期，不证明模型出帧或性能。
  只在用户授权覆盖该自测时执行，失败或释放 unknown 时停止追加测试，保留现场，不自动重启共享服务。
- 模型出帧回归：使用仓库自有 `$room-sglang-source-replay`，先素材准备/哈希核验和 preflight，再 real-sglang run。
  完整命令在 `benchmark/whitebox/SOURCE_REPLAY_INTEGRATION.md`；三 service 地址与测试 Coordinator 两种模式互斥。
  不用 mock 代替真实服务失败，也不把 DiT 管理端口交给通用生成客户端。480/736 分开，缺档明确未测。
- 需要加入已指定模型池时，先离线查看 `attach --state ...`，再在接线授权内执行
  `attach --state ... --execute --yes`（默认 in-place）。只激活本次 owned 服务，复用角色保持原注册身份。
  不擅用 `--recreate`；它不是原地接线的等价重试。
- 浏览器/WebRTC/WTP 业务 E2E 是另一个可选验收层，按用户授权另行执行；不是模型部署的强制 SDK 构建或四角色发布前提。

下面是两项独立变更的完整调用形式；根据本次授权与验收目标选择，不机械执行全部步骤：

```bash
python3 deploy/sglang/orchestrate.py selftest --aliyun-profile shengdong \
  --state /private/deploy/live/state.json --execute --yes
python3 deploy/sglang/orchestrate.py attach --aliyun-profile shengdong \
  --state /private/deploy/live/state.json --execute --yes
```

## 组件监控核验（新建与复用分别记录）

- 新建角色由 `launch.py` 自动设置 `WORLD_MODEL_METRIC_SERVICE=encoder|dit|decoder`；
  内嵌 Qwen 保持 `service="encoder", component="qwen_vlm"`。
  可在角色 `environment` 设置 `WORLD_MODEL_METRIC_LANE`，默认 `default`。
  不手工覆盖 service 或 `PROMETHEUS_MULTIPROC_DIR`；启动器按 worker boot 分配新目录，
  在导入 Prometheus 前创建，同一 DiT 的 ranks 共用，角色/实例/重启之间隔离。
- `receipt.json.metrics` 给出本实例的 URL、service、lane、目录及开关；从实际回执取
  `/metrics` 地址，不假定默认端口。一个共享模型的多 service 只抓 base HTTP 一次。
  SP2 的 rank0 HTTP 可汇总同目录的 rank 样本；默认面板选择 rank0，不能将两个 rank
  的逻辑帧数相加。本编排入口仍只支持 SP1，不因监控配置增加 SP2 部署能力。
- 复用角色不改环境、不清目录、不重启；只读检查现有源码身份和 `/metrics`，缺少指标
  或身份不明时标为“未验证/需后续发布”，不得把健康检查当作监控验收。
- 发正常测试请求后核验组件样本、`service/lane/rank` 与分辨率；未触发的组件允许无样本。
  检查 errors/dropped 的增量。分别记录端点可访问、有效样本、真实模型性能对照和线上
  Grafana 接入状态；不要由其中一项推断其他项通过。
- Prometheus 的 `region/pool` 是抓取配置标签，`instance` 通常由 target 自动添加，
  模型不会自动发现它们。生成/检查抓取配置，确认抓取端网络可达；线上配置变更按授权执行。
  不清理活跃目录；历史目录只在确认所有所属进程已退出后随本次运行目录回收。

指标口径、ms/FPS 查询与抓取示例统一见
`docs/docs/sglang-diffusion/zing/specs/COMPONENT_METRICS.md`，不在技能内复制第二套定义。

## 历史、回滚与交付

- 部署凭据与历史默认归档至按环境隔离的固定 OSS 路径前缀（`selftest`：`oss://leap-world-shanghai/sglang/deployments/selftest` 或泰国 `oss://leap-worldmodel-thailand-single-az/sglang/deployments/selftest`；`test-shanghai-copy`：`oss://leap-world-shanghai/sglang/deployments/test-shanghai-copy`；`prod-shanghai`：`oss://leap-world-shanghai/sglang/deployments/prod-shanghai`；`prod-thailand`：`oss://leap-worldmodel-thailand-single-az/sglang/deployments/prod-thailand`），可选传 `--local-history-dir <dir>` 额外镜像保存到本地目录（PAI 镜像拉取凭据通过 `image_auth_file` 或 `SGLANG_PAI_IMAGE_AUTH_FILE` 注入，历史记录自动脱敏）。使用 `history --environment ...`、`show --environment ... --deployment-id ...`、`diff --environment ... --from-id ... --to-id ...` 可直接从 OSS 回读记录。
- 回滚支持 `--previous` 自动从 OSS（或 `--local-history-dir`）定位当前部署之前的上一可用版本（含升版时已通过 `down` 优雅停止且具备完整 `rollback_bundle` 的紧邻上一版，或 `--deployment-id <id>` 指定目标版本），先运行 `rollback --previous --current-state ... --output <新目录>` 看计划；
  仅在回滚授权内追加 `--execute --yes`。必须具备完整 rollback_bundle，不仅是一份摘要，不能要求重获旧 UUID/Pod/Job 身份，且绝不停止或重置 `reuse` 共享角色。
- 部署后准入核验（全部门禁 Fail-Closed）：检查 `deployment.json` 中 `timings_sec`（`< 600s` 10 分钟 SLA）、新建与复用全部三角色 `/health` 显式返回 `warmed_up: true`（缺失字段或 `false` 均判定未通过），并在 VPC 内运行 `deploy/sglang/verify_traffic_quality.py run --state ... --baseline-dir ... --output-dir ...`（首次基线录制使用独立分离的 `record-baseline` 子命令生成 `baseline_manifest.json`）核验真实 Coordinator 接流会话（`traffic_accepted: true` 才会使 `acceptance_gates.traffic_ready` 通过）及绑定身份基线的逐帧 PSNR/SSIM/luma 画质门禁（产出 `traffic_verification.json`）。
- 清理仅使用保存 state 的 `down --state ... --execute --yes`，仅在该资源清理授权内执行。
  stopped-unretired 状态通过同一 state 重试完成退役；reuse 服务、共享缓存、其他部署和活跃会话不在清理范围。
- 保存 `state.json`、原始 receipt、`deployment.json`、`DEPLOYMENT.md`、history/rollback_bundle、平台回读及测试结果。
  报告完整源码/镜像/模型身份、各角色 create/reuse、实际资源与端点、执行与失败/未执行项、真实耗时和证据路径。
  部署 ready、组生命周期、模型出帧、池注册、浏览器呈现分开汇报；不承诺未经测量的启动 SLA 或性能收益。
