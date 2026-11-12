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
- 新建三角色与仅新建 DiT、借用现有 Encoder/Decoder 都可用。每 service 单活跃会话；E/D 多 service 须有实际进程预算，
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

## 历史、回滚与交付

- 使用 `history --history-dir ...`、`show --history-dir ... --deployment-id ...`、
  `diff --history-dir ... --from-id ... --to-id ...` 回读已持久保存的记录。
- 回滚先运行 `rollback --history-dir ... --deployment-id ... --current-state ... --output <新目录>` 看计划；
  仅在回滚授权内追加 `--execute --yes`。必须具备完整 rollback_bundle，不仅是一份摘要，不能要求重获旧 UUID/Pod/Job 身份。
- 清理仅使用保存 state 的 `down --state ... --execute --yes`，仅在该资源清理授权内执行。
  stopped-unretired 状态通过同一 state 重试完成退役；reuse 服务、共享缓存、其他部署和活跃会话不在清理范围。
- 保存 `state.json`、原始 receipt、`deployment.json`、`DEPLOYMENT.md`、history/rollback_bundle、平台回读及测试结果。
  报告完整源码/镜像/模型身份、各角色 create/reuse、实际资源与端点、执行与失败/未执行项、真实耗时和证据路径。
  部署 ready、组生命周期、模型出帧、池注册、浏览器呈现分开汇报；不承诺未经测量的启动 SLA 或性能收益。
