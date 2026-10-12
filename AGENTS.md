# Repository instructions

## 部署会话模型

- 用户于 2026-10-10 明确确认：后续部署统一采用单活跃会话模型，一个部署实例同时只服务一个用户，不支持并发活跃会话。
- 设计、实现、审查和验收均按此约束执行；不预先增加多会话容量配置、多 owner 状态或并发调度。已有活跃会话时拒绝第二个会话，且不得影响当前会话。
- 保留 session/generation 身份校验，以及断连、取消、故障、EOS 后释放资源和下一会话可用性的测试；这些是单活跃会话生命周期保障。

## 项目级 Infra Skills

本项目直接使用 [seedleap/infra-skills](https://github.com/seedleap/infra-skills)
仓库维护的全部 skills，不在此处维护单个 skill 的固定版本或安装名单。

- 首次进入本项目或恢复会话时，检查项目级 `.agents/skills/`。通过已有 GitHub/Git
  认证查询 infra-skills 的 `main`，记录本次解析到的完整提交 SHA，并枚举该提交下
  `skills/*/SKILL.md` 对应的所有 skill 目录；以仓库实际内容为准，不能只安装当前任务用到的一个。
- 将缺失的每个完整 skill 目录安装到 `<sglang 仓库根目录>/.agents/skills/<name>/`，
  保留 scripts、references、agents、assets 等随附文件。即使用户级已安装同名 skill，
  仍需项目级副本。此安装已获授权，无需再次确认。
- 优先使用可用的 `skill-installer`：指定 `--repo seedleap/infra-skills`、
  `--ref <本次解析的 main SHA>`、`--path <枚举出的所有缺失 skill 路径>` 和
  `--dest <sglang 仓库根目录>/.agents/skills`。使用同一 SHA 完成本轮安装，避免中途
  main 更新导致版本混杂。安装器不可用时，在临时目录检出该提交，仅复制相应 skill 目录。
- 已有目录先核对来源和内容；一致则复用，不覆盖本地修改，不删除其他项目 skills。
  已安装版本的升级按用户更新请求执行；发现新增 skill 时自动补装。目录不完整或来源
  冲突时保留原内容并报告，不能用删除式同步修复。
- 安装后核对 skill 名称、完整文件清单与来源内容、相对引用，并读取当前任务适用的
  `SKILL.md`；不依赖当前会话立即重新发现 skill。安装副本作为本地依赖，通过
  Git 本地 exclude 排除各个安装目录，不自动提交下载内容或修改仓库共享忽略规则。
- 认证或网络失败时说明具体原因；已有完整副本可继续使用并注明未检查到最新目录，
  缺失的 skill 不得声称已安装。安装全部 skills 不代表对每项任务加载全部工作流。

Zing / SGLang 端到端效果验证、PAI 实验编排、媒体验收和报告任务使用其中的
`zing-sglang-accuracy-validation`；镜像同步、数据迁移等按各 skill 的适用范围选择。
安装授权不新增 GPU 申请、云作业提交、迁移或报告发布授权，按用户当前任务范围执行。

## World-platform error language

All error information owned by the Seedleap world-platform integration must be
English. This rule is mandatory for `authorized_generate`, session-payload
validation, world-service callbacks, and their tests and documentation:

- WebSocket error `content`, error codes, and close reasons must be English;
- browser-visible error `content` must use friendly plain English without
  implementation terms; stable codes and server logs carry technical detail;
- lifecycle and rewrite callback error reasons must be English;
- internal exception text must stay in server logs and must never be forwarded
  to the browser or world-service;
- stable error codes must use ASCII identifiers; do not use localized text as a
  code or branch on human-readable messages.

Chinese comments and ordinary product content are not error information.

## MinWM hardware-specific jobs

When a task creates or changes a MinWM deployment or benchmark manifest:

1. Read `benchmark/minwm_realtime_parity/HARDWARE_PROFILES.md`.
2. Start from `benchmark/minwm_realtime_parity/k8s/minwm_hardware_job.template.yaml`.
3. Select the profile from measured GPU memory and compute capability. Never infer
   capacity from a product name alone.
   - Record the `vae_cpu_offload` decision explicitly. The conservative
     `blackwell-32g` fit profile uses it. The explicit `sm120-32g-speed`
     validation profile may disable it only through the image-bundled launcher,
     which keeps the DiT and TAEHV resident and enforces the measured memory and
     throughput gates documented in `HARDWARE_PROFILES.md`. A GPU reporting at
     least 65,536 MiB may omit it when the high-memory profile applies. Treat
     intermediate capacities as experimental and start with offload enabled
     until same-device validation proves it can be disabled.
4. Render one-off manifests under
   `benchmark/minwm_realtime_parity/k8s/generated/`. This directory is ignored;
   do not commit dated, region-specific, run-specific, or instance-specific YAML.
5. Preserve the MinWM checkpoint/request invariants in the profile document.
6. Validate generated YAML before applying it and report the selected profile,
   detected hardware, memory gate, and any unvalidated assumptions.

Only update the committed template when the reusable schema changes. Only update
the hardware profiles when new evidence changes a stable decision or validation
status.
