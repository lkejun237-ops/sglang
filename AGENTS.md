# Repository instructions

## 开发进度提交与远端备份

- 本项目后续开发默认授权将当前任务的代码、测试、脚本、配置和必要文档提交并推送到
  `origin` 的任务分支（新分支默认使用 `codex/` 前缀）；无需逐次询问是否 commit / push。
  使用已有合适的任务分支，禁止直接推送 `main`、强制推送或混入其他任务／他人的改动。
- 备份不以功能完成、测试全绿或正式 PR 为前提。允许 `WIP` checkpoint commit，提交说明
  必须如实记录未完成项和测试状态；正式交付时再整理为便于审查的提交。
- 每完成一个可描述的小阶段、连续开发满 30 分钟，以及启动远端测试／部署、长时间作业、
  暂停或结束当前任务之前，必须对本任务新增及修改的源码执行 commit 并 push。
  不得把未推送代码长期只留在个人电脑、容器、云机器或 OSS 源码包中。
- push 后核对远端分支 SHA 与本次提交一致，才可报告“已备份”，并提供分支或 commit 链接。
  网络、认证或权限失败时明确报告“尚未完成远端备份”和未推送范围；恢复后优先补推，
  不把本地 commit 当成远端备份成功，也不静默继续积累多个开发阶段。
- 远端测试／部署必须记录各源码仓库已推送的完整 commit SHA。可以从该 commit 打包、
  经 OSS 分发或使用 Git CodeSource；所有实际执行的源码覆盖、补丁和启动脚本也必须
  先纳入对应仓库并推送，运行时不得再悄悄修改。发生修复时重新提交、推送并更新运行回执。
  GPU JIT 等自动生成产物记录生成配置和来源，不作为手写源码覆盖。
- 密钥、token、个人数据、模型权重、视频、编译缓存和大型实验产物不得作为备份提交；
  实验产物按既有归档约定保存。只显式暂存本任务文件或 hunk，不使用无差别 `git add .`。
- 本授权不包含 merge、正式发布、发送消息或新增云资源申请。共享 checkout 中无法安全
  隔离本任务改动时，使用独立工作树／独立提交方式，不擅自整理或提交其他人的工作。

## 文档目录约定

- `benchmark/` 只存放基准测试工具、测试输入、复现说明和实验／验收记录。
- 架构设计、接口契约、实现方案及设计分析不得放在 `benchmark/`（包括其子目录）。
  多模态模型的设计与契约放在 `docs/docs/sglang-diffusion/<model>/specs/`，
  实现计划与设计分析放在同级 `plans/`；其他组件按 `docs/` 已有主题结构归档。
- 设计文档可链接到 `benchmark/` 的验证工具和结果。迁移时同步修正仓库内引用、
  相对链接与代码注释；指向固定提交的历史证据链接保留原路径。

## AI Infra 测试机器

- 默认优先使用上海 H100 测试资源，其次使用泰国 B300；用户指定硬件或
  测试必须依赖特定架构时按该要求执行，不把跨硬件结果混作同条件对照。
- 上海 H200 已部署两个实例，供产品团队做 demo case 使用，不作为默认测试资源。
- 访问入口、region、workspace、资源配额和可复制的查询命令统一见
  [AI Infra 测试机器与访问方式](benchmark/AI_INFRA_TEST_MACHINES.md)。先读该文档再找机器，
  不依赖某个 Codex 的本地 memory、SSH 别名或历史作业仍在运行。
- 查询 H100/H200、B300 任务前，必须读取[GPU 任务归属记录](AI_INFRA_TASK_OWNERS.md)，
  按任务实际提交人的数字 `UserId` 识别账号，人员统一使用两字称呼；业务负责人另行记录。
  H100 独立 K3s 节点缺少可关联的创建者身份时标为未知，不按任务名推断。
- 使用前重新核验 GPU 占用和作业归属；机器清单不代表空闲，也不授权停止其他人的任务。

## Worldkit 跨平台部署

- Prefer the canonical Python/browser dependency stages. Never inherit a full
  Room image followed by `rm -rf /app /opt/worldkit` as a dependency base:
  deletion in a later layer retains the old bytes. An existing verified browser
  filesystem may be extracted through `docker/whitebox-browser-dependencies.Dockerfile`,
  whose final FROM scratch stage copies only the cleaned filesystem and retains
  none of the old layers. Verify the resulting dependencies and complete image
  gates. Preserve ordinary BuildKit cache imports and the exact source/SDK pin.
- Before publishing CPU/Room images, run the layer-history gate in
  `deploy/whitebox/image_storage.py` (also enforced by the manual builder and
  registry verification). CPU control images contain no SDK COPY layer; Room
  images contain one. Keep the native image tests and full source from SGLang main; pin Worldkit SDK to dev.
- On shared runtime hosts, never use global `docker system prune`, delete
  container writable layers, or stop Docker/Redis to reclaim space. Use an exact
  reviewed image inventory, preserve all container references and rollback
  images, and verify remote recovery before removing an unused image.

涉及 Worldkit / WTP / Whitebox 集成的服务部署、模型或 Room runtime 发布前，必读
[跨平台依赖、版本与协议变更自查](https://github.com/seedleap/worldkit-monorepo/blob/dev/docs/development/cross-platform-change-checklist.md)。
共享协议或发布输入变更也使用该入口，沿实际生产者和消费者核对冻结 SDK、镜像、模型、
存储和协议身份。它不替代本仓库 H3 部署决策、Whitebox 发布与验收要求，也不授权部署、
重启共享作业或更改活跃会话；源码、实际部署与端到端验收分别记录。

## SGLang 发布与跨仓库排查入口

- SGLang 负责可执行发布代码、模型策略与接口合同；[world-model-all-in-one 知识库](https://github.com/seedleap/world-model-all-in-one)负责环境目录、带时间的现场观测、排查和运维交接。发布命令以本仓库 [deploy/whitebox](deploy/whitebox/README.md) 和[发布手册](docs/docs/sglang-diffusion/zing/plans/whitebox-manual-releases.mdx)为准，不在知识库复制第二套实现；当前目录名描述 Whitebox 集成，不代表通用 SGLang 部署 API。
- 每次先确定 `prod-shanghai`、`prod-thailand` 或 `test-shanghai-copy`，再选择 region；上海正式与副本同属 `cn-shanghai`，不能仅按地域判定目标。机器与路由从[环境目录](https://github.com/seedleap/world-model-all-in-one/blob/main/wiki/platform/environments.md)定位后重新核实。
- 启动前区分同机四卡完整组与 standalone DiT 复用 Encoder/Decoder 的拓扑；后者不调用完整组 launcher，不据此重启既有角色。SGLang 发布固定 main 可达 SHA，SDK 固定 dev 可达 SHA；分别核对各仓库来源和实际运行身份。
- 只读排查复用知识库的 [Coordinator](https://github.com/seedleap/world-model-all-in-one/blob/main/.agents/skills/whitebox-coordinator-ops/SKILL.md)、[Room](https://github.com/seedleap/world-model-all-in-one/blob/main/.agents/skills/whitebox-room-ops/SKILL.md)、[SGLang](https://github.com/seedleap/world-model-all-in-one/blob/main/.agents/skills/whitebox-sglang-ops/SKILL.md) skills 与 [whiteboxctl](https://github.com/seedleap/world-model-all-in-one/blob/main/whitebox_ops/README.md)。跨仓库使用时定位本人可访问的 checkout 并显式读取对应 SKILL.md，不假设其他项目的 skill 自动加载。
- 本仓库 `deploy/whitebox/release.py` 是发布身份校验；知识库 `whitebox_ops/cli.py`、`whitebox_ops/evidence.py` 是带时间的组件观测，清单格式与覆盖范围不同。部分组件 PASS 不代表完整组发布或业务验收通过。
- 用同一 release/trace 关联源码 SHA、镜像 digest、配置摘要、实例与观测时间；分别记录已合并、已构建、已部署、身份核验和业务验收状态。排查经验与脱敏证据按知识库约定回写，历史快照不因源码合并而重标；引用文档不增加线上操作权限。

## 部署会话模型

- 用户于 2026-11-03 确认分离服务池：每个逻辑 Model Slot/Encoder–DiT–Decoder 通信组只属于一个玩家，每个 service 同时最多一个活跃会话。DiT 的 SP ranks 合起来是一个 service；Encoder/Decoder 进程可通过多个独立 service 共享模型，受进程总预算约束。这是对下述旧“部署实例”口径的细化，不允许同一个 service 并发承载多个玩家，也不恢复已停用的旧多 DiT shared API 路由。
- 用户于 2026-10-10 明确确认：后续部署统一采用单活跃会话模型，一个部署实例同时只服务一个用户，不支持并发活跃会话。
- 设计、实现、审查和验收均按此约束执行；不预先增加多会话容量配置、多 owner 状态或并发调度。已有活跃会话时拒绝第二个会话，且不得影响当前会话。
- 保留 session/generation 身份校验，以及断连、取消、故障、EOS 后释放资源和下一会话可用性的测试；这些是单活跃会话生命周期保障。

## Zing PR 性能验证

- 测试模型版本与精确原始路径见 `docs/docs/sglang-diffusion/zing/specs/MODEL_VERSIONS.md`。
  2026-11-06 已登记 1102 DMD step2000 generator；用户确认克君已在泰国部署并有配套
  serving 与 AdaLN 资产，后续测试优先复用。路径和证据见版本登记，使用前核对实际
  模型、table manifest 与 runtime；本次尚未独立完成该组合的 GPU 验收。
  原 1027 step10000 默认模型保留作对照，不将旧结果改标为新版。

- Codex 与部署同学以 `docs/docs/sglang-diffusion/zing/deployment_zing_h3.mdx` 作为
  H100/H200/B300 部署决策入口。对 Zing H3 性能或显存有影响的 PR，必须在同一 PR
  更新该文档受影响的策略、结果或验证状态；未实测时明确标为当前版本未验证，旧证据保留版本边界。
  Recipe 只保留日期/短 SHA、关键配置、主要结果与一个可回查入口，不重复归档完整实验材料。

- 后续 Zing / SGLang 相关 PR 与测试优先使用英伦维护的 benchmark，入口和指标口径见
  `benchmark/zing_h3_causal/CLIENT_BENCHMARK.md`；有限输入使用 `run_benchmark.py`
  / `bench_client.py`，连续 source 使用 `bench_streaming_source.py`，报告沿用配套生成器。
- 后续测试默认模型使用 1027 的 `global_step_010000` AR + DMD diff checkpoint，保持
  `num_frame_per_block=[3,2]`、`split_first_latent=false`；模型身份、修正后的 serving 路径
  和未验证范围见 `docs/docs/sglang-diffusion/zing/deployment_zing_h3.mdx` 的“当前测试模型”。
  不得为适配旧 runtime 静默改成 `true` 或回退旧 checkpoint；无法运行时明确报告阻塞。
- 用户于 2026-10-30 确认：后续 Zing / SGLang 测试默认使用真实请求集合
  `product-case-20261029`（`benchmark/zing_h3_causal/case_sets/product-case-20261029.json`），
  覆盖长 prompt、少图、动态换图和长会话。默认入口为 `bench_streaming_source.py`，
  不传 case 选择参数即运行该集合；保留完整 prompt、参考图顺序及逐帧更新边界。
  原四个 case 仅作为显式 `--case-set whitebox-basic` 的旧集合回归，不再作为默认。
  新集合当前仅冻结 832×480 输入；736 档仍须另行冻结并验证，不得外推。
- 性能测试必须分别覆盖 480 和 736 两档分辨率，记录实际宽×高、case 与输入身份，
  不混合统计，也不以 704、720 或其他档位代替 736。工具或部署尚不支持某档时，
  明确列为未验证及说明原因，不得用另一档外推为通过。
- 每个对 Zing H3 速度或显存有影响的 PR，都必须提供同条件 baseline 与改动后的
  整体 benchmark 指标，并分别报告 Encoder、DiT、Decoder 三部分的性能和显存，
  不能只报告直接修改的组件。每部分给出耗时中位数/P95、样本量及变化；显存给出
  常驻占用、请求峰值及变化，注明 allocated/reserved 或 driver 口径、阶段和逐 rank 数据。
  Encoder 中 Qwen、参考图 VAE、source VAE 分项标注；共进程/共卡无法独立归因时，
  明确报告共享进程或整卡范围，不将其冒称组件独占显存。缺项写明未测原因与补测计划，
  不用 0、估算或其他硬件结果代替；完整报告格式见 `benchmark/zing_h3_causal/CLIENT_BENCHMARK.md`。
  kernel/microbenchmark 只作归因补充，不能替代端到端测试。
- 重点报告 TTFF、E2E、FPS、TPOC、ICL（含中位数/P95 与样本量）；FPS 区分全程与
  首包后口径。连续 source 另关注 source-to-output 延迟、tail drain、credit wait
  和播放停顿等工具已有指标，按实际接口与采集能力报告，缺失项标为未测。
- 固定硬件、checkpoint、输入、采样、并行布局、精度、编译/offload 与计时方式，
  保留预热、重复次数、失败/未执行数、源码 SHA、命令与原始 JSON/报告。
  阶段计时遵循 `benchmark/zing_h3_causal/STAGE_TIMING.md`，明确 Host/CUDA 边界、
  逐 rank 数据和采集开销；嵌套或重叠阶段不得相加当作 E2E。

## 项目自有 Skills

SGLang 专用的 `sglang-pr-review` 在本仓库 `.agents/skills/sglang-pr-review/` 维护并提交，作为项目自有技能。跨项目复用的基础设施技能仍由 infra-skills 维护；下面的本地安装与 exclude 规则仅适用于从该仓库安装的副本。

`sglang-env-deploy`（已授权目标环境的模型三角色部署/复用/回滚）和 `room-sglang-source-replay`
（真实 Room 供帧回放集成测试）同样在本仓库 `.agents/skills/` 维护，与其消费的 CLI 和配置合同同步更新。
部署流程优先使用这两个仓库自有入口；不在 infra-skills 复制一套 SGLang 启动器或部署编排 skill。
镜像同步、跨地域迁移和 checkpoint 导出仍按需复用 infra-skills，不将这些步骤机械加入每次部署。

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
