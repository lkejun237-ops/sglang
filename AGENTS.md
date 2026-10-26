# Repository instructions

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
- 使用前重新核验 GPU 占用和作业归属；机器清单不代表空闲，也不授权停止其他人的任务。

## 部署会话模型

- 用户于 2026-10-10 明确确认：后续部署统一采用单活跃会话模型，一个部署实例同时只服务一个用户，不支持并发活跃会话。
- 设计、实现、审查和验收均按此约束执行；不预先增加多会话容量配置、多 owner 状态或并发调度。已有活跃会话时拒绝第二个会话，且不得影响当前会话。
- 保留 session/generation 身份校验，以及断连、取消、故障、EOS 后释放资源和下一会话可用性的测试；这些是单活跃会话生命周期保障。

## Zing PR 性能验证

- Codex 与部署同学以 `docs/docs/sglang-diffusion/zing/deployment_zing_h3.mdx` 作为
  H100/H200/B300 部署决策入口。对 Zing H3 性能或显存有影响的 PR，必须在同一 PR
  更新该文档受影响的策略、结果或验证状态；未实测时明确标为当前版本未验证，旧证据保留版本边界。
  Recipe 只保留日期/短 SHA、关键配置、主要结果与一个可回查入口，不重复归档完整实验材料。

- 后续 Zing / SGLang 相关 PR 与测试优先使用英伦维护的 benchmark，入口和指标口径见
  `benchmark/zing_h3_causal/CLIENT_BENCHMARK.md`；有限输入使用 `run_benchmark.py`
  / `bench_client.py`，连续 source 使用 `bench_streaming_source.py`，报告沿用配套生成器。
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
