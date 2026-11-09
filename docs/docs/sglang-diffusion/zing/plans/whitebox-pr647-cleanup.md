# PR647 review 处理清单与迁移依据

范围：源码、测试、文档与历史证据归档。未合并main，未部署、重启、删除云资源/运行数据，
也未发送GitHub review回复。SG正式发布仍main，SDK仍dev，WTP仍white_box。

## 六项评论

| 评论 | 处理 | 可核查入口 |
| --- | --- | --- |
| main发布来源 | 构建准入、配置、Docker身份、运行身份、Jenkins准入和verify统一main；main-only构建归档回归 | release.py / test_main_release_review.py |
| 跨地区模型行为 | quantum1/prompt_joint=false、调度开关、AdaLN、媒体与时长统一common；地区覆盖受allowlist限制；实际CLI解析对比 | config.py / common.json / test_main_release_review.py |
| Coordinator验收 | 四角色集合；缺失/错误source/SDK/config/role/digest拒绝 | release.py / test_main_release_review.py |
| 唯一部署入口与SDK清理 | deploy/whitebox唯一发布入口；旧renderer/launcher删除，必要资产/资源校验迁移；SDKoverlay删除 | 下方迁移表与新prepare_assets.py |
| 原始证据外置 | 11个原文件私有OSS归档并逐字节回读验证；摘要保留失败/unknown/版本边界；生成输出在Git外 | [证据索引](../../../../../benchmark/whitebox/history/20261109-evidence-index.json) |
| 新人手册与有效配置 | 两种拓扑独立模板，真实参数解析测试；main/candidate/live分列，Jenkins/Grafana只读核验 | [手册](whitebox-manual-releases.mdx)、[配置记录](whitebox-effective-config-20261109.md) |

代码检查不是上线验收。GPU模型初始化、最终自动调整、画质、480/736对照、完整镜像构建未执行；
配置记录保留具体unknown。Jenkins安装pipeline配置返回403；不把仓库新代码冒称线上已启用。

## 删除、迁移与保留的消费者盘点

盘点基于原PR提交 `bab95179a16b2eeac4d0d199e5c780a073b91c69` 的tracked文件及全仓引用。
所有旧路径作为历史事实归属该提交，不是当前可执行入口。

| 原对象 | 原消费者 | 处理及理由 |
| --- | --- | --- |
| benchmark内旧region renderer及两套region JSON | 旧模型prepare/serve、旧regional测试、EncoderPeer的region专属测试、Docker COPY与手册 | 删除；公共策略由deploy/whitebox生成。保留地区隔离/资源容量/secret边界的当前配置测试 |
| 旧模型prepare实现 | 旧serve、regional测试、Lynn说明 | 迁入prepare_assets.py并改为消费同一bundle；保留源目录不修改、已有输出拒绝、config/decoder hash、causal24fps、导出完整性、Lynn架构、AdaLN表身份与混合采样拒绝；新增bundle/checkpoint/[3,2]绑定 |
| 旧serve及regional专属测试 | 旧latency实验owner、Docker检查及历史手册 | 删除旧三卡/E2/shared拓扑和SIGUSR1恢复流程；当前四卡model_service承担角色/ready/cleanup；standalone角色池使用原生dit_worker。有效GPU归属、心跳租约和EncoderPeer回归迁移保留 |
| latency_serving_owner | 无tracked调用者；只支持旧prepare/serve合同 | 删除已失去支持的旧实验启动器，避免复制旧launcher形成第二入口。历史实验代码仍可从冻结提交恢复 |
| loopit/CLI lane现场JSON、namespace/PVC YAML、旧lane校验器 | 历史验收/发布review、校验器自身 | 完整字节私有归档后删除。它们固定旧集群/Redis/SDK，不是当前模板；现行secret/容量等不变量由bundle/镜像/真实角色回归覆盖，无需复制旧请求fixture |
| CLI Prefill覆盖目录（4个SDK源文件、install、manifest、说明） | installer与说明，当前Docker/Jenkins无调用 | SDK pin已有capture hold、server路径、producer租约重取与测试；删除全部覆盖副本，不搬到其他目录继续覆盖SDK |
| 11个部署helper/回归/依赖文件 | Docker bind/COPY、镜像合同、CI room-init、owner-retirement测试及维护说明 | 同功能迁至deploy/whitebox并改全部消费者；没有删除原生镜像门禁 |
| model-context JSON/MD与4份历史review | 文档导航、历史证据阅读；没有运行时消费者 | 原始字节已归档；保留源码/补丁/输入/模型/结果/失败边界，移出逐事件和调用栈明细；旧操作步骤不再冒充现手册 |
| Grafana JSON、SDK pin、公共config/示例、containerd-zero-size fixture | 构建/运维定义与回归 | 原样保留；JSON或日期命名不是删除理由 |

迁移的helper包括镜像源码投影、Debian镜像源、GPU归属防抢占、Room初始化探针、精确退役维护工具、
CLI图形预检及相应测试/依赖清单。Dockerfile、Jenkins配方输入、CI与相对链接同步更新。
`test_gpu_guard.py`保留禁止抢占/扩大分配，`test_encoder_peer.py`保留鉴权/同节点/隔离回归，
`test_heartbeat_contract.py`仍通过真实ASGI worker reservation接口检查拒绝/释放/下一用户。
原生image检查、非root归档与containerd历史fixture继续保留；可选真实镜像检查不伪称已执行。

## SDK覆盖层退出证据

准确pin `689e81f718e22065877de9af589e6496c712f979` 已包含CaptureHold、HTTP capture控制及
producer重新获取失效输入租约。独立干净worktree使用冻结pnpm lock安装离线依赖，执行
capture-hold（8）、raw-producer（27）、raw-session（5）共40项测试通过，耗时11.54秒。
首次在package目录运行root Vitest配置未发现用例，调整为root完整路径后成功；不把空用例算通过。
未修改SDK代码、pin或其发布分支，也未重建/部署Room。

## 证据和未来输出路径分别处理

本PR已存在并处理：原summary、完整MODEL-CONTEXT、两份现场请求、三份旧YAML、四份历史review，
共11文件以原字节归档；tar.gz为37123字节，SHA256及每个成员SHA256见索引。
原summary的stageEvidence、逐样本/分配栈等转入原始member，p2 overall failed以及Prefill版本
未GPU重验等状态保留。原先更大的视频/journal档案不是本次tar成员，其持久位置未核实，不能混称已归档。

未来输出：release plan、launch/status/epoch日志、实际bindings和Secret、prepared model view、
.whitebox-sdk构建上下文、node_modules、镜像tar/构建缓存及权重媒体均不提交Git。
本轮先检查tracked范围，没有把这些假设生成物冒称已提交。plan/preparation/state输出新增Git外路径
检查（含符号链接解析）；build继续使用临时Git archive。没有扩大共享.gitignore或删除运行数据。

## 验证记录

- 最终修复实现 `eeda44fcada6150b3c6d6a8dd71ed1c601eb9e7f`：435项通过、3项可选镜像检查跳过，65.98秒；覆盖deploy、registry、owner-retirement及Prefill/persistence/本地与Mooncake缓存。此前264项是较早子集，不累加。
- 其中Git archive→Docker build→UID10001读取/执行的可选检查随后单独启用并通过，24.60秒。
  两项仍未执行的是最终CPU和Room候选镜像的payload/import门禁；本轮未完整构建这些发布镜像。
- SDK准确pin的40项capture/producer/session测试通过，11.54秒。
- Jenkins路由23项、post生命周期4项通过；缓存与实际build shell契约独立检查。
- 新手standalone命令通过真实native parser；CI area gate与CPU inventory通过。
- 准确修复SHA的完整PR范围367个Python文件isort7.0.0、ruff0.15.1与format通过；相对链接、原JSON保留字段与归档成员hash检查通过。
- 公共策略48项、迁移后deploy191项是较早子集，不与最终435累加。GPU、完整镜像、Mintlify整站构建和现场模型最终配置未验证。

配置/API证据与最终源分支SHA分别记录；文档/单测通过不能代替上线验收。

## 新增合并前TODO：完整范围与合并门槛

以本地Git冻结main `3a9236633cc35bfe618af8caa13c16180d294717` 对候选做三点diff，
清理后仍约498文件、15.7万新增行。没有把白盒整合请求擅自改成删除运行时的最小部署PR；
也没有把范围盘点冒称全部逐行验证。完整准确统计、模块阅读路线和当前候选SHA放在PR描述中。

| 保留模块 | 审阅入口/关键边界 | 证据与剩余边界 |
| --- | --- | --- |
| 发布/构建 | deploy/whitebox，Docker、Jenkins；main与SDK dev身份、四角色verify、私有输出 | 实际CPU解析、Git archive、原生UID10001、源码阶段build；最终CPU/Room发布镜像待验 |
| Gateway/Coordinator/玩家生命周期 | coordinator.py、worker_reservation.py、player_lifecycle.py；先验证SDK/模型清理回执再归还Slot | 相应用例及历史CI；真实分布式故障/多玩家仍待验 |
| role pool | role_pool/{contracts,store,scheduler,worker,api}.py；一玩家一服务、boot/control epoch、lease与隔离 | 模块风险抽查/CPU合同；原生READY/建组/取消/下一会话未在本轮GPU执行 |
| Source/Prefill/缓存 | source_session.py、stage.py、preparation_cache.py、disk_cache.py、storage_budget.py | 持久化可选性反例已修复；当前CPU回归；缓存性能和实际恢复待GPU |
| Room/SDK/媒体/RTC | whitebox/{cli_room,render_source,room_client}及录制/Go RTC实现 | SDK准确pin40项；运行时范围大，不宣称已逐行审完所有Go/JS/媒体实现；真实浏览器验证待做 |
| CI/历史材料 | 版本化CPU清单、原生门禁、归档索引 | 保留独立fixture及原始失败；各阶段结果绑定其SHA |

main已包含Mooncake后端（#627）及CUDA流/跨rank修复（#648），不将其合并提交当成本PR独立新功能。
本PR对Mooncake主要补10行lookup遥测；主要新增缓存行为在本地容量/磁盘预算/GC和持久化回执。

独立只读review在旧候选871a2457上复现两条P1并指出两条P2：失效TAEH3默认、无L2时成功Prefill
被磁盘描述符抛错、手册观测JSON漏Coordinator、部署recipe仍写旧分支与地区差异。
随后在eeda44fc修正为真实Lynn schema准入、可选磁盘能力与明确unavailable回执，并补真实stage/telemetry和
WebSocket接续回归。独立reviewer对准确修复SHA定向复核4项均PASS，另独立复跑14项通过（最终435的子集）。
独立审查明确：仅全范围盘点与关键模块风险抽查，整体merge仍NEEDS_CHANGES，GPU/最终镜像缺口不因修复而消失。

## 尚待授权资源后的完整验收输入与命令

当前没有新增GPU/云资源或部署共享服务授权，故只准备以下矩阵，不执行。需要已有且明确授权
本轮独占的匹配硬件；不得默认借用H200产品实例、停止训练、增加GPU或清理别人租约。

| 项 | 固定输入/待补字段 |
| --- | --- |
| 源码 | baseline=main 3a9236633cc35bfe618af8caa13c16180d294717；candidate=最终PR完整SHA，各自完整镜像digest |
| 模型 | 1027 step010000 AR+DMD，checkpoint SHA256 89c3a67d77214e57d810e6a4e5554c7c919ae929db0174e6439dfadb53a1d731；[3,2]、split_first_latent=false；Lynn资产完整身份待实际清单 |
| 硬件/布局 | 同一已授权UUID组、逐rankcapability/显存；固定E1/Conditioning/SP2/D1或同一原生role-pool布局；两种拓扑结果不得混合 |
| 公共行为 | BF16、Qwen六项关闭、quantum1/prompt-joint false、resident AdaLN、W6/S0、4→2步、初始3block、mem_fraction_static=0.8；缓存和媒体条件相同 |
| 输入 | product-case-20261029；480与736分别冻结RGB/参考图/Prompt更新边界/seed42与SHA。736实际1280×736，不能只改width参数冒称输入已固定 |
| 资源/接线 | 已批准instance/owner/GPU UUID、私有bindings、真实baseline/candidate URL、role pool注册/兼容/network_domain、可用Host/磁盘预算均待回读，不填猜测值 |
| 采样/指标 | 每档预热1、正式3；TTFF/E2E/FPS全程与首包后/TPOC/ICL中位数/P95/样本数；Encoder细分Qwen/reference VAE/source VAE，DiT/Decoder时延及逐rank常驻/峰值allocated/reserved/driver显存 |

在baseline和candidate各自验证已加载相同模型、输入与配置后，分别执行：

```bash
: "${MODEL_WS:?authorized isolated model WebSocket}"
: "${PRODUCT_MEDIA_ROOT:?frozen product media directory}"
: "${CASESET_JSON:?frozen 480 or 736 case-set JSON}"
: "${RESULT_DIR:?private revision-and-shape-specific output directory}"
PYTHONPATH=python SGLANG_LIGHTWEIGHT_RUNTIME=1 \
python benchmark/zing_h3_causal/bench_streaming_source.py \
  --url "$MODEL_WS" --case-set "$CASESET_JSON" --media-root "$PRODUCT_MEDIA_ROOT" \
  --output-dir "$RESULT_DIR" --seed 42 --paced --timeout 1800 \
  --warmup-runs 1 --measured-runs 3
```

Stage计时和显存采集沿[客户端规范](../../../../../benchmark/zing_h3_causal/CLIENT_BENCHMARK.md)及
[计时边界](../../../../../benchmark/zing_h3_causal/STAGE_TIMING.md)，不把Host墙钟当CUDA阶段，不相加重叠耗时。
main不支持候选新增的一部分缓存CLI；基线配方需按该版本合法参数表达同条件，不能硬传后静默忽略。
真实测试环境还须核查身份、注册READY、首帧、输入、取消/释放、下一玩家接续，以及5个正式case
每玩家可见模型画面后180秒的浏览器/缓存/画质/联机要求。旧GPU报告保留旧SHA，不迁移成新候选结果。

这些缺口目前仍阻止把完整PR宣称为已通过整体合并验收；本次交付是已完成的源码整理、离线验证和
明确的待验清单，不自动merge或激活任何线上服务。

## 对补充逐文件评论的再次核对

p4的10项eventToPresentedSamplesMs已从摘要移出，完整原始数组仍在已核验的私有归档member中；
Git保留sampleCount=10、真实median/min/max与口径，raw_evidence增加archive/member字节数。
processMetrics、localGapFollowUp的RSS/队列/phases当前是first/last/min/max或count/mean/p95聚合，
windows为窗口数18，均不是逐样本数组，按“保留结果、不按字段名批删”保留。
benchmark README的旧confcutdir参数也已改为deploy/whitebox，与实际迁移后测试路径一致。
