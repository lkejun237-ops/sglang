# Qwen 前缀复用监控：仅通过 Prometheus / Grafana 暴露

2026-11-13：按用户要求，#687 `aae1496b9b` 已纳入 #662 的统一交付分支，供后续同次部署测试；
本页保留 prefix 指标合同，合并版的动态计时与共同部署边界见
[TTFF 观测设计](../plans/serving-ttff-breakdown.md)。合并本身不代表已部署或 FC target 已绑定。

## T051 实际采集链路补充（2026-11-12）

T051《B300/H200 Demo TTFF 与 Prompt 切换异常排查》使用的是阿里云托管 Grafana，不能与仓库的 Whitebox 示例看板或 `log.ops.loopit.me` 混为一谈。真实测试看板 UID=`sglang-whitebox-test-shanghai-copy`，目录=`whitebox-native-fc-world-model-data`，数据源=`whitebox-sh-native`。本轮重新读取确认 canSave=true，但 `world_model_conditioning_.*{pool="test-shanghai-copy"}` 查询成功且返回0条序列，尚未接通。

T051 原 Room 链路为：源进程 `/metrics` → 主机采集脚本投影 → FC relay 校验/remote_write → 托管 Prometheus → Grafana 查询。原两层过滤只接受 Room 的指定 stage。2026-11-13 review 后，本 PR 的 conditioning **直连读取**改为 `/metrics` → 认证 reader → relay 一次解析 → 待发送 batch；不经过主机投影，不修改 Room 主机代码。原因是 reader 和 relay 位于同一进程，再解析、重写一次文本没有隔离边界收益。真实 exporter 的83个样本已验证直接 relay 与旧双层路径的 remote_write bytes 一致。若部署现场只能经 Cloud Assistant 跨主机采集，需要另行确定该边界，不能把直连 reader 宣称为现成的主机命令。

接入必须逐层完成以下内容：

1. **生产者与目标身份**：#687 源进程必须是 remote conditioning 服务，重新核对实际运行 SHA、地址/端口、profile、任务前缀缓存配置。T051 记录的独立 Encoder 本地 Qwen 路径不因本 PR 自动获得前缀缓存或指标；不得制造 hit=0 来伪装已支持。
2. **认证抓取**：独立只读采集目标携带管理员 Bearer Secret，不能把 token 放到命令参数、日志或看板；核对 HTTP状态与 Prometheus格式。不能直接借用 Room 的 endpoint/proof/身份。
3. **relay 合同**：为已经验证身份的 conditioning 目标放行下表精确 family/标签；沿现有 sglang 角色上报，以 instance 区分具体端点。保留目标证明、新鲜度、数值有限性、批次大小及标签碰撞检查。原 Room stage 过滤不得放宽。无数据不补0，不把失败/过期快照重新盖时间戳发送。
4. **部署与权限**：构建、部署新的采集代码和目标配置，必要时更新精确 Cloud Assistant command ARN。T051 最后记录为 `whitebox-test-shanghai-copy-observability` 的 RAM 修改被 `ram:CreatePolicyVersion` 拒绝，等待管理员；这不等于 #687 的新采集目标已经获准或已创建。不要覆盖 T051 正在准备的采集包，也不修改正式环境或恢复报警通知。
5. **数据与看板验收**：先读取源端计数增量，再验证 relay 未丢失维度、remote_write成功以及数据源可查询到近期真实样本，最后查看面板。保存看板成功不等于有数据；低流量先用原始Counter验证，再看rate。

| 精确 metric family（统一前缀 world_model_conditioning_） | 允许的源标签 |
| --- | --- |
| encodes_total | prefix_outcome=hit/miss/unknown |
| tokens_total | kind=reused/encoded/total |
| results_total | source=encoding/joined/result_cache |
| reused_tokens_bucket/sum/count | bucket含le，sum/count无标签 |
| encode_duration_seconds_bucket/sum/count | prefix_outcome；bucket另含le |
| queue_duration_seconds_bucket/sum/count | bucket含le，sum/count无标签 |

不需要转发 `_created`；不把 prompt、session、event、token文本或用户身份作为标签。固定身份为 business_line_id=world-model-data、region=cn-shanghai、pool/deployment_id=test-shanghai-copy、service=sglang，instance 来自已验证目标，不接受源端伪造这些字段。三项语义标签缺失会合并不同系列，应拒绝而非默默删除。

`deploy/whitebox/prepare_conditioning_grafana.py` 从真实看板 GET 导出生成更新请求，保留既有 UID、目录、版本和全部原图表，在末尾追加本功能8图表，转换为托管数据源和实际标签体系。不创建独立看板，也不修改业务 API。重复添加会拒绝；保留版本并 overwrite=false，防止覆盖并发编辑。本轮实际导出含46个原面板，已核对全部保留；8条查询经真实数据源执行均成功但无数据。未向 Grafana 写入更新。

### 本轮源码补齐范围与部署边界

用户确认先完成代码、部署之后再做。本轮不改变推理行为，也不为独立 Encoder 新增缓存：

- `conditioning_metric_contract.py` 定义精确 family、语义标签枚举及 exporter 的 histogram bucket 集合，由 relay 使用；reader复用其中的目标范围校验。只接受 exporter 原生标签，身份标签由已验证目标注入；额外标签直接拒绝。未知 family（包括 `_created`）忽略，支持的 family 缺标签、负值、非有限数、显式样本时间戳则拒绝本批。
- `prepare_conditioning_collector.py` 离线构建 T051 PR #662 r3 包的增量。基线 archive SHA256=`a80254563127d1bc76448a390b96ae4984ca7b7100e68371c88cf6056682f402`；同时校验待修改 relay 源码，不匹配立即失败。替换 relay 和现有采集周期入口，新增合同与认证 reader；测试池身份查询只增加实际 DiT 管理端口43310，所有原有来源校验保留。Room host及其他文件逐字节保留。schedule与pool source也逐文件校验SHA256。基线变化必须重新审查并更新 pin，不能绕过 checksum。
- relay 仅对上海测试目标 `service=sglang` 接收 conditioning family，原 Room stage 规则、identity/proof、新鲜度、批次限制和碰撞检测保持。正式环境不能自动接收新 family。
- `conditioning_metric_source.prepare_batch(proof, token_file)` 是同步 FC 入口：先校验当前 proof，再读 Secret，通过已确认私网 IP 的 `/metrics` 抓取。复用采集包及CI已有的 `aiohttp==3.14.5`，`ClientTimeout(total=3)` 覆盖连接、TLS、响应头和全部正文；流式读取上限1 MiB，禁代理/重定向/自动解压。超时取消读取并关闭连接，返回固定错误类别，不留后台抓取线程；同步入口应在没有运行中 asyncio loop 的 FC 调用栈使用。记录实际读取结束时间，原文直接交 relay 生成未发送 batch。不自动 remote_write、不重试、不补0。返回零系列时调用者跳过发送；发送前必须用当前 proof 再执行 `assert_sendable`。

该 reader 已挂入 `copy_production.run_round` 的现有租约采集周期：复用本轮 `CopySources.prove_pool()` 的新鲜注册/健康证据，只接受其中唯一匹配的 Encoder target；由 `CONDITIONING_METRICS_TARGET` 与私密 `CONDITIONING_METRICS_TOKEN` 绑定。两者未配置时禁用，不影响Room；证据过期、目标不匹配或抓取失败时标记unknown，不补0、不发送不明来源样本。发送复用原fenced sink，不增加timer或重试。此实现限定上海副本测试环境，不能用静态地址代替实时身份校验。本轮不生成或修改Cloud Assistant主机命令。

离线准备入口（不会调用云 API）：

```bash
python deploy/whitebox/prepare_conditioning_collector.py --baseline /path/to/pr662-r3/collector.zip --output /path/to/new-output
python deploy/whitebox/prepare_conditioning_grafana.py --export /path/to/dashboard-export.json --output /path/to/dashboard-update.json
uv run --with pytest --with prometheus-client --with aiohttp==3.14.5 python -m pytest deploy/whitebox/test_conditioning_collector.py -q
```

核心测试不再需要私有基线 archive 或环境变量。仓库 `fixtures/conditioning-relay-pr662-r3.py.txt` 保存精确 SHA 校验的真实 relay 源码快照，经过与发布相同的 `patched_sources` 后执行，fixture缺失或hash不符直接失败；它是冻结测试输入，不是第二个可编辑实现。`.gitattributes` 保护跨平台换行字节；来源与更新规则在同目录 README。既有 zing-ci 的 deploy/whitebox 测试会自动覆盖这些用例。完整发布 archive 构建/逐文件保留检查单独执行，不拿小型fixture打包测试冒充发布包验收。

测试使用真实 `ConditioningMetrics` exporter 生成 hit/miss/unknown 与三种结果来源，逐样本核对 relay 后的名称、标签和值；覆盖 Room 不变、正式环境隔离、过期proof、额外标签、碰撞、无数据、真实本地HTTP Bearer/401/重定向/HTML拒绝，以及超大正文。慢响应回归用250ms测试deadline覆盖响应头逐字节返回、正文持续分段返回、完全无响应，要求1.2秒内返回超时且服务器观察到连接断开；默认配置仍为3秒。

截至部署前，采集代码与测试看板已发布初版，FC更新权限已验证；仍待新Encoder目标部署/身份核验、Secret绑定、真实数据入库验收。不能因为离线 reader 能读取或 CPU测试通过就标成线上可观测。

PR #687 在已合并的前缀 KV/输出特征复用之上增加监控。2026-11-12 最终范围为：**只向 Grafana 提供聚合指标，不向业务 API 返回命中字段，不推送就绪事件。** 当前 PR 已撤去此前的 HTTP/NCCL metadata 扩展、Python 结果包装、WebSocket 通知与私有 IPC、视频块统计及 Gateway 转发改动。文档路径保留以兼容已有 review 链接。

缓存算法、失效条件和历史数值差异继续见 [前缀缓存设计](conditioning-prefix-cache.zh-CN.md)。本次不改变 Qwen 模型计算、精度、KV 生命周期或业务响应格式。

## 用户怎么用

Prometheus 抓取 conditioning 服务 `GET /metrics`。沿用服务的 Bearer 认证，凭证配置必须有 `admin: true`：无凭证401、普通租户403、管理员200。指标汇总本实例所有租户，因此不对普通租户开放；registry 随服务实例创建，进程重启后累计值归零。已有业务 HTTP/NCCL 编码请求仍返回原两张量，业务 WebSocket 沿用 main 的格式。

仓库通用模板 `benchmark/whitebox/grafana-dashboard.json` 保留 **Whitebox playtest service** 看板 UID=`whitebox-playtest`、原6个图表、数据源和 cluster_id/service_id 筛选，在底部增加8个图表。它不是 T051 实际线上看板；该环境用前述转换工具更新既有 ARMS 看板，不另外导入此模板创建新看板。通用模板的 cluster/service 发现查询纳入 conditioning 计数器，供直接 Prometheus 抓取的部署使用。

以下 Linux Prometheus 片段合入现有 scrape_configs，替换为真实服务地址及已有 cluster/service 标识；凭据文件内是已配置的 conditioning 管理员 token，不提交凭据。若目标已由平台发现/抓取，更新既有抓取配置，避免同一实例被重复采集。

```yaml
scrape_configs:
  - job_name: zing-conditioning
    metrics_path: /metrics
    authorization:
      type: Bearer
      credentials_file: /run/secrets/conditioning_metrics_token
    static_configs:
      - targets: ["CONDITIONING_HOST:PORT"]
        labels:
          cluster_id: "EXISTING_CLUSTER_ID"
          service_id: "CONDITIONING_SERVICE_ID"
```

cluster_id/service_id 必须与原看板的 target labels 一致。跨主机按部署网络策略配置 TLS 或内网代理。此次指标来自 remote conditioning 服务；只抓 Gateway/视频 API 看不到 encoder 指标。独立 Encoder 的本地 Qwen 路径不因本 PR 自动获得这些统计。

## 指标与分母

以下名称均以 `world_model_conditioning_` 开头，Histogram 自动附加 `_bucket/_sum/_count`。

| 指标后缀 | 类型/标签 | 口径 |
| --- | --- | --- |
| `encodes_total` | Counter；prefix_outcome=hit/miss/unknown | 成功 encoder 任务数，每任务一次；unknown 不进入命中率分母 |
| `tokens_total` | Counter；kind=reused/encoded/total | 已知统计的 token 累计量，包含模板和视觉展开 |
| `reused_tokens` | Histogram | 已知统计每任务复用长度，包含 miss 的0 |
| `encode_duration_seconds` | Histogram；prefix_outcome | executor 提交至结果返回，含 encoder 内预处理、HTTP CPU staging 和返回开销，不含队列等待与队列前请求校验/上传 |
| `queue_duration_seconds` | Histogram | 最终成功任务入队至 executor 提交 |
| `results_total` | Counter；source=encoding/joined/result_cache | 成功 HTTP 响应/NCCL 传输流程完成后的交付次数 |

hit 表示 reused_tokens>0；完全特征复用或仅缩短可 encoded_tokens=0，但仍计一次成功 encoder 任务，因此 encodes 不等于 GPU forward 次数。没有统计的 encoder/未启用任务缓存记 unknown，不能伪装为 miss。预取消费、joined 等待者与完整结果缓存不会重复增加编码/token 计数；交付次数单独按来源记录。

失败任务和入队后未执行的取消任务不进入成功任务计数或耗时直方图。请求已断开但实际编码成功完成仍计一次。完整结果交付失败不增加 results_total。指标固定枚举标签，不使用 session_id/event_id/task_handle/prompt/租户/模型路径。

面板的任务命中率为 hit/(hit+miss)，token 复用比例为 reused/total；使用 rate 并按 job/instance 聚合，分母为0时无数据。复用长度与耗时 P95 为直方图近似值，不提供逐请求查询。unknown 任务速率单独显示。

**encode_duration_seconds 不是纯 Qwen GPU 时间**，也不是 prompt 更新至首帧的切换延迟。不同 prompt/reference 长度的 hit/miss 耗时不能直接推导加速比；整体加速仍需同条件 A/B。抓取健康度 `up` 不证明模型可用。

## 实现与必要性

唯一 encoder executor 完成编码后立即复制 cache 的标量计数，作为 `_forward` 私有返回值的第二项交给 EncodeQueue。队列在设置共享 future 前记录一次指标，只把原 tensor dict 交给等待者；统计不进入业务响应、缓存返回对象、SP receipt 或视频块。该快照避免 task release/下一任务清空缓存后读取到错误计数。

`ConditioningService` 拥有 `ConditioningMetrics` 和独立 registry；`metrics.py` 集中有限标签及聚合定义。`/metrics` 直接导出本服务 registry。只需要服务内部计数和监控抓取，因此移除了业务通知、跨进程观测通道、结果包装/校验与下游传递链。不新增 GPU 同步、张量拷贝或 collective；本次未测热路径开销，不宣称性能零变化。

核查基线 main `df057b0ec3cab9759c2650f8d55d8891b75392d0` 的 `runtime/realtime/critical_path_metrics.py`、`world_gateway_metrics.py` 和已有 Whitebox 看板：本仓库已采用 prometheus_client、有限标签和 `/metrics`，本次沿用该机制。此处是 Diffusion conditioning 私有服务，不声称复用 SRT 文本 RadixCache 调度器，也不增加 OpenAI usage 字段。上游 reviewer 应关注 registry 归属、成功任务分母、认证范围、指标额外开销与两种传输协议不变。

推荐阅读：本设计 → `conditioning/service.py:EncodeQueue._forward/_run`（内部快照与唯一计数点）→ `conditioning/metrics.py`（指标）→ `ConditioningService.__call__/_encode`（抓取鉴权与交付计数）→ `benchmark/whitebox/grafana-dashboard.json`（聚合查询）→ `test_qwen_prefix.py` 与 `test_nccl.py`、注册 `test_conditioning_rpc_core.py`（集成验证）。

采集扩展的阅读入口为 `deploy/whitebox/conditioning_metric_contract.py`（指标与目标合同）→ `prepare_conditioning_collector.py:patched_sources`（relay、现有周期接线与测试池端口，Room行为保留）→ `conditioning_metric_source.py:prepare_batch/_read_metrics`（证明校验、认证抓取、总deadline和未发送batch）→ `test_conditioning_collector.py`。依赖限于标准库和原采集包已有的aiohttp，不从推理runtime导入模型；测试用文件加载方式读取真实exporter，避免加载torch。这里采用checksum限定的离线构建，沿用#662的包准备方式；代价是collector基线更新时需重新审查，不能作为任意版本自动补丁。测试环境专用adapter不视为上游通用部署接口。

## 验证方案与边界

- 真实 uvicorn/TCP + 正式客户端，实际 encode_ids/QwenPrefixCache，LM 为 CPU 因果 oracle：冷启动、替换中部、完全命中、缩短、修改首 token，期望 hit=3/miss=2、reused/encoded/total=8/10/18。快照后故意覆盖 cache 计数为99，抓取结果仍应正确；返回值仍是原两张量 dict。
- 实际 HTTP `/metrics` 抓取并解析 Prometheus 文本，核对401/403/200及固定标签；不把 CPU oracle 当作完整 Qwen 权重验证。
- HTTP framing/NCCL metadata 都不得含 qwen_prefix/result_source；HTTP 缓存只执行2次任务而交付3次，NCCL路径执行3次；8个并发等待者只执行1次，交付1 encoding/7 joined，后续result_cache=1。
- 全量 conditioning 默认 CPU 回归及 Gateway 验证撤回业务协议扩展后的兼容性。当前最终实现结果见下文；之前业务 API/通知版本的测试数不作为本版本验收。
- 看板 JSON 检查 UID/原6个图表/数据源保持、唯一ID、仅8个conditioning图表及无就绪通知指标。早期访问另一 Grafana 超时；本轮已找到真实 ARMS 实例并完成只读 PromQL 检查，实际看板尚未写入，不部署或重启服务。

GPU完整权重、真实NCCL collective、视频画质与性能、生产抓取及Grafana线上展示不在CPU验证覆盖范围内。

## 当前实现实测

指标 runtime 实现 `ad82622ba63b60cd27aebd651d48c7b0fe6b3bf7` 的完整 conditioning 默认 CPU 套件加 Gateway：**2250 passed, 7 warnings, 59 subtests passed in 459.96s (0:07:39)**，JUnit确认0失败、0错误、0跳过。环境为 WSL Linux / Python3.12 / torch CPU，单线程数学库、HF离线。真实 TCP `/metrics` 抓取与鉴权、Qwen cache快照被覆盖后的计数正确、HTTP/NCCL响应无新增字段、合并请求与结果缓存计数均通过。日志/JUnit：`D:/sglang687-grafana-only-20261112/full.log`、`full.xml`。该次8文件范围之后，本轮只追加部署准备工具、合同、reader及对应测试/文档，没有修改业务协议、worker、预取、视频或模型计算。上述旧套件不替代本轮 collector 集成测试，也不代表线上部署已验收。

2026-11-12 旧双层实现的collector/reader/看板测试为28项通过；这是历史证据，不代表下面简化版的验证。真实Grafana只读检查保留于 `D:/sglang687-native-grafana-20261112/` 的 `before.json`、`query-before.json`、`validation.json`；实际8条PromQL查询可执行但0系列，尚未完成真实数据入库验收。

2026-11-13 简化版在未设置任何基线包环境变量时：**33 passed in 5.08s，0失败/错误/跳过**（Windows / Python3.13 / CPU；aiohttp固定3.14.5）。JUnit：`D:/sglang687-simplify-20261113/tests.xml`。包括3项总deadline/断连测试，解决旧socket timeout可被持续小流量绕过的问题。完整真实发布包也单独构建并逐文件验证：仅relay变化，新增合同和reader，包含Room host在内的其余**4034项字节一致**；证据 `archive-verification.json` 和 `collector/build.json`。没有部署、GPU测试或线上写入。

Ruff（含import排序）、format和diff检查通过。全仓 `check_registered_tests.py` 仍报告 `test/registered/kernels/ops/test_fp8_scaled_mm_routing.py` 缺少CI注册；已核对该文件与PR基线完全一致，本轮未修改。该全仓检查不能记为通过。

## 独立 Encoder 接线方案（2026-11-13，开发中）

现有三角色部署的 Encoder 已加载 Qwen，却固定使用无 prefix 的 LocalConditioner。
为使同次部署可以测 prefix，独立 Encoder 将复用既有 QwenPrefixCache 与原 encode_condition，
不另启一份 Qwen 模型或引入远端 RPC。原 remote conditioning 路径保持原样。

启动配置：`SGLANG_ZING_QWEN_PREFIX_BYTES` 为每 service、当前会话的 KV 上限，默认 0 关闭；
本轮候选计划明确设为 4294967296（4 GiB），实际显存和收益尚未测量。
`SGLANG_ZING_CONDITIONING_METRICS_TOKEN_FILE` 为独立只读 Bearer 文件；设置后通过已有管理
HTTP 端口 `/metrics` 暴露相同 family。文件内容不入配置清单、日志或源码。
详细 TTFF 仍由 detailed_timing 控制，与 prefix 预算/聚合指标独立。

所有权：SessionPrefixConditioner 归每个 EncoderConditioning，访问沿已有单线程 executor；
不同 service 不共享 KV，EOS/断连/会话释放与 prefill 完成沿 dispose 清空。完整条件结果缓存
沿用原有 LocalConditioner；其命中仅计 result_cache，不伪造 prefix hit 或编码次数。
换图用真实像素与顺序的 media_identity 使缓存失效；异常清除已可能原地修改的 KV。
没有新增 CUDA 同步；本地 encode_duration 是进入 LocalConditioner 至返回的 Host 墙钟，
包含媒体摘要、CPU 结果缓存处理，不能当纯 Qwen CUDA 时间，也不能直接与 remote RPC 时延
混合比较。独立 Encoder 未测到独立 condition 队列边界，不向 queue Histogram 填零样本。
通过 instance 区分部署拓扑；远端结果语义/管理鉴权不变。

验证计划：生产 EncoderConditioning 入口的切词复用、同条件命中、换图失效、失败后恢复、
dispose 后首次全量、两个 service 隔离及 /metrics 未授权拒绝；默认关闭保持全量路径。
合并前已有测试不能替代这次接线验证。GPU 480/736 同条件输出、三角色耗时与显存、完整
Room/首 RGB/切换以及 Grafana 入库均待验收。上游 review 关注此私有 Zing service 的缓存
生命周期和显存预算，而非将本地部署结构作为通用 SGLang 默认；复用本仓库已有 QwenPrefixCache、
LocalConditioner 与 ConditioningMetrics，不引入第二份 Qwen 算法。

接线首次 CPU 结果：`21cc9ca66627af1edc62d408f0323fe3ed88a5be` 在上海 H100 主机的
无 GPU 隔离容器（2 CPU / 8 GiB，Python 3.12，原生依赖镜像 ID
`sha256:e132494ca7a1fb21b0655aeca65a13b1e3bffa2fb8696330678f59348b610ea1`）运行仓库 CPU
harness，prefix/Encoder API/inference timing 共 **94 passed**。这是 CPU 张量与协议证据，
未占 GPU、不证明真实模型性能。后续自审增加进程退出清理与空凭据拒绝，待最新提交复验。
本机 WSL 加载 PyTorch 异常退出、Windows 缺 Linux resource/orjson 的尝试未计为通过。

### 2026-11-13 采集周期接线验证

实际冻结的collector schedule进入现有run_round后，已验证prefix样本写入且Room流程继续；过期/缺失/多owner证明拒绝抓取，未配置禁用。连同真实HTTP鉴权、deadline与relay检查共35项通过（Windows Python，Prometheus客户端，不加载推理模型）。这不替代FC现场连通和线上样本验收。

### 2026-11-13 实际采集配置与部署边界

周期接线包已发布到 `whitebox-test-shanghai-copy-observability`，下载SHA256与本地包一致。正式环境未改。新的主机只读命令分别验证controls complete=true和模型池registry/capacity/controls complete=true，当前三角色health均200。模型池明确选择 `whitebox-copy-model1027-480`；不能沿用Coordinator历史default路由，它包含过期旧组并导致整个身份证明失败。`patch_test_timing_collector.registered_host_commands`保留两个命令各自的冻结依赖子集、实例和身份校验，只更换测试容器名称、DiT端口和这一路由；不放宽过期校验、不清理其他进程。

FC角色仍需要管理员替换两个精确的只读命令ARN，当前操作者调用ram:CreatePolicyVersion返回NoPermission。看板和FC代码发布不能视为采集入库完成。新部署候选b6d6945ee5b5的prefix/EncoderAPI/分时CPU回归97项通过；尚未启动GPU对照。现场Room/Gateway已由其他部署更新为main32d61c852997，缺少本PR详细计时源码，保留其新改动后才能更新Room；环境变量为1不能代替代码接线。此边界为现场观测，不改变历史测试证据。
