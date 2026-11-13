# Managed Player 并行准备验证（2026-11-12）

基线为当轮最新 `origin/main`：`e9bf5d137f34a03abab55093038b4ba5ee7942a8`；
最终实现提交为 `792c73e37bfdab84e3ea9e997d8d12e964e9806a`。
初轮完整CPU范围在 `f5c79fd2a9dfc3923ba0472a6293686496f90767` 通过；最终取消竞争修正后，
重跑下列 parallel_startup、inference_control、source_conditions、control_prefill 四文件，235项通过。
其他未受该修正影响的文件保留原回执，未重复执行。
本记录只描述本地验证；未构建/部署镜像、未创建真实玩家、未申请GPU或影响训练。

## 行为与覆盖

旧 main 在 CLI 阻塞时等不到 `source.prepare`（新增调度用例先 red）。候选实现
在同样条件下已发送 prepare，CLI 仍未返回；manual 推理仍未开启，未产生模型画面。
18 项专项覆盖可信入口、缺图回退、真实 World 条件匹配/不匹配、真实 F0、
同一父绑定/子连接接续、原子 journal 提交、WS/RTC 就绪、CLI 失败/取消、
关闭前取消、准备失败/超时、清理未知和显式 Prefill 接管。

测试复用真实 SourceInference、SourceSession 调度、PlayStartup、ModelBinding、
SourceConditions 和 RenderJournal，模型协议端是受控 peer；这不等于真实GPU或浏览器验收。

## 最终 CPU 回归

Python 3.14/macOS，`SGLANG_LIGHTWEIGHT_RUNTIME=1`、`PYTHONPATH=python`。
对下表每个文件单独运行 pytest，与 `scripts/ci/zing/run_area_tests.py` 的进程隔离方式一致。
`--confcutdir` 指向 realtime 目录，以使用轻量 Room 检查，不加载父级 native runtime conftest。

| 文件 | 结果 |
| --- | --- |
| `test_whitebox_parallel_startup.py` | 18 passed in 0.23s |
| `test_whitebox_startup.py` | 48 passed in 3.36s |
| `test_whitebox_inference_control.py` | 53 passed in 21.81s |
| `test_whitebox_source_conditions.py` | 151 passed in 71.98s (0:01:11) |
| `test_whitebox_model_binding.py` | 6 passed, 1 skipped in 0.02s |
| `test_whitebox_render_lifecycle.py` | 50 passed in 78.03s (0:01:18) |
| `test_whitebox_render_contract.py` | 38 passed, 1 deselected in 12.75s |
| `test_whitebox_player_control.py` | 105 passed in 3.27s |
| `test_control_prefill.py` | 13 passed in 0.72s |
| `test_whitebox_startup_prewarm.py` | 10 passed in 0.17s |
| `test_whitebox_prewarm_plan.py` | 41 passed in 0.50s |

两轮按用例去重覆盖 **533 passed、1 skipped、1 deselected**。isort、Ruff、git diff --check 通过。
Linux `/proc` 描述符检查按既有平台条件 skip；
`test_worker_prompt_boundary_includes_admitted_prefetch` 依赖本机缺少的 zmq/Torch
原生运行时，明确以 `-k 'not test_worker_prompt_boundary_includes_admitted_prefetch'` 排除，
保留在原有完整 CI 中，没有改测试为通过。

历史失败保留：早期多文件共用单进程执行在玩家控制用例出现 `KeyError` 后源会话未创建，
371 项通过后中止；该根因未单独证实。最终玩家文件独立 105 项通过，新增专项与玩家文件
合跑 121 项通过，render_contract 与同一玩家用例合跑 39 项通过（另1项依赖排除）。
这些结果不包装成任意单进程混跑或全仓测试通过。首轮错误测试路径、deselect 路径和
stdin 诊断不作为验证成功证据。

## 最终取消竞争修正

新增可复现red：候选被主动退役时，等待它的场景消费者也收到 CancelledError。
现在区分“消费者自己取消”与“共享候选退役”：前者照常传播且不取消共享准备；
后者等待原清理确认，令候选失效并允许消费者继续。清理未知仍失败，不提前借用模型。
两种方向均有回归；对应专项最终18项通过。

## 后续验收边界

实际收益取决于预置条件是否与 World 第一组真实条件一致、父 Slot 准入和 CLI 准备时间。
本轮未验证 GPU 480/736 的条件准备、TTFF、角色分项耗时、峰值显存、画质、输入延迟、
真实 RTC 呈现或多玩家 E2E；未改变模型配方，也不承诺固定节省秒数。
发布仍需完整镜像和准确依赖，代码已完成不等于线上已生效。

## 2026-11-13 部署前补充

真实SDK给参考图使用reference-0等上传ID，World使用按序号的Picture标签。部署前补了有序槽位映射：保持原请求、原指纹和图像字节，依然拒绝错序、槽位间隙或内容变化。新增red已复现；专项现为22项通过，相关推理/条件204项通过。

隔离镜像候选source为207f44eae43ffb02a4226acf8241f35f12896eb2，SDK仍31ea27da。首次a631构建因浏览器公网下载超时（124）未完成Room；已核验同版本本机缓存并补显式构建输入，未跳过原生检查。当前仅部署准备完成，尚未切换或做真实GPU/浏览器验收；不将旧531/533回执当作新完整发布通过。

## 2026-11-13 副本切换与 PR 审查修复

15:23（上海）test-shanghai-copy 的 Room/Gateway 已完整切换至
`5ff10650577a8d05e8a0dd8f6d07c2fba327a66d`，SDK 为 dev 可达的
`d67107d90421d3645abd83bd2628874487e93268`。完整源码/SDK/镜像层审计、
原生 CLI、健康和公网身份回读通过。15:29 公网仍为 used0/free0/total0，兼容 Encoder
由另一任务恢复、仍在排队，本轮未新建玩家。详见
[脱敏部署证据](https://github.com/seedleap/world-model-all-in-one/blob/main/evidence/2026-11-13-managed-parallel-startup-test-deploy.json)。
这是组件部署检查，不是正式效果验收；480/736 GPU、TTFF、RTC、画质/显存及真实官网
E2E 均未完成，不能报告提速或无退化。

PR #708 审查指出采集器默认 urllib 重定向可能转发官网 Authorization。使用内存 HTTP
响应和假令牌在旧代码复现跨域 301 后目标请求仍含认证头；没有使用真实凭据或外部服务器。
修复对初始认证请求及每次重定向按 HTTPS 官网 origin（scheme/host/effective port）
校验，跨域、降级和嵌入凭据的 URL 拒绝；同源 HTTPS 正常跳转保留认证，无认证 CDN
请求可在 HTTPS 间跳转。采集器与图片准备工具都验证配置来源，拒绝不可信 origin。
49 项专项覆盖 301/302/303/307/308、跳转链、伪造子域/端口、直接外域请求、正常跳转、
CDN gzip 及拒绝响应关闭，并纳入 AREA CI。该修复只涉及验证工具，不改变运行时模型调度；
此前 5ff 镜像不冒充包含后续工具修复，新增 head 的 CI 必须独立回读。
