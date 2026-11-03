# API Prompt / Reference 输入限制

连续 source API 在 init 与 prompt/reference 更新时检查输入；本 PR 仅修改 API 准入、启动配置及其测试。

## 使用方式

默认纯 prompt 最多40000 tokenizer token，reference最多21张，估计组合预算50000 token。
图数复用既有参数；采用历史H200两卡的25图专项预算时，在原有启动命令增加：

```bash
--zing-prompt-max-tokens 40000 \
--zing-reference-max-images 25 \
--zing-condition-max-tokens 50000
```

三项同时满足：`T ≤ 40000`、`R ≤ 配置图数上限`、`T + 1700 × R + 500 ≤ 50000`。
T 用导出模型的CPU tokenizer计纯prompt，不添加special tokens、不截断。1700是每图统一估算，500是固定预留。
组合上限仅约束API估计预算，不表示校验了实际编码后的完整condition行数；缩小图片不改变每图1700的估计值。
组合超限时提示缩短prompt或减少参考图，不建议通过缩小图片降低这项预算。
例如40k文本+5图估算49k，通过；10k+25图估算53k，拒绝。25600的20FPS推荐线不参与拒绝。
`session.ready.input_budget`公布实际配置及系数。以上不扩容Qwen或KV池，服务已有容量校验继续生效。

## API 内部的检查时机

init先检查预算，再验证图片；更新在准备新参考图、提交prefetch或派发命令前检查。
仅加载CPU tokenizer文件，使用既有有界CPU executor，tokenizer按模型路径缓存；缺少文件时明确失败，不按字符估算。
超限init拒绝会话；超限更新发送带event_id的非致命英文错误，保留当前生成，并沿用较新事件淘汰旧候选的规则。

显式references（含空数组）代表完整参考图快照，直接用其张数。纯prompt更新不携带新图数：
API保存已派发参考图数量的上界，初始化为init图数；派发reference更新时先取新旧图数较大值。
既有输出元数据的event_id确认最新reference更新已经生效后，才降低至该图数。
若更新失败、被覆盖或尚未输出，纯prompt更新继续按较大值检查，可能保守拒绝；可提交显式完整references校验新组合。
这两个计数状态仅属于API会话，不复制图片、不增加队列或worker协议字段。

后端source_session、packer、stage及缓存流程与本次同步的main `772927dc45a0`一致。
不新增实际condition行数检查，不改变KV池、缓存恢复、prefetch消费、模型计算或采样路径。

## 验证和适用范围

计划覆盖文本/图数/组合等于和超出边界、真实CPU tokenizer不截断、CLI覆盖、init早拒绝、
更新拒绝与恢复、显式清空、纯prompt更新、参考图更新未输出/失败/已输出的计数、原缓存与prefill回归。
测试复用既有source adapter文件及现有CI入口，不新增CI清单或执行规则。

此前 `d42c909` 在固定Python3.12 / Torch2.14.1 CPU环境，source adapter、prompt switch cache、preparation cache、prefill API
四份回归共368项通过，包含API边界、真实tokenizer、动态图数确认和旧缓存路径。Ruff/isort/format与受改文档独立编译通过。
合并测试时初次collection因本地文本替换影响import缩进而失败，修正后整组通过；没有修改生产运行方式。
本轮未本地重跑完整conditioning两轮，最新完整回归由PR CI继续执行；旧版后端检查结果不作为当前API方案证据。
2026-11-03修复review指出的完成回调初始化假设：仅在存在预算状态时处理reference计数，生命周期回调仍可独立使用。
保留既有zero-RGB/续跑/取消测试，修复前本地复现AttributeError；修复并同步main后，该文件7项回归通过。
合并保留main的 `_take_source_input()` 拆分，把预算计数接回 `prepare_next_request()`，没有修改后端消费路径。
同日对 `d42c909` 的部署tokenizer与正式WebSocket路由12项CPU验收通过；该结果不替代合并后版本的完整CI。
未运行新的GPU推理；480/736端到端延迟、显存和最大输入容量未验证。
历史H200/W6/S3/split-first/Graph0预算是本项目的估算策略，不代表当前默认部署的显存或20FPS保证。

## Review 依据与阅读顺序

本分支起点为`e4d8047acd9ee854cd023b677674f91ef7e78179`，复用PipelineConfig参数、ReferenceLimits、
ConditionError及BoundedExecutor。已核查官方上游
[SRT Engine@26fde018](https://github.com/sgl-project/sglang/blob/26fde018e53840238624f7810b69ca641531713c/python/sglang/srt/entrypoints/engine.py#L1394)
使用scheduler提供的max_req_input_len进行请求限制；本PR的40k/50k及1700+500属于本项目配置，不是上游通用标准。

先看本文和预算页，再看ZingH3SourceStreamAdapter.on_init/_ingest_prompt_event、
prepare_next_request/on_chunk_complete的图数计数，最后看ZingInputBudget与source adapter测试。
Review重点：tokenizer是否一致、检查是否早于派发、失败更新是否错误降低图数、后端文件是否保持main原样。
