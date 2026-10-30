# Prompt 切换支持不拆首块的 3/2 分块

基线：seedleap/sglang main `480b48c853`（2026-10-30）。

## 使用与行为

模型 runtime 配置 `num_frame_per_block: [3, 2]`、`split_first_latent: false`。
生成从头到尾按 3/2 循环；prompt 更新仍等当前五 latent 周期完成，
保留最新五个生成 latent，按两个 target block（3/2）重建新 prompt 的 KV。
切换后第一个生成块为本地 chunk 2、latent [5,8)，而非原先固定的 chunk 3。
默认值不变；true 继续使用 1/2/2 prefix。Source adapter 的 reference 更新能力声明同步接受 false。只放开 split 标志，仍拒绝其他 block 周期。

## 决策与调用链

沿用 ChunkPlan 作为边界唯一来源。Packer 枚举前五 latent 的完整 chunk，
SourceReadySession 从返回值计算 prefix 块数，并用于切换后的规划和接纳。
Sampler 继续将最新五 latent 以原值填入 clean target，先重建 condition，再提交 history。
不改变历史张量所有权、RNG、codec、切换生效边界或新 prompt 的处理方式。

仅删除校验不能成立：原来的 chunk(3) 会使 false 布局跳到 [8,10)，无法与五 latent prefix 相接。
不新增另一套切分配置；复用现有 layout 和 KV commit。

现有 prompt-joint / prefix CUDA Graph 专用于三个历史块及 source/target 7/8。
两个历史块使用现有串行 fallback；已命中 condition KV 时跳过重复 condition forward，
依次提交两个历史 target。true 的联合执行和失败行为保持兼容。
因此此变更不承诺与 true 相同的 TTFF/切换延迟。

## 验证

计划：CPU 小模型覆盖 prefix block/行标签、split 两种配置、两次 prompt/reference 切换、
最新五 latent 保留、fresh KV 数值对照、中途更新等待周期完成和 EOS 截断。
补充 prompt-joint 开启时 false 的串行回退数值对照。
实际结果：WSL Python 3.12、torch 2.14.1+cpu。两份受影响测试中 202 项通过；
新增回退测试首次因测试 context/role 长度不匹配失败，修正输入后两项均通过
（17.24s）。另有 source adapter 229 项通过（15.65s），含 true/false 的能力声明；合计 433 项通过。CUDA 用例未执行。Ruff 0.13.2 check/format 和
 git diff --check 通过。日志位于本地 outputs/unsplit-prompt-prefix/。
CPU harness 额外屏蔽未使用的 GPU registry 查询，并显式选择 CPU platform；
未替换分块、Sampler、DiT forward、KV 或 session 算法。

CPU 测试显式使用 torch_sdpa，关闭 stable shapes/compiled AdaLN；这与性能默认基线不同，
只证明 CPU 结构与数值语义，不证明真实 checkpoint 的效果、CUDA Graph 或 SP 通信。
### H100 实测（2026-10-30）

受测代码 `1ab209cad67c364b53fff9a9e8c44770c1103929`。H100 四卡布局为
Encoder/Qwen / DiT rank0 / DiT rank1 / Lynn Decoder；用户指定 W6/S3，
与统一基线 W21/S3 的差异明确保留。使用短prompt（水彩→黏土）、无参考图、
seed42、24FPS paced source、360帧；4→2 schedule、stable解码、block_relative/gap12。

- 832×480：预热与正式各360帧，prompt更新各applied一次，两遍RGB exact。
  真实块边界从[0,3)、[3,5)开始；每次prefix为[0,3)、[3,5)，
  本例在全局latent40切换，新块为全局[40,43)、局部[5,8)。
- 正式轮condition cache启动/切换均命中，两个历史块出现一次预期的
  prompt-joint unsupported_layout串行回退；普通cross_chunk无回退。
- 1280×736：四卡/SP2/W6S3、KV单槽/prefix4096在KV预分配时OOM，未进入生成，
  不能算功能通过。更早的480/W21/S3也在KV预分配OOM，不是共卡导致。
- 正式轮init→RGB 2.070s、E2E 19.947s、FPS 18.048、切换0.890s；
  只有一轮正式请求，包含阶段观测及同机另一档并行加载，不作性能收益结论。
- driver生命周期采样峰值（MiB）：Encoder/Qwen共享卡53891、DiT两rank
  68611/67333、Decoder16971；该口径含加载，不是各组件独占allocated峰值。
  更细的常驻/请求峰值、Host区间和计数见PR描述及原始报告。

已尝试480与736；SP1、GPU split=true同条件对照、多参考图与人工画质均未覆盖。
GPU Graph捕获日志存在，未完整采集replay计数，不声称全程命中。
所有本次Pod已清理，原始数据保留在私有OSS：
`oss://leap-world-shanghai/world-model/sglang/evals/pr483-unsplit-h100-20261030-r5/REPORT.md`，
对应r5/readback.tar.gz为480成功、r6/readback.tar.gz为736容量失败。
完整逐轮证据归档在[PR #483](https://github.com/seedleap/sglang/pull/483)，
不将单次实验日志提交入库。

### 主线同步验证（2026-10-30）

合入 `main@e8977c4824`。唯一文本冲突位于部署 recipe 开头：主线新增默认 checkpoint
与导出配置校验说明，本 PR 新增 unsplit 的 H100 验证记录。两部分均保留，先列主线说明，
随后列历史受测结果，并明确不能替代修正 serving 目录的验收；本次不改本 PR 的推理代码。

沿用本地 CPU harness，在 WSL Python3.12 / Torch2.14.1+cpu 环境完整运行
`test_zing_h3_chunks.py`、`test_zing_h3_denoise_graph.py`、`test_zing_h3_source_adapter.py`：
433 passed / 0 failed / 4 skipped，131.70秒，1条环境警告。四个跳过项均要求 CUDA，
未作为已通过；七个 PR Python 文件的 Ruff、format 及 diff 检查通过。
本轮未重跑 GPU，前述480成功及736 OOM仍仅属于 `1ab209cad6` 的历史证据。

## 实现惯例与 review 入口

已核查本次目标 fork main 的 ChunkPlan、Packer、Sampler 和 session 实现；
沿用其 native block 元数据和 serial clean-prefix commit，未新增公共抽象或跨模型依赖。
这属于 fork 自有 Zing 路径，未将其视为 sgl-project 上游已采用的功能。

阅读顺序：packing.py 的 prompt_switch_prefix_chunks → incremental.py 的
_next_model_chunk / _activate_pending_prompt / _prefix_prediction → sampling.py 的
pack_prompt_prefix / rebuild_prompt_prefix → test_zing_h3_chunks.py。
重点检查：前五 latent 恰好被覆盖，下一块从 5 开始，缓存命中时不把首个历史 target 当 condition。
