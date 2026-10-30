# Source 首块合批等待修复

## 行为与使用

沿用现有 `joint_forward=cross_chunk` 与 `joint_wait_ms=auto` 配置，无新增开关。
progressive Encoder 首次派发按当前首块的 source 依赖开始执行；后续派发仍按下一块
需求合批。首块为 1 latent 时，第一个 source 像素帧足够，自动等待预算为 0。
首块未拆分时继续等待其真实依赖：3 latent 需要 9 帧，5 latent 需要 17 帧。
该行为与缓存关闭、未命中、命中无关。

## 原因与边界

旧 `_joint_wait_target` 在首次派发也选择下一块。对于 `[3,2]`、split-first，
首块范围 `[0,1)`，下一块结束于 latent3，错误地把启动合批目标设为 source9。
H100 历史实测 `04233e46a985e0ee1f996f798598c61a188cd6b1` 的两次命中请求因此等待
332.265/333.267 ms 后超时，随后仍只派发 1 帧。

修复复用现有 ChunkPlan 和 1/9/17 codec 依赖映射，仅首次派发选择当前块。
没有新增状态、副本、缓存判断或全局关闭等待；后续联合执行和非 progressive 入口保留。
EOS、prompt、取消、有界预算、纯进度 `more_work` 返回继续由现有等待入口处理。
显式固定预算也不会延迟已满足首块需求的请求：`ready` 判断先于超时等待。

## 验证

测试入口为 `test_zing_h3_source_adapter.py`：核对首块拆分与否的依赖、auto/固定预算下
首 latent 已就绪时不调用异步等待，以及未拆分首块继续等待足够 source。
同时运行 `test_zing_source_decoder_overlap.py`，覆盖相邻的 source 续推/输出等待契约。
2026-10-29，在 WSL Python3.11/Torch2.8 CPU 环境，使用仓库现有
`run_conditioning_cpu_tests.py` 运行上述两份完整文件，236 passed / 0 failed / 0 skipped
（20.84 秒，1 条既有环境警告）；Ruff、format 和 diff 检查通过。受测源码为主线
`4adab592c602f89e87717277a3c0b29af667b35f` 加本次调度修复；该测试 harness
替代不可用的 GPU 导入边界，不替代 CUDA/collective 验证。
2026-10-29，上海 H100（实际 L20Z/CC9.0）对运行源码 `1ffee5c9e1` 完成一条真实
condition（9521→9653 token）的两遍 832×480 / 360 帧验证。沿用 SP2、window6/sink3、
静态 AdaLN、Graph0、先4步后2步及 NCCL，缓存开启。首次合批预算均为0，实际耗时
0.0019/0.0013 ms，均以 `ready` 继续；各44次派发、360帧连续输出并正常结束，
prompt 均在输出第136帧生效，两遍完整 RGB 逐字节一致。

第一遍含首次编译，TTFF 58.162秒；第二遍1.567秒。本轮按用户要求没有 baseline
或缓存命中对照，不能把历史333 ms直接写成已测 TTFF 净收益；没有验收736或baseline
画质差异，阶段计时也不替代关闭探针的正式吞吐benchmark。原始记录及完整配置见
[PR #468](https://github.com/seedleap/sglang/pull/468)。

第二遍 source 首帧编码结束到 DiT 进入仍有410.463 ms。独立 decoder 连接诊断发现，
worker 的 `close` 保留旧 `bridge`，下一次 `open` 替换它时会同步释放旧 NCCL 组。
该诊断的第二次 open 为1441 ms，其中1439 ms发生在旧组释放，新组构造约0.14 ms；
这是独立组件结果，不能直接替代完整推理410.463 ms的逐段归因。后续补充了会话结束时
清空worker传输引用的修复，见[Decoder通信组生命周期](/docs/sglang-diffusion/zing/specs/DECODER_BRIDGE_LIFECYCLE)；
该修复的两卡验证与上述完整推理属于不同运行，不能混作同一提交的端到端结果。
之后对包含清理修复的 `a90d51fcfe` 完成同数据两遍480p完整推理，启动主动等待均小于
0.003ms；第二遍TTFT745.700ms、Decoder建连4.729ms，末帧前仍承担结束清理成本。
最新范围与结果见上述Decoder生命周期说明，旧版本数字保持其原始提交边界。

## 2026-10-30 主线同步验证

合入 `main@e8977c4824`，保留主线 source 480p 编码、独立 reference 缩放及默认配置，
同时保留本 PR 的首次派发水位选择和 Decoder 会话结束清理。Git 自动合并未产生文本冲突。
主线新增的默认参数传递测试会查询设备空闲显存，CPU 回归将该硬件查询固定为80 GiB，
仍执行真实 `ServerArgs` 解析及本地/远程 codec 两种路径的断言，不改生产代码。

在 WSL Python3.11 / Torch2.8 CPU 环境运行 source adapter、source/decoder overlap、
remote decoder 三份完整文件：263 passed / 0 failed / 0 skipped，30.07秒，1条环境警告；
四个 PR Python 文件的 Ruff、format 及 diff 检查通过。
本轮没有重新运行 GPU 或新 checkpoint；前述 H100 结果仍仅属于各自历史受测提交。

## 阅读与评审

先看本说明，再看 Source adapter 的 `_joint_wait_target`、`wait_for_next_chunk`，
最后看上述测试。重点确认首次派发采用当前块、后续采用下一块，且不把未拆分首块
也固定成 1 帧。实现沿用本仓库已有的非 progressive 首次派发特判和现有 codec 映射；
这是私有 Zing 入口的局部修复，不据此宣称社区上游已有同等实现。
