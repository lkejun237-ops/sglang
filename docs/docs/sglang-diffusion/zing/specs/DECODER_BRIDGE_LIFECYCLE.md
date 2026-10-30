# Decoder 会话结束时释放私有通信组引用

## 使用与行为

沿用专用 Lynn decoder 的现有部署，不增加配置。DiT 与 Decoder 服务进程仍通过
ZMQ 交换命令、通过私有 NCCL bridge 传 latent；同一会话的 chunk/prompt 切换复用连接。
模型和服务内部的常驻通信组不属于会话资源，不在本次清理范围。

旧实现的 `abort`/`close` 排空解码任务并清空 `service/token/channel`，但主循环继续持有
`bridge/store`。下一次 `open` 覆盖 `bridge` 时，旧 ProcessGroupNCCL 的引用释放会阻塞
新会话。修复将这两个引用的释放放在会话结束确认前，异常退出也清空它们。
清理开销移动到结束阶段，不代表推理总成本消失。

## 所有权和顺序

`decoder_worker.main` 持有当前会话的 bridge 和 store，解码 `service` 只持有解码状态。
客户端持有另一进程中的对应对象，两端引用分别管理。实际客户端的正常结束也通过
`dispose → abort(session_id)` 释放会话；名字为 abort 不代表本次推理失败。

1. `abort` 先检查 session，`close` 先检查 channel；身份不匹配不得清理当前会话。
2. `drain` 等待解码队列和线程结束，再 `service.dispose()` 释放解码状态。
3. 清空 `service/token/channel` 和 `bridge/store`。
4. 既有控制组确认成功后，回复释放结果；下一次 open 创建自己的通信对象。
5. 服务因异常退出时也清空 bridge/store，不新增退出 collective。

选择在资源的现有所有者处清理，避免新增关闭协议、后台清理线程或跨会话复用状态。
没有添加客户端改动；实测验证了 worker 先释放、客户端收到确认后再释放的既有顺序，
以及当前客户端一秒接收上限内得到匹配会话的 abort 确认。

## 正确性和性能证据

2026-10-29：`test_zing_h3_remote_decoder.py` 的16个用例在受测GPU镜像的CPU环境通过
（2.75秒）；本地WSL Python3.11/Torch2.8 CPU环境与 source adapter、source/decoder
overlap 两份完整回归一起运行，252 passed / 0 failed / 0 skipped，31.84秒，1条环境警告。
新增用例直接运行真实 worker 控制循环，检查 close/匹配abort 在回复前释放传输引用、
下一会话重新创建资源、过期abort保留当前资源，以及失败命令退出时释放资源。

上海H100（实际L20Z/CC9.0，81559MiB）两卡，沿用上一轮真实 Lynn worker 和 NCCL
传输诊断，连续4轮建会话并各传一个 `[1,24,1,30,52]` FP16 latent，74880字节。
前两轮使用close，后两轮使用实际客户端dispose/abort；4轮均通过，匹配abort均得到
`released=true`，没有接收超时。首次连接19.767ms，随后三次3.448/3.396/3.448ms。
Worker会话结束释放耗时约518ms；后两轮abort确认520.330/519.948ms，客户端完整dispose
1222.298/1167.265ms，均为各自范围的观测，不能相加后宣称完整视频收益。

旧版本独立诊断第二次connect为1441.929ms，其中旧组引用释放1438.898ms；它与新实验
使用相同硬件、latent形状及worker，后两轮额外覆盖实际abort路径，但不是多次独立作业
的统计benchmark。原始数据见[PR #468](https://github.com/seedleap/sglang/pull/468)，
运行目录标识 `pr468-decoder-cleanup-h100-20261029-r1`。

上述组件轮没有重新运行完整视频；此前两遍完整推理属于修复前的历史证据。
私有组继续沿用现有引用释放机制，运行时仍有PyTorch底层shutdown警告；本次保证主循环
不跨会话保留旧对象，不声称完成了底层通信器显式shutdown或长期资源泄漏验收。

### 同提交完整推理补测

2026-10-29，运行源码 `a90d51fcfe`，上海H100四卡、SP2、832×480、360帧，
同一条真实condition连续两遍；window6/sink3、静态AdaLN、Graph0、先4步后2步，
cache on、41GB host/32GB disk、NVMe。两遍均完整输出、正常EOS、prompt在第136帧
采用，RGB与前轮同输入一致。第一遍含首次编译，TTFT54.658秒；第二遍745.700ms，
其中Source编码及latent回传149.004ms，回传完成到DiT2.585ms，首DiT353.941ms。
第二遍Decoder建连4.729ms，启动二分支、切换五分支，均未调用conditioning RPC。

第二遍切换863.828ms；GPU前缀恢复125.443ms、五分支276.075ms、第二次去噪89.738ms，
有重叠，不能直接相加为切换总时长。结束成本仍在：客户端dispose1044.606ms，
其中worker bridge释放516.696ms；dispose位于末次派发返回前，最后3帧前间隔1673.389ms，
已计入E2E17.344秒。不能宣称清理总成本消失或完整性能无回退。

原始逐块、Source完成边界、Decoder真实线程CUDA事件和显存范围见
[PR #468](https://github.com/seedleap/sglang/pull/468)，运行标识
`pr468-cache-on-full-h100-20261029-r3`。这是一条480p数据的两遍专项验证；
736、同条件baseline、无探针吞吐、独立画质和长期资源稳定性未验收。

## 社区实践与review重点

核对了SGLang上游 `srt/distributed/parallel_state.py`：正常退出和故障退出区分处理，
GroupCoordinator清理其拥有的组和通信对象；不可在单会话清理中无差别销毁整个分布式环境。
本仓库的encoder服务也在会话释放时清空私有bridge。PyTorch建议协调通信组的清理顺序，
不能因对象构造完成或Python引用清空，就推导整个通信器初始化/显式shutdown已完成。
参考：[SGLang上游](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/distributed/parallel_state.py)、
[PyTorch shutdown](https://docs.pytorch.org/docs/stable/distributed.html#shutdown)。

先读 `decoder_worker.main` 的abort/close/finally，再看
`RemoteLynnDecoder.dispose/_close_channel` 和上述测试。重点确认：drain先于释放、
回复前主循环已清空引用、过期身份不影响当前会话、常驻控制组不受影响；组件计时不替代
完整推理或长期显存结论。
