# Qwen 同机 NCCL 结果通道

基线：`seedleap/sglang` main `a99560cf901be9b1589bcc9537593e9cc3883fb4`。本功能只改变 Qwen 条件结果的传输方式，不改变编码算子，不引入前缀复用或异步编码。

## 使用与边界

目标是让远程 Qwen 的 BF16 `context[rows,5120]` 和 I64 `row_role[rows]` 直接进入 DiT leader GPU，再沿现有 SP GPU 广播。输入 RGB、身份、形状、完成通知仍走 HTTP。新增配置 `result_transport: "nccl"`，服务端与客户端均显式开启；默认 `"http"` 保留原行为。

仅支持同机可信部署、一个 DiT leader、一条持续连接、每次一个请求。服务监听 loopback，仍校验 bearer token、profile、输入身份及容量。source encoder、Qwen、decoder 各自拥有 communicator；共用 GPU 不代表共用进程或 collective 序列。NCCL 实际是否选择 NVLink 由机器拓扑和运行时决定。

在既有服务端 JSON 和 DiT `--conditioning-config` 指向的客户端 JSON 中，各增加：

```json
{"result_transport": "nccl"}
```

这是增量配置，须保留原 token、native、limits 等字段。服务仍使用
`python -m sglang.multimodal_gen.runtime.conditioning.cli serve --config SERVICE_CONFIG --profile PROFILE --host 127.0.0.1 --port 31002`，
DiT 仍使用 `--zing-conditioning-mode remote --conditioning-service-url http://127.0.0.1:31002 --conditioning-config CLIENT_CONFIG --conditioning-expected-profile PROFILE`。
代码身份包含 `native.py`，升级后须通过原 `build-profile` 命令重新生成并同步同一 profile；不能沿用旧代码身份的 profile。
客户端 `conditioning_timeout_s` 不得大于服务 `limits.request_timeout_s`，且其两倍须小于 SP CPU group timeout，为 NCCL 超时后的 drain 留出余量；不是端到端墙钟时间的硬保证。
建组在首次 encode 懒执行；第一请求包含建组/首次 NCCL 初始化，后续请求复用。现有 CLI `warmup` 是 HTTP 结果工具，不用于 NCCL 服务；服务启动的 native warmup 仍执行，部署端用真实 DiT 首次请求预热结果通道。

NCCL 模式不使用现有 CPU 结果 LRU，避免为了缓存重新引入 D2H；DiT 会话内相同输入的现有缓存仍保留。新增 GPU LRU 不在范围内。main 的远程 native profile 仍要求六个 Qwen 执行开关全部关闭；本次不移植独立执行选项功能。

## 接口与所有权

1. 从现有 encoder/decoder transport 提取两 rank 私有建组及 tensor send/recv 到 runtime 层。保留各自 PrefixStore 名称与原调用接口；Qwen 使用独立前缀。
2. 客户端首次使用时创建 loopback TCPStore，通过认证 HTTP open 与服务端建立 Qwen 两 rank group。建组只执行一次，不能重试不确定的发送。
3. encode 仍使用现有输入格式与 admission/encode queue。NCCL 模式 native 输出和队列结果留在 CUDA，不进入 safetensors 或 CPU LRU。
4. 服务先返回流式 HTTP 元数据，客户端校验身份/形状、分配 GPU 接收 buffer；随后 NCCL 按 context、row_role 固定顺序传输。服务发最终完成回执后请求才成功。
5. 服务持有结果到 GPU send 完成；客户端持有接收存储并同步通信 stream 后才交给现有 SP 广播。失败或传输阶段断连会使该通道失效，不在旧 communicator 上接下一个请求。正常完成释放单请求占用，后续 prompt/会话复用已建通道；进程退出释放 communicator。

如果集合通信抛错，失败通道保留可能仍被 CUDA 引用的结果/接收 buffer，并拒绝继续使用；需要重启 Qwen 服务和 DiT worker，不能仅重开 WebSocket。此保守故障策略不提供自动恢复。正常结束一个生成会话不销毁进程级通道，下一会话可以复用。

HTTP 模式继续支持原缓存和并发契约；NCCL 模式显式限定单接收方，第二请求立即拒绝而不插入当前收发序列。只复用编码队列与控制输入，不把 Qwen 塞入 source 的 moments 协议。

## 验证计划与证据

### 与 main 的异步预取合并（2026-10-26）

同步 main `3e5ddd880360`（PR #410）时，保留 HTTP 路径的 session prefetch 与独立 Decoder overlap；NCCL 路径继续在 worker 调用线程通过 `encode_device` 接收 GPU 结果。当前预取器仅持有 HTTP/CPU 结果，因此配置 NCCL 时不启动该预取器，并输出明确 warning；不能把两项开关同时开启当作已经实现异步 NCCL 预取。后续 GPU 预取需另行定义 CUDA stream、结果持有和取消收尾，不在本次冲突解决中实现。

合并回归覆盖 HTTP 预取命中及失败、NCCL 路由不误消费 CPU 预取、两种 SP rank 不启动 HTTP 预取器，以及原有 NCCL 超时和生命周期测试。仓库规定的 Linux conditioning 完整默认套件两轮均 1820/1820 通过，其中 `test_nccl.py` 每轮 22 项；回执为仓库外 `D:/qwen-nccl-work/ci-conditioning-main-merge/receipt.json`。GPU 性能证据仍限于下述历史版本。

### 2026-10-26 超时与空闲结果持有修复

配置和协议保持不变。首次建连、HTTP 元数据和完成回执共用一次调用的绝对 deadline；HTTP 流复用客户端已有的有界异步 I/O 线程，元数据就绪后由调用线程执行 GPU 接收。相比给同步 socket 逐次设置 timeout，这能中断持续分段到达的慢响应。过期后不再开始新的接收；已经提交的 GPU 工作仍须安全收尾，失败缓冲区继续由失效通道持有，不能为了准点返回而提前释放。NCCL 固有超时用于收尾，SP CPU 等待预算继续保留两倍请求超时的余量。

EncodeQueue 只负责把结果交给请求；每次迭代末清除 condition、flight 和 executor future 的局部引用。请求和发送端在真正完成前继续持有结果，空闲队列不再无意保留最近一份 GPU 输出。

修复实测：`test_nccl.py` 的 20 项测试通过，覆盖真实 loopback HTTP 慢元数据、慢完成回执及 EOF 停滞、截止后不进入接收、首次建连消耗请求预算、接收失败保留缓冲区，以及成功发送和无人等待的编码完成后弱引用释放。锁定依赖的 Linux conditioning 默认完整套件连续两轮均 1799/1799 通过，无跳过；回执保存在仓库外 `D:/qwen-nccl-work/ci-conditioning-deadline-fix/receipt.json`。

相同复现探针中，0.2 s 请求预算下的慢 HTTP 响应由修复前约 0.955 s 返回变为约 0.200 s 返回，且没有再调用 GPU receive；原 HTTP 路径同样约 0.200 s 返回。队列空闲时结果弱引用已释放，无需关闭服务。这里 HTTP 为真实 loopback，CUDA/NCCL 为替身、生命周期探针使用 CPU 张量，因此不代表 GPU 故障收尾或显存 allocator 实测。以下原 GPU 性能证据来自修复前版本，新 HTTP 控制调度路径尚未重复 GPU 性能/故障测试。

- 本地：已有 conditioning 回归；新增元数据/身份/容量、单请求互斥、失败后拒绝复用、CPU 路径兼容和结果传输协议测试。
- 多进程：可用时用 Gloo 验证共享建组及张量往返；这不代表 CUDA 生命周期或 NCCL 已验证。
- 已完成同机 Qwen→NCCL→SP2 连续变长结果与三组 prompt1 输出一致性；取消/断连、source/decoder bridge 共存和完整视频 prompt 切换仍待验证。
- 完整流水线待测：按统一基线分别 480/736，固定 checkpoint/输入/布局，比较整体切换、FPS、Encoder/DiT/Decoder 时延及显存。本轮按指定范围仅验证条件链路，未启动完整流水线验收。
- 已执行条件链路专项云测试，见下节；不能将其约 20% 的链路降时延推断为完整视频流水线收益。

### 本地已执行（2026-10-25）

Windows / Python 3.13 / PyTorch 2.7.0：20 项针对性测试通过；PR 收尾重跑仍为 20 项通过（7.89 s，1 条 pytest 空正则匹配警告，错误码另有显式断言）。
包括 `conditioning/test_nccl.py` 的 13 项、`test_review_r1.py` 的 2 项，以及
`test_service_resources.py` 中 admission、容量、取消 drain、默认预算、encoder-only loader 的 5 项。
其中两进程 Gloo 真实传输了三组不同长度的 BF16 context/I64 row_role，未初始化默认 world；
HTTP 流解析与服务控制测试使用真实业务代码，CUDA stream/NCCL receive 使用替身，不能替代 GPU 测试。

标准 pytest 在加载全局 conftest 时被 Windows 缺少 `resource` 阻断。
本地 runner 仅为包路径设置 namespace 并关闭模型 loader 全局 conftest，业务叶模块和 PyTorch 正常执行；
模型配置/profile fixture 仍触发 Linux 专属依赖，未纳入上述通过项，未修改仓库测试环境来掩盖限制。
本机复现入口 `D:/qwen-nccl-work/run_tests.py`，JUnit 记录 `D:/qwen-nccl-work/test-results.xml`。
按仓库配置运行 isort、Ruff format、Ruff `E9,F63,F7,F82,F401,UP037` 与 `git diff --check` 通过。另补发送阶段错误不重复发送 HTTP headers 的回归；NCCL 模式 response_bytes 仅计 HTTP 控制体，不计 GPU payload。

在完整 Linux 环境可运行：

```bash
python -m pytest python/sglang/multimodal_gen/test/unit/conditioning/test_nccl.py -q
```

这是协议/控制与 Gloo 测试命令，不会自行验收真实 NCCL。下面的 GPU 专项覆盖条件链路；完整视频质量/FPS与 source/decoder 共存仍待验证。

## 条件链路 GPU 专项（2026-10-25）

按用户明确缩小的范围，只运行真实 Qwen 编码、HTTP/NCCL 结果传输与 SP2 GPU 广播，不加载 DiT、source VAE、decoder。同机 L20X / SM90 / NV18，申请四卡而实际使用三卡；BF16、torch_sdpa、六开关全关，两组 profile 完全一致。常用三个 case 的 prompt1、480 source S0，输出 5413/6575/6606 行；每组两轮预热，每例十次正式请求。

完整条件链路热态中位数缩短约 20%，编码耗时基本不变，主要消除了 CPU 结果交付开销。三例首个正式请求的 context/row_role 在两个 SP rank 上逐元素一致。首次 NCCL 收发存在明显懒初始化成本，部署须用真实条件链路预热，单独 native warmup 不足以预热 bridge/SP。

| prompt1 case | HTTP 中位数 / P95（ms） | NCCL 中位数 / P95（ms） |
| --- | ---: | ---: |
| hiker yellow raincoat | 958.94 / 994.55 | 767.13 / 773.03 |
| garden stained glass | 1175.38 / 1221.29 | 940.87 / 959.14 |
| forest cyan trees | 1176.92 / 1201.45 | 940.50 / 965.62 |

结果就绪到 leader 返回的中位数从 190–233 ms 降至 1.82–1.87 ms；该阶段包含序列化、复制及协议开销，不能视作纯网络耗时。Qwen 编码仍约 699/864/865 ms，SP 广播约 0.4 ms。首次请求 HTTP 为 1308.53 ms、NCCL 为 1541.35 ms，均不含模型加载。Qwen allocator 峰值 allocated 约 49.50–49.78 GiB → 49.56–49.84 GiB；这不是整卡或整流水线显存。

CPU 准备与 GPU 作业均 Succeeded；GPU 作业 `dlcj1inm3k1yxnv5`。受测产品源码为上述基线加本 PR 改动，冻结文件哈希见实验 `LOCK.json`；PR 收尾整理了文档和 import 格式，十个受影响 Python 文件均与受测归档 AST 一致，未重跑 GPU。环境为 PyTorch 2.13.0+cu130、Transformers 5.12.1，两组服务结果缓存均强制 miss。脚本、配置、逐请求时间戳、可加和均值细分、P95、组件 allocator 峰值和失败启动记录保存在仓库外 `D:/sglang-qwen-nccl-h200-20261025-r4/`；持久归档为 `oss://leap-world-shanghai/world-model/sglang/evals/sglang-qwen-nccl-h200-20261025-r4/`（需内部访问权限），汇总在其 `analysis/RESULTS.md`。

当前 main 的 NVML 查询只接受数字 CUDA mask；本次使用数字 mask 并核对 CUDA/NVML UUID 对应关系，没有移植其他工作树的平台补丁。encoder-only loader 的配置发现仍检查完整导出包；保留非 Qwen 权重的既有 OSS 文件引用，仅复制和加载 Qwen 权重。退出时私有 ProcessGroupNCCL 有未显式销毁提示，进程及作业正常退出，长期资源复用尚未验证。

此专项不测 736、Qwen 六开关全开、完整视频质量/FPS、流水线竞争、取消/断连故障和 source/decoder bridge 共存。配置/调用链没有因此改用模拟传输；测量调用真实 NativeConditionEncoder、ConditioningService、RemoteConditioner 和 SGLang SP group。

## 社区实践与 review

上游 SGLang 的 [CUDA IPC transport](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/multimodal/transport/cuda_ipc.py) 将控制元数据、GPU 存储及完成确认分开；这是生命周期参考，不是可以直接开启的 H3 开关。本仓库已存在 encoder/decoder NCCL bridge，因此复用其传输原语比引入新的 IPC/NIXL 后端更小。未声称此私有协议已被上游接受。

Review 路线：本设计 → `conditioning/worker.py` 实际调用 → `conditioning/client.py` / `conditioning/nccl.py` 通道与结果生命周期 → `conditioning/service.py` / `native.py` 确认没有结果 D2H → `runtime/tensor_bridge.py` 及既有 VAE wrappers → 测试。

重点：接收方是否真实拿到 CUDA tensor；元数据先于 send；异常后是否仍有线程/GPU 引用已释放 buffer；请求是否会交错；旧 HTTP 缓存是否保持；GPU/SP 故障证据是否充分。当前仅有条件链路 GPU 实测，不能代替完整流水线及故障恢复验收。
