# Issue #256 conditioning RPC 使用说明

2026-10-10。**首期仅验收单机多卡、loopback RPC、SP1。** 跨机部署、TLS/mTLS、HA/LB、SP2、SP4 均未验收，不属于本期支持范围；`tls` 配置和非 loopback 的 bind 路径是 experimental，`serve` 启动时会打 warning。

推荐拓扑：GPU0 运行 1 个 TP1 Qwen conditioning service，GPU1…N 运行多个独立 DiT worker，所有 worker 通过 `http://127.0.0.1:<port>` 共享这个服务，并使用同一个 tenant（这样才能共享服务端 LRU 与 singleflight）。

实测（L20D，提交 `cc6b1a1d`，详见 PR #272 描述）：每个 worker 显存减少约 48.1 GiB，稳态出帧速度基本不变；一个 Qwen 实例处理新内容的能力约 15 次/秒（编码串行，单次约 65 ms），增加 worker 只会排队，不会提高 miss 吞吐；同内容的并发请求由 singleflight 合并，同一会话内相同 digest 在 worker 上直接复用，不再发 RPC。

## 管理员 CLI

```bash
python -m sglang.multimodal_gen.runtime.conditioning.cli --help
python -m sglang.multimodal_gen.runtime.conditioning.cli build-profile --model /internal/zing-export --numerical-environment /internal/numeric-environment.json --output /internal/profile.json
python -m sglang.multimodal_gen.runtime.conditioning.cli serve --config /internal/service.secret.json --profile /internal/profile.json --host 127.0.0.1 --port 8093
python -m sglang.multimodal_gen.runtime.conditioning.cli warmup --config /internal/service.secret.json --profile /internal/profile.json --input /internal/warmup.json --url http://127.0.0.1:8093
```

仅查看 help 可在 Mac 上直接执行 `python python/sglang/multimodal_gen/runtime/conditioning/cli.py --help`。命令只有 build-profile/serve/warmup；artifact、precompute、import 和 cache-root 已删除。profile 构建在 CPU 计算现有资产与代码 hash，不复制权重；可选 projection 用 `--projection` 指定。数值环境须包含 `encoder_tp:1`、`gpu`、`cc`、`backend`、`precision`、`software`。build-profile 的 `gpu_validation:NOT_RUN` 不证明硬件验证；NativeConditionEncoder 启动核验实际 TP、folding、backend、BF16、GPU/CC 和 torch/transformers/cuda。

## 服务与 worker 配置

H3 六项优化现由 profile 的 `numerical_environment.execution_options` 冻结，builder
从 `SGLANG_H3_QWEN_*` 环境解析（默认全开）；构建与启动环境须一致。文本/视觉 FA4
均关闭、均开启或仅一项开启时，`backend` 分别为 `torch_sdpa`、`fa4`、`fa4+torch_sdpa`。
`precision=bf16` 与独立 FP8 权重选项并存；加载回执检查实际权重和算子。升级需重建并
分发 profile，详见 [执行契约及验证边界](../plans/QWEN_REMOTE_EXECUTION_OPTIONS.md)。

服务配置为 JSON：`tokens` 映射私有 bearer token 到 `{tenant,admin}`，`native` 指定 `{model_path,gpu_id}`，`limits` 必须显式填写 **ResourceLimits 的全部字段**，所有值有限、正值且 tenant 不大于 global。字段及默认值由实际代码生成，避免复制过期字段：

```python
from dataclasses import asdict
from sglang.multimodal_gen.runtime.conditioning.resources import ResourceLimits
print(asdict(ResourceLimits()))
```

worker 配置使用 `bearer_token`、`conditioning_timeout_s`（默认 10）、`retry_initial_s`（默认 0.05）、`retry_max_s`（默认 0.5）及 `limits` 内 request/response bytes 与 max rows。服务/worker 共用配置时同时包含这两组字段。凭据保存在管理员 secret，勿送到浏览器或写入日志。`issue256_gpu/prep_e2e.py` 是 NVIDIA 环境下生成 profile/配置的辅助脚本，不是已执行实验。

在既有 Zing 启动命令加上：

```bash
--zing-conditioning-mode remote \
--conditioning-service-url http://127.0.0.1:8093 \
--conditioning-expected-profile /internal/profile.json \
--conditioning-config /internal/worker.secret.json
```

默认模式仍为 local；local CPU 缓存由 `--zing-conditioning-local-cache-bytes` 设置，默认 268435456（256 MiB），0 禁用。remote worker 不加载 Qwen、tokenizer 或 processor，且不允许与 text-encoder CPU offload、layerwise offload、encoder folding 配置混用。remote 模式的默认 encoder_parallel auto 会按已有逻辑调整为 replicate。

服务 URL 只有 loopback 地址允许使用明文 `http://`，其他地址必须是 `https://`（客户端构造、`ServerArgs` 校验和 `warmup` 子命令都会拒绝非 loopback 的 `http://`，避免 bearer token 明文传输）。bind 地址与 TLS 配置的校验在加载模型和预热之前完成。

非 loopback 的服务 bind 需要 `tls`（certificate/private_key/ca）或 `internal_auth_proxy_verified:true`。该标志表示管理员已经核验代理，不能代替真实部署验收。worker 使用管理员 endpoint 与 bearer；首版不宣称跨机或客户端 mTLS 已验收。

## Warmup 输入

输入是 JSON 列表，每项为：

```json
[
  {"prompt": "完整 prompt，保留前后空白", "images": ["/internal/ref1.png", "/internal/ref2.png"], "s0_image": "/internal/source-first-frame.png"}
]
```

images 顺序必须与真实会话一致；可省略 images/s0_image。s0_image 转 RGB 后以 video 类型放在参考图前面，实际像素、shape/type 和顺序必须相同才能命中。CLI 不解码完整 source video。固定文本/图片可 warmup；实时 S0 变化时会产生新 key，并受 LRU 限额约束。输出计数为 hit/miss/joined。CLI 验证 profile 与响应，不生成持久离线工件。

## RPC 与容量

认证端点：`POST /v1/conditioning/encode`、`GET /v1/conditioning/profile`、`GET /v1/conditioning/stats`。普通 token 的 stats 只看本 tenant，admin 可看全局。请求必须声明 Content-Length，禁用 Transfer-Encoding。raw media 为 CPU uint8 `[3,1,H,W]`；图片、source S0、合法同值单帧 video clip 可用。remote 不支持多帧 video/audio/video_audio，local 保留原能力。

媒体 FPS 输入域仍为有限 `0 < fps <= 240`，请求 header 保留调用方值。服务在完成输入与 digest 校验后，仅把 remote 单帧 video 的内部采样 FPS 归一为 240.0；T=1 始终选 index 0、时间戳 0 并使用相同 temporal padding，完整原 forward 的 context/row_role 保持一致。该边界处理避免极小合法 FPS 使原采样循环下溢或运行过久；图片与 native encoder 源码不变。

默认预算：active slots global/tenant 32/16；inflight 8 GiB/2 GiB；CPU cache global/tenant 256 MiB/256 MiB；request 512 MiB；response 256 MiB；max rows 16384；queue 16；CPU executor active/waiting 2/4。

每行输出为 `5120*2+8 = 10248` 字节。miss/joined 先预留 `16384*10248 = 167903232` 字节（160.125 MiB），再缩减至实际值；hit 在 clone 前按实际 tensor 字节预留。还有 body、解析/序列化 workspace 与 bounded header 成本。极端 512 MiB body 加最坏输出，2 GiB tenant 至多容纳约 3 个这样的请求，其他 workspace 会进一步收紧。真实并发同时受 slot、CPU tickets 和 queue 限制，不能按 16 slots 推断 16 个最大请求一定准入。

这些账本是逻辑 buffer 所有权预算；不是进程 RSS 或 GPU 显存硬上限。框架、native 模型、allocator、CPU workspace 和编码器暂存仍需硬件测量。CPU缓存读取返回后重新检查预算，过期请求不创建或加入编码任务；queued 最后 waiter 离开则撤销编码；running 不可取消工作 drain 前保持其真实 retained bytes 计费。成功/失败响应都在发送终态后释放请求 slot。

服务端正有限 deadline_ms 安全限制到配置预算，巨大 JSON 整数不会触发浮点溢出；FPS 超出范围返回400。响应序列化完成及开始发送成功状态前再次检查到期，超时走504发送宽限，资源仍持有至终态。

默认 10 秒是同步 RPC 总预算，包含序列化、重试、下载和解码；必须小于实际 SP CPU group timeout。仅对 429/503/连接失败做有界带抖动重试，400/401/409 不重试，不回退 local。单 worker 上其他会话可能同时停顿，性能验收必须包含该影响。

连接、发送、HTTP 状态行/响应头及 raw body 共享绝对 deadline；不跟随重定向，不读取环境代理。每进程固定一个 I/O 线程，最多 32 个请求 ownership；DNS 两线程、总容量四项。同步超时返回后，容量仍占用到异步传输清理结束；系统 DNS 无法强制中断；外层取消不会取消物理 dispatch，queued lookup 保留容量至实际 dequeue 并跳过已取消 OS 调用，dispatch drain 后才回收容量，避免已取消队列无限累积。整个网络 exchange 使用 Python 3.10 支持的 wait_for，分别捕获 asyncio/concurrent.futures 超时类型。服务等待者超时或断连即离开编码队列，早于错误响应发送；发送和 native drain 各自保留其实际资源所有权。

## 错误与验证

公共错误为英文 `{code,message}`；码为 CONDITION_INVALID、CONDITION_UNAUTHORIZED、CONDITION_PROFILE_MISMATCH、CONDITION_BUSY、CONDITION_TIMEOUT、CONDITION_UNAVAILABLE。初次编码失败 fatal:true；中途更新失败 fatal:false，继续旧 conditioning，相同失败事件只通知一次。逐 block 输出 event_id/prompt_version 表示实际生效版本，source provenance 不变。具体网络字段见 [实时契约](realtime-contracts.zh-CN.md)。

CPU 验证：`python benchmark/zing_h3_causal/run_conditioning_cpu_tests.py`；依赖版本见 ISSUE_256_CPU_ENVIRONMENT.json。harness 运行真实 conditioning/ASGI/safetensors 和 fake native/SP 边界，不初始化 NVIDIA registry。CPU 通过不证明 CUDA/SP collectives、Qwen 数值、质量、worker 显存或在线延迟。GPU 辅助脚本已更新为 local/remote/encode，尚未执行；launch_bench.sh 与 launch_e2e.sh 默认 SERVICE_GPU=0、WORKER_GPU=1，使用前须实际检测 GPU 与用户授权。
