# Worker group test Master

入口继续使用 `encoder_test_master.py`。Master 复用现有 source 读取、prompt／参考图事件、独立发送与接收线程、结果记录及视频拼接代码。

旧共享 API 自动选择多个 DiT 的入口已注释停用，请使用下述 worker HTTP 建组流程。
停用原因及已发现的会话释放、超时重试问题见
[部署说明](../../docs/docs/sglang-diffusion/zing/deployment_zing_h3.mdx#当前状态旧共享-api-多-dit-入口停用2026-11-03)。

接入背景与启动流程见 [Zing 分离部署接入指南](../../docs/docs/sglang-diffusion/zing/DISAGGREGATED_SERVING_GUIDE.md)。
通信组接口契约见 [Worker 通信组接口协议](../../docs/docs/sglang-diffusion/zing/specs/WORKER_CONNECTION_API.md)。

## 已有 worker

参数接受 `IP:port` 或完整的管理 HTTP 地址。地址对应一个具体 service，不需要额外提供 service index。

```sh
python benchmark/zing_h3_causal/encoder_test_master.py \
  --encoder-http 10.0.0.11:31000 \
  --dit-http 10.0.0.12:33000 \
  --decoder-http 10.0.0.13:36000 \
  --source-video /data/original-case/source.mp4 \
  --expected-frames 120 --width 832 --height 480 \
  --prompt 'Use the prompt from the original test case.' \
  --output /results/group-test-01
```

测试前先等待健康状态，随后查询三个 service、取得预留、使用响应中的 `service` 原样组成拓扑，并在预留期内发送建连请求。Encoder 输入 WS 和 Decoder 输出 WS 地址都来自查询结果，不自行推算。

请求数据仍由现有 `continuous_source_client.py` 读取并发送；输出沿用独立 Decoder 接收线程，最终保存 RGB、MP4、事件和指标。模型初始化与 HTTP 控制耗时不计入业务客户端原有的计时区间。

## 缺少 worker 时启动

没有提供 HTTP 地址的角色，必须在 `--worker-launch-config` 中提供启动配置。配置会在**运行 Master 的测试主机／已分配的测试容器**内启动原有 worker 进程；模型路径、部署配置和获准使用的 GPU 由测试启动配置指定。

Master 不申请云资源、不查询集群、不扫描机器、不自动选择 GPU。远端资源分配由测试 harness 先完成，再在该测试环境中运行 Master，或把已启动 worker 的 HTTP 地址传给 Master。

```sh
python benchmark/zing_h3_causal/encoder_test_master.py \
  --encoder-http 10.0.0.11:31000 \
  --worker-launch-config /run/test/worker-launch.json \
  --source-video /data/original-case/source.mp4 \
  --expected-frames 120 --width 832 --height 480 \
  --prompt 'Use the prompt from the original test case.' \
  --output /results/group-test-02
```

这个示例复用指定的 encoder，只启动配置中的 DiT 和 decoder。全部不提供 HTTP 地址时，按配置启动三个角色。全部提供时不会读取启动配置，也不会重复启动任何 worker。

配置结构如下。`command` 是 argv 数组，不经 shell 执行；占位符必须替换成测试环境的真实值，不会自动展开环境变量。沿用上一次成功测试的模型、采样、FP8 和 decoder 配置。

```json
{
  "workers": {
    "encoder": {
      "http": "http://127.0.0.1:31000",
      "cwd": "/run/test/source",
      "env": {"CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": "/run/test/source/python"},
      "command": [
        "python3", "-m", "sglang.multimodal_gen.runtime.entrypoints.zing_source_encoder_server",
        "--model", "REPLACE_WITH_MODEL_PATH",
        "--service-count", "1", "--http-port", "31000"
      ]
    },
    "dit": {
      "http": "http://127.0.0.1:33000",
      "cwd": "/run/test/source",
      "env": {"CUDA_VISIBLE_DEVICES": "1", "PYTHONPATH": "/run/test/source/python"},
      "command": [
        "python3", "-m", "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_worker",
        "--model", "REPLACE_WITH_MODEL_PATH",
        "--pipeline-config-path", "REPLACE_WITH_VERIFIED_PIPELINE_CONFIG",
        "--quantization", "fp8", "--http-port", "33000"
      ]
    },
    "decoder": {
      "http": "http://127.0.0.1:36000",
      "cwd": "/run/test/source",
      "env": {"CUDA_VISIBLE_DEVICES": "2", "PYTHONPATH": "/run/test/source/python"},
      "command": [
        "python3", "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=1", "--module",
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.decoder_worker",
        "--model", "REPLACE_WITH_MODEL_PATH",
        "--decoder-config", "REPLACE_WITH_VERIFIED_DECODER_CONFIG_JSON",
        "--service-count", "1", "--http-port", "36000"
      ]
    }
  }
}
```

`CUDA_VISIBLE_DEVICES` 仅表示已获分配、已核验空闲的测试 GPU，示例序号不代表当前机器可用资源。跨主机连接时，在对应 worker 启动命令中设置实际可访问的 `--advertise-host`，管理监听地址与网络策略也必须允许 Master 访问。

启动日志和进程回执默认放在结果目录旁的 `<output>.workers-<随机后缀>/`，可用 `--worker-log-dir` 指定。Master 只终止本次启动的进程，已有 worker 不会被终止。

## 独立 DiT 的 SP2 启动

在已分配两张 GPU 的 DiT 实例内，沿用原模型、pipeline、采样和 FP8 参数，
将启动命令改为以下形式：

```sh
python3 -m torch.distributed.run --standalone --nproc-per-node=2 --module \
  sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_worker \
  --sp-degree 2 --model /path/to/model \
  --pipeline-config-path /path/to/verified-pipeline.json \
  --quantization fp8 --advertise-host REPLACE_WITH_REACHABLE_DIT_IP
```

`--sp-degree` 必须与 torchrun 的进程数一致。本入口使用原有 Ulysses SP；
只有 rank 0 对外提供一个 DiT service，Master 仍只接收一个 DiT HTTP 地址。
Encoder 和 Decoder 的命令及接口不变。直接启动、不传 `--sp-degree` 时仍为 SP1。
两个 rank 均加载原有模型并初始化各自的 KV 分片，内部 SP 通信组在 worker 生命周期内保留；
HTTP `destroy_group` 只解除外部 worker 连接。

本次 SP2 接入已覆盖 CPU 双 rank 的输入顺序、张量广播、建连／排空断连，以及建连、rank 0 传输和进度消息的失败传播。
单 rank 模型执行异常后的 fail-stop 行为尚未完成故障注入验证，不承诺模型故障在线恢复。
2026-11-02 的 H100 E1+SP2+D1 已完成 832×480、FP8、5/30/120s 各三次交替测速，
5s/120s RGB 哈希与此前对应 SP1 case 一致。736、混合 H100/H200、同条件 SP1 性能对照
及逐组件耗时/峰值显存未测；结果和证据入口见
[部署 recipe](../../docs/docs/sglang-diffusion/zing/deployment_zing_h3.mdx#独立-worker-的-sp2-接入h100-480-已测)。

## 保留和复用连接

默认测试结束后调用三个 service 的 `destroy_group`，收到 `cleaning` 后查询至清理完成。该查询会取得新的 10 秒预留，随后自然到期；重复测试建议直接保留通信组，避免反复建连。

```text
--worker-group-id test-group-a --keep-connections
```

成功后保留连接；本次启动的 worker 进程也继续运行。使用相同 HTTP 地址和 `--worker-group-id test-group-a` 再次启动 Master，即可复用已有通信组。保留组必须明确指定匹配的 group ID，不能接管其他组。

测试摘要中的 `worker_group` 记录 group ID、完整拓扑、是否复用及是否保留，不记录预留 token。

## 其他参数

| 参数 | 含义 |
|---|---|
| `--worker-startup-timeout` | 等待 worker 健康状态的超时，默认 900 秒 |
| `--worker-control-timeout` | HTTP 建连与清理等待的超时，默认 600 秒 |
| `--keep-connections` | 测试成功后保留通信组和本次启动的 worker |
| `--worker-group-id` | 指定新组 ID，或明确要复用的已有组 ID |
| `--direct-ws` | 使用原有直接 WS 模式，不管理 worker 连接 |

旧的 Encoder moments／DiT latent 单模块测试继续使用 `--direct-ws --output-kind source_moments` 或 `--direct-ws --output-kind dit_latents`，配合原有 `--url`。Worker group 模式输出 RGB。

## 验证边界

本地测试覆盖真实 HTTP 协议，以及以 CPU 回调代替模型的组合、失败和清理流程。已有 GPU 验证的源码边界和配置见上方部署 recipe；本地协议测试不能替代 GPU 验证。本次提交整理补充了 DiT 清理后的通信故障分类回归，此清理修复尚未重新做 GPU 故障注入。

## 独立 worker 的 Block0 缓存

缓存相关的外部接口统一由对应 Encoder service 提供，沿用 main 的请求与回执格式：
`/v1/zing/cache` 下的列表、删除、清空、preload、unload、status，以及业务 WS 端口上的
`/v1/realtime_video/prefill`。管理操作转发给当前连接的 DiT；尚未建组或正在断连时拒绝操作，
不会查询其他 worker。`destroy_group` 不清理 DiT 的内存／磁盘缓存。

在 DiT 启动时使用 `--zing-block0-cache-mode on`，并按原部署要求配置
`--zing-block0-cache-host-gb`、`--zing-block0-cache-dir`、`--zing-block0-cache-disk-gb`；
缓存目录和预算由 DiT 管理。L1 沿用 main 的分页分配，按实际 prefix 行数占页；
status 的 page_rows、total_pages、free_pages 等字段原样返回，不按旧的固定槽位容量解释。
周期 GC 参数继续沿用 main，同目录只指定一个 GC 实例。
Encoder 可通过 `--pipeline-config-path` 传入原有 pipeline 配置，包括参考图缩放设置。

Encoder 复用 main 的 key 生成函数，通过原 ZMQ 通道查询；条件张量仍使用 NCCL。
命中跳过对应的 Qwen／参考图准备，DiT 使用原缓存 owner 恢复 prefix KV。
source VAE 继续执行。普通请求仍遵循 `cache_write`，prefill 保持 `force` 语义。
`prefill.completed` 由 Encoder 返回，表示 DiT 的 L1 已可用且临时请求已清理；不等待异步 L2 落盘。
视频 RGB 与 source.consumed 仍由 Decoder 返回。

本轮 rebase 后的 H100 E1+SP2+D1、832×480 已完成真实 GPU/NCCL 回归，
覆盖完整 case、prefill、L2 重新加载、更新后的缓存命中及断连重建；源码边界和原始证据见部署 recipe。
736/B300、完整 main/candidate 性能及三组件逐 rank 显存矩阵仍未验证，历史结果不外推到当前版本。
