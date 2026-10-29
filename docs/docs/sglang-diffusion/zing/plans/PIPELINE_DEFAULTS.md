# Zing 流水线默认配置

## 使用与行为

以 `4adab592c6`（main，已合入 #455）为基线，将已有流水线开关设为默认。
远程 Qwen、远程 Source Encoder、独立 Decoder 的部署无需再重复传入下表参数。
本地编码不要求额外启动远程服务。必要的模型、服务地址、认证和部署 profile
仍由现有启动配置提供；本 PR 不创建服务或猜测端口。

| 参数 | 省略时的有效值 |
| --- | --- |
| `--zing-conditioning-prefetch` | remote conditioning 且单活跃会话时 true；local 时 false |
| `--zing-source-encoder-mode` | 存在 remote Encoder 时 overlap；local 不注入 remote 配置 |
| `--zing-source-encoder-prefetch` | remote Encoder overlap 时 true；serial 时 false |
| `--zing-source-async-output` | true |
| `--zing-source-decoder-overlap` | 独立 remote Decoder、quantum=1、单活跃会话时 true |
| `--zing-source-work-quantum` | 1 |
| `--zing-source-joint-forward` | cross_chunk |
| `--zing-source-defer-final` | true |
| `--zing-source-prompt-joint` | true |
| `--realtime-max-sessions` | 1 |
| `--realtime-max-sessions-per-worker` | 1 |

例如在已有四卡启动命令中删除上述显式参数，仍采用同样的流水线策略。
需要串行对照时，在原命令追加：

```bash
--zing-conditioning-prefetch false \
--zing-source-encoder-mode serial \
--zing-source-encoder-prefetch false \
--zing-source-async-output false \
--zing-source-decoder-overlap false \
--zing-source-work-quantum 0 \
--zing-source-joint-forward off \
--zing-source-defer-final false \
--zing-source-prompt-joint false
```

其中显式 encoder mode 仍要求 remote Encoder。quantum=0 时必须同时关闭
defer-final；显式启用不受支持的远程优化仍沿用原有报错。
CLI 显式值优先于 server JSON，新默认值覆盖旧 profile 中的调度值。
quantum/defer-final/prompt-joint 默认值也优先于旧环境变量；需要环境变量回退的
程序化调用可显式传 None，生产启动建议直接用 CLI。

#455 已将 Block0 cache 改为默认 `--zing-block0-cache-mode on` 和 host/disk
字节预算。本 PR 按用户确认沿用该契约，不恢复已删除的
`SGLANG_ZING_BLOCK0_CACHE_MODE=overlap` / `SGLANG_ZING_BLOCK0_CACHE_ENTRIES=2`。
启用 KV pool 的部署仍须按真实资源配置 cache host/disk GB 和已挂载目录；
本 PR 不自动设置容量。其他算子、窗口、采样和硬件默认值见
[性能默认配置](../specs/PERFORMANCE_DEFAULTS.md)。

## 决策和调用链

复用 ServerArgs 声明和 StoreBoolean；不新增开关、调度器或缓存状态。
CLI/from_dict → ServerArgs → Source adapter.on_init → profile 副本及
source command → 既有 worker 执行路径。导出的 profile 文件不被修改。

远程能力在 adapter 解析部署 profile、注入 disagg 端口后才完整可见，
所以 encoder mode/prefetch、conditioning prefetch、decoder overlap 的 None
表示按能力选择默认值。直接在公共 ServerArgs 上把所有布尔值设为 true
会让 local 默认启动触发现有 remote-only 校验，因此不采用。
显式 false、serial、off 始终保留；不以自动默认值吞掉显式错误。

公共会话容量默认保持 8，非 Zing 模型的显式容量继续生效；Zing 通过既有
`_validate_batching()` 校验将两项有效容量限定为 1。资源 owner、取消、EOS 和缓存淘汰
均沿用现有实现，本 PR 不改变状态所有权。

## 验证

验收条件：省略参数可解析；local 不创建 remote 资源；remote 编码/解码启用
对应优化；worker command 收到 q1、defer-final、prompt-joint 和 async output；
显式 false/serial/off 与 server JSON/CLI 覆盖仍生效。

实际结果（2026-10-29）：WSL / Python 3.12 / torch 2.14.1+cpu，基线
`4adab592c6` 加本 PR 的四个 Python 文件改动，227 passed、3 subtests passed，
1 条既有警告，15.78s。覆盖真实参数解析、profile 合成、command 序列化及
预取发布/关闭；补测前一轮为 226 passed。Ruff 0.15.1 check/format、
isort 7.0.0 和 diff whitespace 检查通过。

```bash
SGLANG_DIFFUSION_PLATFORM_OVERRIDE=cpu python \
  benchmark/zing_h3_causal/run_conditioning_cpu_tests.py \
  python/sglang/multimodal_gen/test/unit/conditioning/test_pipeline_cli.py \
  python/sglang/multimodal_gen/test/unit/realtime/test_zing_h3_source_adapter.py \
  -q --disable-warnings --tb=short
```

Windows 原生测试因 Linux `resource` / 本机缺少 cloudpickle 无法收集；
WSL 首次未指定 CPU platform 时误选 CUDA，失败于 CPU wheel 的设备探测。
以上通过结果使用显式 CPU platform，没有修改产品的设备探测逻辑。
CPU harness 屏蔽 GPU 加载边界，不能证明 GPU 数值、性能或长期稳定性。
CPU阶段未运行GPU；其后用户授权的480/SP2追测见下节。736/SP1和默认window21
仍未通过实际模型验收，不宣称相对旧版本的FPS、TTFF或显存收益。

公共容量作用域修正后的定向回归：pipeline CLI / Source adapter 共
229 passed、3 subtests passed（1 条既有警告）。包含非 Zing 默认 8、显式容量覆盖、
真实 Zing ServerArgs 完整校验后的有效值 1，以及校验后的默认策略下发。
该修正没有改变 Zing 的既有容量归一化或 worker 执行代码，未重新运行 GPU。

## 默认配置验收结论与运行修复

本 PR 的配置目标已达到：省略“使用与行为”表中的 11 项参数，真实
ServerArgs、adapter 合成的 profile 和下发的 worker command 均采用预期默认值。
远程专用默认值按能力条件解析，本地路径仍可使用；显式 false/serial/off 保留。
远程 Qwen 单会话的 conditioning prefetch、remote Encoder overlap/prefetch、
独立 Decoder overlap 由现有能力条件启用；worker 实际策略确认
async output、quantum=1、cross_chunk、defer-final、prompt-joint 生效。
Zing 校验后的两项有效会话容量均为 1，公共默认仍为 8；Block0 沿用新版 on 和 host/disk 字节预算。

真实 GPU 使用用户指定的 window6/SP2、离线 AdaLN，省略上述参数，完成
预热和正式请求，连续输出及 prompt 更新均成功；两遍完整 RGB 一致。
这验证默认配置已接入可运行链路，不代表每次请求都触发全部预取或 Graph 路由。
热缓存可能跳过 conditioning 编码，是否配置启用与某次请求是否实际执行应分开判断。
本次不以性能数值作为默认配置变更的验收依据；736/SP1和window21未覆盖。

预热发现既有 Block0 fill 用 CPU `prefix.positions` 选择 CUDA stream，
导致首帧后会话失败。修复改为使用缓存已有 transfer stream 所属设备等待 producer，
未增加复制或同步；回归覆盖 CPU positions、KV 内容和异步完成后发布。
修复后的真实连续推理和 prompt 切换通过；相关 CPU preparation-cache 回归通过。

## 社区与 reviewer 导读

目标是 seedleap/sglang main 的已有 Zing 扩展，不是向 sgl-project/sglang
新增远程调度机制。核对本基线的
[Diffusion 贡献指南](../../contributing.mdx) 和相邻 ServerArgs/StoreBoolean
实现，复用声明与显式 CLI 优先的方式。未核实社区上游是否接受这些部署默认值，
也未将团队单会话策略当作社区通用约束。

推荐依次检查本说明、`runtime/server_args/server_args.py` 的字段与
`add_cli_args`、`zing_h3_source_adapter.py` 的 `_with_source_pipeline_policy`
和 `on_init`，最后看 `conditioning/test_pipeline_cli.py` 与
`realtime/test_zing_h3_source_adapter.py`。代码路径均相对
`python/sglang/multimodal_gen/`（adapter 位于
`runtime/entrypoints/openai/realtime/adapters/`）。
重点确认 local/remote 能力边界、显式关闭、旧 profile 覆盖语义及 Zing 专属容量；
CPU 证据不替代部署硬件上的 480/736 验收。
