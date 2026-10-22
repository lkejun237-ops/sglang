# Zing H3 benchmark 工具范围

设计、接口契约与实现方案见 [Zing 设计文档](../../docs/docs/sglang-diffusion/zing/README.md)。
本目录的文档范围遵守 [benchmark 目录约定](../README.md)。

本目录提供 Zing causal/AR 的开发、benchmark 和验收工具，不由推理运行时导入。
当前部署使用 Lynn streaming decoder；`decoder_config` 必须提供其 checkpoint 和 model_dir。
TAEH3 运行时、专用脚本和测试已移除；历史验收记录仅对应各自冻结的 Git 版本。

- `run_benchmark.py`：加载 checkpoint 配置，准备媒体并运行完整请求。
- `run_encoder_overlap_matrix.py`：分离 source encoder、DiT、Lynn decoder 的实验矩阵。
- `benchmark_lynn_replay.py`：按捕获的 chunk 重放 Lynn decoder，检查重复输出并独立计时。
- `validate_lynn_sp_tiles.py`：Lynn 空间 tile 的 SP 正确性与生命周期验证。

## 综合 benchmark 入口

`benchmark.py` 只是现有工具的命令索引，参数直接传给对应脚本，不引入调度框架。
完整请求、组件测速、算子 profile 和显存诊断分轮执行，结果汇入同一个离线报告。

| 命令 | 回答什么 | 实现与说明 |
|---|---|---|
| `request` | 完整请求的 TTFF、E2E、TTFF 后 FPS | `run_benchmark.py`，[客户端说明](CLIENT_BENCHMARK.md) |
| `source` | 连续 source 的交付、输入节奏和 prompt 更新 | `bench_streaming_source.py`，[连续 source](CONTINUOUS_SOURCE_VALIDATION.md) |
| `dit` | 3/2 phase、source、四步 denoise、clean commit、算子成本 | `bench_dit_phases.py`，[计时说明](STAGE_TIMING.md) |
| `capture` | worker 显存历史、检查点和整卡采样 | `profile_worker.py`，[显存操作](MEMORY_PROFILING.md) |
| `memory` | 新增分配、待释放字节、峰值调用栈与阶段时间轴 | `analyze_memory.py`，[显存操作](MEMORY_PROFILING.md) |
| `report` | 合并比较、切换历史数据、导入/导出 JSON | `build_benchmark_report.py`，[报告格式](STAGE_TIMING.md#离线展示网页) |

```bash
python3 benchmark/zing_h3_causal/benchmark.py request --model-path /path/to/checkpoint
python3 benchmark/zing_h3_causal/benchmark.py dit --help
python3 benchmark/zing_h3_causal/benchmark.py report \
  --include-results '全链路=benchmark/zing_h3_causal/results/issue334-full-chain-20261019.json' \
  --memory-run 'H200=/path/to/replayed-memory' --output report.html
```

连续 source 除原始 `results.json` 外，另存标准 `report.json`，可直接用于 `--include-results`。
不开启诊断时，原有命令、默认测量和结果版本保持不变。算子 profile 和分配历史影响运行节奏，
不参与无采集测速的统计。运行时优化由相应优化 PR 提供，本目录只负责测量与分析。
PR275/283 的历史交付与旧 TAEH3 数据已移至 [BENCHMARK_HISTORY.md](BENCHMARK_HISTORY.md)。

统一 CPU CI 见 `scripts/ci/zing/README.md`。CPU 契约、真实权重 GPU 数值、媒体质量和
端到端性能是分别验证的项目；工具能运行不代表这些验收已经通过。

## 白膜 source + prompt 切换用例

[`prompt_switch_cases.jsonl`](prompt_switch_cases.jsonl) 包含以下三条派生用例。原始基线保留在
[`whitebox_video_long_cases.jsonl`](whitebox_video_long_cases.jsonl)。三例的 user 视频均显式
设置 `clip={"start":0,"end":0}`，只向 Qwen 提供 source 首帧；前两例由原来的整段
视频提示派生，第三例沿用原有首帧提示。初始文本、参考图、白膜 `target.source` 和
输出尺寸保持原样；`target.source` 不设置 clip，保留完整 30 秒运动输入。
三例均为 832×480、720 帧、24 fps，测试 seed 为 42。

| 新 case ID | 原基线（该文件行号） | 第二段的明显变化 |
| --- | --- | --- |
| `zing-whitebox-prompt-switch-wasteland-snow` | `zing-whitebox-video-30s-six-triview-0001`（1） | 晴天棕色荒原 → 暴雪、白色积雪、冷色照明 |
| `zing-whitebox-prompt-switch-village-rain-night` | `zing-whitebox-video-30s-six-triview-0002`（2） | 暖色晴天街区 → 夜晚暴雨、湿地反射、青蓝/品红灯光 |
| `zing-whitebox-prompt-switch-interior-white-cyan` | `zing-whitebox-video-30s-two-triview-0001`（3，来自 PR #299） | 黑红金室内 → 白色陶瓷墙地面、青蓝灯光，主角外套变白 |

目标时间表为零基像素帧 `[0,360)` 使用原 prompt，`[360,720)` 使用新 prompt，
即在 15 秒附近切换一次。采用 PR #303 回放夹具的 **原完整 prompt + `events:` 后缀**
方式：前两例仅替换末尾 `events: N/A` 的占位内容，第三例在原文后新增 `events:`
小节。原始主体定义、场景描述、参考图映射、动作约束和声音要求逐字保留。
WebSocket 的 payload 仍是拼接后的**完整文本**，不能只发送事件那一句。
新增事件只改变天气、照明或表面外观，source 的运动、布局、可见性和镜头路径继续
保持。参考图不切换，仍可能抑制外观变化；表中是测试期望，不是已测得的效果。

例如荒原用例原文末尾的 `events:\nN/A` 改为以下内容，其余原文保持不变：

```text
events:
Large, fluffy snowflakes begin falling across the scene. The snowfall quickly grows into a dense snowstorm. Fresh white snow gradually accumulates on the existing ground, rooftops, tree branches and watchtower, while cold, diffuse light replaces the warm sunlight. The same people and horse continue their ongoing movements, and the camera continues along the original path in <Video 1>.
```

这里的文本小节 `events:` 与外层协议 `type=event, kind=prompt` 是两回事：前者描述
要发生的视觉变化，后者负责把完整更新文本送给服务。初始 `messages` 仍使用原 prompt，
新增事件只出现在 `metadata.prompt_switch.event.payload`，避免从首帧就应用变化。

### 数据格式与格式来源

下面展示数据存储结构；实际媒体 URI、参考图与初始 prompt 见 JSONL。它不是可直接
提交给 source-ready 的 WebSocket 初始化消息，发送前仍需 runner 转换。

```json
{
  "schema_version": 4,
  "sample_id": "zing-whitebox-prompt-switch-interior-white-cyan",
  "messages": [
    {"role": "user", "content": [
      {"id": "<Video 1>", "type": "video", "uri": "原基线白膜视频 URI", "clip": {"start": 0, "end": 0}},
      {"id": "<Picture 1>", "type": "image", "uri": "原基线首帧参考图 URI"},
      {"id": "<Picture 2>", "type": "image", "uri": "原基线主角参考图 URI"},
      {"id": "<Picture 3>", "type": "image", "uri": "原基线林姐参考图 URI"},
      {"type": "text", "text": "原基线完整 prompt"}
    ]},
    {
      "role": "target",
      "source": {"type": "video", "uri": "原基线白膜视频 URI"},
      "content": [{"type": "video", "output": {"frames": 720, "height": 480, "width": 832}}],
      "conditions": [
        {"content_id": "<Picture 1>", "role": "reference", "latent_modalities": ["visual"]},
        {"content_id": "<Picture 2>", "role": "reference", "latent_modalities": ["visual"]},
        {"content_id": "<Picture 3>", "role": "reference", "latent_modalities": ["visual"]}
      ]
    }
  ],
  "metadata": {
    "fps": 24,
    "seed": 42,
    "prompt_switch": {
      "target_pixel_frame": 360,
      "end_pixel_frame": 720,
      "event": {"type": "event", "kind": "prompt", "event_id": 103, "payload": "原基线完整 prompt\n\nevents:\n室内白色与青蓝色变化事件\n"}
    }
  }
}
```

- `schema_version: 4` 和两条 `messages` 沿用本仓库已有白膜 benchmark；运行时由
  [`ZingH3MediaInput._parse_inference_request`](../../python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/zing_h3/media_input.py)
  解析。`target.source` 是全程对齐的白膜轨道，不能只把视频放进 user content。
- `event` 直接采用现有 [`RealtimeEvent`](../../python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py)
  字段；白膜切换测试应对齐 [PR #303](https://github.com/seedleap/sglang/pull/303) 的
  source-ready 路径，由 [`ZingH3SourceStreamAdapter`](../../python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime/adapters/zing_h3_source_adapter.py)
  接收 `kind=prompt` 的字符串 payload。`event_id` 是关联标识，不是帧号。数据中的
  101/102/103 是模板 ID；runner 发送时必须与 `source.append` / `source.end` 共用
  会话内递增的事件 ID 分配器，并记录模板 ID 到实际 ID 的映射，不能直接照搬固定 ID。
- **`metadata.prompt_switch` 及其帧号是本 PR 的 benchmark 约定，不是 SGLang
  服务端内置的定时协议，也不是旧 Wan 的 `text_prompt_interval`。** 三条白膜数据的
  事件后缀、15 秒切换点均为测试编写，没有原始切换标注。后缀拼接规则来自
  PR #303 实验夹具 `event_suffix_test.py::append_event`；大雪事件直接复用同一实验的
  `snow-event.txt`，另外两条事件由 Codex 于 2026-10-14 编写。
  `event_suffix_source` 保存对应 PR、夹具文件名和 SHA-256，便于核对来源。
  `provenance.prompt_switch_derivation` 记录父用例的仓库、完整 commit、文件 SHA-256、
  行号和 sample ID；原始素材来源与 `assets` 的大小/hash 继续保留。前两例追溯到
  Zing `uniedit_whitebox_long_v1` 第 3/8 行，第三例追溯到
  [PR #299](https://github.com/seedleap/sglang/pull/299) 的 WTP `B1-002` 素材与回放证据。
  `streaming_adaptation` 另外记录本次首帧呈现转换；继承的原始 `derivation` 描述父数据
  历史，不代表派生用例仍让 Qwen 查看全片。

复用现有准备器，使用表中的 case ID；准备器会校验并下载已有素材：

```bash
python benchmark/zing_h3_causal/prepare_whitebox_video_smoke.py \
  --manifest benchmark/zing_h3_causal/prompt_switch_cases.jsonl \
  --case zing-whitebox-prompt-switch-wasteland-snow \
  --output-dir /data/whitebox-prompt-switch-wasteland-snow
```

产出 `request.json`（含切换 metadata）和 `receipt.json`。
**它不会执行时间表。当前自动切换 runner 仍未接入，一次性推理只使用原 prompt。**
运行时语义以 [#303 的设计与实现（核对 commit `8bd5aef2`）](https://github.com/seedleap/sglang/blob/8bd5aef2b01ebd059e604fc9028a3ebfb912a0e2/docs/docs/sglang-diffusion/zing/specs/prompt-switch-latent-prefix.zh-CN.md)
为准；该 PR 尚未合入本数据分支，不代表这里已经具备其运行时行为：

- source-ready 入口拒绝直接传入 `zing_request`。runner 必须把素材请求转换为该入口的
  初始化/reference 字段，将白膜解码后通过 source 流协议持续送入同一会话，再发送
  prompt 事件；有限请求能解析成功不等于已经覆盖 #303。三例均可使用 source 首帧 S0
  作为 prompt 视频提示；按流式协议将 source 对齐到 832×480、24 fps，保持 720 帧。
  首个 source packet 自动提供 S0，参考图通过 `references` 保留原 ID 与顺序。
  连续接口不接受独立媒体上传；旧 `user_media` 字段和 `--user-media` 开关已删除。
- `target_pixel_frame=360` 只表达测试希望发生变化的位置。服务不会按该字段排程；
  当前 block 使用旧 prompt 完成，已有最近 **5 个 target latent** 后才在完整边界
  激活新 prompt，历史不足则保持 pending。5 是 latent 数量，不是像素帧数。
- `prompt_switch_prefix_latent_frames=5` 是部署配置；历史由服务从已生成 target 中
  维护，按 `1+2+2` 重建 prefix。数据/事件不上传静态 prefix，也不动态修改这个配置。
- 多次待处理更新采用 latest-wins。验收以输出 block 固化的 `event_id`、
  `prompt_version` 和实际帧范围为准；控制 ACK、墙钟 15 秒或目标帧号都不证明已经生效。
  source 时间线与输出帧必须连续，不能重复输出 prefix 对应的旧帧。

建议与**同样使用首帧提示**、不切换的同 seed 流式基线比较，不与原有限视频的
全片 Qwen 条件结果混用；前两例的 Qwen 条件范围已改变。通过条件为新外观清晰且持续，同时 source 运动与
镜头没有漂移。当前数据分支仅覆盖数据与准备契约；PR #303 有独立回放夹具和实验记录，
不等同于这里已接入通用 runner。本次三条 events 后缀数据尚未完成逐例 GPU 效果验收。
