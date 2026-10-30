# 产品真实请求回放集合

自 2026-10-30 起，`product-case-20261029` 是 Zing / SGLang 测试的**默认集合**。
`bench_streaming_source.py` 不传选择参数或使用 `--case all` 即运行此集合；
原四个 `whitebox-basic` case 保留为显式选择的旧集合。
输入来自北京时间 2026-10-29 上海与泰国部署的真实生成归档；逐条定义见
[`product_case_20261029_cases.jsonl`](product_case_20261029_cases.jsonl)。每条保留 recording ID、
原 manifest / inputs URI 与 SHA256、完整 prompt、图片顺序及资产字节数 / SHA256。
媒体保留在原有内部 OSS 访问范围内，不提交或公开上传视频、参考图。

## 选择与覆盖

| case ID 后缀 | 原 recording ID | 帧数 | 回放时长（24 FPS） | prompt 字符数，逐版本 | 参考图数，逐版本 |
| --- | --- | ---: | ---: | --- | --- |
| `few-images` | `7a3c0ea9d0da42dca1ad8254b0d9c755` | 403 | 16.792 秒 | 1088 | 1 |
| `long-prompt` | `8b63cb6d483c464da388313595788a8e` | 6247 | 260.292 秒 | 9122 → 10736 → 11335 → 11678 → 10799 | 3 → 3 → 3 → 3 → 3 |
| `reference-switch` | `606af3a572244a18a808bf227426cb6f` | 764 | 31.833 秒 | 2556 → 1675 → 2282 → 1675 → 2239 → 1675 → 2239 | 3 → 1 → 2 → 1 → 2 → 1 → 2 |
| `long-session` | `07567e68cb4949708690b2a53a5aeb66` | 15921 | 663.375 秒 | 5504 → 5504 → 5504 → 5504 | 5 → 5 → 5 → 5 |

完整 case ID 为 `zing-product-case-20261029-<后缀>`。字符数是 `len(prompt)`，不是 token 数；
性能分析还需采集实际 Qwen context / DiT prefix rows。长会话保留同长度 prompt 下的
真实 reference 替换与重复版本，不因文字相同而删除事件。

这些输入均为 **832×480**。736 档尚未生成对应冻结输入或运行验证，不能以本集合代替
736 验收；后续补充须使用单独 case 身份并记录实际尺寸与来源，不能静默改写本集合。

## 回放语义

- source 使用每条 render segment 的 `whitebox.mp4`（与输出配对的真实白模画面），
  **不使用生成的 `model.mp4`**，完整逐帧解码，不循环、不裁切、不补帧。
- `inputs.jsonl` 的 `start_frame` 是原输出区间边界。本集合将它映射为配对白模视频的
  解码帧号，再作为回放的 `source_frame_start`；保留所有实际进入输出的条件版本。
  `reference-switch` 的六次更新发生在帧 `34, 289, 374, 544, 561, 629`。
- 这不是原始浏览器 source 流的逐包重演，不恢复原请求发送时间、未生效更新、
  浏览器机位采集停顿或网络延迟。原 VFR 时间轴保留在 provenance 中；例如长会话的
  source PTS 跨度为 704.166105 秒，归档创建至结束为 708.647119 秒；15921 帧
  按 24 FPS 回放对应 663.375 秒。PTS 终点不是时长，必须减去首个输入区间的
  PTS 起点；这三种口径分别记录在 provenance.archive_timing 中，不混称。
- init 保留原 reference ID；每次更新发送完整 reference 列表，列表缩短表示删除旧图。
  prompt 原文不改写，额外图片仅用于准备更新素材，不提前加入初始 reference 条件。
- 使用连续 source 入口。有限输入客户端会明确拒绝此集合，避免静默丢弃换图事件。
  回放发送边界固定；服务实际应用边界仍由协议决定，以 `prompt-updates.json` 的
  回执为准，不宣称重现原服务的逐帧输出或图像质量。

旧集合回归须显式传 `--case-set whitebox-basic`，单个旧 case 可使用 `--case 5s` 等参数。
有限输入工具不支持动态回放，不指定集合时会提示使用连续 source 入口，不能静默回退旧集合。

## CPU 素材准备与预检

使用 Python 3.11+，依赖沿用连续 source 客户端（msgpack、websockets、Pillow），
并需要 FFmpeg / FFprobe、已授权的 Aliyun CLI profile。
以下操作只读原对象，下载到指定缓存，核对全部文件字节数与 SHA256，再完整解码视频
检查宽高和实际帧数；已有损坏缓存明确失败，不静默覆盖。

```bash
python3 benchmark/zing_h3_causal/prepare_product_cases.py \
  --profile shengdong --media-root /data/zing-product-case-media

PYTHONPATH=python SGLANG_LIGHTWEIGHT_RUNTIME=1 \
python3 benchmark/zing_h3_causal/bench_streaming_source.py \
  --media-root /data/zing-product-case-media \
  --output-dir /data/product-case-preflight --preflight-only
```

准备成功回执为缓存根目录的 `product-case-20261029-assets.json`。预检生成每条 case 的
`streaming-case.json` 与本地路径版本 `prompt-updates.json`，包括所有后续换图素材。
预检不连接模型服务，也不创建 GPU 作业。输出目录须为新目录。

## 连接已部署的连续 source 服务

```bash
PYTHONPATH=python SGLANG_LIGHTWEIGHT_RUNTIME=1 \
python3 benchmark/zing_h3_causal/bench_streaming_source.py \
  --url ws://127.0.0.1:30000/v1/realtime_video/generate \
  --media-root /data/zing-product-case-media \
  --output-dir /data/product-case-streaming-run --seed 42 --paced \
  --timeout 1800 --warmup-runs 0 --measured-runs 1
```

上述是单轮检查命令；正式同条件对比使用相同预热 / 重复次数并分别报告每个 case。
客户端会保存原尺寸 RGB：四条合计每轮约 **26.04 GiB**，一轮预热加三轮正式约
**104.15 GiB**，另加日志与报告空间。长会话是完整输入，不能为减少开销改用短片循环。
`--paced` 为每个解码帧赋予 24 FPS 回放节奏，不保留原视频 VFR 墙钟停顿；
`--paced` 与不加该参数的结果按工具已有 `input_mode` 分开统计。

阶段、显存和性能验证沿用 [客户端 benchmark 规范](CLIENT_BENCHMARK.md)，包括
baseline/candidate 的 Encoder、DiT、Decoder 数据，不能仅凭 CPU 预检或素材完整声明
GPU 生成、换图正确性、画质、24 FPS 或性能回归通过。当前 GPU 验证状态为**未验证**。
