# Zing H3 白模视频基本测试

本目录保留两个独立输入 case，不能仅凭 input_revision 区分或混用测试结果。
默认 `zing-whitebox-video-five-triview-0001` 保留 PR #164 的六图契约：全局风格图为 Picture 1，五张三视图为 Picture 2–6，原 prompt 与 conditions 映射不变。
本文件以下介绍显式选择的训练 target 首帧变体；它不覆盖默认六图 case。

另有独立的 [30 秒白模输入](WHITEBOX_VIDEO_LONG.md)，使用
`whitebox_video_long_cases.jsonl` 和显式 `--case` 选择：两条 UniEdit 输入保留全局图 + 六张三视图，
潇潇提供的 WTP 效果检查输入保留全局图 + 主角／林姐两张三视图。

| 用例 | 源素材 | 参考图 | 输出规格 |
| --- | --- | --- | --- |
| `zing-whitebox-video-target-frame0-five-triview-0001` | 5 秒白模游戏视频，1280×720、24 fps | 五张合并的白模／目标外观三视图 | 120 帧，832×480 |

源素材是游戏引擎录制的低多边形海岸场景，人物在场景中行走。五张参考图分别定义人物、巡逻车、站点、灯塔和瞭望台的外观。
输入修订版 **r2**：用户消息的首个媒体是训练 **target** 视频第 0 帧，
保留 `type="video"` 和 `clip.start=clip.end=0`，仅供多模态文本编码器按视频处理；
`target.source` 则向 Source encoder 提供完整的 120 帧视频。它与首个 prompt 视频是两个不同素材。这体现了渲染首帧建立场景语境、白模源像素帧提供运动信息的输入分工；当前请求仍一次传入完整视频。

## 训练数据来源

素材和提示词来自训练数据源 `h3_whitebox_multiview_v1_segment_causal_260918`，原样本 ID 为：

```text
night-human-195-c2bb31df54__style-04__segment-05__attempt-02__p01416_a1a13ef79363__p01416_59f2bc5a5d46__clip02_10_15
```

原始记录位于 `part-00000.jsonl` 第 10 行，该分片的 SHA256 为：

```text
d24cddb259eb7e22dc9b763648934394e98e0818c80a744f79b2fc947176cbed
```

训练记录还包含第六个条件：一张已经渲染的目标风格参考帧。为了贴合本期实际输入，基本测试请求进行了以下转换：

- 移除这张额外参考帧及提示词中对它的引用，保留五张合并三视图。
- 将相对素材路径转换为完整的 S3 URI。
- 将训练目标视频 URI 替换为明确的推理输出规格。

r2 按测试需求将首个 prompt 视频改为训练 target 第 0 帧；这不是原始训练记录的原样复制，
也不是恢复原始的额外全局风格图片。五张三视图及其编号保持不变。

这些转换、素材字节数和 SHA256 均记录在 `whitebox_video_smoke_cases.jsonl` 的来源信息中。
仓库仅保存请求与来源记录；媒体保留在现有的 `leap-worldmodel-thailand` 存储桶中。
模型提示词保留英文，以免翻译改变测试输入的语义。

移除原 `<Picture 1>` 后，原 `<Picture 2>`～`<Picture 6>` 对应新
`<Picture 1>`～`<Picture 5>`。重编号须对原文本一次完成，不能从大到小连续
执行字符串替换，否则会将新编号再次替换，最终全部变成 `<Picture 1>`。
本 PR 早期版本出现过此提取错误，原训练记录的映射正确。回归测试同时核对
三视图定义、主体定义、`retention_analysis`、素材文件和 conditions 的对应关系。
若已生成早期版本的 `request.json`，请更新代码后重新运行准备工具。

## 准备请求

列出固定的测试用例：

```bash
python benchmark/zing_h3_causal/prepare_whitebox_video_smoke.py --list
```

将用例素材准备到持久化本地目录。`fsspec` 使用运行环境已有的标准 S3 凭证链：

```bash
python benchmark/zing_h3_causal/prepare_whitebox_video_smoke.py \
  --case zing-whitebox-video-target-frame0-five-triview-0001 \
  --output-dir /data/zing-smoke/whitebox-video-five-triview
```

若已有存储桶的本地挂载或镜像，可用 `--media-root` 指定存储桶根目录，直接复制本地素材。
该目录下需保留 `world-model/...` 对象键对应的目录层级：

```bash
python benchmark/zing_h3_causal/prepare_whitebox_video_smoke.py \
  --case zing-whitebox-video-target-frame0-five-triview-0001 \
  --media-root /mnt/leap-worldmodel-thailand \
  --output-dir /data/zing-smoke/whitebox-video-five-triview
```

准备工具按固定的字节数和 SHA256 核验每份素材，然后写出：

- `request.json`：可直接运行的 Zing 双消息请求，素材 URI 改为本地 `file://` 路径。
- `media/`：白模 source、训练 target、五张三视图，以及从 target 提取的单帧无损视频 `training-target-frame0.mkv`。
- 准备阶段需要 `ffmpeg`；单帧视频保留解码 RGB 像素，不包含 target 后续帧或音轨。推理请求仅引用该单帧视频，不引用完整 target。
- `receipt.json`：源 URI、本地 URI、字节数、SHA256 和训练来源记录。

使用 `--keep-remote-uris` 可保留远程 URI，仅写出请求与清单，不复制媒体。
这种模式不执行媒体下载、校验或首帧预提取；请求保留训练 target URI 和闭区间 `clip={"start":0,"end":0}`，由推理端解码选取第 0 帧。它与本地准备路径具有相同的媒体角色，但性能测量时必须记录不同的读取成本。

## 运行 SGLang

现有的真实权重验证工具可直接读取生成的请求：

```bash
PYTHONPATH=python python benchmark/zing_h3_causal/validate_real_checkpoint.py \
  sglang \
  --request /data/zing-smoke/whitebox-video-five-triview/request.json \
  --output /data/zing-smoke/results/whitebox-video-five-triview \
  --sglang-root "$PWD" \
  --model /path/to/zing-ar-export \
  --sp-size 1 \
  --sigma-points 50 \
  --solver euler
```

仅当模型为匹配的蒸馏权重时，才选择 `--solver few_step` 并配置对应的去噪步数列表。

## 验收范围

基本技术要求是请求执行完成，输出可正常解码为 120 帧、832×480 的视频。
人工查看源相机和人物运动是否保留，并检查从首帧到片段结束，可见主体是否持续遵循对应的三视图参考。

该用例验证完整源视频请求，采用明确分离的 Prompt／Source 分工，输入版本为 r2。
它不提供离线质量评分，也不验证增量 `source.append`：当前 Zing SGLang 接口在分块输出前仍需接收完整的源对象。

## API 字段与 PR #140 接入注意事项

不要使用 `first_frame`，也不要把这个首帧追加为第六张普通 image reference。
这里“第一个参考输入”是 user.content 的第一个媒体，类型必须为 **video**：

```json
{
  "messages": [
    {"role": "user", "content": [
      {"id": "<Video 1>", "type": "video", "uri": "file:///data/media/training-target-frame0.mkv", "clip": {"start": 0, "end": 0}},
      {"id": "<Picture 1>", "type": "image", "uri": "file:///data/media/ranger.jpg"}
    ]},
    {"role": "target", "source": {"type": "video", "uri": "file:///data/media/source.mp4"},
     "content": [{"type": "video", "output": {"frames": 120, "height": 480, "width": 832}}],
     "conditions": [{"content_id": "<Picture 1>", "role": "reference", "latent_modalities": ["visual"]}]}
  ]
}
```

上例仅展示字段位置，省略了 prompt 和另外四张三视图；实际运行使用准备工具生成的完整 `request.json`，放入 `zing_request`。
`<Video 1>` 不进入 target.conditions，因此不会增加 VAE reference 数量。
文本编码器按媒体 type 路由视频／图片；不是只改变列表顺序即可。
当前 SGLang 的 `ZingH3Codecs.encode_condition` 使用 Qwen video processor；不要仅凭此请求格式声称已经使用 vLLM 后端。

PR #140 的持续 source 接口当前会自动取首个 source RGB 帧作为 prompt 视频。
这与 r2 的训练 target 首帧不同，不能原样复用该行为，也不能把渲染首帧塞进 source.append。
该接口需另行接入独立的 prompt 视频首帧，并在五张普通 references 之前按 video 路径处理。
本 PR 修正冻结输入和完整视频准备路径，不宣称 PR #140 已兼容或已通过 GPU 验收。

两个 sample_id 的 request、receipt、性能基线及数值快照不得混用。#166 初版曾复用默认 ID；请重新准备，首帧变体必须记录独立 sample_id 与 input_revision=2。

## 内部素材预览

[查看输入 r2 与首帧路由说明](https://review-cdn.loopit.com.cn/reports/zing-whitebox-video-case/20260922-r2/index.html)
（仅办公网／公司 VPN）。页面展示训练素材，不是 SGLang 输出报告。
