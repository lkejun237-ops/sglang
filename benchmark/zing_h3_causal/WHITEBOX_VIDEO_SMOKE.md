# Zing H3 白模视频基本测试

本目录固定一个贴近实际输入场景的请求，用于 Zing H3 的基本测试：

| 用例 | 源素材 | 参考图 | 输出规格 |
| --- | --- | --- | --- |
| `zing-whitebox-video-five-triview-0001` | 5 秒白模游戏视频，1280×720、24 fps | 五张合并的白模／目标外观三视图 | 120 帧，832×480 |

源素材是游戏引擎录制的低多边形海岸场景，人物在场景中行走。五张参考图分别定义人物、巡逻车、站点、灯塔和瞭望台的外观。
用户消息通过 `clip.start=clip.end=0` 仅向 Prompt encoder 提供源视频第 0 帧；
`target.source` 则向 Source encoder 提供完整的 120 帧视频。这体现了首帧建立场景语境、后续源像素帧提供运动信息的输入分工；当前请求仍一次传入完整视频。

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
  --output-dir /data/zing-smoke/whitebox-video-five-triview
```

若已有存储桶的本地挂载或镜像，可用 `--media-root` 指定存储桶根目录，直接复制本地素材。
该目录下需保留 `world-model/...` 对象键对应的目录层级：

```bash
python benchmark/zing_h3_causal/prepare_whitebox_video_smoke.py \
  --media-root /mnt/leap-worldmodel-thailand \
  --output-dir /data/zing-smoke/whitebox-video-five-triview
```

准备工具按固定的字节数和 SHA256 核验每份素材，然后写出：

- `request.json`：可直接运行的 Zing 双消息请求，素材 URI 改为本地 `file://` 路径。
- `media/`：源视频与五张三视图各一份。
- `receipt.json`：源 URI、本地 URI、字节数、SHA256 和训练来源记录。

使用 `--keep-remote-uris` 可保留远程 URI，仅写出请求与清单，不复制媒体。
这种模式不执行媒体下载或校验，推理环境仍需具备对应存储桶的访问权限。

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

该用例验证完整源视频请求，采用与训练一致的 Prompt／Source 分工。
它不提供离线质量评分，也不验证增量 `source.append`：当前 Zing SGLang 接口在分块输出前仍需接收完整的源对象。
