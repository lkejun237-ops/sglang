# Zing H3 30 秒白模输入

本清单冻结 UniEdit `whitebox-long-v1` 中两条真实 30 秒输入，以及潇潇提供的一条 WTP 效果检查输入。
命名沿用 `zing-whitebox-video-…-0001`；独立清单为 `whitebox_video_long_cases.jsonl`，默认 5 秒 case 不变。

| Case ID | 场景 | 全局图之后的三视图顺序 |
| --- | --- | --- |
| `zing-whitebox-video-30s-six-triview-0001` | 写实废土 | 拾荒者、马、房屋、瞭望塔、路灯、树 |
| `zing-whitebox-video-30s-six-triview-0002` | 动漫地中海小镇 | 三轮车、蔬菜箱、石桥、摊位、房屋、花箱 |
| `zing-whitebox-video-30s-two-triview-0001` | 刺绣霓虹漫画、室内通道 | 主角（player）、林姐（accounts-colleague） |

前两例均为 **source + 七张图片**：Picture 1 为全局外观参考，Picture 2–7 为六张主体三视图配对 sheet（上排白模、下排目标外观），对应七项 visual reference conditions。
这与默认 5 秒输入的「全局图 + 五张三视图」六图契约不同，不能丢图、重编号或混用基线。

## UniEdit 来源与请求语义

来源固定为 `seedleap/zing` 提交 `32846761dd8d696c80b174137a98ea86433c498d` 的
`src/eval/datasets/uniedit/whitebox_long_v1/uniedit_whitebox_long_v1.jsonl` 第 3、8 行。
原始 sample ID、清单 SHA256、转换记录和每项素材的字节数／SHA256 保存在 provenance 中。
仅重命名 case、将对象键展开为实际核验的 `s3://leap-worldmodel-thailand-single-az/` URI、移除专属 processor cache URI；英文 prompt、图顺序、conditions、输出规格保持原样。
仓库不保存视频和图片二进制文件；素材读取需要该桶的访问权限及正确的 OSS S3 endpoint，或已有本地镜像。

实测 source 为 1280×720、24 fps、720 帧、30.000 秒，**无音轨**；两套共 16 项素材完整解码通过。
原请求输出为 `video`、832×480、720 帧，音频未启用。
Prompt 中的 `<Video 1>` 和 `target.source` 共用完整 source，不设 `clip`，保留原 BIDIR 全片描述。
它不是默认 5 秒输入的 prompt 首帧路径，也不是 target-frame0 r2 变体；若另做 AR 首帧输入或 10 秒截取，必须另建 case/revision 并核验 prompt 与素材，不能覆盖本清单。

## 准备与运行

现有准备器按请求动态遍历素材，支持这两例的八项输入（一个 source + 七图），无需改动准备器。

```bash
python benchmark/zing_h3_causal/prepare_whitebox_video_smoke.py \
  --manifest benchmark/zing_h3_causal/whitebox_video_long_cases.jsonl --list

python benchmark/zing_h3_causal/prepare_whitebox_video_smoke.py \
  --manifest benchmark/zing_h3_causal/whitebox_video_long_cases.jsonl \
  --case zing-whitebox-video-30s-six-triview-0001 \
  --media-root /mnt/leap-worldmodel-thailand-single-az \
  --output-dir /data/zing-smoke/whitebox-video-30s-six-triview-0001
```

`--media-root` 指向保留 `world-model/...` 对象键层级的桶镜像。省略它时，准备器通过 fsspec 与环境已有 S3 凭证读取；`--keep-remote-uris` 只输出远程请求，不执行素材校验。
切换第二例须显式指定 `--case zing-whitebox-video-30s-six-triview-0002` 并使用独立 output-dir。
输出为 `request.json`、八份 `media/` 素材及 `receipt.json`，所有本地媒体须通过固定字节数与 SHA256 检查。

```bash
PYTHONPATH=python python benchmark/zing_h3_causal/validate_real_checkpoint.py \
  sglang \
  --request /data/zing-smoke/whitebox-video-30s-six-triview-0001/request.json \
  --output /data/zing-smoke/results/whitebox-video-30s-six-triview-0001 \
  --sglang-root "$PWD" --model /path/to/zing-ar-export \
  --sp-size 1 --sigma-points 50 --solver euler
```

该入口从请求读取输出帧数、尺寸并传入完整 `zing_request`，不写死五图／六图。
本次只验收资产、请求准备与七图映射，**尚未执行 native/SGLang GPU 推理**。
720 帧生成、七图模型适配、显存、画质、数值与性能均待匹配权重实测；不能据此宣称离线或流式端到端通过。
增量 source、持续 source 像素帧消费和音视频能力需要单独验证。

## WTP 效果检查用例

`zing-whitebox-video-30s-two-triview-0001` 对应世界
`world-ec2bb6fa4c7c3c6d1312f7d601bbedca`、WTP run
`run-muu2ogwl-2e6159c1dc696d4a`，Release 为
`prod-auto-r6-v535-516975f0ab35`，SDK 提交为
`b229d02b1c342852575f11ab4b0642c2aecfa89e`。
冻结素材包为 `mat-stream-20261010-4605ec707f55b165fc057637`，
clip 为 `B1-002-v3-c1`，style 为 `B1-002-v4-s5`。

此例保留实际 **source + 三图**，不补齐默认六图或 UniEdit 七图：
Picture 1 为全局样式首帧，Picture 2 为主角、Picture 3 为林姐的双排三视图
（上排白模形体、下排目标外观），对应三项 visual reference conditions。
使用实际裁切视频 `19b458b4…fd182.mp4`，而非原始录制 `cb9afc67…e839.mp4`；
source 实测为 1280×720、24 fps、720 帧、30.000 秒，无音轨。
Prompt 仅转换 `@视频1`／`@图片1–3` 为 Zing content ID，其他文字原样保留。
与默认六图 AR 请求一致，`clip.start=clip.end=0` 只向 Prompt encoder 提供 source
首帧；`target.source` 保留完整裁切视频。Picture 1 仍是独立图片 reference，
没有替换 prompt 视频或走 target-frame0 r2 路径。
效果检查输出设为 `video`、832×480、720 帧、24 fps。
原 Prompt 的自然收音要求保留，但此例未启用音频，不能作为音视频能力验收。

四项推理输入，以及世界包、世界 manifest、原始录制、操作文档 JSON／Markdown、
冻结素材包和原 Prompt 均已从北京桶 `leap-worldmodel-beijing-single-az` 回读并核对
字节数与 SHA256；视频／图片完整解码通过。`provenance.assets` 仅列四项推理输入，
`provenance.evidence` 保留八项交付证据的稳定 URI 与校验和，准备器不下载重建世界所需的证据。
网页链接返回的部分 JSON 与 OSS 原件字节不同；冻结记录以用户给出的 OSS 原件为准。

```bash
python benchmark/zing_h3_causal/prepare_whitebox_video_smoke.py \
  --manifest benchmark/zing_h3_causal/whitebox_video_long_cases.jsonl \
  --case zing-whitebox-video-30s-two-triview-0001 \
  --media-root /mnt/leap-worldmodel-beijing-single-az \
  --output-dir /data/zing-smoke/whitebox-video-30s-two-triview-0001
```

省略 `--media-root` 时使用北京桶的 OSS S3 endpoint 和已有授权凭据。
输出为 `request.json`、四份输入媒体及 `receipt.json`，可交给上面的验证入口，
替换 request 与 output 路径。后续 H3 推理须核对匹配权重／冻结配置采用
`num_frame_per_block=[3,2]`、`split_first_latent=true`，实际分块为 `1,2,2` 后循环 `3,2`。

复现状态为 `evidence-complete-not-reexecuted`：历史 recorder 未记录 seed，
复现依赖逐帧键盘输入、逐帧相机、SDK 调用日志、起始状态及 1/60 秒固定 timestep。
未来推理的采样 seed 须单独记录，不能冒充 recorder seed，也不声称世界已确定性重放。
操作证据中的林姐跟随动作有 `FOLLOW_NOT_EFFECTIVE` 记录；效果检查以实际视频为准，
不把计划中的动作当成已发生动作。

本次只完成资产与请求准备，尚未执行 native／SGLang GPU 推理。
人工效果检查关注首帧构图、相机与人物根轨迹、主角身份、林姐入镜／离镜时序、
双排参考的下排外观采用，以及长时间闪烁／身份漂移；未提供生成质量达标结论。

素材查看页仅用于本地临时 review；在线交付须另行核实办公网／VPN 访问限制，不使用匿名公网托管。
