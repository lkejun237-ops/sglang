# Zing 视频固定 480p 编码与参考图的可选缩放

这个功能减少条件输入的分辨率，生成视频仍使用用户请求的尺寸。
对应 [#375](https://github.com/seedleap/sglang/issues/375) 和
[#312 第 6 项](https://github.com/seedleap/sglang/issues/312)。

- **source**：用户持续发送的视频帧。实时 `source_stream/source.append` 始终应用 480p 编码上限，没有关闭开关。
- **reference**：请求中选作参考条件的图片。它的开关独立生效。

## 怎么使用

在 Zing 的 PipelineConfig 中设置：

```json
{
  "zing_reference_resize": {"max_short_edge": 480, "max_area": 399360}
}
```

source 固定使用短边 480、面积 399360 的编码上限，不提供原分辨率编码选项。
reference 默认是 `null`，表示不增加额外缩放限制；设置上述配置后独立生效。
配置非法或小到放不下一个模型网格时，在启动或建立会话时拒绝。

`480` 是短边上限，`399360` 是面积上限（832×480）。沿用模型的网格对齐、
宽高比选桶与 Lanczos 缩放规则，不放大小图，也不把所有图片拉伸为同一尺寸。
source 与 reference 仍受模型限制；参考图的请求面积上限继续生效。

实时请求的 `size` 指定输出，`source_stream.width/height` 指定实际上传尺寸：

| 实际上传 | VAE 编码 | 请求的输出 |
|---|---|---|
| 832×480 | 832×480 | 832×480 |
| 1280×736 | 832×480 | 1280×736 |
| 832×480 | 832×480，不再放大 | 1280×736 |

上传尺寸可以与输出尺寸不同；不放大小输入。`session.ready.source_encoding`
会返回实际编码宽高及是否发生缩放，客户端按原协议上传和接收即可。

## 参考图怎么处理

只给 `target.conditions` 选中的 image 增加短边和面积上限。
例如 1280×720 图片选到832×480，1024×832 图片选到608×480。
视频、音频参考和其他媒体沿用当前 main 的处理。

在 `media_input.py` 中先算出最终限制，再复用已有选桶和 resize；
`codecs.py::prepare` 把同一份归一化参考图交给 Qwen、VAE 与后续 prompt 更新。
**当前 main 本来就共享这份归一化图**；关闭 reference 开关表示沿用 main 的默认预处理。
参考图缓存使用相同的有效短边/面积上限选桶，并将这些上限纳入预处理 key。
这样从原图直接缩到最终尺寸，缓存命中与关闭缓存保持相同像素，避免先按默认尺寸缩一次再缩到480p。
固定 source 编码策略与 reference 缩放配置也纳入持久化 Block0 缓存身份。修改 reference
配置或升级 source 策略会使用不同的缓存命名空间，避免恢复旧参考图条件、prefix KV
或旧 source 准入尺寸。

## source 为什么不只是加一次 resize

原来的实时路径共用 source 与输出尺寸。缩小 source 后，需要分清三个用途：

| 尺寸 | 用在哪里 |
|---|---|
| 实际上传尺寸 | RGB 字节数、流控额度、帧序号与输入身份校验 |
| VAE 编码尺寸 | source latent、source token 与 KV 行数 |
| 请求输出尺寸 | target 初始噪声、target latent 与 Decoder 输出 |

`SourceGeometry` 保存这三种尺寸，归当前会话所有。adapter 解析，worker 按模型配置复核。
共享 Encoder 在校验原始帧后、VAE 前缩放；独立 Encoder 的普通和预取入口汇合后再缩放，
并通过 RPC 确认编码尺寸。原始输入的帧数、时间对齐、身份和流控不变。

`SourceChunkBridge` 按输出网格生成 target noise，避免 source 缩小后输出也跟着变小。
KV pool 按 source/target 各自行数做准入，保留容量边界检查。
固定池的预留空间和动态 KV 倍增策略不变，所以 token 减少不代表整卡峰值一定下降。
取消、断连、EOS 与下一会话复用继续沿用已有生命周期。

## 如何验证、先看哪里

| 要保证的行为 | 主要代码与回归测试 |
|---|---|
| source 默认固定 480p、网格对齐、面积上限、不放大 | `media_geometry.py`；`test_zing_h3_media_contract.py` |
| Qwen/VAE/prompt 更新共用参考图 | `media_input.py`、`codecs.py::prepare`；媒体契约测试 |
| 上传、编码、输出尺寸分别正确 | adapter、`source_session.py`；adapter/progressive session 测试 |
| 输出噪声不受 source 尺寸影响 | `source_bridge.py`；`test_zing_source_bridge.py` |
| mixed-grid KV 容量安全 | `zing_kv_pool.py`；KV pool 与 runtime 测试 |
| 独立 Encoder、预取、错误与清理 | `remote_encoder.py`、`encoder_worker.py`；cross-dispatch 与 abort/recovery 测试 |

2026-10-29 固定 source 480p 上限后，媒体、adapter、progressive session、bridge 与 Block0
缓存相关回归共1404项CPU测试通过。覆盖两档输出与三种上传尺寸的组合；GPU效果、性能及显存未验证。

以下为固定 source 策略前的历史验证：2026-10-29 合并 main `4adab592c602` 后，媒体、adapter、progressive session与bridge共1330项CPU回归通过，
包括新增的reference缓存像素一致性检查。额外Encoder/KV分组受本机内存问题中断，未取得完整结果；未重跑GPU。

2026-10-26 Linux CPU 回归：主分组1535项通过；最终媒体用例144项重跑通过（包含在主分组中）；
KV runtime/cache 分组58项通过、6项CUDA测试跳过。CPU 张量、协议及小模型测试不替代
完整模型的多 GPU、Graph、480/736 输出和视频质量验收。已有历史 w6/s3 测试及限制保留在
[PR #370 的测试结果](https://github.com/seedleap/sglang/pull/370)，不视为当前代码验收。

建议先读配置与 `media_geometry.py`，再沿 adapter → session → Encoder 看输入，
最后检查 bridge 和 KV。review 重点是三种尺寸没有混用、reference 关闭策略兼容当前 main、
原始流控不变，以及 prompt 更新和失败恢复仍正确。

配置沿用上游的模型专属
[PipelineConfig 与 validate_server_args](https://github.com/sgl-project/sglang/blob/fd5c52588ff2399d25d31e49cf2121863cf620e3/python/sglang/multimodal_gen/configs/pipeline_configs/base.py)
扩展点（2026-10-26 核对），缩放限制只加入 Zing，不新增通用服务端开关或修改 scheduler。
上游合入仍需审查默认兼容、SP/RPC 一致性、CPU 缩放开销和目标硬件的实际验证。
