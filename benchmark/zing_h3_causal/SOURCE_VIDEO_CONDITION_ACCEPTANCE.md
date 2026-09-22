# Issue #167：显式首帧 video conditioning 验收

固定 native revision：`967df2fd539590b4fa88462ddb67276e5f9a8192`。
`src/h3/modules/condition.py` SHA256：
`df600e78ab897c5384371c6e96d7228b18ef4c9be49f918fc46630c06fafd661`。
该 revision 声明 Transformers 5.5.3、PyTorch 2.9.1、torchvision 0.24.1。
记录实际依赖、checkpoint/config/媒体及测试提交哈希；其他环境上的合同测试不能冒充原生环境验收。

## 范围与复用

`type=video, clip={start:0,end:0}` 是显式用户媒体；即使解码仅一帧，仍走视频 processor。
普通首张 image 保持 Picture 1。source 预览和独立 target 首帧是两种请求，不能按位置互换。
VAE 只消费 target.source 的完整时间轴和 target.conditions 声明的 references。

现有 `ZingH3Codecs.encode_condition` 已读取 temporal_patch_size、同步复制像素和时间戳，
关闭 processor 再次采样；不重复实现。PR #148 持有显式实时传输接口，且负责唯一组合 GPU 验收。
六图 image 请求与 Video 1 + 五图请求分别记录 sample_id/request/hash，不能互用结果。

## 验收顺序及门槛

以下数值门槛在本轮真实 feature/E2E 执行前记录；失败保留原始证据，不事后放宽。

1. CPU 合同：单帧 video、普通 image、混排、独立编号、temporal patch 1/2/4、
   首帧与 VAE 输入隔离。mock 只验证路由，不能证明 learned features 一致。
2. 真实 checkpoint processor：补齐后 RGB、pixel_values、token IDs、grid、
   H3 tags、processor mm_token_type_ids、实际 backend RoPE positions/deltas 完全相等。
   记录 processor 调用一次且 do_sample_frames=False。双方使用相同解码媒体的对照
   不额外声称证明 native 媒体解码器一致。
3. Native 重复性：相同输入/seed/config 重复编码与生成，分别记录逐元素差异。
   若重复性本身超出下述跨实现门槛，则本轮不能判通过，先诊断环境。
4. Condition features：固定 hidden_states[50] 的未归一 residual 语义和 BF16 输出，
   shape/dtype 相同、全部 finite；relative L2 ≤ 0.01 且 cosine ≥ 0.999。
5. 端到端：相同请求、实际初始 noise、schedule、采样参数和模型权重，分别执行 native
   与 serving 条件编码及生成。最终 latent relative L2 ≤ 0.05 且 cosine ≥ 0.995，
   shape/dtype 相同且全部 finite。使用相同 decoder 比较像素；native VAE 与 TAEH3
   的输出不能用于声称 conditioning 数值通过。完整解码、帧数、尺寸、FPS 必须符合请求。
   人工画质独立记录，未审阅为 pending；24 FPS/完整质量研究仍归 #131/#153。

relative L2 = `||candidate-native||₂ / max(||native||₂, 1e-12)`，
cosine 对展平的 FP64 张量计算。记录 max_abs/RMSE 作为诊断，不替换上述门槛。

## CPU 对照入口

`audit_source_video_condition.py` 直接加载固定 native 模块并使用真实 checkpoint
processor，截取当前 serving encoder 输入，保存 `preprocessing.pt`、
`padded_pixels.npz` 和 `report.json`。输出目录必须尚不存在。

```bash
PYTHONPATH=python python benchmark/zing_h3_causal/audit_source_video_condition.py \
  --native-source /FROZEN/native_condition.py \
  --model-root /FROZEN/H3_ROOT \
  --runtime-config /FROZEN/runtime.json \
  --request /FROZEN/request.json \
  --output /NEW/preprocessing-audit
```

H3_ROOT 必须具有原生 tokenizer/processor/text_encoder 子目录；从模型清单核对路径，
不猜路径或下载另一套默认 processor。runtime.json 包含实际 condition_encoder_config。
脚本校验 native 文件哈希，读取真实配置；不加载 Qwen/DiT 权重，
所以成功报告的 feature_validation/e2e_validation 仍为 not_run。

## 当前状态

- CPU 媒体合同：72 passed，其中新增 14 项覆盖上述显式首帧/混排/隔离。
- 本地实际 backend RoPE 三个合成输入（video/image/mixed）相等，Transformers 5.12.1；
  仅属工具验证，不是固定 native 环境证据。
- 真实 checkpoint 预处理、native features、真实 E2E：pending，由组合运行补齐。

## 跨环境参考导出

若 serving 镜像不是 Transformers 5.5.3，保留原审计失败报告，不修改 serving 环境。
先用原环境的 `audit_source_video_condition.py --export-decoded-only` 输出到新目录，
只保存真实解码媒体及请求哈希，不重跑 processor 或完整 CPU 矩阵。
随后在独立固定 5.5.3 的 Python 路径下运行：

```bash
PYTHONPATH=/ISOLATED/transformers553 python \
  benchmark/zing_h3_causal/audit_native_condition_reference.py \
  --serving-audit /PREVIOUS/serving-audit \
  --decoded-media /NEW/decoded/decoded_media.pt \
  --native-source /FROZEN/native_condition.py \
  --model-root /FROZEN/H3_ROOT \
  --output /NEW/native-reference
```

该入口不导入 SGLang，也不加载权重；对照既有 serving 原始张量和 RoPE，检查请求、
模型文件及输入 hash。`preprocessing_equal` 与 `full_native_dependency_match` 分开：
只隔离 Transformers、仍用 serving 的 PyTorch 时，后者保持 false；不能冒充完整
原生依赖环境或 feature/E2E 验收。现有 serving 结果不被覆盖。
