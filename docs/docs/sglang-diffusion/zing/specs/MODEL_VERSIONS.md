# Zing H3 测试模型版本

本页供 Codex 和测试同学选择精确 checkpoint。登记原始权重不代表已经导出、部署或通过 GPU 验收。
默认模型及已验证版本边界见 [部署 recipe](../deployment_zing_h3.mdx#当前测试模型1027-step-010000)。

## 1102 DMD step2000（11 月候选）

2026-11-06 登记。豪泽于 2026-11-04 22:01 在 world model sota 群发布，随后说明部署方式不变；
该说明属于发布方信息，不替代 SGLang 独立验证。
[发布消息](https://applink.feishu.cn/client/chat/open?openChatId=oc_a9dfe6855b615b5fa75d8c22b3da9300&position=13806)。

| 项目 | 精确值 |
| --- | --- |
| 记录标识（用于测试回执） | `zing-h3-dmd-1102-step2000-generator` |
| run | `h3-dmd-sp4-05-1102-ac3d5b28` |
| step / component | `global_step_002000` / `generator`；不是 EMA |
| 原生 Zing 提交（manifest） | `ac3d5b288f05d350485d486474c331ff6480ddd0` |
| 原始权重大小（OSS 对象） | 83,628,149,413 bytes |
| 原始权重 multipart ETag | `55E8B7841DFCB2BB0BB18BB33A91F517-2493`；不是 SHA256 |
| 原始权重 SHA256 | 尚未全量下载计算 |
| 原始配置 | 同 step 下的 `lora_export_config.yaml`；不是 `effective_config.yaml`（查询返回 404） |
| 原始配置 SHA256 | `937c7e86a7b6109345f1a9c29ad99f8d0f70ae3dfb5435d1f3fb2102a37d5956` |
| `manifest.yaml` SHA256 | `ebdfe089758f5ea7c136f3a2182674d52c3206ecf977058595002b05408e598f` |

原始权重（泰国 OSS，endpoint `oss-ap-southeast-7.aliyuncs.com`）：

```text
oss://leap-worldmodel-thailand-single-az/world-model/zing/checkpoints/run-archive/dmd/h3-dmd-sp4-05-1102-ac3d5b28/checkpoints/global_step_002000/generator/model.pt
```

配置、manifest 和 `READY` 位于 `global_step_002000/` 下。2026-11-06 已通过现有
`shengdong` profile 查询对象，并下载小型配置与 manifest 核对字节摘要；未下载 83.6 GB 权重。
训练配置可能含凭证，不把原始 YAML 提交仓库或随报告公开；只记录必要字段和摘要。

### 已核对的模型合同

以下字段来自该 step 的原始配置，不能用旧模型配置覆盖：

- 50 层 MHA，2 层 refiner；`dim=5376`、`num_heads=56`、`head_dim=128`。
- 五模态：`video=0,text=1,audio=2,source_visual=3,condition_visual=4`。
- `num_frame_per_block=[3,2]`、`split_first_latent=false`、RoPE `frame_scale="5/3"`。
- 媒体 FPS 24；`num_train_timestep=1000`，video/audio shift 为 8/3，`ar_condition_timestep=0`。
- 训练 `denoising_step_list=[1000,750,500,250]`。减少步数或采用分块混合 schedule 属于额外实验，须单独记录、验证。
- 配置启用 LoRA，manifest 的 DCP 组件为 `lora` / `fake_scorer_lora`；发布链接指向 portable
  `generator/model.pt`。是否已完整合并 LoRA 仍须从原生导出证据和实际 tensor keys 核验，不能仅凭文件名确认。

### 已有泰国 serving 与 AdaLN 资产

用户确认克君已在泰国部署本模型，已有 SGLang serving 包和配套 AdaLN cache/table。
因此后续测试优先复用已有资产，不将“尚未找到路径”误写为“尚未导出”，也不默认重新转换或计算。

已通过历史 CPU 盘点回执及 PAI 作业核对到的路径：

| 用途 | 泰国 CPFS 路径 |
| --- | --- |
| 部署侧模型目录 | `/cpfs/world-model/sglang/whitebox-serving/model-step2000-ac3d5b28` |
| 测试侧 serving 目录 | `/cpfs/world-model/sglang/evals/product736-newmodel-20261104-r3/model` |
| 同版本另一份测试目录 | `/cpfs/world-model/sglang/evals/product736-newmodel-20261104-r2/model` |
| 既有部署启动入口 | `/cpfs/world-model/sglang/whitebox-serving/streameval-dmdac3d-step2000-prefix0-sink2-w9-20261106-r4/launch_main.sh` |

盘点来源：`oss://leap-worldmodel-thailand-single-az/world-model/sglang/evals/ttff-warmup-quantify-20261105-r1/`
下的 `ASSET_DISCOVERY.json`、`MODEL_DISCOVERY_4bd6.json` 和 `ASSETS.json`。
对应 CPU 盘点作业 `dlc10hznrfrqhdw5`，部署启动作业 `dlc1njpqj2o7ohpb` 的日志明确记录
`deployment: ac3d5b28 step2000 prefix0 sink2 window9`。上述是资产定位依据，
不是所有目录逐文件完整性或当前配置兼容性证明；不能按目录名猜测它们互为完整副本。

**配套 table 已存在的依据是用户对既有部署的确认；本次尚未独立读到其精确路径、manifest、
schedule 和 CUDA provenance。** 使用时从实际部署的 pipeline/profile 或上述启动入口取得
`zing_adaln_table_path` 与模式，核对其绑定的模型/配置摘要、solver/schedule 和运行环境。
不能用其他同为 step2000 的模型（如 `1741c541`、`1103`）或 1027/step2500 的 table 替代。
如实际部署使用 resident AdaLN，则如实记录模式，不把它冒称为已加载离线 table。

### 后续测试如何选择

本版本可显式选择用于后续测试，原 1027 step10000 模型保留作对照。
记录标识 `zing-h3-dmd-1102-step2000-generator` 仅用于报告，不是新增 CLI 模型别名；
客户端连接的服务实际决定权重。

1. 优先取得上述现成 serving 目录及其实际部署配置，核对 `zing_config.json` 的原始 checkpoint、
   权重摘要、`provenance`、chunk/split 合同和公共组件链接。记录实际采用的目录，
   不把原始 `generator/model.pt` 当成 SGLang `--model-path`。
2. 复用匹配的已有 AdaLN 资产，先核对 table manifest、模型/配置摘要、solver/schedule 和 CUDA
   算术环境；模式、路径及 provenance 写入本次部署回执。只有资产缺失或不兼容时才重新导出/生成。
3. 服务加载选定 serving 目录后，客户端显式运行
   `bench_streaming_source.py --case-set product-case-20261029`，按
   [benchmark 契约](../../../../../benchmark/zing_h3_causal/CLIENT_BENCHMARK.md)保留输入和更新边界。
   480 与 736 两档分别验收；群里的“720p”是发布讨论用语，不是 1280×736 已验收的证据。
   A/B 两侧使用同一模型版本；模型效果对比与代码性能对比各自记录。

### 当前可用性边界

已完成：发布出处、OSS 原始权重对象、原始配置/manifest 摘要及配置字段核验；
用户确认的既有泰国部署与配套 AdaLN 资产；历史盘点中的 serving 路径和同版本启动日志定位。

本次未独立完成：现成 serving 包的权重 SHA256/LoRA 合并与完整性复核，配套 table 的精确路径、
manifest 和 runtime 匹配核验，以及当前源码的 480/736 生成效果、TTFF/E2E/FPS/TPOC/ICL、
Encoder/DiT/Decoder 分项耗时和显存验收。这些缺口不等于资产尚未生成；
不继承旧模型数据，不切换现有服务或全局默认模型。
