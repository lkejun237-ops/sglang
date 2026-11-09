# Whitebox 的 Lynn 部署

> 历史配方记录：参数、路径和分支描述只适用于原记录版本，不是当前部署输入。
> 当前操作统一从 [deploy/whitebox](../../../deploy/whitebox/README.md) 进入；本次没有重新进行 GPU 验收。

Lynn 只替换 H3 输出 decoder，不使用 TAEH3 或官方 decoder 生成输出。白模输入编码继续使用 serving 导出包的 `video_vae` 组件；当前 main 的该组件仍包含 encoder/decoder 结构，所以不能宣称官方 decoder 的权重已从加载路径剔除。这不要求另外下载官方 VAE。客户端协议仍是两路 H.264 + states 的 WebSocket，A/B 只选择显示，同一模型 session 持续推进，录制不受显示选择影响。

## 代码、镜像与资产

- PR #298 的 `09c6b3122e73859e6dc3f7887aa390440b290de4` 已以 `d96d910b4f060e0ea952e4f02a8312f8d5ba9afc` 合入 main；white_box 需要包含这份源码，不能仅替换镜像。
- 基础镜像：`zing-thailand-registry-vpc.ap-southeast-7.cr.aliyuncs.com/zing/sglang@sha256:894eb66a6f96755d439e07045c27dac2dd5087d7caa5ea5f4d534bf7a500740a`。
- CPFS 数据源 `d-ak7t158lhzaq7kdwp3` 挂到 `/cpfs`。模型资产必须直接可读，不能用 OSS 中相似名称的文件替代。
- DiT 导出：`/cpfs/world-model/sglang/checkpoints/zing-h3-dmd-step2500-197fc0d3-r1/serving`。
- 本次不使用离线 AdaLN cache，不读取或重建已有的 `adaln-cache`。
- Lynn checkpoint：`/cpfs/world-model/zing/runs/vae_decoder/vae-decoder-w3-ar-image-gt-4gpu-20261010-r1/step-00002000.pt`。
- Lynn model_dir：`/cpfs/world-model/checkpoints/LynnReal-Onmi-light-vae-3af3330d`。

当时的资产准备工具支持 `--decoder-type lynn_streaming`、`--decoder-model-dir`、可选 `--adaln-table`，保留默认 TAEH3 参数以兼容原部署脚本。它创建独立 serving view，不修改源 checkpoint。生成的 pipeline 配置为：

```json
{
  "decoder_config": {
    "type": "lynn_streaming",
    "checkpoint_path": "/cpfs/world-model/zing/runs/vae_decoder/vae-decoder-w3-ar-image-gt-4gpu-20261010-r1/step-00002000.pt",
    "model_dir": "/cpfs/world-model/checkpoints/LynnReal-Onmi-light-vae-3af3330d",
    "precision": "bf16"
  }
}
```

`weights` 不覆盖，沿用 checkpoint 的 `export_ema` 选择；预检和验收记录实际选择。Lynn 常驻 GPU，不开启 offload 或 activation checkpointing。SP2 当前每个 rank 都执行完整 Lynn decode，没有 tile 分片加速；吞吐和显存必须实测。

## 采样兼容性

当前 main 的 `ZingH3RealtimeSampler` 明确拒绝混合 initial/regular schedule 与离线单 schedule AdaLN 表组合。不能绕过该校验，也不能因为 2-step 的数值是 4-step 的子集就当成兼容。

本次明确保留前 3 个 block `[1000,750,500,250]`、后续 `[1000,750]`，不传 `--adaln-table`，不设置 `zing_adaln_table_path`。离线表支持仍保留在工具里，但不属于本次部署。在线 projection cache 是另一个开关，沿用已有 `SGLANG_ZING_ADALN_PROJECTION_CACHE=0`。

W6 / sink3、480p、24fps 模型时间轴、SP2、单活跃 session 保持原约定。是否使用独立 encoder 由资源配置确定，不自动增加第三张卡。

## 预检与发布

1. CPU 预检确认镜像依赖、代码身份、CPFS 组件及符号链接、Lynn checkpoint 配置。不要将 checkpoint 中可能存在的训练凭证原样输出。
2. 核验模型配置、DiT shard 与 Lynn 资产哈希；检查生成的配置没有 `zing_adaln_table_path`，运行命令确实保留混合采样。不开启离线表检查或构建任务。
3. GPU 启动时核对实际 decoder 为 Lynn，完整权重正常加载、没有离线 AdaLN 表路径。GPU 名称不能替代 capability 和显存实测。
4. 验证首帧、持续生成、A/B 连续切换、主动结束、断连、第二个并发会话拒绝、下一个会话可用、尾帧和 OSS 录像。没有实际出帧不能宣称上线成功。
5. 仅外围代码更新时不重建 GPU Job。基础镜像变化需要先确认调度平台的更新方式；API 接受配置不等于保留原 Pod / GPU。需要新实例时，先检查已批准节点的真实空闲配额，禁止抢占或操作训练任务。未准备完成前保留原模型。

这些步骤是发布接口与验收要求，不是本次部署已经完成的证明。
