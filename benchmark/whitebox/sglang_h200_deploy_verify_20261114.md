# 2026-11-14/15 上海 PAI-DLC H200 SGLang 部署、复用与一键回滚实机验收记录

本文档归档 2026-11-14 至 2026-11-15 在上海阿里云 PAI-DLC H200 测试集群（`cn-shanghai`，Workspace `1453237`，Quota `quota3l8bnv9a35n`，`NVIDIA L20X` SM90 / 143,771 MiB）执行的 SGLang Zing H3 四场景端到端部署、复用、升版与跨版本回滚实机验收数据。

---

## 1. 验收范围、隔离边界与硬件环境

- **目标集群与网络域**：
  - PAI-DLC：`cn-shanghai`，Workspace `1453237`，Quota `quota3l8bnv9a35n`（`shanghai-vpc-h100-h200`）
  - VPC 隔离控制面：灵骏宿主机 `e01-cn-2zd4xi8m505`（`10.0.0.121`）上的独立 0-GPU 测试 Coordinator 容器（`:28181`，独立 SQLite + 控制纪元 `h200-verify-epoch-20261114`），全程未触碰 `prod-shanghai`（`10.0.1.31`）、`test-shanghai-copy` 共享 Coordinator（`10.0.1.37:18181`）或 `prod-thailand`。
  - 模型与采样契约：`model1027`（1027 AR + DMD `global_step_010000`，`num_frame_per_block=[3,2]`，`split_first_latent=false`，`zing_denoising_step_list="1000,500"`，单活跃会话）。
  - 回放与画质基准：`benchmark/whitebox/cases/interactive_production_four_cases.json` 中 `shanghai_drive_baseline`（`832×480`，固定 `seed=886410`，`WHITEBOX_INFERENCE_STEPS=2`）。
- **四大硬性验收门禁**：
  1. **10 分钟 SLA**：`up + selftest + attach` 端到端墙钟耗时 `end_to_end_sec < 600s`（`within_10min_sla: true`）。
  2. **三角色已预热**：`encoder`（Source VAE + Qwen 文本编码）、`dit`（`832×480` `T=1` 真实采样会话并释放 KV 池）、`decoder`（`832×480` 真实潜变量流式解码并释放流状态）均通过 `/health` 返回 `warmed_up: true` 与 `warmup.elapsed_s`。
  3. **真实可接流量**：Coordinator `attach` 后精确 `(profile="480", compatibility, network_domain="shanghai-vpc-h100-h200")` 角色池 `roles.model_group.free_slots >= 1`，通过真实 `SourceSession` 完成视频流生成（`dropped_before_model == 0`），并在会话关闭后归还租约恢复 `free_slots == 1`。
  4. **画质不变**：解码会话归档 `spool/*/model.mp4` 全量帧序列，校验帧数、分辨率 `832×480`、非黑屏亮度/纹理/时间动态内禀指标，以及跨部署/回滚相对场景 1 基线的首帧门禁（`PSNR >= 28.0 dB`、`SSIM >= 0.90`）、43 帧自回归均值门禁（`PSNR >= 25.0 dB`、`SSIM >= 0.85`）与全局亮度一致性门禁（`|Δluma| <= 5.0`）。

---

## 2. 四场景实机验收指标总览

| 阶段与场景 | `deployment_id` | 部署源码 Commit SHA | PAI-DLC 角色拓扑与 Job ID | 分阶段耗时 (`timings_sec`) | 预热耗时 (`warmup.elapsed_s`) | 接流会话 (`SourceSession`) | 画质指标 (`832×480`, `seed=886410`) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **场景 1：`stage1_full_v1`**<br>H200 全新建 `1E + 1DiT + 1D` | `deploy-20261114T162550Z-55ca8058` | `0166232ece1e673143da90bde763d1fe3d6c055b` | 新建 E: `dlccla8sr0lrbfwa` (`10.0.0.248:31000`)<br>新建 DiT: `dlccla8sr0hsspvi` (`10.0.0.247:33000`)<br>新建 D: `dlccla8srjan8qqf` (`10.0.0.246:36000`) | `plan=0.694s`<br>`upload=29.871s`<br>`submit=1.838s`<br>`wait_ready=341.212s`<br>`up_total=373.615s`<br>`selftest=20.143s`<br>`attach=131.471s`<br>**`end_to_end=525.229s`** ✅ | `E=33.166s`<br>`DiT=31.458s`<br>`D=7.803s`<br>(`all_warmed_up=true`) | `free_slots=1` → 接流生成 42 帧 (`TTFF=1274.334ms`, `dropped=0`) → 释放后 `free_slots=1` ✅ | 基线 43 帧：<br>`mean_luma=104.949`<br>`min_std=49.716`<br>`motion_mae=3.3414`<br>(`quality_unchanged=true`) ✅ |
| **场景 2a：`stage2_reuse_v1`**<br>复用在线 E/D + 新建 DiT (V1) | `deploy-20261114T171609Z-9f348f52` | `025eb5241e5bf74aa3335d68c059e09ebd1026a9` | 复用 E: `dlccla8sr0lrbfwa`<br>复用 D: `dlccla8srjan8qqf`<br>新建 DiT: `dlc1wnfocr3rie1w` (`10.0.0.249:33000`) | `plan=1.350s`<br>`upload=29.514s`<br>`submit=1.687s`<br>`wait_ready=378.226s`<br>`up_total=410.777s`<br>`selftest=71.782s`<br>`attach=60.875s`<br>**`end_to_end=543.434s`** ✅ | `DiT=30.682s`<br>复用 E/D 保持 `warmed_up=true` | `free_slots=1` → 正常接流生成并释放；随后执行 `down` 仅停 `dlc1wnfocr3rie1w`，复用 E/D 零中断 ✅ | 对比场景 1（43 帧）：<br>首帧 `PSNR=29.6641dB`, `SSIM=0.9210`<br>均值 `PSNR=26.9347dB`, `SSIM=0.883081`<br>`luma_diff=0.527` ✅ |
| **场景 2b：`stage3_reuse_v2`**<br>复用在线 E/D + 升版 DiT (V2) | `deploy-20261114T173244Z-c038aeed` | `55caa2ff0c71f8566900e84ca0f1e5a524d8892b` | 复用 E: `dlccla8sr0lrbfwa`<br>复用 D: `dlccla8srjan8qqf`<br>新建 DiT: `dlc1obxh57vc1xx3` (`10.0.0.250:33000`) | `plan=1.312s`<br>`upload=28.615s`<br>`submit=1.544s`<br>`wait_ready=377.200s`<br>`up_total=408.671s`<br>`selftest=68.972s`<br>`attach=60.453s`<br>**`end_to_end=538.096s`** ✅ | `DiT=31.357s`<br>复用 E/D 保持 `warmed_up=true` | `free_slots=1` → 接流生成 52 帧 (`TTFF=1373.747ms`, `dropped=0`) → 释放后 `free_slots=1` ✅ | 对比场景 1 通过全部内禀与基线 PSNR/SSIM/luma 门禁 (`quality_unchanged=true`) ✅ |
| **场景 3：`stage4_rollback_v1`**<br>`rollback --previous` (V2 → V1) | `deploy-20261114T174549Z-d64ceec2`<br>(`rollback_of`: `deploy-20261114T171609Z-9f348f52`) | `025eb5241e5bf74aa3335d68c059e09ebd1026a9` | 保护复用 E/D 零中断；自动先停 V2 `dlc1obxh57vc1xx3`，再拉起 V1 DiT `dlc2wf56zoa5y3vz` (`10.0.0.251:33000`) | `plan=1.443s`<br>`stop_previous=30.609s`<br>`upload=30.080s`<br>`submit=1.646s`<br>`wait_ready=374.857s`<br>`up_total=408.026s`<br>`selftest=66.554s`<br>`attach=60.504s`<br>**`end_to_end=565.693s`** ✅ | `DiT=31.621s`<br>复用 E/D 全程 `Running` 未重启 | `free_slots=1` → 接流生成 52 帧 (`TTFF=1529.639ms`, `dropped=0`) → 释放后 `free_slots=1` ✅ | 对比场景 1（43 帧）：<br>首帧 `PSNR=31.2631dB`, `SSIM=0.940095`<br>均值 `PSNR=28.6783dB`, `SSIM=0.910882`<br>`luma_diff=1.995` ✅ |

---

## 3. 部署凭据归档与脱敏审计

所有轮次的持久化部署凭据已通过 `--sync-oss` 归档至上海 OSS：

- **历史索引**：`oss://leap-world-shanghai/sglang/orchestration/h200-verify-20261114/history/index.json`
- **逐次部署回滚包与验收回执**：
  - `oss://leap-world-shanghai/sglang/orchestration/h200-verify-20261114/history/deploy-20261114T162550Z-55ca8058.json`（及同名目录下 `DEPLOYMENT.md`、`deployment.json`、`traffic_verification.json`）
  - `oss://leap-world-shanghai/sglang/orchestration/h200-verify-20261114/history/deploy-20261114T171609Z-9f348f52.json`（及同名目录三件套）
  - `oss://leap-world-shanghai/sglang/orchestration/h200-verify-20261114/history/deploy-20261114T173244Z-c038aeed.json`（及同名目录三件套）
  - `oss://leap-world-shanghai/sglang/orchestration/h200-verify-20261114/history/deploy-20261114T174549Z-d64ceec2.json`（及同名目录三件套）
- **密钥零明文核验**：`ACR_IMAGE_AUTH`、`ROLE_POOL_OPERATOR_TOKEN`、`ROLE_POOL_REGISTRATION_TOKEN`、`ROLE_POOL_SIGNING_KEY`、`WHITEBOX_ROOM_TOKEN` 五项凭据仅保留容器/宿主机文件路径引用，归档 JSON/MD 及本地 `state.json` 明文扫描命中数为 `0`。
- **资源清理对账**：验收结束后，本轮创建的全部 6 个 PAI-DLC 作业（`dlccla8srjan8qqf`、`dlccla8sr0hsspvi`、`dlccla8sr0lrbfwa`、`dlc1wnfocr3rie1w`、`dlc1obxh57vc1xx3`、`dlc2wf56zoa5y3vz`）均已通过 `orchestrate.py down --execute --yes` 优雅排空退役并确认为 `Status=Stopped`；Eflo 隔离 Coordinator 容器与临时密钥目录已清理。

---

## 4. 实机踩坑与代码沉淀摘要

1. **PAI-DLC `GetPodLogs` 单行 2048B 截断与嵌套 `runtime.gpu` 键名冲突**：`launch.py` 在回执超长时输出保留身份与 `warmed_up` 的紧凑回执；`orchestrate.py` 的 `last_receipt()` 优先提取含 `GPU-<uuid>` 的硬件指纹并从早期未截断日志补全 `argv`。
2. **全 PAI H200 拓扑无 K3s Pod 时的 VPC 自测中继**：`remote_selftest()` 在无 K3s Pod 时自动通过 Eflo 宿主机上的临时 0-GPU 容器完成三角色 HTTP 通信组建连自测。
3. **DMD 推理步数自动对齐**：`verify_traffic_quality.py` 从部署 `state.json` 的 `zing_denoising_step_list` 自动推导步数注入 `WHITEBOX_INFERENCE_STEPS`，避免 4 步默认值与 2 步 DMD 服务契约冲突触发 `prefill_invalid`。
4. **流式会话归档 `model.mp4` 全量解码与因果自回归门禁**：优先解码 `SourceSession` 落盘的 `spool/*/model.mp4` 获取全部非 IDR 帧，并区分同实例逐位确定性与跨物理 H200 卡 FP8/BF16 自回归累积合理范围。
5. **升级后 `--previous` 回滚目标选择**：将含完整 `rollback_bundle` 的正常下线版本（`phase="stopped"`）与在线版本（`"ready"`/`"attached"`）同级按时间逆序匹配，避免从 V2 回滚时跳过已正常停机的 V1 误回退到更早的 V0。

---

## 5. 未验证范围与边界说明

- 本轮实机验收仅覆盖上海 PAI-DLC H200 上的 `832×480` 分辨率档（`shanghai_drive_baseline`）；`1280×736` 分辨率档尚未在本轮独立编排流中执行，不外推为已验证。
- 本轮使用 VPC 内隔离 0-GPU Coordinator 容器验证真实 `SourceSession` 接流与租约生命周期，未对 `prod-shanghai` 或 `prod-thailand` 执行线上切流。

回滚总耗时修正：原汇总 535.084s 漏计 stop_previous=30.609s，完整总耗时为 565.693s，仍低于 600s；这是对已有阶段记录的统计修正，未重跑 GPU 验收。
