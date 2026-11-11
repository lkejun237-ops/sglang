# 2026-11-10 上海 `test-shanghai-copy` SGLang 三角色实机验收记录

本文档归档 2026-11-10 在上海复制测试环境（`test-shanghai-copy`）执行的 SGLang H100 Encoder + H200 DiT + H100 Decoder 跨机部署、隔离建组自测、4-Case 真实模型回归及 WTP/Room 实时出帧实验记录（从 `deploy/sglang/ORCHESTRATION.md` 迁移归档）。

## 1. 运行环境与逐角色身份回执

- **历史验证提交**：`4916a1a9dd2035e87060ffb8480ee1930a9acf2d`（分支 `codex/deploy-test-environment-sglang`，PR #657）
- **统一启动入口**：`sglang.multimodal_gen.tools.zing_h3_profile_launcher --mode independent` → `zing_h3_role_launcher.py`
- **角色运行身份**：
  - **Encoder**（上海灵骏 H100 `e01-cn-2zd4xi8m505` / `10.0.0.121`）：
    - K3s Pod：`sglang-encoder-f857b47c8937-3`
    - 物理 GPU：`GPU-a276d56c-e2d8-688b-38b7-fda7e0f6c521`
    - 生效 Profile：`h100`
    - 端口：HTTP `:31000` / WS `:30000` / Rendezvous `:32000`
    - 身份：`worker_id: encoder-9a9344ccd07a41bea08733a28fb225f1`，`boot_epoch: b2464978686f489d8adbb7c7bcafb53c`
  - **DiT**（上海 PAI-DLC H200 `cn-shanghai`，workspace `1453237`，quota `quota3l8bnv9a35n`）：
    - PAI-DLC Job：`dlc1ih9nuzu3erbl`（`10.0.0.197`）
    - 物理 GPU：`GPU-31ae5499-2973-a77a-2b80-61fe52b8e01a`（单卡 SP1）
    - 生效 Profile：`h200`（带 `h200_model1027_480_bf16_sp1_block0_cache` 显存策略 `mem_fraction_static=0.875`）
    - 端口：HTTP `:33000` / ZMQ `:34000` / Rendezvous `:35000`
    - 身份：`worker_id: dit-3903c3af2d704b0ca4c37368aad0a76d`，`boot_epoch: 929da71f5ba046e0913c942159ac9076`
  - **Decoder**（上海灵骏 H100 `e01-cn-2zd4xi8m505` / `10.0.0.121`）：
    - K3s Pod：`sglang-decoder-f857b47c8937-1`
    - 物理 GPU：`GPU-92440320-a522-4ce3-6178-9e4943cf8e7e`
    - 生效 Profile：`h100`
    - 端口：HTTP `:36000` / ZMQ `:37000` / Output WS `:38000`
    - 身份：`worker_id: decoder-553fc7c5f94e43788b3a91d20fad4a6b`，`boot_epoch: d2e74bd8f9364177bd3c114e4758f6d2`

## 2. 4-Case 真实模型回归结果（480p 档）

执行命令：`benchmark/whitebox/run_four_case_sglang_suite.py --mode real-sglang`（分辨率 `832×480`，`model1027`）：

| 用例 | 状态 | 关键指标与观测 |
| --- | --- | --- |
| `case1_prefill_cache_and_persistence` | PASS | Step1 cache miss `blocks_built=3`（13283 ms）→ Step2 cache hit `blocks_hit=3, blocks_built=0`（1629 ms，加速 8.15×）→ Step3/4 OSS read-back hit |
| `case2_enter_game_hit_prefill_with_16_refs` | PASS | 16 张参考图入场，首帧命中 prefill cache，TTFF 5690 ms，输出 29 帧 `832×480` 视频 |
| `case3_dynamic_prompt_and_reference_changes` | PASS | 会话内动态切换参考图（ref 0/1/2）与 prompt，连续输出 49 帧 `832×480` 视频，lease 正常释放 |
| `case4_switch_scene_clear_cache_and_rebuild` | PASS | 切场景清空 cache 后重建 `blocks_built=3` 并连续输出 29 帧 |

- 736 档状态：该轮仅执行 `832×480` 档；`736` 档未在该轮执行。

## 3. WTP / Coordinator / Room 浏览器 WebRTC 会话记录

- Coordinator：`http://10.0.1.37:18181`（pool `adaptive-test-20261109`，route `whitebox-copy-model1027-480`）
- Room：`http://10.0.1.34:8092`（容器 `whitebox-sh-copy-room-adaptive-017-default36-r6`）
- WTP：`https://shcopy.world-test-platform.loopit.com.cn`
- 实测会话记录：
  - `admission_id: 2af003006237427890f474f188b0b76b`：WebRTC `model` 轨道生成并投递 425 帧（`failed_frames=0`），断开后正常释放。
  - `admission_id: 59924914e2ad42cebcbc71bd959ec9a6`：WebRTC `model` 轨道生成并投递 366 帧（`failed_frames=0`）。
  - `admission_id: 7992458bdd314bd09978e751b6d838ac`：WebRTC `model` 轨道连续生成并投递 3,137 帧（`failed_frames=0`），会话结束后 `active_sessions=0`。

---

## 4. 2026-11-10/11 阶段三（PR #657）复用服务、唯一 Profile 启动链、浏览器全生命周期与受控回滚验证

本节记录 PR #657 完成阶段一（正确性与唯一启动链 `zing_h3_profile_launcher.py`）和阶段二（通用部署配置、`create`/`reuse` 混合编排、`history/show/diff/rollback`）后的上海 `test-shanghai-copy` 实机验证闭环。

### 4.1 远端源码、部署凭据持久化位置与角色身份（`deploy-20261110T115739Z-88e645ef`）

- **部署源码与归档**：
  - 远端分支：`codex/deploy-test-environment-sglang`
  - 部署源码 Commit：`d180d0071dc387895cce7f8001be7745fd35190f`（含唯一启动链、通用 `target`、`create/reuse`、`history/show/diff/rollback` 及 `coordinator_selftest` 修复）
  - 源码归档 SHA256：`ff250304151c3748ef3a54bf86ed3ad1804ef10392ba0b8ec55995babed630de`
  - OSS 源码包：`oss://leap-world-shanghai/sglang/orchestration/d180d0071dc387895cce7f8001be7745fd35190f/ff250304151c3748ef3a54bf86ed3ad1804ef10392ba0b8ec55995babed630de/source.tar`
  - 镜像 Digest：`whitebox-shanghai-registry-vpc.cn-shanghai.cr.aliyuncs.com/whitebox/sglang@sha256:427cd63816e12bb3a69431db751da924a60cb7c166f9d17702a7cb75370edbdd`
- **部署凭据与回执持久化保存位置（禁止明文密钥，仅存 Secret 文件路径引用）**：
  - **OSS 持久化历史仓库**（`oss://leap-world-shanghai/sglang/orchestration/history/`，Endpoint `oss-cn-shanghai.aliyuncs.com`）：
    - 历史索引：`oss://leap-world-shanghai/sglang/orchestration/history/index.json`
    - 主验证部署完整回滚包与回执：`oss://leap-world-shanghai/sglang/orchestration/history/deploy-20261110T115739Z-88e645ef.json`、`deploy-20261110T115739Z-88e645ef.deployment.json`、`deploy-20261110T115739Z-88e645ef.DEPLOYMENT.md`
    - 受控回滚部署完整回滚包与回执：`oss://leap-world-shanghai/sglang/orchestration/history/deploy-20261110T181155Z-b99af6d7.json`、`deploy-20261110T181155Z-b99af6d7.deployment.json`、`deploy-20261110T181155Z-b99af6d7.DEPLOYMENT.md`
  - **本地工作目录副本**：
    - 历史目录：`/tmp/shanghai-orchestrate-prep/history/`（含 `index.json`、`deploy-20261110T115739Z-88e645ef.json`、`deploy-20261110T181155Z-b99af6d7.json`）
    - 主部署运行态与摘要：`/tmp/shanghai-orchestrate-prep/run-stage3-2step/{state.json,deployment.json,DEPLOYMENT.md}`
    - 回滚部署运行态与摘要：`/tmp/shanghai-orchestrate-prep/run-stage3-rollback/{state.json,deployment.json,DEPLOYMENT.md}`
    - 远端容器/节点内预置 Secret 文件路径引用（写入 `rollback_bundle.secret_references`）：
      - H100 K3s 节点（`encoder`）：`/mnt/zing-nvme/sglang-orchestrate-secrets/{ROLE_POOL_REGISTRATION_TOKEN,ROLE_POOL_SIGNING_KEY}`
      - H200 PAI-DLC 挂载（`dit`）：`/mnt/oss/sglang/orchestration/secrets-20261110/{ROLE_POOL_REGISTRATION_TOKEN,ROLE_POOL_SIGNING_KEY}`
- **逐角色运行身份（最小必要 GPU + 复用现有 Decoder）**：
  - **`encoder`（`create`）**：
    - K3s Pod：`sglang-encoder-88e645efb6c4-0`（UID `7194b4d2-8a4a-40e0-aa3d-e65b2938221e`，上海灵骏 H100 `e01-cn-2zd4xi8m505` / `10.0.0.121`）
    - 物理 GPU：`GPU-92440320-a522-4ce3-6178-9e4943cf8e7e`（GPU 5）
    - Service / Boot 身份：`service_id: encoder-22e6cfc84bc94b1f95e880dd06ae7628/0`，`boot_epoch: b08171db8b574813b7b3690c2befd646`，`service_count: 1`
    - 端点与生效 Profile：`http://10.0.0.121:23320`（ZMQ `:23310`，Rendezvous `:23330`），Profile `h100`
    - 实际启动 `effective_argv`：`unknown`（历史运行仅采集精简日志回执，未回读容器内 `receipt.json`；计划容器启动命令 `planned_argv` 如下）：
      ```bash
      /opt/sglang/bin/python -m sglang.multimodal_gen.tools.zing_h3_profile_launcher --mode independent --component encoder --profile auto --resolution 480 --sp-degree 1 -- --model-path /mnt/zing-nvme/copy-whitebox-release-20261107/runtime/model --service-count 1 --http-port 23320 --advertise-host 10.0.0.121 --worker-id encoder-22e6cfc84bc94b1f95e880dd06ae7628 --port 23310 --rendezvous-port 23330
      ```
  - **`dit`（`create`）**：
    - PAI-DLC Job：`dlcmforuzuanazu8`（`sglang-dit-88e645efb6c4-0`，上海 H200 `10.0.0.218`，workspace `1453237`）
    - 物理 GPU：`GPU-31ae5499-2973-a77a-2b80-61fe52b8e01a`（单卡 SP1）
    - Service / Boot 身份：`service_id: dit-461b4b9b96ad41e69c7d2a41aedf6215/0`，`boot_epoch: 0e777738a9ae48dfb3b62910e5242b17`，`service_count: 1`
    - 端点与生效 Profile：`http://10.0.0.218:33000`（ZMQ `:34000`，Rendezvous `:35000`），Profile `h200`（`mem_fraction_static=0.875`，`denoising_step_list=1000,500`）
    - 实际启动 `effective_argv`：`unknown`（PAI 2 KiB 行长上限下精简日志回执省略 `argv`；计划容器启动命令 `planned_argv` 如下）：
      ```bash
      /opt/sglang/bin/python -m sglang.multimodal_gen.tools.zing_h3_profile_launcher --mode independent --component dit --profile auto --resolution 480 --sp-degree 1 -- --model-path /opt/whitebox/assets/model --service-count 1 --http-port 33000 --advertise-host 10.0.0.218 --worker-id dit-461b4b9b96ad41e69c7d2a41aedf6215 --port 34000 --rendezvous-port 35000 --pipeline-config-path /mnt/oss/sglang/orchestration/assets-20261110/dit-pipeline.json --sp-degree 1 --mem-fraction-static 0.875 --timeout-seconds 300 --zing-block0-cache-budget-root /var/cache/sglang/block0/orchestrate-sp1-20261110 --zing-block0-cache-capacity-dir /var/lib/sglang/capacity --zing-block0-cache-dir /var/cache/sglang/block0/orchestrate-sp1-20261110 --zing-block0-cache-disk-gb 700 --zing-block0-cache-gc-high-watermark 0.9 --zing-block0-cache-gc-low-watermark 0.8 --zing-block0-cache-host-gb 64 --zing-block0-cache-host-total-gb 64 --zing-block0-cache-inflight-gb 128 --zing-block0-cache-max-writers 2 --zing-block0-cache-min-free-gb 128 --zing-block0-cache-mode on --zing-block0-cache-storage local --zing-denoising-step-list 1000,500 --zing-initial-block-count 0 --zing-solver few_step --zing-source-joint-forward cross_chunk
      ```
  - **`decoder`（`reuse`）**：
    - 复用已有常驻 K3s 服务：`adaptive-test-decoder-20261109-r2`（上海灵骏 H100 `10.0.0.121:29210`）
    - 物理 GPU：`GPU-6b33d443-422b-e4e1-d4eb-7f258803a044`（GPU 2）
    - Service / Boot 身份：`service_id: adaptive-test-decoder-20261109-r2/0`，`boot_epoch: 36618b677c36470e9db6ee9ef7f6a86c`，`service_count: 1`
    - 复用保护：未重启、未重注册、未 drain/retire，全程通过 Coordinator 正常租约（`/v1/role-pool/admit` → `/v1/role-pool/close`）借用与释放。

### 4.2 建组自测（`selftest`）与原地挂载（`attach --in-place`）

- **`orchestrate.py selftest`**：
  - 检测到 `decoder` 为 `managed` 复用服务后，自动走 `coordinator-lease-build-quiesce-release` 路径：先将本次新建的 `encoder-22e6cfc84bc94b1f95e880dd06ae7628` 与 `dit-461b4b9b96ad41e69c7d2a41aedf6215` 原地挂载至 Coordinator（`http://10.0.1.37:18181`），再通过确定性 `admission_id` 连续 4 轮调用 `/v1/role-pool/admit` 绑定三角色通信组并调用 `/v1/role-pool/close` 释放，4/4 轮全部成功。
- **`orchestrate.py attach --in-place` & `inspect`**：
  - 原地挂载幂等对账通过，`attachment_mode: "in-place-managed"`，Coordinator `/v1/capacity` `free_slots: 2, active_sessions: 0`，`orchestrate.py inspect` 返回 `rc=0`。

### 4.3 Room–SGLang 供帧回放集成测试（`run_room_sglang_source_replay.py`）与显存观测（按 480 / 736 分档记录）

#### （1）`832×480` 档（PASS，含 Session TTFF 与 Prompt Switch TTFF）

- **执行入口**：[`benchmark/whitebox/run_room_sglang_source_replay.py`](file:///Users/chenshengdong/.gemini/antigravity/worktrees/sglang/deploy_test_environment_sglang/benchmark/whitebox/run_room_sglang_source_replay.py)（原 `run_four_case_sglang_suite.py`，`--mode real-sglang --coordinator-url http://10.0.1.37:18181`，`seed_base=268160`）
- **绑定角色**：`encoder-22e6cfc84bc94b1f95e880dd06ae7628/0` + `dit-461b4b9b96ad41e69c7d2a41aedf6215/0` + `adaptive-test-decoder-20261109-r2/0`
- **结果归档**：`/tmp/shanghai-orchestrate-prep/run-stage3-2step/four-cases/{summary.json,REPORT.md}`（前置条件 Preflight 6 项全绿，`all_leases_released=True`）

**A. Block 0 Cache 生效与持久化指标**：

| 场景 | Cache 结果 | Prefill to Memory (ms) | Prefill to Persisted (ms) |
| --- | --- | ---: | ---: |
| Step 1 上海行 (Cold Prefill) | `miss` | 569.553 | 10675.236 |
| Step 1 上海行 (Warm Prefill) | `l1_hit` | **1.313** | - |
| Step 2 MARVEL (12-Ref Stress Prefill) | `miss` | 3864.253 | - |

**B. Session TTFF 指标（`whitebox_session_ttff_ms`，从 `render.start` 到 Room 首个可解码模型帧）**：

| 案例 | Recording ID | Session TTFF (`whitebox_session_ttff_ms`) |
| --- | --- | ---: |
| Step 1 驾驶：上海行 | `0d256658d28949c8a3f6371df5c47541` | **1888.385 ms** |
| Step 2 跑酷：MARVEL: OVERDRIVE | `b8045d7b8b064756a0081fe4e19643c6` | **1694.283 ms** |
| Step 3 魂系战斗：Dark Horns | `5505ef1c0f214e96a206980eeb048d57` | **1774.522 ms** |
| Step 4 武侠格斗：功夫嘉豪 Seg 1 | `f03d9c0dbd5e46a7819d298d495af73d` | **2030.733 ms** |
| Step 4 武侠格斗：功夫嘉豪 Seg 2 | `083738c8081b49c2bc11141aa8403ac3` | **1811.777 ms** |

**C. Prompt Switch TTFF 与 Dark Horns A/B 对照**：

| 案例 / 切换 | Prompt Version | Cache 结果 | Switch Host (ms) | Prompt Switch TTFF (ms) | Skipped Captures |
| --- | ---: | --- | ---: | ---: | ---: |
| MARVEL Rapid Switch @510 | 2 | `bypass` | 1983.359 | **3008.660** | 47 |
| MARVEL Rapid Switch @544 | 3 | `bypass` | 960.487 | **2004.097** | 23 |
| Dark Horns 3->6 Refs (Arm A: Unprefilled) | 2 | `miss` | 5211.150 | **6324.514** | 125 |
| Dark Horns 3->6 Refs (Arm B: Prefilled) | 2 | `l1_hit` | 2432.913 | **3531.073** | 58 |

- **Dark Horns Prefill Hit 相对 Unprefilled Miss 切换耗时降低**：Switch Host 降低 **`53.31%`**（`5211.150 ms → 2432.913 ms`），Prompt Switch TTFF 降低 **`44.17%`**（`6324.514 ms → 3531.073 ms`）。

**D. 稳态供帧与流控统计**：

| 案例 | Sent Frames | Received Model Frames | Dropped Before Model | Skipped Captures |
| --- | ---: | ---: | ---: | ---: |
| `shanghai_drive` | 56 | 56 | 9 | 0 |
| `marvel_overdrive` | 588 | 588 | 223 | 70 |
| `dark_horns_miss` | 322 | 322 | 203 | 125 |
| `dark_horns_hit` | 322 | 322 | 139 | 58 |
| `wuxia_jiahao` | 52 | 52 | 0 | 0 |

- **三角色逐卡显存（driver `nvidia-smi` 口径，832×480）**：
  - **Encoder**（H100 GPU 5 `GPU-92440320-a522-4ce3-6178-9e4943cf8e7e`，Qwen + 参考图 VAE + Source VAE 同进程）：常驻 `68,815 MiB` / 请求峰值 `73,967 MiB`（单卡总容量 `81,559 MiB`）
  - **DiT**（H200 GPU 0 `GPU-31ae5499-2973-a77a-2b80-61fe52b8e01a`，SP1，`mem_fraction_static=0.875`）：常驻 `110,499 MiB` / 请求峰值 `124,612 MiB`（单卡总容量 `143,771 MiB`）
  - **Decoder**（H100 GPU 2 `GPU-6b33d443-422b-e4e1-d4eb-7f258803a044`，复用常驻服务）：常驻 `13,023 MiB` / 请求峰值 `29,845 MiB`（单卡总容量 `81,559 MiB`）

#### （2）`1280×736` 档（受环境限制未执行，不以 480 外推）

- **状态**：`blocked`（`DRIVER_FATAL:TARGET_PROFILE_UNAVAILABLE`）
- **原因与证据**：`test-shanghai-copy` 线上 Coordinator（`10.0.1.37:18181`）仅配置 `profile_limits: {"480": 2}` 及 `route: "whitebox-copy-model1027-480"`，未注册 `736` 档池与路由。使用 `world-model-all-in-one/tools/e2e/run_product_e2e.py --environment shanghai-copy` 请求 `1280×736` 时，Coordinator `/v1/capacity?profile=736` 直接阻断并记录于 `/tmp/shanghai-orchestrate-prep/run-stage3-2step/e2e-copy-736/browser-evidence.json`。按仓库规范不修改共享 Coordinator 配置，亦不将 `832×480` 结果外推为 `1280×736` 通过。

### 4.4 `world-model-all-in-one` 真实浏览器呈现与单会话生命周期验证

使用知识库 `world-model-all-in-one`（commit `873d631e657c264aa32b8b96374c92bbb724cfe9`）的 `tools/e2e/run_product_e2e.py` 与 `product_e2e_browser.mjs`，驱动真实 Chromium `145.0.7632.6` + 冻结 WorldKit SDK（`689e81f718e22065877de9af589e6496c712f979`）连接 `wss://shcopy.world-test-platform.loopit.com.cn/whitebox/v1/realtime_video/generate?whitebox_control=v2`（控制面版本：Gateway `422fde26f484fc96b1230f79990f78ffa25b4ad0`、Room `412589f54190f03d06d4490ef4a8177d9c39e839`、Coordinator `ccda28a6a3f399a1d6d7b98474fee46d1d775938`），全部绑定本次部署的 `worker_id="dit-461b4b9b96ad41e69c7d2a41aedf6215"`：

| 场景 | 状态 | 关联身份与浏览器/控制面证据（832×480） |
| --- | --- | --- |
| `first_frame` | **PASS** | `model_session_id=e837164974b748b8829c2472aed9d373`，`generation_id=e9bbecb1d71a452cbc8a7c601055efd8`，`source_session_id=8444703e23554350977faf51f0ac420f`；浏览器接收/解码/上屏 `1,105` 帧（`output_frame_start=0`，`pts_us=2527750`，`source_frame_index=68`），Canvas 非黑（`nonBlackRatio=1.0`，`meanLuma=82.90`，截图 `first_frame-1794331674793.png` `sha256=0bf09ec5...`） |
| `input` | **PASS** | 浏览器按键 `"w"` 提交（`sequence=1`）并在上屏模型帧生效（`modelFrame=47`，`sourceFrame=153`，截图 `input-1794331676567.png`） |
| `prompt_switch` | **PASS** | `client.updatePrompt` 切换至 `prompt_version=2`（`prompt_sha256=56ca6f22...`，`first_output_frame=153`）并在 Canvas 呈现（截图 `prompt_switch-1794331684367.png`） |
| `normal_end` | **PASS** | `request_id=2d854d5e-3536-4205-a3d9-982c2382f6f2`，`segment_id=c9028058e0aa4dd8ab242aad037d65c5`，`eos_frames=41`，`segment_state="ended"`，`resource_state="released"` |
| `cancel` | **PASS** | `request_id=af309b0d-5398-477f-baad-6b1f7632cd6e`，`segment_id=9a15d983ae0347bca0a078c22b67084c`，`segment_state="cancelled"`，`resource_state="released"` |
| `exclusive` | **PASS** | 并发第二会话被拒绝（`rejection_code="USER_SESSION_LIMIT"`），主会话不受影响继续上屏（`primary_presented_before=5` → `primary_presented_after=136`，`primary_identities_unchanged=True`） |
| `disconnect` & `next_session` | **模型侧释放与下一次会话上屏 PASS / Room 静态 capability 探针受限** | 断连后 `next_session` 在同一 `worker_id="dit-461b4b9b96ad41e69c7d2a41aedf6215"` 成功建立新会话并解码上屏（截图 `next_session-1794331855786.png` `sha256=5c3f835c...`），Coordinator `/v1/capacity` 确认 `active_sessions=0` 且 `orchestrate.py inspect` 确认三角色均回到 `ready & idle`；但 `test-shanghai-copy` 当前部署的旧版 Room（`412589f54190f03d06d4490ef4a8177d9c39e839`）在 `/v1/whitebox/render/sources/{source_id}` 校验动态会话 token 且不含 PR #620 新增的 `sdk_cleanup_receipt` 字段，导致 `run_product_e2e.py` 静态探针报 `DISCONNECT_CLEANUP_NOT_CONFIRMED` |

- **其余未通过/受阻子项说明（如实记录，不标为全量 E2E 通过）**：
  - `sink`：`SINK_OFF_NOT_COMPLETED_OR_PRESENTED`（`PLAYER_IDLE_TIMEOUT`）。
  - `cache`：因 `test-shanghai-copy` Coordinator 池内同时存在另一台空闲 DiT（`adaptive-test-dit-20261109-r3`），独立 `prefill` 与 `resumeSeg` 两次准入按池哈希落到不同 DiT 实例，触发 `CACHE_MODEL_SEGMENT_HIT_OR_PRESENTATION_MISSING`。
  - 公网 WebRTC 播放页（`WhiteboxPlaytestPage`）：`product_e2e_browser.mjs` 走 `websocket_h264` 传输；公网 WebRTC 页面需交互式 WTP 登录态（未登录返回 `RTC_AUTHORIZATION_FAILED`）。

### 4.5 历史查询（`history` / `show` / `diff`）与受控回滚演练（`rollback`）

在 `/tmp/shanghai-orchestrate-prep/history` 上验证完整历史管理与受控回滚：

1. **回滚计划生成与校验（`rollback` dry-run）**：
   - 目标部署：`deploy-20261110T115739Z-88e645ef`
   - 新生成回滚部署 ID：`deploy-20261110T181155Z-b99af6d7`（`rollback_of: "deploy-20261110T115739Z-88e645ef"`）
   - 计划校验：`current_owned_roles_to_stop: ["dit", "encoder"]`、`current_reused_roles_protected: ["decoder"]`、`preserves_shared_caches: true`、`requires_previous_gpu_or_pod_uid: false`。
2. **受控回滚执行（`rollback --execute --yes`）+ `selftest` + `attach --in-place` + `inspect`**：
   - 自动优雅排空、停止并从 Coordinator retire 旧自有资源（`sglang-encoder-88e645efb6c4-0` 与 `dlcmforuzuanazu8`），将 `deploy-20261110T115739Z-88e645ef` 状态更新为 `stopped`。
   - **复用服务零干扰验证**：复用的 Decoder `adaptive-test-decoder-20261109-r2/0` 在回滚前后保持完全相同的 `boot_epoch: 36618b677c36470e9db6ee9ef7f6a86c`，未发生重启、重注册或 drain/retire。
   - 按固定源码包（`ff250304151c3748ef3a54bf86ed3ad1804ef10392ba0b8ec55995babed630de`）与镜像 digest 拉起新自有资源并完成 `selftest` 与 `attach --in-place`：
     - 新 Encoder：`sglang-encoder-b99af6d77ea3-0`（UID `f8a8a2d3-889d-4b6f-a82c-dc8cb08b5abf`，`service_id: encoder-694c50880a7c48f2b3bca249b038528c/0`，`boot_epoch: 829954e14f034698bf030c9bcf1c387b`，GPU `GPU-92440320-a522-4ce3-6178-9e4943cf8e7e`）
     - 新 DiT：PAI-DLC Job `dlc6ftcejh9j6h3z`（`service_id: dit-13080b4fc17946dfb8c9efccb0fb0755/0`，`boot_epoch: 444a5af541674685bc64ed09652ba761`，GPU `GPU-31ae5499-2973-a77a-2b80-61fe52b8e01a`，`http://10.0.0.188:33000`）
     - 复用 Decoder：`adaptive-test-decoder-20261109-r2/0`（`boot_epoch: 36618b677c36470e9db6ee9ef7f6a86c`）
3. **`history` / `show` / `diff` 核验**：
   - `orchestrate.py history` 准确列出 `deploy-20261110T115739Z-88e645ef`（`phase: stopped`）与 `deploy-20261110T181155Z-b99af6d7`（`phase: ready`，`rollback_of: deploy-20261110T115739Z-88e645ef`）。
   - `orchestrate.py show` 完整返回 schema=1 的 `rollback_bundle`（含源码 SHA/hash/URI、站点镜像 digest、资产 provenance、路由状态、Secret 文件路径引用与角色所有权，不含任何明文密钥）。
   - `orchestrate.py diff` 确认回滚前后 `source_changed: false`、`images_changed: false`、`assets_changed: false`、`route_changed: false`、`roles_changed: false`。

