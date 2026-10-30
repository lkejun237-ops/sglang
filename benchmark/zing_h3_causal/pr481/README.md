# PR #481：B300 SP2 性能验证摘要

## 当前结论

截至 2026-10-30，旧候选 `13cd4ca52a` 在同机 480p ABBA 对照中，首包后服务 FPS
中位数 **30.715→28.520（−7.15%，每版本 n=6）**，不作为 480p 性能升级建议。
候选额外强制开启 fused RoPE/KV，不能把回退单独归因于本 PR 代码。
新增优化随后已限制为 1280×736 目标；**门限后当前版本的完整 GPU 验收尚未完成，PR 保持 Draft**。
下表均为整套服务指标，不是 DiT-only 吞吐，也不证明稳定 DiT ≥20 FPS。

## 版本与测试条件

| 项目 | ABBA 对照条件 |
| --- | --- |
| 基线 / 候选 | main `480b48c8531c956d108fb8a1a524b60e8acb13e4` / candidate `13cd4ca52ac906fe8688b01281e6cd3e07630645`（门限前） |
| 硬件 / 软件 | 泰国 B300（设备报告 L20D、CC10.3、148 SM、275040 MiB/卡）；PyTorch 2.13.0+cu130、CUDA13.0、driver580.105.08；未锁频 |
| 部署 / 模型 | E1 + DiT SP2 + D1，整套四卡；完整50层、BF16、Graph、Block0热缓存，未启用FP8 |
| 输入 / 采样 | 旧 case `30s-wasteland`，832×480、720帧、7参考图、seed123；初始3个block各4步，随后2步；W6/sink3 |
| 客户端 / 测量 | 连续source、unpaced、batch pattern `[1]`、单活跃会话；正式轮无profiler/组件CUDA计时，各臂均采集100ms CPU cgroup数据 |
| 顺序 / 样本 | 同次四卡分配，main→candidate→candidate→main；每臂独立启动、2次预热+3次正式；每版本n=6，正式失败/未执行均0 |
| 配置差异 | candidate设 `SGLANG_ZING_FUSED_CURRENT_ROPE_KV=1`，main unset；不是单因素代码对照 |

Checkpoint SHA256：`2b4d3ef2d9cd6d34c779e4ae0253faae71baab65e7cbe9aeb89063e9ac482e3e`。
输入RGB SHA256：`8e3d2d8bf947a116cc6bdcac67bc31f9d4033eeb79aa765c4b8e73e66911cf2c`。
完整prompt、参考图顺序、模型/AdaLN配置、镜像及源码身份见原始配置审计；此输入不是产品集合验收。

## 核心性能对照

每格为请求级 **中位数 / P95**，每版本n=6；P95使用线性插值，时间单位秒。
ICL先取每请求85个间隔的中位数/P95，再对6个请求值汇总，不混池。

| 指标 | main | candidate | 中位变化 |
| --- | ---: | ---: | ---: |
| TTFF | 1.509370 / 1.688620 | 1.473552 / 1.642161 | −2.37% |
| E2E | 25.032337 / 25.458246 | 26.755509 / 27.143619 | +6.88% |
| 全程服务FPS | 28.763559 / 29.332628 | 26.910550 / 27.202435 | −6.44% |
| 首包后服务FPS | 30.715278 / 31.238987 | 28.519505 / 28.750399 | −7.15% |
| TPOC | 0.275396 / 0.282596 | 0.296598 / 0.301786 | +7.70% |
| 每请求ICL中位数 | 0.264610 / 0.270941 | 0.265421 / 0.269909 | +0.31% |
| 每请求ICL P95 | 0.417081 / 0.423319 | 0.416770 / 0.418386 | −0.07% |

另一次同机三臂对照（各n=3）中，首包后服务FPS中位数：main默认 **30.166**、
main强制fused1 **28.922（−4.12%）**、旧候选默认 **29.404（−2.53%）**。
它与ABBA分配不同，百分比不能相加；尚未分离fixed-Q与输入布局各自贡献。
上述旧case的完整RGB跨臂逐字节一致，不外推到其他输入或当前门限版本。

## 未验证项

- 当前门限版本在 `product-case-20261029` 上的480/736两档完整同条件GPU性能与质量对照；
  736输入为原始480素材bilinear派生到1280×736，不是原生736采集。
- 当前版本逐rank DiT阶段吞吐及稳定≥20 FPS验收；服务FPS与DiT CUDA-event阶段吞吐分别报告。
- Encoder（Qwen、参考图VAE、source VAE）、DiT、Decoder的完整阶段计时及常驻/请求峰值显存；
  冷路径与共享进程/卡级归因须单独标注，不能用历史版本或0代替。
- 480回退根因及各改动独立贡献；H100/H200、SP4性能与质量尚未验收。

## 原始证据入口

内部OSS根目录（未发布公网）：
`oss://leap-worldmodel-thailand-single-az/world-model/sglang/evals/zing-b300-product-20261030/`

- 正式结果：`analysis/ABBA_FORMAL_SUMMARY.json`；四臂原始配置、请求和日志位于
  `r1-regression-{0-main,1-candidate,2-candidate,3-main}-480/`。
- 身份与输出核验：`analysis/config-audit/CONFIG_AUDIT.json`、`ABBA_RGB_VERIFICATION.json`。
- 三臂对照：`FACTOR_LIVE_r3.json`、`FACTOR_READBACK_r3.json`；产品输入身份：`FROZEN_INPUTS.json`。
- 复现入口：归档的 `prepare-regression-r1.py`、`run-regression-r1.py`、`run-regression-r1.sh`
  及对应JobSpec；历史诊断、阶段吞吐定义与重算材料保留在该证据目录，不在本文重复归档。
