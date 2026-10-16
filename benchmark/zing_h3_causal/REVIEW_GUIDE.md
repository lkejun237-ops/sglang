# PR #230 评审入口

2026-10-09 更新：用户入口默认 cross_chunk、等待默认 auto、attention 默认有条件 auto；
配置迁移与支持范围见 [当前策略](../../docs/docs/sglang-diffusion/zing/specs/JOINT_FORWARD.md)。本页固定 SHA 和复现命令保留旧实验语义。

这次请评审两件事：现有 joint forward 的在线语义是否可靠，以及专用 attention
kernel 是否还值得投入。联合主干已支持在线；专用 attention kernel 仍是离线形状
实验，二者不要混为一谈。四步采样保持不变。

## 七项材料完成情况

| 项目 | 入口与状态 |
|---|---|
| 1. 中文正文和重要文档 | PR 正文、联合设计、attention 设计、组件/端到端 A/B、在线报告均为中文；代码和原始日志保留原文 |
| 2. 历次结果和上下文 | PR 正文及下表汇总，失败与波动样本保留 |
| 3. 远端代码、范围和固定版本 | 下方按 PR 实现/后续实验分开，所有链接固定 commit |
| 4. 同口径对照表 | 下表明确基线、在线/离线、正确性和边界 |
| 5. 代码阅读入口 | 下方列出调用链、分支可见范围、FA4 和实验启用位置 |
| 6. 最小复现 | 下方固定环境、输入、命令、预热和原始证据；不需要先重跑 GPU 才能评审 |
| 7. 评审问题和证据缺口 | 最后一节列出三项判断及尚缺的测量 |

“评审材料齐全”不代表 performance 组合无损准入，也不代表专用 kernel 已在线启用。

## 先看这张结果表

各行使用各自冻结基线，不能把不同实验的百分比相加。详细样本和失败历史见链接。

| 实验 | 配置/基线 | 实测收益 | 正确性与限制 |
|---|---|---|---|
| joint 主干，离线 | SP2，off→cross，原 `[1,2,2]`，4 步 | DiT 11.5516→9.3963 s（-18.66%）；API -5.61% | 修复 AdaLN/head 形状后完整 latent/媒体一致；不是在线数据 |
| joint 主干，原在线验收 | SP2，off→cross0，24 FPS paced，确定性编码配置 | 全程 32.9279→31.0218 s（-5.79%） | 五档共 15 个完整 RGB 一致；抽样 block 的全部 50 层 KV/step0 一致 |
| VAE，离线 | deterministic，重复编码→SP2 编码 | source 7.281→3.931 s；API -11.52% | 完整输出一致；performance 的不同结论单列 |
| attention 组件 | 原逐分支 FA4+cat→融合跳块，随机 QKV、热缓存 | 长历史三分支 1.23–1.40× | 测试形状上 FA4 输出一致；无历史有退化，不是业务请求 |
| attention 端到端，离线 | deterministic VAE SP2；ABBA，各组 6 次 | DiT -2.26%；API -1.00%（0.266 s） | 12 次 latent/视频一致；样本重叠，未证明稳定 1% |
| source 编码，在线 r1 | performance+cross，replicated→tile SP2 | source host 3.8616→2.2701 s（-41.21%） | 分工/benchmark 生效；跨组输出不同，非无损结论 |
| joint 主干，在线 r2 | performance+tile SP2，off→source_first/cross | DiT CUDA 区间 -14.19%/-17.92%；全程 -4.62%/-6.10% | 各档 3 次正式+3 次独立诊断；跨组 RGB 不同，预热输入侧编码已不同 |

r2 的 CUDA 区间包括 stream 等待与发射间隙，不是纯 kernel 时间。source host 与
encoder pop CUDA 的范围不同。decoder 约 17.2 s，是当前端到端最大的单项。

## 固定代码与阅读顺序

PR 实现固定版本：`9d1d1b38f545ea4840674a2e29844d0010ef1e87`。
后续实验/报告：`f51114396f`（远端分支 `codex/zing-vae-encoder-retest`）。
在线 GPU r2 的源码基座是 `1224489aea6923b47a5ab01a74b87f65f3fad04b` 加独立
冻结诊断 overlay，不能写成在最新 PR head 上跑了这轮 GPU。

1. [联合语义与历史验收](https://github.com/seedleap/sglang/blob/9d1d1b38f545ea4840674a2e29844d0010ef1e87/benchmark/zing_h3_causal/JOINT_FORWARD.md)。
2. [incremental.py 的 drain_ready](https://github.com/seedleap/sglang/blob/9d1d1b38f545ea4840674a2e29844d0010ef1e87/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/zing_h3/incremental.py)：连续 source 的有界批次入口。
3. [sampling.py 的 _joint_eligible / _ready_cross_prediction](https://github.com/seedleap/sglang/blob/9d1d1b38f545ea4840674a2e29844d0010ef1e87/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/zing_h3/sampling.py)：检查就绪、窗口等条件，构造 clean final/source/step0 联合输入。
4. [zing_h3_joint.py](https://github.com/seedleap/sglang/blob/9d1d1b38f545ea4840674a2e29844d0010ef1e87/python/sglang/multimodal_gen/runtime/models/dits/zing_h3_joint.py)：验证分支、RoPE/布局与 clean 提交；失败不留下半提交会话。
5. [zing_h3.py 的 _joint_attention / _fa4_attention](https://github.com/seedleap/sglang/blob/9d1d1b38f545ea4840674a2e29844d0010ef1e87/python/sglang/multimodal_gen/runtime/models/dits/zing_h3.py)：现有 FA4 仍逐分支调用，最后 cat；合并的是主干遍历，并没有把 attention 自动合成一个 kernel。
6. [实验 hook](https://github.com/seedleap/sglang/blob/f51114396f/benchmark/zing_h3_causal/install_joint_attention_experiment.py) → [FA4/CuTe kernel](https://github.com/seedleap/sglang/blob/f51114396f/benchmark/zing_h3_causal/joint_attention_kernel.py)。hook 替换 `_joint_attention`，不改生产默认。
7. [设计与业内参考](https://github.com/seedleap/sglang/blob/f51114396f/benchmark/zing_h3_causal/JOINT_ATTENTION_DESIGN.md)、[组件 A/B](https://github.com/seedleap/sglang/blob/f51114396f/benchmark/zing_h3_causal/JOINT_ATTENTION_KERNEL_AB.md)、[离线端到端 A/B](https://github.com/seedleap/sglang/blob/f51114396f/benchmark/zing_h3_causal/JOINT_ATTENTION_E2E_AB.md)、[在线 r1/r2](ONLINE_SP2_RETEST.md)。

三分支按 F=上一块 clean final、S=下一块 source、T=下一块 noisy target 排列：

| Q 分支 | 可见 K/V |
|---|---|
| F | H + F |
| S | H + F + S |
| T | H + F + S + T |

各分支内部双向可见。不能替换成普通 token causal mask，S 不能看 T。
只将 clean F/S 提交为持久历史，T 不冒充 clean KV；sampler 只消费 target 预测。
窗口淘汰会改变历史视图，当前三分支合并回退，仍可继续尝试 source_first。

### 在线怎么开、哪些情况下合并

在模型目录的 `zing_realtime_profile.json` 设置：

```json
{"joint_forward": "cross_chunk", "joint_wait_ms": "auto"}
```

这是修改两个字段的示意，保留 profile 其他字段；只合并 source+step0 时设
`source_first`。当前默认 `cross_chunk` + `joint_wait_ms="auto"`，显式等待只接受 (0,1000] ms。
以下历史固定版本结果使用 0，新自动策略尚未做 GPU 性能复测。

连续像素 WebSocket → progressive encoder → `drain_ready` 支持在线跨块合并，
最多暂存两个已就绪 chunk。下一块 source 没就绪则回退，不读未来帧。普通单步
`step()` 会在返回前提交 final，不能跨返回边界做三分支合并。EOS/prompt 更新/
取消会中断等待；SP4、CFG>1、音频等不支持组合有回退。

实际 r1 在线每请求只有 7 次三分支、15 次 source_not_ready 回退。Q 为
780/1560/2340，H=11756–38276，K 最大 39056，均有界。实验 hook 的 `_LENGTHS`
仅有离线 `(880,390)/(390,1760,780)/(780,1760,780)/(780,880,390)`，在线形状
全部不匹配；不是“在线 joint 不支持”，而是“该专用 attention hook 尚未覆盖在线”。

## 最小复现与证据位置

### 先做不占 GPU 的核对

```sh
git fetch origin codex/zing-joint-forward codex/zing-vae-encoder-retest
git show 9d1d1b38f545ea4840674a2e29844d0010ef1e87:python/sglang/multimodal_gen/runtime/models/dits/zing_h3.py
git show f51114396f:benchmark/zing_h3_causal/install_joint_attention_experiment.py
```

在线逐次数据：[r1 JSON](online_sp2_retest_r1.json)、[r2 JSON](online_sp2_retest_r2.json)。
组件和离线原始数据：[组件 JSON](https://github.com/seedleap/sglang/blob/f51114396f/benchmark/zing_h3_causal/joint_attention_kernel_ab.json)、
[端到端 JSON](https://github.com/seedleap/sglang/blob/f51114396f/benchmark/zing_h3_causal/joint_attention_e2e_ab.json)。

### 组件复现（明确是组件诊断）

在冻结实验 checkout、同一 GPU 镜像内执行：

```sh
python benchmark/zing_h3_causal/benchmark_joint_attention_kernel.py --output /tmp/joint-kernel-review-new
```

使用 1 GPU，实测设备 CC=10.3、274113 MiB、148 SM；PyTorch 2.13.0+cu130，
镜像 digest `sha256:4b9bca7dd61784cd84e43ac760db4c458c08a2d6bf22887e3740455fffe9e015`。
镜像内 FA4 interface SHA256
`c2ca5de1e73bb410c8bd560a95865ae1e0e5382f2cc792d16a4ff60f6bcc0c7f`。
随机 QKV seed229、BF16、D128、28 heads；编译后 5 次预热，5×30 次计时及
独立 Graph replay。热路径数据不含首次编译，包含输出 cat；不代替业务验收。

### 在线 r2 复现（后续业务测试的默认入口）

使用同一镜像、2 GPU 和挂载同一私有 CPFS 的 worker。模型权重、六图/source
需具备团队访问权限，不下载到公共托管。现有准备目录为：
`/cpfs/world-model/sglang/evals/zing-online-20261008-prepared-r3`。
其中 `receipt.json` 保存素材校验和、模型来源和三份完整 profile；`PREPARED`
只是准备完成标志，启动脚本仍重新检查素材 SHA 和硬件。缺少该目录时先按仓库
WHITEBOX_VIDEO_SMOKE.md 重建六图资产及 progressive profile，不能换用同名五图 case。

已测工具完整包保存在私有 OSS（挂载到 `/mnt/oss`）的
`world-model/code/sglang/zing-online-20261008-r1/`。复现使用该目录里的 **tools-r4**，
不是较早 tools 包。下面只在有上述资源的 worker 内执行，不会自动申请云资源：

```sh
mkdir -p /workspace/joint-source /workspace/joint-tools
cp /mnt/oss/world-model/code/sglang/zing-online-20261008-r1/source.tar.gz /workspace/source.tar.gz
cp /mnt/oss/world-model/code/sglang/zing-online-20261008-r1/tools-r4.tar.gz /workspace/tools-r4.tar.gz
cd /workspace
python - <<'PYVERIFY'
from pathlib import Path
import hashlib
expected = {
    'source.tar.gz': '82fe6f7d4a4b5694cb60eabb1cc2bbd48ce4047a70aab36b399fb0b9cebed47c',
    'tools-r4.tar.gz': 'fc4aaca94c337d43da08d046095f4b00fd1ded1c9ac62e001251fd08b02f1a95',
}
for name, digest in expected.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
PYVERIFY
tar -xzf source.tar.gz -C joint-source
tar -xzf tools-r4.tar.gz -C joint-tools
# 原 run.sh 的结果目录已存在；只改为新的、不存在的输出目录，保留 prepared-r3。
python - <<'PYPATCH'
from pathlib import Path
p = Path('/workspace/joint-tools/run.sh')
s = p.read_text()
old = '/cpfs/world-model/sglang/evals/zing-online-20261008-r2'
new = '/cpfs/world-model/sglang/evals/zing-online-review-new'
assert old in s and not Path(new).exists()
p.write_text(s.replace(old, new))
PYPATCH
bash /workspace/joint-tools/run.sh
```

源码包 SHA：`82fe6f7d4a4b5694cb60eabb1cc2bbd48ce4047a70aab36b399fb0b9cebed47c`。
工具包 SHA：`fc4aaca94c337d43da08d046095f4b00fd1ded1c9ac62e001251fd08b02f1a95`。
运行前后保留修改后的脚本哈希，不能说修改后的整个包与历史包逐字节相同。

包内 `run.sh/server.py/clients.py/plan.json` 给出了完整启动参数与请求；包括
resident SP2、native stable decoder、source tile auto/performance_auto、FA4、
24 FPS paced source、120 帧 832×480、seed42、四步、window21/sink3、无音频。
每档独立 server：1 次完整预热 + 3 次正式 + 3 次独立诊断；正式不插入额外 CUDA
事件同步，哈希只在预热采集。模型加载和预热（含编译）均不计入正式客户端时延。
TTFF 从建立连接前到首个合法视频批次到达，不是浏览器显示时间。

原 source RGB SHA：`96dcb9c925fcc04f214d975277086a5d8444ae0dfcb31f138f9f8061b8098c6c`。
源视频与六参考图单独校验，不以这一个 RGB 哈希代替六图校验。

原始在线结果位于 bucket `leap-worldmodel-thailand-single-az` 的
`world-model/results/zing-online-20261008-r2/{summary.json,results.tar.gz,receipt.json}`。
归档 SHA：`d887a65859bd8b33c871565a8e0eafb82ca4956fe1400d8a3845f5b36f6847a0`。
包内含双 rank 日志/阶段/tile/形状回执和九份视频。GPU `dlc1nhby8aj6vct4`、
后处理 `dlc138ff9mv8fwdq` 均成功结束；视频/原始素材不发布公网。

离线 kernel ABBA 只作为历史证据，不安排为后续业务基线。其完整启动/工具包在
同一 bucket 的 `world-model/code/sglang/zing-kernel-e2e-20261008-r1/`；执行入口
`run.sh` → `validate_real_checkpoint.py`，模式 `SGLANG_ZING_JOINT_KERNEL_AB=baseline|fused`，
由实验 sitecustomize 显式导入 hook。仅设置环境变量而不导入 hook 不会启用融合。
源码、工具和结果包 SHA 见 [离线报告](https://github.com/seedleap/sglang/blob/f51114396f/benchmark/zing_h3_causal/JOINT_ATTENTION_E2E_AB.md)。

## 请逸文重点判断的三件事

逸文已经给过 [第一轮分析](https://github.com/seedleap/sglang/pull/230#issuecomment-6059879110)，
这里补齐最新在线结果，不要求重复泛泛调研或直接重写 kernel。

1. 组件收益→DiT 收益是否已能用可优化占比解释？旧受影响 FA4 约 1.59 s/rank，
   不能用全部 FA4 时间，也不能跨 rank 相加。旧估算与当前在线 shape 不同。
2. 现有 FA4 之上，是否还存在值得投入的大收益点？一次 launch 不等于历史 KV
   只读一次，现有 FA4 已有 Q tile 复用；缺少 HBM/计算瓶颈证据，不能按三分支猜三倍。
3. 给出继续/暂停的建议，以及最小的下一步实验。如果继续，优先真实在线形状的
   同进程逐调用交替 A/B，分别记录两卡调用数与编译成本；仅用于上界估计的 H=0/
   返回 Q 必须标为数值无效，不能混入正确性或业务性能数据。

证据缺口：融合版尚无新的完整 profiler；计数器受权限限制（逸文已报告
ERR_NVGPUCTRPERM），HBM 假设未实测；离线约 1% 收益存在跨进程波动；在线
专用 kernel 白名单尚未覆盖；performance 输入侧数值差异尚未用固定条件/source
latent 回放隔离。该组合不能宣称无损。不要为了交接重跑同一套无针对性的 GPU 测试。
