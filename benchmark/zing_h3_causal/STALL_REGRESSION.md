# 完整卡顿回归的复现入口

核心回归入口保持不变。Encoder、Decoder 和 DiT 数值专项工具统一放在
[`diagnostics/`](diagnostics/README.md)；该索引同时说明本轮旧版本适配的退役条件。
旧冻结实验继续使用其原始提交和路径，新实验从已推送源码重新准备。

本页说明如何运行与保存 `serving-stall-v1`；判定口径见
[回归规范](../../docs/docs/sglang-diffusion/zing/specs/serving-stall-regression.md)，
组件计时见 [STAGE_TIMING.md](STAGE_TIMING.md)。此工具只用于本轮新建的隔离实例。

需要定位 native／DiT replay 数值差异时，可独立选择
`native_rng_probe: true`，不与其他专用模式混用。它仅运行两档原有 403 帧短 case
的自然 EOS，原生捕获后回放前三块，记录每 rank 前三次 drain 的真实 chunk 身份、
有效 CPU／CUDA RNG 状态、内部重加噪张量及模型 dtype。首 drain 另保存准备后的
输入、实际模型架构、配置及每个参数头尾八个元素，并分块计算全部加载后参数的
SHA256；参数样本相同不能证明全量权重相同。旧探针缺少完整参数哈希时，分析器
须保留未观测状态，不能凭两侧都缺失判定完整参数一致。此哈希含 GPU→CPU 传输与
CPU 开销，只在独立数值诊断首 drain 执行，不能用于正式性能统计。
普通 `forward` 路径保存实际 kwargs、Host metadata、缓存事务和输出；固定 Graph
首次 warmup 另保存 `FixedForwardPlan.execute` 实际输入、计划几何和输出，编译／
CUDA capture 内不读回。已缓存 Graph replay 可能绕过两种方法，分析器必须明确
标为未观测，同时仍比较其他已采输入。计划几何不包含完整历史 KV 存储数值；
不能将这些坐标相同解释为历史 K/V 已逐值相同。
首个非编译、非捕获的固定前向还记录 embedding 与各主 block 输出的完整字节
SHA256 和最多 32 个值，不保留逐层完整副本；同时记录实际 SDP/cuDNN 全局开关。
分析前核对每侧实际观测到一条 embedding 与全部 50 个 block，按第一个不同摘要
定位计算边界。双方都缺失不代表一致。该读回会同步 GPU，只属于数值诊断，
不可用于性能统计，也不能仅凭某层摘要不同认定该层的具体算子有错。
第 0 个 block 进一步保留实际 Linear 输入／输出（含 FP8 tuple）、weight 布局、
quant method/backend/policy 的实际类型及简单配置，以及融合预量化的实际分支。
embedding 和该 block 的 Linear 保留完整 CPU 张量供独立计算核查，其余 block
只留摘要；核对实际覆盖的 Linear 名单，不能把未进入的采集分支当作一致。
固定计划准备后另采集实际 buffer（含 RoPE 和延迟生成的 timestep 频率），
Graph owner 前后记录 route、计划数量、实际输入／输出及会话状态。
route 和缓存数量的历史差异只描述执行路径，不能单独证明数值差异的原因。
读取 RNG 不推进随机流；
噪声 CPU 读回跳过 compile／CUDA capture，并在 replay 预热结束后恢复全部 hook。
此诊断有同步与写盘开销，既不提供正式速度，也不代表完整长 case 或耐久通过。
`diagnostics/dit_replay/inspect_dit_norm_boundary.py` 可在零 GPU 实例独立读取已冻结的首层操作数，
核验双方 hidden、shift/scale 表、indices、norm weight 全量 SHA 与 eps；需要从
checkpoint 取权重时仍必须匹配实际加载后的完整 SHA。CPU 舍入候选不复现 GPU
归约树与乘加融合，候选不一致不能排除对应 GPU 路径，也不构成数值验收。

跨进程逐位比较失败时，默认耐久仍中止。需要独立收齐卡顿证据时，显式选择
`dit_resident_collection: true` 并提供两档原始完整 `capture_root`，只启动四个
专用 DiT rank，Encoder／Decoder 为空。可用 `matched_allocation` 强制核对原
候选分配的全部 GPU UUID；八卡分配时仅四卡参与，四卡闲置仍计入分配成本。
该模式使用 `--collect-soak-on-numerical-mismatch`：完整预热必须结构合法且有限，
所有 rank 同意判定后才开始三轮各至少 600 秒的完整会话工作；数值失败不阻断
另一档的诊断采集。原始 FAIL 保留，不写验收 PASS；两档采齐后整体仍以数值失败
退出。NaN／Inf、shape／dtype／数量错误、执行、EOS 或资源释放失败不会获准继续。
此模式的显存和时延属于诊断 workload，不能冒称 native 数值等价或正式优化收益。
显式设置 `minimum_round_seconds: 0` 时，仅独立收集两档各三次完整上下文回放，
输出为 `D1-*`，不执行或声称十分钟耐久。它可与 baseline 使用同一份冻结 capture，
补齐默认严格耐久中止后尚未执行的另一档；失败和未执行项仍保留，不用补测覆盖。
默认值仍为 600。两种采集均保留数值 FAIL，结构、执行或清理失败立即中止。

## 准备与分配

逐次云作业的配置、回执和冻结源码随原始产物归档，入口见
[PR #580](https://github.com/seedleap/sglang/pull/580)。本目录维护可复用工具和复现方法，
不维护日期、节点和实例绑定的一次性作业清单。整理 PR 不删除原始运行目录或历史证据。

1. 按 [测试机器清单](../AI_INFRA_TEST_MACHINES.md) 核实整机作业、归属与空闲卡，
   排除产品 demo 主机。由调度器分配设备，不借用其他作业的可见卡。
2. 在零 GPU 环境准备固定源码、模型、Lynn、依赖与真实素材。模型身份沿用
   [CLIENT_BENCHMARK.md](CLIENT_BENCHMARK.md)；输入保留四条完整 case，冻结
   两档 RGB 的帧数、实际宽高、字节数、SHA256 与派生命令，不循环或截短长素材。
3. 冻结源码完整提交和对应 Git 对象，使用 `frozen_source.verify_source` 核验全部
   tracked 文件。资产清单记录实际文件的路径、解析路径、大小和 SHA256。
   CPU 环境实际导入 `GPUWorker`、Zing pipeline、Encoder runtime 和 Lynn loader，
   核对两档 geometry、完整 50 层、采样、SP、KV 池与所有启动端口。
4. 生成 `CPU_READY.json`，保存 `passed: true`、`source_revision` 以及
   `controls: {绝对文件路径: SHA256}`。controls 至少绑定最终配置、模型／输入身份
   清单、pipeline、conditioning profile、执行入口及随附 worker/probe；不能手写
   一个空 controls 来跳过预检。GPU 包装入口在首次 CUDA 导入前再次核验源码、
   controls 和资产；原始作业命令、包装入口及其 SHA 一并保存。

配置示例的路径均须替换为本轮冻结路径。`root` 下的 `results` 必须尚不存在，
不能覆盖或混用旧轮结果。下面为 B300 / CC 10.3、SP2 的四张专卡；其他硬件须以
实际 capability 修改 `expected_cc`。

```json
{
  "isolated_instance": true,
  "root": "/data/stall-run/work",
  "source": "/data/stall-run/source",
  "deps": "/data/stall-run/deps",
  "model": "/data/stall-run/model",
  "media": "/data/product-media",
  "frozen_inputs": "/data/stall-run/FROZEN_RGB.json",
  "conditioning_profile": "/data/stall-run/conditioning-profile.json",
  "pipeline": "/data/stall-run/pipeline.json",
  "cpu_receipt": "/data/stall-run/CPU_READY.json",
  "before_import_gpu_apps": "/data/stall-run/GPU_BEFORE_IMPORT.csv",
  "archive_dir": "/mnt/private-stall-run/raw-blobs",
  "diagnostic_switch": "/dev/shm/owned-stall-run-diagnostic",
  "expected_cc": [10, 3],
  "sp": 2,
  "mem_fraction_static": 0.8,
  "role_devices": {"dit": [0, 1], "encoder": [2], "decoder": [3]},
  "allow_shared_role_devices": false,
  "ports": {
    "http": 24000,
    "encoder": 24101,
    "qwen": 24102,
    "decoder": 24103,
    "scheduler": 24105,
    "master": 24106,
    "encoder_master": 24201,
    "decoder_master": 24203,
    "replay_master": 24211
  }
}
```

`role_devices` 使用进程内 CUDA 索引，外层 `CUDA_VISIBLE_DEVICES` 保持调度器提供的
数字设备映射；工具核对 CUDA 与 NVML 的实际 UUID。SP4 专卡配置改为 DiT `[0,1,2,3]`、
Encoder `[4]`、Decoder `[5]`。平台支持碎片分配时，设置 `allocated_gpu_count: 6`，
同时在调度规格中请求六卡，六卡全部用于该完整实例；不能使用其他作业已分配的卡。
若平台须分配八卡，外层包装先核验全部八卡空闲，
保存八卡分配／六卡使用／两卡闲置的独立回执，再只向 Runner 暴露六张使用卡。
不能省略闲置分配的资源成本，也不能仅改 `sp` 而沿用 SP2 capture。

六卡分配／使用，或八卡分配、六卡使用的 SP4 完整套件可用 `stall_job.py prepare --config job.json`
在零 GPU 作业内准备，再用 `stall_job.py run --config <root>/job-config.json` 执行。
配置需提供新的 `root`、私有挂载目录 `output`、已推送的 `source_revision`、
`source_archive`、`source_objects`、绑定两者 SHA256 的 `source_files`，以及
已核验的资产 `reuse_root`、独立 `diagnostic_switch`、`ports` 和 `timeout_seconds`。
源码提交必须包含完整回归工具；不同 runtime/harness 的 main 对照另行绑定两份身份。
准备阶段验证 Git 树、实际模块导入、两轮 CPU 契约和资产实体，冻结所有实际读取的
配置副本；GPU 阶段再次校验，并在首次 CUDA 导入前核验八卡空闲。子进程失败或
超时后清理本次进程树，保留失败与原始元数据；退出码零仍须独立读回验收，完整
RGB、capture tensor 等大型原始数据另按后文留存，不能把元数据压缩包冒充全量归档。

`stall_job.py` 默认仍分配八卡、使用六张专卡。仅有四张空闲卡时，可显式设置
`allocated_gpu_count: 4`、`role_devices: {"dit":[0,1,2,3],"encoder":[1],"decoder":[2]}`
和 `allow_shared_role_devices: true`。这是完整 SP4 的共享角色实验，不是已验证的
专卡替代配方：必须先跑两档原有短请求检查显存、通信与收尾，再进行完整矩阵；
记录每卡实际 UUID 和共享进程，driver 显存不能冒称某个组件独占。正式速度只与同一
共享布局的基线比较，不混入专卡统计。CPU 前置会验证角色准入与分配规模；GPU
入口只检查获配卡，禁止清理其他卡上的任务。独立 DiT 回放前仍停止本实例的 E/D。

main 对照使用相同工具但独立指定 `runtime_source_revision` 和
`runtime_source_archive`，harness 的 `source_revision` 保持为包含测试工具的冻结提交；
两份 archive 与 Git objects 都须列入 `source_files`。另外提供已完成候选的
`capture_root` 和 `matched_allocation`。零 GPU 准备会完整哈希两档 capture 文件并
验证实际 baseline D1 adapter 的导入；GPU 开始前再次核验实体及同一组八卡 UUID。
此模式先运行双档各三次重启的 S3，再运行完整 S1/S2、CUDA 组件诊断及生命周期，
然后独立收集两档冷组件诊断与 D1；不覆盖 baseline 耐久。S3 单次重启或冷诊断
请求失败须保留失败和未执行项，清理成功才继续独立采集；清理失败立即终止。
诊断数据不能抵消原失败。candidate 在正式矩阵前另做冷诊断与完整 capture，
与此模式的阶段顺序不同；每个 case 的预热不能自动证明缓存前置完全相同，
报告须注明并核验后才计算正式轮收益。adapter 只提供有类型的输入递归映射及不冻结的 GC 测量接口，
不得替换 baseline 模型计算、Graph 或 KV 方法；已有该接口的 main 必须重新审查。

`encoder_gc_comparison: true` 选择独立的 Encoder-only 静态 GC 对照，不叠加默认
完整套件，也不能与 main runtime 模式同时设置。同一分配按 off/on 串行执行，
每臂每档完整长 case 预热一次、正式三次、诊断一次；DiT 的 GC 策略始终开启。
正式轮无 CUDA 阶段计时、GC 观察标志关闭；独立诊断启用阶段计时与被动 gen2
记录，raw 预算为 64 GiB，归档前关闭诊断标志。被动观察不改收集策略；事件须
按明确角色和 session 关联，重叠本身不能证明全部停顿都由 GC 导致。
`cpu_contracts` 可指定冻结源码中的 CPU 测试文件，准备会执行两轮并拒绝失败或跳过。

可通过显式 `role_devices` 测试已有 Lynn Decoder 的多卡图块并行。例如八卡内
保持 DiT `[0,1,2,3]`、Encoder `[4]`，将 Decoder 改为 `[5,6,7]`；编排按
Decoder 设备数启动对应数量的本地 worker。此模式要求三个角色使用独立设备，
共享卡回退仍仅允许单 Decoder。必须保存实际角色映射及逐 rank 显存，固定其他
条件与单 Decoder 对照；图块批量大小变化可能影响数值，不能凭相同 checkpoint
或请求成功代替准确性验证。该配置不改变默认布局，性能和准确性须独立验收。

## 执行

在本轮获配的 GPU 作业内、首次导入 CUDA 前保存占用快照。使用预检时的相同
解释器、依赖与源码；不要把以下命令直接放到产品容器执行。

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv \
  > /data/stall-run/GPU_BEFORE_IMPORT.csv

PYTHONPATH=/data/stall-run/deps:/data/stall-run/source/python \
SGLANG_LIGHTWEIGHT_RUNTIME=1 PYTHONDONTWRITEBYTECODE=1 \
python3 /data/stall-run/source/benchmark/zing_h3_causal/run_isolated_stall_suite.py \
  --config /data/stall-run/runner-config.json
```

默认顺序为两档各三次重启与每次三个暖请求、独立 Graph/GC 安全检查、重新采集
两档完整长会话 DiT 输入、完整实例 S1/S2、独立 CUDA 组件诊断、单活跃会话拒绝／
断连恢复，以及两档完整 DiT 三轮回放和常驻耐久。常驻耐久每档三轮，每轮所有 rank
累计至少 600 秒完整会话工作，复用原有长 case；完整会话正常结束后立即开始下一次，
不拼接成更长上下文。S3 位于 capture 前；进程重启和全新容器
的缓存条件分开记录。完整矩阵耗时较长，保留状态与失败回执，不能用短测冒充完成。

隔离启动工具要求 Linux。模型和 CPU 任务子树经 `owned_process.py` 监督；验收必须
同时检查原任务退出码和完整清理回执，不把 launcher 已退出当作子进程已释放。
异常清理保留失败并停止后续复用，不按旧 PID／PGID 清理其他任务。独立复制的
job／CPU-controller 入口先设置并核验 harness，再加载监督依赖；专项 norm 探针
复制执行时须同时交付同版本 `owned_process.py`，CPU 回执绑定实际模块摘要。具体边界见
[卡顿回归契约](../../docs/docs/sglang-diffusion/zing/specs/serving-stall-regression.md)。

第二会话可能在 WebSocket 握手完成后、客户端发送 init 前就被拒绝。生命周期用例
允许该次发送因关闭失败，但仍须读取真实 error 帧，并核对 1008 关闭码及明确的
准入拒绝原因；裸关闭或其他错误不得通过。随后继续检查原会话产出、断连及新会话
完整恢复，不能仅凭第二连接关闭就认定生命周期通过。

Graph/GC 检查后，两档各启动一次独立进程做首请求组件诊断，再进入 capture。
这提供 Qwen／参考 VAE 未命中请求缓存时的实际阶段数据；暖请求缓存命中而没有
执行该组件时标明跳过，不能写成耗时为零。此诊断仍可能使用 S3 填过的磁盘编译
缓存，不冒称全新容器首调用。

Graph 安全检查同时运行 `check_cuda_graph_stream.py`：在真实 denoise/prefix
backend 下申请 256 个普通 stream，再让后台线程在捕获时同步普通 stream，检查
它与进程私有 capture stream 不重合、输出正确且 caller stream 恢复。此项使用
独立 GPU 子进程，不把刻意的并发同步诊断计入正式吞吐或冷请求时长。

当前完整套件显式采用 `mem_fraction_static=0.8`，按实际加载前后显存建立共享 KV
容量；可在冻结配置中明确修改，baseline/candidate 必须相同。旧的固定 pool size、
prefix rows 等字段已由主线移除，不能把旧 pipeline JSON 直接用于新版本。重新
capture 保存实际 slot rows、运行时预算和 hidden widths；D1 必须按该策略重建，
不接受缺少当前 KV 策略的旧 capture 冒充新版本回放。

当前 component capture/replay 只保存静态 sink 策略和 prompt 更新，不支持动态
`sink.off` / `sink.refresh`。开启 capture 时明确拒绝这些控制，沿现有异常释放路径
结束且不生成 `COMPLETE.json`；普通非 capture 请求照常处理。固定 case 的结果不能
外推为动态 sink 回放验收；部署缺省变化也不能改写冻结 capture 中的显式策略。

普通 benchmark 保持默认 2 GiB 留存预算。完整卡顿套件的 S3、正式 S1/S2、capture
与生命周期显式使用 `metrics`：完整接收、校验并哈希每字节 RGB，保存原始事件与
回执，不写完整 RGB 文件。独立组件诊断显式使用 `raw`，为最长 736 请求声明
64 GiB 单请求临时上限，随后按块校验归档；这项完整原始数据留存与普通代表媒体
预算分开。新旧写盘模式不能混算吞吐收益，报告列出每轮实际 artifact mode。

## Decoder 跨会话资源归因

默认 S1/S2 在同一部署进程内串行运行多个会话。修改 Decoder stream／资源生命周期，
或观察到会话结束后 reserved 持续增长时，随默认矩阵追加
`diagnostics/decoder/benchmark_decoder_resident.py`。它直接运行真实 `_DecodeService`、共享内存发布、
消费确认和正常 EOS／释放，隔离 DiT 与网络，用于归因，不能代替完整实例验收。

`--cases` JSON 必须包含 `480`、`736`，每档的 `short`、`long` 分别指向完整
403、15921 帧 capture 目录。每档先完整预热这两条 case，再串行运行 40 次短会话和
3 次完整长会话；模型只加载一次，480 后接 736，每次销毁会话线程后才能开始下一次。
会话之间不执行 `empty_cache`、手动 GC 或进程重启，不拼接／填充上下文。

先在零 GPU 环境传 `--validate-only`，核验完整冻结 runtime、模型权重选择与文件、
capture 完整性和哈希，并生成 `--cpu-ready`；GPU 执行使用同一组参数且移除此开关。
必填参数还包括 `--pipeline`、`--runtime-source`、`--runtime-revision`、
`--expected-gpu-uuid`、`--output`。外层调度器负责先核对整次分配的设备身份和占用，
只将其中一张自有 GPU 暴露给此工具；工具再核对该 GPU UUID。前后版本必须使用同一
物理 GPU、模型、输入、执行选项和会话顺序，每个版本各用一个新模型进程。
分配器环境变量也绑定到 CPU 回执，GPU 执行前变化即拒绝。对照部署配置时，须在
CPU 准备与 GPU 执行中显式使用对应的 allocator 设置，不能把继承的未知默认值
当作生产配置。结果另保存实际 backend 和 allocator settings；固定位置的 snapshot
保留完整结构。旧工具仅保留 segments 时，不据此反推完整分配器配置。

每次会话保留逐块时长／输出哈希、实际 stream／线程、allocated／reserved／峰值、
分配重试／OOM、RSS 与线程数；40 次短会话另在固定位置保存 allocator snapshot。
两条完整预热保留 RGB，后续每字节输出均哈希并与对应预热比较。显存包含固定驻留的
capture 输入，是整进程口径；reserved 增长本身不等于活动 tensor 泄漏。
Host publish 包含未完成的 decode、转换、D2H 和共享内存发布，不冒称纯 GPU decode。
工具写完结果后还必须检查进程退出码和清理结果；有结果文件不能覆盖清理失败。

## 读回与报告

所有 writer 结束后，可用零 GPU 环境从完整原件重建指标：

```bash
python3 benchmark/zing_h3_causal/collect_stall_report_evidence.py \
  --root candidate=/data/candidate --root baseline=/data/baseline \
  --harness-source /data/frozen-harness --source-revision FULL_COMMIT_SHA \
  --output /data/stall-report-evidence.json
```

工具按 case、目录阶段及回执的 warmup／measured phase 分组，保留失败与缺失，
从事件重新计算播放欠载。`stage_request_totals` 是单请求、单 rank、单组件的累计值；
`stage_comparison_groups` 是 span 分布，两者的样本量与 P95 不可互换。
不同组件可嵌套，不相加当作 E2E。allocator 显存保留实际进程／阶段范围。
没有原始时区或时间配对的 driver 采样仅保留整轮未归因统计，不按文件 mtime 推断
请求或分辨率；该采样最大值不是请求峰值。JSON 始终不自动授予整套验收通过。

正式轮的 `zing_source_step` worker 日志保留 `stage_host_ms`，frame header 不透传
这些 Host 计时。定位时可用 `read_stall_run.py --log work/results/instance-dit.log`
配合可重复的 `--source-step-session <精确会话ID>` 与
`--source-step-frame-window 586:1038` 读取原始步骤，另传 `--root`、`--output`。
帧窗口是内容坐标，不是墙钟区间；快照绑定读取前的字节边界及原始行 SHA，日志
后续追加、未收齐会话和不完整行均不能视为最终整件核验或因果结论。
同一日志中的参考图缓存收据另列原始行位置及 logger 前缀，可能包含其他请求；
累计快照须先核对相邻会话／步骤再做差分，不能冒充某张图或全部 rank 的命中记录。
已完成请求的选定事件／回执可用 `--original <相对路径>` 完整导出并核验 SHA；
该选项拒绝读取期间变化的文件、secret/private 路径和大型 tensor，单文件最多
64 MiB、每次最多四件。它不替代整轮 writer 结束后的原件归档与独立核验。

诊断采集使用隔离实例自有 `/dev/shm` 子目录，避免阶段 writer 逐条写共享文件系统
造成队列积压。只在 `isolated_stall_worker.py` 中启用保留原始阶段文件的测量覆盖；
产品入口不启用。结束时先等待 writer 排空，再停止自有角色，将 RAM 中的原始文件
复制到 `results/timing`，保留 `TIMING_RETENTION.json`；未排空仍判采集失败并保存片段。
普通请求仍使用 20 秒 keepalive；接收阶段数据的诊断请求使用有限的总请求超时预算。
combined 通道完成后停止应用 heartbeat，split 通道继续按 input completion 释放。
服务端已停止会话 listener、完成资源清理后，在收集 profile 期间继续消费终态输入，
避免在途 heartbeat 堵住单条接收队列；收集结束后先退出该 reader，再发送 profile
并由原逻辑独占接收 ACK。此修正不覆盖资源清理本身长时间不读输入的情形。

`diagnostics/decoder/run_decoder_layout_pilot.py` 可选比较同一八卡分配中的 E1+SP4+D1 与 E1+SP4+D2，
各自使用互不重叠的端口和独立子进程。先以两个完整 15921 帧 latent 序列比较
当前 Decoder 的逐位输出，并分别计时三轮；再执行两档完整 403 帧短 case，各布局
每档冷诊断一次、预热一次、正式三次和独立 CUDA 诊断三次。纯 Decoder 明确采用
当前生产 Decoder 的 NCCL 策略；不能与此前 MIN4 的专项结果混算。
八卡均计入分配：纯 Decoder 阶段用两卡，实例 D1 用六卡、D2 用七卡。
`diagnostics/decoder/analyze_decoder_parallel.py --worlds 2` 只产生后续完整矩阵的候选，不提供完整验收。

- `results/STATUS.json` 和进程退出码只表示采集进度；`complete` 不代表容量通过。
  核对两档四条 S1、长流 S2 的正式三轮、每次 S3、两档生命周期、所有 D1 rank
  和 Graph/GC 结果；失败与未执行分别列出。
- 正式请求回执、`events.jsonl` 与诊断轮分开。按固定 250 ms 播放缓冲报告所有
  underflow、>250 ms 与 >1 s 事件；供满、实时、冷首请求分别解释，不增大缓冲
  掩盖欠速。三组件阶段、逐 rank 显存和计时边界按规范报告。
- D1 核对完整 15921 帧、1873 块、真实 causal/KV 演进、预热逐块数值检查及正式
  三轮。SP／layout 改变后重新 capture，不拿另一拓扑的数据替代当前配置验收。
  块计时覆盖完整 `append+drain_ready`；ready barrier 仅是已预加载 GPU 输入的
  就绪上界。内部融合 Graph 的独立 phase／step、allocator reset／compaction
  时间保持未测，异常块需另做 nsys/Graph 节点诊断，不拆成 eager 冒充正式结果。
- 保存诊断轮完整 RGB、capture/replay tensor、全部轮次的 Host/CUDA／交付原始记录、GC、profile、命令、
  身份清单和失败样本。大对象归档后从独立客户端逐路径读回，检查有序分块和整件
  SHA256；同 SHA 的不同路径分别验证。tar 包另核对全部成员的大小与 SHA。
- 共享进程／整卡显存范围、1 秒 driver 采样的峰值下界、736 source 编码实际仍为
  832×480 等限制写入报告。只在全部写入结束后生成最终证据完整性结论。

发布报告沿用团队受限 review-cdn 入口，先核实源桶映射与办公网／VPN访问控制；
不把签名 URL、凭据或运行时 secret 配置放入仓库或报告。

终态整批留存可使用 `archive_owned_stall_run.py`：先以 PAI 实时查询建立 proof，
`scope` 为 `Exact PAI job terminal states checked before bulk retention`，
`roots[目录名]` 绑定实际 `root`、`job_id`、`user_id` 与终态 `status`。
传入 `--root`、全新 `--output`、`--proof`、`--owner-user-id`、`--store-root`、
已冻结的 `--harness-source` 和完整 `--source-revision`，在零 GPU 作业内执行。
它保留原件，包含失败片段、tensor、trace 和标准 HTML；权重／缓存／链接资产及
secret/private 路径不打包。既有 RGB 归档也逐个精确对象路径重读，不信任旧的
`verified` 标志；零字节遗留文件仅在原归档实体核验成功后单独记录。
仅剩 `.rgb.archive.json`、原 RGB 已由此前流程移走的记录也纳入实体核验，
标注原件当前缺失；只有有序 part 路径、offset、大小和 SHA 全部相同的组合可复用
本次读取的结果，相同整件 SHA 的不同对象路径必须分别读取。

从另一个零 GPU 实例运行 `verify_owned_stall_archive.py --index <INDEX.json>
--store-root <挂载根目录> --output <新proof路径> --reader-job-id <实际CPU作业ID>`，
重新流式核对所有 part、整件、每个 tar 成员与 RGB，保留失败 proof。
归档和核验成功只证明字节留存完整，不能把失败请求或未执行测试改标为通过。
