# #196 渐进空间 padding 真机验收

完整编码/E2E性能门槛：**通过**。根据用户要求，运行时默认 **reflect2d**，可显式设置 **off** 回退；结果只适用于本次冻结环境。完整输出RGB一致性、媒体规格与业务画质分别报告。

**人工复核更新（2026-09-28）：用户观看后反馈“我肉眼看了没有问题”，记录为用户主观复核通过。此前助手仅据抽帧判定不通过，不能替代用户的观看结论。A/B RGB完全一致；覆盖范围仍为重复源压力流。**

## 无 profiler 性能

同节点独立进程ABBA，每模式每时长六次正式样本，表中取中位数；SP2编码取每次较慢rank。

|SP|源时长|encoder off→candidate 秒|耗时下降|首latent off→candidate ms|编码间隔p95 off→candidate ms|编码门槛|
|---|---|---:|---:|---:|---:|---|
|1|10s|9.437853→8.468109|10.275%|94.273→86.664|277.668→248.037|通过|
|1|60s|55.077421→49.472968|10.176%|94.578→86.674|277.583→248.339|通过|
|2|10s|4.914433→4.412912|10.205%|51.343→46.793|144.308→129.185|通过|
|2|60s|28.604457→25.765635|9.924%|50.886→46.498|144.137→129.392|通过|

|SP|源时长|E2E off→candidate 秒|FPS off→candidate|TTFF off→candidate 秒|输出间隔p95 off→candidate 秒|源到输出p95 off→candidate 秒|时延门槛|RGB逐位|
|---|---|---:|---:|---:|---:|---:|---|---|
|1|10s|44.745→43.675|5.364→5.495|4.337→4.221|4.770→4.647|20.938→20.511|通过|一致|
|1|60s|272.627→267.577|5.282→5.382|4.314→4.233|5.242→5.142|21.284→20.933|通过|一致|
|2|10s|39.415→38.829|6.089→6.181|4.328→4.293|4.279→4.175|18.202→17.833|通过|一致|
|2|60s|232.953→233.928|6.182→6.156|4.340→4.346|4.416→4.457|18.058→18.188|通过|一致|

E2E为localhost当前source.append输入到RGB客户端写出，包含真实六图；不是WAN展示FPS。预热和profile均不进入上表。全部原始重复与范围见metrics.json。

## 全部服务正式重复（秒）

按ABBA中的进程与repeat顺序保留；未丢弃任何慢样本。

|SP|时长|模式|六次正式耗时|
|---|---|---|---|
|1|10s|off|44.320329, 44.440604, 44.777961, 44.712560, 45.349595, 45.376567|
|1|10s|reflect2d|43.305510, 43.373331, 43.840923, 44.245262, 43.509237, 45.630263|
|1|60s|off|272.134952, 272.748948, 272.504527, 277.491718, 277.087982, 272.374715|
|1|60s|reflect2d|266.202431, 267.076655, 267.824484, 268.176860, 270.568276, 267.329272|
|2|10s|off|39.756808, 39.405484, 39.424827, 39.333206, 39.790708, 38.894186|
|2|10s|reflect2d|38.821150, 38.137805, 38.395558, 39.680397, 38.837679, 39.261376|
|2|60s|off|235.491121, 232.029064, 232.075806, 235.213459, 231.315616, 233.829612|
|2|60s|reflect2d|234.096069, 234.178999, 231.975689, 234.435138, 233.759581, 232.307960|

## 全部编码正式重复（秒）

|SP|时长|模式|六次完整编码耗时|
|---|---|---|---|
|1|10s|off|9.441604, 9.432217, 9.429268, 9.454932, 9.445157, 9.434101|
|1|10s|reflect2d|8.464329, 8.463307, 8.469102, 8.469584, 8.468927, 8.467292|
|1|60s|off|54.996002, 55.031344, 55.304269, 55.075661, 55.079180, 55.085875|
|1|60s|reflect2d|49.394248, 49.396518, 49.652305, 49.607394, 49.496249, 49.449687|
|2|10s|off|4.889962, 4.894467, 4.904514, 4.932961, 4.924352, 4.931731|
|2|10s|reflect2d|4.416355, 4.433975, 4.409468, 4.407867, 4.440105, 4.409344|
|2|60s|off|28.610506, 28.646927, 28.596232, 28.627653, 28.572268, 28.598408|
|2|60s|reflect2d|25.762890, 25.734282, 25.764933, 25.766337, 25.773144, 25.780754|

## 数值、显存与缓存

- SP1：140项moments/posterior字节比较，最大绝对误差0.0、relative L2 0.0。
- SP2：280项moments/posterior字节比较，最大绝对误差0.0、relative L2 0.0。
- 独立完整tensor shape/dtype/finite检查共840项；逐tensor SHA与原始.pt SHA均保留。

|SP|模式|请求peak allocated范围GiB|请求peak reserved范围GiB|KV峰值GiB|encoder retained峰值GiB|
|---|---|---:|---:|---:|---:|
|1|off|229.489–229.489|229.922–229.941|94.184–94.184|3.160–3.160|
|1|reflect2d|229.489–229.489|229.922–229.941|94.184–94.184|3.160–3.160|
|2|off|179.264–179.302|179.686–180.057|47.097–47.097|1.584–1.584|
|2|reflect2d|179.264–179.302|179.686–180.057|47.097–47.097|1.584–1.584|

以上是请求期间PyTorch显存统计，不是冷加载全程峰值或整机nvidia-smi显存。独立核对102个session/rank的dispose回执，所有实际输出的保留状态字段（ready、KV、static、codec、bridge latent、decoder、metadata、encoder）归零。96个10/60秒session/rank有非EOS后半段样本，6个短profile没有这类样本但同样通过释放检查。缓存平台期逐步样本及10/60秒峰值集合见metrics.json的cache_plateau_observations；reserved allocator内存不直接当泄漏。60秒后半段：pending blocks保持2–4、bridge bytes为0，A/B每SP的allocated范围一致；详见cache-verification.json。

## 1/2 latent分组

|SP|源时长|模式|latent数|样本数|p50 ms|p95 ms|
|---|---|---|---|---:|---:|---:|
|1|10s|off|1|96|94.186|275.251|
|1|10s|off|2|168|274.759|275.324|
|1|10s|reflect2d|1|96|86.455|245.906|
|1|10s|reflect2d|2|168|245.401|245.836|
|1|60s|off|1|516|94.146|94.477|
|1|60s|off|2|1014|274.731|275.186|
|1|60s|reflect2d|1|516|86.476|86.637|
|1|60s|reflect2d|2|1014|245.516|246.007|
|2|10s|off|1|96|49.955|141.610|
|2|10s|off|2|168|141.477|142.690|
|2|10s|reflect2d|1|96|46.032|126.461|
|2|10s|reflect2d|2|168|126.363|127.216|
|2|60s|off|1|516|49.880|50.555|
|2|60s|off|2|1014|141.182|142.118|
|2|60s|reflect2d|1|516|46.057|46.557|
|2|60s|reflect2d|2|1014|126.488|127.207|

按实际产出的latent数量分桶，包含EOS尾部；1 latent分组不等同于纯首帧延迟。

## 媒体与 trace

全部68份实际RGB输出已读至EOF，校验长度、SHA、帧数，并完整解码；MP4为有损显示副本，另校验832×480/24FPS/video-only与时长。4份A/B并排对照为1664×480。

实际查看SP1/SP2、10/60秒四份A/B并排八帧采样：10秒可辨认人物、地形、建筑和天空，未见A/B新增差异；60秒中后段在两种模式与两个SP均有明显横向拉伸、彩色条带、人物和场景失真。这些是助手当时的抽帧观察，并非经确认的业务缺陷；用户后续观看认为无问题，当前主观验收以用户复核通过记录。RGB完全一致只证明本次padding没有新增数值/画面回归。木平台被Subject5明确要求，不作为错误。人工检查为抽帧而非全程主观播放，完整解码单独通过。见visual-review.json。

保留18份真实逐rank原始CPU/CUDA trace，manifest包含SHA；Perfetto UI验收回执单独交付。编码SP1 off/candidate含133/108个重叠完整事件警告，SP2 off两rank65/83、candidate59/72；UI明确移入overflowtrack而未丢失事件，显示嵌套存在歧义。服务原始trace另有slice_max_depth_exceeded，Perfetto会丢弃过深Python调用栈显示切片；不能称原始服务UI无丢失。原始JSON保留不变，另外提供仅去掉python_function事件的派生查看副本，逐事件摘要验证所有其余CPU/CUDA/阶段事件保持一致；具体原始与派生UI回执见service-perfetto-ui.json。

SP1独立17帧1/8/8 profile中，reflection kernel duration之和194.809→115.737ms；候选同时多2568个FillFunctor kernel。因此该差值只解释反射kernel，不等于完整padding scope或端到端收益。全encoder无profiler表已包含这些额外开销。

## 服务 trace 的实际 Perfetto UI 检查

12份原始服务trace全部实际打开；累计5472个Python调用栈显示切片因深度超限丢弃。12份派生查看副本均已实际打开，深度丢弃为0，剩余警告全为重叠事件移入overflowtrack；这些重叠警告不丢数据，但嵌套布局有歧义。派生副本明确删去python_function事件，不能称其保留Python栈；其余所有事件的分类数量及顺序摘要均与原始一致。

|配置|批次ID前缀 / rank|原始深度丢弃|原始重叠警告|派生重叠警告|
|---|---|---:|---:|---:|
|service-sp1-0-off|4e49b4d9 / 0|688|13337|1278|
|service-sp1-0-off|7d159e65 / 0|188|8448|1410|
|service-sp1-1-reflect2d|32c1f901 / 0|688|13300|1241|
|service-sp1-1-reflect2d|675d63ca / 0|188|8397|1359|
|service-sp2-0-off|46cb6018 / 0|724|15596|2311|
|service-sp2-0-off|46cb6018 / 1|724|14309|1024|
|service-sp2-0-off|b500ab8d / 0|206|8851|1250|
|service-sp2-0-off|b500ab8d / 1|206|9504|1903|
|service-sp2-1-reflect2d|3ff58926 / 0|724|14261|976|
|service-sp2-1-reflect2d|3ff58926 / 1|724|15964|2679|
|service-sp2-1-reflect2d|82ebffb6 / 0|206|9373|1772|
|service-sp2-1-reflect2d|82ebffb6 / 1|206|8600|999|

## 证据位置

- CPFS：`/cpfs/world-model/sglang/evals/zing196-20260924-r1/`。
- 私有OSS：`oss://leap-worldmodel-thailand-single-az/world-model/sglang/evals/zing196-20260924/`，gpu-r1/逐rank原始trace和报告、review-r1/metrics和媒体、service-assets-r1.json权重清单。
- GPU Job `dlccpajuyme5x3y6`；CPU review `dlc4e26mi9s8sjyc`。
- 未发布公共媒体或trace链接；本地完整交付目录见任务最终回复。

# #196 实验复现与证据索引

本记录适用于冻结实现 `359c187223aeb27e2ec3a232ac532e880c37c478`，基线 `b235905dd7f16a387ac9c801340c3368713c46e8`。只切换 `MINIMAX_H3_PROGRESSIVE_SPATIAL_PADDING=off|reflect2d`。当时默认 off；后续仅将默认值改为 reflect2d，显式 A/B 选项和冻结测量结果不变；#195 encode policy 和 #197 tile batching 未混入。

复现时先在独立工作目录检出冻结代码提交 `359c187223aeb27e2ec3a232ac532e880c37c478`，并令 `PYTHONPATH` 指向该目录的 `python/`。后续 PR 提交只追加验收文档；不能直接在不同 HEAD 上沿用旧 `--source-sha`，因为校验器会拒绝不匹配的源码。

## 环境与输入

- PAI 区域 `ap-southeast-7`，workspace 384，GPU Job `dlccpajuyme5x3y6`。
- 镜像 `zing-thailand-registry-vpc.ap-southeast-7.cr.aliyuncs.com/zing/sglang@sha256:4b9bca7dd61784cd84e43ac760db4c458c08a2d6bf22887e3740455fffe9e015`。
- 实测两卡均报告 NVIDIA L20D、CC 10.3、274113 MiB；按实测容量记录，不按名称推断。
- torch 2.13.0+cu130，CUDA13.0，NCCL2.29.7；具体 UUID、驱动和拓扑见 hardware.json/gpus.csv/topology.txt。
- `preflight.json` 保存六图、原始120帧源、派生240/1440帧RGB、VAE配置及权重的SHA。
- `service-assets-r1.json` 保存实际服务包67个文件、163163078786 bytes逐文件SHA。校验在服务启动前完成。
- 案例 `zing-whitebox-video-five-triview-0001`，六张 Picture1–6 reference，832×480、24FPS、4步、seed42、video-only。真实5秒源确定性重播2/12次；同session，不在接缝reset。60秒仅技术压力测试，不能代表自然60秒运动质量。

## 源视频到冻结RGB的独立复现

原素材为1280×720、24FPS、120帧。继承的#177准备脚本明确使用 `fps=24,scale=832:480:flags=bilinear`，导出120帧RGB24；再逐字节重复2/12次。CPU作业 `dlc1meo7e4cgqzfu` 按原命令重解，所得SHA为 `d917780f9c0bb002fbd646b77f23d442d73b5a42d840c4c911cef74f5af7adbb`，与本轮冻结120帧RGB完全一致。原MP4 SHA为 `6579ebd3f1d80455b71e30f9795aa494bceb615ab1302a812430f1199ec80f85`，回执 `source-audit-r2.json`。实际解码器为FFmpeg 6.1.1-3ubuntu5（libavcodec60.31.102、libavfilter9.12.100、libswscale7.5.100），完整构建参数见 `source-transform-provenance.json`。

首次额外审计用了默认bicubic缩放，SHA不同；失败回执source-audit-r1.json保留。定位原准备脚本后补齐bilinear参数才复现一致，未修改在途输入或放宽任何数值门槛。

## 实验控制

每个SP（1和2）分别运行四个新进程 off/reflect2d/reflect2d/off；每个进程预热后，240/1440帧各三次正式无profiler计时。完整encoder包含gather/拼接；服务通过localhost当前source.append协议真实生成，六图均进入请求。服务耗时含客户端写出，不是WAN/浏览器显示FPS。

`gpu-r1.sh` 保存完整环境变量与硬件门槛。`matrix-r2.py` 保存精确命令、ABBA顺序、预热和独立profile。`transport.py` 仅调整WebSocket ping超时，不改变生成语义。服务各配置的 `command.json` 保存启动参数；输入与profile见冻结回执。可用相同只读模型/素材在新输出目录重放控制脚本；不可覆盖本轮证据目录。

单个encoder配置的命令形式如下（具体路径和SHA取本轮preflight.json）：

```bash
MINIMAX_H3_PROGRESSIVE_SPATIAL_PADDING=reflect2d \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  benchmark/zing_h3_causal/validate_spatial_padding.py \
  --model "$MODEL" --rgb "$RGB1440" --rgb-sha256 "$RGB_SHA256" \
  --source-sha 359c187223aeb27e2ec3a232ac532e880c37c478 \
  --mode reflect2d --frames 240 1440 --correctness --profile --output "$NEW_OUTPUT"
```

模型/资产准备、汇总、媒体转码与解码都在零GPU CPU作业进行。GPU只挂CPFS。源代码、权重和原始素材只读使用，未修改原生Zing工作区。未运行SP4，不声明SP4验收。

## 检查口径

运行前冻结门槛：padding BF16/FP16/FP32 bitwise；完整moments/posterior bitwise且finite；full encoder中位耗时改善至少5%，首latent、E2E耗时和输出间隔p95退化不超过5%。保留全部重复值与波动范围，不删除慢样本、不按结果调整容差。

CPU回归 off/reflect2d 各1057通过。当前main测试收集存在循环import，测试启动先import configs.pipeline_configs后调用pytest.main；生产源码没有为此改动。padding参数化测试在实际NVIDIA机器共726项通过，其中366项CPU、360项CUDA；不能将726项全部称为CUDA测试。

Perfetto原始trace保持不变，解压JSON可本地打开。首轮SP1 off/reflect2d分别有133/108项slice_spill_overlapping_complete_event：UI明确为重叠完整事件移到overflowtrack，未丢弃数据；GPU事件嵌套显示有歧义。该trace仅解释算子归因，性能取独立无profiler重复。

所有产物位于私有CPFS/OSS，不生成公共媒体或trace链接。GPU状态Succeeded不能替代数值、媒体、性能验收。

1/2 latent分组按每次实际产出的latent数量分桶，包含EOS尾部；因此1 latent分组的p95可能包含较大的尾部输入，不应把整个分组解释成纯首帧延迟。首latent指标单独报告。

服务输出间隔p95由客户端相邻RGB消息的接收时间计算，是交付批次间隔，不是把一个批次内的帧摊平后的单帧间隔。source-to-output延迟按客户端记录的输入可用时刻计算；本轮为unpaced技术吞吐测试。

服务profile按scheduler dispatch分文件，每个17帧session有两个批次；SP1各一rank、SP2各两rank，合计12份服务原始trace，加6份独立encoder trace共18份。服务原始trace含python_function事件，Perfetto出现slice_max_depth_exceeded；另外提供删去该类事件的派生查看副本。derive-trace-views.py使用流式解析，逐事件规范化摘要与分类计数验证所有其他事件保留；原始文件始终作为权威证据。
