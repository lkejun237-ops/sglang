# Room–SGLang 白膜供帧回放集成测试

入口：`run_room_sglang_source_replay.py`。四案例是当前数据集
`cases/interactive_production_four_cases.json`，不是测试工具的范围名称。

## 覆盖范围

录制 MP4（或 CPU 测试用合成帧）通过 `RecordedVideoEngine` 替代实时 WorldKit 引擎；
进程内实际运行 Room `SourceSession`、`ControlPrefill`、`RenderSegment`，接入真实三角色
Encoder/DiT/Decoder。`mock-worker` 模式仅用于 CPU 协议回归，不能证明真实模型集成通过。
不经过部署后的 Room HTTP 入口、浏览器、WebRTC 或实时 WorldKit，不能将结果写成这些链路的验收。

Room 的 Latest-Wins、credit、参考图归档、切词屏障及清理代码保持复用。
`--flow-mode natural`（默认）直接使用 `SourceSession`，不安装测试专用 admission 闸门，
用于观察真实 Room 策略下的选帧、丢帧、切词延迟。
`--flow-mode deterministic` 使用测试子类，在目标模型输入计数处关闭 admission，
由真实 `kv_ready` 或协商后的 `source_ready` 回执推进闸门，并要求实际切词边界与请求一致；用于精确边界回归。后者仅允许恢复供帧，不代表 KV 已完成；报告保留 `readiness_state` 和真实 `kv_ready` 布尔值。
该测试闸门不是生产 Room 的供帧策略，其等待时间不能用来证明生产吞吐量收益。
两种模式分别记录，不能混合统计。

## 首次准备

需要匹配的 Whitebox 依赖，特别是 `av==17.1.0`。脚本自行设置轻量导入模式；它不会
将远端模型变为 mock。仓库根目录执行：

```bash
export PYTHONPATH=python
/opt/homebrew/bin/python3.11 benchmark/whitebox/run_room_sglang_source_replay.py prepare-media \
  --media-dir /tmp/whitebox-four-cases-media --oss-profile shengdong
```

清单保留归档路径、大小与 SHA-256；主桶为 `leap-world-shanghai`，候选桶为
`leap-worldmodel-shanghai`。候选下载同样必须匹配大小和 SHA。2026-11-10 对上海行 MP4 的
只读查询确认主桶对象存在（15,937,251 字节），候选桶在上海 endpoint 返回 `NoSuchBucket`。
真实模式不允许缺失素材或合成替代。mock 运行默认也只复用已有素材；需要生成缺失文件时，
显式传 `--synthesize-fixtures`（仅 mock-worker 可用）。无论 prepare-media 还是 run 的显式合成，
均保留已有 MP4/参考图字节，只补缺失项；准备回执以 reused_files / synthesized_files 分开记录。

## 三地址直连

三个专用测试 Worker 需要启用 managed role，包含 `/managed/discovery` 和认证
`/managed/identity`。测试机预先配置一次 `ROLE_POOL_SIGNING_SECRET`，密钥不从 Worker 下载。
三个 Worker 自测期间不能注册到另一个 Coordinator，也不能使用正式环境或他人活跃服务。

```bash
/opt/homebrew/bin/python3.11 benchmark/whitebox/run_room_sglang_source_replay.py run \
  --encoder-url "$ENCODER_TEST_HTTP" --dit-url "$DIT_TEST_HTTP" --decoder-url "$DECODER_TEST_HTTP"
```

脚本自动进入真实模式，从 Worker 核验 compatibility、network domain、control epoch；
选择唯一共同支持 832×480、24 FPS、至少 12 图且限制一致的 profile。歧义时可显式 `--profile`。
直连没有路由表，未指定 routing revision 时使用 compatibility 作为模型标识。
地址对应一个具体 service 的 HTTP listener；同一 Encoder 进程有多个 service 时传其中一个，
当前不会自动选择该进程的其他空闲 service。

桥在当前进程内使用 `MemoryRecords`、`RolePoolStore`、`GroupController` 和原生认证 API，
不启动外部 Coordinator、Room 或 Redis。占用、epoch、鉴权或配置不匹配时失败，不抢占旧组。
默认素材目录 `/tmp/whitebox-four-cases-media`，输出 `/tmp/whitebox-four-cases-output`；
建议每次通过 `--output-dir` 指定独立结果目录。

## 已挂载测试池

已挂池时改用 Coordinator，不能同时传三角色地址：

```bash
/opt/homebrew/bin/python3.11 benchmark/whitebox/run_room_sglang_source_replay.py run \
  --mode real-sglang --coordinator-url http://10.0.1.37:18181 \
  --routing-revision "$TEST_ROUTING_REVISION" --compatibility "$TEST_COMPATIBILITY" \
  --profile "$TEST_PROFILE" --network-domain "$TEST_NETWORK_DOMAIN" \
  --room-token-env WHITEBOX_ROOM_TOKEN
```

参数从 `test-shanghai-copy` 部署回执取得，凭据通过环境变量注入。
该入口验证池的申请和释放，不访问已部署 Room `10.0.1.34:8092`。
严禁改指向 `prod-shanghai`、`prod-thailand`，不停止或重启他人 Worker。

## 长度与计时

用户指定录制集合可在 cases JSON 设置 `execution: "recorded-cases"`，保留实际
case_id、prompt、参考图、录制 ID 和媒体哈希。该模式每个 case 连续运行两次且 seed
相同，包含该 case 的全部切词；命中状态以真实回执判断，不因第二次执行就断言命中。
它不添加默认四案例的 12 图压力条件，也不修改没有 `<Video 1>` 标签的原始 prompt。
该集合可显式加 `--legacy-prompt-control`，让同一 Room 使用既有的队列切词协议，
对照同一模型服务、输入、seed 和缓存状态下独立控制的影响。默认接受独立控制能力；
每轮记录请求的模式和实际协商结果。此消融不等于重新部署历史源码 baseline，
自然供帧的选帧仍可能因耗时不同而变化，不能冒充逐帧相同的数值对照。
同一次 suite 只能使用一种已冻结源几何，24 FPS。736 输出使用 832×480 源视频，
在 cases JSON 顶层加 `"model_target": {"width": 1280, "height": 736}`；不能把源视频
放大到 736 绕过准入。省略时保持源与输出尺寸相同。Worker 必须实际声明支持所选
source/target 组合；报告分别记录两者，不能将 480 服务重标为 736。每个完成的 case
立即保存到 summary.json，后续失败保留
已完成结果并标记整体 failed。模型步数通过 `WHITEBOX_INFERENCE_STEPS` 显式匹配
部署（两步 1000,500 对应 2），不能沿用工具的 4 步默认值。

该模式仍有显式 prefill；Session TTFF 不包含该准备阶段。短测仍按下述规则压缩切词
位置，不能称为原始在线 RGB 逐帧复现；保留历史位置需 `--preserve-switch-boundaries`。

短测默认压缩切词目标到 17、34……，最后一次切词后目标保留 18 个输出帧；
natural 模式可能晚于请求边界执行，此时按实际边界追加尾帧。

| 当前案例 | 短测最低目标（24 FPS 输出时间轴） | full 输出帧数目标 |
|---|---|---|
| 上海行 | 18 帧 / 0.75 秒 | 1049 帧 |
| MARVEL | 52 帧 / 约 2.17 秒 | 547 帧 |
| Dark Horns | Miss、Hit 各 35 帧 / 各约 1.46 秒 | 两臂各 369 帧 |
| 功夫嘉豪 | 两段各 18 帧 / 各 0.75 秒 | 第一段 1585、第二段 127 帧 |

`--preserve-switch-boundaries` 保留清单切词位置；`--full-frames` 同时保留这些位置，并对每段
使用历史输出帧数目标，不再额外追加尾帧。模型批量输出和 drain 可能超过目标。
这些均不是墙钟运行时长：prefill、切词、credit 等待会延长运行；Latest-Wins 会丢弃捕获帧。
源视频可循环以收集足够输出，因此 full 不等于 MP4 从头到尾只播放一次，也不是逐帧同源复现。
功夫嘉豪第二段是在同一持续 source 上换条件并重新锁 F0，其 secondary recording 提供条件和
输出长度；并未切换到第二个 MP4。第二段输出使用 `input_kind=same-source-condition-change`、
`source_recording_id`（真实视频源）、`condition_recording_id`（条件来源），不再用一个含混的
recording_id 指代混合输入；prepare 中另有实际 F0 帧号与 RGB SHA。报告明确显示两种来源。
输出分别记录目标、源总捕获数与实际输出。
`--capture-fps 48` 改变捕获时钟，不属于同条件 24 FPS 对照；默认 pacing 为 1 倍速。

## 验收与结果

退出成功后检查 `summary.json` 和 `REPORT.md`：mode、flow_mode、length_mode、源码/输入身份、
目标与实际帧数、错误、租约释放。Session TTFF 从 render.start 到 Room 首个可解码模型帧，
不包含此前的独立 prefill，也不包含浏览器展示延迟。

`--detailed-timing` 在每个 `render.start` 中发送 `detailed_timing=true`，覆盖首段和
同源重开段，后续 prompt 切换继承该段设置；默认不发送该字段，便于与未支持此字段的
旧版本做对照。报告的 `detailed_timing_requested` 只记录请求开关，实际采集需检查
模型回执和归档阶段数据。该开关不设置 `return_stage_timing`。

计时开销对照使用同一冻结输入和配置，分别测原版、PR 详细计时关闭、PR 详细计时开启；
同名 case 的各轮 seed 对齐、cache 冷热分别核验，预热与正式样本分开统计。该回放
使用进程内 Room，不测已部署 Room 的 HTTP scrape、世界资源加载或浏览器显示开销。
缓存 Cold/Warm 和 A/B 必须有真实命中类型和 cache key；预热臂匹配预热 key，F0 SHA 一致。
A/B 使用不同 seed 隔离热缓存，不能作为固定 seed 的生成效果对照。
持久化时间区分 Worker 计时与 client receipt 计时；未观测到持久化回执不能从 L1 Hit 推断已落盘。

CPU 验证入口：

```bash
SGLANG_LIGHTWEIGHT_RUNTIME=1 PYTHONPATH=python /opt/homebrew/bin/python3.11 -m pytest \
  --confcutdir=python/sglang/multimodal_gen/test/unit/realtime \
  python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_render_lifecycle.py \
  python/sglang/multimodal_gen/test/unit/realtime/test_control_prefill.py \
  python/sglang/multimodal_gen/test/unit/realtime/test_role_pool.py
```

父级模型 fixture 通过 `--confcutdir` 隔离；CPU/mock 结果不替代上海混合 GPU、浏览器或部署验收。

2026-11-10 本次本地验证：指定 Python 3.11、PyAV 17.1，三文件主回归
`91 passed, 2 skipped`（缺少本地 redis-server）；加强后的长度测试另跑 `2 passed`。
验证包含自然模式不安装测试闸门、确定性慢参考图切词、full 不额外补尾以及第二段独立长度。
项目 skill 通过格式校验。未运行真实 GPU、目标 Coordinator/Room 或浏览器验收。

素材与来源复核修复：CPU 回归 `92 passed, 2 skipped`（本地缺少 redis-server）。
覆盖默认 mock 复用自定义有效参考图后的逐文件字节不变、显式合成只补缺失项，以及两份不同视频
的 F0 哈希归属与 JSON/报告中视频、条件来源的分别标注。没有增加 GPU 或真实素材 E2E 验收结论。

### 真实引擎供帧与重复次数

`execution: recorded-cases` 复用冻结的 prompt/参考图序列，可指定 `repeats: 3`。
增加 `source_kind: worldkit-live` 后，白膜由生产 `cli_room` 启动真实 Worldkit 引擎生成；
录制视频不参与供帧。`live_source_engine.py` 只持有 CLI/view 生命周期并统计读入帧数，
供帧、暂停恢复、模型 admission 和切词仍走原有 SourceSession。只支持 real-sglang。

使用 `run_live_source_controller.py --root <目录>` 在已有 Room 镜像中创建独立测试容器；
目录包含已推送 SHA 导出的 `source/`、已自检的独立三角色 `state.json`、`cases.json` 和
经过校验的 `media/`。复用 controller 本地凭据，不下载 Worker 密钥，不更改共享服务/路由。

真实引擎报告给出 CLI 启动耗时 `engine_startup_ms` 和从引擎初始化到模型首 RGB 的
`engine_init_to_first_rgb_ms`。手动发送 render.start 的路径另有
`whitebox_session_ttff_ms`，从 render.start 计到模型首 RGB。测量终点位于进程内
Room 收到模型输出的位置。实时游戏帧可能与历史录像逐像素不同。

验证状态：本次扩展的 CPU 回归和真实引擎 GPU 实测以运行回执为准，尚不能由源码推断通过。

测试网与平台网互不连通时，在已有 Room 主机运行 `--prepare-only`，通过官方
CDN 资源模式启动三份真实游戏，预热 SDK 游戏缓存并保留实际 artifact 身份。
复制缓存至模型网络后，仍由正式 CLI 启动真实游戏生成白膜，使用 SDK async cache
update 的本地快照启动路径；每轮必须核对 artifact 身份，变化则失败。
`--render-pod --image <固定 Room digest>` 生成不占 GPU 的 CPU controller Pod，
仅使用节点已有的模型签名凭据。缓存准备不计入每轮首帧；真实游戏进程启动单列。

游戏声明 `modelContext` 时，由游戏提供首个条件、参考图和后续切词。测试在 Room 收到首个模型 RGB 后按原始操作间隔发送 `player_inputs`，记录前三次实际生效的条件和模型返回帧。真实操作可触发与历史不同的游戏结果；按模型输出记录中的 prompt 哈希和有序参考图哈希核对同条件样本。切词边界以游戏实际触发位置为准。首帧从引擎初始化请求计至模型首 RGB，Room 接纳条件至首 RGB 另见分时记录。

录制回放和未声明条件通道的游戏保留手动准备参考图、指定切词边界的流程。
