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
由真实 `kv_ready` 回执推进闸门，并要求实际切词边界与请求一致；用于精确边界回归。
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
