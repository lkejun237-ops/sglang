# 四案例回放测试

入口：`run_four_case_sglang_suite.py`。这是 Room 供帧、控制、缓存和清理的集成测试，
不启动 GPU Worker，不改变路由，不部署 Room。H100 Encoder、H100 Decoder、H200 SP1 DiT
的混合硬件业务效果与性能尚未实测；CPU/mock 通过不代表 GPU 或浏览器验收通过。

## 输入与证据

先运行 `prepare-media --media-dir <目录>`，再运行真实模式。清单保留归档对象路径、大小与
SHA-256；主桶为 `leap-world-shanghai`，候选桶为 `leap-worldmodel-shanghai`。候选路径
下载后也必须满足同一大小和 SHA，不能用合成素材替代。2026-11-10 只读查询上海行 MP4：
主桶对象存在（15,937,251 字节）；候选桶在上海 endpoint 返回 `NoSuchBucket`，不意味着已完成桶镜像。

实时供帧默认 24 FPS、1 倍速；`--capture-fps 48` 明确改变采集时钟，不是同条件的 24 FPS
性能对照。短回归会把切词边界压缩为 17、34……；`--preserve-switch-boundaries` 保留清单边界。
回放器在目标模型输入帧数处关闭 admission，等待真实 `clip-barrier-v1` 接管后恢复，
并要求 Worker 回执的边界与请求一致。该闸门用于确定性回归；闸门等待时间不能作为生产吞吐量收益。

Session TTFF 从 `render.start` 到 Room 的首个可解码模型帧，**不含此前独立 prefill**。
未采集浏览器 presented FPS。缓存 Cold/Warm 与 A/B 必须有实际 cache key 和命中类型；
预热臂还需匹配 prefill key，输入 F0 SHA 必须一致，缺失或不符合预期则失败。
两臂使用不同 seed 隔离已驻留缓存，这不是固定 seed 的生成效果对照。

## 已挂载的上海测试池

只使用 `test-shanghai-copy`。从测试环境部署回执取得 routing revision、compatibility、profile
和 network domain，Room 凭据通过环境变量提供，不写入命令或结果文件。

```bash
export SGLANG_LIGHTWEIGHT_RUNTIME=1
export PYTHONPATH=python
/opt/homebrew/bin/python3.11 benchmark/whitebox/run_four_case_sglang_suite.py run \
  --mode real-sglang --media-dir "$MEDIA_DIR" --output-dir "$RESULT_DIR" \
  --coordinator-url http://10.0.1.37:18181 \
  --routing-revision "$TEST_ROUTING_REVISION" --compatibility "$TEST_COMPATIBILITY" \
  --profile "$TEST_PROFILE" --network-domain "$TEST_NETWORK_DOMAIN" \
  --room-token-env WHITEBOX_ROOM_TOKEN --realtime-pacing
```

此入口通过认证的 `/v1/role-pool/*` 申请、续租和验证释放完整三角色组。
Room `10.0.1.34:8092` 不参与该进程内 Room 测试，因此不能据此宣称已验收该 Room 的部署或浏览器链路。

## 三个隔离 Worker 的本地建组桥

先将三个专用测试 Worker 配置为 managed role，并使用相同的 control epoch 与签名密钥。
Worker 的 `SGLANG_ROLE_POOL_WORKER_CONFIG` 中 compatibility、network domain 和 profile
必须匹配测试参数；密钥通过 `--signing-secret-env` 指定的环境变量读取。
Worker 必须包含本 PR 扩展后的认证 `/managed/identity`，返回实际容量和 profile 限制。
它们在自测期间不得注册到其他 Coordinator；不要对已挂载池使用此模式，应使用上面的 Coordinator 入口。

```bash
/opt/homebrew/bin/python3.11 benchmark/whitebox/run_four_case_sglang_suite.py run \
  --mode real-sglang --media-dir "$MEDIA_DIR" --output-dir "$RESULT_DIR" \
  --encoder-url "$ENCODER_TEST_HTTP" --dit-url "$DIT_TEST_HTTP" --decoder-url "$DECODER_TEST_HTTP" \
  --routing-revision "$TEST_ROUTING_REVISION" --compatibility "$TEST_COMPATIBILITY" \
  --profile "$TEST_PROFILE" --network-domain "$TEST_NETWORK_DOMAIN" \
  --control-epoch "$TEST_CONTROL_EPOCH" --signing-secret-env ROLE_POOL_SIGNING_SECRET
```

桥在当前进程内使用 `MemoryRecords`、`RolePoolStore`、`GroupController` 和原生认证 API，
不启动 Redis 或监听端口。通过签名 identity/observe 获取实际服务身份与状态；通过现有
reserve/build/business/close 协议工作。占用中的 service、错误 epoch、错误签名或配置不匹配
均失败，不接管已有组。返回池模式与显式 Worker 地址写入 `summary.json`，不写入密钥。
异常时仍由原生 ownership fence 和租约 watchdog 保护资源；清理不确定时必须检查 Worker，
不能立即把同一批角色挂载到另一个 Coordinator。

严禁指向 `prod-shanghai`、`prod-thailand`，不允许停止或重启他人 Worker。

## CPU 验证

使用仓库 Whitebox 依赖版本，特别是 `av==17.1.0`。指定 Python 的原装 PyAV 13.1
缺少播放代码所需的 `Flags.global_header`，需要在独立依赖目录或虚拟环境安装正确依赖。

```bash
SGLANG_LIGHTWEIGHT_RUNTIME=1 PYTHONPATH=python /opt/homebrew/bin/python3.11 -m pytest \
  --confcutdir=python/sglang/multimodal_gen/test/unit/realtime \
  python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_render_lifecycle.py \
  python/sglang/multimodal_gen/test/unit/realtime/test_control_prefill.py \
  python/sglang/multimodal_gen/test/unit/realtime/test_role_pool.py
```

`--confcutdir` 隔离父目录依赖模型运行时的 fixture；本组测试不运行模型。角色池测试使用实际
CPU ASGI Worker 与签名/建组/释放代码，并覆盖其他 authority 无法抢占已占 service。

2026-11-10 本地验证：指定 Python 3.11 + 临时依赖目录中的 PyAV 17.1，以上三文件
`82 passed, 2 skipped`；两项 Redis 持久化测试因本机没有 `redis-server` 跳过。
角色池内存建组、认证路由、错误租约拒绝、缺失素材拒绝、OSS 候选桶哈希校验及慢参考图
写入下的精确切词均执行通过。尚未运行上海混合 GPU、目标 Coordinator/Room 或真实素材 E2E。

持久化时间优先使用 Worker 回执；回执无计时时用实际 client receipt 时刻，
以 `persistence_timing_source` 区分。未收到持久化回执时标为 `not_observed`，不从 L1 Hit 推断已落盘。
