# Zing H3 周期性动态 Sink 刷新实现规范（v2）

本文档定义在分支 `feat/zing-periodic-sink`（工作树 `/home/ss80194/work/sglang-zing/feat-periodic-sink`，基线 `origin/main@2dfb1aa538`）上落地**周期性动态 Sink 刷新（含 `copy` 与 `rebuild` 双方法）**的完整实现契约、状态机不变量、文件级改动清单与本地验收闭环。

文档面向执行 Agent（如 `dsh`）与人工 Code Review 双重读者：所有接口签名、字段名、边界时序和红线约束均为精确规定，不得自行发散。

---

## 0. 执行红线与环境约束（必读硬边界）

### 0.1 工作树与分支隔离
1. **唯一允许修改的工作树**：`/home/ss80194/work/sglang-zing/feat-periodic-sink`（分支 `feat/zing-periodic-sink`）。
2. **严禁触碰其他工作树**：
   - `/home/ss80194/work/sglang-zing/zing-periodic-sink-v2`（人工编码现场，严禁读写或清理）
   - `/home/ss80194/work/sglang-zing/zing-graph-workspace-diagnostic`（独立 Graph 诊断现场，严禁触碰）

### 0.2 严禁修改的底层文件清单
本次改动限定在协议适配层、编排队列与 `zing_h3` 采样会话层。**严禁修改以下任何文件**：
- `python/sglang/multimodal_gen/runtime/models/dits/zing_h3_denoise_graph.py`
- `python/sglang/multimodal_gen/runtime/models/dits/zing_h3_prefix_graph.py`
- `python/sglang/multimodal_gen/runtime/models/dits/zing_h3.py`
- `python/sglang/multimodal_gen/runtime/models/dits/zing_h3_cache.py`
- `python/sglang/multimodal_gen/runtime/realtime/zing_kv_pool.py`
- `python/sglang/multimodal_gen/runtime/realtime/zing_window.py`
- 任何 FA4 / Triton / CUDA Graph 底层实现文件

**原因说明（防误判）**：
- `ZingH3KVCache` 已具备完整的 `update_sink("off" | "refresh", current_id=...)`、`validate_prefix_reset()` 与 `reset_history_keep_prefix()`，无需改 `zing_h3_cache.py`。
- `ZingFixedForwardKVPlan` 已按固定 `capacity` 对 K/V 做容量级 padding 并通过 `cu_k` 控制有效长度，窗口增长（`2 -> 3 -> 4 -> 5 -> 6`）**不会**改变 Graph key。
- Sink 刷新后的首个块无 `cache.active`，天然走 `graphs.recovery`（默认 `SGLANG_ZING_CHUNK_GRAPH_BOUNDARY=0` 下已是 eager），稳态流式图（`resume` 2 张 + `deferred` 2 张 + 积压态 `steady` 2 张 = 6 张）在默认 `SGLANG_ZING_DENOISE_CUDA_GRAPH_MAX=32` 下完全常驻复用；测试或文档中**严禁**传入 `--max-graphs 4` / `SGLANG_ZING_DENOISE_CUDA_GRAPH_MAX=4`。

### 0.3 WSL 本地环境与 CI 清单红线
1. **严禁安装任何依赖**：严禁执行 `pip install`、`conda install` 或修改 Python 环境（本机无 `torchvision`/`msgspec`/`orjson`/`flashinfer`，且强装会破坏环境）。
2. **严禁新增未注册的 `test_zing*.py` 文件**：`scripts/ci/zing/run_cpu_tests.py` 的 `inventory_check()` 会扫描所有 `test_zing*.py` 并校验其是否在 `scripts/ci/zing/cpu_test_manifest.json` 注册，擅自新建测试文件会导致 GitHub CPU CI 直接报错 `unknown: ...`。**所有新增单测必须直接追加到以下已注册白名单测试文件中**：
   - `python/sglang/multimodal_gen/test/unit/test_zing_h3_window.py`
   - `python/sglang/multimodal_gen/test/unit/test_zing_h3_queued_dit.py`
   - `python/sglang/multimodal_gen/test/unit/realtime/test_zing_h3_source_adapter.py`
3. **两步独立提交**：
   - **Commit 1**：协议契约 + 环境变量兜底 + 双路径透传 + `SourceReadySession` 周期调度状态机 + `copy`（Impl A）+ 对应单测。
   - **Commit 2**：最近 5 帧 `source + target` 配对环 + `pack_sink_rebuild_prefix` + `rebuild_sink_prefix` + `rebuild`（Impl B）激活与 SP 投票 + 对应单测。

---

## 1. 协议契约、环境变量与双通道透传

### 1.1 能力协商（`session.ready`）
在 `ZingH3SourceStreamAdapter`（`zing_h3_source_adapter.py`）构造 `session.ready` 时：
- 保留现有 `"sink_updates": self.sink_updates`
- 新增 `"auto_sink_updates": self.sink_updates`
- 动态 Sink 支持条件保持不变：未拆首帧的 `[3, 2]` chunk plan（`lengths == (3, 2) and not split_first_latent`）、纯视频（`not audio_enabled`）、`window_chunks is not None and window_chunks >= 4`、初始 `sink_chunks in (0, 2)`，且非 `component_capture` 会话。

### 1.2 客户端 WebSocket 控制事件
在 `zing_h3_source_adapter.py` 的 `ingest_wire_event` 中支持以下三类 Sink 事件（**不引入**冗余的 `sink.on`）：

1. **`sink.off`**
   - Payload：必须为 `{}`。
   - 归一化命令：`dict(operation="off", method="copy", event_id=event_id)`。
2. **`sink.refresh`**
   - Payload：允许 `{}`（向后兼容，默认 `"copy"`）或 `{"method": "copy" | "rebuild"}`。包含未知字段或非法 `method` 时抛 `ValueError`。
   - 归一化命令：`dict(operation="refresh", method=method, event_id=event_id)`。
3. **`sink.auto`**
   - 开启/更新：`{"enabled": True, "interval_cycles": <int>, "method": <"copy" | "rebuild">}`（其中 `"method"` 可选，缺省为 `"copy"`；`enabled` 必须 `type(enabled) is bool`；`interval_cycles` 必须 `type(interval_cycles) is int and interval_cycles >= 1`，严防 `True` 被当成 `1`；不允许出现未知额外字段）。
   - 关闭：严格要求 `payload == {"enabled": False}`（归一化为 `dict(enabled=False, interval_cycles=None, method="copy", event_id=event_id)`）。
   - 确认帧：与 `sink.off` / `sink.refresh` 一致，发送 `dict(type="control_ack", kind="sink.auto", event_id=event_id, stage="accepted")`。

**关键校验顺序不变量**：
- 所有 `self.sink_updates` 检查、payload 结构/类型校验、`self.ledger.validate_event_id(event_id)` 及 `not self.ledger.ended` 检查，**必须在赋值 `self.ledger.event_id = event_id` 之前完成**。任何校验失败不得消耗或污染 `self.ledger.event_id`，也不得修改挂起槽位。

### 1.3 双独立挂起槽位（互不覆盖）
手动单次操作与自动周期配置职责正交，Adapter 层必须维护**两个独立槽位**（各自内部 latest-wins，同一 dispatch 批次内互不覆盖）：
- `self._pending_sink_command` -> `batch.extra["zing_sink_update"]`
- `self._pending_sink_auto_command` -> `batch.extra["zing_sink_auto"]`

**所有挂起检查点必须同时覆盖两个槽位**：
1. `zing_h3_source_adapter.py`：
   - `__init__` 与 `clear_state`：同时初始化/清理 `_pending_sink_command = None` 与 `_pending_sink_auto_command = None`。
   - `ingest_wire_event`：收到合法事件后置位对应槽位并 `self.available.set()`。
   - `has_control_work`：`(self._pending_sink_command is not None or self._pending_sink_auto_command is not None) and self.dispatched_source_frames > 0 and not self.end_dispatched`。（注：在首帧 `source` 发送前 `dispatched_source_frames == 0`，提前到达的 `sink.auto` 暂存在槽位中，随首个 `source` 批次的 `prepare_next_request` 自动捎带下发，不会触发空批次抢跑。）
   - `_wait_joint_source`：任意一个槽位非 `None` 时立即跳出等待。
   - `prepare_next_request`：分别检查并搬运到 `batch.extra["zing_sink_auto"]` 与 `batch.extra["zing_sink_update"]`，随后清空对应属性。
2. `zing_source_encoder_adapter.py`（第 96 行附近）与 `zing_encoder_dit_adapter.py`（`has_control_work` 第 142 行附近）：凡检查 `_pending_sink_command is not None` 之处，同步纳入 `getattr(self, "_pending_sink_auto_command", None) is not None`。

### 1.4 一体化与分离式（Encoder -> QueuedDiT）双路径消费
同批次同时存在 `zing_sink_auto` 与 `zing_sink_update` 时，**统一先应用 `sink.auto`，再应用 `sink.update`**（确保同批先开启自动刷新再手动立即刷新时，手动刷新可基于新配置对齐锚点）：

1. **一体化路径（`source_session.py` 第 1156 行附近）**：
   - 先处理 `sink_auto = batch.extra.get("zing_sink_auto")`：若非空，当 `self.component_capture is not None` 时抛 `ValueError("Component capture does not support dynamic sink updates")`，否则调用：
     ```python
     self.sampler.configure_auto_sink(
         sink_auto["enabled"],
         interval_cycles=sink_auto["interval_cycles"],
         method=sink_auto["method"],
         event_id=sink_auto["event_id"],
     )
     ```
   - 再处理 `sink_command = batch.extra.get("zing_sink_update")`：同样拒绝 `component_capture`，调用：
     ```python
     self.sampler.update_sink(
         sink_command["operation"],
         sink_command["event_id"],
         method=sink_command.get("method", "copy"),
     )
     ```
2. **分离式路径（`zing_source_encoder_server.py` 第 835 行附近 + `queued_dit.py`）**：
   - `zing_source_encoder_server.py`：若 `batch.extra.get("zing_sink_auto")` 非空，校验 `self.dit_link is not None` 后 `send_dit("sink.auto", **sink_auto)`；再按原逻辑转发 `send_dit("sink.update", **sink_command)`。
   - `queued_dit.py`（`QueuedDiTSession`）：
     - 新增 `self.last_sink_auto_event_id = None`。
     - `_put` 中：
       - `kind == "sink.update"`：保持 `self.initial_received` 与 `operation in {"off", "refresh"}` 校验，新增 `method = message.get("method", "copy")` 校验（`method in {"copy", "rebuild"}` 且 `operation == "off"` 时必须 `method == "copy"`），`event_id` 相对 `self.last_sink_event_id` 严格递增。
       - 新增 `kind == "sink.auto"`：要求 `self.initial_received`，校验 `type(enabled) is bool`、`method = message.get("method", "copy") in {"copy", "rebuild"}`、当 `enabled` 为真时 `type(interval_cycles) is int and interval_cycles >= 1`（为假时 `interval_cycles is None`），且 `event_id` 相对 `self.last_sink_auto_event_id` 严格递增；占用并释放 `self.control_tasks`。
     - `_arrange`：`if kind in {"conditioning", "conditioning.cancel", "sink.update", "sink.auto"}: return [message]`。
     - `_process`：新增 `elif task["type"] == "sink.auto":`，在 `try ... finally: self.control_tasks -= 1` 中调用 `self.state.configure_auto_sink(task["enabled"], interval_cycles=task.get("interval_cycles"), method=task.get("method", "copy"), event_id=task["event_id"])`；`sink.update` 分支传入 `method=task.get("method", "copy")`。

### 1.5 服务端环境变量默认兜底（免改客户端联调支持）
在 `SourceReadySession.__init__`（`incremental.py`）末尾，若当前会话满足动态 Sink 支持条件（`self._supports_dynamic_sink()` 为 `True`），读取环境变量作为会话初始默认值：
- `SGLANG_ZING_AUTO_SINK_CYCLES`：未设置、空串或 `"0"` 表示默认关闭；若设置其他值，必须可解析为正整数 `cycles >= 1`（非法字符串或负数抛 `ValueError`）。
- `SGLANG_ZING_AUTO_SINK_METHOD`：缺省 `"copy"`，仅允许 `"copy"` 或 `"rebuild"`（其他值抛 `ValueError`）。
- 当 `cycles >= 1` 时，初始化 `self.auto_sink = AutoSinkConfig(enabled=True, interval_cycles=cycles, method=method, event_id=None)`，`self.sink_anchor_latent = 0`。后续若收到客户端 `sink.auto`，直接通过 `configure_auto_sink` 覆盖该默认配置。

### 1.6 输出帧回执元数据（Receipt Metadata）
1. `ChunkResult`（`incremental.py`）保持原有前 9 个字段位置与默认值完全不变，在末尾追加 4 个带默认值的可选字段：
   ```python
   sink_config_event_id: int | str | None = None
   sink_refresh_count: int = 0
   sink_refresh_method: str | None = None
   sink_refresh_latent: int | None = None
   ```
   含义：
   - `sink_config_event_id`：最近一次已生效 `configure_auto_sink` 的 `event_id`（环境变量默认开启且未收过 `sink.auto` 时为 `None`）。
   - `sink_refresh_count`：本会话内已实际执行的 Sink `refresh`（含自动到期刷新与手动 `refresh`，不含 `off`）累计次数。
   - `sink_refresh_method`：最近一次实际执行的 `refresh` 方法（`"copy"` 或 `"rebuild"`，未发生过 `refresh` 时为 `None`）。
   - `sink_refresh_latent`：最近一次实际执行 `refresh` 时的 `committed_latent` 边界位置（未发生过 `refresh` 时为 `None`）。
2. 将上述 4 个字段与原有的 `("sink_event_id", "sink_chunks", "sink_chunk_ids")` 同步透传至：
   - `source_session.py`（第 1748 行与第 1904 行两处块元数据字典）
   - `dit_decoder.py`（第 35 行 `output_metadata`）
   - `dit_link.py`（第 323 行 `dit.latents` 消息字典）
   - `realtime_output_adapter.py`（第 730 行 header 拷贝元组扩展为 `("sink_event_id", "sink_chunks", "sink_chunk_ids", "sink_config_event_id", "sink_refresh_count", "sink_refresh_method", "sink_refresh_latent")`）

---

## 2. `SourceReadySession` 周期状态机与竞争规则（`incremental.py`）

### 2.1 核心状态字段
在 `SourceReadySession` 中定义：
```python
@dataclass(frozen=True)
class AutoSinkConfig:
    enabled: bool = False
    interval_cycles: int | None = None
    method: str = "copy"
    event_id: int | str | None = None

@dataclass(frozen=True)
class PendingSinkIntent:
    operation: str          # "off" | "refresh"
    method: str             # "copy" | "rebuild"
    event_id: int | str | None
    after_chunk_id: int
    is_auto: bool = False
```
会话实例维护：
- `self.auto_sink = AutoSinkConfig()`
- `self.sink_config_event_id = None`
- `self.sink_anchor_latent = 0`
- `self.sink_refresh_count = 0`
- `self.sink_refresh_method = None`
- `self.sink_refresh_latent = None`
- `self.pending_sink: PendingSinkIntent | None = None`（仅存手动 `update_sink` 写入的挂起意图）
- `self.recent_paired: tuple[int, int, torch.Tensor, torch.Tensor] | None = None`（Commit 2 使用：`(latent_start, latent_end, source_5f, target_5f)`）

辅助判定 `self._supports_dynamic_sink()`：
```python
policy = self.sampler.window_policy
return (
    self.plan.lengths == (3, 2)
    and not self.plan.split_first_latent
    and policy.window_chunks is not None
    and policy.window_chunks >= 4
    and policy.sink_chunks in (0, 2)
    and not self.audio_enabled
)
```

### 2.2 计时、S0 暂停、半周期顺延与锚点重置不变量
1. **周期定义**：`[3, 2]` 下每 `5` 个已提交生成 latent 为 `1` 个 cycle，到期阈值为 `5 * self.auto_sink.interval_cycles`。
2. **何时判定自动刷新到期（`_auto_sink_due`）**：
   同时满足以下全部条件时，自动刷新处于“已到期（due）”状态：
   - `self.auto_sink.enabled` 为 `True`
   - `self.cache is not None` 且 `self.cache.window_policy.sink_chunks == 2`（**在 `S0` 即 `sink_chunks == 0` 期间，自动刷新暂停触发**；冷启动首块尚未创建 `self.cache` 时也不触发）
   - `self.committed_latent >= 5`
   - `self.committed_latent - self.sink_anchor_latent >= 5 * self.auto_sink.interval_cycles`
3. **边界对齐与半周期顺延规则（极其关键，防卡死与破测）**：
   - **手动 `sink.off` 与手动 `sink.refresh(method="copy")`**：保持与主干 100% 兼容——只要 `ready.chunk.chunk_id > intent.after_chunk_id`，在**下一个 chunk 边界立即生效**（无论 `committed_latent % 5` 是否为 `0`，确保既有 `test_external_sink_control_and_prompt_share_the_original_session` 在 latent `8` 处的断言原样通过）。
   - **手动 `sink.refresh(method="rebuild")` 与所有自动到期刷新（`auto_sink` 的 `"copy"` 和 `"rebuild"`）**：**统一只在 `committed_latent % 5 == 0` 的完整 `[3, 2]` 周期边界执行**！
     - 若排空队列后处于半周期（例如 `committed_latent == 8` 或 `103`，`% 5 == 3`），**严禁令 `_next_model_chunk()` 返回 `None` 阻塞供帧**！必须继续按当前局部 epoch 规划下一个 2-latent 块（`self.plan.chunk(self.model_chunk_id)`），跑完该 2-latent 块到达 `committed_latent == 10` 或 `105`（`% 5 == 0`）后立即执行刷新。
4. **`sink_anchor_latent` 重置为 `self.committed_latent` 的 5 个精确时机**：
   1. 调用 `configure_auto_sink(...)` 开启、关闭或修改周期配置时；
   2. 手动 `sink.refresh`（`copy` 或 `rebuild`）实际落地生效时（包括从 `S0` 通过手动 `refresh` 回到 `S2` 时）；
   3. 自动到期刷新（`copy` 或 `rebuild`）实际落地生效时；
   4. `prompt_switch`（`_rebuild_pending_prompt`）完成新局部 epoch 重建时（新 epoch 前 5 帧已是全新锚点，从该边界重新起算周期）；
   5. 会话初始化启用环境变量默认自动 Sink 时（锚点为 `0`）。
5. **多控制事件同边界优先级**：
   `pending_prompt`（Prompt 切换） **>** 手动 `pending_sink` **>** 自动到期刷新（`_auto_sink_due`）。
   - 若在同一个边界上 `pending_prompt` 生效：在 `_rebuild_pending_prompt` 中按原逻辑吸收已就绪的手动 `pending_sink`（决定新 epoch 的 `sink_chunks` 是 `2` 还是 `0`，若手动 `operation == "refresh"` 则同步更新 `sink_event_id`），并将 `self.sink_anchor_latent = self.committed_latent`，本轮已到期的自动刷新直接被这次重建吸收，**绝不连续执行第二次重建**。
   - 若在同一边界上手动 `pending_sink` 已就绪（`chunk_id > after_chunk_id`）：优先执行手动 `pending_sink`（若是 `off` 则转入 `S0` 自动暂停；若是 `refresh` 则更新 `sink_anchor_latent = self.committed_latent`），从而自然消解同边界的自动到期状态。
6. **会话收尾清理**：`dispose()` 中清空 `self.pending_sink = None`、`self.recent_paired = None`。EOS 前未到达可执行边界的挂起/到期刷新随会话结束自然丢弃，绝不无源空跑。

---

## 3. 两种刷新实现细节与分 Commit 改动清单

### Commit 1：协议层 + 周期调度状态机 + `copy`（Impl A）

#### 3.1 `copy` 执行点（`_apply_pending_sink`）
`copy` 不改变局部 `model_chunk_id` 或 `media_origin`，仍在 `_prepare(ready, ...)` **之后**、`with self.rng.random_stream(...)` 块首的 `_apply_pending_sink(ready, current)` 中执行：

1. 计算当前块在 `_apply_pending_sink` 处应执行的 `copy`/`off` 意图 `effective_copy_intent(ready)`：
   - 若 `self.pending_sink is not None and ready.chunk.chunk_id > self.pending_sink.after_chunk_id`：
     - 若 `self.pending_sink.method == "copy"`（含 `"off"` 与手动 `"refresh"+"copy"`），选中该手动意图；
     - 若 `self.pending_sink.method == "rebuild"`，此处不执行（留到 `committed_latent % 5 == 0` 的 `_prepare` 前执行，若当前是半周期 `3->5` 的 2 帧过渡块则直接放行）。
   - 否则（无就绪的手动意图，且手动意图未挂起等待），若 `self._auto_sink_due and self.auto_sink.method == "copy" and ready.chunk.start % 5 == 0`：
     - 选中自动意图 `PendingSinkIntent("refresh", "copy", None, ready.chunk.chunk_id - 1, is_auto=True)`。
2. 若未选中任何意图，直接 `return`。
3. 执行多卡一致性投票：
   ```python
   sampler._agree_sp_boundary(
       cache,
       ready.model_chunk.target_block,
       phase="sink_update",
       ready_route=intent,
   )
   ```
   同时在 `_agree_ready_route` 的 `ready_route` 元组中将原来的 `self.pending_sink` 扩展为 `(self.pending_sink, self.auto_sink, self._auto_sink_due)`，确保所有 SP rank 在 `step` / `drain_ready` 入口对自动到期状态投票一致。
4. 冲刷延迟尾步并更新窗口：
   ```python
   if cache.active is not None:
       final = sampler._deferred_final(cache, self.recent_visual, current.positive)
       sampler._forward(final, cache, "final")
   cache.update_sink(intent.operation, current_id=ready.model_chunk.chunk_id)
   ```
5. 更新状态与回执字段：
   - 若 `not intent.is_auto`：`self.sink_event_id = intent.event_id`，`self.pending_sink = None`。
   - 若 `intent.operation == "refresh"`：
     - `self.sink_anchor_latent = ready.chunk.start`（即当前 `self.committed_latent`）
     - `self.sink_refresh_count += 1`
     - `self.sink_refresh_method = "copy"`
     - `self.sink_refresh_latent = ready.chunk.start`

#### 3.2 Commit 1 涉及文件清单
1. `python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime/adapters/zing_h3_source_adapter.py`
2. `python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime/adapters/zing_source_encoder_adapter.py`
3. `python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime/adapters/zing_encoder_dit_adapter.py`
4. `python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime/realtime_output_adapter.py`
5. `python/sglang/multimodal_gen/runtime/entrypoints/zing_source_encoder_server.py`
6. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/zing_h3/queued_dit.py`
7. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/zing_h3/source_session.py`
8. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/zing_h3/dit_decoder.py`
9. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/zing_h3/dit_link.py`
10. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/zing_h3/incremental.py`（Commit 1 中若收到 `method="rebuild"`，可先在 Commit 1 校验通过协议或直接在 Commit 2 补齐执行；建议按两步连续完成后统一复核，Commit 1 时若调用尚无实现的 `rebuild` 路径可显式 `NotImplementedError`，Commit 2 立即替换为完整实现）。

---

### Commit 2：配对环 + `source + target` Clean KV 重建（Impl B）

#### 3.3 为什么 `rebuild` 必须同时重算 Clean `source` + Clean `target`（`[3, 2]` 共 4 个动态块）
冷启动与 `copy` 路径下，每个已提交 chunk 在 `cache.committed` 中均由 **`source`（奇数块 `1, 3`）+ `target`（偶数块 `2, 4`）配对合并**而成（`ZingH3KVCache.update(..., "final", ...)` 会将 `cache.pending_source` 与当前 `target` 合并进同一个 `CommittedChunk`）。
因此，Sink `rebuild` **绝不能复用仅含 target 的 `pack_prompt_switch_prefix`**，而必须用最近 5 帧配对的 `(source_5f, target_5f)` 按 `[3, 2]` 重新跑完 `block 1 (src) -> block 2 (tgt)` 与 `block 3 (src) -> block 4 (tgt)`，同时调 `cache.reset_history_keep_prefix()` 原地保留 `block 0`（condition + reference），实现与冷启动 Sink 100% 同构的 KV 布局与行数！

#### 3.4 `packing.py`：新增 `pack_sink_rebuild_prefix`
在 `ZingH3Packer`（`packing.py` 第 282 行 `pack_prompt_switch_prefix` 之后）新增：
```python
def pack_sink_rebuild_prefix(
    self,
    inputs: H3InferenceInputs,
    condition: dict[str, torch.Tensor],
    source_prefix: torch.Tensor,
    target_prefix: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], tuple[TemporalChunk, ...]]:
    """Pack five paired clean source+target latents into local epoch blocks (1,2) and (3,4)."""
```
校验与行为要求：
1. 调用 `chunks = self.prompt_switch_prefix_chunks()`（校验 `plan.lengths == (3, 2)`，得到 `chunk 0: [0..3, src=1, tgt=2]` 与 `chunk 1: [3..5, src=3, tgt=4]`）。
2. 对 `source_prefix` 与 `target_prefix` 分别调用 `self._validate_visual(...)`，并要求两侧的 `shape[2]` 都等于 `self.PROMPT_SWITCH_PREFIX_FRAMES`（即 `5`）。两侧保留各自的空间网格，允许 `H/W` 不同。
3. 浅拷贝 `prefix_inputs = copy(inputs)`，设置 `prefix_inputs.positive = condition`、`prefix_inputs.source_visual = source_prefix`，调用 `layout = self.pack_inference_layout(prefix_inputs, condition, target_prefix, None)`。
4. 严格校验 `layout["block_idx"]`：
   - `row_role == SOURCE_VISUAL` 的唯一有序 block 集合必须等于 `[chunk.source_block for chunk in chunks]`（即 `[1, 3]`）；
   - `row_role == TARGET_VISUAL` 的唯一有序 block 集合必须等于 `[chunk.target_block for chunk in chunks]`（即 `[2, 4]`）；
   - 不包含任何 `TARGET_AUDIO` 行。
5. 返回 `(layout, chunks)`。

#### 3.5 `sampling.py`：新增 `rebuild_sink_prefix`
在 `ZingH3Sampler`（`sampling.py` 第 269 行 `rebuild_prompt_prefix` 之后）新增：
```python
def rebuild_sink_prefix(
    self,
    inputs,
    condition,
    source_prefix,
    target_prefix,
    cache,
):
    """Keep resident block 0 prefix and rebuild paired [3, 2] clean source+target sink KV."""
```
精确执行步骤：
1. `source_prefix = source_prefix.float().clone()`，`target_prefix = target_prefix.float().clone()`（与 `initialize` / `_prepare` 一致保持 FP32 master latent）。
2. 调用 `layout, chunks = self.processor.pack_sink_rebuild_prefix(inputs, condition, source_prefix, target_prefix)`。
3. 调用 `cache.reset_history_keep_prefix()`（原地保留 `cache.prefix` 即 block 0，清空 `committed` / `active` / `pending_source`，将 `cache.last_block` 置回 `0`）。
4. 按序遍历 `for chunk in chunks:`（先 `chunk 0` 的 3 帧，再 `chunk 1` 的 2 帧）：
   - 写入并提交 `source_block`（设置 `cache.pending_source`）：
     ```python
     self._set_block_latents(
         layout,
         chunk.source_block,
         source_prefix[:, :, chunk.start : chunk.end],
         None,
     )
     source_packed = self._select_block(
         layout,
         chunk.source_block,
         phase="source",
         video_t=1.0,
         audio_t=None,
     )
     self._forward(source_packed, cache, "source")
     ```
   - 写入并提交 `target_block`（内部自动将 `pending_source` 与 clean `target` 合并提交为一个完整的 `CommittedChunk`，推进 `cache.last_block` 到 `2`、再到 `4`）：
     ```python
     self._commit_clean_prefix(
         layout,
         chunk.target_block,
         target_prefix[:, :, chunk.start : chunk.end],
         cache,
     )
     ```
5. 计算并返回新局部 epoch 的 `(media_origin, latent_end)`：
   ```python
   host = self._layout_index(layout).host
   target = host["row_role"] == int(H3RowRole.TARGET_VISUAL)
   return float(host["position_ids"][target][0, 0]), chunks[-1].end
   ```

#### 3.6 `incremental.py`：配对环、`chunk(2)` 规划与 `_prepare` 前激活
1. **配对环维护（`_remember_paired(ready, target_visual)`）**：
   - 在 `_finish(ready, current, prompt, origin)` 中，紧接 `self._remember_visual(current.visual)` 之后更新 `self.recent_paired`：
     - 取 `src = ready.source.detach().float().clone()` 与 `tgt = current.visual.detach().float().clone()`（校验两者时间帧数均为 `ready.chunk.end - ready.chunk.start`）。
     - 若 `self.recent_paired is None` 或 `self.recent_paired[1] != ready.chunk.start`（不连续时重置），令拼接后范围从 `ready.chunk.start` 开始；否则沿时间维 `dim=2` 拼接 `cat([prev_src, src], dim=2)` 与 `cat([prev_tgt, tgt], dim=2)`。
     - 只保留末尾最多 `5` 个 latent（`keep = min(5, total)`），记录精确全局游标范围 `(ready.chunk.end - keep, ready.chunk.end, src[:, :, -keep:], tgt[:, :, -keep:])`。
   - 新增只读属性 `sink_rebuild_ready`：
     ```python
     @property
     def sink_rebuild_ready(self):
         return (
             self.cache is not None
             and self.cache.prefix is not None
             and self.applied_prompt is self.prompt
             and self.recent_paired is not None
             and self.recent_paired[0] == self.committed_latent - 5
             and self.recent_paired[1] == self.committed_latent
             and self.recent_paired[2].shape[2] == 5
             and self.recent_paired[3].shape[2] == 5
             and self.committed_latent >= 5
             and self.committed_latent % 5 == 0
         )
     ```
2. **`_next_model_chunk()` 规划规则（重设局部 epoch 为 `chunk(2)`）**：
   - 当 `self.pending` **非空**时：
     - 若存在 `self.pending_prompt is not None`、或 `self.pending_sink is not None`、或当前已满足 `self._auto_sink_due`（或加上队列尾部块后即将跨越 `5 * interval_cycles` 阈值：`self.auto_sink.enabled and self.cache is not None and self.cache.window_policy.sink_chunks == 2 and self.pending[-1].chunk.end - self.sink_anchor_latent >= 5 * self.auto_sink.interval_cycles`），返回 `None` 先排空已在队列中的旧块！
     - 否则按原逻辑返回 `self.plan.chunk(self.pending[-1].model_chunk.chunk_id + 1)`。
   - 当 `self.pending` **为空**时：
     - 优先检查 `pending_prompt`（保持原逻辑不变：若就绪返回 `self.plan.chunk(self.prompt_prefix_chunk_count)`）。
     - 次优检查**是否有需要在当前边界执行的 `rebuild`**：
       - 手动 `rebuild`：`self.pending_sink is not None and self.pending_sink.method == "rebuild" and self.committed_chunk_id > self.pending_sink.after_chunk_id`
       - 或自动 `rebuild`：`self.pending_sink is None and self._auto_sink_due and self.auto_sink.method == "rebuild"`
       - 若上述任一成立**且** `self.sink_rebuild_ready` 为 `True`：返回 `self.plan.chunk(self.prompt_prefix_chunk_count)`（即新局部 epoch 的 `chunk(2)`，长度 3 帧，`start=5, end=8`）！
       - 若上述任一成立但 `self.sink_rebuild_ready` 为 `False`（典型场景：刚排空到半周期 `committed_latent == 8` 或 `103`，`% 5 == 3`）：**不返回 `None`**，继续走最后的 `return self.plan.chunk(self.model_chunk_id)` 补齐剩余 2 帧到达 `% 5 == 0` 边界！
3. **`_activate_pending_sink_rebuild(ready)`（在 `_prepare` 前执行）**：
   - 调用位置：在 `_step()`（第 1171 行）和 `drain_ready()` 的两处 `_activate_pending_prompt(...)`（第 1221 行与第 1237 行）**紧随其后、`_prepare(...)` 之前**调用 `self._activate_pending_sink_rebuild(self.pending[0])`。
   - 触发条件：
     - `ready.chunk.start % 5 == 0`、`ready.model_chunk.chunk_id == self.prompt_prefix_chunk_count`、`ready.model_chunk.start == 5` 且 `self.sink_rebuild_ready` 为 `True`；
     - 选出待执行的 `rebuild` 意图（手动 `pending_sink` 优先于自动 `_auto_sink_due`；若刚才 `_activate_pending_prompt` 已经执行了 Prompt 切换重建，则 `sink_anchor_latent` 已对齐且 `pending_sink` 已被吸收，此处直接 `return`，不会二次重建！）。
   - SP 预检与一致性投票（参照 `_rebuild_pending_prompt` 中 `reuse_prefix` 的预检范式）：
     ```python
     error = None
     try:
         _, _, src_prefix, tgt_prefix = self.recent_paired
         layout, _ = self.sampler.processor.pack_sink_rebuild_prefix(
             self.inputs, self.prompt.condition, src_prefix, tgt_prefix
         )
         host = self.sampler._layout_index(layout).host
         prefix = self.cache.validate_prefix_reset()
         block0_pos = host["position_ids"][host["block_idx"] == 0]
         if (
             prefix is None
             or prefix.host_positions is None
             or not torch.equal(prefix.host_positions, block0_pos)
         ):
             raise ValueError("Resident prefix geometry differs from sink rebuild layout")
     except Exception as exc:
         error = exc
     self.sampler._agree_sp_boundary(
         self.cache,
         0,
         phase="sink_rebuild_prepare",
         ready_route=(intent.is_auto, intent.event_id),
         precheck_error=error is not None,
         local_error=error,
     )
     if error is not None:
         raise error
     ```
   - 执行重建与局部游标重置（**注意**：无需冲刷上一块的 `_deferred_final`，因为 `reset_history_keep_prefix()` 会丢弃旧动态块并用 `recent_paired` 中完整的 5 帧 clean `src+tgt` 重新提交 block `1..4`）：
     ```python
     policy = self.cache.window_policy
     self.cache.window_policy = replace(
         policy, sink_chunks=2, sink_start=0, min_chunk_id=0, active_sinks=None
     )
     origin, latent_end = self.sampler.rebuild_sink_prefix(
         self.inputs,
         self.prompt.condition,
         src_prefix,
         tgt_prefix,
         self.cache,
     )
     if latent_end != 5:
         raise ValueError("Sink rebuild prefix does not meet the next model block")
     self.media_origin = origin
     self.model_chunk_id = self.prompt_prefix_chunk_count
     self.epoch_chunk_base = self.committed_chunk_id - self.prompt_prefix_chunk_count
     self._pending_prefix = None
     self._cached_prefix_ready = False
     if not intent.is_auto:
         self.sink_event_id = intent.event_id
         self.pending_sink = None
     self.sink_anchor_latent = ready.chunk.start
     self.sink_refresh_count += 1
     self.sink_refresh_method = "rebuild"
     self.sink_refresh_latent = ready.chunk.start
     ```
   - **异常语义**：`_step()` 与 `drain_ready()` 外层已包有 `except BaseException: self.dispose(); raise`，因此重建中途一旦抛错会自动 `dispose()` 销毁会话，严禁半残状态继续生成。

#### 3.7 Commit 2 涉及文件清单
1. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/zing_h3/packing.py`
2. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/zing_h3/sampling.py`
3. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/zing_h3/incremental.py`

---

## 4. 本地验证命令与必测用例清单（Anti-Stupidity 闭环）

### 4.1 已验证的 WSL 本地秒级测试启动器（严禁改环境）
由于 WSL 编辑机无 `torchvision`/`msgspec`/`flashinfer` 等重依赖，执行单测时使用以下已实测通过的轻量内联启动命令（耗时 `< 8s`）：

1. **纯窗口缓存测试（最快，约 2 秒）**：
   ```bash
   cd /home/ss80194/work/sglang-zing/feat-periodic-sink && \
   SGLANG_LIGHTWEIGHT_RUNTIME=1 PYTHONPATH=python python3 -m pytest --noconftest \
     python/sglang/multimodal_gen/test/unit/test_zing_h3_window.py -q
   ```

2. **`queued_dit` / `incremental` / `sampling` / `packing` 完整 CPU 会话测试（约 7 秒）**：
   ```bash
   cd /home/ss80194/work/sglang-zing/feat-periodic-sink && \
   SGLANG_LIGHTWEIGHT_RUNTIME=1 PYTHONPATH=python python3 -c '
   import sys, types, pathlib, pickle, importlib, transformers.utils.import_utils, pytest
   from sglang._platform_stubs import _MockModule
   class _Field:
       def __init__(self, *, default=None, default_factory=None): self.default, self.factory = default, default_factory
   class _Struct:
       def __init_subclass__(cls, **kw): pass
       def __init__(self, *a, **kw):
           names = list(getattr(self.__class__, "__annotations__", ()))
           for k, v in zip(names, a): setattr(self, k, v)
           for k in names[len(a):]:
               if k in kw: setattr(self, k, kw.pop(k))
               elif hasattr(self.__class__, k):
                   d = getattr(self.__class__, k)
                   setattr(self, k, d.factory() if isinstance(d, _Field) and d.factory else (d.default if isinstance(d, _Field) else d))
           for k, v in kw.items(): setattr(self, k, v)
   for m in ("torchvision", "torchvision.io", "orjson", "pybase64", "imageio", "IPython", "IPython.display", "flashinfer", "deep_gemm", "sglang.multimodal_gen.registry", "sglang.srt.utils.hf_transformers", "sglang.srt.layers.deep_gemm_wrapper", "sglang.srt.layers.deep_gemm_wrapper.configurer"):
       sys.modules.setdefault(m, _MockModule(m))
   ms = _MockModule("msgspec"); ms.Struct = _Struct; ms.field = _Field
   mp = _MockModule("msgspec.msgpack"); mp.encode = pickle.dumps; mp.decode = pickle.loads
   ms.msgpack = mp; sys.modules["msgspec"] = ms; sys.modules["msgspec.msgpack"] = mp
   for pkg, rel in [
       ("sglang.multimodal_gen.configs.pipeline_configs", "configs/pipeline_configs"),
       ("sglang.multimodal_gen.runtime.layers.attention", "runtime/layers/attention"),
       ("sglang.multimodal_gen.runtime.pipelines_core", "runtime/pipelines_core"),
       ("sglang.multimodal_gen.runtime.pipelines_core.stages", "runtime/pipelines_core/stages"),
       ("sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3", "runtime/pipelines_core/stages/model_specific_stages/minimax_h3"),
   ]:
       mod = types.ModuleType(pkg); mod.__path__ = [str(pathlib.Path("python/sglang/multimodal_gen") / rel)]; sys.modules[pkg] = mod
   def _attn(n):
       if n not in {"USPAttention", "LocalAttention", "Attention"}: raise AttributeError(n)
       return getattr(importlib.import_module("sglang.multimodal_gen.runtime.layers.attention.layer"), n)
   sys.modules["sglang.multimodal_gen.runtime.layers.attention"].__getattr__ = _attn
   raise SystemExit(pytest.main(["--noconftest", "python/sglang/multimodal_gen/test/unit/test_zing_h3_queued_dit.py", "-k", "sink", "-q"]))
   '
   ```

3. **`zing_h3_source_adapter` 协议测试 + 静态语法与 Lint 检查**：
   - 将上述命令末尾的文件路径替换为 `python/sglang/multimodal_gen/test/unit/realtime/test_zing_h3_source_adapter.py` 运行协议测试。
   - 每次提交前必须执行：
     ```bash
     python3 -m py_compile <所有修改过的 .py 文件>
     /home/ss80194/mini-sglang/.venv/bin/ruff check <所有修改过的 .py 文件>
     /home/ss80194/mini-sglang/.venv/bin/ruff format --check <所有修改过的 .py 文件>
     python3 scripts/ci/zing/run_cpu_tests.py --check-inventory
     ```
     确认全部零报错退出（`exit code == 0`；若格式不符可直接运行 `/home/ss80194/mini-sglang/.venv/bin/ruff format <文件>`，严禁 `pip install`）。

### 4.2 必须覆盖的 4 组新增单元测试断言
将以下用例分别追加到已注册的现有单测文件中：

1. **协议、双槽独立与非法输入不吞 `event_id`（追加到 `test_zing_h3_source_adapter.py`）**：
   - `session.ready` 同时暴露 `sink_updates` 与 `auto_sink_updates`。
   - `sink.auto` 对 `interval_cycles=0`、`interval_cycles=True`（布尔值伪装整数）、`method="unknown"`、`{"enabled": False, "interval_cycles": 2}` 均抛 `ValueError`，且抛错后 `adapter.ledger.event_id` 保持原值不变，后续复用同一 `event_id` 发合法事件可正常通过。
   - 同一批次连续发送 `sink.auto`（`event_id=2`）与 `sink.refresh`（`{"method": "rebuild"}`, `event_id=3`），`prepare_next_request` 生成的 `batch.extra` 同时包含 `zing_sink_auto` 与 `zing_sink_update`，互不覆盖。
2. **周期 `copy`、半周期顺延、`S0` 暂停与环境变量兜底（追加到 `test_zing_h3_queued_dit.py`）**：
   - `sink.auto(enabled=True, interval_cycles=2, method="copy")`（每 10 个 latent 一刷）：在 `0->3->5->8->10->13->15->18->20` 流式推进下，精确在 `committed_latent == 10` 与 `20` 触发刷新，`sink_refresh_count` 依次变为 `1, 2`，`sink_chunk_ids` 分别切换为对应新锚点全局 chunk ID，且同时覆盖 `defer_final=False`（`step()`）与 `defer_final=True`（`drain_ready()`）。
   - 半周期顺延：先在 `committed_latent == 8`（`8 % 5 == 3`）执行一次手动 `sink.refresh`（重置锚点到 `8`），随后开启 `interval_cycles=1`（阈值 5）：到达 `committed_latent == 13`（`13 - 8 >= 5` 但 `13 % 5 == 3`）时不阻塞供帧，自动跑完 `13..15` 块到达 `committed_latent == 15`（`15 % 5 == 0`）后触发自动刷新并将锚点对齐至 `15`。
   - `S0` 暂停与恢复：`sink.off` 进入 `S0` 后跨越多个周期不触发自动刷新；手动 `sink.refresh` 回到 `S2` 后从恢复位置重新起算周期。
   - 环境变量 `SGLANG_ZING_AUTO_SINK_CYCLES=1` + `SGLANG_ZING_AUTO_SINK_METHOD=copy` 默认生效，且收到 `sink.auto(enabled=False)` 后停止。
3. **同边界优先级与 Prompt Switch 协同（追加到 `test_zing_h3_queued_dit.py`）**：
   - 在同一 5-latent 周期边界同时到达 `conditioning`（Prompt 切换）与自动到期刷新（或手动 `sink.refresh`）时，仅执行一次 Prompt 重建，自动刷新计数不虚增、锚点重置到当前边界。
4. **`rebuild` 的 `source + target` Clean KV 数值等价性与游标连续性（追加到 `test_zing_h3_queued_dit.py`）**：
   - `pack_sink_rebuild_prefix` 与 `rebuild_sink_prefix` 产出的 `cache.prefix` 对象引用保持不变（未重跑 Qwen/block 0），`cache.last_block == 4`，`len(cache.committed) == 2`，两个 `CommittedChunk` 的行数等于 `source + target` 之和（而非仅 `target`！），且数值与用相同 `condition` + 相同 5 帧 `(src, tgt)` 冷启动前两块 clean 提交得到的 KV 逐元素一致。
   - 执行周期性 `rebuild`（含 `pooled_kv=True` 容量池模式）跨 2 个刷新周期后，输出 `ChunkResult` 的全局 `chunk.chunk_id`、`chunk.start/end`、像素范围 `provenance.pixels` 与未刷新会话严格连续无跳变，不会触发池容量泄漏或越界。
