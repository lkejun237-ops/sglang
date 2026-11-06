# Zing 动态 Window/Sink 设计（`sink_refresh` / `sink_off`）

状态：实现完成，2026-11-03。GPU 验收基线为 `0a4623e129`，运行源码为 `d87d75fb16`；提 PR 前已整合 `origin/main@a9a4b2f93e`，整合后的 CPU 回归以 PR checks 为准。默认部署模型为 unsplit `[3, 2]`，`split_first_latent=false`。

---

## 1. 要解决的问题与 Non-goals

### 1.1 要解决的问题

当前 Zing 流式会话的注意力窗口（[`ZingWindowPolicy`](../../../../../python/sglang/multimodal_gen/runtime/realtime/zing_window.py)）固定保留会话最开头的 `sink_chunks` 个块作为背景锚点（Sink），其余窗口额度分配给最近滚动历史（Recent）。

在当前默认部署基线 **unsplit `[3, 2]`（`split_first_latent=false`）** 下，1 个完整的 **5-latent 周期**对应 **2 个 chunk（`3 + 2` 或 `2 + 3`）**。但此前配置仍沿用旧 `1+2+2` 拆首块时代的 `sink_chunks = 3`，导致冷启动 Sink 膨胀为 `3+2+3 = 8` 个 latent，且与 `prompt_switch` 的 5-latent（2 块）Prefix 脱节。

因此，本设计同时解决两个绑定在一起的问题：

1. **在 unsplit `[3, 2]` 基线下将默认开启态 Sink 对齐为 `S = 2`（2 个 chunk = 恒定 5 个 latent）**：使冷启动 Sink、运行中 `sink_refresh` 锁定的 Sink、以及 `prompt_switch` 重建的 Prefix 统一为 **5 个 latent（2 个 chunk）**。
2. **支持运行中动态调整 Sink 锚点（$S \leftrightarrow 0$）**：
   - **`sink_refresh()`（重定锚点）**：将当前最新生成的至多 $S=2$ 个已完成块（5 个 latent）锁定为新 Sink，立即淘汰更早的旧 Sink 与旧历史；后续新生成的块重新积累 Recent。
   - **`sink_off()`（释放锚点）**：释放当前固定的 Sink，保留现有 Recent，后续每生成一块自然增加历史，长满后按纯滚动窗口（`0 + W`）滑动淘汰。

上述切换须满足：**不重算已提交历史 KV、不扩容固定 KV Pool 槽位、不打断输出帧与全局时间戳连续性**。用户已确认允许在切换边界单独完成原本延迟的 final；若因此拆开跨块融合，须报告新增 forward 调用和停顿，不再把这类路径描述为严格零额外调用。

### 1.2 Non-goals（非目标）

- **不开放客户端任意窗口或 Sink 大小调参**：运行中只在固定锚点块数 $S$（unsplit `3+2` 下为 `2`）与 `0` 之间切换，窗口总额度 $W$ 保持不变。
- **不恢复已淘汰历史**：切换只在当前 [`ZingH3KVCache.committed`](../../../../../python/sglang/multimodal_gen/runtime/models/dits/zing_h3_cache.py) 实际留存的块上裁剪，不回放或重算已滑出窗口的历史块。
- **不在去噪步中途切换**：不在单个块的 diffusion step 之间修改窗口，只在完整块边界生效。
- **不新增物理环形缓冲区或独立分区**：不破坏现有 KV 槽位 `[0, _length)` 连续紧凑排列的物理布局。

---

## 2. 重要设计

### 2.1 核心机制：只改保留条件，复用 `_evict_committed` 自动筛选与左移搬运

[`ZingH3KVCache`](../../../../../python/sglang/multimodal_gen/runtime/models/dits/zing_h3_cache.py) 物理上没有独立的“Sink 槽位”与“Recent 槽位”，其淘汰与存储模型为：

1. **按条件筛选**：`_evict_committed(target_block)` 遍历 `self.committed`，保留 `window_policy.retains(chunk.temporal_chunk_id, current_id)` 为 `True` 的块；
2. **原地左移紧凑**：一旦有块被筛除，`_compact_store()` 调用融合算子 `move_store_rows` 将留存块按顺序左移拼到紧邻 `self.prefix` 的 `[0, _length)` 连续区间，并更新各块的 `offset`；
3. **重算相对位置**：`_invalidate_rope(preserve_chunks=True)` 触发下一轮 forward 根据留存块的真实时间戳重新压缩 `block_relative` RoPE 坐标。

因此，`sink_refresh` 与 `sink_off` **无需重新计算 KV，也无需手动管理槽位交换**，只需在块边界更新 `cache.window_policy` 并立即调用一次 `_evict_committed()`。

### 2.2 关键数据结构：`ZingWindowPolicy`

在 [`ZingWindowPolicy`](../../../../../python/sglang/multimodal_gen/runtime/realtime/zing_window.py) 中增加两个整数字段：

- `sink_start: int = 0`：当前 Sink 锚点区间的起始 `temporal_chunk_id`（默认 `0`）。
- `min_chunk_id: int = 0`：已被显式丢弃的历史水位线，`< min_chunk_id` 的块不再保留（默认 `0`）。

```python
@dataclass(frozen=True)
class ZingWindowPolicy:
    window_chunks: int | None = None
    sink_chunks: int = 0
    sink_start: int = 0
    min_chunk_id: int = 0

    def retains(self, chunk_id: int, current_id: int) -> bool:
        return self.window_chunks is None or (
            chunk_id >= self.min_chunk_id
            and (
                self.sink_start <= chunk_id < self.sink_start + self.sink_chunks
                or chunk_id >= current_id - (self.window_chunks - self.sink_chunks - 1)
            )
        )
```

默认参数 `sink_start=0, min_chunk_id=0` 与现有行为完全等价。

### 2.3 状态转移规则与具体例子（基于 unsplit `3+2`，`S = 2`）

在 unsplit `[3, 2]` 基线下，开启态 Sink 块数固定为 $S = 2$（对应恒定 5 个 latent），设窗口总额度为 $W = 5$（即 2 个 Sink + 2 个 Recent + 1 个当前块，或任意配置 $W$），块边界处下一待生成块编号为 `current_id`：

#### （1）`sink_refresh()`：锁定最新至多 $S = 2$ 块（含不足 2 块自动补齐）

- **状态更新**：
  - 从当前 `cache.committed` 中取最后至多 $S = 2$ 个块的起始编号：
    `new_start = committed[-2].temporal_chunk_id`（若仅完成 1 个块则取 `committed[0].temporal_chunk_id`；若 `committed` 为空则取 `current_id`）。
  - 设定 `cache.window_policy = ZingWindowPolicy(W, S, sink_start=new_start, min_chunk_id=new_start)`，并立即执行 `_evict_committed`。
- **例子 A（稳态锁定 `8, 9`）**：块 9 完成后（`committed = [0, 1, 8, 9]`，`current_id = 10`）调用 `sink_refresh()`：
  - `sink_chunks = 2, sink_start = 8, min_chunk_id = 8`。
  - `< 8` 的旧块（`0, 1`）被 `chunk_id >= 8` 立即淘汰；`8, 9`（长度分别为 `3` 和 `2`，共 5 个 latent）左移到槽位开头成为新 Sink。
  - 生成 `10, 11, 12` 时，Recent 依次为 `—`、`10`、`10, 11`；生成 `12` 时恢复满窗口 `2+2+1`。生成 `13` 时，`current_id - 2 = 11`，块 `10` 才正常滑出。默认 `after_step0` 在第 0 步完成后执行这次淘汰。
- **例子 B（历史不足 2 块自动补齐）**：仅完成块 `0`（`current_id = 1`）时调用 `sink_refresh()`：
  - `sink_chunks = 2, sink_start = 0, min_chunk_id = 0`。
  - 块 `0` 保持锁定；随后块 `1` 完成时因满足 `0 <= 1 < 2` 自动补入第 2 个 Sink 名额（凑齐 `3+2=5` 个 latent）；块 `2` 起自动进入 Recent。

#### （2）`sink_off()`：释放当前 Sink，保留现有 Recent 继续滚动

- **状态更新**：
  - 将水位线推过当前已锁定的 Sink 区间：`new_min = max(min_chunk_id, min(sink_start + sink_chunks, current_id))`。
  - 设定 `cache.window_policy = ZingWindowPolicy(W, sink_chunks=0, sink_start=0, min_chunk_id=new_min)`，并立即执行 `_evict_committed`。
  - 若当前已是 `sink_chunks == 0`，则保持原策略不变（无操作）。
- **例子（释放 `8, 9`）**：块 11 完成后（Sink 为 `8, 9`，Recent 为 `10, 11`，`current_id = 12`）调用 `sink_off()`：
  - `sink_chunks = 0, sink_start = 0, min_chunk_id = 10`。
  - `< 10` 的旧 Sink（`8, 9`）立即淘汰；Recent `10, 11` 左移到槽位开头全部保留。
  - 生成 `12, 13, 14` 时，历史依次为 `10, 11`、`10, 11, 12`、`10, 11, 12, 13`，加当前块后窗口从 3 块 → 4 块 → 5 块自然长满 `0+5`；生成 `15` 时 `10` 才正常滑出。默认 `after_step0` 在第 0 步完成后执行这次淘汰。

### 2.4 为什么将默认 `S` 对齐为 `2`（与 5-latent 周期绑定）

仓库中同时存在“按 **5 个 latent** 计数的 Prompt Switch Prefix”和“按 **chunk** 计数的 Sink（此前默认 `sink_chunks=3`）”，两者的错位源于分块配置从 `split_first_latent=true` 向 `false` 的演进：

| 维度 | 旧 split 模型（`1, 2, 2, 3, 2, 3, 2...`） | 现 unsplit 基线（`3, 2, 3, 2, 3, 2...`）配旧 `S=3` | 现 unsplit 基线（`3, 2, 3, 2, 3, 2...`）**对齐 `S=2`** |
| :--- | :--- | :--- | :--- |
| **1 个 5-latent 周期对应的块数** | **3 个 chunk**（`1 + 2 + 2 = 5`） | 2 个 chunk，但 `S=3` 强锁 3 块 | **2 个 chunk**（`3 + 2 = 5`） |
| **任意块边界 `sink_refresh` 锁定的长度** | `5 / 7 / 8` 个 latent 跳变 | `3+2+3 = 8` 或 `2+3+2 = 7` 个 latent 奇偶跳变 | **恒定 `3+2 = 5` 或 `2+3 = 5` 个 latent！** |
| **`prompt_switch`（5-latent Prefix）重建后的 Sink** | 重建 3 块（5 个 latent），刚好填满 `S=3` | 重建 2 块（5 个 latent），导致切换后第 1 个新去噪块 `chunk 2` 也被误锁入第 3 个 Sink 名额 | **重建 2 块（5 个 latent），刚好填满 `S=2`，新生成块直接进 Recent！** |
| **Sink 保留画面长度**<br>（不等于 KV Pool 峰值） | 启动为 5 个 latent | 启动为 8 个 latent | 启动为 5 个 latent；相邻完整两块 refresh 也是 5 个 latent |

**绑定 `S = 2` 的核心收益**：

1. **物理存储保持整块（chunk）淘汰，同时长度严格恒定为 5 个 latent**：在 `3, 2, 3, 2...` 序列中，任意相邻 2 个块必然是一个 `3` 加一个 `2`，因此无论在奇数块还是偶数块边界执行 `sink_refresh`，锁定的新 Sink 永远恒定为 **5 个 latent**，无需等待 5-latent 周期边界，也无需拆碎任何单个 chunk。
2. **与 `prompt_switch` 的 5-latent Prefix 完全同构**：切换 Prompt 重建出的 2 个 Prefix 块（`3+2 = 5` 个 latent）恰好填满 `S = 2`，不再误锁切换后的第 1 个新生成块。
3. **固定锚点画面长度**：相邻完整两块的 Sink 恒为 5 个 latent。KV Pool 仍须覆盖开启、关闭、刷新及 `after_step0` 淘汰前的最大实际行数，不能据 Sink 少 3 个 latent 推导预分配也少 3 个。现有 `peak_rows()` 在 unsplit、无音频、每 latent 一行且不计条件 prefix 的口径下，W6/S3 与 W6/S2 的 `after_step0` 峰值都为 18；W5/S3 为 16，W5/S2 为 15。PR #483 的 OOM 不能据此认定已解决。

### 2.5 生效边界、与 Prompt 切换联动及多卡同步

- **生效边界与跨块融合（`cross_chunk`）排空**：
  - `SourceReadySession` 持有 `pending_sink`（与 `pending_prompt` 独立，同一边界内多次 sink 事件遵循 latest-wins）。
  - 当 `pending_sink` 非空时，与 `pending_prompt` 一样停止跨块预跑入队，等已入队的 `pending` 块完成后，在下一块开始前应用 `window_policy` 更新并执行 `_evict_committed`。
- **与 `prompt_switch` 的关系**：
  - `prompt_switch` 在 5-latent 周期边界调用 `cache.clear()` 并用最新 5 个 latent 重建局部 Epoch（局部 `chunk_id` 重新从 `0` 开始）。
  - 重建后，`sink_start` 与 `min_chunk_id` 重置为 `0`，`sink_chunks` 保持当前开关状态（开启态为 $S$，关闭态为 $0$）。若同一边界同时存在待生效的 `prompt_switch` 与 `sink` 操作，合并在同一次重建中设置最终的 `sink_chunks`。
- **SP 多卡同步**：
  - `_agree_sp_boundary` 同步校验完整 `window_policy`（含 `sink_chunks, sink_start, min_chunk_id`）。

---

## 3. 备选方案及其缺点

### 备选方案 A：复用 `prompt_switch` 的 Clean Prefix 重建实现 `sink_refresh`

- **做法**：在 5-latent 周期边界清空 `cache.committed`，保持当前 Prompt 的 `cache.prefix` 不变，用 `session.recent_visual`（最近 5 个 target latent）重新执行 2~3 次 `t=1.0` 的 clean target forward，作为新局部 Epoch 的开头块写入。
- **缺点**：
  1. **引入额外推理延迟与周期等待**：每次 `sink_refresh` 需多跑 2~3 次 DiT forward，且必须强制等待 5-latent 周期边界（`committed_latent % 5 == 0`），无法在任意完整块边界零开销生效。
  2. **KV 内容与初始 Sink 不一致**：重建出的 Prefix 块只有 `target` KV（丢失了 `source` KV）；且在 unsplit `3+2` + `sink_chunks=3` 下，5-latent Prefix 仅生成 2 个块（`3+2`），会导致刷新后生成的第 1 个新去噪块（`chunk 2`）被一并锁入第 3 个 Sink 名额。
  3. **仍需修改窗口策略支持 `sink_off`**：`sink_off` 要求保留现有 Recent 继续滚动，无法通过 Prefix 重建实现。

### 备选方案 B：S2 平滑过渡语义（`sink_refresh` 时保留旧 Recent 维持 6 块满窗口）

- **做法**：刷新时不丢弃旧 Recent，窗口始终维持 6 块，随新块生成逐块替换旧历史。
- **缺点**：
  1. **旧场景残留污染**：在硬切场景下，旧场景块会在注意力上下文中残留约 5 个块之久，`block_relative` RoPE 的最早时间参考点也要延迟到 `c+5` 才能切到新 Sink。
  2. **过渡状态复杂**：需要额外追踪新旧两代块的交替配额，不如直接推进 `min_chunk_id` 简洁确定。

### 备选方案 C：在 KV Pool 物理显存中划分独立的 Sink 分区与 Recent 环形区

- **做法**：在 `ZingLayerStore` 中固定切分 Sink 物理行与 Recent 环形行，通过指针或 gather 索引访问。
- **缺点**：
  1. **破坏连续前缀假设与 CUDA Graph 契约**：现有 `ZingFixedForwardKVPlan` 要求所有有效历史行紧凑位于 `[0, history_rows)`；且 `[3, 2]` 分块下奇偶块行数不同，固定分区会产生内部碎片，而现有 `_compact_store()` 的融合 Triton 左移算子已无碎片地解决了该问题。

## 4. 已确认的验证范围与素材选点

以下为用户在 2026-11-03 对验收方案 1–5 全部选择 A 后确认的范围；不是已执行或已通过的结果。

实现入口为会话的 `update_sink(operation, event_id)`。连续 source WebSocket 使用 `kind="sink.off"` 或 `kind="sink.refresh"`，`type="event"`、递增 `event_id`、`payload={}`。ready 的 `sink_updates` 表示部署支持；`control_ack.stage="accepted"` 仅表示接纳，实际生效由输出块快照的 `sink_event_id`、`sink_chunks`、`sink_chunk_ids` 判断，后者使用全局生成块编号。集成式 source 与独立 Encoder/DiT 路径复用相同会话逻辑。

首版动态切换限定完整视频原生块、unsplit `[3,2]`、启动 sink 为 0 或 2、窗口至少 4 块；更小窗口无法保证最后两个留存块相邻，拒绝启用。显式部署 profile 仍优先，历史 S3 profile 不会被静默改写；以 S2 启动时须将请求与 pool 的 sink 配置同时设为 2；以 off 启动时两处同时设为 0。音频和旧 split 模型不启用动态 sink。

### 4.1 CPU 与 GPU 分工

- CPU 回归进入 CI，覆盖 sink 操作独立生效、prompt 尚未满足周期边界、同边界组合、cross-chunk 已预执行、deferred final 尚未提交以及连续操作 latest-wins。逐 forward 核对实际可见历史、角色、事件版本与全局输出范围，不只断言 ACK 或 policy 字段；奇偶相位作为低成本参数覆盖。
- GPU 功能和视觉优先：480 室内三组视觉对照、产品 reference-switch 交互、代表脚本串行/优化路径对照，以及取消/EOS 后下一会话可用性。
- 736 使用显式冻结的派生输入，做关键视觉和最复杂交互回归；独立记录输入身份、预处理和结果，不把原始 1280×720 source 称为 736 输入。
- 使用当前 unsplit `[3,2]` 模型。优先 H100；无空闲或容量预检不满足时自动转 B300，同一个 A/B 在相同硬件上重跑。不使用 H200 产品实例。
- 最多同时 4 卡，累计不超过 16 GPU·小时；排队时间单列，到限收尾并列出未测项。完整产品四 case、三次重复的性能 A/B 不纳入本轮，明确保留未完成状态。
- 允许在预算内再筛最多两个视觉候选、调整事件位置、修复实现与测试问题并重跑失败项。禁止重算已提交历史；允许单独完成原本延迟的 final，并报告融合拆分成本。

### 4.2 先看素材，再选择生效区间

视觉首选为 `zing-whitebox-video-30s-two-triview-0001`：原 source 为 1280×720、24 FPS、720 帧、30 秒；480 用例目标为 832×480。原 source SHA-256 为 `19b458b45cf2baafb2352e27fc8f7da6b203f588bcbd1f5f3776afd45f1fd182`，已与清单核对。

原 source：`oss://leap-worldmodel-beijing-single-az/world-model/platform/worldkit-delivery-ledger/world500-20261011-mg720-r4/assets/19b458b45cf2baafb2352e27fc8f7da6b203f588bcbd1f5f3776afd45f1fd182.mp4`。

已检查完整 30 秒逐秒概览，以及 12–24 秒每 0.25 秒的连续帧。画面约在 15.5 秒开始横向转身，18.75 秒左右恢复向前；约 22 秒再次转向。基于这些运动选择首轮候选，而非把历史报告的故障时间直接当事件发送时间：

| 操作 | 原 source 视觉候选 | 选择依据 |
| --- | --- | --- |
| `sink_off` | 14.5–15 秒，source 帧 348–360 | 在第一次明显转身前释放初始锚点，观察随后通道开口、墙体边界和人物遮挡 |
| `sink_refresh` | 约 20 秒，source 帧 480 | 在新视角稳定后锁定；实际选中的两块必须均位于约 18.75–21.75 秒的稳定段，否则顺延到满足条件的首个完整块边界；若已越过该段则放弃此候选 |

表中是待校准的实际生效区域，不是保证在该 source 帧提交事件就立即生效。测试须记录事件发送、模型接纳、实际生效边界和被锁定块的 provenance；客户端 source 提交进度、模型生成进度和输出接收进度分别记录。最终触发位置根据原片观察与真实模型边界共同冻结。

视觉对照保持 prompt 不变，比较固定 S2、off、refresh；prompt/reference 交互另用 `zing-product-case-20261029-reference-switch`，保留其 source 更新帧 `34, 289, 374, 544, 561, 629`，避免 prompt 重建掩盖 sink 本身的效果。历史稀疏注意力结果只作候选线索，不作为当前模型或本功能已通过的证据。

实际执行状态、作业编号和证据边界见[验收记录](dynamic-window-sink.validation.zh-CN.md)。
