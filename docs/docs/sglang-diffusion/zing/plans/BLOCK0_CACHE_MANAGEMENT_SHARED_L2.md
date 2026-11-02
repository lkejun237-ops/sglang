---
title: "Block0 Cache 管理、业务索引与显式预加载"
description: "用户管理缓存、简化内容 key、game/group 索引和实例级 preload/status/unload 的确认方案。"
---

# Block0 Cache 管理、业务索引与显式预加载

## 状态与范围

2026-10-30 汇总本轮讨论确认的目标方案，继续在
[PR #467](https://github.com/seedleap/sglang/pull/467) 完成，保持 Draft。
候选代码已实现新 key、目录、业务索引和新 API。提交 `65ae9daf74` 的缓存专项两轮
378 项通过，零失败/跳过；真实 NVMe 小规模索引/目录探针已通过。完整 CPU CI（`87ec8e38f3`）已通过。
bind/unbind 尚未纳入实现。
验证记录见文末。2026-11-01 已同步 main 并解决冲突，保留 source 缩放和首块调度改动。
参考图 resize 上限作为有效输入的一部分参与 key，避免相同原图在不同缩放设置下误用 KV。
同步后的 `c54e926815` 已通过[缓存专项](https://github.com/seedleap/sglang/actions/runs/41279424307)
（两轮 380 项，零失败/跳过）和[完整 CPU CI](https://github.com/seedleap/sglang/actions/runs/41279424329)
（12,746 项记录，零失败、432 项跳过）。PR 保持 Draft：真实 GPU 视频路径尚待验收。

截至提交 `1889a04f78`，代码已实现同机本地 NVMe L2 管理、force prefill、
单文件原子发布和周期 GC；普通请求 L1 miss 仍会同步加载磁盘，且计算后自动写缓存。
候选代码已按本文调整这些行为。旧模型/layout 命名空间、旧格式和旧进程混跑不提供兼容或迁移。
新格式为 schema `zing-block0-v3` / disk_format=4；旧 `v2/` 目录不加载或计入新容量扫描，
升级时使用新空目录，或由部署方清理旧目录。
旧版本说明见[原实现快照](https://github.com/seedleap/sglang/blob/1889a04f78c950eec209c6ec8650b75410abf74f/docs/docs/sglang-diffusion/zing/plans/BLOCK0_CACHE_MANAGEMENT_SHARED_L2.md)。

部署范围仍是各实例独立 L1、同机共享本地 NVMe L2；不同机器的本地盘相互独立。
跨机器网络存储不在支持范围。每个部署实例只允许一个活跃会话。

## 业务读写路径

| 入口 | 读取 | 新缓存写入 |
| --- | --- | --- |
| 正常视频，`cache_write=false`（默认） | 只查 L1；miss 直接重算，不探测或读取 L2 | 不新增 L1/L2 条目，不登记业务关联 |
| 正常视频，`cache_write=true` | 同上 | 将新计算前缀填入 L1，异步写 L2；有业务标签时登记关联 |
| prefill，`force=false`（默认） | 沿用已有 L1/L2 复用能力 | miss 时计算，填入 L1 并异步写 L2；有标签时登记关联 |
| prefill，`force=true` | 跳过 Block0 L1/L2 命中，强制重算 | 更新本实例 L1，异步覆盖 L2；有标签时登记关联 |
| preload | 已在 L1 就复用，否则显式从 L2 加载 | 不重算模型，不创建新的磁盘 KV |
| unload | 按选择范围定位本实例 L1 条目 | 移除 L1 条目，保留磁盘和业务索引 |

正常视频的 L1-only 规则同时覆盖初始 prompt 和后续 prompt 切换。
`may_contain` 等条件预取提示也不能访问磁盘。
当前正常路径的填充点是新 prompt 首次输出之后，不是专门在换下旧 prompt 时写入；
新开关控制这条填充路径。关闭写入仍允许读取和使用已有 L1。

prefill 成功回执不等待 L2 落盘；force 不影响其他实例已经存在的 L1。
Qwen/参考图编码缓存仍可复用。已有 L1 槽位若被后台写盘占用，复用时仍可能等待，
本轮不引入独立写盘缓冲池或固定延迟承诺。

## 2026-11-01 确认增量：缓存行为日志与 FPS 去重

本轮只补 cache 生命周期日志并移除 preparation key 中重复的 `source.fps`，不扩展会话全链路日志。
模型包 `zing_config` 保持完整；媒体 FPS、抽帧与时间戳等有效输入身份保持原有语义。
删除重复字段会改变本候选版本生成的 key；旧 key 不迁移，按新 key 重新 prefill。

沿用现有 logger，前缀 `zing_cache_event` 后输出单行 JSON。公共字段：
`operation_id`、`instance`、`rank`、`event`、`cache_key`、`result`、`reason`、`elapsed_host_ms`；
有上下文时附 `session_id`、`event_id`、`game_id`、`group_id`。实例身份使用主机名与进程号，
rank 使用现有分布式/磁盘 owner 身份，耗时仅为 host wall time。

| 路径 | 事件及级别 |
| --- | --- |
| 在线读取 | INFO：L1 hit / miss 后重算 / cache disabled；保留部分 SP miss 原因 |
| prefill / L1 填充 | INFO：复用、force 重算、填充完成、LRU 淘汰 |
| 异步 L2 写入 | INFO：开始与成功；WARNING：失败/跳过，保留原始操作关联字段和实际 rank |
| 管理操作 | INFO：preload/unload/delete/clear 开始、结果与数量/耗时；WARNING：部分失败或拒绝 |
| 状态轮询、分片 | DEBUG：status/list、逐 rank 正常结果 |
| GC | 有删除时 INFO 汇总占用、原因、数量与字节；空转 DEBUG；失败 WARNING |

正常 SP 结果仅在各实例 SP rank 0 输出 INFO；其他 rank 的正常记录用 DEBUG，异常在实际 rank 输出 WARNING。
日志上下文是每次操作的小型元数据快照，经现有 scheduler RPC 传递管理操作的 operation_id；
L1 entry 随其生命周期保留该快照，异步写盘线程持有独立上下文，不写入 KV manifest/conditions、业务索引或 key。
不记录 prompt/媒体内容，不逐帧打印，不因日志新增 GPU 同步；prefill 成功和 L2 落盘成功分开记录。

最小闭环走现有 CPU CI：验证 source FPS 去重、模型配置和媒体输入仍隔离；
捕获在线 hit/miss/关闭、force prefill、L1 填充/淘汰、L2 异步成功/失败、管理部分失败/拒绝、GC/轮询的事件与级别。
特别验证操作关联字段跨异步线程不串线、非零 rank 正常日志为 DEBUG、WARNING 保留实际 rank、无 prompt/媒体泄漏。
不追加 GPU/NVMe 验收，既有性能证据保持版本边界；本轮 CPU run 链接与回读回执在 PR 描述中更新。

## Cache key

返回给业务的 `cache_key` 是 64 位小写十六进制 SHA-256 字符串：

```text
cache_key = SHA256(canonical_json({
    cache_schema_version,
    zing_config,
    inputs,
    sp,
    kv_dtype
}))
```

| 字段 | 计算内容 |
| --- | --- |
| `cache_schema_version` | 手工维护的缓存契约版本；KV 语义或存储格式改变时升级 |
| `zing_config` | 直接纳入 `zing_config.json` 的 JSON 内容，排除纯路径记录字段 `checkpoint`；不单独计算 model_id |
| `inputs` | 有效模型输入：prompt、参考媒体内容和条件参数、宽高、配置的参考图 resize 上限，以及存在时的 S0 内容 |
| `sp` | SP 大小及分片方式，例如 Ulysses/Ring 配置 |
| `kv_dtype` | 实际 KV 存储类型 |

JSON 规范化使字段顺序和排版不影响 key；媒体按内容计算，不以部署路径或下载 URL 代替内容身份。
默认值归一化后使用有效输入；操作元数据须在计算输入前剥离。
`game_id`、`group_id`、`event_id`、请求/会话编号、`force`、`cache_write` 均不参与 key。

移除旧 `runtime_policy` 和嵌套的大指纹，不再扫描代码文件、库版本、GPU/环境信息，
也不为 Block0 身份重复扫描整套 Qwen/VAE 权重。此项不改变 AdaLN 等组件自身的加载校验。
部署方保证实际模型与配置符合模型包内的 `zing_config.json`，模型包在服务期间不可变；
外部权重、组件或 LoRA 覆盖须由部署方同步到配置身份中，重启服务后重新生成指纹；
缓存不自动发现运行中修改。

## 目录、索引与生命周期

```text
cache_root/
├── entries/
│   └── <cache_key>/
│       ├── manifest.json
│       ├── rank-0/
│       │   ├── kv.bin
│       │   ├── conditions.pt
│       │   └── ready.json
│       └── rank-1/
│           └── ...
└── index/
    └── <game_id>/
        └── <group_id 或内部未分组标记>/
            └── <cache_key>.json
```

目录图展示逻辑组织；业务 ID 必须安全编码为路径，未分组使用不会与业务 group_id 冲突的内部标记。
不再设置 model/layout/SP/dtype 目录层，也不保留 current 指针或 generation 目录。
manifest 保留输入身份、SP、dtype、形状等可检查的元数据，并用于加载校验。

逻辑数据结构如下：

```text
各实例 L1：cache_key -> 本实例 pinned 槽位与条目
共享 L2：cache_key -> manifest + 各 rank 的 KV/conditioning
业务索引：game_id -> group_id/未分组 -> set(cache_key)
```

同一 key 只有一份磁盘实体和每实例至多一份 L1 条目，可被多个游戏或组引用。
索引仅保存关联，不改变 key、不复制 KV。prefill 命中已有条目时也应登记请求携带的业务关联。
正常视频仅在允许写入且生成了缓存时登记关联；无标签的缓存仍可按 key 管理。
索引采用每条关联独立文件的简单组织，不对一个共享大 JSON 进行无锁读改写，
也不引入跨 KV、索引、L1 的事务或回滚。

| 操作 | L1 | L2 | 索引 |
| --- | --- | --- | --- |
| unload | 移除目标实例对应 key | 保留 | 保留 |
| 按 key 删除磁盘 / clear / GC | 保留 | 删除对应实体 | 保留，允许悬空关联 |
| unbind（待确认） | 保留 | 保留 | 只移除指定关联 |

unload 释放缓存槽位，不释放启动时预分配的 pinned 内存。
同 key 被多组引用时，unload 会改变这些组在该实例的实际 ready 状态，不做分组引用计数。
删除磁盘后，已有 L1 和活跃会话继续使用；在途写入仍可重新生成条目。
索引指向缺失磁盘实体时，preload 返回 missing，不自动计算或修复关联。

## 外部接口与参数

所有 HTTP 路由放在现有 SGLang HTTP 服务，不新增缓存服务或 Web 页面。
客户端不传磁盘目录、model/layout、SP 或 dtype；这些由部署配置和服务端计算决定。
下表描述候选接口；model/layout 管理参数已移除。game_id/group_id 为非空字符串，
最多 128 个 UTF-8 字节，存盘时进行安全编码。
完整 HTTP 状态码、返回值 JSON 字段字典、CLI 与部署红线见
[Block0 缓存管理 API 与部署契约](/docs/sglang-diffusion/zing/specs/block0-cache-management-api)。

### Prefill WebSocket

入口仍为 `/v1/realtime_video/prefill`，沿用现有二进制 MessagePack 请求及
`request_id`、prompt、参考图、source_frame 等字段，回执包含 `cache_key`。
现行协议见 [prefill 契约](/docs/sglang-diffusion/zing/specs/block0-prefill-websocket)。

| 字段 | 必填 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `force` | 否 | `false` | 严格布尔值；强制重算并覆盖 |
| `game_id` | 否 | 无 | 登记该 key 与游戏的关联 |
| `group_id` | 否 | 未分组 | 游戏内分组；传入时必须同时提供 game_id |

成功表示各 rank 的 L1 条目已经就绪，沿用原准入、会话清理与取消处理。
不等待异步 L2 完成。request_id 只关联请求回执，不是缓存身份或幂等键。

### 正常视频会话

在现有会话创建请求上增加以下字段，其他视频事件和 event_id 语义不变：

| 字段 | 必填 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `cache_write` | 否 | `false` | 严格布尔值；控制整个会话的新前缀缓存写入，包括 prompt 切换 |
| `game_id` | 否 | 无 | 开启写入时登记游戏关联 |
| `group_id` | 否 | 未分组 | 开启写入时登记分组；要求同时提供 game_id |

event_id 仍是会话内事件序号，不用于表示游戏、图包或阶段。业务阶段由 group_id 表示。
`cache_write=false` 时业务标签不触发索引写入。

### Preload 与 unload

```http
POST /v1/zing/cache/preload
POST /v1/zing/cache/unload
```

两者共用 JSON body：

```json
{
  "game_id": "game-a",
  "group_id": "stage-1",
  "cache_key": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
}
```

| 字段 | 必填 | 含义 |
| --- | --- | --- |
| `game_id` | 是 | 目标游戏 |
| `group_id` | 否 | 限定到游戏内指定组 |
| `cache_key` | 否 | 限定到单个 key，并验证其属于所选游戏/组 |

只有 game_id 时选择该游戏全部关联，包括未分组条目，并对 key 去重。
game_id + group_id 选择整组；game_id + cache_key 选择游戏内单 key；
三者都有时验证该 key 在指定组中。未登记的 key 不通过这个选择器直接加载。

preload 返回选中的 key、逐项结果与最终 L1 驻留检查。
已在 L1 的条目直接复用；否则加载并校验 L2，各 rank 全部成功才视为 ready。
缺失、不兼容或加载失败逐项返回，不重算、不回滚已成功加载的条目。
成功仅表示 L1 可用，不表示已恢复到 GPU 或永久驻留。

目标 key 去重后超过 L1 总槽位时，在加载前拒绝；否则沿用 LRU 淘汰其他条目。
完成后重新检查整组选中 key 的实际驻留，不能把历史成功次数相加当作 all_ready。
不增加分组 pinning、自动阶段切换或自动卸载；中台驱动 preload/status/unload。
空选择（例如不存在的游戏）返回空列表且 all_ready=false。容量不足返回 HTTP 409 /
`cache_capacity_exceeded`；实例 busy 返回 HTTP 409 / `cache_busy`；不属于指定游戏/组的 key
返回 HTTP 422 / `invalid_cache_selection`；DP>1 的 L1 管理请求返回 HTTP 409 / `cache_requires_dp1`。

### 状态与磁盘管理

| 接口 | 参数 | 作用 |
| --- | --- | --- |
| `GET /v1/zing/cache/status` | 可选 query：game_id、group_id、cache_key | 不传时返回实例 L1 概览；传入时返回对应条目驻留状态；group 要求 game |
| `GET /v1/zing/cache` | 可选 query：cache_key、limit=100、offset=0 | 磁盘条目列表、容量扫描快照；分页最大 limit 沿用现有 1000 |
| `DELETE /v1/zing/cache/entries/{cache_key}` | 路径中的 cache_key | 删除该磁盘实体 |
| `POST /v1/zing/cache/clear` | 无参数 | 清理该部署配置的共享目录中的全部磁盘缓存实体，保留业务索引 |

status 查询目标 worker 的真实 L1，返回 `capacity`（总槽位）、`entries`（所有 rank 都驻留的条目数）、
`cache_keys`、`busy`、`selected[{cache_key, ready}]` 和 `all_ready`；
没有选择 game/group 时可用 cache_key 查询单条。它不等同于 HTTP 进程中的磁盘管理器。
既有 /health 仍只表示服务就绪，不自动绑定某游戏的缓存 ready。
列表与容量快照不是实时磁盘余额，返回采样时间，未知容量不能当作零；
分页不保证并发写删下的快照一致性。

删除与 clear 只删除磁盘实体，不等待活跃会话结束，不强制清空其他实例 L1。
同 key 的磁盘实体被多个组共用，删除会影响这些组后续的 preload。
返回实际 deleted/missing/failed 等结果，允许部分成功，不回滚。
现有薄 CLI 跟随 key/目录契约调整，不另设一套模型调用入口。

### 尚未确认：独立 bind/unbind

以下是补充建议，不作为本轮已确认必做接口：

```http
POST /v1/zing/cache/bind
POST /v1/zing/cache/unbind
```

body 中 game_id、cache_key 必填，group_id 可选；省略 group_id 表示未分组关联。
bind 登记已有 key 的关联；unbind 仅解除指定关联，不删除 KV、其他组关联或 L1。
即使暂不提供这两个接口，prefill/允许缓存写入的视频请求仍会自动登记关联。

## 调度与并发

中台选择目标实例，调用 preload/status 后，将视频流量路由到同一实例。
L1 属于 worker/SP 组；HTTP 进程通过现有调度通信转发，不能直接管理模型内存。
首版限定一个实例对应一个 SP 组（DP=1），复用 scheduler RPC；暂不扩展多 DP 缓存管理。
不让新 L1 管理操作 round-robin 或广播到其他副本。

prefill、preload、unload 等会修改 L1 的操作在实例空闲时串行执行，忙时返回 busy；
status 作为状态读取不抢占活跃会话。不增加后台任务队列或进度服务。
断连和取消沿用已有生命周期清理：已开始的文件读取完成后再释放资源，
保留已成功加载的条目，不增加可中断 IO。

同 key 结果继续视为等价，允许重复计算、并发覆盖和跨写入批次组合分片。
每个文件先写唯一临时文件，写完关闭后原子替换，不原地修改正式文件。
只要求单文件原子性，不要求整组分片同时切换，不增加 flock、租约或严格 OCC。
读取保留 key、格式、shape、dtype、分片齐全及安全 conditioning 校验；
任一 rank 加载失败则该 key 整组 miss/失败。序列化继续使用受限安全加载。
磁盘持久化失败在边界捕获，尽力清理自己的临时文件，保留已成功 L1，不立即重试。

## 容量、GC 与部署

沿用已实现的周期 GC：同一共享目录指定一个实例执行，其内部仅全局 rank 0 清理。
启动扫描一次，此后定期扫描元数据估算逻辑字节数，不读取 KV 内容。
按内部文件最新 mtime 清理超时临时文件/未完成残留，不能只看父目录 mtime。
超过高水位后按近似 LRU 淘汰 key 到低水位。成功删除才扣减估算，竞争或失败由下轮校正。
索引文件不作为 KV 完整或 ready 的证明，GC 不因索引仍引用而阻止淘汰。

写入不全盘扫描、不更新共享精确总计数；管理查询使用最近扫描快照并带时间。
水位不是硬配额，允许两个实例同时写入时暂时超出，不做实时空间预留。
没有 non-current generation 或旧版本保留期；超慢写入被残留清理打断时允许失败。

部署方配置已挂载的本地 NVMe 路径、各实例 L1 预算和磁盘高水位，另需：

1. 指定一个实例启用 `--zing-block0-cache-gc`，其余实例关闭。
2. 给磁盘留余量，覆盖并发写入、临时文件和同盘其他数据；写满按 L2 失败降级。

现有可调参数保持：`--zing-block0-cache-gc-interval-s=60`、
`--zing-block0-cache-gc-stale-s=600`、
`--zing-block0-cache-gc-low-watermark=0.9`；
`--zing-block0-cache-disk-gb` 是清理实例使用的高水位。
不增加自动 GC 选主/接管、L1 失效广播、持久 tombstone 或事务框架。

## 实现清单与验证边界

候选实现清单：

- [x] 简化 key，移除 runtime_policy、大身份扫描和 model/layout 目录层。
- [x] 正常视频 L1-only，包括 may_contain 提示；增加会话级 cache_write。
- [x] 实现 game/group 关联索引，标签与操作元数据不进入 key。
- [x] 接入 preload/status/unload、目标 worker 路由与空闲准入。
- [x] 修改磁盘列表/删除/clear 和 CLI 的参数与目录遍历。
- [x] 同步实际协议、部署说明与测试，保留明确的 CPU/NVMe 与待补 GPU 验证边界。

2026-10-30 已确认第一轮闭环为 CPU CI + 真实 NVMe 小规模验证，复用已有大 KV IO 结果。
CPU 用例覆盖：

- key：JSON 顺序/排版、checkpoint 路径及业务标签不改变 key；
  有效输入、zing_config、SP、dtype 和 schema 改变则 key 改变；
  删除代码/环境扫描后仍保留实际文件加载校验。
- 正常路径：L1 miss 和预取提示均不调用磁盘；cache_write 默认关闭，
  开启后覆盖初始与 prompt 切换的新前缀写入，关闭时仍可命中 L1。
- 索引：关联幂等、同机并发写入、游戏/分组选择和去重、悬空关联及多组共享实体。
- preload/status/unload：容量提前拒绝、部分成功、跨 rank 一致结果、
  最终实际驻留检查、卸载不删除磁盘或索引。
- HTTP/调度：参数校验、目标实例一致性、busy、断连/取消清理及下一请求可用。
- 新目录的删除/clear/GC 保持旧持久化失败和部分成功的容错边界。

上述 CPU 用例已通过[缓存专项 CI](https://github.com/seedleap/sglang/actions/runs/41034480612)，
378 项为两轮合计，包含每轮 156 项缓存/KV pool、6 项 Graph provider 和 27 项 API 用例；
[完整 CPU CI](https://github.com/seedleap/sglang/actions/runs/41033440960) 已通过，源码提交 `87ec8e38f3`。
真实 NVMe 已补验新索引与目录的同机并发、持久化和恢复，
Pod `pr467-managed-cache-20261030-r1` Succeeded / exit 0；记录见[验收文档](/benchmark/zing_h3_causal/BLOCK0_CACHE_MANAGEMENT_VALIDATION)。
后续 GPU 单独验证实际视频路径命中/重算与输出一致性，不把第一轮 CPU 结果当作 GPU 验收。
CPU 回归交 GitHub Actions；WSL 只做编辑、diff、静态检查，不补装运行依赖。
新增真实 NVMe 验证优先覆盖索引持久化/并发与新目录，不因文档更新重跑大 KV IO。
本次 NVMe 作业为零 GPU；新业务路径的 480/736 GPU 性能与数值均未验证。

历史结果保留其源码边界：运行时 `e42b951c15` 缓存专项两轮 362 项、完整 CPU CI 通过；
2026-10-30 工具提交 `08883300b3` 在本地 NVMe 上通过 181 项 CPU 用例、
双进程发布/删除/恢复/GC，以及三轮大 KV IO 精确校验。
合成 SP2 每 key 16.406 GiB，冷读/页缓存热读/写入并 fsync 中位数为 2.840/2.093/7.911 秒；
这些是 pageable CPU Tensor 文件 IO，不是新 preload 或业务 TTFF 的实测。
完整历史回执见[共享缓存验收](/benchmark/zing_h3_causal/BLOCK0_CACHE_MANAGEMENT_VALIDATION)。
