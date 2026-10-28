# Block0 Cache 默认启用与持久身份

## 已确认的范围

2026-10-28，在 PR #445 合入后的 `5e66d5e2c3` 上继续开发。

- GPU 仅保存当前活跃会话 KV。没有 GPU KV pool 时跳过 Block0 Cache，服务继续启动，
  不要求缓存参数、不扫描资产、不分配 pinned KV。
- 配置了 GPU pool 时，Block0 CPU/磁盘缓存默认开启。生产模式只有 `on/off`；
  开启统一沿用原 overlap DMA 恢复，不改变模型计算、切词边界或 `1+2+2` history 提交。
- 必须显式提供整个部署的 pinned KV 预算、已挂载的磁盘根目录、磁盘容量预算。
  缺失、参数不合法、目录未挂载/不可写或 pinned 分配失败，启动退出。
- H100 目标机型 `efg2.NHxcn` 的云端规格是 2048 GB CPU 内存。该机建议起始配置
  800 GB pinned KV/部署，SP2 每 rank 约 400 GB；不是每 rank 800 GB。
  当前可用内存、容器上限与实际 800 GB 分配尚未验证。
- 只做 CPU 缓存正确性验证；本轮不申请 GPU，不重跑 GPU A→B→A 或性能 benchmark。

## 部署参数

先在部署环境挂载 CPFS，并在该文件系统上创建专用缓存根目录。服务不挂载文件系统，
也不自动创建缺失的根目录；存在但未挂在独立 volume 上的 `/cpfs` 空目录不能充当缓存盘。
路径检查确认有非根文件系统挂载祖先；CPFS 类型和持久性由部署方提供的卷保证。

在原有 Zing 服务启动命令末尾添加（NVIDIA CUDA/Linux）：

```bash
--zing-block0-cache-host-gb 800 \
--zing-block0-cache-dir /cpfs/zing/block0-cache \
--zing-block0-cache-disk-gb DISK_BUDGET_GB
```

把 `DISK_BUDGET_GB` 换成该部署拥有的容量数值。所有 `*-gb` 参数使用 **十进制 GB，1 GB = 10^9 bytes**。
这些参数属于部署，在服务启动时传入，不出现在生成请求中。关闭使用
`--zing-block0-cache-mode off`；默认值是 `on`。不再读取原来的
`SGLANG_ZING_BLOCK0_CACHE_MODE`、`SGLANG_ZING_BLOCK0_CACHE_ENTRIES`、
`SGLANG_ZING_DISK_CACHE_*`；原消融 `late/early/overlap` 不再是生产选项。

L1 根据最大 prefix 行数和每 rank KV geometry 算出一个槽位的字节数，
以 `floor(部署预算 / SP度数 / 单槽字节数)` 预分配并复用 pinned 槽位。
剩余不足一个槽位的字节不分配；预算连一个槽位都放不下时启动失败。
该预算是 pinned KV 存储预算，不是整个进程的 RSS 上限：CPU conditioning 快照、
Qwen/参考图小缓存、工作区和文件系统 page cache 另占内存。
同一机器运行多个部署时，分别分配预算，不能每个部署都假定独占整机内存。

## Key 与模型身份

现有严格请求 key 的字段保留：规范化请求快照、媒体内容摘要及顺序、S0 像素摘要、
width/height/fps、冻结的 window/sink/RoPE/source 等策略。
冻结策略中新增模型/布局身份摘要，所以 startup、prefill 和 prompt switch 使用同一规则。
本轮不放宽 fps、无关请求字段或近似媒体匹配。

每次服务启动扫描内容，生成并冻结本 worker 生命周期的身份：

| 部分 | 身份来源 |
| --- | --- |
| DiT | 复用 `adaln_table.checkpoint_verified` 为真的 manifest `identity`，包括 config SHA256 与全部选中 checkpoint 分片 SHA256；resident 模式使用同一 checkpoint resolver/identity 函数 |
| 本地 Qwen | 实际 text_encoder、tokenizer、processor 路径下的完整内容摘要，跟随 symlink；同一内容换挂载路径不会改变摘要 |
| 远程 Qwen | 已握手确认的 conditioning profile 中 model/projection/tokenizer/processor 资产身份；完整 profile ID 同时进入兼容契约 |
| 图像 VAE | 实际 video_vae 文件内容摘要；编码配置、posterior、latent 归一化等有效 runtime/arch 配置进入兼容契约 |
| 额外 LoRA | 若配置，纳入对应文件或目录的内容身份；模型资产在服务期间不可替换 |

Qwen/VAE 扫描由 SP rank 0 执行并广播成功/失败结果；DiT 使用 AdaLN 已验证结果，
不会为 Block0 再读一遍。**AdaLN 现有 loader 自己的逐 rank checkpoint 校验没有改变**。
扫描期间不 materialize GPU 权重、不运行模型，扫描失败让各 rank 一致拒绝启动。

兼容契约独立记录 SP/Ulysses/Ring/TP、dtype、有效模型/编码配置、AdaLN 表摘要/算术与计划、
GPU 型号/CC、PyTorch/CUDA/cuDNN、算术开关、相关库版本、编码选项、实现文件摘要。
首版严格匹配。没有 GPU UUID、进程 ID、storage 地址、输出 event_id、缓存预算或根目录身份。
显式环境变量与默认值即使数值等价，也可能产生保守 miss；本轮不建立兼容性白名单。

## CPFS 目录与 SP2

用户仅指定根目录。程序自动组织：

```text
root/
  .index.lock
  lru_index.json
  .tmp/<rank-write-uuid>/
  v2/<model-sha256>/<layout-sha256>/<key[:2]>/<key>/
    manifest.json
    rank-0/kv.bin
    rank-0/conditions.pt
    rank-1/kv.bin
    rank-1/conditions.pt
    .complete
```

SP2 两 rank 共用模型/key，分别保存完整 prefix 行上的本地 attention-head 分片。
conditions 按 rank 独立保存，不假定复制的 conditioning 字节完全一样。
每次 fill 共享一个 generation，防止不同部署的写入混合成一个条目；已发布条目不再覆盖。
全部预期 rank 的文件、generation、身份和大小都合法，才发布 `.complete`。
缺分片/截断/身份不符/旧格式都返回 miss；SP2→SP1/SP4 不重分片，使用不同布局命名空间。
读取还会核对本地 KV shape/dtype。先取元数据、再升入 L1 期间若发生淘汰或替换，安全 miss。

根目录的同一个 `flock` 保护发布、索引、淘汰和 KV 读入，避免读取已被淘汰的文件；
L2→L1 读入前等待该 pinned 槽位的旧 DMA 结束。多个 rank 的磁盘读可能因此串行，
本轮没有新的 L2 时延结论。

仅按根目录下所有 v2 模型/布局的实际管理字节数做 LRU，包含尚未齐全的条目，不限制条数。
旧格式目录不读取、不删除、不计入新池；部署使用新的专用根目录或另行清理旧文件。
临时写盘空间、其他文件不属于管理预算，部署须预留余量。
启动后磁盘 IO 错误继续沿用 miss/跳过写回路径。既有 CPU 槽位复用等待后台写盘的行为保留，
没有引入写盘队列或承诺慢盘下零阻塞。

### Conditioning 安全格式

`conditions.pt` 使用 `zing-block0-conditions-v1` 数据协议，只写入 CPU Tensor 和基础
容器/标量。`H3InferenceInputs` 与 `ZingKVRequest` 展开为字段字典，读取后仅通过代码
固定的两种构造函数重建；文件不能指定任意 Python 类。读取统一使用
`torch.load(..., weights_only=True)`，不增加自定义类白名单，也不回退到不受限加载。
字段缺失、非法 Tensor/形状、未知记录类型和旧条件格式均为 miss。每 rank manifest
记录条件协议，旧 manifest 在反序列化前拒绝。目录仍由部署控制写权限；本协议不提供
内容签名，也不将受限加载当成任意恶意文件的资源隔离。

本 PR 保留现有 L1/L2 LRU、最大尺寸槽位和后台写盘等待策略；缓存准入、分层热度、
槽位利用率与磁盘索引开销留到后续 PR。

## 验证与交付

代码、用例与验证文档先以 Draft PR 交付。
测试范围、当前结果和未测项见 [CPU 验证记录](/benchmark/zing_h3_causal/BLOCK0_CACHE_DEFAULTS_VALIDATION)。
本设计不提供新的 480/736 性能、GPU 数值或 CPFS 实机验收结论。
