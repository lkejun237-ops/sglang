# 三 case benchmark：2026-10-12 r1

本轮采用完整海岸 5s、废土 30s、地中海小镇 30s。后两者为 PR #261 的七图输入，
不替代旧六图 ID。完整 source、prompt 和参考条件未裁剪或改写。

| 选择器 | 准确 sample ID | 帧数 / FPS | 参考数 | 正式请求 |
|---|---|---:|---:|---:|
| 5s | zing-whitebox-video-five-triview-0001 | 120 / 24 | 6 | 3/3 成功 |
| 30s-wasteland | zing-whitebox-video-30s-six-triview-0001 | 720 / 24 | 7 | 3/3 成功 |
| 30s-town | zing-whitebox-video-30s-six-triview-0002 | 720 / 24 | 7 | 3/3 成功 |

## 部署与执行口径

三个 case 各自启动一个常驻服务，完成一次完整业务预热后串行执行三次正式请求；
并发固定为 1，seed=123，few-step 四步采样，SP2。客户端与服务在同一节点，通过
loopback WebSocket 通信。模型使用 GPU 0/1，GPU 2/3 未参与模型计算；四卡均为
NVIDIA L20D，单卡 torch 可见显存 274113 MiB。运行环境为 torch 2.13.0+cu130、
websockets 15.0.1。

使用本地有限 Encoder/Decoder，没有独立 Encoder 服务。CUDA 采集模式记录逐 rank
阶段，因此这些数字是带采集开销的诊断基线，不是无采集开销性能成绩。

## 客户端结果

以下均为三次正式成功请求的中位数。FPS（全程）是原始客户端记录的 `总帧数 / E2E`；
FPS（首包后）是网页从 chunk 到达时间重算的 `(总帧数 - 首包帧数) / (E2E - TTFF)`。
两种口径都保留，不能混用。

| case | TTFF / s | E2E / s | FPS（全程） | FPS（首包后） | TPOC / s | ICL 中位数 / s |
|---|---:|---:|---:|---:|---:|---:|
| 5s | 10.9239 | 20.5825 | 5.8302 | 12.3206 | 0.4599 | 0.0065 |
| 30s-wasteland | 36.2012 | 100.0196 | 7.1986 | 11.2663 | 0.5025 | 0.0054 |
| 30s-town | 36.8128 | 101.3725 | 7.1025 | 11.1254 | 0.5089 | 0.0057 |

每个 5s 请求收到 22 个媒体 batch；每个 30s 请求收到 128 个。IFL 仍未测：协议只有
逐 batch 到达时间，没有 batch 内逐帧到达或显示时间戳，不能插值伪造逐帧延迟。

## 阶段覆盖与作业状态

每个请求、每个 rank 的严格 ledger 如下：

| case | Encoder | DiT | Decoder | Conditioning | Qwen 独立 span |
|---|---:|---:|---:|---:|---|
| 5s | 7 | 22 | 8 | 1 | 未测，严格要求为 0 |
| 两个 30s | 8 | 128 | 43 | 1 | 未测，严格要求为 0 |

`Encoder` 数量分别等于 reference VAE 加完整 source VAE。帧/latent 连续性、货号、
rank、CUDA 记录和 collector health 均通过。各 rank 的区间会重叠，阶段累计也包含
嵌套工作，不能相加成 E2E 或解释为纯 kernel 时间。

GPU 作业 `dlczc9wcr6ogva9j` 在阿里云控制面终态为 **Failed**，此状态必须保留。
原因是冻结时的旧分析器要求每个请求存在 1 个独立 Qwen span，而当前 main 先通过
本地 `LocalConditioner` 回调计算 Qwen；该工作包含在 Conditioning host 区间内，
但不会进入随后被跳过的 Qwen span。生成子进程本身完成了全部 12 个请求。

CPU 后处理使用更新后的分析器，显式选择 `conditioner-unmeasured`：它严格要求 Qwen
span 恰为 0，并继续执行上述所有 VAE/DiT/Decoder 约束。三组复核均为 `valid=true`，
`qwen_scope=not_measured_conditioner_callback`。这项复核没有补造 Qwen CUDA 时间，也
没有把原 GPU 作业状态改写为成功。

作业记录（UTC+8）：

- CPU 预检 `dlc03h1cbksqy5t7`：14:59:50–15:06:23，Succeeded；23 项素材大小、SHA、
  完整视频解码和图片校验通过。
- GPU `dlczc9wcr6ogva9j`：15:27:00–15:53:46，Failed（旧 Qwen 门禁）；全部生成完成。
- CPU 后处理 `dlc1jbrwgbu8w279`：16:02:53–16:03:57，Succeeded；严格复核和导出通过。

## 媒体与网页

九个正式输出全部转为 H.264 MP4，并通过完整解码：5s 输出为 832×480、24 FPS、
120 帧；30s 输出为 832×480、24 FPS、720 帧。同一 case 的三次输出 SHA 相同，符合
固定 seed 的本次运行结果。三个 case 的中点帧肉眼抽查均为非空、可辨识且符合对应
场景；这仍不是完整时序与画质验收，最终人工判断需要观看完整视频。

真实数据的 Chromium 验收覆盖：三个内置 case 切换、三 case JSON 导入、当前结果
JSON 导出、CSV 下载、无效文件拒绝、桌面渲染、390px 无页面横向溢出、零 page error、
零外部网络请求。展示的参考数、帧数和 chunk 数分别为 6/120/22、7/720/128、
7/720/128。

本地报告：

- `exports/issue253-three-cases-20261012-r1/view/benchmark-report.html`
  SHA256 `d4cda6d855a508d2cb9251aabfeb8a2177a915ca847d84fd0c91cb4913950abe`
- `exports/issue253-three-cases-20261012-r1/view/benchmark-report.json`
  SHA256 `8ebd1e3405c4cc0aabdb931782fa51b16a01c1a17aaabc455c9de45da299e8d1`

云端原始导出、九个视频、三份阶段验证和证据包位于私有 OSS 前缀
`world-model/sglang/evals/issue253-three-cases-20261012-r1/`。`export-receipt.json`
列出的 16 个文件已下载并逐一通过 SHA256 回读校验；证据包 SHA256 为
`1a632b5de1e5607e1f514a4226aaa94f950ed73c0e84059b24483944b6a71788`。

## 版本边界

GPU 运行源码为提交 `aaa141d4b5b3d140f0e079da10264ce470c558b2` 加冻结 overlay。
基础归档 SHA256 为
`44850b6d6896bdd45f5e9b9ca06ec9db6efca5a0f3d06481d518e2eb014ec323`，overlay SHA256
为 `2de46875dced2cbfcb68d7d74db4a353eb5a07e7816d337d1eed5ad4ca10f066`，冻结清单 SHA256
为 `b77721b0bab02a8ded2d3e3c776c329548fd5db3e8cfa3fea4b8fde9946c4c50`。

`deployment.json` 顶层的 `server_source_commit`、`client_overlay_sha256` 和
`source_bundle_sha256` 是继承自 2026-10-10 部署的历史字段；本轮源码身份以同文件的
`source_freeze` 以及上述冻结清单为准。大权重沿用既有 checkpoint，只核验 symlink
目标与元数据，不声称重新逐字节哈希全部权重。

本地最终合同为 56 passed / 1 skipped（缺本地 ffmpeg）；云端另行完成了实际媒体
转码与完整解码。数值/native 对照、人工画质、真实播放器、音频同步和长期性能仍是
独立验收项。本轮音频关闭，三次样本也不足以形成稳定 p99 或容量结论。

## checkpoint 简化入口实测

2026-10-12 追加单入口验证：`run_benchmark.py` 自动读取 checkpoint 配置，启动一个
常驻 SP2 服务，串行运行默认三 case；每 case 一次预热、三次正式请求。阶段采集
关闭，checkpoint 附带本轮 TAEH3 pipeline 配置，复用已校验素材缓存。未在此 GPU
作业验证 native 默认 decoder 或首次 S3 下载。

CPU 准备 `dlcrc79jflt7tfv7`、GPU `dlcpofbijm0f7fe4`、CPU 后处理
`dlc1gvec5nxu9uu8` 均 Succeeded。GPU 时间为北京时间 18:31:38–18:52:43，退出码 0。
全部 12 个请求完整成功，9 个正式 MP4 为 832×480、24 FPS、120/720 帧且完整解码
通过。服务日志记录 application shutdown complete；另有运行时 semaphore 清理警告，
保留在原始 server.log 中。

| case | TTFF / s | E2E / s | 全程 FPS | 首包后 FPS |
|---|---:|---:|---:|---:|
| 5s | 10.8470 | 20.0117 | 5.9965 | 13.2327 |
| 30s-wasteland | 35.3813 | 98.0238 | 7.3452 | 11.4778 |
| 30s-town | 36.4663 | 99.4789 | 7.2377 | 11.5101 |

工件根为 `exports/issue283-checkpoint-entry-20261012-r1/`。`freeze.json` 记录此前
CPFS 源码基座与新入口覆盖文件 SHA；冻结归档无 `.git`，所以入口显式传入完整
`--source-revision`。CPU 预检核验生产 launch 模块导入及三个 case preflight。
实际配置、启动命令、全部请求与原始 chunk 均在 `cloud/evidence/` 中。

入口生成的 `cloud/report.html` 与 `cloud/report.json` 已通过真实数据浏览器检查：
三个 case 切换、三 case 导入、JSON 导出、390px 布局、零页面错误与零外部请求。
13 个导出文件下载后逐项匹配回执 SHA。HTML SHA256 为
`81a70d5cd72de0b748d612f700e610dfd22244138c5bdf29534f0581ddcde935`，证据包 SHA256 为
`ecd525b428dbf01342e3f53477c64d59dd35337819e4c506e69268dc1b06c614`。

提交前修正了报告模板对导入字段的 HTML 转义。原始云端报告保留不改，使用同一批
接收记录在本地重建 `view/report.html`；恶意 JSON 浏览器回归在原报告上失败、
在修复后报告上通过。该显示修正没有改变客户端或模型计时数据。
