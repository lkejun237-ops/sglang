# 多地域手工发布

SGLang 的生产源码分支使用 `main`。本仓库 PR CI 执行测试；云平台 webhook/自动发布规则须另行回读。镜像构建、推送和地域启用分别核验，不以源码合并当作已部署。
Worldkit CLI/Stream Host 使用其 `dev` 分支可达的不可变提交，无需先合入 SDK 的 `white_box` 分支。
手工构建器和 Jenkins 都会校验提交的祖先关系；`sdk-revision.txt` 独立于 SGLang `main` 提交固定 SDK 的精确源码版本。

持续维护的操作流程和验证边界见
[Whitebox 手工发布](../../docs/docs/sglang-diffusion/zing/plans/whitebox-manual-releases.mdx)。

上海 CPU 镜像及存储变更、受保护资源、回滚边界和短时检查的限制，见
[2026-11-04 部署说明](SHANGHAI-CPU-20261104.zh.md)。

```sh
python deploy/whitebox/release.py --help
python deploy/whitebox/release.py plan --region shanghai --bindings /private/shanghai-bindings.json --output /private/shanghai-release.json
python deploy/whitebox/model_service.py --bundle /private/shanghai-release.json
python benchmark/zing_h3_causal/run_conditioning_cpu_tests.py -q deploy/whitebox
```

以上生成计划的命令不会启动服务。`build` 只构建本地镜像。
下载受限时，`build --pip-index-url <公共HTTPS索引> --go-proxy <公共HTTPS代理>`
可显式选择构建期下载源；URL 不得携带凭据、查询参数或片段。未设置保持原有源，
不改变运行环境、依赖 pin、TLS/Go 校验和验证或镜像测试；先在实际构建机核验可达性，
示例与阶段范围见上方手工发布入口。
可重复的 `--build-host HOST:IP` 只转发构建期 `--add-host`，默认无映射；不改宿主机、
daemon 或运行时配置，不接受 `host-gateway`。仅使用已核实地址并保留原 HTTPS Host/SNI
和正常证书校验，具体镜像示例及验证边界同见手工发布入口。
`model_service.py --run` 会在已经分配的四卡作业内启动所有模型角色；
`plan`、`build`、合并 CI 和 `verify` 都不会调用它。

`bindings.example.json` 中是占位值，不是线上端点。实际 bindings、凭据、准备好的资产和生成的计划均保存在 Git 之外。
地区 JSON 只接受硬件要求、存储路径及资源预算，模型行为、媒体与玩家期限统一放在 common.json。
旧地区启动器、现场请求和 SDK 覆盖副本已移除，见[清理清单](../../docs/docs/sglang-diffusion/zing/plans/whitebox-pr647-cleanup.md)。

模型策略默认使用 `quantization: "none"`。经过审查的原生在线 FP8 发布需将
`model.quantization` 设为 `"fp8"`，并选择 `model.quantization_scope`
（`ffn`、`fc1`、`interior-qkv-fc1` 或 `ffn-attention`）。
启动器只向 DiT 添加 `--quantization fp8 --direct-gpu-weight-loading false`，
并在模型环境中固定 `SGLANG_ZING_QUANT_SCOPE`。保留原始源码 checkpoint，
以及 Encoder、conditioning 和 Decoder 的精度；该策略不选择序列化 FP8 权重、
量化 KV、MXFP8 或 NVFP4。缺少这些字段的已有 bundle 继续使用 BF16 启动计划。
配置校验通过不代表 GPU 速度或画质一致性已通过验证。

## 项目音频测试的发布输入

项目音频发布通过 `benchmark/whitebox/sdk-revision.txt`，将 CLI/Stream Host 固定到
已审查的 SDK `dev` 提交 `689e81f718e22065877de9af589e6496c712f979`。
它使用配套 SG 音频桥接引入的可选封存音频 manifest 和 source 时钟快照。
CPU 控制服务及 Room 镜像都应从最终合入的 SG `main` 提交构建；
源码集成和版本固定本身不会激活任何服务。

原项目音频任务的首次验收范围仅限现有上海 `shcopy` 测试路由，且关闭手动推理。
保持生产域名、Coordinator、worker 和外部模型服务不变。
分别验证就绪状态、已鉴权资源的 hash、音频 source 帧证据、Player 实际可听效果，
以及刷新和退出后的清理。镜像及原生组件检查、最终 CI 仍为必需项；
固定此版本不代表已部署、音频可听或 GPU 性能已通过验证。

## 镜像层证据

镜像层检查会保存解析后的不可变本地镜像，并读取其真实配置历史。
支持传统 Docker config ID 和 containerd 存储中的 OCI index，
并校验 descriptor/config SHA-256 及所选平台。
`docker history` 显示 `0B` 不代表该层是 `empty_layer`：
containerd 存储可能将真实 COPY 层的大小显示为零。
只有配置中的 `empty_layer` 元数据决定该分类，非空历史项还必须与配置中的文件系统层数量匹配。
现有的单份源码树、CPU 镜像不复制 SDK、Room 镜像只复制一份 SDK 的规则保持不变，
隐藏的旧镜像层也在检查范围内。

检查期间会在构建机上临时导出镜像，完成后删除归档。
保存前会立即检查实际临时文件系统（可能与 DockerRootDir 不同），
仍保留 15 GiB 空间，并采用保守的 CPU/Room 待写入空间预算。
将 TMPDIR 指向空间充足的私有构建目录；临时文件系统空间不足时，
会在写入镜像层之前失败。最终配置的 OS/架构必须与所选平台一致，
传统镜像和直接 config 格式的镜像均需满足这一要求。
保留正常的 Docker 根目录剩余空间检查和 BuildKit 缓存；
不要为绕过显示格式问题而跳过镜像校验或修改服务源码。
已保存的 containerd 元数据夹具
`deploy/whitebox/fixtures/containerd-zero-size-history.json`
可复现大小显示为零的情况，不包含镜像层内容或私有配置。

## 运行用户的归档权限

手工构建器支持操作者使用 077 等私有 umask。
Python 的 `tarfile.data_filter` 会有意移除目录权限信息，
直接解包因此可能创建 root 所有、权限为 0700 的源码目录。
发布导出器保留该过滤器对路径、链接、类型和所有权的安全限制，
并恢复 Git 归档中每个目录的安全权限（`mode & 0755`）。
文件可执行位及安全的相对符号链接仍沿用原过滤器行为；
不会递归设置 0777，也不会修改操作者的 umask。

运行时源码映射会拒绝运行用户无法遍历或读取的源码根目录及子目录。
该检查在发布源码树前执行，避免仅由 root 运行的构建测试将不可读的服务判为通过。
需显式启用的本地 Docker 回归覆盖：umask 077 下的 Git 归档、实际 bind mount 映射，
以及 UID10001 下的模块/符号链接读取和可执行文件调用：

```sh
WHITEBOX_TEST_DOCKER_EXPORT=1 PYTHONPATH=python SGLANG_LIGHTWEIGHT_RUNTIME=1 \
  python -m pytest deploy/whitebox/test_release.py \
  -k actual_runtime_uid --confcutdir=deploy/whitebox
```

## main 发布身份与地域参数

解析不可变提交前，先获取 SGLang `origin/main` 和 Worldkit `origin/dev`。
仅存在于 main 的 SGLang 修复可以直接构建，无需复制回 white_box。
新的 CPU、Room 和模型镜像声明 `source_branch=main`；
配置后的运行时及发布核验使用同一身份。历史产物保留原始身份，不重新标记。
隔离的 `codex/` 运行时身份仍需沿用已有的显式开发授权开关。

`verify` 要求提供 Gateway、Coordinator、Room 和 model 的观测信息，
包括每个角色实际使用的镜像 digest、源码/SDK SHA 和配置指纹。
Coordinator 缺失或版本过旧都会导致失败，不能判为发布成功。

两地统一使用公共策略：quantum=1、prompt-joint=false，开启 prefetch、异步输出、
Decoder overlap 和 defer-final；AdaLN=resident、W6/S0、每玩家期限1800秒及公共媒体预算。
地区文件不能覆盖这些行为。若将公共策略显式关闭，CLI也会传递0/false；实际参数解析测试
覆盖两地一致性和相反父进程环境变量。当前候选的GPU初始化后最终值、延迟与画质仍未验证。

## 选择正确的启动方式

- 已分配的同机四卡完整组：`prepare_assets.py`消费同一bundle，然后由`model_service.py`
  按 Encoder → Conditioning → Decoder → DiT → heartbeat 等待ready后逐项启动；就绪后进程并行。
- 测试环境仅新增 standalone DiT、复用既有Encoder/Decoder：使用原生`dit_worker`及
  `SGLANG_ROLE_POOL_WORKER_CONFIG`。不得调用整组启动器或重启已有Encoder/Decoder/Coordinator。
  完整字段来源、命令与验收/回滚边界见上方唯一手册。
- 独立源码归档、OSS 分发及三角色启动：见 [原生角色启动器](../sglang/README.md)。
  managed Encoder/Decoder 的接线与验收边界见唯一手册 C 节；当前启动器不透传 pool 配置或 Secret。

实际bindings、Secret、准备视图、plan和进程日志均在Git外。plan、资产准备和模型state输出
会检查该边界；构建继续使用临时Git archive，不扩大共享.gitignore。
