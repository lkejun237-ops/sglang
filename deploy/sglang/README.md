# 通用 Encoder、DiT、Decoder 启动器

`launch.py` 直接启动现有原生 worker，不导入 `deploy/whitebox`、Room 或 Worldkit SDK。
单进程服务只允许一个活跃会话；通信组、签名连接、缓存和媒体协议仍由现有 worker 提供。
本目录负责源码分发、启动和本角色的 readiness，不代替 Coordinator 注册、通信组建连或生成验收。

## 稳定入口

| 命令 | 输入 | 输出／动作 |
| --- | --- | --- |
| `prepare` | 本地仓库、完整 40 位 commit、OSS prefix | `source.tar`、`source.json`；只执行 git archive，不上传 |
| `render` | source manifest、role config、site config | 平台 JSON、角色配置和 `render.json`；离线 dry-run，不提交 |
| `run` | 已验证解包目录、role config、commit/hash | 启动原生 worker，等待 `/health`，输出 JSON 回执，转发退出码 |

`run` 是容器内部入口，通常由渲染的 bootstrap 调用。解释器固定为目标镜像的
`/opt/sglang/bin/python`（Python ≥3.10），镜像必须已有匹配的 torch/CUDA、原生扩展和
`ossutil 2.x`。准备端只用 Python 标准库和 Git。H100 节点选择器兼容 Python 3.6。
依赖镜像只提供依赖：镜像中的 `/opt/whitebox/python` 不作为 SGLang 实际源码。
本 PR 不构建新的镜像；基础镜像必须按 digest 核验依赖、ABI、ossutil 和资产挂载。
旧 Whitebox Dockerfile 的 SDK 构建参数和 ENTRYPOINT 不是本入口的依赖。

## 依赖环境与源码身份

`transport-deps`、`deps` 是历史实验使用的目录名，不是启动器要求的固定目录。
当前启动器将 worker 的 `PYTHONPATH` 固定为已验证的 `source/python`，不继承外部依赖目录。
需要的第三方包和原生扩展应预先安装到目标镜像的 `/opt/sglang/bin/python` 环境，固定版本与
wheel/镜像摘要，并检查 ABI；不能仅设置旧实验的 `DEPS_DIR` 就认为本入口会加载它。

H3 模型侧 transport 的精确 pin 可查
[`requirements-h3-transport.txt`](../whitebox/requirements-h3-transport.txt)，当前为
`uvicorn==0.39.0`、`websockets==15.0.1`；支持范围以
[`SourceTransportBudget`](../../python/sglang/multimodal_gen/runtime/realtime/source_transport.py)
和 [`python/pyproject.toml`](../../python/pyproject.toml) 为准。
Gateway/Coordinator/Room 的 `benchmark/whitebox/requirements.txt` 以及控制面 CPU 测试的
`scripts/ci/zing/requirements-role-pool.txt` 使用独立解释器环境，不能安装进 GPU 模型环境覆盖 transport。
这里引用 Whitebox 的现有 pin 作为依赖核验入口，不引入 Whitebox bundle 或 SDK 运行依赖。

镜像准入先在同镜像、同解释器、同源码路径做零 GPU 的真实导入、transport 版本、WebSocket 回环、
Lynn loader 和角色 CLI 检查；CPU 测试的 platform stubs 不进入部署环境。
这些检查与本启动器运行时的 GPU preflight 分开记录，均不能代替模型加载和三角色生成验收。
`source/benchmark/zing_h3_causal` 仅供单独运行 Master/客户端时使用；worker 不需要该工具导入目录。

镜像内 `release.json` 只描述构建时源码。当前 worker 环境白名单不保留
`WHITEBOX_RELEASE_MANIFEST`，原生版本接口因此报告 `unmanaged`；保留镜像原 manifest，
用运行回执中的实际 commit、源码包摘要和导入文件路径/hash 证明本次源码身份。
实验记录另外保存依赖清单、pipeline/profile 身份和独立的 harness SHA；本入口不支持运行时
overlay/hooks，需要改源码时先提交推送，再生成新源码包，避免回执与实际执行代码分离。
Whitebox 正式发布仍使用完整 main 镜像、bundle 和四角色版本核验，不能拿本入口回执代替
`release.py verify`；见[唯一发布手册](../../docs/docs/sglang-diffusion/zing/plans/whitebox-manual-releases.mdx)。

## 固定模型与资产边界

三角色均固定 `zing-h3-dmd-1102-step2000-generator`，即
`h3-dmd-sp4-05-1102-ac3d5b28/global_step_002000/generator`，非 EMA。
原始权重：

```text
oss://leap-worldmodel-thailand-single-az/world-model/zing/checkpoints/run-archive/dmd/h3-dmd-sp4-05-1102-ac3d5b28/checkpoints/global_step_002000/generator/model.pt
```

已有泰国 serving 优先查：
`/cpfs/world-model/sglang/whitebox-serving/model-step2000-ac3d5b28`，或
`/cpfs/world-model/sglang/evals/product736-newmodel-20261104-r3/model`；同轮 Lynn 在相邻 `lynn`。
已有配套 AdaLN table，应从实际 pipeline/profile 核对，而不是重新生成或使用 1027 旧表。
历史入口为
`/cpfs/world-model/sglang/whitebox-serving/streameval-dmdac3d-step2000-prefix0-sink2-w9-20261106-r4/launch_main.sh`。
2026-11-09 只读 `GetJob(dlc1njpqj2o7ohpb)` 确认状态 `Stopped`，UserCommand 只引用该脚本，
未返回 pipeline/table 内容。**上海持久 serving 包和 table 精确路径仍未核实**，不能直接套用泰国 CPFS。

`examples/` 是故意不能直接执行的配置骨架；`REQUIRES_*` 必须由资产/平台准备环节替换。
运行者不填写 GPU UUID、worker UUID、IP 或随机 Job/Pod 名。尚未核实的路径、镜像 digest、
CPU/内存规格、卷和 Secret 挂载保留为阻断项，不提供虚假的可运行默认值。

资产 owner 提供非敏感 `assets.json`，角色配置冻结其 SHA256。结构为：

```json
{
  "schema": 1,
  "model_id": "zing-h3-dmd-1102-step2000-generator",
  "component": "generator",
  "ema": false,
  "model": "/verified-shanghai/model",
  "files": {
    "/verified-shanghai/model/zing_config.json": "<sha256>",
    "/verified-shanghai/model/model.safetensors": "<sha256>",
    "/verified-shanghai/dit-pipeline.json": "<sha256>",
    "/verified-shanghai/decoder.json": "<sha256>",
    "/verified-shanghai/lynn/weights.safetensors": "<sha256>",
    "/verified-shanghai/adaln/manifest.json": "<sha256>",
    "/verified-shanghai/adaln/table.safetensors": "<sha256>"
  },
  "adaln": {
    "mode": "weights_free",
    "table": "/verified-shanghai/adaln",
    "model_id": "zing-h3-dmd-1102-step2000-generator",
    "schedule": {"copy": "exact schedule object from the verified table manifest"},
    "cuda_provenance": {"copy": "exact arithmetic object from the verified table manifest"}
  }
}
```

示例文件名不是实际资产清单；所有模型 shard、Qwen/VAE/Lynn 等外部组件、配置和 table 都须列入。
启动器逐文件验 SHA256，并要求 model 下全部可发现的 `.safetensors/.pt/.bin`、模型配置及本角色配置受清单覆盖。
`model_id` 是资产 owner 对原始导出来源的声明，不能代替原始 checkpoint 到 serving 的导出证据；
不能通过给旧权重改标签获得验证。各角色使用同一份已审阅的资产回执，避免新旧模型混配。
Lynn 等外部路径完整性由 owner 的清单保证，不猜测任意 decoder 配置中的路径语义。

DiT 的 table manifest 的 schedule/arithmetic 必须与资产回执一致，且 manifest 和 table 文件均验 hash。
实际模型、LoRA、schedule、运行 CUDA/GPU 的兼容性继续由原生 AdaLN loader 校验，不绕过原生 gate。
如已验证部署实际使用 resident 模式，应记录 `"adaln":{"mode":"resident"}`，pipeline 中不得同时设置 table 路径。
Encoder 只接受其独立 pipeline（可以省略）；Decoder 读取独立 JSON 后作为 `--decoder-config` 传入。
不得把含 Lynn decoder_config 的 DiT pipeline 无条件传给 Encoder。

## 准备与 OSS 分发

先 commit/push 全部源码并核对远端 SHA，再在该提交的 checkout 执行：

```sh
SOURCE_COMMIT=$(git rev-parse HEAD)
python3 deploy/sglang/launch.py prepare --repo "$PWD" --commit "$SOURCE_COMMIT" \
  --oss-prefix oss://YOUR_AUTHORIZED_BUCKET/sglang/releases --output /private/release/source
SOURCE_URI=$(python3 -c 'import json; print(json.load(open("/private/release/source/source.json"))["uri"])')
ossutil cp /private/release/source/source.tar "$SOURCE_URI"
```

输出目录必须不存在，40 位 SHA 不接受分支名/短 SHA；Git archive 不包含未提交文件、模型或 Secret。
URI 包含完整 commit 和 tar SHA256，消费端始终固定摘要，不使用 latest 或工作区覆盖。
只向已授权的 bucket/prefix 上传；OSS 认证由现有 workload identity/配置链提供，不出现在参数或回执中。
服务端禁止覆盖/WORM 策略由已有 bucket 管理；内容寻址本身不是存储层写保护。
[OSS PutObject 文档](https://www.alibabacloud.com/help/en/oss/developer-reference/putobject)
说明 forbid-overwrite 在启用或暂停版本控制的 bucket 上不生效，不能将此参数当作普适 WORM。
对象若被错误覆盖，容器端摘要检查会拒绝启动。上传失败或结果不确定时先核对对象，不改写 manifest 指向其他内容。

渲染必须使用选定 commit 的相同 `launch.py/source.py/select_gpu.py`，工具摘要不一致会拒绝。
容器先下载至私有临时目录，校验整个 tar，然后安全解包；拒绝路径逃逸、特殊文件和逃逸 symlink。
不修改镜像目录，不覆盖共享源码。随后在该目录执行 launcher，清除继承 PYTHONPATH/PYTHONHOME/LD_PRELOAD，
以同一解释器和环境检查真实 package/worker 的导入路径与文件 SHA256，再启动 worker。

## 上海固定布局与离线渲染

| 角色 | 平台 | 卡数/进程 | 实测硬件 gate | 管理端口 |
| --- | --- | --- | --- | --- |
| Encoder | 上海 H100 K3s | 1 / 1 | CC9.0、CUDA 可见显存 ≥75 GiB | 31000 |
| Decoder | 上海 H100 K3s | 1 / 单进程 torchrun | CC9.0、≥75 GiB | 36000 |
| DiT | 上海 H200 PAI | 1 / SP1 | CC9.0、≥130 GiB | 33000 |

以上是容量检查门槛，**不是 480/736 生成或显存验收**。DiT 示例显式选择 FP8；所有模型/schedule 参数由已核验 pipeline 提供。
H200 SP1 全常驻 736 历史有 OOM；当前 480、736、混合节点 NCCL、吞吐/显存及生成效果均未在本版本验收。
通用 `run` 不限定 GPU 型号；上海 `render` 固化表中的硬件约束。

```sh
# /private/release 下的 role/site 文件由准备端从现有平台和资产回执生成；不把 Secret 值写进去。
python3 deploy/sglang/launch.py render --config /private/release/encoder.json \
  --manifest /private/release/source/source.json --site /private/release/h100-site.json \
  --output /private/release/encoder-plan
# Decoder 同上，换 decoder.json 和输出目录；DiT 换 dit.json 与 h200-site.json。
```

H100 site 提供已有 namespace、digest 镜像、`volumes` 和 `volume_mounts`（模型/配置及 ossutil 认证文件的只读挂载）。
H200 site 的 `job_spec`/`data_sources` 从当前已核实的单卡配额规格、数据源回读，保留一 Pod、一 GPU、Never restart；
不得从旧四卡 Whitebox JSON 直接改 GPU 数冒充已验证规格。上海挂载以 `GetJob.DataSources` 为准。
仅路径配置不证明镜像依赖、挂载或跨节点网络已就绪。

## 提交命令（执行会占用 GPU，须另有当次部署授权）

`render` 只输出材料；本 PR 未执行以下提交命令。提交记录和 render 目录保存在共享的持久运行目录；
同一次计划只提交一次，API 超时/UNKNOWN 时按返回身份或唯一名称查询，不自动重试创建。

H100 通过 Eflo 在指定节点内执行脚本，不使用本机 kube context：

```sh
aliyun eflo-controller RunCommand --profile shengdong --region cn-shanghai \
  --body "$(cat /private/release/encoder-plan/eflo.json)" > /private/release/encoder-submit.json
# 从返回 JSON 读取 InvokeId，再查同一个 invocation；不得只以请求受理为成功。
aliyun eflo-controller DescribeInvocations --profile shengdong --region cn-shanghai \
  --InvokeId "$INVOKE_ID" --IncludeOutput true --ContentEncoding PlainText
```

节点脚本持有本工具共享的 flock，读 nvidia-smi 和全 namespace 的 Pod 声明，排除已有进程、显存非零和 Pending Pod 已声明的 UUID；
然后分配一张卡，使用 NVIDIA runtime、hostNetwork、`imagePullPolicy: Never`，先 kubectl dry-run 再创建。
非 UUID 或含糊的既有 GPU 声明会退出，请平台 owner 先核验，不强行抢卡。
该锁不能约束其他工具/管理员；实际创建前仍需遵守现有资源归属和调度协调规则。
Pod IP/Name/UID 用 downward API，随机 Pod 名和 worker UUID 自动生成。

H200 使用 PAI REST POST 入口（沿用现有 CLI 提交流程）：

```sh
PAI_UA="AlibabaCloud-Agent-Skills/alibabacloud-pai-dlc-job/$(uuidgen | tr '[:upper:]' '[:lower:]')"
aliyun pai-dlc POST /api/v1/jobs --profile shengdong --region cn-shanghai \
  --user-agent "$PAI_UA" --header Content-Type=application/json \
  --body "$(cat /private/release/dit-plan/pai.json)" > /private/release/dit-submit.json
# 从返回 JSON 读取 JobId；回读 JobSpecs/DataSources，与已冻结 pai.json 对账。
aliyun pai-dlc GetJob --profile shengdong --region cn-shanghai \
  --user-agent "$PAI_UA" --JobId "$JOB_ID"
```

固定 workspace `1453237`、quota `quota3l8bnv9a35n`；真实可用规格和占用必须重新核验。
PAI 沿用平台分配的可见 GPU，不枚举宿主机“空闲卡”；IP 由 POD_IP 或内核到 H100 地址 `10.0.0.121` 的路由选择取得，
UDP connect 不发送数据。跨节点 HTTP/WS/ZMQ/TCPStore/NCCL 网络仍需平台验收；端点不是 loopback。

## 回执与失败行为

容器输出 `starting`、本角色 `ready`、`exited/failed` JSON，同时保存于解包目录旁 `receipt.json`。
包含 release/boot 身份、commit、源码包和 launcher hash、期望镜像 digest、模型资产回执 hash、角色配置/table 信息、
白名单 argv/env、实际导入来源/hash、Python/torch/CUDA、GPU 信息、IP/HTTP、PID、Pod 名/UID 和可取得的 Job ID。
PAI 未注入 Job ID/Pod UID 时字段为 null；外层必须与 `dit-submit.json` 和 GetJob 的 Job/Pod 记录关联，不能声称身份已完整。
日志由平台保存，外层流水线应收集回执及平台提交回执；临时目录不是永久归档接口。

preflight/摘要/导入/硬件检查失败不启动 worker。已有角色端口被占用时退出，防止读到其他实例的 health。
worker 启动退出、角色不匹配或健康超时使 launcher 失败；SIGINT/SIGTERM 终止本次子进程组，30 秒后强制清理。
父进程退出也清理 torchrun 后代；不发送任何现有服务的 stop/restart，也不申请额外 GPU。
`ready` 仅代表该角色加载就绪；三角色出帧需另跑原生建组与生成协议，不能把 `bench_client.py` 直接连到 DiT 管理端口。

```sh
python3 -m unittest discover -s deploy/sglang -p 'test_*.py'
```

专用 CPU CI 覆盖本目录。GPU、OSS 实际传输、Eflo/PAI 提交和模型生成未在本 PR 运行。


## v1 接口合同与 infra-skills #17

所有公共 JSON 均使用 `schema: 1`。消费者必须拒绝其他 major schema；新增可选回执字段可忽略，
role/site 配置及 source manifest 的未知键会拒绝。改变必填项、语义或命令名需提升 schema 并同步使用者；
当前 v1 是固定上述 Zing H3 模型与上海配方，不能据“通用”二字外推为任意模型/任意云平台支持。

| JSON | 必填字段 | 可选字段／语义 |
| --- | --- | --- |
| role | schema, role, model_id, model, asset_receipt, asset_receipt_sha256 | pipeline（DiT必填）、decoder_config（Decoder必填）、quantization（仅DiT，可选fp8）、environment、startup_timeout、hardware |
| H100 site | schema, image, namespace, volumes, volume_mounts | 现有卷/Secret引用；不接受Secret值 |
| H200 site | schema, image, job_spec, data_sources | job_spec仅Type/PodCount/QuotaId/RestartPolicy/ResourceConfig；固定Worker/1/指定quota/Never |
| source manifest | schema, commit, sha256, uri, tools | uri须包含commit/hash；tools为三个启动工具的文件SHA256 |
| asset receipt | schema, model_id, component, ema, model, files | DiT还需adaln；files为绝对路径到SHA256；owner保留原始export来源证据 |
| render receipt | schema, name, role, source, model_id, status | status仅rendered-not-submitted；输出目录与输入role/source共同持久保存 |
| runtime receipt | schema, release_id, boot_epoch, role_config_sha256, commit, archive_sha256, launcher_sha256, model_id, asset_receipt_sha256, role, worker_id, argv, environment, runtime, gpu, ip, http, pid, pod | 其余资产/平台字段见下文；按状态逐步补齐，不存在的数据为null |

`hardware` 可指定 capability 数组和 minimum_memory_gib；上海 renderer 统一覆盖为固定配方门槛。
role 中 model/asset_receipt 路径须为绝对路径，其余角色资产必须由文件hash清单覆盖；摘要必须为完整小写hex。
环境键白名单见 `ENV_ALLOW`，配置端不能覆盖平台CUDA/NVIDIA分配、PATH/HOME/LD_LIBRARY_PATH；
Secret从现有平台挂载供 bootstrap 的 ossutil 认证使用，不透传给模型worker，也不序列化进回执。

runtime 的 `boot_epoch` 是本次 launcher 的启动身份，`worker_boot_epoch` 目前为null，不能伪装成原生模型池epoch。
`image_requested` 是渲染时指定的digest，`image_actual` 目前为null，必须由外层回读Pod的实际imageID/平台记录后关联；
不能把环境中的期望镜像值当作平台实测。`job`/`pod_uid` 缺失时同样由外层补齐证据，保留原始回执。
`runtime` 是目标解释器实测版本、导入文件path/hash、CUDA capability与显存；`gpu`来自容器内nvidia-smi，包含UUID。
`role_config_sha256` 使用排序键、紧凑分隔符的JSON UTF-8计算，不是原配置文件的字节hash。

`prepare/render` 成功退出0；配置/验证/平台命令失败返回非0，不产生“已部署”状态。
`run` 在健康前失败返回非0，健康后转发worker退出码；signal退出返回128+signal。
回执状态是 starting → ready → exited，或 starting/ready → failed；preflight失败还未创建worker，只有错误退出和日志。
就绪后进程退出、超时、signal只清理本launcher的进程组；Pod/Job删除和回滚不在本工具执行范围。
外层仅可按已保存的精确PodUID/JobId和归属清理本次资源，不通过名称前缀批量清理或触碰共享租约。

[infra-skills #17](https://github.com/seedleap/infra-skills/issues/17) 负责步骤编排，直接调用本目录，不复制归档、选卡或renderer。
world-model-all-in-one负责现场环境、观测和证据回写。开发验证引用已推送PR完整SHA；正式发布在合并后引用main可达完整SHA。

| 范围 | 当前状态 |
| --- | --- |
| prepare/render/run、归档校验、导入核验、原生worker argv、H100选卡、回执和失败退出 | 已实现；CPU合同测试通过 |
| OSS上传 | 没有专用publish命令；外层调用既有ossutil并记录上传结果；容器下载/验hash已实现 |
| 平台提交/查重/UNKNOWN对账、实际imageID与JobPod回读、回执长期归档 | 本PR提供请求文件与准确CLI；没有自动submit/inspect/rollback命令，由外层按授权执行 |
| 模型池注册/签名接线、实际出帧、浏览器业务E2E | 独立验收层；不作为启动器ready含义，不由本PR伪造 |
| 上海1102 serving/table精确路径和model/schedule/CUDA provenance、480/736及SP1显存、OSS/Eflo/PAI实测 | 未验收；正式部署前须关闭对应缺口 |

#17可以先完成skill设计；命令集成需对照本PR最终main SHA，底层缺口反馈本PR，不能在skill里发明不存在的publish/submit/verify命令。
