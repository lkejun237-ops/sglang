# AI Infra 测试机器与访问方式

最近核验：2026-10-22。供本仓库开发者和 Codex 查找测试资源。
**优先上海 H200 / H100，其次泰国 B300**。明确指定硬件或需要特定 CUDA 架构的测试
按实验要求选机；同条件 baseline 与改动后测试使用相同硬件，不跨机器型号计算性能收益。

## 资源入口

| 优先级 | 测试资源 | 地域与资源标识 | 访问方式 |
| --- | --- | --- | --- |
| 1 | 上海 H200，实际设备名 `NVIDIA L20X` | `cn-shanghai`；PAI workspace `1453237`；ResourceId / QuotaId `quota3l8bnv9a35n` | PAI-DLC 查询作业，再进入该作业的 Web Terminal |
| 1 | 上海 H100，实际设备名 `NVIDIA L20Z` | `cn-shanghai` / `cn-shanghai-b`；灵骏集群 `i119114381787650669980`；节点 `e01-cn-2zd4xi8m505` | `eflo-controller RunCommand` / `DescribeInvocations`；独立 K3s 节点，无需 SSH |
| 2 | 泰国 B300 测试目标，实际设备名 `NVIDIA L20D` | `ap-southeast-7`；PAI GPU workspace `384`；ResourceId / QuotaId `quota1betfyi1ilu`；CPU 辅助 workspace `385` | PAI-DLC；CPFS 数据源 `d-ak7t158lhzaq7kdwp3`、版本 `v1`，挂载 `/cpfs` |

上海 L20X 按用户确认的 H200 使用；保留原始设备名便于核验。
已有真实 CUDA 查询：上海 H100/H200 为 SM90；泰国目标为 CC 10.3。
上海 H100 driver 显存为 81,559 MiB/卡，H200 为 143,771 MiB/卡；
泰国已有 CUDA 可见容量为 274,113 MiB/卡。上述是历史硬件证据，
新测试仍记录实际 GPU UUID、运行时 capability、显存和软件版本；不要只凭商品名选编译架构。
Zing 的配置、显存余量和结果边界另见
[部署 recipe](../docs/docs/sglang-diffusion/zing/deployment_zing_h3.mdx)。

## 登录与前提

- 浏览器入口：[阿里云 SSO](http://auth.ops.loopit.me/aliyun-sso)，选择飞书登录，再进入对应地域的 PAI / 灵骏控制台。
- CLI 使用已授权的 `shengdong` profile。先执行 `aliyun configure list` 确认本机配置；
  该名称是本机约定，其他人的机器需先配置自己的授权身份，不能仅复制 profile 名就获得权限。
- 浏览器退出登录不代表 CLI 凭据失效。不要把 AK/SK、token、kubeconfig 或带鉴权参数的
  Web Terminal 临时 URL 提交到仓库。
- 当前环境没有 DSW。PAI 路径的 CPU 准备和后处理使用零 GPU CPU 作业；
  不为此索要 DSW MCP 地址或 token。提交前分别核验 CPU/GPU 的配额、挂载和规格，
  不把 GPU 作业配置仅改 GPU=0 就当作 CPU 模板。

以下命令面向已安装 Aliyun CLI 及对应插件的终端。示例显式指定 region，
不依赖 profile 的默认地域。执行前可用 `aliyun <product> <action> --help` 核对本机参数。

## 上海 H200：PAI-DLC

先查当前作业，不能把某个旧 Job ID 当作永久机器入口：

```sh
AI_INFRA_SESSION_ID=$(uuidgen | tr '[:upper:]' '[:lower:]')
AI_INFRA_UA="AlibabaCloud-Agent-Skills/alibabacloud-pai-dlc-job/$AI_INFRA_SESSION_ID"
aliyun pai-dlc ListJobs --profile shengdong --region cn-shanghai \
  --WorkspaceId 1453237 --ShowOwn false --PageNumber 1 --PageSize 100 \
  --user-agent "$AI_INFRA_UA" \
  --cli-query '{TotalCount:TotalCount,Jobs:Jobs[].{JobId:JobId,Name:DisplayName,Status:Status,ResourceId:ResourceId}}'
```

按 `ResourceId=quota3l8bnv9a35n` 找到目标作业；需要完整清单时按 `TotalCount`
递增 `PageNumber`。核对归属与用途后，用真实的 Job ID 查询 Pod：

```sh
JOB_ID=REPLACE_WITH_TARGET_JOB_ID
aliyun pai-dlc GetJob --profile shengdong --region cn-shanghai \
  --JobId "$JOB_ID" --user-agent "$AI_INFRA_UA" \
  --cli-query '{WorkspaceId:WorkspaceId,ResourceId:ResourceId,Status:Status,Pods:Pods[].{PodId:PodId,Status:Status},DataSources:DataSources,JobSpecs:JobSpecs}'

# 选 Running 的工作 Pod，不选 aimaster；该操作返回临时访问链接，只在本地使用。
POD_ID=REPLACE_WITH_RUNNING_WORKER_POD_ID
aliyun pai-dlc GetWebTerminal --profile shengdong --region cn-shanghai \
  --JobId "$JOB_ID" --PodId "$POD_ID" --IsShared false --user-agent "$AI_INFRA_UA"
```

也可在已登录的 PAI 控制台按 Job ID 找到该 Pod 的终端。
`GetWebTerminal` 链接有效期很短，失效后重新获取，不把链接写进交接文档。
进入终端后先运行 `nvidia-smi` 查看当前容器可见 GPU 与进程。
PAI 容器视图不等于整机空闲，不能据此直接占用其他卡。
上海已有作业使用 `oss://leap-world-shanghai/`、挂载 `/mnt/oss`；
具体作业的挂载以 `GetJob.DataSources` 为准，不把泰国 CPFS 或 OSS 地址直接套到上海。

## 上海 H100：独立灵骏 K3s 节点

该节点不在上述 PAI 作业清单内；只查 workspace `1453237` 会漏掉它。
主机名 `world-model-b27mys7b-0000`，内网 IP `10.0.0.121`，
运行 containerd / nerdctl 与 K3s。通过下面的节点命令入口检查，无需另找 SSH 密码：

```sh
aliyun eflo-controller RunCommand --profile shengdong --region cn-shanghai \
  --NodeIdList '["e01-cn-2zd4xi8m505"]' \
  --Name ai-infra-readonly-gpu-check --RepeatMode Once --Timeout 30 \
  --ContentEncoding PlainText \
  --CommandContent 'hostname; nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv'

# 将上一步返回的 InvokeId 填入此处；未到终态时继续查询同一个 InvokeId。
INVOKE_ID=REPLACE_WITH_INVOKE_ID
aliyun eflo-controller DescribeInvocations --profile shengdong --region cn-shanghai \
  --InvokeId "$INVOKE_ID" --IncludeOutput true --ContentEncoding PlainText
```

节点历史配置没有 `nvidia.com/gpu` allocatable 资源；不能据 Kubernetes GPU request
为零推断物理 GPU 空闲。测试前同时检查显存与 compute process，按空闲 GPU 的 UUID 隔离。
已有验证使用临时 Pod 的 `runtimeClassName: nvidia` 和精确的 `NVIDIA_VISIBLE_DEVICES`。
既有服务容器只暴露它获配的卡，在容器内修改 `CUDA_VISIBLE_DEVICES` 不能访问未暴露的卡。
不要把历史空闲卡号当作当前可用卡，也不要修改既有服务的 GPU 可见范围。

## 泰国 B300：后备测试入口

沿用上面的 PAI 查询与终端流程，将 region 改为 `ap-southeast-7`、workspace 改为 `384`，
按 ResourceId `quota1betfyi1ilu` 核对资源。CPU 辅助作业还需查 workspace `385`。
CPFS 数据源和挂载见资源表；提交前从目标 workspace 的真实数据源配置复核。
OSS 默认使用 `oss://leap-worldmodel-thailand-single-az/`，先核对实际对象，
不要沿用旧桶名 `leap-worldmodel-thailand`。

## 使用与交接

- 使用前重新查询作业、节点和 GPU 占用，按实际提交账号与任务上下文确认归属；任务名不是归属证明。
- 本页提供访问与发现方式，不新增启动 GPU 作业、停止他人作业或变更服务的授权。
  测试结束只清理本次创建的进程和资源。
- 交接保留 region、workspace、ResourceId、Job/Pod 或 Node ID、代码 SHA、命令与产物路径；
  不依赖某个开发者私有 memory 文件或本机绝对路径作为唯一入口。
- 无权限、无空闲资源或目标硬件不支持时明确报告；切换泰国后重新建立该硬件的 baseline。
