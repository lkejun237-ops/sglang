# AI Infra 测试机器与访问方式

最近核验：2026-10-22。供本仓库开发者和 Codex 查找测试资源。
资源用途更新：2026-10-26（用户确认）。**优先上海 H100，其次泰国 B300**。
上海 H200 已部署两个实例，供产品团队做 demo case 使用，不作为默认测试资源。
明确指定硬件或需要特定 CUDA 架构的测试
按实验要求选机；同条件 baseline 与改动后测试使用相同硬件，不跨机器型号计算性能收益。

查询任务前先读[GPU 任务归属记录](../AI_INFRA_TASK_OWNERS.md)，
按实际提交人的数字 `UserId` 区分克君、`dev_api` 及其他账号，业务负责人另行记录。
H100 独立 K3s 节点须查可验证的创建者身份；缺少证据时标为未知。

## 资源入口

| 优先级 | 测试资源 | 地域与资源标识 | 访问方式 |
| --- | --- | --- | --- |
| 1 | 上海 H100，实际设备名 `NVIDIA L20Z` | `cn-shanghai` / `cn-shanghai-b`；灵骏集群 `i119114381787650669980`；节点 `e01-cn-2zd4xi8m505` | `eflo-controller RunCommand` / `DescribeInvocations`；独立 K3s 节点，无需 SSH |
| 2 | 泰国 B300 测试目标，实际设备名 `NVIDIA L20D` | `ap-southeast-7`；PAI GPU workspace `384`；ResourceId / QuotaId `quota1betfyi1ilu`；CPU 辅助 workspace `385` | PAI-DLC；CPFS 数据源 `d-ak7t158lhzaq7kdwp3`、版本 `v1`，挂载 `/cpfs` |
| 不作为默认测试资源 | 上海 H200，实际设备名 `NVIDIA L20X`；两个实例供产品 demo case 使用 | `cn-shanghai`；PAI workspace `1453237`；ResourceId / QuotaId `quota3l8bnv9a35n` | PAI-DLC 查询作业，再进入该作业的 Web Terminal |

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

## 上海 H200：产品 demo 部署（PAI-DLC）

该机器已部署两个实例，供产品团队做 demo case 使用。以下保留访问与核验方式，
不代表可占用测试；测试默认先使用上海 H100，其次泰国 B300。

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

### 从本地执行节点脚本

完整路径是 **本机 Aliyun CLI → Eflo → 节点内 kubectl → 隔离测试 Pod**，
不是本机 kubectl 直连，也不是 PAI-DLC。不要在本机当前的 AWS kube context 上执行下面的 kubectl。
将节点上要执行的命令保存为本地 `h100-node.sh`，通过同一个通道发送：

```sh
H100_SCRIPT=h100-node.sh
aliyun eflo-controller RunCommand --profile shengdong --region cn-shanghai \
  --NodeIdList '["e01-cn-2zd4xi8m505"]' \
  --Name h100-node-script --RepeatMode Once --Timeout 60 \
  --ContentEncoding PlainText --CommandContent "$(cat "$H100_SCRIPT")"
```

使用返回的 `InvokeId` 按上节 `DescribeInvocations` 读回。`RunCommand` 受理成功不等于
脚本执行成功；同时检查 invocation 状态和 `InvokeNodes.InvokeNode[].ExitCode/Output`。
脚本与输出有大小限制，大文件和长日志应留在节点或产物存储，分段读取。
宿主机历史 Python 为 3.6，调用 subprocess 时使用 `universal_newlines=True`，
不使用 Python 3.7 才支持的 `text=True`；PyTorch 测试在镜像内执行。

### 在空闲卡上创建隔离测试 Pod

仅在当前任务已授权 GPU 测试时执行。先通过上述节点脚本查询：

```sh
# 这些命令在节点内执行；只读发现 namespace、runtime 和现有镜像。
kubectl get node world-model-b27mys7b-0000
kubectl get runtimeclass nvidia
kubectl -n minwm-realtime get pods \
  -o custom-columns='NAME:.metadata.name,STATUS:.status.phase,IMAGE:.spec.containers[*].image'
crictl images
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

选择已获准且显存为零、没有 compute process 的 GPU UUID，并与其他测试使用者协调，
避免同时选中同一张卡。该节点没有设备插件的 GPU 分配保护，检查与创建之间也不是原子操作。
从节点已有镜像中选择带 PyTorch/CUDA 的固定 digest，不把历史镜像当作永久可拉取入口。
下面是 `h100-node.sh` 的单卡最小示例，替换 UUID 和缓存镜像后发送；
`imagePullPolicy: Never` 使缓存缺失明确失败，不隐式拉取镜像。

```sh
set -eu
GPU_UUID=REPLACE_WITH_IDLE_GPU_UUID
IMAGE=REPLACE_WITH_CACHED_IMAGE_AT_SHA256
POD_NAME=h100-torch-smoke-$(date +%s)

# 创建前再次确认目标卡；不只看 utilization=0。
test "$(nvidia-smi --id="$GPU_UUID" --query-gpu=memory.used --format=csv,noheader,nounits)" -eq 0
COMPUTE_UUIDS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
if printf '%s\n' "$COMPUTE_UUIDS" | grep -Fqx "$GPU_UUID"; then
  echo 'Selected GPU has a compute process' >&2
  exit 1
fi
cat > "/tmp/$POD_NAME.yaml" <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: $POD_NAME
  namespace: minwm-realtime
spec:
  nodeName: world-model-b27mys7b-0000
  runtimeClassName: nvidia
  restartPolicy: Never
  activeDeadlineSeconds: 90
  containers:
    - name: smoke
      image: $IMAGE
      imagePullPolicy: Never
      env:
        - name: NVIDIA_VISIBLE_DEVICES
          value: "$GPU_UUID"
        - name: NVIDIA_DRIVER_CAPABILITIES
          value: compute,utility
        - name: OMP_NUM_THREADS
          value: "1"
      resources:
        requests: {cpu: "1", memory: 1Gi}
        limits: {cpu: "2", memory: 4Gi}
      command: [python3, -u, -c]
      args:
        - |
          import torch
          assert torch.cuda.is_available() and torch.cuda.device_count() == 1
          print(torch.__version__, torch.version.cuda, torch.cuda.get_device_properties(0), flush=True)
          a = torch.arange(256 * 256, dtype=torch.float32).reshape(256, 256) / 65536
          x = a.cuda()
          torch.testing.assert_close((x @ x.T).cpu(), a @ a.T, rtol=1e-4, atol=1e-4)
          y = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
          assert torch.isfinite(y @ y.T).all().item()
          torch.cuda.synchronize()
          print('TORCH_SMOKE_PASS', flush=True)
EOF
kubectl create --dry-run=client -f "/tmp/$POD_NAME.yaml"
kubectl create -f "/tmp/$POD_NAME.yaml"
echo "POD_NAME=$POD_NAME GPU_UUID=$GPU_UUID"
```

将返回的 Pod 名和 GPU UUID 保存到本次交接。后续仍通过 Eflo 发送节点脚本，读取
`kubectl -n minwm-realtime get pod "$POD_NAME" -o json` 和
`kubectl -n minwm-realtime logs "$POD_NAME"`。只有 Pod 为 `Succeeded`、
容器退出码为 0 且日志有 `TORCH_SMOKE_PASS` 才算通过；Pending/Running 不是通过。
失败时读取该 Pod 的 `describe` 和日志，不反复创建新 Pod。
取回日志后仅删除本次 Pod（`kubectl -n minwm-realtime delete pod "$POD_NAME"`）及
`/tmp/$POD_NAME.yaml`，再核对目标 UUID 显存/进程释放和既有服务状态。
90 秒 deadline 是兜底，不代替失败后的清理。

上述访问流程来源于 2026-10-16 的四卡测试：先在既有 `denoiser-0` 容器中设置
`CUDA_VISIBLE_DEVICES`，得到 CUDA 不可用；改用独立 Pod、`runtimeClassName: nvidia`
与 `NVIDIA_VISIBLE_DEVICES` 后，四卡 FP32 CPU 对齐和 FP16 matmul 全部通过，
Pod 清理后显存释放。该历史运行使用 PyTorch `2.11.0+cu130` / CUDA `13.0`，
实测 CC 9.0，纠正了早期 `nvidia-smi` 的 CC 8.9 读数。
本节将其收敛为单卡复现示例；该示例本次只做静态检查，不冒称重新跑过 GPU。

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
