# Aliyun Shanghai K3s Realtime Zing Chain

This overlay mirrors the Beijing public chain at
`https://zing-model.loopit.com.cn/` for the Shanghai Lingjun cluster
`i119114381787650669980`.

Target public domain:

```text
https://zing-model-shanghai.loopit.com.cn/
```

DNS for this deployment points at the Shanghai public gateway ECS
`zing-shanghai-vae-l` EIP `8.133.210.94`.

## Target Shape

- Region: `cn-shanghai`
- Lingjun cluster: `i119114381787650669980`
- Kubernetes: self-managed k3s
- Namespace: `minwm-realtime`
- Model: `wan22-5b-stage3-dmd-47-0808-2fb2cfec2a2`
- Resolution: `1280x704`
- FPS: `24`
- Denoiser: 4 workers, each `sp=2`, `tp=1`, `ulysses=2`, `ring=1`
- Overall denoiser concurrency: `4`
- VAE: 2 low-cost 24 GB GPU workers, each capacity `16`, total VAE capacity `32`
- VAE codec path: TAEHV + direct H.264/fMP4 output, same browser path as Beijing
- Coordinator cache: Redis, not process-local memory. The live Shanghai rollout
  uses an external Redis service on `10.0.1.21:6379` because DockerHub Redis
  image pulls are not reliable from the Lingjun node.

The Beijing chain currently advertises `832x480`, H.264 WebSocket playback, and
single-Zing mode. Shanghai keeps the same gateway/WebUI protocol and changes the
runtime size to `1280x704`.

## Preflight Findings From 2026-08-25

The Lingjun API reports the cluster as:

- Cluster name: `wprld-model`
- Cluster type: `Lite`
- State: `Running`
- VPC: `vpc-uf6bmsqo2f8jwk6o6me6n`
- vSwitch: `vsw-uf6pwtvw6hoboz5hcxg2x`, `cn-shanghai-b`, `10.0.0.0/24`
- Security group: `sg-uf65itprxseo4lm028h9`
- Node: `e01-cn-2zd4xi8m505`, hostname `world-model-b27mys7b-0000`
- Node private IP: `10.0.0.121`
- Reported machine type: `efg2.NHxcn`, `8 x L20Z`, 80 GB each

The last item does not match the requested H100. Do a final `nvidia-smi` on the
node before production traffic. If it is actually L20Z, the manifests still run
SP2 across the 8 GPUs, but H100 latency/capacity expectations should not be
used.

The local workstation currently cannot reach `10.0.0.121` directly by ping or
SSH, so run the deployment from a VPC-reachable bastion, from Cloud Assistant /
Lingjun RunCommand, or from an ECS instance in the same VPC.

## Low-Cost VAE GPU Pick

Recommended VAE nodes:

| Count | Type | GPU | Zones with stock checked | Price checked |
| --- | --- | --- | --- | --- |
| 2 | `ecs.gn7i-c8g1.2xlarge` | `1 x NVIDIA A10 24GB` | `cn-shanghai-l`, `cn-shanghai-m` | about `9.8476 CNY/hour/node` with 300G ESSD system disk |

This is the lowest practical 24 GB single-GPU ECS shape found in Shanghai. The
cheaper `sgn8ia` vGPU sizes below 24 GB are not recommended for VAE because they
fall below the AWS L4-like 24 GB memory target.

The current VPC only has a `cn-shanghai-b` vSwitch. To place A10 nodes where
stock exists, create:

- `10.0.1.0/24` in `cn-shanghai-l`
- `10.0.2.0/24` in `cn-shanghai-m`

## Files

- `deploy_control_plane.sh`: installs/reconciles k3s on the denoiser/control
  node, creates secrets, uploads runtime/WebUI source patches, and applies
  manifests.
- `provision_vae_ecs.sh`: guarded helper for vSwitch/security-group/ECS writes.
  Defaults to dry-run.
- `install_k3s_vae_agent.sh`: run on each VAE ECS node after k3s server token is
  available.
- `python/sglang/multimodal_gen/apps/realtime_webui/feishu_auth.py`: optional
  Feishu OAuth login gate embedded in the WebUI backend.
- `manifests/control-plane.yaml.tpl`: coordinator, gateway, WebUI, RuntimeClass.
- `manifests/denoiser-sp2.yaml.tpl`: one DaemonSet Pod on the Lingjun 8-GPU
  node, with four SP2 denoiser containers pinned to GPU pairs.
- `manifests/vae-workers-a10.yaml.tpl`: two VAE replicas spread one per A10 node.

## Suggested Execution Order

Bootstrap control plane first, without waiting for VAE nodes:

```bash
SSH_HOST=root@10.0.0.121 \
PUBLIC_GATEWAY_BASE_URL=https://zing-model-shanghai.loopit.com.cn \
APPLY_GPU_WORKER_TEMPLATES=false \
bash benchmark/minwm_realtime_async_vae/aliyun/k8s-shanghai/deploy_control_plane.sh
```

Get the k3s server token:

```bash
ssh root@10.0.0.121 'sudo cat /data/k3s/server/node-token'
```

Preview the VAE ECS creation:

```bash
bash benchmark/minwm_realtime_async_vae/aliyun/k8s-shanghai/provision_vae_ecs.sh
```

After explicit approval, create the two VAE nodes:

```bash
DRY_RUN=false \
CONFIRM_ALIYUN_WRITES=create-shanghai-zing-vae-ecs \
bash benchmark/minwm_realtime_async_vae/aliyun/k8s-shanghai/provision_vae_ecs.sh
```

Join each VAE ECS node:

```bash
K3S_TOKEN='<token-from-server>' \
bash benchmark/minwm_realtime_async_vae/aliyun/k8s-shanghai/install_k3s_vae_agent.sh
```

Then apply the full runtime:

```bash
SSH_HOST=root@10.0.0.121 \
PUBLIC_GATEWAY_BASE_URL=https://zing-model-shanghai.loopit.com.cn \
REDIS_URL=redis://10.0.1.21:6379/0 \
DEPLOY_REDIS=false \
bash benchmark/minwm_realtime_async_vae/aliyun/k8s-shanghai/deploy_control_plane.sh
```

## Feishu WebUI Auth

The Shanghai WebUI public domain is protected by optional Feishu OAuth inside
the WebUI backend. Enable it per region with `WEBUI_AUTH_ENABLED=true`; keep it
unset or `false` for regions that should remain unchanged. Sessions are stored
in Redis, not local process memory.

The machine-to-machine gateway API stays public for the expected integration
paths:

- `/v1/models`
- `/v1/slots`
- `/v1/realtime_video/authorized_generate`
- `/backends/<backend>/v1/models`
- `/backends/<backend>/v1/slots`
- `/backends/<backend>/v1/realtime_video/authorized_generate`

Create the Kubernetes secret out of band; do not commit these values. The
current Shanghai deployment reuses this secret from the WebUI pod:

```bash
kubectl -n minwm-realtime create secret generic zing-auth-secrets \
  --from-literal=FEISHU_APP_ID='<feishu-app-id>' \
  --from-literal=FEISHU_APP_SECRET='<feishu-app-secret>' \
  --from-literal=AUTH_COOKIE_SECRET='<random-48-byte-url-safe-secret>' \
  --from-literal=AUTH_REDIS_URL='<redis-url>' \
  --dry-run=client -o yaml | kubectl apply -f -
```

The Feishu app must allow this redirect URI:

```text
https://zing-model-shanghai.loopit.com.cn/auth/callback
```

Optional allowlist controls are supported via secret/env values:

```text
AUTH_ALLOWED_TENANT_KEY=<tenant_key>
AUTH_ALLOWED_EMAIL_DOMAINS=loopit.com.cn
AUTH_ALLOWED_USERS=<open_id_or_user_id_or_union_id_or_email>,...
```

Use region-specific values for separate Beijing and Shanghai login states, for
example:

```text
AUTH_PUBLIC_BASE_URL=https://zing-model-shanghai.loopit.com.cn
AUTH_COOKIE_NAME=zing_webui_session
AUTH_REDIS_URL=<same-redis-url-used-by-coordinator>
AUTH_REDIS_KEY_PREFIX=zing:shanghai:webui-auth
```

The live Caddy entry for the WebUI domain simply proxies to `zing-webui`; the
WebUI backend decides which paths are public and which require login. The
standalone `zing-auth` deployment is no longer used.

## Required Host Paths

On the denoiser/control Lingjun node:

```text
/data/zing-realtime/model-cache/zing/model/_READY
/data/zing-realtime/model-cache/zing/model/model_index.json
/data/zing-realtime/model-cache/zing/model/transformer/
/data/zing-realtime/taehv/taew2_2.pth
```

On each VAE ECS node:

```text
/data/zing-realtime/taehv/taew2_2.pth
```

## Writes That Need Human Confirmation

Before running live deployment, confirm these exact mutations:

- Create two vSwitches in VPC `vpc-uf6bmsqo2f8jwk6o6me6n` if absent:
  `10.0.1.0/24` in `cn-shanghai-l` and `10.0.2.0/24` in `cn-shanghai-m`.
- Add security-group ingress to `sg-uf65itprxseo4lm028h9` from `10.0.0.0/16`
  for TCP `6443`, UDP `8472`, and TCP `10250`.
- Create two pay-as-you-go ECS instances:
  `zing-shanghai-vae-l` and `zing-shanghai-vae-m`,
  type `ecs.gn7i-c8g1.2xlarge`, image
  `ubuntu_24_04_x64_100G_with_gpu_driver_and_cuda_alibase_20260519.vhd`,
  key pair `world-model`, 300G ESSD system disk.
- Install/reconcile k3s on Lingjun node `e01-cn-2zd4xi8m505` and join the two
  ECS nodes as agents.
- Install and start Redis on `zing-shanghai-vae-l` bound to
  `127.0.0.1` and `10.0.1.21:6379`, with AOF enabled, for coordinator state.
- Apply Kubernetes objects in namespace `minwm-realtime`.

Rollback is straightforward: delete the Kubernetes namespace/manifests, stop and
release the two ECS VAE instances, delete the two VAE vSwitches if empty, and
remove the internal k3s/Redis security-group rules if no other cluster uses
them. For Redis, stop/disable `redis-server`, restore
`/etc/redis/redis.conf.zing.bak.*`, and keep or delete the Redis data directory
according to audit requirements.
