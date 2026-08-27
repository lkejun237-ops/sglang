# Aliyun Shanghai K3s Realtime Zing Chain

This overlay mirrors the Beijing public chain at
`https://zing-model.loopit.com.cn/` for the Shanghai Lingjun cluster
`i119114381787650669980`.

Target public domain:

```text
https://zing-model-shanghai.loopit.com.cn/
```

DNS for this deployment currently points at EIP `8.133.210.94`. The target
topology keeps that EIP on a small CPU edge ECS for Caddy instead of on a GPU
worker, and uses an ApsaraDB Redis instance for coordinator/auth state.

## Target Shape

- Region: `cn-shanghai`
- Lingjun cluster: `i119114381787650669980`
- Kubernetes: self-managed k3s
- Namespace: `minwm-realtime`
- Model: `wan22-5b-stage3-dmd-47-0808-2fb2cfec2a2`
- Resolution: `1280x704`
- FPS: `24`
- Denoiser: seven single-GPU workers on GPUs `0-6`, each `sp=1`
- Overall denoiser concurrency: `7`
- VAE: one worker on GPU `7`, capacity-limited to `7`
- VAE codec path: TAEHV + direct H.264/fMP4 output, same browser path as Beijing
- Coordinator cache: Redis, not process-local memory. Match the Beijing
  production pattern by using an ApsaraDB Redis instance and injecting the
  password through the `zing-redis` Kubernetes Secret.
- Edge: Caddy remains a temporary Shanghai-only public edge until this chain is
  moved behind ACK/ALB/ASM/Istio like the Beijing online chain.

The Beijing chain currently advertises `832x480`, H.264 WebSocket playback, and
single-Zing mode. Shanghai keeps the same gateway/WebUI protocol and changes the
runtime size to `1280x704`.

## Zing + LingBot Comparison Preflight

The committed control-plane template now gives the shared Gateway the verified
LingBot routing identity:

```text
LINGBOT2_MODEL_REVISION=robbyant/lingbot-world-v2-14b-causal-fast-diffusers
LINGBOT2_VAE_FINGERPRINT=taew2_1-d26151e7
```

This only makes the authenticated LingBot route routable after compatible
workers register with the Coordinator. It does not claim that the current GPU
layout can run both models. The current `zing-denoiser-sp1x7` plus
`zing-vae-a10` layout consumes all eight GPUs (`0-6` plus `7`). Before enabling
the world-service comparison switch, the deployment owner must choose and test
an explicit GPU partition, then add:

- a LingBot Denoiser using the already validated 720P parameters in
  `benchmark/minwm_realtime_async_vae/k8s/lingbot2-h100-denoiser.yaml`;
- a LingBot VAE registered with fingerprint `taew2_1-d26151e7`, using
  `taew2_1.pth` as in `benchmark/minwm_realtime_async_vae/k8s/l4-vae.yaml`;
- an authoritative person-level capacity source. The current Coordinator
  exposes model-session counts only, so it cannot truthfully report one person
  for a Zing + LingBot pair.

Do not guess a GPU split or divide session counts by two. Keep the Admin switch
off until both worker pools are Ready, a real dual WebSocket run produces
`1280x704` Zing and exact `1280x720` LingBot output, and `/v1/slots.used_slots`
increases by exactly one for that pair.

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

The console reports the H100 inventory with the `L20Z` label in this environment.
The current runtime target is still seven single-GPU denoisers (`sp=1`) plus one
VAE on the same 8-GPU Lingjun node.

The local workstation currently cannot reach `10.0.0.121` directly by ping or
SSH, so run the deployment from a VPC-reachable bastion, from Cloud Assistant /
Lingjun RunCommand, or from an ECS instance in the same VPC.

## Historical Low-Cost VAE GPU Pick

The original low-cost VAE plan used separate A10 nodes:

| Count | Type | GPU | Zones with stock checked | Price checked |
| --- | --- | --- | --- | --- |
| 2 | `ecs.gn7i-c8g1.2xlarge` | `1 x NVIDIA A10 24GB` | `cn-shanghai-l`, `cn-shanghai-m` | about `9.8476 CNY/hour/node` with 300G ESSD system disk |

This is no longer the target runtime shape. The live chain now keeps VAE on the
Lingjun/H100 node's GPU `7`; the A10 nodes should be released after Caddy and
Redis are migrated away from them.

The original plan needed extra vSwitches to place A10 nodes where stock existed:

- `10.0.1.0/24` in `cn-shanghai-l`
- `10.0.2.0/24` in `cn-shanghai-m`

For the current target, keep only the existing `cn-shanghai-l` subnet needed by
the temporary Caddy edge until the old A10 host is released.

## Files

- `deploy_control_plane.sh`: installs/reconciles k3s on the denoiser/control
  node, creates secrets, uploads runtime/WebUI source patches, and applies
  manifests.
- `provision_vae_ecs.sh`: historical guarded helper for the old separate A10 VAE
  plan. Defaults to dry-run.
- `provision_managed_redis.sh`: guarded helper for the Shanghai ApsaraDB Redis
  instance. Defaults to dry-run and never prints the Redis password.
- `provision_edge_caddy_ecs.sh`: guarded helper for the small CPU ECS that will
  own Caddy and the public EIP. Defaults to dry-run.
- `install_k3s_vae_agent.sh`: historical helper for joining old VAE ECS nodes.
- `python/sglang/multimodal_gen/apps/realtime_webui/feishu_auth.py`: optional
  Feishu OAuth login gate embedded in the WebUI backend.
- `manifests/control-plane.yaml.tpl`: coordinator, gateway, WebUI, RuntimeClass.
- `manifests/denoiser-sp2.yaml.tpl`: historical SP2 worker template.
- `manifests/vae-workers-a10.yaml.tpl`: current 7+1 worker template; despite the
  filename, the live target runs seven denoisers on GPUs `0-6` and one VAE on
  GPU `7`.

## Suggested Execution Order

Bootstrap the control plane first, without touching GPU worker templates:

```bash
SSH_HOST=root@10.0.0.121 \
PUBLIC_GATEWAY_BASE_URL=https://zing-model-shanghai.loopit.com.cn \
APPLY_GPU_WORKER_TEMPLATES=false \
bash benchmark/minwm_realtime_async_vae/aliyun/k8s-shanghai/deploy_control_plane.sh
```

Create or locate the managed Redis instance first. Dry-run:

```bash
bash benchmark/minwm_realtime_async_vae/aliyun/k8s-shanghai/provision_managed_redis.sh
```

Live creation requires explicit approval and a password supplied out of band:

```bash
DRY_RUN=false \
CONFIRM_ALIYUN_WRITES=create-shanghai-managed-redis \
REDIS_PASSWORD='<redis-password>' \
bash benchmark/minwm_realtime_async_vae/aliyun/k8s-shanghai/provision_managed_redis.sh
```

Create the CPU edge ECS for Caddy. Dry-run:

```bash
bash benchmark/minwm_realtime_async_vae/aliyun/k8s-shanghai/provision_edge_caddy_ecs.sh
```

Live creation:

```bash
DRY_RUN=false \
CONFIRM_ALIYUN_WRITES=create-shanghai-edge-caddy-ecs \
bash benchmark/minwm_realtime_async_vae/aliyun/k8s-shanghai/provision_edge_caddy_ecs.sh
```

Then apply the full runtime with managed Redis:

```bash
SSH_HOST=root@10.0.0.121 \
PUBLIC_GATEWAY_BASE_URL=https://zing-model-shanghai.loopit.com.cn \
REDIS_HOST='<redis-connection-domain>' \
REDIS_USERNAME=default \
REDIS_PASSWORD='<redis-password>' \
DEPLOY_REDIS=false \
bash benchmark/minwm_realtime_async_vae/aliyun/k8s-shanghai/deploy_control_plane.sh
```

## Feishu WebUI Auth

The Shanghai WebUI public domain is protected by optional Feishu OAuth inside
the WebUI backend. Enable it per region with `WEBUI_AUTH_ENABLED=true`; keep it
unset or `false` for regions that should remain unchanged. Sessions are stored
in Redis, not local process memory.

The public Caddy route must send only these machine-to-machine paths directly to
`zing-gateway`; do not use a `/backends/*` wildcard because legacy unauthenticated
Demo routes remain registered inside the cluster:

- `/backends/minwm/v1/realtime_video/authorized_generate`
- `/backends/lingbot2/v1/realtime_video/authorized_generate`
- `/v1/slots`

Create the Kubernetes secret out of band; do not commit these values. The
current Shanghai deployment reuses this secret from the WebUI pod:

```bash
kubectl -n minwm-realtime create secret generic zing-auth-secrets \
  --from-literal=FEISHU_APP_ID='<feishu-app-id>' \
  --from-literal=FEISHU_APP_SECRET='<feishu-app-secret>' \
  --from-literal=AUTH_COOKIE_SECRET='<random-48-byte-url-safe-secret>' \
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
AUTH_REDIS_HOST=<same-redis-host-used-by-coordinator>
AUTH_REDIS_USERNAME=default
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

Historical A10 VAE nodes, if that plan is ever re-enabled:

```text
/data/zing-realtime/taehv/taew2_2.pth
```

## Writes That Need Human Confirmation

Before running live deployment, confirm these exact current mutations:

- Create one ApsaraDB Redis Open-Source Edition instance in
  `cn-shanghai-l`, VPC `vpc-uf6bmsqo2f8jwk6o6me6n`, vSwitch
  `vsw-uf6pd2zo6sbf8pfldxnz3`, class `redis.master.stand.default`,
  capacity `4096` MB, pay-as-you-go, with a password supplied out of band.
- Append Redis default whitelist entry `10.0.0.0/24`, so coordinator/WebUI Pods
  on the Lingjun control node can reach it through VPC private networking.
- Create one pay-as-you-go CPU ECS `zing-shanghai-edge-caddy`, type
  `ecs.e-c1m2.large`, image `aliyun_3_x64_20G_alibase_20260720.vhd`,
  key pair `world-model`, 40G ESSD system disk, no public bandwidth.
- Install/configure Caddy on `zing-shanghai-edge-caddy`, then move EIP
  `8.133.210.94` from the historical A10 host to this CPU edge ECS.
- After Redis and Caddy are verified on their new homes, delete the historical
  A10 k3s node and release the remaining A10 ECS.
- Reconcile k3s/Kubernetes objects on Lingjun node `e01-cn-2zd4xi8m505`.
- Apply Kubernetes objects in namespace `minwm-realtime`.

Rollback is straightforward: point `REDIS_HOST` back to the previous Redis
endpoint and roll `zing-coordinator`/`zing-webui`, move EIP `8.133.210.94` back
to the previous ECS if needed, then delete the new CPU ECS and managed Redis
only after confirming no live sessions depend on them. For the historical A10
nodes, delete the Kubernetes node object, release the ECS instances, delete any
now-empty legacy VAE vSwitch, and remove the internal k3s/security-group rules
if no other cluster uses them.
