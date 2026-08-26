#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
SSH_HOST="${SSH_HOST:?set SSH_HOST=root@<shanghai-k3s-control-host-or-bastion-target>}"
REMOTE_DIR="${REMOTE_DIR:-/root/zing-k8s-shanghai}"
NAMESPACE="${NAMESPACE:-minwm-realtime}"
PUBLIC_WEB_HOST="${PUBLIC_WEB_HOST:-${SSH_HOST#*@}}"
PUBLIC_GATEWAY_PORT="${PUBLIC_GATEWAY_PORT:-18080}"
PUBLIC_GATEWAY_BASE_URL="${PUBLIC_GATEWAY_BASE_URL:-https://zing-model-shanghai.loopit.com.cn}"
ACR_INSTANCE_ID="${ACR_INSTANCE_ID:-cri-ghpj9pt8jwhxdk0e}"
ACR_REGION="${ACR_REGION:-cn-beijing}"
ACR_REGISTRY="${ACR_REGISTRY:-loopit-registry-bj-registry.cn-beijing.cr.aliyuncs.com}"
ACR_REPOSITORY="${ACR_REPOSITORY:-minwm/sglang-minwm-realtime}"
GPU_RUNTIME_IMAGE="${GPU_RUNTIME_IMAGE:-${RUNTIME_IMAGE:-${ACR_REGISTRY}/${ACR_REPOSITORY}:minwm-r3-20260820-amd64}}"
CONTROL_IMAGE="${CONTROL_IMAGE:-${ACR_REGISTRY}/${ACR_REPOSITORY}:gateway-c7ae70a65a2d}"
WEBUI_IMAGE="${WEBUI_IMAGE:-${ACR_REGISTRY}/${ACR_REPOSITORY}:webui-c7ae70a65a2d}"
AUTH_IMAGE="${AUTH_IMAGE:-${CONTROL_IMAGE}}"
REDIS_IMAGE="${REDIS_IMAGE:-redis:7.2-alpine}"
REDIS_URL="${REDIS_URL:-redis://zing-redis:6379/0}"
REDIS_KEY_PREFIX="${REDIS_KEY_PREFIX:-zing:shanghai:coordinator}"
DEPLOY_REDIS="${DEPLOY_REDIS:-true}"
WORLD_STUDIO_WEBUI_BASE_URL="${WORLD_STUDIO_WEBUI_BASE_URL:-http://zing-webui:18080}"
NVIDIA_DEVICE_PLUGIN_IMAGE="${NVIDIA_DEVICE_PLUGIN_IMAGE:-nvcr.io/nvidia/k8s-device-plugin:v0.17.1}"
REALTIME_TARGET_FPS="${REALTIME_TARGET_FPS:-24}"
REALTIME_SIZE="${REALTIME_SIZE:-1280x704}"
H264_LIVE_EDGE_TARGET_MS="${H264_LIVE_EDGE_TARGET_MS:-1000}"
H264_LIVE_EDGE_SEEK_THRESHOLD_MS="${H264_LIVE_EDGE_SEEK_THRESHOLD_MS:-1800}"
UI_CONFIG_JSON="${UI_CONFIG_JSON:-{\"generationModes\":[\"i2v\"],\"defaultGenerationMode\":\"i2v\",\"modelSlots\":[\"minwm\"],\"lockModelSlots\":true,\"size\":\"${REALTIME_SIZE}\",\"targetFps\":${REALTIME_TARGET_FPS},\"sessionMaxLifetimeSeconds\":70,\"playbackAckEnabled\":false,\"h264WebSocketEnabled\":true,\"h264DirectGatewayEnabled\":true,\"h264WebSocketBaseUrl\":\"${PUBLIC_GATEWAY_BASE_URL}\",\"h264CompressedBitrateKbps\":3000,\"h264CompressedCrf\":20,\"h264CompressedPreset\":\"fast\",\"h264CompressedGopSeconds\":2,\"h264CompressedVbvBufferMs\":250,\"h264WebSocketLiveEdgeTargetMs\":${H264_LIVE_EDGE_TARGET_MS},\"h264WebSocketSeekThresholdMs\":${H264_LIVE_EDGE_SEEK_THRESHOLD_MS},\"singleExperience\":true,\"tabScopedUserIds\":true,\"smoothCatchupRateMax\":1.1,\"dualModels\":{\"minwm\":{\"label\":\"Zing\",\"size\":\"${REALTIME_SIZE}\",\"targetFps\":${REALTIME_TARGET_FPS},\"sinkSize\":8,\"windowFrames\":32,\"continuous\":true,\"h264StartupDropFrames\":0}}}}"
DENOISER_CAPACITY_LIMIT="${DENOISER_CAPACITY_LIMIT:-7}"
VAE_CAPACITY_LIMIT="${VAE_CAPACITY_LIMIT:-7}"
VAE_WORKER_CAPACITY="${VAE_WORKER_CAPACITY:-16}"
CONTROL_NODE_LABEL_KEY="${CONTROL_NODE_LABEL_KEY:-seedleap.ai/zing-control}"
CONTROL_NODE_LABEL_VALUE="${CONTROL_NODE_LABEL_VALUE:-true}"
CONTROL_PLANE_NODE_LABEL_KEY="${CONTROL_PLANE_NODE_LABEL_KEY:-${CONTROL_NODE_LABEL_KEY}}"
CONTROL_PLANE_NODE_LABEL_VALUE="${CONTROL_PLANE_NODE_LABEL_VALUE:-${CONTROL_NODE_LABEL_VALUE}}"
COORDINATOR_REPLICAS="${COORDINATOR_REPLICAS:-2}"
GATEWAY_REPLICAS="${GATEWAY_REPLICAS:-2}"
AUTH_REPLICAS="${AUTH_REPLICAS:-2}"
AUTH_PUBLIC_BASE_URL="${AUTH_PUBLIC_BASE_URL:-https://zing-model-shanghai.loopit.com.cn}"
AUTH_COOKIE_NAME="${AUTH_COOKIE_NAME:-zing_webui_session}"
AUTH_REDIS_KEY_PREFIX="${AUTH_REDIS_KEY_PREFIX:-zing:shanghai:webui-auth}"
AUTH_SESSION_TTL_SECONDS="${AUTH_SESSION_TTL_SECONDS:-86400}"
DENOISER_GPU_POOL="${DENOISER_GPU_POOL:-aliyun-shanghai-denoiser}"
VAE_GPU_POOL="${VAE_GPU_POOL:-${DENOISER_GPU_POOL}}"
VAE_NVIDIA_VISIBLE_DEVICES="${VAE_NVIDIA_VISIBLE_DEVICES:-7}"
WEBUI_PROMPT_REWRITER_PATH="${WEBUI_PROMPT_REWRITER_PATH:-/data/zing-realtime/secrets/prompt-rewriter-vertex.json}"
WEBUI_DOUBAO_ARK_PATH="${WEBUI_DOUBAO_ARK_PATH:-/data/zing-realtime/secrets/doubao-ark.json}"
WEBUI_WORLD_IMAGE_CONFIG_PATH="${WEBUI_WORLD_IMAGE_CONFIG_PATH:-/data/zing-realtime/secrets/world-image-model-config.json}"
VIDEO_PROMPT_REWRITE_PROVIDER="${VIDEO_PROMPT_REWRITE_PROVIDER:-doubao}"
VIDEO_PROMPT_REWRITE_TIMEOUT_SECONDS="${VIDEO_PROMPT_REWRITE_TIMEOUT_SECONDS:-60}"
CREATE_WORLD_DESCRIPTION_PROVIDER="${CREATE_WORLD_DESCRIPTION_PROVIDER:-${VIDEO_PROMPT_REWRITE_PROVIDER}}"
CREATE_WORLD_IMAGE_CONFIG="${CREATE_WORLD_IMAGE_CONFIG:-/run/secrets/realtime-webui/world-image-model-config.json}"
DOUBAO_ARK_CREDENTIALS="${DOUBAO_ARK_CREDENTIALS:-/run/secrets/realtime-webui/doubao-ark.json}"
DOUBAO_REQUEST_TIMEOUT_SECONDS="${DOUBAO_REQUEST_TIMEOUT_SECONDS:-60}"
WEBUI_NO_PROXY="${WEBUI_NO_PROXY:-localhost,127.0.0.1,::1,.svc,.svc.cluster.local,.cluster.local,zing-gateway,zing-webui,world-realtime-gateway,world-studio-webui,ark.cn-beijing.volces.com,.volces.com,.cognitiveservices.azure.com}"
K3S_PAUSE_IMAGE="${K3S_PAUSE_IMAGE:-registry.cn-hangzhou.aliyuncs.com/google_containers/pause:3.10}"
K3S_DATA_DIR="${K3S_DATA_DIR:-/data/k3s}"
KUBELET_ROOT_DIR="${KUBELET_ROOT_DIR:-/data/kubelet}"
K3S_DISABLE_LOCAL_STORAGE="${K3S_DISABLE_LOCAL_STORAGE:-true}"
K3S_NODE_IP="${K3S_NODE_IP:-}"
K3S_FLANNEL_IFACE="${K3S_FLANNEL_IFACE:-}"
K3S_HTTP_PROXY="${K3S_HTTP_PROXY:-}"
K3S_HTTPS_PROXY="${K3S_HTTPS_PROXY:-${K3S_HTTP_PROXY}}"
K3S_NO_PROXY="${K3S_NO_PROXY:-localhost,127.0.0.1,::1,10.0.0.0/16,10.42.0.0/16,10.43.0.0/16,.svc,.svc.cluster.local,.cluster.local}"
K3S_COREDNS_IMAGE="${K3S_COREDNS_IMAGE:-registry.cn-hangzhou.aliyuncs.com/rancher/mirrored-coredns-coredns:1.14.6}"
K3S_METRICS_SERVER_IMAGE="${K3S_METRICS_SERVER_IMAGE:-registry.cn-hangzhou.aliyuncs.com/google_containers/metrics-server:v0.9.0}"
K3S_LOCAL_PATH_PROVISIONER_IMAGE="${K3S_LOCAL_PATH_PROVISIONER_IMAGE:-registry.cn-hangzhou.aliyuncs.com/rancher/local-path-provisioner:v0.0.36}"
K3S_BUSYBOX_IMAGE="${K3S_BUSYBOX_IMAGE:-registry.cn-hangzhou.aliyuncs.com/rancher/mirrored-library-busybox:1.37.0}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-1800s}"
APPLY_NVIDIA_DEVICE_PLUGIN="${APPLY_NVIDIA_DEVICE_PLUGIN:-false}"
APPLY_GPU_WORKER_TEMPLATES="${APPLY_GPU_WORKER_TEMPLATES:-true}"
APPLY_RUNTIME_SOURCE_PATCH="${APPLY_RUNTIME_SOURCE_PATCH:-true}"
APPLY_AUTH_SOURCE_PATCH="${APPLY_AUTH_SOURCE_PATCH:-true}"
APPLY_WEBUI_STATIC_PATCH="${APPLY_WEBUI_STATIC_PATCH:-true}"
APPLY_ACR_SECRET="${APPLY_ACR_SECRET:-true}"
WEBUI_STATIC_PATCH_NAME="${WEBUI_STATIC_PATCH_NAME:-zing-webui-static-patch}"
WEBUI_ASSET_PATCH_NAME="${WEBUI_ASSET_PATCH_NAME:-zing-webui-preset-assets-patch}"
WEBUI_STATIC_PATCH_VERSION="${WEBUI_STATIC_PATCH_VERSION:-single-zing-v10-shanghai-redis-20260825}"
WEBUI_STATIC_SOURCE_DIR="${WEBUI_STATIC_SOURCE_DIR:-${REPO_ROOT}/python/sglang/multimodal_gen/apps/realtime_webui}"
RUNTIME_SOURCE_PATCH_VERSION="${RUNTIME_SOURCE_PATCH_VERSION:-runtime-playbackack-mainali-dd4a65a8730b-20260826}"
AUTH_SOURCE_PATCH_VERSION="${AUTH_SOURCE_PATCH_VERSION:-auth-feishu-shanghai-20260826-3}"
AUTH_SOURCE_FILE="${AUTH_SOURCE_FILE:-${SCRIPT_DIR}/auth/feishu_auth_server.py}"
RUNTIME_REALTIME_SOURCE_DIR="${RUNTIME_REALTIME_SOURCE_DIR:-${REPO_ROOT}/python/sglang/multimodal_gen/runtime/realtime}"
RUNTIME_REALTIME_STATES_SOURCE_DIR="${RUNTIME_REALTIME_STATES_SOURCE_DIR:-${RUNTIME_REALTIME_SOURCE_DIR}/states}"
RUNTIME_ENTRYPOINT_SOURCE_DIR="${RUNTIME_ENTRYPOINT_SOURCE_DIR:-${REPO_ROOT}/python/sglang/multimodal_gen/runtime/entrypoints}"
RUNTIME_OPENAI_REALTIME_ENTRYPOINT_SOURCE_DIR="${RUNTIME_OPENAI_REALTIME_ENTRYPOINT_SOURCE_DIR:-${RUNTIME_ENTRYPOINT_SOURCE_DIR}/openai/realtime}"
RUNTIME_UTILS_SOURCE_DIR="${RUNTIME_UTILS_SOURCE_DIR:-${REPO_ROOT}/python/sglang/multimodal_gen/runtime/utils}"
RUNTIME_MODELS_DITS_SOURCE_DIR="${RUNTIME_MODELS_DITS_SOURCE_DIR:-${REPO_ROOT}/python/sglang/multimodal_gen/runtime/models/dits}"
RUNTIME_PIPELINES_SOURCE_DIR="${RUNTIME_PIPELINES_SOURCE_DIR:-${REPO_ROOT}/python/sglang/multimodal_gen/runtime/pipelines}"
RUNTIME_PIPELINE_MINWM_STAGE_SOURCE_DIR="${RUNTIME_PIPELINE_MINWM_STAGE_SOURCE_DIR:-${REPO_ROOT}/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minwm}"
RUNTIME_MANAGERS_SOURCE_DIR="${RUNTIME_MANAGERS_SOURCE_DIR:-${REPO_ROOT}/python/sglang/multimodal_gen/runtime/managers}"
RUNTIME_CONFIGS_SOURCE_DIR="${RUNTIME_CONFIGS_SOURCE_DIR:-${REPO_ROOT}/python/sglang/multimodal_gen/configs}"
RUNTIME_PIPELINE_CONFIGS_SOURCE_DIR="${RUNTIME_PIPELINE_CONFIGS_SOURCE_DIR:-${RUNTIME_CONFIGS_SOURCE_DIR}/pipeline_configs}"
RUNTIME_SAMPLE_CONFIGS_SOURCE_DIR="${RUNTIME_SAMPLE_CONFIGS_SOURCE_DIR:-${RUNTIME_CONFIGS_SOURCE_DIR}/sample}"
JIT_DIFFUSION_SOURCE_DIR="${JIT_DIFFUSION_SOURCE_DIR:-${REPO_ROOT}/python/sglang/jit_kernel/diffusion}"
JIT_CSRC_DIFFUSION_SOURCE_DIR="${JIT_CSRC_DIFFUSION_SOURCE_DIR:-${REPO_ROOT}/python/sglang/jit_kernel/csrc/diffusion}"
MINWM_PROFILE_LAUNCHER_SOURCE="${MINWM_PROFILE_LAUNCHER_SOURCE:-${REPO_ROOT}/python/sglang/multimodal_gen/tools/minwm_profile_launcher.py}"
RUNTIME_SHIMS_SOURCE_DIR="${RUNTIME_SHIMS_SOURCE_DIR:-${SCRIPT_DIR}/runtime-shims}"
H264_FFMPEG_BIN="${H264_FFMPEG_BIN:-/usr/local/lib/python3.12/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2}"
SSH_COMMON_ARGS=()
if [[ -n "${SSH_CONFIG_FILE:-}" ]]; then
  SSH_COMMON_ARGS+=(-F "${SSH_CONFIG_FILE}")
fi
if [[ -n "${SSH_PROXY_COMMAND:-}" ]]; then
  SSH_COMMON_ARGS+=(-o "ProxyCommand=${SSH_PROXY_COMMAND}")
elif [[ -n "${SSH_PROXY_JUMP:-}" ]]; then
  SSH_COMMON_ARGS+=(-J "${SSH_PROXY_JUMP}")
fi
if [[ -n "${SSH_KNOWN_HOSTS_FILE:-}" ]]; then
  SSH_COMMON_ARGS+=(-o "UserKnownHostsFile=${SSH_KNOWN_HOSTS_FILE}")
fi
if [[ -n "${SSH_GLOBAL_KNOWN_HOSTS_FILE:-}" ]]; then
  SSH_COMMON_ARGS+=(-o "GlobalKnownHostsFile=${SSH_GLOBAL_KNOWN_HOSTS_FILE}")
fi
if [[ "${SSH_STRICT_HOST_KEY_CHECKING:-accept-new}" != "" ]]; then
  SSH_COMMON_ARGS+=(-o "StrictHostKeyChecking=${SSH_STRICT_HOST_KEY_CHECKING}")
fi

ssh_remote() {
  command ssh -o BatchMode=yes "${SSH_COMMON_ARGS[@]}" "$@"
}

scp_remote() {
  command scp -q "${SSH_COMMON_ARGS[@]}" "$@"
}

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

render_template() {
  local template="$1"
  local output="$2"
  NAMESPACE="${NAMESPACE}" \
  CONTROL_IMAGE="${CONTROL_IMAGE}" \
  AUTH_IMAGE="${AUTH_IMAGE}" \
  GPU_RUNTIME_IMAGE="${GPU_RUNTIME_IMAGE}" \
  WEBUI_IMAGE="${WEBUI_IMAGE}" \
  REDIS_IMAGE="${REDIS_IMAGE}" \
  REDIS_URL="${REDIS_URL}" \
  REDIS_KEY_PREFIX="${REDIS_KEY_PREFIX}" \
  WORLD_STUDIO_WEBUI_BASE_URL="${WORLD_STUDIO_WEBUI_BASE_URL}" \
  NVIDIA_DEVICE_PLUGIN_IMAGE="${NVIDIA_DEVICE_PLUGIN_IMAGE}" \
  REALTIME_TARGET_FPS="${REALTIME_TARGET_FPS}" \
  REALTIME_SIZE="${REALTIME_SIZE}" \
  DENOISER_CAPACITY_LIMIT="${DENOISER_CAPACITY_LIMIT}" \
  VAE_CAPACITY_LIMIT="${VAE_CAPACITY_LIMIT}" \
  VAE_WORKER_CAPACITY="${VAE_WORKER_CAPACITY}" \
  CONTROL_NODE_LABEL_KEY="${CONTROL_NODE_LABEL_KEY}" \
  CONTROL_NODE_LABEL_VALUE="${CONTROL_NODE_LABEL_VALUE}" \
  CONTROL_PLANE_NODE_LABEL_KEY="${CONTROL_PLANE_NODE_LABEL_KEY}" \
  CONTROL_PLANE_NODE_LABEL_VALUE="${CONTROL_PLANE_NODE_LABEL_VALUE}" \
  COORDINATOR_REPLICAS="${COORDINATOR_REPLICAS}" \
  GATEWAY_REPLICAS="${GATEWAY_REPLICAS}" \
  AUTH_REPLICAS="${AUTH_REPLICAS}" \
  AUTH_PUBLIC_BASE_URL="${AUTH_PUBLIC_BASE_URL}" \
  AUTH_COOKIE_NAME="${AUTH_COOKIE_NAME}" \
  AUTH_REDIS_KEY_PREFIX="${AUTH_REDIS_KEY_PREFIX}" \
  AUTH_SESSION_TTL_SECONDS="${AUTH_SESSION_TTL_SECONDS}" \
  DENOISER_GPU_POOL="${DENOISER_GPU_POOL}" \
  VAE_GPU_POOL="${VAE_GPU_POOL}" \
  VIDEO_PROMPT_REWRITE_PROVIDER="${VIDEO_PROMPT_REWRITE_PROVIDER}" \
  VIDEO_PROMPT_REWRITE_TIMEOUT_SECONDS="${VIDEO_PROMPT_REWRITE_TIMEOUT_SECONDS}" \
  CREATE_WORLD_DESCRIPTION_PROVIDER="${CREATE_WORLD_DESCRIPTION_PROVIDER}" \
  CREATE_WORLD_IMAGE_CONFIG="${CREATE_WORLD_IMAGE_CONFIG}" \
  DOUBAO_ARK_CREDENTIALS="${DOUBAO_ARK_CREDENTIALS}" \
  DOUBAO_REQUEST_TIMEOUT_SECONDS="${DOUBAO_REQUEST_TIMEOUT_SECONDS}" \
  WEBUI_NO_PROXY="${WEBUI_NO_PROXY}" \
  UI_CONFIG_JSON="${UI_CONFIG_JSON}" \
  H264_FFMPEG_BIN="${H264_FFMPEG_BIN}" \
  RUNTIME_SOURCE_PATCH_VERSION="${RUNTIME_SOURCE_PATCH_VERSION}" \
  AUTH_SOURCE_PATCH_VERSION="${AUTH_SOURCE_PATCH_VERSION}" \
  VAE_NVIDIA_VISIBLE_DEVICES="${VAE_NVIDIA_VISIBLE_DEVICES}" \
  envsubst '${NAMESPACE} ${CONTROL_IMAGE} ${AUTH_IMAGE} ${GPU_RUNTIME_IMAGE} ${WEBUI_IMAGE} ${REDIS_IMAGE} ${REDIS_URL} ${REDIS_KEY_PREFIX} ${WORLD_STUDIO_WEBUI_BASE_URL} ${NVIDIA_DEVICE_PLUGIN_IMAGE} ${REALTIME_TARGET_FPS} ${REALTIME_SIZE} ${DENOISER_CAPACITY_LIMIT} ${VAE_CAPACITY_LIMIT} ${VAE_WORKER_CAPACITY} ${CONTROL_NODE_LABEL_KEY} ${CONTROL_NODE_LABEL_VALUE} ${CONTROL_PLANE_NODE_LABEL_KEY} ${CONTROL_PLANE_NODE_LABEL_VALUE} ${COORDINATOR_REPLICAS} ${GATEWAY_REPLICAS} ${AUTH_REPLICAS} ${AUTH_PUBLIC_BASE_URL} ${AUTH_COOKIE_NAME} ${AUTH_REDIS_KEY_PREFIX} ${AUTH_SESSION_TTL_SECONDS} ${DENOISER_GPU_POOL} ${VAE_GPU_POOL} ${VAE_NVIDIA_VISIBLE_DEVICES} ${VIDEO_PROMPT_REWRITE_PROVIDER} ${VIDEO_PROMPT_REWRITE_TIMEOUT_SECONDS} ${CREATE_WORLD_DESCRIPTION_PROVIDER} ${CREATE_WORLD_IMAGE_CONFIG} ${DOUBAO_ARK_CREDENTIALS} ${DOUBAO_REQUEST_TIMEOUT_SECONDS} ${WEBUI_NO_PROXY} ${UI_CONFIG_JSON} ${H264_FFMPEG_BIN} ${RUNTIME_SOURCE_PATCH_VERSION} ${AUTH_SOURCE_PATCH_VERSION}' \
    <"${template}" >"${output}"
}

filter_redis_documents() {
  local manifest="$1"
  if [[ "${DEPLOY_REDIS}" == "true" ]]; then
    return
  fi
  python3 - "${manifest}" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
docs = path.read_text().split("\n---\n")
kept = []
for doc in docs:
    is_redis_doc = (
        re.search(r"(?m)^kind:\s+(Service|Deployment)\s*$", doc)
        and re.search(r"(?m)^\s+name:\s+zing-redis\s*$", doc)
    )
    if not is_redis_doc:
        kept.append(doc)
path.write_text("\n---\n".join(kept))
PY
}

install_k3s() {
  log "ensuring k3s is installed on ${SSH_HOST}"
  ssh_remote "${SSH_HOST}" "K3S_PAUSE_IMAGE='${K3S_PAUSE_IMAGE}' K3S_DATA_DIR='${K3S_DATA_DIR}' KUBELET_ROOT_DIR='${KUBELET_ROOT_DIR}' K3S_DISABLE_LOCAL_STORAGE='${K3S_DISABLE_LOCAL_STORAGE}' K3S_NODE_IP='${K3S_NODE_IP}' K3S_FLANNEL_IFACE='${K3S_FLANNEL_IFACE}' K3S_HTTP_PROXY='${K3S_HTTP_PROXY}' K3S_HTTPS_PROXY='${K3S_HTTPS_PROXY}' K3S_NO_PROXY='${K3S_NO_PROXY}' bash -s" <<'REMOTE'
set -Eeuo pipefail
mkdir -p "${K3S_DATA_DIR}" "${KUBELET_ROOT_DIR}"
if ! command -v k3s >/dev/null 2>&1 && [[ -x /root/k3s ]]; then
  install -m 0755 /root/k3s /usr/local/bin/k3s
fi
if [[ -n "${K3S_HTTP_PROXY}" ]]; then
  export HTTP_PROXY="${K3S_HTTP_PROXY}" http_proxy="${K3S_HTTP_PROXY}"
fi
if [[ -n "${K3S_HTTPS_PROXY}" ]]; then
  export HTTPS_PROXY="${K3S_HTTPS_PROXY}" https_proxy="${K3S_HTTPS_PROXY}"
fi
export NO_PROXY="${K3S_NO_PROXY}" no_proxy="${K3S_NO_PROXY}"
disable_local_storage_arg=""
if [[ "${K3S_DISABLE_LOCAL_STORAGE}" == "true" ]]; then
  disable_local_storage_arg=" --disable=local-storage"
fi
node_ip_arg=""
if [[ -n "${K3S_NODE_IP:-}" ]]; then
  node_ip_arg=" --node-ip ${K3S_NODE_IP} --advertise-address ${K3S_NODE_IP}"
fi
flannel_iface_arg=""
if [[ -n "${K3S_FLANNEL_IFACE:-}" ]]; then
  flannel_iface_arg=" --flannel-iface ${K3S_FLANNEL_IFACE}"
fi
k3s_exec="server --write-kubeconfig-mode 644 --disable=traefik${disable_local_storage_arg} --pause-image ${K3S_PAUSE_IMAGE} --data-dir ${K3S_DATA_DIR} --kubelet-arg root-dir=${KUBELET_ROOT_DIR}${node_ip_arg}${flannel_iface_arg}"
if command -v k3s >/dev/null 2>&1 && ! systemctl list-unit-files k3s.service >/dev/null 2>&1 && [[ -f /root/k3s-offline-install.sh ]]; then
  INSTALL_K3S_SKIP_DOWNLOAD=true \
    INSTALL_K3S_EXEC="${k3s_exec}" \
    sh /root/k3s-offline-install.sh
elif ! command -v k3s >/dev/null 2>&1; then
  curl -sfL https://rancher-mirror.rancher.cn/k3s/k3s-install.sh \
    | INSTALL_K3S_MIRROR=cn \
      INSTALL_K3S_EXEC="${k3s_exec}" \
      sh -
fi
mkdir -p /etc/systemd/system/k3s.service.d
tmp_unit="$(mktemp)"
{
  printf '%s\n' '[Service]'
  [[ -n "${K3S_HTTP_PROXY}" ]] && printf 'Environment=%q\n' "HTTP_PROXY=${K3S_HTTP_PROXY}"
  [[ -n "${K3S_HTTP_PROXY}" ]] && printf 'Environment=%q\n' "http_proxy=${K3S_HTTP_PROXY}"
  [[ -n "${K3S_HTTPS_PROXY}" ]] && printf 'Environment=%q\n' "HTTPS_PROXY=${K3S_HTTPS_PROXY}"
  [[ -n "${K3S_HTTPS_PROXY}" ]] && printf 'Environment=%q\n' "https_proxy=${K3S_HTTPS_PROXY}"
  printf 'Environment=%q\n' "NO_PROXY=${K3S_NO_PROXY}"
  printf 'Environment=%q\n' "no_proxy=${K3S_NO_PROXY}"
  printf '%s\n' 'ExecStart='
  printf '%s\n' "ExecStart=/usr/local/bin/k3s ${k3s_exec}"
} >"${tmp_unit}"
unit_changed=false
if ! cmp -s "${tmp_unit}" /etc/systemd/system/k3s.service.d/10-zing-runtime.conf; then
  mv "${tmp_unit}" /etc/systemd/system/k3s.service.d/10-zing-runtime.conf
  unit_changed=true
else
  rm -f "${tmp_unit}"
fi
systemctl daemon-reload
if [[ "${unit_changed}" == "true" ]] || ! systemctl is-active --quiet k3s; then
  systemctl restart k3s
fi
for _ in $(seq 1 60); do
  kubectl get node >/dev/null 2>&1 && exit 0
  sleep 2
done
systemctl status k3s --no-pager
exit 1
REMOTE
}

configure_k3s_system_images() {
  log "ensuring k3s system images use mainland-accessible mirrors"
  ssh_remote "${SSH_HOST}" \
    "K3S_DATA_DIR='${K3S_DATA_DIR}' \
     K3S_DISABLE_LOCAL_STORAGE='${K3S_DISABLE_LOCAL_STORAGE}' \
     K3S_COREDNS_IMAGE='${K3S_COREDNS_IMAGE}' \
     K3S_METRICS_SERVER_IMAGE='${K3S_METRICS_SERVER_IMAGE}' \
     K3S_LOCAL_PATH_PROVISIONER_IMAGE='${K3S_LOCAL_PATH_PROVISIONER_IMAGE}' \
     K3S_BUSYBOX_IMAGE='${K3S_BUSYBOX_IMAGE}' \
     bash -s" <<'REMOTE'
set -Eeuo pipefail
changed="$(python3 - <<'PY'
import os
import re
from pathlib import Path

data_dir = Path(os.environ["K3S_DATA_DIR"])
targets = [
    (
        data_dir / "server/manifests/coredns.yaml",
        r'image:\s*"[^"]*mirrored-coredns-coredns:1\.14\.6"',
        f'image: "{os.environ["K3S_COREDNS_IMAGE"]}"',
    ),
    (
        data_dir / "server/manifests/local-storage.yaml",
        r'image:\s*"[^"]*local-path-provisioner:v0\.0\.36"',
        f'image: "{os.environ["K3S_LOCAL_PATH_PROVISIONER_IMAGE"]}"',
    ),
    (
        data_dir / "server/manifests/local-storage.yaml",
        r'image:\s*"[^"]*mirrored-library-busybox:1\.37\.0"',
        f'image: "{os.environ["K3S_BUSYBOX_IMAGE"]}"',
    ),
    (
        data_dir / "server/manifests/metrics-server/metrics-server-deployment.yaml",
        r'image:\s*"[^"]*mirrored-metrics-server:v0\.9\.0"',
        f'image: "{os.environ["K3S_METRICS_SERVER_IMAGE"]}"',
    ),
    (
        data_dir / "server/manifests/metrics-server/metrics-server-deployment.yaml",
        r'image:\s*"[^"]*metrics-server:v0\.9\.0"',
        f'image: "{os.environ["K3S_METRICS_SERVER_IMAGE"]}"',
    ),
]
did_change = False
for path, pattern, replacement in targets:
    if not path.exists():
        continue
    text = path.read_text()
    new = re.sub(pattern, replacement, text)
    if new != text:
        path.write_text(new)
        did_change = True
print("true" if did_change else "false")
PY
)"
if [[ "${changed}" == "true" ]]; then
  kubectl apply -f "${K3S_DATA_DIR}/server/manifests/coredns.yaml"
  if [[ "${K3S_DISABLE_LOCAL_STORAGE}" != "true" && -f "${K3S_DATA_DIR}/server/manifests/local-storage.yaml" ]]; then
    kubectl apply -f "${K3S_DATA_DIR}/server/manifests/local-storage.yaml"
  fi
  kubectl apply -f "${K3S_DATA_DIR}/server/manifests/metrics-server/metrics-server-deployment.yaml"
  kubectl -n kube-system rollout restart deploy/coredns deploy/metrics-server
  if [[ "${K3S_DISABLE_LOCAL_STORAGE}" != "true" ]]; then
    kubectl -n kube-system rollout restart deploy/local-path-provisioner || true
  fi
fi
if [[ "${K3S_DISABLE_LOCAL_STORAGE}" == "true" ]]; then
  kubectl -n kube-system delete deploy/local-path-provisioner --ignore-not-found=true
  kubectl delete storageclass/local-path --ignore-not-found=true
fi
kubectl -n kube-system rollout status deploy/coredns --timeout=240s
kubectl -n kube-system rollout status deploy/metrics-server --timeout=120s || true
REMOTE
}

label_control_node() {
  log "labeling k3s control node as Shanghai Zing denoiser/control"
  ssh_remote "${SSH_HOST}" \
    "CONTROL_NODE_LABEL_KEY='${CONTROL_NODE_LABEL_KEY}' \
     CONTROL_NODE_LABEL_VALUE='${CONTROL_NODE_LABEL_VALUE}' \
     DENOISER_GPU_POOL='${DENOISER_GPU_POOL}' \
     bash -s" <<'REMOTE'
set -Eeuo pipefail
node="$(kubectl get node "$(hostname)" -o jsonpath='{.metadata.name}' 2>/dev/null || true)"
if [[ -z "${node}" ]]; then
  node="$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')"
fi
kubectl label node "${node}" \
  "${CONTROL_NODE_LABEL_KEY}=${CONTROL_NODE_LABEL_VALUE}" \
  seedleap.ai/gpu-worker=true \
  "seedleap.ai/gpu-pool=${DENOISER_GPU_POOL}" \
  --overwrite
REMOTE
}

prepare_remote() {
  log "preparing remote k8s assets"
  ssh_remote "${SSH_HOST}" "mkdir -p '${REMOTE_DIR}'"
}

create_acr_secret() {
  log "creating/updating ACR image pull secret"
  auth_json="$(aliyun cr GetAuthorizationToken --RegionId "${ACR_REGION}" --InstanceId "${ACR_INSTANCE_ID}")"
  user="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["TempUsername"])' <<<"${auth_json}")"
  token="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["AuthorizationToken"])' <<<"${auth_json}")"
  local secret_yaml
  secret_yaml="$(mktemp)"
  kubectl -n "${NAMESPACE}" create secret docker-registry acr-pull \
    --docker-server="${ACR_REGISTRY}" \
    --docker-username="${user}" \
    --docker-password="${token}" \
    --dry-run=client -o yaml >"${secret_yaml}"
  scp_remote "${secret_yaml}" "${SSH_HOST}:${REMOTE_DIR}/acr-pull.yaml"
  rm -f "${secret_yaml}"
  ssh_remote "${SSH_HOST}" \
    "NAMESPACE='${NAMESPACE}' REMOTE_DIR='${REMOTE_DIR}' bash -s" <<'REMOTE'
set -Eeuo pipefail
kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1 || kubectl create namespace "${NAMESPACE}"
kubectl apply -f "${REMOTE_DIR}/acr-pull.yaml"
REMOTE
}

create_optional_webui_secret() {
  log "creating/updating optional webui secret"
  ssh_remote "${SSH_HOST}" \
    "NAMESPACE='${NAMESPACE}' WEBUI_PROMPT_REWRITER_PATH='${WEBUI_PROMPT_REWRITER_PATH}' WEBUI_DOUBAO_ARK_PATH='${WEBUI_DOUBAO_ARK_PATH}' WEBUI_WORLD_IMAGE_CONFIG_PATH='${WEBUI_WORLD_IMAGE_CONFIG_PATH}' bash -s" <<'REMOTE'
set -Eeuo pipefail
if [[ ! -f "${WEBUI_PROMPT_REWRITER_PATH}" && ! -f "${WEBUI_DOUBAO_ARK_PATH}" && ! -f "${WEBUI_WORLD_IMAGE_CONFIG_PATH}" ]]; then
  echo "no optional webui secret files found; leaving webui secret unchanged"
  exit 0
fi
if ! kubectl -n "${NAMESPACE}" get secret webui-secrets >/dev/null 2>&1; then
  kubectl -n "${NAMESPACE}" create secret generic webui-secrets
fi
patch_secret_file() {
  local key="$1"
  local path="$2"
  local encoded
  if [[ ! -f "${path}" ]]; then
    return
  fi
  encoded="$(base64 -w0 "${path}" 2>/dev/null || base64 "${path}" | tr -d '\n')"
  kubectl -n "${NAMESPACE}" patch secret webui-secrets \
    --type merge \
    -p "{\"data\":{\"${key}\":\"${encoded}\"}}"
}
patch_secret_file prompt-rewriter-vertex.json "${WEBUI_PROMPT_REWRITER_PATH}"
patch_secret_file doubao-ark.json "${WEBUI_DOUBAO_ARK_PATH}"
patch_secret_file world-image-model-config.json "${WEBUI_WORLD_IMAGE_CONFIG_PATH}"
REMOTE
}

apply_runtime_source_patch() {
  if [[ "${APPLY_RUNTIME_SOURCE_PATCH}" != "true" ]]; then
    log "runtime source patch disabled"
    return
  fi

  log "uploading runtime source patch"
  local remote_patch_dir="${REMOTE_DIR}/runtime-source-patch"
  local entrypoint_files=(
    realtime_gateway_server.py
    realtime_coordinator_server.py
    realtime_vae_server.py
  )
  local openai_realtime_entrypoint_files=(
    realtime_video_api.py
    generate_session.py
  )
  local utils_files=(
    logging_utils.py
    realtime_video.py
    realtime_trace.py
  )
  local realtime_state_files=(
    causal.py
  )
  local model_dit_files=(
    causal_wanvideo.py
    minwm.py
    minwm_action.py
    minwm_kv_cache.py
  )
  local pipeline_files=(
    minwm_causal_dmd_pipeline.py
  )
  local minwm_stage_files=(
    minwm_causal_denoising.py
  )
  local manager_files=(
    gpu_worker.py
  )
  local config_files=(
    minwm_resolution_buckets.py
  )
  local pipeline_config_files=(
    base.py
    minwm.py
  )
  local sample_config_files=(
    minwm.py
  )
  local jit_diffusion_files=(
    minwm_rmsnorm.py
    minwm_rotary.py
  )
  local jit_csrc_diffusion_files=(
    minwm_rmsnorm.cuh
    minwm_rotary.cuh
  )
  local file
  for file in "${entrypoint_files[@]}"; do
    if [[ ! -f "${RUNTIME_ENTRYPOINT_SOURCE_DIR}/${file}" ]]; then
      printf 'missing runtime entrypoint file: %s\n' "${RUNTIME_ENTRYPOINT_SOURCE_DIR}/${file}" >&2
      return 1
    fi
  done
  for file in "${openai_realtime_entrypoint_files[@]}"; do
    if [[ ! -f "${RUNTIME_OPENAI_REALTIME_ENTRYPOINT_SOURCE_DIR}/${file}" ]]; then
      printf 'missing OpenAI realtime entrypoint file: %s\n' "${RUNTIME_OPENAI_REALTIME_ENTRYPOINT_SOURCE_DIR}/${file}" >&2
      return 1
    fi
  done
  for file in "${utils_files[@]}"; do
    if [[ ! -f "${RUNTIME_UTILS_SOURCE_DIR}/${file}" ]]; then
      printf 'missing runtime utils file: %s\n' "${RUNTIME_UTILS_SOURCE_DIR}/${file}" >&2
      return 1
    fi
  done
  for file in "${realtime_state_files[@]}"; do
    if [[ ! -f "${RUNTIME_REALTIME_STATES_SOURCE_DIR}/${file}" ]]; then
      printf 'missing runtime realtime state file: %s\n' "${RUNTIME_REALTIME_STATES_SOURCE_DIR}/${file}" >&2
      return 1
    fi
  done
  for file in "${model_dit_files[@]}"; do
    if [[ ! -f "${RUNTIME_MODELS_DITS_SOURCE_DIR}/${file}" ]]; then
      printf 'missing runtime model DIT file: %s\n' "${RUNTIME_MODELS_DITS_SOURCE_DIR}/${file}" >&2
      return 1
    fi
  done
  for file in "${pipeline_files[@]}"; do
    if [[ ! -f "${RUNTIME_PIPELINES_SOURCE_DIR}/${file}" ]]; then
      printf 'missing runtime pipeline file: %s\n' "${RUNTIME_PIPELINES_SOURCE_DIR}/${file}" >&2
      return 1
    fi
  done
  for file in "${minwm_stage_files[@]}"; do
    if [[ ! -f "${RUNTIME_PIPELINE_MINWM_STAGE_SOURCE_DIR}/${file}" ]]; then
      printf 'missing MinWM pipeline stage file: %s\n' "${RUNTIME_PIPELINE_MINWM_STAGE_SOURCE_DIR}/${file}" >&2
      return 1
    fi
  done
  for file in "${manager_files[@]}"; do
    if [[ ! -f "${RUNTIME_MANAGERS_SOURCE_DIR}/${file}" ]]; then
      printf 'missing runtime manager file: %s\n' "${RUNTIME_MANAGERS_SOURCE_DIR}/${file}" >&2
      return 1
    fi
  done
  for file in "${config_files[@]}"; do
    if [[ ! -f "${RUNTIME_CONFIGS_SOURCE_DIR}/${file}" ]]; then
      printf 'missing runtime config file: %s\n' "${RUNTIME_CONFIGS_SOURCE_DIR}/${file}" >&2
      return 1
    fi
  done
  for file in "${pipeline_config_files[@]}"; do
    if [[ ! -f "${RUNTIME_PIPELINE_CONFIGS_SOURCE_DIR}/${file}" ]]; then
      printf 'missing pipeline config file: %s\n' "${RUNTIME_PIPELINE_CONFIGS_SOURCE_DIR}/${file}" >&2
      return 1
    fi
  done
  for file in "${sample_config_files[@]}"; do
    if [[ ! -f "${RUNTIME_SAMPLE_CONFIGS_SOURCE_DIR}/${file}" ]]; then
      printf 'missing sample config file: %s\n' "${RUNTIME_SAMPLE_CONFIGS_SOURCE_DIR}/${file}" >&2
      return 1
    fi
  done
  for file in "${jit_diffusion_files[@]}"; do
    if [[ ! -f "${JIT_DIFFUSION_SOURCE_DIR}/${file}" ]]; then
      printf 'missing JIT diffusion file: %s\n' "${JIT_DIFFUSION_SOURCE_DIR}/${file}" >&2
      return 1
    fi
  done
  for file in "${jit_csrc_diffusion_files[@]}"; do
    if [[ ! -f "${JIT_CSRC_DIFFUSION_SOURCE_DIR}/${file}" ]]; then
      printf 'missing JIT csrc diffusion file: %s\n' "${JIT_CSRC_DIFFUSION_SOURCE_DIR}/${file}" >&2
      return 1
    fi
  done
  if [[ ! -f "${MINWM_PROFILE_LAUNCHER_SOURCE}" ]]; then
    printf 'missing MinWM profile launcher file: %s\n' "${MINWM_PROFILE_LAUNCHER_SOURCE}" >&2
    return 1
  fi
  for file in envs.py prometheus_client.py sitecustomize.py; do
    if [[ ! -f "${RUNTIME_SHIMS_SOURCE_DIR}/${file}" ]]; then
      printf 'missing runtime shim file: %s\n' "${RUNTIME_SHIMS_SOURCE_DIR}/${file}" >&2
      return 1
    fi
  done

  ssh_remote "${SSH_HOST}" \
    "rm -rf '${remote_patch_dir}' && mkdir -p '${remote_patch_dir}/realtime' '${remote_patch_dir}/realtime-states' '${remote_patch_dir}/entrypoints' '${remote_patch_dir}/openai-realtime-entrypoints' '${remote_patch_dir}/utils' '${remote_patch_dir}/models-dits' '${remote_patch_dir}/pipelines' '${remote_patch_dir}/pipeline-minwm-stage' '${remote_patch_dir}/managers' '${remote_patch_dir}/configs' '${remote_patch_dir}/pipeline-configs' '${remote_patch_dir}/sample-configs' '${remote_patch_dir}/jit-diffusion' '${remote_patch_dir}/jit-csrc-diffusion' '${remote_patch_dir}/tools' '${remote_patch_dir}/shims'"
  scp_remote "${RUNTIME_REALTIME_SOURCE_DIR}"/*.py "${SSH_HOST}:${remote_patch_dir}/realtime/"
  local realtime_state_sources=()
  for file in "${realtime_state_files[@]}"; do
    realtime_state_sources+=("${RUNTIME_REALTIME_STATES_SOURCE_DIR}/${file}")
  done
  scp_remote "${realtime_state_sources[@]}" "${SSH_HOST}:${remote_patch_dir}/realtime-states/"
  local entrypoint_sources=()
  for file in "${entrypoint_files[@]}"; do
    entrypoint_sources+=("${RUNTIME_ENTRYPOINT_SOURCE_DIR}/${file}")
  done
  scp_remote "${entrypoint_sources[@]}" "${SSH_HOST}:${remote_patch_dir}/entrypoints/"
  local openai_realtime_entrypoint_sources=()
  for file in "${openai_realtime_entrypoint_files[@]}"; do
    openai_realtime_entrypoint_sources+=("${RUNTIME_OPENAI_REALTIME_ENTRYPOINT_SOURCE_DIR}/${file}")
  done
  scp_remote "${openai_realtime_entrypoint_sources[@]}" "${SSH_HOST}:${remote_patch_dir}/openai-realtime-entrypoints/"
  local utils_sources=()
  for file in "${utils_files[@]}"; do
    utils_sources+=("${RUNTIME_UTILS_SOURCE_DIR}/${file}")
  done
  scp_remote "${utils_sources[@]}" "${SSH_HOST}:${remote_patch_dir}/utils/"
  local model_dit_sources=()
  for file in "${model_dit_files[@]}"; do
    model_dit_sources+=("${RUNTIME_MODELS_DITS_SOURCE_DIR}/${file}")
  done
  scp_remote "${model_dit_sources[@]}" "${SSH_HOST}:${remote_patch_dir}/models-dits/"
  local pipeline_sources=()
  for file in "${pipeline_files[@]}"; do
    pipeline_sources+=("${RUNTIME_PIPELINES_SOURCE_DIR}/${file}")
  done
  scp_remote "${pipeline_sources[@]}" "${SSH_HOST}:${remote_patch_dir}/pipelines/"
  local minwm_stage_sources=()
  for file in "${minwm_stage_files[@]}"; do
    minwm_stage_sources+=("${RUNTIME_PIPELINE_MINWM_STAGE_SOURCE_DIR}/${file}")
  done
  scp_remote "${minwm_stage_sources[@]}" "${SSH_HOST}:${remote_patch_dir}/pipeline-minwm-stage/"
  local manager_sources=()
  for file in "${manager_files[@]}"; do
    manager_sources+=("${RUNTIME_MANAGERS_SOURCE_DIR}/${file}")
  done
  scp_remote "${manager_sources[@]}" "${SSH_HOST}:${remote_patch_dir}/managers/"
  local config_sources=()
  for file in "${config_files[@]}"; do
    config_sources+=("${RUNTIME_CONFIGS_SOURCE_DIR}/${file}")
  done
  scp_remote "${config_sources[@]}" "${SSH_HOST}:${remote_patch_dir}/configs/"
  local pipeline_config_sources=()
  for file in "${pipeline_config_files[@]}"; do
    pipeline_config_sources+=("${RUNTIME_PIPELINE_CONFIGS_SOURCE_DIR}/${file}")
  done
  scp_remote "${pipeline_config_sources[@]}" "${SSH_HOST}:${remote_patch_dir}/pipeline-configs/"
  local sample_config_sources=()
  for file in "${sample_config_files[@]}"; do
    sample_config_sources+=("${RUNTIME_SAMPLE_CONFIGS_SOURCE_DIR}/${file}")
  done
  scp_remote "${sample_config_sources[@]}" "${SSH_HOST}:${remote_patch_dir}/sample-configs/"
  local jit_diffusion_sources=()
  for file in "${jit_diffusion_files[@]}"; do
    jit_diffusion_sources+=("${JIT_DIFFUSION_SOURCE_DIR}/${file}")
  done
  scp_remote "${jit_diffusion_sources[@]}" "${SSH_HOST}:${remote_patch_dir}/jit-diffusion/"
  local jit_csrc_diffusion_sources=()
  for file in "${jit_csrc_diffusion_files[@]}"; do
    jit_csrc_diffusion_sources+=("${JIT_CSRC_DIFFUSION_SOURCE_DIR}/${file}")
  done
  scp_remote "${jit_csrc_diffusion_sources[@]}" "${SSH_HOST}:${remote_patch_dir}/jit-csrc-diffusion/"
  scp_remote "${MINWM_PROFILE_LAUNCHER_SOURCE}" "${SSH_HOST}:${remote_patch_dir}/tools/"
  scp_remote \
    "${RUNTIME_SHIMS_SOURCE_DIR}/envs.py" \
    "${RUNTIME_SHIMS_SOURCE_DIR}/prometheus_client.py" \
    "${RUNTIME_SHIMS_SOURCE_DIR}/sitecustomize.py" \
    "${SSH_HOST}:${remote_patch_dir}/shims/"

  ssh_remote "${SSH_HOST}" \
    "NAMESPACE='${NAMESPACE}' REMOTE_PATCH_DIR='${remote_patch_dir}' bash -s" <<'REMOTE'
set -Eeuo pipefail
: >"${REMOTE_PATCH_DIR}/utils/__init__.py"
kubectl -n "${NAMESPACE}" delete configmap \
  zing-runtime-realtime-patch \
  zing-runtime-realtime-states-patch \
  zing-runtime-entrypoint-patch \
  zing-runtime-openai-realtime-entrypoint-patch \
  zing-runtime-utils-patch \
  zing-runtime-model-dits-patch \
  zing-runtime-pipelines-patch \
  zing-runtime-pipeline-minwm-stage-patch \
  zing-runtime-managers-patch \
  zing-runtime-configs-patch \
  zing-runtime-pipeline-configs-patch \
  zing-runtime-sample-configs-patch \
  zing-jit-diffusion-patch \
  zing-jit-csrc-diffusion-patch \
  zing-runtime-tools-patch \
  zing-runtime-envs-patch \
  zing-runtime-dep-shims \
  --ignore-not-found=true
kubectl -n "${NAMESPACE}" create configmap zing-runtime-realtime-patch \
  --from-file="${REMOTE_PATCH_DIR}/realtime"
kubectl -n "${NAMESPACE}" create configmap zing-runtime-realtime-states-patch \
  --from-file="${REMOTE_PATCH_DIR}/realtime-states"
kubectl -n "${NAMESPACE}" create configmap zing-runtime-entrypoint-patch \
  --from-file="${REMOTE_PATCH_DIR}/entrypoints/realtime_gateway_server.py" \
  --from-file="${REMOTE_PATCH_DIR}/entrypoints/realtime_coordinator_server.py" \
  --from-file="${REMOTE_PATCH_DIR}/entrypoints/realtime_vae_server.py"
kubectl -n "${NAMESPACE}" create configmap zing-runtime-openai-realtime-entrypoint-patch \
  --from-file="${REMOTE_PATCH_DIR}/openai-realtime-entrypoints/realtime_video_api.py" \
  --from-file="${REMOTE_PATCH_DIR}/openai-realtime-entrypoints/generate_session.py"
kubectl -n "${NAMESPACE}" create configmap zing-runtime-utils-patch \
  --from-file="${REMOTE_PATCH_DIR}/utils"
kubectl -n "${NAMESPACE}" create configmap zing-runtime-model-dits-patch \
  --from-file="${REMOTE_PATCH_DIR}/models-dits"
kubectl -n "${NAMESPACE}" create configmap zing-runtime-pipelines-patch \
  --from-file="${REMOTE_PATCH_DIR}/pipelines"
kubectl -n "${NAMESPACE}" create configmap zing-runtime-pipeline-minwm-stage-patch \
  --from-file="${REMOTE_PATCH_DIR}/pipeline-minwm-stage"
kubectl -n "${NAMESPACE}" create configmap zing-runtime-managers-patch \
  --from-file="${REMOTE_PATCH_DIR}/managers"
kubectl -n "${NAMESPACE}" create configmap zing-runtime-configs-patch \
  --from-file="${REMOTE_PATCH_DIR}/configs"
kubectl -n "${NAMESPACE}" create configmap zing-runtime-pipeline-configs-patch \
  --from-file="${REMOTE_PATCH_DIR}/pipeline-configs"
kubectl -n "${NAMESPACE}" create configmap zing-runtime-sample-configs-patch \
  --from-file="${REMOTE_PATCH_DIR}/sample-configs"
kubectl -n "${NAMESPACE}" create configmap zing-jit-diffusion-patch \
  --from-file="${REMOTE_PATCH_DIR}/jit-diffusion"
kubectl -n "${NAMESPACE}" create configmap zing-jit-csrc-diffusion-patch \
  --from-file="${REMOTE_PATCH_DIR}/jit-csrc-diffusion"
kubectl -n "${NAMESPACE}" create configmap zing-runtime-tools-patch \
  --from-file="${REMOTE_PATCH_DIR}/tools/minwm_profile_launcher.py"
kubectl -n "${NAMESPACE}" create configmap zing-runtime-envs-patch \
  --from-file="${REMOTE_PATCH_DIR}/shims/envs.py"
kubectl -n "${NAMESPACE}" create configmap zing-runtime-dep-shims \
  --from-file="${REMOTE_PATCH_DIR}/shims/prometheus_client.py" \
  --from-file="${REMOTE_PATCH_DIR}/shims/sitecustomize.py"
REMOTE
}

apply_auth_source_patch() {
  if [[ "${APPLY_AUTH_SOURCE_PATCH}" != "true" ]]; then
    log "auth source patch disabled"
    return
  fi
  if [[ ! -f "${AUTH_SOURCE_FILE}" ]]; then
    printf 'missing auth source file: %s\n' "${AUTH_SOURCE_FILE}" >&2
    return 1
  fi

  log "uploading Feishu auth source patch"
  local remote_auth_dir="${REMOTE_DIR}/auth-source-patch"
  ssh_remote "${SSH_HOST}" "rm -rf '${remote_auth_dir}' && mkdir -p '${remote_auth_dir}'"
  scp_remote "${AUTH_SOURCE_FILE}" "${SSH_HOST}:${remote_auth_dir}/feishu_auth_server.py"
  ssh_remote "${SSH_HOST}" \
    "NAMESPACE='${NAMESPACE}' REMOTE_AUTH_DIR='${remote_auth_dir}' bash -s" <<'REMOTE'
set -Eeuo pipefail
kubectl -n "${NAMESPACE}" delete configmap zing-auth-source --ignore-not-found=true
kubectl -n "${NAMESPACE}" create configmap zing-auth-source \
  --from-file="${REMOTE_AUTH_DIR}/feishu_auth_server.py"
REMOTE
}

render_denoiser_sp1x7_manifest() {
  local output="$1"
  NAMESPACE="${NAMESPACE}" \
  GPU_RUNTIME_IMAGE="${GPU_RUNTIME_IMAGE}" \
  RUNTIME_SOURCE_PATCH_VERSION="${RUNTIME_SOURCE_PATCH_VERSION}" \
  DENOISER_GPU_POOL="${DENOISER_GPU_POOL}" \
  REALTIME_TARGET_FPS="${REALTIME_TARGET_FPS}" \
  REALTIME_SIZE="${REALTIME_SIZE}" \
  python3 - "${output}" <<'PY'
import json
import os
import sys

output = sys.argv[1]
namespace = os.environ["NAMESPACE"]
image = os.environ["GPU_RUNTIME_IMAGE"]
patch_version = os.environ["RUNTIME_SOURCE_PATCH_VERSION"]
gpu_pool = os.environ["DENOISER_GPU_POOL"]
target_fps = os.environ["REALTIME_TARGET_FPS"]
size = os.environ["REALTIME_SIZE"]
app_name = "zing-denoiser-sp1x7"
worker_count = 7


def field_ref(path):
    return {"valueFrom": {"fieldRef": {"fieldPath": path}}}


def env(name, value):
    return {"name": name, "value": value}


def mount(name, path, sub_path=None):
    item = {"name": name, "mountPath": path, "readOnly": True}
    if sub_path is not None:
        item["subPath"] = sub_path
    return item


def patch_mounts(prefix):
    base = f"{prefix}/sglang"
    return [
        mount(
            "runtime-tools-patch",
            f"{base}/multimodal_gen/tools/minwm_profile_launcher.py",
            "minwm_profile_launcher.py",
        ),
        mount(
            "runtime-openai-realtime-entrypoint-patch",
            f"{base}/multimodal_gen/runtime/entrypoints/openai/realtime/realtime_video_api.py",
            "realtime_video_api.py",
        ),
        mount(
            "runtime-openai-realtime-entrypoint-patch",
            f"{base}/multimodal_gen/runtime/entrypoints/openai/realtime/generate_session.py",
            "generate_session.py",
        ),
        mount(
            "runtime-realtime-patch",
            f"{base}/multimodal_gen/runtime/realtime/critical_path_metrics.py",
            "critical_path_metrics.py",
        ),
        mount(
            "runtime-realtime-patch",
            f"{base}/multimodal_gen/runtime/realtime/async_vae_protocol.py",
            "async_vae_protocol.py",
        ),
        mount(
            "runtime-realtime-patch",
            f"{base}/multimodal_gen/runtime/realtime/async_vae_client.py",
            "async_vae_client.py",
        ),
        mount(
            "runtime-realtime-patch",
            f"{base}/multimodal_gen/runtime/realtime/causal_kv_cache_pool.py",
            "causal_kv_cache_pool.py",
        ),
        mount(
            "runtime-realtime-states-patch",
            f"{base}/multimodal_gen/runtime/realtime/states/causal.py",
            "causal.py",
        ),
        mount(
            "runtime-utils-patch",
            f"{base}/multimodal_gen/runtime/utils/realtime_video.py",
            "realtime_video.py",
        ),
        mount(
            "runtime-model-dits-patch",
            f"{base}/multimodal_gen/runtime/models/dits/causal_wanvideo.py",
            "causal_wanvideo.py",
        ),
        mount(
            "runtime-model-dits-patch",
            f"{base}/multimodal_gen/runtime/models/dits/minwm.py",
            "minwm.py",
        ),
        mount(
            "runtime-model-dits-patch",
            f"{base}/multimodal_gen/runtime/models/dits/minwm_action.py",
            "minwm_action.py",
        ),
        mount(
            "runtime-model-dits-patch",
            f"{base}/multimodal_gen/runtime/models/dits/minwm_kv_cache.py",
            "minwm_kv_cache.py",
        ),
        mount(
            "runtime-pipelines-patch",
            f"{base}/multimodal_gen/runtime/pipelines/minwm_causal_dmd_pipeline.py",
            "minwm_causal_dmd_pipeline.py",
        ),
        mount(
            "runtime-pipeline-minwm-stage-patch",
            f"{base}/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minwm/minwm_causal_denoising.py",
            "minwm_causal_denoising.py",
        ),
        mount(
            "runtime-managers-patch",
            f"{base}/multimodal_gen/runtime/managers/gpu_worker.py",
            "gpu_worker.py",
        ),
        mount(
            "runtime-configs-patch",
            f"{base}/multimodal_gen/configs/minwm_resolution_buckets.py",
            "minwm_resolution_buckets.py",
        ),
        mount(
            "runtime-pipeline-configs-patch",
            f"{base}/multimodal_gen/configs/pipeline_configs/base.py",
            "base.py",
        ),
        mount(
            "runtime-pipeline-configs-patch",
            f"{base}/multimodal_gen/configs/pipeline_configs/minwm.py",
            "minwm.py",
        ),
        mount(
            "runtime-sample-configs-patch",
            f"{base}/multimodal_gen/configs/sample/minwm.py",
            "minwm.py",
        ),
        mount(
            "jit-diffusion-patch",
            f"{prefix}/sglang/jit_kernel/diffusion/minwm_rotary.py",
            "minwm_rotary.py",
        ),
        mount(
            "jit-diffusion-patch",
            f"{prefix}/sglang/jit_kernel/diffusion/minwm_rmsnorm.py",
            "minwm_rmsnorm.py",
        ),
        mount(
            "jit-csrc-diffusion-patch",
            f"{prefix}/sglang/jit_kernel/csrc/diffusion/minwm_rotary.cuh",
            "minwm_rotary.cuh",
        ),
        mount(
            "jit-csrc-diffusion-patch",
            f"{prefix}/sglang/jit_kernel/csrc/diffusion/minwm_rmsnorm.cuh",
            "minwm_rmsnorm.cuh",
        ),
    ]


def worker_command(index):
    port = 30000 + index * 10
    master_port = 30200 + index * 10
    scheduler_port = 5615 + index * 10
    return f"""set -euo pipefail
test -f /models/minwm-tianpeng-gap12/_READY
test -f /models/minwm-tianpeng-gap12/model_index.json
test -d /models/minwm-tianpeng-gap12/transformer
exec 9>/var/run/minwm-startup-lock/denoiser.lock
flock -x 9
python3 -m sglang.multimodal_gen.tools.minwm_profile_launcher \\
  --profile auto \\
  --taehv-checkpoint-path /models/taehv/taew2_2.pth \\
  -- \\
  --model-path /models/minwm-tianpeng-gap12 \\
  --num-gpus 1 \\
  --tp-size 1 \\
  --sp-degree 1 \\
  --ulysses-degree 1 \\
  --ring-degree 1 \\
  --enable-cuda-graph true \\
  --batching-max-size 1 \\
  --batching-delay-ms 2 \\
  --realtime-vae-backend taehv_remote \\
  --realtime-vae-transport websocket \\
  --realtime-session-idle-timeout-s 90 \\
  --realtime-session-max-lifetime-s 70 \\
  --realtime-admission-wait-s 10 \\
  --host 0.0.0.0 \\
  --port {port} \\
  --master-port {master_port} \\
  --scheduler-port {scheduler_port} \\
  --strict-ports &
child=$!
terminate() {{
  kill -TERM "${{child}}" 2>/dev/null || true
  wait "${{child}}" || true
  exit 143
}}
trap terminate TERM INT
until curl --fail --silent --max-time 2 http://127.0.0.1:{port}/health >/dev/null; do
  if ! kill -0 "${{child}}" 2>/dev/null; then
    wait "${{child}}"
    exit $?
  fi
  sleep 2
done
flock -u 9
wait "${{child}}"
"""


def worker_env(index):
    return [
        env("PYTHONUNBUFFERED", "1"),
        env("PYTHONPATH", "/opt/sglang/python:/sgl-workspace/sglang/python"),
        env("NVIDIA_DRIVER_CAPABILITIES", "compute,utility"),
        env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        env("SGLANG_DISABLE_PDEATHSIG", "1"),
        env("SGLANG_USE_SGL_FA3_KERNEL", "1"),
        env("OMP_NUM_THREADS", "4"),
        env("MKL_NUM_THREADS", "4"),
        env("OPENBLAS_NUM_THREADS", "4"),
        env("NUMEXPR_NUM_THREADS", "4"),
        env("VECLIB_MAXIMUM_THREADS", "4"),
        env("TOKENIZERS_PARALLELISM", "false"),
        env("WORKER_EPOCH_FILE", "/var/run/minwm-worker/epoch"),
        env("MINWM_ATTENTION_IMPL", "packed"),
        env("MINWM_SEGMENT_COMPILE", "true"),
        env("MINWM_CACHE_ROTATED_K", "true"),
        env("MINWM_PRECOMPUTE_CACHE_ROPE", "true"),
        env("MINWM_CACHE_PACKED_METADATA", "true"),
        env("MINWM_PACKED_ATTENTION_DETERMINISTIC", "false"),
        env("MINWM_RMSNORM_IMPL", "auto"),
        env("MINWM_PARITY_DETERMINISTIC", "false"),
        env("MINWM_NATIVE_COMPONENTS", ""),
        env("MINWM_RUNTIME_ALIGNMENT_LOG", "1"),
        env("SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D", "false"),
        env("NCCL_DEBUG", "WARN"),
        env("NCCL_PROTO", "Simple"),
        env("REALTIME_MAX_SESSIONS", "1"),
        env(
            "REALTIME_UI_CONFIG_JSON",
            json.dumps(
                {
                    "targetFps": int(target_fps),
                    "size": size,
                    "dualModels": {"minwm": {"sinkSize": 8, "windowFrames": 32}},
                },
                separators=(",", ":"),
            ),
        ),
        env("NVIDIA_VISIBLE_DEVICES", str(index)),
    ]


def worker_mounts(index):
    mounts = [
        {"name": f"worker-epoch-denoiser-{index}", "mountPath": "/var/run/minwm-worker"},
        {"name": "startup-lock", "mountPath": "/var/run/minwm-startup-lock"},
        {"name": "model", "mountPath": "/models/minwm-tianpeng-gap12", "readOnly": True},
        {"name": "taehv", "mountPath": "/models/taehv", "readOnly": True},
        {"name": "shm", "mountPath": "/dev/shm"},
    ]
    mounts.extend(patch_mounts("/opt/sglang/python"))
    mounts.extend(patch_mounts("/sgl-workspace/sglang/python"))
    return mounts


init_containers = []
containers = []
for index in range(worker_count):
    port = 30000 + index * 10
    init_containers.append(
        {
            "name": f"denoiser-heartbeat-{index}",
            "image": image,
            "imagePullPolicy": "IfNotPresent",
            "restartPolicy": "Always",
            "command": ["/bin/sh", "-ec"],
            "args": [
                "exec python3 -m sglang.multimodal_gen.runtime.entrypoints.realtime_worker_heartbeat "
                "--coordinator-url=http://zing-coordinator:18081 "
                f"--health-url=http://127.0.0.1:{port}/health "
                f"--state-url=http://127.0.0.1:{port}/v1/realtime_worker/state "
                f"--worker-id=$(NODE_NAME)-denoiser-{index} "
                "--worker-epoch-file=/var/run/minwm-worker/epoch "
                "--role=denoiser "
                f"--endpoint=ws://$(POD_IP):{port}/v1/realtime_video/generate "
                f"--reservation-endpoint=http://$(POD_IP):{port}/v1/realtime_worker "
                "--node-name=$(NODE_NAME) "
                "--capacity=1 "
                "--model-revision=wan22-5b-stage3-dmd-47-0808-2fb2cfec2a2 "
                "--vae-fingerprint=taew2_2-d053e216 "
                "--interval-s=5"
            ],
            "env": [
                env("PYTHONPATH", "/opt/sglang/python:/sgl-workspace/sglang/python"),
                env("SGLANG_LIGHTWEIGHT_RUNTIME", "1"),
                env("NVIDIA_VISIBLE_DEVICES", "void"),
                {"name": "POD_IP", **field_ref("status.podIP")},
                {"name": "NODE_NAME", **field_ref("spec.nodeName")},
            ],
            "resources": {
                "requests": {"cpu": "25m", "memory": "64Mi"},
                "limits": {"cpu": "200m", "memory": "256Mi"},
            },
            "volumeMounts": [
                {
                    "name": f"worker-epoch-denoiser-{index}",
                    "mountPath": "/var/run/minwm-worker",
                }
            ],
        }
    )
    containers.append(
        {
            "name": f"denoiser-{index}",
            "image": image,
            "imagePullPolicy": "IfNotPresent",
            "workingDir": "/opt/sglang",
            "command": ["/bin/bash", "-lc"],
            "args": [worker_command(index)],
            "env": worker_env(index),
            "ports": [{"name": f"denoiser-{index}", "containerPort": port}],
            "startupProbe": {
                "httpGet": {"path": "/health", "port": f"denoiser-{index}"},
                "periodSeconds": 10,
                "timeoutSeconds": 5,
                "failureThreshold": 540,
            },
            "readinessProbe": {
                "httpGet": {"path": "/health", "port": f"denoiser-{index}"},
                "periodSeconds": 10,
                "timeoutSeconds": 5,
                "failureThreshold": 6,
            },
            "livenessProbe": {
                "httpGet": {"path": "/health", "port": f"denoiser-{index}"},
                "initialDelaySeconds": 60,
                "periodSeconds": 30,
                "timeoutSeconds": 5,
                "failureThreshold": 6,
            },
            "resources": {
                "requests": {"cpu": "16", "memory": "180Gi"},
                "limits": {"cpu": "24", "memory": "240Gi"},
            },
            "securityContext": {"capabilities": {"add": ["SYS_ADMIN"]}},
            "volumeMounts": worker_mounts(index),
        }
    )

volumes = [
    {"name": f"worker-epoch-denoiser-{index}", "emptyDir": {}}
    for index in range(worker_count)
]
volumes.extend(
    [
        {
            "name": "startup-lock",
            "hostPath": {
                "path": "/var/run/minwm-startup-lock",
                "type": "DirectoryOrCreate",
            },
        },
        {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "256Gi"}},
        {
            "name": "model",
            "hostPath": {
                "path": "/data/zing-realtime/model-cache/zing/model",
                "type": "Directory",
            },
        },
        {
            "name": "taehv",
            "hostPath": {"path": "/data/zing-realtime/taehv", "type": "Directory"},
        },
    ]
)
for name in [
    "runtime-realtime-patch",
    "runtime-realtime-states-patch",
    "runtime-openai-realtime-entrypoint-patch",
    "runtime-utils-patch",
    "runtime-model-dits-patch",
    "runtime-pipelines-patch",
    "runtime-pipeline-minwm-stage-patch",
    "runtime-managers-patch",
    "runtime-configs-patch",
    "runtime-pipeline-configs-patch",
    "runtime-sample-configs-patch",
    "jit-diffusion-patch",
    "jit-csrc-diffusion-patch",
    "runtime-tools-patch",
]:
    volumes.append({"name": name, "configMap": {"name": f"zing-{name}"}})

manifest = {
    "apiVersion": "apps/v1",
    "kind": "DaemonSet",
    "metadata": {
        "name": app_name,
        "namespace": namespace,
        "labels": {
            "app.kubernetes.io/name": app_name,
            "app.kubernetes.io/part-of": "minwm-realtime",
        },
    },
    "spec": {
        "selector": {"matchLabels": {"app.kubernetes.io/name": app_name}},
        "updateStrategy": {
            "type": "RollingUpdate",
            "rollingUpdate": {"maxUnavailable": 1},
        },
        "template": {
            "metadata": {
                "annotations": {"seedleap.ai/runtime-source-patch": patch_version},
                "labels": {
                    "app.kubernetes.io/name": app_name,
                    "app.kubernetes.io/part-of": "minwm-realtime",
                    "seedleap.ai/worker-role": "denoiser",
                    "seedleap.ai/gpu-topology": "7p1-sp1-local-vae",
                },
            },
            "spec": {
                "serviceAccountName": "zing-realtime",
                "runtimeClassName": "nvidia",
                "terminationGracePeriodSeconds": 120,
                "nodeSelector": {"seedleap.ai/gpu-pool": gpu_pool},
                "tolerations": [
                    {
                        "key": "nvidia.com/gpu",
                        "operator": "Exists",
                        "effect": "NoSchedule",
                    }
                ],
                "initContainers": init_containers,
                "containers": containers,
                "volumes": volumes,
            },
        },
    },
}

with open(output, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, separators=(",", ":"))
    handle.write("\n")
PY
}

apply_manifests() {
  log "rendering and applying control-plane manifests"
  local rendered
  rendered="$(mktemp)"
  render_template "${SCRIPT_DIR}/manifests/control-plane.yaml.tpl" "${rendered}"
  filter_redis_documents "${rendered}"
  scp_remote "${rendered}" "${SSH_HOST}:${REMOTE_DIR}/control-plane.yaml"
  rm -f "${rendered}"
  ssh_remote "${SSH_HOST}" "kubectl apply -f '${REMOTE_DIR}/control-plane.yaml'"
  if [[ "${APPLY_NVIDIA_DEVICE_PLUGIN}" == "true" ]]; then
    log "rendering and applying NVIDIA device plugin"
    rendered="$(mktemp)"
    render_template "${SCRIPT_DIR}/manifests/nvidia-device-plugin.yaml.tpl" "${rendered}"
    scp_remote "${rendered}" "${SSH_HOST}:${REMOTE_DIR}/nvidia-device-plugin.yaml"
    rm -f "${rendered}"
    ssh_remote "${SSH_HOST}" "kubectl apply -f '${REMOTE_DIR}/nvidia-device-plugin.yaml'"
  fi
  if [[ "${APPLY_GPU_WORKER_TEMPLATES}" == "true" ]]; then
    log "rendering and applying GPU worker templates"
    rendered="$(mktemp)"
    render_denoiser_sp1x7_manifest "${rendered}"
    scp_remote "${rendered}" "${SSH_HOST}:${REMOTE_DIR}/denoiser-sp1x7.json"
    rm -f "${rendered}"
    rendered="$(mktemp)"
    render_template "${SCRIPT_DIR}/manifests/vae-workers-a10.yaml.tpl" "${rendered}"
    scp_remote "${rendered}" "${SSH_HOST}:${REMOTE_DIR}/vae-workers-a10.yaml"
    rm -f "${rendered}"
    ssh_remote "${SSH_HOST}" "kubectl -n '${NAMESPACE}' delete ds/zing-denoiser-sp2 ds/zing-denoiser-5090-sp2 ds/zing-vae-5090-dual deploy/zing-vae-5090 sts/zing-denoiser-5090 svc/zing-denoiser-5090 --ignore-not-found=true"
    ssh_remote "${SSH_HOST}" "kubectl apply -f '${REMOTE_DIR}/denoiser-sp1x7.json'"
    ssh_remote "${SSH_HOST}" "kubectl apply -f '${REMOTE_DIR}/vae-workers-a10.yaml'"
  fi
}

apply_webui_static_patch() {
  if [[ "${APPLY_WEBUI_STATIC_PATCH}" != "true" ]]; then
    log "webui static patch disabled"
    return
  fi

  log "uploading single-Zing webui static patch"
  local remote_patch_dir="${REMOTE_DIR}/webui-static-patch"
  local files=(
    index.html
    styles.css
    playback_controller.js
    model_session.js
    dual_model_controller.js
    h264_websocket_session.js
    happy_oyster_session.js
    session_lifetime.js
    prompt_rewrite_controller.js
    world_rules_controller.js
    fullscreen_controller.js
    trace_topology.js
    trace_transport.js
    app.js
    doubao_provider.py
    prompt_rewriter.py
    server.py
    world_creator.py
  )
  local asset_root="assets/presets/v1/feishu-internal-demo-20260825"
  local asset_files=()
  if [[ -d "${WEBUI_STATIC_SOURCE_DIR}/${asset_root}" ]]; then
    while IFS= read -r asset_file; do
      asset_files+=("${asset_file}")
    done < <(cd "${WEBUI_STATIC_SOURCE_DIR}" && find "${asset_root}" -type f | sort)
  fi
  local local_static_patch_dir
  local local_asset_patch_dir
  local_static_patch_dir="$(mktemp -d)"
  local_asset_patch_dir="$(mktemp -d)"
  local key
  local file
  for file in "${files[@]}"; do
    if [[ ! -f "${WEBUI_STATIC_SOURCE_DIR}/${file}" ]]; then
      printf 'missing webui static file: %s\n' "${WEBUI_STATIC_SOURCE_DIR}/${file}" >&2
      return 1
    fi
    key="${file//\//__}"
    cp "${WEBUI_STATIC_SOURCE_DIR}/${file}" "${local_static_patch_dir}/${key}"
  done
  local asset_file
  for asset_file in "${asset_files[@]}"; do
    if [[ ! -f "${WEBUI_STATIC_SOURCE_DIR}/${asset_file}" ]]; then
      printf 'missing webui static asset: %s\n' "${WEBUI_STATIC_SOURCE_DIR}/${asset_file}" >&2
      return 1
    fi
    key="${asset_file//\//__}"
    cp "${WEBUI_STATIC_SOURCE_DIR}/${asset_file}" "${local_asset_patch_dir}/${key}"
  done
  local static_files_json
  static_files_json="$(python3 - "${files[@]}" <<'PY'
import json
import sys

files = sys.argv[1:]
files.insert(7, "happy_oyster_sdk.js")
print(json.dumps(files, separators=(",", ":")))
PY
)"
  local asset_files_json
  asset_files_json="$(python3 - "${asset_files[@]}" <<'PY'
import json
import sys

print(json.dumps(sys.argv[1:], separators=(",", ":")))
PY
)"

  ssh_remote "${SSH_HOST}" "rm -rf '${remote_patch_dir}' && mkdir -p '${remote_patch_dir}/static' '${remote_patch_dir}/assets'"
  scp_remote "${local_static_patch_dir}"/* "${SSH_HOST}:${remote_patch_dir}/static/"
  if ((${#asset_files[@]})); then
    scp_remote "${local_asset_patch_dir}"/* "${SSH_HOST}:${remote_patch_dir}/assets/"
  fi
  rm -rf "${local_static_patch_dir}" "${local_asset_patch_dir}"
  ssh_remote "${SSH_HOST}" \
    "NAMESPACE='${NAMESPACE}' REMOTE_PATCH_DIR='${remote_patch_dir}' PATCH_NAME='${WEBUI_STATIC_PATCH_NAME}' ASSET_PATCH_NAME='${WEBUI_ASSET_PATCH_NAME}' ASSET_FILE_COUNT='${#asset_files[@]}' bash -s" <<'REMOTE'
set -Eeuo pipefail
printf '%s\n' 'globalThis.HappyOysterSDK = globalThis.HappyOysterSDK || {};' \
  >"${REMOTE_PATCH_DIR}/static/happy_oyster_sdk.js"
kubectl -n "${NAMESPACE}" delete configmap "${PATCH_NAME}" --ignore-not-found=true
kubectl -n "${NAMESPACE}" create configmap "${PATCH_NAME}" --from-file="${REMOTE_PATCH_DIR}/static"
if [[ "${ASSET_FILE_COUNT}" != "0" ]]; then
  kubectl -n "${NAMESPACE}" delete configmap "${ASSET_PATCH_NAME}" --ignore-not-found=true
  kubectl -n "${NAMESPACE}" create configmap "${ASSET_PATCH_NAME}" --from-file="${REMOTE_PATCH_DIR}/assets"
else
  kubectl -n "${NAMESPACE}" delete configmap "${ASSET_PATCH_NAME}" --ignore-not-found=true
fi
REMOTE

  WEBUI_STATIC_PATCH_NAME="${WEBUI_STATIC_PATCH_NAME}" \
  WEBUI_ASSET_PATCH_NAME="${WEBUI_ASSET_PATCH_NAME}" \
  WEBUI_STATIC_PATCH_VERSION="${WEBUI_STATIC_PATCH_VERSION}" \
  WEBUI_STATIC_FILES_JSON="${static_files_json}" \
  WEBUI_ASSET_FILES_JSON="${asset_files_json}" \
  WEBUI_IMAGE="${WEBUI_IMAGE}" \
  python3 - <<'PY' | ssh_remote "${SSH_HOST}" \
    "kubectl -n '${NAMESPACE}' patch deploy zing-webui --type=strategic --patch-file /dev/stdin"
import json
import os

base = "/opt/sglang/python/sglang/multimodal_gen/apps/realtime_webui"
static_files = json.loads(os.environ["WEBUI_STATIC_FILES_JSON"])
asset_files = json.loads(os.environ["WEBUI_ASSET_FILES_JSON"])
volumes = [
    {
        "name": "webui-static-patch",
        "configMap": {"name": os.environ["WEBUI_STATIC_PATCH_NAME"]},
    }
]
volume_mounts = [
    {
        "name": "webui-static-patch",
        "mountPath": f"{base}/{name}",
        "subPath": name.replace("/", "__"),
        "readOnly": True,
    }
    for name in static_files
]
if asset_files:
    volumes.append({
        "name": "webui-preset-assets-patch",
        "configMap": {"name": os.environ["WEBUI_ASSET_PATCH_NAME"]},
    })
    volume_mounts.extend([
        {
            "name": "webui-preset-assets-patch",
            "mountPath": f"{base}/{name}",
            "subPath": name.replace("/", "__"),
            "readOnly": True,
        }
        for name in asset_files
    ])
patch = {
    "spec": {
        "template": {
            "metadata": {
                "annotations": {
                    "seedleap.ai/webui-static-patch": os.environ["WEBUI_STATIC_PATCH_VERSION"]
                }
            },
            "spec": {
                "volumes": volumes,
                "containers": [
                    {
                        "name": "webui",
                        "image": os.environ["WEBUI_IMAGE"],
                        "volumeMounts": volume_mounts,
                    }
                ],
            },
        }
    }
}
print(json.dumps(patch, separators=(",", ":")))
PY
}

wait_rollout() {
  log "waiting for k8s rollouts"
  ssh_remote "${SSH_HOST}" \
    "kubectl -n '${NAMESPACE}' rollout status deploy/zing-coordinator --timeout='${ROLLOUT_TIMEOUT}' && \
     kubectl -n '${NAMESPACE}' rollout status deploy/zing-gateway --timeout='${ROLLOUT_TIMEOUT}' && \
     kubectl -n '${NAMESPACE}' rollout status deploy/zing-auth --timeout='${ROLLOUT_TIMEOUT}' && \
     kubectl -n '${NAMESPACE}' rollout status deploy/zing-webui --timeout='${ROLLOUT_TIMEOUT}'"
  if [[ "${APPLY_GPU_WORKER_TEMPLATES}" == "true" ]]; then
    ssh_remote "${SSH_HOST}" \
      "kubectl -n '${NAMESPACE}' rollout status ds/zing-denoiser-sp1x7 --timeout='${ROLLOUT_TIMEOUT}' && \
       kubectl -n '${NAMESPACE}' rollout status deploy/zing-vae-a10 --timeout='${ROLLOUT_TIMEOUT}'"
  fi
}

print_status() {
  ssh_remote "${SSH_HOST}" \
    "kubectl -n '${NAMESPACE}' get pods -o wide && \
     kubectl -n '${NAMESPACE}' get svc && \
     curl -fsS http://127.0.0.1/runtime-config.js >/dev/null && \
     curl -fsS http://127.0.0.1:18080/healthz >/dev/null"
  log "webui: https://zing-model-shanghai.loopit.com.cn/?mode=i2v&playback=smooth_timeline"
  log "gateway: ${PUBLIC_GATEWAY_BASE_URL}"
}

main() {
  install_k3s
  configure_k3s_system_images
  label_control_node
  prepare_remote
  if [[ "${APPLY_ACR_SECRET}" == "true" ]]; then
    create_acr_secret
  else
    log "ACR image pull secret refresh disabled; using existing secret"
  fi
  create_optional_webui_secret
  apply_runtime_source_patch
  apply_auth_source_patch
  apply_manifests
  apply_webui_static_patch
  wait_rollout
  print_status
}

main "$@"
