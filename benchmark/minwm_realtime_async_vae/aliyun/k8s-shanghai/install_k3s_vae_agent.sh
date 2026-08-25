#!/usr/bin/env bash
set -Eeuo pipefail

K3S_SERVER_URL="${K3S_SERVER_URL:-https://10.0.0.121:6443}"
K3S_TOKEN="${K3S_TOKEN:?set K3S_TOKEN from /data/k3s/server/node-token on the k3s server}"
VAE_GPU_POOL="${VAE_GPU_POOL:-aliyun-shanghai-vae}"
K3S_PAUSE_IMAGE="${K3S_PAUSE_IMAGE:-registry.cn-hangzhou.aliyuncs.com/google_containers/pause:3.10}"
K3S_DATA_DIR="${K3S_DATA_DIR:-/data/k3s}"
KUBELET_ROOT_DIR="${KUBELET_ROOT_DIR:-/data/kubelet}"
K3S_NODE_IP="${K3S_NODE_IP:-}"
K3S_FLANNEL_IFACE="${K3S_FLANNEL_IFACE:-}"
K3S_HTTP_PROXY="${K3S_HTTP_PROXY:-}"
K3S_HTTPS_PROXY="${K3S_HTTPS_PROXY:-${K3S_HTTP_PROXY}}"
K3S_NO_PROXY="${K3S_NO_PROXY:-localhost,127.0.0.1,::1,10.0.0.0/16,10.42.0.0/16,10.43.0.0/16,.svc,.svc.cluster.local,.cluster.local}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_gpu_runtime() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf 'nvidia-smi not found; use an Aliyun GPU image with NVIDIA driver/CUDA installed.\n' >&2
    exit 1
  fi
  nvidia-smi
  if ! command -v nvidia-container-runtime >/dev/null 2>&1; then
    printf 'nvidia-container-runtime not found; install NVIDIA Container Runtime before k3s starts.\n' >&2
    exit 1
  fi
}

install_agent() {
  mkdir -p "${K3S_DATA_DIR}" "${KUBELET_ROOT_DIR}" /data/zing-realtime/taehv
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

  local k3s_exec
  local node_ip_arg=""
  if [[ -n "${K3S_NODE_IP}" ]]; then
    node_ip_arg=" --node-ip ${K3S_NODE_IP}"
  fi
  local flannel_iface_arg=""
  if [[ -n "${K3S_FLANNEL_IFACE}" ]]; then
    flannel_iface_arg=" --flannel-iface ${K3S_FLANNEL_IFACE}"
  fi
  k3s_exec="agent --server ${K3S_SERVER_URL} --token ${K3S_TOKEN} --pause-image ${K3S_PAUSE_IMAGE} --data-dir ${K3S_DATA_DIR} --kubelet-arg root-dir=${KUBELET_ROOT_DIR}${node_ip_arg}${flannel_iface_arg} --node-label seedleap.ai/gpu-worker=true --node-label seedleap.ai/gpu-pool=${VAE_GPU_POOL}"

  if command -v k3s >/dev/null 2>&1 && ! systemctl list-unit-files k3s-agent.service >/dev/null 2>&1 && [[ -f /root/k3s-offline-install.sh ]]; then
    INSTALL_K3S_SKIP_DOWNLOAD=true \
      INSTALL_K3S_EXEC="${k3s_exec}" \
      sh /root/k3s-offline-install.sh
  elif ! command -v k3s >/dev/null 2>&1; then
    curl -sfL https://rancher-mirror.rancher.cn/k3s/k3s-install.sh \
      | INSTALL_K3S_MIRROR=cn \
        INSTALL_K3S_EXEC="${k3s_exec}" \
        sh -
  fi

  mkdir -p /etc/systemd/system/k3s-agent.service.d
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
  if ! cmp -s "${tmp_unit}" /etc/systemd/system/k3s-agent.service.d/10-zing-runtime.conf; then
    mv "${tmp_unit}" /etc/systemd/system/k3s-agent.service.d/10-zing-runtime.conf
    unit_changed=true
  else
    rm -f "${tmp_unit}"
  fi

  systemctl daemon-reload
  if [[ "${unit_changed}" == "true" ]] || ! systemctl is-active --quiet k3s-agent; then
    systemctl restart k3s-agent
  fi
}

verify_runtime_detection() {
  local config="${K3S_DATA_DIR}/agent/etc/containerd/config.toml"
  for _ in $(seq 1 60); do
    if [[ -f "${config}" ]] && grep -q nvidia "${config}"; then
      log "k3s containerd detected NVIDIA runtime"
      return 0
    fi
    sleep 2
  done
  printf 'k3s did not render an NVIDIA runtime into %s; check nvidia-container-runtime PATH and restart k3s-agent.\n' "${config}" >&2
  return 1
}

main() {
  require_gpu_runtime
  install_agent
  verify_runtime_detection
  if [[ ! -f /data/zing-realtime/taehv/taew2_2.pth ]]; then
    log "warning: /data/zing-realtime/taehv/taew2_2.pth is missing; copy it before starting VAE pods"
  fi
  log "k3s VAE agent joined ${K3S_SERVER_URL}"
}

main "$@"
