#!/usr/bin/env bash
set -Eeuo pipefail

REGION="${REGION:-cn-shanghai}"
LINGJUN_CLUSTER_ID="${LINGJUN_CLUSTER_ID:-i119114381787650669980}"
VPC_ID="${VPC_ID:-vpc-uf6bmsqo2f8jwk6o6me6n}"
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-sg-uf65itprxseo4lm028h9}"
KEY_PAIR_NAME="${KEY_PAIR_NAME:-world-model}"

# Lowest practical 24 GB GPU choice found in cn-shanghai during preflight:
# ecs.gn7i-c8g1.2xlarge, 8 vCPU / 30 GiB / 1 x NVIDIA A10 24 GB.
INSTANCE_TYPE="${INSTANCE_TYPE:-ecs.gn7i-c8g1.2xlarge}"
IMAGE_ID="${IMAGE_ID:-ubuntu_24_04_x64_100G_with_gpu_driver_and_cuda_alibase_20260519.vhd}"
SYSTEM_DISK_SIZE="${SYSTEM_DISK_SIZE:-300}"
INTERNET_MAX_BANDWIDTH_OUT="${INTERNET_MAX_BANDWIDTH_OUT:-0}"
SPOT_STRATEGY="${SPOT_STRATEGY:-NoSpot}"
SPOT_PRICE_LIMIT="${SPOT_PRICE_LIMIT:-}"

VAE_ZONE_1="${VAE_ZONE_1:-cn-shanghai-l}"
VAE_ZONE_2="${VAE_ZONE_2:-cn-shanghai-m}"
VAE_CIDR_1="${VAE_CIDR_1:-10.0.1.0/24}"
VAE_CIDR_2="${VAE_CIDR_2:-10.0.2.0/24}"
VAE_NAME_1="${VAE_NAME_1:-zing-shanghai-vae-l}"
VAE_NAME_2="${VAE_NAME_2:-zing-shanghai-vae-m}"

DRY_RUN="${DRY_RUN:-true}"
CONFIRM_ALIYUN_WRITES="${CONFIRM_ALIYUN_WRITES:-}"
REQUIRED_CONFIRMATION="create-shanghai-zing-vae-ecs"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

require_write_confirmation() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    return
  fi
  if [[ "${CONFIRM_ALIYUN_WRITES}" != "${REQUIRED_CONFIRMATION}" ]]; then
    cat >&2 <<EOF
Refusing to run Aliyun writes.

Set:
  DRY_RUN=false CONFIRM_ALIYUN_WRITES=${REQUIRED_CONFIRMATION}

This creates or changes:
  - vSwitch ${VAE_CIDR_1} in ${VAE_ZONE_1} and ${VAE_CIDR_2} in ${VAE_ZONE_2}, if absent
  - security-group ingress for k3s node traffic inside 10.0.0.0/16
  - two pay-as-you-go ECS GPU instances, ${INSTANCE_TYPE}, one per zone
EOF
    exit 2
  fi
}

run_write() {
  printf '+' >&2
  local arg
  for arg in "$@"; do
    printf ' %q' "${arg}" >&2
  done
  printf '\n' >&2
  if [[ "${DRY_RUN}" == "true" ]]; then
    return 0
  fi
  "$@"
}

find_vswitch_by_cidr() {
  local zone="$1"
  local cidr="$2"
  aliyun ecs DescribeVSwitches \
    --RegionId "${REGION}" \
    --VpcId "${VPC_ID}" \
    --ZoneId "${zone}" \
    | python3 -c 'import json,sys
data=json.load(sys.stdin)
cidr=sys.argv[1]
for item in data.get("VSwitches", {}).get("VSwitch", []):
    if item.get("CidrBlock") == cidr:
        print(item.get("VSwitchId", ""))
        break' "${cidr}"
}

ensure_vswitch() {
  local zone="$1"
  local cidr="$2"
  local name="$3"
  local existing
  existing="$(find_vswitch_by_cidr "${zone}" "${cidr}")"
  if [[ -n "${existing}" ]]; then
    log "found vSwitch ${existing} for ${zone} ${cidr}"
    printf '%s\n' "${existing}"
    return 0
  fi
  log "will create vSwitch ${name} in ${zone} ${cidr}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    run_write aliyun ecs CreateVSwitch \
      --RegionId "${REGION}" \
      --VpcId "${VPC_ID}" \
      --ZoneId "${zone}" \
      --CidrBlock "${cidr}" \
      --VSwitchName "${name}" \
      --ClientToken "${name}-20260825"
    printf 'DRY_RUN_VSWITCH_%s\n' "${zone##*-}"
    return 0
  fi
  run_write aliyun ecs CreateVSwitch \
    --RegionId "${REGION}" \
    --VpcId "${VPC_ID}" \
    --ZoneId "${zone}" \
    --CidrBlock "${cidr}" \
    --VSwitchName "${name}" \
    --ClientToken "${name}-20260825" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["VSwitchId"])'
}

authorize_rule() {
  local proto="$1"
  local port_range="$2"
  local description="$3"
  local output
  log "ensuring security-group rule ${proto} ${port_range} from 10.0.0.0/16 (${description})"
  printf '+ aliyun ecs AuthorizeSecurityGroup --RegionId %q --SecurityGroupId %q --IpProtocol %q --PortRange %q --SourceCidrIp 10.0.0.0/16 --Policy accept --Description %q\n' \
    "${REGION}" "${SECURITY_GROUP_ID}" "${proto}" "${port_range}" "${description}" >&2
  if [[ "${DRY_RUN}" == "true" ]]; then
    return 0
  fi
  if ! output="$(aliyun ecs AuthorizeSecurityGroup \
      --RegionId "${REGION}" \
      --SecurityGroupId "${SECURITY_GROUP_ID}" \
      --IpProtocol "${proto}" \
      --PortRange "${port_range}" \
      --SourceCidrIp 10.0.0.0/16 \
      --Policy accept \
      --Description "${description}" 2>&1)"; then
    if grep -q 'InvalidPermission.Duplicate' <<<"${output}"; then
      log "security-group rule already exists"
      return 0
    fi
    printf '%s\n' "${output}" >&2
    return 1
  fi
}

find_instance_by_name() {
  local name="$1"
  aliyun ecs DescribeInstances \
    --RegionId "${REGION}" \
    --VpcId "${VPC_ID}" \
    --InstanceName "${name}" \
    | python3 -c 'import json,sys
name=sys.argv[1]
for item in json.load(sys.stdin).get("Instances", {}).get("Instance", []):
    if item.get("InstanceName") == name:
        print(item.get("InstanceId", ""))
        break' "${name}"
}

ensure_instance() {
  local name="$1"
  local zone="$2"
  local vswitch_id="$3"
  local existing
  existing="$(find_instance_by_name "${name}")"
  if [[ -n "${existing}" ]]; then
    log "found existing ECS ${name}: ${existing}"
    return 0
  fi

  local cmd=(
    aliyun ecs RunInstances
    --RegionId "${REGION}"
    --ZoneId "${zone}"
    --InstanceType "${INSTANCE_TYPE}"
    --ImageId "${IMAGE_ID}"
    --SecurityGroupId "${SECURITY_GROUP_ID}"
    --VSwitchId "${vswitch_id}"
    --InstanceName "${name}"
    --HostName "${name}"
    --Amount 1
    --MinAmount 1
    --InstanceChargeType PostPaid
    --InternetMaxBandwidthOut "${INTERNET_MAX_BANDWIDTH_OUT}"
    --SystemDisk.Category cloud_essd
    --SystemDisk.Size "${SYSTEM_DISK_SIZE}"
    --SystemDisk.PerformanceLevel PL0
    --KeyPairName "${KEY_PAIR_NAME}"
    --Tag.1.Key seedleap-role
    --Tag.1.Value zing-shanghai-vae
    --Tag.2.Key lingjun-cluster-id
    --Tag.2.Value "${LINGJUN_CLUSTER_ID}"
    --ClientToken "${name}-20260825"
  )
  if [[ "${SPOT_STRATEGY}" != "NoSpot" ]]; then
    cmd+=(--SpotStrategy "${SPOT_STRATEGY}")
    if [[ -n "${SPOT_PRICE_LIMIT}" ]]; then
      cmd+=(--SpotPriceLimit "${SPOT_PRICE_LIMIT}")
    fi
  fi

  log "will create ECS ${name} in ${zone} using ${INSTANCE_TYPE}"
  run_write "${cmd[@]}"
}

main() {
  require_write_confirmation
  log "dry-run: ${DRY_RUN}"
  log "target VPC: ${VPC_ID}; security group: ${SECURITY_GROUP_ID}"

  authorize_rule tcp 6443/6443 "k3s api inside world_model_vpc"
  authorize_rule udp 8472/8472 "k3s flannel vxlan inside world_model_vpc"
  authorize_rule tcp 10250/10250 "k3s kubelet metrics inside world_model_vpc"

  vsw1="$(ensure_vswitch "${VAE_ZONE_1}" "${VAE_CIDR_1}" "${VAE_NAME_1}-vsw")"
  vsw2="$(ensure_vswitch "${VAE_ZONE_2}" "${VAE_CIDR_2}" "${VAE_NAME_2}-vsw")"

  ensure_instance "${VAE_NAME_1}" "${VAE_ZONE_1}" "${vsw1}"
  ensure_instance "${VAE_NAME_2}" "${VAE_ZONE_2}" "${vsw2}"

  log "done. Join the two ECS nodes to k3s with install_k3s_vae_agent.sh after the server token is available."
}

main "$@"
