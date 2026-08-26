#!/usr/bin/env bash
set -Eeuo pipefail

REGION="${REGION:-cn-shanghai}"
LINGJUN_CLUSTER_ID="${LINGJUN_CLUSTER_ID:-i119114381787650669980}"
VPC_ID="${VPC_ID:-vpc-uf6bmsqo2f8jwk6o6me6n}"
VSWITCH_ID="${VSWITCH_ID:-vsw-uf6pd2zo6sbf8pfldxnz3}"
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-sg-uf65itprxseo4lm028h9}"
ZONE_ID="${ZONE_ID:-cn-shanghai-l}"
KEY_PAIR_NAME="${KEY_PAIR_NAME:-world-model}"

INSTANCE_NAME="${INSTANCE_NAME:-zing-shanghai-edge-caddy}"
INSTANCE_TYPE="${INSTANCE_TYPE:-ecs.e-c1m2.large}"
IMAGE_ID="${IMAGE_ID:-aliyun_3_x64_20G_alibase_20260720.vhd}"
SYSTEM_DISK_SIZE="${SYSTEM_DISK_SIZE:-40}"
SYSTEM_DISK_CATEGORY="${SYSTEM_DISK_CATEGORY:-cloud_essd}"
SYSTEM_DISK_PERFORMANCE_LEVEL="${SYSTEM_DISK_PERFORMANCE_LEVEL:-PL0}"
INTERNET_MAX_BANDWIDTH_OUT="${INTERNET_MAX_BANDWIDTH_OUT:-0}"
CLIENT_TOKEN="${CLIENT_TOKEN:-zing-shanghai-edge-caddy-20260826}"

DRY_RUN="${DRY_RUN:-true}"
CONFIRM_ALIYUN_WRITES="${CONFIRM_ALIYUN_WRITES:-}"
REQUIRED_CONFIRMATION="create-shanghai-edge-caddy-ecs"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

print_cmd() {
  printf '+' >&2
  local arg
  for arg in "$@"; do
    printf ' %q' "${arg}" >&2
  done
  printf '\n' >&2
}

require_write_confirmation() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    return
  fi
  if [[ "${CONFIRM_ALIYUN_WRITES}" != "${REQUIRED_CONFIRMATION}" ]]; then
    cat >&2 <<EOF
Refusing to run Aliyun ECS writes.

Set:
  DRY_RUN=false CONFIRM_ALIYUN_WRITES=${REQUIRED_CONFIRMATION}

This creates or changes:
  - One pay-as-you-go CPU ECS named ${INSTANCE_NAME}
  - Region/zone: ${REGION}/${ZONE_ID}
  - VPC/vSwitch/security group: ${VPC_ID}/${VSWITCH_ID}/${SECURITY_GROUP_ID}
  - Instance type/image: ${INSTANCE_TYPE}/${IMAGE_ID}
  - System disk: ${SYSTEM_DISK_SIZE} GiB ${SYSTEM_DISK_CATEGORY} ${SYSTEM_DISK_PERFORMANCE_LEVEL}
  - Public bandwidth: ${INTERNET_MAX_BANDWIDTH_OUT}; the existing EIP is moved separately after Caddy is configured.
EOF
    exit 2
  fi
}

find_instance_by_name() {
  aliyun ecs DescribeInstances \
    --RegionId "${REGION}" \
    --VpcId "${VPC_ID}" \
    --InstanceName "${INSTANCE_NAME}" \
    | python3 -c 'import json,sys
name=sys.argv[1]
for item in json.load(sys.stdin).get("Instances", {}).get("Instance", []):
    if item.get("InstanceName") == name and item.get("Status") != "Deleted":
        print(item.get("InstanceId", ""))
        break' "${INSTANCE_NAME}"
}

create_instance() {
  local existing
  existing="$(find_instance_by_name)"
  if [[ -n "${existing}" ]]; then
    log "found existing ECS ${INSTANCE_NAME}: ${existing}"
    printf '%s\n' "${existing}"
    return 0
  fi

  local cmd=(
    aliyun ecs RunInstances
    --RegionId "${REGION}"
    --ZoneId "${ZONE_ID}"
    --InstanceType "${INSTANCE_TYPE}"
    --ImageId "${IMAGE_ID}"
    --SecurityGroupId "${SECURITY_GROUP_ID}"
    --VSwitchId "${VSWITCH_ID}"
    --InstanceName "${INSTANCE_NAME}"
    --HostName "${INSTANCE_NAME}"
    --Amount 1
    --MinAmount 1
    --InstanceChargeType PostPaid
    --InternetMaxBandwidthOut "${INTERNET_MAX_BANDWIDTH_OUT}"
    --SystemDisk.Category "${SYSTEM_DISK_CATEGORY}"
    --SystemDisk.Size "${SYSTEM_DISK_SIZE}"
    --SystemDisk.PerformanceLevel "${SYSTEM_DISK_PERFORMANCE_LEVEL}"
    --KeyPairName "${KEY_PAIR_NAME}"
    --Tag.1.Key seedleap-role
    --Tag.1.Value zing-shanghai-edge
    --Tag.2.Key lingjun-cluster-id
    --Tag.2.Value "${LINGJUN_CLUSTER_ID}"
    --ClientToken "${CLIENT_TOKEN}"
  )

  if [[ "${DRY_RUN}" == "true" ]]; then
    cmd+=(--DryRun true)
    print_cmd "${cmd[@]}"
    local output status
    set +e
    output="$("${cmd[@]}" 2>&1)"
    status=$?
    set -e
    if [[ "${output}" == *"DryRunOperation"* ]]; then
      log "ECS create dry-run passed"
      return 0
    fi
    printf '%s\n' "${output}" >&2
    return "${status}"
  fi

  print_cmd "${cmd[@]}"
  "${cmd[@]}" | python3 -c 'import json,sys
data=json.load(sys.stdin)
ids=data.get("InstanceIdSets", {}).get("InstanceIdSet", [])
print(ids[0] if ids else data.get("InstanceId", ""))'
}

wait_instance_running() {
  local instance_id="$1"
  if [[ "${DRY_RUN}" == "true" || -z "${instance_id}" ]]; then
    return
  fi
  local status
  for _ in {1..60}; do
    status="$(aliyun ecs DescribeInstances --RegionId "${REGION}" --InstanceIds "[\"${instance_id}\"]" \
      | python3 -c 'import json,sys
items=json.load(sys.stdin).get("Instances", {}).get("Instance", [])
print(items[0].get("Status", "") if items else "")')"
    log "ECS ${instance_id} status: ${status}"
    if [[ "${status}" == "Running" ]]; then
      return
    fi
    if [[ "${status}" == "Stopped" ]]; then
      aliyun ecs StartInstance --RegionId "${REGION}" --InstanceId "${instance_id}" >/dev/null
    fi
    sleep 5
  done
  printf 'Timed out waiting for ECS %s to become Running.\n' "${instance_id}" >&2
  return 1
}

print_summary() {
  local instance_id="$1"
  if [[ "${DRY_RUN}" == "true" || -z "${instance_id}" ]]; then
    return
  fi
  aliyun ecs DescribeInstances \
    --RegionId "${REGION}" \
    --InstanceIds "[\"${instance_id}\"]" \
    | python3 -c 'import json,sys
for item in json.load(sys.stdin).get("Instances", {}).get("Instance", []):
    private_ips=item.get("VpcAttributes", {}).get("PrivateIpAddress", {}).get("IpAddress", [])
    print("InstanceId:", item.get("InstanceId", ""))
    print("InstanceName:", item.get("InstanceName", ""))
    print("PrivateIp:", private_ips[0] if private_ips else "")
    print("Status:", item.get("Status", ""))
    print("ZoneId:", item.get("ZoneId", ""))'
}

main() {
  require_write_confirmation
  log "dry-run: ${DRY_RUN}"
  local instance_id
  instance_id="$(create_instance)"
  wait_instance_running "${instance_id}"
  print_summary "${instance_id}"
}

main "$@"
