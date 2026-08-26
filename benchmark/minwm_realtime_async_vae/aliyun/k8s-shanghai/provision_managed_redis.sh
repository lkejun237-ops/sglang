#!/usr/bin/env bash
set -Eeuo pipefail

REGION="${REGION:-cn-shanghai}"
VPC_ID="${VPC_ID:-vpc-uf6bmsqo2f8jwk6o6me6n}"
VSWITCH_ID="${VSWITCH_ID:-vsw-uf6pd2zo6sbf8pfldxnz3}"
ZONE_ID="${ZONE_ID:-cn-shanghai-l}"
INSTANCE_NAME="${INSTANCE_NAME:-zing-shanghai-realtime-coordinator}"
INSTANCE_CLASS="${INSTANCE_CLASS:-redis.master.stand.default}"
CAPACITY="${CAPACITY:-4096}"
ENGINE_VERSION="${ENGINE_VERSION:-5.0}"
NODE_TYPE="${NODE_TYPE:-double}"
CHARGE_TYPE="${CHARGE_TYPE:-PostPaid}"
NETWORK_TYPE="${NETWORK_TYPE:-VPC}"
PORT="${PORT:-6379}"
REDIS_SECURITY_IPS="${REDIS_SECURITY_IPS:-10.0.0.0/24}"
TOKEN="${TOKEN:-zing-shanghai-redis-20260826}"

DRY_RUN="${DRY_RUN:-true}"
CONFIRM_ALIYUN_WRITES="${CONFIRM_ALIYUN_WRITES:-}"
REQUIRED_CONFIRMATION="create-shanghai-managed-redis"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

print_cmd_redacted() {
  local redact_next=false
  local arg
  printf '+' >&2
  for arg in "$@"; do
    if [[ "${redact_next}" == "true" ]]; then
      printf ' %q' '***' >&2
      redact_next=false
      continue
    fi
    printf ' %q' "${arg}" >&2
    if [[ "${arg}" == "--Password" ]]; then
      redact_next=true
    fi
  done
  printf '\n' >&2
}

require_write_confirmation() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    return
  fi
  if [[ "${CONFIRM_ALIYUN_WRITES}" != "${REQUIRED_CONFIRMATION}" ]]; then
    cat >&2 <<EOF
Refusing to run Aliyun Redis writes.

Set:
  DRY_RUN=false CONFIRM_ALIYUN_WRITES=${REQUIRED_CONFIRMATION} REDIS_PASSWORD='<secret>'

This creates or changes:
  - A pay-as-you-go Redis Open-Source Edition instance named ${INSTANCE_NAME}
  - Region/zone: ${REGION}/${ZONE_ID}
  - VPC/vSwitch: ${VPC_ID}/${VSWITCH_ID}
  - Class/capacity: ${INSTANCE_CLASS}/${CAPACITY} MB
  - Port: ${PORT}
  - Default whitelist append: ${REDIS_SECURITY_IPS}

The password is read from REDIS_PASSWORD and is never printed.
EOF
    exit 2
  fi
  if [[ -z "${REDIS_PASSWORD:-}" ]]; then
    printf 'REDIS_PASSWORD is required for live Redis creation.\n' >&2
    exit 2
  fi
}

find_instance_id() {
  aliyun r-kvstore DescribeInstances \
    --RegionId "${REGION}" \
    --SearchKey "${INSTANCE_NAME}" \
    --PageSize 50 \
    | python3 -c 'import json,sys
name=sys.argv[1]
vpc=sys.argv[2]
for item in json.load(sys.stdin).get("Instances", {}).get("KVStoreInstance", []):
    if item.get("InstanceName") == name and item.get("VpcId") == vpc:
        print(item.get("InstanceId", ""))
        break' "${INSTANCE_NAME}" "${VPC_ID}"
}

describe_instance_field() {
  local instance_id="$1"
  local field="$2"
  aliyun r-kvstore DescribeInstances \
    --RegionId "${REGION}" \
    --InstanceIds "${instance_id}" \
    --PageSize 1 \
    | python3 -c 'import json,sys
items=json.load(sys.stdin).get("Instances", {}).get("KVStoreInstance", [])
print(items[0].get(sys.argv[1], "") if items else "")' "${field}"
}

create_instance() {
  local existing
  existing="$(find_instance_id)"
  if [[ -n "${existing}" ]]; then
    log "found existing Redis ${INSTANCE_NAME}: ${existing}"
    printf '%s\n' "${existing}"
    return 0
  fi

  local password="${REDIS_PASSWORD:-DryrunOnly123!}"
  local cmd=(
    aliyun r-kvstore CreateInstance
    --RegionId "${REGION}"
    --ZoneId "${ZONE_ID}"
    --VpcId "${VPC_ID}"
    --VSwitchId "${VSWITCH_ID}"
    --InstanceName "${INSTANCE_NAME}"
    --InstanceType Redis
    --EngineVersion "${ENGINE_VERSION}"
    --InstanceClass "${INSTANCE_CLASS}"
    --Capacity "${CAPACITY}"
    --NetworkType "${NETWORK_TYPE}"
    --ChargeType "${CHARGE_TYPE}"
    --NodeType "${NODE_TYPE}"
    --Port "${PORT}"
    --Appendonly yes
    --Password "${password}"
    --Token "${TOKEN}"
    --Tag.1.Key managed_by
    --Tag.1.Value codex
    --Tag.2.Key project
    --Tag.2.Value world-model
    --Tag.3.Key component
    --Tag.3.Value realtime-coordinator-state
    --Tag.4.Key region
    --Tag.4.Value "${REGION}"
  )

  if [[ "${DRY_RUN}" == "true" ]]; then
    cmd+=(--DryRun true)
    print_cmd_redacted "${cmd[@]}"
    local output status
    set +e
    output="$("${cmd[@]}" 2>&1)"
    status=$?
    set -e
    if [[ "${output}" == *"DryRunOperation"* ]]; then
      log "Redis create dry-run passed"
      return 0
    fi
    printf '%s\n' "${output}" >&2
    return "${status}"
  fi

  print_cmd_redacted "${cmd[@]}"
  "${cmd[@]}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["InstanceId"])'
}

wait_instance_normal() {
  local instance_id="$1"
  if [[ "${DRY_RUN}" == "true" || -z "${instance_id}" ]]; then
    return
  fi
  local status
  for _ in {1..90}; do
    status="$(describe_instance_field "${instance_id}" InstanceStatus)"
    log "Redis ${instance_id} status: ${status}"
    if [[ "${status}" == "Normal" ]]; then
      return
    fi
    if [[ "${status}" == "Error" || "${status}" == "Released" ]]; then
      printf 'Redis instance entered terminal bad status: %s\n' "${status}" >&2
      return 1
    fi
    sleep 10
  done
  printf 'Timed out waiting for Redis %s to become Normal.\n' "${instance_id}" >&2
  return 1
}

ensure_security_ips() {
  local instance_id="$1"
  if [[ "${DRY_RUN}" == "true" || -z "${instance_id}" ]]; then
    print_cmd_redacted aliyun r-kvstore ModifySecurityIps \
      --InstanceId "${instance_id:-<new-instance-id>}" \
      --SecurityIpGroupName default \
      --ModifyMode Append \
      --SecurityIps "${REDIS_SECURITY_IPS}"
    return
  fi

  local current
  current="$(aliyun r-kvstore DescribeSecurityIps --InstanceId "${instance_id}" \
    --RegionId "${REGION}" \
    | python3 -c 'import json,sys
for group in json.load(sys.stdin).get("SecurityIpGroups", {}).get("SecurityIpGroup", []):
    if group.get("SecurityIpGroupName") == "default":
        print(group.get("SecurityIpList", ""))
        break')"
  if tr ',' '\n' <<<"${current}" | grep -Fxq "${REDIS_SECURITY_IPS}"; then
    log "Redis whitelist already contains ${REDIS_SECURITY_IPS}"
    return
  fi
  print_cmd_redacted aliyun r-kvstore ModifySecurityIps \
    --RegionId "${REGION}" \
    --InstanceId "${instance_id}" \
    --SecurityIpGroupName default \
    --ModifyMode Append \
    --SecurityIps "${REDIS_SECURITY_IPS}"
  aliyun r-kvstore ModifySecurityIps \
    --RegionId "${REGION}" \
    --InstanceId "${instance_id}" \
    --SecurityIpGroupName default \
    --ModifyMode Append \
    --SecurityIps "${REDIS_SECURITY_IPS}" >/dev/null
}

print_summary() {
  local instance_id="$1"
  if [[ "${DRY_RUN}" == "true" || -z "${instance_id}" ]]; then
    return
  fi
  aliyun r-kvstore DescribeInstances \
    --RegionId "${REGION}" \
    --InstanceIds "${instance_id}" \
    --PageSize 1 \
    | python3 -c 'import json,sys
items=json.load(sys.stdin).get("Instances", {}).get("KVStoreInstance", [])
for item in items:
    print("InstanceId:", item.get("InstanceId", ""))
    print("InstanceName:", item.get("InstanceName", ""))
    print("ConnectionDomain:", item.get("ConnectionDomain", ""))
    print("PrivateIp:", item.get("PrivateIp", ""))
    print("Port:", item.get("Port", ""))
    print("Status:", item.get("InstanceStatus", ""))'
}

main() {
  require_write_confirmation
  log "dry-run: ${DRY_RUN}"
  local instance_id
  instance_id="$(create_instance)"
  wait_instance_normal "${instance_id}"
  ensure_security_ips "${instance_id}"
  print_summary "${instance_id}"
}

main "$@"
