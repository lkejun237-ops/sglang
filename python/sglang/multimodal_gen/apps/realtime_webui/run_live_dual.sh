#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUAL_GATEWAY_HTTP="${REALTIME_DUAL_GATEWAY_HTTP:-http://k8s-minwmrea-zingling-081cdcf247-5c21747e962f7bd4.elb.us-east-2.amazonaws.com}"

if [[ "${DUAL_GATEWAY_HTTP}" == https://* ]]; then
  DEFAULT_GATEWAY_WS="wss://${DUAL_GATEWAY_HTTP#https://}"
else
  DEFAULT_GATEWAY_WS="ws://${DUAL_GATEWAY_HTTP#http://}"
fi
DUAL_GATEWAY_WS="${REALTIME_DUAL_GATEWAY_WS:-${DEFAULT_GATEWAY_WS}}"

export WEBUI_PORT="${WEBUI_PORT:-18083}"
export REALTIME_UPSTREAM_HTTP="${DUAL_GATEWAY_HTTP%/}"
export REALTIME_UPSTREAM_WS="${DUAL_GATEWAY_WS%/}"
export MINWM_UPSTREAM_HTTP="${DUAL_GATEWAY_HTTP%/}/backends/minwm"
export MINWM_UPSTREAM_WS="${DUAL_GATEWAY_WS%/}/backends/minwm"
export LINGBOT2_UPSTREAM_HTTP="${DUAL_GATEWAY_HTTP%/}/backends/lingbot2"
export LINGBOT2_UPSTREAM_WS="${DUAL_GATEWAY_WS%/}/backends/lingbot2"
export VIDEO_PROMPT_REWRITE_PROVIDER="${VIDEO_PROMPT_REWRITE_PROVIDER:-doubao}"
if [[ -z "${VIDEO_PROMPT_REWRITE_MODEL:-}" ]]; then
  if [[ "${VIDEO_PROMPT_REWRITE_PROVIDER}" == "doubao" ]]; then
    export VIDEO_PROMPT_REWRITE_MODEL="doubao-seed-2-0-mini-260428"
  else
    export VIDEO_PROMPT_REWRITE_MODEL="gemini-3.1-flash-lite"
  fi
fi
if [[ -z "${CREATE_WORLD_DESCRIPTION_PROVIDER:-}" ]]; then
  if [[ "${VIDEO_PROMPT_REWRITE_PROVIDER}" == "local" ]]; then
    export CREATE_WORLD_DESCRIPTION_PROVIDER="vertex"
  else
    export CREATE_WORLD_DESCRIPTION_PROVIDER="${VIDEO_PROMPT_REWRITE_PROVIDER}"
  fi
fi
if [[ -z "${CREATE_WORLD_DESCRIPTION_MODEL:-}" ]]; then
  if [[ "${CREATE_WORLD_DESCRIPTION_PROVIDER}" == "${VIDEO_PROMPT_REWRITE_PROVIDER}" ]]; then
    export CREATE_WORLD_DESCRIPTION_MODEL="${VIDEO_PROMPT_REWRITE_MODEL}"
  elif [[ "${CREATE_WORLD_DESCRIPTION_PROVIDER}" == "doubao" ]]; then
    export CREATE_WORLD_DESCRIPTION_MODEL="doubao-seed-2-0-mini-260428"
  else
    export CREATE_WORLD_DESCRIPTION_MODEL="gemini-3.1-flash-lite"
  fi
fi
export DOUBAO_ARK_CREDENTIALS="${DOUBAO_ARK_CREDENTIALS:-${SCRIPT_DIR}/../realtime_webui_secrets/doubao-ark.json}"
if [[ -z "${REALTIME_UI_CONFIG_JSON:-}" ]]; then
  export REALTIME_UI_CONFIG_JSON="{\"generationModes\":[\"i2v\",\"t2v\"],\"defaultGenerationMode\":\"i2v\",\"t2vFrameStep\":4,\"t2vDefaultNumFrames\":121,\"sessionMaxLifetimeSeconds\":90,\"playbackAckEnabled\":false,\"h264WebSocketEnabled\":true,\"h264WebSocketBaseUrl\":\"https://zing-world-studio.loopit.me\",\"h264CompressedBitrateKbps\":3000,\"h264CompressedCrf\":20,\"h264CompressedPreset\":\"fast\",\"h264CompressedGopSeconds\":2,\"h264CompressedVbvBufferMs\":250,\"h264WebSocketLiveEdgeTargetMs\":80,\"h264WebSocketSeekThresholdMs\":260,\"singleExperience\":false,\"smoothCatchupRateMax\":1.1,\"dualModels\":{\"minwm\":{\"label\":\"Zing\",\"sinkSize\":8,\"windowFrames\":32,\"h264StartupDropFrames\":0},\"lingbot2\":{\"label\":\"LingBot2\",\"targetFps\":16,\"sinkSize\":9,\"windowFrames\":18,\"h264StartupDropFrames\":8}}}"
fi

exec python3 "${SCRIPT_DIR}/server.py"
