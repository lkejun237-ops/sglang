#!/usr/bin/env bash
# Run on an existing K3s GPU node through its authorized command channel.
# SOURCE_ARCHIVE_B64 is a git archive of TOOL_SHA, not a runtime source patch.
# This creates only one temporary Pod and ConfigMap. After retrieving logs,
# remove those exact names and verify the selected GPU was released.
set -euo pipefail
: "${POD_NAME:?}" "${GPU_UUID:?}" "${IMAGE:?}" "${TOOL_SHA:?}"
: "${RUNTIME_SHA:?}" "${SOURCE_ARCHIVE_B64:?}"
NAMESPACE=${NAMESPACE:-minwm-realtime}

test "$(nvidia-smi --id="$GPU_UUID" --query-gpu=memory.used --format=csv,noheader,nounits)" -eq 0
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fqx "$GPU_UUID"; then
    echo 'Selected GPU is in use' >&2
    exit 1
fi
# Check CRI's lookup, not just a running container's historical imageID: image
# aliases can disappear while an existing container keeps its snapshot lease.
k3s crictl --config /dev/null \
    --runtime-endpoint unix:///run/k3s/containerd/containerd.sock \
    --image-endpoint unix:///run/k3s/containerd/containerd.sock inspecti "$IMAGE" >/dev/null
# Never pull a new image just for this short collector test.
kubectl -n "$NAMESPACE" create configmap "$POD_NAME-source" \
    --from-literal=archive="$SOURCE_ARCHIVE_B64"
trap 'kubectl -n "$NAMESPACE" delete configmap "$POD_NAME-source" --ignore-not-found' ERR
cat <<EOF | kubectl -n "$NAMESPACE" create -f -
apiVersion: v1
kind: Pod
metadata:
  name: $POD_NAME
  labels:
    experiment: zing-component-metrics-probe
spec:
  nodeName: world-model-b27mys7b-0000
  runtimeClassName: nvidia
  restartPolicy: Never
  activeDeadlineSeconds: 300
  containers:
    - name: probe
      image: $IMAGE
      imagePullPolicy: Never
      env:
        - {name: NVIDIA_VISIBLE_DEVICES, value: "$GPU_UUID"}
        - {name: NVIDIA_DRIVER_CAPABILITIES, value: "compute,utility"}
        - {name: SGLANG_LIGHTWEIGHT_RUNTIME, value: "1"}
        - {name: OMP_NUM_THREADS, value: "1"}
        - {name: TOOL_SHA, value: "$TOOL_SHA"}
        - {name: RUNTIME_SHA, value: "$RUNTIME_SHA"}
      resources:
        requests: {cpu: "1", memory: 1Gi}
        limits: {cpu: "2", memory: 4Gi}
      volumeMounts:
        - {name: source, mountPath: /probe-input, readOnly: true}
      command: [/bin/bash, -ec]
      args:
        - |
          /opt/sglang/bin/python - <<'PY'
          import base64,io,tarfile,pathlib
          root=pathlib.Path('/tmp/zing-metrics-probe');root.mkdir()
          data=base64.b64decode(pathlib.Path('/probe-input/archive').read_bytes())
          with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
              archive.extractall(root,filter='data')
          PY
          export PYTHONPATH=/tmp/zing-metrics-probe/python
          cd /tmp/zing-metrics-probe
          for mode in native multiprocess; do
            if [ "\$mode" = multiprocess ]; then
              export PROMETHEUS_MULTIPROC_DIR=/tmp/zing-metrics-probe/prom-mmap
              mkdir "\$PROMETHEUS_MULTIPROC_DIR"
            else
              unset PROMETHEUS_MULTIPROC_DIR
            fi
            /opt/sglang/bin/python benchmark/zing_h3_causal/bench_component_metrics.py \
              --device cuda --runtime-sha "\$RUNTIME_SHA" --tool-sha "\$TOOL_SHA" \
              --output "/tmp/zing-metrics-probe/\$mode.json"
          done
          /opt/sglang/bin/python - <<'PY'
          import base64,gzip,json,pathlib
          root=pathlib.Path('/tmp/zing-metrics-probe')
          results={mode:json.loads((root/(mode+'.json')).read_text()) for mode in ('native','multiprocess')}
          print('PROBE_RESULT_BASE64='+base64.b64encode(gzip.compress(json.dumps(results).encode())).decode(),flush=True)
          PY
          echo COMPONENT_METRICS_PROBE_PASS
  volumes:
    - name: source
      configMap: {name: $POD_NAME-source}
EOF
trap - ERR
kubectl -n "$NAMESPACE" get pod "$POD_NAME" -o jsonpath='{.metadata.uid}{"\n"}'
