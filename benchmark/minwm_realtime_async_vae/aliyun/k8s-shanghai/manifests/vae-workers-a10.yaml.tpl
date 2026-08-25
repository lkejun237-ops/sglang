apiVersion: apps/v1
kind: Deployment
metadata:
  name: zing-vae-a10
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/name: zing-vae-a10
    app.kubernetes.io/part-of: minwm-realtime
spec:
  replicas: 1
  revisionHistoryLimit: 3
  progressDeadlineSeconds: 1800
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 0
  selector:
    matchLabels:
      app.kubernetes.io/name: zing-vae-a10
  template:
    metadata:
      annotations:
        seedleap.ai/runtime-source-patch: ${RUNTIME_SOURCE_PATCH_VERSION}
      labels:
        app.kubernetes.io/name: zing-vae-a10
        app.kubernetes.io/part-of: minwm-realtime
        seedleap.ai/worker-role: vae
        seedleap.ai/gpu-topology: 1p1-local-vae
    spec:
      serviceAccountName: zing-realtime
      runtimeClassName: nvidia
      terminationGracePeriodSeconds: 90
      nodeSelector:
        seedleap.ai/gpu-pool: ${VAE_GPU_POOL}
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector:
                matchLabels:
                  app.kubernetes.io/name: zing-vae-a10
              topologyKey: kubernetes.io/hostname
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      initContainers:
        - name: vae-heartbeat
          image: ${GPU_RUNTIME_IMAGE}
          imagePullPolicy: IfNotPresent
          restartPolicy: Always
          command: [/bin/sh, -ec]
          args:
            - >-
              exec python3 -m sglang.multimodal_gen.runtime.entrypoints.realtime_worker_heartbeat
              --coordinator-url=http://zing-coordinator:18081
              --health-url=http://127.0.0.1:18082/health
              --state-url=http://127.0.0.1:18082/v1/realtime_worker/state
              --worker-id=$(POD_UID)
              --worker-epoch-file=/var/run/minwm-worker/epoch
              --role=vae
              --endpoint=ws://$(POD_IP):18082/v1/realtime_vae/decode
              --reservation-endpoint=http://$(POD_IP):18082/v1/realtime_worker
              --node-name=$(NODE_NAME)
              --capacity=${VAE_WORKER_CAPACITY}
              --model-revision=all
              --vae-fingerprint=taew2_2-d053e216
              --interval-s=5
          env:
            - {name: PYTHONPATH, value: /opt/sglang/python}
            - {name: SGLANG_LIGHTWEIGHT_RUNTIME, value: "1"}
            - {name: NVIDIA_VISIBLE_DEVICES, value: void}
            - name: POD_UID
              valueFrom: {fieldRef: {fieldPath: metadata.uid}}
            - name: POD_IP
              valueFrom: {fieldRef: {fieldPath: status.podIP}}
            - name: NODE_NAME
              valueFrom: {fieldRef: {fieldPath: spec.nodeName}}
          resources:
            requests: {cpu: 25m, memory: 64Mi}
            limits: {cpu: 200m, memory: 256Mi}
          volumeMounts:
            - {name: worker-epoch, mountPath: /var/run/minwm-worker}
      containers:
        - name: vae
          image: ${GPU_RUNTIME_IMAGE}
          imagePullPolicy: IfNotPresent
          command: [python3]
          args:
            - -m
            - sglang.multimodal_gen.runtime.entrypoints.realtime_vae_server
            - --decoder-backend=taehv
            - --checkpoint-path=/models/taehv/taew2_2.pth
            - --device=cuda
            - --dtype=bfloat16
            - --max-sessions=${VAE_WORKER_CAPACITY}
            - --queue-depth-per-session=1
            - --encoded-frames-per-batch=1
            - --max-message-mb=64
            - --host=0.0.0.0
            - --port=18082
            - --direct-h264-output
            - --direct-h264-trigger-output-format=jpeg
            - --h264-ffmpeg-bin=${H264_FFMPEG_BIN}
            - --h264-fps=${REALTIME_TARGET_FPS}
            - --h264-bitrate-kbps=3000
            - --h264-crf=20
            - --h264-preset=fast
            - --h264-gop-seconds=2
            - --h264-vbv-buffer-ms=250
            - --h264-max-frame-age-ms=250
          env:
            - {name: PYTHONUNBUFFERED, value: "1"}
            - {name: PYTHONPATH, value: /opt/sglang/python}
            - {name: SGLANG_LIGHTWEIGHT_RUNTIME, value: "1"}
            - {name: NVIDIA_VISIBLE_DEVICES, value: "${VAE_NVIDIA_VISIBLE_DEVICES}"}
            - {name: NVIDIA_DRIVER_CAPABILITIES, value: "compute,utility"}
            - {name: WORKER_EPOCH_FILE, value: /var/run/minwm-worker/epoch}
            - {name: WORLD_MODEL_METRIC_SERVICE, value: vae}
            - {name: WORLD_MODEL_METRIC_MODEL, value: minwm}
            - {name: WORLD_MODEL_METRIC_LANE, value: aliyun-shanghai}
          ports:
            - {name: vae, containerPort: 18082}
          startupProbe:
            httpGet: {path: /health, port: vae}
            periodSeconds: 5
            timeoutSeconds: 5
            failureThreshold: 180
          readinessProbe:
            httpGet: {path: /health, port: vae}
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 6
          livenessProbe:
            httpGet: {path: /health, port: vae}
            initialDelaySeconds: 30
            periodSeconds: 30
            timeoutSeconds: 5
            failureThreshold: 6
          resources:
            requests: {cpu: "6", memory: 24Gi}
            limits: {cpu: "8", memory: 32Gi}
          volumeMounts:
            - {name: worker-epoch, mountPath: /var/run/minwm-worker}
            - {name: taehv, mountPath: /models/taehv, readOnly: true}
            - {name: runtime-realtime-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/runtime/realtime/async_vae_client.py, subPath: async_vae_client.py, readOnly: true}
            - {name: runtime-realtime-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/runtime/realtime/async_vae_protocol.py, subPath: async_vae_protocol.py, readOnly: true}
            - {name: runtime-realtime-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/runtime/realtime/async_vae_worker.py, subPath: async_vae_worker.py, readOnly: true}
            - {name: runtime-realtime-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/runtime/realtime/async_vae_metrics.py, subPath: async_vae_metrics.py, readOnly: true}
            - {name: runtime-realtime-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/runtime/realtime/h264_media_pipeline.py, subPath: h264_media_pipeline.py, readOnly: true}
            - {name: runtime-realtime-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/runtime/realtime/worker_reservation.py, subPath: worker_reservation.py, readOnly: true}
            - {name: runtime-realtime-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/runtime/realtime/exact_vae_backend.py, subPath: exact_vae_backend.py, readOnly: true}
            - {name: runtime-realtime-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/runtime/realtime/critical_path_metrics.py, subPath: critical_path_metrics.py, readOnly: true}
            - {name: runtime-entrypoint-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/runtime/entrypoints/realtime_vae_server.py, subPath: realtime_vae_server.py, readOnly: true}
            - {name: runtime-utils-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/runtime/utils/logging_utils.py, subPath: logging_utils.py, readOnly: true}
            - {name: runtime-utils-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/runtime/utils/realtime_trace.py, subPath: realtime_trace.py, readOnly: true}
            - {name: runtime-utils-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/runtime/utils/realtime_video.py, subPath: realtime_video.py, readOnly: true}
            - {name: runtime-envs-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/envs.py, subPath: envs.py, readOnly: true}
            - {name: runtime-dep-shims, mountPath: /opt/sglang/python/prometheus_client.py, subPath: prometheus_client.py, readOnly: true}
            - {name: runtime-dep-shims, mountPath: /opt/sglang/python/sitecustomize.py, subPath: sitecustomize.py, readOnly: true}
      volumes:
        - {name: worker-epoch, emptyDir: {}}
        - {name: runtime-realtime-patch, configMap: {name: zing-runtime-realtime-patch}}
        - {name: runtime-entrypoint-patch, configMap: {name: zing-runtime-entrypoint-patch}}
        - {name: runtime-utils-patch, configMap: {name: zing-runtime-utils-patch}}
        - {name: runtime-envs-patch, configMap: {name: zing-runtime-envs-patch}}
        - {name: runtime-dep-shims, configMap: {name: zing-runtime-dep-shims}}
        - name: taehv
          hostPath:
            path: /data/zing-realtime/taehv
            type: Directory
