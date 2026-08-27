apiVersion: v1
kind: Namespace
metadata:
  name: ${NAMESPACE}
  labels:
    app.kubernetes.io/part-of: minwm-realtime
    seedleap.ai/environment: aliyun-shanghai-zing
---
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: nvidia
handler: nvidia
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: zing-realtime
  namespace: ${NAMESPACE}
imagePullSecrets:
  - name: acr-pull
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: zing-realtime-node-reader
rules:
  - apiGroups: [""]
    resources: ["nodes"]
    verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: zing-realtime-node-reader-${NAMESPACE}
subjects:
  - kind: ServiceAccount
    name: zing-realtime
    namespace: ${NAMESPACE}
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: zing-realtime-node-reader
---
apiVersion: v1
kind: Service
metadata:
  name: zing-redis
  namespace: ${NAMESPACE}
spec:
  type: ClusterIP
  selector:
    app.kubernetes.io/name: zing-redis
  ports:
    - name: redis
      port: 6379
      targetPort: redis
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zing-redis
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/name: zing-redis
    app.kubernetes.io/part-of: minwm-realtime
spec:
  replicas: 1
  revisionHistoryLimit: 2
  strategy:
    type: Recreate
  selector:
    matchLabels:
      app.kubernetes.io/name: zing-redis
  template:
    metadata:
      labels:
        app.kubernetes.io/name: zing-redis
        app.kubernetes.io/part-of: minwm-realtime
    spec:
      nodeSelector:
        ${CONTROL_NODE_LABEL_KEY}: "${CONTROL_NODE_LABEL_VALUE}"
      terminationGracePeriodSeconds: 30
      containers:
        - name: redis
          image: ${REDIS_IMAGE}
          imagePullPolicy: IfNotPresent
          command: [redis-server]
          args:
            - --appendonly
            - "yes"
            - --dir
            - /data
            - --save
            - "60"
            - "1"
          ports:
            - {name: redis, containerPort: 6379}
          startupProbe:
            tcpSocket: {port: redis}
            periodSeconds: 2
            failureThreshold: 60
          readinessProbe:
            tcpSocket: {port: redis}
            periodSeconds: 5
          livenessProbe:
            tcpSocket: {port: redis}
            periodSeconds: 15
          resources:
            requests: {cpu: 100m, memory: 128Mi}
            limits: {cpu: "1", memory: 1Gi}
          volumeMounts:
            - {name: redis-data, mountPath: /data}
      volumes:
        - name: redis-data
          hostPath:
            path: /data/zing-realtime/redis
            type: DirectoryOrCreate
---
apiVersion: v1
kind: Service
metadata:
  name: zing-coordinator
  namespace: ${NAMESPACE}
spec:
  type: ClusterIP
  selector:
    app.kubernetes.io/name: zing-coordinator
  ports:
    - name: http
      port: 18081
      targetPort: http
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zing-coordinator
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/name: zing-coordinator
    app.kubernetes.io/part-of: minwm-realtime
spec:
  replicas: ${COORDINATOR_REPLICAS}
  revisionHistoryLimit: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app.kubernetes.io/name: zing-coordinator
  template:
    metadata:
      annotations:
        seedleap.ai/runtime-source-patch: ${RUNTIME_SOURCE_PATCH_VERSION}
      labels:
        app.kubernetes.io/name: zing-coordinator
        app.kubernetes.io/part-of: minwm-realtime
    spec:
      serviceAccountName: zing-realtime
      nodeSelector:
        ${CONTROL_PLANE_NODE_LABEL_KEY}: "${CONTROL_PLANE_NODE_LABEL_VALUE}"
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: kubernetes.io/hostname
          whenUnsatisfiable: ScheduleAnyway
          labelSelector:
            matchLabels:
              app.kubernetes.io/name: zing-coordinator
      terminationGracePeriodSeconds: 30
      containers:
        - name: coordinator
          image: ${CONTROL_IMAGE}
          imagePullPolicy: IfNotPresent
          command: [/bin/sh, -ec]
          args:
            - >-
              exec python3 -m sglang.multimodal_gen.runtime.entrypoints.realtime_coordinator_server
              --host=0.0.0.0
              --port=18081
              --backend=redis
              --redis-host=${REDIS_HOST}
              --redis-port=${REDIS_PORT}
              --redis-username=${REDIS_USERNAME}
              --redis-password-env=REDIS_PASSWORD
              --redis-db=${REDIS_DB}
              --redis-key-prefix="${REDIS_KEY_PREFIX}"
              --redis-state-ttl-s=${REDIS_STATE_TTL_SECONDS}
              --redis-lock-timeout-s=${REDIS_LOCK_TIMEOUT_SECONDS}
              ${REDIS_SSL_ARG}
              --ttl-s=30
              --worker-ttl-s=15
              --wait-timeout-s=10
              --candidate-limit=64
              --denoiser-capacity-limit=${DENOISER_CAPACITY_LIMIT}
              --vae-capacity-limit=${VAE_CAPACITY_LIMIT}
          env:
            - {name: PYTHONUNBUFFERED, value: "1"}
            - {name: PYTHONPATH, value: /opt/sglang/python}
            - {name: SGLANG_LIGHTWEIGHT_RUNTIME, value: "1"}
            - {name: OTEL_SERVICE_NAME, value: zing-coordinator}
            - name: REDIS_PASSWORD
              valueFrom:
                secretKeyRef: {name: zing-redis, key: password, optional: true}
          ports:
            - {name: http, containerPort: 18081}
          startupProbe:
            httpGet: {path: /healthz, port: http}
            periodSeconds: 2
            failureThreshold: 60
          readinessProbe:
            httpGet: {path: /healthz, port: http}
            periodSeconds: 5
          livenessProbe:
            httpGet: {path: /healthz, port: http}
            periodSeconds: 15
          resources:
            requests: {cpu: 250m, memory: 256Mi}
            limits: {cpu: "2", memory: 2Gi}
          volumeMounts:
            - {name: runtime-realtime-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/runtime/realtime, readOnly: true}
            - {name: runtime-entrypoint-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/runtime/entrypoints/realtime_coordinator_server.py, subPath: realtime_coordinator_server.py, readOnly: true}
            - {name: runtime-utils-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/runtime/utils, readOnly: true}
            - {name: runtime-envs-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/envs.py, subPath: envs.py, readOnly: true}
            - {name: runtime-dep-shims, mountPath: /opt/sglang/python/prometheus_client.py, subPath: prometheus_client.py, readOnly: true}
            - {name: runtime-dep-shims, mountPath: /opt/sglang/python/sitecustomize.py, subPath: sitecustomize.py, readOnly: true}
      volumes:
        - {name: runtime-realtime-patch, configMap: {name: zing-runtime-realtime-patch}}
        - {name: runtime-entrypoint-patch, configMap: {name: zing-runtime-entrypoint-patch}}
        - {name: runtime-utils-patch, configMap: {name: zing-runtime-utils-patch}}
        - {name: runtime-envs-patch, configMap: {name: zing-runtime-envs-patch}}
        - {name: runtime-dep-shims, configMap: {name: zing-runtime-dep-shims}}
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: zing-coordinator
  namespace: ${NAMESPACE}
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: zing-coordinator
---
apiVersion: v1
kind: Service
metadata:
  name: zing-gateway
  namespace: ${NAMESPACE}
spec:
  type: NodePort
  selector:
    app.kubernetes.io/name: zing-gateway
  ports:
    - name: http
      port: 18080
      targetPort: http
      nodePort: 30080
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zing-gateway
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/name: zing-gateway
    app.kubernetes.io/part-of: minwm-realtime
spec:
  replicas: ${GATEWAY_REPLICAS}
  revisionHistoryLimit: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app.kubernetes.io/name: zing-gateway
  template:
    metadata:
      annotations:
        seedleap.ai/runtime-source-patch: ${RUNTIME_SOURCE_PATCH_VERSION}
      labels:
        app.kubernetes.io/name: zing-gateway
        app.kubernetes.io/part-of: minwm-realtime
    spec:
      serviceAccountName: zing-realtime
      nodeSelector:
        ${CONTROL_PLANE_NODE_LABEL_KEY}: "${CONTROL_PLANE_NODE_LABEL_VALUE}"
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: kubernetes.io/hostname
          whenUnsatisfiable: ScheduleAnyway
          labelSelector:
            matchLabels:
              app.kubernetes.io/name: zing-gateway
      terminationGracePeriodSeconds: 30
      initContainers:
        - name: install-world-platform-dependencies
          image: ${CONTROL_IMAGE}
          imagePullPolicy: IfNotPresent
          command: [python3, -m, pip, install]
          args:
            - --disable-pip-version-check
            - --no-cache-dir
            - "--only-binary=:all:"
            - --index-url=https://mirrors.aliyun.com/pypi/simple/
            - --target=/opt/sglang/runtime-deps
            - cryptography==46.0.7
          volumeMounts:
            - {name: runtime-python-deps, mountPath: /opt/sglang/runtime-deps}
      containers:
        - name: gateway
          image: ${CONTROL_IMAGE}
          imagePullPolicy: IfNotPresent
          command: [/bin/sh, -ec]
          args:
            - >-
              exec python3 -m sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server
              --host=0.0.0.0
              --port=18080
              --coordinator-url=http://zing-coordinator:18081
              --model-revision=${MODEL_REVISION}
              --lingbot2-model-revision=${LINGBOT2_MODEL_REVISION}
              --vae-fingerprint=taew2_2-d053e216
              --lingbot2-vae-fingerprint=${LINGBOT2_VAE_FINGERPRINT}
              --internal-output-url=ws://${POD_IP}:18080/v1/internal/realtime_output
              --output-queue-depth=64
              --output-enqueue-timeout-s=0
              --output-drain-timeout-s=70
              --lease-renew-interval-s=10
              --release-grace-s=0.5
              --max-admission-waiters=64
              --ui-config-json="$RUNTIME_UI_CONFIG_JSON"
              --world-studio-webui-base-url="$WORLD_STUDIO_WEBUI_BASE_URL"
          env:
            - {name: PYTHONUNBUFFERED, value: "1"}
            - {name: PYTHONPATH, value: /opt/sglang/runtime-deps:/opt/sglang/python}
            - {name: SGLANG_LIGHTWEIGHT_RUNTIME, value: "1"}
            - {name: OTEL_SERVICE_NAME, value: zing-gateway}
            - {name: WORLD_STUDIO_WEBUI_BASE_URL, value: "${WORLD_STUDIO_WEBUI_BASE_URL}"}
            - name: POD_IP
              valueFrom:
                fieldRef:
                  fieldPath: status.podIP
            - name: RUNTIME_UI_CONFIG_JSON
              value: '${UI_CONFIG_JSON}'
          ports:
            - name: http
              containerPort: 18080
          startupProbe:
            httpGet: {path: /healthz, port: http}
            periodSeconds: 2
            failureThreshold: 60
          readinessProbe:
            httpGet: {path: /healthz, port: http}
            periodSeconds: 5
          livenessProbe:
            httpGet: {path: /healthz, port: http}
            periodSeconds: 15
          resources:
            requests: {cpu: 500m, memory: 512Mi}
            limits: {cpu: "4", memory: 4Gi}
          volumeMounts:
            - {name: runtime-python-deps, mountPath: /opt/sglang/runtime-deps, readOnly: true}
            - {name: runtime-realtime-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/runtime/realtime, readOnly: true}
            - {name: runtime-entrypoint-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/runtime/entrypoints/realtime_gateway_server.py, subPath: realtime_gateway_server.py, readOnly: true}
            - {name: runtime-utils-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/runtime/utils, readOnly: true}
            - {name: runtime-envs-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/envs.py, subPath: envs.py, readOnly: true}
            - {name: runtime-dep-shims, mountPath: /opt/sglang/python/prometheus_client.py, subPath: prometheus_client.py, readOnly: true}
            - {name: runtime-dep-shims, mountPath: /opt/sglang/python/sitecustomize.py, subPath: sitecustomize.py, readOnly: true}
      volumes:
        - {name: runtime-python-deps, emptyDir: {sizeLimit: 64Mi}}
        - {name: runtime-realtime-patch, configMap: {name: zing-runtime-realtime-patch}}
        - {name: runtime-entrypoint-patch, configMap: {name: zing-runtime-entrypoint-patch}}
        - {name: runtime-utils-patch, configMap: {name: zing-runtime-utils-patch}}
        - {name: runtime-envs-patch, configMap: {name: zing-runtime-envs-patch}}
        - {name: runtime-dep-shims, configMap: {name: zing-runtime-dep-shims}}
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: zing-gateway
  namespace: ${NAMESPACE}
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: zing-gateway
---
apiVersion: v1
kind: Service
metadata:
  name: zing-webui
  namespace: ${NAMESPACE}
spec:
  type: ClusterIP
  selector:
    app.kubernetes.io/name: zing-webui
  ports:
    - name: http
      port: 18080
      targetPort: http
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zing-webui
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/name: zing-webui
    app.kubernetes.io/part-of: minwm-realtime
spec:
  replicas: 1
  revisionHistoryLimit: 3
  strategy:
    type: Recreate
  selector:
    matchLabels:
      app.kubernetes.io/name: zing-webui
  template:
    metadata:
      labels:
        app.kubernetes.io/name: zing-webui
        app.kubernetes.io/part-of: minwm-realtime
    spec:
      serviceAccountName: zing-realtime
      nodeSelector:
        ${CONTROL_NODE_LABEL_KEY}: "${CONTROL_NODE_LABEL_VALUE}"
      automountServiceAccountToken: false
      terminationGracePeriodSeconds: 10
      securityContext:
        runAsNonRoot: true
        runAsUser: 65532
        runAsGroup: 65532
        fsGroup: 65532
        fsGroupChangePolicy: OnRootMismatch
      containers:
        - name: webui
          image: ${WEBUI_IMAGE}
          imagePullPolicy: IfNotPresent
          command: [python]
          args:
            - /opt/sglang/python/sglang/multimodal_gen/apps/realtime_webui/server.py
          envFrom:
            - secretRef:
                name: zing-auth-secrets
                optional: true
          env:
            - {name: PYTHONUNBUFFERED, value: "1"}
            - {name: PYTHONPATH, value: /opt/sglang/python}
            - {name: WEBUI_PORT, value: "18080"}
            - {name: REALTIME_UPSTREAM_HTTP, value: "http://zing-gateway:18080"}
            - {name: REALTIME_UPSTREAM_WS, value: "ws://zing-gateway:18080"}
            - {name: MINWM_UPSTREAM_HTTP, value: "http://zing-gateway:18080/backends/minwm"}
            - {name: MINWM_UPSTREAM_WS, value: "ws://zing-gateway:18080/backends/minwm"}
            - {name: LINGBOT2_UPSTREAM_HTTP, value: ""}
            - {name: LINGBOT2_UPSTREAM_WS, value: ""}
            - {name: VIDEO_PROMPT_REWRITE_PROVIDER, value: ${VIDEO_PROMPT_REWRITE_PROVIDER}}
            - {name: VIDEO_PROMPT_REWRITE_TIMEOUT_SECONDS, value: "${VIDEO_PROMPT_REWRITE_TIMEOUT_SECONDS}"}
            - {name: CREATE_WORLD_DESCRIPTION_PROVIDER, value: ${CREATE_WORLD_DESCRIPTION_PROVIDER}}
            - {name: VIDEO_PROMPT_REWRITE_CREDENTIALS, value: /run/secrets/realtime-webui/prompt-rewriter-vertex.json}
            - {name: DOUBAO_ARK_CREDENTIALS, value: ${DOUBAO_ARK_CREDENTIALS}}
            - {name: DOUBAO_REQUEST_TIMEOUT_SECONDS, value: "${DOUBAO_REQUEST_TIMEOUT_SECONDS}"}
            - {name: CREATE_WORLD_IMAGE_CONFIG, value: ${CREATE_WORLD_IMAGE_CONFIG}}
            - {name: WEBUI_AUTH_ENABLED, value: "${WEBUI_AUTH_ENABLED}"}
            - {name: AUTH_PUBLIC_BASE_URL, value: "${AUTH_PUBLIC_BASE_URL}"}
            - {name: AUTH_COOKIE_NAME, value: "${AUTH_COOKIE_NAME}"}
            - {name: AUTH_REDIS_URL, value: "${AUTH_REDIS_URL}"}
            - {name: AUTH_REDIS_HOST, value: "${AUTH_REDIS_HOST}"}
            - {name: AUTH_REDIS_PORT, value: "${AUTH_REDIS_PORT}"}
            - {name: AUTH_REDIS_USERNAME, value: "${AUTH_REDIS_USERNAME}"}
            - {name: AUTH_REDIS_DB, value: "${AUTH_REDIS_DB}"}
            - {name: AUTH_REDIS_SSL, value: "${AUTH_REDIS_SSL}"}
            - {name: AUTH_REDIS_PASSWORD_ENV, value: AUTH_REDIS_PASSWORD}
            - name: AUTH_REDIS_PASSWORD
              valueFrom:
                secretKeyRef: {name: zing-redis, key: password, optional: true}
            - {name: AUTH_REDIS_KEY_PREFIX, value: "${AUTH_REDIS_KEY_PREFIX}"}
            - {name: AUTH_SESSION_TTL_SECONDS, value: "${AUTH_SESSION_TTL_SECONDS}"}
            - name: REALTIME_UI_CONFIG_JSON
              value: '${UI_CONFIG_JSON}'
            - {name: NO_PROXY, value: "${WEBUI_NO_PROXY}"}
            - {name: no_proxy, value: "${WEBUI_NO_PROXY}"}
            - {name: WORLD_MODEL_METRIC_SERVICE, value: world-studio-webui}
            - {name: WORLD_MODEL_METRIC_LANE, value: aliyun-shanghai}
          ports:
            - name: http
              containerPort: 18080
              hostPort: 80
          startupProbe:
            httpGet: {path: /healthz, port: http}
            periodSeconds: 2
            failureThreshold: 60
          readinessProbe:
            httpGet: {path: /healthz, port: http}
            periodSeconds: 5
          livenessProbe:
            httpGet: {path: /healthz, port: http}
            periodSeconds: 15
          resources:
            requests: {cpu: 250m, memory: 384Mi}
            limits: {cpu: "2", memory: 2Gi}
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities: {drop: [ALL]}
          volumeMounts:
            - {name: webui-secrets, mountPath: /run/secrets/realtime-webui, readOnly: true}
            - {name: generated-images, mountPath: /opt/sglang/python/sglang/multimodal_gen/apps/realtime_webui_generated}
            - {name: tmp, mountPath: /tmp}
      volumes:
        - name: webui-secrets
          secret:
            secretName: webui-secrets
            optional: true
        - {name: generated-images, emptyDir: {sizeLimit: 1Gi}}
        - {name: tmp, emptyDir: {sizeLimit: 512Mi}}
