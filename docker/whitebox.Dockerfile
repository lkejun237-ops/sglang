FROM node:24-bookworm-slim AS node

FROM python:3.14-slim-bookworm AS base
ARG PIP_INDEX_URL
ENV PYTHONUNBUFFERED=1 SGLANG_LIGHTWEIGHT_RUNTIME=1 PYTHONPATH=/app/python
WORKDIR /app
COPY benchmark/whitebox/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN useradd --uid 10001 --create-home whitebox && mkdir -p /var/lib/sglang/whitebox && chown -R whitebox:whitebox /var/lib/sglang
USER whitebox
EXPOSE 18080 18081 18089
ENTRYPOINT ["python", "/app/run_service.py"]

FROM base AS immediate-encoder-build
USER root
RUN --mount=type=bind,source=deploy/whitebox/debian_main_mirror.py,target=/tmp/debian_main_mirror.py,readonly \
    python /tmp/debian_main_mirror.py -- sh -ec 'sed -i '"'"'s/^Types: deb$/Types: deb deb-src/'"'"' /etc/apt/sources.list.d/debian.sources && \
        apt-get update && \
        apt-get install -y --no-install-recommends gcc libc6-dev make nasm pkg-config dpkg-dev libx264-dev && \
        mkdir /tmp/x264-source && \
        cd /tmp/x264-source && \
        apt-get source "x264=$(dpkg-query -W -f='"'"'${source:Version}'"'"' libx264-dev)" && \
        cd x264-* && \
        ./configure --prefix=/opt/whitebox-codec --enable-static --enable-pic --disable-cli --disable-opencl && \
        make -j2 && \
        make install'
COPY python/sglang/multimodal_gen/runtime/realtime/whitebox/native/x264_bridge.c /tmp/x264_bridge.c
RUN cc -O2 -Wall -Wextra -Werror -fPIC -shared /tmp/x264_bridge.c \
      -o /tmp/libwhitebox_x264.so -I/opt/whitebox-codec/include \
      -Wl,--exclude-libs,ALL /opt/whitebox-codec/lib/libx264.a \
      -lm -lpthread -ldl && \
    PKG_CONFIG_PATH=/opt/whitebox-codec/lib/pkgconfig pkg-config --modversion x264 > /tmp/whitebox-codec-version.txt

FROM golang:1.25-bookworm AS rtc-builder
ARG GOPROXY
WORKDIR /build
COPY python/sglang/multimodal_gen/runtime/realtime/whitebox/rtc_sender /build
RUN go test ./... && CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags='-s -w' -o /tmp/whitebox-rtc .

# A test-only edit can rerun this cheap projection. Its unchanged output is the
# runtime COPY input; the complete test tree remains a separate checks input.
FROM base AS runtime-source
# Bind-mounted inputs do not invalidate every supported builder cache.
# Pin projection to the same source revision used by the release manifest.
ARG WHITEBOX_SOURCE_SHA=unmanaged
USER root
RUN --mount=type=bind,source=python/sglang,target=/source/sglang,readonly \
    --mount=type=bind,source=deploy/whitebox/project_runtime_source.py,target=/tmp/project_runtime_source.py,readonly \
    test -n "$WHITEBOX_SOURCE_SHA" && \
    python /tmp/project_runtime_source.py --source /source/sglang --destination /runtime-source/sglang

FROM base AS runtime
COPY --from=runtime-source /runtime-source/sglang /app/python/sglang
COPY benchmark/whitebox/run_service.py /app/run_service.py
USER root
ARG WHITEBOX_SOURCE_SHA=unmanaged
ARG WHITEBOX_SOURCE_BRANCH=main
ARG WHITEBOX_SDK_SHA=unmanaged
RUN python -c 'import json,sys; json.dump(dict(schema_version=1,source_branch=sys.argv[4],source_sha=sys.argv[1],sdk_sha=sys.argv[2]),open(sys.argv[3],"w"))' "$WHITEBOX_SOURCE_SHA" "$WHITEBOX_SDK_SHA" /app/release.json "$WHITEBOX_SOURCE_BRANCH"
ENV WHITEBOX_RELEASE_MANIFEST=/app/release.json
USER whitebox


# Every published image exercises the actual CPU control-plane contracts.
# Test dependencies stay in this intermediate stage.
FROM runtime AS checks
USER root
ARG WHITEBOX_TEST_ALLOW_DEVELOPMENT_RELEASE=0
ENV WHITEBOX_ALLOW_DEVELOPMENT_RELEASE=$WHITEBOX_TEST_ALLOW_DEVELOPMENT_RELEASE
# Frozen publication bridge contracts exercise their actual Node wire encoder.
# Keep this test dependency out of the Python-only control-plane runtime.
COPY --from=node /usr/local/bin/node /usr/local/bin/node
RUN node --version
# Disposable Redis is a checks-only dependency, never part of the runtime.
RUN --mount=type=bind,source=deploy/whitebox/debian_main_mirror.py,target=/tmp/debian_main_mirror.py,readonly \
    python /tmp/debian_main_mirror.py -- sh -ec 'apt-get update && apt-get install -y --no-install-recommends redis-server && rm -rf /var/lib/apt/lists/*'
COPY python/sglang/multimodal_gen/test /app/python/sglang/multimodal_gen/test
COPY docker/whitebox.Dockerfile /app/docker/whitebox.Dockerfile
COPY deploy/whitebox /app/deploy/whitebox
RUN pip install --no-cache-dir pytest==9.0.2 pyzmq==27.2.0 && \
    python -m pytest \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_async_archive.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_room_pipeline.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_player_control.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_player_idle.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_capture_wait.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_control_prefill.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_condition_refresh.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_prewarm_plan.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_startup_prewarm.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_room_control.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_startup.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_startup_retirement.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_player_room_lifecycle.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_failed_member_retirement.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_prefetch.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_prefill_gateway.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_runtime_management.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_native_rtc.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_rtc_lifecycle.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_rtc_pipeline.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_interactive.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_recording_process.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_compositor.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_rooms.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_room_registry.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_native.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_client_recording.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_realtime_coordinator.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_realtime_coordinator_server.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_realtime_gateway_server.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_realtime_trace.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_world_gateway_metrics.py \
      --confcutdir=python/sglang/multimodal_gen/test/unit/realtime -q && \
    python -m pytest \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_parallel_startup.py \
      --confcutdir=python/sglang/multimodal_gen/test/unit/realtime -q && \
    python -m pytest deploy/whitebox/test_prepare_assets.py \
      deploy/whitebox/test_room_instance.py \
      deploy/whitebox/test_gpu_guard.py \
      deploy/whitebox/test_encoder_peer.py \
      deploy/whitebox/test_debian_main_mirror.py \
      deploy/whitebox/test_runtime_source_projection.py \
      deploy/whitebox/test_runtime_image_contract.py \
      --confcutdir=deploy/whitebox -q && \
    touch /tmp/whitebox-contracts-passed

FROM runtime AS release
COPY --from=checks /tmp/whitebox-contracts-passed /app/whitebox-contracts-passed

# CI checks out the exact revision from benchmark/whitebox/sdk-revision.txt.
# Credentials are used only by Git checkout, never copied into this context.
FROM node AS sdk
RUN corepack enable && corepack prepare pnpm@10.14.0 --activate
WORKDIR /opt/worldkit
COPY .whitebox-sdk/ ./
RUN pnpm install --frozen-lockfile --ignore-scripts && rm -rf asset-library/tests/fixtures/atlas-recording

FROM base AS room-process-init
USER root
# Chromium and compiler descendants can outlive their direct CLI parent.
# PID 1 must reap those orphans without consuming asyncio-owned child exits.
RUN --mount=type=bind,source=deploy/whitebox/debian_main_mirror.py,target=/tmp/debian_main_mirror.py,readonly \
    python /tmp/debian_main_mirror.py -- sh -ec 'apt-get update && \
        apt-get install -y --no-install-recommends tini && \
        rm -rf /var/lib/apt/lists/*'
USER whitebox
ENTRYPOINT ["/usr/bin/tini", "--", "python", "/app/run_service.py"]

FROM room-process-init AS room-browser
USER root
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules/npm /opt/npm
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/playwright
# GPU devices and matching driver libraries are supplied by the container runtime.
# These are userspace loaders only; never install or replace a host NVIDIA driver.
RUN --mount=type=bind,source=deploy/whitebox/debian_main_mirror.py,target=/tmp/debian_main_mirror.py,readonly \
    python /tmp/debian_main_mirror.py -- sh -ec 'apt-get update && \
        apt-get install -y --no-install-recommends ffmpeg libvulkan1 vulkan-tools libegl1 libglx0 libgl1 xvfb libgl1-mesa-dri libglx-mesa0 libegl-mesa0 && \
        rm -rf /var/lib/apt/lists/*'
# Sealed multiplayer admission re-probes every recorded segment through ffprobe.
# Exercise the SDK's exact metadata fields before publishing a Room image.
RUN ffmpeg -nostdin -hide_banner -loglevel error -f lavfi -i color=black:s=16x16:r=1 \
      -frames:v 1 -c:v libx264 -pix_fmt yuv420p /tmp/whitebox-media-check.mp4 && \
    ffprobe -v error -select_streams v:0 -count_frames \
      -show_entries stream=width,height,nb_read_frames:format=duration -of json /tmp/whitebox-media-check.mp4 > /tmp/whitebox-media-check.json && \
    python -c 'import json; v=json.load(open("/tmp/whitebox-media-check.json")); s=v["streams"][0]; assert (s["width"],s["height"],int(s["nb_read_frames"])) == (16,16,1); assert float(v["format"]["duration"]) > 0' && \
    rm /tmp/whitebox-media-check.mp4 /tmp/whitebox-media-check.json
# Keep this version aligned with the pinned SDK's resolved Playwright version.
ARG PLAYWRIGHT_DOWNLOAD_HOST
RUN --mount=type=bind,source=deploy/whitebox/debian_main_mirror.py,target=/tmp/debian_main_mirror.py,readonly \
    python /tmp/debian_main_mirror.py -- sh -ec 'node /opt/npm/bin/npx-cli.js --yes --package=playwright@1.62.1 playwright install --with-deps chromium && \
        chmod -R a+rX /opt/playwright && \
        rm -rf /opt/npm /root/.npm /var/lib/apt/lists/*'

FROM room-browser AS room-runtime
COPY --from=sdk /opt/worldkit /opt/worldkit
COPY --from=runtime-source /runtime-source/sglang /app/python/sglang
COPY --from=immediate-encoder-build /tmp/libwhitebox_x264.so /app/python/sglang/multimodal_gen/runtime/realtime/whitebox/native/libwhitebox_x264.so
COPY --from=immediate-encoder-build /tmp/whitebox-codec-version.txt /app/WHITEBOX_CODEC_VERSION
COPY --from=rtc-builder /tmp/whitebox-rtc /app/whitebox-rtc
COPY benchmark/whitebox/run_service.py /app/run_service.py
USER root
ARG WHITEBOX_SOURCE_SHA=unmanaged
ARG WHITEBOX_SOURCE_BRANCH=main
ARG WHITEBOX_SDK_SHA=unmanaged
RUN python -c 'import json,sys; json.dump(dict(schema_version=1,source_branch=sys.argv[4],source_sha=sys.argv[1],sdk_sha=sys.argv[2]),open(sys.argv[3],"w"))' "$WHITEBOX_SOURCE_SHA" "$WHITEBOX_SDK_SHA" /app/release.json "$WHITEBOX_SOURCE_BRANCH"
ENV WHITEBOX_RELEASE_MANIFEST=/app/release.json
USER whitebox
COPY --from=checks /tmp/whitebox-contracts-passed /app/whitebox-contracts-passed
ENV WHITEBOX_CLI_ENTRYPOINT=/opt/worldkit/packages/cli/bin/worldkit.mjs \
    WHITEBOX_ROOM_SPOOL=/var/lib/sglang/whitebox/rooms \
    WHITEBOX_ROLE=room
USER whitebox
EXPOSE 8092

FROM room-runtime AS room-checks
USER root
ARG WHITEBOX_TEST_ALLOW_DEVELOPMENT_RELEASE=0
ENV WHITEBOX_ALLOW_DEVELOPMENT_RELEASE=$WHITEBOX_TEST_ALLOW_DEVELOPMENT_RELEASE
COPY python/sglang/multimodal_gen/test /app/python/sglang/multimodal_gen/test
# The CLI fixture uses ps to track and reap the browser and encoder processes.
RUN --mount=type=bind,source=deploy/whitebox/debian_main_mirror.py,target=/tmp/debian_main_mirror.py,readonly \
    python /tmp/debian_main_mirror.py -- sh -ec 'apt-get update && \
        apt-get install -y --no-install-recommends procps && rm -rf /var/lib/apt/lists/*'
RUN pip install --no-cache-dir pytest==9.0.2 pyzmq==27.2.0
# The sealed cutscene regression generates and verifies media with the Room's FFmpeg and Chromium.
RUN --mount=type=bind,source=.whitebox-sdk/asset-library/tests/fixtures/atlas-recording,target=/opt/worldkit/asset-library/tests/fixtures/atlas-recording,readonly \
    --mount=type=tmpfs,target=/tmp/atlas-fixtures \
    cd /opt/worldkit && export WORLDKIT_PLAYGROUND_CACHE_ROOT=/tmp/atlas-fixtures XDG_RUNTIME_DIR=/tmp/room-check-xdg && \
    install -d -m 0700 "$XDG_RUNTIME_DIR" && \
    node asset-library/tools/prepare-test-fixtures.mjs && node node_modules/vitest/vitest.mjs run \
      packages/stream-host/tests/capture-hold.test.ts \
      packages/stream-host/tests/raw-producer.test.ts packages/stream-host/tests/raw-session.test.ts
USER whitebox
# A real sealed world, two real native-pixel CLI processes, no server UI compositor.
# All HTTP, Coordinator, H3 and archive dependencies are local fixtures.
RUN WHITEBOX_TEST_CLI_SDK=/opt/worldkit python -m pytest \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_cli_integration.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_runtime_management.py \
      --confcutdir=python/sglang/multimodal_gen/test/unit/realtime -p no:cacheprovider -q && \
    WHITEBOX_TEST_CLI_SDK=/opt/worldkit python -m pytest \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_room_registry.py \
      -k native_cli_keeps_running \
      --confcutdir=python/sglang/multimodal_gen/test/unit/realtime -p no:cacheprovider -q && \
    touch /tmp/whitebox-cli-passed
RUN WHITEBOX_REQUIRE_NATIVE_X264=1 python -m pytest \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_immediate_capture.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_capture_clock.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_stream_archive.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_independent_archive.py \
      python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_independent_resolution.py \
      --confcutdir=python/sglang/multimodal_gen/test/unit/realtime -p no:cacheprovider -q && \
    touch /tmp/whitebox-immediate-codec-passed

FROM room-runtime AS room
COPY --from=room-checks /tmp/whitebox-cli-passed /app/whitebox-cli-passed
COPY --from=room-checks /tmp/whitebox-immediate-codec-passed /app/whitebox-immediate-codec-passed
COPY deploy/whitebox/cli-gpu-preflight.mjs /app/cli-gpu-preflight.mjs
