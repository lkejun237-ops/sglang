# Normalize native artifact permissions on a complete candidate Room image.
# This target does not copy or replace application or SDK source. Preserve the
# candidate's release manifest and record the input image in the build receipt.
ARG ROOM_IMAGE
FROM ${ROOM_IMAGE} AS room
USER root
RUN chmod 0755 /app/python/sglang/multimodal_gen/runtime/realtime/whitebox/native/libwhitebox_x264.so && \
    chmod 0644 /app/WHITEBOX_CODEC_VERSION
USER whitebox
RUN --network=none python -c 'from sglang.multimodal_gen.runtime.realtime.whitebox.native_capture import library; library(); from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import RenderHost'
