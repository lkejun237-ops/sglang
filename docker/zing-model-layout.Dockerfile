# Add only the verified model layout to an immutable GPU dependency image.
# model-layout/ contains small metadata files and symlinks to read-only mounts;
# weights, credentials and serving source must not be included in this context.
# Serving source is independently verified by deploy/sglang/source.py at startup.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY --link model-layout/ /var/lib/sglang/weight-releases/
