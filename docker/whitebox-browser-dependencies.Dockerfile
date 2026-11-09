# Repack an already verified CPU Room artifact into a fresh dependency root.
# The final stage MUST NOT inherit the source image's deleted SDK/app layers.
# No application code is taken from this target: the ordinary whitebox.Dockerfile
# still copies the full selected main source and pinned SDK and runs gates.
ARG DEPENDENCY_SOURCE
FROM ${DEPENDENCY_SOURCE} AS dependency-files
USER root
WORKDIR /
RUN rm -rf /app /opt/worldkit /var/lib/sglang/whitebox /root/.cache /home/whitebox/.cache && \
    find /tmp /var/tmp -mindepth 1 -maxdepth 1 -exec rm -rf -- {} + && \
    test -x /usr/local/bin/python && test -x /usr/local/bin/node && \
    test -x /usr/bin/tini && test -d /opt/playwright && \
    test "$(id -u whitebox)" = 10001 && \
    python -c 'import importlib.util; assert importlib.util.find_spec("sglang") is None'

FROM scratch AS browser-root
COPY --from=dependency-files / /
ARG DEPENDENCY_SOURCE
LABEL com.seedleap.whitebox.dependency-kind="browser-root-v1" \
      com.seedleap.whitebox.dependency-source="${DEPENDENCY_SOURCE}"
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LANG=C.UTF-8 PYTHONUNBUFFERED=1 SGLANG_LIGHTWEIGHT_RUNTIME=1 \
    PYTHONPATH=/app/python PLAYWRIGHT_BROWSERS_PATH=/opt/playwright \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
USER whitebox
ENTRYPOINT ["/usr/bin/tini", "--", "python", "/app/run_service.py"]
