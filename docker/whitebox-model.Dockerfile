# Supply a verified CUDA/torch/kernel dependency image by digest at manual build.
# Its old SGLang checkout is never used as the source of this release.
ARG MODEL_BASE_IMAGE
FROM ${MODEL_BASE_IMAGE} AS model
USER root
WORKDIR /opt/whitebox
# Keep the bounded realtime WebSocket backend inside the repository's supported
# ranges even when the CUDA dependency image ships newer HTTP packages.
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN python -m pip install --no-cache-dir --index-url "$PIP_INDEX_URL" \
      uvicorn==0.39.0 websockets==15.0.1 \
      diffusers==0.41.0 huggingface_hub==1.33.0
COPY python/sglang /opt/whitebox/python/sglang
COPY deploy/whitebox /opt/whitebox/deploy/whitebox
ARG WHITEBOX_SOURCE_SHA=unmanaged
ARG WHITEBOX_SOURCE_BRANCH=main
ARG WHITEBOX_SDK_SHA=unmanaged
RUN python -c 'import json,sys; json.dump(dict(schema_version=1,source_branch=sys.argv[4],source_sha=sys.argv[1],sdk_sha=sys.argv[2]),open(sys.argv[3],"w"))' "$WHITEBOX_SOURCE_SHA" "$WHITEBOX_SDK_SHA" /opt/whitebox/release.json "$WHITEBOX_SOURCE_BRANCH"
ENV PYTHONPATH=/opt/whitebox/python \
    WHITEBOX_RELEASE_MANIFEST=/opt/whitebox/release.json \
    SGLANG_LIGHTWEIGHT_RUNTIME=1
RUN python -c 'import pathlib, torch, transformers, zmq, sglang; assert pathlib.Path(sglang.__file__).resolve() == pathlib.Path("/opt/whitebox/python/sglang/__init__.py"); print(torch.__version__)'
RUN python -c 'from sglang.multimodal_gen.runtime.realtime.source_transport import SourceTransportBudget; SourceTransportBudget().websocket_options()'
# --help stops before the Decoder loads Lynn. Import its real implementation so
# an older dependency base cannot defer this failure until GPUs are allocated.
RUN python -c 'import diffusers; assert diffusers.__version__ == "0.41.0"; import sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.loader'
# Model weights, regional bundle, logs, conditioning secrets and local cache are
# mounted independently. The builder does not read or alter any of them.
ENTRYPOINT ["python", "/opt/whitebox/deploy/whitebox/model_service.py"]
