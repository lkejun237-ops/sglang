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
      diffusers==0.41.0 huggingface_hub==1.33.0 \
      mooncake-transfer-engine==0.3.13 boto3==1.43.111
# Install the hash-locked prebuilt binary seedleap-kernels wheel before copying
# SGLang source so routine Python changes reuse this cached image layer.
ARG KERNELS_WHEEL_FIND_LINKS=https://zing-cdn.loopit.com.cn/releases/seedleap-kernels/simple/seedleap-kernels/index.html
COPY deploy/whitebox/kernels-wheel.lock.txt /opt/whitebox/deploy/whitebox/kernels-wheel.lock.txt
RUN python -m pip install --no-cache-dir --no-index --no-deps --force-reinstall --only-binary=:all: --require-hashes \
      --find-links "$KERNELS_WHEEL_FIND_LINKS" \
      -r /opt/whitebox/deploy/whitebox/kernels-wheel.lock.txt
# The pinned Mooncake wheel links CUDA 12 even in a CUDA 13 model image. Keep
# its runtime in a separate directory; do not replace Torch's CUDA 13 package.
RUN python -m pip install --no-cache-dir --no-deps --index-url "$PIP_INDEX_URL" \
      --target /opt/mooncake-cuda12 nvidia-cuda-runtime-cu12==12.8.90
RUN python -c 'from pathlib import Path; Path("/etc/ld.so.conf.d/mooncake-cuda12.conf").write_text("/opt/mooncake-cuda12/nvidia/cuda_runtime/lib\n")' && ldconfig
COPY python/sglang /opt/whitebox/python/sglang
COPY deploy/whitebox /opt/whitebox/deploy/whitebox
COPY benchmark/zing_h3_causal/check_native_cache_tiers.py /opt/whitebox/benchmark/zing_h3_causal/check_native_cache_tiers.py
COPY benchmark/zing_h3_causal/cache_validation_service.py benchmark/zing_h3_causal/worker_group_master.py benchmark/zing_h3_causal/check_cache_oss.py /opt/whitebox/benchmark/zing_h3_causal/
RUN python /opt/whitebox/deploy/whitebox/verify_cache_dependencies.py --allow-missing-driver
ARG WHITEBOX_SOURCE_SHA=unmanaged
ARG WHITEBOX_SOURCE_BRANCH=main
ARG WHITEBOX_SDK_SHA=unmanaged
RUN python -c 'import json,sys; json.dump(dict(schema_version=1,source_branch=sys.argv[4],source_sha=sys.argv[1],sdk_sha=sys.argv[2]),open(sys.argv[3],"w"))' "$WHITEBOX_SOURCE_SHA" "$WHITEBOX_SDK_SHA" /opt/whitebox/release.json "$WHITEBOX_SOURCE_BRANCH"
ENV PYTHONPATH=/opt/whitebox/python \
    WHITEBOX_RELEASE_MANIFEST=/opt/whitebox/release.json \
    SGLANG_LIGHTWEIGHT_RUNTIME=1 \
    SEEDLEAP_KERNELS_DISABLE_JIT=1
RUN python -c 'import pathlib, torch, transformers, zmq, sglang; assert pathlib.Path(sglang.__file__).resolve() == pathlib.Path("/opt/whitebox/python/sglang/__init__.py"); print(torch.__version__)'
RUN python -c 'from sglang.multimodal_gen.runtime.realtime.source_transport import SourceTransportBudget; SourceTransportBudget().websocket_options()'
# --help stops before the Decoder loads Lynn. Import its real implementation so
# an older dependency base cannot defer this failure until GPUs are allocated.
RUN python -c 'import diffusers; assert diffusers.__version__ == "0.41.0"; import sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.loader'
# Verify the installed prebuilt binary seedleap-kernels wheel matches the lockfile
# and ABI contract (no runtime nvcc/JIT compilation permitted in production).
RUN python /opt/whitebox/deploy/whitebox/verify_model_kernels.py \
      --manifest /opt/whitebox/release.json \
      --lockfile /opt/whitebox/deploy/whitebox/kernels-wheel.lock.txt
# Model weights, regional bundle, logs, conditioning secrets and local cache are
# mounted independently. The builder does not read or alter any of them.
ENTRYPOINT ["python", "/opt/whitebox/deploy/whitebox/model_service.py"]
