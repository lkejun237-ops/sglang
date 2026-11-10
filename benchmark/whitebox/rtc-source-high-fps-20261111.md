# Source capture 720p high-rate CPU validation (2026-11-11)

Scope: remove the H.264 Level 3.1 admission gate from source capture rates. Base
SGLang `0710ad66d2` (`origin/main` when work began); task branch
`codex/rtc-source-high-fps-20261111`. No GPU allocation, deployment, model,
Coordinator or active source was touched. No SDK pin was changed.

## Environment and observations

- macOS arm64, Python 3.14.10, PyAV 17.1.0, libavcodec 62.28.101.
- SVT-AV1 4.1.0; software libx264 / libvpx-vp9 / libsvtav1 encoders.
- Official Go 1.27.3 darwin/arm64 archive checked against the go.dev release
  manifest SHA256; native sender built from this checkout. Shared Python
  environment was reused without installing or changing its dependencies.
- Real 1280×720 at both 36 and 60 FPS: three emitted AUs decoded into three
  correctly sized frames for each codec. Source timing and encoder rate were
  supplied explicitly. This is an ability check, not a throughput benchmark.
- H.264 actual SPS profile/level: `42c020` (Level 3.2), obtained from libx264,
  not a rewritten Level 3.1 label. Default source settings and explicit H.264
  both use actual SPS for native source negotiation when outside the old envelope.
- Fresh encoded AUs for each rate passed existing native TCP ICE/DTLS/SRTP
  negotiation and elementary-byte roundtrip, including explicit session codecs
  and dynamic H.264 profile (8 subcases per rate). These use localhost Pion peers,
  not a browser or a production Room.
- TDD: 13 new assertions failed on the old rate gates before the production
  changes. Lifecycle tests cover initial/default and explicit admission,
  managed default source preparation, frozen profile reuse before encoder
  readiness, immutable overrides, authorization and reconnect cleanup.
- Review regression: managed requests may omit both resolution and encoding
  while the deployment defaults to 1280×720. Two new cases reproduced missing
  preflight before the fix (explicit request FPS60 and omitted FPS inheriting
  environment60). They now freeze one probe/SPS before creating one source and
  reuse it for source.bind; omitted 480p defaults still perform no probe.
  Follow-up regression run: 185 passed across managed control, RTC lifecycle
  and session encoding; four unchanged real-binary preflight cases skipped
  because the already-validated temporary native binary had been cleaned.
  Ruff and diff whitespace checks passed.

Validation results: **297 Python tests passed**, one existing native-x264 bridge
integration skipped because that optional library is not installed; no changed
codec test was skipped. Four real native+software preflight cases passed. Both
fresh fixture TCP runs and the full native Go suite passed. Ruff passed with
only the pre-existing test E731 rule excluded; `git diff --check` passed.

## Reproduction

Use an isolated environment with the project's lightweight runtime dependencies,
PyAV software codecs and Go. Run from the repository root; replace the example
paths with private temporary paths. Generated media and binaries stay outside Git.

```bash
export PYTHONPATH=python
export SGLANG_LIGHTWEIGHT_RUNTIME=1
python benchmark/whitebox/rtc_source_high_fps.py --output-dir /tmp/rtc-high-fps/fixtures
(cd python/sglang/multimodal_gen/runtime/realtime/whitebox/rtc_sender && go build -o /tmp/rtc-high-fps/whitebox-rtc .)
export WHITEBOX_TEST_RTC_BINARY=/tmp/rtc-high-fps/whitebox-rtc
pytest --confcutdir=python/sglang/multimodal_gen/test/unit/realtime -q \
  python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_session_encoding.py \
  python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_resolution.py \
  python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_player_control.py \
  python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_source_encoding.py \
  python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_rtc_lifecycle.py \
  python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_native_rtc.py \
  python/sglang/multimodal_gen/test/unit/realtime/test_whitebox_error_origins.py
(cd python/sglang/multimodal_gen/runtime/realtime/whitebox/rtc_sender && \
  WHITEBOX_SOURCE_CODEC_FIXTURE=/tmp/rtc-high-fps/fixtures/source-720p-36.json \
  go test -count=1 -run TestSourceCodecsNegotiateAndCarryElementaryAUOverTCP .)
(cd python/sglang/multimodal_gen/runtime/realtime/whitebox/rtc_sender && \
  WHITEBOX_SOURCE_CODEC_FIXTURE=/tmp/rtc-high-fps/fixtures/source-720p-60.json \
  go test -count=1 -run TestSourceCodecsNegotiateAndCarryElementaryAUOverTCP .)
(cd python/sglang/multimodal_gen/runtime/realtime/whitebox/rtc_sender && go test -count=1 ./...)
```

The existing probe limits (5-second native capabilities, 15-second real codec
subprocess), concurrency bound, native media budgets and GOP 1–24 policy remain.
The GOP cap is a separate audited restriction, not changed by this task.

Model geometry, model bitrate, immediate native model encoder selection and its
fixed bridge settings remain unchanged. No new 480/736 GPU timing, memory or
model-effect evidence was collected; browser sustained FPS, long sessions,
production codec libraries and deployed acceptance still require their own checks.
Source capture at 60 FPS does not establish model output or playback at 60 FPS.
