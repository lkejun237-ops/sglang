# Repository instructions

## World-platform error language

All error information owned by the Seedleap world-platform integration must be
English. This rule is mandatory for `authorized_generate`, session-payload
validation, world-service callbacks, and their tests and documentation:

- WebSocket error `content`, error codes, and close reasons must be English;
- browser-visible error `content` must use friendly plain English without
  implementation terms; stable codes and server logs carry technical detail;
- lifecycle and rewrite callback error reasons must be English;
- internal exception text must stay in server logs and must never be forwarded
  to the browser or world-service;
- stable error codes must use ASCII identifiers; do not use localized text as a
  code or branch on human-readable messages.

When any of these values or close codes change, update
`world-service/docs/asyncapi-gateway.yaml` in the frontend contract repository.
Chinese comments and ordinary product content are not error information.

## MinWM hardware-specific jobs

When a task creates or changes a MinWM deployment or benchmark manifest:

1. Read `benchmark/minwm_realtime_parity/HARDWARE_PROFILES.md`.
2. Start from `benchmark/minwm_realtime_parity/k8s/minwm_hardware_job.template.yaml`.
3. Select the profile from measured GPU memory and compute capability. Never infer
   capacity from a product name alone.
   - Record the `vae_cpu_offload` decision explicitly. The conservative
     `blackwell-32g` fit profile uses it. The explicit `sm120-32g-speed`
     validation profile may disable it only through the image-bundled launcher,
     which keeps the DiT and TAEHV resident and enforces the measured memory and
     throughput gates documented in `HARDWARE_PROFILES.md`. A GPU reporting at
     least 65,536 MiB may omit it when the high-memory profile applies. Treat
     intermediate capacities as experimental and start with offload enabled
     until same-device validation proves it can be disabled.
4. Render one-off manifests under
   `benchmark/minwm_realtime_parity/k8s/generated/`. This directory is ignored;
   do not commit dated, region-specific, run-specific, or instance-specific YAML.
5. Preserve the MinWM checkpoint/request invariants in the profile document.
6. Validate generated YAML before applying it and report the selected profile,
   detected hardware, memory gate, and any unvalidated assumptions.

Only update the committed template when the reusable schema changes. Only update
the hardware profiles when new evidence changes a stable decision or validation
status.
