# Realtime Zing deployment notes

This document captures the current single-Zing realtime deployment shape used
for the B300/B200 lab chain. It is intentionally narrow: production services may
use different account, namespace, and admission settings.

## B300 single-node baseline

- Public WebUI: `https://zing-b300-world-studio.loopit.me/`
- Namespace: `minwm-zing-b300-sp1-20260819`
- Transport: WebSocket + direct VAE H.264 encode, no bridge service in the
  active media path.
- Zing denoiser: 7 replicas, one GPU per replica, `sp=1`.
- VAE: 1 replica on the same B300 node, one GPU, async output path.
- WebUI/gameplay: 1280x704, 24 FPS, `sink=8`, `window=32`,
  `sessionMaxLifetimeSeconds=70`.
- H.264 profile used by the lab WebUI: 3000 kbps, CRF 20, `fast`, GOP 2s,
  VBV buffer 250 ms.
- Effective admission while only the B300 node is present: 7 denoiser slots and
  7 VAE slots.

## B200 Spot expansion target

The optional B200 Spot expansion keeps the same code path and adds:

- Zing denoiser: 7 additional replicas, one GPU per replica, `sp=1`.
- VAE: 1 additional GPU, capacity aligned to the 7 extra denoiser sessions.
- WebUI admission target: 14 sessions after the B200 node is actually Ready.

Do not count the B200 capacity as available until the node has joined the
cluster and the `zing-denoiser-b200-*` and `zing-vae-b200-*` pods are Ready.
During Spot `UnfulfillableCapacity`, the Kubernetes resources may exist but the
actual serving capacity remains the B300 baseline.

## Authorized World Run recording

When the world-platform route is enabled, Gateway also requires:

```text
RUN_VIDEO_S3_BUCKET=<dedicated run-video bucket>
RUN_VIDEO_S3_REGION=us-west-2
```

Gateway refuses to start when only part of the world-platform and recording
configuration is present. The bucket must be in the Gateway region. Its IAM
role needs only multipart create/upload/complete/abort access under
`runs/*`; object reads belong to the CDN origin, not Gateway. Enable S3 Block
Public Access, default server-side encryption, and a lifecycle rule that aborts
incomplete multipart uploads after one day.

Only a verified `authorized_generate` token with
`session.record_video=true` creates a recorder. Legacy/showcase sessions and
tokens with `record_video=false` do not record. Each complete Run is uploaded
directly as:

```text
s3://<bucket>/runs/<run_id>/video.mp4
```

The recording path is a fail-open side tap after Gateway output validation. It
does not proxy through world-service, re-encode media, or wait for S3 before
forwarding to the browser. Per recording, an 8 MiB byte queue feeds sequential
8 MiB multipart parts in a dedicated eight-thread S3 executor. The steady raw
media ceiling is therefore about 16 MiB per active recording, excluding SDK,
Python-object, and thread-stack overhead. If the queue fills or S3 fails, the
multipart upload is aborted while realtime playback continues.

Gateway completes an upload only after the final `media_chunk_complete` emitted
after FFmpeg stdout EOF. It then sends the existing HMAC-authenticated
`POST /internal/v1/runs/recorded` callback. Session cleanup does not wait for
this finalization; process shutdown drains detached recorders for at most ten
seconds before relying on the bucket lifecycle rule.

Deploy in this order:

1. Provision the bucket, CDN origin, IAM policy, encryption, and incomplete
   multipart lifecycle rule.
2. Apply world-service migration `0012_run_video_uri.sql` and deploy the
   callback/history implementation.
3. Deploy Gateway with both recording environment variables.
4. Verify one recorded World Run reaches `CompleteMultipartUpload`, callback,
   and CDN playback; verify one Showcase Run creates no multipart upload.

At current 3000 kbps encoding, eight concurrent recordings add about 24 Mbps
(3 MB/s) of same-region S3 upload traffic. Before raising admission above eight
sessions, repeat an RSS and event-loop-lag load test; do not compensate for a
slow S3 path by increasing the queue without new measurements.

## Consumed-reservation watchdog

Realtime workers should not let consumed reservations leak forever. A stale
consumed token is unsafe to simply delete because the corresponding session may
still be executing on GPU. The worker-local watchdog therefore:

1. Keeps normal unconsumed reservation expiry unchanged.
2. Tracks `oldest_consumed_age_s`, `stale_consumed_reservations`, and
   `last_progress_age_s` in the worker snapshot and heartbeat.
3. Marks the worker lifecycle as `failed` once a consumed reservation exceeds
   `max_consumed_age_s`.
4. Exposes failed lifecycle through health checks so the container supervisor
   can restart the worker instead of admitting more work onto a half-dead
   process.

Recommended value for the current 70-second game sessions:

```text
max_consumed_age_s = 120
```

This keeps a 50-second grace window after normal session lifetime. Denoiser
workers should use this strictly because capacity is one session per process.
VAE workers can use the same limit; if stale VAE reservations accumulate, the
worker fails health and restarts rather than silently over-admitting sessions.

## Rolling-update compatibility

Gateway and coordinator assignment parsing must tolerate additional heartbeat
or worker-slot fields. During rolling deploys, the coordinator may return a
newer payload schema before every gateway replica has the matching dataclass.
Filtering unknown fields avoids transient WebSocket init failures such as:

```text
WorkerSlot.__init__() got an unexpected keyword argument 'oldest_consumed_age_s'
```

## B300 H.264 ConfigMap overlays

### Retired overlay, 2026-08-21

The first tail-drain fix was temporarily shipped to the B300 lab namespace as
ConfigMap `realtime-h264-drain-fix-20260821`. It overlaid
`h264_media_pipeline.py` on `zing-vae` / `zing-vae-b200` and
`realtime_gateway_server.py` on `gateway`.

It was retired before 2026-08-24: the ConfigMap and all three mounts are gone,
the VAE image contains that fix, and `gateway` retains
`--output-drain-timeout-s=8`.

### Live EOF-barrier overlay, 2026-08-24 (remove after the next image)

Strict production E2E found that a complete 78-chunk / 1249-frame session
delivered only 1244 complete fMP4 fragments. Denoiser and VAE decode both
finished all 1249 frames. The loss was confined to H.264 finalization:

- FFmpeg input flag `-fflags nobuffer` deterministically discarded one frame;
- final `media_chunk_complete` reached Gateway before FFmpeg stdout EOF, so
  Gateway could tear down the output route before the remaining fragments.

Until an image containing the EOF-barrier fix is rolled out, ConfigMap
`realtime-h264-eof-fix-20260824` overlays only
`.../runtime/realtime/h264_media_pipeline.py` on `zing-vae` and the scaled-to-zero
`zing-vae-b200` template. The patch removes `nobuffer` and sends final completion
only after the FIFO queue drained, stdout reached EOF, and FFmpeg exited with
status zero. `gateway`, coordinator, and denoisers are not overlaid.

This overlay carries the usual `subPath` hazard:

**A `subPath` ConfigMap mount pins that file forever.** It survives pod
restarts, scaling, and `kubectl set image` — which is what makes it a usable
hotfix, but it also means a newer image's copy of the same file will be silently
shadowed by the 2026-08-24 snapshot. The overlay is an escape hatch, not an
upgrade path.

Required follow-up once an image containing the EOF-barrier fix is rolled out:

1. Remove the `h264-eof-fix` volume and its `volumeMount` from `zing-vae` and
   `zing-vae-b200` in the same rollout that changes the image.
2. Delete ConfigMap `realtime-h264-eof-fix-20260824`.
3. Keep `gateway --output-drain-timeout-s=8`.
4. Run strict E2E at least three times and require `media_batch` count to equal
   complete `moof+mdat` fragment count on every run.

Verification while the overlay is active:

```sh
kubectl -n minwm-zing-b300-sp1-20260819 exec deploy/zing-vae -c vae -- \
  sh -c '! grep -q nobuffer "$1" && grep -q _pending_final_completion "$1"' sh \
  /opt/sglang/python/sglang/multimodal_gen/runtime/realtime/h264_media_pipeline.py
```

The namespace has no manifest under version control; the live objects are the
source of truth, so this note is the only record of the overlay.
