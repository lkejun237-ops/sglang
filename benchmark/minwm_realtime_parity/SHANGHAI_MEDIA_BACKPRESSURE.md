# Shanghai Server-Side Media Backpressure

## Scope

Base release `41b3aa800efb`, deployed 2026-09-07 to namespace `minwm-realtime`
in Shanghai Lingjun cluster `i119114381787650669980`.

## Event Policy Follow-up

VAE-only release `9c20e4411675` removes unconditional event-ID-based frame
invalidation. `event_id` remains unchanged in output metadata and is not an
invalidation boundary. The queue keeps fresh frames in arrival order even when
new input events are sampled. No input coalescing or public protocol change is
introduced. Existing queue-capacity, age, startup-trim and finalization behavior
remains in effect, including the base release's negotiated backpressure.

Only `h264_media_pipeline.py` is replaced in the `vae` and `lingbot2-vae`
containers by immutable ConfigMap `zing-vae-drop-policy-9c20e4411675`.
Gateway, Caddy, denoisers, coordinator, WebUI, frame rate and bitrates are
unchanged by this follow-up. GPU allocation and CPU-offload settings are also
unchanged; no new hardware profile or fit assumption is introduced.

Server logs now report `queue_full`, `latency_expired`, and `startup_trim`
cumulative counts. Logs are throttled to one update per five seconds and include
an unconditional final summary. Existing browser-facing counters are preserved.

Verification: 77 focused tests passed, including eight event-policy tests. Both
production VAE images passed isolated tests using their FFmpeg 4.2.2: 30 event
updates produced 61 encoded/decoded frames with zero drops, and the separate
three-second blocked-output test recovered to a decodable latest frame.
These are synthetic-frame probes, not a real-client gameplay acceptance test.

Follow-up artifacts and the **policy-only** rollback are on the Shanghai node:

```text
/root/zing-k8s-shanghai/changes/vae-drop-policy-9c20e4411675/
```

After checking there are no active sessions, restore just the previous VAE Pod
template with its release-guarded `vae-rollback.json`. This keeps the base
backpressure release and the LingBot bitrate reduction:

```sh
kubectl -n minwm-realtime patch deployment zing-vae-a10 --type=json --patch-file=/root/zing-k8s-shanghai/changes/vae-drop-policy-9c20e4411675/vae-rollback.json
kubectl -n minwm-realtime rollout status deployment/zing-vae-a10
```

The base-release rollback below is broader and removes its backpressure changes.

## Base Release Scope

- Workloads: `zing-gateway` and both containers of `zing-vae-a10`.
- No client, `authorized_generate` authentication, coordinator, Redis, denoiser,
  model, resolution, or GPU layout changes.
- Zing: 1280x704, 24 fps, H.264 maxrate 6500 kbps.
- LingBot: 832x480, 24 fps, H.264 maxrate 3250 kbps.
- Rates are encoder caps, not guaranteed constant throughput.

## Behavior

The VAE negotiates `media_delivery_ack` on its internal Gateway output socket.
The Gateway acknowledges a media payload only after the output route drains,
including the currently pending browser socket write. The VAE serializes its
next media payload behind that credit. This is neither `control_ack` nor a
browser playback acknowledgement, and does not change the public wire protocol.
Older gateways that do not negotiate the option retain their existing behavior.

While the output write is blocked, the next RGB frame remains in the bounded,
droppable pre-encode queue. Old RGB frames are discarded before entering FFmpeg;
the latest frame is retained. H.264/fMP4 chunks are never independently deleted.
Frame indexes remain continuous across encoded frames.

There is no fixed sleep or FPS pacing. A writable connection sends immediately.
Network congestion can reduce the number of frames encoded and delivered.
Session deadlines and dead-peer handling remain in effect; congestion does not
add the ordinary five-second internal RPC timeout to media payload delivery.

## Settings

VAE flags, for both models:

```text
--h264-gateway-backpressure
--h264-max-frame-age-ms=250
--h264-drain-timeout-s=15
```

The 15-second setting applies to final pipeline draining, not playback buffering.
Both new settings are opt-in; other deployments keep the previous defaults.

Gateway environment:

```text
SGLANG_GATEWAY_SOCKET_SNDBUF=131072
```

The opt-in Uvicorn WebSocket protocol also sets TCP_NOTSENT_LOWAT to 32768
where supported and asyncio write watermarks to 65536/16384 bytes. Linux reports
a doubled SO_SNDBUF of 262144 bytes for the requested 131072-byte buffer.

On the dedicated Caddy edge only, `/etc/sysctl.d/90-zing-media-buffers.conf`:

```text
net.ipv4.tcp_wmem = 4096 16384 262144
net.ipv4.tcp_rmem = 4096 131072 262144
net.ipv4.tcp_notsent_lowat = 32768
```

These bounds prevent large server-side TCP buffers from hiding downstream
backpressure. They do not clear bytes already delivered to the browser, change
the client's playback buffer, or guarantee a fixed end-to-end latency under
arbitrarily low bandwidth. Kernel semantics are documented in the
[Linux IP sysctl reference](https://docs.kernel.org/networking/ip-sysctl.html).

## Deployment and Verification

The release uses immutable ConfigMap `zing-media-backpressure-41b3aa800efb`
mounted with subPath files in Deployment templates. It survives Pod replacement;
it is not an edit inside a running container. Future image or manifest rollouts
must explicitly preserve or supersede this overlay.

The production Gateway image contains newer code than this branch. Its release
file was built from the actual production file, replacing only the internal
output handler and optional Uvicorn startup configuration. Do not replace the
entire production Gateway with the older branch file.

Pre-deployment verification:

- 69 focused unit and integration tests passed.
- Real FFmpeg encoding/decoding survived a three-second blocked output.
- The actual Gateway image passed legacy-protocol and in-flight credit tests.
- The actual VAE image's FFmpeg 4.2.2 dropped stale RGB frames and decoded the
  latest frame without deleting encoded payloads.
- Server-side deployment dry runs and runtime source-baseline checks passed.

Post-deployment checks verify source hashes, readiness, flags and isolated
Gateway/VAE probes. These probes use synthetic frames and memory-only sessions;
they do not create production model sessions or modify Redis. A real client
weak-network gameplay session remains a separate acceptance check.

## Rollback

Node backups and hash manifest:

```text
/root/zing-k8s-shanghai/changes/media-backpressure-41b3aa800efb/
```

After confirming no active sessions, roll back VAE first, then Gateway on the
Shanghai control node. The patches check the current release annotation before
restoring the saved complete Pod template:

```sh
kubectl -n minwm-realtime patch deployment zing-vae-a10 --type=json --patch-file=/root/zing-k8s-shanghai/changes/media-backpressure-41b3aa800efb/vae-rollback.json
kubectl -n minwm-realtime rollout status deployment/zing-vae-a10
kubectl -n minwm-realtime patch deployment zing-gateway --type=json --patch-file=/root/zing-k8s-shanghai/changes/media-backpressure-41b3aa800efb/gateway-rollback.json
kubectl -n minwm-realtime rollout status deployment/zing-gateway
```

The VAE backup already includes LingBot's 3250 kbps bitrate change.
On the dedicated edge, restore sysctls from
`/root/zing-media-41b3aa800efb/tcp-before.conf` and remove only this release's
`90-zing-media-buffers.conf` after checking its contents. Do not change GPU-node
sysctls. Keep the immutable ConfigMap until no running Pod references it.
