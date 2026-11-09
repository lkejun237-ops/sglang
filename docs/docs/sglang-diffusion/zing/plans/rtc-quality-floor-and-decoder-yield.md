# RTC quality floor and bounded decoder delivery

## User-visible behavior

The model track keeps an adaptive encoding budget of at least 1 Mbps and at most
its configured 2.5 Mbps ceiling. The whitebox track retains its 1 Mbps ceiling
and has a 300 kbps quality floor. These are encoder budgets, not promised network
throughput or CBR padding. Explicitly lower deployment ceilings still win.

The Thai 2026-10-30 repair also applies a 1.5 Mbps minimum to native RTP pacing,
including protocol headroom, so the sender does not throttle a 1 Mbps model plus
300 kbps preview behind a smaller pacing budget. Higher receiver feedback remains
adaptive up to the existing deployment ceiling. Raw `receiver_bitrate` is kept
separate from applied `target_bitrate` / `transport_target_bitrate`; a quality
floor is not a measured link capacity. Existing packet/access-unit byte and age
limits remain unchanged. A truly insufficient path can still reduce delivered
FPS; the floor does not create network capacity.

## Evidence and choices

A reported late-session mosaic was present in the verified OSS archive, with no
H.264 decoding errors. The model payload fell from approximately 2.1–2.4 Mbps
to 0.27 Mbps and whitebox to approximately 0.11 Mbps. The previous weighted
350 kbps minimum allocation permitted 250/100 kbps model/whitebox targets.
A local 240-frame native x264 replay reproduced 36 nearly-flat whitebox frames
at 100 kbps, and zero at 300 kbps. This supports a modest source floor, while
reserving the larger quality budget for the model. It does not establish the
cause of the receiver's bandwidth estimate collapse or prove weak-network FPS.

Another recorded session delivered raw model frames in 8/9-frame groups despite
metadata declaring a one-latent decode quantum. The implementation was absent.
The opt-in `SGLANG_ZING_EXPERIMENTAL_DECODE_WORK_QUANTUM=1` now limits Lynn to
one causal latent per scheduler dispatch. It requires source work quantum 1.
The temporal protocol emits 1+4+4+4+4 valid frames for a 17-frame clip; outputs
are not split after decoding the entire pending RGB batch. Undecoded latents
remain bounded and resume before another DiT block. A condition-switch barrier
cannot advance past pending RGB. EOS returns every valid frame before disposal.
The default quantum 0 preserves the existing unbounded decode behavior.

The change does not alter weights, diffusion steps, source PTS, DiT chunking,
Encoder 1+8+8 slicing, H.264 inter-frame compression or UI/frame identity.
Async GPU-to-CPU materialization remains the deployed setting; this change does
not weaken the local-IPC guard or assume the public HTTP bind address is local.

## Verification

- Real tiny Lynn streams: bounded and unbounded decoding are bitwise equal,
  including partial final latents, multiple clips, provenance and EOS.
- Worker/adapter tests: bounded work, empty continuation, cancellation, prompt
  barrier and disposal remain valid. Synthetic wiring tests are not GPU evidence.
- Native x264 integration: rate changes preserve frame count, source identity,
  decodability and inter-frame GOPs. Feedback hysteresis cannot retain a target
  below its floor; failed/unready updates are retried without claiming applied.
- Production acceptance: after the separate active deployment finishes, merge
  its final source, verify deployed identities, then test at least ten different
  existing worlds sequentially, each for 190 seconds after first model picture.
  Record actual payload/receiver bitrate windows, source/raw/decoded/display FPS,
  input-receipt-to-visible-frame latency, RTT, freezes, losses and quality.
  Label latency as input mapping, not proof of semantic action completion.
- Inspect startup separately, retain failed attempts, and compare early and late
  windows. Report 832×480 independently. The online playtest service does not
  offer a 736 profile; that profile remains unverified.

No production acceptance is implied by local tests or this design.

## Thai hard-floor repair (2026-10-30)

The incident Room image `50fa3a3bc713` used 1 Mbps/300 kbps only as preferred
floors and allowed 100 kbps per codec. The new overlay `fb23724eea03` restores
hard encoder budgets and matching native pacing, retaining TCP/443, packet queue
bounds, two-hour play configuration, GPU allocation, models and independent RGB
archives. In this incident the backend archive is clear, unlike the older
archive evidence above: do not conflate the two recordings or versions.

Native Go tests cover low-feedback pacing/encoder agreement and truthful
measurement fields. Runtime checks cover the Linux binary, ten budget cases and
20 decoded H.264 frames at a 1 Mbps target. Browser acceptance and the resulting
budget/quality observations are recorded separately; code constants alone are
not a claim about actual wire throughput or image quality on every client.

### Regional bitrate configuration

Room reads `WHITEBOX_MODEL_MIN_BITRATE` (default 1000000 bps) and the existing
`WHITEBOX_PLAYBACK_MAX_BITRATE`. Native RTC reads `WHITEBOX_RTC_MIN_SEND_BITRATE`
(default 1500000) and `WHITEBOX_RTC_MAX_SEND_BITRATE` (default 4200000).
Shanghai uses 2500000 / 4000000 for model encoding, and 3200000 / 5600000 for
combined transport pacing, including protocol headroom. Preview stays 300000–1000000.
Receiver estimates remain telemetry; applied send budgets do not guarantee network
throughput or pad simple frames to a constant bitrate. Queue age/byte limits remain.
Thailand retains existing defaults. Changing these environment variables requires
a Room restart; model workers and archive bitrate do not change.
