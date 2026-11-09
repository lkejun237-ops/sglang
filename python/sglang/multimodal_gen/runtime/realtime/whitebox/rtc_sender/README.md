# Native RTC sender

Bandwidth feedback uses the browser's native receiver estimator (`goog-remb`).
Pion's native pacer, NACK responder and RTCP reports remain in charge of media
transport. The standard absolute-send-time extension is written at packet
departure, after pacing. The application forwards the native estimate to its
existing encoder controls, bounded by the configured ceilings. Missing feedback
stops media after ten seconds rather than continuing at an uncontrolled rate.

The interceptor dependency is pinned to upstream commit
`8d41ec1246a48507996ab59a9b4bea84e4d5f2e4`, which includes
[Pion interceptor #480](https://github.com/pion/interceptor/pull/480).
The v0.1.49 feedback adapter counts status-vector padding beyond TWCC's
`PacketStatusCount` as unreceived packets. This can feed false loss into GCC
and collapse the external encoder's bitrate despite a healthy RTP path.

The pin remains useful for the shared interceptors. The sender-side GCC path
was removed after browser tests still showed large transient loss estimates and
poor bitrate recovery. Do not compensate by inventing a loss filter, disabling
feedback, or forcing the encoder above the available bandwidth. Re-run real
browser bitrate, quality, and recovery tests when changing the transport stack.
