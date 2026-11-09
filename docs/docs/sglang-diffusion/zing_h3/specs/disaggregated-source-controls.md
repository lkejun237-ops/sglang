# Source controls on managed Encoder / DiT / Decoder groups

The existing player and native protocols remain the public contract. A player
owns one parent Model Slot; Prefill, generation and scene children use its serial
compute lock. Child completion never releases the player's parent allocation.
The separate upload Prefill endpoint is a model-only admission: it never creates
an SDK member and cannot supply or bypass a real player's SDK cleanup receipt.

## Conditions and startup

Room reserves a complete codec boundary for an ordered Prompt/reference update.
Encoder dispatches the remaining old source, then sends an ordered synchronous
condition barrier to DiT. DiT checks its accepted and committed positions, drains
pending source, calls the original `activate_prompt_at_source_boundary`, and
confirms CUDA completion and identical SP receipts before `kv_ready`. Room then
resumes with a fresh capture. No future source frame is needed to unblock this
barrier. Identical complete conditions keep the original resident-prefix reuse
path; their receipt is `bypass`, not an L1 hit.

`source.prepare` v1/v2 uses the existing immutable preview/identity contract.
Encoder prepares the original Qwen/reference condition and waits for the connected
DiT's confirmation. `source.prepared(level=condition)` is not an assertion that a
future generation's GPU KV has already been restored. Duplicate preparation must
have identical pixels **and dimensions**. v1 retains the exact first-frame check;
v2 retains its independently sized preview semantics. Old peers lacking the
negotiated capabilities fail explicitly instead of silently omitting the fence.

The Room's existing 5-second refresh scheduler and Sink on/refresh/off controls
are reused. They do not create another player, open another Slot, change the
sampling schedule, or extend input-idle/player deadlines.

## Public upload Prefill

The same signed ticket, binary upload, status/idempotency and persistence messages
remain in use. The Gateway selects the configured model backend through the
existing lease factory. A model-only group records its operation at admission;
that immutable identity cannot be converted into a player allocation.

`memory_ready` is exposed after positive child-cleanup evidence. The caller can
then release all native roles without an invented SDK receipt. Pending disk
writeback is observed through the original allocation and boot identity, within
the original bounded deadline. A terminal Prefill may only inspect persistence;
it cannot renew capacity, access another generation, open a generation socket,
or change a later player's owner. Ordinary players still require their actual
SDK retirement proof before terminal release.

## Cache evidence and acceptance

Prefill reuses the original Block0 cache and model key. L1 lookup, L2 promotion,
Host publication and GPU restoration remain distinct. A second bookkeeping L1
lookup must not erase a first L2 promotion. Public `model.cache` uses the existing
sanitizer; native per-rank `DIT_CACHE_RECEIPT` retains transfer-event timing without
synchronizing merely to collect a metric. An unavailable CUDA timing stays absent.
Expected cache admission failure is a request rejection, not permission to
restart or quarantine a healthy model group.

Required validation uses real input identities and request/worker epochs: both
user WebSocket and CLI-originated Prefill; ten distinct keys warmed **before**
recall, all ten classified as L1 hits; separate L2 read/promotion and GPU restore;
5-second refresh; Sink on/refresh/off; scene changes, cancellation, expiry,
reconnect, next-player reuse, recording and sustained visual quality. CPU protocol
tests do not prove CUDA, checkpoint quality or real-game CLI output. No runtime
configuration switch substitutes for the missing native protocol implementation.

Current implementation work has CPU protocol evidence. GPU/SP2, real-game media
and ten-entry resident-hit acceptance remain pending until recorded in the PR.
