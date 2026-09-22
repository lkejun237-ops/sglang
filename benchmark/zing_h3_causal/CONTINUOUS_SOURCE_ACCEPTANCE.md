# PR #140: shared runtime reuse and final functional acceptance

This checklist supersedes the historical completion statement in
CONTINUOUS_SOURCE_VALIDATION.md. Historical results retain their original SHAs;
none count as execution of the new frozen runtime after the shared cancellation change.

## Existing owners reused

| Responsibility | Existing owner / entry point | Validation reused |
| --- | --- | --- |
| In-flight chunks | GenerateSession.new_chunk / generate_chunk_completed | test_realtime_runtime |
| Generation, send overlap and EOS send completion | _generate_loop_local / _send_output_and_log | test_realtime_runtime / test_realtime_output_transport |
| Serialized headers, payloads and controls; bounded send backpressure | _LockedRealtimeWebSocket.send_group | existing transport/runtime tests; actual slow consumer |
| In-flight execution cancellation | shared _process_realtime_generation_batch waits for worker response before existing cleanup | new delayed completion/error regression; real disconnect |
| Worker state release | _cleanup_realtime_session → ReleaseRealtimeSessionReq → GPUWorker.release_realtime_session → RealtimeSessionCache.release | existing release/registry tests plus H3 evidence |
| Temporary reference files | GenerateSession.input_temp_dir, existing cleanup | new in-flight cleanup regression; cleanup trace |
| Admission and reservation leases | generate.finally → controller.release / _release_gateway_reservation | existing admission/reservation tests; fresh connection after cleanup |
| Remote VAE credit | existing async VAE coordinator only when configured | not used: this H3 source route explicitly requires local decode |

No separate H3 cancellation, disconnect, admission or writer state machine.
The shared generation RPC remains owned through cancellation because cancelling
its socket wait does not stop executing GPU kernels. During cleanup the existing watchdog renews the admission lease, so a block
that outlives its original TTL cannot admit a replacement prematurely. Client
idle and playable-lifetime limits remain unchanged. This may delay admission
until the active block actually finishes; it does not resume an old session.

## H3-specific state and ownership

| State | Owner | Release / ownership boundary |
| --- | --- | --- |
| Source frame/byte credit and append provenance | adapter SourceLedger | accepted is sent before pending work becomes visible; consumed only covers complete appends handed to bounded bridge; dispose clears records |
| Pending RGB, source readiness, pending prompt, reference IDs/paths | one ZingH3SourceStreamAdapter per connection | existing adapter.dispose, common temporary directory cleanup |
| Encoder pixels, bridge latents, decoder cache, source-to-output block mapping | ZingSourceStreamState in shared worker session registry | source.end drains encoder → bridge → sampler → decoder; exception/EOS/registry release invokes idempotent dispose |
| DiT KV, prompt/static tensors and bounded ready chunk | SourceReadySession | existing dispose; window 21 / sink 3 and block-relative gap 12 |

Reference IDs and order are preserved. Prompt video context remains the first
actually appended RGB frame (832×480), not a separately pre-uploaded video.
The codec's dependency/lookahead watermark never independently grants credit:
`bridge.accepted_pixels` is the consumed ownership watermark.

## Frozen-run requirements

- Corrected PR #141 case `zing-whitebox-video-five-triview-0001`, case SHA256
  `02716a977c5dd3a3f52b26da617d75204a2b38ef69bd3654d4de9dd2db270ff1`.
- Five original references, original prompt, verified six asset hashes;
  832×480 RGB, ODE step1000 actual `[1,2,2]`, four steps, TAEH3 fp16.
- Freeze code SHA, run scripts, image digest and config before execution.
  First SP1 correctness, then minimal SP2. No SP8, no performance matrix.
- Real weights and MessagePack; actual single-frame / irregular paced append.
  Never submit the complete source as an init payload.
- 10/60-second technical streams may concatenate the five-second source only
  with explicit segment boundaries and a derived hash. They prove accounting
  and bounded state, not continuous-scene visual quality.

| Required gate | Frozen runtime result |
| --- | --- |
| Existing shared runtime, release, admission, transport tests; new cancellation regression | passed; evidence below |
| Source/reference mapping, first frame context, single-frame and irregular append | passed; evidence below |
| Frame/byte limits, slow-consumer backpressure, accepted before referenced output, consumed at full append boundaries | passed; evidence below |
| Pause has no fabricated source/output; resumed ranges remain consecutive | passed; evidence below |
| EOS short tails and full case: exact valid frame count and explicit completed | passed; evidence below |
| Cancellation while executing, EOS/disconnect race, initialized-session exception cleanup | passed; evidence below |
| Worker encoder/bridge/decoder/KV release, temporary file removal, admission reuse | passed; evidence below |
| Two live sessions: distinct IDs, independent source ledgers, output lengths and release | passed; evidence below |
| 10/60-second bounded-state evidence with declared concatenation | passed; evidence below |
| Final SHA CI and all owned cloud jobs terminal | passed; evidence below |

Basic input/output correctness and resource-release safety remain PR #140 gates.
24 FPS / performance is tracked in issue #131 and related optimization issues;
complete subjective/native quality review is issue #153. PR #148 owns the real
progressive encoder and same-version clip17/progressive A/B; its historical
runs are not substituted for this final functional run. No new temporary
whitebox review webpage will be published.

## Final frozen runtime evidence

Tested source: `74d19dde391717b5f57c8690a6027768e2cd3f6d`, including main
`9a0a1b3641` and the applicable merged PR #149 / PR #151 fixes. Subsequent
commits only record acceptance documentation; GPU runs are attributed to this
exact frozen source, not relabeled as execution of a later documentation SHA.

Image: `zing-thailand-registry-vpc.ap-southeast-7.cr.aliyuncs.com/zing/sglang@sha256:4b9bca7dd61784cd84e43ac760db4c458c08a2d6bf22887e3740455fffe9e015`.
Measured hardware: NVIDIA L20D, CC 10.3, 274113 MiB per GPU; Torch 2.13.0+cu130,
CUDA 13.0, uvicorn 0.52.4. No server timeout or acceptance threshold was relaxed.

| Final-source job | Terminal state | Result |
| --- | --- | --- |
| CPU `dlc1l770nexylor8` | Succeeded / exit 0 | 1075 tests, including shared runtime/admission/reservation/transport and delayed execution/error/lease cleanup regressions; all six original asset hashes verified |
| SP1 `dlcpaj47edvof66q` | Succeeded / exit 0 | 9 video cases, 10 lifecycle cases plus the second live-session peer |
| SP2 `dlcukfm6i8hjypg0` | Succeeded / exit 0 | 4 video cases, 6 lifecycle cases plus peer, and 4 distinct-source baseline/concurrent isolation requests |

SP1 covers full 120 frames; single-frame appends to 18 frames; tails 1/2/16/17/34;
51-frame slow consumption (2 seconds per received output batch); 240-frame
technical concatenation. SP2 covers full 120, single-frame tail 18, slow 51,
and 1440-frame technical concatenation. All four source counters agree, with
explicit completed, continuous output ranges/PTS/provenance, accepted before
referenced output and consumed only at complete append boundaries. Both
120-frame MP4s fully decode as 832x480 / 24 FPS / 5 seconds; this is a media
format check, not inference speed or subjective-quality acceptance.

SP1 lifecycle: invalid sequence, in-flight disconnect, fresh session after it,
EOS/disconnect race, fresh session after it, same-user concurrent rejection,
pause/prompt/resume (51 frames), error after real output, two live sessions
(17/34 independent frames), five-reference completion. SP2 repeats cancellation,
EOS race, fresh sessions, two live sessions and five-reference completion.
The logs prove cancellation begins after scheduler dispatch and before the
worker step finishes; cleanup then waits for the response. SP1 cancellation
cleanup measured 3393.710 / 3395.720 ms. CPU regression separately holds an
in-flight operation across more than two lease TTLs: new admission remains
rejected until execution and release finish, then succeeds.

SP2 content isolation also uses two different source/seed streams: A is the
first 17 real frames, seed 42; B reverses those 17 frames, seed 43 (technical
isolation only). Each concurrent output is byte-identical to its own isolated
baseline, and A/B outputs differ:

- A output SHA256: `ccf11eee9583d5f97ba647c01be5a2fa314f970284843b661b2077b2ada1aa86`.
- B output SHA256: `ed649d9af1e4d655871ed22d68fae8d59930968f8f78f2ba85484605e9f1a0a6`.

### Bounds, backpressure and release

- 10-second input pending maximum 68 frames; 60-second maximum 67; negotiated
  limit 68 (81,469,440 RGB bytes). Slow SP1 socket writes actually blocked for
  3.834 / 10.045 seconds, demonstrating shared send backpressure.
- Long-stream KV reaches 21 temporal chunks and stays bounded. On SP2, each
  rank has 47 full-window samples at exactly 42,681,302,400 retained KV bytes;
  allocated GPU memory varies only about 40.7 MB over those samples. This
  describes retained session state, not allocator/model-weight release to zero.
- Long-stream maxima: encoder pixels <=16, decoder latents 1, metadata batches 5,
  pending blocks 2; bridge buffered latents are zero at recorded dispatch ends.
- All 25 rank/EOS records for the video/isolation-request matrix clear KV,
  static conditions, codec/bridge/decoder storage, metadata and pending blocks.
  All 49 actual H3 rank/state disposal records (including cancellation/error)
  clear the recorded retained-state fields. Invalid-before-initialization
  correctly has no worker H3 state to dispose.
- Every tested session, including isolation peers, confirms scheduler release
  acknowledgment and removal of its reference temporary files. Fresh admission
  and the independent baseline comparison verify clean reuse. All owned GPU
  jobs are terminal; no SP8 was submitted.

The 10/60-second sources concatenate the original five-second/120-frame RGB
segment 2/12 times. Boundaries at every 120 frames and derived hashes are stored
in each source receipt. They are technical lifecycle/resource tests, not
continuous-scene visual-quality evidence.

### Provenance, failures and reproduction

Checkpoint export: `/cpfs/world-model/sglang/checkpoints/zing131-ode-step1000-r1`;
original step1000 model SHA256
`09b0348a66f567c3f708cf6298a794950bb7e3e6e506fc00aa6824db5e75b0ba`.
Original video SHA256
`6579ebd3f1d80455b71e30f9795aa494bceb615ab1302a812430f1199ec80f85`.
All reference hashes, full resolved configuration and runtime manifests are
in the receipts; the exact source overlay SHA256 is
`c1bb026eee4c2968c0f8537c9e0df55ab58df120d4e5e51795ad002cb217e15b`.

Private artifacts: `oss://leap-worldmodel-thailand/world-model/sglang/evals/zing-h3-dmd/zing140-final-20260922/`,
with `cpu-r3`, `sp1-r2`, `sp2-r1`, and `acceptance-final/evidence.zip`.
The bundle includes exact service/client commands, manifests, job terminal
receipts, media validation and logs, plus the independent result verifier.
It can be downloaded using the team's existing OSS identity and Thailand
endpoint. No new HTML review page is published.

Earlier attempts remain recorded: CPU `dlcxw2bd65eb0rfj` Failed (1072 passed,
two old test adapters missing base hooks); CPU `dlcza1oui0whr0bw` Succeeded
(1074 passed, predecessor code); SP1 `dlc5va4uidy1ednc` Stopped by this task
when the lease-TTL gap was found before accepting GPU evidence. These are
not counted as final-source GPU acceptance. The new final CPU adds the lease
regression, reaching 1075. Complete subjective quality, native parity and
24 FPS remain not executed/not passed in this functional run, tracked by
issue #153 / issue #131. PR #148's progressive encoder/A-B is not duplicated.
