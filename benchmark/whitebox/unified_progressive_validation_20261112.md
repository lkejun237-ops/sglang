# Unified progressive flow: Shanghai test validation

Status: implemented and deployed to `test-shanghai-copy`; the user reported that
quick playtesting worked and asked to stop testing and prepare a PR. Services
remain available. Neither production region nor training jobs were changed.
This is a server-side validation report, not full browser/RTC acceptance.

## Frozen identities

- Implementation: `35886fadf4580b1bdb965dc904506888a27fc055` +
  `e3b1ae48ab51ae9a436b10dd33185c1b3d22fea6`, based on main
  `4229aa4df50258c6c25ada02e340ef006733331c`.
- Complete deployed source: `a7d61072d3b6e5c3548bb2f8c01f1f305bab8733` on
  `codex/whitebox-unified-progressive-20261111`; its runtime is unchanged from
  the tested implementation. SDK: `31ea27da6cc7bdfbe235da28f688d1580ed3f3b6`,
  verified reachable from SDK `origin/dev`.
- GPU image: `sha256:1fd79ac1985312b6ad596e5a494e2f7f5227bcc42d0bb211174cfec8df1a5b98`.
- Room image: `sha256:3654ec3684969128740c42324d32b3c46f524c0ac067312477a5ea16120238cb`.
- One private test model group: H100 Encoder + H200 SP2 DiT + H100 Decoder.
  Model 1027 AR/DMD step10000, BF16, `[3,2]`, `split_first_latent=false`, W6/S0,
  `mem_fraction_static=.8`; weights and model settings were preserved.
- World: `world-d94f64601bff9a3b867f82adb3f55123`, source `interactive-production`,
  832×480, CLI36, seed42, manual model enable followed by the same repeated
  W/D/S/A press/release script. Observer uses the pinned SDK's real WhiteboxClient.
- PR integration base: main `a6e2617083d6bd75f0e08cccd4602b15db521ccb`.
  The two feature commits applied without conflicts. This final main integration
  was not redeployed or given another GPU run after the user stopped testing.

Machine-specific operators and failed-attempt receipts remain on the original
experiment branch; they are not part of the proposed reusable implementation.

## Observed behavior

The actual Decoder→DiT→Encoder→Room handshake declared
`valid-source-frames-v1` and `clip17-source-flow-v1`: input groups `[1,8,8]`,
first-output dependency 9 frames, progressive output enabled. The retained hard
budget was input25 + output reserve4; high-water occupancy was26, below29.
The adaptive recording declared negotiated native demand and effective mode
`adaptive`, with accepted/dispatched source ends equal at release, no queued
source frames, and zero integral. This was not an unverified fallback mislabeled
as adaptive. These contracts apply to both deployment topologies; Thailand itself
was not changed or retested in this run.

All three observers ran for 190 seconds **after the first model packet**, sent
real inputs through the real CLI, received real model output, ended with no client
error, and released their own source/model resources. Sequential sessions reused
the same model processes without a fault restart. Timing starts at server-side
packet receipt, not browser presentation.

## Sequential same-hardware comparison

Counts use the strict common 190.000-second window, including stalls. Drain frames
and the observer's small timer overrun are excluded. Steady-state figures are
separately labeled; the first ten seconds remain in the whole-window result.

| Measurement | Adaptive A1, first run | Latest B1 | Adaptive A2, warm repeat |
| --- | ---: | ---: | ---: |
| Unique model frames in 190s | 4,996 | 5,177 | 4,964 |
| Model packet FPS | 26.295 | 27.247 | 26.126 |
| FPS after first 10s | 27.206 | 27.294 | 26.067 |
| Receive gap P95 / max, ms | 203 / 2,314 | 212 / 871 | 207 / 591 |
| Source PTS gap P95, ms | 85.255 | 109.885 | 109.955 |
| Capture→raw output mean / P95, ms | 817 / 907 | 792 / 909 | 815 / 957 |
| Keydown→model receipt P50 / P95, ms | 892 / 1,022 | 928 / 987 | 946 / 1,129 |
| Matched keydown samples / emitted edges | 29 / 33 | 33 / 33 | 31 / 33 |

No sustained decline to the previously observed 13FPS range occurred. A1's first
10-second window was only9.9FPS; subsequent windows were26.2–28.2FPS. A2's
10-second windows were24.6–27.2FPS and did not show a continuing downward trend.

**Observed costs are retained:** warm A2 had4.1% lower throughput than B1,
5.3% higher capture→raw P95 and14.4% higher matched-input P95. A1 had a much worse
initial maximum gap. A2's source PTS P95 was approximately unchanged, not a proven
smoothing gain. These sequential single runs do not establish statistical
significance or unconditional performance improvement; initial frames, warm-up
and live-world evolution were not bit-identical. Unmatched inputs are not silently
counted as responsive. Receipt mapping is not proof of a visible action or WAN/
browser latency. The old runtime could not supply an immediately comparable
baseline because the previous test Decoder was absent; `latest` here uses the same
new candidate runtime, not the old release.

## Recordings and quality

A1 and B1 model MP4s matched their manifests' SHA256 and decoded completely:

- A1: 5,018 frames, 192.777410s,
  `6adea38ed808b0df189a9dc3476e4926d908c57f38f6329ae14b5767894a8924`.
- B1: 5,203 frames, 191.274320s,
  `02f829c87777eef1d81605a17166286541703ddbe35df5db81424f9444a35cfb`.

Both manifests report complete recording/upload. Private observer evidence for all
three runs is retained under the task-owned OSS evaluation prefix. The current
recorder is diskless: an absent local MP4 directory does not mean upload failed;
resolve the exact recording ID through its archive index and verify the objects.

**Quality did not pass:** late model frames in both modes show strong outlines,
high-frequency texture and structural distortion. Early/middle/late screenshots
were retained. Similar long-session deterioration was already documented on
November9 with the old model/runtime; that is evidence of a pre-existing symptom,
not proof that the candidate cannot worsen it. The user subsequently reported
quick browser playtesting as normal; no timed browser quality/RTC evidence was
collected by this observer.

## Validation coverage and retained failures

- [Zing CPU CI run43047219482](https://github.com/seedleap/sglang/actions/runs/43047219482)
  passed on implementation `e3b1ae48ab51ae9a436b10dd33185c1b3d22fea6`.
- Full GPU image: 5,541 exported source files verified. Room: 11,815 source/SDK
  files and links verified; layer-history gate confirmed one source and one SDK
  copy. Embedded CLI checks17 passed; native codec checks51 passed. Runtime code
  was never patched in place.
- CPU regressions cover native-demand identity/order fencing, real assembler
  demand, slow/absent supply, true capacity changes, control/EOS wakeups,
  direct-output Decoder progressive publication and legacy peers.
- Two H200 attempts failed before model loading because two mutable shared
  metadata objects no longer matched the original asset manifest. A complete
  30-object audit identified the drift. Exact original metadata hashes were
  recovered into a new immutable deployment prefix; the final job verified all30
  assets. Shared objects and model weights were not edited. Failed attempts remain
  recorded, and are not counted as successful model tests.
- macOS broad-run startup timing and IPC-path failures are preserved in private
  logs; the same startup timing failure reproduced on base main. Short-path focused
  reruns and the Linux CI passed. Initial malformed local/read-only probes and an
  interrupted security-rule readback were reconciled; unknown mutations were not
  blindly repeated.
- Public whitebox sessions appeared between comparison runs. Idle guards prevented
  their interruption; restoration waited for their natural exit. Public admission
  was reopened after exact observer release proof, with1 free logical Slot.

Not covered: 736 output, Thailand B300, full five-case browser acceptance, browser
presentation/RTC/WAN RTT, independent live Prefill/L1/L2 recall, Prompt/Scene change
GPU acceptance, multiplayer, and per-component Encoder/DiT/Decoder latency and
allocator peak memory. Single-slot validation does not imply multiplayer capacity.
No extra GPUs were allocated to fill those gaps. The test environment is retained
until the user asks to release it; this PR does not authorize production rollout.

Trace IDs: A1 `ecc98cea73ed49f98719ccb212222573`, B1
`942eb215134e46269a0229031023ed9f`, A2 `5e48536669fb45ea8b881cf7ba12c20b`.
