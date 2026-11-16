# Entry-only scene transition CPU verification

## Scope

A valid single-player scene inventory may contain `transitions: []`.
Room now accepts 0–128 story rules while retaining nonempty entries, declared
initial/targets, video coverage, request identity, generation/revision checks
and the existing presentation-before-activation path.

The companion CLI projects the declared entry inventory without requiring a
story rule. Paired PRs: [Worldkit #1265](https://github.com/seedleap/worldkit-monorepo/pull/1265)
and [SGLang #734](https://github.com/seedleap/sglang/pull/734).

## Actual results — 2026-11-16

Each row identifies a separate execution and its actual source versions.

| Verification | Result | Executed sources |
| --- | --- | --- |
| Room transition unit file | 17/17; original story/video/recovery/failure matrix retained, plus entry-only and invalid initial/entry cases | SGLang `b1aa8c84f126d8957e3973c88fb4cbeb6d3efcbe` |
| Real inference-off Yard→Forest→Yard | 6/6 | Worldkit `67c06237c4af2aad2c96a22bb21052d9d38d87d7` / SGLang `afeca13198cdc5538fcc1fedae3b37dfea3f16e9` |
| Original native story/video flow | 13/13, first attempt on the merged sources | Worldkit `d59c24048e65a03c1a8dba6a294fef4fbae5aadc` / SGLang `59d4f2f1a4f59f2e664ce2abe7ce2bd4c19cfe5a` |

The 13-check pair includes Worldkit dev
`db4a5b7e6ce21ba222a2ec932add3796b64a42d2` and SGLang main
`6247343cc2fd096cdb6a91ef62955f90d1fb2e64`. Later changes only update
documentation. The entry-only result retains its earlier source boundary.

The entry-only flow clicks the actual scene directory, checks both directions
under a 650 ms presentation hold, rejects attempted input and keeps
activation/ticks/damage unchanged. Uncovering restores input and activates
exactly once, 0→1→2. Native Player callbacks match both presentation receipts;
catalog preview images do not substitute for actual presentation. There is no
video or model lease/child/compute.

The 13 checks cover no-model video navigation, authored events/playback, both
video/prefill completion orders, covered gameplay/input/activation, frozen
condition identity, serialized children, real F0 distinct from S0, one parent
across A→B→A→B, fresh-Player recovery and exit while prefill is held.

CLI exit is 0, without forced kill or cleanup errors. Sources, Rooms, native
leases, worker children and CLI process groups are gone; ports are released and
subsequent process inspection finds no residue. The story parent is released.
Its only terminal wire error is deliberate `PLAYER_CANCELLED`. Two friendly
browser error callbacks and a runtime `WebSocketDisconnect (1011)` log are
retained. Recovery/exit assertions pass; those records are not described as
zero-error output or as a separately fixed warning.

## Test boundary and evidence

Only fixed Workbench CPU `i-uf6hh7iq2z2jcjyt7h3a` (`10.0.1.32`, Shanghai) is
used. A fresh namespace contains loopback and an internal virtual NIC pair, with
both ends/gateway inside that same namespace and no host peer/external interface.
Credentials and proxies are cleared; source/build/temp/cache files are isolated.

The CLI, Room, native RTC, Player and scene-directory component are real.
WTP/CDN and model/cache responses are local CPU fixtures. There are no
production/GPU requests, worker registrations or local CI runs. This verifies
protocol and lifecycle behavior, not deployed WTP, real GPU cache hits or model
quality. The final run verifies 6,923 SDK files and 1,778 present Room files;
3,748 unrelated Room catalog files are omitted from the sparse snapshot.

The original sealed inputs and their delivered runtime are unchanged:

- Entry-only SHA-256: `e417a4075b44807038885d01b398e073776a0b7ce5f959750c5eade530bee9a2`.
- Story/video SHA-256: `ca6459c00d2b23be7d74135f3957215b31b4813807ef698eb4222390b8be566e`.

[Replay and complete paired evidence](https://github.com/seedleap/worldkit-monorepo/blob/codex/scene-direct-jump-sdk-20261116/docs/superpowers/plans/2026-11-16-scene-entry-direct-jump.md).

Private archives are retained outside Git:

| Evidence | Local archive | SHA-256 |
| --- | --- | --- |
| Room/CLI unit regressions and controls | `/private/tmp/scene-direct-core-evidence-20261116.tar.gz` | `72da4606855abe3a93d49682cabac0c5f8e7aa7d84cc5c561350373951833ec2` |
| Entry-only native flow, 66 hashed files | `/private/tmp/scene-direct-isolation-evidence-20261116.tar.gz` | `3591ff11e365b1f084096cbbd2af3490244924d27fd27b427ee93f7460f6eda0` |
| Final story/video flow, 80 hashed files and sealed input | `/private/tmp/scene-direct-story-final-evidence-20261116.tar.gz` | `42225b0c48fd2c38278c67a2ee41b224238a47973fc0fe88ec1dd2a235dc225b` |

Archive and per-file hashes were verified after download.
