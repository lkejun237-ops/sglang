<!-- Thank you for your contribution! Please follow these guidelines to enhance your pull request. If anything is unclear, submit your PR and reach out to maintainers for assistance. Join our Slack community at https://slack.sglang.io to discuss further. -->

## Motivation

<!-- Describe the purpose and goals of this pull request. -->

## Modifications

<!-- Detail the changes made in this pull request. -->

## Accuracy Tests

<!-- If this pull request affects model outputs (e.g., changes to the kernel or model forward code), provide accuracy test results. -->

## Speed Tests and Profiling

<!-- If this pull request impacts inference speed, provide benchmarking and profiling results. -->

<!-- Zing: use YingLun's benchmark (benchmark/zing_h3_causal/CLIENT_BENCHMARK.md).
Report 480 and 736 resolution results separately, with exact width x height.
For Zing H3 performance or memory changes, update
docs/docs/sglang-diffusion/zing/deployment_zing_h3.mdx in this PR: affected
strategy/results or an explicit unverified status. Keep recipe evidence concise
(date/short SHA, hardware/shape, key settings, result and one traceable link).
For every Zing H3 PR affecting speed or GPU memory, compare baseline vs. change
for both E2E metrics (TTFF, E2E, full-run/post-first-batch FPS, TPOC, ICL
median/P95 and sample counts) and ALL THREE components: Encoder, DiT, Decoder.
For each component, report latency median/P95, sample count, resident and
request-peak GPU memory, plus before/after deltas. Break Encoder into Qwen,
reference VAE and source VAE. Identify per-rank scope, allocated/reserved vs.
driver memory, warmup/load/request boundaries and shared-process/device limits.
Do not label shared memory as component-exclusive or sum independent peaks.
State missing measurements, reasons and follow-up plans; unchanged code alone
is not measurement evidence. See CLIENT_BENCHMARK.md for the report table.
For continuous source, include the applicable source-to-output, tail-drain,
credit-wait and playback-stall metrics. State missing/unsupported measurements.
Keep hardware, checkpoint, cases, sampling and deployment settings comparable;
include source SHAs, commands, warmup/repeat counts, failures and raw JSON/report
links. Identify Host/CUDA timing and collection overhead. See AGENTS.md for the
full contract; microbenchmarks alone do not establish E2E improvement. -->

## Checklist

- [ ] For Zing H3 speed/memory changes, report baseline/candidate latency and resident/peak memory for Encoder, DiT and Decoder; identify missing data and shared-memory attribution limits.
- [ ] For Zing H3 performance/memory changes, update the [deployment recipe](https://github.com/seedleap/sglang/blob/main/docs/docs/sglang-diffusion/zing/deployment_zing_h3.mdx), including unverified hardware/configurations.
- [ ] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [ ] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [ ] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [ ] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed).
- [ ] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).

## Review and Merge Process

1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process).
2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers.
3. Trigger CI tests with [comments](https://docs.sglang.io/developer_guide/contribution_guide.html#how-to-trigger-ci-tests) or contact authorized users to do so.
   - Common commands include `/tag-and-rerun-ci`, `/tag-run-ci-label`, `/rerun-failed-ci`
4. After green CI and required approvals, ask Merge Oncalls or people with Write permission to merge the PR.
