# Benchmark 目录约定

本目录保存基准测试工具、测试输入、复现说明和实验／验收记录。
架构设计、接口契约、实现方案及设计分析统一放在 `docs/` 对应主题下，
不放在 `benchmark/` 的任何子目录中。

多模态模型按以下结构归档：

| 文档内容 | 位置 |
| --- | --- |
| 架构、接口、运行时契约与设计说明 | `docs/docs/sglang-diffusion/<model>/specs/` |
| 实现计划与设计分析 | `docs/docs/sglang-diffusion/<model>/plans/` |
| 测试方法、输入、测量结果与验收证据 | `benchmark/<suite>/` |

设计与测试材料通过链接相互引用，避免在两个目录维护同一份设计。
实验记录中的实现背景和数值门槛可以与证据保存在一起；长期设计和接口的维护入口应在 `docs/`。

模型设计入口：[MinWM](../docs/docs/sglang-diffusion/minwm/README.md)、
[Zing](../docs/docs/sglang-diffusion/zing/README.md)。
