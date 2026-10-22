# Lynn Decoder 的精确 attention 准备路径

Lynn 的流式 Decoder 在每个 tile batch 内跨 transformer 层复用 RoPE 几何，并直接
生成有界、独立存储的 pre-RoPE K/V 历史。在 SM90（H100/H200）与 SM103（B300）
设备上，对满足下述形状和布局契约的输入启用两个
Triton kernel 合并 Q RoPE、历史与当前 K/V 拼接、K RoPE 和下一步历史写入。
attention 仍调用原有 backend；不修改 FlashAttention 的 kernel 或调度。

## 调用链与两个 kernel 的分工

这两个 Triton kernel 是本 PR 新增的实现，定义在
[`fused_preparation.py`](../../../../../python/sglang/multimodal_gen/runtime/models/vaes/lynn_streaming/fused_preparation.py)。
[`attention.py::streaming_block`](../../../../../python/sglang/multimodal_gen/runtime/models/vaes/lynn_streaming/attention.py#L130)
在准入检查通过后调用 Python 包装函数 `prepare()`；包装函数分配输出张量，并用
`kernel[grid](...)` 语法实际发射两个 GPU kernel：

| kernel / 发射位置 | 一次完成的工作 |
| --- | --- |
| [`_query_rope`](../../../../../python/sglang/multimodal_gen/runtime/models/vaes/lynn_streaming/fused_preparation.py#L122) | 读取当前 Q，完成 half-split RoPE 的逐元素乘加和未旋转尾部复制，写出旋转后的 Q |
| [`_keys_values`](../../../../../python/sglang/multimodal_gen/runtime/models/vaes/lynn_streaming/fused_preparation.py#L136) | 按 token 索引直接读取历史或当前 K/V，替代单独的拼接；用当前相对位置系数旋转历史及当前 K，写出 attention 的 K/V；同时将保留窗口内的原始 pre-RoPE K/V 写入下一步历史 |

以下为解释用伪代码，完整流程仍有两次准备 kernel launch 和一次 attention 调用：

```python
# model.py 先生成 cos/sin，同一 tile batch 的各层复用这些系数。
query_rope, key_rope = build_or_reuse_rope_coefficients(history_frames)
q_rot = query_kernel(q, query_rope)
k_rot, all_v, next_k, next_v = kv_kernel(k, v, history_k, history_v, key_rope)
output = existing_attention_backend(q_rot, k_rot, all_v)
```

RoPE 系数的生成位于
[`model.py::_decode_tiles`](../../../../../python/sglang/multimodal_gen/runtime/models/vaes/lynn_streaming/model.py#L429)，
不在这两个 kernel 内。Q/K 投影和归一化也在调用前完成。`_rotate_values` 是两个
kernel 共用的 Triton JIT 辅助函数，不产生第三次独立 launch。下一步历史保存未旋转
的 K，使其能在下一 latent 使用新的相对时间重新旋转；它是 Decoder 历史写出，
与 DiT 的 KV 池或 CUDA Graph 回滚备份分别处理。

## 数值与可见域契约

- 当前块包含 256 个空间 token 和 5 个特殊 token。attention 读取全部当前 token；
  只有空间 token 进入下一步历史。
- 历史保留原始、归一化后且尚未旋转的 K，以及原始 V。下一 latent 按相对时间
  重新旋转历史，不能跨 latent 复用已旋转 K 或旧坐标。
- D64 的前 48 维使用 half-split RoPE，配对前后各 24 维；后 16 维原样传递。
  系数先转换为 BF16，两个乘积分别舍入为 BF16，再做加法并舍入为 BF16。
  融合 kernel 禁用 FMA，保持 Torch 路径的舍入顺序。
- 历史只拥有保留窗口的存储，不通过 slice 延长过期帧或特殊 token 的存储寿命。
  tile 的切片可以共享同一 tile batch 的有效存储，不共享 attention 临时输出。
- RoPE 几何缓存只存在于一次 tile batch 解码内，以历史帧数区分；不增加跨会话状态。

## 融合路径的条件

`attention._can_fuse_preparation` 检查 CUDA BF16、B 为 6/14/30、Q261/H32/D64、
空间 token 数 256、历史上限 512，且历史长度为 0/256/512。Q/K/V 和历史必须
连续且位于同一设备；cos/sin 必须同形状、同 stride、同设备，类型为 FP32 或
BF16，形状为 `[B, sequence, 1, 48]`。设备计算能力为 9.0 或 10.3，不限定 SM
数量。融合核仅使用逐元素加载、存储与 BF16 算术，不依赖 Blackwell 专用指令或
特定 SM 数量；硬件条件由运行路径和启动预编译共用。其他计算能力仍走 Torch。
形状、布局、窗口和精确舍入约束保持不变，不因放宽硬件条件而放宽。

其他形状、设备或布局使用 Torch 准备路径，同时保留层间 RoPE 几何复用和紧凑历史。
没有新增 backend、环境变量、自动调优或会话容量配置。

Torch 路径仍服务未测硬件、不同 tile batch、窗口和布局，因而不是待删除的旧接口。
选择在 attention 之前融合准备，是为了保留现有 backend 的数值与调度契约；
不直接依赖 FA4 私有 launcher。几何缓存只存于局部变量，历史只保留下一步需要的
pre-RoPE K/V，避免引入跨 latent 的旋转有效性状态。

## 服务启动预编译

`ZingH3CausalPipeline.create_pipeline_stages` 在构造 sampler 后、分配 KV 池前调用
`StreamingDecoder.warmup_preparation`，完成后才注册 stage 并进入 worker 就绪流程。
因此新增准备核的 JIT 不由用户第一、第二、第三个 latent 承担；通用
`--warmup-mode off` 只关闭请求式预热，不跳过这项模型初始化。

在符合融合条件的设备与模型配置上，预编译 B6/14/30 × 历史 0/256/512 ×
FP32/BF16 系数，共 18 种输入组合。系数由模型自身的 RoPE 生成，沿用真实布局；
模型权重此时仍可驻留 CPU，只分配有界的临时 GPU 准备张量，不执行 transformer
或创建会话。完成后同步当前设备，临时张量随函数返回释放。其他设备或配置返回 0，
继续使用原有 Torch 路径。

编译失败直接终止初始化，不能带着未完成的预热进入 worker 就绪状态。这个过程
不引入后台编译线程、请求级自动调优或跨会话张量状态，也不改变既有 FA4 / DiT
预热策略。编译成本计入服务启动时间，不能据此宣称已有 attention、GEMM 或 DiT 的
冷启动成本消失。

## 验证入口

CPU 测试覆盖非零残差权重、历史 aging、重复调用、清理、无历史窗口以及不完整的
最后一组 tiles。GPU 测试覆盖三档 batch、三种历史长度、两种系数类型及回退条件，
比较全部五个输出与 Torch 参考的严格相等性，并检查特殊 token 和存储独立性。
SM90/SM103 均执行该 GPU 测试矩阵及真实启动预编译测试，不以固定 SM 数跳过；
CPU 测试通过模拟设备属性检查两个架构及不同 SM 数的准入、其他架构的拒绝，
以及不支持设备在启动预编译时不分配 CUDA 张量。

### 当前硬件验证状态

- H100/H200（SM90）：生产源码 `c3cc5714eb` 已分别通过实机准备核精确比较、18 组合启动
  预编译及完整 26 层 Decoder 的 SP2 双 rank 回放；覆盖 832×480 与 1312×736，
  包括组件级 EOS、取消、断连、故障清理和下一 stream 检查。
- SM103：同一生产源码已通过准备核/Graph 单测、480/736 Decoder 回放与显存诊断；
  实测设备报告为 NVIDIA L20D、CC10.3、148 SM。明确标识的 B300 产品设备及其他
  SM 数配置仍待验证，不按报告名称推定产品身份。
- 上述组件回放不代替当前英伦 benchmark 的端到端、完整服务准确性和单活跃会话
  生命周期验收。736 的冻结 latent 来自历史 10 层 DiT 诊断，Decoder 仍为完整 26 层。
  各设备性能、显存及完整服务容量分别验收，不跨设备外推。

逐轮原始证据、精确源码身份和未验证范围以 PR 的验证记录为准；代码支持范围
不等于所有设备与完整服务均已验收。

```sh
PYTHONPATH=python SGLANG_LIGHTWEIGHT_RUNTIME=1 python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_zing_lynn_rope_reuse.py \
  python/sglang/multimodal_gen/test/unit/test_zing_lynn_fused_preparation.py
```

归档校验与当前 benchmark 的复现入口见
[验证工具](../../../../../benchmark/zing_h3_causal/lynn_decoder_attention/README.md)；
逐次实验结果、原始证据和当前版本验收状态见
[PR #356](https://github.com/seedleap/sglang/pull/356)。
