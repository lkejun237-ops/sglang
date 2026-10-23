#include <sgl_kernel/tensor.h>

#include <sgl_kernel/runtime.cuh>

#include <sgl_kernel/impl/norm.cuh>

namespace sglang {

struct ZingQKNormRopeKVParams {
  const bf16_t *q, *k, *v, *qw, *kw, *coefficients;
  const void* rows;
  bf16_t *out_q, *raw_k, *rotated_k, *out_v;
  int64_t qs, qh, ks, kh, vs, vh;
  uint32_t tokens, heads, clean;
  float eps;
};

// Use the same warp reduction and BF16 norm boundary as fused_inplace_qknorm.
// A warp owns one Q or K head; its K worker also persists raw K and copies V.
template <int kRotary, typename Id>
__global__ void zing_qknorm_rope_kv(const ZingQKNormRopeKVParams __grid_constant__ p) {
  using namespace device;
  using Storage = norm::StorageType<bf16_t, 128>;
  constexpr int kWarps = 4;
  constexpr int kLanes = kRotary / 4;
  constexpr int kHalfLanes = kLanes / 2;
  constexpr uint32_t kMask = 0xffffffffu >> (32 - kLanes);
  const uint32_t lane = threadIdx.x % 32;
  const uint32_t worker = blockIdx.x * kWarps + threadIdx.x / 32;
  for (uint32_t i = worker; i < p.tokens * p.heads * 2; i += gridDim.x * kWarps) {
    const uint32_t token = i / (2 * p.heads);
    const uint32_t head = i % p.heads;
    const bool is_q = i % (2 * p.heads) < p.heads;
    const auto input = is_q ? p.q + token * p.qs + head * p.qh : p.k + token * p.ks + head * p.kh;
    const auto weight = is_q ? p.qw : p.kw;
    auto normalized = norm::apply_norm_warp<128>(load_as<Storage>(input, lane), load_as<Storage>(weight, lane), p.eps);
    const int64_t row = is_q ? token : static_cast<const Id*>(p.rows)[token];
    const int64_t offset = (row * p.heads + head) * 128;
    if (!is_q) {
      if (token < p.clean) store_as<Storage>(p.raw_k + offset, normalized, lane);
      const auto value = load_as<Storage>(p.v + token * p.vs + head * p.vh, lane);
      store_as<Storage>(p.out_v + offset, value, lane);
    }
    if (lane < kLanes) {
      const uint32_t partner = lane < kHalfLanes ? lane + kHalfLanes : lane - kHalfLanes;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        auto other = normalized[j];
        auto bits = reinterpret_cast<const uint32_t&>(other);
        bits = __shfl_sync(kMask, bits, partner);
        reinterpret_cast<uint32_t&>(other) = bits;
        auto& x = unpack(normalized[j]);
        const auto& y = unpack(other);
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          const int coeff = token * kRotary + (lane % kHalfLanes) * 4 + 2 * j + c;
          const float cos = __bfloat162float(p.coefficients[coeff]);
          const float sin = __bfloat162float(p.coefficients[coeff + kRotary / 2]);
          const auto a = __float2bfloat16_rn(__fmul_rn(__bfloat162float(x[c]), cos));
          const float partner_value = __bfloat162float(y[c]) * (lane < kHalfLanes ? -1.0f : 1.0f);
          const auto b = __float2bfloat16_rn(__fmul_rn(partner_value, sin));
          x[c] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(a), __bfloat162float(b)));
        }
      }
    }
    store_as<Storage>((is_q ? p.out_q : p.rotated_k) + offset, normalized, lane);
  }
}

template <int kRotary>
struct ZingQKNormRopeKVKernel {
  static_assert(kRotary > 0 && kRotary <= 128 && kRotary % 8 == 0);
  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView qw,
      const tvm::ffi::TensorView kw,
      const tvm::ffi::TensorView coefficients,
      const tvm::ffi::TensorView rows,
      const tvm::ffi::TensorView out_q,
      const tvm::ffi::TensorView raw_k,
      const tvm::ffi::TensorView rotated_k,
      const tvm::ffi::TensorView out_v,
      int64_t clean,
      float eps) {
    using namespace host;
    auto N = SymbolicSize{"tokens"}, H = SymbolicSize{"heads"}, C = SymbolicSize{"capacity"};
    auto QS = SymbolicSize{"qs"}, QH = SymbolicSize{"qh"};
    auto KS = SymbolicSize{"ks"}, KH = SymbolicSize{"kh"};
    auto VS = SymbolicSize{"vs"}, VH = SymbolicSize{"vh"};
    auto device = SymbolicDevice{};
    auto id_type = SymbolicDType{};
    device.set_options<kDLCUDA>();
    TensorMatcher({N, H, 128}).with_strides({QS, QH, 1}).with_dtype<bf16_t>().with_device(device).verify(q);
    TensorMatcher({N, H, 128}).with_strides({KS, KH, 1}).with_dtype<bf16_t>().with_device(device).verify(k);
    TensorMatcher({N, H, 128}).with_strides({VS, VH, 1}).with_dtype<bf16_t>().with_device(device).verify(v);
    TensorMatcher({128}).with_dtype<bf16_t>().with_device(device).verify(qw).verify(kw);
    TensorMatcher({N, kRotary}).with_dtype<bf16_t>().with_device(device).verify(coefficients);
    TensorMatcher({N}).with_dtype<int32_t, int64_t>(id_type).with_device(device).verify(rows);
    TensorMatcher({N, H, 128}).with_dtype<bf16_t>().with_device(device).verify(out_q);
    TensorMatcher({C, H, 128}).with_dtype<bf16_t>().with_device(device).verify(raw_k).verify(rotated_k).verify(out_v);
    RuntimeCheck(clean >= 0 && clean <= N.unwrap(), "Invalid clean prefix length");
    if (N.unwrap() == 0 || H.unwrap() == 0) return;
    const ZingQKNormRopeKVParams p{
        static_cast<const bf16_t*>(q.data_ptr()),
        static_cast<const bf16_t*>(k.data_ptr()),
        static_cast<const bf16_t*>(v.data_ptr()),
        static_cast<const bf16_t*>(qw.data_ptr()),
        static_cast<const bf16_t*>(kw.data_ptr()),
        static_cast<const bf16_t*>(coefficients.data_ptr()),
        rows.data_ptr(),
        static_cast<bf16_t*>(out_q.data_ptr()),
        static_cast<bf16_t*>(raw_k.data_ptr()),
        static_cast<bf16_t*>(rotated_k.data_ptr()),
        static_cast<bf16_t*>(out_v.data_ptr()),
        QS.unwrap(),
        QH.unwrap(),
        KS.unwrap(),
        KH.unwrap(),
        VS.unwrap(),
        VH.unwrap(),
        static_cast<uint32_t>(N.unwrap()),
        static_cast<uint32_t>(H.unwrap()),
        static_cast<uint32_t>(clean),
        eps};
    const auto kernel =
        id_type.is_type<int32_t>() ? zing_qknorm_rope_kv<kRotary, int32_t> : zing_qknorm_rope_kv<kRotary, int64_t>;
    const auto max_blocks = runtime::get_sm_count(device.unwrap().device_id) * runtime::get_blocks_per_sm(kernel, 128);
    const auto blocks = std::min(max_blocks, div_ceil(p.tokens * p.heads * 2, 4u));
    LaunchKernel(blocks, 128, device.unwrap())(kernel, p);
  }
};

}  // namespace sglang
