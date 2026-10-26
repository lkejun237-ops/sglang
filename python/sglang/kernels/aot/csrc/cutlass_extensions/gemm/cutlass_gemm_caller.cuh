// Adapted from
// https://github.com/vllm-project/vllm/blob/main/csrc/quantization/cutlass_w8a8/c3x/cutlass_gemm_caller.cuh

#pragma once

// clang-format will break include orders
// clang-format off
#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>

#include <optional>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

// clang-format on

/**
 * Helper function for checking CUTLASS errors
 */
#define CUTLASS_CHECK(status)                                                       \
  {                                                                                 \
    cutlass::Status error = status;                                                 \
    TORCH_CHECK(error == cutlass::Status::kSuccess, cutlassGetStatusString(error)); \
  }

template <typename GemmKernel>
void cutlass_gemm_caller(
    torch::Device device,
    cute::Shape<int, int, int, int> prob_shape,
    typename GemmKernel::MainloopArguments mainloop_args,
    typename GemmKernel::EpilogueArguments epilogue_args,
    typename GemmKernel::TileSchedulerArguments scheduler = {},
    std::optional<torch::Tensor> workspace_opt = std::nullopt) {
  c10::cuda::CUDAGuard device_guard(device);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device.index();
  hw_info.sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, prob_shape, mainloop_args, epilogue_args, hw_info, scheduler};

  // Launch the CUTLASS GEMM kernel.
  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  GemmOp gemm_op;
  CUTLASS_CHECK(gemm_op.can_implement(args));

  size_t workspace_size = gemm_op.get_workspace_size(args);
  torch::Tensor workspace;
  bool is_internal_workspace = false;
  if (workspace_opt.has_value() && workspace_opt->defined()) {
    TORCH_CHECK(
        workspace_opt->device() == device,
        "Provided workspace device (",
        workspace_opt->device(),
        ") does not match target device (",
        device,
        ")");
    TORCH_CHECK(
        static_cast<size_t>(workspace_opt->nbytes()) >= workspace_size,
        "Provided workspace is smaller than required CUTLASS workspace: ",
        workspace_opt->nbytes(), " < ", workspace_size);
    workspace = *workspace_opt;
  } else {
    auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(device);
    workspace = torch::empty(workspace_size, workspace_options);
    is_internal_workspace = true;
  }

  auto stream = at::cuda::getCurrentCUDAStream(device.index());

  CUTLASS_CHECK(gemm_op.initialize(args, workspace.data_ptr(), stream));
  CUTLASS_CHECK(gemm_op.run(stream));

  if (is_internal_workspace && workspace_size > 0) {
    c10::cuda::CUDACachingAllocator::recordStream(workspace.storage().data_ptr(), stream);
  }
}
