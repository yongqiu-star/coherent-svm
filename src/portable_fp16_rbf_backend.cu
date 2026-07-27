// Copyright 2026 Jiang Yongchun
// SPDX-License-Identifier: Apache-2.0
//
// A path-independent reference backend for the coherent-FP16 protocol.
// It intentionally uses cuBLAS plus explicit CUDA epilogues instead of the
// unpublished architecture-tuned SM120 kernel used for the paper timings.
// The purpose of this file is semantic reproducibility and clean integration.

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <new>

namespace {

struct PortableRbfContext {
  const __half* x{};          // row-major [rows, dim]
  const __half* working{};    // row-major [columns, dim]
  int rows{};
  int columns{};
  int dim{};
  cublasHandle_t cublas{};
  float* dot_panel{};         // column-major [rows, columns]
};

inline int cuda_status(cudaError_t status) { return static_cast<int>(status); }

inline int cublas_status(cublasStatus_t status)
{
  return status == CUBLAS_STATUS_SUCCESS ? 0 : static_cast<int>(cudaErrorUnknown);
}

int compute_dot_panel(PortableRbfContext* context, int active_columns, cudaStream_t stream)
{
  if (active_columns <= 0 || active_columns > context->columns) {
    return cuda_status(cudaErrorInvalidValue);
  }
  auto status = cublasSetStream(context->cublas, stream);
  if (status != CUBLAS_STATUS_SUCCESS) { return cublas_status(status); }

  constexpr float alpha = 1.0f;
  constexpr float beta  = 0.0f;

  // Row-major X[rows,dim] is interpreted as a column-major [dim,rows]
  // matrix and transposed. Row-major W[columns,dim] is interpreted as a
  // column-major [dim,columns] matrix. The output is column-major [rows,columns].
  status = cublasGemmEx(context->cublas,
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        context->rows,
                        active_columns,
                        context->dim,
                        &alpha,
                        context->x,
                        CUDA_R_16F,
                        context->dim,
                        context->working,
                        CUDA_R_16F,
                        context->dim,
                        &beta,
                        context->dot_panel,
                        CUDA_R_32F,
                        context->rows,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return cublas_status(status);
}

template <typename CacheT>
__device__ CacheT encode_cache(float value);

template <>
__device__ float encode_cache<float>(float value)
{
  return value;
}

template <>
__device__ __half encode_cache<__half>(float value)
{
  return __float2half_rn(value);
}

template <typename CacheT>
__global__ void materialize_slotted_kernel(const float* __restrict__ dot,
                                           const float* __restrict__ norm_x,
                                           const float* __restrict__ norm_w,
                                           const int* __restrict__ slots,
                                           CacheT* __restrict__ cache,
                                           int rows,
                                           int columns,
                                           float gamma)
{
  const size_t total = static_cast<size_t>(rows) * columns;
  for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const int row    = static_cast<int>(index % rows);
    const int column = static_cast<int>(index / rows);
    const int slot   = slots == nullptr ? column : slots[column];
    const float distance = fmaxf(norm_x[row] + norm_w[column] - 2.0f * dot[index], 0.0f);
    cache[static_cast<size_t>(slot) * rows + row] = encode_cache<CacheT>(expf(-gamma * distance));
  }
}

__global__ void apply_float_cache_kernel(const float* __restrict__ cache,
                                         const float* __restrict__ delta,
                                         const float* __restrict__ initial,
                                         float* __restrict__ output,
                                         int rows,
                                         int columns)
{
  for (int row = blockIdx.x * blockDim.x + threadIdx.x;
       row < rows;
       row += blockDim.x * gridDim.x) {
    float update = initial == nullptr ? 0.0f : initial[row];
    for (int column = 0; column < columns; ++column) {
      update = fmaf(cache[static_cast<size_t>(column) * rows + row],
                    delta[column],
                    update);
    }
    output[row] = update;
  }
}

template <typename CacheT>
int materialize(PortableRbfContext* context,
                const float* norm_x,
                const float* norm_w,
                const int* slots,
                CacheT* cache,
                int active_columns,
                float gamma,
                cudaStream_t stream)
{
  const int status = compute_dot_panel(context, active_columns, stream);
  if (status != 0) { return status; }
  constexpr int threads = 256;
  const size_t elements = static_cast<size_t>(context->rows) * active_columns;
  const int blocks = static_cast<int>(std::min<size_t>((elements + threads - 1) / threads, 65535));
  materialize_slotted_kernel<<<blocks, threads, 0, stream>>>(context->dot_panel,
                                                             norm_x,
                                                             norm_w,
                                                             slots,
                                                             cache,
                                                             context->rows,
                                                             active_columns,
                                                             gamma);
  return cuda_status(cudaPeekAtLastError());
}

}  // namespace

extern "C" void* fp16_tma_rbf_create_context(const __half* x,
                                               const __half* working,
                                               int rows,
                                               int columns,
                                               int dim)
{
  if (x == nullptr || working == nullptr || rows <= 0 || columns <= 0 || dim <= 0) {
    return nullptr;
  }
  auto* context = new (std::nothrow) PortableRbfContext;
  if (context == nullptr) { return nullptr; }
  context->x       = x;
  context->working = working;
  context->rows    = rows;
  context->columns = columns;
  context->dim     = dim;

  if (cublasCreate(&context->cublas) != CUBLAS_STATUS_SUCCESS ||
      cudaMalloc(&context->dot_panel,
                 static_cast<size_t>(rows) * columns * sizeof(float)) != cudaSuccess) {
    if (context->cublas != nullptr) { cublasDestroy(context->cublas); }
    if (context->dot_panel != nullptr) { cudaFree(context->dot_panel); }
    delete context;
    return nullptr;
  }
  return context;
}

extern "C" int fp16_tma_rbf_launch(void* opaque,
                                    const float* norm_x,
                                    const float* norm_w,
                                    const float* delta,
                                    const float* initial,
                                    float* /*partial*/,
                                    float* output,
                                    float gamma,
                                    void* stream_pointer)
{
  auto* context = static_cast<PortableRbfContext*>(opaque);
  if (context == nullptr || norm_x == nullptr || norm_w == nullptr ||
      delta == nullptr || output == nullptr) {
    return cuda_status(cudaErrorInvalidValue);
  }
  auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  const int status = compute_dot_panel(context, context->columns, stream);
  if (status != 0) { return status; }
  materialize_slotted_kernel<<<std::min((context->rows * context->columns + 255) / 256, 65535),
                                        256,
                                        0,
                                        stream>>>(context->dot_panel,
                                                   norm_x,
                                                   norm_w,
                                                   nullptr,
                                                   context->dot_panel,
                                                   context->rows,
                                                   context->columns,
                                                   gamma);
  apply_float_cache_kernel<<<std::min((context->rows + 255) / 256, 65535), 256, 0, stream>>>(
    context->dot_panel, delta, initial, output, context->rows, context->columns);
  return cuda_status(cudaPeekAtLastError());
}

extern "C" int fp16_tma_rbf_launch_cache(void* opaque,
                                          const float* norm_x,
                                          const float* norm_w,
                                          const float* delta,
                                          float* cache,
                                          float* /*partial*/,
                                          float* output,
                                          int active_columns,
                                          float gamma,
                                          void* stream_pointer)
{
  auto* context = static_cast<PortableRbfContext*>(opaque);
  if (context == nullptr || norm_x == nullptr || norm_w == nullptr ||
      delta == nullptr || cache == nullptr || output == nullptr) {
    return cuda_status(cudaErrorInvalidValue);
  }
  auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  int status = materialize(context,
                           norm_x,
                           norm_w,
                           nullptr,
                           cache,
                           active_columns,
                           gamma,
                           stream);
  if (status != 0) { return status; }
  apply_float_cache_kernel<<<std::min((context->rows + 255) / 256, 65535), 256, 0, stream>>>(
    cache, delta, nullptr, output, context->rows, active_columns);
  return cuda_status(cudaPeekAtLastError());
}

extern "C" int fp16_tma_rbf_materialize_cache(void* opaque,
                                               const float* norm_x,
                                               const float* norm_w,
                                               const float* /*padded_delta*/,
                                               float* cache,
                                               int active_columns,
                                               float gamma,
                                               void* stream_pointer)
{
  auto* context = static_cast<PortableRbfContext*>(opaque);
  if (context == nullptr || norm_x == nullptr || norm_w == nullptr || cache == nullptr) {
    return cuda_status(cudaErrorInvalidValue);
  }
  return materialize(context,
                     norm_x,
                     norm_w,
                     nullptr,
                     cache,
                     active_columns,
                     gamma,
                     reinterpret_cast<cudaStream_t>(stream_pointer));
}

extern "C" int fp16_tma_rbf_materialize_slotted_half_cache(
  void* opaque,
  const float* norm_x,
  const float* norm_w,
  const float* /*padded_delta*/,
  const int* slots,
  __half* cache,
  int active_columns,
  float gamma,
  void* stream_pointer)
{
  auto* context = static_cast<PortableRbfContext*>(opaque);
  if (context == nullptr || norm_x == nullptr || norm_w == nullptr ||
      slots == nullptr || cache == nullptr) {
    return cuda_status(cudaErrorInvalidValue);
  }
  return materialize(context,
                     norm_x,
                     norm_w,
                     slots,
                     cache,
                     active_columns,
                     gamma,
                     reinterpret_cast<cudaStream_t>(stream_pointer));
}

extern "C" int fp16_tma_rbf_materialize_slotted_float_cache(
  void* opaque,
  const float* norm_x,
  const float* norm_w,
  const float* /*padded_delta*/,
  const int* slots,
  float* cache,
  int active_columns,
  float gamma,
  void* stream_pointer)
{
  auto* context = static_cast<PortableRbfContext*>(opaque);
  if (context == nullptr || norm_x == nullptr || norm_w == nullptr ||
      slots == nullptr || cache == nullptr) {
    return cuda_status(cudaErrorInvalidValue);
  }
  return materialize(context,
                     norm_x,
                     norm_w,
                     slots,
                     cache,
                     active_columns,
                     gamma,
                     reinterpret_cast<cudaStream_t>(stream_pointer));
}

extern "C" void fp16_tma_rbf_destroy_context(void* opaque)
{
  auto* context = static_cast<PortableRbfContext*>(opaque);
  if (context == nullptr) { return; }
  if (context->dot_panel != nullptr) { cudaFree(context->dot_panel); }
  if (context->cublas != nullptr) { cublasDestroy(context->cublas); }
  delete context;
}
