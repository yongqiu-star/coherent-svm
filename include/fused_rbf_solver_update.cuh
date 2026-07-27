/*
 * Copyright 2026 Jiang Yongchun
 * SPDX-License-Identifier: Apache-2.0
 *
 * Experimental dense FP32 RBF gradient update for cuML C-SVC.
 *
 * Computes f[i] += sum_j delta[j] * exp(-gamma * ||x_i - w_j||^2)
 * directly, without materializing the N-by-m kernel panel. Both X and W use
 * the column-major layout used by cuML's SVM implementation.
 */
#pragma once

#include <raft/util/cuda_utils.cuh>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>

// Runtime entry points compiled from the SM120 FP16 TMA kernel.  The tensor
// maps bind stable row-major X and working-set buffers once per fit.
extern "C" void* fp16_tma_rbf_create_context(const __half* a,
                                               const __half* b_col,
                                               int m,
                                               int n,
                                               int k);
extern "C" int fp16_tma_rbf_launch(void* context,
                               const float* norm_a,
                               const float* norm_b,
                               const float* delta,
                               const float* f_in,
                               float* partial,
                               float* out,
                               float gamma,
                               void* stream);
extern "C" int fp16_tma_rbf_launch_cache(void* context,
                                     const float* norm_a,
                                     const float* norm_b,
                                     const float* delta,
                                     float* cache,
                                     float* partial,
                                     float* output,
                                     int active_n,
                                     float gamma,
                                     void* stream);
extern "C" int fp16_tma_rbf_materialize_cache(void* context,
                                          const float* norm_a,
                                          const float* norm_b,
                                          const float* padded_delta,
                                          float* cache,
                                          int active_n,
                                          float gamma,
                                          void* stream);
extern "C" int fp16_tma_rbf_materialize_slotted_half_cache(
  void* context,
  const float* norm_a,
  const float* norm_b,
  const float* padded_delta,
  const int* slots,
  __half* cache,
  int active_n,
  float gamma,
  void* stream);
extern "C" int fp16_tma_rbf_materialize_slotted_float_cache(
  void* context,
  const float* norm_a,
  const float* norm_b,
  const float* padded_delta,
  const int* slots,
  float* cache,
  int active_n,
  float gamma,
  void* stream);
extern "C" void fp16_tma_rbf_destroy_context(void* context);

namespace ML::SVM::fused_rbf {

// cuML stores dense SVM inputs as [rows, dim] column-major.  The FP16 TMA
// mainloop consumes row-major K-contiguous records, so this one-time transform
// converts FP32 to FP16 and zero-pads K to a multiple of 64.
__global__ void repack_col_major_x(const float* __restrict__ source,
                                   __half* __restrict__ target,
                                   int rows,
                                   int dim,
                                   int dim_padded)
{
  const size_t elements = static_cast<size_t>(rows) * dim_padded;
  for (size_t index = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
       index < elements;
       index += static_cast<size_t>(blockDim.x) * gridDim.x) {
    const int row     = static_cast<int>(index / dim_padded);
    const int feature = static_cast<int>(index - static_cast<size_t>(row) * dim_padded);
    target[index] = feature < dim
                      ? __float2half_rn(source[row + static_cast<size_t>(feature) * rows])
                      : __float2half_rn(0.0f);
  }
}

// Gather the active delta-alpha rows into the stable row-major buffer bound to
// the TMA descriptor.  Padding is written explicitly because the selected
// bucket can be larger than nnz(delta_alpha).
__global__ void gather_fp16_tma_working_set(const __half* __restrict__ x_row_major,
                                            const int* __restrict__ indices,
                                            __half* __restrict__ working_set,
                                            int active_cols,
                                            int padded_cols,
                                            int dim_padded)
{
  const size_t elements = static_cast<size_t>(padded_cols) * dim_padded;
  for (size_t index = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
       index < elements;
       index += static_cast<size_t>(blockDim.x) * gridDim.x) {
    const int col     = static_cast<int>(index / dim_padded);
    const int feature = static_cast<int>(index - static_cast<size_t>(col) * dim_padded);
    working_set[index] = col < active_cols
                           ? x_row_major[static_cast<size_t>(indices[col]) * dim_padded + feature]
                           : __float2half_rn(0.0f);
  }
}

__global__ void prepare_fp16_tma_aux(const float* __restrict__ norm_x,
                                const int* __restrict__ indices,
                                const float* __restrict__ active_delta,
                                float* __restrict__ norm_w,
                                float* __restrict__ padded_delta,
                                int active_cols,
                                int padded_cols)
{
  for (int col = blockIdx.x * blockDim.x + threadIdx.x;
       col < padded_cols;
       col += blockDim.x * gridDim.x) {
    norm_w[col]       = col < active_cols ? norm_x[indices[col]] : 0.0f;
    padded_delta[col] = col < active_cols ? active_delta[col] : 0.0f;
  }
}

__global__ void build_signed_alpha(const float* __restrict__ alpha,
                                   const float* __restrict__ labels,
                                   float* __restrict__ signed_alpha,
                                   int rows)
{
  for (int row = blockIdx.x * blockDim.x + threadIdx.x;
       row < rows;
       row += blockDim.x * gridDim.x) {
    signed_alpha[row] = alpha[row] * labels[row];
  }
}

inline void launch_fp16_tma_repack(const float* source,
                                   __half* target,
                                   int rows,
                                   int dim,
                                   int dim_padded,
                                   cudaStream_t stream)
{
  constexpr int threads = 256;
  const size_t elements  = static_cast<size_t>(rows) * dim_padded;
  const int blocks = std::min<int>((elements + threads - 1) / threads, 65535);
  repack_col_major_x<<<blocks, threads, 0, stream>>>(source, target, rows, dim, dim_padded);
}

inline void launch_fp16_tma_gather(const __half* x_row_major,
                                   const int* indices,
                                   const float* active_delta,
                                   const float* norm_x,
                                   __half* working_set,
                                   float* norm_w,
                                   float* padded_delta,
                                   int active_cols,
                                   int padded_cols,
                                   int dim_padded,
                                   cudaStream_t stream)
{
  constexpr int threads = 256;
  const size_t elements  = static_cast<size_t>(padded_cols) * dim_padded;
  const int blocks = std::min<int>((elements + threads - 1) / threads, 65535);
  gather_fp16_tma_working_set<<<blocks, threads, 0, stream>>>(
    x_row_major, indices, working_set, active_cols, padded_cols, dim_padded);
  prepare_fp16_tma_aux<<<std::max(1, (padded_cols + threads - 1) / threads), threads, 0, stream>>>(
    norm_x, indices, active_delta, norm_w, padded_delta, active_cols, padded_cols);
}

inline void launch_signed_alpha(const float* alpha,
                                const float* labels,
                                float* signed_alpha,
                                int rows,
                                cudaStream_t stream)
{
  constexpr int threads = 256;
  const int blocks = std::min(raft::ceildiv(rows, threads), 65535);
  build_signed_alpha<<<blocks, threads, 0, stream>>>(alpha, labels, signed_alpha, rows);
}

template <int ROWS_PER_THREAD, int LANES>
__launch_bounds__(256) __global__ void update_d18_col_major(
  const float* __restrict__ x,
  const float* __restrict__ working_set,
  const float* __restrict__ norm_x,
  const float* __restrict__ norm_w,
  const float* __restrict__ delta,
  float* __restrict__ f,
  int rows,
  int cols,
  float gamma)
{
  constexpr int D      = 18;
  constexpr int TILE_M = 32;
  extern __shared__ float shared[];
  float* shared_w     = shared;
  float* shared_norm  = shared_w + TILE_M * D;
  float* shared_delta = shared_norm + TILE_M;

  const int first_row = blockIdx.x * blockDim.x * ROWS_PER_THREAD + threadIdx.x;
  float x_values[ROWS_PER_THREAD][D];
  float row_norms[ROWS_PER_THREAD];
  float accumulators[ROWS_PER_THREAD][LANES];

#pragma unroll
  for (int r = 0; r < ROWS_PER_THREAD; ++r) {
    const int row = first_row + r * blockDim.x;
    row_norms[r]  = row < rows ? norm_x[row] : 0.0f;
#pragma unroll
    for (int k = 0; k < D; ++k) {
      x_values[r][k] = row < rows ? x[row + static_cast<size_t>(k) * rows] : 0.0f;
    }
#pragma unroll
    for (int lane = 0; lane < LANES; ++lane) {
      accumulators[r][lane] = 0.0f;
    }
  }

  for (int tile_begin = 0; tile_begin < cols; tile_begin += TILE_M) {
    const int tile_cols = min(TILE_M, cols - tile_begin);
    for (int index = threadIdx.x; index < tile_cols * D; index += blockDim.x) {
      const int local_col = index / D;
      const int feature   = index - local_col * D;
      shared_w[index] =
        working_set[tile_begin + local_col + static_cast<size_t>(feature) * cols];
    }
    for (int local_col = threadIdx.x; local_col < tile_cols; local_col += blockDim.x) {
      shared_norm[local_col]  = norm_w[tile_begin + local_col];
      shared_delta[local_col] = delta[tile_begin + local_col];
    }
    __syncthreads();

    for (int local_col = 0; local_col < tile_cols; ++local_col) {
      const int lane = (tile_begin + local_col) % LANES;
#pragma unroll
      for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        float dot = 0.0f;
#pragma unroll
        for (int k = 0; k < D; ++k) {
          dot = fmaf(x_values[r][k], shared_w[local_col * D + k], dot);
        }
        const float distance = row_norms[r] + shared_norm[local_col] - 2.0f * dot;
        accumulators[r][lane] =
          fmaf(expf(-gamma * distance), shared_delta[local_col], accumulators[r][lane]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < ROWS_PER_THREAD; ++r) {
    const int row = first_row + r * blockDim.x;
    if (row < rows) {
      float update = 0.0f;
#pragma unroll
      for (int lane = 0; lane < LANES; ++lane) {
        update += accumulators[r][lane];
      }
      f[row] += update;
    }
  }
}

inline void launch(const float* x,
                   const float* working_set,
                   const float* norm_x,
                   const float* norm_w,
                   const float* delta,
                   float* f,
                   int rows,
                   int cols,
                   float gamma,
                   cudaStream_t stream)
{
  constexpr int threads         = 256;
  constexpr int rows_per_thread = 2;
  constexpr int lanes           = 4;
  constexpr int shared_floats   = 32 * 18 + 2 * 32;
  const int blocks = raft::ceildiv(rows, threads * rows_per_thread);
  update_d18_col_major<rows_per_thread, lanes>
    <<<blocks, threads, shared_floats * sizeof(float), stream>>>(
      x, working_set, norm_x, norm_w, delta, f, rows, cols, gamma);
}

// Consume a [cols, tile_rows] column-major dot-product panel.  A warp owns
// one training row, so lanes read consecutive working-set columns and reduce
// the RBF-weighted update without writing an N-by-cols kernel table.
__launch_bounds__(128) __global__ void reduce_transposed_panel(
  const float* __restrict__ panel,
  const float* __restrict__ norm_x,
  const float* __restrict__ norm_w,
  const float* __restrict__ delta,
  float* __restrict__ f,
  int row_offset,
  int tile_rows,
  int cols,
  float gamma)
{
  const int lane            = threadIdx.x & 31;
  const int warp            = threadIdx.x >> 5;
  constexpr int block_warps = 4;
  const int local_row       = blockIdx.x * block_warps + warp;
  if (local_row >= tile_rows) return;

  const int row       = row_offset + local_row;
  const float row_norm = norm_x[row];
  float update         = 0.0f;
  for (int col = lane; col < cols; col += 32) {
    const float dot = panel[col + static_cast<size_t>(local_row) * cols];
    const float distance = row_norm + norm_w[col] - 2.0f * dot;
    update = fmaf(expf(-gamma * distance), delta[col], update);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    update += __shfl_down_sync(0xffffffffu, update, offset);
  }
  if (lane == 0) { f[row] += update; }
}

// High-dimensional path: retain cuBLAS's tuned TF32 Tensor-Core mainloop, but
// materialize only one row tile in transposed layout and immediately consume
// it with the fused RBF reduction above.
inline void launch_tiled(cublasHandle_t cublas_handle,
                         const float* x,
                         const float* working_set,
                         const float* norm_x,
                         const float* norm_w,
                         const float* delta,
                         float* f,
                         float* panel,
                         int rows,
                         int dim,
                         int cols,
                         int tile_rows,
                         float gamma,
                         cudaStream_t stream,
                         cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F_FAST_TF32)
{
  constexpr int threads = 128;
  constexpr int warps   = threads / 32;
  const float alpha     = 1.0f;
  const float beta      = 0.0f;

  cublasPointerMode_t old_pointer_mode;
  RAFT_CUBLAS_TRY(cublasGetPointerMode(cublas_handle, &old_pointer_mode));
  if (old_pointer_mode != CUBLAS_POINTER_MODE_HOST) {
    RAFT_CUBLAS_TRY(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));
  }

  for (int row_offset = 0; row_offset < rows; row_offset += tile_rows) {
    const int current_rows = std::min(tile_rows, rows - row_offset);
    RAFT_CUBLAS_TRY(cublasGemmEx(cublas_handle,
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_T,
                                 cols,
                                 current_rows,
                                 dim,
                                 &alpha,
                                 working_set,
                                 CUDA_R_32F,
                                 cols,
                                 x + row_offset,
                                 CUDA_R_32F,
                                 rows,
                                 &beta,
                                 panel,
                                 CUDA_R_32F,
                                 cols,
                                 compute_type,
                                 compute_type == CUBLAS_COMPUTE_32F_FAST_TF32
                                   ? CUBLAS_GEMM_DEFAULT_TENSOR_OP
                                   : CUBLAS_GEMM_DEFAULT));
    const int blocks = raft::ceildiv(current_rows, warps);
    reduce_transposed_panel<<<blocks, threads, 0, stream>>>(
      panel, norm_x, norm_w, delta, f, row_offset, current_rows, cols, gamma);
  }

  if (old_pointer_mode != CUBLAS_POINTER_MODE_HOST) {
    RAFT_CUBLAS_TRY(cublasSetPointerMode(cublas_handle, old_pointer_mode));
  }
}

}  // namespace ML::SVM::fused_rbf
