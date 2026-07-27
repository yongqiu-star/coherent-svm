/*
 * Copyright 2026 Jiang Yongchun
 * SPDX-License-Identifier: Apache-2.0
 *
 * Experimental on-demand FP32 RBF column cache for dense C-SVC.
 *
 * A column is generated once with a TF32 Tensor-Core GEMM, transformed to
 * RBF values in place, and then reused through a bandwidth-bound update.
 */
#pragma once

#include <raft/util/cuda_utils.cuh>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>

namespace ML::SVM::fused_rbf::column_cache {

__launch_bounds__(256) __global__ void convert_fp32_matrix_to_fp16(
  const float* __restrict__ input, __half* __restrict__ output, size_t elements)
{
  for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<size_t>(gridDim.x) * blockDim.x) {
    output[index] = __float2half_rn(input[index]);
  }
}

// Expand the persistent FP16 scalar representation into the FP32 tile consumed
// by SmoBlockSolve.  This is deliberately a representation conversion only:
// the RBF values have already been evaluated and rounded by the FP16 TMA
// materializer, so the block solver sees exactly the same per-entry values as
// the full-gradient cache path.
__launch_bounds__(256) __global__ void expand_fp16_matrix_to_fp32(
  const __half* __restrict__ input, float* __restrict__ output, size_t elements)
{
  for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<size_t>(gridDim.x) * blockDim.x) {
    output[index] = __half2float(input[index]);
  }
}

__launch_bounds__(256) __global__ void gather_fp16_rows(
  const __half* __restrict__ matrix,
  __half* __restrict__ working_set,
  const int* __restrict__ indices,
  int rows,
  int dim,
  int count)
{
  const size_t elements = static_cast<size_t>(dim) * count;
  for (size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < elements;
       linear += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const int output_row = static_cast<int>(linear % count);
    const int feature    = static_cast<int>(linear / count);
    working_set[linear] =
      matrix[static_cast<size_t>(feature) * rows + indices[output_row]];
  }
}

inline void prepare_fp16_matrix(const float* input,
                                __half* output,
                                size_t elements,
                                cudaStream_t stream)
{
  constexpr int threads = 256;
  const int blocks = static_cast<int>(
    std::min<size_t>((elements + threads - 1) / threads, 65535));
  convert_fp32_matrix_to_fp16<<<blocks, threads, 0, stream>>>(input, output, elements);
}

inline void expand_fp16_matrix(const __half* input,
                               float* output,
                               size_t elements,
                               cudaStream_t stream)
{
  if (elements == 0) { return; }
  constexpr int threads = 256;
  const int blocks = static_cast<int>(
    std::min<size_t>((elements + threads - 1) / threads, 65535));
  expand_fp16_matrix_to_fp32<<<blocks, threads, 0, stream>>>(input, output, elements);
}

inline void gather_fp16_working_set(const __half* matrix,
                                    __half* working_set,
                                    const int* indices,
                                    int rows,
                                    int dim,
                                    int count,
                                    cudaStream_t stream)
{
  if (count <= 0) { return; }
  constexpr int threads = 256;
  const size_t elements  = static_cast<size_t>(dim) * count;
  const int blocks = static_cast<int>(
    std::min<size_t>((elements + threads - 1) / threads, 65535));
  gather_fp16_rows<<<blocks, threads, 0, stream>>>(
    matrix, working_set, indices, rows, dim, count);
}

__launch_bounds__(256) __global__ void transform_store_and_update(
  float* __restrict__ columns,
  const float* __restrict__ norm_x,
  const float* __restrict__ norm_w,
  const float* __restrict__ delta,
  float* __restrict__ output,
  int rows,
  int count,
  float gamma)
{
  extern __shared__ float auxiliary[];
  float* shared_norm        = auxiliary;
  float* shared_delta       = auxiliary + count;
  for (int column = threadIdx.x; column < count; column += blockDim.x) {
    shared_norm[column]  = norm_w[column];
    shared_delta[column] = delta[column];
  }
  __syncthreads();

  for (int row = blockIdx.x * blockDim.x + threadIdx.x;
       row < rows;
       row += gridDim.x * blockDim.x) {
    const float row_norm = norm_x[row];
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    int column = 0;
    for (; column + 3 < count; column += 4) {
      const size_t offset0 = static_cast<size_t>(column) * rows + row;
      const size_t offset1 = offset0 + rows;
      const size_t offset2 = offset1 + rows;
      const size_t offset3 = offset2 + rows;
      const float value0 = expf(-gamma * fmaxf(row_norm + shared_norm[column] -
                                                2.0f * columns[offset0], 0.0f));
      const float value1 = expf(-gamma * fmaxf(row_norm + shared_norm[column + 1] -
                                                2.0f * columns[offset1], 0.0f));
      const float value2 = expf(-gamma * fmaxf(row_norm + shared_norm[column + 2] -
                                                2.0f * columns[offset2], 0.0f));
      const float value3 = expf(-gamma * fmaxf(row_norm + shared_norm[column + 3] -
                                                2.0f * columns[offset3], 0.0f));
      columns[offset0] = value0;
      columns[offset1] = value1;
      columns[offset2] = value2;
      columns[offset3] = value3;
      sum0 = fmaf(value0, shared_delta[column], sum0);
      sum1 = fmaf(value1, shared_delta[column + 1], sum1);
      sum2 = fmaf(value2, shared_delta[column + 2], sum2);
      sum3 = fmaf(value3, shared_delta[column + 3], sum3);
    }
    float update = (sum0 + sum1) + (sum2 + sum3);
    for (; column < count; ++column) {
      const size_t offset = static_cast<size_t>(column) * rows + row;
      const float value = expf(-gamma * fmaxf(row_norm + shared_norm[column] -
                                               2.0f * columns[offset], 0.0f));
      columns[offset] = value;
      update = fmaf(value, shared_delta[column], update);
    }
    output[row] += update;
  }
}

__launch_bounds__(256) __global__ void transform_store_only(
  float* __restrict__ columns,
  const float* __restrict__ norm_x,
  const float* __restrict__ norm_w,
  int rows,
  int count,
  float gamma)
{
  extern __shared__ float shared_norm[];
  for (int column = threadIdx.x; column < count; column += blockDim.x) {
    shared_norm[column] = norm_w[column];
  }
  __syncthreads();

  for (int row = blockIdx.x * blockDim.x + threadIdx.x;
       row < rows;
       row += gridDim.x * blockDim.x) {
    const float row_norm = norm_x[row];
    for (int column = 0; column < count; ++column) {
      const size_t offset = static_cast<size_t>(column) * rows + row;
      columns[offset] = expf(-gamma * fmaxf(row_norm + shared_norm[column] -
                                             2.0f * columns[offset], 0.0f));
    }
  }
}

__launch_bounds__(256) __global__ void apply_cached_columns(
  const float* __restrict__ cache,
  const int* __restrict__ slots,
  const float* __restrict__ delta,
  float* __restrict__ output,
  int rows,
  int count)
{
  extern __shared__ unsigned char storage[];
  int* shared_slots   = reinterpret_cast<int*>(storage);
  float* shared_delta = reinterpret_cast<float*>(shared_slots + count);
  for (int column = threadIdx.x; column < count; column += blockDim.x) {
    shared_slots[column] = slots[column];
    shared_delta[column] = delta[column];
  }
  __syncthreads();

  for (int row = blockIdx.x * blockDim.x + threadIdx.x;
       row < rows;
       row += gridDim.x * blockDim.x) {
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    int column = 0;
    for (; column + 3 < count; column += 4) {
      sum0 = fmaf(cache[static_cast<size_t>(shared_slots[column]) * rows + row],
                  shared_delta[column], sum0);
      sum1 = fmaf(cache[static_cast<size_t>(shared_slots[column + 1]) * rows + row],
                  shared_delta[column + 1], sum1);
      sum2 = fmaf(cache[static_cast<size_t>(shared_slots[column + 2]) * rows + row],
                  shared_delta[column + 2], sum2);
      sum3 = fmaf(cache[static_cast<size_t>(shared_slots[column + 3]) * rows + row],
                  shared_delta[column + 3], sum3);
    }
    float update = (sum0 + sum1) + (sum2 + sum3);
    for (; column < count; ++column) {
      update = fmaf(cache[static_cast<size_t>(shared_slots[column]) * rows + row],
                    shared_delta[column], update);
    }
    output[row] += update;
  }
}

// Reproduce the FP16 TMA epilogue's 128-column hierarchy: four lane-local sums
// per 64-column half, the same shuffle-tree pairing, then a left-to-right
// reduction across 128-column tiles. This keeps the SMO trajectory invariant
// when a kernel column is served from cache instead of recomputed.
__launch_bounds__(256) __global__ void apply_cached_columns_tiled_order(
  const float* __restrict__ cache,
  const int* __restrict__ slots,
  const float* __restrict__ delta,
  float* __restrict__ output,
  int rows,
  int count)
{
  extern __shared__ unsigned char storage[];
  int* shared_slots   = reinterpret_cast<int*>(storage);
  float* shared_delta = reinterpret_cast<float*>(shared_slots + count);
  for (int column = threadIdx.x; column < count; column += blockDim.x) {
    shared_slots[column] = slots[column];
    shared_delta[column] = delta[column];
  }
  __syncthreads();

  const int tile_count = (count + 127) / 128;
  for (int row = blockIdx.x * blockDim.x + threadIdx.x;
       row < rows;
       row += gridDim.x * blockDim.x) {
    float total = 0.0f;
    for (int tile = 0; tile < tile_count; ++tile) {
      float half_sum[2];
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        float lane_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int lane = 0; lane < 4; ++lane) {
#pragma unroll
          for (int wn = 0; wn < 4; ++wn) {
            const int base = tile * 128 + half * 64 + wn * 16 + lane * 2;
            const int columns[4] = {base, base + 1, base + 8, base + 9};
#pragma unroll
            for (int item = 0; item < 4; ++item) {
              const int column = columns[item];
              if (column < count) {
                lane_sum[lane] =
                  fmaf(cache[static_cast<size_t>(shared_slots[column]) * rows + row],
                       shared_delta[column],
                       lane_sum[lane]);
              }
            }
          }
        }
        half_sum[half] =
          (lane_sum[0] + lane_sum[2]) + (lane_sum[1] + lane_sum[3]);
      }
      total += half_sum[0] + half_sum[1];
    }
    output[row] = output[row] + total;
  }
}

// Same accumulation hierarchy as apply_cached_columns_tiled_order, but the
// persistent representation is FP16. Conversion and accumulation remain FP32.
__launch_bounds__(256) __global__ void apply_cached_columns_tiled_order_fp16(
  const __half* __restrict__ cache,
  const int* __restrict__ slots,
  const float* __restrict__ delta,
  float* __restrict__ output,
  int rows,
  int count)
{
  extern __shared__ unsigned char storage[];
  int* shared_slots   = reinterpret_cast<int*>(storage);
  float* shared_delta = reinterpret_cast<float*>(shared_slots + count);
  for (int column = threadIdx.x; column < count; column += blockDim.x) {
    shared_slots[column] = slots[column];
    shared_delta[column] = delta[column];
  }
  __syncthreads();

  const int tile_count = (count + 127) / 128;
  for (int row = blockIdx.x * blockDim.x + threadIdx.x;
       row < rows;
       row += gridDim.x * blockDim.x) {
    float total = 0.0f;
    for (int tile = 0; tile < tile_count; ++tile) {
      float half_sum[2];
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        float lane_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int lane = 0; lane < 4; ++lane) {
#pragma unroll
          for (int wn = 0; wn < 4; ++wn) {
            const int base = tile * 128 + half * 64 + wn * 16 + lane * 2;
            const int columns[4] = {base, base + 1, base + 8, base + 9};
#pragma unroll
            for (int item = 0; item < 4; ++item) {
              const int column = columns[item];
              if (column < count) {
                const float value = __half2float(
                  cache[static_cast<size_t>(shared_slots[column]) * rows + row]);
                lane_sum[lane] =
                  fmaf(value, shared_delta[column], lane_sum[lane]);
              }
            }
          }
        }
        half_sum[half] =
          (lane_sum[0] + lane_sum[2]) + (lane_sum[1] + lane_sum[3]);
      }
      total += half_sum[0] + half_sum[1];
    }
    output[row] = output[row] + total;
  }
}

__launch_bounds__(256) __global__ void scatter_cached_columns(
  const float* __restrict__ source,
  float* __restrict__ cache,
  const int* __restrict__ slots,
  int rows,
  int count)
{
  const int column = blockIdx.y;
  for (int row = blockIdx.x * blockDim.x + threadIdx.x;
       row < rows;
       row += gridDim.x * blockDim.x) {
    cache[static_cast<size_t>(slots[column]) * rows + row] =
      source[static_cast<size_t>(column) * rows + row];
  }
}

__launch_bounds__(256) __global__ void gather_cached_square_columns(
  const float* __restrict__ cache,
  const int* __restrict__ slots,
  const int* __restrict__ positions,
  const int* __restrict__ working_set_indices,
  float* __restrict__ square_tile,
  int rows,
  int working_set_size,
  int hit_count)
{
  const size_t elements = static_cast<size_t>(working_set_size) * hit_count;
  for (size_t linear = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
       linear < elements;
       linear += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const int row_position = static_cast<int>(linear % working_set_size);
    const int hit          = static_cast<int>(linear / working_set_size);
    const int source_row   = working_set_indices[row_position];
    const int target_col   = positions[hit];
    square_tile[row_position + static_cast<size_t>(target_col) * working_set_size] =
      cache[static_cast<size_t>(slots[hit]) * rows + source_row];
  }
}

__launch_bounds__(256) __global__ void gather_cached_square_columns_fp16(
  const __half* __restrict__ cache,
  const int* __restrict__ slots,
  const int* __restrict__ positions,
  const int* __restrict__ working_set_indices,
  float* __restrict__ square_tile,
  int rows,
  int working_set_size,
  int hit_count)
{
  const size_t elements = static_cast<size_t>(working_set_size) * hit_count;
  for (size_t linear = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
       linear < elements;
       linear += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const int row_position = static_cast<int>(linear % working_set_size);
    const int hit          = static_cast<int>(linear / working_set_size);
    const int source_row   = working_set_indices[row_position];
    const int target_col   = positions[hit];
    square_tile[row_position + static_cast<size_t>(target_col) * working_set_size] =
      __half2float(cache[static_cast<size_t>(slots[hit]) * rows + source_row]);
  }
}

__launch_bounds__(256) __global__ void transform_scatter_square_columns(
  const float* __restrict__ dot_panel,
  const float* __restrict__ norm_working_set,
  const float* __restrict__ norm_missing,
  const int* __restrict__ missing_positions,
  float* __restrict__ square_tile,
  int working_set_size,
  int missing_count,
  float gamma)
{
  const size_t elements = static_cast<size_t>(working_set_size) * missing_count;
  for (size_t linear = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
       linear < elements;
       linear += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const int row_position = static_cast<int>(linear % working_set_size);
    const int missing      = static_cast<int>(linear / working_set_size);
    const int target_col   = missing_positions[missing];
    const float distance = norm_working_set[row_position] + norm_missing[missing] -
                           2.0f * dot_panel[linear];
    square_tile[row_position + static_cast<size_t>(target_col) * working_set_size] =
      expf(-gamma * distance);
  }
}

inline void materialize(cublasHandle_t cublas_handle,
                        const float* x,
                        const float* working_set,
                        const float* norm_x,
                        const float* norm_w,
                        const float* delta,
                        float* output,
                        float* cache_destination,
                        int rows,
                        int dim,
                        int count,
                        float gamma,
                        cudaStream_t stream)
{
  if (count <= 0) { return; }
  const float alpha = 1.0f;
  const float beta  = 0.0f;
  cublasPointerMode_t old_pointer_mode;
  RAFT_CUBLAS_TRY(cublasGetPointerMode(cublas_handle, &old_pointer_mode));
  if (old_pointer_mode != CUBLAS_POINTER_MODE_HOST) {
    RAFT_CUBLAS_TRY(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));
  }
  RAFT_CUBLAS_TRY(cublasGemmEx(cublas_handle,
                               CUBLAS_OP_N,
                               CUBLAS_OP_T,
                               rows,
                               count,
                               dim,
                               &alpha,
                               x,
                               CUDA_R_32F,
                               rows,
                               working_set,
                               CUDA_R_32F,
                               count,
                               &beta,
                               cache_destination,
                               CUDA_R_32F,
                               rows,
                               CUBLAS_COMPUTE_32F_FAST_TF32,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  constexpr int threads = 256;
  const int blocks = std::min(raft::ceildiv(rows, threads), 65535);
  transform_store_and_update<<<blocks, threads, 2 * count * sizeof(float), stream>>>(
    cache_destination, norm_x, norm_w, delta, output, rows, count, gamma);
  if (old_pointer_mode != CUBLAS_POINTER_MODE_HOST) {
    RAFT_CUBLAS_TRY(cublasSetPointerMode(cublas_handle, old_pointer_mode));
  }
}

inline void materialize_only(cublasHandle_t cublas_handle,
                             const float* x,
                             const float* working_set,
                             const float* norm_x,
                             const float* norm_w,
                             float* cache_destination,
                             int rows,
                             int dim,
                             int count,
                             float gamma,
                             cublasComputeType_t compute_type,
                             cudaStream_t stream)
{
  if (count <= 0) { return; }
  const float alpha = 1.0f;
  const float beta  = 0.0f;
  cublasPointerMode_t old_pointer_mode;
  RAFT_CUBLAS_TRY(cublasGetPointerMode(cublas_handle, &old_pointer_mode));
  if (old_pointer_mode != CUBLAS_POINTER_MODE_HOST) {
    RAFT_CUBLAS_TRY(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));
  }
  RAFT_CUBLAS_TRY(cublasGemmEx(cublas_handle,
                               CUBLAS_OP_N,
                               CUBLAS_OP_T,
                               rows,
                               count,
                               dim,
                               &alpha,
                               x,
                               CUDA_R_32F,
                               rows,
                               working_set,
                               CUDA_R_32F,
                               count,
                               &beta,
                               cache_destination,
                               CUDA_R_32F,
                               rows,
                               compute_type,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  constexpr int threads = 256;
  const int blocks = std::min(raft::ceildiv(rows, threads), 65535);
  transform_store_only<<<blocks, threads, count * sizeof(float), stream>>>(
    cache_destination, norm_x, norm_w, rows, count, gamma);
  if (old_pointer_mode != CUBLAS_POINTER_MODE_HOST) {
    RAFT_CUBLAS_TRY(cublasSetPointerMode(cublas_handle, old_pointer_mode));
  }
}

inline void materialize_only_fp16(cublasHandle_t cublas_handle,
                                  const __half* x,
                                  const __half* working_set,
                                  const float* norm_x,
                                  const float* norm_w,
                                  float* cache_destination,
                                  int rows,
                                  int dim,
                                  int count,
                                  float gamma,
                                  cudaStream_t stream)
{
  if (count <= 0) { return; }
  const float alpha = 1.0f;
  const float beta  = 0.0f;
  cublasPointerMode_t old_pointer_mode;
  RAFT_CUBLAS_TRY(cublasGetPointerMode(cublas_handle, &old_pointer_mode));
  if (old_pointer_mode != CUBLAS_POINTER_MODE_HOST) {
    RAFT_CUBLAS_TRY(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));
  }
  RAFT_CUBLAS_TRY(cublasGemmEx(cublas_handle,
                               CUBLAS_OP_N,
                               CUBLAS_OP_T,
                               rows,
                               count,
                               dim,
                               &alpha,
                               x,
                               CUDA_R_16F,
                               rows,
                               working_set,
                               CUDA_R_16F,
                               count,
                               &beta,
                               cache_destination,
                               CUDA_R_32F,
                               rows,
                               CUBLAS_COMPUTE_32F,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  constexpr int threads = 256;
  const int blocks = std::min(raft::ceildiv(rows, threads), 65535);
  transform_store_only<<<blocks, threads, count * sizeof(float), stream>>>(
    cache_destination, norm_x, norm_w, rows, count, gamma);
  if (old_pointer_mode != CUBLAS_POINTER_MODE_HOST) {
    RAFT_CUBLAS_TRY(cublasSetPointerMode(cublas_handle, old_pointer_mode));
  }
}

// FP16 fallback for small working sets already packed for the TMA path.
// Both inputs are row-major with a K-contiguous, zero-padded leading
// dimension.  Viewed as column-major matrices they are Kxrows and Kxcount,
// so A^T*B produces the column-major [rows,count] cache layout directly.
inline void materialize_only_fp16_row_major(cublasHandle_t cublas_handle,
                                            const __half* x_row_major,
                                            const __half* working_set_row_major,
                                            const float* norm_x,
                                            const float* norm_w,
                                            float* cache_destination,
                                            int rows,
                                            int dim_padded,
                                            int count,
                                            float gamma,
                                            cudaStream_t stream)
{
  if (count <= 0) { return; }
  const float alpha = 1.0f;
  const float beta  = 0.0f;
  cublasPointerMode_t old_pointer_mode;
  RAFT_CUBLAS_TRY(cublasGetPointerMode(cublas_handle, &old_pointer_mode));
  if (old_pointer_mode != CUBLAS_POINTER_MODE_HOST) {
    RAFT_CUBLAS_TRY(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));
  }
  RAFT_CUBLAS_TRY(cublasGemmEx(cublas_handle,
                               CUBLAS_OP_T,
                               CUBLAS_OP_N,
                               rows,
                               count,
                               dim_padded,
                               &alpha,
                               x_row_major,
                               CUDA_R_16F,
                               dim_padded,
                               working_set_row_major,
                               CUDA_R_16F,
                               dim_padded,
                               &beta,
                               cache_destination,
                               CUDA_R_32F,
                               rows,
                               CUBLAS_COMPUTE_32F,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  constexpr int threads = 256;
  const int blocks = std::min(raft::ceildiv(rows, threads), 65535);
  transform_store_only<<<blocks, threads, count * sizeof(float), stream>>>(
    cache_destination, norm_x, norm_w, rows, count, gamma);
  if (old_pointer_mode != CUBLAS_POINTER_MODE_HOST) {
    RAFT_CUBLAS_TRY(cublasSetPointerMode(cublas_handle, old_pointer_mode));
  }
}

inline void apply(const float* cache,
                  const int* slots,
                  const float* delta,
                  float* output,
                  int rows,
                  int count,
                  cudaStream_t stream)
{
  if (count <= 0) { return; }
  constexpr int threads = 256;
  const int blocks = std::min(raft::ceildiv(rows, threads), 65535);
  apply_cached_columns<<<blocks, threads, 2 * count * sizeof(float), stream>>>(
    cache, slots, delta, output, rows, count);
}

inline void apply_tiled_order(const float* cache,
                            const int* slots,
                            const float* delta,
                            float* output,
                            int rows,
                            int count,
                            cudaStream_t stream)
{
  if (count <= 0) { return; }
  constexpr int threads = 256;
  const int blocks = std::min(raft::ceildiv(rows, threads), 65535);
  apply_cached_columns_tiled_order<<<blocks, threads, 2 * count * sizeof(float), stream>>>(
    cache, slots, delta, output, rows, count);
}

inline void apply_tiled_order_fp16(const __half* cache,
                                   const int* slots,
                                   const float* delta,
                                   float* output,
                                   int rows,
                                   int count,
                                   cudaStream_t stream)
{
  if (count <= 0) { return; }
  constexpr int threads = 256;
  const int blocks = std::min(raft::ceildiv(rows, threads), 65535);
  apply_cached_columns_tiled_order_fp16<<<blocks, threads,
                                          2 * count * sizeof(float), stream>>>(
    cache, slots, delta, output, rows, count);
}

// FP16 persistent-cache hits are always expanded into an FP32 block-solve
// tile. The active solver currently builds that tile through the stock FP32
// path; this helper preserves the contract if cache-hit reuse is enabled there.
inline void gather_square_tile_hits_fp16(const __half* cache,
                                         const int* hit_slots,
                                         const int* hit_positions,
                                         const int* working_set_indices,
                                         float* square_tile,
                                         int rows,
                                         int working_set_size,
                                         int hit_count,
                                         cudaStream_t stream)
{
  if (hit_count <= 0) { return; }
  constexpr int threads = 256;
  const size_t elements = static_cast<size_t>(working_set_size) * hit_count;
  const int blocks = std::min<int>(
    static_cast<int>((elements + threads - 1) / threads), 65535);
  gather_cached_square_columns_fp16<<<blocks, threads, 0, stream>>>(cache,
                                                                    hit_slots,
                                                                    hit_positions,
                                                                    working_set_indices,
                                                                    square_tile,
                                                                    rows,
                                                                    working_set_size,
                                                                    hit_count);
}

inline void scatter(const float* source,
                    float* cache,
                    const int* slots,
                    int rows,
                    int count,
                    cudaStream_t stream)
{
  if (count <= 0) { return; }
  constexpr int threads = 256;
  const dim3 blocks(std::min(raft::ceildiv(rows, threads), 65535), count);
  scatter_cached_columns<<<blocks, threads, 0, stream>>>(
    source, cache, slots, rows, count);
}

inline void build_square_tile(cublasHandle_t cublas_handle,
                              const float* working_set,
                              const float* missing_working_set,
                              const float* norm_working_set,
                              const float* norm_missing,
                              const int* working_set_indices,
                              const float* cache,
                              const int* hit_slots,
                              const int* hit_positions,
                              const int* missing_positions,
                              float* missing_dot_panel,
                              float* square_tile,
                              int rows,
                              int dim,
                              int working_set_size,
                              int hit_count,
                              int missing_count,
                              float gamma,
                              cudaStream_t stream)
{
  constexpr int threads = 256;
  if (hit_count > 0) {
    const size_t elements = static_cast<size_t>(working_set_size) * hit_count;
    const int blocks = std::min<int>(
      static_cast<int>((elements + threads - 1) / threads), 65535);
    gather_cached_square_columns<<<blocks, threads, 0, stream>>>(cache,
                                                                 hit_slots,
                                                                 hit_positions,
                                                                 working_set_indices,
                                                                 square_tile,
                                                                 rows,
                                                                 working_set_size,
                                                                 hit_count);
  }
  if (missing_count <= 0) { return; }

  const float alpha = 1.0f;
  const float beta  = 0.0f;
  cublasPointerMode_t old_pointer_mode;
  RAFT_CUBLAS_TRY(cublasGetPointerMode(cublas_handle, &old_pointer_mode));
  if (old_pointer_mode != CUBLAS_POINTER_MODE_HOST) {
    RAFT_CUBLAS_TRY(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));
  }
  RAFT_CUBLAS_TRY(cublasGemmEx(cublas_handle,
                               CUBLAS_OP_N,
                               CUBLAS_OP_T,
                               working_set_size,
                               missing_count,
                               dim,
                               &alpha,
                               working_set,
                               CUDA_R_32F,
                               working_set_size,
                               missing_working_set,
                               CUDA_R_32F,
                               missing_count,
                               &beta,
                               missing_dot_panel,
                               CUDA_R_32F,
                               working_set_size,
                               CUBLAS_COMPUTE_32F,
                               CUBLAS_GEMM_DEFAULT));
  const size_t elements = static_cast<size_t>(working_set_size) * missing_count;
  const int blocks = std::min<int>(
    static_cast<int>((elements + threads - 1) / threads), 65535);
  transform_scatter_square_columns<<<blocks, threads, 0, stream>>>(missing_dot_panel,
                                                                   norm_working_set,
                                                                   norm_missing,
                                                                   missing_positions,
                                                                   square_tile,
                                                                   working_set_size,
                                                                   missing_count,
                                                                   gamma);
  if (old_pointer_mode != CUBLAS_POINTER_MODE_HOST) {
    RAFT_CUBLAS_TRY(cublasSetPointerMode(cublas_handle, old_pointer_mode));
  }
}

}  // namespace ML::SVM::fused_rbf::column_cache
