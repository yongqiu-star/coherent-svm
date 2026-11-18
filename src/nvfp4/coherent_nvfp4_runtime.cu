// Standalone coherent NVFP4 primitive runtime for sm_120a.
//
// The frozen v10h mainloop is included without modification. This wrapper owns
// one immutable packed representation of X and uses it for both X*W^T and
// W*W^T. Both views apply the same BF16-dot -> RBF -> FP16 map.

#ifndef COHERENT_NVFP4_V10H_SOURCE
#error "COHERENT_NVFP4_V10H_SOURCE must name the frozen v10h .cu source"
#endif

#define main coherent_nvfp4_frozen_v10h_main
#include COHERENT_NVFP4_V10H_SOURCE
#undef main

#include "coherent_nvfp4_runtime.h"

#include <array>
#include <new>

namespace {

constexpr std::uint32_t kUnitScaleWord = 0x38383838u;

__device__ __forceinline__ float encode_decode_e4m3_scale(float maximum)
{
  if (!(maximum > 0.0f)) { return 1.0f; }
  const float requested = fmaxf(maximum * (1.0f / 6.0f), 1.0e-6f);
  const __nv_fp8_e4m3 encoded(requested);
  const float decoded = static_cast<float>(encoded);
  return decoded > 0.0f ? decoded : 1.0e-6f;
}

__device__ __forceinline__ float decode_e4m3(std::uint8_t bits)
{
  __nv_fp8_e4m3 value{};
  value.__x = bits;
  return static_cast<float>(value);
}

__device__ __forceinline__ float decode_e2m1(std::uint8_t bits)
{
  __nv_fp4_e2m1 value{};
  value.__x = bits & 0x0fu;
  return static_cast<float>(value);
}

__device__ __forceinline__ float decode_packed_value(
  const std::uint8_t* packed,
  const std::uint32_t* scale_row_major,
  int row,
  int feature,
  int features)
{
  const int row_bytes = features / PACK_RATIO;
  const std::uint8_t byte =
    packed[static_cast<std::size_t>(row) * row_bytes + feature / 2];
  const std::uint8_t nibble =
    (feature & 1) ? static_cast<std::uint8_t>(byte >> 4)
                  : static_cast<std::uint8_t>(byte & 0x0f);
  const int word = feature / MMA_K;
  const int group = (feature % MMA_K) / SCALE_GROUP;
  const std::uint32_t scale_word =
    scale_row_major[static_cast<std::size_t>(row) * (features / MMA_K) + word];
  const std::uint8_t scale_bits =
    static_cast<std::uint8_t>((scale_word >> (8 * group)) & 0xffu);
  return decode_e2m1(nibble) * decode_e4m3(scale_bits);
}

__global__ void quantize_rows(const float* __restrict__ source,
                              std::uint8_t* __restrict__ packed,
                              std::uint32_t* __restrict__ scale_row_major,
                              int rows,
                              int features)
{
  const int words_per_row = features / MMA_K;
  const long long total = static_cast<long long>(rows) * words_per_row;
  for (long long task = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       task < total;
       task += static_cast<long long>(blockDim.x) * gridDim.x) {
    const int row = static_cast<int>(task / words_per_row);
    const int word = static_cast<int>(task - static_cast<long long>(row) * words_per_row);
    const int first = word * MMA_K;
    float scales[4];
#pragma unroll
    for (int group = 0; group < 4; ++group) {
      float maximum = 0.0f;
#pragma unroll
      for (int lane = 0; lane < 16; ++lane) {
        maximum = fmaxf(maximum,
                        fabsf(source[static_cast<long long>(row) * features + first +
                                     group * 16 + lane]));
      }
      scales[group] = encode_decode_e4m3_scale(maximum);
    }
    const __nv_fp8x4_e4m3 packed_scale(
      make_float4(scales[0], scales[1], scales[2], scales[3]));
    scale_row_major[task] = packed_scale.__x;

    std::uint16_t* destination = reinterpret_cast<std::uint16_t*>(
      packed + static_cast<long long>(row) * (features / 2) + word * (MMA_K / 2));
#pragma unroll
    for (int group4 = 0; group4 < MMA_K / 4; ++group4) {
      const int feature = first + group4 * 4;
      const float scale = scales[group4 / 4];
      const __nv_fp4x4_e2m1 encoded(make_float4(
        source[static_cast<long long>(row) * features + feature + 0] / scale,
        source[static_cast<long long>(row) * features + feature + 1] / scale,
        source[static_cast<long long>(row) * features + feature + 2] / scale,
        source[static_cast<long long>(row) * features + feature + 3] / scale));
      destination[group4] = *reinterpret_cast<const std::uint16_t*>(&encoded);
    }
  }
}

__global__ void quantize_columns(const float* __restrict__ source,
                                 std::uint8_t* __restrict__ packed,
                                 std::uint32_t* __restrict__ scale_row_major,
                                 int rows,
                                 int source_features,
                                 int padded_features)
{
  const int words_per_row = padded_features / MMA_K;
  const long long total = static_cast<long long>(rows) * words_per_row;
  for (long long task = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       task < total;
       task += static_cast<long long>(blockDim.x) * gridDim.x) {
    const int row = static_cast<int>(task / words_per_row);
    const int word = static_cast<int>(task - static_cast<long long>(row) * words_per_row);
    const int first = word * MMA_K;
    float scales[4];
#pragma unroll
    for (int group = 0; group < 4; ++group) {
      float maximum = 0.0f;
#pragma unroll
      for (int lane = 0; lane < 16; ++lane) {
        const int feature = first + group * 16 + lane;
        const float value = feature < source_features
                              ? source[static_cast<long long>(feature) * rows + row]
                              : 0.0f;
        maximum = fmaxf(maximum, fabsf(value));
      }
      scales[group] = encode_decode_e4m3_scale(maximum);
    }
    const __nv_fp8x4_e4m3 packed_scale(
      make_float4(scales[0], scales[1], scales[2], scales[3]));
    scale_row_major[task] = packed_scale.__x;

    std::uint16_t* destination = reinterpret_cast<std::uint16_t*>(
      packed + static_cast<long long>(row) * (padded_features / 2) +
      word * (MMA_K / 2));
#pragma unroll
    for (int group4 = 0; group4 < MMA_K / 4; ++group4) {
      const int feature = first + group4 * 4;
      float values[4];
#pragma unroll
      for (int lane = 0; lane < 4; ++lane) {
        const int source_feature = feature + lane;
        values[lane] = source_feature < source_features
                         ? source[static_cast<long long>(source_feature) * rows + row]
                         : 0.0f;
      }
      const float scale = scales[group4 / 4];
      const __nv_fp4x4_e2m1 encoded(make_float4(values[0] / scale,
                                                values[1] / scale,
                                                values[2] / scale,
                                                values[3] / scale));
      destination[group4] = *reinterpret_cast<const std::uint16_t*>(&encoded);
    }
  }
}

__global__ void repack_scales(const std::uint32_t* __restrict__ row_major,
                              std::uint32_t* __restrict__ tiled,
                              int rows,
                              int features)
{
  const int words_per_row = features / MMA_K;
  const int k_tiles = features / BK;
  const long long total = static_cast<long long>(rows) * words_per_row;
  for (long long task = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       task < total;
       task += static_cast<long long>(blockDim.x) * gridDim.x) {
    const int row = static_cast<int>(task / words_per_row);
    const int word = static_cast<int>(task - static_cast<long long>(row) * words_per_row);
    const int row_tile = row / BM;
    const int row_local = row - row_tile * BM;
    const int k_tile = word / K_FRAGS;
    const int fragment = word - k_tile * K_FRAGS;
    const std::size_t destination =
      (static_cast<std::size_t>(row_tile) * k_tiles + k_tile) * (BM * K_FRAGS) +
      row_local * K_FRAGS + fragment;
    tiled[destination] = row_major[task];
  }
}

__global__ void coherent_row_norms(const std::uint8_t* __restrict__ packed,
                                   const std::uint32_t* __restrict__ scale_row_major,
                                   float* __restrict__ norms,
                                   int rows,
                                   int features)
{
  constexpr int warps_per_block = 8;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warp_stride = gridDim.x * warps_per_block;
  for (int row = blockIdx.x * warps_per_block + warp; row < rows; row += warp_stride) {
    float sum = 0.0f;
    for (int feature = lane; feature < features; feature += 32) {
      const float value =
        decode_packed_value(packed, scale_row_major, row, feature, features);
      sum = fmaf(value, value, sum);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) { norms[row] = sum; }
  }
}

__global__ void gather_packed_rows(const std::uint8_t* __restrict__ source,
                                   const int* __restrict__ indices,
                                   std::uint8_t* __restrict__ destination,
                                   int active_rows,
                                   int padded_rows,
                                   int row_bytes)
{
  const long long total = static_cast<long long>(padded_rows) * row_bytes;
  for (long long task = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       task < total;
       task += static_cast<long long>(blockDim.x) * gridDim.x) {
    const int row = static_cast<int>(task / row_bytes);
    const int byte = static_cast<int>(task - static_cast<long long>(row) * row_bytes);
    destination[task] = row < active_rows
                          ? source[static_cast<long long>(indices[row]) * row_bytes + byte]
                          : std::uint8_t{0};
  }
}

__global__ void gather_scales(const std::uint32_t* __restrict__ source_row_major,
                              const int* __restrict__ indices,
                              std::uint32_t* __restrict__ destination_tiled,
                              int active_rows,
                              int padded_rows,
                              int features)
{
  const int words_per_row = features / MMA_K;
  const int k_tiles = features / BK;
  const long long total = static_cast<long long>(padded_rows) * words_per_row;
  for (long long task = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       task < total;
       task += static_cast<long long>(blockDim.x) * gridDim.x) {
    const int row = static_cast<int>(task / words_per_row);
    const int word = static_cast<int>(task - static_cast<long long>(row) * words_per_row);
    const int row_tile = row / BN;
    const int row_local = row - row_tile * BN;
    const int k_tile = word / K_FRAGS;
    const int fragment = word - k_tile * K_FRAGS;
    const std::size_t destination =
      (static_cast<std::size_t>(row_tile) * k_tiles + k_tile) * (BN * K_FRAGS) +
      row_local * K_FRAGS + fragment;
    destination_tiled[destination] = row < active_rows
                                       ? source_row_major[
                                           static_cast<std::size_t>(indices[row]) *
                                             words_per_row +
                                           word]
                                       : kUnitScaleWord;
  }
}

__global__ void gather_row_scales(const std::uint32_t* __restrict__ source,
                                  const int* __restrict__ selected,
                                  std::uint32_t* __restrict__ target,
                                  int active,
                                  int padded,
                                  int words)
{
  const std::size_t total = static_cast<std::size_t>(padded) * words;
  for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                           threadIdx.x;
       index < total;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const int row = static_cast<int>(index / words);
    const int word = static_cast<int>(index - static_cast<std::size_t>(row) * words);
    target[index] = row < active
                      ? source[static_cast<std::size_t>(selected[row]) * words + word]
                      : kUnitScaleWord;
  }
}

__global__ void gather_metadata(const float* __restrict__ norms,
                                const int* __restrict__ indices,
                                float* __restrict__ gathered_norms,
                                int* __restrict__ gathered_indices,
                                int active_rows,
                                int padded_rows)
{
  for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < padded_rows;
       row += blockDim.x * gridDim.x) {
    const int source_row = row < active_rows ? indices[row] : -1;
    gathered_indices[row] = source_row;
    gathered_norms[row] = source_row >= 0 ? norms[source_row] : 0.0f;
  }
}

void encode_bf16_panel(CUtensorMap* map,
                       __nv_bfloat16* output,
                       int rows,
                       int columns)
{
  std::uint64_t global_dim[3] = {64ull, std::uint64_t(columns / 64), std::uint64_t(rows)};
  std::uint64_t global_stride[2] = {
    std::uint64_t(64) * sizeof(__nv_bfloat16),
    std::uint64_t(columns) * sizeof(__nv_bfloat16),
  };
  std::uint32_t box_dim[3] = {64u, std::uint32_t(BN / 64), std::uint32_t(BM_WG)};
  std::uint32_t element_stride[3] = {1u, 1u, 1u};
  CHECK_DRV(cuTensorMapEncodeTiled(map,
                                  CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                  3,
                                  output,
                                  global_dim,
                                  global_stride,
                                  box_dim,
                                  element_stride,
                                  CU_TENSOR_MAP_INTERLEAVE_NONE,
                                  CU_TENSOR_MAP_SWIZZLE_128B,
                                  CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

__device__ __forceinline__ __half rbf_to_half(float norm_a,
                                               float norm_b,
                                               float dot,
                                               float gamma,
                                               bool diagonal)
{
  if (diagonal) { return __float2half_rn(1.0f); }
  const float distance = fmaxf(norm_a + norm_b - 2.0f * dot, 0.0f);
  return __float2half_rn(expf(-gamma * distance));
}

__global__ void transform_kww(const __nv_bfloat16* __restrict__ panel,
                              const float* __restrict__ norms,
                              __half* __restrict__ output,
                              int active_cols,
                              int panel_stride,
                              float gamma)
{
  const std::size_t elements = static_cast<std::size_t>(active_cols) * active_cols;
  for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                           threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const int row = static_cast<int>(index / active_cols);
    const int col = static_cast<int>(index - static_cast<std::size_t>(row) * active_cols);
    const float dot = __bfloat162float(
      panel[static_cast<std::size_t>(row) * panel_stride + col]);
    output[index] = rbf_to_half(norms[row], norms[col], dot, gamma, row == col);
  }
}

__global__ void transform_slotted(const __nv_bfloat16* __restrict__ panel,
                                  const float* __restrict__ norm_x,
                                  const float* __restrict__ norm_w,
                                  const int* __restrict__ selected_indices,
                                  const int* __restrict__ slots,
                                  __half* __restrict__ cache,
                                  int rows,
                                  int active_cols,
                                  int panel_stride,
                                  float gamma)
{
  const std::size_t elements = static_cast<std::size_t>(rows) * active_cols;
  for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                           threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const int row = static_cast<int>(index / active_cols);
    const int col = static_cast<int>(index - static_cast<std::size_t>(row) * active_cols);
    const float dot = __bfloat162float(
      panel[static_cast<std::size_t>(row) * panel_stride + col]);
    cache[static_cast<std::size_t>(slots[col]) * rows + row] = rbf_to_half(
      norm_x[row], norm_w[col], dot, gamma, row == selected_indices[col]);
  }
}

// Read the temporary BF16 panel in row-major coalesced order, transpose one
// 64x32 tile through shared memory, then write the slot-major FP16 cache with
// coalesced row stores.  A small persistent grid avoids launching tens of
// thousands of mostly tiny CTAs for every SMO cache-miss batch.
__global__ void transform_slotted_tiled(const __nv_bfloat16* __restrict__ panel,
                                        const float* __restrict__ norm_x,
                                        const float* __restrict__ norm_w,
                                        const int* __restrict__ selected_indices,
                                        const int* __restrict__ slots,
                                        __half* __restrict__ cache,
                                        int rows,
                                        int active_cols,
                                        int panel_stride,
                                        float gamma)
{
  constexpr int tile_rows = 64;
  constexpr int tile_columns = 32;
  // Four-half padding keeps every column 8-byte aligned.  This permits
  // conflict-light transpose stores and aligned half2 drains.
  __shared__ __half transposed[tile_columns][tile_rows + 4];
  __shared__ float row_norm[tile_rows];
  __shared__ float column_norm[tile_columns];
  __shared__ int column_index[tile_columns];
  __shared__ int column_slot[tile_columns];

  const int tiles_m = (rows + tile_rows - 1) / tile_rows;
  const int tiles_n = (active_cols + tile_columns - 1) / tile_columns;
  const int total_tiles = tiles_m * tiles_n;
  for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
    const int tile_m = tile / tiles_n;
    const int tile_n = tile - tile_m * tiles_n;
    const int row_base = tile_m * tile_rows;
    const int column_base = tile_n * tile_columns;
    if (threadIdx.x < tile_rows) {
      const int row = row_base + threadIdx.x;
      row_norm[threadIdx.x] = row < rows ? norm_x[row] : 0.0f;
    }
    if (threadIdx.x < tile_columns) {
      const int column = column_base + threadIdx.x;
      column_norm[threadIdx.x] =
        column < active_cols ? norm_w[column] : 0.0f;
      column_index[threadIdx.x] =
        column < active_cols ? selected_indices[column] : -1;
      column_slot[threadIdx.x] = column < active_cols ? slots[column] : 0;
    }
    __syncthreads();

    constexpr int column_pairs = tile_columns / 2;
    for (int position = threadIdx.x; position < tile_rows * column_pairs;
         position += blockDim.x) {
      const int row_local = position / column_pairs;
      const int column_local =
        2 * (position - row_local * column_pairs);
      const int row = row_base + row_local;
      const int column = column_base + column_local;
      __half value0 = __float2half_rn(0.0f);
      __half value1 = __float2half_rn(0.0f);
      if (row < rows) {
        const __nv_bfloat162 packed_dot =
          *reinterpret_cast<const __nv_bfloat162*>(
            panel + static_cast<std::size_t>(row) * panel_stride + column);
        const float2 dot = __bfloat1622float2(packed_dot);
        if (column < active_cols) {
          value0 = rbf_to_half(row_norm[row_local],
                               column_norm[column_local],
                               dot.x,
                               gamma,
                               row == column_index[column_local]);
        }
        if (column + 1 < active_cols) {
          value1 = rbf_to_half(row_norm[row_local],
                               column_norm[column_local + 1],
                               dot.y,
                               gamma,
                               row == column_index[column_local + 1]);
        }
      }
      transposed[column_local][row_local] = value0;
      transposed[column_local + 1][row_local] = value1;
    }
    __syncthreads();

    constexpr int row_pairs = tile_rows / 2;
    for (int position = threadIdx.x; position < tile_columns * row_pairs;
         position += blockDim.x) {
      const int column_local = position / row_pairs;
      const int row_local = 2 * (position - column_local * row_pairs);
      const int row = row_base + row_local;
      const int column = column_base + column_local;
      if (row + 1 < rows && column < active_cols) {
        const std::uint32_t packed_value =
          *reinterpret_cast<const std::uint32_t*>(
            &transposed[column_local][row_local]);
        *reinterpret_cast<std::uint32_t*>(
          cache + static_cast<std::size_t>(column_slot[column_local]) * rows + row) =
          packed_value;
      }
    }
    __syncthreads();
  }
}

// Frozen scalar reference used only by the A/B gate.
__global__ void transform_slotted_tiled_scalar(
  const __nv_bfloat16* __restrict__ panel,
  const float* __restrict__ norm_x,
  const float* __restrict__ norm_w,
  const int* __restrict__ selected_indices,
  const int* __restrict__ slots,
  __half* __restrict__ cache,
  int rows,
  int active_cols,
  int panel_stride,
  float gamma)
{
  constexpr int tile_rows = 64;
  constexpr int tile_columns = 32;
  __shared__ __half transposed[tile_columns][tile_rows + 1];
  __shared__ float row_norm[tile_rows];
  __shared__ float column_norm[tile_columns];
  __shared__ int column_index[tile_columns];
  __shared__ int column_slot[tile_columns];

  const int tiles_m = (rows + tile_rows - 1) / tile_rows;
  const int tiles_n = (active_cols + tile_columns - 1) / tile_columns;
  const int total_tiles = tiles_m * tiles_n;
  for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
    const int tile_m = tile / tiles_n;
    const int tile_n = tile - tile_m * tiles_n;
    const int row_base = tile_m * tile_rows;
    const int column_base = tile_n * tile_columns;
    if (threadIdx.x < tile_rows) {
      const int row = row_base + threadIdx.x;
      row_norm[threadIdx.x] = row < rows ? norm_x[row] : 0.0f;
    }
    if (threadIdx.x < tile_columns) {
      const int column = column_base + threadIdx.x;
      column_norm[threadIdx.x] = column < active_cols ? norm_w[column] : 0.0f;
      column_index[threadIdx.x] =
        column < active_cols ? selected_indices[column] : -1;
      column_slot[threadIdx.x] = column < active_cols ? slots[column] : 0;
    }
    __syncthreads();

    for (int position = threadIdx.x; position < tile_rows * tile_columns;
         position += blockDim.x) {
      const int row_local = position / tile_columns;
      const int column_local = position - row_local * tile_columns;
      const int row = row_base + row_local;
      const int column = column_base + column_local;
      __half value = __float2half_rn(0.0f);
      if (row < rows && column < active_cols) {
        const float dot = __bfloat162float(
          panel[static_cast<std::size_t>(row) * panel_stride + column]);
        value = rbf_to_half(row_norm[row_local],
                            column_norm[column_local],
                            dot,
                            gamma,
                            row == column_index[column_local]);
      }
      transposed[column_local][row_local] = value;
    }
    __syncthreads();

    for (int position = threadIdx.x; position < tile_rows * tile_columns;
         position += blockDim.x) {
      const int column_local = position / tile_rows;
      const int row_local = position - column_local * tile_rows;
      const int row = row_base + row_local;
      const int column = column_base + column_local;
      if (row < rows && column < active_cols) {
        cache[static_cast<std::size_t>(column_slot[column_local]) * rows + row] =
          transposed[column_local][row_local];
      }
    }
    __syncthreads();
  }
}

__global__ void reference_pairs(const std::uint8_t* __restrict__ packed_x,
                                const std::uint32_t* __restrict__ scale_x_row,
                                const float* __restrict__ norm_x,
                                const std::uint8_t* __restrict__ packed_w,
                                const std::uint32_t* __restrict__ scale_w_row,
                                const float* __restrict__ norm_w,
                                const int* __restrict__ selected_indices,
                                const int* __restrict__ row_indices,
                                __half* __restrict__ output,
                                int sample_rows,
                                int active_cols,
                                int features,
                                float gamma)
{
  const std::size_t elements = static_cast<std::size_t>(sample_rows) * active_cols;
  for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                           threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const int sample = static_cast<int>(index / active_cols);
    const int col = static_cast<int>(index - static_cast<std::size_t>(sample) * active_cols);
    const int row = row_indices[sample];
    float dot = 0.0f;
    for (int feature = 0; feature < features; ++feature) {
      const float a =
        decode_packed_value(packed_x, scale_x_row, row, feature, features);
      const float b =
        decode_packed_value(packed_w, scale_w_row, col, feature, features);
      dot = fmaf(a, b, dot);
    }
    output[index] = rbf_to_half(
      norm_x[row], norm_w[col], dot, gamma, row == selected_indices[col]);
  }
}


__global__
__launch_bounds__(384, 1)
void sm120_nvfp4_rbf_fused_kernel(
    const __grid_constant__ CUtensorMap tmaA,
    const __grid_constant__ CUtensorMap tmaB,
    const uint32_t* __restrict__ scale_A_tiled,
    const uint32_t* __restrict__ scale_B_tiled,
    const int* __restrict__ tile_ids,
    const float* __restrict__ norm_x,
    const float* __restrict__ norm_w,
    const int* __restrict__ selected_indices,
    const int* __restrict__ slots,
    __half* __restrict__ cache,
    int cache_rows,
    int active_cols,
    int K,
    int total_tiles,
    float gamma)
{
    __shared__ __align__(128) uint8_t smA[PIPE_STAGES][BM * ROW_BYTES];
    __shared__ __align__(128) uint8_t smB[PIPE_STAGES][BN * ROW_BYTES];
    __shared__ __align__(16)  uint32_t smSA[PIPE_STAGES][SA_WORDS];
    __shared__ __align__(16)  uint32_t smSB[PIPE_STAGES][SB_WORDS];
    __shared__ __align__(8)   uint64_t mbar_full[PIPE_STAGES];
    __shared__ __align__(8)   uint64_t mbar_empty[PIPE_STAGES];
    __shared__ __align__(16)  __half sK[NUM_CONSUMER_WG][BN][BM_WG + 2];
    __shared__ float smNormA[NUM_CONSUMER_WG][BM_WG];
    __shared__ float smNormB[NUM_CONSUMER_WG][BN];
    __shared__ int smSelB[NUM_CONSUMER_WG][BN];
    __shared__ int smSlots[NUM_CONSUMER_WG][BN];

    const int tid = threadIdx.x;
    const int wg_idx = tid / 128;
    const int warp_id = tid / 32;
    const int lane = tid & 31;
    const int k_tiles = K / BK;

    constexpr uint32_t TMA_A_BYTES = BM * ROW_BYTES;
    constexpr uint32_t TMA_B_BYTES = BN * ROW_BYTES;
    constexpr uint32_t TOTAL_TMA = TMA_A_BYTES + TMA_B_BYTES + SA_BYTES + SB_BYTES;

    if (tid < PIPE_STAGES) {
        init_bar(&mbar_full[tid], 1);
        init_bar(&mbar_empty[tid], CONSUMER_WARPS);
    }
    asm volatile("bar.sync 0;");

    const uint64_t tmapA = reinterpret_cast<uint64_t>(&tmaA);
    const uint64_t tmapB = reinterpret_cast<uint64_t>(&tmaB);

    uint64_t sA[PIPE_STAGES], sB[PIPE_STAGES], mF[PIPE_STAGES];
    uint64_t sSA[PIPE_STAGES], sSB[PIPE_STAGES];
    #pragma unroll
    for (int s = 0; s < PIPE_STAGES; ++s) {
        sA[s]  = static_cast<uint64_t>(__cvta_generic_to_shared(&smA[s][0]));
        sB[s]  = static_cast<uint64_t>(__cvta_generic_to_shared(&smB[s][0]));
        sSA[s] = static_cast<uint64_t>(__cvta_generic_to_shared(&smSA[s][0]));
        sSB[s] = static_cast<uint64_t>(__cvta_generic_to_shared(&smSB[s][0]));
        mF[s]  = static_cast<uint64_t>(__cvta_generic_to_shared(&mbar_full[s]));
    }

    const size_t sa_stride = static_cast<size_t>(k_tiles) * SA_WORDS;
    const size_t sb_stride = static_cast<size_t>(k_tiles) * SB_WORDS;

    if (wg_idx == 0) {
        if (warp_id == 0 && lane == 0) {
            int qidx = 0, p_empty = 0;
            for (int i = blockIdx.x; i < total_tiles; i += gridDim.x) {
                const int packed = __ldg(tile_ids + i);
                const int mt_idx = (packed >> 16) & 0xFFFF;
                const int nt_idx = packed & 0xFFFF;
                const int block_m = mt_idx * BM;
                const int block_n = nt_idx * BN;
                const uint64_t gSA = reinterpret_cast<uint64_t>(scale_A_tiled + size_t(mt_idx) * sa_stride);
                const uint64_t gSB = reinterpret_cast<uint64_t>(scale_B_tiled + size_t(nt_idx) * sb_stride);

                for (int kt = 0; kt < k_tiles; ++kt) {
                    if (qidx == PIPE_STAGES) { qidx = 0; p_empty ^= 1; }
                    wait_bar(&mbar_empty[qidx], p_empty);
                    if (lane == 0) arrive_tx(&mbar_full[qidx], TOTAL_TMA);
                    const int k_off = kt * BK_PACKED;
                    const uint64_t gSA_k = gSA + size_t(kt) * SA_BYTES;
                    const uint64_t gSB_k = gSB + size_t(kt) * SB_BYTES;
                    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];\n"
                        ::"l"(sA[qidx]),"l"(tmapA),"r"(k_off),"r"(block_m),"l"(mF[qidx]):"memory");
                    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];\n"
                        ::"l"(sB[qidx]),"l"(tmapB),"r"(k_off),"r"(block_n),"l"(mF[qidx]):"memory");
                    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];\n"
                        ::"l"(sSA[qidx]),"l"(gSA_k),"r"(SA_BYTES),"l"(mF[qidx]):"memory");
                    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];\n"
                        ::"l"(sSB[qidx]),"l"(gSB_k),"r"(SB_BYTES),"l"(mF[qidx]):"memory");
                    ++qidx;
                }
            }
        }
    } else {
        const int consumer_wg_id = wg_idx - 1;
        const int tid_in_wg = tid - (wg_idx * 128);
        const int warp_id_in_wg = warp_id - PRODUCER_WARPS - consumer_wg_id * CONSUMER_WARPS_PER_WG;
        const int cwn = warp_id_in_wg;

        const int gid = lane >> 2;
        const int tig = lane & 3;
        const int lane_n8 = lane & 7;
        const int sa_rl = (tig == 1) ? (gid + 8) : gid;

        if (lane == 0) {
            #pragma unroll
            for (int q = 0; q < PIPE_STAGES; ++q) arrive(&mbar_empty[q]);
        }

        float c_frag[WM_TILES][WN_TILES][8];
        #pragma unroll 4
        for (int wm=0;wm<WM_TILES;++wm)
            #pragma unroll 2
            for (int wn=0;wn<WN_TILES;++wn)
                #pragma unroll 8
                for (int i=0;i<8;++i) c_frag[wm][wn][i]=0.f;

        int qidx = 0, p_full = 0;
        const int bar_pre  = consumer_wg_id + 3;
        const int bar_post = consumer_wg_id + 5;

        for (int i = blockIdx.x; i < total_tiles; i += gridDim.x) {
            const int packed = __ldg(tile_ids + i);
            const int mt_idx = (packed >> 16) & 0xFFFF;
            const int nt_idx = packed & 0xFFFF;
            const int block_m = mt_idx * BM;
            const int block_n = nt_idx * BN;

            for (int kt = 0; kt < k_tiles; ++kt) {
                if (qidx == PIPE_STAGES) { qidx = 0; p_full ^= 1; }
                wait_bar(&mbar_full[qidx], p_full);

                const uint32_t smA_b = cvta_shared_u32(&smA[qidx][0]);
                const uint32_t smB_b = cvta_shared_u32(&smB[qidx][0]);
                const uint32_t* cSA = smSA[qidx];
                const uint32_t* cSB = smSB[qidx];

                uint32_t sa_c[WM_TILES][K_FRAGS];
                #pragma unroll 4
                for (int wm=0;wm<WM_TILES;++wm) {
                    const int sr = consumer_wg_id*BM_WG + wm*MMA_M + sa_rl;
                    #pragma unroll 2
                    for (int kf=0;kf<K_FRAGS;++kf)
                        sa_c[wm][kf] = cSA[sr*K_FRAGS+kf];
                }
                uint32_t sb0_c[WN_TILES][K_FRAGS], sb1_c[WN_TILES][K_FRAGS];
                #pragma unroll 2
                for (int wn=0;wn<WN_TILES;++wn) {
                    const int sc0 = cwn*WN_TILES*MMA_N + wn*MMA_N + gid;
                    const int sc1 = sc0 + 8;
                    #pragma unroll 2
                    for (int kf=0;kf<K_FRAGS;++kf) {
                        sb0_c[wn][kf] = cSB[sc0*K_FRAGS+kf];
                        sb1_c[wn][kf] = cSB[sc1*K_FRAGS+kf];
                    }
                }

                #pragma unroll 2
                for (int kf = 0; kf < K_FRAGS; ++kf) {
                    uint32_t br[WN_TILES][4];
                    #pragma unroll 2
                    for (int wn=0;wn<WN_TILES;++wn) {
                        const int nb = cwn*WN_TILES*MMA_N + wn*MMA_N;
                        const int brow = nb + lane_n8 + ((lane>>4)*8);
                        const int bkb = kf*FRAG_BYTES + (((lane>>3)&1)*16);
                        const int bl = brow*ROW_BYTES + bkb;
                        const int bmask = ((brow >> 1) & 3) << 4;
                        ldmatrix_x4(br[wn], smB_b + uint32_t(bl ^ bmask));
                    }

                    uint32_t a0[4], a1[4], a2[4], a3[4];
                    {
                        const int akb = (lane>>4)*16 + kf*FRAG_BYTES;
                        const int ar0 = consumer_wg_id*BM_WG + 0*MMA_M + (lane&15);
                        const int ar1 = consumer_wg_id*BM_WG + 1*MMA_M + (lane&15);
                        const int ar2 = consumer_wg_id*BM_WG + 2*MMA_M + (lane&15);
                        const int ar3 = consumer_wg_id*BM_WG + 3*MMA_M + (lane&15);
                        const int am0 = ((ar0 >> 1) & 3) << 4;
                        const int am1 = ((ar1 >> 1) & 3) << 4;
                        const int am2 = ((ar2 >> 1) & 3) << 4;
                        const int am3 = ((ar3 >> 1) & 3) << 4;
                        ldmatrix_x4(a0, smA_b + uint32_t((ar0*ROW_BYTES + akb) ^ am0));
                        ldmatrix_x4(a1, smA_b + uint32_t((ar1*ROW_BYTES + akb) ^ am1));
                        ldmatrix_x4(a2, smA_b + uint32_t((ar2*ROW_BYTES + akb) ^ am2));
                        ldmatrix_x4(a3, smA_b + uint32_t((ar3*ROW_BYTES + akb) ^ am3));
                    }

                    const uint32_t sa0 = sa_c[0][kf];
                    const uint32_t sa1 = sa_c[1][kf];
                    const uint32_t sa2 = sa_c[2][kf];
                    const uint32_t sa3 = sa_c[3][kf];

                    mma_nvfp4(c_frag[0][0][0],c_frag[0][0][1],c_frag[0][0][2],c_frag[0][0][3],
                              a0[0],a0[1],a0[2],a0[3], br[0][0],br[0][1],
                              c_frag[0][0][0],c_frag[0][0][1],c_frag[0][0][2],c_frag[0][0][3],
                              sa0, sb0_c[0][kf]);
                    mma_nvfp4(c_frag[0][0][4],c_frag[0][0][5],c_frag[0][0][6],c_frag[0][0][7],
                              a0[0],a0[1],a0[2],a0[3], br[0][2],br[0][3],
                              c_frag[0][0][4],c_frag[0][0][5],c_frag[0][0][6],c_frag[0][0][7],
                              sa0, sb1_c[0][kf]);
                    mma_nvfp4(c_frag[0][1][0],c_frag[0][1][1],c_frag[0][1][2],c_frag[0][1][3],
                              a0[0],a0[1],a0[2],a0[3], br[1][0],br[1][1],
                              c_frag[0][1][0],c_frag[0][1][1],c_frag[0][1][2],c_frag[0][1][3],
                              sa0, sb0_c[1][kf]);
                    mma_nvfp4(c_frag[0][1][4],c_frag[0][1][5],c_frag[0][1][6],c_frag[0][1][7],
                              a0[0],a0[1],a0[2],a0[3], br[1][2],br[1][3],
                              c_frag[0][1][4],c_frag[0][1][5],c_frag[0][1][6],c_frag[0][1][7],
                              sa0, sb1_c[1][kf]);

                    mma_nvfp4(c_frag[1][0][0],c_frag[1][0][1],c_frag[1][0][2],c_frag[1][0][3],
                              a1[0],a1[1],a1[2],a1[3], br[0][0],br[0][1],
                              c_frag[1][0][0],c_frag[1][0][1],c_frag[1][0][2],c_frag[1][0][3],
                              sa1, sb0_c[0][kf]);
                    mma_nvfp4(c_frag[1][0][4],c_frag[1][0][5],c_frag[1][0][6],c_frag[1][0][7],
                              a1[0],a1[1],a1[2],a1[3], br[0][2],br[0][3],
                              c_frag[1][0][4],c_frag[1][0][5],c_frag[1][0][6],c_frag[1][0][7],
                              sa1, sb1_c[0][kf]);
                    mma_nvfp4(c_frag[1][1][0],c_frag[1][1][1],c_frag[1][1][2],c_frag[1][1][3],
                              a1[0],a1[1],a1[2],a1[3], br[1][0],br[1][1],
                              c_frag[1][1][0],c_frag[1][1][1],c_frag[1][1][2],c_frag[1][1][3],
                              sa1, sb0_c[1][kf]);
                    mma_nvfp4(c_frag[1][1][4],c_frag[1][1][5],c_frag[1][1][6],c_frag[1][1][7],
                              a1[0],a1[1],a1[2],a1[3], br[1][2],br[1][3],
                              c_frag[1][1][4],c_frag[1][1][5],c_frag[1][1][6],c_frag[1][1][7],
                              sa1, sb1_c[1][kf]);

                    mma_nvfp4(c_frag[2][0][0],c_frag[2][0][1],c_frag[2][0][2],c_frag[2][0][3],
                              a2[0],a2[1],a2[2],a2[3], br[0][0],br[0][1],
                              c_frag[2][0][0],c_frag[2][0][1],c_frag[2][0][2],c_frag[2][0][3],
                              sa2, sb0_c[0][kf]);
                    mma_nvfp4(c_frag[2][0][4],c_frag[2][0][5],c_frag[2][0][6],c_frag[2][0][7],
                              a2[0],a2[1],a2[2],a2[3], br[0][2],br[0][3],
                              c_frag[2][0][4],c_frag[2][0][5],c_frag[2][0][6],c_frag[2][0][7],
                              sa2, sb1_c[0][kf]);
                    mma_nvfp4(c_frag[2][1][0],c_frag[2][1][1],c_frag[2][1][2],c_frag[2][1][3],
                              a2[0],a2[1],a2[2],a2[3], br[1][0],br[1][1],
                              c_frag[2][1][0],c_frag[2][1][1],c_frag[2][1][2],c_frag[2][1][3],
                              sa2, sb0_c[1][kf]);
                    mma_nvfp4(c_frag[2][1][4],c_frag[2][1][5],c_frag[2][1][6],c_frag[2][1][7],
                              a2[0],a2[1],a2[2],a2[3], br[1][2],br[1][3],
                              c_frag[2][1][4],c_frag[2][1][5],c_frag[2][1][6],c_frag[2][1][7],
                              sa2, sb1_c[1][kf]);

                    mma_nvfp4(c_frag[3][0][0],c_frag[3][0][1],c_frag[3][0][2],c_frag[3][0][3],
                              a3[0],a3[1],a3[2],a3[3], br[0][0],br[0][1],
                              c_frag[3][0][0],c_frag[3][0][1],c_frag[3][0][2],c_frag[3][0][3],
                              sa3, sb0_c[0][kf]);
                    mma_nvfp4(c_frag[3][0][4],c_frag[3][0][5],c_frag[3][0][6],c_frag[3][0][7],
                              a3[0],a3[1],a3[2],a3[3], br[0][2],br[0][3],
                              c_frag[3][0][4],c_frag[3][0][5],c_frag[3][0][6],c_frag[3][0][7],
                              sa3, sb1_c[0][kf]);
                    mma_nvfp4(c_frag[3][1][0],c_frag[3][1][1],c_frag[3][1][2],c_frag[3][1][3],
                              a3[0],a3[1],a3[2],a3[3], br[1][0],br[1][1],
                              c_frag[3][1][0],c_frag[3][1][1],c_frag[3][1][2],c_frag[3][1][3],
                              sa3, sb0_c[1][kf]);
                    mma_nvfp4(c_frag[3][1][4],c_frag[3][1][5],c_frag[3][1][6],c_frag[3][1][7],
                              a3[0],a3[1],a3[2],a3[3], br[1][2],br[1][3],
                              c_frag[3][1][4],c_frag[3][1][5],c_frag[3][1][6],c_frag[3][1][7],
                              sa3, sb1_c[1][kf]);
                }
                if (lane == 0) arrive(&mbar_empty[qidx]);
                ++qidx;
            }

            // Ensure prior tile sK drain + metadata reads finished before overwriting
            asm volatile("bar.sync %0, 128;\n" :: "r"(bar_pre) : "memory");

            const int wg_row_base = block_m + consumer_wg_id * BM_WG;
            if (tid_in_wg < BM_WG) {
                const int gr = wg_row_base + tid_in_wg;
                smNormA[consumer_wg_id][tid_in_wg] = (gr < cache_rows) ? norm_x[gr] : 0.0f;
            }
            {
                const int col = block_n + tid_in_wg;
                if (col < active_cols) {
                    smNormB[consumer_wg_id][tid_in_wg] = norm_w[col];
                    smSelB[consumer_wg_id][tid_in_wg]  = selected_indices[col];
                    smSlots[consumer_wg_id][tid_in_wg] = slots[col];
                } else {
                    smNormB[consumer_wg_id][tid_in_wg] = 0.0f;
                    smSelB[consumer_wg_id][tid_in_wg]  = -1;
                    smSlots[consumer_wg_id][tid_in_wg] = 0;
                }
            }
            asm volatile("bar.sync %0, 128;\n" :: "r"(bar_post) : "memory");

            #pragma unroll
            for (int wm = 0; wm < WM_TILES; ++wm) {
                const int r0 = wm * MMA_M + gid;
                const int r1 = r0 + 8;
                const int gr0 = wg_row_base + r0;
                const int gr1 = wg_row_base + r1;
                const float na0 = smNormA[consumer_wg_id][r0];
                const float na1 = smNormA[consumer_wg_id][r1];
                #pragma unroll
                for (int wn = 0; wn < WN_TILES; ++wn) {
                    const int c0 = cwn * WN_TILES * MMA_N + wn * MMA_N + tig * 2;
                    const int c1 = c0 + 1;
                    const int c2 = c0 + 8;
                    const int c3 = c2 + 1;
                    const float nb0 = smNormB[consumer_wg_id][c0];
                    const float nb1 = smNormB[consumer_wg_id][c1];
                    const float nb2 = smNormB[consumer_wg_id][c2];
                    const float nb3 = smNormB[consumer_wg_id][c3];
                    const int sb0 = smSelB[consumer_wg_id][c0];
                    const int sb1 = smSelB[consumer_wg_id][c1];
                    const int sb2 = smSelB[consumer_wg_id][c2];
                    const int sb3 = smSelB[consumer_wg_id][c3];
                    float* cr = c_frag[wm][wn];
                    sK[consumer_wg_id][c0][r0] = rbf_to_half(na0, nb0, cr[0], gamma, gr0 == sb0);
                    sK[consumer_wg_id][c1][r0] = rbf_to_half(na0, nb1, cr[1], gamma, gr0 == sb1);
                    sK[consumer_wg_id][c0][r1] = rbf_to_half(na1, nb0, cr[2], gamma, gr1 == sb0);
                    sK[consumer_wg_id][c1][r1] = rbf_to_half(na1, nb1, cr[3], gamma, gr1 == sb1);
                    sK[consumer_wg_id][c2][r0] = rbf_to_half(na0, nb2, cr[4], gamma, gr0 == sb2);
                    sK[consumer_wg_id][c3][r0] = rbf_to_half(na0, nb3, cr[5], gamma, gr0 == sb3);
                    sK[consumer_wg_id][c2][r1] = rbf_to_half(na1, nb2, cr[6], gamma, gr1 == sb2);
                    sK[consumer_wg_id][c3][r1] = rbf_to_half(na1, nb3, cr[7], gamma, gr1 == sb3);
                }
            }

            asm volatile("bar.sync %0, 128;\n" :: "r"(bar_post) : "memory");

            constexpr int row_pairs = BM_WG / 2;  // 32
            for (int p = tid_in_wg; p < BN * row_pairs; p += 128) {
                const int col_local = p >> 5;
                const int row_local = (p & 31) * 2;
                const int gr = wg_row_base + row_local;
                if (block_n + col_local < active_cols && gr < cache_rows) {
                    __half* dst = cache + static_cast<size_t>(smSlots[consumer_wg_id][col_local]) * cache_rows + gr;
                    if (gr + 1 < cache_rows && ((cache_rows & 1) == 0)) {
                        const uint32_t packed_val = *reinterpret_cast<const uint32_t*>(
                            &sK[consumer_wg_id][col_local][row_local]);
                        *reinterpret_cast<uint32_t*>(dst) = packed_val;
                    } else {
                        dst[0] = sK[consumer_wg_id][col_local][row_local];
                        if (gr + 1 < cache_rows) {
                            dst[1] = sK[consumer_wg_id][col_local][row_local + 1];
                        }
                    }
                }
            }

            #pragma unroll 4
            for (int wm=0;wm<WM_TILES;++wm)
                #pragma unroll 2
                for (int wn=0;wn<WN_TILES;++wn)
                    #pragma unroll 8
                    for (int i=0;i<8;++i) c_frag[wm][wn][i]=0.f;
        }
    }
}

struct Context {
  int rows = 0;
  int max_cols = 0;
  int features = 0;
  int sms = 0;
  std::uint8_t* packed_x = nullptr;
  std::uint8_t* packed_w = nullptr;
  std::uint32_t* scale_x_row = nullptr;
  std::uint32_t* scale_x_tiled = nullptr;
  std::uint32_t* scale_w_row = nullptr;
  std::uint32_t* scale_w_tiled = nullptr;
  float* norm_x = nullptr;
  float* norm_w = nullptr;
  int* selected_indices = nullptr;
  int* identity_indices = nullptr;
  __nv_bfloat16* panel_full = nullptr;
  __nv_bfloat16* panel_square = nullptr;
  CUtensorMap tma_x{};
  CUtensorMap tma_w{};
  CUtensorMap tma_full{};
  CUtensorMap tma_square{};
  std::array<int*, 8> full_tile_ids{};
  std::array<int, 8> full_tile_counts{};
  std::array<int*, 8> square_tile_ids{};
  std::array<int, 8> square_tile_counts{};
};

void release_context(Context* context)
{
  if (!context) { return; }
  for (int* pointer : context->square_tile_ids) { cudaFree(pointer); }
  for (int* pointer : context->full_tile_ids) { cudaFree(pointer); }
  cudaFree(context->panel_square);
  cudaFree(context->panel_full);
  cudaFree(context->identity_indices);
  cudaFree(context->selected_indices);
  cudaFree(context->norm_w);
  cudaFree(context->norm_x);
  cudaFree(context->scale_w_tiled);
  cudaFree(context->scale_w_row);
  cudaFree(context->scale_x_tiled);
  cudaFree(context->scale_x_row);
  cudaFree(context->packed_w);
  cudaFree(context->packed_x);
  delete context;
}

int launch_blocks(std::size_t work, int threads)
{
  return static_cast<int>(std::min<std::size_t>((work + threads - 1) / threads, 65535));
}

bool allocate_context(Context* context)
{
  const std::size_t packed_x_bytes =
    static_cast<std::size_t>(context->rows) * (context->features / 2);
  const std::size_t packed_w_bytes =
    static_cast<std::size_t>(context->max_cols) * (context->features / 2);
  const std::size_t scale_x_words =
    static_cast<std::size_t>(context->rows) * (context->features / MMA_K);
  const std::size_t scale_w_words =
    static_cast<std::size_t>(context->max_cols) * (context->features / MMA_K);
  const std::size_t full_elements =
    static_cast<std::size_t>(context->rows) * context->max_cols;
  const std::size_t square_elements =
    static_cast<std::size_t>(context->max_cols) * context->max_cols;
  return cudaMalloc(&context->packed_x, packed_x_bytes) == cudaSuccess &&
         cudaMalloc(&context->packed_w, packed_w_bytes) == cudaSuccess &&
         cudaMalloc(&context->scale_x_row, scale_x_words * sizeof(std::uint32_t)) ==
           cudaSuccess &&
         cudaMalloc(&context->scale_x_tiled, scale_x_words * sizeof(std::uint32_t)) ==
           cudaSuccess &&
         cudaMalloc(&context->scale_w_row, scale_w_words * sizeof(std::uint32_t)) ==
           cudaSuccess &&
         cudaMalloc(&context->scale_w_tiled, scale_w_words * sizeof(std::uint32_t)) ==
           cudaSuccess &&
         cudaMalloc(&context->norm_x,
                    static_cast<std::size_t>(context->rows) * sizeof(float)) == cudaSuccess &&
         cudaMalloc(&context->norm_w,
                    static_cast<std::size_t>(context->max_cols) * sizeof(float)) ==
           cudaSuccess &&
         cudaMalloc(&context->selected_indices,
                    static_cast<std::size_t>(context->max_cols) * sizeof(int)) ==
           cudaSuccess &&
         cudaMalloc(&context->identity_indices,
                    static_cast<std::size_t>(context->max_cols) * sizeof(int)) ==
           cudaSuccess &&
         cudaMalloc(&context->panel_full,
                    full_elements * sizeof(__nv_bfloat16)) == cudaSuccess &&
         cudaMalloc(&context->panel_square,
                    square_elements * sizeof(__nv_bfloat16)) == cudaSuccess;
}

bool build_schedules(Context* context)
{
  const int max_tiles = context->max_cols / BN;
  const int full_tiles_m = context->rows / BM;
  for (int tile_count = 1; tile_count <= max_tiles; ++tile_count) {
    std::vector<int> full_ids;
    std::vector<int> square_ids;
    // The keeper's Hilbert constructor walks next_pow2(max(M,N))^2 cells.
    // For SVM's very tall and skinny X*W panels (e.g. 4688x1..8 tiles), that
    // turns a tiny schedule into hundreds of millions of host iterations.
    // An 8-row tiled swizzle is O(actual tiles), preserves adjacent-row
    // locality and gives the persistent mainloop the same complete tile set.
    build_sched_tiled_swizzle(full_tiles_m, tile_count, 8, full_ids);
    build_sched_tiled_swizzle(tile_count, tile_count, 8, square_ids);
    context->full_tile_counts[tile_count - 1] = static_cast<int>(full_ids.size());
    context->square_tile_counts[tile_count - 1] = static_cast<int>(square_ids.size());
    if (cudaMalloc(&context->full_tile_ids[tile_count - 1],
                   full_ids.size() * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&context->square_tile_ids[tile_count - 1],
                   square_ids.size() * sizeof(int)) != cudaSuccess ||
        cudaMemcpy(context->full_tile_ids[tile_count - 1],
                   full_ids.data(),
                   full_ids.size() * sizeof(int),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(context->square_tile_ids[tile_count - 1],
                   square_ids.data(),
                   square_ids.size() * sizeof(int),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      return false;
    }
  }
  return true;
}

Context* create_empty_context(int rows, int max_cols, int features)
{
  if (rows <= 0 || max_cols <= 0 || features <= 0 || rows % BM || max_cols % BN ||
      max_cols / BN > 8 || features % BK) {
    return nullptr;
  }
  if (cuInit(0) != CUDA_SUCCESS) { return nullptr; }
  auto* context = new (std::nothrow) Context;
  if (!context) { return nullptr; }
  context->rows = rows;
  context->max_cols = max_cols;
  context->features = features;
  int device = 0;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, device) != cudaSuccess ||
      properties.major != 12 || properties.minor != 0) {
    release_context(context);
    return nullptr;
  }
  context->sms = properties.multiProcessorCount;
  if (!allocate_context(context)) {
    release_context(context);
    return nullptr;
  }
  return context;
}

__global__ void fill_identity_indices(int* indices, int count)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) { indices[i] = i; }
}

bool finish_context(Context* context, cudaStream_t stream)
{
  constexpr int threads = 256;
  const std::size_t quant_tasks =
    static_cast<std::size_t>(context->rows) * (context->features / MMA_K);
  fill_identity_indices<<<launch_blocks(context->max_cols, threads), threads, 0, stream>>>(context->identity_indices, context->max_cols);
  repack_scales<<<launch_blocks(quant_tasks, threads), threads, 0, stream>>>(
    context->scale_x_row, context->scale_x_tiled, context->rows, context->features);
  coherent_row_norms<<<launch_blocks(context->rows, 8), threads, 0, stream>>>(
    context->packed_x,
    context->scale_x_row,
    context->norm_x,
    context->rows,
    context->features);
  if (cudaPeekAtLastError() != cudaSuccess) { return false; }

  create_tma(&context->tma_x,
             context->packed_x,
             context->features / PACK_RATIO,
             context->rows,
             BK_PACKED,
             BM,
             CU_TENSOR_MAP_SWIZZLE_64B);
  create_tma(&context->tma_w,
             context->packed_w,
             context->features / PACK_RATIO,
             context->max_cols,
             BK_PACKED,
             BN,
             CU_TENSOR_MAP_SWIZZLE_64B);
  encode_bf16_panel(
    &context->tma_full, context->panel_full, context->rows, context->max_cols);
  encode_bf16_panel(&context->tma_square,
                    context->panel_square,
                    context->max_cols,
                    context->max_cols);
  return build_schedules(context);
}

}  // namespace

extern "C" void* coherent_nvfp4_create_context(const float* x_row_major,
                                                int rows,
                                                int max_cols,
                                                int features,
                                                void* stream_pointer)
{
  if (!x_row_major) { return nullptr; }
  auto* context = create_empty_context(rows, max_cols, features);
  if (!context) { return nullptr; }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  constexpr int threads = 256;
  const std::size_t quant_tasks =
    static_cast<std::size_t>(rows) * (features / MMA_K);
  quantize_rows<<<launch_blocks(quant_tasks, threads), threads, 0, stream>>>(
    x_row_major, context->packed_x, context->scale_x_row, rows, features);
  if (!finish_context(context, stream)) {
    release_context(context);
    return nullptr;
  }
  return context;
}

extern "C" void* coherent_nvfp4_create_context_col_major(
  const float* x_col_major,
  int rows,
  int max_cols,
  int source_features,
  void* stream_pointer)
{
  if (!x_col_major || source_features <= 0) { return nullptr; }
  const int padded_features =
    ((source_features + BK - 1) / BK) * BK;
  auto* context = create_empty_context(rows, max_cols, padded_features);
  if (!context) { return nullptr; }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  constexpr int threads = 256;
  const std::size_t quant_tasks =
    static_cast<std::size_t>(rows) * (padded_features / MMA_K);
  quantize_columns<<<launch_blocks(quant_tasks, threads), threads, 0, stream>>>(
    x_col_major,
    context->packed_x,
    context->scale_x_row,
    rows,
    source_features,
    padded_features);
  if (!finish_context(context, stream)) {
    release_context(context);
    return nullptr;
  }
  return context;
}

extern "C" void* coherent_nvfp4_create_subcontext(void* opaque_parent,
                                                  const int* row_indices,
                                                  int active_rows,
                                                  int padded_rows,
                                                  void* stream_pointer)
{
  auto* parent = static_cast<Context*>(opaque_parent);
  if (!parent || !row_indices || active_rows <= 0 || active_rows > parent->rows ||
      padded_rows < active_rows || padded_rows % BM) {
    return nullptr;
  }
  auto* context = create_empty_context(padded_rows, parent->max_cols, parent->features);
  if (!context) { return nullptr; }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  constexpr int threads = 256;
  const int row_bytes = parent->features / PACK_RATIO;
  const std::size_t packed_tasks =
    static_cast<std::size_t>(padded_rows) * row_bytes;
  gather_packed_rows<<<launch_blocks(packed_tasks, threads), threads, 0, stream>>>(
    parent->packed_x,
    row_indices,
    context->packed_x,
    active_rows,
    padded_rows,
    row_bytes);
  const int words_per_row = parent->features / MMA_K;
  const std::size_t scale_tasks =
    static_cast<std::size_t>(padded_rows) * words_per_row;
  gather_row_scales<<<launch_blocks(scale_tasks, threads), threads, 0, stream>>>(
    parent->scale_x_row,
    row_indices,
    context->scale_x_row,
    active_rows,
    padded_rows,
    words_per_row);
  if (!finish_context(context, stream)) {
    release_context(context);
    return nullptr;
  }
  return context;
}

extern "C" int coherent_nvfp4_gather(void* opaque_context,
                                      const int* indices,
                                      int active_cols,
                                      int padded_cols,
                                      void* stream_pointer)
{
  auto* context = static_cast<Context*>(opaque_context);
  if (!context || !indices || active_cols <= 0 || padded_cols < active_cols ||
      padded_cols > context->max_cols || padded_cols % BN) {
    return int(cudaErrorInvalidValue);
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  constexpr int threads = 256;
  const int row_bytes = context->features / PACK_RATIO;
  const std::size_t packed_tasks = static_cast<std::size_t>(padded_cols) * row_bytes;
  gather_packed_rows<<<launch_blocks(packed_tasks, threads), threads, 0, stream>>>(
    context->packed_x,
    indices,
    context->packed_w,
    active_cols,
    padded_cols,
    row_bytes);
  const std::size_t scale_tasks =
    static_cast<std::size_t>(padded_cols) * (context->features / MMA_K);
  gather_scales<<<launch_blocks(scale_tasks, threads), threads, 0, stream>>>(
    context->scale_x_row,
    indices,
    context->scale_w_tiled,
    active_cols,
    padded_cols,
    context->features);
  // Keep a row-major scale copy for the independent packed scalar oracle.
  const int words_per_row = context->features / MMA_K;
  const std::size_t row_scale_tasks =
    static_cast<std::size_t>(padded_cols) * words_per_row;
  gather_row_scales<<<launch_blocks(row_scale_tasks, threads), threads, 0, stream>>>(
    context->scale_x_row,
    indices,
    context->scale_w_row,
    active_cols,
    padded_cols,
    words_per_row);
  gather_metadata<<<launch_blocks(padded_cols, threads), threads, 0, stream>>>(
    context->norm_x,
    indices,
    context->norm_w,
    context->selected_indices,
    active_cols,
    padded_cols);
  return int(cudaPeekAtLastError());
}

extern "C" int coherent_nvfp4_materialize_kww_fp16(void* opaque_context,
                                                    __half* output,
                                                    int active_cols,
                                                    int padded_cols,
                                                    float gamma,
                                                    void* stream_pointer)
{
  auto* context = static_cast<Context*>(opaque_context);
  if (!context || !output || active_cols <= 0 || padded_cols < active_cols ||
      padded_cols > context->max_cols || padded_cols % BN) {
    return int(cudaErrorInvalidValue);
  }
  const int schedule = padded_cols / BN - 1;
  const int total_tiles = context->square_tile_counts[schedule];
  const int grid = std::min(total_tiles, context->sms);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  sm120_nvfp4_rbf_fused_kernel<<<grid, THREADS, 0, stream>>>(
    context->tma_w,
    context->tma_w,
    context->scale_w_tiled,
    context->scale_w_tiled,
    context->square_tile_ids[schedule],
    context->norm_w,
    context->norm_w,
    context->identity_indices,
    context->identity_indices,
    output,
    active_cols,
    active_cols,
    context->features,
    total_tiles,
    gamma);
  return int(cudaPeekAtLastError());
}

extern "C" int coherent_nvfp4_materialize_slotted_fp16(void* opaque_context,
                                                        const int* slots,
                                                        __half* cache,
                                                        int active_cols,
                                                        int padded_cols,
                                                        float gamma,
                                                        void* stream_pointer)
{
  auto* context = static_cast<Context*>(opaque_context);
  if (!context || !slots || !cache || active_cols <= 0 || padded_cols < active_cols ||
      padded_cols > context->max_cols || padded_cols % BN) {
    return int(cudaErrorInvalidValue);
  }
  const int schedule = padded_cols / BN - 1;
  const int total_tiles = context->full_tile_counts[schedule];
  const int grid = std::min(total_tiles, context->sms);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  sm120_nvfp4_rbf_fused_kernel<<<grid, THREADS, 0, stream>>>(
    context->tma_x,
    context->tma_w,
    context->scale_x_tiled,
    context->scale_w_tiled,
    context->full_tile_ids[schedule],
    context->norm_x,
    context->norm_w,
    context->selected_indices,
    slots,
    cache,
    context->rows,
    active_cols,
    context->features,
    total_tiles,
    gamma);
  return int(cudaPeekAtLastError());
}

extern "C" int coherent_nvfp4_transform_slotted_fp16_vec2(
  void* opaque_context,
  const int* slots,
  __half* cache,
  int active_cols,
  float gamma,
  void* stream_pointer)
{
  auto* context = static_cast<Context*>(opaque_context);
  if (!context || !slots || !cache || active_cols <= 0 ||
      active_cols > context->max_cols) {
    return int(cudaErrorInvalidValue);
  }
  constexpr int threads = 256;
  constexpr int tile_rows = 64;
  constexpr int tile_columns = 32;
  const int total_tiles = ((context->rows + tile_rows - 1) / tile_rows) *
                          ((active_cols + tile_columns - 1) / tile_columns);
  const int grid = std::min(total_tiles, context->sms * 4);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  transform_slotted_tiled<<<grid, threads, 0, stream>>>(context->panel_full,
                                                        context->norm_x,
                                                        context->norm_w,
                                                        context->selected_indices,
                                                        slots,
                                                        cache,
                                                        context->rows,
                                                        active_cols,
                                                        context->max_cols,
                                                        gamma);
  return int(cudaPeekAtLastError());
}

extern "C" int coherent_nvfp4_transform_slotted_fp16_legacy(
  void* opaque_context,
  const int* slots,
  __half* cache,
  int active_cols,
  float gamma,
  void* stream_pointer)
{
  auto* context = static_cast<Context*>(opaque_context);
  if (!context || !slots || !cache || active_cols <= 0 ||
      active_cols > context->max_cols) {
    return int(cudaErrorInvalidValue);
  }
  constexpr int threads = 256;
  constexpr int tile_rows = 64;
  constexpr int tile_columns = 32;
  const int total_tiles = ((context->rows + tile_rows - 1) / tile_rows) *
                          ((active_cols + tile_columns - 1) / tile_columns);
  const int grid = std::min(total_tiles, context->sms * 4);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  transform_slotted_tiled_scalar<<<grid, threads, 0, stream>>>(
    context->panel_full,
    context->norm_x,
    context->norm_w,
    context->selected_indices,
    slots,
    cache,
    context->rows,
    active_cols,
    context->max_cols,
    gamma);
  return int(cudaPeekAtLastError());
}

extern "C" int coherent_nvfp4_materialize_slotted_fp16_legacy(
  void* opaque_context,
  const int* slots,
  __half* cache,
  int active_cols,
  int padded_cols,
  float gamma,
  void* stream_pointer)
{
  auto* context = static_cast<Context*>(opaque_context);
  if (!context || !slots || !cache || active_cols <= 0 ||
      padded_cols < active_cols || padded_cols > context->max_cols ||
      padded_cols % BN) {
    return int(cudaErrorInvalidValue);
  }
  const int schedule = padded_cols / BN - 1;
  const int total_tiles = context->full_tile_counts[schedule];
  const int grid = std::min(total_tiles, context->sms);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  sm120_nvfp4_gemm_v10h_kernel<<<grid, THREADS, 0, stream>>>(
    context->tma_x,
    context->tma_w,
    context->tma_full,
    context->panel_full,
    context->scale_x_tiled,
    context->scale_w_tiled,
    context->full_tile_ids[schedule],
    context->rows,
    padded_cols,
    context->features,
    total_tiles,
    0,
    0,
    nullptr);
  constexpr int threads = 256;
  constexpr int tile_rows = 64;
  constexpr int tile_columns = 32;
  const int total_transform_tiles =
    ((context->rows + tile_rows - 1) / tile_rows) *
    ((active_cols + tile_columns - 1) / tile_columns);
  const int transform_grid =
    std::min(total_transform_tiles, context->sms * 4);
  transform_slotted_tiled_scalar<<<transform_grid, threads, 0, stream>>>(
    context->panel_full,
    context->norm_x,
    context->norm_w,
    context->selected_indices,
    slots,
    cache,
    context->rows,
    active_cols,
    context->max_cols,
    gamma);
  return int(cudaPeekAtLastError());
}

extern "C" int coherent_nvfp4_materialize_slotted_prefix_fp16(
  void* opaque_context,
  const int* slots,
  __half* cache,
  int prefix_rows,
  int active_cols,
  int padded_cols,
  float gamma,
  void* stream_pointer)
{
  auto* context = static_cast<Context*>(opaque_context);
  constexpr int schedule_group_tiles = 8;
  if (!context || !slots || !cache || prefix_rows <= 0 ||
      prefix_rows > context->rows || prefix_rows % BM ||
      (prefix_rows / BM) % schedule_group_tiles || active_cols <= 0 ||
      padded_cols < active_cols || padded_cols > context->max_cols ||
      padded_cols % BN) {
    return int(cudaErrorInvalidValue);
  }
  const int schedule = padded_cols / BN - 1;
  const int total_tiles = (prefix_rows / BM) * (padded_cols / BN);
  const int grid = std::min(total_tiles, context->sms);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_pointer);

  // build_sched_tiled_swizzle emits complete groups of eight M tiles in
  // increasing order.  A prefix containing an integral number of groups is
  // therefore exactly the first total_tiles entries of the full schedule.
  sm120_nvfp4_rbf_fused_kernel<<<grid, THREADS, 0, stream>>>(
    context->tma_x,
    context->tma_w,
    context->scale_x_tiled,
    context->scale_w_tiled,
    context->full_tile_ids[schedule],
    context->norm_x,
    context->norm_w,
    context->selected_indices,
    slots,
    cache,
    prefix_rows,
    active_cols,
    context->features,
    total_tiles,
    gamma);
  return int(cudaPeekAtLastError());
}

extern "C" int coherent_nvfp4_reference_pairs_fp16(void* opaque_context,
                                                    const int* row_indices,
                                                    int sample_rows,
                                                    __half* output,
                                                    int active_cols,
                                                    float gamma,
                                                    void* stream_pointer)
{
  auto* context = static_cast<Context*>(opaque_context);
  if (!context || !row_indices || !output || sample_rows <= 0 || active_cols <= 0 ||
      active_cols > context->max_cols) {
    return int(cudaErrorInvalidValue);
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  constexpr int threads = 128;
  const std::size_t elements = static_cast<std::size_t>(sample_rows) * active_cols;
  reference_pairs<<<launch_blocks(elements, threads), threads, 0, stream>>>(
    context->packed_x,
    context->scale_x_row,
    context->norm_x,
    context->packed_w,
    context->scale_w_row,
    context->norm_w,
    context->selected_indices,
    row_indices,
    output,
    sample_rows,
    active_cols,
    context->features,
    gamma);
  return int(cudaPeekAtLastError());
}

extern "C" void coherent_nvfp4_destroy_context(void* opaque_context)
{
  release_context(static_cast<Context*>(opaque_context));
}
