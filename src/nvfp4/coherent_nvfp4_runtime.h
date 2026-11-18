#pragma once

#include <cuda_fp16.h>

extern "C" void* coherent_nvfp4_create_context(const float* x_row_major,
                                                int rows,
                                                int max_cols,
                                                int features,
                                                void* stream);

// cuML dense SVM inputs are column-major: source[row + feature * rows].  This
// factory quantizes that layout directly and zero-pads the feature dimension to
// the 128-wide native NVFP4 mainloop contract without materializing a second
// FP32 copy of X.
extern "C" void* coherent_nvfp4_create_context_col_major(
  const float* x_col_major,
  int rows,
  int max_cols,
  int source_features,
  void* stream);

extern "C" int coherent_nvfp4_gather(void* context,
                                      const int* indices,
                                      int active_cols,
                                      int padded_cols,
                                      void* stream);

extern "C" int coherent_nvfp4_materialize_kww_fp16(void* context,
                                                    __half* output,
                                                    int active_cols,
                                                    int padded_cols,
                                                    float gamma,
                                                    void* stream);

extern "C" int coherent_nvfp4_materialize_slotted_fp16(void* context,
                                                        const int* slots,
                                                        __half* cache,
                                                        int active_cols,
                                                        int padded_cols,
                                                        float gamma,
                                                        void* stream);

// Materialize only a row prefix of a larger immutable context.  This is used
// by coherent inference: the prefix contains query rows while the remaining
// rows hold gathered support vectors.  prefix_rows must contain a whole number
// of eight-row tile groups (8 * 128 rows), matching the frozen tiled schedule.
extern "C" int coherent_nvfp4_materialize_slotted_prefix_fp16(
  void* context,
  const int* slots,
  __half* cache,
  int prefix_rows,
  int active_cols,
  int padded_cols,
  float gamma,
  void* stream);

extern "C" int coherent_nvfp4_reference_pairs_fp16(void* context,
                                                    const int* row_indices,
                                                    int sample_rows,
                                                    __half* output,
                                                    int active_cols,
                                                    float gamma,
                                                    void* stream);

extern "C" void* coherent_nvfp4_create_subcontext(void* parent_context,
                                                  const int* row_indices,
                                                  int active_rows,
                                                  int padded_rows,
                                                  void* stream);

extern "C" void coherent_nvfp4_destroy_context(void* context);
