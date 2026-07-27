cmake_minimum_required(VERSION 3.26)

project(coherent_svm_overlay LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)
find_package(raft CONFIG REQUIRED)
find_package(rmm CONFIG REQUIRED)
find_package(rapids_logger CONFIG REQUIRED)

foreach(variable CUML_SOURCE CUML_WHEEL_INCLUDE CUVS_INCLUDE LIBCUML_SO)
  if(NOT DEFINED ${variable})
    message(FATAL_ERROR "${variable} must be provided; see scripts/build.sh")
  endif()
endforeach()

add_library(coherent_rbf_backend OBJECT src/portable_fp16_rbf_backend.cu)
set_target_properties(coherent_rbf_backend PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_ARCHITECTURES native)
target_compile_options(coherent_rbf_backend PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_library(coherent_svm SHARED
  "${CUML_SOURCE}/cpp/src/svm/svr.cu"
  $<TARGET_OBJECTS:coherent_rbf_backend>)

target_include_directories(coherent_svm PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CUML_SOURCE}/cpp/src"
  "${CUML_SOURCE}/cpp/src_prims"
  "${CUML_SOURCE}/cpp/src/svm"
  "${CUML_SOURCE}/cpp/include"
  "${CUML_WHEEL_INCLUDE}"
  "${CUVS_INCLUDE}")

target_compile_definitions(coherent_svm PRIVATE
  CUML_ENABLE_GPU
  DISABLE_CUSPARSE_DEPRECATED
  NVTX_ENABLED)
target_compile_options(coherent_svm PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

target_link_libraries(coherent_svm PRIVATE
  "${LIBCUML_SO}"
  raft::raft
  rmm::rmm
  rapids_logger::rapids_logger
  CUDA::cublas
  CUDA::cuda_driver
  CUDA::cusolver
  CUDA::cusparse
  CUDA::curand)

set_target_properties(coherent_svm PROPERTIES OUTPUT_NAME coherent_svm)

