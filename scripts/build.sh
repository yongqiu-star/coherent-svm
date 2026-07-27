#!/usr/bin/env bash
# Copyright 2026 Jiang Yongchun
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

: "${CUML_SOURCE:?set CUML_SOURCE to a patched cuML 26.06.00 source tree}"
: "${CUVS_INCLUDE:?set CUVS_INCLUDE to the cuVS source or installed include directory}"

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
SITE=$($PYTHON -c 'import site; print(site.getsitepackages()[0])')
BUILD_DIR=${BUILD_DIR:-$ROOT/build}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
CUDACXX=${CUDACXX:-$CUDA_HOME/bin/nvcc}
CMAKE=${CMAKE:-cmake}

"$CMAKE" -S "$ROOT" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" \
  -DCUDAToolkit_ROOT="$CUDA_HOME" \
  -DCMAKE_PREFIX_PATH="$SITE/libraft/lib64/cmake;$SITE/librmm/lib64/cmake;\
$SITE/rapids_logger/lib64/cmake" \
  -Draft_DIR="$SITE/libraft/lib64/cmake/raft" \
  -Drmm_DIR="$SITE/librmm/lib64/cmake/rmm" \
  -Drapids_logger_DIR="$SITE/rapids_logger/lib64/cmake/rapids_logger" \
  -DCUML_SOURCE="$CUML_SOURCE" \
  -DCUML_WHEEL_INCLUDE="$SITE/libcuml/include" \
  -DCUVS_INCLUDE="$CUVS_INCLUDE" \
  -DLIBCUML_SO="$SITE/libcuml/lib64/libcuml.so"

"$CMAKE" --build "$BUILD_DIR" -j "${JOBS:-8}"
