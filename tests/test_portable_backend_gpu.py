# Copyright 2026 Jiang Yongchun
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest


cp = pytest.importorskip("cupy")
pytest.importorskip("cuml")  # Loads wheel-provided RAPIDS shared-library dependencies.


def device_pointer(array: cp.ndarray) -> ctypes.c_void_p:
    return ctypes.c_void_p(int(array.data.ptr))


@pytest.mark.gpu
def test_portable_backend_matches_explicit_rbf() -> None:
    library_path = os.environ.get("COHERENT_SVM_LIBRARY")
    if not library_path:
        pytest.skip("set COHERENT_SVM_LIBRARY to the built overlay")
    library = ctypes.CDLL(str(Path(library_path).resolve()))

    library.fp16_tma_rbf_create_context.restype = ctypes.c_void_p
    library.fp16_tma_rbf_create_context.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    library.fp16_tma_rbf_materialize_slotted_float_cache.restype = ctypes.c_int
    library.fp16_tma_rbf_materialize_slotted_float_cache.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    library.fp16_tma_rbf_materialize_slotted_half_cache.restype = ctypes.c_int
    library.fp16_tma_rbf_materialize_slotted_half_cache.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    library.fp16_tma_rbf_launch_cache.restype = ctypes.c_int
    library.fp16_tma_rbf_launch_cache.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    library.fp16_tma_rbf_destroy_context.restype = None
    library.fp16_tma_rbf_destroy_context.argtypes = [ctypes.c_void_p]

    rows, columns, dim = 257, 17, 31
    gamma = np.float32(1.0 / dim)
    rng = cp.random.RandomState(7)
    features = cp.ascontiguousarray(rng.standard_normal((rows, dim)).astype(cp.float16))
    working = cp.ascontiguousarray(rng.standard_normal((columns, dim)).astype(cp.float16))
    norm_x = cp.sum(features.astype(cp.float32) ** 2, axis=1)
    norm_w = cp.sum(working.astype(cp.float32) ** 2, axis=1)
    delta = rng.standard_normal(columns).astype(cp.float32)
    slots = cp.arange(columns - 1, -1, -1, dtype=cp.int32)
    padded_delta = cp.zeros(columns, dtype=cp.float32)
    stream = ctypes.c_void_p(int(cp.cuda.get_current_stream().ptr))

    context = library.fp16_tma_rbf_create_context(
        device_pointer(features), device_pointer(working), rows, columns, dim
    )
    assert context
    try:
        float_cache = cp.empty(rows * columns, dtype=cp.float32)
        status = library.fp16_tma_rbf_materialize_slotted_float_cache(
            context,
            device_pointer(norm_x),
            device_pointer(norm_w),
            device_pointer(padded_delta),
            device_pointer(slots),
            device_pointer(float_cache),
            columns,
            ctypes.c_float(float(gamma)),
            stream,
        )
        assert status == 0

        dot = features.astype(cp.float32) @ working.astype(cp.float32).T
        expected = cp.exp(-gamma * cp.maximum(norm_x[:, None] + norm_w[None, :] - 2 * dot, 0))
        slotted_expected = cp.empty((columns, rows), dtype=cp.float32)
        slotted_expected[slots] = expected.T
        cp.testing.assert_allclose(
            float_cache.reshape(columns, rows), slotted_expected, rtol=2e-4, atol=2e-4
        )

        half_cache = cp.empty(rows * columns, dtype=cp.float16)
        status = library.fp16_tma_rbf_materialize_slotted_half_cache(
            context,
            device_pointer(norm_x),
            device_pointer(norm_w),
            device_pointer(padded_delta),
            device_pointer(slots),
            device_pointer(half_cache),
            columns,
            ctypes.c_float(float(gamma)),
            stream,
        )
        assert status == 0
        cp.testing.assert_allclose(
            half_cache.reshape(columns, rows).astype(cp.float32),
            slotted_expected.astype(cp.float16).astype(cp.float32),
            rtol=2e-3,
            atol=2e-3,
        )

        contiguous_cache = cp.empty(rows * columns, dtype=cp.float32)
        partial = cp.empty(1, dtype=cp.float32)
        output = cp.empty(rows, dtype=cp.float32)
        status = library.fp16_tma_rbf_launch_cache(
            context,
            device_pointer(norm_x),
            device_pointer(norm_w),
            device_pointer(delta),
            device_pointer(contiguous_cache),
            device_pointer(partial),
            device_pointer(output),
            columns,
            ctypes.c_float(float(gamma)),
            stream,
        )
        assert status == 0
        cp.cuda.get_current_stream().synchronize()
        cp.testing.assert_allclose(output, expected @ delta, rtol=3e-4, atol=3e-4)
    finally:
        library.fp16_tma_rbf_destroy_context(context)
