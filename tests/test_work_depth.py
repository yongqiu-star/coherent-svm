# Copyright 2026 Jiang Yongchun
# SPDX-License-Identifier: Apache-2.0
import runpy
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "studies" / "work-depth" / "metrics.py"
WorkDepthSample = runpy.run_path(MODULE)["WorkDepthSample"]


def test_weighted_progress_distinguishes_global_and_local_data() -> None:
    sample = WorkDepthSample(10.0, 8.0, 2, 100, 20, 10.0, 0.1)
    assert sample.weighted_kernel_work == 30.0
    assert sample.progress_per_kernel_item == 2.0 / 30.0
    assert sample.progress_per_dependent_stage == 0.1
