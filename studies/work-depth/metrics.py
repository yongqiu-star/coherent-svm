# Copyright 2026 Jiang Yongchun
# SPDX-License-Identifier: Apache-2.0
"""Mechanism-level metrics for dependency-limited iterative GPU solvers."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkDepthSample:
    objective_before: float
    objective_after: float
    global_columns: int
    local_rows: int
    dependent_stages: int
    global_column_cost: float
    local_row_cost: float

    @property
    def objective_progress(self) -> float:
        return self.objective_before - self.objective_after

    @property
    def weighted_kernel_work(self) -> float:
        return (
            self.global_columns * self.global_column_cost
            + self.local_rows * self.local_row_cost
        )

    @property
    def progress_per_kernel_item(self) -> float:
        work = self.weighted_kernel_work
        return self.objective_progress / work if work else float("inf")

    @property
    def progress_per_dependent_stage(self) -> float:
        return self.objective_progress / self.dependent_stages

