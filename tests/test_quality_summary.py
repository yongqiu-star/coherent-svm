# Copyright 2026 Jiang Yongchun
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from coherent_svm.cli import paired_quality


def test_paired_quality_counts_and_correlation(tmp_path) -> None:
    labels = np.array([-1, -1, 1, 1], dtype=np.float32)
    baseline_prediction = np.array([-1, 1, 1, 1], dtype=np.float32)
    candidate_prediction = np.array([-1, -1, -1, 1], dtype=np.float32)
    baseline = tmp_path / "baseline.npz"
    candidate = tmp_path / "candidate.npz"
    np.savez(baseline, labels=labels, prediction=baseline_prediction, score=[-2, 1, 2, 3])
    np.savez(candidate, labels=labels, prediction=candidate_prediction, score=[-1, -0.5, -0.1, 2])

    result = paired_quality(baseline, candidate)

    assert result["rows"] == 4
    assert result["prediction_disagreements"] == 2
    assert result["baseline_only_correct"] == 1
    assert result["candidate_only_correct"] == 1
    assert result["mcnemar_exact_p"] == 1.0
    assert -1.0 <= result["score_pearson_correlation"] <= 1.0
