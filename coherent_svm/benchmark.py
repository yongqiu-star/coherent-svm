# Copyright 2026 Jiang Yongchun
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cupy as cp
import numpy as np
from cuml.svm import SVC


def first_int(value: object) -> int:
    if hasattr(value, "get"):
        value = value.get()
    return int(np.asarray(value).reshape(-1)[0])


def synthetic(rows: int, dim: int, seed: int) -> tuple[cp.ndarray, cp.ndarray]:
    rng = cp.random.RandomState(seed)
    features = rng.standard_normal((rows, dim), dtype=cp.float32)
    signal_dim = min(dim, 64)
    score = features[:, :signal_dim].sum(axis=1) / math.sqrt(signal_dim)
    score += 0.5 * rng.standard_normal(rows, dtype=cp.float32)
    labels = cp.where(score >= 0, 1.0, -1.0).astype(cp.float32)
    return cp.ascontiguousarray(features), cp.ascontiguousarray(labels)


def prepared(root: Path, rows: int) -> tuple[cp.ndarray, cp.ndarray]:
    features = np.load(root / "x_standardized_f32.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(root / "y_f32.npy", mmap_mode="r", allow_pickle=False)
    if rows > features.shape[0] or labels.shape[0] != features.shape[0]:
        raise ValueError("prepared dataset does not contain the requested rows")
    return (
        cp.asarray(features[:rows], dtype=cp.float32),
        cp.asarray(labels[:rows], dtype=cp.float32),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--held-out-dir", type=Path)
    parser.add_argument("--held-out-rows", type=int)
    parser.add_argument("--quality-output", type=Path)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--cache-size-mib", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=60_000)
    parser.add_argument("--eval-rows", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prepared_dir is None:
        features, labels = synthetic(args.rows, args.dim, args.seed)
    else:
        features, labels = prepared(args.prepared_dir, args.rows)
        args.dim = int(features.shape[1])
    cp.cuda.get_current_stream().synchronize()

    gamma = args.gamma if args.gamma is not None else 1.0 / args.dim
    model = SVC(
        C=args.c,
        kernel="rbf",
        gamma=gamma,
        tol=args.tol,
        cache_size=args.cache_size_mib,
        max_iter=args.max_iter,
        nochange_steps=1000,
        output_type="cupy",
        verbose=False,
    )
    cp.cuda.get_current_stream().synchronize()
    started = time.perf_counter()
    model.fit(features, labels, convert_dtype=False)
    cp.cuda.get_current_stream().synchronize()
    fit_seconds = time.perf_counter() - started

    eval_rows = min(args.eval_rows, args.rows)
    prediction = model.predict(features[:eval_rows], convert_dtype=False)
    cp.cuda.get_current_stream().synchronize()
    accuracy = float(cp.mean((prediction == labels[:eval_rows]).astype(cp.float32)).get())
    properties = cp.cuda.runtime.getDeviceProperties(0)

    record = {
        "method": args.method,
        "dataset_id": args.dataset_id,
        "device_name": properties["name"].decode(),
        "compute_capability": f"{properties['major']}.{properties['minor']}",
        "rows": args.rows,
        "dim": args.dim,
        "gamma": gamma,
        "c": args.c,
        "tol": args.tol,
        "fit_seconds": fit_seconds,
        "fit_status": first_int(model.fit_status_),
        "iterations": first_int(model.n_iter_),
        "support_vectors": first_int(model.n_support_),
        "evaluation_rows": eval_rows,
        "training_accuracy_sample": accuracy,
    }
    if args.held_out_dir is not None:
        if args.held_out_rows is None or args.quality_output is None:
            raise ValueError("held-out evaluation requires --held-out-rows and --quality-output")
        held_features, held_labels = prepared(args.held_out_dir, args.held_out_rows)
        held_prediction = model.predict(held_features, convert_dtype=False)
        held_score = model.decision_function(held_features, convert_dtype=False)
        cp.cuda.get_current_stream().synchronize()
        host_labels = cp.asnumpy(held_labels).reshape(-1)
        host_prediction = cp.asnumpy(held_prediction).reshape(-1)
        host_score = cp.asnumpy(held_score).reshape(-1)
        from sklearn.metrics import roc_auc_score

        record["held_out_rows"] = int(host_labels.size)
        record["held_out_accuracy"] = float(np.mean(host_prediction == host_labels))
        record["held_out_roc_auc"] = float(roc_auc_score(host_labels > 0, host_score))
        args.quality_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.quality_output,
            labels=host_labels,
            prediction=host_prediction,
            score=host_score,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
