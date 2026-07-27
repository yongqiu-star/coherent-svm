# Copyright 2026 Jiang Yongchun
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import site
import statistics
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .provenance import public_manifest, sha256_file, write_json


ROOT = Path(__file__).resolve().parents[1]
CONTROLLED_ENV = {
    "CUML_SVM_DIRECT_SLOTTED_CACHE",
    "CUML_SVM_FP16_PERSISTENT_CACHE",
    "CUML_SVM_FP16_COHERENT_TILE",
    "CUML_SVM_COLUMN_CACHE_COLUMNS",
    "CUML_SVM_COLUMN_CACHE_GIB",
}


def rapids_library_path(environment: dict[str, str]) -> str:
    """Make wheel-provided RAPIDS dependencies visible to the preload process."""
    candidates: list[Path] = []
    for site_root in map(Path, site.getsitepackages()):
        candidates.extend(path for path in site_root.glob("*/lib64") if path.is_dir())
    inherited = [
        Path(path)
        for path in environment.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if path
    ]
    ordered = list(dict.fromkeys([*sorted(candidates), *inherited]))
    return os.pathsep.join(map(str, ordered))


def load_config(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return tomllib.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def doctor(_: argparse.Namespace) -> int:
    checks: dict[str, object] = {
        "python": sys.version.split()[0],
        "nvcc": shutil.which("nvcc") is not None,
        "nvidia_smi": shutil.which("nvidia-smi") is not None,
    }
    for package in ("cupy", "cuml", "numpy", "nvtx"):
        try:
            module = __import__(package)
            checks[package] = getattr(module, "__version__", "available")
        except Exception as error:  # pragma: no cover - environment diagnostic
            checks[package] = f"unavailable: {type(error).__name__}"
    print(json.dumps(checks, indent=2, sort_keys=True))
    unavailable = any(str(value).startswith("unavailable") for value in checks.values())
    return 0 if checks["nvcc"] and not unavailable else 2


def method_environment(method: str, cache_columns: int) -> dict[str, str]:
    common = {
        "CUML_SVM_COLUMN_CACHE_COLUMNS": str(cache_columns),
        "CUML_SVM_FP16_TMA_RBF": "1",
        "CUML_SVM_FP16_TMA_REPAIR": "0",
        "CUML_SVM_FUSED_RBF_UPDATE": "0",
        "CUML_SVM_BOUNDED_CACHE": "1",
    }
    if method == "stock_fp32":
        common["CUML_SVM_FP16_TMA_RBF"] = "0"
        common["CUML_SVM_BOUNDED_CACHE"] = "0"
    elif method == "direct_fp32":
        common["CUML_SVM_DIRECT_SLOTTED_CACHE"] = "1"
    elif method == "persistent_fp16":
        common["CUML_SVM_FP16_PERSISTENT_CACHE"] = "1"
    elif method == "coherent_fp16":
        common["CUML_SVM_FP16_PERSISTENT_CACHE"] = "1"
        common["CUML_SVM_FP16_COHERENT_TILE"] = "1"
    else:
        raise ValueError(f"unsupported method: {method}")
    return common


def balanced_order(methods: list[str], repeat: int) -> list[str]:
    shift = repeat % len(methods)
    return methods[shift:] + methods[:shift]


def summarize(records: list[dict], baseline: str, candidate: str) -> dict:
    grouped = {
        method: [float(item["fit_seconds"]) for item in records if item["method"] == method]
        for method in (baseline, candidate)
    }
    base_median = statistics.median(grouped[baseline])
    candidate_median = statistics.median(grouped[candidate])
    rng = np.random.default_rng(0)
    bootstrap = []
    for _ in range(10_000):
        base = rng.choice(grouped[baseline], len(grouped[baseline]), replace=True)
        cand = rng.choice(grouped[candidate], len(grouped[candidate]), replace=True)
        bootstrap.append(float(np.median(base) / np.median(cand)))
    return {
        "baseline": {"method": baseline, "median_fit_seconds": base_median},
        "candidate": {"method": candidate, "median_fit_seconds": candidate_median},
        "speedup": base_median / candidate_median,
        "speedup_ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "repeats": len(grouped[baseline]),
    }


def paired_quality(baseline_file: Path, candidate_file: Path) -> dict[str, float | int]:
    baseline = np.load(baseline_file, allow_pickle=False)
    candidate = np.load(candidate_file, allow_pickle=False)
    labels = baseline["labels"].reshape(-1)
    if not np.array_equal(labels, candidate["labels"].reshape(-1)):
        raise ValueError("paired quality files use different labels")
    base_prediction = baseline["prediction"].reshape(-1)
    candidate_prediction = candidate["prediction"].reshape(-1)
    base_correct = base_prediction == labels
    candidate_correct = candidate_prediction == labels
    baseline_only = int(np.count_nonzero(base_correct & ~candidate_correct))
    candidate_only = int(np.count_nonzero(~base_correct & candidate_correct))
    discordant = baseline_only + candidate_only
    if discordant:
        from scipy.stats import binomtest

        mcnemar_p = float(
            binomtest(baseline_only, discordant, p=0.5, alternative="two-sided").pvalue
        )
    else:
        mcnemar_p = 1.0
    correlation = float(
        np.corrcoef(baseline["score"].reshape(-1), candidate["score"].reshape(-1))[0, 1]
    )
    return {
        "rows": int(labels.size),
        "prediction_disagreements": int(np.count_nonzero(base_prediction != candidate_prediction)),
        "baseline_only_correct": baseline_only,
        "candidate_only_correct": candidate_only,
        "mcnemar_exact_p": mcnemar_p,
        "score_pearson_correlation": correlation,
    }


def reproduce(args: argparse.Namespace) -> int:
    config, digest = load_config(args.config)
    config_id = config["id"]
    dataset = config["dataset"]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / config_id / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    prepared_dir = None
    if dataset["kind"] == "prepared":
        if args.data_root is None:
            raise ValueError("--data-root is required for a prepared dataset")
        prepared_dir = args.data_root / dataset["relative_path"]

    manifest = public_manifest(
        repository=ROOT,
        config_id=config_id,
        config_digest=digest,
        library=args.library,
        dataset_id=dataset["id"],
    )
    write_json(run_dir / "manifest.json", manifest)
    measurements = run_dir / "measurements.jsonl"
    held_out_relative = dataset.get("held_out_relative_path")
    quality_dir = run_dir / ".quality" if held_out_relative else None

    methods = list(config["experiment"]["methods"])
    for repeat in range(int(config["experiment"]["repeats"])):
        for method in balanced_order(methods, repeat):
            environment = os.environ.copy()
            for key in CONTROLLED_ENV:
                environment.pop(key, None)
            environment.update(
                method_environment(method, int(config["solver"]["cache_columns"]))
            )
            environment["LD_LIBRARY_PATH"] = rapids_library_path(environment)
            environment["LD_PRELOAD"] = str(args.library.resolve())
            command = [
                sys.executable,
                "-m",
                "coherent_svm.benchmark",
                "--method",
                method,
                "--dataset-id",
                dataset["id"],
                "--rows",
                str(dataset["rows"]),
                "--dim",
                str(dataset["dim"]),
                "--c",
                str(config["solver"]["c"]),
                "--tol",
                str(config["solver"]["tol"]),
                "--cache-size-mib",
                str(config["solver"].get("cache_size_mib", 1024.0)),
                "--max-iter",
                str(config["solver"]["max_iter"]),
                "--eval-rows",
                str(config["evaluation"]["rows"]),
                "--output",
                str(measurements),
            ]
            if prepared_dir is not None:
                command += ["--prepared-dir", str(prepared_dir)]
            if held_out_relative:
                if args.data_root is None:
                    raise ValueError("--data-root is required for held-out evaluation")
                command += [
                    "--held-out-dir",
                    str(args.data_root / held_out_relative),
                    "--held-out-rows",
                    str(dataset["held_out_rows"]),
                    "--quality-output",
                    str(quality_dir / f"{method}-{repeat}.npz"),
                ]
            subprocess.run(command, check=True, cwd=ROOT, env=environment)

    records = [json.loads(line) for line in measurements.read_text().splitlines()]
    summary = summarize(records, methods[0], methods[-1])
    summary["config_id"] = config_id
    if quality_dir is not None:
        summary["paired_quality"] = paired_quality(
            quality_dir / f"{methods[0]}-0.npz",
            quality_dir / f"{methods[-1]}-0.npz",
        )
        shutil.rmtree(quality_dir)
    write_json(run_dir / "summary.json", summary)
    checks = {
        "all_runs_completed": len(records) == len(methods) * int(config["experiment"]["repeats"]),
        "all_fit_status_zero": all(item["fit_status"] == 0 for item in records),
        "all_finite": all(np.isfinite(item["fit_seconds"]) for item in records),
    }
    write_json(run_dir / "checks.json", checks)
    hashes = {
        name: sha256_file(run_dir / name)
        for name in ("manifest.json", "measurements.jsonl", "summary.json", "checks.json")
    }
    write_json(run_dir / "artifacts.sha256", hashes)
    print(run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir.name)
    return 0 if all(checks.values()) else 3


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="coherent-svm")
    commands = root.add_subparsers(dest="command", required=True)
    doctor_parser = commands.add_parser("doctor")
    doctor_parser.set_defaults(function=doctor)
    reproduce_parser = commands.add_parser("reproduce")
    reproduce_parser.add_argument("--config", type=Path, required=True)
    reproduce_parser.add_argument("--library", type=Path, required=True)
    reproduce_parser.add_argument("--data-root", type=Path)
    reproduce_parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    reproduce_parser.set_defaults(function=reproduce)
    return root


def main() -> None:
    args = parser().parse_args()
    raise SystemExit(args.function(args))


if __name__ == "__main__":
    main()
