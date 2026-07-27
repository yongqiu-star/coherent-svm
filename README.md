# Coherent SVM

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Technical report](https://img.shields.io/badge/technical_report-PDF-b31b1b.svg)](report/svm_gpu_technical_report.pdf)

**Coherent low-precision execution for nonlinear RBF-SVM training on modern GPUs.**
This repository contains the public research artifact and the complete
[technical report](report/svm_gpu_technical_report.pdf).

> **Central insight:** low precision is not merely a faster GEMM option. In an
> iterative solver, it defines the approximate operator that drives every
> state-changing decision.

## Why this matters

GPU RBF-SVM training alternates between two very different forms of work:

1. generating and reusing expensive kernel values; and
2. making state-dependent SMO decisions inside a serial working-set solver.

A naive mixed-precision implementation stores global kernel columns in FP16
but constructs the local working-set matrix independently in FP32. The solver
then makes local decisions with one kernel operator and applies them globally
with another. Our coherent path instead uses the same FP16 representation
boundary for both cached `K[:, W]` columns and local `K[W, W]` values, while
retaining FP32 accumulation, gradients, dual variables, and committed SMO
steps.

```mermaid
flowchart LR
    A[Select working set W] --> B[Local K W,W<br/>coherent FP16 boundary]
    B --> C[FP32 SMO block solve]
    C --> D[Delta alpha]
    D --> E[Global K :,W cache<br/>same FP16 boundary]
    E --> F[FP32 gradient update]
    F --> A
```

This is the difference between **isolated low-precision kernels** and a
**solver-wide coherent numerical dataflow**.

## Results at a glance

The report's primary experiment uses EMBER-65K (`65,536 x 2,381`) on an
NVIDIA RTX PRO 6000 Blackwell GPU.

| Evidence | FP32 reference | Coherent FP16 | Result |
| --- | ---: | ---: | ---: |
| Matched fit time, 7 balanced fresh-process runs | 0.486954 s | 0.360385 s | **1.351x** (95% CI: 1.263x-1.403x) |
| Held-out accuracy, 200K examples | 0.830980 | 0.830895 | -0.000085 |
| Held-out ROC AUC | 0.93525055 | 0.93516107 | -0.00008948 |

The timing row compares coherent FP16 with a matched direct-FP32-cache
control. The held-out rows compare saved stock-FP32 and coherent-FP16 models;
they are a separate quality experiment, not the timing denominator.

After accelerating kernel generation and reuse, `SmoBlockSolveCoarsened`
becomes the dominant GPU kernel and accounts for **40.3%** of the measured
steady solver phase. This exposes the next limit: dependent SMO decision depth,
not Tensor Core throughput.

## What the study found

- **Coherence before precision.** Local and global kernel values must obey one
  numerical contract; accelerating only GEMM can increase solver work.
- **Optimize progress, not activity.** The useful metric is convergence
  progress per weighted kernel-data item, not occupancy or pair updates per
  second.
- **Bottlenecks migrate.** Faster kernel materialization exposes a serial
  pair-selection and update chain inside the block solver.
- **The best path is phase- and shape-dependent.** Reuse, materialization, and
  low-precision choices do not have one globally optimal dispatch rule.

The report also records negative results—including incoherent half-precision,
uniform 8-bit kernel caches, speculative prefetch, and naive batched pair
updates—so the artifact documents the applicability boundary rather than only
the winning configuration.

## Read the report

- [Technical report (PDF)](report/svm_gpu_technical_report.pdf)
- [LaTeX source](report/svm_gpu_technical_report.tex)
- [Report build instructions](report/README.md)

The report explains the motivation, SMO dataflow, profiling evidence,
coherent-FP16 mechanism, quality contract, work-depth bottleneck, rejected
hypotheses, and open research questions.

## What is released

This is a clean, path-independent integration overlay for RAPIDS cuML
26.06.00, not a standalone SVM implementation. It includes:

- the cuML integration patch;
- coherent column-cache and working-set dataflow;
- a portable CUDA/cuBLAS reference backend;
- public experiment configurations and normalized result schemas;
- synthetic smoke and EMBER-65K benchmark paths; and
- work-depth analysis code.

The architecture-tuned SM120 backend used for the report's headline timing is
**not included** because its third-party provenance has not yet been cleared.
The released portable backend validates numerical semantics and integration,
but it does **not** reproduce the reported 1.351x speedup. See [NOTICE](NOTICE)
for the exact provenance and redistribution boundary.

## Requirements

- Linux with a compute-capability 12.0 (`sm_120`) NVIDIA GPU for the complete
  patched-solver `reproduce` path;
- CUDA 13.x, CMake 3.26+, Ninja, and Python 3.11+;
- cuML/libcuml 26.06.00 with RAFT, RMM, cuVS, and rapids_logger headers; and
- [`uv`](https://docs.astral.sh/uv/) for the locked Python environment.

The unpublished tuned backend targets `sm_120a`. The standalone portable
backend uses cuBLAS and builds for the local CUDA architecture, but the current
cuML overlay activates its complete coherent solver path only on compute
capability 12.0. `coherent-svm doctor` reports this gate explicitly. Dense
coherent runs additionally require at least 65,536 rows and a
Tensor-Core-suitable feature shape.

## Quick start

```bash
git clone https://github.com/yongqiu-star/coherent-svm.git
cd coherent-svm

uv sync --frozen --extra gpu --extra quality --extra dev
uv run coherent-svm doctor

export CUML_SOURCE=/path/to/cuml-26.06.00
export CUVS_INCLUDE=/path/to/cuvs/include
git -C "$CUML_SOURCE" apply \
  "$PWD/patches/cuml-26.06.00/coherent-fp16.patch"

PYTHON="$PWD/.venv/bin/python" scripts/build.sh

uv run coherent-svm reproduce \
  --config artifact/configs/smoke-synthetic.toml \
  --library build/libcoherent_svm.so \
  --output-root results
```

Validate the portable backend against an explicit RBF oracle:

```bash
COHERENT_SVM_LIBRARY=build/libcoherent_svm.so uv run pytest -q
```

For the primary study, prepare EMBER locally and use
`artifact/configs/ember65k-ab.toml`. The held-out protocol is encoded by
`artifact/configs/ember65k-quality.toml`; the expected array layout is
documented in [artifact/README.md](artifact/README.md). Raw data, generated
arrays, binaries, and profiler reports are intentionally excluded from Git.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/`, `include/` | Portable coherent low-precision backend |
| `patches/cuml-26.06.00/` | cuML integration and decision-path patches |
| `coherent_svm/` | CLI, benchmark runner, and sanitized provenance |
| `artifact/` | Reproduction configs, schemas, and reference summaries |
| `studies/work-depth/` | Serial-depth and kernel-data accounting |
| `tests/` | Public-tree, numerical, quality, and GPU integration tests |
| `report/` | Technical-report PDF and complete LaTeX source |

Each reproduction writes a normalized result bundle:

```text
results/<config-id>/<run-id>/
|-- manifest.json
|-- measurements.jsonl
|-- summary.json
|-- checks.json
`-- artifacts.sha256
```

The manifest retains the public source revision and dirty state, configuration
and library digests, logical dataset identifier, Python version, and platform
class. Per-process measurements contain the device and compute capability,
shape, solver parameters, timing, iteration count, support-vector count, and
sampled accuracy. `checks.json` records completion, fit status, and finite
timing. These files deliberately exclude hostnames, usernames, GPU UUIDs,
private remotes, and absolute paths.

## Numerical claim boundary

The coherent solver reaches its declared low-precision stopping criterion and
is evaluated through held-out prediction quality. It is not an exact solution
to the original FP32 RBF objective. The report therefore treats an independent
FP32 KKT audit as a model-difference diagnostic, not as the convergence
definition for `coherent_fp16`. The independent audit reported in the PDF is
not automated by the current public runner.

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff). The public
artifact is released under [Apache-2.0](LICENSE); cuML-derived files retain
their upstream notices and modification boundaries.
