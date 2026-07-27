# Coherent SVM

Public repository: <https://github.com/yongqiu-star/coherent-svm>

This repository is a clean, path-independent research artifact for studying
coherent low-precision execution in nonlinear RBF SVM training. It is an
integration overlay for RAPIDS cuML 26.06.00 rather than a standalone SVM
implementation.

The central idea is simple:

> Low precision is not a GEMM option. It defines the approximate kernel
> operator that drives every state-changing decision.

A naive design stores global kernel columns in FP16 while constructing the
local working-set matrix independently in FP32. The two parts of the solver
then optimize against different numerical operators. `coherent_fp16` instead
rounds both the persistent cache and the local `K[W,W]` values at the same
representation boundary while retaining FP32 accumulators, gradients, dual
variables, and committed SMO steps.

## What is included

- a cuML 26.06.00 integration patch;
- the coherent column-cache and working-set dataflow;
- a portable cuBLAS/CUDA reference backend;
- public experiment configurations and normalized result schemas;
- a synthetic smoke path and an EMBER-65K benchmark harness; and
- work--depth study code that distinguishes global columns, local rows, and
  dependent decision stages.

The architecture-tuned SM120 backend used for the technical-report timing is
not included in this snapshot because its third-party provenance has not yet
been cleared. The portable backend is deliberately simple and materializes a
dot-product panel. It validates semantics and integration but must not be used
to reproduce or advertise the reported 1.351x speedup.

## Requirements

- Linux and an NVIDIA GPU;
- CUDA 13.x, CMake 3.26+, and Ninja;
- cuML/libcuml 26.06.00 with its RAFT, RMM, cuVS, and rapids_logger headers;
- Python 3.11+; the measured environment used Python 3.12.

The optimized unpublished backend targets `sm_120a`. The included reference
backend uses cuBLAS and can build for the local CUDA architecture.

The current cuML overlay exposes the coherent path only for dense workloads
with at least 65,536 rows and a Tensor-Core-suitable feature shape. The public
synthetic smoke therefore uses 65,536 x 256. The separately studied compact
low-dimensional kernel is not part of this clean release.

## Quick start

```bash
uv sync --frozen --extra gpu --extra quality --extra dev
uv run coherent-svm doctor

export CUML_SOURCE=/path/to/cuml-26.06.00
export CUVS_INCLUDE=/path/to/cuvs/include
git -C "$CUML_SOURCE" apply \
  "$PWD/patches/cuml-26.06.00/coherent-fp16.patch"

CUML_SOURCE="$CUML_SOURCE" scripts/build.sh

uv run coherent-svm reproduce \
  --config artifact/configs/smoke-synthetic.toml \
  --library build/libcoherent_svm.so \
  --output-root results
```

Validate the clean reference backend against an explicit RBF oracle with:

```bash
COHERENT_SVM_LIBRARY=build/libcoherent_svm.so uv run pytest -q
```

For the primary study, prepare EMBER locally and use
`artifact/configs/ember65k-ab.toml`. The held-out protocol is encoded by
`artifact/configs/ember65k-quality.toml`; its expected array layout is
documented in `artifact/README.md`. Raw data, generated arrays, binaries, and
profiler reports are intentionally excluded from version control.

## Result contract

Each reproduction writes:

```text
results/<config-id>/<run-id>/
├── manifest.json
├── measurements.jsonl
├── summary.json
├── checks.json
└── artifacts.sha256
```

Public manifests record software revisions, dependency versions, normalized
arguments, hardware class, dataset identifiers, and numerical checks. They do
not record hostnames, user names, GPU UUIDs, private remotes, or absolute local
paths.

## Interpretation boundary

The coherent solver reaches its own low-precision stopping criterion and is
evaluated through held-out prediction quality. It is not an exact solution to
the original FP32 RBF objective. The report therefore treats the FP32 KKT
audit as a model-difference diagnostic, not as the defining convergence test
for `coherent_fp16`.

See `NOTICE` for provenance and redistribution boundaries.
