# Artifact contract

The artifact separates three questions that are often conflated in GPU
optimization papers:

1. Does the implementation execute the intended numerical operator?
2. Does the complete solver retain the declared quality contract?
3. Is the architecture-tuned backend faster under a matched denominator?

The public portable backend answers the first question and exercises the
end-to-end integration path. The configurations below encode the second and
third measurement protocols. The unpublished SM120 backend is required to
reproduce the report's headline timing.

## Configurations

| File | Purpose |
| --- | --- |
| `smoke-synthetic.toml` | Build, ABI, finite-output, and small numerical checks |
| `ember65k-ab.toml` | Balanced direct-FP32 versus coherent-FP16 timing protocol |
| `ember65k-quality.toml` | Stock-FP32 versus coherent-FP16 held-out quality protocol |

Prepared datasets are external to the repository and use this layout:

```text
<data-root>/ember2018/
├── probe-65536/
│   ├── x_standardized_f32.npy
│   └── y_f32.npy
└── test-200000/
    ├── x_standardized_f32.npy
    └── y_f32.npy
```

Both feature arrays are row-major `float32`; labels are `float32` values in
`{-1,+1}`. The held-out split must be transformed with the training split's
statistics. Raw EMBER data and derived arrays are not redistributed here.

## Output

Each run contains a normalized manifest, per-process measurements, a summary,
checks, and checksums. Quality runs also compute prediction disagreements,
the exact paired McNemar statistic, and score correlation; temporary score
arrays are removed after aggregation.

Absolute paths are accepted as command inputs but are not copied into the
public manifest. Wall-clock values are observations, not cross-machine pass
conditions.
