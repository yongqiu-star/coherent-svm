# Technical Report

- [Read the PDF](svm_gpu_technical_report.pdf)
- [Open the main LaTeX source](svm_gpu_technical_report.tex)

The report is titled **Optimizing RBF SVM Training on Modern GPUs: Coherent
Low Precision, Kernel Reuse, and the Limits of Parallel SMO**. It accompanies
the public artifact in the repository root.

Build it with a standard TeX Live installation:

```bash
make
```

Remove generated LaTeX files with `make clean`, or remove the PDF as well with
`make distclean`. The committed PDF is built from the source in this directory;
development-machine paths, private run identifiers, raw datasets, and binary
artifacts are intentionally excluded.
