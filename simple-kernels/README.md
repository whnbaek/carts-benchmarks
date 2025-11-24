# Simple Kernels

Collection of CARTS-authored microbenchmarks inspired by classic kernels. Each directory contains a self-contained C file, a three-line Makefile (see `Makefile`), and leverages the shared CARTS pipeline targets (`seq`, `metadata`, `parallel`, `concurrency`, `concurrency-opt`, `clean`).

| Kernel | Source Inspiration | GitHub | Notes |
|--------|--------------------|--------|-------|
| `stream/` | STREAM Triad | [https://www.cs.virginia.edu/stream/](https://www.cs.virginia.edu/stream/) | Memory-bandwidth bound vector triad with adjustable length (`N`) and repetitions (`REPS`). |
| `gemm/` | BLAS SGEMM Reference | [https://www.netlib.org/blas/](https://www.netlib.org/blas/) | Naive GEMM (`C = αAB + βC`) using flat arrays and OpenMP `parallel for`. |
| `axpy/` | BLAS AXPY | [https://www.netlib.org/blas/](https://www.netlib.org/blas/) | SAXPY-style vector update (`y = αx + y`) for ultra-fast regression tests. |

## Usage

```bash
# Build and run STREAM triad
make -C simple-kernels/stream clean all

# Override vector length via CFLAGS
make -C simple-kernels/stream all CFLAGS="-DN=1048576 -DREPS=10"

# Build GEMM (defaults to N=256)
make -C simple-kernels/gemm all
make -C simple-kernels/gemm all CFLAGS="-DN=512"

# Run SAXPY with a smaller vector
make -C simple-kernels/axpy all
make -C simple-kernels/axpy all CFLAGS="-DLENGTH=262144 -DALPHA=2.0f"
```

Both kernels allocate and initialize their own buffers, emit lightweight checksums, and are designed to run quickly (seconds) for CI smoke tests. When contributing additional microbenchmarks, follow the same structure and update this README plus `benchmarks_manifest.yaml`.
