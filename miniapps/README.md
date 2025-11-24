# Miniapps

Compact application-style kernels that sit between the microbenchmarks and full suites. Each miniapp exposes the same shared CARTS pipeline targets and is intended to exercise more realistic memory footprints without lengthy runtimes.

## stencil2d

- **Path**: `miniapps/stencil2d`
- **Origin**: Inspired by the 2D heat/stencil kernels from [ParRes Kernels](https://github.com/ParRes/Kernels) and DOE miniapps.
- **Pattern**: 5-point stencil with configurable domain/iterations (compile-time macros).
- **Use cases**: Validate OpenMP lowering, metadata reuse, and runtime concurrency for mid-sized workloads.

### Build

```bash
make -C external/carts-benchmarks/miniapps/stencil2d clean all
make -C external/carts-benchmarks/miniapps/stencil2d all CFLAGS="-DN=1024 -DTSTEPS=50"
```

### Next Miniapps

- `heat3d` (3D Jacobi) from ParRes Kernels
- `miniFE` (conjugate gradient) from [Mantevo/miniFE](https://github.com/Mantevo/miniFE)
- `miniGMG` / `AMG2013` for hierarchical solvers

Please document provenance and update `benchmarks_manifest.yaml` whenever a new miniapp is added.
