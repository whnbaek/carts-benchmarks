# SPECFEM3D Stress Update

- **Origin**: Inspired by the stress tensor update routines in [SPECFEM3D](https://github.com/SPECFEM/specfem3d).
- **Pattern**: Update the six independent stress components based on velocity gradients (Hooke's law in isotropic media).
- **Use case**: Exercises mixed-derivative stencils, Lamé parameter lookups, and memory bandwidth patterns similar to SPECFEM's finite-difference backend.

## Build

```bash
make -C external/carts-benchmarks/specfem3d_stress clean all
make -C external/carts-benchmarks/specfem3d_stress CFLAGS="-DNX=64 -DNY=48 -DNZ=48 -DDT=5e-4" all
```

Artifacts mirror other single-file benchmarks (MLIR + logs under `build/` and `logs/` respectively).
