# SeisSol Volume Integral

- **Origin**: Inspired by the element-local ADER-DG kernels in [SeisSol/SeisSol](https://github.com/SeisSol/SeisSol), specifically the volume integral that multiplies precomputed gradient matrices with DOFs.
- **Pattern**: Matrix-matrix operations (`fluxMatrix * (gradMatrix * dofs)`) performed per element, stressing register tiling and memory layouts similar to SeisSol's generated kernels.
- **Use case**: Complements stencil-based seismic kernels with dense-tensor contractions typical of ADER-DG solvers.

## Build

```bash
make -C external/carts-benchmarks/seissol_volume_integral clean all
make -C external/carts-benchmarks/seissol_volume_integral CFLAGS="-DN_ELEMENTS=32 -DN_BASIS=20 -DN_QUAD=25" all
```
