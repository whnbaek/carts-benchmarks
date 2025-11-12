# seidel-2d - Gauss-Seidel 2D Stencil

## Description

2D Gauss-Seidel iterative solver with 9-point stencil averaging.

## Original Source

**Repository**: [Polybench-ACC](https://github.com/cavazos-lab/PolyBench-ACC)
**Original Path**: `OpenMP/stencils/seidel-2d/`

**Polybench Project**: http://polybench.sourceforge.net/

## Algorithm

9-point averaging stencil:
```
A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
         + A[i][j-1]   + A[i][j]   + A[i][j+1]
         + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1]) / 9.0
```

## References

- **Repository**: https://github.com/cavazos-lab/PolyBench-ACC
- **Gauss-Seidel Method**: Classical iterative solver for linear systems
