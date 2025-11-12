# atax - Matrix Transpose and Vector Multiplication

## Description

Computes `y = A^T * (A * x)` - matrix transpose times matrix-vector product.

## Original Source

**Repository**: [Polybench-ACC](https://github.com/cavazos-lab/PolyBench-ACC)
**Original Path**: `OpenMP/linear-algebra/kernels/atax/`
**License**: See Polybench LICENSE file

**Polybench Project**:
- **Official Site**: http://polybench.sourceforge.net/
- **Citation**: Pouchet, L.N., et al. "Polybench: The polyhedral benchmark suite." IMPACT, 2012.

## Algorithm

```
Input: A[NX×NY], x[NY]
Output: y[NY]

tmp[NX] = A * x
y[NY] = A^T * tmp
```

## Problem Sizes

- **MINI**: 32×32
- **SMALL**: 128×128
- **STANDARD**: 4000×4000
- **LARGE**: 8000×8000
- **EXTRALARGE**: 100000×100000

## References

- **Repository**: https://github.com/cavazos-lab/PolyBench-ACC
- **Polybench**: http://polybench.sourceforge.net/
