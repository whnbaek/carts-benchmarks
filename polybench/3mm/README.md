# 3mm - Three Matrix Multiplications

## Description

Performs three chained matrix multiplications:
- `E = A * B`
- `F = C * D`
- `G = E * F`

## Original Source

**Repository**: [Polybench-ACC](https://github.com/cavazos-lab/PolyBench-ACC)
**Original Path**: `OpenMP/linear-algebra/kernels/3mm/`
**License**: See Polybench LICENSE file

**Polybench Project**:
- **Official Site**: http://polybench.sourceforge.net/
- **Original Paper**: "Polybench: The polyhedral benchmark suite" (Pouchet et al.)
- **Citation**: Pouchet, L.N., et al. "Polybench: The polyhedral benchmark suite." In Workshop on Polyhedral Compilation Techniques (IMPACT), 2012.

## Algorithm

```
Input: A[NI×NK], B[NK×NJ], C[NJ×NM], D[NM×NL]
Output: G[NI×NL]

E[NI×NJ] = A * B
F[NJ×NL] = C * D
G[NI×NL] = E * F
```

## Problem Sizes

- **MINI**: 32×32 matrices
- **SMALL**: 128×128 matrices
- **STANDARD**: 1024×1024 matrices
- **LARGE**: 2000×2000 matrices
- **EXTRALARGE**: 4000×4000 matrices

## OpenMP Parallelization

Uses `#pragma omp parallel private(j, k)` with multiple `#pragma omp for` regions within a single parallel region.

## Use in Machine Learning

Three chained matrix multiplications test:
- Dependency chains across multiple operations
- Intermediate result management
- Memory allocation patterns for temporary matrices

## CARTS Compatibility

- ✅ No global variables
- ✅ Clean parameter passing
- ✅ OpenMP parallel for
- ✅ Static functions

## References

- **Polybench Suite**: http://polybench.sourceforge.net/
- **Repository**: https://github.com/cavazos-lab/PolyBench-ACC
- **Original Authors**: Louis-Noël Pouchet, Ohio State University
