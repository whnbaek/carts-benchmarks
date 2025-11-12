# bicg - BiConjugate Gradient Sub-kernel

## Description

BiCG sub-kernel: `s = A^T * r` and `q = A * p`

## Original Source

**Repository**: [Polybench-ACC](https://github.com/cavazos-lab/PolyBench-ACC)
**Original Path**: `OpenMP/linear-algebra/kernels/bicg/`

**Polybench Project**: http://polybench.sourceforge.net/

## Algorithm

```
Input: A[NX×NY], r[NX], p[NY]
Output: s[NY], q[NX]

s = A^T * r
q = A * p
```

## References

- **Repository**: https://github.com/cavazos-lab/PolyBench-ACC
- **BiCGStab Algorithm**: van der Vorst, H. A. (1992). "Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG"
