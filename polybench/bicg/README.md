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

## Problem Sizes

| Size | Dimensions | Description |
|------|------------|-------------|
| **MINI** | 32×32 | Minimal size for quick testing |
| **SMALL** | 128×128 | Small problem size |
| **MEDIUM** | 1024×1024 | Standard problem size (default) |
| **LARGE** | 2000×2000 | Large problem size |
| **EXTRALARGE** | 4000×4000 | Extra large problem size |

## Building and Running

### Build with CARTS pipeline

```bash
# Build small size (128×128)
make small

# Build medium size (1024×1024) - default
make medium

# Build large size (2000×2000)
make large

# Build all pipeline stages (seq, metadata, parallel, concurrency)
make all
```

### Build individual stages

```bash
# Generate sequential MLIR
make seq

# Collect runtime metadata
make metadata

# Generate parallel MLIR
make parallel

# Run concurrency analysis
make concurrency

# Run optimized concurrency analysis
make concurrency-opt
```

### Clean build artifacts

```bash
make clean
```

## References

- **Repository**: https://github.com/cavazos-lab/PolyBench-ACC
- **BiCGStab Algorithm**: van der Vorst, H. A. (1992). "Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG"
