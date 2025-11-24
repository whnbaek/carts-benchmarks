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

| Size | Dimensions | Description |
|------|------------|-------------|
| **MINI** | 32×32 | Minimal size for quick testing |
| **SMALL** | 128×128 | Small problem size |
| **MEDIUM** | 4000×4000 | Standard problem size (default) |
| **LARGE** | 8000×8000 | Large problem size |
| **EXTRALARGE** | 100000×100000 | Extra large problem size |

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
- **Polybench**: http://polybench.sourceforge.net/
