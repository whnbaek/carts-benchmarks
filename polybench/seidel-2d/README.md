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
- **Gauss-Seidel Method**: Classical iterative solver for linear systems
