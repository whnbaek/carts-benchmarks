# fdtd-2d - 2D Finite Difference Time Domain

## Description

2D FDTD solver for electromagnetic field simulation. Time-stepping stencil computation.

## Original Source

**Repository**: [Polybench-ACC](https://github.com/cavazos-lab/PolyBench-ACC)
**Original Path**: `OpenMP/stencils/fdtd-2d/`

**Polybench Project**: http://polybench.sourceforge.net/

## Algorithm

3-point stencil applied iteratively over time steps for electric (ex, ey) and magnetic (hz) fields.

```
Input: ex[NX×NY], ey[NX×NY], hz[NX×NY], TSTEPS
Output: ex, ey, hz (modified after TSTEPS iterations)

For t = 0 to TSTEPS-1:
  Update ey field (3-point stencil)
  Update ex field (3-point stencil)
  Update hz field (5-point stencil)
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
- **FDTD Method**: Yee, K. (1966). "Numerical solution of initial boundary value problems involving Maxwell's equations"
