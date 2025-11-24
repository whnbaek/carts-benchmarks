# PRK Stencil - Configurable 2D Stencil

## Description

2D stencil computation with compile-time configurable radius and patterns (star or compact).

## Original Source

**Repository**: [Parallel Research Kernels (PRK)](https://github.com/ParRes/Kernels)
**Original Path**: `OPENMP/Stencil/`
**License**: BSD 3-Clause (see repository)

**PRK Project**:
- **Repository**: https://github.com/ParRes/Kernels
- **Official Site**: https://github.com/ParRes/Kernels
- **Purpose**: Suite of simple, portable kernels for parallel programming research

## Authors

- Rob Van der Wijngaart (Intel Corporation)
- Jeff Hammond
- And contributors

## Algorithm

Applies a stencil pattern to a 2D grid:
```
For each interior point (i, j):
  OUT[i][j] = Σ WEIGHT[di][dj] * IN[i+di][j+dj]
              for (di, dj) in stencil pattern
```

Stencil patterns:
- **Star**: 4-point (radius 1) or more
- **Compact**: Includes diagonals

## Compile-Time Configuration

```c
#define RADIUS 2      // Stencil radius
#define STAR          // Use star pattern (vs compact)
#define DOUBLE        // Use double precision (vs float)
```

## Command-Line Arguments

```bash
./stencil <# threads> <# iterations> <array dimension>
```

## Problem Sizes

This benchmark uses runtime arguments for problem size configuration:

| Size | Configuration | Description |
|------|--------------|-------------|
| **SMALL** | `./prk-stencil.exe 4 100 1000` | 1000×1000 grid |
| **MEDIUM** | `./prk-stencil.exe 4 100 5000` | 5000×5000 grid |
| **LARGE** | `./prk-stencil.exe 4 100 10000` | 10000×10000 grid |

**Note**: Problem size is specified at runtime, not compile-time.

## Building and Running

### Build with CARTS pipeline

```bash
# Build all pipeline stages
make all

# The small/medium/large targets provide usage hints
make small   # Shows command for small problem
make medium  # Shows command for medium problem
make large   # Shows command for large problem
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

### Run with different problem sizes

```bash
# After building, run with command-line arguments:
# 4 threads, 100 iterations, 1000×1000 grid
./prk-stencil.exe 4 100 1000

# 8 threads, 200 iterations, 5000×5000 grid
./prk-stencil.exe 8 200 5000
```

### Clean build artifacts

```bash
make clean

# Also remove generated loop body files
make veryclean
```

## Use in Computing

Stencil computations are fundamental in:
- Partial differential equation solvers
- Image processing
- Computational fluid dynamics
- Climate modeling
- Seismic wave propagation (similar to SW4lite in CARTS)

## CARTS Compatibility

- ✅ No global variables
- ✅ Clean parameter passing
- ✅ OpenMP parallel for
- ✅ Macro-based array indexing (like sw4lite_rhs4sg_revNW)
- ✅ Uses restrict qualifiers

## Key Features

- **Simple, clean implementation**: Educational focus
- **Configurable stencil radius**: Test different dependency patterns
- **Multiple data types**: Float or double
- **Performance measurement**: Built-in timing

## References

- **Repository**: https://github.com/ParRes/Kernels
- **PRK Paper**: Van der Wijngaart, R.F., and Mattson, T.G. "The Parallel Research Kernels." HPCC, 2014.
- **License**: BSD 3-Clause License

## Citation

```
Van der Wijngaart, Rob F., and Timothy G. Mattson.
"The Parallel Research Kernels."
High Performance Computing and Communications (HPCC), 2014.
```
