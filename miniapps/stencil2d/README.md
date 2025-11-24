# Stencil2D - 2D Five-Point Stencil

## Description

A 2D five-point stencil computation that iteratively updates grid points based on their neighbors. This miniapp represents a fundamental computational pattern found in many scientific applications including heat diffusion, wave propagation, and image processing.

## Algorithm

The stencil applies a weighted average of each point and its four neighbors (north, south, east, west):

```
For each timestep t in [0, TSTEPS):
  For each interior point (i, j):
    out[i,j] = 0.2 * (in[i,j] + in[i-1,j] + in[i+1,j] +
                                 in[i,j-1] + in[i,j+1])
  Swap in and out arrays
```

### Stencil Pattern

```
        N (i-1, j)
        |
W (i,j-1) - C (i,j) - E (i,j+1)
        |
        S (i+1, j)
```

Each cell is updated using the weighted average of itself (C) and its four neighbors (N, S, E, W).

## Use in Scientific Computing

Five-point stencils are fundamental in:
- **Heat Equation**: Modeling thermal diffusion
- **Laplace/Poisson Equations**: Electrostatics, fluid pressure
- **Wave Equations**: Seismic modeling, acoustics
- **Image Processing**: Smoothing, edge detection
- **Finite Difference Methods**: Discretizing partial differential equations
- **Computational Fluid Dynamics**: Pressure solvers, diffusion

### Why Stencils Matter

Stencil computations:
- Represent 30-40% of HPC workloads
- Challenge memory hierarchies (spatial locality)
- Test compiler optimization (loop tiling, vectorization)
- Exhibit diverse parallelism patterns

## Problem Sizes

| Size | Configuration | Description |
|------|--------------|-------------|
| **SMALL** | N=1000, TSTEPS=50 | Small grid (1M points) |
| **MEDIUM** | N=5000, TSTEPS=50 | Medium grid (25M points) - default |
| **LARGE** | N=10000, TSTEPS=50 | Large grid (100M points) |

**Memory footprint**: 2 × N × N × sizeof(float) bytes (two grids for double-buffering)

## Building and Running

### Build with CARTS pipeline

```bash
# Build small size (N=1000)
make small

# Build medium size (N=5000) - default
make medium

# Build large size (N=10000)
make large

# Build all pipeline stages
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

### Direct compilation (without CARTS pipeline)

```bash
# Compile directly with GCC
gcc -O2 -fopenmp stencil2d.c -o stencil2d -DN=1024 -DTSTEPS=50

# Run
./stencil2d
# Output: checksum=...
```

## Performance Characteristics

- **Compute**: O(N² × TSTEPS) floating-point operations
- **Memory**: O(N²) memory footprint (2 grids)
- **Bandwidth**: Memory-bandwidth bound for large N
- **Arithmetic Intensity**: Low (~0.4 FLOP/byte)
- **Parallelism**: Row-level parallelism with OpenMP
- **Dependencies**: Each timestep depends on previous timestep

### Memory Access Pattern

- **Spatial locality**: Each point accesses its neighbors
- **Temporal locality**: Limited (only via time-stepping)
- **Cache behavior**: Sensitive to grid size and cache capacity
- **Optimal tiling**: Benefits from blocking/tiling optimizations

## Implementation Details

### Boundary Conditions

- **Dirichlet boundaries**: Fixed values at grid edges
- **Interior updates**: Only points with all four neighbors updated
- **Grid range**: Updated region is [1, N-1) in both dimensions

### Double Buffering

The code uses two grids (A and B) that are swapped each timestep:
```
timestep 0: read from A, write to B
timestep 1: read from B, write to A
...
```

This avoids read-after-write hazards and enables parallelization.

## CARTS Compatibility

- ✅ No global variables
- ✅ Clean parameter passing
- ✅ OpenMP parallelization
- ✅ Simple memory access patterns
- ✅ Representative of real applications
- ✅ Good test for loop tiling and locality optimization

## Source Inspiration

Inspired by stencil kernels from:
- **Parallel Research Kernels (PRK)**: [https://github.com/ParRes/Kernels](https://github.com/ParRes/Kernels)
- **DOE Miniapps**: Various DOE proxy applications
- **PolyBench**: Heat-2D and Jacobi-2D benchmarks

## Related Benchmarks

| Benchmark | Pattern | Difference |
|-----------|---------|------------|
| **stencil2d** (this) | 5-point, explicit | Simple, fixed timesteps |
| **PRK Stencil** | Configurable radius | More complex stencil patterns |
| **PolyBench Jacobi-2D** | Jacobi iteration | Similar but different weights |
| **SW4lite** | 3D seismic | Production stencil code |

## Advanced Topics

### Optimization Opportunities

- **Loop tiling**: Improve cache reuse
- **Vectorization**: SIMD for inner loop
- **Fusion**: Combine multiple stencils
- **Temporal blocking**: Reduce memory traffic
- **GPU offload**: Excellent GPU candidate

### Roofline Analysis

This kernel is typically memory-bound:
```
Arithmetic Intensity = 5 FLOPs / (5 reads + 1 write) × 4 bytes
                     ≈ 0.42 FLOP/byte (very low)
```

Performance limited by memory bandwidth, not compute throughput.

## References

- Datta, K., et al. "Stencil computation optimization and auto-tuning on state-of-the-art multicore architectures." SC '08.
- Christen, M., et al. "PATUS: A code generation and autotuning framework for parallel iterative stencil computations on modern microarchitectures." IPDPS 2011.
- ParRes Kernels: https://github.com/ParRes/Kernels
