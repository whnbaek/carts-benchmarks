# AXPY - Vector Scale and Add

## Description

AXPY implements the BLAS Level 1 operation `y = alpha * x + y`, where `x` and `y` are vectors and `alpha` is a scalar. This is one of the most fundamental operations in linear algebra and serves as a building block for more complex algorithms.

## Algorithm

The AXPY operation performs element-wise computation:

```
For i in [0, LENGTH):
  y[i] = alpha * x[i] + y[i]
```

This operation is trivially parallelizable with OpenMP `parallel for` since each iteration is independent.

## BLAS Level 1 Operation

AXPY stands for "A times X Plus Y" and is a standard BLAS (Basic Linear Algebra Subprograms) Level 1 operation:
- **Level 1**: Vector-vector operations (O(n) work, O(n) data)
- **Level 2**: Matrix-vector operations (O(n²) work, O(n²) data)
- **Level 3**: Matrix-matrix operations (O(n³) work, O(n²) data)

AXPY is used in:
- Iterative solvers (e.g., conjugate gradient)
- Neural network backpropagation
- Physics simulations
- Any algorithm requiring vector updates

## Problem Sizes

| Size | Configuration | Description |
|------|--------------|-------------|
| **SMALL** | N=1000 | Small problem size (fast testing) |
| **MEDIUM** | N=10000 | Medium problem size (default) |
| **LARGE** | N=100000 | Large problem size |

**Note**: The source code uses `LENGTH` instead of `N`. The build system maps N to LENGTH.

## Building and Running

### Build with CARTS pipeline

```bash
# Build small size
make small

# Build medium size (default)
make medium

# Build large size
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
gcc -O2 -fopenmp axpy.c -o axpy -DLENGTH=1048576 -DALPHA=1.2345f

# Run
./axpy
# Output: axpy checksum=...
```

## Performance Characteristics

- **Compute**: O(n) floating-point operations
- **Memory**: O(n) memory footprint (2 vectors)
- **Bandwidth**: Memory-bandwidth bound (low arithmetic intensity)
- **Parallelism**: Embarrassingly parallel, perfect load balance
- **Cache**: Streaming access pattern, good cache utilization

## CARTS Compatibility

- ✅ No global variables
- ✅ Clean parameter passing
- ✅ OpenMP parallelization
- ✅ Simple memory access patterns
- ✅ Perfect for testing parallel-for lowering

## Source Inspiration

- **BLAS Reference**: [https://www.netlib.org/blas/](https://www.netlib.org/blas/)
- **Operation**: SAXPY (single precision) / DAXPY (double precision)
- **Standard**: Defined in the BLAS standard since 1979

## References

- Lawson, C.L., et al. "Basic Linear Algebra Subprograms for Fortran Usage." ACM TOMS, 1979.
- Dongarra, J.J., et al. "A Set of Level 3 Basic Linear Algebra Subprograms." ACM TOMS, 1990.
