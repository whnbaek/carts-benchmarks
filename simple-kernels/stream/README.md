# STREAM - Memory Bandwidth Benchmark

## Description

STREAM benchmark measures sustained memory bandwidth using a simple vector triad operation: `a[i] = b[i] + scalar * c[i]`. This is the industry-standard benchmark for measuring achievable memory bandwidth on different computer architectures.

## Algorithm

The STREAM Triad operation performs:

```
For i in [0, N):
  a[i] = b[i] + scalar * c[i]
```

This operation is repeated multiple times (REPS) to measure sustained bandwidth rather than peak burst performance.

## Memory Bandwidth Testing

STREAM is the de facto standard for memory bandwidth measurement because:
- **Simple operation**: Minimal computation, maximum memory stress
- **Three arrays**: Tests read and write bandwidth simultaneously
- **Streaming access**: Sequential access pattern, no cache reuse
- **Industry standard**: Results comparable across systems and vendors
- **Arithmetic intensity**: Very low (1 FP op per 3 memory operations)

### Why Triad?

STREAM includes multiple operations (Copy, Scale, Add, Triad), but **Triad** is the most important:
- **Copy**: `a[i] = b[i]` - 2 memory operations
- **Scale**: `a[i] = scalar * b[i]` - 2 memory operations, 1 FP operation
- **Add**: `a[i] = b[i] + c[i]` - 3 memory operations
- **Triad**: `a[i] = b[i] + scalar * c[i]` - 3 memory operations, 2 FP operations

Triad is most representative of real scientific applications that combine loads, stores, and arithmetic.

## Problem Sizes

| Size | Configuration | Description |
|------|--------------|-------------|
| **SMALL** | N=1000 | Small problem size (fits in cache) |
| **MEDIUM** | N=10000 | Medium problem size |
| **LARGE** | N=100000 | Large problem size (exceeds cache) |

**Note**: For accurate bandwidth measurement, arrays should exceed last-level cache size. Use N=10M+ for production benchmarking.

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
# Compile directly with GCC (1M elements, 5 repetitions)
gcc -O2 -fopenmp stream.c -o stream -DN=1048576 -DREPS=5

# For accurate bandwidth measurement (10M elements)
gcc -O2 -fopenmp stream.c -o stream -DN=10485760 -DREPS=10

# Run
./stream
# Output: checksum=...
```

## Performance Characteristics

- **Compute**: O(n) floating-point operations per iteration
- **Memory**: O(n) per array, 3 arrays total
- **Bandwidth**: Memory-bandwidth bound (primary metric)
- **Arithmetic Intensity**: ~0.33 FLOP/byte (very low)
- **Parallelism**: Embarrassingly parallel, good load balance

### Interpreting Results

```
Bandwidth (GB/s) = (Bytes transferred) / (Time in seconds) / 10^9

For Triad with N elements:
Bytes = N * sizeof(float) * 3 operations (2 reads + 1 write)
      = N * 4 * 3 = 12N bytes
```

## Use in Computing

STREAM is used for:
- **Hardware procurement**: Vendor comparisons
- **System validation**: Verify memory subsystem performance
- **Performance modeling**: Roofline analysis baseline
- **Compiler testing**: Memory optimization validation
- **Supercomputer rankings**: Memory-bound workload representative

### Real-World Applications with Similar Patterns:
- **Computational Fluid Dynamics**: Explicit time-stepping schemes
- **Weather/Climate**: Grid-based updates
- **Molecular Dynamics**: Force calculations
- **Seismic Processing**: Wave propagation
- **Machine Learning**: Large tensor operations

## CARTS Compatibility

- ✅ No global variables
- ✅ Clean parameter passing
- ✅ OpenMP parallelization
- ✅ Simple streaming memory access
- ✅ Perfect for memory subsystem analysis

## Source Inspiration

- **STREAM Benchmark**: [https://www.cs.virginia.edu/stream/](https://www.cs.virginia.edu/stream/)
- **Author**: Dr. John D. McCalpin (University of Virginia)
- **First Release**: 1991-1995
- **Current Version**: STREAM2 (ongoing updates)

## Historical Context

STREAM was developed in the 1990s when memory bandwidth became a critical bottleneck. It has become the standard because:
- Simple, reproducible, and portable
- Measures sustained rather than peak bandwidth
- Industry-wide adoption and recognition
- Results database for comparison

## References

- McCalpin, J.D. "Memory Bandwidth and Machine Balance in Current High Performance Computers." IEEE TCCA Newsletter, 1995.
- McCalpin, J.D. "STREAM: Sustainable Memory Bandwidth in High Performance Computers." Technical Report, 1991-2007.
- STREAM Official Website: https://www.cs.virginia.edu/stream/

## Citation

```
McCalpin, John D.
"Memory Bandwidth and Machine Balance in Current High Performance Computers."
IEEE Technical Committee on Computer Architecture (TCCA) Newsletter, 1995.
```
