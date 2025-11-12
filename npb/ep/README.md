# NPB EP - Embarrassingly Parallel Benchmark

## Description

Embarrassingly parallel Monte Carlo computation that generates random numbers and computes statistics. The EP (Embarrassingly Parallel) benchmark tests raw computational power with minimal communication.

## Original Source

**Benchmark Suite**: [NAS Parallel Benchmarks (NPB)](https://www.nas.nasa.gov/software/npb.html)
**Institution**: NASA Advanced Supercomputing (NAS) Division
**License**: NASA Open Source Agreement (NOSA)

**NPB Project**:
- **Official Site**: https://www.nas.nasa.gov/software/npb.html
- **Purpose**: Evaluate performance of parallel supercomputers
- **Designed**: To represent computational fluid dynamics (CFD) workloads
- **Versions**: Multiple implementations (MPI, OpenMP, CUDA, etc.)

## Algorithm

The EP benchmark performs independent random number generation and classification:

```
1. Initialize random number generator state

2. For each of N samples (in parallel):
   - Generate pair of random numbers (x, y)
   - Compute r² = x² + y²
   - Classify into bins based on r value
   - Accumulate statistics

3. Reduce statistics across all threads

4. Report results
```

### Monte Carlo Pattern

This is a classic Monte Carlo computation:
```
For i in [0, N):
  x = random()
  y = random()

  if (x² + y² <= 1.0):
    inside_circle += 1

Estimate π ≈ 4 * (inside_circle / N)
```

## Why "Embarrassingly Parallel"?

**Definition**: A problem is "embarrassingly parallel" when:
- No dependencies between iterations
- No communication needed during computation
- Perfect linear speedup is theoretically possible
- Work can be divided arbitrarily

**EP Characteristics**:
- Each sample is independent
- Only final reduction needed
- Minimal memory access (mostly compute)
- Tests raw CPU computational throughput

## Configuration

```c
#ifndef M
#define M 24      // 2^M samples (default: 16M samples)
#endif
```

Problem sizes:
- **Mini**: M=20 → 1M samples
- **Small**: M=24 → 16M samples
- **Medium**: M=28 → 256M samples
- **Large**: M=32 → 4B samples

## Build

```bash
cd npb/ep/
gcc -O2 -fopenmp -lm ep.c -o ep -DM=24
```

## Usage

```bash
./ep

# Output:
# pi=3.141592654 N=16777216
```

## Use in Computing

Embarrassingly parallel benchmarks are used to:
- **Baseline Performance**: Measure peak computational throughput
- **Scaling Studies**: Test parallel efficiency without communication overhead
- **Monte Carlo**: Template for Monte Carlo simulations
- **Comparison**: Reference point for communication-intensive benchmarks

### Real-World EP Applications:
- **Monte Carlo**: Financial modeling, particle physics
- **Parameter Sweeps**: Running simulations with different parameters
- **Ray Tracing**: Independent pixel/ray computations
- **Machine Learning**: Some hyperparameter searches
- **Cryptography**: Password cracking, hash searches

## CARTS Compatibility

- ✅ No global variables (all local)
- ✅ OpenMP parallel for with reduction
- ✅ Simple, clean implementation
- ✅ Perfect test case for parallel-for lowering
- ✅ Minimal dependencies

## Key Features

- **Minimal communication**: Only reduction at end
- **Compute-intensive**: Tests arithmetic throughput
- **Deterministic RNG**: Reproducible results
- **Simple reduction**: Single reduction variable

## CARTS Testing Focus

### Memory Access Patterns
- **Local computation**: Each thread uses local variables
- **Minimal memory traffic**: Only loop counter and accumulator
- **Cache-friendly**: No shared data structure access

### Dependencies
- **None during loop**: Iterations are independent
- **Reduction at end**: Sum across threads
- **Embarrassingly parallel**: Ideal for parallel-for

### Parallelization Strategy
- **Static scheduling**: Work evenly divided
- **No synchronization**: Until final reduction
- **Perfect load balance**: All iterations equal cost

## Performance Characteristics

- **Compute**: O(N) arithmetic operations
- **Memory**: O(1) per thread
- **Communication**: O(P) for reduction (P = threads)
- **Speedup**: Near-linear (limited only by reduction)
- **Cache**: Minimal cache usage

## Comparison with Other NPB Kernels

| Kernel | Pattern | Communication | Difficulty |
|--------|---------|---------------|------------|
| **EP** | Monte Carlo | None | Trivial |
| **CG** | Conjugate Gradient | High | Medium |
| **MG** | Multigrid | Medium | Medium |
| **FT** | FFT | High (all-to-all) | Hard |
| **IS** | Integer Sort | High (irregular) | Hard |

## NPB Class Sizes

NPB uses class sizes (S, W, A, B, C, D, E):
- **Class S**: Small for development
- **Class W**: Workstation size
- **Class A**: Standard (small)
- **Class B**: Standard (medium)
- **Class C**: Standard (large)
- **Class D**: Very large
- **Class E**: Extreme

This implementation uses custom M parameter instead.

## Historical Context

**NAS Parallel Benchmarks** (1991):
- Developed at NASA Ames Research Center
- Designed to evaluate supercomputers for CFD
- Became de facto standard for parallel benchmarks
- Influenced HPC procurement decisions

**EP Kernel Purpose**:
- Establish performance baseline
- Test overhead of parallel runtime
- Verify scalability without communication bottleneck

## References

- **NPB Official**: https://www.nas.nasa.gov/software/npb.html
- **NPB Technical Report**: Bailey et al., "The NAS Parallel Benchmarks" (1991)
- **GitHub Mirror**: https://github.com/benchmark-subsetting/NPB3.0-omp-C

## Citation

### NAS Parallel Benchmarks
```
Bailey, David H., et al.
"The NAS parallel benchmarks."
The International Journal of Supercomputing Applications 5.3 (1991): 63-73.
```

### NPB Technical Report
```
Bailey, David, et al.
"The NAS Parallel Benchmarks - Summary and Preliminary Results."
NASA Technical Report RNR-94-007, 1994.
```

### OpenMP Version
```
Jin, H., Frumkin, M., and Yan, J.
"The OpenMP implementation of NAS parallel benchmarks and its performance."
NASA Technical Report NAS-99-011, 1999.
```
