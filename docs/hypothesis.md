# Hypothesis: Impact of Delayed Optimizations on Sequential Kernel Performance

## Background

CARTS (Compiler for ARTS Runtime System) transforms OpenMP parallel code into a task-based execution model using Data Blocks (DBs) for data partitioning. This transformation requires **delaying classical compiler optimizations** such as:

- **CSE (Common Subexpression Elimination)**: Reusing previously computed values
- **DCE (Dead Code Elimination)**: Removing unreachable or unused code
- **Loop Invariant Code Motion**: Moving invariant computations out of loops
- **Other MLIR-level optimizations**: Various canonicalization passes

### Why Are Optimizations Delayed?

DB partitioning in CARTS requires:
1. Splitting data into partitioned blocks for distributed memory
2. Creating separate EDTs (Event-Driven Tasks) for each partition
3. Managing data dependencies between partitions

These transformations must happen **before** certain optimizations can be applied, because:
- CSE might merge expressions that access different DB partitions
- DCE might remove code that appears dead but is needed for DB synchronization
- Loop transformations might interfere with partition boundaries

### The Final -O3 Compilation

After CARTS generates LLVM IR (`.ll` file), we compile it with `clang -O3`. The question is:

> **Can LLVM's -O3 recover all the optimizations that were delayed at the MLIR level?**

## Hypothesis

**H0 (Null)**: LLVM -O3 fully recovers delayed optimizations; CARTS and native OpenMP have equivalent sequential kernel performance.

**H1 (Alternative)**: Delayed MLIR optimizations result in suboptimal LLVM IR that -O3 cannot fully optimize, causing slower sequential kernel execution in CARTS-generated code.

## Experimental Design

### What We Measure

We instrument benchmarks with two levels of timing:

```c
#pragma omp parallel
{
    CARTS_PARALLEL_TIMER_START("kernel_name");    // T_parallel

    #pragma omp task
    {
        CARTS_TASK_TIMER_START("kernel_name:compute");  // T_task

        // Pure sequential computation (kernel work)
        kernel_computation(...);

        CARTS_TASK_TIMER_STOP("kernel_name:compute");
    }
    #pragma omp taskwait

    CARTS_PARALLEL_TIMER_STOP("kernel_name");
}
```

This gives us:
- **T_parallel**: Total time in parallel region (includes task creation overhead)
- **T_task**: Pure sequential computation time (excludes overhead)
- **T_overhead = T_parallel - T_task**: Task scheduling/creation overhead

### Comparison Setup

For each benchmark, we compare:

| Version | Description |
|---------|-------------|
| **Native OpenMP** | Source compiled directly with `clang -O3 -fopenmp` |
| **CARTS-generated** | Source → CARTS pipeline → `.ll` → `clang -O3` |

### Key Metrics

1. **Sequential Efficiency Ratio**:
   ```
   η = T_task(CARTS) / T_task(OpenMP)
   ```
   - η ≈ 1.0 → LLVM recovers optimizations well
   - η > 1.0 → CARTS sequential code is slower (delayed opts hurt)
   - η < 1.0 → CARTS actually faster (unlikely, would indicate different code paths)

2. **Overhead Ratio**:
   ```
   overhead_ratio = T_overhead(CARTS) / T_overhead(OpenMP)
   ```
   - Measures task creation/scheduling overhead difference

3. **Per-Worker Variance**:
   - High variance across workers may indicate load imbalance
   - Compare variance between CARTS and OpenMP

## Expected Outcomes

### Scenario A: Optimizations Recovered (η ≈ 1.0)
- LLVM -O3 successfully optimizes the generated code
- Delayed MLIR optimizations don't matter for final performance
- **Implication**: Current CARTS approach is sound

### Scenario B: Partial Recovery (1.0 < η < 1.5)
- Some optimization opportunities lost at MLIR level
- -O3 recovers most but not all optimizations
- **Implication**: May want to investigate which patterns aren't recovered

### Scenario C: Significant Degradation (η > 1.5)
- Delayed optimizations significantly impact sequential performance
- LLVM cannot recover what MLIR could have optimized
- **Implication**: Need to reconsider optimization ordering in CARTS pipeline

## Output Format

Timer output follows this pattern:
```
parallel.<name>[worker=<id>]: <time>s
task.<name>[worker=<id>]: <time>s
```

Example output:
```
parallel.gemm[worker=0]: 0.001234s
task.gemm:compute[worker=0]: 0.001100s
parallel.gemm[worker=1]: 0.001256s
task.gemm:compute[worker=1]: 0.001120s
kernel.gemm: 0.002500s
checksum: 1.234567e+06
```

## Enabling Instrumentation

Timing is **disabled by default** to avoid overhead in production runs.

To enable timing instrumentation:
```bash
# Compile-time enable
clang -DCARTS_ENABLE_TIMING ...

# Or in source before include
#define CARTS_ENABLE_TIMING
#include "arts/Utils/Benchmarks/CartsBenchmarks.h"
```

## Benchmarks to Instrument

Priority benchmarks for this experiment:
1. **polybench/gemm** - Dense matrix multiplication (compute-bound)
2. **polybench/jacobi2d** - Stencil computation (memory-bound)
3. **polybench/2mm** - Chained matrix operations
4. **kastors-jacobi** - Task-parallel Jacobi iterations

## Analysis Postprocessing

The `benchmark_runner.py` script parses timing output and computes:
- Per-worker statistics (mean, stddev, min, max)
- Aggregated statistics across workers
- Efficiency ratios (CARTS vs OpenMP)
- Overhead breakdown

See `analyze_timing_results()` function for implementation.

## References

- CARTS compiler pipeline: `tools/run/carts-run.cpp`
- Timing macros: `include/arts/Utils/Benchmarks/CartsBenchmarks.h`
- DB partitioning passes: `lib/arts/Passes/CreateDbs.cpp`
- Optimization passes: `lib/arts/Passes/` (CSE, DCE, etc.)
