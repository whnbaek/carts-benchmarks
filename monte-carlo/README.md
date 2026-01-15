# Monte Carlo Ensemble Benchmark

This suite contains Monte Carlo-style benchmarks designed to demonstrate the **scalability wall** pattern.

## Scalability Wall Pattern

These benchmarks demonstrate where OpenMP becomes non-viable and CARTS enables continued scaling:

1. **Problem size scales beyond single-node memory capacity**
2. **OpenMP execution fails with out-of-memory errors**
3. **CARTS continues scaling by distributing memory across nodes**

## Benchmarks

### ensemble

Large-scale Monte Carlo simulations where each sample allocates a state matrix inside the parallel loop.

**Key Pattern:**
```c
#pragma omp parallel for reduction(+: global_sum) schedule(dynamic)
for (unsigned s = 0; s < NUM_SAMPLES; s++) {
    // Allocate sample state INSIDE loop (distributed by CARTS)
    double **state = malloc(STATE_DIM * sizeof(double*));
    for (unsigned i = 0; i < STATE_DIM; i++)
        state[i] = malloc(STATE_DIM * sizeof(double));

    // ... computation ...

    global_sum += reduce_state(state);
    free_state(state);
}
```

**Size Configurations:**
- `small`: 1,000 samples × 1024² = ~8 GB
- `medium`: 10,000 samples × 1024² = ~80 GB (exceeds single node)
- `large`: 50,000 samples × 1024² = ~400 GB
- `extralarge`: 100,000 samples × 1024² = ~800 GB

## Building

```bash
# Build all benchmarks with default (small) size
make

# Build with specific size
cd ensemble && make small
cd ensemble && make medium     # Requires multi-node for OpenMP
cd ensemble && make large      # Requires multi-node for OpenMP
cd ensemble && make extralarge # Requires multi-node for OpenMP
```

## Expected Results

| Samples | Memory | OpenMP | CARTS |
|---------|--------|--------|-------|
| 1,000 | ~8 GB | Works | Works |
| 10,000 | ~80 GB | OOM | Works (2+ nodes) |
| 50,000 | ~400 GB | OOM | Works (8+ nodes) |
| 100,000 | ~800 GB | OOM | Works (16+ nodes) |

## Why This Pattern Matters

The Monte Carlo ensemble pattern represents many real-world HPC workloads:
- Uncertainty quantification simulations
- Parameter sweeps
- Ensemble weather forecasting
- Molecular dynamics sampling

In all these cases, the aggregate memory requirement exceeds single-node capacity while individual samples are independent—perfect for CARTS distributed execution.

## References

- Cardosi & Bramas, "Specx: a C++ task-based runtime system for heterogeneous distributed architectures" (2024)
- Valero-Lara et al., "Asynchronous distributed task managers" (2020)
