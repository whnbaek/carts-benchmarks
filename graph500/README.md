# Graph500-Inspired Benchmarks

This suite contains Graph500-inspired benchmarks adapted for CARTS to demonstrate the **scalability wall** pattern.

## Scalability Wall Pattern

These benchmarks demonstrate where OpenMP becomes non-viable and CARTS enables continued scaling:

1. **Problem size scales beyond single-node memory capacity**
2. **OpenMP execution fails with out-of-memory errors**
3. **CARTS continues scaling by distributing memory across nodes**

## Benchmarks

### graph-gen

Parallel graph generation where each vertex's adjacency list is allocated inside the parallel loop.

**Key Pattern:**
```c
#pragma omp parallel for schedule(dynamic) reduction(+: total_edges)
for (uint64_t v = 0; v < num_vertices; v++) {
    // Allocate INSIDE loop - distributed by CARTS
    adj_list[v] = malloc(edges_per_vertex * sizeof(uint64_t));
    // ... generate edges ...
}
```

**Size Configurations:**
- `small`: Scale 20 (2^20 = 1M vertices, ~130 MB)
- `medium`: Scale 24 (2^24 = 16M vertices, ~2 GB)
- `large`: Scale 26 (2^26 = 64M vertices, ~8 GB)
- `extralarge`: Scale 28 (2^28 = 256M vertices, ~32 GB) - exceeds single-node memory

## Building

```bash
# Build all benchmarks with default (small) size
make

# Build with specific size
cd graph-gen && make small
cd graph-gen && make large
cd graph-gen && make extralarge  # Requires multi-node for OpenMP
```

## Expected Results

| Scale | Memory | OpenMP | CARTS |
|-------|--------|--------|-------|
| 20 | ~130 MB | Works | Works |
| 24 | ~2 GB | Works | Works |
| 26 | ~8 GB | Works (slow) | Works |
| 28 | ~32 GB | OOM | Works (distributed) |
| 32 | ~1 TB | OOM | Works (distributed) |

## References

- [Graph500 Benchmark Specification](https://graph500.org)
- Thoman et al., "A taxonomy of task-based parallel programming technologies for HPC" (2018)
