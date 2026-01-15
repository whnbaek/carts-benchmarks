# lulesh example analysis

Walk through these steps and fix any problem that you find in the way

1. **Navigate to the lulesh example directory:**

   ```bash
   cd ~/Documents/carts/external/carts-benchmarks/lulesh
   ```

2. **Build carts if any changes were made:**

   ```bash
   carts build
   ```

   If there is no `lulesh.mlir` run:

   ```bash
   carts cgeist lulesh.c -DMINI_DATASET -O0 --print-debug-info -S --raise-scf-to-affine -I. -I../common -I../utilities &> lulesh_seq.mlir
   carts run lulesh_seq.mlir --collect-metadata &> lulesh_arts_metadata.mlir
   carts cgeist lulesh.c -DMINI_DATASET -O0 --print-debug-info -S -fopenmp --raise-scf-to-affine -I. -I../common -I../utilities &> lulesh.mlir
   ```

3. **Run the pipeline and stop after any stage**
   Run the pipeline and stop after any stage.

   For example, check canonicalize-memrefs:
   ```bash
   carts run lulesh.mlir --canonicalize-memrefs &> lulesh_canonicalize_memrefs.mlir
   ```
   Check that array-of-arrays are rewritten to explicit memref dimensions:
   ```mlir
   // nodelist: Index_t** -> memref<?x?xi32> (outer = element, inner = 8)
   %nodelist = memref.alloc(%numElem, %c8) : memref<?x?xi32>
   ```

   For example, analyze the create-dbs pipeline:
   ```bash
   carts run lulesh.mlir --create-dbs &> lulesh_create_dbs.mlir
   ```
   Check that `arts.db_alloc` uses outer dims as `sizes[...]` and inner dims
   in `elementSizes[...]`, and that `arts.db_ref` indexes the outer dimension
   before accessing the inner memref.

4. **Concurrency-opt checkpoint**
   ```bash
   carts run lulesh.mlir --concurrency-opt &> lulesh_concurrency_opt.mlir
   ```
   Check that arrays tied to the parallel loop are chunked (non-zero
   `arts.partition = #arts.promotion_mode<chunked>` and `db_acquire` offsets
   are not always 0). If everything is coarse, inspect the access pattern
   and offset hints for the first dynamic index.

5. **Relax partitioning heuristics (dbpartitioning)**
   If allocations are still coarse-grained, relax the heuristics so chunking
   is allowed when offsets/sizes clearly map to the parallel loop.

   Inspect partitioning signals:
   ```bash
   rg -n "arts.db_alloc|arts.db_acquire|arts.partition" lulesh_concurrency_opt.mlir
   ```

   Update the decision rules (see `docs/heuristics/partitioning/partitioning.md`):
   - Treat an allocation as chunkable if at least one acquire has valid
     `offset_hints`/`size_hints` and passes offset validation.
   - Allow coarse acquires to coexist with a chunked allocation (full acquire
     for non-parallel regions), instead of forcing coarse for the entire DB.
   - If metadata for distinct allocations is merged (e.g., helper allocation
     wrappers), include callsite info in the allocation id so heuristics see
     the correct per-array access pattern.

   Re-run `--concurrency-opt` and confirm arrays tied to parallel loops
   get `arts.partition = #arts.promotion_mode<chunked>`.

6. **Optional: print init values**
   The code includes debug hooks for initialization and per-cycle summaries.
   Enable them with:
   ```bash
   CARTS_LULESH_DEBUG=1 artsConfig=arts.cfg ./lulesh_arts -s 3 -i 5
   ```

7. **Finally lets carts execute and check**

   ```bash
   carts execute lulesh.c -O3 -DMINI_DATASET -I. -I../common -I../utilities
   artsConfig=arts.cfg ./lulesh_arts -s 3 -i 5
   ```

8. **Run with carts benchmarks and check**

   ```bash
   carts benchmarks run lulesh --size small
   ```

Notes

- LULESH uses array-of-arrays (e.g., `Index_t **nodelist`, `Real_t **fx_elem`)
  so `canonicalize-memrefs` can materialize explicit memref dimensions.
- `CARTS_QSTOP` can be overridden with `-DCARTS_QSTOP=...`. The Makefile sets
  `-DCARTS_QSTOP=1.0e+30`.
- `carts benchmarks run` uses `SMALL_ARGS`, `MEDIUM_ARGS`, etc from the
  Makefile (e.g., `SMALL_ARGS := -s 10 -i 10`).
