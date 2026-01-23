# lulesh example analysis

Walk through these steps and fix any problem that you find in the way

---

## Bug Fix: ARTS crash with indirect indices (Fixed)

**Problem:** `lulesh_arts` crashed in `__arts_edt_3` with a null dependency
pointer.

**Root Cause:** dbpartitioning allowed chunked acquires when the index was
derived from a memory load (indirect gather via `nodelist`). The dependency
indexing then referenced depv slots that were never acquired.

**Fix:** Treat indirect indices (values derived from `memref.load`/`llvm.load`)
as non-partitionable by default. This keeps arrays like `x/y/z` coarse even
when offset/size hints exist. Use `--partition-fallback=fine` to allow
element-wise partitioning for non-affine accesses when you want to explore
performance tradeoffs.

**Result:** `carts benchmarks run lulesh --size small` passes (ARTS+OMP
verification succeeds).

---

## Bug Fix: polygeist.subindex Not Traced in CreateDbs (Fixed)

**Problem:** lulesh crashed with "EDT region uses external value that is not a
block argument or dependency" during the CreateDbs pass.

**Root Cause:** `ValueUtils::getUnderlyingValueImpl()` didn't handle
`polygeist::SubIndexOp`. When CreateDbs tried to collect allocations used
inside EDTs, it couldn't trace through `polygeist.subindex` operations to find
the underlying `memref.alloc`. This caused allocations accessed via subindex
to not be converted to datablocks.

**Affected Pattern:** 2D array accesses like `fx_elem[k]` which generate:
```mlir
%alloc = memref.alloc(%size) : memref<?xf64>
%subindex = polygeist.subindex %alloc[%k] : memref<?xf64> -> memref<?xf64>
```

**Fix:** Added handling for `polygeist::SubIndexOp` in `getUnderlyingValueImpl()`
in `/opt/carts/lib/arts/Utils/ValueUtils.cpp`:
```cpp
else if (auto subindex = dyn_cast<polygeist::SubIndexOp>(op))
  return getUnderlyingValueImpl(subindex.getSource(), visited, depth + 1);
```

**Comparison with mixed_access:** The `mixed_access` example worked because it
has 0 `polygeist.subindex` ops - its array accesses don't generate this pattern.

---

## Bug: Local Alloca Inside Parallel Loop (Open)

**Problem:** ARTS transformations fail with "EDT region uses external value that is not a block argument or dependency" for local stack arrays declared inside parallel loop bodies.

**Error Message:**
```
error: 'memref.store' op EDT region uses external value '%255 = "memref.alloca"()
... : () -> memref<3x8xf64>' that is not a block argument or dependency.
```

**Affected Pattern:**
```c
#pragma omp parallel for firstprivate(numElem)
for (Index_t k = 0; k < numElem; ++k) {
    Real_t B[3][8];      // <-- Local alloca (memref<3x8xf64>)
    Real_t x_local[8];   // <-- Local alloca
    Real_t y_local[8];   // <-- Local alloca
    Real_t z_local[8];   // <-- Local alloca

    CollectNodesToElemNodes(x, y, z, nodelist[k], x_local, y_local, z_local);
    CalcElemShapeFunctionDerivatives(x_local, y_local, z_local, B, &determ[k]);
    // ...
}
```

**Root Cause:** When this parallel loop is converted to an EDT:
1. The `memref.alloca` for `B[3][8]` is created outside the EDT region
2. The EDT body tries to use this alloca
3. CARTS validation fails because the alloca isn't passed as a block argument or dependency

**Affected Locations:**
- `lulesh.c:638` inside `IntegrateStressForElems()`
- `lulesh.c:1431` (another `Real_t B[3][8]` inside a parallel loop)

**Status:** Open - requires compiler fix in CreateDbs or EDT outlining pass.

**Expected Fix:** The CARTS transformation should handle local allocas inside parallel loop bodies by either:
1. **Cloning the alloca inside the EDT** - Each task gets its own private copy
2. **Converting to thread-local storage** - Explicit privatization
3. **Hoisting and passing as dependency** - Less efficient but correct

**Workaround:** None currently available. The local arrays must be handled by the compiler.

---

1. **Navigate to the lulesh example directory:**

   ```bash
   cd /Users/randreshg/Documents/carts/external/carts-benchmarks/lulesh
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

   If you need to inspect initialization values, enable the debug prints:
   ```bash
   CARTS_LULESH_DEBUG=1 ./lulesh_arts -s 3 -i 1
   ```
   This prints `e/p/q/v/volo/nodalMass` and `nodelist` values after init.

4. **Concurrency-opt checkpoint**
   ```bash
   carts run lulesh.mlir --concurrency-opt &> lulesh_concurrency_opt.mlir
   ```
   Check that arrays tied to the parallel loop are chunked only when the
   access is direct. For indirect gathers (e.g., `x/y/z` indexed by
   `nodelist`), `arts.partition` should be `none` (coarse) and `db_acquire`
   offsets should be `%c0` by default. To experiment with element-wise fallback:
   ```bash
   carts run lulesh.mlir --concurrency-opt --partition-fallback=fine \
     &> lulesh_concurrency_opt_fine.mlir
   ```

5. **Finally lets carts execute and check**

   ```bash
   carts execute lulesh.c -O3 -DMINI_DATASET -I. -I../common -I../utilities
   artsConfig=arts.cfg ./lulesh_arts -s 3 -i 5
   ```

6. **Run with carts benchmarks and check**

   ```bash
   carts benchmarks run lulesh --size small
   ```
