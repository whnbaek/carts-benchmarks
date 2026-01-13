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

   If there is no array.mlir run:

   ```bash
      carts cgeist lulesh.c -DMINI_DATASET -O0 --print-debug-info -S --raise-scf-to-affine -I. -I../common -I../utilities > lulesh_seq.mlir
      carts run lulesh_seq.mlir --collect-metadata > lulesh_arts_metadata.mlir
      carts cgeist lulesh.c -DMINI_DATASET -O0 --print-debug-info -S -fopenmp --raise-scf-to-affine -I. -I../common -I../utilities > lulesh.mlir
   ```

3. **CanonicalizeMemrefs checkpoint:**
    ```bash
      carts run lulesh.mlir --canonicalize-memrefs &> lulesh_canonicalize_memrefs.mlir
    ```
    Are all the memrefs uses  canonicalized???
    ultrathink

4. **ConvertOpenMPtoARTS checkpoint:**
    ```bash
      carts run lulesh.mlir --openmp-to-arts &> lulesh_convert_openmp_to_arts.mlir
    ```

5. **CreateDbs checkpoint:**
    ```bash
      carts run lulesh.mlir --create-dbs &> lulesh_create_dbs.mlir
    ```
    Analyze all create dbs... Check if there is any access to a memref within an EDT that is not through a datablock.... Be very thorough here.

6. **Concurrency-opt checkpoint:**
    ```bash
      carts run lulesh.mlir --concurrency-opt &> lulesh_concurrency_opt.mlir
    ```
5. **Finally lets carts execute and check**
```bash
    carts execute lulesh.c -O3 -DMINI_DATASET -I. -I../common -I../utilities
   ./lulesh_arts
```

6. **Run it with carts example and check**

```bash
    carts benchmarks run polybench/2mm
```