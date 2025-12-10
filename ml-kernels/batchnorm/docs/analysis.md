# batchnorm example analysis

Walk through these steps and fix any problem that you find in the way

1. **Navigate to the batchnorm example directory:**

   ```bash
   cd ~/Documents/carts/external/carts-benchmarks/ml-kernels/batchnorm
   ```

2. **Build carts if any changes were made:**

   ```bash
   carts build
   ```

   If there is no batchnorm.mlir run:

   ```bash
      carts cgeist batchnorm.c -DMINI_DATASET -O0 --print-debug-info -S --raise-scf-to-affine -I. -I../common -I../utilities &> batchnorm_seq.mlir
      carts run batchnorm_seq.mlir --collect-metadata &> batchnorm_arts_metadata.mlir
      carts cgeist batchnorm.c -DMINI_DATASET -O0 --print-debug-info -S -fopenmp --raise-scf-to-affine -I. -I../common -I../utilities > batchnorm.mlir
   ```

3. **Run the pipeline and stop after any stage**
    Run the pipeline and stop after any stage.

   For example, lets analyze the create-dbs pipeline
    ```bash
      carts run batchnorm.mlir --create-dbs --debug-only=canonicalize-memrefs &> batchnorm_create-dbs.mlir
    ```
   Lets investigate why the canonicalize-memrefs pass is failing. Check if the code has something we do not provide support for. Remember that the canonicalize-memrefs should be able to canonicalize memrefs of N dimensions.

4. **Finally lets carts execute and check**
```bash
    carts execute batchnorm.c -O3 -DMINI_DATASET -I. -I../common -I../utilities
   ./batchnorm_arts
```
