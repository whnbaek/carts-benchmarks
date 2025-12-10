# activations example analysis

Walk through these steps and fix any problem that you find in the way

1. **Navigate to the activations example directory:**

   ```bash
   cd ~/Documents/carts/external/carts-benchmarks/ml-kernels/activations
   ```

2. **Build carts if any changes were made:**

   ```bash
   carts build
   ```

   If there is no array.mlir run:

   ```bash
      carts cgeist activations.c -DMINI_DATASET -O0 --print-debug-info -S --raise-scf-to-affine -I. -I../common -I../utilities &> activations_seq.mlir
      carts run activations_seq.mlir --collect-metadata &> activations_arts_metadata.mlir
      carts cgeist activations.c -DMINI_DATASET -O0 --print-debug-info -S -fopenmp --raise-scf-to-affine -I. -I../common -I../utilities > activations.mlir
   ```

3. **Run the pipeline and stop after any stage**
    Run the pipeline and stop after any stage.

   For example, lets analyze the create-dbs pipeline
    ```bash
      carts run activations.mlir --create-dbs &> activations_create-dbs.mlir
    ```

    Analyze the comments within the summarized output.
    ```mlir
   module attributes {...} {
   ...
   func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
      ...
      /// Check that all the DbAllocOps respected the control deps in the original code. In the .c code we have
      /// row dependencies
      %guid, %ptr = arts.db_alloc[<in>, <heap>, <read>] route(%c0_i32 : i32) sizes[%13] elementType(f64) elementSizes[%c100] {...} : (memref<?xi64>, memref<?xmemref<memref<?xf64>>>)
      %guid_4, %ptr_5 = arts.db_alloc[<inout>, <heap>, <write>] route(%c0_i32 : i32) sizes[%13] elementType(f64) elementSizes[%c100] {...} : (memref<?xi64>, memref<?xmemref<memref<?xf64>>>)
      %guid_6, %ptr_7 = arts.db_alloc[<inout>, <heap>, <write>] route(%c0_i32 : i32) sizes[%13] elementType(f64) elementSizes[%c100] {...} : (memref<?xi64>, memref<?xmemref<memref<?xf64>>>)
      %alloc = memref.alloc(%13) : memref<?x100xf64>
      scf.for %arg0 = %c0 to %c100 step %c1 {
         scf.for %arg1 = %c0 to %c100 step %c1 {
         /// Notice that we dbref the outer dimension of the db and then store the value to the inner dimension
         %31 = arts.db_ref %ptr[%arg0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
         memref.store %cst_3, %31[%arg1] : memref<?xf64>
         %32 = arts.db_ref %ptr_5[%arg0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
         memref.store %cst_3, %32[%arg1] : memref<?xf64>
         %33 = arts.db_ref %ptr_7[%arg0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
         memref.store %cst_3, %33[%arg1] : memref<?xf64>
         memref.store %cst_3, %alloc[%arg0, %arg1] : memref<?x100xf64>
         } {...}
      } {...}
      ...
      %17 = arts.epoch {
         scf.for %arg0 = %c1 to %c11 step %c1 {
         scf.for %arg1 = %c0 to %c100 step %c1 {
            /// Acquire unew[i] - we acquired the whole row unew[i] and added the index %arg1 to the indices
            %guid_8, %ptr_9 = arts.db_acquire[<in>] (%guid_6 : memref<?xi64>, %ptr_7 : memref<?xmemref<memref<?xf64>>>) indices[%arg1] offsets[%c0] sizes[%c1] {arts.twin_diff = false} -> (memref<?xi64>, memref<?xmemref<memref<?xf64>>>)
            /// Acquire u[i] - we acquired the whole row u[i] and added the index %arg1 to the indices
            %guid_10, %ptr_11 = arts.db_acquire[<out>] (%guid_4 : memref<?xi64>, %ptr_5 : memref<?xmemref<memref<?xf64>>>) indices[%arg1] offsets[%c0] sizes[%c1] {arts.twin_diff = false} -> (memref<?xi64>, memref<?xmemref<memref<?xf64>>>)
            /// EDT that computes u[i] = unew[i]
            arts.edt <task> <intranode> route(%c0_i32) (%ptr_9, %ptr_11) : memref<?xmemref<memref<?xf64>>>, memref<?xmemref<memref<?xf64>>> {
            ^bb0(%arg2: memref<?xmemref<memref<?xf64>>>, %arg3: memref<?xmemref<memref<?xf64>>>):
               scf.for %arg4 = %c0 to %c100 step %c1 {
               /// Load unew[i][j]
               %31 = arts.db_ref %arg2[%c0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
               %32 = memref.load %31[%arg4] : memref<?xf64>
               /// Load u[i][j]
               %33 = arts.db_ref %arg3[%c0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
               /// Store u[i][j] = unew[i][j]
               memref.store %32, %33[%arg4] : memref<?xf64>
               } {...}
               arts.db_release(%arg2) : memref<?xmemref<memref<?xf64>>>
               arts.db_release(%arg3) : memref<?xmemref<memref<?xf64>>>
            }
         } {...}
         scf.for %arg1 = %c0 to %c100 step %c1 {
            ...
            /// IN: Acquire f[i] - we acquired the whole row f[i] and added the index %arg1 to the indices
            %guid_8, %ptr_9 = arts.db_acquire[<in>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<memref<?xf64>>>) indices[%arg1] offsets[%c0] sizes[%c1] {arts.twin_diff = false} -> (memref<?xi64>, memref<?xmemref<memref<?xf64>>>)
            /// OUT: Acquire unew[i] - we acquired the whole row unew[i] and added the index %arg1 to the indices
            %guid_10, %ptr_11 = arts.db_acquire[<out>] (%guid_6 : memref<?xi64>, %ptr_7 : memref<?xmemref<memref<?xf64>>>) indices[%arg1] offsets[%c0] sizes[%c1] {arts.twin_diff = false} -> (memref<?xi64>, memref<?xmemref<memref<?xf64>>>)
            /// IN: Acquire u[i-1] - we acquired the whole row u[i-1] and added the index %32 to the indices
            %guid_12, %ptr_13 = arts.db_acquire[<in>] (%guid_4 : memref<?xi64>, %ptr_5 : memref<?xmemref<memref<?xf64>>>) indices[%32] offsets[%c0] sizes[%c1] {arts.twin_diff = false} -> (memref<?xi64>, memref<?xmemref<memref<?xf64>>>)
            /// IN: Acquire u[i] - we acquired the whole row u[i] and added the index %arg1 to the indices
            %guid_14, %ptr_15 = arts.db_acquire[<in>] (%guid_4 : memref<?xi64>, %ptr_5 : memref<?xmemref<memref<?xf64>>>) indices[%arg1] offsets[%c0] sizes[%c1] {arts.twin_diff = false} -> (memref<?xi64>, memref<?xmemref<memref<?xf64>>>)
            /// IN: Acquire u[i+1] - we acquired the whole row u[i+1] and added the index %33 to the indices
            %guid_16, %ptr_17 = arts.db_acquire[<in>] (%guid_4 : memref<?xi64>, %ptr_5 : memref<?xmemref<memref<?xf64>>>) indices[%33] offsets[%c0] sizes[%c1] {arts.twin_diff = false} -> (memref<?xi64>, memref<?xmemref<memref<?xf64>>>)
            /// EDT that computes u[i][j] = 0.25 * (u[i-1][j] + u[i][j+1] + u[i][j-1] + u[i+1][j] + f[i][j] * dx * dy)
            arts.edt <task> <intranode> route(%c0_i32) (%ptr_9, %ptr_11, %ptr_13, %ptr_15, %ptr_17) : memref<?xmemref<memref<?xf64>>>, memref<?xmemref<memref<?xf64>>>, memref<?xmemref<memref<?xf64>>>, memref<?xmemref<memref<?xf64>>>, memref<?xmemref<memref<?xf64>>> {
            ^bb0(%arg2: memref<?xmemref<memref<?xf64>>>, %arg3: memref<?xmemref<memref<?xf64>>>, %arg4: memref<?xmemref<memref<?xf64>>>, %arg5: memref<?xmemref<memref<?xf64>>>, %arg6: memref<?xmemref<memref<?xf64>>>):
               scf.for %arg7 = %c0 to %c100 step %c1 {
               %35 = arith.index_cast %arg7 : index to i32
               %36 = arith.cmpi eq, %arg1, %c0 : index
               %37 = scf.if %36 -> (i1) {
                  scf.yield %true : i1
               } else {
                  %42 = arith.cmpi eq, %35, %c0_i32 : i32
                  scf.yield %42 : i1
               }
               %38 = arith.xori %37, %true : i1
               %39 = arith.andi %38, %34 : i1
               %40 = arith.ori %37, %39 : i1
               %41 = scf.if %40 -> (i1) {
                  scf.yield %true : i1
               } else {
                  %42 = arith.cmpi eq, %35, %c99_i32 : i32
                  scf.yield %42 : i1
               }
               scf.if %41 {
                  /// Load f[i][j]
                  %42 = arts.db_ref %arg2[%c0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
                  %43 = memref.load %42[%arg7] : memref<?xf64>
                  /// Load unew[i][j]
                  %44 = arts.db_ref %arg3[%c0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
                  /// Store unew[i][j] = f[i][j]
                  memref.store %43, %44[%arg7] : memref<?xf64>
               } else {
                  /// Load u[i-1][j]
                  %42 = arts.db_ref %arg4[%c0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
                  /// Load u[i][j+1]
                  %43 = memref.load %42[%arg7] : memref<?xf64>
                  /// Load u[i][j-1]
                  %44 = arith.addi %arg7, %c1 : index
                  /// Load u[i+1][j]
                  %45 = arts.db_ref %arg5[%c0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
                  %46 = memref.load %45[%44] : memref<?xf64>
                  %47 = arith.addf %43, %46 : f64
                  %48 = arith.addi %arg7, %c-1 : index
                  %49 = memref.load %45[%48] : memref<?xf64>
                  %50 = arith.addf %47, %49 : f64
                  /// Load f[i][j]
                  %51 = arts.db_ref %arg6[%c0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
                  %52 = memref.load %51[%arg7] : memref<?xf64>
                  %53 = arith.addf %50, %52 : f64
                  /// Load unew[i][j]
                  %54 = arts.db_ref %arg2[%c0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
                  %55 = memref.load %54[%arg7] : memref<?xf64>
                  %56 = arith.mulf %55, %cst_0 : f64
                  %57 = arith.mulf %56, %cst_0 : f64
                  %58 = arith.addf %53, %57 : f64
                  %59 = arith.mulf %58, %cst : f64
                  /// Store unew[i][j] = 0.25 * (u[i-1][j] + u[i][j+1] + u[i][j-1] + u[i+1][j] + f[i][j] * dx * dy)
                  %60 = arts.db_ref %arg3[%c0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
                  memref.store %59, %60[%arg7] : memref<?xf64>
               }
               } {...}
               arts.db_release(%arg2) : memref<?xmemref<memref<?xf64>>>
               arts.db_release(%arg3) : memref<?xmemref<memref<?xf64>>>
               arts.db_release(%arg4) : memref<?xmemref<memref<?xf64>>>
               arts.db_release(%arg5) : memref<?xmemref<memref<?xf64>>>
               arts.db_release(%arg6) : memref<?xmemref<memref<?xf64>>>
            }
         } {...}
         } {...}
      } : i64
      scf.for %arg0 = %c0 to %c100 step %c1 {
         scf.for %arg1 = %c0 to %c100 step %c1 {
         %31 = arts.db_ref %ptr_7[%arg0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
         %32 = memref.load %31[%arg1] : memref<?xf64>
         memref.store %32, %alloc[%arg0, %arg1] : memref<?x100xf64>
         } {...}
      } {...}
      scf.for %arg0 = %c0 to %c100 step %c1 {
         scf.for %arg1 = %c0 to %c100 step %c1 {
         %31 = arts.db_ref %ptr_5[%arg0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
         memref.store %cst_3, %31[%arg1] : memref<?xf64>
         %32 = arts.db_ref %ptr_7[%arg0] : memref<?xmemref<memref<?xf64>>> -> memref<?xf64>
         memref.store %cst_3, %32[%arg1] : memref<?xf64>
         } {...}
      } {...}
      ...
      return %30 : i32
   }
   }
    ```

4. **Finally lets carts execute and check**
```bash
    carts execute activations.c -O3 -DMINI_DATASET -I. -I../common -I../utilities
   ./activations_arts
```
