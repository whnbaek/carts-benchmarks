# Convolution-2d example analysis

Walk through these steps and fix any problem that you find in the way

1. **Navigate to the convolution-2d example directory:**

   ```bash
   cd ~/Documents/carts/external/carts-benchmarks/polybench/convolution-2d
   ```

2. **Build carts if any changes were made:**

   ```bash
   carts build
   ```

   If there is no array.mlir run:

   ```bash
      carts cgeist convolution-2d.c -DMINI_DATASET -O0 --print-debug-info -S --raise-scf-to-affine -I. -I../common -I../utilities &> convolution-2d_seq.mlir
      carts run convolution-2d_seq.mlir --collect-metadata &> convolution-2d_arts_metadata.mlir
      carts cgeist convolution-2d.c -DMINI_DATASET -O0 --print-debug-info -S -fopenmp --raise-scf-to-affine -I. -I../common -I../utilities &> convolution-2d.mlir
   ```

3. **Run the pipeline and stop after any stage**
    Run the pipeline and stop after any stage.

   For example, lets analyze the concurrency pipeline
    ```bash
      carts run convolution-2d.mlir --concurrency &> convolution-2d_concurrency.mlir
    ```

    Check the output of the concurrency pipeline.
    ```mlir
    module attributes {...} {
    ...
    func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
        ...
        /// Array A - Coarse grained allocation
        %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>] route(%c0_i32 : i32) sizes[%c1] elementType(f32) elementSizes[%4, %c64] {...} : (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
        /// Array B - Coarse grained allocation
        %guid_9, %ptr_10 = arts.db_alloc[<inout>, <heap>, <write>] route(%c0_i32 : i32) sizes[%c1] elementType(f32) elementSizes[%4, %c64] {...} : (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
        /// Initialize array A
        scf.for %arg2 = %c0 to %c64 step %c1 {
        %7 = arith.index_cast %arg2 : index to i32
        scf.for %arg3 = %c0 to %c64 step %c1 {
            %8 = arith.index_cast %arg3 : index to i32
            %9 = arith.addi %7, %8 : i32
            %10 = arith.sitofp %9 : i32 to f32
            %11 = arith.divf %10, %cst_8 : f32
            /// Since Array A is a coarse grained allocation we use a db_ref with index 0, and then we have a single element access (the 0th element). Remember that the db_ref iterates over the outer dimensions (the sizes).
            %12 = arts.db_ref %ptr[%c0] : memref<?xmemref<memref<?x?xf32>>> -> memref<?x?xf32>
            memref.store %11, %12[%arg2, %arg3] : memref<?x?xf32>
        } {...}
        } {...}
        /// Parallel Work before the arts.for loop. This will me removed later on by the edt pass in the concurrency-opt pipeline.
        %guid_11, %ptr_12 = arts.db_acquire[<in>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<memref<?x?xf32>>>) offsets[%c0] sizes[%c1] -> (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
        %guid_13, %ptr_14 = arts.db_acquire[<out>] (%guid_9 : memref<?xi64>, %ptr_10 : memref<?xmemref<memref<?x?xf32>>>) offsets[%c0] sizes[%c1] -> (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
        arts.edt <parallel> <intranode> route(%c0_i32) (%ptr_12, %ptr_14) : memref<?xmemref<memref<?x?xf32>>>, memref<?xmemref<memref<?x?xf32>>> attributes {workers = #arts.workers<16>} {
        ^bb0(%arg2: memref<?xmemref<memref<?x?xf32>>>, %arg3: memref<?xmemref<memref<?x?xf32>>>):
        arts.db_release(%arg2) : memref<?xmemref<memref<?x?xf32>>>
        arts.db_release(%arg3) : memref<?xmemref<memref<?x?xf32>>>
        }
        /// Epochs are used to create the tasks and the parallel_for
        %5 = arts.epoch {
        scf.for %arg2 = %c0 to %c16 step %c1 {
            ...
            /// If the current worker has work to do, we acquire the input and output dbs and create a task.
            scf.if %13 {
            /// For work
            %guid_15, %ptr_16 = arts.db_acquire[<in>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<memref<?x?xf32>>>) offsets[%c0] sizes[%c1] offset_hints[%7] size_hints[%12] -> (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
            %guid_17, %ptr_18 = arts.db_acquire[<out>] (%guid_9 : memref<?xi64>, %ptr_10 : memref<?xmemref<memref<?x?xf32>>>) offsets[%c0] sizes[%c1] offset_hints[%7] size_hints[%12] -> (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
            arts.edt <task> <intranode> route(%c0_i32) (%ptr_16, %ptr_18) : memref<?xmemref<memref<?x?xf32>>>, memref<?xmemref<memref<?x?xf32>>> {
            ^bb0(%arg3: memref<?xmemref<memref<?x?xf32>>>, %arg4: memref<?xmemref<memref<?x?xf32>>>):
                scf.for %arg5 = %c0 to %12 step %c1 {
                %...
                scf.for %arg6 = %c1 to %c63 step %c1 {
                    ...
                } {...}
                } {...}
                arts.db_release(%arg3) : memref<?xmemref<memref<?x?xf32>>>
                arts.db_release(%arg4) : memref<?xmemref<memref<?x?xf32>>>
            }
            }
        }
        } : i64
        ...
        arts.db_free(%guid_9) : memref<?xi64>
        arts.db_free(%ptr_10) : memref<?xmemref<memref<?x?xf32>>>
        arts.db_free(%guid) : memref<?xi64>
        arts.db_free(%ptr) : memref<?xmemref<memref<?x?xf32>>>
        return %c0_i32 : i32
    }
    ...
    }
    ```
4. **Concurrency-opt checkpoint:**
    ```bash
      carts run convolution-2d.mlir --concurrency-opt &> convolution-2d_concurrency_opt.mlir
    ```
    Check the output of the concurrency optimization pipeline.
    ```mlir
      module attributes {...} {
        ...
        func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
            ...
            /// Notice that this DB allocation is coarse grained - Stencil allocation.
            %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>] route(%c0_i32 : i32) sizes[%c1] elementType(f32) elementSizes[%4, %c64] {...} : (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
            /// Notice that this DB allocation is coarse grained - Stencil allocation.
            %guid_9, %ptr_10 = arts.db_alloc[<inout>, <heap>, <write>] route(%c0_i32 : i32) sizes[%4] elementType(f32) elementSizes[%c64] {...} : (memref<?xi64>, memref<?xmemref<memref<?xf32>>>)
            scf.for %arg2 = %c0 to %c64 step %c1 {
            %7 = arith.index_cast %arg2 : index to i32
            scf.for %arg3 = %c0 to %c64 step %c1 {
                %8 = arith.index_cast %arg3 : index to i32
                %9 = arith.addi %7, %8 : i32
                %10 = arith.sitofp %9 : i32 to f32
                %11 = arith.divf %10, %cst_8 : f32
                %12 = arts.db_ref %ptr[%c0] : memref<?xmemref<memref<?x?xf32>>> -> memref<?x?xf32>
                memref.store %11, %12[%arg2, %arg3] : memref<?x?xf32>
            } {...}
            } {...}
            %5 = arts.epoch {
            ...
            scf.if %13 {
                /// Input DB Acquire - I wouldve expected the alloc to be IN as well...
                %guid_11, %ptr_12 = arts.db_acquire[<in>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<memref<?x?xf32>>>) offsets[%c0] sizes[%c1] offset_hints[%7] size_hints[%12] {arts.twin_diff = false} -> (memref<?xi64>, memref<?xmemref<memref<?x?xf32>>>)
                /// Output DB Acquire - Fine grained! This is expected...
                %guid_13, %ptr_14 = arts.db_acquire[<out>] (%guid_9 : memref<?xi64>, %ptr_10 : memref<?xmemref<memref<?xf32>>>) offsets[%7] sizes[%12] offset_hints[%7] size_hints[%12] {arts.twin_diff = false} -> (memref<?xi64>, memref<?xmemref<memref<?xf32>>>)
                arts.edt <task> <intranode> route(%c0_i32) (%ptr_12, %ptr_14) : memref<?xmemref<memref<?x?xf32>>>, memref<?xmemref<memref<?xf32>>> attributes {arts.id = 35 : i64} {
                ^bb0(%arg3: memref<?xmemref<memref<?x?xf32>>>, %arg4: memref<?xmemref<memref<?xf32>>>):
                    scf.for %arg5 = %c0 to %12 step %c1 {
                    ...
                    scf.for %arg6 = %c1 to %c63 step %c1 {
                        /// Coarse grained, then dbref to get the inner dimension.
                        %22 = arts.db_ref %arg3[%c0] : memref<?xmemref<memref<?x?xf32>>> -> memref<?x?xf32>
                        %23 = memref.load %22[%18, %21] : memref<?x?xf32>
                        ...
                        /// Fine grained, then dbref to get the inner dimension.
                        %61 = arts.db_ref %arg4[%60] : memref<?xmemref<memref<?xf32>>> -> memref<?xf32>
                        memref.store %59, %61[%arg6] : memref<?xf32>
                    } {...}
                    } {...}
                    arts.db_release(%arg3) : memref<?xmemref<memref<?x?xf32>>>
                    arts.db_release(%arg4) : memref<?xmemref<memref<?xf32>>>
                }
                }
            } {...}
            } : i64
            ...
            arts.db_free(%guid) : memref<?xi64>
            arts.db_free(%ptr) : memref<?xmemref<memref<?x?xf32>>>
            arts.db_free(%guid_9) : memref<?xi64>
            arts.db_free(%ptr_10) : memref<?xmemref<memref<?xf32>>>
            return %c0_i32 : i32
        }
    }
    ```
    Lets investigate if this is OK, right now we are having some segmentation faults when running the program.

4. **Finally lets carts execute and check**
```bash
    carts execute convolution-2d.c -O3 -DMINI_DATASET -I. -I../common -I../utilities
   ./convolution-2d_arts
```
