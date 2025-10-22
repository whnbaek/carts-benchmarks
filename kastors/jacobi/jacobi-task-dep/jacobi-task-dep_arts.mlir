module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:o-i64:64-i128:128-n32:64-S128", llvm.target_triple = "arm64-apple-macosx15.0.0", "polygeist.target-cpu" = "apple-m1", "polygeist.target-features" = "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"} {
  func.func @sweep(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<?xf64>, %arg5: i32, %arg6: i32, %arg7: memref<?xf64>, %arg8: memref<?xf64>, %arg9: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c8 = arith.constant 8 : index
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %c-1_i32 = arith.constant -1 : i32
    %true = arith.constant true
    %cst = arith.constant 2.500000e-01 : f64
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %0 = polygeist.memref2pointer %arg4 : memref<?xf64> to !llvm.ptr
    %1 = polygeist.pointer2memref %0 : !llvm.ptr to memref<?x?xf64>
    %2 = polygeist.memref2pointer %arg7 : memref<?xf64> to !llvm.ptr
    %3 = polygeist.pointer2memref %2 : !llvm.ptr to memref<?x?xf64>
    %4 = polygeist.memref2pointer %arg8 : memref<?xf64> to !llvm.ptr
    %5 = polygeist.pointer2memref %4 : !llvm.ptr to memref<?x?xf64>
    arts.edt <parallel> <internode> route(%c0_i32) {
      arts.barrier
      arts.edt <single> <intranode> route(%c0_i32) {
        %6 = arith.addi %arg5, %c1_i32 : i32
        %7 = arith.addi %arg6, %c1_i32 : i32
        %8 = arith.index_cast %7 : i32 to index
        %9 = arith.index_cast %6 : i32 to index
        scf.for %arg10 = %9 to %8 step %c1 {
          %10 = arith.index_cast %arg0 : i32 to index
          scf.for %arg11 = %c0 to %10 step %c1 {
            %11 = arts.db_control[<in>] (%5 : memref<?x?xf64>) elementType(f64) elementTypeSize(%c8 : index) indices[%arg11, %c0] offsets[] sizes[] : memref<f64>
            %12 = arts.db_control[<out>] (%3 : memref<?x?xf64>) elementType(f64) elementTypeSize(%c8 : index) indices[%arg11, %c0] offsets[] sizes[] : memref<f64>
            arts.edt <task> <intranode> route(%c0_i32) (%11, %12) : memref<f64>, memref<f64> {
              %13 = arith.index_cast %arg1 : i32 to index
              scf.for %arg12 = %c0 to %13 step %c1 {
                %14 = memref.load %5[%arg11, %arg12] : memref<?x?xf64>
                memref.store %14, %3[%arg11, %arg12] : memref<?x?xf64>
              }
            }
          }
          scf.for %arg11 = %c0 to %10 step %c1 {
            %11 = arith.index_cast %arg11 : index to i32
            %12 = arts.db_control[<in>] (%1 : memref<?x?xf64>) elementType(f64) elementTypeSize(%c8 : index) indices[%arg11, %c0] offsets[] sizes[] : memref<f64>
            %13 = arith.addi %11, %c-1_i32 : i32
            %14 = arith.index_cast %13 : i32 to index
            %15 = arts.db_control[<in>] (%3 : memref<?x?xf64>) elementType(f64) elementTypeSize(%c8 : index) indices[%14, %c0] offsets[] sizes[] : memref<f64>
            %16 = arts.db_control[<in>] (%3 : memref<?x?xf64>) elementType(f64) elementTypeSize(%c8 : index) indices[%arg11, %c0] offsets[] sizes[] : memref<f64>
            %17 = arith.addi %11, %c1_i32 : i32
            %18 = arith.index_cast %17 : i32 to index
            %19 = arts.db_control[<in>] (%3 : memref<?x?xf64>) elementType(f64) elementTypeSize(%c8 : index) indices[%18, %c0] offsets[] sizes[] : memref<f64>
            %20 = arts.db_control[<out>] (%5 : memref<?x?xf64>) elementType(f64) elementTypeSize(%c8 : index) indices[%arg11, %c0] offsets[] sizes[] : memref<f64>
            arts.edt <task> <intranode> route(%c0_i32) (%12, %15, %16, %19, %20) : memref<f64>, memref<f64>, memref<f64>, memref<f64>, memref<f64> {
              %21 = arith.cmpi eq, %11, %c0_i32 : i32
              %22 = arith.index_cast %arg1 : i32 to index
              scf.for %arg12 = %c0 to %22 step %c1 {
                %23 = arith.index_cast %arg12 : index to i32
                %24 = scf.if %21 -> (i1) {
                  scf.yield %true : i1
                } else {
                  %27 = arith.cmpi eq, %23, %c0_i32 : i32
                  scf.yield %27 : i1
                }
                %25 = scf.if %24 -> (i1) {
                  scf.yield %true : i1
                } else {
                  %27 = arith.addi %arg0, %c-1_i32 : i32
                  %28 = arith.cmpi eq, %11, %27 : i32
                  scf.yield %28 : i1
                }
                %26 = scf.if %25 -> (i1) {
                  scf.yield %true : i1
                } else {
                  %27 = arith.addi %arg1, %c-1_i32 : i32
                  %28 = arith.cmpi eq, %23, %27 : i32
                  scf.yield %28 : i1
                }
                scf.if %26 {
                  %27 = memref.load %1[%arg11, %arg12] : memref<?x?xf64>
                  memref.store %27, %5[%arg11, %arg12] : memref<?x?xf64>
                } else {
                  %27 = memref.load %3[%14, %arg12] : memref<?x?xf64>
                  %28 = arith.addi %23, %c1_i32 : i32
                  %29 = arith.index_cast %28 : i32 to index
                  %30 = memref.load %3[%arg11, %29] : memref<?x?xf64>
                  %31 = arith.addf %27, %30 : f64
                  %32 = arith.addi %23, %c-1_i32 : i32
                  %33 = arith.index_cast %32 : i32 to index
                  %34 = memref.load %3[%arg11, %33] : memref<?x?xf64>
                  %35 = arith.addf %31, %34 : f64
                  %36 = memref.load %3[%18, %arg12] : memref<?x?xf64>
                  %37 = arith.addf %35, %36 : f64
                  %38 = memref.load %1[%arg11, %arg12] : memref<?x?xf64>
                  %39 = arith.mulf %38, %arg2 : f64
                  %40 = arith.mulf %39, %arg3 : f64
                  %41 = arith.addf %37, %40 : f64
                  %42 = arith.mulf %41, %cst : f64
                  memref.store %42, %5[%arg11, %arg12] : memref<?x?xf64>
                }
              }
            }
          }
        }
      }
      arts.barrier
    }
    return
  }
}
