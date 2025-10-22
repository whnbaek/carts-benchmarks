module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:o-i64:64-i128:128-n32:64-S128", llvm.target_triple = "arm64-apple-macosx16.0.0", "polygeist.target-cpu" = "apple-m1", "polygeist.target-features" = "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"} {
  func.func @sweep_seq(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<?xf64>, %arg5: i32, %arg6: i32, %arg7: memref<?xf64>, %arg8: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c-1_i32 = arith.constant -1 : i32
    %true = arith.constant true
    %cst = arith.constant 2.500000e-01 : f64
    %c0_i32 = arith.constant 0 : i32
    %0 = polygeist.memref2pointer %arg4 : memref<?xf64> to !llvm.ptr
    %1 = polygeist.pointer2memref %0 : !llvm.ptr to memref<?x?xf64>
    %2 = polygeist.memref2pointer %arg7 : memref<?xf64> to !llvm.ptr
    %3 = polygeist.pointer2memref %2 : !llvm.ptr to memref<?x?xf64>
    %4 = polygeist.memref2pointer %arg8 : memref<?xf64> to !llvm.ptr
    %5 = polygeist.pointer2memref %4 : !llvm.ptr to memref<?x?xf64>
    %6 = arith.addi %arg5, %c1_i32 : i32
    %7 = arith.addi %arg6, %c1_i32 : i32
    %8 = arith.index_cast %7 : i32 to index
    %9 = arith.index_cast %6 : i32 to index
    scf.for %arg9 = %9 to %8 step %c1 {
      %10 = arith.index_cast %arg0 : i32 to index
      scf.for %arg10 = %c0 to %10 step %c1 {
        %11 = arith.index_cast %arg1 : i32 to index
        scf.for %arg11 = %c0 to %11 step %c1 {
          %12 = memref.load %5[%arg10, %arg11] : memref<?x?xf64>
          memref.store %12, %3[%arg10, %arg11] : memref<?x?xf64>
        }
      }
      scf.for %arg10 = %c0 to %10 step %c1 {
        %11 = arith.index_cast %arg10 : index to i32
        %12 = arith.index_cast %arg1 : i32 to index
        scf.for %arg11 = %c0 to %12 step %c1 {
          %13 = arith.index_cast %arg11 : index to i32
          %14 = arith.cmpi eq, %11, %c0_i32 : i32
          %15 = scf.if %14 -> (i1) {
            scf.yield %true : i1
          } else {
            %18 = arith.cmpi eq, %13, %c0_i32 : i32
            scf.yield %18 : i1
          }
          %16 = scf.if %15 -> (i1) {
            scf.yield %true : i1
          } else {
            %18 = arith.addi %arg0, %c-1_i32 : i32
            %19 = arith.cmpi eq, %11, %18 : i32
            scf.yield %19 : i1
          }
          %17 = scf.if %16 -> (i1) {
            scf.yield %true : i1
          } else {
            %18 = arith.addi %arg1, %c-1_i32 : i32
            %19 = arith.cmpi eq, %13, %18 : i32
            scf.yield %19 : i1
          }
          scf.if %17 {
            %18 = memref.load %1[%arg10, %arg11] : memref<?x?xf64>
            memref.store %18, %5[%arg10, %arg11] : memref<?x?xf64>
          } else {
            %18 = arith.addi %11, %c-1_i32 : i32
            %19 = arith.index_cast %18 : i32 to index
            %20 = memref.load %3[%19, %arg11] : memref<?x?xf64>
            %21 = arith.addi %13, %c1_i32 : i32
            %22 = arith.index_cast %21 : i32 to index
            %23 = memref.load %3[%arg10, %22] : memref<?x?xf64>
            %24 = arith.addf %20, %23 : f64
            %25 = arith.addi %13, %c-1_i32 : i32
            %26 = arith.index_cast %25 : i32 to index
            %27 = memref.load %3[%arg10, %26] : memref<?x?xf64>
            %28 = arith.addf %24, %27 : f64
            %29 = arith.addi %11, %c1_i32 : i32
            %30 = arith.index_cast %29 : i32 to index
            %31 = memref.load %3[%30, %arg11] : memref<?x?xf64>
            %32 = arith.addf %28, %31 : f64
            %33 = memref.load %1[%arg10, %arg11] : memref<?x?xf64>
            %34 = arith.mulf %33, %arg2 : f64
            %35 = arith.mulf %34, %arg3 : f64
            %36 = arith.addf %32, %35 : f64
            %37 = arith.mulf %36, %cst : f64
            memref.store %37, %5[%arg10, %arg11] : memref<?x?xf64>
          }
        }
      }
    }
    return
  }
}
