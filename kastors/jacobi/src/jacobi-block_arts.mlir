module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:o-i64:64-i128:128-n32:64-S128", llvm.target_triple = "arm64-apple-macosx15.0.0", "polygeist.target-cpu" = "apple-m1", "polygeist.target-features" = "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"} {
  llvm.mlir.global internal constant @str2("(i < nx) && (j < ny)\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("poisson.h\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str0("copy_block\00") {addr_space = 0 : i32}
  func.func @sweep(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<?xf64>, %arg5: i32, %arg6: i32, %arg7: memref<?xf64>, %arg8: memref<?xf64>, %arg9: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi eq, %arg9, %c0_i32 : i32
    %1 = arith.select %0, %arg0, %arg9 : i32
    %2 = arith.divsi %arg0, %1 : i32
    %3 = arith.divsi %arg1, %1 : i32
    %4 = arith.addi %arg5, %c1_i32 : i32
    %5 = arith.addi %arg6, %c1_i32 : i32
    %6 = arith.index_cast %5 : i32 to index
    %7 = arith.index_cast %4 : i32 to index
    scf.for %arg10 = %7 to %6 step %c1 {
      %8 = arith.index_cast %2 : i32 to index
      scf.for %arg11 = %c0 to %8 step %c1 {
        %9 = arith.index_cast %arg11 : index to i32
        %10 = arith.index_cast %3 : i32 to index
        scf.for %arg12 = %c0 to %10 step %c1 {
          %11 = arith.index_cast %arg12 : index to i32
          func.call @copy_block(%arg0, %arg1, %9, %11, %arg7, %arg8, %1) : (i32, i32, i32, i32, memref<?xf64>, memref<?xf64>, i32) -> ()
        }
      }
      scf.for %arg11 = %c0 to %8 step %c1 {
        %9 = arith.index_cast %arg11 : index to i32
        %10 = arith.index_cast %3 : i32 to index
        scf.for %arg12 = %c0 to %10 step %c1 {
          %11 = arith.index_cast %arg12 : index to i32
          func.call @compute_estimate(%9, %11, %arg7, %arg8, %arg4, %arg2, %arg3, %arg0, %arg1, %1) : (i32, i32, memref<?xf64>, memref<?xf64>, memref<?xf64>, f64, f64, i32, i32, i32) -> ()
        }
      }
    }
    return
  }
  func.func private @copy_block(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: i32) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %c72_i32 = arith.constant 72 : i32
    %false = arith.constant false
    %0 = polygeist.memref2pointer %arg4 : memref<?xf64> to !llvm.ptr
    %1 = polygeist.pointer2memref %0 : !llvm.ptr to memref<?x?xf64>
    %2 = polygeist.memref2pointer %arg5 : memref<?xf64> to !llvm.ptr
    %3 = polygeist.pointer2memref %2 : !llvm.ptr to memref<?x?xf64>
    %4 = arith.muli %arg2, %arg6 : i32
    %5 = arith.muli %arg3, %arg6 : i32
    %6 = arith.addi %4, %arg6 : i32
    %7 = arith.addi %5, %arg6 : i32
    %8 = arith.index_cast %6 : i32 to index
    %9 = arith.index_cast %4 : i32 to index
    scf.for %arg7 = %9 to %8 step %c1 {
      %10 = arith.index_cast %arg7 : index to i32
      %11 = arith.cmpi slt, %10, %arg0 : i32
      %12 = arith.index_cast %7 : i32 to index
      %13 = arith.index_cast %5 : i32 to index
      scf.for %arg8 = %13 to %12 step %c1 {
        %14 = scf.if %11 -> (i1) {
          %17 = arith.index_cast %arg8 : index to i32
          %18 = arith.cmpi slt, %17, %arg1 : i32
          scf.yield %18 : i1
        } else {
          scf.yield %false : i1
        }
        %15 = arith.xori %14, %true : i1
        scf.if %15 {
          %17 = llvm.mlir.addressof @str0 : !llvm.ptr
          %18 = llvm.mlir.addressof @str1 : !llvm.ptr
          %19 = llvm.mlir.addressof @str2 : !llvm.ptr
          %20 = polygeist.pointer2memref %17 : !llvm.ptr to memref<?xi8>
          %21 = polygeist.pointer2memref %18 : !llvm.ptr to memref<?xi8>
          %22 = polygeist.pointer2memref %19 : !llvm.ptr to memref<?xi8>
          func.call @__assert_rtn(%20, %21, %c72_i32, %22) : (memref<?xi8>, memref<?xi8>, i32, memref<?xi8>) -> ()
        }
        %16 = memref.load %3[%arg7, %arg8] : memref<?x?xf64>
        memref.store %16, %1[%arg7, %arg8] : memref<?x?xf64>
      }
    }
    return
  }
  func.func private @compute_estimate(%arg0: i32, %arg1: i32, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: f64, %arg6: f64, %arg7: i32, %arg8: i32, %arg9: i32) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c1 = arith.constant 1 : index
    %c-1_i32 = arith.constant -1 : i32
    %true = arith.constant true
    %cst = arith.constant 2.500000e-01 : f64
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = polygeist.memref2pointer %arg4 : memref<?xf64> to !llvm.ptr
    %1 = polygeist.pointer2memref %0 : !llvm.ptr to memref<?x?xf64>
    %2 = polygeist.memref2pointer %arg2 : memref<?xf64> to !llvm.ptr
    %3 = polygeist.pointer2memref %2 : !llvm.ptr to memref<?x?xf64>
    %4 = polygeist.memref2pointer %arg3 : memref<?xf64> to !llvm.ptr
    %5 = polygeist.pointer2memref %4 : !llvm.ptr to memref<?x?xf64>
    %6 = arith.muli %arg0, %arg9 : i32
    %7 = arith.muli %arg1, %arg9 : i32
    %8 = arith.addi %6, %arg9 : i32
    %9 = arith.addi %7, %arg9 : i32
    %10 = arith.index_cast %8 : i32 to index
    %11 = arith.index_cast %6 : i32 to index
    scf.for %arg10 = %11 to %10 step %c1 {
      %12 = arith.index_cast %arg10 : index to i32
      %13 = arith.cmpi eq, %12, %c0_i32 : i32
      %14 = arith.index_cast %9 : i32 to index
      %15 = arith.index_cast %7 : i32 to index
      scf.for %arg11 = %15 to %14 step %c1 {
        %16 = arith.index_cast %arg11 : index to i32
        %17 = scf.if %13 -> (i1) {
          scf.yield %true : i1
        } else {
          %20 = arith.cmpi eq, %16, %c0_i32 : i32
          scf.yield %20 : i1
        }
        %18 = scf.if %17 -> (i1) {
          scf.yield %true : i1
        } else {
          %20 = arith.addi %arg7, %c-1_i32 : i32
          %21 = arith.cmpi eq, %12, %20 : i32
          scf.yield %21 : i1
        }
        %19 = scf.if %18 -> (i1) {
          scf.yield %true : i1
        } else {
          %20 = arith.addi %arg8, %c-1_i32 : i32
          %21 = arith.cmpi eq, %16, %20 : i32
          scf.yield %21 : i1
        }
        scf.if %19 {
          %20 = memref.load %1[%arg10, %arg11] : memref<?x?xf64>
          memref.store %20, %5[%arg10, %arg11] : memref<?x?xf64>
        } else {
          %20 = arith.addi %12, %c-1_i32 : i32
          %21 = arith.index_cast %20 : i32 to index
          %22 = memref.load %3[%21, %arg11] : memref<?x?xf64>
          %23 = arith.addi %16, %c1_i32 : i32
          %24 = arith.index_cast %23 : i32 to index
          %25 = memref.load %3[%arg10, %24] : memref<?x?xf64>
          %26 = arith.addf %22, %25 : f64
          %27 = arith.addi %16, %c-1_i32 : i32
          %28 = arith.index_cast %27 : i32 to index
          %29 = memref.load %3[%arg10, %28] : memref<?x?xf64>
          %30 = arith.addf %26, %29 : f64
          %31 = arith.addi %12, %c1_i32 : i32
          %32 = arith.index_cast %31 : i32 to index
          %33 = memref.load %3[%32, %arg11] : memref<?x?xf64>
          %34 = arith.addf %30, %33 : f64
          %35 = memref.load %1[%arg10, %arg11] : memref<?x?xf64>
          %36 = arith.mulf %35, %arg5 : f64
          %37 = arith.mulf %36, %arg6 : f64
          %38 = arith.addf %34, %37 : f64
          %39 = arith.mulf %38, %cst : f64
          memref.store %39, %5[%arg10, %arg11] : memref<?x?xf64>
        }
      }
    }
    return
  }
  func.func private @__assert_rtn(memref<?xi8>, memref<?xi8>, i32, memref<?xi8>) attributes {llvm.linkage = #llvm.linkage<external>}
}
