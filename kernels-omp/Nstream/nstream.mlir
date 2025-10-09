module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:o-i64:64-i128:128-n32:64-S128", llvm.target_triple = "arm64-apple-macosx15.0.0", "polygeist.target-cpu" = "apple-m1", "polygeist.target-features" = "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"} {
  llvm.mlir.global internal constant @str15("(a & (~a+1)) == a\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str14("par-res-kern_general.h\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str13("prk_get_alignment\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str12("PRK_ALIGNMENT\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str11("Solution validates\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str10("Failed Validation on output array\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str9("Rate (MB/s): %lf Avg time (s): %lf\0A\00") {addr_space = 0 : i32}
  memref.global "private" @c : memref<1xmemref<?xf64>> = uninitialized
  memref.global "private" @b : memref<1xmemref<?xf64>> = uninitialized
  llvm.mlir.global internal constant @str8("ERROR: Could not allocate %ld words for vectors\0A\00") {addr_space = 0 : i32}
  memref.global "private" @a : memref<1xmemref<?xf64>> = uninitialized
  llvm.mlir.global internal constant @str7("ERROR: Incvalid array offset: %ld\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str6("ERROR: Invalid vector length: %ld\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str5("ERROR: Invalid number of iterations: %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str4("ERROR: Invalid number of threads: %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str3("Usage:  %s <# threads> <# iterations> <vector length> <offset>\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2("OpenMP stream triad: A = B + scalar*C\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("2.17\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str0("Parallel Research Kernels version %s\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %c8_i64 = arith.constant 8 : i64
    %cst = arith.constant 3.200000e+01 : f64
    %cst_0 = arith.constant 9.9999999999999995E-7 : f64
    %cst_1 = arith.constant 3.000000e+00 : f64
    %cst_2 = arith.constant 2.000000e+00 : f64
    %cst_3 = arith.constant 0.000000e+00 : f64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c0_i64 = arith.constant 0 : i64
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    %c5_i32 = arith.constant 5 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = llvm.mlir.undef : f64
    %1 = llvm.mlir.addressof @str0 : !llvm.ptr
    %2 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
    %3 = llvm.mlir.addressof @str1 : !llvm.ptr
    %4 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<5 x i8>
    %5 = llvm.call @printf(%2, %4) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    %6 = llvm.mlir.addressof @str2 : !llvm.ptr
    %7 = llvm.getelementptr %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<39 x i8>
    %8 = llvm.call @printf(%7) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %9 = arith.cmpi ne, %arg0, %c5_i32 : i32
    scf.if %9 {
      %54 = llvm.mlir.addressof @str3 : !llvm.ptr
      %55 = llvm.getelementptr %54[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x i8>
      %56 = affine.load %arg1[0] : memref<?xmemref<?xi8>>
      %57 = polygeist.memref2pointer %56 : memref<?xi8> to !llvm.ptr
      %58 = llvm.call @printf(%55, %57) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %10 = affine.load %arg1[1] : memref<?xmemref<?xi8>>
    %11 = call @atoi(%10) : (memref<?xi8>) -> i32
    %12 = affine.load %arg1[2] : memref<?xmemref<?xi8>>
    %13 = call @atoi(%12) : (memref<?xi8>) -> i32
    %14 = affine.load %arg1[3] : memref<?xmemref<?xi8>>
    %15 = call @atol(%14) : (memref<?xi8>) -> i64
    %16 = affine.load %arg1[4] : memref<?xmemref<?xi8>>
    %17 = call @atol(%16) : (memref<?xi8>) -> i64
    %18 = arith.cmpi slt, %11, %c1_i32 : i32
    %19 = scf.if %18 -> (i1) {
      scf.yield %true : i1
    } else {
      %54 = arith.cmpi sgt, %11, %c512_i32 : i32
      scf.yield %54 : i1
    }
    scf.if %19 {
      %54 = llvm.mlir.addressof @str4 : !llvm.ptr
      %55 = llvm.getelementptr %54[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
      %56 = llvm.call @printf(%55, %11) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %20 = arith.cmpi slt, %13, %c1_i32 : i32
    scf.if %20 {
      %54 = llvm.mlir.addressof @str5 : !llvm.ptr
      %55 = llvm.getelementptr %54[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<41 x i8>
      %56 = llvm.call @printf(%55, %13) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %21 = arith.cmpi slt, %15, %c0_i64 : i64
    scf.if %21 {
      %54 = llvm.mlir.addressof @str6 : !llvm.ptr
      %55 = llvm.getelementptr %54[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<35 x i8>
      %56 = llvm.call @printf(%55, %15) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %22 = arith.cmpi slt, %17, %c0_i64 : i64
    scf.if %22 {
      %54 = llvm.mlir.addressof @str7 : !llvm.ptr
      %55 = llvm.getelementptr %54[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<35 x i8>
      %56 = llvm.call @printf(%55, %17) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    call @omp_set_num_threads(%11) : (i32) -> ()
    %23 = arith.muli %15, %c3_i64 : i64
    %24 = arith.muli %17, %c2_i64 : i64
    %25 = arith.addi %23, %24 : i64
    %26 = arith.muli %25, %c8_i64 : i64
    %27 = memref.get_global @a : memref<1xmemref<?xf64>>
    %28 = call @prk_malloc(%26) : (i64) -> memref<?xi8>
    %29 = polygeist.memref2pointer %28 : memref<?xi8> to !llvm.ptr
    %30 = polygeist.pointer2memref %29 : !llvm.ptr to memref<?xf64>
    affine.store %30, %27[0] : memref<1xmemref<?xf64>>
    %31 = memref.get_global @a : memref<1xmemref<?xf64>>
    %32 = affine.load %31[0] : memref<1xmemref<?xf64>>
    %33 = polygeist.memref2pointer %32 : memref<?xf64> to !llvm.ptr
    %34 = llvm.mlir.zero : !llvm.ptr
    %35 = llvm.icmp "eq" %33, %34 : !llvm.ptr
    scf.if %35 {
      %54 = llvm.mlir.addressof @str8 : !llvm.ptr
      %55 = llvm.getelementptr %54[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<49 x i8>
      %56 = arith.muli %15, %c3_i64 : i64
      %57 = arith.muli %17, %c2_i64 : i64
      %58 = arith.addi %56, %57 : i64
      %59 = llvm.call @printf(%55, %58) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %36 = memref.get_global @b : memref<1xmemref<?xf64>>
    %37 = memref.get_global @a : memref<1xmemref<?xf64>>
    %38 = affine.load %37[0] : memref<1xmemref<?xf64>>
    %39 = arith.index_cast %15 : i64 to index
    %40 = arith.index_cast %17 : i64 to index
    %41 = arith.addi %40, %39 : index
    %42 = polygeist.subindex %38[%41] () : memref<?xf64> -> memref<?xf64>
    affine.store %42, %36[0] : memref<1xmemref<?xf64>>
    %43 = memref.get_global @c : memref<1xmemref<?xf64>>
    %44 = memref.get_global @b : memref<1xmemref<?xf64>>
    %45 = affine.load %44[0] : memref<1xmemref<?xf64>>
    %46 = arith.index_cast %15 : i64 to index
    %47 = arith.index_cast %17 : i64 to index
    %48 = arith.addi %47, %46 : index
    %49 = polygeist.subindex %45[%48] () : memref<?xf64> -> memref<?xf64>
    affine.store %49, %43[0] : memref<1xmemref<?xf64>>
    omp.parallel   {
      func.call @bail_out(%c0_i32) : (i32) -> ()
      %54 = arith.index_cast %15 : i64 to index
      omp.wsloop   for  (%arg2) : index = (%c0) to (%54) step (%c1) {
        %58 = memref.get_global @a : memref<1xmemref<?xf64>>
        %59 = affine.load %58[0] : memref<1xmemref<?xf64>>
        memref.store %cst_3, %59[%arg2] : memref<?xf64>
        %60 = memref.get_global @b : memref<1xmemref<?xf64>>
        %61 = affine.load %60[0] : memref<1xmemref<?xf64>>
        memref.store %cst_2, %61[%arg2] : memref<?xf64>
        %62 = memref.get_global @c : memref<1xmemref<?xf64>>
        %63 = affine.load %62[0] : memref<1xmemref<?xf64>>
        memref.store %cst_2, %63[%arg2] : memref<?xf64>
        omp.yield
      }
      %55 = arith.extsi %13 : i32 to i64
      %56 = arith.addi %55, %c1_i64 : i64
      %57 = arith.index_cast %56 : i64 to index
      scf.for %arg2 = %c0 to %57 step %c1 {
        %58 = arith.index_cast %15 : i64 to index
        omp.wsloop   for  (%arg3) : index = (%c0) to (%58) step (%c1) {
          %59 = memref.get_global @a : memref<1xmemref<?xf64>>
          %60 = affine.load %59[0] : memref<1xmemref<?xf64>>
          %61 = memref.get_global @b : memref<1xmemref<?xf64>>
          %62 = affine.load %61[0] : memref<1xmemref<?xf64>>
          %63 = memref.load %62[%arg3] : memref<?xf64>
          %64 = memref.get_global @c : memref<1xmemref<?xf64>>
          %65 = affine.load %64[0] : memref<1xmemref<?xf64>>
          %66 = memref.load %65[%arg3] : memref<?xf64>
          %67 = arith.mulf %66, %cst_1 : f64
          %68 = arith.addf %63, %67 : f64
          %69 = memref.load %60[%arg3] : memref<?xf64>
          %70 = arith.addf %69, %68 : f64
          memref.store %70, %60[%arg3] : memref<?xf64>
          omp.yield
        }
      }
      omp.terminator
    }
    %50 = arith.sitofp %15 : i64 to f64
    %51 = arith.mulf %50, %cst : f64
    %52 = call @checkTRIADresults(%13, %15) : (i32, i64) -> i32
    %53 = arith.cmpi ne, %52, %c0_i32 : i32
    scf.if %53 {
      %54 = arith.sitofp %13 : i32 to f64
      %55 = arith.divf %0, %54 : f64
      %56 = llvm.mlir.addressof @str9 : !llvm.ptr
      %57 = llvm.getelementptr %56[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<36 x i8>
      %58 = arith.mulf %51, %cst_0 : f64
      %59 = arith.divf %58, %55 : f64
      %60 = llvm.call @printf(%57, %59, %55) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64) -> i32
    } else {
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    return %c0_i32 : i32
  }
  func.func private @exit(i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @atoi(memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @atol(memref<?xi8>) -> i64 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @omp_set_num_threads(i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @prk_malloc(%arg0: i64) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<1xmemref<?xi8>>
    %0 = call @prk_get_alignment() : () -> i32
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = polygeist.pointer2memref %1 : !llvm.ptr to memref<?xi8>
    affine.store %2, %alloca[0] : memref<1xmemref<?xi8>>
    %cast = memref.cast %alloca : memref<1xmemref<?xi8>> to memref<?xmemref<?xi8>>
    %3 = arith.extsi %0 : i32 to i64
    %4 = call @posix_memalign(%cast, %3, %arg0) : (memref<?xmemref<?xi8>>, i64, i64) -> i32
    %5 = arith.cmpi ne, %4, %c0_i32 : i32
    scf.if %5 {
      %7 = llvm.mlir.zero : !llvm.ptr
      %8 = polygeist.pointer2memref %7 : !llvm.ptr to memref<?xi8>
      affine.store %8, %alloca[0] : memref<1xmemref<?xi8>>
    }
    %6 = affine.load %alloca[0] : memref<1xmemref<?xi8>>
    return %6 : memref<?xi8>
  }
  func.func private @bail_out(i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @checkTRIADresults(%arg0: i32, %arg1: i64) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 8.000000e+00 : f64
    %true = arith.constant true
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 1.000000e-08 : f64
    %0 = arith.addi %arg0, %c1_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = scf.for %arg2 = %c0 to %1 step %c1 iter_args(%arg3 = %cst_0) -> (f64) {
      %14 = arith.addf %arg3, %cst : f64
      scf.yield %14 : f64
    }
    %3 = arith.sitofp %arg1 : i64 to f64
    %4 = arith.mulf %2, %3 : f64
    %5 = arith.index_cast %arg1 : i64 to index
    %6 = scf.for %arg2 = %c0 to %5 step %c1 iter_args(%arg3 = %cst_0) -> (f64) {
      %14 = memref.get_global @a : memref<1xmemref<?xf64>>
      %15 = affine.load %14[0] : memref<1xmemref<?xf64>>
      %16 = memref.load %15[%arg2] : memref<?xf64>
      %17 = arith.addf %arg3, %16 : f64
      scf.yield %17 : f64
    }
    %7 = arith.subf %4, %6 : f64
    %8 = arith.cmpf oge, %7, %cst_0 : f64
    %9 = scf.if %8 -> (f64) {
      scf.yield %7 : f64
    } else {
      %14 = arith.negf %7 : f64
      scf.yield %14 : f64
    }
    %10 = arith.divf %9, %6 : f64
    %11 = arith.cmpf ogt, %10, %cst_1 : f64
    %12 = arith.xori %11, %true : i1
    %13 = arith.extui %12 : i1 to i32
    scf.if %11 {
      %14 = llvm.mlir.addressof @str10 : !llvm.ptr
      %15 = llvm.getelementptr %14[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<35 x i8>
      %16 = llvm.call @printf(%15) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    } else {
      %14 = llvm.mlir.addressof @str11 : !llvm.ptr
      %15 = llvm.getelementptr %14[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<20 x i8>
      %16 = llvm.call @printf(%15) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    }
    return %13 : i32
  }
  func.func private @prk_get_alignment() -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c8_i32 = arith.constant 8 : i32
    %false = arith.constant false
    %c107_i32 = arith.constant 107 : i32
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = llvm.mlir.addressof @str12 : !llvm.ptr
    %1 = polygeist.pointer2memref %0 : !llvm.ptr to memref<?xi8>
    %2 = call @getenv(%1) : (memref<?xi8>) -> memref<?xi8>
    %3 = llvm.mlir.zero : !llvm.ptr
    %4 = polygeist.memref2pointer %2 : memref<?xi8> to !llvm.ptr
    %5 = llvm.icmp "ne" %4, %3 : !llvm.ptr
    %6:2 = scf.if %5 -> (i32, i1) {
      %12 = func.call @atoi(%2) : (memref<?xi8>) -> i32
      %13 = arith.cmpi slt, %12, %c8_i32 : i32
      scf.yield %12, %13 : i32, i1
    } else {
      scf.yield %c64_i32, %false : i32, i1
    }
    %7 = arith.select %6#1, %c8_i32, %6#0 : i32
    %8 = arith.xori %7, %c-1_i32 : i32
    %9 = arith.addi %8, %c1_i32 : i32
    %10 = arith.andi %7, %9 : i32
    %11 = arith.cmpi ne, %10, %7 : i32
    scf.if %11 {
      %12 = llvm.mlir.addressof @str13 : !llvm.ptr
      %13 = llvm.mlir.addressof @str14 : !llvm.ptr
      %14 = llvm.mlir.addressof @str15 : !llvm.ptr
      %15 = polygeist.pointer2memref %12 : !llvm.ptr to memref<?xi8>
      %16 = polygeist.pointer2memref %13 : !llvm.ptr to memref<?xi8>
      %17 = polygeist.pointer2memref %14 : !llvm.ptr to memref<?xi8>
      func.call @__assert_rtn(%15, %16, %c107_i32, %17) : (memref<?xi8>, memref<?xi8>, i32, memref<?xi8>) -> ()
    }
    return %7 : i32
  }
  func.func private @posix_memalign(memref<?xmemref<?xi8>>, i64, i64) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @getenv(memref<?xi8>) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @__assert_rtn(memref<?xi8>, memref<?xi8>, i32, memref<?xi8>) attributes {llvm.linkage = #llvm.linkage<external>}
}
