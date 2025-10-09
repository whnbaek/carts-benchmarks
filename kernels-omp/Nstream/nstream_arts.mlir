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
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.000000e-08 : f64
    %cst_0 = arith.constant 8.000000e+00 : f64
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %c8_i64 = arith.constant 8 : i64
    %cst_1 = arith.constant 3.200000e+01 : f64
    %cst_2 = arith.constant 9.9999999999999995E-7 : f64
    %cst_3 = arith.constant 3.000000e+00 : f64
    %cst_4 = arith.constant 2.000000e+00 : f64
    %cst_5 = arith.constant 0.000000e+00 : f64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c0_i64 = arith.constant 0 : i64
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    %c5_i32 = arith.constant 5 : i32
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
      %58 = llvm.mlir.addressof @str3 : !llvm.ptr
      %59 = llvm.getelementptr %58[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x i8>
      %60 = memref.load %arg1[%c0] : memref<?xmemref<?xi8>>
      %61 = polygeist.memref2pointer %60 : memref<?xi8> to !llvm.ptr
      %62 = llvm.call @printf(%59, %61) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %10 = memref.load %arg1[%c1] : memref<?xmemref<?xi8>>
    %11 = call @atoi(%10) : (memref<?xi8>) -> i32
    %12 = memref.load %arg1[%c2] : memref<?xmemref<?xi8>>
    %13 = call @atoi(%12) : (memref<?xi8>) -> i32
    %14 = memref.load %arg1[%c3] : memref<?xmemref<?xi8>>
    %15 = call @atol(%14) : (memref<?xi8>) -> i64
    %16 = memref.load %arg1[%c4] : memref<?xmemref<?xi8>>
    %17 = call @atol(%16) : (memref<?xi8>) -> i64
    %18 = arith.cmpi slt, %11, %c1_i32 : i32
    %19 = scf.if %18 -> (i1) {
      scf.yield %true : i1
    } else {
      %58 = arith.cmpi sgt, %11, %c512_i32 : i32
      scf.yield %58 : i1
    }
    scf.if %19 {
      %58 = llvm.mlir.addressof @str4 : !llvm.ptr
      %59 = llvm.getelementptr %58[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
      %60 = llvm.call @printf(%59, %11) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %20 = arith.cmpi slt, %13, %c1_i32 : i32
    scf.if %20 {
      %58 = llvm.mlir.addressof @str5 : !llvm.ptr
      %59 = llvm.getelementptr %58[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<41 x i8>
      %60 = llvm.call @printf(%59, %13) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %21 = arith.cmpi slt, %15, %c0_i64 : i64
    scf.if %21 {
      %58 = llvm.mlir.addressof @str6 : !llvm.ptr
      %59 = llvm.getelementptr %58[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<35 x i8>
      %60 = llvm.call @printf(%59, %15) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %22 = arith.cmpi slt, %17, %c0_i64 : i64
    scf.if %22 {
      %58 = llvm.mlir.addressof @str7 : !llvm.ptr
      %59 = llvm.getelementptr %58[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<35 x i8>
      %60 = llvm.call @printf(%59, %17) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
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
    memref.store %30, %27[%c0] : memref<1xmemref<?xf64>>
    %31 = memref.load %27[%c0] : memref<1xmemref<?xf64>>
    %32 = polygeist.memref2pointer %31 : memref<?xf64> to !llvm.ptr
    %33 = llvm.mlir.zero : !llvm.ptr
    %34 = llvm.icmp "eq" %32, %33 : !llvm.ptr
    scf.if %34 {
      %58 = llvm.mlir.addressof @str8 : !llvm.ptr
      %59 = llvm.getelementptr %58[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<49 x i8>
      %60 = llvm.call @printf(%59, %25) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %35 = memref.get_global @b : memref<1xmemref<?xf64>>
    %36 = memref.load %27[%c0] : memref<1xmemref<?xf64>>
    %37 = arith.index_cast %15 : i64 to index
    %38 = arith.index_cast %17 : i64 to index
    %39 = arith.addi %38, %37 : index
    %40 = polygeist.subindex %36[%39] () : memref<?xf64> -> memref<?xf64>
    memref.store %40, %35[%c0] : memref<1xmemref<?xf64>>
    %41 = memref.get_global @c : memref<1xmemref<?xf64>>
    %42 = memref.load %35[%c0] : memref<1xmemref<?xf64>>
    %43 = polygeist.subindex %42[%39] () : memref<?xf64> -> memref<?xf64>
    memref.store %43, %41[%c0] : memref<1xmemref<?xf64>>
    arts.edt <parallel> <internode> route(%c0_i32) {
      func.call @bail_out(%c0_i32) : (i32) -> ()
      arts.for(%c0) to(%37) step(%c1) {{
      ^bb0(%arg2: index):
        %61 = memref.load %27[%c0] : memref<1xmemref<?xf64>>
        memref.store %cst_5, %61[%arg2] : memref<?xf64>
        %62 = memref.load %35[%c0] : memref<1xmemref<?xf64>>
        memref.store %cst_4, %62[%arg2] : memref<?xf64>
        %63 = memref.load %41[%c0] : memref<1xmemref<?xf64>>
        memref.store %cst_4, %63[%arg2] : memref<?xf64>
      }}
      %58 = arith.extsi %13 : i32 to i64
      %59 = arith.addi %58, %c1_i64 : i64
      %60 = arith.index_cast %59 : i64 to index
      scf.for %arg2 = %c0 to %60 step %c1 {
        arts.for(%c0) to(%37) step(%c1) {{
        ^bb0(%arg3: index):
          %61 = memref.load %27[%c0] : memref<1xmemref<?xf64>>
          %62 = memref.load %35[%c0] : memref<1xmemref<?xf64>>
          %63 = memref.load %62[%arg3] : memref<?xf64>
          %64 = memref.load %41[%c0] : memref<1xmemref<?xf64>>
          %65 = memref.load %64[%arg3] : memref<?xf64>
          %66 = arith.mulf %65, %cst_3 : f64
          %67 = arith.addf %63, %66 : f64
          %68 = memref.load %61[%arg3] : memref<?xf64>
          %69 = arith.addf %68, %67 : f64
          memref.store %69, %61[%arg3] : memref<?xf64>
        }}
      }
    }
    %44 = arith.sitofp %15 : i64 to f64
    %45 = arith.mulf %44, %cst_1 : f64
    %46 = arith.addi %13, %c1_i32 : i32
    %47 = arith.index_cast %46 : i32 to index
    %48 = scf.for %arg2 = %c0 to %47 step %c1 iter_args(%arg3 = %cst_5) -> (f64) {
      %58 = arith.addf %arg3, %cst_0 : f64
      scf.yield %58 : f64
    }
    %49 = arith.sitofp %15 : i64 to f64
    %50 = arith.mulf %48, %49 : f64
    %51 = arith.index_cast %15 : i64 to index
    %52 = scf.for %arg2 = %c0 to %51 step %c1 iter_args(%arg3 = %cst_5) -> (f64) {
      %58 = memref.get_global @a : memref<1xmemref<?xf64>>
      %59 = memref.load %58[%c0] : memref<1xmemref<?xf64>>
      %60 = memref.load %59[%arg2] : memref<?xf64>
      %61 = arith.addf %arg3, %60 : f64
      scf.yield %61 : f64
    }
    %53 = arith.subf %50, %52 : f64
    %54 = arith.cmpf oge, %53, %cst_5 : f64
    %55 = scf.if %54 -> (f64) {
      scf.yield %53 : f64
    } else {
      %58 = arith.negf %53 : f64
      scf.yield %58 : f64
    }
    %56 = arith.divf %55, %52 : f64
    %57 = arith.cmpf ogt, %56, %cst : f64
    scf.if %57 {
      %58 = llvm.mlir.addressof @str10 : !llvm.ptr
      %59 = llvm.getelementptr %58[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<35 x i8>
      %60 = llvm.call @printf(%59) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    } else {
      %58 = llvm.mlir.addressof @str11 : !llvm.ptr
      %59 = llvm.getelementptr %58[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<20 x i8>
      %60 = llvm.call @printf(%59) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %61 = arith.sitofp %13 : i32 to f64
      %62 = arith.divf %0, %61 : f64
      %63 = llvm.mlir.addressof @str9 : !llvm.ptr
      %64 = llvm.getelementptr %63[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<36 x i8>
      %65 = arith.mulf %45, %cst_2 : f64
      %66 = arith.divf %65, %62 : f64
      %67 = llvm.call @printf(%64, %66, %62) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64) -> i32
    }
    return %c0_i32 : i32
  }
  func.func private @exit(i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @atoi(memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @atol(memref<?xi8>) -> i64 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @omp_set_num_threads(i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @prk_malloc(%arg0: i64) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<1xmemref<?xi8>>
    %0 = call @prk_get_alignment() : () -> i32
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = polygeist.pointer2memref %1 : !llvm.ptr to memref<?xi8>
    memref.store %2, %alloca[%c0] : memref<1xmemref<?xi8>>
    %cast = memref.cast %alloca : memref<1xmemref<?xi8>> to memref<?xmemref<?xi8>>
    %3 = arith.extsi %0 : i32 to i64
    %4 = call @posix_memalign(%cast, %3, %arg0) : (memref<?xmemref<?xi8>>, i64, i64) -> i32
    %5 = arith.cmpi ne, %4, %c0_i32 : i32
    scf.if %5 {
      memref.store %2, %alloca[%c0] : memref<1xmemref<?xi8>>
    }
    %6 = memref.load %alloca[%c0] : memref<1xmemref<?xi8>>
    return %6 : memref<?xi8>
  }
  func.func private @bail_out(i32) attributes {llvm.linkage = #llvm.linkage<external>}
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
