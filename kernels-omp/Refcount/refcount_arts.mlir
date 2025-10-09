module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:o-i64:64-i128:128-n32:64-S128", llvm.target_triple = "arm64-apple-macosx15.0.0", "polygeist.target-cpu" = "apple-m1", "polygeist.target-features" = "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"} {
  llvm.mlir.global internal constant @str16("(a & (~a+1)) == a\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str15("par-res-kern_general.h\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str14("prk_get_alignment\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str13("PRK_ALIGNMENT\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str12("Rate (MCPUPs/s): %lf time (s): %lf\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str11("Solution validates\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str10("should be %13.10lf, %13.10lf\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str9("ERROR: Incorrect or inconsistent counter values %13.10lf %13.10lf; \00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str8("ERROR: Thread %d encountered errors in private work\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str7("ERROR: could not allocate space for counters\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str6("ERROR: Could not allocate %ld words for private streams\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str5("ERROR: iterations must be >= 1 : %zu \0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str4("ERROR: Invalid number of threads: %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str3("Usage: %s <# threads> <# counter pair updates> <private stream size>\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2("OpenMP exclusive access test RefCount, shared counters\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("2.17\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str0("Parallel Research Kernels version %s\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  func.func @private_stream(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: i64) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 3.000000e+00 : f64
    %0 = arith.index_cast %arg3 : i64 to index
    scf.for %arg4 = %c0 to %0 step %c1 {
      %1 = memref.load %arg1[%arg4] : memref<?xf64>
      %2 = memref.load %arg2[%arg4] : memref<?xf64>
      %3 = arith.mulf %2, %cst : f64
      %4 = arith.addf %1, %3 : f64
      %5 = memref.load %arg0[%arg4] : memref<?xf64>
      %6 = arith.addf %5, %4 : f64
      memref.store %6, %arg0[%arg4] : memref<?xf64>
    }
    return
  }
  func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    %c16_i64 = arith.constant 16 : i64
    %c8_i64 = arith.constant 8 : i64
    %cst = arith.constant 9.9999999999999995E-7 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %c2_i64 = arith.constant 2 : i64
    %c29_i32 = arith.constant 29 : i32
    %cst_1 = arith.constant 0.000000e+00 : f64
    %c1_i64 = arith.constant 1 : i64
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant 9.9999999999999995E-8 : f64
    %0 = llvm.mlir.undef : i64
    %1 = llvm.mlir.undef : i32
    %alloca = memref.alloca() : memref<f64>
    %2 = llvm.mlir.undef : f64
    memref.store %2, %alloca[] : memref<f64>
    %3 = llvm.mlir.addressof @str0 : !llvm.ptr
    %4 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
    %5 = llvm.mlir.addressof @str1 : !llvm.ptr
    %6 = llvm.getelementptr %5[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<5 x i8>
    %7 = llvm.call @printf(%4, %6) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    %8 = llvm.mlir.addressof @str2 : !llvm.ptr
    %9 = llvm.getelementptr %8[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<56 x i8>
    %10 = llvm.call @printf(%9) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %11 = arith.cmpi ne, %arg0, %c4_i32 : i32
    %12 = arith.cmpi eq, %arg0, %c4_i32 : i32
    %13 = arith.select %11, %c1_i32, %1 : i32
    scf.if %11 {
      %14 = llvm.mlir.addressof @str3 : !llvm.ptr
      %15 = llvm.getelementptr %14[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<70 x i8>
      %16 = memref.load %arg1[%c0] : memref<?xmemref<?xi8>>
      %17 = polygeist.memref2pointer %16 : memref<?xi8> to !llvm.ptr
      %18 = llvm.call @printf(%15, %17) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    }
    scf.if %12 {
      %14 = memref.load %arg1[%c1] : memref<?xmemref<?xi8>>
      %15 = func.call @atoi(%14) : (memref<?xi8>) -> i32
      %16 = arith.cmpi slt, %15, %c1_i32 : i32
      %17 = scf.if %16 -> (i1) {
        scf.yield %true : i1
      } else {
        %32 = arith.cmpi sgt, %15, %c512_i32 : i32
        scf.yield %32 : i1
      }
      scf.if %17 {
        %32 = llvm.mlir.addressof @str4 : !llvm.ptr
        %33 = llvm.getelementptr %32[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
        %34 = llvm.call @printf(%33, %15) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
        func.call @exit(%c1_i32) : (i32) -> ()
      }
      %18 = memref.load %arg1[%c2] : memref<?xmemref<?xi8>>
      %19 = func.call @atol(%18) : (memref<?xi8>) -> i64
      %20 = arith.cmpi slt, %19, %c1_i64 : i64
      scf.if %20 {
        %32 = llvm.mlir.addressof @str5 : !llvm.ptr
        %33 = llvm.getelementptr %32[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<39 x i8>
        %34 = llvm.call @printf(%33, %19) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
        func.call @exit(%c1_i32) : (i32) -> ()
      }
      func.call @omp_set_num_threads(%15) : (i32) -> ()
      arts.edt <parallel> <internode> route(%c0_i32) {
        func.call @bail_out(%c0_i32) : (i32) -> ()
        %32 = func.call @sysconf(%c29_i32) : (i32) -> i64
        %33 = arith.addi %32, %c8_i64 : i64
        %34 = polygeist.typeSize memref<?xi8> : index
        %35 = arith.index_cast %34 : index to i64
        %36 = arith.addi %33, %35 : i64
        %37 = func.call @prk_malloc(%36) : (i64) -> memref<?xi8>
        %38 = polygeist.memref2pointer %37 : memref<?xi8> to !llvm.ptr
        %39 = polygeist.pointer2memref %38 : !llvm.ptr to memref<?xf64>
        %40 = llvm.mlir.zero : !llvm.ptr
        %41:2 = scf.while (%arg2 = %32, %arg3 = %39) : (i64, memref<?xf64>) -> (i64, memref<?xf64>) {
          %61 = polygeist.memref2pointer %arg3 : memref<?xf64> to !llvm.ptr
          %62 = llvm.icmp "eq" %61, %40 : !llvm.ptr
          %63 = arith.cmpi sgt, %arg2, %c16_i64 : i64
          %64 = arith.andi %62, %63 : i1
          scf.condition(%64) %arg2, %arg3 : i64, memref<?xf64>
        } do {
        ^bb0(%arg2: i64, %arg3: memref<?xf64>):
          %61 = arith.divsi %arg2, %c2_i64 : i64
          %62 = arith.addi %61, %c8_i64 : i64
          %63 = arith.addi %62, %35 : i64
          %64 = func.call @prk_malloc(%63) : (i64) -> memref<?xi8>
          %65 = polygeist.memref2pointer %64 : memref<?xi8> to !llvm.ptr
          %66 = polygeist.pointer2memref %65 : !llvm.ptr to memref<?xf64>
          scf.yield %61, %66 : i64, memref<?xf64>
        }
        %42 = polygeist.memref2pointer %41#1 : memref<?xf64> to !llvm.ptr
        %43 = llvm.icmp "eq" %42, %40 : !llvm.ptr
        scf.if %43 {
          %61 = llvm.mlir.addressof @str7 : !llvm.ptr
          %62 = llvm.getelementptr %61[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<46 x i8>
          %63 = llvm.call @printf(%62) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
          func.call @exit(%c1_i32) : (i32) -> ()
        }
        %44 = arith.divui %41#0, %c8_i64 : i64
        %45 = arith.index_cast %44 : i64 to index
        memref.store %cst_0, %41#1[%c0] : memref<?xf64>
        memref.store %cst_1, %41#1[%45] : memref<?xf64>
        %46 = memref.load %41#1[%c0] : memref<?xf64>
        %47 = arith.addf %46, %cst_0 : f64
        memref.store %47, %41#1[%c0] : memref<?xf64>
        %48 = memref.load %41#1[%45] : memref<?xf64>
        %49 = arith.addf %48, %cst_0 : f64
        memref.store %49, %41#1[%45] : memref<?xf64>
        arts.barrier
        arts.edt <single> <intranode> route(%c0_i32) {
          %61 = func.call @wtime() : () -> f64
          memref.store %61, %alloca[] : memref<f64>
        }
        arts.barrier
        %50 = arith.addi %19, %c1_i64 : i64
        %51 = arith.index_cast %50 : i64 to index
        scf.for %arg2 = %c1 to %51 step %c1 {
          %61 = memref.load %41#1[%c0] : memref<?xf64>
          %62 = arith.addf %61, %cst_0 : f64
          memref.store %62, %41#1[%c0] : memref<?xf64>
          %63 = memref.load %41#1[%45] : memref<?xf64>
          %64 = arith.addf %63, %cst_0 : f64
          memref.store %64, %41#1[%45] : memref<?xf64>
        }
        arts.barrier
        arts.edt <single> <intranode> route(%c0_i32) {
          %61 = func.call @wtime() : () -> f64
          %62 = memref.load %alloca[] : memref<f64>
          %63 = arith.subf %61, %62 : f64
          memref.store %63, %alloca[] : memref<f64>
        }
        arts.barrier
        func.call @bail_out(%c0_i32) : (i32) -> ()
        %52 = arith.addi %19, %c2_i64 : i64
        %53 = arith.sitofp %52 : i64 to f64
        %54 = arith.sitofp %50 : i64 to f64
        %55 = memref.load %41#1[%c0] : memref<?xf64>
        %56 = arith.subf %55, %53 : f64
        %57 = arith.cmpf oge, %56, %cst_1 : f64
        %58 = scf.if %57 -> (f64) {
          scf.yield %56 : f64
        } else {
          %61 = arith.negf %56 : f64
          scf.yield %61 : f64
        }
        %59 = arith.cmpf ogt, %58, %cst_2 : f64
        %60 = scf.if %59 -> (i1) {
          scf.yield %true : i1
        } else {
          %61 = memref.load %41#1[%45] : memref<?xf64>
          %62 = arith.subf %61, %54 : f64
          %63 = arith.cmpf oge, %62, %cst_1 : f64
          %64 = scf.if %63 -> (f64) {
            %66 = memref.load %41#1[%45] : memref<?xf64>
            %67 = arith.subf %66, %54 : f64
            scf.yield %67 : f64
          } else {
            %66 = memref.load %41#1[%45] : memref<?xf64>
            %67 = arith.subf %66, %54 : f64
            %68 = arith.negf %67 : f64
            scf.yield %68 : f64
          }
          %65 = arith.cmpf ogt, %64, %cst_2 : f64
          scf.yield %65 : i1
        }
        scf.if %60 {
          %61 = llvm.mlir.addressof @str9 : !llvm.ptr
          %62 = llvm.getelementptr %61[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<68 x i8>
          %63 = memref.load %41#1[%c0] : memref<?xf64>
          %64 = memref.load %41#1[%45] : memref<?xf64>
          %65 = llvm.call @printf(%62, %63, %64) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64) -> i32
          %66 = llvm.mlir.addressof @str10 : !llvm.ptr
          %67 = llvm.getelementptr %66[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<30 x i8>
          %68 = llvm.call @printf(%67, %53, %54) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64) -> i32
        }
      }
      %21 = llvm.mlir.addressof @str11 : !llvm.ptr
      %22 = llvm.getelementptr %21[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<20 x i8>
      %23 = llvm.call @printf(%22) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %24 = arith.muli %19, %0 : i64
      %25 = llvm.mlir.addressof @str12 : !llvm.ptr
      %26 = llvm.getelementptr %25[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<36 x i8>
      %27 = arith.sitofp %24 : i64 to f64
      %28 = memref.load %alloca[] : memref<f64>
      %29 = arith.divf %27, %28 : f64
      %30 = arith.mulf %29, %cst : f64
      %31 = llvm.call @printf(%26, %30, %28) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64) -> i32
      func.call @exit(%c0_i32) : (i32) -> ()
    }
    return %13 : i32
  }
  func.func private @atoi(memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @exit(i32) attributes {llvm.linkage = #llvm.linkage<external>}
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
  func.func private @sysconf(i32) -> i64 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @wtime() -> f64 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @prk_get_alignment() -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c8_i32 = arith.constant 8 : i32
    %false = arith.constant false
    %c107_i32 = arith.constant 107 : i32
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = llvm.mlir.addressof @str13 : !llvm.ptr
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
      %12 = llvm.mlir.addressof @str14 : !llvm.ptr
      %13 = llvm.mlir.addressof @str15 : !llvm.ptr
      %14 = llvm.mlir.addressof @str16 : !llvm.ptr
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
