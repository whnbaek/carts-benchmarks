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
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 3.000000e+00 : f64
    %c0_i64 = arith.constant 0 : i64
    %0 = scf.while (%arg4 = %c0_i64) : (i64) -> i64 {
      %1 = arith.cmpi slt, %arg4, %arg3 : i64
      scf.condition(%1) %arg4 : i64
    } do {
    ^bb0(%arg4: i64):
      %1 = arith.index_cast %arg4 : i64 to index
      %2 = memref.load %arg1[%1] : memref<?xf64>
      %3 = memref.load %arg2[%1] : memref<?xf64>
      %4 = arith.mulf %3, %cst : f64
      %5 = arith.addf %2, %4 : f64
      %6 = memref.load %arg0[%1] : memref<?xf64>
      %7 = arith.addf %6, %5 : f64
      memref.store %7, %arg0[%1] : memref<?xf64>
      %8 = arith.addi %arg4, %c1_i64 : i64
      scf.yield %8 : i64
    }
    return
  }
  func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 8.000000e+00 : f64
    %true = arith.constant true
    %c16_i64 = arith.constant 16 : i64
    %c8_i64 = arith.constant 8 : i64
    %cst_0 = arith.constant 9.9999999999999995E-7 : f64
    %cst_1 = arith.constant 1.000000e+00 : f64
    %c2_i64 = arith.constant 2 : i64
    %c29_i32 = arith.constant 29 : i32
    %cst_2 = arith.constant 0.000000e+00 : f64
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_3 = arith.constant 9.9999999999999995E-8 : f64
    %0 = llvm.mlir.undef : i64
    %1 = llvm.mlir.undef : i32
    %alloca = memref.alloca() : memref<f64>
    %2 = llvm.mlir.undef : f64
    affine.store %2, %alloca[] : memref<f64>
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
      %16 = affine.load %arg1[0] : memref<?xmemref<?xi8>>
      %17 = polygeist.memref2pointer %16 : memref<?xi8> to !llvm.ptr
      %18 = llvm.call @printf(%15, %17) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    }
    scf.if %12 {
      %14 = affine.load %arg1[1] : memref<?xmemref<?xi8>>
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
      %18 = affine.load %arg1[2] : memref<?xmemref<?xi8>>
      %19 = func.call @atol(%18) : (memref<?xi8>) -> i64
      %20 = arith.cmpi slt, %19, %c1_i64 : i64
      scf.if %20 {
        %32 = llvm.mlir.addressof @str5 : !llvm.ptr
        %33 = llvm.getelementptr %32[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<39 x i8>
        %34 = llvm.call @printf(%33, %19) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
        func.call @exit(%c1_i32) : (i32) -> ()
      }
      func.call @omp_set_num_threads(%15) : (i32) -> ()
      omp.parallel   {
        %alloca_4 = memref.alloca() : memref<memref<?xf64>>
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
        %41 = polygeist.typeSize memref<?xi8> : index
        %42 = arith.index_cast %41 : index to i64
        %43:2 = scf.while (%arg2 = %32, %arg3 = %39) : (i64, memref<?xf64>) -> (i64, memref<?xf64>) {
          %67 = polygeist.memref2pointer %arg3 : memref<?xf64> to !llvm.ptr
          %68 = llvm.icmp "eq" %67, %40 : !llvm.ptr
          %69 = arith.cmpi sgt, %arg2, %c16_i64 : i64
          %70 = arith.andi %68, %69 : i1
          scf.condition(%70) %arg2, %arg3 : i64, memref<?xf64>
        } do {
        ^bb0(%arg2: i64, %arg3: memref<?xf64>):
          %67 = arith.divsi %arg2, %c2_i64 : i64
          %68 = arith.addi %67, %c8_i64 : i64
          %69 = arith.addi %68, %42 : i64
          %70 = func.call @prk_malloc(%69) : (i64) -> memref<?xi8>
          %71 = polygeist.memref2pointer %70 : memref<?xi8> to !llvm.ptr
          %72 = polygeist.pointer2memref %71 : !llvm.ptr to memref<?xf64>
          scf.yield %67, %72 : i64, memref<?xf64>
        }
        %44 = polygeist.memref2pointer %43#1 : memref<?xf64> to !llvm.ptr
        %45 = llvm.mlir.zero : !llvm.ptr
        %46 = llvm.icmp "eq" %44, %45 : !llvm.ptr
        scf.if %46 {
          %67 = llvm.mlir.addressof @str7 : !llvm.ptr
          %68 = llvm.getelementptr %67[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<46 x i8>
          %69 = llvm.call @printf(%68) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
          func.call @exit(%c1_i32) : (i32) -> ()
        }
        %47 = arith.divui %43#0, %c8_i64 : i64
        %48 = arith.index_cast %47 : i64 to index
        affine.store %cst_1, %43#1[0] : memref<?xf64>
        memref.store %cst_2, %43#1[%48] : memref<?xf64>
        %49 = affine.load %43#1[0] : memref<?xf64>
        %50 = arith.addf %49, %cst_1 : f64
        affine.store %50, %43#1[0] : memref<?xf64>
        %51 = memref.load %43#1[%48] : memref<?xf64>
        %52 = arith.addf %51, %cst_1 : f64
        memref.store %52, %43#1[%48] : memref<?xf64>
        omp.barrier
        omp.master {
          %67 = func.call @wtime() : () -> f64
          affine.store %67, %alloca[] : memref<f64>
          omp.terminator
        }
        omp.barrier
        %53 = scf.while (%arg2 = %c1_i64) : (i64) -> i64 {
          %67 = arith.cmpi sle, %arg2, %19 : i64
          scf.condition(%67) %arg2 : i64
        } do {
        ^bb0(%arg2: i64):
          %67 = affine.load %43#1[0] : memref<?xf64>
          %68 = arith.addf %67, %cst_1 : f64
          affine.store %68, %43#1[0] : memref<?xf64>
          %69 = memref.load %43#1[%48] : memref<?xf64>
          %70 = arith.addf %69, %cst_1 : f64
          memref.store %70, %43#1[%48] : memref<?xf64>
          %71 = arith.addi %arg2, %c1_i64 : i64
          scf.yield %71 : i64
        }
        omp.barrier
        omp.master {
          %67 = func.call @wtime() : () -> f64
          %68 = affine.load %alloca[] : memref<f64>
          %69 = arith.subf %67, %68 : f64
          affine.store %69, %alloca[] : memref<f64>
          omp.terminator
        }
        omp.barrier
        %54:2 = scf.while (%arg2 = %cst_2, %arg3 = %c0_i64) : (f64, i64) -> (f64, i64) {
          %67 = arith.cmpi sle, %arg3, %19 : i64
          scf.condition(%67) %arg2, %arg3 : f64, i64
        } do {
        ^bb0(%arg2: f64, %arg3: i64):
          %67 = arith.addf %arg2, %cst : f64
          %68 = arith.addi %arg3, %c1_i64 : i64
          scf.yield %67, %68 : f64, i64
        }
        %55:2 = scf.while (%arg2 = %c0_i32, %arg3 = %c0_i64) : (i32, i64) -> (i32, i64) {
          %67 = arith.cmpi slt, %arg3, %c0_i64 : i64
          scf.condition(%67) %arg2, %arg3 : i32, i64
        } do {
        ^bb0(%arg2: i32, %arg3: i64):
          %67 = affine.load %alloca_4[] : memref<memref<?xf64>>
          %68 = arith.index_cast %arg3 : i64 to index
          %69 = memref.load %67[%68] : memref<?xf64>
          %70 = arith.subf %69, %54#0 : f64
          %71 = arith.cmpf oge, %70, %cst_2 : f64
          %72 = scf.if %71 -> (f64) {
            %79 = memref.load %67[%68] : memref<?xf64>
            %80 = arith.subf %79, %54#0 : f64
            scf.yield %80 : f64
          } else {
            %79 = memref.load %67[%68] : memref<?xf64>
            %80 = arith.subf %79, %54#0 : f64
            %81 = arith.negf %80 : f64
            scf.yield %81 : f64
          }
          %73 = arith.cmpf ogt, %72, %cst_3 : f64
          %74 = arith.extui %73 : i1 to i32
          %75 = arith.cmpi sgt, %74, %arg2 : i32
          %76 = scf.if %75 -> (i32) {
            %79 = memref.load %67[%68] : memref<?xf64>
            %80 = arith.subf %79, %54#0 : f64
            %81 = arith.cmpf oge, %80, %cst_2 : f64
            %82 = scf.if %81 -> (f64) {
              %85 = memref.load %67[%68] : memref<?xf64>
              %86 = arith.subf %85, %54#0 : f64
              scf.yield %86 : f64
            } else {
              %85 = memref.load %67[%68] : memref<?xf64>
              %86 = arith.subf %85, %54#0 : f64
              %87 = arith.negf %86 : f64
              scf.yield %87 : f64
            }
            %83 = arith.cmpf ogt, %82, %cst_3 : f64
            %84 = arith.extui %83 : i1 to i32
            scf.yield %84 : i32
          } else {
            scf.yield %arg2 : i32
          }
          %77 = arith.addi %arg2, %76 : i32
          %78 = arith.addi %arg3, %c1_i64 : i64
          scf.yield %77, %78 : i32, i64
        }
        %56 = arith.cmpi sgt, %55#0, %c0_i32 : i32
        scf.if %56 {
          %67 = llvm.mlir.addressof @str8 : !llvm.ptr
          %68 = llvm.getelementptr %67[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<53 x i8>
          %69 = func.call @omp_get_thread_num() : () -> i32
          %70 = llvm.call @printf(%68, %69) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
        }
        func.call @bail_out(%55#0) : (i32) -> ()
        %57 = arith.addi %19, %c2_i64 : i64
        %58 = arith.sitofp %57 : i64 to f64
        %59 = arith.addi %19, %c1_i64 : i64
        %60 = arith.sitofp %59 : i64 to f64
        %61 = affine.load %43#1[0] : memref<?xf64>
        %62 = arith.subf %61, %58 : f64
        %63 = arith.cmpf oge, %62, %cst_2 : f64
        %64 = scf.if %63 -> (f64) {
          %67 = arith.subf %61, %58 : f64
          scf.yield %67 : f64
        } else {
          %67 = arith.subf %61, %58 : f64
          %68 = arith.negf %67 : f64
          scf.yield %68 : f64
        }
        %65 = arith.cmpf ogt, %64, %cst_3 : f64
        %66 = scf.if %65 -> (i1) {
          scf.yield %true : i1
        } else {
          %67 = memref.load %43#1[%48] : memref<?xf64>
          %68 = arith.subf %67, %60 : f64
          %69 = arith.cmpf oge, %68, %cst_2 : f64
          %70 = scf.if %69 -> (f64) {
            %72 = memref.load %43#1[%48] : memref<?xf64>
            %73 = arith.subf %72, %60 : f64
            scf.yield %73 : f64
          } else {
            %72 = memref.load %43#1[%48] : memref<?xf64>
            %73 = arith.subf %72, %60 : f64
            %74 = arith.negf %73 : f64
            scf.yield %74 : f64
          }
          %71 = arith.cmpf ogt, %70, %cst_3 : f64
          scf.yield %71 : i1
        }
        scf.if %66 {
          %67 = llvm.mlir.addressof @str9 : !llvm.ptr
          %68 = llvm.getelementptr %67[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<68 x i8>
          %69 = affine.load %43#1[0] : memref<?xf64>
          %70 = memref.load %43#1[%48] : memref<?xf64>
          %71 = llvm.call @printf(%68, %69, %70) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64) -> i32
          %72 = llvm.mlir.addressof @str10 : !llvm.ptr
          %73 = llvm.getelementptr %72[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<30 x i8>
          %74 = llvm.call @printf(%73, %58, %60) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64) -> i32
        }
        omp.terminator
      }
      %21 = llvm.mlir.addressof @str11 : !llvm.ptr
      %22 = llvm.getelementptr %21[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<20 x i8>
      %23 = llvm.call @printf(%22) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %24 = arith.muli %19, %0 : i64
      %25 = llvm.mlir.addressof @str12 : !llvm.ptr
      %26 = llvm.getelementptr %25[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<36 x i8>
      %27 = arith.sitofp %24 : i64 to f64
      %28 = affine.load %alloca[] : memref<f64>
      %29 = arith.divf %27, %28 : f64
      %30 = arith.mulf %29, %cst_0 : f64
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
  func.func private @sysconf(i32) -> i64 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @wtime() -> f64 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @omp_get_thread_num() -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
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
