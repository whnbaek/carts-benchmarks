module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:o-i64:64-i128:128-n32:64-S128", llvm.target_triple = "arm64-apple-macosx15.0.0", "polygeist.target-cpu" = "apple-m1", "polygeist.target-features" = "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"} {
  llvm.mlir.global internal constant @str23("(a & (~a+1)) == a\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str22("par-res-kern_general.h\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str21("prk_get_alignment\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str20("PRK_ALIGNMENT\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str19("ERROR: array sum = %d, reference value = %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str18("Rate (Mops/s) without branches: %lf time (s): %lf\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str17("Rate (Mops/s) with branches:    %lf time (s): %lf\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str16("Solution validates\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str15("ERROR: Thread %d failed to allocate space for vector\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str14("no_vector, or ins_heavy\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str13("Wrong branch type: %s; choose vector_stop, vector_go, \00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str12("ins_heavy\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str11("no_vector\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str10("vector_go\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str9("vector_stop\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str8("ERROR: loop length must be >= 1 : %d \0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str7("ERROR: Iterations must be positive and even : %d \0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str6("ERROR: Invalid number of threads: %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str5("branching type: vector_go, vector_stop, no_vector, ins_heavy\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str4("<branching type>\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str3("Usage:     %s <# threads> <# iterations> <vector length>\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2("OpenMP Branching Bonanza\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("2.17\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str0("Parallel Research Kernels version %s\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c-8_i32 = arith.constant -8 : i32
    %true = arith.constant true
    %c4_i64 = arith.constant 4 : i64
    %cst = arith.constant 1.000000e+06 : f64
    %c8_i32 = arith.constant 8 : i32
    %c7_i32 = arith.constant 7 : i32
    %c3_i32 = arith.constant 3 : i32
    %c99_i32 = arith.constant 99 : i32
    %c88_i32 = arith.constant 88 : i32
    %c77_i32 = arith.constant 77 : i32
    %c66_i32 = arith.constant 66 : i32
    %c2_i32 = arith.constant 2 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    %c5_i32 = arith.constant 5 : i32
    %alloca = memref.alloca() : memref<1xi32>
    %0 = llvm.mlir.undef : i32
    memref.store %0, %alloca[%c0] : memref<1xi32>
    %alloca_0 = memref.alloca() : memref<1xi32>
    memref.store %0, %alloca_0[%c0] : memref<1xi32>
    %alloca_1 = memref.alloca() : memref<i32>
    %alloca_2 = memref.alloca() : memref<i32>
    %1 = llvm.mlir.undef : f64
    memref.store %c0_i32, %alloca_2[] : memref<i32>
    memref.store %c0_i32, %alloca_1[] : memref<i32>
    %2 = llvm.mlir.addressof @str0 : !llvm.ptr
    %3 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
    %4 = llvm.mlir.addressof @str1 : !llvm.ptr
    %5 = llvm.getelementptr %4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<5 x i8>
    %6 = llvm.call @printf(%3, %5) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    %7 = llvm.mlir.addressof @str2 : !llvm.ptr
    %8 = llvm.getelementptr %7[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<26 x i8>
    %9 = llvm.call @printf(%8) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %10 = arith.cmpi ne, %arg0, %c5_i32 : i32
    scf.if %10 {
      %36 = llvm.mlir.addressof @str3 : !llvm.ptr
      %37 = llvm.getelementptr %36[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<57 x i8>
      %38 = memref.load %arg1[%c0] : memref<?xmemref<?xi8>>
      %39 = polygeist.memref2pointer %38 : memref<?xi8> to !llvm.ptr
      %40 = llvm.call @printf(%37, %39) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
      %41 = llvm.mlir.addressof @str4 : !llvm.ptr
      %42 = llvm.getelementptr %41[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<18 x i8>
      %43 = llvm.call @printf(%42) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %44 = llvm.mlir.addressof @str5 : !llvm.ptr
      %45 = llvm.getelementptr %44[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<62 x i8>
      %46 = llvm.call @printf(%45) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %11 = memref.load %arg1[%c1] : memref<?xmemref<?xi8>>
    %12 = call @atoi(%11) : (memref<?xi8>) -> i32
    %13 = arith.cmpi slt, %12, %c1_i32 : i32
    %14 = scf.if %13 -> (i1) {
      scf.yield %true : i1
    } else {
      %36 = arith.cmpi sgt, %12, %c512_i32 : i32
      scf.yield %36 : i1
    }
    scf.if %14 {
      %36 = llvm.mlir.addressof @str6 : !llvm.ptr
      %37 = llvm.getelementptr %36[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
      %38 = llvm.call @printf(%37, %12) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    call @omp_set_num_threads(%12) : (i32) -> ()
    %15 = memref.load %arg1[%c2] : memref<?xmemref<?xi8>>
    %16 = call @atoi(%15) : (memref<?xi8>) -> i32
    %17 = arith.cmpi slt, %16, %c1_i32 : i32
    %18 = scf.if %17 -> (i1) {
      scf.yield %true : i1
    } else {
      %36 = arith.remsi %16, %c2_i32 : i32
      %37 = arith.cmpi eq, %36, %c1_i32 : i32
      scf.yield %37 : i1
    }
    scf.if %18 {
      %36 = llvm.mlir.addressof @str7 : !llvm.ptr
      %37 = llvm.getelementptr %36[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<51 x i8>
      %38 = llvm.call @printf(%37, %16) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %19 = memref.load %arg1[%c3] : memref<?xmemref<?xi8>>
    %20 = call @atoi(%19) : (memref<?xi8>) -> i32
    %21 = arith.cmpi slt, %20, %c1_i32 : i32
    scf.if %21 {
      %36 = llvm.mlir.addressof @str8 : !llvm.ptr
      %37 = llvm.getelementptr %36[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<39 x i8>
      %38 = llvm.call @printf(%37, %20) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %22 = memref.load %arg1[%c4] : memref<?xmemref<?xi8>>
    %23 = llvm.mlir.addressof @str9 : !llvm.ptr
    %24 = polygeist.pointer2memref %23 : !llvm.ptr to memref<?xi8>
    %25 = call @strcmp(%22, %24) : (memref<?xi8>, memref<?xi8>) -> i32
    %26 = arith.cmpi eq, %25, %c0_i32 : i32
    %27 = scf.if %26 -> (i32) {
      scf.yield %c66_i32 : i32
    } else {
      %36 = llvm.mlir.addressof @str10 : !llvm.ptr
      %37 = polygeist.pointer2memref %36 : !llvm.ptr to memref<?xi8>
      %38 = func.call @strcmp(%22, %37) : (memref<?xi8>, memref<?xi8>) -> i32
      %39 = arith.cmpi eq, %38, %c0_i32 : i32
      %40 = scf.if %39 -> (i32) {
        scf.yield %c77_i32 : i32
      } else {
        %41 = llvm.mlir.addressof @str11 : !llvm.ptr
        %42 = polygeist.pointer2memref %41 : !llvm.ptr to memref<?xi8>
        %43 = func.call @strcmp(%22, %42) : (memref<?xi8>, memref<?xi8>) -> i32
        %44 = arith.cmpi eq, %43, %c0_i32 : i32
        %45 = scf.if %44 -> (i32) {
          scf.yield %c88_i32 : i32
        } else {
          %46 = llvm.mlir.addressof @str12 : !llvm.ptr
          %47 = polygeist.pointer2memref %46 : !llvm.ptr to memref<?xi8>
          %48 = func.call @strcmp(%22, %47) : (memref<?xi8>, memref<?xi8>) -> i32
          %49 = arith.cmpi ne, %48, %c0_i32 : i32
          %50 = arith.select %49, %0, %c99_i32 : i32
          scf.if %49 {
            %51 = llvm.mlir.addressof @str13 : !llvm.ptr
            %52 = llvm.getelementptr %51[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<55 x i8>
            %53 = polygeist.memref2pointer %22 : memref<?xi8> to !llvm.ptr
            %54 = llvm.call @printf(%52, %53) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
            %55 = llvm.mlir.addressof @str14 : !llvm.ptr
            %56 = llvm.getelementptr %55[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<25 x i8>
            %57 = llvm.call @printf(%56) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
            func.call @exit(%c1_i32) : (i32) -> ()
          }
          scf.yield %50 : i32
        }
        scf.yield %45 : i32
      }
      scf.yield %40 : i32
    }
    arts.edt <parallel> <internode> route(%c0_i32) {
      %36 = memref.load %alloca_1[] : memref<i32>
      func.call @bail_out(%36) : (i32) -> ()
      %37 = arts.get_current_node -> i32
      %38 = arith.muli %20, %c2_i32 : i32
      %39 = arith.extsi %38 : i32 to i64
      %40 = arith.muli %39, %c4_i64 : i64
      %41 = func.call @prk_malloc(%40) : (i64) -> memref<?xi8>
      %42 = polygeist.memref2pointer %41 : memref<?xi8> to !llvm.ptr
      %43 = polygeist.pointer2memref %42 : !llvm.ptr to memref<?xi32>
      %44 = llvm.mlir.zero : !llvm.ptr
      %45 = llvm.icmp "eq" %42, %44 : !llvm.ptr
      %46 = arith.select %45, %c1_i32, %36 : i32
      scf.if %45 {
        %49 = llvm.mlir.addressof @str15 : !llvm.ptr
        %50 = llvm.getelementptr %49[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<54 x i8>
        %51 = llvm.call @printf(%50, %37) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
        memref.store %c1_i32, %alloca_1[] : memref<i32>
      }
      func.call @bail_out(%46) : (i32) -> ()
      %47 = arith.index_cast %20 : i32 to index
      scf.for %arg2 = %c0 to %47 step %c1 {
        %49 = arith.index_cast %arg2 : index to i32
        %50 = arith.andi %49, %c7_i32 : i32
        %51 = arith.subi %c3_i32, %50 : i32
        %52 = llvm.getelementptr %42[%49] : (!llvm.ptr, i32) -> !llvm.ptr, i32
        llvm.store %51, %52 : i32, !llvm.ptr
        %53 = arith.addi %arg2, %47 : index
        %54 = arith.index_cast %53 : index to i32
        %55 = llvm.getelementptr %42[%54] : (!llvm.ptr, i32) -> !llvm.ptr, i32
        llvm.store %49, %55 : i32, !llvm.ptr
      }
      scf.execute_region {
        cf.switch %27 : i32, [
          default: ^bb2,
          99: ^bb1
        ]
      ^bb1:  // pred: ^bb0
        %cast = memref.cast %alloca_0 : memref<1xi32> to memref<?xi32>
        %cast_3 = memref.cast %alloca : memref<1xi32> to memref<?xi32>
        %49 = func.call @fill_vec(%43, %20, %16, %c1_i32, %cast, %cast_3) : (memref<?xi32>, i32, i32, i32, memref<?xi32>, memref<?xi32>) -> i32
        cf.br ^bb2
      ^bb2:  // 2 preds: ^bb0, ^bb1
        scf.yield
      }
      scf.execute_region {
        cf.switch %27 : i32, [
          default: ^bb2,
          99: ^bb1
        ]
      ^bb1:  // pred: ^bb0
        %cast = memref.cast %alloca_0 : memref<1xi32> to memref<?xi32>
        %cast_3 = memref.cast %alloca : memref<1xi32> to memref<?xi32>
        %49 = func.call @fill_vec(%43, %20, %16, %c0_i32, %cast, %cast_3) : (memref<?xi32>, i32, i32, i32, memref<?xi32>, memref<?xi32>) -> i32
        cf.br ^bb2
      ^bb2:  // 2 preds: ^bb0, ^bb1
        scf.yield
      }
      memref.store %c0_i32, %alloca_2[] : memref<i32>
      %48 = scf.for %arg2 = %c0 to %47 step %c1 iter_args(%arg3 = %c0_i32) -> (i32) {
        %49 = arith.index_cast %arg2 : index to i32
        %50 = llvm.getelementptr %42[%49] : (!llvm.ptr, i32) -> !llvm.ptr, i32
        %51 = llvm.load %50 : !llvm.ptr -> i32
        %52 = arith.addi %arg3, %51 : i32
        memref.store %52, %alloca_2[] : memref<i32>
        scf.yield %52 : i32
      }
    }
    %28 = arith.remsi %20, %c8_i32 : i32
    %29 = arith.addi %28, %c-8_i32 : i32
    %30 = arith.muli %28, %29 : i32
    %31 = arith.addi %30, %20 : i32
    %32 = arith.divsi %31, %c2_i32 : i32
    %33 = arith.muli %32, %0 : i32
    %34 = memref.load %alloca_2[] : memref<i32>
    %35 = arith.cmpi eq, %34, %33 : i32
    scf.if %35 {
      %36 = llvm.mlir.addressof @str16 : !llvm.ptr
      %37 = llvm.getelementptr %36[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<20 x i8>
      %38 = llvm.call @printf(%37) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %39 = llvm.mlir.addressof @str17 : !llvm.ptr
      %40 = llvm.getelementptr %39[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<51 x i8>
      %41 = arith.mulf %1, %cst : f64
      %42 = arith.divf %1, %41 : f64
      %43 = llvm.call @printf(%40, %42, %1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64) -> i32
      %44 = llvm.mlir.addressof @str18 : !llvm.ptr
      %45 = llvm.getelementptr %44[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<51 x i8>
      %46 = llvm.call @printf(%45, %42, %1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64) -> i32
    } else {
      %36 = llvm.mlir.addressof @str19 : !llvm.ptr
      %37 = llvm.getelementptr %36[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<45 x i8>
      %38 = llvm.call @printf(%37, %34, %33) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32) -> i32
    }
    call @exit(%c0_i32) : (i32) -> ()
    return %0 : i32
  }
  func.func private @exit(i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @atoi(memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @omp_set_num_threads(i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @strcmp(memref<?xi8>, memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @bail_out(i32) attributes {llvm.linkage = #llvm.linkage<external>}
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
  func.func private @fill_vec(memref<?xi32>, i32, i32, i32, memref<?xi32>, memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @prk_get_alignment() -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c8_i32 = arith.constant 8 : i32
    %false = arith.constant false
    %c107_i32 = arith.constant 107 : i32
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = llvm.mlir.addressof @str20 : !llvm.ptr
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
      %12 = llvm.mlir.addressof @str21 : !llvm.ptr
      %13 = llvm.mlir.addressof @str22 : !llvm.ptr
      %14 = llvm.mlir.addressof @str23 : !llvm.ptr
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
