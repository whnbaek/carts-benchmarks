module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:o-i64:64-i128:128-n32:64-S128", llvm.target_triple = "arm64-apple-macosx15.0.0", "polygeist.target-cpu" = "apple-m1", "polygeist.target-features" = "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"} {
  llvm.mlir.global internal constant @str18("(a & (~a+1)) == a\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str17("par-res-kern_general.h\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str16("prk_get_alignment\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str15("PRK_ALIGNMENT\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str14("0\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str13("Rate (synch/s): %e, time (s): %lf\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str12("Solution validates\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str11("Incorrect checksum: %d instead of %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str10("ERROR: Thread %d could not allocate space for private string\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str9("ERROR: Could not allocate space for concatenation string: %ld\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str8("ERROR: Could not allocate space for scramble string\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str7("ERROR: length of string %ld must be multiple of # threads: %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str6("ERROR: iterations must be >= 1 : %d \0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str5("ERROR: Invalid number of threads: %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str4("Usage: %s <# threads> <# iterations> <scramble string length>\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str3("OpenMP global synchronization test\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2("2.17\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("Parallel Research Kernels version %s\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @str0("27638472638746283742712311207892\00") {addr_space = 0 : i32}
  func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %true = arith.constant true
    %c57_i8 = arith.constant 57 : i8
    %c0_i8 = arith.constant 0 : i8
    %c32_i32 = arith.constant 32 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : i32
    %alloca = memref.alloca() : memref<i32>
    %1 = llvm.mlir.undef : f64
    %2 = llvm.mlir.addressof @str0 : !llvm.ptr
    %3 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<33 x i8>
    affine.store %c0_i32, %alloca[] : memref<i32>
    %4 = llvm.mlir.addressof @str1 : !llvm.ptr
    %5 = llvm.getelementptr %4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
    %6 = llvm.mlir.addressof @str2 : !llvm.ptr
    %7 = llvm.getelementptr %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<5 x i8>
    %8 = llvm.call @printf(%5, %7) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    %9 = llvm.mlir.addressof @str3 : !llvm.ptr
    %10 = llvm.getelementptr %9[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<36 x i8>
    %11 = llvm.call @printf(%10) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %12 = arith.cmpi ne, %arg0, %c4_i32 : i32
    scf.if %12 {
      %49 = llvm.mlir.addressof @str4 : !llvm.ptr
      %50 = llvm.getelementptr %49[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<63 x i8>
      %51 = affine.load %arg1[0] : memref<?xmemref<?xi8>>
      %52 = polygeist.memref2pointer %51 : memref<?xi8> to !llvm.ptr
      %53 = llvm.call @printf(%50, %52) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %13 = affine.load %arg1[1] : memref<?xmemref<?xi8>>
    %14 = call @atoi(%13) : (memref<?xi8>) -> i32
    %15 = arith.cmpi slt, %14, %c1_i32 : i32
    %16 = scf.if %15 -> (i1) {
      scf.yield %true : i1
    } else {
      %49 = arith.cmpi sgt, %14, %c512_i32 : i32
      scf.yield %49 : i1
    }
    scf.if %16 {
      %49 = llvm.mlir.addressof @str5 : !llvm.ptr
      %50 = llvm.getelementptr %49[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
      %51 = llvm.call @printf(%50, %14) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    call @omp_set_num_threads(%14) : (i32) -> ()
    %17 = affine.load %arg1[2] : memref<?xmemref<?xi8>>
    %18 = call @atoi(%17) : (memref<?xi8>) -> i32
    %19 = arith.cmpi slt, %18, %c1_i32 : i32
    scf.if %19 {
      %49 = llvm.mlir.addressof @str6 : !llvm.ptr
      %50 = llvm.getelementptr %49[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
      %51 = llvm.call @printf(%50, %18) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %20 = affine.load %arg1[3] : memref<?xmemref<?xi8>>
    %21 = call @atol(%20) : (memref<?xi8>) -> i64
    %22 = arith.extsi %14 : i32 to i64
    %23 = arith.cmpi slt, %21, %22 : i64
    %24 = scf.if %23 -> (i1) {
      scf.yield %true : i1
    } else {
      %49 = arith.remsi %21, %22 : i64
      %50 = arith.cmpi ne, %49, %c0_i64 : i64
      scf.yield %50 : i1
    }
    scf.if %24 {
      %49 = llvm.mlir.addressof @str7 : !llvm.ptr
      %50 = llvm.getelementptr %49[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<63 x i8>
      %51 = llvm.call @printf(%50, %21, %14) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64, i32) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %25 = arith.divsi %21, %22 : i64
    %26 = arith.addi %25, %c1_i64 : i64
    %27 = call @prk_malloc(%26) : (i64) -> memref<?xi8>
    %28 = llvm.mlir.zero : !llvm.ptr
    %29 = polygeist.memref2pointer %27 : memref<?xi8> to !llvm.ptr
    %30 = llvm.icmp "eq" %29, %28 : !llvm.ptr
    scf.if %30 {
      %49 = llvm.mlir.addressof @str8 : !llvm.ptr
      %50 = llvm.getelementptr %49[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<53 x i8>
      %51 = llvm.call @printf(%50) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %31 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
      %49 = arith.extsi %arg2 : i32 to i64
      %50 = arith.cmpi slt, %49, %25 : i64
      scf.condition(%50) %arg2 : i32
    } do {
    ^bb0(%arg2: i32):
      %49 = arith.index_cast %arg2 : i32 to index
      %50 = arith.remsi %arg2, %c32_i32 : i32
      %51 = arith.index_cast %50 : i32 to index
      %52 = arith.index_cast %51 : index to i64
      %53 = llvm.getelementptr %3[%52] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %54 = llvm.load %53 : !llvm.ptr -> i8
      memref.store %54, %27[%49] : memref<?xi8>
      %55 = arith.addi %arg2, %c1_i32 : i32
      scf.yield %55 : i32
    }
    %32 = arith.index_cast %25 : i64 to index
    affine.store %c0_i8, %27[symbol(%32)] : memref<?xi8>
    %33 = arith.addi %21, %c1_i64 : i64
    %34 = call @prk_malloc(%33) : (i64) -> memref<?xi8>
    %35 = polygeist.memref2pointer %34 : memref<?xi8> to !llvm.ptr
    %36 = llvm.icmp "eq" %35, %28 : !llvm.ptr
    scf.if %36 {
      %49 = llvm.mlir.addressof @str9 : !llvm.ptr
      %50 = llvm.getelementptr %49[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<63 x i8>
      %51 = arith.addi %21, %c1_i64 : i64
      %52 = llvm.call @printf(%50, %51) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %37 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
      %49 = arith.extsi %arg2 : i32 to i64
      %50 = arith.cmpi slt, %49, %21 : i64
      scf.condition(%50) %arg2 : i32
    } do {
    ^bb0(%arg2: i32):
      %49 = arith.index_cast %arg2 : i32 to index
      memref.store %c57_i8, %34[%49] : memref<?xi8>
      %50 = arith.addi %arg2, %c1_i32 : i32
      scf.yield %50 : i32
    }
    %38 = arith.index_cast %21 : i64 to index
    affine.store %c0_i8, %34[symbol(%38)] : memref<?xi8>
    omp.parallel   {
      %49 = func.call @omp_get_thread_num() : () -> i32
      %50 = func.call @prk_malloc(%26) : (i64) -> memref<?xi8>
      %51 = polygeist.memref2pointer %50 : memref<?xi8> to !llvm.ptr
      %52 = llvm.icmp "eq" %51, %28 : !llvm.ptr
      scf.if %52 {
        %65 = llvm.mlir.addressof @str10 : !llvm.ptr
        %66 = llvm.getelementptr %65[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<62 x i8>
        %67 = llvm.call @printf(%66, %49) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
        affine.store %c1_i32, %alloca[] : memref<i32>
      }
      %53 = affine.load %alloca[] : memref<i32>
      func.call @bail_out(%53) : (i32) -> ()
      %54 = polygeist.memref2pointer %50 : memref<?xi8> to !llvm.ptr
      %55 = func.call @__builtin_object_size(%50, %c1_i32) : (memref<?xi8>, i32) -> i64
      %56 = func.call @__strcpy_chk(%54, %29, %55) : (!llvm.ptr, !llvm.ptr, i64) -> memref<?xi8>
      func.call @bail_out(%53) : (i32) -> ()
      %57 = arith.extsi %49 : i32 to i64
      %58 = arith.muli %57, %25 : i64
      %59 = arith.index_cast %58 : i64 to index
      %60 = polygeist.subindex %34[%59] () : memref<?xi8> -> memref<?xi8>
      %61 = polygeist.memref2pointer %34 : memref<?xi8> to !llvm.ptr
      %62 = llvm.getelementptr %61[%58] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %63 = polygeist.memref2pointer %50 : memref<?xi8> to !llvm.ptr
      %64 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
        %65 = arith.cmpi slt, %arg2, %18 : i32
        scf.condition(%65) %arg2 : i32
      } do {
      ^bb0(%arg2: i32):
        %65 = func.call @__builtin_object_size(%60, %c1_i32) : (memref<?xi8>, i32) -> i64
        %66 = func.call @__strncpy_chk(%62, %63, %25, %65) : (!llvm.ptr, !llvm.ptr, i64, i64) -> memref<?xi8>
        %67 = scf.while (%arg3 = %c0_i32) : (i32) -> i32 {
          %69 = arith.extsi %arg3 : i32 to i64
          %70 = arith.cmpi slt, %69, %25 : i64
          scf.condition(%70) %arg3 : i32
        } do {
        ^bb0(%arg3: i32):
          %69 = arith.index_cast %arg3 : i32 to index
          %70 = arith.muli %arg3, %0 : i32
          %71 = arith.addi %49, %70 : i32
          %72 = arith.index_cast %71 : i32 to index
          %73 = memref.load %34[%72] : memref<?xi8>
          memref.store %73, %50[%69] : memref<?xi8>
          %74 = arith.addi %arg3, %c1_i32 : i32
          scf.yield %74 : i32
        }
        %68 = arith.addi %arg2, %c1_i32 : i32
        scf.yield %68 : i32
      }
      omp.terminator
    }
    %39:2 = scf.while (%arg2 = %c0_i32, %arg3 = %c0_i32) : (i32, i32) -> (i32, i32) {
      %49 = arith.extsi %arg3 : i32 to i64
      %50 = arith.cmpi slt, %49, %25 : i64
      scf.condition(%50) %arg2, %arg3 : i32, i32
    } do {
    ^bb0(%arg2: i32, %arg3: i32):
      %49 = arith.index_cast %arg3 : i32 to index
      %50 = memref.load %27[%49] : memref<?xi8>
      %51 = func.call @chartoi(%50) : (i8) -> i32
      %52 = arith.addi %arg2, %51 : i32
      %53 = arith.addi %arg3, %c1_i32 : i32
      scf.yield %52, %53 : i32, i32
    }
    %40 = polygeist.memref2pointer %34 : memref<?xi8> to !llvm.ptr
    %41:2 = scf.while (%arg2 = %c0_i64, %arg3 = %c0_i32) : (i64, i32) -> (i32, i64) {
      %49 = func.call @strlen(%40) : (!llvm.ptr) -> i64
      %50 = arith.cmpi slt, %arg2, %49 : i64
      scf.condition(%50) %arg3, %arg2 : i32, i64
    } do {
    ^bb0(%arg2: i32, %arg3: i64):
      %49 = arith.index_cast %arg3 : i64 to index
      %50 = memref.load %34[%49] : memref<?xi8>
      %51 = func.call @chartoi(%50) : (i8) -> i32
      %52 = arith.addi %arg2, %51 : i32
      %53 = arith.addi %arg3, %c1_i64 : i64
      scf.yield %53, %52 : i64, i32
    }
    %42 = arith.muli %39#0, %0 : i32
    %43 = arith.cmpi ne, %41#0, %42 : i32
    scf.if %43 {
      %49 = llvm.mlir.addressof @str11 : !llvm.ptr
      %50 = llvm.getelementptr %49[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
      %51 = llvm.call @printf(%50, %41#0, %42) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    } else {
      %49 = llvm.mlir.addressof @str12 : !llvm.ptr
      %50 = llvm.getelementptr %49[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<20 x i8>
      %51 = llvm.call @printf(%50) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    }
    %44 = llvm.mlir.addressof @str13 : !llvm.ptr
    %45 = llvm.getelementptr %44[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<35 x i8>
    %46 = arith.sitofp %18 : i32 to f64
    %47 = arith.divf %46, %1 : f64
    %48 = llvm.call @printf(%45, %47, %1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64) -> i32
    call @exit(%c0_i32) : (i32) -> ()
    return %0 : i32
  }
  func.func private @exit(i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @atoi(memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @omp_set_num_threads(i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @atol(memref<?xi8>) -> i64 attributes {llvm.linkage = #llvm.linkage<external>}
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
  func.func private @omp_get_thread_num() -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @bail_out(i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @__builtin_object_size(memref<?xi8>, i32) -> i64 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @__strcpy_chk(!llvm.ptr, !llvm.ptr, i64) -> memref<?xi8>
  func.func private @__strncpy_chk(!llvm.ptr, !llvm.ptr, i64, i64) -> memref<?xi8>
  func.func private @chartoi(%arg0: i8) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %alloca = memref.alloca() : memref<2xi8>
    %0 = llvm.mlir.addressof @str14 : !llvm.ptr
    %1 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i8>
    %2 = llvm.load %1 : !llvm.ptr -> i8
    affine.store %2, %alloca[0] : memref<2xi8>
    %3 = llvm.getelementptr %0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i8>
    %4 = llvm.load %3 : !llvm.ptr -> i8
    affine.store %4, %alloca[1] : memref<2xi8>
    affine.store %arg0, %alloca[0] : memref<2xi8>
    %cast = memref.cast %alloca : memref<2xi8> to memref<?xi8>
    %5 = call @atoi(%cast) : (memref<?xi8>) -> i32
    return %5 : i32
  }
  func.func private @strlen(!llvm.ptr) -> i64
  func.func private @prk_get_alignment() -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c8_i32 = arith.constant 8 : i32
    %false = arith.constant false
    %c107_i32 = arith.constant 107 : i32
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = llvm.mlir.addressof @str15 : !llvm.ptr
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
      %12 = llvm.mlir.addressof @str16 : !llvm.ptr
      %13 = llvm.mlir.addressof @str17 : !llvm.ptr
      %14 = llvm.mlir.addressof @str18 : !llvm.ptr
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
