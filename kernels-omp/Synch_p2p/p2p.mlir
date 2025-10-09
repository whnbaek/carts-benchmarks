module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:o-i64:64-i128:128-n32:64-S128", llvm.target_triple = "arm64-apple-macosx15.0.0", "polygeist.target-cpu" = "apple-m1", "polygeist.target-features" = "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"} {
  llvm.mlir.global internal constant @str18("(a & (~a+1)) == a\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str17("par-res-kern_general.h\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str16("prk_get_alignment\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str15("PRK_ALIGNMENT\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str14("Rate (MFlops/s): %lf Avg time (s): %lf\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str13("Solution validates\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str12("ERROR: checksum %lf does not match verification value %lf\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str11("ERROR: COuld not allocate space for synchronization flags\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str10("ERROR: Could not allocate space for array of slice boundaries\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str9("First grid dimension %ld smaller than number of threads requested: %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str8("ERROR: Could not allocate space for vector: %ld\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str7("ERROR: grid dimensions must be positive: %ld, %ld \0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str6("ERROR: iterations must be >= 1 : %d \0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str5("ERROR: Invalid number of threads: %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str4("<second array dimension> [group factor]\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str3("Usage: %s <# threads> <# iterations> <first array dimension> \00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2("OpenMP pipeline execution on 2D grid\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("2.17\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str0("Parallel Research Kernels version %s\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1_i32 = arith.constant -1 : i32
    %c-2_i64 = arith.constant -2 : i64
    %c-1_i64 = arith.constant -1 : i64
    %true = arith.constant true
    %c64_i64 = arith.constant 64 : i64
    %c8_i64 = arith.constant 8 : i64
    %c4_i64 = arith.constant 4 : i64
    %cst = arith.constant 2.000000e-06 : f64
    %cst_0 = arith.constant -1.000000e+00 : f64
    %c16_i32 = arith.constant 16 : i32
    %cst_1 = arith.constant 0.000000e+00 : f64
    %c2_i32 = arith.constant 2 : i32
    %c1_i64 = arith.constant 1 : i64
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    %false = arith.constant false
    %c6_i32 = arith.constant 6 : i32
    %c5_i32 = arith.constant 5 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant 1.000000e-08 : f64
    %0 = llvm.mlir.undef : i32
    %1 = llvm.mlir.undef : f64
    %2 = llvm.mlir.addressof @str0 : !llvm.ptr
    %3 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
    %4 = llvm.mlir.addressof @str1 : !llvm.ptr
    %5 = llvm.getelementptr %4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<5 x i8>
    %6 = llvm.call @printf(%3, %5) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    %7 = llvm.mlir.addressof @str2 : !llvm.ptr
    %8 = llvm.getelementptr %7[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
    %9 = llvm.call @printf(%8) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %10 = arith.cmpi ne, %arg0, %c5_i32 : i32
    %11 = arith.cmpi ne, %arg0, %c6_i32 : i32
    %12 = arith.andi %10, %11 : i1
    %13 = arith.select %12, %c1_i32, %0 : i32
    scf.if %12 {
      %14 = llvm.mlir.addressof @str3 : !llvm.ptr
      %15 = llvm.getelementptr %14[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<62 x i8>
      %16 = affine.load %arg1[0] : memref<?xmemref<?xi8>>
      %17 = polygeist.memref2pointer %16 : memref<?xi8> to !llvm.ptr
      %18 = llvm.call @printf(%15, %17) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
      %19 = llvm.mlir.addressof @str4 : !llvm.ptr
      %20 = llvm.getelementptr %19[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<41 x i8>
      %21 = llvm.call @printf(%20) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    } else {
      %14 = affine.load %arg1[1] : memref<?xmemref<?xi8>>
      %15 = func.call @atoi(%14) : (memref<?xi8>) -> i32
      %16 = arith.cmpi slt, %15, %c1_i32 : i32
      %17 = scf.if %16 -> (i1) {
        scf.yield %true : i1
      } else {
        %90 = arith.cmpi sgt, %15, %c512_i32 : i32
        scf.yield %90 : i1
      }
      scf.if %17 {
        %90 = llvm.mlir.addressof @str5 : !llvm.ptr
        %91 = llvm.getelementptr %90[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
        %92 = llvm.call @printf(%91, %15) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
        func.call @exit(%c1_i32) : (i32) -> ()
      }
      func.call @omp_set_num_threads(%15) : (i32) -> ()
      %18 = affine.load %arg1[2] : memref<?xmemref<?xi8>>
      %19 = func.call @atoi(%18) : (memref<?xi8>) -> i32
      %20 = arith.cmpi slt, %19, %c1_i32 : i32
      scf.if %20 {
        %90 = llvm.mlir.addressof @str6 : !llvm.ptr
        %91 = llvm.getelementptr %90[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
        %92 = llvm.call @printf(%91, %19) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
        func.call @exit(%c1_i32) : (i32) -> ()
      }
      %21 = affine.load %arg1[3] : memref<?xmemref<?xi8>>
      %22 = func.call @atol(%21) : (memref<?xi8>) -> i64
      %23 = affine.load %arg1[4] : memref<?xmemref<?xi8>>
      %24 = func.call @atol(%23) : (memref<?xi8>) -> i64
      %25 = arith.cmpi slt, %22, %c1_i64 : i64
      %26 = scf.if %25 -> (i1) {
        scf.yield %true : i1
      } else {
        %90 = arith.cmpi slt, %24, %c1_i64 : i64
        scf.yield %90 : i1
      }
      scf.if %26 {
        %90 = llvm.mlir.addressof @str7 : !llvm.ptr
        %91 = llvm.getelementptr %90[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<52 x i8>
        %92 = llvm.call @printf(%91, %22, %24) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64, i64) -> i32
        func.call @exit(%c1_i32) : (i32) -> ()
      }
      %27 = arith.cmpi eq, %arg0, %c6_i32 : i32
      %28:2 = scf.if %27 -> (i32, i1) {
        %90 = affine.load %arg1[5] : memref<?xmemref<?xi8>>
        %91 = func.call @atoi(%90) : (memref<?xi8>) -> i32
        %92 = arith.cmpi slt, %91, %c1_i32 : i32
        %93:2 = scf.if %92 -> (i32, i1) {
          scf.yield %c1_i32, %false : i32, i1
        } else {
          %94 = arith.extsi %91 : i32 to i64
          %95 = arith.cmpi sge, %94, %24 : i64
          %96 = scf.if %95 -> (i32) {
            %98 = arith.addi %24, %c-1_i64 : i64
            %99 = arith.trunci %98 : i64 to i32
            scf.yield %99 : i32
          } else {
            scf.yield %91 : i32
          }
          %97 = arith.cmpi sgt, %96, %c1_i32 : i32
          scf.yield %96, %97 : i32, i1
        }
        scf.yield %93#0, %93#1 : i32, i1
      } else {
        scf.yield %c1_i32, %false : i32, i1
      }
      %29 = arith.muli %22, %c8_i64 : i64
      %30 = arith.muli %29, %24 : i64
      %31 = func.call @prk_malloc(%30) : (i64) -> memref<?xi8>
      %32 = polygeist.memref2pointer %31 : memref<?xi8> to !llvm.ptr
      %33 = llvm.mlir.zero : !llvm.ptr
      %34 = llvm.icmp "eq" %32, %33 : !llvm.ptr
      scf.if %34 {
        %90 = llvm.mlir.addressof @str8 : !llvm.ptr
        %91 = llvm.getelementptr %90[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<49 x i8>
        %92 = llvm.call @printf(%91, %30) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
        func.call @exit(%c1_i32) : (i32) -> ()
      }
      %35 = arith.extsi %15 : i32 to i64
      %36 = arith.cmpi slt, %22, %35 : i64
      scf.if %36 {
        %90 = llvm.mlir.addressof @str9 : !llvm.ptr
        %91 = llvm.getelementptr %90[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<71 x i8>
        %92 = llvm.call @printf(%91, %22, %15) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64, i32) -> i32
        func.call @exit(%c1_i32) : (i32) -> ()
      }
      %37 = arith.muli %15, %c2_i32 : i32
      %38 = arith.extsi %37 : i32 to i64
      %39 = arith.muli %38, %c4_i64 : i64
      %40 = func.call @prk_malloc(%39) : (i64) -> memref<?xi8>
      %41 = polygeist.memref2pointer %40 : memref<?xi8> to !llvm.ptr
      %42 = llvm.mlir.zero : !llvm.ptr
      %43 = llvm.icmp "eq" %41, %42 : !llvm.ptr
      scf.if %43 {
        %90 = llvm.mlir.addressof @str10 : !llvm.ptr
        %91 = llvm.getelementptr %90[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<63 x i8>
        %92 = llvm.call @printf(%91) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
        func.call @exit(%c1_i32) : (i32) -> ()
      }
      %44 = arith.index_cast %15 : i32 to index
      llvm.store %c0_i32, %41 : i32, !llvm.ptr
      %45 = arith.extsi %15 : i32 to i64
      %46 = arith.divsi %22, %45 : i64
      %47 = arith.trunci %46 : i64 to i32
      %48 = arith.remsi %22, %45 : i64
      %49 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
        %90 = arith.cmpi slt, %arg2, %15 : i32
        scf.condition(%90) %arg2 : i32
      } do {
      ^bb0(%arg2: i32):
        %90 = arith.extsi %arg2 : i32 to i64
        %91 = arith.cmpi slt, %90, %48 : i64
        %92 = scf.if %91 -> (i32) {
          %103 = arith.addi %47, %c1_i32 : i32
          scf.yield %103 : i32
        } else {
          scf.yield %47 : i32
        }
        %93 = arith.cmpi sgt, %arg2, %c0_i32 : i32
        scf.if %93 {
          %103 = arith.addi %arg2, %c-1_i32 : i32
          %104 = arith.index_cast %103 : i32 to index
          %105 = arith.addi %104, %44 : index
          %106 = arith.index_cast %105 : index to i32
          %107 = llvm.getelementptr %41[%106] : (!llvm.ptr, i32) -> !llvm.ptr, i32
          %108 = llvm.load %107 : !llvm.ptr -> i32
          %109 = arith.addi %108, %c1_i32 : i32
          %110 = llvm.getelementptr %41[%arg2] : (!llvm.ptr, i32) -> !llvm.ptr, i32
          llvm.store %109, %110 : i32, !llvm.ptr
        }
        %94 = arith.index_cast %arg2 : i32 to index
        %95 = llvm.getelementptr %41[%arg2] : (!llvm.ptr, i32) -> !llvm.ptr, i32
        %96 = llvm.load %95 : !llvm.ptr -> i32
        %97 = arith.addi %96, %92 : i32
        %98 = arith.addi %97, %c-1_i32 : i32
        %99 = arith.addi %94, %44 : index
        %100 = arith.index_cast %99 : index to i32
        %101 = llvm.getelementptr %41[%100] : (!llvm.ptr, i32) -> !llvm.ptr, i32
        llvm.store %98, %101 : i32, !llvm.ptr
        %102 = arith.addi %arg2, %c1_i32 : i32
        scf.yield %102 : i32
      }
      %50 = arith.extsi %15 : i32 to i64
      %51 = arith.muli %50, %c64_i64 : i64
      %52 = arith.muli %51, %24 : i64
      %53 = func.call @prk_malloc(%52) : (i64) -> memref<?xi8>
      %54 = polygeist.memref2pointer %53 : memref<?xi8> to !llvm.ptr
      %55 = llvm.mlir.zero : !llvm.ptr
      %56 = llvm.icmp "eq" %54, %55 : !llvm.ptr
      scf.if %56 {
        %90 = llvm.mlir.addressof @str11 : !llvm.ptr
        %91 = llvm.getelementptr %90[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<59 x i8>
        %92 = llvm.call @printf(%91) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
        func.call @exit(%c1_i32) : (i32) -> ()
      }
      omp.parallel   {
        func.call @bail_out(%c0_i32) : (i32) -> ()
        %90 = func.call @omp_get_thread_num() : () -> i32
        %91 = arith.index_cast %90 : i32 to index
        %92 = llvm.getelementptr %41[%90] : (!llvm.ptr, i32) -> !llvm.ptr, i32
        %93 = arith.addi %91, %44 : index
        %94 = arith.index_cast %93 : index to i32
        %95 = llvm.getelementptr %41[%94] : (!llvm.ptr, i32) -> !llvm.ptr, i32
        %96 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
          %115 = arith.extsi %arg2 : i32 to i64
          %116 = arith.cmpi slt, %115, %24 : i64
          scf.condition(%116) %arg2 : i32
        } do {
        ^bb0(%arg2: i32):
          %115 = arith.extsi %arg2 : i32 to i64
          %116 = llvm.load %92 : !llvm.ptr -> i32
          %117 = arith.muli %115, %22 : i64
          %118 = scf.while (%arg3 = %116) : (i32) -> i32 {
            %120 = llvm.load %95 : !llvm.ptr -> i32
            %121 = arith.cmpi sle, %arg3, %120 : i32
            scf.condition(%121) %arg3 : i32
          } do {
          ^bb0(%arg3: i32):
            %120 = arith.extsi %arg3 : i32 to i64
            %121 = arith.addi %120, %117 : i64
            %122 = arith.index_cast %121 : i64 to index
            %123 = arith.index_cast %122 : index to i32
            %124 = llvm.getelementptr %32[%123] : (!llvm.ptr, i32) -> !llvm.ptr, f64
            llvm.store %cst_1, %124 : f64, !llvm.ptr
            %125 = arith.addi %arg3, %c1_i32 : i32
            scf.yield %125 : i32
          }
          %119 = arith.addi %arg2, %c1_i32 : i32
          scf.yield %119 : i32
        }
        %97 = arith.cmpi eq, %90, %c0_i32 : i32
        scf.if %97 {
          %115 = llvm.getelementptr %41[%90] : (!llvm.ptr, i32) -> !llvm.ptr, i32
          %116 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
            %117 = arith.extsi %arg2 : i32 to i64
            %118 = arith.cmpi slt, %117, %24 : i64
            scf.condition(%118) %arg2 : i32
          } do {
          ^bb0(%arg2: i32):
            %117 = arith.extsi %arg2 : i32 to i64
            %118 = llvm.load %115 : !llvm.ptr -> i32
            %119 = arith.extsi %118 : i32 to i64
            %120 = arith.muli %117, %22 : i64
            %121 = arith.addi %119, %120 : i64
            %122 = arith.index_cast %121 : i64 to index
            %123 = arith.sitofp %arg2 : i32 to f64
            %124 = arith.index_cast %122 : index to i32
            %125 = llvm.getelementptr %32[%124] : (!llvm.ptr, i32) -> !llvm.ptr, f64
            llvm.store %123, %125 : f64, !llvm.ptr
            %126 = arith.addi %arg2, %c1_i32 : i32
            scf.yield %126 : i32
          }
        }
        %98 = arith.index_cast %90 : i32 to index
        %99 = llvm.getelementptr %41[%90] : (!llvm.ptr, i32) -> !llvm.ptr, i32
        %100 = llvm.load %99 : !llvm.ptr -> i32
        %101 = arith.addi %98, %44 : index
        %102 = arith.index_cast %101 : index to i32
        %103 = llvm.getelementptr %41[%102] : (!llvm.ptr, i32) -> !llvm.ptr, i32
        %104 = scf.while (%arg2 = %100) : (i32) -> i32 {
          %115 = llvm.load %103 : !llvm.ptr -> i32
          %116 = arith.cmpi sle, %arg2, %115 : i32
          scf.condition(%116) %arg2 : i32
        } do {
        ^bb0(%arg2: i32):
          %115 = arith.sitofp %arg2 : i32 to f64
          %116 = llvm.getelementptr %32[%arg2] : (!llvm.ptr, i32) -> !llvm.ptr, f64
          llvm.store %115, %116 : f64, !llvm.ptr
          %117 = arith.addi %arg2, %c1_i32 : i32
          scf.yield %117 : i32
        }
        %105 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
          %115 = arith.extsi %arg2 : i32 to i64
          %116 = arith.cmpi slt, %115, %24 : i64
          scf.condition(%116) %arg2 : i32
        } do {
        ^bb0(%arg2: i32):
          %115 = arith.muli %arg2, %0 : i32
          %116 = arith.addi %90, %115 : i32
          %117 = arith.muli %116, %c16_i32 : i32
          %118 = llvm.getelementptr %54[%117] : (!llvm.ptr, i32) -> !llvm.ptr, i32
          llvm.store %c0_i32, %118 : i32, !llvm.ptr
          %119 = arith.addi %arg2, %c1_i32 : i32
          scf.yield %119 : i32
        }
        %106 = arith.extsi %28#0 : i32 to i64
        %107 = llvm.getelementptr %41[%90] : (!llvm.ptr, i32) -> !llvm.ptr, i32
        %108 = arith.index_cast %90 : i32 to index
        %109 = arith.addi %108, %44 : index
        %110 = arith.index_cast %109 : index to i32
        %111 = llvm.getelementptr %41[%110] : (!llvm.ptr, i32) -> !llvm.ptr, i32
        %112 = arith.addi %0, %c-1_i32 : i32
        %113 = arith.cmpi eq, %90, %112 : i32
        %114 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
          %115 = arith.cmpi sle, %arg2, %19 : i32
          scf.condition(%115) %arg2 : i32
        } do {
        ^bb0(%arg2: i32):
          %115 = scf.while (%arg3 = %c1_i32) : (i32) -> i32 {
            %117 = arith.extsi %arg3 : i32 to i64
            %118 = arith.cmpi slt, %117, %24 : i64
            scf.condition(%118) %arg3 : i32
          } do {
          ^bb0(%arg3: i32):
            %117 = arith.extsi %arg3 : i32 to i64
            %118 = arith.subi %24, %117 : i64
            %119 = arith.cmpi slt, %106, %118 : i64
            %120 = arith.select %119, %106, %118 : i64
            %121 = arith.trunci %120 : i64 to i32
            %122 = arith.addi %arg3, %121 : i32
            %123 = scf.while (%arg4 = %arg3) : (i32) -> i32 {
              %125 = arith.cmpi slt, %arg4, %122 : i32
              scf.condition(%125) %arg4 : i32
            } do {
            ^bb0(%arg4: i32):
              %125 = llvm.load %107 : !llvm.ptr -> i32
              %126 = arith.cmpi sgt, %125, %c1_i32 : i32
              %127 = scf.if %126 -> (i32) {
                %135 = llvm.getelementptr %41[%90] : (!llvm.ptr, i32) -> !llvm.ptr, i32
                %136 = llvm.load %135 : !llvm.ptr -> i32
                scf.yield %136 : i32
              } else {
                scf.yield %c1_i32 : i32
              }
              %128 = arith.extsi %arg4 : i32 to i64
              %129 = arith.muli %128, %22 : i64
              %130 = arith.addi %arg4, %c-1_i32 : i32
              %131 = arith.extsi %130 : i32 to i64
              %132 = arith.muli %131, %22 : i64
              %133 = scf.while (%arg5 = %127) : (i32) -> i32 {
                %135 = llvm.load %111 : !llvm.ptr -> i32
                %136 = arith.cmpi sle, %arg5, %135 : i32
                scf.condition(%136) %arg5 : i32
              } do {
              ^bb0(%arg5: i32):
                %135 = arith.extsi %arg5 : i32 to i64
                %136 = arith.addi %135, %129 : i64
                %137 = arith.index_cast %136 : i64 to index
                %138 = arith.addi %arg5, %c-1_i32 : i32
                %139 = arith.extsi %138 : i32 to i64
                %140 = arith.addi %139, %129 : i64
                %141 = arith.index_cast %140 : i64 to index
                %142 = arith.index_cast %141 : index to i32
                %143 = llvm.getelementptr %32[%142] : (!llvm.ptr, i32) -> !llvm.ptr, f64
                %144 = llvm.load %143 : !llvm.ptr -> f64
                %145 = arith.addi %135, %132 : i64
                %146 = arith.index_cast %145 : i64 to index
                %147 = arith.index_cast %146 : index to i32
                %148 = llvm.getelementptr %32[%147] : (!llvm.ptr, i32) -> !llvm.ptr, f64
                %149 = llvm.load %148 : !llvm.ptr -> f64
                %150 = arith.addf %144, %149 : f64
                %151 = arith.addi %139, %132 : i64
                %152 = arith.index_cast %151 : i64 to index
                %153 = arith.index_cast %152 : index to i32
                %154 = llvm.getelementptr %32[%153] : (!llvm.ptr, i32) -> !llvm.ptr, f64
                %155 = llvm.load %154 : !llvm.ptr -> f64
                %156 = arith.subf %150, %155 : f64
                %157 = arith.index_cast %137 : index to i32
                %158 = llvm.getelementptr %32[%157] : (!llvm.ptr, i32) -> !llvm.ptr, f64
                llvm.store %156, %158 : f64, !llvm.ptr
                %159 = arith.addi %arg5, %c1_i32 : i32
                scf.yield %159 : i32
              }
              %134 = arith.addi %arg4, %c1_i32 : i32
              scf.yield %134 : i32
            }
            %124 = arith.addi %arg3, %28#0 : i32
            scf.yield %124 : i32
          }
          scf.if %113 {
            %117 = arith.addi %22, %c-1_i64 : i64
            %118 = arith.addi %24, %c-1_i64 : i64
            %119 = arith.muli %118, %22 : i64
            %120 = arith.addi %117, %119 : i64
            %121 = arith.index_cast %120 : i64 to index
            %122 = arith.index_cast %121 : index to i32
            %123 = llvm.getelementptr %32[%122] : (!llvm.ptr, i32) -> !llvm.ptr, f64
            %124 = llvm.load %123 : !llvm.ptr -> f64
            %125 = arith.negf %124 : f64
            llvm.store %125, %32 : f64, !llvm.ptr
          }
          %116 = arith.addi %arg2, %c1_i32 : i32
          scf.yield %116 : i32
        }
        omp.terminator
      }
      %57 = arith.addi %19, %c1_i32 : i32
      %58 = arith.extsi %57 : i32 to i64
      %59 = arith.addi %24, %22 : i64
      %60 = arith.addi %59, %c-2_i64 : i64
      %61 = arith.muli %58, %60 : i64
      %62 = arith.sitofp %61 : i64 to f64
      %63 = arith.addi %22, %c-1_i64 : i64
      %64 = arith.addi %24, %c-1_i64 : i64
      %65 = arith.muli %64, %22 : i64
      %66 = arith.addi %63, %65 : i64
      %67 = arith.index_cast %66 : i64 to index
      %68 = arith.index_cast %67 : index to i32
      %69 = llvm.getelementptr %32[%68] : (!llvm.ptr, i32) -> !llvm.ptr, f64
      %70 = llvm.load %69 : !llvm.ptr -> f64
      %71 = arith.subf %70, %62 : f64
      %72 = math.absf %71 : f64
      %73 = arith.divf %72, %62 : f64
      %74 = arith.cmpf ogt, %73, %cst_2 : f64
      scf.if %74 {
        %90 = llvm.mlir.addressof @str12 : !llvm.ptr
        %91 = llvm.getelementptr %90[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<59 x i8>
        %92 = arith.index_cast %67 : index to i32
        %93 = llvm.getelementptr %32[%92] : (!llvm.ptr, i32) -> !llvm.ptr, f64
        %94 = llvm.load %93 : !llvm.ptr -> f64
        %95 = llvm.call @printf(%91, %94, %62) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64) -> i32
        func.call @exit(%c1_i32) : (i32) -> ()
      }
      %75 = llvm.mlir.addressof @str13 : !llvm.ptr
      %76 = llvm.getelementptr %75[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<20 x i8>
      %77 = llvm.call @printf(%76) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %78 = arith.sitofp %19 : i32 to f64
      %79 = arith.divf %1, %78 : f64
      %80 = scf.if %28#1 -> (f64) {
        %90 = arith.mulf %79, %cst_0 : f64
        scf.yield %90 : f64
      } else {
        scf.yield %79 : f64
      }
      %81 = llvm.mlir.addressof @str14 : !llvm.ptr
      %82 = llvm.getelementptr %81[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<40 x i8>
      %83 = arith.addi %22, %c-1_i64 : i64
      %84 = arith.addi %24, %c-1_i64 : i64
      %85 = arith.muli %83, %84 : i64
      %86 = arith.sitofp %85 : i64 to f64
      %87 = arith.mulf %86, %cst : f64
      %88 = arith.divf %87, %80 : f64
      %89 = llvm.call @printf(%82, %88, %80) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64) -> i32
      func.call @exit(%c0_i32) : (i32) -> ()
    }
    return %13 : i32
  }
  func.func private @atoi(memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @exit(i32) attributes {llvm.linkage = #llvm.linkage<external>}
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
  func.func private @bail_out(i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @omp_get_thread_num() -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
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
