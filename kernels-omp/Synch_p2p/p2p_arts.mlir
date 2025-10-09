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
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
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
      %16 = memref.load %arg1[%c0] : memref<?xmemref<?xi8>>
      %17 = polygeist.memref2pointer %16 : memref<?xi8> to !llvm.ptr
      %18 = llvm.call @printf(%15, %17) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
      %19 = llvm.mlir.addressof @str4 : !llvm.ptr
      %20 = llvm.getelementptr %19[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<41 x i8>
      %21 = llvm.call @printf(%20) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    } else {
      %14 = memref.load %arg1[%c1] : memref<?xmemref<?xi8>>
      %15 = func.call @atoi(%14) : (memref<?xi8>) -> i32
      %16 = arith.cmpi slt, %15, %c1_i32 : i32
      %17 = scf.if %16 -> (i1) {
        scf.yield %true : i1
      } else {
        %83 = arith.cmpi sgt, %15, %c512_i32 : i32
        scf.yield %83 : i1
      }
      scf.if %17 {
        %83 = llvm.mlir.addressof @str5 : !llvm.ptr
        %84 = llvm.getelementptr %83[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
        %85 = llvm.call @printf(%84, %15) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
        func.call @exit(%c1_i32) : (i32) -> ()
      }
      func.call @omp_set_num_threads(%15) : (i32) -> ()
      %18 = memref.load %arg1[%c2] : memref<?xmemref<?xi8>>
      %19 = func.call @atoi(%18) : (memref<?xi8>) -> i32
      %20 = arith.cmpi slt, %19, %c1_i32 : i32
      scf.if %20 {
        %83 = llvm.mlir.addressof @str6 : !llvm.ptr
        %84 = llvm.getelementptr %83[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
        %85 = llvm.call @printf(%84, %19) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
        func.call @exit(%c1_i32) : (i32) -> ()
      }
      %21 = memref.load %arg1[%c3] : memref<?xmemref<?xi8>>
      %22 = func.call @atol(%21) : (memref<?xi8>) -> i64
      %23 = memref.load %arg1[%c4] : memref<?xmemref<?xi8>>
      %24 = func.call @atol(%23) : (memref<?xi8>) -> i64
      %25 = arith.cmpi slt, %22, %c1_i64 : i64
      %26 = scf.if %25 -> (i1) {
        scf.yield %true : i1
      } else {
        %83 = arith.cmpi slt, %24, %c1_i64 : i64
        scf.yield %83 : i1
      }
      scf.if %26 {
        %83 = llvm.mlir.addressof @str7 : !llvm.ptr
        %84 = llvm.getelementptr %83[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<52 x i8>
        %85 = llvm.call @printf(%84, %22, %24) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64, i64) -> i32
        func.call @exit(%c1_i32) : (i32) -> ()
      }
      %27 = arith.cmpi eq, %arg0, %c6_i32 : i32
      %28:2 = scf.if %27 -> (i32, i1) {
        %83 = memref.load %arg1[%c5] : memref<?xmemref<?xi8>>
        %84 = func.call @atoi(%83) : (memref<?xi8>) -> i32
        %85 = arith.cmpi slt, %84, %c1_i32 : i32
        %86:2 = scf.if %85 -> (i32, i1) {
          scf.yield %c1_i32, %false : i32, i1
        } else {
          %87 = arith.extsi %84 : i32 to i64
          %88 = arith.cmpi sge, %87, %24 : i64
          %89 = scf.if %88 -> (i32) {
            %91 = arith.addi %24, %c-1_i64 : i64
            %92 = arith.trunci %91 : i64 to i32
            scf.yield %92 : i32
          } else {
            scf.yield %84 : i32
          }
          %90 = arith.cmpi sgt, %89, %c1_i32 : i32
          scf.yield %89, %90 : i32, i1
        }
        scf.yield %86#0, %86#1 : i32, i1
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
        %83 = llvm.mlir.addressof @str8 : !llvm.ptr
        %84 = llvm.getelementptr %83[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<49 x i8>
        %85 = llvm.call @printf(%84, %30) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
        func.call @exit(%c1_i32) : (i32) -> ()
      }
      %35 = arith.extsi %15 : i32 to i64
      %36 = arith.cmpi slt, %22, %35 : i64
      scf.if %36 {
        %83 = llvm.mlir.addressof @str9 : !llvm.ptr
        %84 = llvm.getelementptr %83[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<71 x i8>
        %85 = llvm.call @printf(%84, %22, %15) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64, i32) -> i32
        func.call @exit(%c1_i32) : (i32) -> ()
      }
      %37 = arith.muli %15, %c2_i32 : i32
      %38 = arith.extsi %37 : i32 to i64
      %39 = arith.muli %38, %c4_i64 : i64
      %40 = func.call @prk_malloc(%39) : (i64) -> memref<?xi8>
      %41 = polygeist.memref2pointer %40 : memref<?xi8> to !llvm.ptr
      %42 = llvm.icmp "eq" %41, %33 : !llvm.ptr
      scf.if %42 {
        %83 = llvm.mlir.addressof @str10 : !llvm.ptr
        %84 = llvm.getelementptr %83[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<63 x i8>
        %85 = llvm.call @printf(%84) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
        func.call @exit(%c1_i32) : (i32) -> ()
      }
      %43 = arith.index_cast %15 : i32 to index
      llvm.store %c0_i32, %41 : i32, !llvm.ptr
      %44 = arith.divsi %22, %35 : i64
      %45 = arith.trunci %44 : i64 to i32
      %46 = arith.remsi %22, %35 : i64
      scf.for %arg2 = %c0 to %43 step %c1 {
        %83 = arith.index_cast %arg2 : index to i32
        %84 = arith.extsi %83 : i32 to i64
        %85 = arith.cmpi slt, %84, %46 : i64
        %86 = scf.if %85 -> (i32) {
          %95 = arith.addi %45, %c1_i32 : i32
          scf.yield %95 : i32
        } else {
          scf.yield %45 : i32
        }
        %87 = arith.cmpi sgt, %83, %c0_i32 : i32
        scf.if %87 {
          %95 = arith.addi %83, %c-1_i32 : i32
          %96 = arith.index_cast %95 : i32 to index
          %97 = arith.addi %96, %43 : index
          %98 = arith.index_cast %97 : index to i32
          %99 = llvm.getelementptr %41[%98] : (!llvm.ptr, i32) -> !llvm.ptr, i32
          %100 = llvm.load %99 : !llvm.ptr -> i32
          %101 = arith.addi %100, %c1_i32 : i32
          %102 = llvm.getelementptr %41[%83] : (!llvm.ptr, i32) -> !llvm.ptr, i32
          llvm.store %101, %102 : i32, !llvm.ptr
        }
        %88 = llvm.getelementptr %41[%83] : (!llvm.ptr, i32) -> !llvm.ptr, i32
        %89 = llvm.load %88 : !llvm.ptr -> i32
        %90 = arith.addi %89, %86 : i32
        %91 = arith.addi %90, %c-1_i32 : i32
        %92 = arith.addi %arg2, %43 : index
        %93 = arith.index_cast %92 : index to i32
        %94 = llvm.getelementptr %41[%93] : (!llvm.ptr, i32) -> !llvm.ptr, i32
        llvm.store %91, %94 : i32, !llvm.ptr
      }
      %47 = arith.muli %35, %c64_i64 : i64
      %48 = arith.muli %47, %24 : i64
      %49 = func.call @prk_malloc(%48) : (i64) -> memref<?xi8>
      %50 = polygeist.memref2pointer %49 : memref<?xi8> to !llvm.ptr
      %51 = llvm.icmp "eq" %50, %33 : !llvm.ptr
      scf.if %51 {
        %83 = llvm.mlir.addressof @str11 : !llvm.ptr
        %84 = llvm.getelementptr %83[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<59 x i8>
        %85 = llvm.call @printf(%84) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
        func.call @exit(%c1_i32) : (i32) -> ()
      }
      arts.edt <parallel> <internode> route(%c0_i32) {
        func.call @bail_out(%c0_i32) : (i32) -> ()
        %83 = arts.get_current_node -> i32
        %84 = arith.index_cast %83 : i32 to index
        %85 = llvm.getelementptr %41[%83] : (!llvm.ptr, i32) -> !llvm.ptr, i32
        %86 = arith.addi %84, %43 : index
        %87 = arith.index_cast %86 : index to i32
        %88 = llvm.getelementptr %41[%87] : (!llvm.ptr, i32) -> !llvm.ptr, i32
        %89 = arith.index_cast %24 : i64 to index
        scf.for %arg2 = %c0 to %89 step %c1 {
          %98 = arith.index_cast %arg2 : index to i32
          %99 = arith.extsi %98 : i32 to i64
          %100 = llvm.load %85 : !llvm.ptr -> i32
          %101 = arith.muli %99, %22 : i64
          %102 = scf.while (%arg3 = %100) : (i32) -> i32 {
            %103 = llvm.load %88 : !llvm.ptr -> i32
            %104 = arith.cmpi sle, %arg3, %103 : i32
            scf.condition(%104) %arg3 : i32
          } do {
          ^bb0(%arg3: i32):
            %103 = arith.extsi %arg3 : i32 to i64
            %104 = arith.addi %103, %101 : i64
            %105 = arith.index_cast %104 : i64 to index
            %106 = arith.index_cast %105 : index to i32
            %107 = llvm.getelementptr %32[%106] : (!llvm.ptr, i32) -> !llvm.ptr, f64
            llvm.store %cst_1, %107 : f64, !llvm.ptr
            %108 = arith.addi %arg3, %c1_i32 : i32
            scf.yield %108 : i32
          }
        }
        %90 = arith.cmpi eq, %83, %c0_i32 : i32
        scf.if %90 {
          scf.for %arg2 = %c0 to %89 step %c1 {
            %98 = arith.index_cast %arg2 : index to i32
            %99 = arith.extsi %98 : i32 to i64
            %100 = llvm.load %85 : !llvm.ptr -> i32
            %101 = arith.extsi %100 : i32 to i64
            %102 = arith.muli %99, %22 : i64
            %103 = arith.addi %101, %102 : i64
            %104 = arith.index_cast %103 : i64 to index
            %105 = arith.sitofp %98 : i32 to f64
            %106 = arith.index_cast %104 : index to i32
            %107 = llvm.getelementptr %32[%106] : (!llvm.ptr, i32) -> !llvm.ptr, f64
            llvm.store %105, %107 : f64, !llvm.ptr
          }
        }
        %91 = llvm.load %85 : !llvm.ptr -> i32
        %92 = scf.while (%arg2 = %91) : (i32) -> i32 {
          %98 = llvm.load %88 : !llvm.ptr -> i32
          %99 = arith.cmpi sle, %arg2, %98 : i32
          scf.condition(%99) %arg2 : i32
        } do {
        ^bb0(%arg2: i32):
          %98 = arith.sitofp %arg2 : i32 to f64
          %99 = llvm.getelementptr %32[%arg2] : (!llvm.ptr, i32) -> !llvm.ptr, f64
          llvm.store %98, %99 : f64, !llvm.ptr
          %100 = arith.addi %arg2, %c1_i32 : i32
          scf.yield %100 : i32
        }
        scf.for %arg2 = %c0 to %89 step %c1 {
          %98 = arith.index_cast %arg2 : index to i32
          %99 = arith.muli %98, %0 : i32
          %100 = arith.addi %83, %99 : i32
          %101 = arith.muli %100, %c16_i32 : i32
          %102 = llvm.getelementptr %50[%101] : (!llvm.ptr, i32) -> !llvm.ptr, i32
          llvm.store %c0_i32, %102 : i32, !llvm.ptr
        }
        %93 = arith.extsi %28#0 : i32 to i64
        %94 = arith.addi %0, %c-1_i32 : i32
        %95 = arith.cmpi eq, %83, %94 : i32
        %96 = arith.addi %19, %c1_i32 : i32
        %97 = arith.index_cast %96 : i32 to index
        scf.for %arg2 = %c0 to %97 step %c1 {
          %98 = arith.index_cast %28#0 : i32 to index
          scf.for %arg3 = %c1 to %89 step %98 {
            %99 = arith.subi %arg3, %c1 : index
            %100 = arith.divui %99, %98 : index
            %101 = arith.muli %100, %98 : index
            %102 = arith.addi %101, %c1 : index
            %103 = arith.index_cast %102 : index to i32
            %104 = arith.extsi %103 : i32 to i64
            %105 = arith.subi %24, %104 : i64
            %106 = arith.cmpi slt, %93, %105 : i64
            %107 = arith.select %106, %93, %105 : i64
            %108 = arith.trunci %107 : i64 to i32
            %109 = arith.addi %103, %108 : i32
            %110 = arith.index_cast %109 : i32 to index
            scf.for %arg4 = %102 to %110 step %c1 {
              %111 = arith.index_cast %arg4 : index to i32
              %112 = llvm.load %85 : !llvm.ptr -> i32
              %113 = arith.cmpi sgt, %112, %c1_i32 : i32
              %114 = scf.if %113 -> (i32) {
                %121 = llvm.load %85 : !llvm.ptr -> i32
                scf.yield %121 : i32
              } else {
                scf.yield %c1_i32 : i32
              }
              %115 = arith.extsi %111 : i32 to i64
              %116 = arith.muli %115, %22 : i64
              %117 = arith.addi %111, %c-1_i32 : i32
              %118 = arith.extsi %117 : i32 to i64
              %119 = arith.muli %118, %22 : i64
              %120 = scf.while (%arg5 = %114) : (i32) -> i32 {
                %121 = llvm.load %88 : !llvm.ptr -> i32
                %122 = arith.cmpi sle, %arg5, %121 : i32
                scf.condition(%122) %arg5 : i32
              } do {
              ^bb0(%arg5: i32):
                %121 = arith.extsi %arg5 : i32 to i64
                %122 = arith.addi %121, %116 : i64
                %123 = arith.index_cast %122 : i64 to index
                %124 = arith.addi %arg5, %c-1_i32 : i32
                %125 = arith.extsi %124 : i32 to i64
                %126 = arith.addi %125, %116 : i64
                %127 = arith.index_cast %126 : i64 to index
                %128 = arith.index_cast %127 : index to i32
                %129 = llvm.getelementptr %32[%128] : (!llvm.ptr, i32) -> !llvm.ptr, f64
                %130 = llvm.load %129 : !llvm.ptr -> f64
                %131 = arith.addi %121, %119 : i64
                %132 = arith.index_cast %131 : i64 to index
                %133 = arith.index_cast %132 : index to i32
                %134 = llvm.getelementptr %32[%133] : (!llvm.ptr, i32) -> !llvm.ptr, f64
                %135 = llvm.load %134 : !llvm.ptr -> f64
                %136 = arith.addf %130, %135 : f64
                %137 = arith.addi %125, %119 : i64
                %138 = arith.index_cast %137 : i64 to index
                %139 = arith.index_cast %138 : index to i32
                %140 = llvm.getelementptr %32[%139] : (!llvm.ptr, i32) -> !llvm.ptr, f64
                %141 = llvm.load %140 : !llvm.ptr -> f64
                %142 = arith.subf %136, %141 : f64
                %143 = arith.index_cast %123 : index to i32
                %144 = llvm.getelementptr %32[%143] : (!llvm.ptr, i32) -> !llvm.ptr, f64
                llvm.store %142, %144 : f64, !llvm.ptr
                %145 = arith.addi %arg5, %c1_i32 : i32
                scf.yield %145 : i32
              }
            }
          }
          scf.if %95 {
            %99 = arith.addi %22, %c-1_i64 : i64
            %100 = arith.addi %24, %c-1_i64 : i64
            %101 = arith.muli %100, %22 : i64
            %102 = arith.addi %99, %101 : i64
            %103 = arith.index_cast %102 : i64 to index
            %104 = arith.index_cast %103 : index to i32
            %105 = llvm.getelementptr %32[%104] : (!llvm.ptr, i32) -> !llvm.ptr, f64
            %106 = llvm.load %105 : !llvm.ptr -> f64
            %107 = arith.negf %106 : f64
            llvm.store %107, %32 : f64, !llvm.ptr
          }
        }
      }
      %52 = arith.addi %19, %c1_i32 : i32
      %53 = arith.extsi %52 : i32 to i64
      %54 = arith.addi %24, %22 : i64
      %55 = arith.addi %54, %c-2_i64 : i64
      %56 = arith.muli %53, %55 : i64
      %57 = arith.sitofp %56 : i64 to f64
      %58 = arith.addi %22, %c-1_i64 : i64
      %59 = arith.addi %24, %c-1_i64 : i64
      %60 = arith.muli %59, %22 : i64
      %61 = arith.addi %58, %60 : i64
      %62 = arith.index_cast %61 : i64 to index
      %63 = arith.index_cast %62 : index to i32
      %64 = llvm.getelementptr %32[%63] : (!llvm.ptr, i32) -> !llvm.ptr, f64
      %65 = llvm.load %64 : !llvm.ptr -> f64
      %66 = arith.subf %65, %57 : f64
      %67 = math.absf %66 : f64
      %68 = arith.divf %67, %57 : f64
      %69 = arith.cmpf ogt, %68, %cst_2 : f64
      scf.if %69 {
        %83 = llvm.mlir.addressof @str12 : !llvm.ptr
        %84 = llvm.getelementptr %83[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<59 x i8>
        %85 = llvm.load %64 : !llvm.ptr -> f64
        %86 = llvm.call @printf(%84, %85, %57) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64) -> i32
        func.call @exit(%c1_i32) : (i32) -> ()
      }
      %70 = llvm.mlir.addressof @str13 : !llvm.ptr
      %71 = llvm.getelementptr %70[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<20 x i8>
      %72 = llvm.call @printf(%71) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %73 = arith.sitofp %19 : i32 to f64
      %74 = arith.divf %1, %73 : f64
      %75 = scf.if %28#1 -> (f64) {
        %83 = arith.mulf %74, %cst_0 : f64
        scf.yield %83 : f64
      } else {
        scf.yield %74 : f64
      }
      %76 = llvm.mlir.addressof @str14 : !llvm.ptr
      %77 = llvm.getelementptr %76[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<40 x i8>
      %78 = arith.muli %58, %59 : i64
      %79 = arith.sitofp %78 : i64 to f64
      %80 = arith.mulf %79, %cst : f64
      %81 = arith.divf %80, %75 : f64
      %82 = llvm.call @printf(%77, %81, %75) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64) -> i32
      func.call @exit(%c0_i32) : (i32) -> ()
    }
    return %13 : i32
  }
  func.func private @atoi(memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @exit(i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @omp_set_num_threads(i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @atol(memref<?xi8>) -> i64 attributes {llvm.linkage = #llvm.linkage<external>}
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
