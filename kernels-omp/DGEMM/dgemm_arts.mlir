module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:o-i64:64-i128:128-n32:64-S128", llvm.target_triple = "arm64-apple-macosx15.0.0", "polygeist.target-cpu" = "apple-m1", "polygeist.target-features" = "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"} {
  llvm.mlir.global internal constant @str15("(a & (~a+1)) == a\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str14("par-res-kern_general.h\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str13("prk_get_alignment\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str12("PRK_ALIGNMENT\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str11("Rate (MFlops/s): %lf  Avg time (s): %lf\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str10("Solution validates\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str9("ERROR: Checksum = %lf, Reference checksum = %lf\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str8("Could not allocate space for matrix tiles on thread %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str7("ERROR: Could not allocate space for global matrices\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str6("ERROR: Matrix order must be positive: %ld\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str5("ERROR: Iterations must be positive : %d \0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str4("ERROR: Invalid number of threads: %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str3("Usage: %s <# threads> <# iterations> <matrix order> [tile size]\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2("OpenMP Dense matrix-matrix multiplication\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("2.17\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str0("Parallel Research Kernels version %s\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  memref.global "private" @"main@static@C" : memref<1xmemref<?xf64>> = uninitialized
  memref.global "private" @"main@static@B" : memref<1xmemref<?xf64>> = uninitialized
  memref.global "private" @"main@static@A" : memref<1xmemref<?xf64>> = uninitialized
  func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %true = arith.constant true
    %c8_i64 = arith.constant 8 : i64
    %cst = arith.constant 9.9999999999999995E-7 : f64
    %cst_0 = arith.constant 2.000000e+00 : f64
    %c3_i32 = arith.constant 3 : i32
    %c12_i32 = arith.constant 12 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_1 = arith.constant 1.000000e+00 : f64
    %cst_2 = arith.constant 2.500000e-01 : f64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst_3 = arith.constant 1.000000e-08 : f64
    %cst_4 = arith.constant 0.000000e+00 : f64
    %0 = llvm.mlir.undef : f64
    %1 = llvm.mlir.undef : i32
    %2 = memref.get_global @"main@static@C" : memref<1xmemref<?xf64>>
    %3 = memref.get_global @"main@static@B" : memref<1xmemref<?xf64>>
    %4 = memref.get_global @"main@static@A" : memref<1xmemref<?xf64>>
    %alloca = memref.alloca() : memref<i32>
    memref.store %c0_i32, %alloca[] : memref<i32>
    %5 = llvm.mlir.addressof @str0 : !llvm.ptr
    %6 = llvm.getelementptr %5[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
    %7 = llvm.mlir.addressof @str1 : !llvm.ptr
    %8 = llvm.getelementptr %7[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<5 x i8>
    %9 = llvm.call @printf(%6, %8) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    %10 = llvm.mlir.addressof @str2 : !llvm.ptr
    %11 = llvm.getelementptr %10[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<43 x i8>
    %12 = llvm.call @printf(%11) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %13 = arith.cmpi ne, %arg0, %c4_i32 : i32
    %14 = arith.cmpi ne, %arg0, %c5_i32 : i32
    %15 = arith.andi %13, %14 : i1
    scf.if %15 {
      %76 = llvm.mlir.addressof @str3 : !llvm.ptr
      %77 = llvm.getelementptr %76[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<65 x i8>
      %78 = memref.load %arg1[%c0] : memref<?xmemref<?xi8>>
      %79 = polygeist.memref2pointer %78 : memref<?xi8> to !llvm.ptr
      %80 = llvm.call @printf(%77, %79) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %16 = memref.load %arg1[%c1] : memref<?xmemref<?xi8>>
    %17 = call @atoi(%16) : (memref<?xi8>) -> i32
    %18 = arith.cmpi slt, %17, %c1_i32 : i32
    %19 = scf.if %18 -> (i1) {
      scf.yield %true : i1
    } else {
      %76 = arith.cmpi sgt, %17, %c512_i32 : i32
      scf.yield %76 : i1
    }
    scf.if %19 {
      %76 = llvm.mlir.addressof @str4 : !llvm.ptr
      %77 = llvm.getelementptr %76[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
      %78 = llvm.call @printf(%77, %17) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    call @omp_set_num_threads(%17) : (i32) -> ()
    %20 = memref.load %arg1[%c2] : memref<?xmemref<?xi8>>
    %21 = call @atoi(%20) : (memref<?xi8>) -> i32
    %22 = arith.cmpi slt, %21, %c1_i32 : i32
    scf.if %22 {
      %76 = llvm.mlir.addressof @str5 : !llvm.ptr
      %77 = llvm.getelementptr %76[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<42 x i8>
      %78 = llvm.call @printf(%77, %21) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %23 = memref.load %arg1[%c3] : memref<?xmemref<?xi8>>
    %24 = call @atol(%23) : (memref<?xi8>) -> i64
    %25 = arith.cmpi slt, %24, %c0_i64 : i64
    %26 = scf.if %25 -> (i64) {
      %76 = arith.subi %c0_i64, %24 : i64
      scf.yield %76 : i64
    } else {
      scf.yield %24 : i64
    }
    %27 = arith.cmpi slt, %26, %c1_i64 : i64
    scf.if %27 {
      %76 = llvm.mlir.addressof @str6 : !llvm.ptr
      %77 = llvm.getelementptr %76[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<43 x i8>
      %78 = llvm.call @printf(%77, %26) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %28 = arith.muli %26, %26 : i64
    %29 = arith.muli %28, %c8_i64 : i64
    %30 = call @prk_malloc(%29) : (i64) -> memref<?xi8>
    %31 = polygeist.memref2pointer %30 : memref<?xi8> to !llvm.ptr
    %32 = polygeist.pointer2memref %31 : !llvm.ptr to memref<?xf64>
    memref.store %32, %4[%c0] : memref<1xmemref<?xf64>>
    %33 = call @prk_malloc(%29) : (i64) -> memref<?xi8>
    %34 = polygeist.memref2pointer %33 : memref<?xi8> to !llvm.ptr
    %35 = polygeist.pointer2memref %34 : !llvm.ptr to memref<?xf64>
    memref.store %35, %3[%c0] : memref<1xmemref<?xf64>>
    %36 = call @prk_malloc(%29) : (i64) -> memref<?xi8>
    %37 = polygeist.memref2pointer %36 : memref<?xi8> to !llvm.ptr
    %38 = polygeist.pointer2memref %37 : !llvm.ptr to memref<?xf64>
    memref.store %38, %2[%c0] : memref<1xmemref<?xf64>>
    %39 = memref.load %4[%c0] : memref<1xmemref<?xf64>>
    %40 = polygeist.memref2pointer %39 : memref<?xf64> to !llvm.ptr
    %41 = llvm.mlir.zero : !llvm.ptr
    %42 = llvm.icmp "eq" %40, %41 : !llvm.ptr
    %43 = scf.if %42 -> (i1) {
      scf.yield %true : i1
    } else {
      %76 = memref.load %3[%c0] : memref<1xmemref<?xf64>>
      %77 = polygeist.memref2pointer %76 : memref<?xf64> to !llvm.ptr
      %78 = llvm.icmp "eq" %77, %41 : !llvm.ptr
      scf.yield %78 : i1
    }
    %44 = scf.if %43 -> (i1) {
      scf.yield %true : i1
    } else {
      %76 = llvm.icmp "eq" %37, %41 : !llvm.ptr
      scf.yield %76 : i1
    }
    scf.if %44 {
      %76 = llvm.mlir.addressof @str7 : !llvm.ptr
      %77 = llvm.getelementptr %76[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<53 x i8>
      %78 = llvm.call @printf(%77) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %45 = arith.sitofp %26 : i64 to f64
    %46 = arith.mulf %45, %cst_2 : f64
    %47 = arith.mulf %46, %45 : f64
    %48 = arith.mulf %47, %45 : f64
    %49 = arith.subf %45, %cst_1 : f64
    %50 = arith.mulf %48, %49 : f64
    %51 = arith.mulf %50, %49 : f64
    %52 = arith.trunci %26 : i64 to i32
    %53 = arith.index_cast %52 : i32 to index
    arts.edt <parallel> <internode> route(%c0_i32) {
      arts.for(%c0) to(%53) step(%c1) {{
      ^bb0(%arg2: index):
        %76 = arith.index_cast %arg2 : index to i32
        %77 = arith.extsi %76 : i32 to i64
        %78 = arith.muli %26, %77 : i64
        %79 = arith.sitofp %76 : i32 to f64
        %80 = arith.index_cast %26 : i64 to index
        scf.for %arg3 = %c0 to %80 step %c1 {
          %81 = arith.index_cast %arg3 : index to i32
          %82 = arith.extsi %81 : i32 to i64
          %83 = memref.load %4[%c0] : memref<1xmemref<?xf64>>
          %84 = arith.addi %82, %78 : i64
          %85 = arith.index_cast %84 : i64 to index
          %86 = memref.load %3[%c0] : memref<1xmemref<?xf64>>
          memref.store %79, %86[%85] : memref<?xf64>
          %87 = memref.load %86[%85] : memref<?xf64>
          memref.store %87, %83[%85] : memref<?xf64>
          %88 = memref.load %2[%c0] : memref<1xmemref<?xf64>>
          memref.store %cst_4, %88[%85] : memref<?xf64>
        }
      }}
    }
    %54 = arith.cmpi eq, %arg0, %c5_i32 : i32
    %55:3 = scf.if %54 -> (i32, i1, i1) {
      %76 = memref.load %arg1[%c4] : memref<?xmemref<?xi8>>
      %77 = func.call @atoi(%76) : (memref<?xi8>) -> i32
      %78 = arith.cmpi sgt, %77, %c0_i32 : i32
      scf.yield %77, %78, %78 : i32, i1, i1
    } else {
      scf.yield %c32_i32, %true, %true : i32, i1, i1
    }
    arts.edt <parallel> <internode> route(%c0_i32) {
      %alloca_5 = memref.alloca() : memref<memref<?xf64>>
      %alloca_6 = memref.alloca() : memref<memref<?xf64>>
      %alloca_7 = memref.alloca() : memref<memref<?xf64>>
      scf.if %55#2 {
        %79 = arith.addi %55#0, %c12_i32 : i32
        %80 = arith.muli %55#0, %79 : i32
        %81 = arith.muli %80, %c3_i32 : i32
        %82 = arith.extsi %81 : i32 to i64
        %83 = arith.muli %82, %c8_i64 : i64
        %84 = func.call @prk_malloc(%83) : (i64) -> memref<?xi8>
        %85 = polygeist.memref2pointer %84 : memref<?xi8> to !llvm.ptr
        %86 = polygeist.pointer2memref %85 : !llvm.ptr to memref<?xf64>
        memref.store %86, %alloca_7[] : memref<memref<?xf64>>
        %87 = llvm.icmp "eq" %85, %41 : !llvm.ptr
        scf.if %87 {
          memref.store %c1_i32, %alloca[] : memref<i32>
          %93 = llvm.mlir.addressof @str8 : !llvm.ptr
          %94 = llvm.getelementptr %93[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<56 x i8>
          %95 = arts.get_current_node -> i32
          %96 = llvm.call @printf(%94, %95) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
        }
        %88 = memref.load %alloca[] : memref<i32>
        func.call @bail_out(%88) : (i32) -> ()
        %89 = arith.index_cast %80 : i32 to index
        %90 = polygeist.subindex %86[%89] () : memref<?xf64> -> memref<?xf64>
        memref.store %90, %alloca_6[] : memref<memref<?xf64>>
        %91 = arith.addi %89, %89 : index
        %92 = polygeist.subindex %86[%91] () : memref<?xf64> -> memref<?xf64>
        memref.store %92, %alloca_5[] : memref<memref<?xf64>>
      }
      %76 = memref.load %alloca[] : memref<i32>
      func.call @bail_out(%76) : (i32) -> ()
      scf.if %25 {
        func.call @exit(%c0_i32) : (i32) -> ()
      }
      %77 = arith.addi %21, %c1_i32 : i32
      %78 = arith.index_cast %77 : i32 to index
      scf.for %arg2 = %c0 to %78 step %c1 {
        scf.if %55#1 {
          %79 = arith.subi %c1_i32, %55#0 : i32
          %80 = arith.extsi %79 : i32 to i64
          %81 = arith.subi %26, %80 : i64
          %82 = arith.extsi %55#0 : i32 to i64
          %83 = arith.divsi %81, %82 : i64
          %84 = arith.trunci %83 : i64 to i32
          %85 = arith.muli %84, %55#0 : i32
          %86 = arith.index_cast %85 : i32 to index
          %87 = arith.index_cast %55#0 : i32 to index
          arts.for(%c0) to(%86) step(%87) {{
          ^bb0(%arg3: index):
            %88 = arith.index_cast %arg3 : index to i32
            %89 = arith.addi %88, %55#0 : i32
            %90 = arith.extsi %89 : i32 to i64
            %91 = arith.cmpi slt, %90, %26 : i64
            %92 = arith.select %91, %90, %26 : i64
            %93 = arith.addi %55#0, %c12_i32 : i32
            %94 = arith.index_cast %26 : i64 to index
            scf.for %arg4 = %c0 to %94 step %87 {
              %95 = arith.divui %arg4, %87 : index
              %96 = arith.muli %95, %87 : index
              %97 = arith.index_cast %96 : index to i32
              %98 = arith.addi %97, %55#0 : i32
              %99 = arith.extsi %98 : i32 to i64
              %100 = arith.cmpi slt, %99, %26 : i64
              %101 = arith.select %100, %99, %26 : i64
              %102 = arith.index_cast %92 : i64 to index
              scf.for %arg5 = %arg3 to %102 step %c1 {
                %103 = arith.subi %arg5, %arg3 : index
                %104 = arith.index_cast %103 : index to i32
                %105 = arith.index_cast %arg5 : index to i32
                %106 = arith.extsi %105 : i32 to i64
                %107 = arith.muli %26, %106 : i64
                %108 = arith.index_cast %101 : i64 to index
                scf.for %arg6 = %96 to %108 step %c1 {
                  %109 = arith.subi %arg6, %96 : index
                  %110 = arith.index_cast %109 : index to i32
                  %111 = arith.index_cast %arg6 : index to i32
                  %112 = arith.extsi %111 : i32 to i64
                  %113 = memref.load %alloca_6[] : memref<memref<?xf64>>
                  %114 = arith.muli %93, %110 : i32
                  %115 = arith.addi %104, %114 : i32
                  %116 = arith.index_cast %115 : i32 to index
                  %117 = memref.load %3[%c0] : memref<1xmemref<?xf64>>
                  %118 = arith.addi %112, %107 : i64
                  %119 = arith.index_cast %118 : i64 to index
                  %120 = memref.load %117[%119] : memref<?xf64>
                  memref.store %120, %113[%116] : memref<?xf64>
                }
              }
              scf.for %arg5 = %c0 to %94 step %87 {
                %103 = arith.divui %arg5, %87 : index
                %104 = arith.muli %103, %87 : index
                %105 = arith.index_cast %104 : index to i32
                %106 = arith.addi %105, %55#0 : i32
                %107 = arith.extsi %106 : i32 to i64
                %108 = arith.cmpi slt, %107, %26 : i64
                %109 = arith.select %108, %107, %26 : i64
                %110 = arith.index_cast %101 : i64 to index
                scf.for %arg6 = %96 to %110 step %c1 {
                  %111 = arith.subi %arg6, %96 : index
                  %112 = arith.index_cast %111 : index to i32
                  %113 = arith.index_cast %arg6 : index to i32
                  %114 = arith.extsi %113 : i32 to i64
                  %115 = arith.muli %93, %112 : i32
                  %116 = arith.muli %26, %114 : i64
                  %117 = arith.index_cast %109 : i64 to index
                  scf.for %arg7 = %104 to %117 step %c1 {
                    %118 = arith.subi %arg7, %104 : index
                    %119 = arith.index_cast %118 : index to i32
                    %120 = arith.index_cast %arg7 : index to i32
                    %121 = arith.extsi %120 : i32 to i64
                    %122 = memref.load %alloca_7[] : memref<memref<?xf64>>
                    %123 = arith.addi %119, %115 : i32
                    %124 = arith.index_cast %123 : i32 to index
                    %125 = memref.load %4[%c0] : memref<1xmemref<?xf64>>
                    %126 = arith.addi %121, %116 : i64
                    %127 = arith.index_cast %126 : i64 to index
                    %128 = memref.load %125[%127] : memref<?xf64>
                    memref.store %128, %122[%124] : memref<?xf64>
                  }
                }
                scf.for %arg6 = %arg3 to %102 step %c1 {
                  %111 = arith.subi %arg6, %arg3 : index
                  %112 = arith.index_cast %111 : index to i32
                  %113 = arith.muli %93, %112 : i32
                  %114 = arith.index_cast %109 : i64 to index
                  scf.for %arg7 = %104 to %114 step %c1 {
                    %115 = arith.subi %arg7, %104 : index
                    %116 = arith.index_cast %115 : index to i32
                    %117 = memref.load %alloca_5[] : memref<memref<?xf64>>
                    %118 = arith.addi %116, %113 : i32
                    %119 = arith.index_cast %118 : i32 to index
                    memref.store %cst_4, %117[%119] : memref<?xf64>
                  }
                }
                scf.for %arg6 = %96 to %110 step %c1 {
                  %111 = arith.subi %arg6, %96 : index
                  %112 = arith.index_cast %111 : index to i32
                  %113 = arith.muli %93, %112 : i32
                  scf.for %arg7 = %arg3 to %102 step %c1 {
                    %114 = arith.subi %arg7, %arg3 : index
                    %115 = arith.index_cast %114 : index to i32
                    %116 = arith.muli %93, %115 : i32
                    %117 = arith.addi %115, %113 : i32
                    %118 = arith.index_cast %117 : i32 to index
                    %119 = arith.index_cast %109 : i64 to index
                    scf.for %arg8 = %104 to %119 step %c1 {
                      %120 = arith.subi %arg8, %104 : index
                      %121 = arith.index_cast %120 : index to i32
                      %122 = memref.load %alloca_5[] : memref<memref<?xf64>>
                      %123 = arith.addi %121, %116 : i32
                      %124 = arith.index_cast %123 : i32 to index
                      %125 = memref.load %alloca_7[] : memref<memref<?xf64>>
                      %126 = arith.addi %121, %113 : i32
                      %127 = arith.index_cast %126 : i32 to index
                      %128 = memref.load %125[%127] : memref<?xf64>
                      %129 = memref.load %alloca_6[] : memref<memref<?xf64>>
                      %130 = memref.load %129[%118] : memref<?xf64>
                      %131 = arith.mulf %128, %130 : f64
                      %132 = memref.load %122[%124] : memref<?xf64>
                      %133 = arith.addf %132, %131 : f64
                      memref.store %133, %122[%124] : memref<?xf64>
                    }
                  }
                }
                scf.for %arg6 = %arg3 to %102 step %c1 {
                  %111 = arith.subi %arg6, %arg3 : index
                  %112 = arith.index_cast %111 : index to i32
                  %113 = arith.index_cast %arg6 : index to i32
                  %114 = arith.extsi %113 : i32 to i64
                  %115 = arith.muli %26, %114 : i64
                  %116 = arith.muli %93, %112 : i32
                  %117 = arith.index_cast %109 : i64 to index
                  scf.for %arg7 = %104 to %117 step %c1 {
                    %118 = arith.subi %arg7, %104 : index
                    %119 = arith.index_cast %118 : index to i32
                    %120 = arith.index_cast %arg7 : index to i32
                    %121 = arith.extsi %120 : i32 to i64
                    %122 = memref.load %2[%c0] : memref<1xmemref<?xf64>>
                    %123 = arith.addi %121, %115 : i64
                    %124 = arith.index_cast %123 : i64 to index
                    %125 = memref.load %alloca_5[] : memref<memref<?xf64>>
                    %126 = arith.addi %119, %116 : i32
                    %127 = arith.index_cast %126 : i32 to index
                    %128 = memref.load %125[%127] : memref<?xf64>
                    %129 = memref.load %122[%124] : memref<?xf64>
                    %130 = arith.addf %129, %128 : f64
                    memref.store %130, %122[%124] : memref<?xf64>
                  }
                }
              }
            }
          }}
        } else {
          arts.for(%c0) to(%53) step(%c1) {{
          ^bb0(%arg3: index):
            %79 = arith.index_cast %arg3 : index to i32
            %80 = arith.extsi %79 : i32 to i64
            %81 = arith.muli %26, %80 : i64
            %82 = arith.index_cast %26 : i64 to index
            scf.for %arg4 = %c0 to %82 step %c1 {
              %83 = arith.index_cast %arg4 : index to i32
              %84 = arith.extsi %83 : i32 to i64
              %85 = arith.muli %26, %84 : i64
              %86 = arith.addi %84, %81 : i64
              %87 = arith.index_cast %86 : i64 to index
              scf.for %arg5 = %c0 to %82 step %c1 {
                %88 = arith.index_cast %arg5 : index to i32
                %89 = arith.extsi %88 : i32 to i64
                %90 = memref.load %2[%c0] : memref<1xmemref<?xf64>>
                %91 = arith.addi %89, %81 : i64
                %92 = arith.index_cast %91 : i64 to index
                %93 = memref.load %4[%c0] : memref<1xmemref<?xf64>>
                %94 = arith.addi %89, %85 : i64
                %95 = arith.index_cast %94 : i64 to index
                %96 = memref.load %93[%95] : memref<?xf64>
                %97 = memref.load %3[%c0] : memref<1xmemref<?xf64>>
                %98 = memref.load %97[%87] : memref<?xf64>
                %99 = arith.mulf %96, %98 : f64
                %100 = memref.load %90[%92] : memref<?xf64>
                %101 = arith.addf %100, %99 : f64
                memref.store %101, %90[%92] : memref<?xf64>
              }
            }
          }}
        }
      }
    }
    %56 = arith.index_cast %26 : i64 to index
    %57 = scf.for %arg2 = %c0 to %56 step %c1 iter_args(%arg3 = %cst_4) -> (f64) {
      %76 = arith.index_cast %arg2 : index to i32
      %77 = arith.extsi %76 : i32 to i64
      %78 = arith.muli %26, %77 : i64
      %79 = scf.for %arg4 = %c0 to %56 step %c1 iter_args(%arg5 = %arg3) -> (f64) {
        %80 = arith.index_cast %arg4 : index to i32
        %81 = arith.extsi %80 : i32 to i64
        %82 = memref.load %2[%c0] : memref<1xmemref<?xf64>>
        %83 = arith.addi %81, %78 : i64
        %84 = arith.index_cast %83 : i64 to index
        %85 = memref.load %82[%84] : memref<?xf64>
        %86 = arith.addf %arg5, %85 : f64
        scf.yield %86 : f64
      }
      scf.yield %79 : f64
    }
    %58 = arith.addi %21, %c1_i32 : i32
    %59 = arith.sitofp %58 : i32 to f64
    %60 = arith.mulf %51, %59 : f64
    %61 = arith.subf %57, %60 : f64
    %62 = arith.divf %61, %60 : f64
    %63 = arith.cmpf oge, %62, %cst_4 : f64
    %64 = scf.if %63 -> (f64) {
      scf.yield %62 : f64
    } else {
      %76 = arith.negf %62 : f64
      scf.yield %76 : f64
    }
    %65 = arith.cmpf ogt, %64, %cst_3 : f64
    scf.if %65 {
      %76 = llvm.mlir.addressof @str9 : !llvm.ptr
      %77 = llvm.getelementptr %76[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<49 x i8>
      %78 = llvm.call @printf(%77, %57, %60) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    } else {
      %76 = llvm.mlir.addressof @str10 : !llvm.ptr
      %77 = llvm.getelementptr %76[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<20 x i8>
      %78 = llvm.call @printf(%77) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    }
    %66 = arith.mulf %45, %cst_0 : f64
    %67 = arith.mulf %66, %45 : f64
    %68 = arith.mulf %67, %45 : f64
    %69 = arith.sitofp %21 : i32 to f64
    %70 = arith.divf %0, %69 : f64
    %71 = llvm.mlir.addressof @str11 : !llvm.ptr
    %72 = llvm.getelementptr %71[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<41 x i8>
    %73 = arith.mulf %68, %cst : f64
    %74 = arith.divf %73, %70 : f64
    %75 = llvm.call @printf(%72, %74, %70) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64) -> i32
    call @exit(%c0_i32) : (i32) -> ()
    return %1 : i32
  }
  func.func private @exit(i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @atoi(memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
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
