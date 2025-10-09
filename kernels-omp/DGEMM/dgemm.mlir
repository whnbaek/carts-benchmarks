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
    %true = arith.constant true
    %c1 = arith.constant 1 : index
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
    %c0 = arith.constant 0 : index
    %0 = llvm.mlir.undef : f64
    %1 = llvm.mlir.undef : i32
    %2 = memref.get_global @"main@static@C" : memref<1xmemref<?xf64>>
    %3 = memref.get_global @"main@static@B" : memref<1xmemref<?xf64>>
    %4 = memref.get_global @"main@static@A" : memref<1xmemref<?xf64>>
    %alloca = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
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
      %75 = llvm.mlir.addressof @str3 : !llvm.ptr
      %76 = llvm.getelementptr %75[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<65 x i8>
      %77 = affine.load %arg1[0] : memref<?xmemref<?xi8>>
      %78 = polygeist.memref2pointer %77 : memref<?xi8> to !llvm.ptr
      %79 = llvm.call @printf(%76, %78) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %16 = affine.load %arg1[1] : memref<?xmemref<?xi8>>
    %17 = call @atoi(%16) : (memref<?xi8>) -> i32
    %18 = arith.cmpi slt, %17, %c1_i32 : i32
    %19 = scf.if %18 -> (i1) {
      scf.yield %true : i1
    } else {
      %75 = arith.cmpi sgt, %17, %c512_i32 : i32
      scf.yield %75 : i1
    }
    scf.if %19 {
      %75 = llvm.mlir.addressof @str4 : !llvm.ptr
      %76 = llvm.getelementptr %75[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
      %77 = llvm.call @printf(%76, %17) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    call @omp_set_num_threads(%17) : (i32) -> ()
    %20 = affine.load %arg1[2] : memref<?xmemref<?xi8>>
    %21 = call @atoi(%20) : (memref<?xi8>) -> i32
    %22 = arith.cmpi slt, %21, %c1_i32 : i32
    scf.if %22 {
      %75 = llvm.mlir.addressof @str5 : !llvm.ptr
      %76 = llvm.getelementptr %75[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<42 x i8>
      %77 = llvm.call @printf(%76, %21) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %23 = affine.load %arg1[3] : memref<?xmemref<?xi8>>
    %24 = call @atol(%23) : (memref<?xi8>) -> i64
    %25 = arith.cmpi slt, %24, %c0_i64 : i64
    %26 = scf.if %25 -> (i64) {
      %75 = arith.subi %c0_i64, %24 : i64
      scf.yield %75 : i64
    } else {
      scf.yield %24 : i64
    }
    %27 = arith.cmpi slt, %26, %c1_i64 : i64
    scf.if %27 {
      %75 = llvm.mlir.addressof @str6 : !llvm.ptr
      %76 = llvm.getelementptr %75[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<43 x i8>
      %77 = llvm.call @printf(%76, %26) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %28 = arith.muli %26, %26 : i64
    %29 = arith.muli %28, %c8_i64 : i64
    %30 = call @prk_malloc(%29) : (i64) -> memref<?xi8>
    %31 = polygeist.memref2pointer %30 : memref<?xi8> to !llvm.ptr
    %32 = polygeist.pointer2memref %31 : !llvm.ptr to memref<?xf64>
    affine.store %32, %4[0] : memref<1xmemref<?xf64>>
    %33 = call @prk_malloc(%29) : (i64) -> memref<?xi8>
    %34 = polygeist.memref2pointer %33 : memref<?xi8> to !llvm.ptr
    %35 = polygeist.pointer2memref %34 : !llvm.ptr to memref<?xf64>
    affine.store %35, %3[0] : memref<1xmemref<?xf64>>
    %36 = call @prk_malloc(%29) : (i64) -> memref<?xi8>
    %37 = polygeist.memref2pointer %36 : memref<?xi8> to !llvm.ptr
    %38 = polygeist.pointer2memref %37 : !llvm.ptr to memref<?xf64>
    affine.store %38, %2[0] : memref<1xmemref<?xf64>>
    %39 = affine.load %4[0] : memref<1xmemref<?xf64>>
    %40 = polygeist.memref2pointer %39 : memref<?xf64> to !llvm.ptr
    %41 = llvm.mlir.zero : !llvm.ptr
    %42 = llvm.icmp "eq" %40, %41 : !llvm.ptr
    %43 = scf.if %42 -> (i1) {
      scf.yield %true : i1
    } else {
      %75 = affine.load %3[0] : memref<1xmemref<?xf64>>
      %76 = polygeist.memref2pointer %75 : memref<?xf64> to !llvm.ptr
      %77 = llvm.icmp "eq" %76, %41 : !llvm.ptr
      scf.yield %77 : i1
    }
    %44 = scf.if %43 -> (i1) {
      scf.yield %true : i1
    } else {
      %75 = llvm.icmp "eq" %37, %41 : !llvm.ptr
      scf.yield %75 : i1
    }
    scf.if %44 {
      %75 = llvm.mlir.addressof @str7 : !llvm.ptr
      %76 = llvm.getelementptr %75[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<53 x i8>
      %77 = llvm.call @printf(%76) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
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
    scf.parallel (%arg2) = (%c0) to (%53) step (%c1) {
      %75 = arith.index_cast %arg2 : index to i32
      %76 = arith.extsi %75 : i32 to i64
      %77 = arith.muli %26, %76 : i64
      %78 = arith.sitofp %75 : i32 to f64
      %79 = scf.while (%arg3 = %c0_i32) : (i32) -> i32 {
        %80 = arith.extsi %arg3 : i32 to i64
        %81 = arith.cmpi slt, %80, %26 : i64
        scf.condition(%81) %arg3 : i32
      } do {
      ^bb0(%arg3: i32):
        %80 = arith.extsi %arg3 : i32 to i64
        %81 = affine.load %4[0] : memref<1xmemref<?xf64>>
        %82 = arith.addi %80, %77 : i64
        %83 = arith.index_cast %82 : i64 to index
        %84 = affine.load %3[0] : memref<1xmemref<?xf64>>
        memref.store %78, %84[%83] : memref<?xf64>
        %85 = memref.load %84[%83] : memref<?xf64>
        memref.store %85, %81[%83] : memref<?xf64>
        %86 = affine.load %2[0] : memref<1xmemref<?xf64>>
        memref.store %cst_4, %86[%83] : memref<?xf64>
        %87 = arith.addi %arg3, %c1_i32 : i32
        scf.yield %87 : i32
      }
      scf.yield
    }
    %54 = arith.cmpi eq, %arg0, %c5_i32 : i32
    %55:3 = scf.if %54 -> (i32, i1, i1) {
      %75 = affine.load %arg1[4] : memref<?xmemref<?xi8>>
      %76 = func.call @atoi(%75) : (memref<?xi8>) -> i32
      %77 = arith.cmpi sgt, %76, %c0_i32 : i32
      %78 = arith.cmpi sgt, %76, %c0_i32 : i32
      scf.yield %76, %77, %78 : i32, i1, i1
    } else {
      scf.yield %c32_i32, %true, %true : i32, i1, i1
    }
    omp.parallel   {
      %alloca_5 = memref.alloca() : memref<memref<?xf64>>
      %alloca_6 = memref.alloca() : memref<memref<?xf64>>
      %alloca_7 = memref.alloca() : memref<memref<?xf64>>
      scf.if %55#2 {
        %77 = arith.addi %55#0, %c12_i32 : i32
        %78 = arith.muli %55#0, %77 : i32
        %79 = arith.muli %78, %c3_i32 : i32
        %80 = arith.extsi %79 : i32 to i64
        %81 = arith.muli %80, %c8_i64 : i64
        %82 = func.call @prk_malloc(%81) : (i64) -> memref<?xi8>
        %83 = polygeist.memref2pointer %82 : memref<?xi8> to !llvm.ptr
        %84 = polygeist.pointer2memref %83 : !llvm.ptr to memref<?xf64>
        affine.store %84, %alloca_7[] : memref<memref<?xf64>>
        %85 = llvm.icmp "eq" %83, %41 : !llvm.ptr
        scf.if %85 {
          affine.store %c1_i32, %alloca[] : memref<i32>
          %96 = llvm.mlir.addressof @str8 : !llvm.ptr
          %97 = llvm.getelementptr %96[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<56 x i8>
          %98 = func.call @omp_get_thread_num() : () -> i32
          %99 = llvm.call @printf(%97, %98) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
        }
        %86 = affine.load %alloca[] : memref<i32>
        func.call @bail_out(%86) : (i32) -> ()
        %87 = arith.addi %55#0, %c12_i32 : i32
        %88 = arith.muli %55#0, %87 : i32
        %89 = arith.index_cast %88 : i32 to index
        %90 = polygeist.subindex %84[%89] () : memref<?xf64> -> memref<?xf64>
        affine.store %90, %alloca_6[] : memref<memref<?xf64>>
        %91 = arith.addi %55#0, %c12_i32 : i32
        %92 = arith.muli %55#0, %91 : i32
        %93 = arith.index_cast %92 : i32 to index
        %94 = arith.addi %93, %89 : index
        %95 = polygeist.subindex %84[%94] () : memref<?xf64> -> memref<?xf64>
        affine.store %95, %alloca_5[] : memref<memref<?xf64>>
      }
      %75 = affine.load %alloca[] : memref<i32>
      func.call @bail_out(%75) : (i32) -> ()
      scf.if %25 {
        func.call @exit(%c0_i32) : (i32) -> ()
      }
      %76 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
        %77 = arith.cmpi sle, %arg2, %21 : i32
        scf.condition(%77) %arg2 : i32
      } do {
      ^bb0(%arg2: i32):
        scf.if %55#1 {
          %78 = arith.subi %c1_i32, %55#0 : i32
          %79 = arith.extsi %78 : i32 to i64
          %80 = arith.subi %26, %79 : i64
          %81 = arith.extsi %55#0 : i32 to i64
          %82 = arith.divsi %80, %81 : i64
          %83 = arith.trunci %82 : i64 to i32
          %84 = arith.muli %83, %55#0 : i32
          %85 = arith.index_cast %84 : i32 to index
          %86 = arith.index_cast %55#0 : i32 to index
          omp.wsloop   for  (%arg3) : index = (%c0) to (%85) step (%86) {
            %87 = arith.index_cast %arg3 : index to i32
            %88 = arith.addi %87, %55#0 : i32
            %89 = arith.extsi %88 : i32 to i64
            %90 = arith.cmpi slt, %89, %26 : i64
            %91 = arith.select %90, %89, %26 : i64
            %92 = arith.addi %55#0, %c12_i32 : i32
            %93 = arith.addi %55#0, %c12_i32 : i32
            %94 = arith.addi %87, %55#0 : i32
            %95 = arith.extsi %94 : i32 to i64
            %96 = arith.cmpi slt, %95, %26 : i64
            %97 = arith.select %96, %95, %26 : i64
            %98 = arith.addi %55#0, %c12_i32 : i32
            %99 = arith.addi %87, %55#0 : i32
            %100 = arith.extsi %99 : i32 to i64
            %101 = arith.cmpi slt, %100, %26 : i64
            %102 = arith.select %101, %100, %26 : i64
            %103 = arith.addi %55#0, %c12_i32 : i32
            %104 = arith.addi %87, %55#0 : i32
            %105 = arith.extsi %104 : i32 to i64
            %106 = arith.cmpi slt, %105, %26 : i64
            %107 = arith.select %106, %105, %26 : i64
            %108 = arith.addi %55#0, %c12_i32 : i32
            %109 = scf.while (%arg4 = %c0_i32) : (i32) -> i32 {
              %110 = arith.extsi %arg4 : i32 to i64
              %111 = arith.cmpi slt, %110, %26 : i64
              scf.condition(%111) %arg4 : i32
            } do {
            ^bb0(%arg4: i32):
              %110 = arith.addi %arg4, %55#0 : i32
              %111 = arith.extsi %110 : i32 to i64
              %112 = arith.cmpi slt, %111, %26 : i64
              %113 = arith.select %112, %111, %26 : i64
              %114:2 = scf.while (%arg5 = %87, %arg6 = %c0_i32) : (i32, i32) -> (i32, i32) {
                %125 = arith.extsi %arg5 : i32 to i64
                %126 = arith.cmpi slt, %125, %91 : i64
                scf.condition(%126) %arg5, %arg6 : i32, i32
              } do {
              ^bb0(%arg5: i32, %arg6: i32):
                %125 = arith.extsi %arg5 : i32 to i64
                %126 = arith.muli %26, %125 : i64
                %127:2 = scf.while (%arg7 = %arg4, %arg8 = %c0_i32) : (i32, i32) -> (i32, i32) {
                  %130 = arith.extsi %arg7 : i32 to i64
                  %131 = arith.cmpi slt, %130, %113 : i64
                  scf.condition(%131) %arg7, %arg8 : i32, i32
                } do {
                ^bb0(%arg7: i32, %arg8: i32):
                  %130 = arith.extsi %arg7 : i32 to i64
                  %131 = affine.load %alloca_6[] : memref<memref<?xf64>>
                  %132 = arith.muli %92, %arg8 : i32
                  %133 = arith.addi %arg6, %132 : i32
                  %134 = arith.index_cast %133 : i32 to index
                  %135 = affine.load %3[0] : memref<1xmemref<?xf64>>
                  %136 = arith.addi %130, %126 : i64
                  %137 = arith.index_cast %136 : i64 to index
                  %138 = memref.load %135[%137] : memref<?xf64>
                  memref.store %138, %131[%134] : memref<?xf64>
                  %139 = arith.addi %arg8, %c1_i32 : i32
                  %140 = arith.addi %arg7, %c1_i32 : i32
                  scf.yield %140, %139 : i32, i32
                }
                %128 = arith.addi %arg6, %c1_i32 : i32
                %129 = arith.addi %arg5, %c1_i32 : i32
                scf.yield %129, %128 : i32, i32
              }
              %115 = arith.addi %arg4, %55#0 : i32
              %116 = arith.extsi %115 : i32 to i64
              %117 = arith.cmpi slt, %116, %26 : i64
              %118 = arith.select %117, %116, %26 : i64
              %119 = arith.addi %arg4, %55#0 : i32
              %120 = arith.extsi %119 : i32 to i64
              %121 = arith.cmpi slt, %120, %26 : i64
              %122 = arith.select %121, %120, %26 : i64
              %123 = scf.while (%arg5 = %c0_i32) : (i32) -> i32 {
                %125 = arith.extsi %arg5 : i32 to i64
                %126 = arith.cmpi slt, %125, %26 : i64
                scf.condition(%126) %arg5 : i32
              } do {
              ^bb0(%arg5: i32):
                %125 = arith.addi %arg5, %55#0 : i32
                %126 = arith.extsi %125 : i32 to i64
                %127 = arith.cmpi slt, %126, %26 : i64
                %128 = arith.select %127, %126, %26 : i64
                %129:2 = scf.while (%arg6 = %arg4, %arg7 = %c0_i32) : (i32, i32) -> (i32, i32) {
                  %146 = arith.extsi %arg6 : i32 to i64
                  %147 = arith.cmpi slt, %146, %118 : i64
                  scf.condition(%147) %arg6, %arg7 : i32, i32
                } do {
                ^bb0(%arg6: i32, %arg7: i32):
                  %146 = arith.extsi %arg6 : i32 to i64
                  %147 = arith.muli %93, %arg7 : i32
                  %148 = arith.muli %26, %146 : i64
                  %149:2 = scf.while (%arg8 = %arg5, %arg9 = %c0_i32) : (i32, i32) -> (i32, i32) {
                    %152 = arith.extsi %arg8 : i32 to i64
                    %153 = arith.cmpi slt, %152, %128 : i64
                    scf.condition(%153) %arg8, %arg9 : i32, i32
                  } do {
                  ^bb0(%arg8: i32, %arg9: i32):
                    %152 = arith.extsi %arg8 : i32 to i64
                    %153 = affine.load %alloca_7[] : memref<memref<?xf64>>
                    %154 = arith.addi %arg9, %147 : i32
                    %155 = arith.index_cast %154 : i32 to index
                    %156 = affine.load %4[0] : memref<1xmemref<?xf64>>
                    %157 = arith.addi %152, %148 : i64
                    %158 = arith.index_cast %157 : i64 to index
                    %159 = memref.load %156[%158] : memref<?xf64>
                    memref.store %159, %153[%155] : memref<?xf64>
                    %160 = arith.addi %arg9, %c1_i32 : i32
                    %161 = arith.addi %arg8, %c1_i32 : i32
                    scf.yield %161, %160 : i32, i32
                  }
                  %150 = arith.addi %arg7, %c1_i32 : i32
                  %151 = arith.addi %arg6, %c1_i32 : i32
                  scf.yield %151, %150 : i32, i32
                }
                %130 = arith.addi %arg5, %55#0 : i32
                %131 = arith.extsi %130 : i32 to i64
                %132 = arith.cmpi slt, %131, %26 : i64
                %133 = arith.select %132, %131, %26 : i64
                %134:2 = scf.while (%arg6 = %87, %arg7 = %c0_i32) : (i32, i32) -> (i32, i32) {
                  %146 = arith.extsi %arg6 : i32 to i64
                  %147 = arith.cmpi slt, %146, %97 : i64
                  scf.condition(%147) %arg6, %arg7 : i32, i32
                } do {
                ^bb0(%arg6: i32, %arg7: i32):
                  %146 = arith.muli %98, %arg7 : i32
                  %147:2 = scf.while (%arg8 = %arg5, %arg9 = %c0_i32) : (i32, i32) -> (i32, i32) {
                    %150 = arith.extsi %arg8 : i32 to i64
                    %151 = arith.cmpi slt, %150, %133 : i64
                    scf.condition(%151) %arg8, %arg9 : i32, i32
                  } do {
                  ^bb0(%arg8: i32, %arg9: i32):
                    %150 = affine.load %alloca_5[] : memref<memref<?xf64>>
                    %151 = arith.addi %arg9, %146 : i32
                    %152 = arith.index_cast %151 : i32 to index
                    memref.store %cst_4, %150[%152] : memref<?xf64>
                    %153 = arith.addi %arg9, %c1_i32 : i32
                    %154 = arith.addi %arg8, %c1_i32 : i32
                    scf.yield %154, %153 : i32, i32
                  }
                  %148 = arith.addi %arg7, %c1_i32 : i32
                  %149 = arith.addi %arg6, %c1_i32 : i32
                  scf.yield %149, %148 : i32, i32
                }
                %135 = arith.addi %arg5, %55#0 : i32
                %136 = arith.extsi %135 : i32 to i64
                %137 = arith.cmpi slt, %136, %26 : i64
                %138 = arith.select %137, %136, %26 : i64
                %139:2 = scf.while (%arg6 = %arg4, %arg7 = %c0_i32) : (i32, i32) -> (i32, i32) {
                  %146 = arith.extsi %arg6 : i32 to i64
                  %147 = arith.cmpi slt, %146, %122 : i64
                  scf.condition(%147) %arg6, %arg7 : i32, i32
                } do {
                ^bb0(%arg6: i32, %arg7: i32):
                  %146 = arith.muli %103, %arg7 : i32
                  %147:2 = scf.while (%arg8 = %87, %arg9 = %c0_i32) : (i32, i32) -> (i32, i32) {
                    %150 = arith.extsi %arg8 : i32 to i64
                    %151 = arith.cmpi slt, %150, %102 : i64
                    scf.condition(%151) %arg8, %arg9 : i32, i32
                  } do {
                  ^bb0(%arg8: i32, %arg9: i32):
                    %150 = arith.muli %103, %arg9 : i32
                    %151 = arith.addi %arg9, %146 : i32
                    %152 = arith.index_cast %151 : i32 to index
                    %153:2 = scf.while (%arg10 = %arg5, %arg11 = %c0_i32) : (i32, i32) -> (i32, i32) {
                      %156 = arith.extsi %arg10 : i32 to i64
                      %157 = arith.cmpi slt, %156, %138 : i64
                      scf.condition(%157) %arg10, %arg11 : i32, i32
                    } do {
                    ^bb0(%arg10: i32, %arg11: i32):
                      %156 = affine.load %alloca_5[] : memref<memref<?xf64>>
                      %157 = arith.addi %arg11, %150 : i32
                      %158 = arith.index_cast %157 : i32 to index
                      %159 = affine.load %alloca_7[] : memref<memref<?xf64>>
                      %160 = arith.addi %arg11, %146 : i32
                      %161 = arith.index_cast %160 : i32 to index
                      %162 = memref.load %159[%161] : memref<?xf64>
                      %163 = affine.load %alloca_6[] : memref<memref<?xf64>>
                      %164 = memref.load %163[%152] : memref<?xf64>
                      %165 = arith.mulf %162, %164 : f64
                      %166 = memref.load %156[%158] : memref<?xf64>
                      %167 = arith.addf %166, %165 : f64
                      memref.store %167, %156[%158] : memref<?xf64>
                      %168 = arith.addi %arg11, %c1_i32 : i32
                      %169 = arith.addi %arg10, %c1_i32 : i32
                      scf.yield %169, %168 : i32, i32
                    }
                    %154 = arith.addi %arg9, %c1_i32 : i32
                    %155 = arith.addi %arg8, %c1_i32 : i32
                    scf.yield %155, %154 : i32, i32
                  }
                  %148 = arith.addi %arg7, %c1_i32 : i32
                  %149 = arith.addi %arg6, %c1_i32 : i32
                  scf.yield %149, %148 : i32, i32
                }
                %140 = arith.addi %arg5, %55#0 : i32
                %141 = arith.extsi %140 : i32 to i64
                %142 = arith.cmpi slt, %141, %26 : i64
                %143 = arith.select %142, %141, %26 : i64
                %144:2 = scf.while (%arg6 = %87, %arg7 = %c0_i32) : (i32, i32) -> (i32, i32) {
                  %146 = arith.extsi %arg6 : i32 to i64
                  %147 = arith.cmpi slt, %146, %107 : i64
                  scf.condition(%147) %arg6, %arg7 : i32, i32
                } do {
                ^bb0(%arg6: i32, %arg7: i32):
                  %146 = arith.extsi %arg6 : i32 to i64
                  %147 = arith.muli %26, %146 : i64
                  %148 = arith.muli %108, %arg7 : i32
                  %149:2 = scf.while (%arg8 = %arg5, %arg9 = %c0_i32) : (i32, i32) -> (i32, i32) {
                    %152 = arith.extsi %arg8 : i32 to i64
                    %153 = arith.cmpi slt, %152, %143 : i64
                    scf.condition(%153) %arg8, %arg9 : i32, i32
                  } do {
                  ^bb0(%arg8: i32, %arg9: i32):
                    %152 = arith.extsi %arg8 : i32 to i64
                    %153 = affine.load %2[0] : memref<1xmemref<?xf64>>
                    %154 = arith.addi %152, %147 : i64
                    %155 = arith.index_cast %154 : i64 to index
                    %156 = affine.load %alloca_5[] : memref<memref<?xf64>>
                    %157 = arith.addi %arg9, %148 : i32
                    %158 = arith.index_cast %157 : i32 to index
                    %159 = memref.load %156[%158] : memref<?xf64>
                    %160 = memref.load %153[%155] : memref<?xf64>
                    %161 = arith.addf %160, %159 : f64
                    memref.store %161, %153[%155] : memref<?xf64>
                    %162 = arith.addi %arg9, %c1_i32 : i32
                    %163 = arith.addi %arg8, %c1_i32 : i32
                    scf.yield %163, %162 : i32, i32
                  }
                  %150 = arith.addi %arg7, %c1_i32 : i32
                  %151 = arith.addi %arg6, %c1_i32 : i32
                  scf.yield %151, %150 : i32, i32
                }
                %145 = arith.addi %arg5, %55#0 : i32
                scf.yield %145 : i32
              }
              %124 = arith.addi %arg4, %55#0 : i32
              scf.yield %124 : i32
            }
            omp.yield
          }
        } else {
          %78 = arith.trunci %26 : i64 to i32
          %79 = arith.index_cast %78 : i32 to index
          omp.wsloop   for  (%arg3) : index = (%c0) to (%79) step (%c1) {
            %80 = arith.index_cast %arg3 : index to i32
            %81 = arith.extsi %80 : i32 to i64
            %82 = arith.muli %26, %81 : i64
            %83 = scf.while (%arg4 = %c0_i32) : (i32) -> i32 {
              %84 = arith.extsi %arg4 : i32 to i64
              %85 = arith.cmpi slt, %84, %26 : i64
              scf.condition(%85) %arg4 : i32
            } do {
            ^bb0(%arg4: i32):
              %84 = arith.extsi %arg4 : i32 to i64
              %85 = arith.muli %26, %84 : i64
              %86 = arith.addi %84, %82 : i64
              %87 = arith.index_cast %86 : i64 to index
              %88 = scf.while (%arg5 = %c0_i32) : (i32) -> i32 {
                %90 = arith.extsi %arg5 : i32 to i64
                %91 = arith.cmpi slt, %90, %26 : i64
                scf.condition(%91) %arg5 : i32
              } do {
              ^bb0(%arg5: i32):
                %90 = arith.extsi %arg5 : i32 to i64
                %91 = affine.load %2[0] : memref<1xmemref<?xf64>>
                %92 = arith.addi %90, %82 : i64
                %93 = arith.index_cast %92 : i64 to index
                %94 = affine.load %4[0] : memref<1xmemref<?xf64>>
                %95 = arith.addi %90, %85 : i64
                %96 = arith.index_cast %95 : i64 to index
                %97 = memref.load %94[%96] : memref<?xf64>
                %98 = affine.load %3[0] : memref<1xmemref<?xf64>>
                %99 = memref.load %98[%87] : memref<?xf64>
                %100 = arith.mulf %97, %99 : f64
                %101 = memref.load %91[%93] : memref<?xf64>
                %102 = arith.addf %101, %100 : f64
                memref.store %102, %91[%93] : memref<?xf64>
                %103 = arith.addi %arg5, %c1_i32 : i32
                scf.yield %103 : i32
              }
              %89 = arith.addi %arg4, %c1_i32 : i32
              scf.yield %89 : i32
            }
            omp.yield
          }
        }
        %77 = arith.addi %arg2, %c1_i32 : i32
        scf.yield %77 : i32
      }
      omp.terminator
    }
    %56:2 = scf.while (%arg2 = %cst_4, %arg3 = %c0_i32) : (f64, i32) -> (f64, i32) {
      %75 = arith.extsi %arg3 : i32 to i64
      %76 = arith.cmpi slt, %75, %26 : i64
      scf.condition(%76) %arg2, %arg3 : f64, i32
    } do {
    ^bb0(%arg2: f64, %arg3: i32):
      %75 = arith.extsi %arg3 : i32 to i64
      %76 = arith.muli %26, %75 : i64
      %77:2 = scf.while (%arg4 = %arg2, %arg5 = %c0_i32) : (f64, i32) -> (f64, i32) {
        %79 = arith.extsi %arg5 : i32 to i64
        %80 = arith.cmpi slt, %79, %26 : i64
        scf.condition(%80) %arg4, %arg5 : f64, i32
      } do {
      ^bb0(%arg4: f64, %arg5: i32):
        %79 = arith.extsi %arg5 : i32 to i64
        %80 = affine.load %2[0] : memref<1xmemref<?xf64>>
        %81 = arith.addi %79, %76 : i64
        %82 = arith.index_cast %81 : i64 to index
        %83 = memref.load %80[%82] : memref<?xf64>
        %84 = arith.addf %arg4, %83 : f64
        %85 = arith.addi %arg5, %c1_i32 : i32
        scf.yield %84, %85 : f64, i32
      }
      %78 = arith.addi %arg3, %c1_i32 : i32
      scf.yield %77#0, %78 : f64, i32
    }
    %57 = arith.addi %21, %c1_i32 : i32
    %58 = arith.sitofp %57 : i32 to f64
    %59 = arith.mulf %51, %58 : f64
    %60 = arith.subf %56#0, %59 : f64
    %61 = arith.divf %60, %59 : f64
    %62 = arith.cmpf oge, %61, %cst_4 : f64
    %63 = scf.if %62 -> (f64) {
      scf.yield %61 : f64
    } else {
      %75 = arith.negf %61 : f64
      scf.yield %75 : f64
    }
    %64 = arith.cmpf ogt, %63, %cst_3 : f64
    scf.if %64 {
      %75 = llvm.mlir.addressof @str9 : !llvm.ptr
      %76 = llvm.getelementptr %75[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<49 x i8>
      %77 = llvm.call @printf(%76, %56#0, %59) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    } else {
      %75 = llvm.mlir.addressof @str10 : !llvm.ptr
      %76 = llvm.getelementptr %75[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<20 x i8>
      %77 = llvm.call @printf(%76) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    }
    %65 = arith.mulf %45, %cst_0 : f64
    %66 = arith.mulf %65, %45 : f64
    %67 = arith.mulf %66, %45 : f64
    %68 = arith.sitofp %21 : i32 to f64
    %69 = arith.divf %0, %68 : f64
    %70 = llvm.mlir.addressof @str11 : !llvm.ptr
    %71 = llvm.getelementptr %70[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<41 x i8>
    %72 = arith.mulf %67, %cst : f64
    %73 = arith.divf %72, %69 : f64
    %74 = llvm.call @printf(%71, %73, %69) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64) -> i32
    call @exit(%c0_i32) : (i32) -> ()
    return %1 : i32
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
