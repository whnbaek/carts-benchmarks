module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:o-i64:64-i128:128-n32:64-S128", llvm.target_triple = "arm64-apple-macosx15.0.0", "polygeist.target-cpu" = "apple-m1", "polygeist.target-features" = "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"} {
  llvm.mlir.global internal constant @str21("(a & (~a+1)) == a\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str20("par-res-kern_general.h\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str19("prk_get_alignment\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str18("PRK_ALIGNMENT\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str17("Rate (MFlops/s): %lf  Avg time (s): %lf\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str16("Solution validates\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str15("ERROR: Vector sum = %lf, Reference vector sum = %lf\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str14("ERROR: Could not allocate space for column indices: %16llu\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str13("ERROR: Cannot represent space for column indices: %zu\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str12("ERROR: Could not allocate space for vectors: %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str11("ERROR: Cannot represent space for vectors: %zu\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str10("ERROR: Could not allocate space for sparse matrix: %16llu\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str9("ERROR: Cannot represent space for matrix: %zu\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str8("ERROR: Grid extent %d smaller than stencil diameter 2*%d+1= %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str7("ERROR: Stencil radius must be non-negative: %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str6("ERROR: Log of grid size must be greater than or equal to zero: %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str5("ERROR: Iterations must be positive : %d \0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str4("ERROR: Invalid number of threads: %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str3("Usage: %s <# threads> <# iterations> <2log grid size> <stencil radius>\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2("OpenMP Sparse matrix-vector multiplication\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("2.17\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str0("Parallel Research Kernels version %s\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %true = arith.constant true
    %c16_i64 = arith.constant 16 : i64
    %c8 = arith.constant 8 : index
    %c8_i64 = arith.constant 8 : i64
    %cst = arith.constant 2.000000e+00 : f64
    %cst_0 = arith.constant 9.9999999999999995E-7 : f64
    %cst_1 = arith.constant 5.000000e-01 : f64
    %cst_2 = arith.constant 1.000000e+00 : f64
    %c4_i64 = arith.constant 4 : i64
    %c3_i64 = arith.constant 3 : i64
    %cst_3 = arith.constant 0.000000e+00 : f64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    %c5_i32 = arith.constant 5 : i32
    %cst_4 = arith.constant 1.000000e-08 : f64
    %0 = llvm.mlir.undef : i32
    %1 = llvm.mlir.undef : f64
    %2 = llvm.mlir.addressof @str0 : !llvm.ptr
    %3 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
    %4 = llvm.mlir.addressof @str1 : !llvm.ptr
    %5 = llvm.getelementptr %4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<5 x i8>
    %6 = llvm.call @printf(%3, %5) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    %7 = llvm.mlir.addressof @str2 : !llvm.ptr
    %8 = llvm.getelementptr %7[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<44 x i8>
    %9 = llvm.call @printf(%8) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %10 = arith.cmpi ne, %arg0, %c5_i32 : i32
    scf.if %10 {
      %74 = llvm.mlir.addressof @str3 : !llvm.ptr
      %75 = llvm.getelementptr %74[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<72 x i8>
      %76 = memref.load %arg1[%c0] : memref<?xmemref<?xi8>>
      %77 = polygeist.memref2pointer %76 : memref<?xi8> to !llvm.ptr
      %78 = llvm.call @printf(%75, %77) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %11 = memref.load %arg1[%c1] : memref<?xmemref<?xi8>>
    %12 = call @atoi(%11) : (memref<?xi8>) -> i32
    %13 = arith.cmpi slt, %12, %c1_i32 : i32
    %14 = scf.if %13 -> (i1) {
      scf.yield %true : i1
    } else {
      %74 = arith.cmpi sgt, %12, %c512_i32 : i32
      scf.yield %74 : i1
    }
    scf.if %14 {
      %74 = llvm.mlir.addressof @str4 : !llvm.ptr
      %75 = llvm.getelementptr %74[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
      %76 = llvm.call @printf(%75, %12) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    call @omp_set_num_threads(%12) : (i32) -> ()
    %15 = memref.load %arg1[%c2] : memref<?xmemref<?xi8>>
    %16 = call @atoi(%15) : (memref<?xi8>) -> i32
    %17 = arith.cmpi slt, %16, %c1_i32 : i32
    scf.if %17 {
      %74 = llvm.mlir.addressof @str5 : !llvm.ptr
      %75 = llvm.getelementptr %74[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<42 x i8>
      %76 = llvm.call @printf(%75, %16) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %18 = memref.load %arg1[%c3] : memref<?xmemref<?xi8>>
    %19 = call @atoi(%18) : (memref<?xi8>) -> i32
    %20 = arith.shli %c1_i32, %19 : i32
    %21 = arith.cmpi slt, %19, %c0_i32 : i32
    scf.if %21 {
      %74 = llvm.mlir.addressof @str6 : !llvm.ptr
      %75 = llvm.getelementptr %74[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<67 x i8>
      %76 = llvm.call @printf(%75, %19) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %22 = arith.muli %20, %20 : i32
    %23 = arith.extsi %22 : i32 to i64
    %24 = memref.load %arg1[%c4] : memref<?xmemref<?xi8>>
    %25 = call @atoi(%24) : (memref<?xi8>) -> i32
    %26 = arith.cmpi slt, %25, %c0_i32 : i32
    scf.if %26 {
      %74 = llvm.mlir.addressof @str7 : !llvm.ptr
      %75 = llvm.getelementptr %74[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<48 x i8>
      %76 = llvm.call @printf(%75, %20) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %27 = arith.muli %25, %c2_i32 : i32
    %28 = arith.addi %27, %c1_i32 : i32
    %29 = arith.cmpi slt, %20, %28 : i32
    scf.if %29 {
      %74 = llvm.mlir.addressof @str8 : !llvm.ptr
      %75 = llvm.getelementptr %74[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x i8>
      %76 = llvm.call @printf(%75, %20, %25, %28) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32, i32) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %30 = arith.muli %25, %c4_i32 : i32
    %31 = arith.addi %30, %c1_i32 : i32
    %32 = arith.extsi %31 : i32 to i64
    %33 = arith.muli %23, %32 : i64
    %34 = arith.muli %33, %c8_i64 : i64
    %35 = arith.divui %34, %c8_i64 : i64
    %36 = arith.cmpi ne, %35, %33 : i64
    scf.if %36 {
      %74 = llvm.mlir.addressof @str9 : !llvm.ptr
      %75 = llvm.getelementptr %74[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<47 x i8>
      %76 = llvm.call @printf(%75, %34) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %37 = call @prk_malloc(%34) : (i64) -> memref<?xi8>
    %38 = polygeist.memref2pointer %37 : memref<?xi8> to !llvm.ptr
    %39 = llvm.mlir.zero : !llvm.ptr
    %40 = llvm.icmp "eq" %38, %39 : !llvm.ptr
    scf.if %40 {
      %74 = llvm.mlir.addressof @str10 : !llvm.ptr
      %75 = llvm.getelementptr %74[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<59 x i8>
      %76 = llvm.call @printf(%75, %33) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %41 = arith.muli %23, %c16_i64 : i64
    %42 = arith.divui %41, %c8_i64 : i64
    %43 = arith.muli %23, %c2_i64 : i64
    %44 = arith.cmpi ne, %42, %43 : i64
    scf.if %44 {
      %74 = llvm.mlir.addressof @str11 : !llvm.ptr
      %75 = llvm.getelementptr %74[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<48 x i8>
      %76 = llvm.call @printf(%75, %41) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %45 = call @prk_malloc(%41) : (i64) -> memref<?xi8>
    %46 = polygeist.memref2pointer %45 : memref<?xi8> to !llvm.ptr
    %47 = llvm.icmp "eq" %46, %39 : !llvm.ptr
    scf.if %47 {
      %74 = llvm.mlir.addressof @str12 : !llvm.ptr
      %75 = llvm.getelementptr %74[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<49 x i8>
      %76 = arith.trunci %43 : i64 to i32
      %77 = llvm.call @printf(%75, %76) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %48 = arith.index_cast %22 : i32 to index
    scf.if %36 {
      %74 = llvm.mlir.addressof @str13 : !llvm.ptr
      %75 = llvm.getelementptr %74[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<55 x i8>
      %76 = llvm.call @printf(%75, %34) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %49 = call @prk_malloc(%34) : (i64) -> memref<?xi8>
    %50 = polygeist.memref2pointer %49 : memref<?xi8> to !llvm.ptr
    %51 = llvm.icmp "eq" %50, %39 : !llvm.ptr
    scf.if %51 {
      %74 = llvm.mlir.addressof @str14 : !llvm.ptr
      %75 = llvm.getelementptr %74[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<60 x i8>
      %76 = llvm.call @printf(%75, %34) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    arts.edt <parallel> <internode> route(%c0_i32) {
      func.call @bail_out(%c0_i32) : (i32) -> ()
      arts.for(%c0) to(%48) step(%c1) {{
      ^bb0(%arg2: index):
        %76 = arith.index_cast %arg2 : index to i32
        %77 = llvm.getelementptr %46[%76] : (!llvm.ptr, i32) -> !llvm.ptr, f64
        llvm.store %cst_3, %77 : f64, !llvm.ptr
        %78 = llvm.load %77 : !llvm.ptr -> f64
        %79 = arith.addi %arg2, %48 : index
        %80 = arith.index_cast %79 : index to i32
        %81 = llvm.getelementptr %46[%80] : (!llvm.ptr, i32) -> !llvm.ptr, f64
        llvm.store %78, %81 : f64, !llvm.ptr
      }}
      arts.for(%c0) to(%48) step(%c1) {{
      ^bb0(%arg2: index):
        %76 = arith.index_cast %arg2 : index to i64
        %77 = arith.extsi %20 : i32 to i64
        %78 = arith.divsi %76, %77 : i64
        %79 = arith.remsi %76, %77 : i64
        %80 = arith.muli %76, %32 : i64
        %81 = arith.index_cast %80 : i64 to index
        %82 = arith.extui %19 : i32 to i64
        %83 = arith.shli %78, %82 : i64
        %84 = arith.addi %79, %83 : i64
        %85 = arith.index_cast %81 : index to i32
        %86 = llvm.getelementptr %50[%85] : (!llvm.ptr, i32) -> !llvm.ptr, i64
        llvm.store %84, %86 : i64, !llvm.ptr
        %87 = arith.addi %25, %c1_i32 : i32
        %88 = arith.index_cast %87 : i32 to index
        scf.for %arg3 = %c1 to %88 step %c1 {
          %98 = arith.subi %arg3, %c1 : index
          %99 = arith.muli %98, %c4 : index
          %100 = arith.addi %81, %99 : index
          %101 = arith.index_cast %100 : index to i64
          %102 = arith.index_cast %arg3 : index to i32
          %103 = arith.addi %101, %c1_i64 : i64
          %104 = arith.index_cast %103 : i64 to index
          %105 = arith.extsi %102 : i32 to i64
          %106 = arith.addi %79, %105 : i64
          %107 = arith.remsi %106, %77 : i64
          %108 = arith.addi %107, %83 : i64
          %109 = arith.index_cast %104 : index to i32
          %110 = llvm.getelementptr %50[%109] : (!llvm.ptr, i32) -> !llvm.ptr, i64
          llvm.store %108, %110 : i64, !llvm.ptr
          %111 = arith.addi %101, %c2_i64 : i64
          %112 = arith.index_cast %111 : i64 to index
          %113 = arith.subi %79, %105 : i64
          %114 = arith.addi %113, %77 : i64
          %115 = arith.remsi %114, %77 : i64
          %116 = arith.addi %115, %83 : i64
          %117 = arith.index_cast %112 : index to i32
          %118 = llvm.getelementptr %50[%117] : (!llvm.ptr, i32) -> !llvm.ptr, i64
          llvm.store %116, %118 : i64, !llvm.ptr
          %119 = arith.addi %101, %c3_i64 : i64
          %120 = arith.index_cast %119 : i64 to index
          %121 = arith.addi %78, %105 : i64
          %122 = arith.remsi %121, %77 : i64
          %123 = arith.shli %122, %82 : i64
          %124 = arith.addi %79, %123 : i64
          %125 = arith.index_cast %120 : index to i32
          %126 = llvm.getelementptr %50[%125] : (!llvm.ptr, i32) -> !llvm.ptr, i64
          llvm.store %124, %126 : i64, !llvm.ptr
          %127 = arith.addi %101, %c4_i64 : i64
          %128 = arith.index_cast %127 : i64 to index
          %129 = arith.subi %78, %105 : i64
          %130 = arith.addi %129, %77 : i64
          %131 = arith.remsi %130, %77 : i64
          %132 = arith.shli %131, %82 : i64
          %133 = arith.addi %79, %132 : i64
          %134 = arith.index_cast %128 : index to i32
          %135 = llvm.getelementptr %50[%134] : (!llvm.ptr, i32) -> !llvm.ptr, i64
          llvm.store %133, %135 : i64, !llvm.ptr
        }
        %89 = arith.muli %81, %c8 : index
        %90 = arith.index_cast %89 : index to i64
        %91 = llvm.getelementptr %50[%90] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %92 = polygeist.pointer2memref %91 : !llvm.ptr to memref<?xi8>
        %93 = polygeist.get_func @compare : !llvm.ptr
        %94 = polygeist.pointer2memref %93 : !llvm.ptr to memref<?x!llvm.func<i32 (memref<?xi8>, memref<?xi8>)>>
        func.call @qsort(%92, %32, %c8_i64, %94) : (memref<?xi8>, i64, i64, memref<?x!llvm.func<i32 (memref<?xi8>, memref<?xi8>)>>) -> ()
        %95 = arith.addi %76, %c1_i64 : i64
        %96 = arith.muli %95, %32 : i64
        %97 = arith.index_cast %96 : i64 to index
        scf.for %arg3 = %81 to %97 step %c1 {
          %98 = arith.index_cast %arg3 : index to i32
          %99 = llvm.getelementptr %50[%98] : (!llvm.ptr, i32) -> !llvm.ptr, i64
          %100 = llvm.load %99 : !llvm.ptr -> i64
          %101 = arith.addi %100, %c1_i64 : i64
          %102 = arith.sitofp %101 : i64 to f64
          %103 = arith.divf %cst_2, %102 : f64
          %104 = llvm.getelementptr %38[%98] : (!llvm.ptr, i32) -> !llvm.ptr, f64
          llvm.store %103, %104 : f64, !llvm.ptr
        }
      }}
      %74 = arith.addi %16, %c1_i32 : i32
      %75 = arith.index_cast %74 : i32 to index
      scf.for %arg2 = %c0 to %75 step %c1 {
        arts.for(%c0) to(%48) step(%c1) {{
        ^bb0(%arg3: index):
          %76 = arith.index_cast %arg3 : index to i64
          %77 = arith.addi %76, %c1_i64 : i64
          %78 = arith.sitofp %77 : i64 to f64
          %79 = arith.index_cast %arg3 : index to i32
          %80 = llvm.getelementptr %46[%79] : (!llvm.ptr, i32) -> !llvm.ptr, f64
          %81 = llvm.load %80 : !llvm.ptr -> f64
          %82 = arith.addf %81, %78 : f64
          llvm.store %82, %80 : f64, !llvm.ptr
        }}
        arts.for(%c0) to(%48) step(%c1) {{
        ^bb0(%arg3: index):
          %76 = arith.index_cast %arg3 : index to i64
          %77 = arith.muli %32, %76 : i64
          %78 = arith.addi %77, %32 : i64
          %79 = arith.index_cast %78 : i64 to index
          %80 = arith.index_cast %77 : i64 to index
          %81 = scf.for %arg4 = %80 to %79 step %c1 iter_args(%arg5 = %cst_3) -> (f64) {
            %87 = arith.index_cast %arg4 : index to i32
            %88 = llvm.getelementptr %38[%87] : (!llvm.ptr, i32) -> !llvm.ptr, f64
            %89 = llvm.load %88 : !llvm.ptr -> f64
            %90 = llvm.getelementptr %50[%87] : (!llvm.ptr, i32) -> !llvm.ptr, i64
            %91 = llvm.load %90 : !llvm.ptr -> i64
            %92 = arith.index_cast %91 : i64 to index
            %93 = arith.index_cast %92 : index to i32
            %94 = llvm.getelementptr %46[%93] : (!llvm.ptr, i32) -> !llvm.ptr, f64
            %95 = llvm.load %94 : !llvm.ptr -> f64
            %96 = arith.mulf %89, %95 : f64
            %97 = arith.addf %arg5, %96 : f64
            scf.yield %97 : f64
          }
          %82 = arith.addi %arg3, %48 : index
          %83 = arith.index_cast %82 : index to i32
          %84 = llvm.getelementptr %46[%83] : (!llvm.ptr, i32) -> !llvm.ptr, f64
          %85 = llvm.load %84 : !llvm.ptr -> f64
          %86 = arith.addf %85, %81 : f64
          llvm.store %86, %84 : f64, !llvm.ptr
        }}
      }
    }
    %52 = arith.sitofp %33 : i64 to f64
    %53 = arith.mulf %52, %cst_1 : f64
    %54 = arith.addi %16, %c1_i32 : i32
    %55 = arith.sitofp %54 : i32 to f64
    %56 = arith.mulf %53, %55 : f64
    %57 = arith.addi %16, %c2_i32 : i32
    %58 = arith.sitofp %57 : i32 to f64
    %59 = arith.mulf %56, %58 : f64
    %60 = arith.index_cast %22 : i32 to index
    %61 = scf.for %arg2 = %c0 to %60 step %c1 iter_args(%arg3 = %cst_3) -> (f64) {
      %74 = arith.addi %arg2, %48 : index
      %75 = arith.index_cast %74 : index to i32
      %76 = llvm.getelementptr %46[%75] : (!llvm.ptr, i32) -> !llvm.ptr, f64
      %77 = llvm.load %76 : !llvm.ptr -> f64
      %78 = arith.addf %arg3, %77 : f64
      scf.yield %78 : f64
    }
    %62 = arith.subf %61, %59 : f64
    %63 = arith.cmpf oge, %62, %cst_3 : f64
    %64 = scf.if %63 -> (f64) {
      scf.yield %62 : f64
    } else {
      %74 = arith.negf %62 : f64
      scf.yield %74 : f64
    }
    %65 = arith.cmpf ogt, %64, %cst_4 : f64
    scf.if %65 {
      %74 = llvm.mlir.addressof @str15 : !llvm.ptr
      %75 = llvm.getelementptr %74[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<53 x i8>
      %76 = llvm.call @printf(%75, %61, %59) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    } else {
      %74 = llvm.mlir.addressof @str16 : !llvm.ptr
      %75 = llvm.getelementptr %74[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<20 x i8>
      %76 = llvm.call @printf(%75) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    }
    %66 = arith.sitofp %16 : i32 to f64
    %67 = arith.divf %1, %66 : f64
    %68 = llvm.mlir.addressof @str17 : !llvm.ptr
    %69 = llvm.getelementptr %68[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<41 x i8>
    %70 = arith.mulf %52, %cst : f64
    %71 = arith.mulf %70, %cst_0 : f64
    %72 = arith.divf %71, %67 : f64
    %73 = llvm.call @printf(%69, %72, %67) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64) -> i32
    call @exit(%c0_i32) : (i32) -> ()
    return %0 : i32
  }
  func.func private @exit(i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @atoi(memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
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
  func.func private @qsort(memref<?xi8>, i64, i64, memref<?x!llvm.func<i32 (memref<?xi8>, memref<?xi8>)>>) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @compare(%arg0: memref<?xi8>, %arg1: memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c-1_i32 = arith.constant -1 : i32
    %0 = polygeist.memref2pointer %arg0 : memref<?xi8> to !llvm.ptr
    %1 = llvm.load %0 : !llvm.ptr -> i64
    %2 = polygeist.memref2pointer %arg1 : memref<?xi8> to !llvm.ptr
    %3 = llvm.load %2 : !llvm.ptr -> i64
    %4 = arith.cmpi slt, %1, %3 : i64
    %5 = scf.if %4 -> (i32) {
      scf.yield %c-1_i32 : i32
    } else {
      %6 = arith.cmpi sgt, %1, %3 : i64
      %7 = arith.extui %6 : i1 to i32
      scf.yield %7 : i32
    }
    return %5 : i32
  }
  func.func private @prk_get_alignment() -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c8_i32 = arith.constant 8 : i32
    %false = arith.constant false
    %c107_i32 = arith.constant 107 : i32
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = llvm.mlir.addressof @str18 : !llvm.ptr
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
      %12 = llvm.mlir.addressof @str19 : !llvm.ptr
      %13 = llvm.mlir.addressof @str20 : !llvm.ptr
      %14 = llvm.mlir.addressof @str21 : !llvm.ptr
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
