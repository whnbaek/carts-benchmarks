module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:o-i64:64-i128:128-n32:64-S128", llvm.target_triple = "arm64-apple-macosx15.0.0", "polygeist.target-cpu" = "apple-m1", "polygeist.target-features" = "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"} {
  llvm.mlir.global internal constant @str15("(a & (~a+1)) == a\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str14("par-res-kern_general.h\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str13("prk_get_alignment\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str12("PRK_ALIGNMENT\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str11("ERROR: Aggregate squared error %lf exceeds threshold %e\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str10("Rate (MB/s): %lf Avg time (s): %lf\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str9("Solution validates\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str8(" ERROR: cannot allocate space for output matrix: %ld\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str7(" ERROR: cannot allocate space for input matrix: %ld\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str6("ERROR: Matrix Order must be greater than 0 : %zu \0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str5("ERROR: iterations must be >= 1 : %d \0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str4("ERROR: Invalid number of threads: %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str3("Usage: %s <# threads> <# iterations> <matrix order> [tile size]\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2("OpenMP Matrix transpose: B = A^T\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("2.17\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str0("Parallel Research Kernels version %s\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %c1 = arith.constant 1 : index
    %c8_i64 = arith.constant 8 : i64
    %cst = arith.constant 1.600000e+01 : f64
    %cst_0 = arith.constant 9.9999999999999995E-7 : f64
    %cst_1 = arith.constant 1.000000e+00 : f64
    %cst_2 = arith.constant 0.000000e+00 : f64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    %false = arith.constant false
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst_3 = arith.constant 1.000000e-08 : f64
    %c32_i32 = arith.constant 32 : i32
    %c0 = arith.constant 0 : index
    %0 = llvm.mlir.undef : i32
    %1 = llvm.mlir.undef : f64
    %2 = llvm.mlir.addressof @str0 : !llvm.ptr
    %3 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
    %4 = llvm.mlir.addressof @str1 : !llvm.ptr
    %5 = llvm.getelementptr %4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<5 x i8>
    %6 = llvm.call @printf(%3, %5) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    %7 = llvm.mlir.addressof @str2 : !llvm.ptr
    %8 = llvm.getelementptr %7[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<34 x i8>
    %9 = llvm.call @printf(%8) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %10 = arith.cmpi ne, %arg0, %c4_i32 : i32
    %11 = arith.cmpi ne, %arg0, %c5_i32 : i32
    %12 = arith.andi %10, %11 : i1
    scf.if %12 {
      %45 = llvm.mlir.addressof @str3 : !llvm.ptr
      %46 = llvm.getelementptr %45[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<65 x i8>
      %47 = affine.load %arg1[0] : memref<?xmemref<?xi8>>
      %48 = polygeist.memref2pointer %47 : memref<?xi8> to !llvm.ptr
      %49 = llvm.call @printf(%46, %48) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %13 = affine.load %arg1[1] : memref<?xmemref<?xi8>>
    %14 = call @atoi(%13) : (memref<?xi8>) -> i32
    %15 = arith.cmpi slt, %14, %c1_i32 : i32
    %16 = scf.if %15 -> (i1) {
      scf.yield %true : i1
    } else {
      %45 = arith.cmpi sgt, %14, %c512_i32 : i32
      scf.yield %45 : i1
    }
    scf.if %16 {
      %45 = llvm.mlir.addressof @str4 : !llvm.ptr
      %46 = llvm.getelementptr %45[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
      %47 = llvm.call @printf(%46, %14) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    call @omp_set_num_threads(%14) : (i32) -> ()
    %17 = affine.load %arg1[2] : memref<?xmemref<?xi8>>
    %18 = call @atoi(%17) : (memref<?xi8>) -> i32
    %19 = arith.cmpi slt, %18, %c1_i32 : i32
    scf.if %19 {
      %45 = llvm.mlir.addressof @str5 : !llvm.ptr
      %46 = llvm.getelementptr %45[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
      %47 = llvm.call @printf(%46, %18) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %20 = affine.load %arg1[3] : memref<?xmemref<?xi8>>
    %21 = call @atoi(%20) : (memref<?xi8>) -> i32
    %22 = arith.extsi %21 : i32 to i64
    %23 = arith.cmpi sle, %22, %c0_i64 : i64
    scf.if %23 {
      %45 = llvm.mlir.addressof @str6 : !llvm.ptr
      %46 = llvm.getelementptr %45[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<51 x i8>
      %47 = llvm.call @printf(%46, %22) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %24 = arith.cmpi eq, %arg0, %c5_i32 : i32
    %25:2 = scf.if %24 -> (i32, i1) {
      %45 = affine.load %arg1[4] : memref<?xmemref<?xi8>>
      %46 = func.call @atoi(%45) : (memref<?xi8>) -> i32
      %47 = arith.cmpi sgt, %46, %c0_i32 : i32
      scf.yield %46, %47 : i32, i1
    } else {
      scf.yield %c32_i32, %true : i32, i1
    }
    %26 = scf.if %25#1 -> (i1) {
      %45 = arith.extsi %25#0 : i32 to i64
      %46 = arith.cmpi slt, %45, %22 : i64
      scf.yield %46 : i1
    } else {
      scf.yield %false : i1
    }
    %27 = arith.extsi %26 : i1 to i32
    %28 = arith.cmpi eq, %27, %c0_i32 : i32
    %29 = arith.select %28, %21, %25#0 : i32
    %30 = arith.muli %22, %22 : i64
    %31 = arith.muli %30, %c8_i64 : i64
    %32 = call @prk_malloc(%31) : (i64) -> memref<?xi8>
    %33 = polygeist.memref2pointer %32 : memref<?xi8> to !llvm.ptr
    %34 = llvm.mlir.zero : !llvm.ptr
    %35 = llvm.icmp "eq" %33, %34 : !llvm.ptr
    scf.if %35 {
      %45 = llvm.mlir.addressof @str7 : !llvm.ptr
      %46 = llvm.getelementptr %45[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<53 x i8>
      %47 = llvm.call @printf(%46, %31) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %36 = call @prk_malloc(%31) : (i64) -> memref<?xi8>
    %37 = polygeist.memref2pointer %36 : memref<?xi8> to !llvm.ptr
    %38 = polygeist.pointer2memref %37 : !llvm.ptr to memref<?xf64>
    %39 = llvm.icmp "eq" %37, %34 : !llvm.ptr
    scf.if %39 {
      %45 = llvm.mlir.addressof @str8 : !llvm.ptr
      %46 = llvm.getelementptr %45[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<54 x i8>
      %47 = llvm.call @printf(%46, %31) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    %40 = arith.sitofp %22 : i64 to f64
    %41 = arith.mulf %40, %cst : f64
    %42 = arith.mulf %41, %40 : f64
    omp.parallel   {
      func.call @bail_out(%c0_i32) : (i32) -> ()
      scf.if %26 {
        %46 = arith.extsi %29 : i32 to i64
        %47 = arith.subi %c1_i64, %46 : i64
        %48 = arith.subi %22, %47 : i64
        %49 = arith.divsi %48, %46 : i64
        %50 = arith.muli %49, %46 : i64
        %51 = arith.index_cast %50 : i64 to index
        %52 = arith.index_cast %29 : i32 to index
        omp.wsloop   for  (%arg2) : index = (%c0) to (%51) step (%52) {
          %53 = arith.index_cast %arg2 : index to i64
          %54 = arith.extsi %29 : i32 to i64
          %55 = arith.addi %53, %54 : i64
          %56 = arith.cmpi slt, %22, %55 : i64
          %57 = arith.select %56, %22, %55 : i64
          %58 = arith.extsi %29 : i32 to i64
          %59 = scf.while (%arg3 = %c0_i64) : (i64) -> i64 {
            %60 = arith.cmpi slt, %arg3, %22 : i64
            scf.condition(%60) %arg3 : i64
          } do {
          ^bb0(%arg3: i64):
            %60 = arith.addi %arg3, %54 : i64
            %61 = arith.cmpi slt, %22, %60 : i64
            %62 = arith.select %61, %22, %60 : i64
            %63 = scf.while (%arg4 = %53) : (i64) -> i64 {
              %65 = arith.cmpi slt, %arg4, %57 : i64
              scf.condition(%65) %arg4 : i64
            } do {
            ^bb0(%arg4: i64):
              %65 = arith.muli %22, %arg4 : i64
              %66 = scf.while (%arg5 = %arg3) : (i64) -> i64 {
                %68 = arith.cmpi slt, %arg5, %62 : i64
                scf.condition(%68) %arg5 : i64
              } do {
              ^bb0(%arg5: i64):
                %68 = arith.addi %arg5, %65 : i64
                %69 = arith.index_cast %68 : i64 to index
                %70 = arith.addi %65, %arg5 : i64
                %71 = arith.sitofp %70 : i64 to f64
                %72 = arith.index_cast %69 : index to i32
                %73 = llvm.getelementptr %33[%72] : (!llvm.ptr, i32) -> !llvm.ptr, f64
                llvm.store %71, %73 : f64, !llvm.ptr
                %74 = arith.index_cast %69 : index to i32
                %75 = llvm.getelementptr %37[%74] : (!llvm.ptr, i32) -> !llvm.ptr, f64
                llvm.store %cst_2, %75 : f64, !llvm.ptr
                %76 = arith.addi %arg5, %c1_i64 : i64
                scf.yield %76 : i64
              }
              %67 = arith.addi %arg4, %c1_i64 : i64
              scf.yield %67 : i64
            }
            %64 = arith.addi %arg3, %58 : i64
            scf.yield %64 : i64
          }
          omp.yield
        }
      } else {
        %46 = arith.index_cast %21 : i32 to index
        omp.wsloop   for  (%arg2) : index = (%c0) to (%46) step (%c1) {
          %47 = arith.index_cast %arg2 : index to i64
          %48 = arith.muli %22, %47 : i64
          %49 = scf.while (%arg3 = %c0_i64) : (i64) -> i64 {
            %50 = arith.cmpi slt, %arg3, %22 : i64
            scf.condition(%50) %arg3 : i64
          } do {
          ^bb0(%arg3: i64):
            %50 = arith.addi %arg3, %48 : i64
            %51 = arith.index_cast %50 : i64 to index
            %52 = arith.addi %48, %arg3 : i64
            %53 = arith.sitofp %52 : i64 to f64
            %54 = arith.index_cast %51 : index to i32
            %55 = llvm.getelementptr %33[%54] : (!llvm.ptr, i32) -> !llvm.ptr, f64
            llvm.store %53, %55 : f64, !llvm.ptr
            %56 = arith.index_cast %51 : index to i32
            %57 = llvm.getelementptr %37[%56] : (!llvm.ptr, i32) -> !llvm.ptr, f64
            llvm.store %cst_2, %57 : f64, !llvm.ptr
            %58 = arith.addi %arg3, %c1_i64 : i64
            scf.yield %58 : i64
          }
          omp.yield
        }
      }
      %45 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
        %46 = arith.cmpi sle, %arg2, %18 : i32
        scf.condition(%46) %arg2 : i32
      } do {
      ^bb0(%arg2: i32):
        scf.if %26 {
          %47 = arith.extsi %29 : i32 to i64
          %48 = arith.subi %c1_i64, %47 : i64
          %49 = arith.subi %22, %48 : i64
          %50 = arith.divsi %49, %47 : i64
          %51 = arith.muli %50, %47 : i64
          %52 = arith.index_cast %51 : i64 to index
          %53 = arith.index_cast %29 : i32 to index
          omp.wsloop   for  (%arg3) : index = (%c0) to (%52) step (%53) {
            %54 = arith.index_cast %arg3 : index to i64
            %55 = arith.extsi %29 : i32 to i64
            %56 = arith.addi %54, %55 : i64
            %57 = arith.cmpi slt, %22, %56 : i64
            %58 = arith.select %57, %22, %56 : i64
            %59 = arith.extsi %29 : i32 to i64
            %60 = scf.while (%arg4 = %c0_i64) : (i64) -> i64 {
              %61 = arith.cmpi slt, %arg4, %22 : i64
              scf.condition(%61) %arg4 : i64
            } do {
            ^bb0(%arg4: i64):
              %61 = arith.addi %arg4, %55 : i64
              %62 = arith.cmpi slt, %22, %61 : i64
              %63 = arith.select %62, %22, %61 : i64
              %64 = scf.while (%arg5 = %54) : (i64) -> i64 {
                %66 = arith.cmpi slt, %arg5, %58 : i64
                scf.condition(%66) %arg5 : i64
              } do {
              ^bb0(%arg5: i64):
                %66 = arith.muli %22, %arg5 : i64
                %67 = scf.while (%arg6 = %arg4) : (i64) -> i64 {
                  %69 = arith.cmpi slt, %arg6, %63 : i64
                  scf.condition(%69) %arg6 : i64
                } do {
                ^bb0(%arg6: i64):
                  %69 = arith.addi %arg6, %66 : i64
                  %70 = arith.index_cast %69 : i64 to index
                  %71 = arith.muli %22, %arg6 : i64
                  %72 = arith.addi %arg5, %71 : i64
                  %73 = arith.index_cast %72 : i64 to index
                  %74 = arith.index_cast %73 : index to i32
                  %75 = llvm.getelementptr %33[%74] : (!llvm.ptr, i32) -> !llvm.ptr, f64
                  %76 = llvm.load %75 : !llvm.ptr -> f64
                  %77 = arith.index_cast %70 : index to i32
                  %78 = llvm.getelementptr %37[%77] : (!llvm.ptr, i32) -> !llvm.ptr, f64
                  %79 = llvm.load %78 : !llvm.ptr -> f64
                  %80 = arith.addf %79, %76 : f64
                  %81 = arith.index_cast %70 : index to i32
                  %82 = llvm.getelementptr %37[%81] : (!llvm.ptr, i32) -> !llvm.ptr, f64
                  llvm.store %80, %82 : f64, !llvm.ptr
                  %83 = arith.index_cast %73 : index to i32
                  %84 = llvm.getelementptr %33[%83] : (!llvm.ptr, i32) -> !llvm.ptr, f64
                  %85 = llvm.load %84 : !llvm.ptr -> f64
                  %86 = arith.addf %85, %cst_1 : f64
                  %87 = arith.index_cast %73 : index to i32
                  %88 = llvm.getelementptr %33[%87] : (!llvm.ptr, i32) -> !llvm.ptr, f64
                  llvm.store %86, %88 : f64, !llvm.ptr
                  %89 = arith.addi %arg6, %c1_i64 : i64
                  scf.yield %89 : i64
                }
                %68 = arith.addi %arg5, %c1_i64 : i64
                scf.yield %68 : i64
              }
              %65 = arith.addi %arg4, %59 : i64
              scf.yield %65 : i64
            }
            omp.yield
          }
        } else {
          %47 = arith.index_cast %21 : i32 to index
          omp.wsloop   for  (%arg3) : index = (%c0) to (%47) step (%c1) {
            %48 = arith.index_cast %arg3 : index to i64
            %49 = arith.muli %22, %48 : i64
            %50 = scf.while (%arg4 = %c0_i64) : (i64) -> i64 {
              %51 = arith.cmpi slt, %arg4, %22 : i64
              scf.condition(%51) %arg4 : i64
            } do {
            ^bb0(%arg4: i64):
              %51 = arith.addi %arg4, %49 : i64
              %52 = arith.index_cast %51 : i64 to index
              %53 = arith.muli %22, %arg4 : i64
              %54 = arith.addi %48, %53 : i64
              %55 = arith.index_cast %54 : i64 to index
              %56 = arith.index_cast %55 : index to i32
              %57 = llvm.getelementptr %33[%56] : (!llvm.ptr, i32) -> !llvm.ptr, f64
              %58 = llvm.load %57 : !llvm.ptr -> f64
              %59 = arith.index_cast %52 : index to i32
              %60 = llvm.getelementptr %37[%59] : (!llvm.ptr, i32) -> !llvm.ptr, f64
              %61 = llvm.load %60 : !llvm.ptr -> f64
              %62 = arith.addf %61, %58 : f64
              %63 = arith.index_cast %52 : index to i32
              %64 = llvm.getelementptr %37[%63] : (!llvm.ptr, i32) -> !llvm.ptr, f64
              llvm.store %62, %64 : f64, !llvm.ptr
              %65 = arith.index_cast %55 : index to i32
              %66 = llvm.getelementptr %33[%65] : (!llvm.ptr, i32) -> !llvm.ptr, f64
              %67 = llvm.load %66 : !llvm.ptr -> f64
              %68 = arith.addf %67, %cst_1 : f64
              %69 = arith.index_cast %55 : index to i32
              %70 = llvm.getelementptr %33[%69] : (!llvm.ptr, i32) -> !llvm.ptr, f64
              llvm.store %68, %70 : f64, !llvm.ptr
              %71 = arith.addi %arg4, %c1_i64 : i64
              scf.yield %71 : i64
            }
            omp.yield
          }
        }
        %46 = arith.addi %arg2, %c1_i32 : i32
        scf.yield %46 : i32
      }
      omp.terminator
    }
    %43 = call @test_results(%22, %38, %18) : (i64, memref<?xf64>, i32) -> f64
    call @prk_free(%36) : (memref<?xi8>) -> ()
    call @prk_free(%32) : (memref<?xi8>) -> ()
    %44 = arith.cmpf olt, %43, %cst_3 : f64
    scf.if %44 {
      %45 = llvm.mlir.addressof @str9 : !llvm.ptr
      %46 = llvm.getelementptr %45[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<20 x i8>
      %47 = llvm.call @printf(%46) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %48 = arith.sitofp %18 : i32 to f64
      %49 = arith.divf %1, %48 : f64
      %50 = llvm.mlir.addressof @str10 : !llvm.ptr
      %51 = llvm.getelementptr %50[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<36 x i8>
      %52 = arith.mulf %42, %cst_0 : f64
      %53 = arith.divf %52, %49 : f64
      %54 = llvm.call @printf(%51, %53, %49) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64) -> i32
      func.call @exit(%c0_i32) : (i32) -> ()
    } else {
      %45 = llvm.mlir.addressof @str11 : !llvm.ptr
      %46 = llvm.getelementptr %45[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<57 x i8>
      %47 = llvm.call @printf(%46, %43, %cst_3) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64) -> i32
      func.call @exit(%c1_i32) : (i32) -> ()
    }
    return %0 : i32
  }
  func.func private @exit(i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @atoi(memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
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
  func.func private @test_results(%arg0: i64, %arg1: memref<?xf64>, %arg2: i32) -> f64 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 2.000000e+00 : f64
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant 0.000000e+00 : f64
    %c0 = arith.constant 0 : index
    %alloca = memref.alloca() : memref<f64>
    %alloca_1 = memref.alloca() : memref<1xf64>
    %cast = memref.cast %alloca_1 : memref<1xf64> to memref<?xf64>
    affine.store %cst_0, %alloca_1[0] : memref<1xf64>
    %0 = arith.addi %arg2, %c1_i32 : i32
    %1 = arith.sitofp %0 : i32 to f64
    %2 = arith.sitofp %arg2 : i32 to f64
    %3 = arith.mulf %1, %2 : f64
    %4 = arith.divf %3, %cst : f64
    %5 = arith.index_cast %arg0 : i64 to index
    affine.store %cst_0, %alloca[] : memref<f64>
    %6 = arith.extsi %arg2 : i32 to i64
    %7 = arith.addi %6, %c1_i64 : i64
    scf.parallel (%arg3) = (%c0) to (%5) step (%c1) {
      %10 = arith.index_cast %arg3 : index to i64
      %11 = arith.muli %arg0, %10 : i64
      %12 = scf.while (%arg4 = %c0_i64) : (i64) -> i64 {
        %13 = arith.cmpi slt, %arg4, %arg0 : i64
        scf.condition(%13) %arg4 : i64
      } do {
      ^bb0(%arg4: i64):
        %13 = arith.addi %arg4, %11 : i64
        %14 = arith.index_cast %13 : i64 to index
        %15 = memref.load %arg1[%14] : memref<?xf64>
        %16 = arith.muli %arg4, %arg0 : i64
        %17 = arith.addi %16, %10 : i64
        %18 = arith.muli %17, %7 : i64
        %19 = arith.sitofp %18 : i64 to f64
        %20 = arith.addf %19, %4 : f64
        %21 = arith.subf %15, %20 : f64
        %22 = arith.cmpf oge, %21, %cst_0 : f64
        %23 = scf.if %22 -> (f64) {
          %26 = memref.load %arg1[%14] : memref<?xf64>
          %27 = arith.subf %26, %20 : f64
          scf.yield %27 : f64
        } else {
          %26 = memref.load %arg1[%14] : memref<?xf64>
          %27 = arith.subf %26, %20 : f64
          %28 = arith.negf %27 : f64
          scf.yield %28 : f64
        }
        %24 = arith.addf %23, %cst_0 : f64
        affine.store %24, %alloca[] : memref<f64>
        %25 = arith.addi %arg4, %c1_i64 : i64
        scf.yield %25 : i64
      }
      scf.yield
    }
    %8 = affine.load %alloca[] : memref<f64>
    omp.atomic.update   %cast : memref<?xf64> {
    ^bb0(%arg3: f64):
      %10 = arith.addf %arg3, %8 : f64
      omp.yield(%10 : f64)
    }
    %9 = affine.load %alloca_1[0] : memref<1xf64>
    return %9 : f64
  }
  func.func private @prk_free(%arg0: memref<?xi8>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    memref.dealloc %arg0 : memref<?xi8>
    return
  }
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
