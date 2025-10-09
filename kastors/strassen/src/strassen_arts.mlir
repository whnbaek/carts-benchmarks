module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:o-i64:64-i128:128-n32:64-S128", llvm.target_triple = "arm64-apple-macosx15.0.0", "polygeist.target-cpu" = "apple-m1", "polygeist.target-features" = "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"} {
  func.func @MultiplyByDivideAndConquer(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.shrui %arg3, %c1_i32 : i32
    %1 = arith.cmpi ugt, %0, %c16_i32 : i32
    %2 = arith.index_cast %0 : i32 to index
    %3 = polygeist.subindex %arg1[%2] () : memref<?xf64> -> memref<?xf64>
    %4 = arith.muli %arg5, %0 : i32
    %5 = arith.index_cast %4 : i32 to index
    %6 = polygeist.subindex %arg1[%5] () : memref<?xf64> -> memref<?xf64>
    %7 = arith.addi %2, %5 : index
    %8 = polygeist.subindex %arg1[%7] () : memref<?xf64> -> memref<?xf64>
    %9 = polygeist.subindex %arg2[%2] () : memref<?xf64> -> memref<?xf64>
    %10 = arith.muli %arg6, %0 : i32
    %11 = arith.index_cast %10 : i32 to index
    %12 = polygeist.subindex %arg2[%11] () : memref<?xf64> -> memref<?xf64>
    %13 = arith.addi %2, %11 : index
    %14 = polygeist.subindex %arg2[%13] () : memref<?xf64> -> memref<?xf64>
    %15 = polygeist.subindex %arg0[%2] () : memref<?xf64> -> memref<?xf64>
    %16 = arith.muli %arg4, %0 : i32
    %17 = arith.index_cast %16 : i32 to index
    %18 = polygeist.subindex %arg0[%17] () : memref<?xf64> -> memref<?xf64>
    %19 = arith.addi %2, %17 : index
    %20 = polygeist.subindex %arg0[%19] () : memref<?xf64> -> memref<?xf64>
    scf.if %1 {
      func.call @MultiplyByDivideAndConquer(%arg0, %arg1, %arg2, %0, %arg4, %arg5, %arg6, %arg7) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32, i32) -> ()
      func.call @MultiplyByDivideAndConquer(%15, %arg1, %9, %0, %arg4, %arg5, %arg6, %arg7) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32, i32) -> ()
      func.call @MultiplyByDivideAndConquer(%20, %6, %9, %0, %arg4, %arg5, %arg6, %arg7) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32, i32) -> ()
      func.call @MultiplyByDivideAndConquer(%18, %6, %arg2, %0, %arg4, %arg5, %arg6, %arg7) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32, i32) -> ()
      func.call @MultiplyByDivideAndConquer(%arg0, %3, %12, %0, %arg4, %arg5, %arg6, %c1_i32) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32, i32) -> ()
      func.call @MultiplyByDivideAndConquer(%15, %3, %14, %0, %arg4, %arg5, %arg6, %c1_i32) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32, i32) -> ()
      func.call @MultiplyByDivideAndConquer(%20, %8, %14, %0, %arg4, %arg5, %arg6, %c1_i32) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32, i32) -> ()
      func.call @MultiplyByDivideAndConquer(%18, %8, %12, %0, %arg4, %arg5, %arg6, %c1_i32) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32, i32) -> ()
    } else {
      %21 = arith.cmpi ne, %arg7, %c0_i32 : i32
      scf.if %21 {
        func.call @FastAdditiveNaiveMatrixMultiply(%arg0, %arg1, %arg2, %0, %arg4, %arg5, %arg6) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32) -> ()
        func.call @FastAdditiveNaiveMatrixMultiply(%15, %arg1, %9, %0, %arg4, %arg5, %arg6) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32) -> ()
        func.call @FastAdditiveNaiveMatrixMultiply(%20, %6, %9, %0, %arg4, %arg5, %arg6) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32) -> ()
        func.call @FastAdditiveNaiveMatrixMultiply(%18, %6, %arg2, %0, %arg4, %arg5, %arg6) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32) -> ()
      } else {
        func.call @FastNaiveMatrixMultiply(%arg0, %arg1, %arg2, %0, %arg4, %arg5, %arg6) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32) -> ()
        func.call @FastNaiveMatrixMultiply(%15, %arg1, %9, %0, %arg4, %arg5, %arg6) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32) -> ()
        func.call @FastNaiveMatrixMultiply(%20, %6, %9, %0, %arg4, %arg5, %arg6) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32) -> ()
        func.call @FastNaiveMatrixMultiply(%18, %6, %arg2, %0, %arg4, %arg5, %arg6) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32) -> ()
      }
      func.call @FastAdditiveNaiveMatrixMultiply(%arg0, %3, %12, %0, %arg4, %arg5, %arg6) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32) -> ()
      func.call @FastAdditiveNaiveMatrixMultiply(%15, %3, %14, %0, %arg4, %arg5, %arg6) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32) -> ()
      func.call @FastAdditiveNaiveMatrixMultiply(%20, %8, %14, %0, %arg4, %arg5, %arg6) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32) -> ()
      func.call @FastAdditiveNaiveMatrixMultiply(%18, %8, %12, %0, %arg4, %arg5, %arg6) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32) -> ()
    }
    return
  }
  func.func private @FastAdditiveNaiveMatrixMultiply(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c3_i32 = arith.constant 3 : i32
    %c8 = arith.constant 8 : index
    %0 = arith.shli %arg6, %c3_i32 : i32
    %1 = arith.extui %0 : i32 to i64
    %2 = arith.shli %arg5, %c3_i32 : i32
    %3 = arith.extui %2 : i32 to i64
    %4 = arith.shli %arg3, %c3_i32 : i32
    %5 = arith.extui %4 : i32 to i64
    %6 = arith.subi %arg4, %arg3 : i32
    %7 = arith.shli %6, %c3_i32 : i32
    %8 = arith.extui %7 : i32 to i64
    %9 = arith.index_cast %arg3 : i32 to index
    %10:2 = scf.for %arg7 = %c0 to %9 step %c1 iter_args(%arg8 = %arg1, %arg9 = %arg0) -> (memref<?xf64>, memref<?xf64>) {
      %11:2 = scf.for %arg10 = %c0 to %9 step %c8 iter_args(%arg11 = %arg8, %arg12 = %arg9) -> (memref<?xf64>, memref<?xf64>) {
        %22 = polygeist.subindex %arg2[%arg10] () : memref<?xf64> -> memref<?xf64>
        %23 = memref.load %arg12[%c0] : memref<?xf64>
        %24 = memref.load %arg12[%c1] : memref<?xf64>
        %25 = memref.load %arg12[%c2] : memref<?xf64>
        %26 = memref.load %arg12[%c3] : memref<?xf64>
        %27 = memref.load %arg12[%c4] : memref<?xf64>
        %28 = memref.load %arg12[%c5] : memref<?xf64>
        %29 = memref.load %arg12[%c6] : memref<?xf64>
        %30 = memref.load %arg12[%c7] : memref<?xf64>
        %31:10 = scf.for %arg13 = %c0 to %9 step %c1 iter_args(%arg14 = %30, %arg15 = %29, %arg16 = %28, %arg17 = %27, %arg18 = %26, %arg19 = %25, %arg20 = %24, %arg21 = %23, %arg22 = %22, %arg23 = %arg11) -> (f64, f64, f64, f64, f64, f64, f64, f64, memref<?xf64>, memref<?xf64>) {
          %38 = polygeist.subindex %arg23[%c1] () : memref<?xf64> -> memref<?xf64>
          %39 = memref.load %arg23[%c0] : memref<?xf64>
          %40 = memref.load %arg22[%c0] : memref<?xf64>
          %41 = arith.mulf %39, %40 : f64
          %42 = arith.addf %arg21, %41 : f64
          %43 = memref.load %arg22[%c1] : memref<?xf64>
          %44 = arith.mulf %39, %43 : f64
          %45 = arith.addf %arg20, %44 : f64
          %46 = memref.load %arg22[%c2] : memref<?xf64>
          %47 = arith.mulf %39, %46 : f64
          %48 = arith.addf %arg19, %47 : f64
          %49 = memref.load %arg22[%c3] : memref<?xf64>
          %50 = arith.mulf %39, %49 : f64
          %51 = arith.addf %arg18, %50 : f64
          %52 = memref.load %arg22[%c4] : memref<?xf64>
          %53 = arith.mulf %39, %52 : f64
          %54 = arith.addf %arg17, %53 : f64
          %55 = memref.load %arg22[%c5] : memref<?xf64>
          %56 = arith.mulf %39, %55 : f64
          %57 = arith.addf %arg16, %56 : f64
          %58 = memref.load %arg22[%c6] : memref<?xf64>
          %59 = arith.mulf %39, %58 : f64
          %60 = arith.addf %arg15, %59 : f64
          %61 = memref.load %arg22[%c7] : memref<?xf64>
          %62 = arith.mulf %39, %61 : f64
          %63 = arith.addf %arg14, %62 : f64
          %64 = polygeist.memref2pointer %arg22 : memref<?xf64> to !llvm.ptr
          %65 = llvm.ptrtoint %64 : !llvm.ptr to i64
          %66 = arith.addi %65, %1 : i64
          %67 = llvm.inttoptr %66 : i64 to !llvm.ptr
          %68 = polygeist.pointer2memref %67 : !llvm.ptr to memref<?xf64>
          scf.yield %63, %60, %57, %54, %51, %48, %45, %42, %68, %38 : f64, f64, f64, f64, f64, f64, f64, f64, memref<?xf64>, memref<?xf64>
        }
        %32 = polygeist.memref2pointer %31#9 : memref<?xf64> to !llvm.ptr
        %33 = llvm.ptrtoint %32 : !llvm.ptr to i64
        %34 = arith.subi %33, %5 : i64
        %35 = llvm.inttoptr %34 : i64 to !llvm.ptr
        %36 = polygeist.pointer2memref %35 : !llvm.ptr to memref<?xf64>
        memref.store %31#7, %arg12[%c0] : memref<?xf64>
        memref.store %31#6, %arg12[%c1] : memref<?xf64>
        memref.store %31#5, %arg12[%c2] : memref<?xf64>
        memref.store %31#4, %arg12[%c3] : memref<?xf64>
        memref.store %31#3, %arg12[%c4] : memref<?xf64>
        memref.store %31#2, %arg12[%c5] : memref<?xf64>
        memref.store %31#1, %arg12[%c6] : memref<?xf64>
        memref.store %31#0, %arg12[%c7] : memref<?xf64>
        %37 = polygeist.subindex %arg12[%c8] () : memref<?xf64> -> memref<?xf64>
        scf.yield %36, %37 : memref<?xf64>, memref<?xf64>
      }
      %12 = polygeist.memref2pointer %11#0 : memref<?xf64> to !llvm.ptr
      %13 = llvm.ptrtoint %12 : !llvm.ptr to i64
      %14 = arith.addi %13, %3 : i64
      %15 = llvm.inttoptr %14 : i64 to !llvm.ptr
      %16 = polygeist.pointer2memref %15 : !llvm.ptr to memref<?xf64>
      %17 = polygeist.memref2pointer %11#1 : memref<?xf64> to !llvm.ptr
      %18 = llvm.ptrtoint %17 : !llvm.ptr to i64
      %19 = arith.addi %18, %8 : i64
      %20 = llvm.inttoptr %19 : i64 to !llvm.ptr
      %21 = polygeist.pointer2memref %20 : !llvm.ptr to memref<?xf64>
      scf.yield %16, %21 : memref<?xf64>, memref<?xf64>
    }
    return
  }
  func.func private @FastNaiveMatrixMultiply(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3_i32 = arith.constant 3 : i32
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %0 = arith.shli %arg6, %c3_i32 : i32
    %1 = arith.extui %0 : i32 to i64
    %2 = arith.shli %arg5, %c3_i32 : i32
    %3 = arith.extui %2 : i32 to i64
    %4 = arith.shli %arg3, %c3_i32 : i32
    %5 = arith.extui %4 : i32 to i64
    %6 = arith.subi %arg4, %arg3 : i32
    %7 = arith.shli %6, %c3_i32 : i32
    %8 = arith.extui %7 : i32 to i64
    %9 = arith.index_cast %arg3 : i32 to index
    %10:2 = scf.for %arg7 = %c0 to %9 step %c1 iter_args(%arg8 = %arg1, %arg9 = %arg0) -> (memref<?xf64>, memref<?xf64>) {
      %11:2 = scf.for %arg10 = %c0 to %9 step %c8 iter_args(%arg11 = %arg8, %arg12 = %arg9) -> (memref<?xf64>, memref<?xf64>) {
        %22 = polygeist.subindex %arg2[%arg10] () : memref<?xf64> -> memref<?xf64>
        %23 = polygeist.subindex %arg11[%c1] () : memref<?xf64> -> memref<?xf64>
        %24 = memref.load %arg11[%c0] : memref<?xf64>
        %25 = memref.load %arg2[%arg10] : memref<?xf64>
        %26 = arith.mulf %24, %25 : f64
        %27 = arith.addi %arg10, %c1 : index
        %28 = memref.load %arg2[%27] : memref<?xf64>
        %29 = arith.mulf %24, %28 : f64
        %30 = arith.addi %arg10, %c2 : index
        %31 = memref.load %arg2[%30] : memref<?xf64>
        %32 = arith.mulf %24, %31 : f64
        %33 = arith.addi %arg10, %c3 : index
        %34 = memref.load %arg2[%33] : memref<?xf64>
        %35 = arith.mulf %24, %34 : f64
        %36 = arith.addi %arg10, %c4 : index
        %37 = memref.load %arg2[%36] : memref<?xf64>
        %38 = arith.mulf %24, %37 : f64
        %39 = arith.addi %arg10, %c5 : index
        %40 = memref.load %arg2[%39] : memref<?xf64>
        %41 = arith.mulf %24, %40 : f64
        %42 = arith.addi %arg10, %c6 : index
        %43 = memref.load %arg2[%42] : memref<?xf64>
        %44 = arith.mulf %24, %43 : f64
        %45 = arith.addi %arg10, %c7 : index
        %46 = memref.load %arg2[%45] : memref<?xf64>
        %47 = arith.mulf %24, %46 : f64
        %48:10 = scf.for %arg13 = %c1 to %9 step %c1 iter_args(%arg14 = %47, %arg15 = %44, %arg16 = %41, %arg17 = %38, %arg18 = %35, %arg19 = %32, %arg20 = %29, %arg21 = %26, %arg22 = %22, %arg23 = %23) -> (f64, f64, f64, f64, f64, f64, f64, f64, memref<?xf64>, memref<?xf64>) {
          %55 = polygeist.subindex %arg23[%c1] () : memref<?xf64> -> memref<?xf64>
          %56 = memref.load %arg23[%c0] : memref<?xf64>
          %57 = polygeist.memref2pointer %arg22 : memref<?xf64> to !llvm.ptr
          %58 = llvm.ptrtoint %57 : !llvm.ptr to i64
          %59 = arith.addi %58, %1 : i64
          %60 = llvm.inttoptr %59 : i64 to !llvm.ptr
          %61 = polygeist.pointer2memref %60 : !llvm.ptr to memref<?xf64>
          %62 = llvm.load %60 : !llvm.ptr -> f64
          %63 = arith.mulf %56, %62 : f64
          %64 = arith.addf %arg21, %63 : f64
          %65 = llvm.getelementptr %60[1] : (!llvm.ptr) -> !llvm.ptr, f64
          %66 = llvm.load %65 : !llvm.ptr -> f64
          %67 = arith.mulf %56, %66 : f64
          %68 = arith.addf %arg20, %67 : f64
          %69 = llvm.getelementptr %60[2] : (!llvm.ptr) -> !llvm.ptr, f64
          %70 = llvm.load %69 : !llvm.ptr -> f64
          %71 = arith.mulf %56, %70 : f64
          %72 = arith.addf %arg19, %71 : f64
          %73 = llvm.getelementptr %60[3] : (!llvm.ptr) -> !llvm.ptr, f64
          %74 = llvm.load %73 : !llvm.ptr -> f64
          %75 = arith.mulf %56, %74 : f64
          %76 = arith.addf %arg18, %75 : f64
          %77 = llvm.getelementptr %60[4] : (!llvm.ptr) -> !llvm.ptr, f64
          %78 = llvm.load %77 : !llvm.ptr -> f64
          %79 = arith.mulf %56, %78 : f64
          %80 = arith.addf %arg17, %79 : f64
          %81 = llvm.getelementptr %60[5] : (!llvm.ptr) -> !llvm.ptr, f64
          %82 = llvm.load %81 : !llvm.ptr -> f64
          %83 = arith.mulf %56, %82 : f64
          %84 = arith.addf %arg16, %83 : f64
          %85 = llvm.getelementptr %60[6] : (!llvm.ptr) -> !llvm.ptr, f64
          %86 = llvm.load %85 : !llvm.ptr -> f64
          %87 = arith.mulf %56, %86 : f64
          %88 = arith.addf %arg15, %87 : f64
          %89 = llvm.getelementptr %60[7] : (!llvm.ptr) -> !llvm.ptr, f64
          %90 = llvm.load %89 : !llvm.ptr -> f64
          %91 = arith.mulf %56, %90 : f64
          %92 = arith.addf %arg14, %91 : f64
          scf.yield %92, %88, %84, %80, %76, %72, %68, %64, %61, %55 : f64, f64, f64, f64, f64, f64, f64, f64, memref<?xf64>, memref<?xf64>
        }
        %49 = polygeist.memref2pointer %48#9 : memref<?xf64> to !llvm.ptr
        %50 = llvm.ptrtoint %49 : !llvm.ptr to i64
        %51 = arith.subi %50, %5 : i64
        %52 = llvm.inttoptr %51 : i64 to !llvm.ptr
        %53 = polygeist.pointer2memref %52 : !llvm.ptr to memref<?xf64>
        memref.store %48#7, %arg12[%c0] : memref<?xf64>
        memref.store %48#6, %arg12[%c1] : memref<?xf64>
        memref.store %48#5, %arg12[%c2] : memref<?xf64>
        memref.store %48#4, %arg12[%c3] : memref<?xf64>
        memref.store %48#3, %arg12[%c4] : memref<?xf64>
        memref.store %48#2, %arg12[%c5] : memref<?xf64>
        memref.store %48#1, %arg12[%c6] : memref<?xf64>
        memref.store %48#0, %arg12[%c7] : memref<?xf64>
        %54 = polygeist.subindex %arg12[%c8] () : memref<?xf64> -> memref<?xf64>
        scf.yield %53, %54 : memref<?xf64>, memref<?xf64>
      }
      %12 = polygeist.memref2pointer %11#0 : memref<?xf64> to !llvm.ptr
      %13 = llvm.ptrtoint %12 : !llvm.ptr to i64
      %14 = arith.addi %13, %3 : i64
      %15 = llvm.inttoptr %14 : i64 to !llvm.ptr
      %16 = polygeist.pointer2memref %15 : !llvm.ptr to memref<?xf64>
      %17 = polygeist.memref2pointer %11#1 : memref<?xf64> to !llvm.ptr
      %18 = llvm.ptrtoint %17 : !llvm.ptr to i64
      %19 = arith.addi %18, %8 : i64
      %20 = llvm.inttoptr %19 : i64 to !llvm.ptr
      %21 = polygeist.pointer2memref %20 : !llvm.ptr to memref<?xf64>
      scf.yield %16, %21 : memref<?xf64>, memref<?xf64>
    }
    return
  }
  func.func @matrix_multiply(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg3 : i32 to index
    scf.for %arg4 = %c0 to %0 step %c1 {
      %1 = arith.index_cast %arg4 : index to i32
      %2 = arith.muli %1, %arg3 : i32
      scf.for %arg5 = %c0 to %0 step %c1 {
        %3 = arith.index_cast %arg5 : index to i32
        %4 = scf.for %arg6 = %c0 to %0 step %c1 iter_args(%arg7 = %cst) -> (f64) {
          %7 = arith.index_cast %arg6 : index to i32
          %8 = arith.addi %2, %7 : i32
          %9 = arith.index_cast %8 : i32 to index
          %10 = memref.load %arg0[%9] : memref<?xf64>
          %11 = arith.muli %7, %arg3 : i32
          %12 = arith.addi %11, %3 : i32
          %13 = arith.index_cast %12 : i32 to index
          %14 = memref.load %arg1[%13] : memref<?xf64>
          %15 = arith.mulf %10, %14 : f64
          %16 = arith.addf %arg7, %15 : f64
          scf.yield %16 : f64
        }
        %5 = arith.addi %2, %3 : i32
        %6 = arith.index_cast %5 : i32 to index
        memref.store %4, %arg2[%6] : memref<?xf64>
      }
    }
    return
  }
  func.func @run(%arg0: memref<?x!llvm.struct<(i32, i32, memref<?xi8>, i32, i32, i32, i32)>>, %arg1: memref<?xi64>, %arg2: memref<?xi64>) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 9.9999999999999995E-7 : f64
    %c1_i32 = arith.constant 1 : i32
    %false = arith.constant false
    %true = arith.constant true
    %cst_0 = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c8 = arith.constant 8 : index
    %c8_i64 = arith.constant 8 : i64
    %cst_1 = arith.constant 1.000000e+06 : f64
    %c4_i32 = arith.constant 4 : i32
    %c64_i32 = arith.constant 64 : i32
    %c256_i32 = arith.constant 256 : i32
    %c0_i64 = arith.constant 0 : i64
    %alloca = memref.alloca() : memref<1x!llvm.struct<(i64, i32)>>
    %alloca_2 = memref.alloca() : memref<1x!llvm.struct<(i64, i32)>>
    memref.store %c0_i64, %arg1[%c0] : memref<?xi64>
    memref.store %c0_i64, %arg2[%c0] : memref<?xi64>
    %0 = polygeist.memref2pointer %arg0 : memref<?x!llvm.struct<(i32, i32, memref<?xi8>, i32, i32, i32, i32)>> to !llvm.ptr
    %1 = llvm.getelementptr %0[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi8>, i32, i32, i32, i32)>
    %2 = llvm.load %1 : !llvm.ptr -> i32
    %3 = arith.cmpi sle, %2, %c0_i32 : i32
    %4 = llvm.getelementptr %0[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi8>, i32, i32, i32, i32)>
    %5 = llvm.load %4 : !llvm.ptr -> i32
    %6 = arith.cmpi sle, %5, %c0_i32 : i32
    %7 = llvm.getelementptr %0[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi8>, i32, i32, i32, i32)>
    %8 = llvm.load %7 : !llvm.ptr -> i32
    %9 = arith.cmpi sle, %8, %c0_i32 : i32
    %10 = arith.select %3, %c256_i32, %2 : i32
    scf.if %3 {
      llvm.store %c256_i32, %1 : i32, !llvm.ptr
    }
    %11 = arith.select %6, %c64_i32, %5 : i32
    scf.if %6 {
      llvm.store %c64_i32, %4 : i32, !llvm.ptr
    }
    %12 = arith.select %9, %c4_i32, %8 : i32
    scf.if %9 {
      llvm.store %c4_i32, %7 : i32, !llvm.ptr
    }
    %13 = arith.muli %10, %10 : i32
    %14 = arith.extsi %13 : i32 to i64
    %15 = arith.muli %14, %c8_i64 : i64
    %16 = arith.index_cast %15 : i64 to index
    %17 = arith.divui %16, %c8 : index
    %alloc = memref.alloc(%17) : memref<?xf64>
    %alloc_3 = memref.alloc(%17) : memref<?xf64>
    %alloc_4 = memref.alloc(%17) : memref<?xf64>
    %cast = memref.cast %alloca_2 : memref<1x!llvm.struct<(i64, i32)>> to memref<?x!llvm.struct<(i64, i32)>>
    %18 = llvm.mlir.zero : !llvm.ptr
    %19 = polygeist.pointer2memref %18 : !llvm.ptr to memref<?xi8>
    %20 = call @gettimeofday(%cast, %19) : (memref<?x!llvm.struct<(i64, i32)>>, memref<?xi8>) -> i32
    call @strassen_main_par(%alloc_4, %alloc, %alloc_3, %10, %11, %12) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32) -> ()
    %cast_5 = memref.cast %alloca : memref<1x!llvm.struct<(i64, i32)>> to memref<?x!llvm.struct<(i64, i32)>>
    %21 = call @gettimeofday(%cast_5, %19) : (memref<?x!llvm.struct<(i64, i32)>>, memref<?xi8>) -> i32
    %22 = polygeist.memref2pointer %alloca : memref<1x!llvm.struct<(i64, i32)>> to !llvm.ptr
    %23 = llvm.load %22 : !llvm.ptr -> i64
    %24 = polygeist.memref2pointer %alloca_2 : memref<1x!llvm.struct<(i64, i32)>> to !llvm.ptr
    %25 = llvm.load %24 : !llvm.ptr -> i64
    %26 = arith.subi %23, %25 : i64
    %27 = arith.sitofp %26 : i64 to f64
    %28 = llvm.getelementptr %22[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i32)>
    %29 = llvm.load %28 : !llvm.ptr -> i32
    %30 = llvm.getelementptr %24[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i32)>
    %31 = llvm.load %30 : !llvm.ptr -> i32
    %32 = arith.subi %29, %31 : i32
    %33 = arith.sitofp %32 : i32 to f64
    %34 = arith.divf %33, %cst_1 : f64
    %35 = arith.addf %27, %34 : f64
    %36 = llvm.load %0 : !llvm.ptr -> i32
    %37 = arith.cmpi ne, %36, %c0_i32 : i32
    scf.if %37 {
      %alloc_6 = memref.alloc(%17) : memref<?xf64>
      %38 = arith.index_cast %10 : i32 to index
      scf.for %arg3 = %c0 to %38 step %c1 {
        %45 = arith.index_cast %arg3 : index to i32
        %46 = arith.muli %45, %10 : i32
        scf.for %arg4 = %c0 to %38 step %c1 {
          %47 = arith.index_cast %arg4 : index to i32
          %48 = scf.for %arg5 = %c0 to %38 step %c1 iter_args(%arg6 = %cst_0) -> (f64) {
            %51 = arith.index_cast %arg5 : index to i32
            %52 = arith.addi %46, %51 : i32
            %53 = arith.index_cast %52 : i32 to index
            %54 = memref.load %alloc[%53] : memref<?xf64>
            %55 = arith.muli %51, %10 : i32
            %56 = arith.addi %55, %47 : i32
            %57 = arith.index_cast %56 : i32 to index
            %58 = memref.load %alloc_3[%57] : memref<?xf64>
            %59 = arith.mulf %54, %58 : f64
            %60 = arith.addf %arg6, %59 : f64
            scf.yield %60 : f64
          }
          %49 = arith.addi %46, %47 : i32
          %50 = arith.index_cast %49 : i32 to index
          memref.store %48, %alloc_6[%50] : memref<?xf64>
        }
      }
      %39 = llvm.getelementptr %0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi8>, i32, i32, i32, i32)>
      %40 = llvm.mlir.undef : f64
      %41 = llvm.mlir.undef : i32
      %42 = arith.cmpi sgt, %10, %c0_i32 : i32
      %43:2 = scf.if %42 -> (i1, i32) {
        %45 = arith.index_cast %10 : i32 to index
        %46:4 = scf.for %arg3 = %c0 to %45 step %c1 iter_args(%arg4 = %true, %arg5 = %41, %arg6 = %40, %arg7 = %true) -> (i1, i32, f64, i1) {
          %47:4 = scf.if %arg7 -> (i1, i32, f64, i1) {
            %48 = arith.index_cast %arg3 : index to i32
            %49:5 = scf.for %arg8 = %c0 to %45 step %c1 iter_args(%arg9 = %arg6, %arg10 = %arg4, %arg11 = %arg5, %arg12 = %true, %arg13 = %true) -> (f64, i1, i32, i1, i1) {
              %50:5 = scf.if %arg13 -> (f64, i1, i32, i1, i1) {
                %51 = arith.index_cast %arg8 : index to i32
                %52 = arith.muli %48, %10 : i32
                %53 = arith.addi %52, %51 : i32
                %54 = arith.index_cast %53 : i32 to index
                %55 = memref.load %alloc_4[%54] : memref<?xf64>
                %56 = arith.muli %48, %10 : i32
                %57 = arith.addi %56, %51 : i32
                %58 = arith.index_cast %57 : i32 to index
                %59 = memref.load %alloc_6[%58] : memref<?xf64>
                %60 = arith.subf %55, %59 : f64
                %61 = arith.cmpf olt, %60, %cst_0 : f64
                %62 = scf.if %61 -> (f64) {
                  %69 = arith.negf %60 : f64
                  scf.yield %69 : f64
                } else {
                  scf.yield %60 : f64
                }
                %63 = arith.divf %62, %55 : f64
                %64 = arith.cmpf ogt, %63, %cst : f64
                %65 = arith.xori %64, %true : i1
                %66 = arith.andi %65, %arg10 : i1
                %67 = arith.select %64, %c0_i32, %arg5 : i32
                %68 = arith.andi %65, %arg12 : i1
                scf.yield %63, %66, %67, %68, %65 : f64, i1, i32, i1, i1
              } else {
                scf.yield %arg9, %arg10, %arg11, %arg12, %false : f64, i1, i32, i1, i1
              }
              scf.yield %50#0, %50#1, %50#2, %50#3, %50#4 : f64, i1, i32, i1, i1
            }
            scf.yield %49#1, %49#2, %49#0, %49#3 : i1, i32, f64, i1
          } else {
            scf.yield %arg4, %arg5, %arg6, %false : i1, i32, f64, i1
          }
          scf.yield %47#0, %47#1, %47#2, %47#3 : i1, i32, f64, i1
        }
        scf.yield %46#0, %46#1 : i1, i32
      } else {
        scf.yield %true, %41 : i1, i32
      }
      %44 = arith.select %43#0, %c1_i32, %43#1 : i32
      llvm.store %44, %39 : i32, !llvm.ptr
      memref.dealloc %alloc_6 : memref<?xf64>
    }
    memref.dealloc %alloc : memref<?xf64>
    memref.dealloc %alloc_3 : memref<?xf64>
    memref.dealloc %alloc_4 : memref<?xf64>
    return %35 : f64
  }
  func.func private @gettimeofday(memref<?x!llvm.struct<(i64, i32)>>, memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @strassen_main_par(memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32) attributes {llvm.linkage = #llvm.linkage<external>}
}
