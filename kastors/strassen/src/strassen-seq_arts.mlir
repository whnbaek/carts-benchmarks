module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:o-i64:64-i128:128-n32:64-S128", llvm.target_triple = "arm64-apple-macosx15.0.0", "polygeist.target-cpu" = "apple-m1", "polygeist.target-features" = "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"} {
  func.func @strassen_main_seq(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: i32, %arg4: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    call @OptimizedStrassenMultiply_seq(%arg2, %arg0, %arg1, %arg3, %arg3, %arg3, %arg3, %arg4) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32, i32) -> ()
    return
  }
  func.func private @OptimizedStrassenMultiply_seq(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8_i64 = arith.constant 8 : i64
    %c11_i32 = arith.constant 11 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = arith.divui %arg3, %c2_i32 : i32
    %1 = arith.extui %0 : i32 to i64
    %2 = arith.muli %1, %c8_i64 : i64
    %3 = arith.muli %2, %1 : i64
    %4 = arith.trunci %3 : i64 to i32
    %5 = arith.cmpi ule, %arg3, %arg7 : i32
    %6 = arith.cmpi ugt, %arg3, %arg7 : i32
    scf.if %5 {
      func.call @MultiplyByDivideAndConquer(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %c0_i32) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32, i32) -> ()
    }
    scf.if %6 {
      %7 = arith.index_cast %0 : i32 to index
      %8 = polygeist.subindex %arg1[%7] () : memref<?xf64> -> memref<?xf64>
      %9 = polygeist.subindex %arg0[%7] () : memref<?xf64> -> memref<?xf64>
      %10 = arith.muli %arg5, %0 : i32
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.muli %arg6, %0 : i32
      %13 = arith.index_cast %12 : i32 to index
      %14 = polygeist.subindex %arg2[%13] () : memref<?xf64> -> memref<?xf64>
      %15 = arith.muli %arg4, %0 : i32
      %16 = arith.index_cast %15 : i32 to index
      %17 = polygeist.subindex %arg0[%16] () : memref<?xf64> -> memref<?xf64>
      %18 = arith.addi %7, %11 : index
      %19 = polygeist.subindex %arg1[%18] () : memref<?xf64> -> memref<?xf64>
      %20 = arith.addi %7, %13 : index
      %21 = polygeist.subindex %arg2[%20] () : memref<?xf64> -> memref<?xf64>
      %22 = arith.addi %7, %16 : index
      %23 = polygeist.subindex %arg0[%22] () : memref<?xf64> -> memref<?xf64>
      %24 = arith.muli %4, %c11_i32 : i32
      %25 = arith.extui %24 : i32 to i64
      %26 = arith.index_cast %25 : i64 to index
      %alloc = memref.alloc(%26) : memref<?xi8>
      %27 = polygeist.memref2pointer %alloc : memref<?xi8> to !llvm.ptr
      %28 = polygeist.pointer2memref %27 : !llvm.ptr to memref<?xf64>
      %29 = arith.index_cast %4 : i32 to index
      %30 = arith.index_cast %29 : index to i64
      %31 = llvm.getelementptr %27[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %32 = polygeist.pointer2memref %31 : !llvm.ptr to memref<?xf64>
      %33 = llvm.getelementptr %31[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %34 = polygeist.pointer2memref %33 : !llvm.ptr to memref<?xf64>
      %35 = llvm.getelementptr %33[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %36 = polygeist.pointer2memref %35 : !llvm.ptr to memref<?xf64>
      %37 = llvm.getelementptr %35[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %38 = polygeist.pointer2memref %37 : !llvm.ptr to memref<?xf64>
      %39 = llvm.getelementptr %37[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %40 = polygeist.pointer2memref %39 : !llvm.ptr to memref<?xf64>
      %41 = llvm.getelementptr %39[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %42 = polygeist.pointer2memref %41 : !llvm.ptr to memref<?xf64>
      %43 = llvm.getelementptr %41[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %44 = polygeist.pointer2memref %43 : !llvm.ptr to memref<?xf64>
      %45 = llvm.getelementptr %43[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %46 = polygeist.pointer2memref %45 : !llvm.ptr to memref<?xf64>
      %47 = llvm.getelementptr %45[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %48 = polygeist.pointer2memref %47 : !llvm.ptr to memref<?xf64>
      %49 = llvm.getelementptr %47[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %50 = polygeist.pointer2memref %49 : !llvm.ptr to memref<?xf64>
      scf.for %arg8 = %c0 to %7 step %c1 {
        %51 = arith.index_cast %arg8 : index to i32
        %52 = arith.muli %51, %0 : i32
        %53 = arith.muli %arg5, %51 : i32
        %54 = arith.muli %51, %arg5 : i32
        %55 = arith.muli %0, %51 : i32
        %56 = arith.muli %51, %arg6 : i32
        scf.for %arg9 = %c0 to %7 step %c1 {
          %57 = arith.index_cast %arg9 : index to i32
          %58 = arith.addi %52, %57 : i32
          %59 = arith.addi %53, %57 : i32
          %60 = arith.index_cast %59 : i32 to index
          %61 = arith.addi %60, %11 : index
          %62 = memref.load %arg1[%61] : memref<?xf64>
          %63 = arith.addi %60, %18 : index
          %64 = memref.load %arg1[%63] : memref<?xf64>
          %65 = arith.addf %62, %64 : f64
          %66 = llvm.getelementptr %27[%58] : (!llvm.ptr, i32) -> !llvm.ptr, f64
          llvm.store %65, %66 : f64, !llvm.ptr
          %67 = llvm.load %66 : !llvm.ptr -> f64
          %68 = memref.load %arg1[%60] : memref<?xf64>
          %69 = arith.subf %67, %68 : f64
          %70 = llvm.getelementptr %31[%58] : (!llvm.ptr, i32) -> !llvm.ptr, f64
          llvm.store %69, %70 : f64, !llvm.ptr
          %71 = arith.addi %54, %57 : i32
          %72 = arith.index_cast %71 : i32 to index
          %73 = arith.addi %72, %7 : index
          %74 = memref.load %arg1[%73] : memref<?xf64>
          %75 = arith.addi %55, %57 : i32
          %76 = llvm.getelementptr %31[%75] : (!llvm.ptr, i32) -> !llvm.ptr, f64
          %77 = llvm.load %76 : !llvm.ptr -> f64
          %78 = arith.subf %74, %77 : f64
          %79 = llvm.getelementptr %35[%58] : (!llvm.ptr, i32) -> !llvm.ptr, f64
          llvm.store %78, %79 : f64, !llvm.ptr
          %80 = arith.addi %56, %57 : i32
          %81 = arith.index_cast %80 : i32 to index
          %82 = arith.addi %81, %7 : index
          %83 = memref.load %arg2[%82] : memref<?xf64>
          %84 = memref.load %arg2[%81] : memref<?xf64>
          %85 = arith.subf %83, %84 : f64
          %86 = llvm.getelementptr %37[%58] : (!llvm.ptr, i32) -> !llvm.ptr, f64
          llvm.store %85, %86 : f64, !llvm.ptr
          %87 = arith.addi %81, %20 : index
          %88 = memref.load %arg2[%87] : memref<?xf64>
          %89 = llvm.load %86 : !llvm.ptr -> f64
          %90 = arith.subf %88, %89 : f64
          %91 = llvm.getelementptr %39[%58] : (!llvm.ptr, i32) -> !llvm.ptr, f64
          llvm.store %90, %91 : f64, !llvm.ptr
          %92 = llvm.load %91 : !llvm.ptr -> f64
          %93 = arith.addi %81, %13 : index
          %94 = memref.load %arg2[%93] : memref<?xf64>
          %95 = arith.subf %92, %94 : f64
          %96 = llvm.getelementptr %43[%58] : (!llvm.ptr, i32) -> !llvm.ptr, f64
          llvm.store %95, %96 : f64, !llvm.ptr
          %97 = memref.load %arg1[%60] : memref<?xf64>
          %98 = memref.load %arg1[%61] : memref<?xf64>
          %99 = arith.subf %97, %98 : f64
          %100 = llvm.getelementptr %33[%58] : (!llvm.ptr, i32) -> !llvm.ptr, f64
          llvm.store %99, %100 : f64, !llvm.ptr
          %101 = memref.load %arg2[%87] : memref<?xf64>
          %102 = memref.load %arg2[%82] : memref<?xf64>
          %103 = arith.subf %101, %102 : f64
          %104 = llvm.getelementptr %41[%58] : (!llvm.ptr, i32) -> !llvm.ptr, f64
          llvm.store %103, %104 : f64, !llvm.ptr
        }
      }
      func.call @OptimizedStrassenMultiply_seq(%46, %arg1, %arg2, %0, %0, %arg5, %arg6, %arg7) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32, i32) -> ()
      func.call @OptimizedStrassenMultiply_seq(%48, %28, %38, %0, %0, %0, %0, %arg7) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32, i32) -> ()
      func.call @OptimizedStrassenMultiply_seq(%50, %32, %40, %0, %0, %0, %0, %arg7) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32, i32) -> ()
      func.call @OptimizedStrassenMultiply_seq(%23, %34, %42, %0, %arg4, %0, %0, %arg7) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32, i32) -> ()
      func.call @OptimizedStrassenMultiply_seq(%arg0, %8, %14, %0, %arg4, %arg5, %arg6, %arg7) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32, i32) -> ()
      func.call @OptimizedStrassenMultiply_seq(%9, %36, %21, %0, %arg4, %0, %arg6, %arg7) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32, i32) -> ()
      func.call @OptimizedStrassenMultiply_seq(%17, %19, %44, %0, %arg4, %arg5, %0, %arg7) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32, i32) -> ()
      scf.for %arg8 = %c0 to %7 step %c1 {
        %51 = arith.index_cast %arg8 : index to i32
        %52 = arith.muli %arg4, %51 : i32
        %53 = arith.muli %51, %0 : i32
        scf.for %arg9 = %c0 to %7 step %c1 {
          %54 = arith.index_cast %arg9 : index to i32
          %55 = arith.addi %52, %54 : i32
          %56 = arith.index_cast %55 : i32 to index
          %57 = arith.addi %53, %54 : i32
          %58 = llvm.getelementptr %45[%57] : (!llvm.ptr, i32) -> !llvm.ptr, f64
          %59 = llvm.load %58 : !llvm.ptr -> f64
          %60 = memref.load %arg0[%56] : memref<?xf64>
          %61 = arith.addf %60, %59 : f64
          memref.store %61, %arg0[%56] : memref<?xf64>
          %62 = llvm.getelementptr %47[%57] : (!llvm.ptr, i32) -> !llvm.ptr, f64
          %63 = llvm.load %62 : !llvm.ptr -> f64
          %64 = llvm.getelementptr %49[%57] : (!llvm.ptr, i32) -> !llvm.ptr, f64
          %65 = llvm.load %64 : !llvm.ptr -> f64
          %66 = arith.addf %63, %65 : f64
          %67 = llvm.load %58 : !llvm.ptr -> f64
          %68 = arith.addf %66, %67 : f64
          %69 = arith.addi %56, %7 : index
          %70 = memref.load %arg0[%69] : memref<?xf64>
          %71 = arith.addf %70, %68 : f64
          memref.store %71, %arg0[%69] : memref<?xf64>
          %72 = arith.addi %56, %16 : index
          %73 = memref.load %arg0[%72] : memref<?xf64>
          %74 = arith.negf %73 : f64
          %75 = arith.addi %56, %22 : index
          %76 = memref.load %arg0[%75] : memref<?xf64>
          %77 = arith.addf %74, %76 : f64
          %78 = llvm.load %64 : !llvm.ptr -> f64
          %79 = arith.addf %77, %78 : f64
          %80 = llvm.load %58 : !llvm.ptr -> f64
          %81 = arith.addf %79, %80 : f64
          memref.store %81, %arg0[%72] : memref<?xf64>
          %82 = llvm.load %62 : !llvm.ptr -> f64
          %83 = llvm.load %64 : !llvm.ptr -> f64
          %84 = arith.addf %82, %83 : f64
          %85 = llvm.load %58 : !llvm.ptr -> f64
          %86 = arith.addf %84, %85 : f64
          %87 = memref.load %arg0[%75] : memref<?xf64>
          %88 = arith.addf %87, %86 : f64
          memref.store %88, %arg0[%75] : memref<?xf64>
        }
      }
      memref.dealloc %alloc : memref<?xi8>
    }
    return
  }
  func.func private @MultiplyByDivideAndConquer(memref<?xf64>, memref<?xf64>, memref<?xf64>, i32, i32, i32, i32, i32) attributes {llvm.linkage = #llvm.linkage<external>}
}
