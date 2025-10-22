module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:o-i64:64-i128:128-n32:64-S128", llvm.target_triple = "arm64-apple-macosx15.0.0", "polygeist.target-cpu" = "apple-m1", "polygeist.target-features" = "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"} {
  func.func @rmsnorm(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 9.99999974E-6 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = scf.for %arg4 = %c0 to %0 step %c1 iter_args(%arg5 = %cst_1) -> (f32) {
      %8 = memref.load %arg1[%arg4] : memref<?xf32>
      %9 = arith.mulf %8, %8 : f32
      %10 = arith.addf %arg5, %9 : f32
      scf.yield %10 : f32
    }
    %2 = arith.sitofp %arg3 : i32 to f32
    %3 = arith.divf %1, %2 : f32
    %4 = arith.addf %3, %cst_0 : f32
    %5 = math.sqrt %4 : f32
    %6 = arith.divf %cst, %5 : f32
    %7 = arith.index_cast %arg3 : i32 to index
    scf.for %arg4 = %c0 to %7 step %c1 {
      %8 = memref.load %arg2[%arg4] : memref<?xf32>
      %9 = memref.load %arg1[%arg4] : memref<?xf32>
      %10 = arith.mulf %6, %9 : f32
      %11 = arith.mulf %8, %10 : f32
      memref.store %11, %arg0[%arg4] : memref<?xf32>
    }
    return
  }
  func.func @softmax(%arg0: memref<?xf32>, %arg1: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = affine.load %arg0[0] : memref<?xf32>
    %1 = arith.index_cast %arg1 : i32 to index
    %2 = scf.for %arg2 = %c1 to %1 step %c1 iter_args(%arg3 = %0) -> (f32) {
      %6 = memref.load %arg0[%arg2] : memref<?xf32>
      %7 = arith.cmpf ogt, %6, %arg3 : f32
      %8 = scf.if %7 -> (f32) {
        %9 = memref.load %arg0[%arg2] : memref<?xf32>
        scf.yield %9 : f32
      } else {
        scf.yield %arg3 : f32
      }
      scf.yield %8 : f32
    }
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = scf.for %arg2 = %c0 to %3 step %c1 iter_args(%arg3 = %cst) -> (f32) {
      %6 = memref.load %arg0[%arg2] : memref<?xf32>
      %7 = arith.subf %6, %2 : f32
      %8 = math.exp %7 : f32
      memref.store %8, %arg0[%arg2] : memref<?xf32>
      %9 = memref.load %arg0[%arg2] : memref<?xf32>
      %10 = arith.addf %arg3, %9 : f32
      scf.yield %10 : f32
    }
    %5 = arith.index_cast %arg1 : i32 to index
    scf.for %arg2 = %c0 to %5 step %c1 {
      %6 = memref.load %arg0[%arg2] : memref<?xf32>
      %7 = arith.divf %6, %4 : f32
      memref.store %7, %arg0[%arg2] : memref<?xf32>
    }
    return
  }
  func.func @matmul(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32, %arg4: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %arg4 : i32 to index
    scf.for %arg5 = %c0 to %0 step %c1 {
      %1 = arith.index_cast %arg5 : index to i32
      %2 = arith.index_cast %arg3 : i32 to index
      %3 = scf.for %arg6 = %c0 to %2 step %c1 iter_args(%arg7 = %cst) -> (f32) {
        %4 = arith.index_cast %arg6 : index to i32
        %5 = arith.muli %1, %arg3 : i32
        %6 = arith.addi %5, %4 : i32
        %7 = arith.index_cast %6 : i32 to index
        %8 = memref.load %arg2[%7] : memref<?xf32>
        %9 = memref.load %arg1[%arg6] : memref<?xf32>
        %10 = arith.mulf %8, %9 : f32
        %11 = arith.addf %arg7, %10 : f32
        scf.yield %11 : f32
      }
      memref.store %3, %arg0[%arg5] : memref<?xf32>
    }
    return
  }
  func.func @forward(%arg0: memref<?x!llvm.struct<(struct<(i32, i32, i32, i32, i32, i32, i32)>, !llvm.struct<(memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>, !llvm.struct<(memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>)>>, %arg1: i32, %arg2: i32) -> memref<?xf32> attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i64 = arith.constant 4 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %c2_i32 = arith.constant 2 : i32
    %cst_0 = arith.constant 1.000000e+04 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = polygeist.memref2pointer %arg0 : memref<?x!llvm.struct<(struct<(i32, i32, i32, i32, i32, i32, i32)>, !llvm.struct<(memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>, !llvm.struct<(memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>)>> to !llvm.ptr
    %1 = polygeist.memref2pointer %arg0 : memref<?x!llvm.struct<(struct<(i32, i32, i32, i32, i32, i32, i32)>, !llvm.struct<(memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>, !llvm.struct<(memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>)>> to !llvm.ptr
    %2 = llvm.getelementptr %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i32, i32, i32, i32, i32, i32, i32)>, !llvm.struct<(memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>, !llvm.struct<(memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>)>
    %3 = polygeist.memref2pointer %arg0 : memref<?x!llvm.struct<(struct<(i32, i32, i32, i32, i32, i32, i32)>, !llvm.struct<(memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>, !llvm.struct<(memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>)>> to !llvm.ptr
    %4 = llvm.getelementptr %3[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i32, i32, i32, i32, i32, i32, i32)>, !llvm.struct<(memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>, !llvm.struct<(memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>)>
    %5 = llvm.load %4 : !llvm.ptr -> memref<?xf32>
    %6 = llvm.load %0 : !llvm.ptr -> i32
    %7 = llvm.load %0 : !llvm.ptr -> i32
    %8 = llvm.getelementptr %0[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %9 = llvm.load %8 : !llvm.ptr -> i32
    %10 = arith.muli %7, %9 : i32
    %11 = llvm.getelementptr %0[3] : (!llvm.ptr) -> !llvm.ptr, i32
    %12 = llvm.load %11 : !llvm.ptr -> i32
    %13 = arith.divsi %10, %12 : i32
    %14 = llvm.getelementptr %0[3] : (!llvm.ptr) -> !llvm.ptr, i32
    %15 = llvm.load %14 : !llvm.ptr -> i32
    %16 = llvm.getelementptr %0[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %17 = llvm.load %16 : !llvm.ptr -> i32
    %18 = arith.divsi %15, %17 : i32
    %19 = llvm.getelementptr %0[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %20 = llvm.load %19 : !llvm.ptr -> i32
    %21 = llvm.getelementptr %0[3] : (!llvm.ptr) -> !llvm.ptr, i32
    %22 = llvm.load %21 : !llvm.ptr -> i32
    %23 = arith.divsi %6, %22 : i32
    %24 = llvm.load %2 : !llvm.ptr -> memref<?xf32>
    %25 = arith.muli %arg1, %6 : i32
    %26 = arith.index_cast %25 : i32 to index
    %27 = polygeist.memref2pointer %5 : memref<?xf32> to !llvm.ptr
    %28 = polygeist.pointer2memref %27 : !llvm.ptr to memref<?xi8>
    %29 = arith.muli %26, %c4 : index
    %30 = arith.index_cast %29 : index to i64
    %31 = polygeist.memref2pointer %24 : memref<?xf32> to !llvm.ptr
    %32 = llvm.getelementptr %31[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %33 = arith.extsi %6 : i32 to i64
    %34 = arith.muli %33, %c4_i64 : i64
    %35 = call @__builtin_object_size(%28, %c0_i32) : (memref<?xi8>, i32) -> i64
    %36 = call @__memcpy_chk(%27, %32, %34, %35) : (!llvm.ptr, !llvm.ptr, i64, i64) -> memref<?xi8>
    %37 = llvm.getelementptr %0[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %38 = llvm.getelementptr %4[1] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %39 = llvm.getelementptr %2[1] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %40 = llvm.getelementptr %0[6] : (!llvm.ptr) -> !llvm.ptr, i32
    %41 = llvm.getelementptr %4[10] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %42 = arith.muli %arg2, %13 : i32
    %43 = arith.index_cast %42 : i32 to index
    %44 = llvm.getelementptr %4[6] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %45 = llvm.getelementptr %4[11] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %46 = llvm.getelementptr %4[7] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %47 = llvm.getelementptr %4[5] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %48 = llvm.getelementptr %4[1] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %49 = llvm.getelementptr %2[3] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %50 = llvm.getelementptr %4[6] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %51 = llvm.getelementptr %4[1] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %52 = llvm.getelementptr %2[4] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %53 = llvm.getelementptr %4[7] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %54 = llvm.getelementptr %4[1] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %55 = llvm.getelementptr %2[5] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %56 = arith.index_cast %6 : i32 to index
    %57 = llvm.getelementptr %0[3] : (!llvm.ptr) -> !llvm.ptr, i32
    %58 = llvm.getelementptr %4[5] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %59 = llvm.getelementptr %4[8] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %60 = llvm.getelementptr %0[6] : (!llvm.ptr) -> !llvm.ptr, i32
    %61 = arith.addi %arg2, %c1_i32 : i32
    %62 = llvm.getelementptr %4[1] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %63 = arith.extsi %23 : i32 to i64
    %64 = arith.muli %63, %c4_i64 : i64
    %65 = llvm.getelementptr %4[2] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %66 = llvm.getelementptr %4[1] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %67 = llvm.getelementptr %2[6] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %68 = arith.index_cast %6 : i32 to index
    %69 = llvm.getelementptr %4[1] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %70 = llvm.getelementptr %2[2] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %71 = llvm.getelementptr %4[3] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %72 = llvm.getelementptr %4[1] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %73 = llvm.getelementptr %2[7] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %74 = llvm.getelementptr %4[4] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %75 = llvm.getelementptr %4[1] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %76 = llvm.getelementptr %2[9] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %77 = arith.index_cast %20 : i32 to index
    %78 = llvm.getelementptr %4[1] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %79 = llvm.getelementptr %4[3] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %80 = llvm.getelementptr %2[8] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %81 = arith.index_cast %6 : i32 to index
    %82 = scf.while (%arg3 = %c0_i32) : (i32) -> i32 {
      %94 = llvm.load %37 : !llvm.ptr -> i32
      %95 = arith.cmpi slt, %arg3, %94 : i32
      scf.condition(%95) %arg3 : i32
    } do {
    ^bb0(%arg3: i32):
      %94 = llvm.load %38 : !llvm.ptr -> memref<?xf32>
      %95 = llvm.load %39 : !llvm.ptr -> memref<?xf32>
      %96 = arith.muli %arg3, %6 : i32
      %97 = arith.index_cast %96 : i32 to index
      %98 = polygeist.subindex %95[%97] () : memref<?xf32> -> memref<?xf32>
      func.call @rmsnorm(%94, %5, %98, %6) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32) -> ()
      %99 = llvm.load %40 : !llvm.ptr -> i32
      %100 = arith.muli %arg3, %99 : i32
      %101 = arith.muli %100, %13 : i32
      %102 = llvm.load %41 : !llvm.ptr -> memref<?xf32>
      %103 = arith.index_cast %101 : i32 to index
      %104 = arith.addi %43, %103 : index
      %105 = polygeist.subindex %102[%104] () : memref<?xf32> -> memref<?xf32>
      llvm.store %105, %44 : memref<?xf32>, !llvm.ptr
      %106 = llvm.load %45 : !llvm.ptr -> memref<?xf32>
      %107 = polygeist.subindex %106[%104] () : memref<?xf32> -> memref<?xf32>
      llvm.store %107, %46 : memref<?xf32>, !llvm.ptr
      %108 = llvm.load %47 : !llvm.ptr -> memref<?xf32>
      %109 = llvm.load %48 : !llvm.ptr -> memref<?xf32>
      %110 = llvm.load %49 : !llvm.ptr -> memref<?xf32>
      %111 = arith.muli %96, %6 : i32
      %112 = arith.index_cast %111 : i32 to index
      %113 = polygeist.subindex %110[%112] () : memref<?xf32> -> memref<?xf32>
      func.call @matmul(%108, %109, %113, %6, %6) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, i32) -> ()
      %114 = llvm.load %50 : !llvm.ptr -> memref<?xf32>
      %115 = llvm.load %51 : !llvm.ptr -> memref<?xf32>
      %116 = llvm.load %52 : !llvm.ptr -> memref<?xf32>
      %117 = arith.muli %96, %13 : i32
      %118 = arith.index_cast %117 : i32 to index
      %119 = polygeist.subindex %116[%118] () : memref<?xf32> -> memref<?xf32>
      func.call @matmul(%114, %115, %119, %6, %13) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, i32) -> ()
      %120 = llvm.load %53 : !llvm.ptr -> memref<?xf32>
      %121 = llvm.load %54 : !llvm.ptr -> memref<?xf32>
      %122 = llvm.load %55 : !llvm.ptr -> memref<?xf32>
      %123 = polygeist.subindex %122[%118] () : memref<?xf32> -> memref<?xf32>
      func.call @matmul(%120, %121, %123, %6, %13) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, i32) -> ()
      scf.for %arg4 = %c0 to %56 step %c2 {
        %161 = arith.index_cast %arg4 : index to i32
        %162 = arith.remsi %161, %23 : i32
        %163 = arith.sitofp %162 : i32 to f32
        %164 = arith.sitofp %23 : i32 to f32
        %165 = arith.divf %163, %164 : f32
        %166 = math.powf %cst_0, %165 : f32
        %167 = arith.divf %cst_1, %166 : f32
        %168 = arith.sitofp %arg2 : i32 to f32
        %169 = arith.mulf %168, %167 : f32
        %170 = func.call @cosf(%169) : (f32) -> f32
        %171 = func.call @sinf(%169) : (f32) -> f32
        %172 = arith.cmpi slt, %161, %13 : i32
        %173 = arith.select %172, %c2_i32, %c1_i32 : i32
        %174 = arith.index_cast %173 : i32 to index
        scf.for %arg5 = %c0 to %174 step %c1 {
          %175 = arith.index_cast %arg5 : index to i32
          %176 = arith.cmpi eq, %175, %c0_i32 : i32
          %177 = scf.if %176 -> (memref<?xf32>) {
            %188 = llvm.getelementptr %4[5] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
            %189 = llvm.load %188 : !llvm.ptr -> memref<?xf32>
            scf.yield %189 : memref<?xf32>
          } else {
            %188 = llvm.getelementptr %4[6] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
            %189 = llvm.load %188 : !llvm.ptr -> memref<?xf32>
            scf.yield %189 : memref<?xf32>
          }
          %178 = memref.load %177[%arg4] : memref<?xf32>
          %179 = arith.addi %161, %c1_i32 : i32
          %180 = arith.index_cast %179 : i32 to index
          %181 = memref.load %177[%180] : memref<?xf32>
          %182 = arith.mulf %178, %170 : f32
          %183 = arith.mulf %181, %171 : f32
          %184 = arith.subf %182, %183 : f32
          memref.store %184, %177[%arg4] : memref<?xf32>
          %185 = arith.mulf %178, %171 : f32
          %186 = arith.mulf %181, %170 : f32
          %187 = arith.addf %185, %186 : f32
          memref.store %187, %177[%180] : memref<?xf32>
        }
      }
      %124 = llvm.load %57 : !llvm.ptr -> i32
      %125 = arith.index_cast %124 : i32 to index
      %126 = arith.cmpi sgt, %125, %c0 : index
      scf.if %126 {
        %161 = llvm.load %58 : !llvm.ptr -> memref<?xf32>
        %162 = llvm.load %59 : !llvm.ptr -> memref<?xf32>
        %163 = llvm.load %60 : !llvm.ptr -> i32
        %164 = llvm.getelementptr %4[10] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
        %165 = arith.index_cast %101 : i32 to index
        %166 = arith.sitofp %23 : i32 to f32
        %167 = math.sqrt %166 : f32
        %168 = llvm.getelementptr %4[11] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
        %169 = arith.index_cast %101 : i32 to index
        scf.parallel (%arg4) = (%c0) to (%125) step (%c1) {
          %170 = arith.index_cast %arg4 : index to i32
          %171 = arith.muli %170, %23 : i32
          %172 = arith.index_cast %171 : i32 to index
          %173 = arith.muli %170, %163 : i32
          %174 = arith.index_cast %173 : i32 to index
          %175 = polygeist.subindex %162[%174] () : memref<?xf32> -> memref<?xf32>
          %176 = arith.addi %arg2, %c1_i32 : i32
          %177 = arith.index_cast %176 : i32 to index
          scf.for %arg5 = %c0 to %177 step %c1 {
            %190 = arith.index_cast %arg5 : index to i32
            %191 = llvm.load %164 : !llvm.ptr -> memref<?xf32>
            %192 = arith.muli %190, %13 : i32
            %193 = arith.index_cast %192 : i32 to index
            %194 = arith.divsi %170, %18 : i32
            %195 = arith.muli %194, %23 : i32
            %196 = arith.index_cast %195 : i32 to index
            %197 = arith.addi %196, %193 : index
            %198 = arith.addi %197, %165 : index
            %199 = arith.index_cast %23 : i32 to index
            %200 = scf.for %arg6 = %c0 to %199 step %c1 iter_args(%arg7 = %cst) -> (f32) {
              %203 = arith.addi %arg6, %172 : index
              %204 = memref.load %161[%203] : memref<?xf32>
              %205 = arith.addi %arg6, %198 : index
              %206 = memref.load %191[%205] : memref<?xf32>
              %207 = arith.mulf %204, %206 : f32
              %208 = arith.addf %arg7, %207 : f32
              scf.yield %208 : f32
            }
            %201 = arith.divf %200, %167 : f32
            %202 = arith.addi %arg5, %174 : index
            memref.store %201, %162[%202] : memref<?xf32>
          }
          func.call @softmax(%175, %61) : (memref<?xf32>, i32) -> ()
          %178 = llvm.load %62 : !llvm.ptr -> memref<?xf32>
          %179 = arith.muli %170, %23 : i32
          %180 = arith.index_cast %179 : i32 to index
          %181 = arith.muli %180, %c4 : index
          %182 = arith.index_cast %181 : index to i64
          %183 = polygeist.memref2pointer %178 : memref<?xf32> to !llvm.ptr
          %184 = llvm.getelementptr %183[%182] : (!llvm.ptr, i64) -> !llvm.ptr, i8
          %185 = polygeist.pointer2memref %184 : !llvm.ptr to memref<?xi8>
          %186 = func.call @__builtin_object_size(%185, %c0_i32) : (memref<?xi8>, i32) -> i64
          %187 = func.call @__memset_chk(%184, %c0_i32, %64, %186) : (!llvm.ptr, i32, i64, i64) -> memref<?xi8>
          %188 = arith.addi %arg2, %c1_i32 : i32
          %189 = arith.index_cast %188 : i32 to index
          scf.for %arg5 = %c0 to %189 step %c1 {
            %190 = arith.index_cast %arg5 : index to i32
            %191 = llvm.load %168 : !llvm.ptr -> memref<?xf32>
            %192 = arith.muli %190, %13 : i32
            %193 = arith.index_cast %192 : i32 to index
            %194 = arith.divsi %170, %18 : i32
            %195 = arith.muli %194, %23 : i32
            %196 = arith.index_cast %195 : i32 to index
            %197 = arith.addi %196, %193 : index
            %198 = arith.addi %197, %169 : index
            %199 = arith.addi %arg5, %174 : index
            %200 = memref.load %162[%199] : memref<?xf32>
            %201 = arith.index_cast %23 : i32 to index
            scf.for %arg6 = %c0 to %201 step %c1 {
              %202 = arith.addi %arg6, %198 : index
              %203 = memref.load %191[%202] : memref<?xf32>
              %204 = arith.mulf %200, %203 : f32
              %205 = arith.addi %arg6, %180 : index
              %206 = memref.load %178[%205] : memref<?xf32>
              %207 = arith.addf %206, %204 : f32
              %208 = arith.addi %arg6, %180 : index
              memref.store %207, %178[%208] : memref<?xf32>
            }
          }
          scf.yield
        }
      }
      %127 = llvm.load %65 : !llvm.ptr -> memref<?xf32>
      %128 = llvm.load %66 : !llvm.ptr -> memref<?xf32>
      %129 = llvm.load %67 : !llvm.ptr -> memref<?xf32>
      %130 = arith.muli %arg3, %6 : i32
      %131 = arith.muli %130, %6 : i32
      %132 = arith.index_cast %131 : i32 to index
      %133 = polygeist.subindex %129[%132] () : memref<?xf32> -> memref<?xf32>
      func.call @matmul(%127, %128, %133, %6, %6) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, i32) -> ()
      scf.for %arg4 = %c0 to %68 step %c1 {
        %161 = llvm.getelementptr %4[2] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
        %162 = llvm.load %161 : !llvm.ptr -> memref<?xf32>
        %163 = memref.load %162[%arg4] : memref<?xf32>
        %164 = memref.load %5[%arg4] : memref<?xf32>
        %165 = arith.addf %164, %163 : f32
        memref.store %165, %5[%arg4] : memref<?xf32>
      }
      %134 = llvm.load %69 : !llvm.ptr -> memref<?xf32>
      %135 = llvm.load %70 : !llvm.ptr -> memref<?xf32>
      %136 = arith.muli %arg3, %6 : i32
      %137 = arith.index_cast %136 : i32 to index
      %138 = polygeist.subindex %135[%137] () : memref<?xf32> -> memref<?xf32>
      func.call @rmsnorm(%134, %5, %138, %6) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32) -> ()
      %139 = llvm.load %71 : !llvm.ptr -> memref<?xf32>
      %140 = llvm.load %72 : !llvm.ptr -> memref<?xf32>
      %141 = llvm.load %73 : !llvm.ptr -> memref<?xf32>
      %142 = arith.muli %arg3, %6 : i32
      %143 = arith.muli %142, %20 : i32
      %144 = arith.index_cast %143 : i32 to index
      %145 = polygeist.subindex %141[%144] () : memref<?xf32> -> memref<?xf32>
      func.call @matmul(%139, %140, %145, %6, %20) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, i32) -> ()
      %146 = llvm.load %74 : !llvm.ptr -> memref<?xf32>
      %147 = llvm.load %75 : !llvm.ptr -> memref<?xf32>
      %148 = llvm.load %76 : !llvm.ptr -> memref<?xf32>
      %149 = arith.muli %arg3, %6 : i32
      %150 = arith.muli %149, %20 : i32
      %151 = arith.index_cast %150 : i32 to index
      %152 = polygeist.subindex %148[%151] () : memref<?xf32> -> memref<?xf32>
      func.call @matmul(%146, %147, %152, %6, %20) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, i32) -> ()
      scf.for %arg4 = %c0 to %77 step %c1 {
        %161 = llvm.getelementptr %4[3] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
        %162 = llvm.load %161 : !llvm.ptr -> memref<?xf32>
        %163 = memref.load %162[%arg4] : memref<?xf32>
        %164 = arith.negf %163 : f32
        %165 = math.exp %164 : f32
        %166 = arith.addf %165, %cst_1 : f32
        %167 = arith.divf %cst_1, %166 : f32
        %168 = arith.mulf %163, %167 : f32
        %169 = llvm.getelementptr %4[4] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
        %170 = llvm.load %169 : !llvm.ptr -> memref<?xf32>
        %171 = memref.load %170[%arg4] : memref<?xf32>
        %172 = arith.mulf %168, %171 : f32
        memref.store %172, %162[%arg4] : memref<?xf32>
      }
      %153 = llvm.load %78 : !llvm.ptr -> memref<?xf32>
      %154 = llvm.load %79 : !llvm.ptr -> memref<?xf32>
      %155 = llvm.load %80 : !llvm.ptr -> memref<?xf32>
      %156 = arith.muli %arg3, %6 : i32
      %157 = arith.muli %156, %20 : i32
      %158 = arith.index_cast %157 : i32 to index
      %159 = polygeist.subindex %155[%158] () : memref<?xf32> -> memref<?xf32>
      func.call @matmul(%153, %154, %159, %20, %6) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, i32) -> ()
      scf.for %arg4 = %c0 to %81 step %c1 {
        %161 = llvm.getelementptr %4[1] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
        %162 = llvm.load %161 : !llvm.ptr -> memref<?xf32>
        %163 = memref.load %162[%arg4] : memref<?xf32>
        %164 = memref.load %5[%arg4] : memref<?xf32>
        %165 = arith.addf %164, %163 : f32
        memref.store %165, %5[%arg4] : memref<?xf32>
      }
      %160 = arith.addi %arg3, %c1_i32 : i32
      scf.yield %160 : i32
    }
    %83 = llvm.getelementptr %2[10] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %84 = llvm.load %83 : !llvm.ptr -> memref<?xf32>
    call @rmsnorm(%5, %5, %84, %6) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32) -> ()
    %85 = llvm.getelementptr %4[9] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %86 = llvm.load %85 : !llvm.ptr -> memref<?xf32>
    %87 = llvm.getelementptr %2[11] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %88 = llvm.load %87 : !llvm.ptr -> memref<?xf32>
    %89 = llvm.load %0 : !llvm.ptr -> i32
    %90 = llvm.getelementptr %0[5] : (!llvm.ptr) -> !llvm.ptr, i32
    %91 = llvm.load %90 : !llvm.ptr -> i32
    call @matmul(%86, %5, %88, %89, %91) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, i32) -> ()
    %92 = llvm.getelementptr %4[9] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %93 = llvm.load %92 : !llvm.ptr -> memref<?xf32>
    return %93 : memref<?xf32>
  }
  func.func private @__builtin_object_size(memref<?xi8>, i32) -> i64 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @__memcpy_chk(!llvm.ptr, !llvm.ptr, i64, i64) -> memref<?xi8>
  func.func private @cosf(f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @sinf(f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @__memset_chk(!llvm.ptr, i32, i64, i64) -> memref<?xi8>
}
