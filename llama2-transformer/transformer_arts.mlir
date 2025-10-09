module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:o-i64:64-i128:128-n32:64-S128", llvm.target_triple = "arm64-apple-macosx15.0.0", "polygeist.target-cpu" = "apple-m1", "polygeist.target-features" = "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"} {
  func.func @rmsnorm(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 9.99999974E-6 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = scf.for %arg4 = %c0 to %0 step %c1 iter_args(%arg5 = %cst_1) -> (f32) {
      %7 = memref.load %arg1[%arg4] : memref<?xf32>
      %8 = arith.mulf %7, %7 : f32
      %9 = arith.addf %arg5, %8 : f32
      scf.yield %9 : f32
    }
    %2 = arith.sitofp %arg3 : i32 to f32
    %3 = arith.divf %1, %2 : f32
    %4 = arith.addf %3, %cst_0 : f32
    %5 = math.sqrt %4 : f32
    %6 = arith.divf %cst, %5 : f32
    scf.for %arg4 = %c0 to %0 step %c1 {
      %7 = memref.load %arg2[%arg4] : memref<?xf32>
      %8 = memref.load %arg1[%arg4] : memref<?xf32>
      %9 = arith.mulf %6, %8 : f32
      %10 = arith.mulf %7, %9 : f32
      memref.store %10, %arg0[%arg4] : memref<?xf32>
    }
    return
  }
  func.func @softmax(%arg0: memref<?xf32>, %arg1: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.load %arg0[%c0] : memref<?xf32>
    %1 = arith.index_cast %arg1 : i32 to index
    %2 = scf.for %arg2 = %c1 to %1 step %c1 iter_args(%arg3 = %0) -> (f32) {
      %4 = memref.load %arg0[%arg2] : memref<?xf32>
      %5 = arith.cmpf ogt, %4, %arg3 : f32
      %6 = scf.if %5 -> (f32) {
        %7 = memref.load %arg0[%arg2] : memref<?xf32>
        scf.yield %7 : f32
      } else {
        scf.yield %arg3 : f32
      }
      scf.yield %6 : f32
    }
    %3 = scf.for %arg2 = %c0 to %1 step %c1 iter_args(%arg3 = %cst) -> (f32) {
      %4 = memref.load %arg0[%arg2] : memref<?xf32>
      %5 = arith.subf %4, %2 : f32
      %6 = math.exp %5 : f32
      memref.store %6, %arg0[%arg2] : memref<?xf32>
      %7 = memref.load %arg0[%arg2] : memref<?xf32>
      %8 = arith.addf %arg3, %7 : f32
      scf.yield %8 : f32
    }
    scf.for %arg2 = %c0 to %1 step %c1 {
      %4 = memref.load %arg0[%arg2] : memref<?xf32>
      %5 = arith.divf %4, %3 : f32
      memref.store %5, %arg0[%arg2] : memref<?xf32>
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
    %cst = arith.constant 9.99999974E-6 : f32
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i64 = arith.constant 4 : i64
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 1.000000e+04 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = polygeist.memref2pointer %arg0 : memref<?x!llvm.struct<(struct<(i32, i32, i32, i32, i32, i32, i32)>, !llvm.struct<(memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>, !llvm.struct<(memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>)>> to !llvm.ptr
    %1 = llvm.getelementptr %0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i32, i32, i32, i32, i32, i32, i32)>, !llvm.struct<(memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>, !llvm.struct<(memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>)>
    %2 = llvm.getelementptr %0[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i32, i32, i32, i32, i32, i32, i32)>, !llvm.struct<(memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>, !llvm.struct<(memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>)>)>
    %3 = llvm.load %2 : !llvm.ptr -> memref<?xf32>
    %4 = llvm.load %0 : !llvm.ptr -> i32
    %5 = llvm.getelementptr %0[4] : (!llvm.ptr) -> !llvm.ptr, i32
    %6 = llvm.load %5 : !llvm.ptr -> i32
    %7 = arith.muli %4, %6 : i32
    %8 = llvm.getelementptr %0[3] : (!llvm.ptr) -> !llvm.ptr, i32
    %9 = llvm.load %8 : !llvm.ptr -> i32
    %10 = arith.divsi %7, %9 : i32
    %11 = arith.divsi %9, %6 : i32
    %12 = llvm.getelementptr %0[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %13 = llvm.load %12 : !llvm.ptr -> i32
    %14 = arith.divsi %4, %9 : i32
    %15 = llvm.load %1 : !llvm.ptr -> memref<?xf32>
    %16 = arith.muli %arg1, %4 : i32
    %17 = arith.index_cast %16 : i32 to index
    %18 = polygeist.memref2pointer %3 : memref<?xf32> to !llvm.ptr
    %19 = polygeist.pointer2memref %18 : !llvm.ptr to memref<?xi8>
    %20 = arith.muli %17, %c4 : index
    %21 = arith.index_cast %20 : index to i64
    %22 = polygeist.memref2pointer %15 : memref<?xf32> to !llvm.ptr
    %23 = llvm.getelementptr %22[%21] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %24 = arith.extsi %4 : i32 to i64
    %25 = arith.muli %24, %c4_i64 : i64
    %26 = call @__builtin_object_size(%19, %c0_i32) : (memref<?xi8>, i32) -> i64
    %27 = call @__memcpy_chk(%18, %23, %25, %26) : (!llvm.ptr, !llvm.ptr, i64, i64) -> memref<?xi8>
    %28 = llvm.getelementptr %0[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %29 = llvm.getelementptr %2[1] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %30 = llvm.getelementptr %1[1] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %31 = llvm.getelementptr %0[6] : (!llvm.ptr) -> !llvm.ptr, i32
    %32 = llvm.getelementptr %2[10] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %33 = arith.muli %arg2, %10 : i32
    %34 = arith.index_cast %33 : i32 to index
    %35 = llvm.getelementptr %2[6] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %36 = llvm.getelementptr %2[11] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %37 = llvm.getelementptr %2[7] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %38 = llvm.getelementptr %2[5] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %39 = llvm.getelementptr %1[3] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %40 = llvm.getelementptr %1[4] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %41 = llvm.getelementptr %1[5] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %42 = arith.index_cast %4 : i32 to index
    %43 = llvm.getelementptr %2[8] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %44 = arith.addi %arg2, %c1_i32 : i32
    %45 = arith.extsi %14 : i32 to i64
    %46 = arith.muli %45, %c4_i64 : i64
    %47 = llvm.getelementptr %2[2] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %48 = llvm.getelementptr %1[6] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %49 = llvm.getelementptr %1[2] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %50 = llvm.getelementptr %2[3] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %51 = llvm.getelementptr %1[7] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %52 = llvm.getelementptr %2[4] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %53 = llvm.getelementptr %1[9] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %54 = arith.index_cast %13 : i32 to index
    %55 = llvm.getelementptr %1[8] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %56 = scf.while (%arg3 = %c0_i32) : (i32) -> i32 {
      %75 = llvm.load %28 : !llvm.ptr -> i32
      %76 = arith.cmpi slt, %arg3, %75 : i32
      scf.condition(%76) %arg3 : i32
    } do {
    ^bb0(%arg3: i32):
      %75 = llvm.load %29 : !llvm.ptr -> memref<?xf32>
      %76 = llvm.load %30 : !llvm.ptr -> memref<?xf32>
      %77 = arith.muli %arg3, %4 : i32
      %78 = arith.index_cast %77 : i32 to index
      %79 = arith.index_cast %4 : i32 to index
      %80 = scf.for %arg4 = %c0 to %79 step %c1 iter_args(%arg5 = %cst_0) -> (f32) {
        %142 = memref.load %3[%arg4] : memref<?xf32>
        %143 = arith.mulf %142, %142 : f32
        %144 = arith.addf %arg5, %143 : f32
        scf.yield %144 : f32
      }
      %81 = arith.sitofp %4 : i32 to f32
      %82 = arith.divf %80, %81 : f32
      %83 = arith.addf %82, %cst : f32
      %84 = math.sqrt %83 : f32
      %85 = arith.divf %cst_2, %84 : f32
      scf.for %arg4 = %c0 to %79 step %c1 {
        %142 = arith.addi %arg4, %78 : index
        %143 = memref.load %76[%142] : memref<?xf32>
        %144 = memref.load %3[%arg4] : memref<?xf32>
        %145 = arith.mulf %85, %144 : f32
        %146 = arith.mulf %143, %145 : f32
        memref.store %146, %75[%arg4] : memref<?xf32>
      }
      %86 = llvm.load %31 : !llvm.ptr -> i32
      %87 = arith.muli %arg3, %86 : i32
      %88 = arith.muli %87, %10 : i32
      %89 = llvm.load %32 : !llvm.ptr -> memref<?xf32>
      %90 = arith.index_cast %88 : i32 to index
      %91 = arith.addi %34, %90 : index
      %92 = polygeist.subindex %89[%91] () : memref<?xf32> -> memref<?xf32>
      llvm.store %92, %35 : memref<?xf32>, !llvm.ptr
      %93 = llvm.load %36 : !llvm.ptr -> memref<?xf32>
      %94 = polygeist.subindex %93[%91] () : memref<?xf32> -> memref<?xf32>
      llvm.store %94, %37 : memref<?xf32>, !llvm.ptr
      %95 = llvm.load %38 : !llvm.ptr -> memref<?xf32>
      %96 = llvm.load %29 : !llvm.ptr -> memref<?xf32>
      %97 = llvm.load %39 : !llvm.ptr -> memref<?xf32>
      %98 = arith.muli %77, %4 : i32
      %99 = arith.index_cast %98 : i32 to index
      %100 = arith.index_cast %4 : i32 to index
      scf.for %arg4 = %c0 to %100 step %c1 {
        %142 = arith.index_cast %arg4 : index to i32
        %143 = arith.index_cast %4 : i32 to index
        %144 = scf.for %arg5 = %c0 to %143 step %c1 iter_args(%arg6 = %cst_0) -> (f32) {
          %145 = arith.index_cast %arg5 : index to i32
          %146 = arith.muli %142, %4 : i32
          %147 = arith.addi %146, %145 : i32
          %148 = arith.index_cast %147 : i32 to index
          %149 = arith.addi %148, %99 : index
          %150 = memref.load %97[%149] : memref<?xf32>
          %151 = memref.load %96[%arg5] : memref<?xf32>
          %152 = arith.mulf %150, %151 : f32
          %153 = arith.addf %arg6, %152 : f32
          scf.yield %153 : f32
        }
        memref.store %144, %95[%arg4] : memref<?xf32>
      }
      %101 = llvm.load %35 : !llvm.ptr -> memref<?xf32>
      %102 = llvm.load %29 : !llvm.ptr -> memref<?xf32>
      %103 = llvm.load %40 : !llvm.ptr -> memref<?xf32>
      %104 = arith.muli %77, %10 : i32
      %105 = arith.index_cast %104 : i32 to index
      %106 = arith.index_cast %10 : i32 to index
      scf.for %arg4 = %c0 to %106 step %c1 {
        %142 = arith.index_cast %arg4 : index to i32
        %143 = arith.index_cast %4 : i32 to index
        %144 = scf.for %arg5 = %c0 to %143 step %c1 iter_args(%arg6 = %cst_0) -> (f32) {
          %145 = arith.index_cast %arg5 : index to i32
          %146 = arith.muli %142, %4 : i32
          %147 = arith.addi %146, %145 : i32
          %148 = arith.index_cast %147 : i32 to index
          %149 = arith.addi %148, %105 : index
          %150 = memref.load %103[%149] : memref<?xf32>
          %151 = memref.load %102[%arg5] : memref<?xf32>
          %152 = arith.mulf %150, %151 : f32
          %153 = arith.addf %arg6, %152 : f32
          scf.yield %153 : f32
        }
        memref.store %144, %101[%arg4] : memref<?xf32>
      }
      %107 = llvm.load %37 : !llvm.ptr -> memref<?xf32>
      %108 = llvm.load %29 : !llvm.ptr -> memref<?xf32>
      %109 = llvm.load %41 : !llvm.ptr -> memref<?xf32>
      %110 = arith.index_cast %10 : i32 to index
      scf.for %arg4 = %c0 to %110 step %c1 {
        %142 = arith.index_cast %arg4 : index to i32
        %143 = arith.index_cast %4 : i32 to index
        %144 = scf.for %arg5 = %c0 to %143 step %c1 iter_args(%arg6 = %cst_0) -> (f32) {
          %145 = arith.index_cast %arg5 : index to i32
          %146 = arith.muli %142, %4 : i32
          %147 = arith.addi %146, %145 : i32
          %148 = arith.index_cast %147 : i32 to index
          %149 = arith.addi %148, %105 : index
          %150 = memref.load %109[%149] : memref<?xf32>
          %151 = memref.load %108[%arg5] : memref<?xf32>
          %152 = arith.mulf %150, %151 : f32
          %153 = arith.addf %arg6, %152 : f32
          scf.yield %153 : f32
        }
        memref.store %144, %107[%arg4] : memref<?xf32>
      }
      scf.for %arg4 = %c0 to %42 step %c2 {
        %142 = arith.index_cast %arg4 : index to i32
        %143 = arith.remsi %142, %14 : i32
        %144 = arith.sitofp %143 : i32 to f32
        %145 = arith.sitofp %14 : i32 to f32
        %146 = arith.divf %144, %145 : f32
        %147 = math.powf %cst_1, %146 : f32
        %148 = arith.divf %cst_2, %147 : f32
        %149 = arith.sitofp %arg2 : i32 to f32
        %150 = arith.mulf %149, %148 : f32
        %151 = func.call @cosf(%150) : (f32) -> f32
        %152 = func.call @sinf(%150) : (f32) -> f32
        %153 = arith.cmpi slt, %142, %10 : i32
        %154 = arith.select %153, %c2_i32, %c1_i32 : i32
        %155 = arith.index_cast %154 : i32 to index
        scf.for %arg5 = %c0 to %155 step %c1 {
          %156 = arith.index_cast %arg5 : index to i32
          %157 = arith.cmpi eq, %156, %c0_i32 : i32
          %158 = scf.if %157 -> (memref<?xf32>) {
            %169 = llvm.load %38 : !llvm.ptr -> memref<?xf32>
            scf.yield %169 : memref<?xf32>
          } else {
            %169 = llvm.load %35 : !llvm.ptr -> memref<?xf32>
            scf.yield %169 : memref<?xf32>
          }
          %159 = memref.load %158[%arg4] : memref<?xf32>
          %160 = arith.addi %142, %c1_i32 : i32
          %161 = arith.index_cast %160 : i32 to index
          %162 = memref.load %158[%161] : memref<?xf32>
          %163 = arith.mulf %159, %151 : f32
          %164 = arith.mulf %162, %152 : f32
          %165 = arith.subf %163, %164 : f32
          memref.store %165, %158[%arg4] : memref<?xf32>
          %166 = arith.mulf %159, %152 : f32
          %167 = arith.mulf %162, %151 : f32
          %168 = arith.addf %166, %167 : f32
          memref.store %168, %158[%161] : memref<?xf32>
        }
      }
      %111 = llvm.load %8 : !llvm.ptr -> i32
      %112 = arith.index_cast %111 : i32 to index
      %113 = arith.cmpi sgt, %112, %c0 : index
      scf.if %113 {
        %142 = llvm.load %38 : !llvm.ptr -> memref<?xf32>
        %143 = llvm.load %43 : !llvm.ptr -> memref<?xf32>
        %144 = llvm.load %31 : !llvm.ptr -> i32
        %145 = arith.sitofp %14 : i32 to f32
        %146 = math.sqrt %145 : f32
        arts.edt <parallel> <internode> route(%c0_i32) {
          arts.for(%c0) to(%112) step(%c1) {{
          ^bb0(%arg4: index):
            %147 = arith.index_cast %arg4 : index to i32
            %148 = arith.muli %147, %14 : i32
            %149 = arith.index_cast %148 : i32 to index
            %150 = arith.muli %147, %144 : i32
            %151 = arith.index_cast %150 : i32 to index
            %152 = arith.index_cast %44 : i32 to index
            scf.for %arg5 = %c0 to %152 step %c1 {
              %165 = arith.index_cast %arg5 : index to i32
              %166 = llvm.load %32 : !llvm.ptr -> memref<?xf32>
              %167 = arith.muli %165, %10 : i32
              %168 = arith.index_cast %167 : i32 to index
              %169 = arith.divsi %147, %11 : i32
              %170 = arith.muli %169, %14 : i32
              %171 = arith.index_cast %170 : i32 to index
              %172 = arith.addi %171, %168 : index
              %173 = arith.addi %172, %90 : index
              %174 = arith.index_cast %14 : i32 to index
              %175 = scf.for %arg6 = %c0 to %174 step %c1 iter_args(%arg7 = %cst_0) -> (f32) {
                %178 = arith.addi %arg6, %149 : index
                %179 = memref.load %142[%178] : memref<?xf32>
                %180 = arith.addi %arg6, %173 : index
                %181 = memref.load %166[%180] : memref<?xf32>
                %182 = arith.mulf %179, %181 : f32
                %183 = arith.addf %arg7, %182 : f32
                scf.yield %183 : f32
              }
              %176 = arith.divf %175, %146 : f32
              %177 = arith.addi %arg5, %151 : index
              memref.store %176, %143[%177] : memref<?xf32>
            }
            %153 = memref.load %143[%151] : memref<?xf32>
            %154 = arith.index_cast %44 : i32 to index
            %155 = scf.for %arg5 = %c1 to %154 step %c1 iter_args(%arg6 = %153) -> (f32) {
              %165 = arith.addi %arg5, %151 : index
              %166 = memref.load %143[%165] : memref<?xf32>
              %167 = arith.cmpf ogt, %166, %arg6 : f32
              %168 = scf.if %167 -> (f32) {
                %169 = arith.addi %arg5, %151 : index
                %170 = memref.load %143[%169] : memref<?xf32>
                scf.yield %170 : f32
              } else {
                scf.yield %arg6 : f32
              }
              scf.yield %168 : f32
            }
            %156 = scf.for %arg5 = %c0 to %154 step %c1 iter_args(%arg6 = %cst_0) -> (f32) {
              %165 = arith.addi %arg5, %151 : index
              %166 = memref.load %143[%165] : memref<?xf32>
              %167 = arith.subf %166, %155 : f32
              %168 = math.exp %167 : f32
              %169 = arith.addi %arg5, %151 : index
              memref.store %168, %143[%169] : memref<?xf32>
              %170 = arith.addi %arg5, %151 : index
              %171 = memref.load %143[%170] : memref<?xf32>
              %172 = arith.addf %arg6, %171 : f32
              scf.yield %172 : f32
            }
            scf.for %arg5 = %c0 to %154 step %c1 {
              %165 = arith.addi %arg5, %151 : index
              %166 = memref.load %143[%165] : memref<?xf32>
              %167 = arith.divf %166, %156 : f32
              %168 = arith.addi %arg5, %151 : index
              memref.store %167, %143[%168] : memref<?xf32>
            }
            %157 = llvm.load %29 : !llvm.ptr -> memref<?xf32>
            %158 = arith.muli %149, %c4 : index
            %159 = arith.index_cast %158 : index to i64
            %160 = polygeist.memref2pointer %157 : memref<?xf32> to !llvm.ptr
            %161 = llvm.getelementptr %160[%159] : (!llvm.ptr, i64) -> !llvm.ptr, i8
            %162 = polygeist.pointer2memref %161 : !llvm.ptr to memref<?xi8>
            %163 = func.call @__builtin_object_size(%162, %c0_i32) : (memref<?xi8>, i32) -> i64
            %164 = func.call @__memset_chk(%161, %c0_i32, %46, %163) : (!llvm.ptr, i32, i64, i64) -> memref<?xi8>
            scf.for %arg5 = %c0 to %152 step %c1 {
              %165 = arith.index_cast %arg5 : index to i32
              %166 = llvm.load %36 : !llvm.ptr -> memref<?xf32>
              %167 = arith.muli %165, %10 : i32
              %168 = arith.index_cast %167 : i32 to index
              %169 = arith.divsi %147, %11 : i32
              %170 = arith.muli %169, %14 : i32
              %171 = arith.index_cast %170 : i32 to index
              %172 = arith.addi %171, %168 : index
              %173 = arith.addi %172, %90 : index
              %174 = arith.addi %arg5, %151 : index
              %175 = memref.load %143[%174] : memref<?xf32>
              %176 = arith.index_cast %14 : i32 to index
              scf.for %arg6 = %c0 to %176 step %c1 {
                %177 = arith.addi %arg6, %173 : index
                %178 = memref.load %166[%177] : memref<?xf32>
                %179 = arith.mulf %175, %178 : f32
                %180 = arith.addi %arg6, %149 : index
                %181 = memref.load %157[%180] : memref<?xf32>
                %182 = arith.addf %181, %179 : f32
                memref.store %182, %157[%180] : memref<?xf32>
              }
            }
          }}
        }
      }
      %114 = llvm.load %47 : !llvm.ptr -> memref<?xf32>
      %115 = llvm.load %29 : !llvm.ptr -> memref<?xf32>
      %116 = llvm.load %48 : !llvm.ptr -> memref<?xf32>
      %117 = arith.index_cast %4 : i32 to index
      scf.for %arg4 = %c0 to %117 step %c1 {
        %142 = arith.index_cast %arg4 : index to i32
        %143 = arith.index_cast %4 : i32 to index
        %144 = scf.for %arg5 = %c0 to %143 step %c1 iter_args(%arg6 = %cst_0) -> (f32) {
          %145 = arith.index_cast %arg5 : index to i32
          %146 = arith.muli %142, %4 : i32
          %147 = arith.addi %146, %145 : i32
          %148 = arith.index_cast %147 : i32 to index
          %149 = arith.addi %148, %99 : index
          %150 = memref.load %116[%149] : memref<?xf32>
          %151 = memref.load %115[%arg5] : memref<?xf32>
          %152 = arith.mulf %150, %151 : f32
          %153 = arith.addf %arg6, %152 : f32
          scf.yield %153 : f32
        }
        memref.store %144, %114[%arg4] : memref<?xf32>
      }
      scf.for %arg4 = %c0 to %42 step %c1 {
        %142 = llvm.load %47 : !llvm.ptr -> memref<?xf32>
        %143 = memref.load %142[%arg4] : memref<?xf32>
        %144 = memref.load %3[%arg4] : memref<?xf32>
        %145 = arith.addf %144, %143 : f32
        memref.store %145, %3[%arg4] : memref<?xf32>
      }
      %118 = llvm.load %29 : !llvm.ptr -> memref<?xf32>
      %119 = llvm.load %49 : !llvm.ptr -> memref<?xf32>
      %120 = arith.index_cast %4 : i32 to index
      %121 = scf.for %arg4 = %c0 to %120 step %c1 iter_args(%arg5 = %cst_0) -> (f32) {
        %142 = memref.load %3[%arg4] : memref<?xf32>
        %143 = arith.mulf %142, %142 : f32
        %144 = arith.addf %arg5, %143 : f32
        scf.yield %144 : f32
      }
      %122 = arith.sitofp %4 : i32 to f32
      %123 = arith.divf %121, %122 : f32
      %124 = arith.addf %123, %cst : f32
      %125 = math.sqrt %124 : f32
      %126 = arith.divf %cst_2, %125 : f32
      scf.for %arg4 = %c0 to %120 step %c1 {
        %142 = arith.addi %arg4, %78 : index
        %143 = memref.load %119[%142] : memref<?xf32>
        %144 = memref.load %3[%arg4] : memref<?xf32>
        %145 = arith.mulf %126, %144 : f32
        %146 = arith.mulf %143, %145 : f32
        memref.store %146, %118[%arg4] : memref<?xf32>
      }
      %127 = llvm.load %50 : !llvm.ptr -> memref<?xf32>
      %128 = llvm.load %29 : !llvm.ptr -> memref<?xf32>
      %129 = llvm.load %51 : !llvm.ptr -> memref<?xf32>
      %130 = arith.muli %77, %13 : i32
      %131 = arith.index_cast %130 : i32 to index
      %132 = arith.index_cast %13 : i32 to index
      scf.for %arg4 = %c0 to %132 step %c1 {
        %142 = arith.index_cast %arg4 : index to i32
        %143 = arith.index_cast %4 : i32 to index
        %144 = scf.for %arg5 = %c0 to %143 step %c1 iter_args(%arg6 = %cst_0) -> (f32) {
          %145 = arith.index_cast %arg5 : index to i32
          %146 = arith.muli %142, %4 : i32
          %147 = arith.addi %146, %145 : i32
          %148 = arith.index_cast %147 : i32 to index
          %149 = arith.addi %148, %131 : index
          %150 = memref.load %129[%149] : memref<?xf32>
          %151 = memref.load %128[%arg5] : memref<?xf32>
          %152 = arith.mulf %150, %151 : f32
          %153 = arith.addf %arg6, %152 : f32
          scf.yield %153 : f32
        }
        memref.store %144, %127[%arg4] : memref<?xf32>
      }
      %133 = llvm.load %52 : !llvm.ptr -> memref<?xf32>
      %134 = llvm.load %29 : !llvm.ptr -> memref<?xf32>
      %135 = llvm.load %53 : !llvm.ptr -> memref<?xf32>
      %136 = arith.index_cast %13 : i32 to index
      scf.for %arg4 = %c0 to %136 step %c1 {
        %142 = arith.index_cast %arg4 : index to i32
        %143 = arith.index_cast %4 : i32 to index
        %144 = scf.for %arg5 = %c0 to %143 step %c1 iter_args(%arg6 = %cst_0) -> (f32) {
          %145 = arith.index_cast %arg5 : index to i32
          %146 = arith.muli %142, %4 : i32
          %147 = arith.addi %146, %145 : i32
          %148 = arith.index_cast %147 : i32 to index
          %149 = arith.addi %148, %131 : index
          %150 = memref.load %135[%149] : memref<?xf32>
          %151 = memref.load %134[%arg5] : memref<?xf32>
          %152 = arith.mulf %150, %151 : f32
          %153 = arith.addf %arg6, %152 : f32
          scf.yield %153 : f32
        }
        memref.store %144, %133[%arg4] : memref<?xf32>
      }
      scf.for %arg4 = %c0 to %54 step %c1 {
        %142 = llvm.load %50 : !llvm.ptr -> memref<?xf32>
        %143 = memref.load %142[%arg4] : memref<?xf32>
        %144 = arith.negf %143 : f32
        %145 = math.exp %144 : f32
        %146 = arith.addf %145, %cst_2 : f32
        %147 = arith.divf %cst_2, %146 : f32
        %148 = arith.mulf %143, %147 : f32
        %149 = llvm.load %52 : !llvm.ptr -> memref<?xf32>
        %150 = memref.load %149[%arg4] : memref<?xf32>
        %151 = arith.mulf %148, %150 : f32
        memref.store %151, %142[%arg4] : memref<?xf32>
      }
      %137 = llvm.load %29 : !llvm.ptr -> memref<?xf32>
      %138 = llvm.load %50 : !llvm.ptr -> memref<?xf32>
      %139 = llvm.load %55 : !llvm.ptr -> memref<?xf32>
      %140 = arith.index_cast %4 : i32 to index
      scf.for %arg4 = %c0 to %140 step %c1 {
        %142 = arith.index_cast %arg4 : index to i32
        %143 = arith.index_cast %13 : i32 to index
        %144 = scf.for %arg5 = %c0 to %143 step %c1 iter_args(%arg6 = %cst_0) -> (f32) {
          %145 = arith.index_cast %arg5 : index to i32
          %146 = arith.muli %142, %13 : i32
          %147 = arith.addi %146, %145 : i32
          %148 = arith.index_cast %147 : i32 to index
          %149 = arith.addi %148, %131 : index
          %150 = memref.load %139[%149] : memref<?xf32>
          %151 = memref.load %138[%arg5] : memref<?xf32>
          %152 = arith.mulf %150, %151 : f32
          %153 = arith.addf %arg6, %152 : f32
          scf.yield %153 : f32
        }
        memref.store %144, %137[%arg4] : memref<?xf32>
      }
      scf.for %arg4 = %c0 to %42 step %c1 {
        %142 = llvm.load %29 : !llvm.ptr -> memref<?xf32>
        %143 = memref.load %142[%arg4] : memref<?xf32>
        %144 = memref.load %3[%arg4] : memref<?xf32>
        %145 = arith.addf %144, %143 : f32
        memref.store %145, %3[%arg4] : memref<?xf32>
      }
      %141 = arith.addi %arg3, %c1_i32 : i32
      scf.yield %141 : i32
    }
    %57 = llvm.getelementptr %1[10] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %58 = llvm.load %57 : !llvm.ptr -> memref<?xf32>
    %59 = arith.index_cast %4 : i32 to index
    %60 = scf.for %arg3 = %c0 to %59 step %c1 iter_args(%arg4 = %cst_0) -> (f32) {
      %75 = memref.load %3[%arg3] : memref<?xf32>
      %76 = arith.mulf %75, %75 : f32
      %77 = arith.addf %arg4, %76 : f32
      scf.yield %77 : f32
    }
    %61 = arith.sitofp %4 : i32 to f32
    %62 = arith.divf %60, %61 : f32
    %63 = arith.addf %62, %cst : f32
    %64 = math.sqrt %63 : f32
    %65 = arith.divf %cst_2, %64 : f32
    scf.for %arg3 = %c0 to %59 step %c1 {
      %75 = memref.load %58[%arg3] : memref<?xf32>
      %76 = memref.load %3[%arg3] : memref<?xf32>
      %77 = arith.mulf %65, %76 : f32
      %78 = arith.mulf %75, %77 : f32
      memref.store %78, %3[%arg3] : memref<?xf32>
    }
    %66 = llvm.getelementptr %2[9] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %67 = llvm.load %66 : !llvm.ptr -> memref<?xf32>
    %68 = llvm.getelementptr %1[11] : (!llvm.ptr) -> !llvm.ptr, memref<?xf32>
    %69 = llvm.load %68 : !llvm.ptr -> memref<?xf32>
    %70 = llvm.load %0 : !llvm.ptr -> i32
    %71 = llvm.getelementptr %0[5] : (!llvm.ptr) -> !llvm.ptr, i32
    %72 = llvm.load %71 : !llvm.ptr -> i32
    %73 = arith.index_cast %72 : i32 to index
    scf.for %arg3 = %c0 to %73 step %c1 {
      %75 = arith.index_cast %arg3 : index to i32
      %76 = arith.index_cast %70 : i32 to index
      %77 = scf.for %arg4 = %c0 to %76 step %c1 iter_args(%arg5 = %cst_0) -> (f32) {
        %78 = arith.index_cast %arg4 : index to i32
        %79 = arith.muli %75, %70 : i32
        %80 = arith.addi %79, %78 : i32
        %81 = arith.index_cast %80 : i32 to index
        %82 = memref.load %69[%81] : memref<?xf32>
        %83 = memref.load %3[%arg4] : memref<?xf32>
        %84 = arith.mulf %82, %83 : f32
        %85 = arith.addf %arg5, %84 : f32
        scf.yield %85 : f32
      }
      memref.store %77, %67[%arg3] : memref<?xf32>
    }
    %74 = llvm.load %66 : !llvm.ptr -> memref<?xf32>
    return %74 : memref<?xf32>
  }
  func.func private @__builtin_object_size(memref<?xi8>, i32) -> i64 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @__memcpy_chk(!llvm.ptr, !llvm.ptr, i64, i64) -> memref<?xi8>
  func.func private @cosf(f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @sinf(f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @__memset_chk(!llvm.ptr, i32, i64, i64) -> memref<?xi8>
}
