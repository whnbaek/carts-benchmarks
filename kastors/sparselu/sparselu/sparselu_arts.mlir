module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:o-i64:64-i128:128-n32:64-S128", llvm.target_triple = "arm64-apple-macosx15.0.0", "polygeist.target-cpu" = "apple-m1", "polygeist.target-features" = "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"} {
  func.func @allocate_clean_block(%arg0: i32) -> memref<?xf32> attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c4_i64 = arith.constant 4 : i64
    %c101_i32 = arith.constant 101 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %0 = arith.muli %arg0, %arg0 : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = arith.muli %1, %c4_i64 : i64
    %3 = arith.index_cast %2 : i64 to index
    %4 = arith.divui %3, %c4 : index
    %alloc = memref.alloc(%4) : memref<?xf32>
    %5 = llvm.mlir.zero : !llvm.ptr
    %6 = polygeist.memref2pointer %alloc : memref<?xf32> to !llvm.ptr
    %7 = llvm.icmp "ne" %6, %5 : !llvm.ptr
    scf.if %7 {
      %8 = arith.index_cast %arg0 : i32 to index
      %9 = scf.for %arg1 = %c0 to %8 step %c1 iter_args(%arg2 = %alloc) -> (memref<?xf32>) {
        %10 = scf.for %arg3 = %c0 to %8 step %c1 iter_args(%arg4 = %arg2) -> (memref<?xf32>) {
          memref.store %cst, %arg4[%c0] : memref<?xf32>
          %11 = polygeist.subindex %arg4[%c1] () : memref<?xf32> -> memref<?xf32>
          scf.yield %11 : memref<?xf32>
        }
        scf.yield %10 : memref<?xf32>
      }
    } else {
      func.call @exit(%c101_i32) : (i32) -> ()
    }
    return %alloc : memref<?xf32>
  }
  func.func private @exit(i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @lu0(%arg0: memref<?xf32>, %arg1: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %arg1 : i32 to index
    scf.for %arg2 = %c0 to %0 step %c1 {
      %1 = arith.index_cast %arg2 : index to i32
      %2 = arith.addi %1, %c1_i32 : i32
      %3 = arith.muli %1, %arg1 : i32
      %4 = arith.addi %3, %1 : i32
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %2 : i32 to index
      scf.for %arg3 = %6 to %0 step %c1 {
        %7 = arith.index_cast %arg3 : index to i32
        %8 = arith.muli %7, %arg1 : i32
        %9 = arith.addi %8, %1 : i32
        %10 = arith.index_cast %9 : i32 to index
        %11 = memref.load %arg0[%10] : memref<?xf32>
        %12 = memref.load %arg0[%5] : memref<?xf32>
        %13 = arith.divf %11, %12 : f32
        memref.store %13, %arg0[%10] : memref<?xf32>
        scf.for %arg4 = %6 to %0 step %c1 {
          %14 = arith.index_cast %arg4 : index to i32
          %15 = arith.addi %8, %14 : i32
          %16 = arith.index_cast %15 : i32 to index
          %17 = memref.load %arg0[%16] : memref<?xf32>
          %18 = memref.load %arg0[%10] : memref<?xf32>
          %19 = arith.addi %3, %14 : i32
          %20 = arith.index_cast %19 : i32 to index
          %21 = memref.load %arg0[%20] : memref<?xf32>
          %22 = arith.mulf %18, %21 : f32
          %23 = arith.subf %17, %22 : f32
          memref.store %23, %arg0[%16] : memref<?xf32>
        }
      }
    }
    return
  }
  func.func @bdiv(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %arg2 : i32 to index
    scf.for %arg3 = %c0 to %0 step %c1 {
      %1 = arith.index_cast %arg3 : index to i32
      %2 = arith.muli %1, %arg2 : i32
      scf.for %arg4 = %c0 to %0 step %c1 {
        %3 = arith.index_cast %arg4 : index to i32
        %4 = arith.addi %2, %3 : i32
        %5 = arith.index_cast %4 : i32 to index
        %6 = memref.load %arg1[%5] : memref<?xf32>
        %7 = arith.muli %3, %arg2 : i32
        %8 = arith.addi %7, %3 : i32
        %9 = arith.index_cast %8 : i32 to index
        %10 = memref.load %arg0[%9] : memref<?xf32>
        %11 = arith.divf %6, %10 : f32
        memref.store %11, %arg1[%5] : memref<?xf32>
        %12 = arith.addi %3, %c1_i32 : i32
        %13 = arith.index_cast %12 : i32 to index
        scf.for %arg5 = %13 to %0 step %c1 {
          %14 = arith.index_cast %arg5 : index to i32
          %15 = arith.addi %2, %14 : i32
          %16 = arith.index_cast %15 : i32 to index
          %17 = memref.load %arg1[%16] : memref<?xf32>
          %18 = memref.load %arg1[%5] : memref<?xf32>
          %19 = arith.addi %7, %14 : i32
          %20 = arith.index_cast %19 : i32 to index
          %21 = memref.load %arg0[%20] : memref<?xf32>
          %22 = arith.mulf %18, %21 : f32
          %23 = arith.subf %17, %22 : f32
          memref.store %23, %arg1[%16] : memref<?xf32>
        }
      }
    }
    return
  }
  func.func @bmod(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg3 : i32 to index
    scf.for %arg4 = %c0 to %0 step %c1 {
      %1 = arith.index_cast %arg4 : index to i32
      %2 = arith.muli %1, %arg3 : i32
      scf.for %arg5 = %c0 to %0 step %c1 {
        %3 = arith.index_cast %arg5 : index to i32
        %4 = arith.addi %2, %3 : i32
        %5 = arith.index_cast %4 : i32 to index
        scf.for %arg6 = %c0 to %0 step %c1 {
          %6 = arith.index_cast %arg6 : index to i32
          %7 = memref.load %arg2[%5] : memref<?xf32>
          %8 = arith.addi %2, %6 : i32
          %9 = arith.index_cast %8 : i32 to index
          %10 = memref.load %arg0[%9] : memref<?xf32>
          %11 = arith.muli %6, %arg3 : i32
          %12 = arith.addi %11, %3 : i32
          %13 = arith.index_cast %12 : i32 to index
          %14 = memref.load %arg1[%13] : memref<?xf32>
          %15 = arith.mulf %10, %14 : f32
          %16 = arith.subf %7, %15 : f32
          memref.store %16, %arg2[%5] : memref<?xf32>
        }
      }
    }
    return
  }
  func.func @fwd(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %arg2 : i32 to index
    scf.for %arg3 = %c0 to %0 step %c1 {
      %1 = arith.index_cast %arg3 : index to i32
      scf.for %arg4 = %c0 to %0 step %c1 {
        %2 = arith.index_cast %arg4 : index to i32
        %3 = arith.addi %2, %c1_i32 : i32
        %4 = arith.muli %2, %arg2 : i32
        %5 = arith.addi %4, %1 : i32
        %6 = arith.index_cast %5 : i32 to index
        %7 = arith.index_cast %3 : i32 to index
        scf.for %arg5 = %7 to %0 step %c1 {
          %8 = arith.index_cast %arg5 : index to i32
          %9 = arith.muli %8, %arg2 : i32
          %10 = arith.addi %9, %1 : i32
          %11 = arith.index_cast %10 : i32 to index
          %12 = memref.load %arg1[%11] : memref<?xf32>
          %13 = arith.addi %9, %2 : i32
          %14 = arith.index_cast %13 : i32 to index
          %15 = memref.load %arg0[%14] : memref<?xf32>
          %16 = memref.load %arg1[%6] : memref<?xf32>
          %17 = arith.mulf %15, %16 : f32
          %18 = arith.subf %12, %17 : f32
          memref.store %18, %arg1[%11] : memref<?xf32>
        }
      }
    }
    return
  }
  func.func @run(%arg0: memref<?x!llvm.struct<(i32, i32, memref<?xi8>, i32, i32, i32)>>, %arg1: memref<?xi64>, %arg2: memref<?xi64>) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.000000e+06 : f64
    %c64_i32 = arith.constant 64 : i32
    %alloca = memref.alloca() : memref<1xmemref<?xmemref<?xf32>>>
    %alloca_0 = memref.alloca() : memref<1x!llvm.struct<(i64, i32)>>
    %alloca_1 = memref.alloca() : memref<1x!llvm.struct<(i64, i32)>>
    %alloca_2 = memref.alloca() : memref<memref<?xmemref<?xf32>>>
    %0 = polygeist.memref2pointer %arg0 : memref<?x!llvm.struct<(i32, i32, memref<?xi8>, i32, i32, i32)>> to !llvm.ptr
    %1 = llvm.getelementptr %0[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi8>, i32, i32, i32)>
    %2 = llvm.load %1 : !llvm.ptr -> i32
    %3 = arith.cmpi sle, %2, %c0_i32 : i32
    %4 = arith.select %3, %c64_i32, %2 : i32
    scf.if %3 {
      llvm.store %c64_i32, %1 : i32, !llvm.ptr
    }
    %5 = llvm.getelementptr %0[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi8>, i32, i32, i32)>
    %6 = llvm.load %5 : !llvm.ptr -> i32
    %7 = arith.cmpi sle, %6, %c0_i32 : i32
    %8 = arith.select %7, %c64_i32, %6 : i32
    scf.if %7 {
      llvm.store %c64_i32, %5 : i32, !llvm.ptr
    }
    %cast = memref.cast %alloca_1 : memref<1x!llvm.struct<(i64, i32)>> to memref<?x!llvm.struct<(i64, i32)>>
    %9 = llvm.mlir.zero : !llvm.ptr
    %10 = polygeist.pointer2memref %9 : !llvm.ptr to memref<?xi8>
    %11 = call @gettimeofday(%cast, %10) : (memref<?x!llvm.struct<(i64, i32)>>, memref<?xi8>) -> i32
    %12 = memref.load %alloca_2[] : memref<memref<?xmemref<?xf32>>>
    call @sparselu_par_call(%12, %4, %8) : (memref<?xmemref<?xf32>>, i32, i32) -> ()
    %cast_3 = memref.cast %alloca_0 : memref<1x!llvm.struct<(i64, i32)>> to memref<?x!llvm.struct<(i64, i32)>>
    %13 = call @gettimeofday(%cast_3, %10) : (memref<?x!llvm.struct<(i64, i32)>>, memref<?xi8>) -> i32
    %14 = polygeist.memref2pointer %alloca_0 : memref<1x!llvm.struct<(i64, i32)>> to !llvm.ptr
    %15 = llvm.load %14 : !llvm.ptr -> i64
    %16 = polygeist.memref2pointer %alloca_1 : memref<1x!llvm.struct<(i64, i32)>> to !llvm.ptr
    %17 = llvm.load %16 : !llvm.ptr -> i64
    %18 = arith.subi %15, %17 : i64
    %19 = arith.sitofp %18 : i64 to f64
    %20 = llvm.getelementptr %14[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i32)>
    %21 = llvm.load %20 : !llvm.ptr -> i32
    %22 = llvm.getelementptr %16[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i32)>
    %23 = llvm.load %22 : !llvm.ptr -> i32
    %24 = arith.subi %21, %23 : i32
    %25 = arith.sitofp %24 : i32 to f64
    %26 = arith.divf %25, %cst : f64
    %27 = arith.addf %19, %26 : f64
    %28 = llvm.load %0 : !llvm.ptr -> i32
    %29 = arith.cmpi ne, %28, %c0_i32 : i32
    scf.if %29 {
      %cast_4 = memref.cast %alloca : memref<1xmemref<?xmemref<?xf32>>> to memref<?xmemref<?xmemref<?xf32>>>
      func.call @sparselu_init(%cast_4, %4, %8) : (memref<?xmemref<?xmemref<?xf32>>>, i32, i32) -> ()
      %30 = memref.load %alloca[%c0] : memref<1xmemref<?xmemref<?xf32>>>
      func.call @sparselu_seq_call(%30, %4, %8) : (memref<?xmemref<?xf32>>, i32, i32) -> ()
      %31 = llvm.getelementptr %0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi8>, i32, i32, i32)>
      %32 = memref.load %alloca[%c0] : memref<1xmemref<?xmemref<?xf32>>>
      %33 = func.call @sparselu_check(%32, %12, %4, %8) : (memref<?xmemref<?xf32>>, memref<?xmemref<?xf32>>, i32, i32) -> i32
      llvm.store %33, %31 : i32, !llvm.ptr
    }
    return %27 : f64
  }
  func.func private @gettimeofday(memref<?x!llvm.struct<(i64, i32)>>, memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @sparselu_par_call(memref<?xmemref<?xf32>>, i32, i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @sparselu_init(%arg0: memref<?xmemref<?xmemref<?xf32>>>, %arg1: i32, %arg2: i32) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c0 = arith.constant 0 : index
    %0 = polygeist.typeSize memref<?xf32> : index
    %1 = arith.muli %arg1, %arg1 : i32
    %2 = arith.extsi %1 : i32 to i64
    %3 = arith.index_cast %0 : index to i64
    %4 = arith.muli %2, %3 : i64
    %5 = arith.index_cast %4 : i64 to index
    %6 = arith.divui %5, %0 : index
    %alloc = memref.alloc(%6) : memref<?xmemref<?xf32>>
    memref.store %alloc, %arg0[%c0] : memref<?xmemref<?xmemref<?xf32>>>
    call @genmat(%alloc, %arg1, %arg2) : (memref<?xmemref<?xf32>>, i32, i32) -> ()
    return
  }
  func.func private @sparselu_seq_call(memref<?xmemref<?xf32>>, i32, i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @sparselu_check(%arg0: memref<?xmemref<?xf32>>, %arg1: memref<?xmemref<?xf32>>, %arg2: i32, %arg3: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 9.9999999999999995E-7 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %false = arith.constant false
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = llvm.mlir.zero : !llvm.ptr
    %1:2 = scf.while (%arg4 = %c1_i32, %arg5 = %c0_i32) : (i32, i32) -> (i32, i32) {
      %2 = arith.cmpi slt, %arg5, %arg2 : i32
      %3 = arith.cmpi ne, %arg4, %c0_i32 : i32
      %4 = arith.andi %2, %3 : i1
      scf.condition(%4) %arg4, %arg5 : i32, i32
    } do {
    ^bb0(%arg4: i32, %arg5: i32):
      %2 = arith.muli %arg5, %arg2 : i32
      %3:2 = scf.while (%arg6 = %arg4, %arg7 = %c0_i32) : (i32, i32) -> (i32, i32) {
        %5 = arith.cmpi slt, %arg7, %arg2 : i32
        %6 = arith.cmpi ne, %arg6, %c0_i32 : i32
        %7 = arith.andi %5, %6 : i1
        scf.condition(%7) %arg6, %arg7 : i32, i32
      } do {
      ^bb0(%arg6: i32, %arg7: i32):
        %5 = arith.addi %2, %arg7 : i32
        %6 = arith.index_cast %5 : i32 to index
        %7 = memref.load %arg0[%6] : memref<?xmemref<?xf32>>
        %8 = polygeist.memref2pointer %7 : memref<?xf32> to !llvm.ptr
        %9 = llvm.icmp "eq" %8, %0 : !llvm.ptr
        %10 = scf.if %9 -> (i32) {
          %14 = memref.load %arg1[%6] : memref<?xmemref<?xf32>>
          %15 = polygeist.memref2pointer %14 : memref<?xf32> to !llvm.ptr
          %16 = llvm.icmp "ne" %15, %0 : !llvm.ptr
          %17 = arith.select %16, %c0_i32, %arg6 : i32
          scf.yield %17 : i32
        } else {
          scf.yield %arg6 : i32
        }
        %11 = llvm.icmp "ne" %8, %0 : !llvm.ptr
        %12 = scf.if %11 -> (i32) {
          %14 = memref.load %arg1[%6] : memref<?xmemref<?xf32>>
          %15 = polygeist.memref2pointer %14 : memref<?xf32> to !llvm.ptr
          %16 = llvm.icmp "eq" %15, %0 : !llvm.ptr
          %17 = arith.select %16, %c0_i32, %10 : i32
          %18 = llvm.icmp "ne" %15, %0 : !llvm.ptr
          %19 = scf.if %18 -> (i32) {
            %20 = memref.load %arg0[%6] : memref<?xmemref<?xf32>>
            %21 = memref.load %arg1[%6] : memref<?xmemref<?xf32>>
            %22 = llvm.mlir.undef : f32
            %23 = llvm.mlir.undef : i32
            %24 = arith.cmpi sgt, %arg3, %c0_i32 : i32
            %25:2 = scf.if %24 -> (i1, i32) {
              %27 = arith.index_cast %arg3 : i32 to index
              %28:4 = scf.for %arg8 = %c0 to %27 step %c1 iter_args(%arg9 = %true, %arg10 = %23, %arg11 = %22, %arg12 = %true) -> (i1, i32, f32, i1) {
                %29:4 = scf.if %arg12 -> (i1, i32, f32, i1) {
                  %30 = arith.index_cast %arg8 : index to i32
                  %31:5 = scf.for %arg13 = %c0 to %27 step %c1 iter_args(%arg14 = %arg11, %arg15 = %arg9, %arg16 = %arg10, %arg17 = %true, %arg18 = %true) -> (f32, i1, i32, i1, i1) {
                    %32:5 = scf.if %arg18 -> (f32, i1, i32, i1, i1) {
                      %33 = arith.index_cast %arg13 : index to i32
                      %34 = arith.muli %30, %arg3 : i32
                      %35 = arith.addi %34, %33 : i32
                      %36 = arith.index_cast %35 : i32 to index
                      %37 = memref.load %20[%36] : memref<?xf32>
                      %38 = memref.load %21[%36] : memref<?xf32>
                      %39 = arith.subf %37, %38 : f32
                      %40 = arith.extf %39 : f32 to f64
                      %41 = arith.cmpf oeq, %40, %cst : f64
                      %42:5 = scf.if %41 -> (f32, i1, i32, i1, i1) {
                        scf.yield %arg14, %arg15, %arg16, %arg17, %true : f32, i1, i32, i1, i1
                      } else {
                        %43 = arith.cmpf olt, %40, %cst : f64
                        %44 = scf.if %43 -> (f32) {
                          %48 = arith.negf %39 : f32
                          scf.yield %48 : f32
                        } else {
                          scf.yield %39 : f32
                        }
                        %45 = memref.load %20[%36] : memref<?xf32>
                        %46 = arith.cmpf oeq, %45, %cst_0 : f32
                        %47:5 = scf.if %46 -> (f32, i1, i32, i1, i1) {
                          scf.yield %39, %arg15, %arg16, %arg17, %true : f32, i1, i32, i1, i1
                        } else {
                          %48 = memref.load %20[%36] : memref<?xf32>
                          %49 = arith.divf %44, %48 : f32
                          %50 = arith.extf %49 : f32 to f64
                          %51 = arith.cmpf ogt, %50, %cst_1 : f64
                          %52 = arith.xori %51, %true : i1
                          %53 = arith.andi %52, %arg15 : i1
                          %54 = arith.select %51, %c0_i32, %arg16 : i32
                          %55 = arith.andi %52, %arg17 : i1
                          scf.yield %49, %53, %54, %55, %52 : f32, i1, i32, i1, i1
                        }
                        scf.yield %47#0, %47#1, %47#2, %47#3, %47#4 : f32, i1, i32, i1, i1
                      }
                      scf.yield %42#0, %42#1, %42#2, %42#3, %42#4 : f32, i1, i32, i1, i1
                    } else {
                      scf.yield %arg14, %arg15, %arg16, %arg17, %false : f32, i1, i32, i1, i1
                    }
                    scf.yield %32#0, %32#1, %32#2, %32#3, %32#4 : f32, i1, i32, i1, i1
                  }
                  scf.yield %31#1, %31#2, %31#0, %31#3 : i1, i32, f32, i1
                } else {
                  scf.yield %arg9, %arg10, %arg11, %false : i1, i32, f32, i1
                }
                scf.yield %29#0, %29#1, %29#2, %29#3 : i1, i32, f32, i1
              }
              scf.yield %28#0, %28#1 : i1, i32
            } else {
              scf.yield %true, %23 : i1, i32
            }
            %26 = arith.select %25#0, %c1_i32, %25#1 : i32
            scf.yield %26 : i32
          } else {
            scf.yield %17 : i32
          }
          scf.yield %19 : i32
        } else {
          scf.yield %10 : i32
        }
        %13 = arith.addi %arg7, %c1_i32 : i32
        scf.yield %12, %13 : i32, i32
      }
      %4 = arith.addi %arg5, %c1_i32 : i32
      scf.yield %3#0, %4 : i32, i32
    }
    return %1#0 : i32
  }
  func.func private @genmat(%arg0: memref<?xmemref<?xf32>>, %arg1: i32, %arg2: i32) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c-1_i32 = arith.constant -1 : i32
    %true = arith.constant true
    %c4 = arith.constant 4 : index
    %c4_i64 = arith.constant 4 : i64
    %cst = arith.constant 1.638400e+04 : f64
    %cst_0 = arith.constant 3.276800e+04 : f64
    %c65536_i32 = arith.constant 65536 : i32
    %c3125_i32 = arith.constant 3125 : i32
    %c101_i32 = arith.constant 101 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1325_i32 = arith.constant 1325 : i32
    %alloca = memref.alloca() : memref<i32>
    %0 = llvm.mlir.undef : i32
    memref.store %0, %alloca[] : memref<i32>
    %1 = arith.index_cast %arg1 : i32 to index
    scf.for %arg3 = %c0 to %1 step %c1 {
      %2 = arith.index_cast %arg3 : index to i32
      scf.for %arg4 = %c0 to %1 step %c1 {
        %3 = arith.index_cast %arg4 : index to i32
        memref.store %c1325_i32, %alloca[] : memref<i32>
        arts.edt <task> <intranode> route(%c0_i32) {
          %4 = arith.cmpi slt, %2, %3 : i32
          %5 = arith.remsi %2, %c3_i32 : i32
          %6 = arith.cmpi ne, %5, %c0_i32 : i32
          %7 = arith.andi %4, %6 : i1
          %8 = arith.xori %7, %true : i1
          %9 = arith.cmpi sgt, %2, %3 : i32
          %10 = arith.remsi %3, %c3_i32 : i32
          %11 = arith.cmpi ne, %10, %c0_i32 : i32
          %12 = arith.andi %9, %11 : i1
          %13 = arith.xori %12, %true : i1
          %14 = arith.andi %13, %8 : i1
          %15 = arith.remsi %2, %c2_i32 : i32
          %16 = arith.cmpi ne, %15, %c1_i32 : i32
          %17 = arith.andi %16, %14 : i1
          %18 = arith.remsi %3, %c2_i32 : i32
          %19 = arith.cmpi ne, %18, %c1_i32 : i32
          %20 = arith.andi %19, %17 : i1
          %21 = arith.cmpi eq, %2, %3 : i32
          %22 = arith.cmpi ne, %2, %3 : i32
          %23 = arith.andi %22, %20 : i1
          %24 = arith.ori %21, %23 : i1
          %25 = arith.addi %3, %c-1_i32 : i32
          %26 = arith.cmpi eq, %2, %25 : i32
          %27 = arith.cmpi ne, %2, %25 : i32
          %28 = arith.andi %27, %24 : i1
          %29 = arith.ori %26, %28 : i1
          %30 = arith.addi %2, %c-1_i32 : i32
          %31 = arith.cmpi eq, %30, %3 : i32
          %32 = arith.cmpi ne, %30, %3 : i32
          %33 = arith.andi %32, %29 : i1
          %34 = arith.ori %31, %33 : i1
          scf.if %34 {
            %35 = arith.muli %2, %arg1 : i32
            %36 = arith.addi %35, %3 : i32
            %37 = arith.index_cast %36 : i32 to index
            %38 = arith.muli %arg2, %arg2 : i32
            %39 = arith.extsi %38 : i32 to i64
            %40 = arith.muli %39, %c4_i64 : i64
            %41 = arith.index_cast %40 : i64 to index
            %42 = arith.divui %41, %c4 : index
            %alloc = memref.alloc(%42) : memref<?xf32>
            memref.store %alloc, %arg0[%37] : memref<?xmemref<?xf32>>
            %43 = memref.load %arg0[%37] : memref<?xmemref<?xf32>>
            %44 = llvm.mlir.zero : !llvm.ptr
            %45 = polygeist.memref2pointer %43 : memref<?xf32> to !llvm.ptr
            %46 = llvm.icmp "eq" %45, %44 : !llvm.ptr
            scf.if %46 {
              func.call @exit(%c101_i32) : (i32) -> ()
            }
            %47 = memref.load %arg0[%37] : memref<?xmemref<?xf32>>
            %48 = arith.index_cast %arg2 : i32 to index
            %49 = scf.for %arg5 = %c0 to %48 step %c1 iter_args(%arg6 = %47) -> (memref<?xf32>) {
              %50 = scf.for %arg7 = %c0 to %48 step %c1 iter_args(%arg8 = %arg6) -> (memref<?xf32>) {
                %51 = memref.load %alloca[] : memref<i32>
                %52 = arith.muli %51, %c3125_i32 : i32
                %53 = arith.remsi %52, %c65536_i32 : i32
                memref.store %53, %alloca[] : memref<i32>
                %54 = arith.sitofp %53 : i32 to f64
                %55 = arith.subf %54, %cst_0 : f64
                %56 = arith.divf %55, %cst : f64
                %57 = arith.truncf %56 : f64 to f32
                memref.store %57, %arg8[%c0] : memref<?xf32>
                %58 = polygeist.subindex %arg8[%c1] () : memref<?xf32> -> memref<?xf32>
                scf.yield %58 : memref<?xf32>
              }
              scf.yield %50 : memref<?xf32>
            }
          } else {
            %35 = arith.muli %2, %arg1 : i32
            %36 = arith.addi %35, %3 : i32
            %37 = arith.index_cast %36 : i32 to index
            %38 = llvm.mlir.zero : !llvm.ptr
            %39 = polygeist.pointer2memref %38 : !llvm.ptr to memref<?xf32>
            memref.store %39, %arg0[%37] : memref<?xmemref<?xf32>>
          }
        }
      }
    }
    arts.barrier
    return
  }
}
