module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:o-i64:64-i128:128-n32:64-S128", llvm.target_triple = "arm64-apple-macosx16.0.0", "polygeist.target-cpu" = "apple-m1", "polygeist.target-features" = "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"} {
  llvm.mlir.global internal constant @str3("Implementation error: a[%d]=%d > a[%d]=%d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2("%.4f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("Error: Could not allocate array of size %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str0("Usage: %s array-size\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c-1_i32 = arith.constant -1 : i32
    %false = arith.constant false
    %true = arith.constant true
    %c4 = arith.constant 4 : index
    %c4_i64 = arith.constant 4 : i64
    %c0_i32 = arith.constant 0 : i32
    %c314159_i32 = arith.constant 314159 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = llvm.mlir.undef : i32
    %1 = arith.cmpi ne, %arg0, %c2_i32 : i32
    %2 = arith.cmpi eq, %arg0, %c2_i32 : i32
    %3 = arith.select %1, %c1_i32, %0 : i32
    scf.if %1 {
      %6 = llvm.mlir.addressof @str0 : !llvm.ptr
      %7 = llvm.getelementptr %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<22 x i8>
      %8 = memref.load %arg1[%c0] : memref<?xmemref<?xi8>>
      %9 = polygeist.memref2pointer %8 : memref<?xi8> to !llvm.ptr
      %10 = llvm.call @printf(%7, %9) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    }
    %4:2 = scf.if %2 -> (i1, i32) {
      %6 = memref.load %arg1[%c1] : memref<?xmemref<?xi8>>
      %7 = func.call @atoi(%6) : (memref<?xi8>) -> i32
      %8 = arith.extsi %7 : i32 to i64
      %9 = arith.muli %8, %c4_i64 : i64
      %10 = arith.index_cast %9 : i64 to index
      %11 = arith.divui %10, %c4 : index
      %alloc = memref.alloc(%11) : memref<?xi32>
      %alloc_0 = memref.alloc(%11) : memref<?xi32>
      %12 = llvm.mlir.zero : !llvm.ptr
      %13 = polygeist.memref2pointer %alloc : memref<?xi32> to !llvm.ptr
      %14 = llvm.icmp "eq" %13, %12 : !llvm.ptr
      %15 = scf.if %14 -> (i1) {
        scf.yield %true : i1
      } else {
        %17 = polygeist.memref2pointer %alloc_0 : memref<?xi32> to !llvm.ptr
        %18 = llvm.icmp "eq" %17, %12 : !llvm.ptr
        scf.yield %18 : i1
      }
      %16:2 = scf.if %15 -> (i1, i32) {
        %17 = llvm.mlir.addressof @str1 : !llvm.ptr
        %18 = llvm.getelementptr %17[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<44 x i8>
        %19 = llvm.call @printf(%18, %7) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
        scf.yield %false, %3 : i1, i32
      } else {
        func.call @srand(%c314159_i32) : (i32) -> ()
        %17 = arith.index_cast %7 : i32 to index
        scf.for %arg2 = %c0 to %17 step %c1 {
          %26 = func.call @rand() : () -> i32
          %27 = arith.remsi %26, %7 : i32
          memref.store %27, %alloc[%arg2] : memref<?xi32>
        }
        %18 = func.call @get_time() : () -> f64
        func.call @mergesort_serial(%alloc, %7, %alloc_0) : (memref<?xi32>, i32, memref<?xi32>) -> ()
        %19 = func.call @get_time() : () -> f64
        %20 = llvm.mlir.addressof @str2 : !llvm.ptr
        %21 = llvm.getelementptr %20[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<6 x i8>
        %22 = arith.subf %19, %18 : f64
        %23 = llvm.call @printf(%21, %22) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
        %24 = arith.cmpi sgt, %7, %c1_i32 : i32
        %25:2 = scf.if %24 -> (i1, i32) {
          %26:3 = scf.for %arg2 = %c1 to %17 step %c1 iter_args(%arg3 = %true, %arg4 = %3, %arg5 = %true) -> (i1, i32, i1) {
            %27:3 = scf.if %arg5 -> (i1, i32, i1) {
              %28 = arith.index_cast %arg2 : index to i32
              %29 = arith.addi %28, %c-1_i32 : i32
              %30 = arith.index_cast %29 : i32 to index
              %31 = memref.load %alloc[%30] : memref<?xi32>
              %32 = memref.load %alloc[%arg2] : memref<?xi32>
              %33 = arith.cmpi sgt, %31, %32 : i32
              %34 = arith.cmpi sle, %31, %32 : i32
              %35 = arith.andi %34, %arg3 : i1
              %36 = arith.select %33, %c1_i32, %arg4 : i32
              scf.if %33 {
                %37 = llvm.mlir.addressof @str3 : !llvm.ptr
                %38 = llvm.getelementptr %37[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<43 x i8>
                %39 = llvm.call @printf(%38, %29, %31, %28, %32) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32, i32, i32) -> i32
              }
              scf.yield %35, %36, %34 : i1, i32, i1
            } else {
              scf.yield %arg3, %arg4, %false : i1, i32, i1
            }
            scf.yield %27#0, %27#1, %27#2 : i1, i32, i1
          }
          scf.yield %26#0, %26#1 : i1, i32
        } else {
          scf.yield %true, %3 : i1, i32
        }
        scf.yield %25#0, %25#1 : i1, i32
      }
      scf.yield %16#0, %16#1 : i1, i32
    } else {
      scf.yield %false, %3 : i1, i32
    }
    %5 = arith.select %4#0, %c0_i32, %4#1 : i32
    return %5 : i32
  }
  func.func private @atoi(memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @srand(i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @rand() -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @get_time() -> f64 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @mergesort_serial(%arg0: memref<?xi32>, %arg1: i32, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %c-1_i32 = arith.constant -1 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %false = arith.constant false
    %c2_i32 = arith.constant 2 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = arith.cmpi sle, %arg1, %c64_i32 : i32
    %1 = arith.cmpi sgt, %arg1, %c64_i32 : i32
    scf.if %0 {
      %2 = arith.index_cast %arg1 : i32 to index
      scf.for %arg3 = %c0 to %2 step %c1 {
        %3 = arith.index_cast %arg3 : index to i32
        %4 = memref.load %arg0[%arg3] : memref<?xi32>
        %5 = arith.addi %3, %c-1_i32 : i32
        %6 = arith.cmpi sge, %5, %c0_i32 : i32
        %7 = scf.if %6 -> (i32) {
          %10:2 = scf.for %arg4 = %c0 to %arg3 step %c1 iter_args(%arg5 = %5, %arg6 = %true) -> (i32, i1) {
            %11:2 = scf.if %arg6 -> (i32, i1) {
              %12 = arith.index_cast %arg5 : i32 to index
              %13 = memref.load %arg0[%12] : memref<?xi32>
              %14 = arith.cmpi sgt, %13, %4 : i32
              %15 = scf.if %14 -> (i32) {
                %16 = arith.addi %arg5, %c1_i32 : i32
                %17 = arith.index_cast %16 : i32 to index
                %18 = memref.load %arg0[%12] : memref<?xi32>
                memref.store %18, %arg0[%17] : memref<?xi32>
                %19 = arith.addi %arg5, %c-1_i32 : i32
                scf.yield %19 : i32
              } else {
                scf.yield %arg5 : i32
              }
              scf.yield %15, %14 : i32, i1
            } else {
              scf.yield %arg5, %false : i32, i1
            }
            scf.yield %11#0, %11#1 : i32, i1
          }
          scf.yield %10#0 : i32
        } else {
          scf.yield %5 : i32
        }
        %8 = arith.addi %7, %c1_i32 : i32
        %9 = arith.index_cast %8 : i32 to index
        memref.store %4, %arg0[%9] : memref<?xi32>
      }
    }
    scf.if %1 {
      %2 = arith.divsi %arg1, %c2_i32 : i32
      func.call @mergesort_serial(%arg0, %2, %arg2) : (memref<?xi32>, i32, memref<?xi32>) -> ()
      %3 = arith.index_cast %2 : i32 to index
      %4 = polygeist.subindex %arg0[%3] () : memref<?xi32> -> memref<?xi32>
      %5 = arith.subi %arg1, %2 : i32
      func.call @mergesort_serial(%4, %5, %arg2) : (memref<?xi32>, i32, memref<?xi32>) -> ()
      func.call @merge(%arg0, %arg1, %arg2) : (memref<?xi32>, i32, memref<?xi32>) -> ()
    }
    return
  }
  func.func @insertion_sort(%arg0: memref<?xi32>, %arg1: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %false = arith.constant false
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c-1_i32 = arith.constant -1 : i32
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.index_cast %arg1 : i32 to index
    scf.for %arg2 = %c0 to %0 step %c1 {
      %1 = arith.index_cast %arg2 : index to i32
      %2 = memref.load %arg0[%arg2] : memref<?xi32>
      %3 = arith.addi %1, %c-1_i32 : i32
      %4 = arith.cmpi sge, %3, %c0_i32 : i32
      %5 = scf.if %4 -> (i32) {
        %8:2 = scf.for %arg3 = %c0 to %arg2 step %c1 iter_args(%arg4 = %3, %arg5 = %true) -> (i32, i1) {
          %9:2 = scf.if %arg5 -> (i32, i1) {
            %10 = arith.index_cast %arg4 : i32 to index
            %11 = memref.load %arg0[%10] : memref<?xi32>
            %12 = arith.cmpi sgt, %11, %2 : i32
            %13 = scf.if %12 -> (i32) {
              %14 = arith.addi %arg4, %c1_i32 : i32
              %15 = arith.index_cast %14 : i32 to index
              %16 = memref.load %arg0[%10] : memref<?xi32>
              memref.store %16, %arg0[%15] : memref<?xi32>
              %17 = arith.addi %arg4, %c-1_i32 : i32
              scf.yield %17 : i32
            } else {
              scf.yield %arg4 : i32
            }
            scf.yield %13, %12 : i32, i1
          } else {
            scf.yield %arg4, %false : i32, i1
          }
          scf.yield %9#0, %9#1 : i32, i1
        }
        scf.yield %8#0 : i32
      } else {
        scf.yield %3 : i32
      }
      %6 = arith.addi %5, %c1_i32 : i32
      %7 = arith.index_cast %6 : i32 to index
      memref.store %2, %arg0[%7] : memref<?xi32>
    }
    return
  }
  func.func @merge(%arg0: memref<?xi32>, %arg1: i32, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c4_i64 = arith.constant 4 : i64
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.divsi %arg1, %c2_i32 : i32
    %1:5 = scf.while (%arg3 = %c0_i32, %arg4 = %0, %arg5 = %c0_i32) : (i32, i32, i32) -> (i32, i32, i32, i32, i32) {
      %12 = arith.cmpi slt, %arg5, %0 : i32
      %13 = arith.cmpi slt, %arg4, %arg1 : i32
      %14 = arith.andi %12, %13 : i1
      %15:5 = scf.if %14 -> (i32, i32, i32, i32, i32) {
        %16 = arith.index_cast %arg5 : i32 to index
        %17 = memref.load %arg0[%16] : memref<?xi32>
        %18 = arith.index_cast %arg4 : i32 to index
        %19 = memref.load %arg0[%18] : memref<?xi32>
        %20 = arith.cmpi slt, %17, %19 : i32
        %21:2 = scf.if %20 -> (i32, i32) {
          %24 = arith.index_cast %arg3 : i32 to index
          %25 = memref.load %arg0[%16] : memref<?xi32>
          memref.store %25, %arg2[%24] : memref<?xi32>
          %26 = arith.addi %arg5, %c1_i32 : i32
          scf.yield %arg4, %26 : i32, i32
        } else {
          %24 = arith.index_cast %arg3 : i32 to index
          %25 = memref.load %arg0[%18] : memref<?xi32>
          memref.store %25, %arg2[%24] : memref<?xi32>
          %26 = arith.addi %arg4, %c1_i32 : i32
          scf.yield %26, %arg5 : i32, i32
        }
        %22 = arith.addi %arg3, %c1_i32 : i32
        %23 = llvm.mlir.undef : i32
        scf.yield %22, %21#0, %21#1, %23, %23 : i32, i32, i32, i32, i32
      } else {
        scf.yield %arg3, %arg4, %arg5, %arg3, %arg5 : i32, i32, i32, i32, i32
      }
      scf.condition(%14) %15#0, %15#1, %15#2, %15#3, %15#4 : i32, i32, i32, i32, i32
    } do {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32):
      scf.yield %arg3, %arg4, %arg5 : i32, i32, i32
    }
    %2:3 = scf.while (%arg3 = %1#3, %arg4 = %1#4) : (i32, i32) -> (i32, i32, i32) {
      %12 = arith.cmpi slt, %arg4, %0 : i32
      %13:3 = scf.if %12 -> (i32, i32, i32) {
        %14 = arith.index_cast %arg3 : i32 to index
        %15 = arith.index_cast %arg4 : i32 to index
        %16 = memref.load %arg0[%15] : memref<?xi32>
        memref.store %16, %arg2[%14] : memref<?xi32>
        %17 = arith.addi %arg4, %c1_i32 : i32
        %18 = arith.addi %arg3, %c1_i32 : i32
        %19 = llvm.mlir.undef : i32
        scf.yield %18, %17, %19 : i32, i32, i32
      } else {
        scf.yield %arg3, %arg4, %arg3 : i32, i32, i32
      }
      scf.condition(%12) %13#0, %13#1, %13#2 : i32, i32, i32
    } do {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):
      scf.yield %arg3, %arg4 : i32, i32
    }
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = arith.index_cast %1#1 : i32 to index
    scf.for %arg3 = %4 to %3 step %c1 {
      %12 = arith.subi %arg3, %4 : index
      %13 = arith.index_cast %2#2 : i32 to index
      %14 = arith.addi %13, %12 : index
      %15 = memref.load %arg0[%arg3] : memref<?xi32>
      memref.store %15, %arg2[%14] : memref<?xi32>
    }
    %5 = polygeist.memref2pointer %arg0 : memref<?xi32> to !llvm.ptr
    %6 = polygeist.pointer2memref %5 : !llvm.ptr to memref<?xi8>
    %7 = polygeist.memref2pointer %arg2 : memref<?xi32> to !llvm.ptr
    %8 = arith.extsi %arg1 : i32 to i64
    %9 = arith.muli %8, %c4_i64 : i64
    %10 = call @__builtin_object_size(%6, %c0_i32) : (memref<?xi8>, i32) -> i64
    %11 = call @__memcpy_chk(%5, %7, %9, %10) : (!llvm.ptr, !llvm.ptr, i64, i64) -> memref<?xi8>
    return
  }
  func.func private @__builtin_object_size(memref<?xi8>, i32) -> i64 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @__memcpy_chk(!llvm.ptr, !llvm.ptr, i64, i64) -> memref<?xi8>
}
