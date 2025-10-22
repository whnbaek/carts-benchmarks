module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:o-i64:64-i128:128-n32:64-S128", llvm.target_triple = "arm64-apple-macosx15.0.0", "polygeist.target-cpu" = "apple-m1", "polygeist.target-features" = "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"} {
  func.func @sparselu_par_call(%arg0: memref<?xmemref<?xf32>>, %arg1: i32, %arg2: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    arts.edt <parallel> <internode> route(%c0_i32) {
      arts.barrier
      arts.edt <single> <intranode> route(%c0_i32) {
        %0 = llvm.mlir.zero : !llvm.ptr
        %1 = arith.index_cast %arg1 : i32 to index
        scf.for %arg3 = %c0 to %1 step %c1 {
          %2 = arith.index_cast %arg3 : index to i32
          %3 = arith.muli %2, %arg1 : i32
          %4 = arith.addi %3, %2 : i32
          %5 = arith.index_cast %4 : i32 to index
          %6 = memref.load %arg0[%5] : memref<?xmemref<?xf32>>
          func.call @lu0(%6, %arg2) : (memref<?xf32>, i32) -> ()
          %7 = arith.addi %2, %c1_i32 : i32
          %8 = arith.index_cast %7 : i32 to index
          scf.for %arg4 = %8 to %1 step %c1 {
            %9 = arith.index_cast %arg4 : index to i32
            %10 = arith.addi %3, %9 : i32
            %11 = arith.index_cast %10 : i32 to index
            %12 = memref.load %arg0[%11] : memref<?xmemref<?xf32>>
            %13 = polygeist.memref2pointer %12 : memref<?xf32> to !llvm.ptr
            %14 = llvm.icmp "ne" %13, %0 : !llvm.ptr
            scf.if %14 {
              arts.edt <task> <intranode> route(%c0_i32) {
                %15 = memref.load %arg0[%5] : memref<?xmemref<?xf32>>
                %16 = memref.load %arg0[%11] : memref<?xmemref<?xf32>>
                func.call @fwd(%15, %16, %arg2) : (memref<?xf32>, memref<?xf32>, i32) -> ()
              }
            }
          }
          scf.for %arg4 = %8 to %1 step %c1 {
            %9 = arith.index_cast %arg4 : index to i32
            %10 = arith.muli %9, %arg1 : i32
            %11 = arith.addi %10, %2 : i32
            %12 = arith.index_cast %11 : i32 to index
            %13 = memref.load %arg0[%12] : memref<?xmemref<?xf32>>
            %14 = polygeist.memref2pointer %13 : memref<?xf32> to !llvm.ptr
            %15 = llvm.icmp "ne" %14, %0 : !llvm.ptr
            scf.if %15 {
              arts.edt <task> <intranode> route(%c0_i32) {
                %16 = memref.load %arg0[%5] : memref<?xmemref<?xf32>>
                %17 = memref.load %arg0[%12] : memref<?xmemref<?xf32>>
                func.call @bdiv(%16, %17, %arg2) : (memref<?xf32>, memref<?xf32>, i32) -> ()
              }
            }
          }
          arts.barrier
          scf.for %arg4 = %8 to %1 step %c1 {
            %9 = arith.index_cast %arg4 : index to i32
            %10 = arith.muli %9, %arg1 : i32
            %11 = arith.addi %10, %2 : i32
            %12 = arith.index_cast %11 : i32 to index
            %13 = memref.load %arg0[%12] : memref<?xmemref<?xf32>>
            %14 = polygeist.memref2pointer %13 : memref<?xf32> to !llvm.ptr
            %15 = llvm.icmp "ne" %14, %0 : !llvm.ptr
            scf.if %15 {
              scf.for %arg5 = %8 to %1 step %c1 {
                %16 = arith.index_cast %arg5 : index to i32
                %17 = arith.addi %3, %16 : i32
                %18 = arith.index_cast %17 : i32 to index
                %19 = memref.load %arg0[%18] : memref<?xmemref<?xf32>>
                %20 = polygeist.memref2pointer %19 : memref<?xf32> to !llvm.ptr
                %21 = llvm.icmp "ne" %20, %0 : !llvm.ptr
                scf.if %21 {
                  arts.edt <task> <intranode> route(%c0_i32) {
                    %22 = arith.addi %10, %16 : i32
                    %23 = arith.index_cast %22 : i32 to index
                    %24 = memref.load %arg0[%23] : memref<?xmemref<?xf32>>
                    %25 = polygeist.memref2pointer %24 : memref<?xf32> to !llvm.ptr
                    %26 = llvm.icmp "eq" %25, %0 : !llvm.ptr
                    scf.if %26 {
                      %30 = func.call @allocate_clean_block(%arg2) : (i32) -> memref<?xf32>
                      memref.store %30, %arg0[%23] : memref<?xmemref<?xf32>>
                    }
                    %27 = memref.load %arg0[%12] : memref<?xmemref<?xf32>>
                    %28 = memref.load %arg0[%18] : memref<?xmemref<?xf32>>
                    %29 = memref.load %arg0[%23] : memref<?xmemref<?xf32>>
                    func.call @bmod(%27, %28, %29, %arg2) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32) -> ()
                  }
                }
              }
            }
          }
          arts.barrier
        }
      }
      arts.barrier
    }
    return
  }
  func.func private @lu0(memref<?xf32>, i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @fwd(memref<?xf32>, memref<?xf32>, i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @bdiv(memref<?xf32>, memref<?xf32>, i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @allocate_clean_block(i32) -> memref<?xf32> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @bmod(memref<?xf32>, memref<?xf32>, memref<?xf32>, i32) attributes {llvm.linkage = #llvm.linkage<external>}
}
