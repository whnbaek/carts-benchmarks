module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:o-i64:64-i128:128-n32:64-S128", llvm.target_triple = "arm64-apple-macosx15.0.0", "polygeist.target-cpu" = "apple-m1", "polygeist.target-features" = "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"} {
  llvm.mlir.global internal constant @str0("sparselu\00") {addr_space = 0 : i32}
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %alloca = memref.alloca() : memref<1x!llvm.struct<(i32, i32, memref<?xi8>, i32, i32, i32)>>
    %0 = polygeist.memref2pointer %alloca : memref<1x!llvm.struct<(i32, i32, memref<?xi8>, i32, i32, i32)>> to !llvm.ptr
    %1 = llvm.getelementptr %0[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi8>, i32, i32, i32)>
    %2 = llvm.mlir.addressof @str0 : !llvm.ptr
    %3 = polygeist.pointer2memref %2 : !llvm.ptr to memref<?xi8>
    llvm.store %3, %1 : memref<?xi8>, !llvm.ptr
    %4 = llvm.getelementptr %0[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi8>, i32, i32, i32)>
    llvm.store %c1_i32, %4 : i32, !llvm.ptr
    %5 = llvm.getelementptr %0[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi8>, i32, i32, i32)>
    llvm.store %c128_i32, %5 : i32, !llvm.ptr
    %6 = llvm.getelementptr %0[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi8>, i32, i32, i32)>
    llvm.store %c16_i32, %6 : i32, !llvm.ptr
    llvm.store %c0_i32, %0 : i32, !llvm.ptr
    %7 = llvm.getelementptr %0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, memref<?xi8>, i32, i32, i32)>
    llvm.store %c1_i32, %7 : i32, !llvm.ptr
    return %c0_i32 : i32
  }
}
