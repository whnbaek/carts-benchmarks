module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:o-i64:64-i128:128-n32:64-S128", llvm.target_triple = "arm64-apple-macosx16.0.0", "polygeist.target-cpu" = "apple-m1", "polygeist.target-features" = "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"} {
  memref.global "private" @rms_final_weight : memref<64xf32> = uninitialized
  memref.global "private" @w2 : memref<32768xf32> = uninitialized
  memref.global "private" @w3 : memref<32768xf32> = uninitialized
  memref.global "private" @w1 : memref<32768xf32> = uninitialized
  memref.global "private" @wv : memref<8192xf32> = uninitialized
  memref.global "private" @wk : memref<8192xf32> = uninitialized
  memref.global "private" @wo : memref<8192xf32> = uninitialized
  memref.global "private" @wq : memref<8192xf32> = uninitialized
  memref.global "private" @rms_ffn_weight : memref<128xf32> = uninitialized
  memref.global "private" @rms_att_weight : memref<128xf32> = uninitialized
  memref.global "private" @token_embedding_table : memref<16384xf32> = uninitialized
  memref.global "private" @value_cache : memref<4096xf32> = uninitialized
  memref.global "private" @key_cache : memref<4096xf32> = uninitialized
  memref.global "private" @logits : memref<256xf32> = uninitialized
  memref.global "private" @att_buf : memref<128xf32> = uninitialized
  memref.global "private" @q_buf : memref<64xf32> = uninitialized
  memref.global "private" @hb2 : memref<256xf32> = uninitialized
  memref.global "private" @hb : memref<256xf32> = uninitialized
  memref.global "private" @xb2 : memref<64xf32> = uninitialized
  memref.global "private" @xb : memref<64xf32> = uninitialized
  memref.global "private" @x : memref<64xf32> = uninitialized
  llvm.mlir.global internal constant @str10("All tests completed successfully!\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str9("Matmul test: [1,2,3; 4,5,6] @ [1,1,1] = [%.1f, %.1f]\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str8("Softmax test: [1.0, 2.0, 3.0, 4.0] -> [%.4f, %.4f, %.4f, %.4f]\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str7("RMSNorm test: [%.4f, %.4f, %.4f, %.4f] -> [%.4f, %.4f, %.4f, %.4f]\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str6("\0ATesting individual functions...\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str5("\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str4("%.4f \00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str3("Forward pass completed. First 10 logits: \00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2("Testing forward pass...\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("Configuration: dim=%d, hidden_dim=%d, n_layers=%d, n_heads=%d, vocab_size=%d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str0("Testing isolated Transformer neural network functions\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c10 = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3_i32 = arith.constant 3 : i32
    %cst = arith.constant 6.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e+00 : f32
    %cst_1 = arith.constant 4.000000e+00 : f32
    %cst_2 = arith.constant 3.000000e+00 : f32
    %cst_3 = arith.constant 2.000000e+00 : f32
    %cst_4 = arith.constant 1.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c42_i32 = arith.constant 42 : i32
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    %c256_i32 = arith.constant 256 : i32
    %c64_i32 = arith.constant 64 : i32
    %alloca = memref.alloca() : memref<2xf32>
    %alloca_5 = memref.alloca() : memref<3xf32>
    %alloca_6 = memref.alloca() : memref<6xf32>
    %alloca_7 = memref.alloca() : memref<4xf32>
    %alloca_8 = memref.alloca() : memref<4xf32>
    %alloca_9 = memref.alloca() : memref<4xf32>
    %alloca_10 = memref.alloca() : memref<4xf32>
    %0 = llvm.mlir.addressof @str0 : !llvm.ptr
    %1 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<55 x i8>
    %2 = llvm.call @printf(%1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %3 = llvm.mlir.addressof @str1 : !llvm.ptr
    %4 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<78 x i8>
    %5 = llvm.call @printf(%4, %c64_i32, %c256_i32, %c2_i32, %c4_i32, %c256_i32) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
    call @initialize_state() : () -> ()
    call @srand(%c42_i32) : (i32) -> ()
    call @initialize_test_data() : () -> ()
    %6 = llvm.mlir.addressof @str2 : !llvm.ptr
    %7 = llvm.getelementptr %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<25 x i8>
    %8 = llvm.call @printf(%7) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %9 = call @forward(%c42_i32, %c0_i32) : (i32, i32) -> memref<?xf32>
    %10 = llvm.mlir.addressof @str3 : !llvm.ptr
    %11 = llvm.getelementptr %10[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<42 x i8>
    %12 = llvm.call @printf(%11) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    scf.for %arg0 = %c0 to %c10 step %c1 {
      %59 = llvm.mlir.addressof @str4 : !llvm.ptr
      %60 = llvm.getelementptr %59[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<6 x i8>
      %61 = memref.load %9[%arg0] : memref<?xf32>
      %62 = arith.extf %61 : f32 to f64
      %63 = llvm.call @printf(%60, %62) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    }
    %13 = llvm.mlir.addressof @str5 : !llvm.ptr
    %14 = llvm.getelementptr %13[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i8>
    %15 = llvm.call @printf(%14) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %16 = llvm.mlir.addressof @str6 : !llvm.ptr
    %17 = llvm.getelementptr %16[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<34 x i8>
    %18 = llvm.call @printf(%17) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    affine.store %cst_4, %alloca_10[0] : memref<4xf32>
    affine.store %cst_3, %alloca_10[1] : memref<4xf32>
    affine.store %cst_2, %alloca_10[2] : memref<4xf32>
    affine.store %cst_1, %alloca_10[3] : memref<4xf32>
    affine.store %cst_4, %alloca_9[0] : memref<4xf32>
    affine.store %cst_4, %alloca_9[1] : memref<4xf32>
    affine.store %cst_4, %alloca_9[2] : memref<4xf32>
    affine.store %cst_4, %alloca_9[3] : memref<4xf32>
    %cast = memref.cast %alloca_8 : memref<4xf32> to memref<?xf32>
    %cast_11 = memref.cast %alloca_10 : memref<4xf32> to memref<?xf32>
    %cast_12 = memref.cast %alloca_9 : memref<4xf32> to memref<?xf32>
    call @rmsnorm(%cast, %cast_11, %cast_12, %c4_i32) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32) -> ()
    %19 = llvm.mlir.addressof @str7 : !llvm.ptr
    %20 = llvm.getelementptr %19[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<68 x i8>
    %21 = affine.load %alloca_10[0] : memref<4xf32>
    %22 = arith.extf %21 : f32 to f64
    %23 = affine.load %alloca_10[1] : memref<4xf32>
    %24 = arith.extf %23 : f32 to f64
    %25 = affine.load %alloca_10[2] : memref<4xf32>
    %26 = arith.extf %25 : f32 to f64
    %27 = affine.load %alloca_10[3] : memref<4xf32>
    %28 = arith.extf %27 : f32 to f64
    %29 = affine.load %alloca_8[0] : memref<4xf32>
    %30 = arith.extf %29 : f32 to f64
    %31 = affine.load %alloca_8[1] : memref<4xf32>
    %32 = arith.extf %31 : f32 to f64
    %33 = affine.load %alloca_8[2] : memref<4xf32>
    %34 = arith.extf %33 : f32 to f64
    %35 = affine.load %alloca_8[3] : memref<4xf32>
    %36 = arith.extf %35 : f32 to f64
    %37 = llvm.call @printf(%20, %22, %24, %26, %28, %30, %32, %34, %36) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64, f64, f64, f64, f64, f64, f64) -> i32
    affine.store %cst_4, %alloca_7[0] : memref<4xf32>
    affine.store %cst_3, %alloca_7[1] : memref<4xf32>
    affine.store %cst_2, %alloca_7[2] : memref<4xf32>
    affine.store %cst_1, %alloca_7[3] : memref<4xf32>
    %cast_13 = memref.cast %alloca_7 : memref<4xf32> to memref<?xf32>
    call @softmax(%cast_13, %c4_i32) : (memref<?xf32>, i32) -> ()
    %38 = llvm.mlir.addressof @str8 : !llvm.ptr
    %39 = llvm.getelementptr %38[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x i8>
    %40 = affine.load %alloca_7[0] : memref<4xf32>
    %41 = arith.extf %40 : f32 to f64
    %42 = affine.load %alloca_7[1] : memref<4xf32>
    %43 = arith.extf %42 : f32 to f64
    %44 = affine.load %alloca_7[2] : memref<4xf32>
    %45 = arith.extf %44 : f32 to f64
    %46 = affine.load %alloca_7[3] : memref<4xf32>
    %47 = arith.extf %46 : f32 to f64
    %48 = llvm.call @printf(%39, %41, %43, %45, %47) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64, f64, f64) -> i32
    affine.store %cst_4, %alloca_6[0] : memref<6xf32>
    affine.store %cst_3, %alloca_6[1] : memref<6xf32>
    affine.store %cst_2, %alloca_6[2] : memref<6xf32>
    affine.store %cst_1, %alloca_6[3] : memref<6xf32>
    affine.store %cst_0, %alloca_6[4] : memref<6xf32>
    affine.store %cst, %alloca_6[5] : memref<6xf32>
    affine.store %cst_4, %alloca_5[0] : memref<3xf32>
    affine.store %cst_4, %alloca_5[1] : memref<3xf32>
    affine.store %cst_4, %alloca_5[2] : memref<3xf32>
    %cast_14 = memref.cast %alloca : memref<2xf32> to memref<?xf32>
    %cast_15 = memref.cast %alloca_5 : memref<3xf32> to memref<?xf32>
    %cast_16 = memref.cast %alloca_6 : memref<6xf32> to memref<?xf32>
    call @matmul(%cast_14, %cast_15, %cast_16, %c3_i32, %c2_i32) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, i32) -> ()
    %49 = llvm.mlir.addressof @str9 : !llvm.ptr
    %50 = llvm.getelementptr %49[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<54 x i8>
    %51 = affine.load %alloca[0] : memref<2xf32>
    %52 = arith.extf %51 : f32 to f64
    %53 = affine.load %alloca[1] : memref<2xf32>
    %54 = arith.extf %53 : f32 to f64
    %55 = llvm.call @printf(%50, %52, %54) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64) -> i32
    %56 = llvm.mlir.addressof @str10 : !llvm.ptr
    %57 = llvm.getelementptr %56[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<35 x i8>
    %58 = llvm.call @printf(%57) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    return %c0_i32 : i32
  }
  func.func private @initialize_state() attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c4096_i32 = arith.constant 4096 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = memref.get_global @x : memref<64xf32>
    %cast = memref.cast %0 : memref<64xf32> to memref<?xf32>
    call @zero_floats(%cast, %c64_i32) : (memref<?xf32>, i32) -> ()
    %1 = memref.get_global @xb : memref<64xf32>
    %cast_0 = memref.cast %1 : memref<64xf32> to memref<?xf32>
    call @zero_floats(%cast_0, %c64_i32) : (memref<?xf32>, i32) -> ()
    %2 = memref.get_global @xb2 : memref<64xf32>
    %cast_1 = memref.cast %2 : memref<64xf32> to memref<?xf32>
    call @zero_floats(%cast_1, %c64_i32) : (memref<?xf32>, i32) -> ()
    %3 = memref.get_global @hb : memref<256xf32>
    %cast_2 = memref.cast %3 : memref<256xf32> to memref<?xf32>
    call @zero_floats(%cast_2, %c256_i32) : (memref<?xf32>, i32) -> ()
    %4 = memref.get_global @hb2 : memref<256xf32>
    %cast_3 = memref.cast %4 : memref<256xf32> to memref<?xf32>
    call @zero_floats(%cast_3, %c256_i32) : (memref<?xf32>, i32) -> ()
    %5 = memref.get_global @q_buf : memref<64xf32>
    %cast_4 = memref.cast %5 : memref<64xf32> to memref<?xf32>
    call @zero_floats(%cast_4, %c64_i32) : (memref<?xf32>, i32) -> ()
    %6 = memref.get_global @att_buf : memref<128xf32>
    %cast_5 = memref.cast %6 : memref<128xf32> to memref<?xf32>
    call @zero_floats(%cast_5, %c128_i32) : (memref<?xf32>, i32) -> ()
    %7 = memref.get_global @logits : memref<256xf32>
    %cast_6 = memref.cast %7 : memref<256xf32> to memref<?xf32>
    call @zero_floats(%cast_6, %c256_i32) : (memref<?xf32>, i32) -> ()
    %8 = memref.get_global @key_cache : memref<4096xf32>
    %cast_7 = memref.cast %8 : memref<4096xf32> to memref<?xf32>
    call @zero_floats(%cast_7, %c4096_i32) : (memref<?xf32>, i32) -> ()
    %9 = memref.get_global @value_cache : memref<4096xf32>
    %cast_8 = memref.cast %9 : memref<4096xf32> to memref<?xf32>
    call @zero_floats(%cast_8, %c4096_i32) : (memref<?xf32>, i32) -> ()
    return
  }
  func.func private @srand(i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @initialize_test_data() attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c0 = arith.constant 0 : index
    %c16384 = arith.constant 16384 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4096 = arith.constant 4096 : index
    %c-50_i32 = arith.constant -50 : i32
    %c64 = arith.constant 64 : index
    %c4096_i32 = arith.constant 4096 : i32
    %cst = arith.constant 1.000000e+00 : f32
    %c64_i32 = arith.constant 64 : i32
    %cst_0 = arith.constant 5.000000e+01 : f32
    %c100_i32 = arith.constant 100 : i32
    %cst_1 = arith.constant 0.00999999977 : f32
    %c16384_i32 = arith.constant 16384 : i32
    scf.for %arg0 = %c0 to %c16384 step %c1 {
      %0 = memref.get_global @token_embedding_table : memref<16384xf32>
      %1 = func.call @rand() : () -> i32
      %2 = arith.remsi %1, %c100_i32 : i32
      %3 = arith.addi %2, %c-50_i32 : i32
      %4 = arith.sitofp %3 : i32 to f32
      %5 = arith.mulf %4, %cst_1 : f32
      %6 = arith.divf %5, %cst_0 : f32
      memref.store %6, %0[%arg0] : memref<16384xf32>
    }
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = arith.index_cast %arg0 : index to i32
      scf.for %arg1 = %c0 to %c64 step %c1 {
        %1 = arith.index_cast %arg1 : index to i32
        %2 = memref.get_global @rms_att_weight : memref<128xf32>
        %3 = arith.muli %0, %c64_i32 : i32
        %4 = arith.addi %3, %1 : i32
        %5 = arith.index_cast %4 : i32 to index
        memref.store %cst, %2[%5] : memref<128xf32>
        %6 = memref.get_global @rms_ffn_weight : memref<128xf32>
        memref.store %cst, %6[%5] : memref<128xf32>
      }
      scf.for %arg1 = %c0 to %c4096 step %c1 {
        %1 = arith.index_cast %arg1 : index to i32
        %2 = memref.get_global @wq : memref<8192xf32>
        %3 = arith.muli %0, %c4096_i32 : i32
        %4 = arith.addi %3, %1 : i32
        %5 = arith.index_cast %4 : i32 to index
        %6 = func.call @rand() : () -> i32
        %7 = arith.remsi %6, %c100_i32 : i32
        %8 = arith.addi %7, %c-50_i32 : i32
        %9 = arith.sitofp %8 : i32 to f32
        %10 = arith.mulf %9, %cst_1 : f32
        %11 = arith.divf %10, %cst_0 : f32
        memref.store %11, %2[%5] : memref<8192xf32>
        %12 = memref.get_global @wo : memref<8192xf32>
        %13 = func.call @rand() : () -> i32
        %14 = arith.remsi %13, %c100_i32 : i32
        %15 = arith.addi %14, %c-50_i32 : i32
        %16 = arith.sitofp %15 : i32 to f32
        %17 = arith.mulf %16, %cst_1 : f32
        %18 = arith.divf %17, %cst_0 : f32
        memref.store %18, %12[%5] : memref<8192xf32>
      }
      scf.for %arg1 = %c0 to %c4096 step %c1 {
        %1 = arith.index_cast %arg1 : index to i32
        %2 = memref.get_global @wk : memref<8192xf32>
        %3 = arith.muli %0, %c4096_i32 : i32
        %4 = arith.addi %3, %1 : i32
        %5 = arith.index_cast %4 : i32 to index
        %6 = func.call @rand() : () -> i32
        %7 = arith.remsi %6, %c100_i32 : i32
        %8 = arith.addi %7, %c-50_i32 : i32
        %9 = arith.sitofp %8 : i32 to f32
        %10 = arith.mulf %9, %cst_1 : f32
        %11 = arith.divf %10, %cst_0 : f32
        memref.store %11, %2[%5] : memref<8192xf32>
        %12 = memref.get_global @wv : memref<8192xf32>
        %13 = func.call @rand() : () -> i32
        %14 = arith.remsi %13, %c100_i32 : i32
        %15 = arith.addi %14, %c-50_i32 : i32
        %16 = arith.sitofp %15 : i32 to f32
        %17 = arith.mulf %16, %cst_1 : f32
        %18 = arith.divf %17, %cst_0 : f32
        memref.store %18, %12[%5] : memref<8192xf32>
      }
      scf.for %arg1 = %c0 to %c16384 step %c1 {
        %1 = arith.index_cast %arg1 : index to i32
        %2 = memref.get_global @w1 : memref<32768xf32>
        %3 = arith.muli %0, %c16384_i32 : i32
        %4 = arith.addi %3, %1 : i32
        %5 = arith.index_cast %4 : i32 to index
        %6 = func.call @rand() : () -> i32
        %7 = arith.remsi %6, %c100_i32 : i32
        %8 = arith.addi %7, %c-50_i32 : i32
        %9 = arith.sitofp %8 : i32 to f32
        %10 = arith.mulf %9, %cst_1 : f32
        %11 = arith.divf %10, %cst_0 : f32
        memref.store %11, %2[%5] : memref<32768xf32>
        %12 = memref.get_global @w3 : memref<32768xf32>
        %13 = func.call @rand() : () -> i32
        %14 = arith.remsi %13, %c100_i32 : i32
        %15 = arith.addi %14, %c-50_i32 : i32
        %16 = arith.sitofp %15 : i32 to f32
        %17 = arith.mulf %16, %cst_1 : f32
        %18 = arith.divf %17, %cst_0 : f32
        memref.store %18, %12[%5] : memref<32768xf32>
      }
      scf.for %arg1 = %c0 to %c16384 step %c1 {
        %1 = arith.index_cast %arg1 : index to i32
        %2 = memref.get_global @w2 : memref<32768xf32>
        %3 = arith.muli %0, %c16384_i32 : i32
        %4 = arith.addi %3, %1 : i32
        %5 = arith.index_cast %4 : i32 to index
        %6 = func.call @rand() : () -> i32
        %7 = arith.remsi %6, %c100_i32 : i32
        %8 = arith.addi %7, %c-50_i32 : i32
        %9 = arith.sitofp %8 : i32 to f32
        %10 = arith.mulf %9, %cst_1 : f32
        %11 = arith.divf %10, %cst_0 : f32
        memref.store %11, %2[%5] : memref<32768xf32>
      }
    }
    scf.for %arg0 = %c0 to %c64 step %c1 {
      %0 = memref.get_global @rms_final_weight : memref<64xf32>
      memref.store %cst, %0[%arg0] : memref<64xf32>
    }
    return
  }
  func.func private @forward(%arg0: i32, %arg1: i32) -> memref<?xf32> attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %c2048_i32 = arith.constant 2048 : i32
    %cst = arith.constant 1.600000e+01 : f32
    %c4 = arith.constant 4 : index
    %cst_0 = arith.constant 4.000000e+00 : f32
    %c4096_i32 = arith.constant 4096 : i32
    %c16384_i32 = arith.constant 16384 : i32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 1.000000e+04 : f32
    %cst_3 = arith.constant 1.000000e+00 : f32
    %c16_i32 = arith.constant 16 : i32
    %c256_i32 = arith.constant 256 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = memref.get_global @x : memref<64xf32>
    %cast = memref.cast %0 : memref<64xf32> to memref<?xf32>
    %1 = memref.get_global @xb : memref<64xf32>
    %cast_4 = memref.cast %1 : memref<64xf32> to memref<?xf32>
    %2 = memref.get_global @xb2 : memref<64xf32>
    %cast_5 = memref.cast %2 : memref<64xf32> to memref<?xf32>
    %3 = memref.get_global @hb : memref<256xf32>
    %cast_6 = memref.cast %3 : memref<256xf32> to memref<?xf32>
    %4 = memref.get_global @hb2 : memref<256xf32>
    %cast_7 = memref.cast %4 : memref<256xf32> to memref<?xf32>
    %5 = memref.get_global @q_buf : memref<64xf32>
    %cast_8 = memref.cast %5 : memref<64xf32> to memref<?xf32>
    %6 = memref.get_global @att_buf : memref<128xf32>
    %7 = arith.muli %arg0, %c64_i32 : i32
    scf.for %arg2 = %c0 to %c64 step %c1 {
      %12 = arith.index_cast %arg2 : index to i32
      %13 = memref.get_global @token_embedding_table : memref<16384xf32>
      %14 = arith.addi %7, %12 : i32
      %15 = arith.index_cast %14 : i32 to index
      %16 = memref.load %13[%15] : memref<16384xf32>
      memref.store %16, %0[%arg2] : memref<64xf32>
    }
    scf.for %arg2 = %c0 to %c2 step %c1 {
      %12 = arith.index_cast %arg2 : index to i32
      %13 = memref.get_global @rms_att_weight : memref<128xf32>
      %14 = arith.muli %12, %c64_i32 : i32
      %15 = arith.index_cast %14 : i32 to index
      %16 = polygeist.subindex %13[%15] () : memref<128xf32> -> memref<?xf32>
      func.call @rmsnorm(%cast_4, %cast, %16, %c64_i32) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32) -> ()
      %17 = arith.muli %12, %c2048_i32 : i32
      %18 = arith.muli %arg1, %c64_i32 : i32
      %19 = arith.addi %17, %18 : i32
      %20 = memref.get_global @wq : memref<8192xf32>
      %21 = arith.muli %12, %c4096_i32 : i32
      %22 = arith.index_cast %21 : i32 to index
      %23 = polygeist.subindex %20[%22] () : memref<8192xf32> -> memref<?xf32>
      func.call @matmul(%cast_8, %cast_4, %23, %c64_i32, %c64_i32) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, i32) -> ()
      %24 = memref.get_global @key_cache : memref<4096xf32>
      %25 = arith.index_cast %19 : i32 to index
      %26 = polygeist.subindex %24[%25] () : memref<4096xf32> -> memref<?xf32>
      %27 = memref.get_global @wk : memref<8192xf32>
      %28 = polygeist.subindex %27[%22] () : memref<8192xf32> -> memref<?xf32>
      func.call @matmul(%26, %cast_4, %28, %c64_i32, %c64_i32) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, i32) -> ()
      %29 = memref.get_global @value_cache : memref<4096xf32>
      %30 = polygeist.subindex %29[%25] () : memref<4096xf32> -> memref<?xf32>
      %31 = memref.get_global @wv : memref<8192xf32>
      %32 = polygeist.subindex %31[%22] () : memref<8192xf32> -> memref<?xf32>
      func.call @matmul(%30, %cast_4, %32, %c64_i32, %c64_i32) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, i32) -> ()
      scf.for %arg3 = %c0 to %c64 step %c2 {
        %56 = arith.index_cast %arg3 : index to i32
        %57 = arith.remsi %56, %c16_i32 : i32
        %58 = arith.sitofp %57 : i32 to f32
        %59 = arith.divf %58, %cst : f32
        %60 = math.powf %cst_2, %59 : f32
        %61 = arith.divf %cst_3, %60 : f32
        %62 = arith.sitofp %arg1 : i32 to f32
        %63 = arith.mulf %62, %61 : f32
        %64 = func.call @cosf(%63) : (f32) -> f32
        %65 = func.call @sinf(%63) : (f32) -> f32
        scf.for %arg4 = %c0 to %c2 step %c1 {
          %66 = arith.index_cast %arg4 : index to i32
          %67 = arith.cmpi eq, %66, %c0_i32 : i32
          %68 = arith.select %67, %cast_8, %26 : memref<?xf32>
          %69 = scf.if %67 -> (f32) {
            %79 = memref.load %5[%arg3] : memref<64xf32>
            scf.yield %79 : f32
          } else {
            %79 = arith.addi %arg3, %25 : index
            %80 = memref.load %24[%79] : memref<4096xf32>
            scf.yield %80 : f32
          }
          %70 = arith.addi %56, %c1_i32 : i32
          %71 = arith.index_cast %70 : i32 to index
          %72 = scf.if %67 -> (f32) {
            %79 = memref.load %5[%71] : memref<64xf32>
            scf.yield %79 : f32
          } else {
            %79 = arith.addi %71, %25 : index
            %80 = memref.load %24[%79] : memref<4096xf32>
            scf.yield %80 : f32
          }
          %73 = arith.mulf %69, %64 : f32
          %74 = arith.mulf %72, %65 : f32
          %75 = arith.subf %73, %74 : f32
          memref.store %75, %68[%arg3] : memref<?xf32>
          %76 = arith.mulf %69, %65 : f32
          %77 = arith.mulf %72, %64 : f32
          %78 = arith.addf %76, %77 : f32
          memref.store %78, %68[%71] : memref<?xf32>
        }
      }
      %33 = arith.addi %arg1, %c1_i32 : i32
      %34 = memref.get_global @key_cache : memref<4096xf32>
      %35 = memref.get_global @value_cache : memref<4096xf32>
      scf.parallel (%arg3) = (%c0) to (%c4) step (%c1) {
        %56 = arith.index_cast %arg3 : index to i32
        %57 = arith.muli %56, %c16_i32 : i32
        %58 = arith.index_cast %57 : i32 to index
        %59 = arith.muli %56, %c32_i32 : i32
        %60 = arith.index_cast %59 : i32 to index
        %61 = polygeist.subindex %6[%60] () : memref<128xf32> -> memref<?xf32>
        %62 = arith.addi %arg1, %c1_i32 : i32
        %63 = arith.index_cast %62 : i32 to index
        scf.for %arg4 = %c0 to %63 step %c1 {
          %69 = arith.index_cast %arg4 : index to i32
          %70 = arith.muli %69, %c64_i32 : i32
          %71 = arith.addi %17, %70 : i32
          %72 = arith.muli %56, %c16_i32 : i32
          %73 = arith.addi %71, %72 : i32
          %74 = arith.index_cast %73 : i32 to index
          %75 = scf.for %arg5 = %c0 to %c16 step %c1 iter_args(%arg6 = %cst_1) -> (f32) {
            %78 = arith.addi %arg5, %58 : index
            %79 = memref.load %5[%78] : memref<64xf32>
            %80 = arith.addi %arg5, %74 : index
            %81 = memref.load %34[%80] : memref<4096xf32>
            %82 = arith.mulf %79, %81 : f32
            %83 = arith.addf %arg6, %82 : f32
            scf.yield %83 : f32
          }
          %76 = arith.divf %75, %cst_0 : f32
          %77 = arith.addi %arg4, %60 : index
          memref.store %76, %6[%77] : memref<128xf32>
        }
        func.call @softmax(%61, %33) : (memref<?xf32>, i32) -> ()
        %64 = arith.muli %56, %c16_i32 : i32
        %65 = arith.index_cast %64 : i32 to index
        %66 = polygeist.subindex %1[%65] () : memref<64xf32> -> memref<?xf32>
        func.call @zero_floats(%66, %c16_i32) : (memref<?xf32>, i32) -> ()
        %67 = arith.addi %arg1, %c1_i32 : i32
        %68 = arith.index_cast %67 : i32 to index
        scf.for %arg4 = %c0 to %68 step %c1 {
          %69 = arith.index_cast %arg4 : index to i32
          %70 = arith.muli %69, %c64_i32 : i32
          %71 = arith.addi %17, %70 : i32
          %72 = arith.muli %56, %c16_i32 : i32
          %73 = arith.addi %71, %72 : i32
          %74 = arith.index_cast %73 : i32 to index
          %75 = arith.addi %arg4, %60 : index
          %76 = memref.load %6[%75] : memref<128xf32>
          scf.for %arg5 = %c0 to %c16 step %c1 {
            %77 = arith.addi %arg5, %74 : index
            %78 = memref.load %35[%77] : memref<4096xf32>
            %79 = arith.mulf %76, %78 : f32
            %80 = arith.addi %arg5, %65 : index
            %81 = memref.load %1[%80] : memref<64xf32>
            %82 = arith.addf %81, %79 : f32
            %83 = arith.addi %arg5, %65 : index
            memref.store %82, %1[%83] : memref<64xf32>
          }
        }
        scf.yield
      }
      %36 = memref.get_global @wo : memref<8192xf32>
      %37 = arith.muli %12, %c4096_i32 : i32
      %38 = arith.index_cast %37 : i32 to index
      %39 = polygeist.subindex %36[%38] () : memref<8192xf32> -> memref<?xf32>
      func.call @matmul(%cast_5, %cast_4, %39, %c64_i32, %c64_i32) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, i32) -> ()
      scf.for %arg3 = %c0 to %c64 step %c1 {
        %56 = memref.load %2[%arg3] : memref<64xf32>
        %57 = memref.load %0[%arg3] : memref<64xf32>
        %58 = arith.addf %57, %56 : f32
        memref.store %58, %0[%arg3] : memref<64xf32>
      }
      %40 = memref.get_global @rms_ffn_weight : memref<128xf32>
      %41 = arith.muli %12, %c64_i32 : i32
      %42 = arith.index_cast %41 : i32 to index
      %43 = polygeist.subindex %40[%42] () : memref<128xf32> -> memref<?xf32>
      func.call @rmsnorm(%cast_4, %cast, %43, %c64_i32) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32) -> ()
      %44 = memref.get_global @w1 : memref<32768xf32>
      %45 = arith.muli %12, %c16384_i32 : i32
      %46 = arith.index_cast %45 : i32 to index
      %47 = polygeist.subindex %44[%46] () : memref<32768xf32> -> memref<?xf32>
      func.call @matmul(%cast_6, %cast_4, %47, %c64_i32, %c256_i32) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, i32) -> ()
      %48 = memref.get_global @w3 : memref<32768xf32>
      %49 = arith.muli %12, %c16384_i32 : i32
      %50 = arith.index_cast %49 : i32 to index
      %51 = polygeist.subindex %48[%50] () : memref<32768xf32> -> memref<?xf32>
      func.call @matmul(%cast_7, %cast_4, %51, %c64_i32, %c256_i32) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, i32) -> ()
      scf.for %arg3 = %c0 to %c256 step %c1 {
        %56 = memref.load %3[%arg3] : memref<256xf32>
        %57 = arith.negf %56 : f32
        %58 = math.exp %57 : f32
        %59 = arith.addf %58, %cst_3 : f32
        %60 = arith.divf %cst_3, %59 : f32
        %61 = arith.mulf %56, %60 : f32
        %62 = memref.load %4[%arg3] : memref<256xf32>
        %63 = arith.mulf %61, %62 : f32
        memref.store %63, %3[%arg3] : memref<256xf32>
      }
      %52 = memref.get_global @w2 : memref<32768xf32>
      %53 = arith.muli %12, %c16384_i32 : i32
      %54 = arith.index_cast %53 : i32 to index
      %55 = polygeist.subindex %52[%54] () : memref<32768xf32> -> memref<?xf32>
      func.call @matmul(%cast_4, %cast_6, %55, %c256_i32, %c64_i32) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, i32) -> ()
      scf.for %arg3 = %c0 to %c64 step %c1 {
        %56 = memref.load %1[%arg3] : memref<64xf32>
        %57 = memref.load %0[%arg3] : memref<64xf32>
        %58 = arith.addf %57, %56 : f32
        memref.store %58, %0[%arg3] : memref<64xf32>
      }
    }
    %8 = memref.get_global @rms_final_weight : memref<64xf32>
    %cast_9 = memref.cast %8 : memref<64xf32> to memref<?xf32>
    call @rmsnorm(%cast, %cast, %cast_9, %c64_i32) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32) -> ()
    %9 = memref.get_global @logits : memref<256xf32>
    %cast_10 = memref.cast %9 : memref<256xf32> to memref<?xf32>
    %10 = memref.get_global @token_embedding_table : memref<16384xf32>
    %cast_11 = memref.cast %10 : memref<16384xf32> to memref<?xf32>
    call @matmul(%cast_10, %cast, %cast_11, %c64_i32, %c256_i32) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, i32) -> ()
    %11 = memref.get_global @logits : memref<256xf32>
    %cast_12 = memref.cast %11 : memref<256xf32> to memref<?xf32>
    return %cast_12 : memref<?xf32>
  }
  func.func private @rmsnorm(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<internal>} {
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
  func.func private @softmax(%arg0: memref<?xf32>, %arg1: i32) attributes {llvm.linkage = #llvm.linkage<internal>} {
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
  func.func private @matmul(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32, %arg4: i32) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg4 : i32 to index
    scf.parallel (%arg5) = (%c0) to (%0) step (%c1) {
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
      scf.yield
    }
    return
  }
  func.func private @zero_floats(%arg0: memref<?xf32>, %arg1: i32) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %arg1 : i32 to index
    scf.for %arg2 = %c0 to %0 step %c1 {
      memref.store %cst, %arg0[%arg2] : memref<?xf32>
    }
    return
  }
  func.func private @rand() -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @cosf(f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @sinf(f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>}
}
