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
    %c0_i32 = arith.constant 0 : i32
    %c2688_i32 = arith.constant 2688 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst = arith.constant 1.000000e+04 : f32
    %cst_0 = arith.constant 1.600000e+01 : f32
    %c2048_i32 = arith.constant 2048 : i32
    %c16 = arith.constant 16 : index
    %cst_1 = arith.constant 6.400000e+01 : f32
    %cst_2 = arith.constant 4.000000e+00 : f64
    %cst_3 = arith.constant 3.000000e+00 : f64
    %cst_4 = arith.constant 2.000000e+00 : f64
    %cst_5 = arith.constant 1.000000e+00 : f64
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %cst_6 = arith.constant 4.000000e+00 : f32
    %cst_7 = arith.constant 9.99999974E-6 : f32
    %c16384_i32 = arith.constant 16384 : i32
    %cst_8 = arith.constant 0.00999999977 : f32
    %c100_i32 = arith.constant 100 : i32
    %cst_9 = arith.constant 5.000000e+01 : f32
    %c4096_i32 = arith.constant 4096 : i32
    %c-50_i32 = arith.constant -50 : i32
    %c16384 = arith.constant 16384 : index
    %cst_10 = arith.constant 0.000000e+00 : f32
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c4096 = arith.constant 4096 : index
    %c5 = arith.constant 5 : index
    %c3 = arith.constant 3 : index
    %c10 = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3_i32 = arith.constant 3 : i32
    %cst_11 = arith.constant 6.000000e+00 : f32
    %cst_12 = arith.constant 5.000000e+00 : f32
    %cst_13 = arith.constant 3.000000e+00 : f32
    %cst_14 = arith.constant 2.000000e+00 : f32
    %cst_15 = arith.constant 1.000000e+00 : f32
    %c42_i32 = arith.constant 42 : i32
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    %c256_i32 = arith.constant 256 : i32
    %c64_i32 = arith.constant 64 : i32
    %alloca = memref.alloca() : memref<2xf32>
    %alloca_16 = memref.alloca() : memref<3xf32>
    %alloca_17 = memref.alloca() : memref<6xf32>
    %alloca_18 = memref.alloca() : memref<4xf32>
    %alloca_19 = memref.alloca() : memref<4xf32>
    %0 = llvm.mlir.addressof @str0 : !llvm.ptr
    %1 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<55 x i8>
    %2 = llvm.call @printf(%1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %3 = llvm.mlir.addressof @str1 : !llvm.ptr
    %4 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<78 x i8>
    %5 = llvm.call @printf(%4, %c64_i32, %c256_i32, %c2_i32, %c4_i32, %c256_i32) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
    %6 = memref.get_global @x : memref<64xf32>
    scf.for %arg0 = %c0 to %c64 step %c1 {
      memref.store %cst_10, %6[%arg0] : memref<64xf32>
    }
    %7 = memref.get_global @xb : memref<64xf32>
    scf.for %arg0 = %c0 to %c64 step %c1 {
      memref.store %cst_10, %7[%arg0] : memref<64xf32>
    }
    %8 = memref.get_global @xb2 : memref<64xf32>
    scf.for %arg0 = %c0 to %c64 step %c1 {
      memref.store %cst_10, %8[%arg0] : memref<64xf32>
    }
    %9 = memref.get_global @hb : memref<256xf32>
    scf.for %arg0 = %c0 to %c256 step %c1 {
      memref.store %cst_10, %9[%arg0] : memref<256xf32>
    }
    %10 = memref.get_global @hb2 : memref<256xf32>
    scf.for %arg0 = %c0 to %c256 step %c1 {
      memref.store %cst_10, %10[%arg0] : memref<256xf32>
    }
    %11 = memref.get_global @q_buf : memref<64xf32>
    scf.for %arg0 = %c0 to %c64 step %c1 {
      memref.store %cst_10, %11[%arg0] : memref<64xf32>
    }
    %12 = memref.get_global @att_buf : memref<128xf32>
    scf.for %arg0 = %c0 to %c128 step %c1 {
      memref.store %cst_10, %12[%arg0] : memref<128xf32>
    }
    %13 = memref.get_global @logits : memref<256xf32>
    scf.for %arg0 = %c0 to %c256 step %c1 {
      memref.store %cst_10, %13[%arg0] : memref<256xf32>
    }
    %14 = memref.get_global @key_cache : memref<4096xf32>
    scf.for %arg0 = %c0 to %c4096 step %c1 {
      memref.store %cst_10, %14[%arg0] : memref<4096xf32>
    }
    %15 = memref.get_global @value_cache : memref<4096xf32>
    scf.for %arg0 = %c0 to %c4096 step %c1 {
      memref.store %cst_10, %15[%arg0] : memref<4096xf32>
    }
    call @srand(%c42_i32) : (i32) -> ()
    scf.for %arg0 = %c0 to %c16384 step %c1 {
      %73 = memref.get_global @token_embedding_table : memref<16384xf32>
      %74 = func.call @rand() : () -> i32
      %75 = arith.remsi %74, %c100_i32 : i32
      %76 = arith.addi %75, %c-50_i32 : i32
      %77 = arith.sitofp %76 : i32 to f32
      %78 = arith.mulf %77, %cst_8 : f32
      %79 = arith.divf %78, %cst_9 : f32
      memref.store %79, %73[%arg0] : memref<16384xf32>
    }
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %73 = arith.index_cast %arg0 : index to i32
      scf.for %arg1 = %c0 to %c64 step %c1 {
        %74 = arith.index_cast %arg1 : index to i32
        %75 = memref.get_global @rms_att_weight : memref<128xf32>
        %76 = arith.muli %73, %c64_i32 : i32
        %77 = arith.addi %76, %74 : i32
        %78 = arith.index_cast %77 : i32 to index
        memref.store %cst_15, %75[%78] : memref<128xf32>
        %79 = memref.get_global @rms_ffn_weight : memref<128xf32>
        memref.store %cst_15, %79[%78] : memref<128xf32>
      }
      scf.for %arg1 = %c0 to %c4096 step %c1 {
        %74 = arith.index_cast %arg1 : index to i32
        %75 = memref.get_global @wq : memref<8192xf32>
        %76 = arith.muli %73, %c4096_i32 : i32
        %77 = arith.addi %76, %74 : i32
        %78 = arith.index_cast %77 : i32 to index
        %79 = func.call @rand() : () -> i32
        %80 = arith.remsi %79, %c100_i32 : i32
        %81 = arith.addi %80, %c-50_i32 : i32
        %82 = arith.sitofp %81 : i32 to f32
        %83 = arith.mulf %82, %cst_8 : f32
        %84 = arith.divf %83, %cst_9 : f32
        memref.store %84, %75[%78] : memref<8192xf32>
        %85 = memref.get_global @wo : memref<8192xf32>
        %86 = func.call @rand() : () -> i32
        %87 = arith.remsi %86, %c100_i32 : i32
        %88 = arith.addi %87, %c-50_i32 : i32
        %89 = arith.sitofp %88 : i32 to f32
        %90 = arith.mulf %89, %cst_8 : f32
        %91 = arith.divf %90, %cst_9 : f32
        memref.store %91, %85[%78] : memref<8192xf32>
      }
      scf.for %arg1 = %c0 to %c4096 step %c1 {
        %74 = arith.index_cast %arg1 : index to i32
        %75 = memref.get_global @wk : memref<8192xf32>
        %76 = arith.muli %73, %c4096_i32 : i32
        %77 = arith.addi %76, %74 : i32
        %78 = arith.index_cast %77 : i32 to index
        %79 = func.call @rand() : () -> i32
        %80 = arith.remsi %79, %c100_i32 : i32
        %81 = arith.addi %80, %c-50_i32 : i32
        %82 = arith.sitofp %81 : i32 to f32
        %83 = arith.mulf %82, %cst_8 : f32
        %84 = arith.divf %83, %cst_9 : f32
        memref.store %84, %75[%78] : memref<8192xf32>
        %85 = memref.get_global @wv : memref<8192xf32>
        %86 = func.call @rand() : () -> i32
        %87 = arith.remsi %86, %c100_i32 : i32
        %88 = arith.addi %87, %c-50_i32 : i32
        %89 = arith.sitofp %88 : i32 to f32
        %90 = arith.mulf %89, %cst_8 : f32
        %91 = arith.divf %90, %cst_9 : f32
        memref.store %91, %85[%78] : memref<8192xf32>
      }
      scf.for %arg1 = %c0 to %c16384 step %c1 {
        %74 = arith.index_cast %arg1 : index to i32
        %75 = memref.get_global @w1 : memref<32768xf32>
        %76 = arith.muli %73, %c16384_i32 : i32
        %77 = arith.addi %76, %74 : i32
        %78 = arith.index_cast %77 : i32 to index
        %79 = func.call @rand() : () -> i32
        %80 = arith.remsi %79, %c100_i32 : i32
        %81 = arith.addi %80, %c-50_i32 : i32
        %82 = arith.sitofp %81 : i32 to f32
        %83 = arith.mulf %82, %cst_8 : f32
        %84 = arith.divf %83, %cst_9 : f32
        memref.store %84, %75[%78] : memref<32768xf32>
        %85 = memref.get_global @w3 : memref<32768xf32>
        %86 = func.call @rand() : () -> i32
        %87 = arith.remsi %86, %c100_i32 : i32
        %88 = arith.addi %87, %c-50_i32 : i32
        %89 = arith.sitofp %88 : i32 to f32
        %90 = arith.mulf %89, %cst_8 : f32
        %91 = arith.divf %90, %cst_9 : f32
        memref.store %91, %85[%78] : memref<32768xf32>
      }
      scf.for %arg1 = %c0 to %c16384 step %c1 {
        %74 = arith.index_cast %arg1 : index to i32
        %75 = memref.get_global @w2 : memref<32768xf32>
        %76 = arith.muli %73, %c16384_i32 : i32
        %77 = arith.addi %76, %74 : i32
        %78 = arith.index_cast %77 : i32 to index
        %79 = func.call @rand() : () -> i32
        %80 = arith.remsi %79, %c100_i32 : i32
        %81 = arith.addi %80, %c-50_i32 : i32
        %82 = arith.sitofp %81 : i32 to f32
        %83 = arith.mulf %82, %cst_8 : f32
        %84 = arith.divf %83, %cst_9 : f32
        memref.store %84, %75[%78] : memref<32768xf32>
      }
    }
    scf.for %arg0 = %c0 to %c64 step %c1 {
      %73 = memref.get_global @rms_final_weight : memref<64xf32>
      memref.store %cst_15, %73[%arg0] : memref<64xf32>
    }
    %16 = llvm.mlir.addressof @str2 : !llvm.ptr
    %17 = llvm.getelementptr %16[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<25 x i8>
    %18 = llvm.call @printf(%17) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %cast = memref.cast %11 : memref<64xf32> to memref<?xf32>
    scf.for %arg0 = %c0 to %c64 step %c1 {
      %73 = arith.index_cast %arg0 : index to i32
      %74 = memref.get_global @token_embedding_table : memref<16384xf32>
      %75 = arith.addi %73, %c2688_i32 : i32
      %76 = arith.index_cast %75 : i32 to index
      %77 = memref.load %74[%76] : memref<16384xf32>
      memref.store %77, %6[%arg0] : memref<64xf32>
    }
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %73 = arith.index_cast %arg0 : index to i32
      %74 = memref.get_global @rms_att_weight : memref<128xf32>
      %75 = arith.muli %73, %c64_i32 : i32
      %76 = arith.index_cast %75 : i32 to index
      %77 = scf.for %arg1 = %c0 to %c64 step %c1 iter_args(%arg2 = %cst_10) -> (f32) {
        %102 = memref.load %6[%arg1] : memref<64xf32>
        %103 = arith.mulf %102, %102 : f32
        %104 = arith.addf %arg2, %103 : f32
        scf.yield %104 : f32
      }
      %78 = arith.divf %77, %cst_1 : f32
      %79 = arith.addf %78, %cst_7 : f32
      %80 = math.sqrt %79 : f32
      %81 = arith.divf %cst_15, %80 : f32
      scf.for %arg1 = %c0 to %c64 step %c1 {
        %102 = arith.addi %arg1, %76 : index
        %103 = memref.load %74[%102] : memref<128xf32>
        %104 = memref.load %6[%arg1] : memref<64xf32>
        %105 = arith.mulf %81, %104 : f32
        %106 = arith.mulf %103, %105 : f32
        memref.store %106, %7[%arg1] : memref<64xf32>
      }
      %82 = arith.muli %73, %c2048_i32 : i32
      %83 = memref.get_global @wq : memref<8192xf32>
      %84 = arith.muli %73, %c4096_i32 : i32
      %85 = arith.index_cast %84 : i32 to index
      arts.edt <parallel> <internode> route(%c0_i32) attributes {no_verify = #arts.no_verify} {
        arts.for(%c0) to(%c64) step(%c1) {{
        ^bb0(%arg1: index):
          %102 = arith.index_cast %arg1 : index to i32
          %103 = scf.for %arg2 = %c0 to %c64 step %c1 iter_args(%arg3 = %cst_10) -> (f32) {
            %104 = arith.index_cast %arg2 : index to i32
            %105 = arith.muli %102, %c64_i32 : i32
            %106 = arith.addi %105, %104 : i32
            %107 = arith.index_cast %106 : i32 to index
            %108 = arith.addi %107, %85 : index
            %109 = memref.load %83[%108] : memref<8192xf32>
            %110 = memref.load %7[%arg2] : memref<64xf32>
            %111 = arith.mulf %109, %110 : f32
            %112 = arith.addf %arg3, %111 : f32
            scf.yield %112 : f32
          }
          memref.store %103, %11[%arg1] : memref<64xf32>
        }}
      }
      %86 = arith.index_cast %82 : i32 to index
      %87 = polygeist.subindex %14[%86] () : memref<4096xf32> -> memref<?xf32>
      %88 = memref.get_global @wk : memref<8192xf32>
      arts.edt <parallel> <internode> route(%c0_i32) attributes {no_verify = #arts.no_verify} {
        arts.for(%c0) to(%c64) step(%c1) {{
        ^bb0(%arg1: index):
          %102 = arith.index_cast %arg1 : index to i32
          %103 = scf.for %arg2 = %c0 to %c64 step %c1 iter_args(%arg3 = %cst_10) -> (f32) {
            %105 = arith.index_cast %arg2 : index to i32
            %106 = arith.muli %102, %c64_i32 : i32
            %107 = arith.addi %106, %105 : i32
            %108 = arith.index_cast %107 : i32 to index
            %109 = arith.addi %108, %85 : index
            %110 = memref.load %88[%109] : memref<8192xf32>
            %111 = memref.load %7[%arg2] : memref<64xf32>
            %112 = arith.mulf %110, %111 : f32
            %113 = arith.addf %arg3, %112 : f32
            scf.yield %113 : f32
          }
          %104 = arith.addi %arg1, %86 : index
          memref.store %103, %14[%104] : memref<4096xf32>
        }}
      }
      %89 = memref.get_global @wv : memref<8192xf32>
      arts.edt <parallel> <internode> route(%c0_i32) attributes {no_verify = #arts.no_verify} {
        arts.for(%c0) to(%c64) step(%c1) {{
        ^bb0(%arg1: index):
          %102 = arith.index_cast %arg1 : index to i32
          %103 = scf.for %arg2 = %c0 to %c64 step %c1 iter_args(%arg3 = %cst_10) -> (f32) {
            %105 = arith.index_cast %arg2 : index to i32
            %106 = arith.muli %102, %c64_i32 : i32
            %107 = arith.addi %106, %105 : i32
            %108 = arith.index_cast %107 : i32 to index
            %109 = arith.addi %108, %85 : index
            %110 = memref.load %89[%109] : memref<8192xf32>
            %111 = memref.load %7[%arg2] : memref<64xf32>
            %112 = arith.mulf %110, %111 : f32
            %113 = arith.addf %arg3, %112 : f32
            scf.yield %113 : f32
          }
          %104 = arith.addi %arg1, %86 : index
          memref.store %103, %15[%104] : memref<4096xf32>
        }}
      }
      scf.for %arg1 = %c0 to %c64 step %c2 {
        %102 = arith.index_cast %arg1 : index to i32
        %103 = arith.remsi %102, %c16_i32 : i32
        %104 = arith.sitofp %103 : i32 to f32
        %105 = arith.divf %104, %cst_0 : f32
        %106 = math.powf %cst, %105 : f32
        %107 = arith.divf %cst_15, %106 : f32
        %108 = arith.mulf %107, %cst_10 : f32
        %109 = func.call @cosf(%108) : (f32) -> f32
        %110 = func.call @sinf(%108) : (f32) -> f32
        scf.for %arg2 = %c0 to %c2 step %c1 {
          %111 = arith.index_cast %arg2 : index to i32
          %112 = arith.cmpi eq, %111, %c0_i32 : i32
          %113 = arith.select %112, %cast, %87 : memref<?xf32>
          %114 = scf.if %112 -> (f32) {
            %124 = memref.load %11[%arg1] : memref<64xf32>
            scf.yield %124 : f32
          } else {
            %124 = arith.addi %arg1, %86 : index
            %125 = memref.load %14[%124] : memref<4096xf32>
            scf.yield %125 : f32
          }
          %115 = arith.addi %102, %c1_i32 : i32
          %116 = arith.index_cast %115 : i32 to index
          %117 = scf.if %112 -> (f32) {
            %124 = memref.load %11[%116] : memref<64xf32>
            scf.yield %124 : f32
          } else {
            %124 = arith.addi %116, %86 : index
            %125 = memref.load %14[%124] : memref<4096xf32>
            scf.yield %125 : f32
          }
          %118 = arith.mulf %114, %109 : f32
          %119 = arith.mulf %117, %110 : f32
          %120 = arith.subf %118, %119 : f32
          memref.store %120, %113[%arg1] : memref<?xf32>
          %121 = arith.mulf %114, %110 : f32
          %122 = arith.mulf %117, %109 : f32
          %123 = arith.addf %121, %122 : f32
          memref.store %123, %113[%116] : memref<?xf32>
        }
      }
      arts.edt <parallel> <internode> route(%c0_i32) attributes {no_verify = #arts.no_verify} {
        arts.for(%c0) to(%c4) step(%c1) {{
        ^bb0(%arg1: index):
          %102 = arith.index_cast %arg1 : index to i32
          %103 = arith.muli %102, %c16_i32 : i32
          %104 = arith.index_cast %103 : i32 to index
          %105 = arith.muli %102, %c32_i32 : i32
          %106 = arith.index_cast %105 : i32 to index
          %107 = arith.addi %82, %103 : i32
          %108 = arith.index_cast %107 : i32 to index
          %109 = scf.for %arg2 = %c0 to %c16 step %c1 iter_args(%arg3 = %cst_10) -> (f32) {
            %118 = arith.addi %arg2, %104 : index
            %119 = memref.load %11[%118] : memref<64xf32>
            %120 = arith.addi %arg2, %108 : index
            %121 = memref.load %14[%120] : memref<4096xf32>
            %122 = arith.mulf %119, %121 : f32
            %123 = arith.addf %arg3, %122 : f32
            scf.yield %123 : f32
          }
          %110 = arith.divf %109, %cst_6 : f32
          memref.store %110, %12[%106] : memref<128xf32>
          %111 = memref.load %12[%106] : memref<128xf32>
          %112 = arith.subf %111, %111 : f32
          %113 = math.exp %112 : f32
          memref.store %113, %12[%106] : memref<128xf32>
          %114 = memref.load %12[%106] : memref<128xf32>
          %115 = arith.addf %114, %cst_10 : f32
          %116 = arith.divf %114, %115 : f32
          memref.store %116, %12[%106] : memref<128xf32>
          scf.for %arg2 = %c0 to %c16 step %c1 {
            %118 = arith.addi %arg2, %104 : index
            memref.store %cst_10, %7[%118] : memref<64xf32>
          }
          %117 = memref.load %12[%106] : memref<128xf32>
          scf.for %arg2 = %c0 to %c16 step %c1 {
            %118 = arith.addi %arg2, %108 : index
            %119 = memref.load %15[%118] : memref<4096xf32>
            %120 = arith.mulf %117, %119 : f32
            %121 = arith.addi %arg2, %104 : index
            %122 = memref.load %7[%121] : memref<64xf32>
            %123 = arith.addf %122, %120 : f32
            memref.store %123, %7[%121] : memref<64xf32>
          }
        }}
      }
      %90 = memref.get_global @wo : memref<8192xf32>
      arts.edt <parallel> <internode> route(%c0_i32) attributes {no_verify = #arts.no_verify} {
        arts.for(%c0) to(%c64) step(%c1) {{
        ^bb0(%arg1: index):
          %102 = arith.index_cast %arg1 : index to i32
          %103 = scf.for %arg2 = %c0 to %c64 step %c1 iter_args(%arg3 = %cst_10) -> (f32) {
            %104 = arith.index_cast %arg2 : index to i32
            %105 = arith.muli %102, %c64_i32 : i32
            %106 = arith.addi %105, %104 : i32
            %107 = arith.index_cast %106 : i32 to index
            %108 = arith.addi %107, %85 : index
            %109 = memref.load %90[%108] : memref<8192xf32>
            %110 = memref.load %7[%arg2] : memref<64xf32>
            %111 = arith.mulf %109, %110 : f32
            %112 = arith.addf %arg3, %111 : f32
            scf.yield %112 : f32
          }
          memref.store %103, %8[%arg1] : memref<64xf32>
        }}
      }
      scf.for %arg1 = %c0 to %c64 step %c1 {
        %102 = memref.load %8[%arg1] : memref<64xf32>
        %103 = memref.load %6[%arg1] : memref<64xf32>
        %104 = arith.addf %103, %102 : f32
        memref.store %104, %6[%arg1] : memref<64xf32>
      }
      %91 = memref.get_global @rms_ffn_weight : memref<128xf32>
      %92 = scf.for %arg1 = %c0 to %c64 step %c1 iter_args(%arg2 = %cst_10) -> (f32) {
        %102 = memref.load %6[%arg1] : memref<64xf32>
        %103 = arith.mulf %102, %102 : f32
        %104 = arith.addf %arg2, %103 : f32
        scf.yield %104 : f32
      }
      %93 = arith.divf %92, %cst_1 : f32
      %94 = arith.addf %93, %cst_7 : f32
      %95 = math.sqrt %94 : f32
      %96 = arith.divf %cst_15, %95 : f32
      scf.for %arg1 = %c0 to %c64 step %c1 {
        %102 = arith.addi %arg1, %76 : index
        %103 = memref.load %91[%102] : memref<128xf32>
        %104 = memref.load %6[%arg1] : memref<64xf32>
        %105 = arith.mulf %96, %104 : f32
        %106 = arith.mulf %103, %105 : f32
        memref.store %106, %7[%arg1] : memref<64xf32>
      }
      %97 = memref.get_global @w1 : memref<32768xf32>
      %98 = arith.muli %73, %c16384_i32 : i32
      %99 = arith.index_cast %98 : i32 to index
      arts.edt <parallel> <internode> route(%c0_i32) attributes {no_verify = #arts.no_verify} {
        arts.for(%c0) to(%c256) step(%c1) {{
        ^bb0(%arg1: index):
          %102 = arith.index_cast %arg1 : index to i32
          %103 = scf.for %arg2 = %c0 to %c64 step %c1 iter_args(%arg3 = %cst_10) -> (f32) {
            %104 = arith.index_cast %arg2 : index to i32
            %105 = arith.muli %102, %c64_i32 : i32
            %106 = arith.addi %105, %104 : i32
            %107 = arith.index_cast %106 : i32 to index
            %108 = arith.addi %107, %99 : index
            %109 = memref.load %97[%108] : memref<32768xf32>
            %110 = memref.load %7[%arg2] : memref<64xf32>
            %111 = arith.mulf %109, %110 : f32
            %112 = arith.addf %arg3, %111 : f32
            scf.yield %112 : f32
          }
          memref.store %103, %9[%arg1] : memref<256xf32>
        }}
      }
      %100 = memref.get_global @w3 : memref<32768xf32>
      arts.edt <parallel> <internode> route(%c0_i32) attributes {no_verify = #arts.no_verify} {
        arts.for(%c0) to(%c256) step(%c1) {{
        ^bb0(%arg1: index):
          %102 = arith.index_cast %arg1 : index to i32
          %103 = scf.for %arg2 = %c0 to %c64 step %c1 iter_args(%arg3 = %cst_10) -> (f32) {
            %104 = arith.index_cast %arg2 : index to i32
            %105 = arith.muli %102, %c64_i32 : i32
            %106 = arith.addi %105, %104 : i32
            %107 = arith.index_cast %106 : i32 to index
            %108 = arith.addi %107, %99 : index
            %109 = memref.load %100[%108] : memref<32768xf32>
            %110 = memref.load %7[%arg2] : memref<64xf32>
            %111 = arith.mulf %109, %110 : f32
            %112 = arith.addf %arg3, %111 : f32
            scf.yield %112 : f32
          }
          memref.store %103, %10[%arg1] : memref<256xf32>
        }}
      }
      scf.for %arg1 = %c0 to %c256 step %c1 {
        %102 = memref.load %9[%arg1] : memref<256xf32>
        %103 = arith.negf %102 : f32
        %104 = math.exp %103 : f32
        %105 = arith.addf %104, %cst_15 : f32
        %106 = arith.divf %cst_15, %105 : f32
        %107 = arith.mulf %102, %106 : f32
        %108 = memref.load %10[%arg1] : memref<256xf32>
        %109 = arith.mulf %107, %108 : f32
        memref.store %109, %9[%arg1] : memref<256xf32>
      }
      %101 = memref.get_global @w2 : memref<32768xf32>
      arts.edt <parallel> <internode> route(%c0_i32) attributes {no_verify = #arts.no_verify} {
        arts.for(%c0) to(%c64) step(%c1) {{
        ^bb0(%arg1: index):
          %102 = arith.index_cast %arg1 : index to i32
          %103 = scf.for %arg2 = %c0 to %c256 step %c1 iter_args(%arg3 = %cst_10) -> (f32) {
            %104 = arith.index_cast %arg2 : index to i32
            %105 = arith.muli %102, %c256_i32 : i32
            %106 = arith.addi %105, %104 : i32
            %107 = arith.index_cast %106 : i32 to index
            %108 = arith.addi %107, %99 : index
            %109 = memref.load %101[%108] : memref<32768xf32>
            %110 = memref.load %9[%arg2] : memref<256xf32>
            %111 = arith.mulf %109, %110 : f32
            %112 = arith.addf %arg3, %111 : f32
            scf.yield %112 : f32
          }
          memref.store %103, %7[%arg1] : memref<64xf32>
        }}
      }
      scf.for %arg1 = %c0 to %c64 step %c1 {
        %102 = memref.load %7[%arg1] : memref<64xf32>
        %103 = memref.load %6[%arg1] : memref<64xf32>
        %104 = arith.addf %103, %102 : f32
        memref.store %104, %6[%arg1] : memref<64xf32>
      }
    }
    %19 = memref.get_global @rms_final_weight : memref<64xf32>
    %20 = scf.for %arg0 = %c0 to %c64 step %c1 iter_args(%arg1 = %cst_10) -> (f32) {
      %73 = memref.load %6[%arg0] : memref<64xf32>
      %74 = arith.mulf %73, %73 : f32
      %75 = arith.addf %arg1, %74 : f32
      scf.yield %75 : f32
    }
    %21 = arith.divf %20, %cst_1 : f32
    %22 = arith.addf %21, %cst_7 : f32
    %23 = math.sqrt %22 : f32
    %24 = arith.divf %cst_15, %23 : f32
    scf.for %arg0 = %c0 to %c64 step %c1 {
      %73 = memref.load %19[%arg0] : memref<64xf32>
      %74 = memref.load %6[%arg0] : memref<64xf32>
      %75 = arith.mulf %24, %74 : f32
      %76 = arith.mulf %73, %75 : f32
      memref.store %76, %6[%arg0] : memref<64xf32>
    }
    %25 = memref.get_global @token_embedding_table : memref<16384xf32>
    arts.edt <parallel> <internode> route(%c0_i32) attributes {no_verify = #arts.no_verify} {
      arts.for(%c0) to(%c256) step(%c1) {{
      ^bb0(%arg0: index):
        %73 = arith.index_cast %arg0 : index to i32
        %74 = scf.for %arg1 = %c0 to %c64 step %c1 iter_args(%arg2 = %cst_10) -> (f32) {
          %75 = arith.index_cast %arg1 : index to i32
          %76 = arith.muli %73, %c64_i32 : i32
          %77 = arith.addi %76, %75 : i32
          %78 = arith.index_cast %77 : i32 to index
          %79 = memref.load %25[%78] : memref<16384xf32>
          %80 = memref.load %6[%arg1] : memref<64xf32>
          %81 = arith.mulf %79, %80 : f32
          %82 = arith.addf %arg2, %81 : f32
          scf.yield %82 : f32
        }
        memref.store %74, %13[%arg0] : memref<256xf32>
      }}
    }
    %26 = llvm.mlir.addressof @str3 : !llvm.ptr
    %27 = llvm.getelementptr %26[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<42 x i8>
    %28 = llvm.call @printf(%27) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    scf.for %arg0 = %c0 to %c10 step %c1 {
      %73 = llvm.mlir.addressof @str4 : !llvm.ptr
      %74 = llvm.getelementptr %73[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<6 x i8>
      %75 = memref.load %13[%arg0] : memref<256xf32>
      %76 = arith.extf %75 : f32 to f64
      %77 = llvm.call @printf(%74, %76) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    }
    %29 = llvm.mlir.addressof @str5 : !llvm.ptr
    %30 = llvm.getelementptr %29[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i8>
    %31 = llvm.call @printf(%30) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %32 = llvm.mlir.addressof @str6 : !llvm.ptr
    %33 = llvm.getelementptr %32[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<34 x i8>
    %34 = llvm.call @printf(%33) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    memref.store %cst_15, %alloca_19[%c0] : memref<4xf32>
    memref.store %cst_14, %alloca_19[%c1] : memref<4xf32>
    memref.store %cst_13, %alloca_19[%c2] : memref<4xf32>
    memref.store %cst_6, %alloca_19[%c3] : memref<4xf32>
    %35 = scf.for %arg0 = %c0 to %c4 step %c1 iter_args(%arg1 = %cst_10) -> (f32) {
      %73 = memref.load %alloca_19[%arg0] : memref<4xf32>
      %74 = arith.mulf %73, %73 : f32
      %75 = arith.addf %arg1, %74 : f32
      scf.yield %75 : f32
    }
    %36 = arith.divf %35, %cst_6 : f32
    %37 = arith.addf %36, %cst_7 : f32
    %38 = math.sqrt %37 : f32
    %39 = arith.divf %cst_15, %38 : f32
    %40 = llvm.mlir.addressof @str7 : !llvm.ptr
    %41 = llvm.getelementptr %40[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<68 x i8>
    %42 = arith.extf %39 : f32 to f64
    %43 = arith.mulf %39, %cst_14 : f32
    %44 = arith.extf %43 : f32 to f64
    %45 = arith.mulf %39, %cst_13 : f32
    %46 = arith.extf %45 : f32 to f64
    %47 = arith.mulf %39, %cst_6 : f32
    %48 = arith.extf %47 : f32 to f64
    %49 = llvm.call @printf(%41, %cst_5, %cst_4, %cst_3, %cst_2, %42, %44, %46, %48) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64, f64, f64, f64, f64, f64, f64) -> i32
    memref.store %cst_15, %alloca_18[%c0] : memref<4xf32>
    memref.store %cst_14, %alloca_18[%c1] : memref<4xf32>
    memref.store %cst_13, %alloca_18[%c2] : memref<4xf32>
    memref.store %cst_6, %alloca_18[%c3] : memref<4xf32>
    %50 = scf.for %arg0 = %c1 to %c4 step %c1 iter_args(%arg1 = %cst_15) -> (f32) {
      %73 = memref.load %alloca_18[%arg0] : memref<4xf32>
      %74 = arith.cmpf ogt, %73, %arg1 : f32
      %75 = scf.if %74 -> (f32) {
        %76 = memref.load %alloca_18[%arg0] : memref<4xf32>
        scf.yield %76 : f32
      } else {
        scf.yield %arg1 : f32
      }
      scf.yield %75 : f32
    }
    %51 = scf.for %arg0 = %c0 to %c4 step %c1 iter_args(%arg1 = %cst_10) -> (f32) {
      %73 = memref.load %alloca_18[%arg0] : memref<4xf32>
      %74 = arith.subf %73, %50 : f32
      %75 = math.exp %74 : f32
      memref.store %75, %alloca_18[%arg0] : memref<4xf32>
      %76 = memref.load %alloca_18[%arg0] : memref<4xf32>
      %77 = arith.addf %arg1, %76 : f32
      scf.yield %77 : f32
    }
    scf.for %arg0 = %c0 to %c4 step %c1 {
      %73 = memref.load %alloca_18[%arg0] : memref<4xf32>
      %74 = arith.divf %73, %51 : f32
      memref.store %74, %alloca_18[%arg0] : memref<4xf32>
    }
    %52 = llvm.mlir.addressof @str8 : !llvm.ptr
    %53 = llvm.getelementptr %52[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x i8>
    %54 = memref.load %alloca_18[%c0] : memref<4xf32>
    %55 = arith.extf %54 : f32 to f64
    %56 = memref.load %alloca_18[%c1] : memref<4xf32>
    %57 = arith.extf %56 : f32 to f64
    %58 = memref.load %alloca_18[%c2] : memref<4xf32>
    %59 = arith.extf %58 : f32 to f64
    %60 = memref.load %alloca_18[%c3] : memref<4xf32>
    %61 = arith.extf %60 : f32 to f64
    %62 = llvm.call @printf(%53, %55, %57, %59, %61) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64, f64, f64) -> i32
    memref.store %cst_15, %alloca_17[%c0] : memref<6xf32>
    memref.store %cst_14, %alloca_17[%c1] : memref<6xf32>
    memref.store %cst_13, %alloca_17[%c2] : memref<6xf32>
    memref.store %cst_6, %alloca_17[%c3] : memref<6xf32>
    memref.store %cst_12, %alloca_17[%c4] : memref<6xf32>
    memref.store %cst_11, %alloca_17[%c5] : memref<6xf32>
    memref.store %cst_15, %alloca_16[%c0] : memref<3xf32>
    memref.store %cst_15, %alloca_16[%c1] : memref<3xf32>
    memref.store %cst_15, %alloca_16[%c2] : memref<3xf32>
    arts.edt <parallel> <internode> route(%c0_i32) attributes {no_verify = #arts.no_verify} {
      arts.for(%c0) to(%c2) step(%c1) {{
      ^bb0(%arg0: index):
        %73 = arith.index_cast %arg0 : index to i32
        %74 = scf.for %arg1 = %c0 to %c3 step %c1 iter_args(%arg2 = %cst_10) -> (f32) {
          %75 = arith.index_cast %arg1 : index to i32
          %76 = arith.muli %73, %c3_i32 : i32
          %77 = arith.addi %76, %75 : i32
          %78 = arith.index_cast %77 : i32 to index
          %79 = memref.load %alloca_17[%78] : memref<6xf32>
          %80 = memref.load %alloca_16[%arg1] : memref<3xf32>
          %81 = arith.mulf %79, %80 : f32
          %82 = arith.addf %arg2, %81 : f32
          scf.yield %82 : f32
        }
        memref.store %74, %alloca[%arg0] : memref<2xf32>
      }}
    }
    %63 = llvm.mlir.addressof @str9 : !llvm.ptr
    %64 = llvm.getelementptr %63[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<54 x i8>
    %65 = memref.load %alloca[%c0] : memref<2xf32>
    %66 = arith.extf %65 : f32 to f64
    %67 = memref.load %alloca[%c1] : memref<2xf32>
    %68 = arith.extf %67 : f32 to f64
    %69 = llvm.call @printf(%64, %66, %68) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64) -> i32
    %70 = llvm.mlir.addressof @str10 : !llvm.ptr
    %71 = llvm.getelementptr %70[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<35 x i8>
    %72 = llvm.call @printf(%71) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    return %c0_i32 : i32
  }
  func.func private @srand(i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @rand() -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @cosf(f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @sinf(f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>}
}
