"builtin.module"() ({
  "func.func"() <{function_type = (memref<?xmemref<?xf32>>, i32, i32) -> (), sym_name = "sparselu_par_call"}> ({
  ^bb0(%arg0: memref<?xmemref<?xf32>>, %arg1: i32, %arg2: i32):
    %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %1 = "arith.constant"() <{value = 0 : index}> : () -> index
    %2 = "arith.constant"() <{value = 1 : index}> : () -> index
    %3 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    "arts.edt"(%0) <{concurrency = #arts.concurrency<internode>, type = #arts.edt_type<parallel>}> ({
      "arts.barrier"() : () -> ()
      "arts.edt"(%0) <{concurrency = #arts.concurrency<intranode>, type = #arts.edt_type<single>}> ({
        %4 = "arith.muli"(%arg2, %arg2) : (i32, i32) -> i32
        %5 = "arith.index_cast"(%4) : (i32) -> index
        %6 = "llvm.mlir.zero"() : () -> !llvm.ptr
        %7 = "arith.index_cast"(%arg1) : (i32) -> index
        "scf.for"(%1, %7, %2) ({
        ^bb0(%arg3: index):
          %8 = "arith.index_cast"(%arg3) : (index) -> i32
          %9 = "arith.muli"(%8, %arg1) : (i32, i32) -> i32
          %10 = "arith.addi"(%9, %8) : (i32, i32) -> i32
          %11 = "arith.index_cast"(%10) : (i32) -> index
          %12 = "memref.subview"(%arg0, %11, %5) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<?xmemref<?xf32>>, index, index) -> memref<?xmemref<?xf32>, strided<[1], offset: ?>>
          %13 = "memref.cast"(%12) : (memref<?xmemref<?xf32>, strided<[1], offset: ?>>) -> memref<?xmemref<?xf32>, strided<[?], offset: ?>>
          "omp.task"(%13) <{depends = [#omp<clause_task_depend(taskdependinout)>], operandSegmentSizes = array<i32: 0, 0, 0, 0, 1, 0, 0>}> ({
            %16 = "memref.load"(%arg0, %11) : (memref<?xmemref<?xf32>>, index) -> memref<?xf32>
            "func.call"(%16, %arg2) <{callee = @lu0}> : (memref<?xf32>, i32) -> ()
            "arts.yield"() : () -> ()
          }) : (memref<?xmemref<?xf32>, strided<[?], offset: ?>>) -> ()
          %14 = "arith.addi"(%8, %3) : (i32, i32) -> i32
          %15 = "arith.index_cast"(%14) : (i32) -> index
          "scf.for"(%15, %7, %2) ({
          ^bb0(%arg4: index):
            %16 = "arith.index_cast"(%arg4) : (index) -> i32
            %17 = "arith.addi"(%9, %16) : (i32, i32) -> i32
            %18 = "arith.index_cast"(%17) : (i32) -> index
            %19 = "memref.load"(%arg0, %18) : (memref<?xmemref<?xf32>>, index) -> memref<?xf32>
            %20 = "polygeist.memref2pointer"(%19) : (memref<?xf32>) -> !llvm.ptr
            %21 = "llvm.icmp"(%20, %6) <{predicate = 1 : i64}> : (!llvm.ptr, !llvm.ptr) -> i1
            "scf.if"(%21) ({
              %22 = "memref.subview"(%arg0, %18, %5) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<?xmemref<?xf32>>, index, index) -> memref<?xmemref<?xf32>, strided<[1], offset: ?>>
              %23 = "memref.cast"(%22) : (memref<?xmemref<?xf32>, strided<[1], offset: ?>>) -> memref<?xmemref<?xf32>, strided<[?], offset: ?>>
              "omp.task"(%13, %23) <{depends = [#omp<clause_task_depend(taskdependin)>, #omp<clause_task_depend(taskdependinout)>], operandSegmentSizes = array<i32: 0, 0, 0, 0, 2, 0, 0>}> ({
                %24 = "memref.load"(%arg0, %11) : (memref<?xmemref<?xf32>>, index) -> memref<?xf32>
                %25 = "memref.load"(%arg0, %18) : (memref<?xmemref<?xf32>>, index) -> memref<?xf32>
                "func.call"(%24, %25, %arg2) <{callee = @fwd}> : (memref<?xf32>, memref<?xf32>, i32) -> ()
                "arts.yield"() : () -> ()
              }) : (memref<?xmemref<?xf32>, strided<[?], offset: ?>>, memref<?xmemref<?xf32>, strided<[?], offset: ?>>) -> ()
              "scf.yield"() : () -> ()
            }, {
            }) : (i1) -> ()
            "scf.yield"() : () -> ()
          }) : (index, index, index) -> ()
          "scf.for"(%15, %7, %2) ({
          ^bb0(%arg4: index):
            %16 = "arith.index_cast"(%arg4) : (index) -> i32
            %17 = "arith.muli"(%16, %arg1) : (i32, i32) -> i32
            %18 = "arith.addi"(%17, %8) : (i32, i32) -> i32
            %19 = "arith.index_cast"(%18) : (i32) -> index
            %20 = "memref.load"(%arg0, %19) : (memref<?xmemref<?xf32>>, index) -> memref<?xf32>
            %21 = "polygeist.memref2pointer"(%20) : (memref<?xf32>) -> !llvm.ptr
            %22 = "llvm.icmp"(%21, %6) <{predicate = 1 : i64}> : (!llvm.ptr, !llvm.ptr) -> i1
            "scf.if"(%22) ({
              %23 = "memref.subview"(%arg0, %19, %5) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<?xmemref<?xf32>>, index, index) -> memref<?xmemref<?xf32>, strided<[1], offset: ?>>
              %24 = "memref.cast"(%23) : (memref<?xmemref<?xf32>, strided<[1], offset: ?>>) -> memref<?xmemref<?xf32>, strided<[?], offset: ?>>
              "omp.task"(%13, %24) <{depends = [#omp<clause_task_depend(taskdependin)>, #omp<clause_task_depend(taskdependinout)>], operandSegmentSizes = array<i32: 0, 0, 0, 0, 2, 0, 0>}> ({
                %25 = "memref.load"(%arg0, %11) : (memref<?xmemref<?xf32>>, index) -> memref<?xf32>
                %26 = "memref.load"(%arg0, %19) : (memref<?xmemref<?xf32>>, index) -> memref<?xf32>
                "func.call"(%25, %26, %arg2) <{callee = @bdiv}> : (memref<?xf32>, memref<?xf32>, i32) -> ()
                "arts.yield"() : () -> ()
              }) : (memref<?xmemref<?xf32>, strided<[?], offset: ?>>, memref<?xmemref<?xf32>, strided<[?], offset: ?>>) -> ()
              "scf.yield"() : () -> ()
            }, {
            }) : (i1) -> ()
            "scf.yield"() : () -> ()
          }) : (index, index, index) -> ()
          "scf.for"(%15, %7, %2) ({
          ^bb0(%arg4: index):
            %16 = "arith.index_cast"(%arg4) : (index) -> i32
            %17 = "arith.muli"(%16, %arg1) : (i32, i32) -> i32
            %18 = "arith.addi"(%17, %8) : (i32, i32) -> i32
            %19 = "arith.index_cast"(%18) : (i32) -> index
            %20 = "memref.load"(%arg0, %19) : (memref<?xmemref<?xf32>>, index) -> memref<?xf32>
            %21 = "polygeist.memref2pointer"(%20) : (memref<?xf32>) -> !llvm.ptr
            %22 = "llvm.icmp"(%21, %6) <{predicate = 1 : i64}> : (!llvm.ptr, !llvm.ptr) -> i1
            "scf.if"(%22) ({
              "scf.for"(%15, %7, %2) ({
              ^bb0(%arg5: index):
                %23 = "arith.index_cast"(%arg5) : (index) -> i32
                %24 = "arith.addi"(%9, %23) : (i32, i32) -> i32
                %25 = "arith.index_cast"(%24) : (i32) -> index
                %26 = "memref.load"(%arg0, %25) : (memref<?xmemref<?xf32>>, index) -> memref<?xf32>
                %27 = "polygeist.memref2pointer"(%26) : (memref<?xf32>) -> !llvm.ptr
                %28 = "llvm.icmp"(%27, %6) <{predicate = 1 : i64}> : (!llvm.ptr, !llvm.ptr) -> i1
                "scf.if"(%28) ({
                  %29 = "arith.addi"(%17, %23) : (i32, i32) -> i32
                  %30 = "arith.index_cast"(%29) : (i32) -> index
                  %31 = "memref.load"(%arg0, %30) : (memref<?xmemref<?xf32>>, index) -> memref<?xf32>
                  %32 = "polygeist.memref2pointer"(%31) : (memref<?xf32>) -> !llvm.ptr
                  %33 = "llvm.icmp"(%32, %6) <{predicate = 0 : i64}> : (!llvm.ptr, !llvm.ptr) -> i1
                  "scf.if"(%33) ({
                    %40 = "func.call"(%arg2) <{callee = @allocate_clean_block}> : (i32) -> memref<?xf32>
                    "memref.store"(%40, %arg0, %30) : (memref<?xf32>, memref<?xmemref<?xf32>>, index) -> ()
                    "scf.yield"() : () -> ()
                  }, {
                  }) : (i1) -> ()
                  %34 = "memref.subview"(%arg0, %19, %5) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<?xmemref<?xf32>>, index, index) -> memref<?xmemref<?xf32>, strided<[1], offset: ?>>
                  %35 = "memref.cast"(%34) : (memref<?xmemref<?xf32>, strided<[1], offset: ?>>) -> memref<?xmemref<?xf32>, strided<[?], offset: ?>>
                  %36 = "memref.subview"(%arg0, %25, %5) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<?xmemref<?xf32>>, index, index) -> memref<?xmemref<?xf32>, strided<[1], offset: ?>>
                  %37 = "memref.cast"(%36) : (memref<?xmemref<?xf32>, strided<[1], offset: ?>>) -> memref<?xmemref<?xf32>, strided<[?], offset: ?>>
                  %38 = "memref.subview"(%arg0, %30, %5) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<?xmemref<?xf32>>, index, index) -> memref<?xmemref<?xf32>, strided<[1], offset: ?>>
                  %39 = "memref.cast"(%38) : (memref<?xmemref<?xf32>, strided<[1], offset: ?>>) -> memref<?xmemref<?xf32>, strided<[?], offset: ?>>
                  "omp.task"(%35, %37, %39) <{depends = [#omp<clause_task_depend(taskdependin)>, #omp<clause_task_depend(taskdependin)>, #omp<clause_task_depend(taskdependinout)>], operandSegmentSizes = array<i32: 0, 0, 0, 0, 3, 0, 0>}> ({
                    %40 = "memref.load"(%arg0, %19) : (memref<?xmemref<?xf32>>, index) -> memref<?xf32>
                    %41 = "memref.load"(%arg0, %25) : (memref<?xmemref<?xf32>>, index) -> memref<?xf32>
                    %42 = "memref.load"(%arg0, %30) : (memref<?xmemref<?xf32>>, index) -> memref<?xf32>
                    "func.call"(%40, %41, %42, %arg2) <{callee = @bmod}> : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32) -> ()
                    "arts.yield"() : () -> ()
                  }) : (memref<?xmemref<?xf32>, strided<[?], offset: ?>>, memref<?xmemref<?xf32>, strided<[?], offset: ?>>, memref<?xmemref<?xf32>, strided<[?], offset: ?>>) -> ()
                  "scf.yield"() : () -> ()
                }, {
                }) : (i1) -> ()
                "scf.yield"() : () -> ()
              }) : (index, index, index) -> ()
              "scf.yield"() : () -> ()
            }, {
            }) : (i1) -> ()
            "scf.yield"() : () -> ()
          }) : (index, index, index) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "arts.barrier"() : () -> ()
        "arts.yield"() : () -> ()
      }) : (i32) -> ()
      "arts.barrier"() : () -> ()
      "arts.yield"() : () -> ()
    }) : (i32) -> ()
    "func.return"() : () -> ()
  }) {llvm.linkage = #llvm.linkage<external>} : () -> ()
  "func.func"() <{function_type = (memref<?xf32>, i32) -> (), sym_name = "lu0", sym_visibility = "private"}> ({
  }) {llvm.linkage = #llvm.linkage<external>} : () -> ()
  "func.func"() <{function_type = (memref<?xf32>, memref<?xf32>, i32) -> (), sym_name = "fwd", sym_visibility = "private"}> ({
  }) {llvm.linkage = #llvm.linkage<external>} : () -> ()
  "func.func"() <{function_type = (memref<?xf32>, memref<?xf32>, i32) -> (), sym_name = "bdiv", sym_visibility = "private"}> ({
  }) {llvm.linkage = #llvm.linkage<external>} : () -> ()
  "func.func"() <{function_type = (i32) -> memref<?xf32>, sym_name = "allocate_clean_block", sym_visibility = "private"}> ({
  }) {llvm.linkage = #llvm.linkage<external>} : () -> ()
  "func.func"() <{function_type = (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32) -> (), sym_name = "bmod", sym_visibility = "private"}> ({
  }) {llvm.linkage = #llvm.linkage<external>} : () -> ()
}) {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:o-i64:64-i128:128-n32:64-S128", llvm.target_triple = "arm64-apple-macosx15.0.0", "polygeist.target-cpu" = "apple-m1", "polygeist.target-features" = "+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz"} : () -> ()
