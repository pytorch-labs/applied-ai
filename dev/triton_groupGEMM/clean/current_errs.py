"""
Running test_backward_pass
Test setup - G: 2, M: 1024, N: 512, K: 128
Input x shape: torch.Size([1024, 128])
2025-03-10 15:44:52,120 - INFO - Weight w shape: torch.Size([1024, 128])
2025-03-10 15:44:52,121 - INFO - Group sizes: tensor([512, 512], device='cuda:0', dtype=torch.int32)
2025-03-10 15:44:52,121 - INFO - Running forward pass
2025-03-10 15:44:52,508 - INFO - Forward result shape: torch.Size([1024, 1024])
2025-03-10 15:44:52,508 - INFO - Created gradient with shape: torch.Size([1024, 1024])
2025-03-10 15:44:52,508 - INFO - Running backward pass directly
2025-03-10 15:44:52,508 - INFO - Starting grouped_gemm_backward with fixed configurations
2025-03-10 15:44:52,508 - INFO - Large computation detected: True
2025-03-10 15:44:52,508 - INFO - M_bucket: 1024, NUM_SMS: 16
2025-03-10 15:44:52,508 - INFO - Computing grad_x with triton kernel
/data/users/less/applied-ai/dev/triton_groupGEMM/clean/tgrouped_gemm_backwards.py:115:24: error: cp.async does not support transfers smaller than 4 bytes; calculated this as 2 bytes
                        grad_output_ptr
                       ^
/data/users/less/applied-ai/dev/triton_groupGEMM/clean/tgrouped_gemm_backwards.py:115:24: error: failed to legalize operation 'ttg.async_copy_global_to_local' that was explicitly marked illegal
                        grad_output_ptr
                       ^
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_kernel_grouped_gemm_backward_x(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c128_i32 = arith.constant 128 : i32
    %c384_i32 = arith.constant 384 : i32
    %cst = arith.constant dense<64> : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<64> : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %c-1_i32 = arith.constant -1 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked1>
    %c16_i32 = arith.constant 16 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<64x64xbf16, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c512_i32 = arith.constant 512 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_3 = arith.constant dense<128> : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cst_4 = arith.constant dense<1024> : tensor<64x1xi32, #blocked>
    %cst_5 = arith.constant dense<128> : tensor<64x1xi32, #blocked>
    %0 = tt.get_program_id x : i32
    %1:3 = scf.for %arg5 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg6 = %0, %arg7 = %c0_i32, %arg8 = %c0_i32) -> (i32, i32, i32)  : i32 {
      %2 = tt.addptr %arg4, %arg5 : !tt.ptr<i32>, i32
      %3 = tt.load %2 : !tt.ptr<i32>
      %4 = arith.cmpi sgt, %3, %c0_i32 : i32
      %5:2 = scf.if %4 -> (i32, i32) {
        %7 = arith.muli %arg5, %c512_i32 : i32
        %8 = arith.addi %3, %c63_i32 : i32
        %9 = arith.divsi %8, %c64_i32 : i32
        %10 = arith.muli %9, %c2_i32 : i32
        %11 = arith.addi %arg7, %10 : i32
        %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %14 = tt.splat %3 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %15 = tt.splat %arg8 : i32 -> tensor<64x1xi32, #blocked>
        %16 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>, #blocked>
        %17 = tt.expand_dims %13 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
        %18 = tt.expand_dims %12 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
        %19 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>, #blocked>
        %20 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>, #blocked>
        %21 = scf.while (%arg9 = %arg6) : (i32) -> i32 {
          %22 = arith.cmpi sge, %arg9, %arg7 : i32
          %23 = arith.cmpi slt, %arg9, %11 : i32
          %24 = arith.andi %22, %23 : i1
          scf.condition(%24) %arg9 : i32
        } do {
        ^bb0(%arg9: i32):
          %22 = arith.subi %arg9, %arg7 : i32
          %23 = arith.remsi %22, %9 : i32
          %24 = arith.divsi %22, %9 : i32
          %25 = arith.muli %23, %c64_i32 : i32
          %26 = tt.splat %25 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %27 = arith.addi %26, %12 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %28 = arith.muli %24, %c64_i32 : i32
          %29 = tt.splat %28 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
          %30 = arith.addi %29, %13 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
          %31 = arith.cmpi slt, %27, %14 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %32 = arith.cmpi slt, %30, %cst_3 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
          %33 = tt.expand_dims %31 {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi1, #blocked>
          %34 = tt.broadcast %33 : tensor<64x1xi1, #blocked> -> tensor<64x64xi1, #blocked>
          %35 = tt.expand_dims %27 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
          %36 = arith.addi %15, %35 : tensor<64x1xi32, #blocked>
          %37 = arith.muli %36, %cst_4 : tensor<64x1xi32, #blocked>
          %38 = tt.addptr %16, %37 : tensor<64x1x!tt.ptr<bf16>, #blocked>, tensor<64x1xi32, #blocked>
          %39 = tt.broadcast %38 : tensor<64x1x!tt.ptr<bf16>, #blocked> -> tensor<64x64x!tt.ptr<bf16>, #blocked>
          %40 = tt.expand_dims %32 {axis = 0 : i32} : tensor<64xi1, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi1, #blocked>
          %41 = tt.broadcast %40 : tensor<1x64xi1, #blocked> -> tensor<64x64xi1, #blocked>
          %42 = tt.expand_dims %30 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
          %43 = tt.broadcast %42 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
          %44 = ttg.local_alloc  : () -> !ttg.memdesc<2x64x64xbf16, #shared, #smem, mutable>
          %45 = ttg.local_alloc  : () -> !ttg.memdesc<2x64x64xbf16, #shared, #smem, mutable>
          %46 = arith.cmpi slt, %13, %cst_0 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
          %47 = arith.cmpi slt, %12, %cst : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %48 = tt.expand_dims %46 {axis = 0 : i32} : tensor<64xi1, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi1, #blocked>
          %49 = tt.broadcast %48 : tensor<1x64xi1, #blocked> -> tensor<64x64xi1, #blocked>
          %50 = arith.andi %34, %49 : tensor<64x64xi1, #blocked>
          %51 = tt.splat %7 : i32 -> tensor<1x64xi32, #blocked>
          %52 = arith.addi %51, %17 : tensor<1x64xi32, #blocked>
          %53 = tt.broadcast %52 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
          %54 = tt.addptr %39, %53 : tensor<64x64x!tt.ptr<bf16>, #blocked>, tensor<64x64xi32, #blocked>
          %55 = ttg.memdesc_subview %44[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x64x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable, 2x64x64>
          %56 = ttg.async_copy_global_to_local %54, %55 mask %50 other %cst_2 : tensor<64x64x!tt.ptr<bf16>, #blocked> -> <64x64xbf16, #shared, #smem, mutable, 2x64x64>
          %57 = ttg.async_commit_group %56
          %58 = tt.expand_dims %47 {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi1, #blocked>
          %59 = tt.broadcast %58 : tensor<64x1xi1, #blocked> -> tensor<64x64xi1, #blocked>
          %60 = arith.andi %59, %41 : tensor<64x64xi1, #blocked>
          %61 = tt.splat %7 : i32 -> tensor<64x1xi32, #blocked>
          %62 = arith.addi %61, %18 : tensor<64x1xi32, #blocked>
          %63 = arith.muli %62, %cst_5 : tensor<64x1xi32, #blocked>
          %64 = tt.addptr %19, %63 : tensor<64x1x!tt.ptr<bf16>, #blocked>, tensor<64x1xi32, #blocked>
          %65 = tt.broadcast %64 : tensor<64x1x!tt.ptr<bf16>, #blocked> -> tensor<64x64x!tt.ptr<bf16>, #blocked>
          %66 = tt.addptr %65, %43 : tensor<64x64x!tt.ptr<bf16>, #blocked>, tensor<64x64xi32, #blocked>
          %67 = ttg.memdesc_subview %45[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x64x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable, 2x64x64>
          %68 = ttg.async_copy_global_to_local %66, %67 mask %60 other %cst_2 : tensor<64x64x!tt.ptr<bf16>, #blocked> -> <64x64xbf16, #shared, #smem, mutable, 2x64x64>
          %69 = ttg.async_commit_group %68
          %70 = arith.addi %7, %c64_i32 : i32
          %71 = tt.splat %70 : i32 -> tensor<1x64xi32, #blocked>
          %72 = arith.addi %71, %17 : tensor<1x64xi32, #blocked>
          %73 = tt.broadcast %72 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
          %74 = tt.addptr %39, %73 : tensor<64x64x!tt.ptr<bf16>, #blocked>, tensor<64x64xi32, #blocked>
          %75 = ttg.memdesc_subview %44[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x64x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable, 2x64x64>
          %76 = ttg.async_copy_global_to_local %74, %75 mask %50 other %cst_2 : tensor<64x64x!tt.ptr<bf16>, #blocked> -> <64x64xbf16, #shared, #smem, mutable, 2x64x64>
          %77 = ttg.async_commit_group %76
          %78 = tt.splat %70 : i32 -> tensor<64x1xi32, #blocked>
          %79 = arith.addi %78, %18 : tensor<64x1xi32, #blocked>
          %80 = arith.muli %79, %cst_5 : tensor<64x1xi32, #blocked>
          %81 = tt.addptr %19, %80 : tensor<64x1x!tt.ptr<bf16>, #blocked>, tensor<64x1xi32, #blocked>
          %82 = tt.broadcast %81 : tensor<64x1x!tt.ptr<bf16>, #blocked> -> tensor<64x64x!tt.ptr<bf16>, #blocked>
          %83 = tt.addptr %82, %43 : tensor<64x64x!tt.ptr<bf16>, #blocked>, tensor<64x64xi32, #blocked>
          %84 = ttg.memdesc_subview %45[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x64x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable, 2x64x64>
          %85 = ttg.async_copy_global_to_local %83, %84 mask %60 other %cst_2 : tensor<64x64x!tt.ptr<bf16>, #blocked> -> <64x64xbf16, #shared, #smem, mutable, 2x64x64>
          %86 = ttg.async_commit_group %85
          %87:5 = scf.for %arg10 = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%arg11 = %cst_1, %arg12 = %c1_i32, %arg13 = %c-1_i32, %arg14 = %69, %arg15 = %86) -> (tensor<64x64xf32, #blocked1>, i32, i32, !ttg.async.token, !ttg.async.token)  : i32 {
            %97 = arith.cmpi slt, %arg10, %c384_i32 : i32
            %98 = arith.addi %arg13, %c1_i32 : i32
            %99 = arith.cmpi slt, %98, %c2_i32 : i32
            %100 = arith.select %99, %98, %c0_i32 : i32
            %101 = ttg.async_wait %arg14 {num = 2 : i32}
            %102 = ttg.memdesc_subview %44[%100, %c0_i32, %c0_i32] : !ttg.memdesc<2x64x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable, 2x64x64>
            %103 = ttg.local_load %102 token %101 : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable, 2x64x64> -> tensor<64x64xbf16, #blocked>
            %104 = ttg.memdesc_subview %45[%100, %c0_i32, %c0_i32] : !ttg.memdesc<2x64x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable, 2x64x64>
            %105 = ttg.local_load %104 token %101 : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable, 2x64x64> -> tensor<64x64xbf16, #blocked>
            %106 = arith.extf %103 : tensor<64x64xbf16, #blocked> to tensor<64x64xf32, #blocked>
            %107 = ttg.local_alloc %106 : (tensor<64x64xf32, #blocked>) -> !ttg.memdesc<64x64xf32, #shared, #smem>
            %108 = arith.extf %105 : tensor<64x64xbf16, #blocked> to tensor<64x64xf32, #blocked>
            %109 = ttg.local_alloc %108 : (tensor<64x64xf32, #blocked>) -> !ttg.memdesc<64x64xf32, #shared, #smem>
            %110 = ttg.local_load %107 : !ttg.memdesc<64x64xf32, #shared, #smem> -> tensor<64x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
            %111 = ttg.local_load %109 : !ttg.memdesc<64x64xf32, #shared, #smem> -> tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>
            %112 = tt.dot %110, %111, %arg11 : tensor<64x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<64x64xf32, #blocked1>
            %113 = arith.addi %arg12, %c1_i32 : i32
            %114 = arith.cmpi slt, %113, %c2_i32 : i32
            %115 = arith.select %114, %113, %c0_i32 : i32
            %116 = arith.subi %c384_i32, %arg10 : i32
            %117 = arith.minsi %116, %c64_i32 : i32
            %118 = tt.splat %117 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
            %119 = tt.splat %117 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
            %120 = arith.cmpi slt, %13, %118 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
            %121 = arith.cmpi slt, %12, %119 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
            %122 = tt.expand_dims %120 {axis = 0 : i32} : tensor<64xi1, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi1, #blocked>
            %123 = tt.broadcast %122 : tensor<1x64xi1, #blocked> -> tensor<64x64xi1, #blocked>
            %124 = arith.andi %34, %123 : tensor<64x64xi1, #blocked>
            %125 = arith.addi %arg10, %c128_i32 : i32
            %126 = arith.addi %7, %125 : i32
            %127 = tt.splat %126 : i32 -> tensor<1x64xi32, #blocked>
            %128 = arith.addi %127, %17 : tensor<1x64xi32, #blocked>
            %129 = tt.broadcast %128 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
            %130 = tt.addptr %39, %129 : tensor<64x64x!tt.ptr<bf16>, #blocked>, tensor<64x64xi32, #blocked>
            %131 = ttg.memdesc_subview %44[%115, %c0_i32, %c0_i32] : !ttg.memdesc<2x64x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable, 2x64x64>
            %132 = tt.splat %97 : i1 -> tensor<64x64xi1, #blocked>
            %133 = arith.andi %132, %124 : tensor<64x64xi1, #blocked>
            %134 = ttg.async_copy_global_to_local %130, %131 mask %133 other %cst_2 : tensor<64x64x!tt.ptr<bf16>, #blocked> -> <64x64xbf16, #shared, #smem, mutable, 2x64x64>
            %135 = ttg.async_commit_group %134
            %136 = tt.expand_dims %121 {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi1, #blocked>
            %137 = tt.broadcast %136 : tensor<64x1xi1, #blocked> -> tensor<64x64xi1, #blocked>
            %138 = arith.andi %137, %41 : tensor<64x64xi1, #blocked>
            %139 = tt.splat %126 : i32 -> tensor<64x1xi32, #blocked>
            %140 = arith.addi %139, %18 : tensor<64x1xi32, #blocked>
            %141 = arith.muli %140, %cst_5 : tensor<64x1xi32, #blocked>
            %142 = tt.addptr %19, %141 : tensor<64x1x!tt.ptr<bf16>, #blocked>, tensor<64x1xi32, #blocked>
            %143 = tt.broadcast %142 : tensor<64x1x!tt.ptr<bf16>, #blocked> -> tensor<64x64x!tt.ptr<bf16>, #blocked>
            %144 = tt.addptr %143, %43 : tensor<64x64x!tt.ptr<bf16>, #blocked>, tensor<64x64xi32, #blocked>
            %145 = ttg.memdesc_subview %45[%115, %c0_i32, %c0_i32] : !ttg.memdesc<2x64x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable, 2x64x64>
            %146 = arith.andi %132, %138 : tensor<64x64xi1, #blocked>
            %147 = ttg.async_copy_global_to_local %144, %145 mask %146 other %cst_2 : tensor<64x64x!tt.ptr<bf16>, #blocked> -> <64x64xbf16, #shared, #smem, mutable, 2x64x64>
            %148 = ttg.async_commit_group %147
            scf.yield %112, %115, %100, %arg15, %148 : tensor<64x64xf32, #blocked1>, i32, i32, !ttg.async.token, !ttg.async.token
          }
          %88 = ttg.async_wait  {num = 0 : i32}
          ttg.local_dealloc %45 : !ttg.memdesc<2x64x64xbf16, #shared, #smem, mutable>
          ttg.local_dealloc %44 : !ttg.memdesc<2x64x64xbf16, #shared, #smem, mutable>
          %89 = arith.andi %34, %41 : tensor<64x64xi1, #blocked>
          %90 = arith.muli %36, %cst_5 : tensor<64x1xi32, #blocked>
          %91 = tt.addptr %20, %90 : tensor<64x1x!tt.ptr<bf16>, #blocked>, tensor<64x1xi32, #blocked>
          %92 = tt.broadcast %91 : tensor<64x1x!tt.ptr<bf16>, #blocked> -> tensor<64x64x!tt.ptr<bf16>, #blocked>
          %93 = tt.addptr %92, %43 : tensor<64x64x!tt.ptr<bf16>, #blocked>, tensor<64x64xi32, #blocked>
          %94 = arith.truncf %87#0 : tensor<64x64xf32, #blocked1> to tensor<64x64xbf16, #blocked1>
          %95 = ttg.convert_layout %94 : tensor<64x64xbf16, #blocked1> -> tensor<64x64xbf16, #blocked>
          tt.store %93, %95, %89 : tensor<64x64x!tt.ptr<bf16>, #blocked>
          %96 = arith.addi %arg9, %c16_i32 : i32
          scf.yield %96 : i32
        }
        scf.yield %21, %11 : i32, i32
      } else {
        scf.yield %arg6, %arg7 : i32, i32
      }
      %6 = arith.addi %arg8, %3 : i32
      scf.yield %5#0, %5#1, %6 : i32, i32, i32
    }
    tt.return
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "builtin.module(triton-nvidia-mma-lowering, tritongpu-combine-tensor-select-and-if, tritongpu-allocate-warp-groups, convert-scf-to-cf, allocate-shared-memory, triton-tensor-memory-allocation, tritongpu-global-scratch-memory-allocation, convert-triton-gpu-to-llvm{compute-capability=90 ptx-version=84}, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, cse, convert-nv-gpu-to-llvm, convert-warp-specialize-to-llvm, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, cse, symbol-dce, enable-line-info)",
      disable_threading: false,
      verify_each: true
    }
  }
#-}
/data/users/less/applied-ai/dev/triton_groupGEMM/clean/tgrouped_gemm_backwards.py:22:0: error: Failures have been detected while processing an MLIR pass pipeline
/data/users/less/applied-ai/dev/triton_groupGEMM/clean/tgrouped_gemm_backwards.py:22:0: note: Pipeline failed while executing [`ConvertTritonGPUToLLVM` on 'builtin.module' operation]: reproducer generated at `std::errs, please share the reproducer above with Triton project.`
2025-03-10 15:44:52,594 - ERROR - Error in backward_x kernel: PassManager::run failed
2025-03-10 15:44:52,594 - INFO - Falling back to PyTorch for grad_x
2025-03-10 15:44:52,645 - INFO - Computing grad_w with triton kernel
2025-03-10 15:44:53,049 - INFO - grad_w computation successful with triton
2025-03-10 15:44:53,143 - INFO - Gradients allclose check - grad_x: True, grad_w: True
2025-03-10 15:44:53,143 - INFO - ✓ Gradients match the PyTorch reference (allclose check passed)
2025-03-10 15:44:53,143 - INFO - Gradient shapes - grad_x: torch.Size([1024, 128]), grad_w: torch.Size([1024, 128])
2025-03-10 15:44:53,143 - INFO - Running PyTorch reference implementation
2025-03-10 15:44:53,373 - INFO - Comparing gradients with PyTorch reference
2025-03-10 15:44:53,383 - INFO - Maximum gradient error - grad_x: 0.25, grad_w: 0.125
2025-03-10 15:44:53,383 - INFO - Gradients allclose check - grad_x: True, grad_w: True
2025-03-10 15:44:53,383 - INFO - ✓ Gradients match the PyTorch reference (allclose check passed)
2025-03-10 15:44:53,383 - INFO - Test succeeded
"""
