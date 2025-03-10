"""
2025-03-09 22:04:04,396 - INFO - Weight w shape: torch.Size([512, 128])
2025-03-09 22:04:04,471 - INFO - Group sizes: tensor([128, 128], device='cuda:0', dtype=torch.int32)
2025-03-09 22:04:04,471 - INFO - Running forward pass
2025-03-09 22:04:05,180 - INFO - Forward result shape: torch.Size([256, 512])
2025-03-09 22:04:05,181 - INFO - Created gradient with shape: torch.Size([256, 512])
2025-03-09 22:04:05,181 - INFO - Running backward pass directly
2025-03-09 22:04:05,181 - INFO - Starting grouped_gemm_backward with fixed configurations
2025-03-09 22:04:05,181 - INFO - Group count: 2
2025-03-09 22:04:05,181 - INFO - Input shapes - x: torch.Size([256, 128]), w: torch.Size([512, 128]), grad_output: torch.Size([256, 512])
2025-03-09 22:04:05,181 - INFO - N per group: 256
2025-03-09 22:04:05,189 - INFO - M_bucket: 256, NUM_SMS: 132
2025-03-09 22:04:05,189 - INFO - Computing grad_x with triton kernel
2025-03-09 22:04:05,430 - INFO - grad_x computation successful with triton
2025-03-09 22:04:05,430 - INFO - Computing grad_w with triton kernel
2025-03-09 22:04:05,441 - ERROR - Error in backward_w kernel: at 117:35:
                    # Load x_t [K, M] block
                    # Shape: [BLOCK_SIZE_N, BLOCK_SIZE_K]
                    x_t_block = tl.load(
                        x_t_ptr
                        + offs_n[:, None] * M_bucket
                        + (M_start_offset + k_offset + offs_k[None, :]),
                        mask=n_mask[:, None] & k_mask[None, :],
                        other=0.0,
                    )

                    # Matrix multiplication: (grad_y_block.T @ x_t_block)
                    accumulator += tl.dot(
                                   ^
2025-03-09 22:04:05,441 - INFO - Falling back to PyTorch for grad_w
2025-03-09 22:04:06,377 - INFO - Gradient shapes - grad_x: torch.Size([256, 128]), grad_w: torch.Size([512, 128])
2025-03-09 22:04:06,377 - INFO - Running PyTorch reference implementation
2025-03-09 22:04:06,870 - INFO - Comparing gradients with PyTorch reference
2025-03-09 22:04:06,973 - INFO - Maximum gradient error - grad_x: 292864.0, grad_w: 0.0
2025-03-09 22:04:06,973 - ERROR - ✗ Gradient mismatch above tolerance threshold
2025-03-09 22:04:06,973 - INFO - Test succeeded
(tritondev) [less@devgpu115.cco2 /data/users/less/applied-ai/dev/triton_groupGEMM/clean (lessw/gg_backward_pass)]$ python fast_debug.py
TMA benchmarks will be running with experimental grid constant TMA descriptor.
Running test_backward_pass
Test setup - G: 2, M: 256, N: 256, K: 128
Input x shape: torch.Size([256, 128])
2025-03-09 22:21:05,683 - INFO - Weight w shape: torch.Size([512, 128])
2025-03-09 22:21:05,689 - INFO - Group sizes: tensor([128, 128], device='cuda:0', dtype=torch.int32)
2025-03-09 22:21:05,689 - INFO - Running forward pass
2025-03-09 22:21:06,189 - INFO - Forward result shape: torch.Size([256, 512])
2025-03-09 22:21:06,190 - INFO - Created gradient with shape: torch.Size([256, 512])
2025-03-09 22:21:06,190 - INFO - Running backward pass directly
2025-03-09 22:21:06,190 - INFO - Starting grouped_gemm_backward with fixed configurations
2025-03-09 22:21:06,190 - INFO - Group count: 2
2025-03-09 22:21:06,190 - INFO - Input shapes - x: torch.Size([256, 128]), w: torch.Size([512, 128]), grad_output: torch.Size([256, 512])
2025-03-09 22:21:06,190 - INFO - N per group: 256
2025-03-09 22:21:06,190 - INFO - M_bucket: 256, NUM_SMS: 132
2025-03-09 22:21:06,190 - INFO - Computing grad_x with triton kernel
/data/users/less/applied-ai/dev/triton_groupGEMM/clean/tgrouped_gemm_backwards.py:132:24: error: cp.async does not support transfers smaller than 4 bytes; calculated this as 2 bytes
                        grad_output_ptr
                       ^
/data/users/less/applied-ai/dev/triton_groupGEMM/clean/tgrouped_gemm_backwards.py:132:24: error: failed to legalize operation 'ttg.async_copy_global_to_local' that was explicitly marked illegal
                        grad_output_ptr
                       ^
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 8]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_kernel_grouped_gemm_backward_x(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %c132_i32 = arith.constant 132 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c63_i32 = arith.constant 63 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x64xbf16, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_1 = arith.constant dense<128> : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cst_2 = arith.constant dense<128> : tensor<32x1xi32, #blocked>
    %cst_3 = arith.constant dense<128> : tensor<64x1xi32, #blocked>
    %c-1_i32 = arith.constant -1 : i32
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<64x32xbf16, #blocked1>
    %c192_i32 = arith.constant 192 : i32
    %cst_5 = arith.constant dense<32> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_6 = arith.constant dense<32> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %cst_7 = arith.constant dense<512> : tensor<64x1xi32, #blocked1>
    %0 = tt.get_program_id x : i32
    %1:3 = scf.for %arg5 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg6 = %c0_i32, %arg7 = %0, %arg8 = %c0_i32) -> (i32, i32, i32)  : i32 {
      %2 = tt.addptr %arg4, %arg5 : !tt.ptr<i32>, i32
      %3 = tt.load %2 : !tt.ptr<i32>
      %4 = arith.addi %arg6, %3 : i32
      %5 = arith.cmpi sgt, %3, %c0_i32 : i32
      %6:2 = scf.if %5 -> (i32, i32) {
        %7 = arith.muli %arg5, %c256_i32 : i32
        %8 = arith.addi %3, %c63_i32 : i32
        %9 = arith.divsi %8, %c64_i32 : i32
        %10 = arith.muli %9, %c2_i32 : i32
        %11 = arith.addi %arg8, %10 : i32
        %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %14 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %15 = tt.splat %3 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %16 = tt.splat %3 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %17 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
        %18 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %19 = tt.splat %arg6 : i32 -> tensor<64x1xi32, #blocked1>
        %20 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>, #blocked1>
        %21 = tt.expand_dims %17 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
        %22 = tt.expand_dims %18 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
        %23 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked>
        %24 = tt.splat %arg6 : i32 -> tensor<64x1xi32, #blocked>
        %25 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>, #blocked>
        %26 = scf.while (%arg9 = %arg7) : (i32) -> i32 {
          %27 = arith.cmpi sge, %arg9, %arg8 : i32
          %28 = arith.cmpi slt, %arg9, %11 : i32
          %29 = arith.andi %27, %28 : i1
          scf.condition(%29) %arg9 : i32
        } do {
        ^bb0(%arg9: i32):
          %27 = arith.subi %arg9, %arg8 : i32
          %28 = arith.remsi %27, %9 : i32
          %29 = arith.divsi %27, %9 : i32
          %30 = arith.muli %28, %c64_i32 : i32
          %31 = tt.splat %30 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
          %32 = tt.splat %30 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %33 = arith.addi %31, %12 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
          %34 = arith.addi %32, %13 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %35 = arith.muli %29, %c64_i32 : i32
          %36 = tt.splat %35 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
          %37 = arith.addi %36, %14 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
          %38 = arith.cmpi slt, %33, %15 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
          %39 = arith.cmpi slt, %34, %16 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %40 = arith.cmpi slt, %37, %cst_1 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
          %41 = tt.expand_dims %38 {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi1, #blocked1>
          %42 = tt.broadcast %41 : tensor<64x1xi1, #blocked1> -> tensor<64x32xi1, #blocked1>
          %43 = tt.expand_dims %33 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
          %44 = arith.addi %19, %43 : tensor<64x1xi32, #blocked1>
          %45 = arith.muli %44, %cst_7 : tensor<64x1xi32, #blocked1>
          %46 = tt.addptr %20, %45 : tensor<64x1x!tt.ptr<bf16>, #blocked1>, tensor<64x1xi32, #blocked1>
          %47 = tt.broadcast %46 : tensor<64x1x!tt.ptr<bf16>, #blocked1> -> tensor<64x32x!tt.ptr<bf16>, #blocked1>
          %48 = tt.expand_dims %40 {axis = 0 : i32} : tensor<64xi1, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi1, #blocked>
          %49 = tt.broadcast %48 : tensor<1x64xi1, #blocked> -> tensor<32x64xi1, #blocked>
          %50 = tt.expand_dims %37 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
          %51 = tt.broadcast %50 : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked>
          %52 = ttg.local_alloc  : () -> !ttg.memdesc<2x64x32xbf16, #shared, #smem, mutable>
          %53 = ttg.local_alloc  : () -> !ttg.memdesc<2x32x64xbf16, #shared1, #smem, mutable>
          %54 = arith.cmpi slt, %17, %cst_6 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
          %55 = arith.cmpi slt, %18, %cst_5 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %56 = tt.expand_dims %54 {axis = 0 : i32} : tensor<32xi1, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi1, #blocked1>
          %57 = tt.broadcast %56 : tensor<1x32xi1, #blocked1> -> tensor<64x32xi1, #blocked1>
          %58 = arith.andi %42, %57 : tensor<64x32xi1, #blocked1>
          %59 = tt.splat %7 : i32 -> tensor<1x32xi32, #blocked1>
          %60 = arith.addi %59, %21 : tensor<1x32xi32, #blocked1>
          %61 = tt.broadcast %60 : tensor<1x32xi32, #blocked1> -> tensor<64x32xi32, #blocked1>
          %62 = tt.addptr %47, %61 : tensor<64x32x!tt.ptr<bf16>, #blocked1>, tensor<64x32xi32, #blocked1>
          %63 = ttg.memdesc_subview %52[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x64x32xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xbf16, #shared, #smem, mutable, 2x64x32>
          %64 = ttg.async_copy_global_to_local %62, %63 mask %58 other %cst_4 : tensor<64x32x!tt.ptr<bf16>, #blocked1> -> <64x32xbf16, #shared, #smem, mutable, 2x64x32>
          %65 = ttg.async_commit_group %64
          %66 = tt.expand_dims %55 {axis = 1 : i32} : tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi1, #blocked>
          %67 = tt.broadcast %66 : tensor<32x1xi1, #blocked> -> tensor<32x64xi1, #blocked>
          %68 = arith.andi %67, %49 : tensor<32x64xi1, #blocked>
          %69 = tt.splat %7 : i32 -> tensor<32x1xi32, #blocked>
          %70 = arith.addi %69, %22 : tensor<32x1xi32, #blocked>
          %71 = arith.muli %70, %cst_2 : tensor<32x1xi32, #blocked>
          %72 = tt.addptr %23, %71 : tensor<32x1x!tt.ptr<bf16>, #blocked>, tensor<32x1xi32, #blocked>
          %73 = tt.broadcast %72 : tensor<32x1x!tt.ptr<bf16>, #blocked> -> tensor<32x64x!tt.ptr<bf16>, #blocked>
          %74 = tt.addptr %73, %51 : tensor<32x64x!tt.ptr<bf16>, #blocked>, tensor<32x64xi32, #blocked>
          %75 = ttg.memdesc_subview %53[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x32x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x64xbf16, #shared1, #smem, mutable, 2x32x64>
          %76 = ttg.async_copy_global_to_local %74, %75 mask %68 other %cst_0 : tensor<32x64x!tt.ptr<bf16>, #blocked> -> <32x64xbf16, #shared1, #smem, mutable, 2x32x64>
          %77 = ttg.async_commit_group %76
          %78 = arith.addi %7, %c32_i32 : i32
          %79 = tt.splat %78 : i32 -> tensor<1x32xi32, #blocked1>
          %80 = arith.addi %79, %21 : tensor<1x32xi32, #blocked1>
          %81 = tt.broadcast %80 : tensor<1x32xi32, #blocked1> -> tensor<64x32xi32, #blocked1>
          %82 = tt.addptr %47, %81 : tensor<64x32x!tt.ptr<bf16>, #blocked1>, tensor<64x32xi32, #blocked1>
          %83 = ttg.memdesc_subview %52[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x64x32xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xbf16, #shared, #smem, mutable, 2x64x32>
          %84 = ttg.async_copy_global_to_local %82, %83 mask %58 other %cst_4 : tensor<64x32x!tt.ptr<bf16>, #blocked1> -> <64x32xbf16, #shared, #smem, mutable, 2x64x32>
          %85 = ttg.async_commit_group %84
          %86 = tt.splat %78 : i32 -> tensor<32x1xi32, #blocked>
          %87 = arith.addi %86, %22 : tensor<32x1xi32, #blocked>
          %88 = arith.muli %87, %cst_2 : tensor<32x1xi32, #blocked>
          %89 = tt.addptr %23, %88 : tensor<32x1x!tt.ptr<bf16>, #blocked>, tensor<32x1xi32, #blocked>
          %90 = tt.broadcast %89 : tensor<32x1x!tt.ptr<bf16>, #blocked> -> tensor<32x64x!tt.ptr<bf16>, #blocked>
          %91 = tt.addptr %90, %51 : tensor<32x64x!tt.ptr<bf16>, #blocked>, tensor<32x64xi32, #blocked>
          %92 = ttg.memdesc_subview %53[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x32x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x64xbf16, #shared1, #smem, mutable, 2x32x64>
          %93 = ttg.async_copy_global_to_local %91, %92 mask %68 other %cst_0 : tensor<32x64x!tt.ptr<bf16>, #blocked> -> <32x64xbf16, #shared1, #smem, mutable, 2x32x64>
          %94 = ttg.async_commit_group %93
          %95:5 = scf.for %arg10 = %c0_i32 to %c256_i32 step %c32_i32 iter_args(%arg11 = %cst, %arg12 = %c1_i32, %arg13 = %c-1_i32, %arg14 = %77, %arg15 = %94) -> (tensor<64x64xf32, #mma>, i32, i32, !ttg.async.token, !ttg.async.token)  : i32 {
            %111 = arith.cmpi slt, %arg10, %c192_i32 : i32
            %112 = arith.addi %arg13, %c1_i32 : i32
            %113 = arith.cmpi slt, %112, %c2_i32 : i32
            %114 = arith.select %113, %112, %c0_i32 : i32
            %115 = ttg.async_wait %arg14 {num = 2 : i32}
            %116 = ttg.memdesc_subview %52[%114, %c0_i32, %c0_i32] : !ttg.memdesc<2x64x32xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xbf16, #shared, #smem, mutable, 2x64x32>
            %117 = ttg.local_load %116 : !ttg.memdesc<64x32xbf16, #shared, #smem, mutable, 2x64x32> -> tensor<64x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
            %118 = ttg.memdesc_subview %53[%114, %c0_i32, %c0_i32] : !ttg.memdesc<2x32x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x64xbf16, #shared1, #smem, mutable, 2x32x64>
            %119 = ttg.local_load %118 token %115 : !ttg.memdesc<32x64xbf16, #shared1, #smem, mutable, 2x32x64> -> tensor<32x64xbf16, #blocked>
            %120 = arith.extf %117 : tensor<64x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> to tensor<64x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
            %121 = arith.extf %119 : tensor<32x64xbf16, #blocked> to tensor<32x64xf32, #blocked>
            %122 = ttg.local_alloc %121 : (tensor<32x64xf32, #blocked>) -> !ttg.memdesc<32x64xf32, #shared2, #smem>
            ttng.fence_async_shared {bCluster = false}
            %123 = ttng.warp_group_dot %120, %122, %arg11 {inputPrecision = 0 : i32, isAsync = true} : tensor<64x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * !ttg.memdesc<32x64xf32, #shared2, #smem> -> tensor<64x64xf32, #mma>
            %124:2 = ttng.warp_group_dot_wait %123, %122 {pendings = 0 : i32} : tensor<64x64xf32, #mma>, !ttg.memdesc<32x64xf32, #shared2, #smem>
            %125 = arith.addi %arg12, %c1_i32 : i32
            %126 = arith.cmpi slt, %125, %c2_i32 : i32
            %127 = arith.select %126, %125, %c0_i32 : i32
            %128 = arith.subi %c192_i32, %arg10 : i32
            %129 = arith.minsi %128, %c32_i32 : i32
            %130 = tt.splat %129 : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
            %131 = tt.splat %129 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
            %132 = arith.cmpi slt, %17, %130 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
            %133 = arith.cmpi slt, %18, %131 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
            %134 = tt.expand_dims %132 {axis = 0 : i32} : tensor<32xi1, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi1, #blocked1>
            %135 = tt.broadcast %134 : tensor<1x32xi1, #blocked1> -> tensor<64x32xi1, #blocked1>
            %136 = arith.andi %42, %135 : tensor<64x32xi1, #blocked1>
            %137 = arith.addi %arg10, %c64_i32 : i32
            %138 = arith.addi %7, %137 : i32
            %139 = tt.splat %138 : i32 -> tensor<1x32xi32, #blocked1>
            %140 = arith.addi %139, %21 : tensor<1x32xi32, #blocked1>
            %141 = tt.broadcast %140 : tensor<1x32xi32, #blocked1> -> tensor<64x32xi32, #blocked1>
            %142 = tt.addptr %47, %141 : tensor<64x32x!tt.ptr<bf16>, #blocked1>, tensor<64x32xi32, #blocked1>
            %143 = ttg.memdesc_subview %52[%127, %c0_i32, %c0_i32] : !ttg.memdesc<2x64x32xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xbf16, #shared, #smem, mutable, 2x64x32>
            %144 = tt.splat %111 : i1 -> tensor<64x32xi1, #blocked1>
            %145 = arith.andi %144, %136 : tensor<64x32xi1, #blocked1>
            %146 = ttg.async_copy_global_to_local %142, %143 mask %145 other %cst_4 : tensor<64x32x!tt.ptr<bf16>, #blocked1> -> <64x32xbf16, #shared, #smem, mutable, 2x64x32>
            %147 = ttg.async_commit_group %146
            %148 = tt.expand_dims %133 {axis = 1 : i32} : tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi1, #blocked>
            %149 = tt.broadcast %148 : tensor<32x1xi1, #blocked> -> tensor<32x64xi1, #blocked>
            %150 = arith.andi %149, %49 : tensor<32x64xi1, #blocked>
            %151 = tt.splat %138 : i32 -> tensor<32x1xi32, #blocked>
            %152 = arith.addi %151, %22 : tensor<32x1xi32, #blocked>
            %153 = arith.muli %152, %cst_2 : tensor<32x1xi32, #blocked>
            %154 = tt.addptr %23, %153 : tensor<32x1x!tt.ptr<bf16>, #blocked>, tensor<32x1xi32, #blocked>
            %155 = tt.broadcast %154 : tensor<32x1x!tt.ptr<bf16>, #blocked> -> tensor<32x64x!tt.ptr<bf16>, #blocked>
            %156 = tt.addptr %155, %51 : tensor<32x64x!tt.ptr<bf16>, #blocked>, tensor<32x64xi32, #blocked>
            %157 = ttg.memdesc_subview %53[%127, %c0_i32, %c0_i32] : !ttg.memdesc<2x32x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x64xbf16, #shared1, #smem, mutable, 2x32x64>
            %158 = tt.splat %111 : i1 -> tensor<32x64xi1, #blocked>
            %159 = arith.andi %158, %150 : tensor<32x64xi1, #blocked>
            %160 = ttg.async_copy_global_to_local %156, %157 mask %159 other %cst_0 : tensor<32x64x!tt.ptr<bf16>, #blocked> -> <32x64xbf16, #shared1, #smem, mutable, 2x32x64>
            %161 = ttg.async_commit_group %160
            scf.yield %124#0, %127, %114, %arg15, %161 : tensor<64x64xf32, #mma>, i32, i32, !ttg.async.token, !ttg.async.token
          }
          %96 = ttg.async_wait  {num = 0 : i32}
          ttg.local_dealloc %53 : !ttg.memdesc<2x32x64xbf16, #shared1, #smem, mutable>
          ttg.local_dealloc %52 : !ttg.memdesc<2x64x32xbf16, #shared, #smem, mutable>
          %97 = tt.expand_dims %39 {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi1, #blocked>
          %98 = tt.broadcast %97 : tensor<64x1xi1, #blocked> -> tensor<64x64xi1, #blocked>
          %99 = tt.broadcast %48 : tensor<1x64xi1, #blocked> -> tensor<64x64xi1, #blocked>
          %100 = arith.andi %98, %99 : tensor<64x64xi1, #blocked>
          %101 = tt.expand_dims %34 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
          %102 = arith.addi %24, %101 : tensor<64x1xi32, #blocked>
          %103 = arith.muli %102, %cst_3 : tensor<64x1xi32, #blocked>
          %104 = tt.addptr %25, %103 : tensor<64x1x!tt.ptr<bf16>, #blocked>, tensor<64x1xi32, #blocked>
          %105 = tt.broadcast %104 : tensor<64x1x!tt.ptr<bf16>, #blocked> -> tensor<64x64x!tt.ptr<bf16>, #blocked>
          %106 = tt.broadcast %50 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
          %107 = tt.addptr %105, %106 : tensor<64x64x!tt.ptr<bf16>, #blocked>, tensor<64x64xi32, #blocked>
          %108 = arith.truncf %95#0 : tensor<64x64xf32, #mma> to tensor<64x64xbf16, #mma>
          %109 = ttg.convert_layout %108 : tensor<64x64xbf16, #mma> -> tensor<64x64xbf16, #blocked>
          tt.store %107, %109, %100 : tensor<64x64x!tt.ptr<bf16>, #blocked>
          %110 = arith.addi %arg9, %c132_i32 : i32
          scf.yield %110 : i32
        }
        scf.yield %26, %11 : i32, i32
      } else {
        scf.yield %arg7, %arg8 : i32, i32
      }
      scf.yield %4, %6#0, %6#1 : i32, i32, i32
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
2025-03-09 22:21:06,271 - ERROR - Error in backward_x kernel: PassManager::run failed
2025-03-09 22:21:06,271 - INFO - Falling back to PyTorch for grad_x
2025-03-09 22:21:06,477 - INFO - Computing grad_w with triton kernel
2025-03-09 22:21:06,867 - INFO - grad_w computation successful with triton
2025-03-09 22:21:06,867 - INFO - Gradient shapes - grad_x: torch.Size([256, 128]), grad_w: torch.Size([512, 128])
2025-03-09 22:21:06,867 - INFO - Running PyTorch reference implementation
2025-03-09 22:21:07,107 - INFO - Comparing gradients with PyTorch reference
2025-03-09 22:21:07,138 - INFO - Maximum gradient error - grad_x: 0.0, grad_w: 47.25
2025-03-09 22:21:07,138 - ERROR - ✗ Gradient mismatch above tolerance threshold
2025-03-09 22:21:07,138 - INFO - Test succeeded
"""
