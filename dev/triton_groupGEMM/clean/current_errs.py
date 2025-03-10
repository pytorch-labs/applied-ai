"""
TMA benchmarks will be running with experimental grid constant TMA descriptor.
Running test_backward_pass
Test setup - G: 2, M: 256, N: 256, K: 128
Input x shape: torch.Size([256, 128])
2025-03-09 21:33:59,073 - INFO - Weight w shape: torch.Size([512, 128])
2025-03-09 21:33:59,123 - INFO - Group sizes: tensor([128, 128], device='cuda:0', dtype=torch.int32)
2025-03-09 21:33:59,123 - INFO - Running forward pass
2025-03-09 21:33:59,602 - INFO - Forward result shape: torch.Size([256, 512])
2025-03-09 21:33:59,602 - INFO - Created gradient with shape: torch.Size([256, 512])
2025-03-09 21:33:59,602 - INFO - Running backward pass directly
2025-03-09 21:33:59,602 - INFO - Starting grouped_gemm_backward with fixed configurations
2025-03-09 21:33:59,603 - INFO - Group count: 2
2025-03-09 21:33:59,603 - INFO - Input shapes - x: torch.Size([256, 128]), w: torch.Size([512, 128]), grad_output: torch.Size([256, 512])
2025-03-09 21:33:59,603 - INFO - N per group: 256
2025-03-09 21:33:59,612 - INFO - M_bucket: 256, NUM_SMS: 132
2025-03-09 21:33:59,612 - INFO - Computing grad_x with triton kernel
2025-03-09 21:33:59,682 - INFO - grad_x computation successful with triton
2025-03-09 21:33:59,682 - INFO - Computing grad_w with triton kernel
2025-03-09 21:33:59,695 - ERROR - Error in backward_w kernel: at 117:35:
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
2025-03-09 21:33:59,695 - INFO - Falling back to PyTorch for grad_w
2025-03-09 21:34:00,505 - ERROR - TMA descriptor setup failed: The expanded size of the tensor (128) must match the existing size (256) at non-singleton dimension 1.  Target sizes: [256, 128].  Tensor sizes: [128, 256]
2025-03-09 21:34:00,505 - INFO - Falling back to PyTorch implementation
2025-03-09 21:34:00,505 - INFO - Using PyTorch fallback for grouped GEMM backward
2025-03-09 21:34:00,508 - ERROR - Unexpected error in grouped_gemm_backward: The expanded size of the tensor (128) must match the existing size (256) at non-singleton dimension 1.  Target sizes: [256, 128].  Tensor sizes: [128, 256]
2025-03-09 21:34:00,508 - INFO - Falling back to PyTorch implementation
2025-03-09 21:34:00,508 - INFO - Using PyTorch fallback for grouped GEMM backward
2025-03-09 21:34:00,508 - ERROR - Test failed with error: The expanded size of the tensor (128) must match the existing size (256) at non-singleton dimension 1.  Target sizes: [256, 128].  Tensor sizes: [128, 256]
2025-03-09 21:34:00,510 - ERROR - Traceback (most recent call last):
  File "/data/users/less/triton/python/triton/language/core.py", line 34, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/data/users/less/triton/python/triton/language/core.py", line 1814, in dot
    return semantic.dot(input, other, acc, input_precision, max_num_imprecise_acc, out_dtype, _builder)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/users/less/triton/python/triton/language/semantic.py", line 1566, in dot
    assert lhs.shape[-1].value == rhs.shape[
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: First input shape (['constexpr[64]', 'constexpr[32]']) and second input shape ['constexpr[64]', 'constexpr[32]'] are not compatible for matmul (second index of first shape (32) must be equal to first index of second shape (64)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/data/users/less/applied-ai/dev/triton_groupGEMM/clean/tgrouped_gemm_backwards.py", line 610, in grouped_gemm_backward
    _kernel_grouped_gemm_backward_w[grid](
  File "/data/users/less/triton/python/triton/runtime/jit.py", line 336, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/users/less/triton/python/triton/runtime/jit.py", line 563, in run
    kernel = self.compile(src, target=target, options=options.__dict__)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/users/less/triton/python/triton/compiler/compiler.py", line 278, in compile
    module = src.make_ir(options, codegen_fns, module_map, context)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/users/less/triton/python/triton/compiler/compiler.py", line 81, in make_ir
    return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
triton.compiler.errors.CompilationError: at 117:35:
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

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/data/users/less/applied-ai/dev/triton_groupGEMM/clean/tgrouped_gemm_backwards.py", line 628, in grouped_gemm_backward
    _compute_grad_w_pytorch(grad_output, x, m_sizes, grad_w)
  File "/data/users/less/applied-ai/dev/triton_groupGEMM/clean/tgrouped_gemm_backwards.py", line 374, in _compute_grad_w_pytorch
    grad_w[n_start:n_end] = (
    ~~~~~~^^^^^^^^^^^^^^^
RuntimeError: The expanded size of the tensor (128) must match the existing size (256) at non-singleton dimension 1.  Target sizes: [256, 128].  Tensor sizes: [128, 256]

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/data/users/less/applied-ai/dev/triton_groupGEMM/clean/tgrouped_gemm_backwards.py", line 633, in grouped_gemm_backward
    return _pytorch_fallback_backward(grad_output, x, w, m_sizes)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/users/less/applied-ai/dev/triton_groupGEMM/clean/tgrouped_gemm_backwards.py", line 409, in _pytorch_fallback_backward
    grad_w[n_start:n_end] = (
    ~~~~~~^^^^^^^^^^^^^^^
RuntimeError: The expanded size of the tensor (128) must match the existing size (256) at non-singleton dimension 1.  Target sizes: [256, 128].  Tensor sizes: [128, 256]

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/data/users/less/applied-ai/dev/triton_groupGEMM/clean/fast_debug.py", line 60, in test_backward_pass
    grad_x, grad_w = grouped_gemm_backward(grad_output, x, w, m_sizes)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/users/less/applied-ai/dev/triton_groupGEMM/clean/tgrouped_gemm_backwards.py", line 648, in grouped_gemm_backward
    return _pytorch_fallback_backward(grad_output, x, w, m_sizes)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/users/less/applied-ai/dev/triton_groupGEMM/clean/tgrouped_gemm_backwards.py", line 409, in _pytorch_fallback_backward
    grad_w[n_start:n_end] = (
    ~~~~~~^^^^^^^^^^^^^^^
RuntimeError: The expanded size of the tensor (128) must match the existing size (256) at non-singleton dimension 1.  Target sizes: [256, 128].  Tensor sizes: [128, 256]

2025-03-09 21:34:00,510 - INFO - Test failed
"""
