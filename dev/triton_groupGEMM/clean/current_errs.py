"""
Test setup - G: 2, M: 256, N: 256, K: 128
Input x shape: torch.Size([256, 128])
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
"""
