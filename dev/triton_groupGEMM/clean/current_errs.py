"""
Running test_backward_pass
Test setup - G: 1, M: 256, N: 256, K: 64
Input x shape: torch.Size([256, 64])
2025-03-10 13:04:44,458 - INFO - Weight w shape: torch.Size([256, 64])
2025-03-10 13:04:44,459 - INFO - Group sizes: tensor([256], device='cuda:0', dtype=torch.int32)
2025-03-10 13:04:44,460 - INFO - Running forward pass
2025-03-10 13:04:44,849 - INFO - Forward result shape: torch.Size([256, 256])
2025-03-10 13:04:44,850 - INFO - Created gradient with shape: torch.Size([256, 256])
2025-03-10 13:04:44,850 - INFO - Running backward pass directly
2025-03-10 13:04:44,850 - INFO - Starting grouped_gemm_backward with fixed configurations
2025-03-10 13:04:44,850 - INFO - Large computation detected: True
2025-03-10 13:04:44,850 - INFO - M_bucket: 256, NUM_SMS: 16
2025-03-10 13:04:44,850 - INFO - Computing grad_x with triton kernel
2025-03-10 13:04:44,862 - ERROR - Error in backward_x kernel: at 141:30:
                    )
                    """
                    grad_output_block = tl.load(
                        grad_output_ptr
                        + (m_start_offset + offs_m)[:, None] * (N * G)
                        + (n_start_offset + n_offset + offs_n)[None, :],
                        mask=m_mask[:, None] & n_mask[None, :],
                        other=0.0,
                    )

                    # Load weights block with explicit memory access pattern
                    w_block = tl.load(
                              ^
2025-03-10 13:04:44,862 - INFO - Falling back to PyTorch for grad_x
2025-03-10 13:04:44,914 - INFO - Computing grad_w with triton kernel
2025-03-10 13:04:45,279 - INFO - grad_w computation successful with triton
2025-03-10 13:04:45,369 - INFO - Gradients allclose check - grad_x: True, grad_w: True
2025-03-10 13:04:45,369 - INFO - ✓ Gradients match the PyTorch reference (allclose check passed)
2025-03-10 13:04:45,369 - INFO - Gradient shapes - grad_x: torch.Size([256, 64]), grad_w: torch.Size([256, 64])
2025-03-10 13:04:45,369 - INFO - Running PyTorch reference implementation
/home/less/.conda/envs/tritondev/lib/python3.12/site-packages/torch/autograd/graph.py:824: UserWarning: Attempting to run cuBLAS, but there was no current CUDA context! Attempting to set the primary context... (Triggered internally at /pytorch/aten/src/ATen/cuda/CublasHandlePool.cpp:181.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
2025-03-10 13:04:45,592 - INFO - Comparing gradients with PyTorch reference
2025-03-10 13:04:45,599 - INFO - Maximum gradient error - grad_x: 0.0009765625, grad_w: 1.9073486328125e-06
2025-03-10 13:04:45,599 - INFO - Gradients allclose check - grad_x: True, grad_w: True
2025-03-10 13:04:45,599 - INFO - ✓ Gradients match the PyTorch reference (allclose check passed)
2025-03-10 13:04:45,599 - INFO - Test succeeded
"""
