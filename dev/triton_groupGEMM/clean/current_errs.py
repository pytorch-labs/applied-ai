"""
Test setup - G: 4, M: 2048, N: 2048, K: 1024
Input x shape: torch.Size([2048, 1024])
2025-03-12 17:42:00,819 - INFO - Weight w shape: torch.Size([8192, 1024])
2025-03-12 17:42:00,821 - INFO - Group sizes: tensor([512, 512, 512, 512], device='cuda:0', dtype=torch.int32)
2025-03-12 17:42:00,821 - INFO - Running forward pass
2025-03-12 17:42:01,248 - INFO - Forward result shape: torch.Size([2048, 8192])
2025-03-12 17:42:01,248 - INFO - Created gradient with shape: torch.Size([2048, 8192])
2025-03-12 17:42:01,248 - INFO - Running backward pass directly
2025-03-12 17:42:01,248 - INFO - Starting grouped_gemm_backward with TMA-enabled scheduling
2025-03-12 17:42:01,248 - INFO - TMA support detected on GPU with 132 SMs
2025-03-12 17:42:01,249 - INFO - EVEN_K optimization enabled: True (K=1024)
2025-03-12 17:42:01,258 - INFO - Computing grad_x with TMA-enabled kernel
2025-03-12 17:42:01,263 - INFO - SUCCESS!! grad_X computation successful with TMA-enabled kernel
2025-03-12 17:42:01,263 - INFO - Computing grad_w with TMA-enabled kernel
2025-03-12 17:42:01,267 - INFO - SUCCESS!! grad_W computation successful with TMA-enabled kernel
2025-03-12 17:42:01,267 - INFO - Gradient shapes - grad_x: torch.Size([2048, 1024]), grad_w: torch.Size([8192, 1024])
2025-03-12 17:42:01,267 - INFO - Running PyTorch reference implementation
2025-03-12 17:42:01,565 - INFO - Comparing gradients with PyTorch reference
2025-03-12 17:42:01,587 - INFO - Maximum gradient error - grad_x: 231.0, grad_w: 120.5
2025-03-12 17:42:01,626 - INFO - Gradients allclose check - grad_x: False, grad_w: False
2025-03-12 17:42:01,626 - ERROR - ✗ FAILURE: Gradient mismatch detected in allclose check
2025-03-12 17:42:01,629 - ERROR - Largest grad_x difference at (np.int64(1951), np.int64(933)): 0.0 vs 231.0
2025-03-12 17:42:01,635 - ERROR - Zeros in grad_x: 1949696/2097152 (92.97%)
2025-03-12 17:42:01,635 - ERROR - Zeros in x_autograd.grad: 0/2097152 (0.00%)
2025-03-12 17:42:01,635 - ERROR - Largest grad_w difference at (np.int64(5413), np.int64(856)): 0.0 vs 120.5
2025-03-12 17:42:01,636 - ERROR - Zeros in grad_w: 7847936/8388608 (93.55%)
2025-03-12 17:42:01,636 - ERROR - Zeros in w_autograd.grad: 0/8388608 (0.00%)
2025-03-12 17:42:01,636 - INFO - Test failed

Success case:
Running test_backward_pass
Test setup - G: 4, M: 2048, N: 2048, K: 64
Input x shape: torch.Size([2048, 64])
2025-03-12 17:46:57,421 - INFO - Weight w shape: torch.Size([8192, 64])
2025-03-12 17:46:57,423 - INFO - Group sizes: tensor([512, 512, 512, 512], device='cuda:0', dtype=torch.int32)
2025-03-12 17:46:57,423 - INFO - Running forward pass
2025-03-12 17:46:57,847 - INFO - Forward result shape: torch.Size([2048, 8192])
2025-03-12 17:46:57,848 - INFO - Created gradient with shape: torch.Size([2048, 8192])
2025-03-12 17:46:57,848 - INFO - Running backward pass directly
2025-03-12 17:46:57,848 - INFO - Starting grouped_gemm_backward with TMA-enabled scheduling
2025-03-12 17:46:57,848 - INFO - TMA support detected on GPU with 132 SMs
2025-03-12 17:46:57,848 - INFO - EVEN_K optimization enabled: True (K=64)
2025-03-12 17:46:57,857 - INFO - Computing grad_x with TMA-enabled kernel
2025-03-12 17:46:57,865 - INFO - completed: grad_X computation successful with TMA-enabled kernel
2025-03-12 17:46:57,865 - INFO - Computing grad_w with TMA-enabled kernel
2025-03-12 17:46:57,873 - INFO - completed:  grad_W computation successful with TMA-enabled kernel
2025-03-12 17:46:57,873 - INFO - Gradient shapes - grad_x: torch.Size([2048, 64]), grad_w: torch.Size([8192, 64])
2025-03-12 17:46:57,873 - INFO - Running PyTorch reference implementation
2025-03-12 17:46:58,224 - INFO - Comparing gradients with PyTorch reference
2025-03-12 17:46:58,246 - INFO - Maximum gradient error - grad_x: 0.25, grad_w: 0.25
2025-03-12 17:46:58,282 - INFO - Gradients allclose check - grad_x: True, grad_w: True
2025-03-12 17:46:58,282 - INFO - ✓ SUCCESS: Gradients match the PyTorch reference (allclose check passed)
2025-03-12 17:46:58,285 - ERROR - Largest grad_x difference at (np.int64(615), np.int64(32)): -46.0 vs -46.25
2025-03-12 17:46:58,290 - ERROR - Zeros in grad_x: 0/131072 (0.00%)
2025-03-12 17:46:58,290 - ERROR - Zeros in x_autograd.grad: 0/131072 (0.00%)
2025-03-12 17:46:58,290 - ERROR - Largest grad_w difference at (np.int64(2609), np.int64(39)): -45.75 vs -46.0
2025-03-12 17:46:58,291 - ERROR - Zeros in grad_w: 0/524288 (0.00%)
2025-03-12 17:46:58,291 - ERROR - Zeros in w_autograd.grad: 0/524288 (0.00%)
2025-03-12 17:46:58,291 - INFO - Test succeeded

Mid case:
Test setup - G: 4, M: 2048, N: 2048, K: 128
Input x shape: torch.Size([2048, 128])
2025-03-12 17:47:45,891 - INFO - Weight w shape: torch.Size([8192, 128])
2025-03-12 17:47:45,893 - INFO - Group sizes: tensor([512, 512, 512, 512], device='cuda:0', dtype=torch.int32)
2025-03-12 17:47:45,893 - INFO - Running forward pass
2025-03-12 17:47:46,318 - INFO - Forward result shape: torch.Size([2048, 8192])
2025-03-12 17:47:46,319 - INFO - Created gradient with shape: torch.Size([2048, 8192])
2025-03-12 17:47:46,319 - INFO - Running backward pass directly
2025-03-12 17:47:46,319 - INFO - Starting grouped_gemm_backward with TMA-enabled scheduling
2025-03-12 17:47:46,319 - INFO - TMA support detected on GPU with 132 SMs
2025-03-12 17:47:46,319 - INFO - EVEN_K optimization enabled: True (K=128)
2025-03-12 17:47:46,328 - INFO - Computing grad_x with TMA-enabled kernel
2025-03-12 17:47:46,333 - INFO - completed: grad_X computation successful with TMA-enabled kernel
2025-03-12 17:47:46,333 - INFO - Computing grad_w with TMA-enabled kernel
2025-03-12 17:47:46,337 - INFO - completed:  grad_W computation successful with TMA-enabled kernel
2025-03-12 17:47:46,337 - INFO - Gradient shapes - grad_x: torch.Size([2048, 128]), grad_w: torch.Size([8192, 128])
2025-03-12 17:47:46,337 - INFO - Running PyTorch reference implementation
2025-03-12 17:47:46,640 - INFO - Comparing gradients with PyTorch reference
2025-03-12 17:47:46,662 - INFO - Maximum gradient error - grad_x: 218.0, grad_w: 111.0
2025-03-12 17:47:46,700 - INFO - Gradients allclose check - grad_x: False, grad_w: False
2025-03-12 17:47:46,700 - ERROR - ✗ FAILURE: Gradient mismatch detected in allclose check
2025-03-12 17:47:46,704 - ERROR - Largest grad_x difference at (np.int64(1632), np.int64(29)): 0.0 vs 218.0
2025-03-12 17:47:46,709 - ERROR - Zeros in grad_x: 114688/262144 (43.75%)
2025-03-12 17:47:46,709 - ERROR - Zeros in x_autograd.grad: 0/262144 (0.00%)
2025-03-12 17:47:46,709 - ERROR - Largest grad_w difference at (np.int64(6603), np.int64(53)): 0.0 vs -111.0
2025-03-12 17:47:46,709 - ERROR - Zeros in grad_w: 507904/1048576 (48.44%)
2025-03-12 17:47:46,709 - ERROR - Zeros in w_autograd.grad: 0/1048576 (0.00%)
2025-03-12 17:47:46,709 - INFO - Test failed
"""
