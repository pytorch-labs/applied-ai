"""
Running test_backward_pass
Test setup - G: 2, M: 128, N: 128, K: 32
Input x shape: torch.Size([128, 32])
2025-03-10 08:31:29,705 - INFO - Weight w shape: torch.Size([256, 32])
2025-03-10 08:31:29,712 - INFO - Group sizes: tensor([64, 64], device='cuda:0', dtype=torch.int32)
2025-03-10 08:31:29,712 - INFO - Running forward pass
2025-03-10 08:31:30,277 - INFO - Forward result shape: torch.Size([128, 256])
2025-03-10 08:31:30,277 - INFO - Created gradient with shape: torch.Size([128, 256])
2025-03-10 08:31:30,277 - INFO - Running backward pass directly
2025-03-10 08:31:30,277 - INFO - Starting grouped_gemm_backward with fixed configurations
2025-03-10 08:31:30,277 - INFO - Using PyTorch fallback for grouped GEMM backward with FP32 precision
2025-03-10 08:31:30,277 - INFO - PyTorch fallback dims - G: 2, M: 128, N: 128, K_x: 32, K_w: 32
2025-03-10 08:31:30,362 - INFO - Gradient shapes - grad_x: torch.Size([128, 32]), grad_w: torch.Size([256, 32])
2025-03-10 08:31:30,362 - INFO - Running PyTorch reference implementation
2025-03-10 08:31:30,639 - INFO - Comparing gradients with PyTorch reference
2025-03-10 08:31:30,722 - INFO - grad W compare: grad_w=tensor([[ -2.1094,   7.5000,  -5.9062,  ...,   3.9688,   3.1406,  -3.3438],
        [ 10.1875,  -1.6719,   3.8281,  ...,   1.9922,  12.1250,   2.0156],
        [ -8.0625,  -0.2285,   5.7188,  ..., -16.2500,   8.0625,  10.1875],
        ...,
        [  0.2383,  -3.7500,   7.3125,  ..., -18.3750,   1.2422, -18.1250],
        [  9.9375,   4.0938,   9.7500,  ...,  -2.7500,  12.8125,  -2.9531],
        [ -9.3750, -12.0625,  -7.5312,  ...,  -8.1250,  -0.9688,   7.5312]],
       device='cuda:0', dtype=torch.bfloat16, grad_fn=<CopyBackwards>), w_autograd=tensor([[ 0.1807, -2.7500,  2.7656,  ...,  0.8789,  0.4805,  0.4062],
        [ 0.1465, -0.5117,  1.1250,  ..., -1.8594,  0.1270, -0.5586],
        [-1.2188,  0.0055,  0.2480,  ...,  2.0625, -0.3438,  0.1318],
        ...,
        [-0.0688,  0.5859, -0.0781,  ..., -2.3594, -0.6094, -0.6250],
        [ 0.1582,  1.4141, -0.2373,  ..., -1.1953, -1.8125,  1.1094],
        [ 0.7695, -0.4707,  0.5508,  ..., -0.1768, -1.2266,  0.3594]],
       device='cuda:0', dtype=torch.bfloat16, requires_grad=True)
2025-03-10 08:31:30,724 - INFO - grad X compare: grad_x=tensor([[  0.3711,  -6.4688,   0.1562,  ...,  36.7500,  11.1250,  14.5000],
        [-12.8750,  -0.1445, -12.9375,  ..., -11.5000,  -7.7500,  10.6250],
        [ -6.4375,  -3.7344, -16.5000,  ...,   4.5625,  -2.3281,  17.0000],
        ...,
        [ -2.0156,   2.1875,   1.7031,  ...,  10.3125,   1.4766,  12.9375],
        [ 13.9375,   2.9688,  12.3125,  ..., -12.8750,  -1.1641,  -8.4375],
        [  8.4375,  -0.5508, -33.2500,  ...,  20.5000,   5.8125,  -7.1875]],
       device='cuda:0', dtype=torch.bfloat16, grad_fn=<CopyBackwards>), x_autograd=tensor([[-0.7305,  0.7031,  0.1738,  ..., -0.3789, -1.2656,  0.2793],
        [ 1.0547, -0.3066,  0.5977,  ...,  0.4805, -1.7031,  0.7227],
        [ 0.3008, -1.1797, -0.2598,  ...,  0.4531, -0.0145, -0.1289],
        ...,
        [-1.7266,  0.8242, -0.0400,  ...,  0.2471, -1.3906, -0.9648],
        [ 3.0156,  1.0000,  0.9531,  ..., -0.4199,  0.7930, -0.9180],
        [ 0.1953, -1.7109, -0.1006,  ..., -0.1128, -1.6797, -0.0767]],
       device='cuda:0', dtype=torch.bfloat16, requires_grad=True)
2025-03-10 08:31:30,725 - INFO - Maximum gradient error - grad_x: 0.125, grad_w: 0.001953125
2025-03-10 08:31:30,725 - ERROR - ✗ Gradient mismatch above tolerance threshold
2025-03-10 08:31:30,725 - INFO - Test succeeded
(tritondev) [less@devgpu115.cco2 /data/users/less/applied-ai/dev/triton_groupGEMM/clean (lessw/gg_backward_pass)]$ python fast_debug.py
TMA benchmarks will be running with experimental grid constant TMA descriptor.
Running test_backward_pass
Test setup - G: 2, M: 64, N: 128, K: 32
Input x shape: torch.Size([64, 32])
2025-03-10 08:31:45,109 - INFO - Weight w shape: torch.Size([256, 32])
2025-03-10 08:31:45,111 - INFO - Group sizes: tensor([32, 32], device='cuda:0', dtype=torch.int32)
2025-03-10 08:31:45,111 - INFO - Running forward pass
2025-03-10 08:31:45,639 - INFO - Forward result shape: torch.Size([64, 256])
2025-03-10 08:31:45,639 - INFO - Created gradient with shape: torch.Size([64, 256])
2025-03-10 08:31:45,639 - INFO - Running backward pass directly
2025-03-10 08:31:45,639 - INFO - Starting grouped_gemm_backward with fixed configurations
2025-03-10 08:31:45,639 - INFO - Using PyTorch fallback for grouped GEMM backward with FP32 precision
2025-03-10 08:31:45,639 - INFO - PyTorch fallback dims - G: 2, M: 64, N: 128, K_x: 32, K_w: 32
2025-03-10 08:31:45,723 - INFO - Gradient shapes - grad_x: torch.Size([64, 32]), grad_w: torch.Size([256, 32])
2025-03-10 08:31:45,724 - INFO - Running PyTorch reference implementation
2025-03-10 08:31:46,008 - INFO - Comparing gradients with PyTorch reference
2025-03-10 08:31:46,090 - INFO - grad W compare: grad_w=tensor([[  0.3633,   0.1143,   3.2812,  ...,  -2.4219,  -0.3008,   5.5312],
        [ 10.8125,  -0.1934, -13.1875,  ...,   7.4688,  -1.9531,  -3.6094],
        [ -6.9375,   5.9062,   6.3750,  ...,  -1.0469,   0.7969,  -4.1875],
        ...,
        [ -7.4062,  -9.3125,   4.6875,  ...,   1.4062,  -0.1455, -10.6875],
        [  2.2812,  -1.1406,  -0.9531,  ...,   0.1406,   1.2812,  -6.0312],
        [  2.0781,   3.1562,  -4.7188,  ...,  -2.1406,   5.1875,   9.6875]],
       device='cuda:0', dtype=torch.bfloat16, grad_fn=<CopyBackwards>), w_autograd=tensor([[ 0.5039,  0.8672, -0.7383,  ...,  0.1914,  0.3535,  0.4785],
        [ 1.2500,  0.5820,  0.3125,  ...,  2.4219,  1.5469,  0.7383],
        [-1.0391, -0.0703, -1.9062,  ...,  1.3438, -0.9492,  2.3750],
        ...,
        [-0.4238,  0.5352,  0.2734,  ...,  0.5547,  0.7305, -0.8242],
        [-1.1719,  0.2910, -0.1377,  ..., -1.7578,  2.0781,  0.9570],
        [ 0.7461,  0.4824,  1.0156,  ...,  0.7500, -0.5117,  1.1172]],
       device='cuda:0', dtype=torch.bfloat16, requires_grad=True)
2025-03-10 08:31:46,092 - INFO - grad X compare: grad_x=tensor([[  6.4062,   1.3750, -11.0625,  ...,   2.4844, -25.3750,  13.9375],
        [-28.2500,   1.2656,  -2.9375,  ...,   2.1719,  -5.6875,  10.3750],
        [-21.3750,   9.2500,   5.5938,  ...,  11.8125,  -8.9375,  -8.2500],
        ...,
        [-25.7500,  -4.4375,   9.1875,  ...,  11.8125,  -6.0000,  25.0000],
        [ 20.6250,  11.1875,   4.3438,  ...,  -9.9375,  -2.0312,   4.0938],
        [ 12.6250,  -8.2500,  -5.1562,  ...,   4.4688,  -5.3125,  -5.8125]],
       device='cuda:0', dtype=torch.bfloat16, grad_fn=<CopyBackwards>), x_autograd=tensor([[ 1.0000, -0.4414, -0.2910,  ...,  0.7617, -2.0000, -0.4590],
        [-0.6562,  0.4395,  2.5000,  ..., -2.0000,  0.4941, -0.5195],
        [ 0.6445, -0.2754,  0.5625,  ..., -1.0938,  0.6016,  0.5820],
        ...,
        [ 0.6641, -2.0156,  1.0312,  ..., -0.4902, -2.2031, -0.8867],
        [ 1.1875, -1.1406,  0.1680,  ..., -0.5312, -0.1060, -0.0864],
        [ 0.2432, -0.5039, -0.9531,  ..., -2.3750,  0.8281,  0.3594]],
       device='cuda:0', dtype=torch.bfloat16, requires_grad=True)
2025-03-10 08:31:46,092 - INFO - Maximum gradient error - grad_x: 1.9073486328125e-06, grad_w: 0.0
2025-03-10 08:31:46,092 - INFO - ✓ Gradients match the PyTorch reference
2025-03-10 08:31:46,092 - INFO - Test succeeded
"""
