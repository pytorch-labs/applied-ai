"""
Test setup - G: 2, M: 64, N: 128, K: 32
Input x shape: torch.Size([64, 32])
2025-03-10 08:48:18,721 - INFO - Weight w shape: torch.Size([256, 32])
2025-03-10 08:48:18,725 - INFO - Group sizes: tensor([32, 32], device='cuda:0', dtype=torch.int32)
2025-03-10 08:48:18,725 - INFO - Running forward pass
2025-03-10 08:48:19,119 - INFO - Forward result shape: torch.Size([64, 256])
2025-03-10 08:48:19,119 - INFO - Created gradient with shape: torch.Size([64, 256])
2025-03-10 08:48:19,119 - INFO - Running backward pass directly
2025-03-10 08:48:19,119 - INFO - Starting grouped_gemm_backward with fixed configurations
2025-03-10 08:48:19,119 - INFO - Large computation detected: False
2025-03-10 08:48:19,119 - INFO - Using PyTorch fallback for grouped GEMM backward with high precision
2025-03-10 08:48:19,119 - INFO - PyTorch fallback dims - G: 2, M: 64, N: 128, K_x: 32, K_w: 32
2025-03-10 08:48:19,281 - INFO - Gradient shapes - grad_x: torch.Size([64, 32]), grad_w: torch.Size([256, 32])
2025-03-10 08:48:19,281 - INFO - Running PyTorch reference implementation
2025-03-10 08:48:19,567 - INFO - Comparing gradients with PyTorch reference
2025-03-10 08:48:19,644 - INFO - grad W compare: grad_w=tensor([[ -4.5625,  -4.9688,   6.1250,  ...,  10.0625,  -7.0625,  -0.9258],
        [  0.5430,  -4.5625,  -1.4453,  ...,   1.9297,   1.1484,   8.8750],
        [  4.9375,   5.1562,   2.2188,  ...,   0.7773, -12.9375,  -0.2578],
        ...,
        [ -4.2188,  -7.3750,  -2.1875,  ...,   1.6641,  -0.1338,   9.5000],
        [ -5.1562,   6.0938,  -7.5938,  ..., -13.5000,   2.2031,   1.2891],
        [ -5.0938,  10.5625,   9.1250,  ...,  -3.6406,  -2.2188,   4.2500]],
       device='cuda:0', dtype=torch.bfloat16, grad_fn=<CopyBackwards>), w_autograd=tensor([[ 1.1797, -0.3535,  0.3750,  ...,  0.0684,  0.0271,  0.2148],
        [ 2.6250,  0.5039, -1.3906,  ..., -0.4277,  2.6719,  1.7578],
        [-0.5117, -1.2969,  1.4766,  ...,  1.2578, -0.7266, -0.0801],
        ...,
        [-0.0645,  0.5898, -0.0474,  ..., -0.0869, -0.1328,  0.2715],
        [-0.4023, -0.7695, -0.9102,  ...,  0.3906, -1.3750, -0.3652],
        [ 0.2168, -0.0171, -1.0547,  ...,  0.6406,  0.6914, -1.5000]],
       device='cuda:0', dtype=torch.bfloat16, requires_grad=True)
2025-03-10 08:48:19,647 - INFO - grad X compare: grad_x=tensor([[ 21.6250, -15.9375,  20.8750,  ...,  -0.0625,  -2.0156,   5.2188],
        [-21.5000, -20.6250,   7.7188,  ...,  11.6875,  -2.4688, -32.2500],
        [ -2.9062,   5.7812,   4.4688,  ...,  11.3750,   7.8750,   3.1406],
        ...,
        [  8.9375,  -5.3750,  25.1250,  ...,  -2.3750,   6.9375, -10.5000],
        [ 10.7500,  20.8750, -24.5000,  ...,   3.4375,  -3.9375,  -3.8750],
        [  2.5156,  -4.1250,   2.3594,  ...,   5.5000,  -9.7500, -34.0000]],
       device='cuda:0', dtype=torch.bfloat16, grad_fn=<CopyBackwards>), x_autograd=tensor([[-0.7656,  1.0391, -0.4512,  ...,  0.1641, -0.8711,  0.6602],
        [-0.7539,  0.4727, -0.0156,  ...,  0.8867,  0.2344,  0.7461],
        [ 0.6914, -0.8750, -1.3672,  ..., -0.3184,  1.0938,  0.9961],
        ...,
        [ 1.0781, -0.7031, -0.9922,  ..., -0.1934, -0.1211,  0.5781],
        [ 1.8281, -0.0796,  0.2373,  ..., -0.7852, -1.5000, -0.4434],
        [-0.6641,  0.2246,  0.5234,  ..., -0.2773,  0.5508, -2.2031]],
       device='cuda:0', dtype=torch.bfloat16, requires_grad=True)
2025-03-10 08:48:19,647 - INFO - Maximum gradient error - grad_x: 0.0, grad_w: 0.0
2025-03-10 08:48:19,647 - INFO - ✓ Gradients match the PyTorch reference
2025-03-10 08:48:19,647 - INFO - Test succeeded
(tritondev) [less@devgpu115.cco2 /data/users/less/applied-ai/dev/triton_groupGEMM/clean (lessw/gg_backward_pass)]$ python fast_debug.py
TMA benchmarks will be running with experimental grid constant TMA descriptor.
Running test_backward_pass
Test setup - G: 2, M: 256, N: 128, K: 32
Input x shape: torch.Size([256, 32])
2025-03-10 08:48:49,768 - INFO - Weight w shape: torch.Size([256, 32])
2025-03-10 08:48:49,770 - INFO - Group sizes: tensor([128, 128], device='cuda:0', dtype=torch.int32)
2025-03-10 08:48:49,770 - INFO - Running forward pass
2025-03-10 08:48:50,289 - INFO - Forward result shape: torch.Size([256, 256])
2025-03-10 08:48:50,289 - INFO - Created gradient with shape: torch.Size([256, 256])
2025-03-10 08:48:50,289 - INFO - Running backward pass directly
2025-03-10 08:48:50,289 - INFO - Starting grouped_gemm_backward with fixed configurations
2025-03-10 08:48:50,289 - INFO - Large computation detected: True
2025-03-10 08:48:50,289 - INFO - Using high precision computation for large matrices
2025-03-10 08:48:50,289 - INFO - Using PyTorch fallback for grouped GEMM backward with high precision
2025-03-10 08:48:50,289 - INFO - PyTorch fallback dims - G: 2, M: 256, N: 128, K_x: 32, K_w: 32
2025-03-10 08:48:50,373 - INFO - Gradient shapes - grad_x: torch.Size([256, 32]), grad_w: torch.Size([256, 32])
2025-03-10 08:48:50,374 - INFO - Running PyTorch reference implementation
2025-03-10 08:48:50,620 - INFO - Comparing gradients with PyTorch reference
2025-03-10 08:48:50,684 - INFO - grad W compare: grad_w=tensor([[ 12.3125, -10.8125,   7.5312,  ...,  -4.7188,  -8.5000,  -0.0408],
        [  4.1250,  -6.8125,  -0.5859,  ..., -11.7500,  -1.0078,   0.9844],
        [-22.3750, -20.7500,  -5.6875,  ...,   3.8906,  15.6250,   4.5938],
        ...,
        [-15.3750,  12.4375,  -6.9375,  ...,   2.5938,  16.8750,  -1.0078],
        [ -0.1562,  -2.3750,  -1.6641,  ..., -16.2500,  15.5000,  -9.7500],
        [ -6.3750,   1.6562,  21.8750,  ...,   2.5156,   6.8125,  -9.3750]],
       device='cuda:0', dtype=torch.bfloat16, grad_fn=<CopyBackwards>), w_autograd=tensor([[ 0.0054, -2.2500,  0.8359,  ...,  0.8711, -0.7500, -0.1074],
        [ 0.1436, -1.2656, -0.0947,  ..., -1.2656,  1.0625,  1.7109],
        [ 0.6484, -0.6016,  0.4160,  ...,  0.0630,  1.0938, -0.7109],
        ...,
        [-0.6250,  1.8672, -0.7070,  ...,  0.7344,  0.0679, -1.1094],
        [ 0.7461, -0.0092, -0.9453,  ...,  0.2891,  0.5625, -0.9336],
        [-0.5234, -0.9844,  0.7070,  ..., -0.0164, -2.2344,  0.3223]],
       device='cuda:0', dtype=torch.bfloat16, requires_grad=True)
2025-03-10 08:48:50,686 - INFO - grad X compare: grad_x=tensor([[  7.0938,   8.1875,  -6.4062,  ...,  -5.7188,  15.0625,   7.6562],
        [-16.6250,  -3.1875,  -2.1719,  ...,   1.8203,  19.1250,  -2.3594],
        [ -8.1875, -24.2500,  -4.5312,  ..., -15.6250,   3.3125, -12.5625],
        ...,
        [ 11.9375,  -7.6875,   0.0894,  ...,   3.0781,  -0.2119,  -6.2812],
        [  2.8594, -13.3125,  -7.3750,  ...,  -8.0000, -11.1875,  -8.6875],
        [ 11.1875,  21.1250,   3.5469,  ..., -14.0625,   0.7617,   2.7344]],
       device='cuda:0', dtype=torch.bfloat16, grad_fn=<CopyBackwards>), x_autograd=tensor([[-4.3750e-01, -1.1094e+00, -8.1250e-01,  ...,  1.6875e+00,
          4.4141e-01, -7.6953e-01],
        [ 1.5312e+00, -1.4648e-01, -2.7084e-04,  ...,  1.6309e-01,
          7.8125e-01, -9.3750e-01],
        [ 8.0859e-01, -9.6094e-01, -7.0312e-01,  ...,  6.1328e-01,
          1.3281e+00, -2.4375e+00],
        ...,
        [ 1.2109e+00, -1.7773e-01,  8.5547e-01,  ...,  1.4219e+00,
          2.6562e-01,  1.0156e+00],
        [-6.5918e-02,  1.8750e-01, -6.7578e-01,  ...,  1.3359e+00,
          5.7031e-01,  1.4531e+00],
        [ 1.6328e+00,  1.2266e+00, -1.9922e-01,  ..., -2.9688e-01,
          1.6953e+00,  1.1328e+00]], device='cuda:0', dtype=torch.bfloat16,
       requires_grad=True)
2025-03-10 08:48:50,686 - INFO - Maximum gradient error - grad_x: 0.0, grad_w: 0.0
2025-03-10 08:48:50,686 - INFO - ✓ Gradients match the PyTorch reference
2025-03-10 08:48:50,686 - INFO - Test succeeded
(tritondev) [less@devgpu115.cco2 /data/users/less/applied-ai/dev/triton_groupGEMM/clean (lessw/gg_backward_pass)]$ python fast_debug.py
TMA benchmarks will be running with experimental grid constant TMA descriptor.
Running test_backward_pass
Test setup - G: 2, M: 256, N: 128, K: 64
Input x shape: torch.Size([256, 64])
2025-03-10 08:49:04,107 - INFO - Weight w shape: torch.Size([256, 64])
2025-03-10 08:49:04,109 - INFO - Group sizes: tensor([128, 128], device='cuda:0', dtype=torch.int32)
2025-03-10 08:49:04,109 - INFO - Running forward pass
2025-03-10 08:49:04,490 - INFO - Forward result shape: torch.Size([256, 256])
2025-03-10 08:49:04,490 - INFO - Created gradient with shape: torch.Size([256, 256])
2025-03-10 08:49:04,490 - INFO - Running backward pass directly
2025-03-10 08:49:04,490 - INFO - Starting grouped_gemm_backward with fixed configurations
2025-03-10 08:49:04,490 - INFO - Large computation detected: True
2025-03-10 08:49:04,490 - INFO - Using high precision computation for large matrices
2025-03-10 08:49:04,491 - INFO - Using PyTorch fallback for grouped GEMM backward with high precision
2025-03-10 08:49:04,491 - INFO - PyTorch fallback dims - G: 2, M: 256, N: 128, K_x: 64, K_w: 64
2025-03-10 08:49:04,574 - INFO - Gradient shapes - grad_x: torch.Size([256, 64]), grad_w: torch.Size([256, 64])
2025-03-10 08:49:04,574 - INFO - Running PyTorch reference implementation
2025-03-10 08:49:04,807 - INFO - Comparing gradients with PyTorch reference
2025-03-10 08:49:04,867 - INFO - grad W compare: grad_w=tensor([[-13.4375, -16.1250,  -0.3320,  ...,   4.8750, -31.7500,  -0.3965],
        [  1.1953, -22.1250,   7.9062,  ..., -11.2500,  -8.9375,   8.6250],
        [ 10.4375,  12.5000,  12.8750,  ...,   1.8047,   0.9883,  -8.8750],
        ...,
        [ 12.7500,  -6.5000,   0.0444,  ...,   9.8125,  28.3750, -12.5000],
        [-12.1250,  18.2500,   0.4434,  ...,  -4.7188,  -6.8750,   0.5547],
        [ -9.5625,  -2.0156, -24.8750,  ...,   9.3750,  -7.6875,  -1.2500]],
       device='cuda:0', dtype=torch.bfloat16, grad_fn=<CopyBackwards>), w_autograd=tensor([[ 7.5781e-01,  8.1250e-01,  1.2578e+00,  ..., -8.9453e-01,
         -8.4375e-01,  1.1719e+00],
        [ 6.4453e-02, -2.6406e+00, -1.5469e+00,  ..., -1.3906e+00,
          1.2344e+00,  2.4805e-01],
        [ 3.7695e-01,  1.8125e+00,  1.5747e-02,  ..., -1.3281e+00,
         -2.9688e-01,  1.8234e-03],
        ...,
        [ 4.5508e-01, -8.1641e-01,  1.4688e+00,  ...,  8.3984e-01,
         -1.0234e+00, -3.1836e-01],
        [-4.1602e-01,  7.5000e-01, -1.3438e+00,  ..., -2.1777e-01,
         -9.1309e-02,  7.1875e-01],
        [ 4.3164e-01, -5.2344e-01, -6.5234e-01,  ...,  5.9375e-01,
          5.3906e-01,  6.1328e-01]], device='cuda:0', dtype=torch.bfloat16,
       requires_grad=True)
2025-03-10 08:49:04,869 - INFO - grad X compare: grad_x=tensor([[ -5.9062,  -2.2344,  17.7500,  ...,  25.2500, -17.0000, -17.2500],
        [ -4.7188,   4.1875,  -8.8750,  ...,  12.1250,  12.2500, -13.4375],
        [ -8.3750,   1.8047,  -2.7344,  ...,   8.0625,  -5.4062, -10.3750],
        ...,
        [ -3.5000,  -2.2188,   6.1562,  ...,   8.5625,  22.1250,  -3.7969],
        [ 10.6875,   9.8125,  13.1250,  ...,  -7.0312,   6.2188,  20.7500],
        [  9.6250, -13.2500,   2.6250,  ...,   7.8438, -10.7500,  -0.2021]],
       device='cuda:0', dtype=torch.bfloat16, grad_fn=<CopyBackwards>), x_autograd=tensor([[-2.2812, -1.3203,  0.9883,  ...,  2.0781, -0.8789, -1.1094],
        [ 0.0204, -1.3281,  1.6953,  ..., -0.3828,  1.3281, -0.8203],
        [-2.5781,  0.9023,  0.4824,  ..., -0.1807, -0.1465, -0.6250],
        ...,
        [ 0.3613,  1.0938,  2.0000,  ...,  0.4004, -0.0201, -1.0156],
        [ 1.5938,  0.6055,  0.0679,  ...,  0.3730,  0.0481, -0.8984],
        [-0.1245,  0.0693, -0.5234,  ...,  0.9961, -0.0481, -0.7422]],
       device='cuda:0', dtype=torch.bfloat16, requires_grad=True)
2025-03-10 08:49:04,869 - INFO - Maximum gradient error - grad_x: 9.5367431640625e-07, grad_w: 7.62939453125e-06
2025-03-10 08:49:04,869 - INFO - ✓ Gradients match the PyTorch reference
2025-03-10 08:49:04,869 - INFO - Test succeeded
"""
