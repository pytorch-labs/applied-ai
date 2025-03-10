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

With latest:
2025-03-10 09:23:30,391 - ERROR - Error in backward_x kernel: PassManager::run failed
2025-03-10 09:23:30,391 - INFO - Falling back to PyTorch for grad_x
2025-03-10 09:23:30,440 - INFO - Computing grad_w with triton kernel
2025-03-10 09:23:30,820 - INFO - grad_w computation successful with triton
2025-03-10 09:23:30,842 - INFO - Gradient shapes - grad_x: torch.Size([256, 128]), grad_w: torch.Size([1024, 128])
2025-03-10 09:23:30,842 - INFO - Running PyTorch reference implementation
/home/less/.conda/envs/tritondev/lib/python3.12/site-packages/torch/autograd/graph.py:824: UserWarning: Attempting to run cuBLAS, but there was no current CUDA context! Attempting to set the primary context... (Triggered internally at /pytorch/aten/src/ATen/cuda/CublasHandlePool.cpp:181.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
2025-03-10 09:23:31,087 - INFO - Comparing gradients with PyTorch reference
2025-03-10 09:23:31,156 - INFO - grad W compare: grad_w=tensor([[-24.7500,  -6.7812, -14.2500,  ...,  -4.3125,  15.6875, -22.1250],
        [ -9.0625,   3.6562,   4.2500,  ...,   1.8750,  -0.5117,   6.5938],
        [ 14.0625, -12.5625,  -3.8750,  ...,  19.8750, -10.3125, -11.1875],
        ...,
        [  4.7188,  -5.4688, -21.2500,  ...,   0.1670,  -9.6875, -17.2500],
        [-13.8125,  -2.5938, -12.8125,  ..., -13.0625,   3.7812,  11.2500],
        [ -9.9375,  -5.5625,   4.8750,  ...,  -0.0635,  -3.9531,  20.7500]],
       device='cuda:0', dtype=torch.bfloat16), w_autograd=tensor([[ 1.3125, -0.3359, -1.4844,  ..., -0.7266, -0.2734,  0.8594],
        [-2.2656, -0.2490, -0.7891,  ..., -0.8789, -0.5430, -0.8828],
        [ 0.8359, -0.9375,  1.3125,  ..., -1.4844, -1.2734,  1.6094],
        ...,
        [ 1.0938, -0.3223, -0.9648,  ...,  1.5156,  0.7031,  0.2100],
        [ 1.8047, -1.5078,  0.4688,  ...,  1.1953,  0.9844, -0.7031],
        [ 0.0110, -1.3906, -0.1758,  ..., -2.7188,  1.0547, -0.7461]],
       device='cuda:0', dtype=torch.bfloat16, requires_grad=True)
2025-03-10 09:23:31,158 - INFO - grad X compare: grad_x=tensor([[ 35.0000,  26.1250,  27.2500,  ..., -58.5000,   7.3125,  26.1250],
        [ 17.1250, -32.0000,  -2.8125,  ...,  19.2500,  -4.7500,  56.5000],
        [ 11.4375,  10.1875,  15.5625,  ...,   2.7969,  10.9375, -22.5000],
        ...,
        [-24.7500,  44.5000,   8.3125,  ...,   4.2500,  11.1875,   5.9375],
        [-26.0000, -29.2500,  -6.8125,  ...,  26.7500, -33.2500, -16.7500],
        [-19.2500,  22.5000,  -0.0815,  ...,   9.6875, -19.7500,   3.3438]],
       device='cuda:0', dtype=torch.bfloat16, grad_fn=<CopyBackwards>), x_autograd=tensor([[ 0.7773,  1.3047,  1.1250,  ..., -1.7422,  0.9922,  0.7539],
        [ 0.4297,  0.4121, -1.8750,  ..., -1.2578, -1.0312, -1.0859],
        [-0.3457, -1.6797, -0.4434,  ...,  1.4219, -0.5742, -1.4844],
        ...,
        [ 0.7070, -1.2969, -0.7578,  ..., -1.0469,  0.5938, -1.4922],
        [-0.0698, -0.6133, -0.3457,  ...,  1.0469,  0.6680,  1.2734],
        [-1.1797,  1.5234, -0.2236,  ..., -0.4199, -1.0312,  0.5938]],
       device='cuda:0', dtype=torch.bfloat16, requires_grad=True)
2025-03-10 09:23:31,158 - INFO - Maximum gradient error - grad_x: 0.25, grad_w: 0.125
2025-03-10 09:23:31,158 - ERROR - ✗ Gradient mismatch above tolerance threshold
2025-03-10 09:23:31,158 - INFO - Test succeeded
(tritondev) [less@devgpu115.cco2 /data/users/less/applied-ai/dev/triton_groupGEMM/clean (lessw/gg_backward_pass)]$ python fast_debug.py
TMA benchmarks will be running with experimental grid constant TMA descriptor.
Running test_backward_pass
Test setup - G: 1, M: 256, N: 256, K: 64
Input x shape: torch.Size([256, 64])
2025-03-10 09:23:56,992 - INFO - Weight w shape: torch.Size([256, 64])
2025-03-10 09:23:56,994 - INFO - Group sizes: tensor([256], device='cuda:0', dtype=torch.int32)
2025-03-10 09:23:56,994 - INFO - Running forward pass
2025-03-10 09:23:57,531 - INFO - Forward result shape: torch.Size([256, 256])
2025-03-10 09:23:57,531 - INFO - Created gradient with shape: torch.Size([256, 256])
2025-03-10 09:23:57,531 - INFO - Running backward pass directly
2025-03-10 09:23:57,531 - INFO - Starting grouped_gemm_backward with fixed configurations
2025-03-10 09:23:57,531 - INFO - Large computation detected: True
2025-03-10 09:23:57,532 - INFO - M_bucket: 256, NUM_SMS: 16
2025-03-10 09:23:57,532 - INFO - Computing grad_x with triton kernel
/data/users/less/applied-ai/dev/triton_groupGEMM/clean/tgrouped_gemm_backwards.py:132:24: error: cp.async does not support transfers smaller than 4 bytes; calculated this as 2 bytes
                        grad_output_ptr
                       ^
/data/users/less/applied-ai/dev/triton_groupGEMM/clean/tgrouped_gemm_backwards.py:132:24: error: failed to legalize operation 'ttg.async_copy_global_to_local' that was explicitly marked illegal
                        grad_output_ptr
                       ^
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_kernel_grouped_gemm_backward_x(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c192_i32 = arith.constant 192 : i32
    %cst = arith.constant dense<32> : tensor<32x1xi32, #blocked>
    %cst_0 = arith.constant dense<32> : tensor<1x32xi32, #blocked1>
    %cst_1 = arith.constant dense<32> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_2 = arith.constant dense<32> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked2>
    %c16_i32 = arith.constant 16 : i32
    %c63_i32 = arith.constant 63 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<32x64xbf16, #blocked>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<64x32xbf16, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_6 = arith.constant dense<64> : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cst_7 = arith.constant dense<256> : tensor<64x1xi32, #blocked1>
    %cst_8 = arith.constant dense<64> : tensor<32x1xi32, #blocked>
    %cst_9 = arith.constant dense<64> : tensor<64x1xi32, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = tt.load %arg4 : !tt.ptr<i32>
    %2 = arith.cmpi sgt, %1, %c0_i32 : i32
    scf.if %2 {
      %3 = arith.addi %1, %c63_i32 : i32
      %4 = arith.divsi %3, %c64_i32 : i32
      %5 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %6 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %7 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %8 = tt.splat %1 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %9 = tt.splat %1 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %10 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %11 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %12 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>, #blocked1>
      %13 = tt.expand_dims %10 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
      %14 = tt.expand_dims %11 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
      %15 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked>
      %16 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>, #blocked>
      %17 = scf.while (%arg5 = %0) : (i32) -> i32 {
        %18 = arith.cmpi sge, %arg5, %c0_i32 : i32
        %19 = arith.cmpi slt, %arg5, %4 : i32
        %20 = arith.andi %18, %19 : i1
        scf.condition(%20) %arg5 : i32
      } do {
      ^bb0(%arg5: i32):
        %18 = arith.remsi %arg5, %4 : i32
        %19 = arith.divsi %arg5, %4 : i32
        %20 = arith.muli %18, %c64_i32 : i32
        %21 = tt.splat %20 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %22 = tt.splat %20 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %23 = arith.addi %21, %5 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %24 = arith.addi %22, %6 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %25 = arith.muli %19, %c64_i32 : i32
        %26 = tt.splat %25 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %27 = arith.addi %26, %7 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %28 = arith.cmpi slt, %23, %8 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %29 = arith.cmpi slt, %24, %9 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %30 = arith.cmpi slt, %27, %cst_6 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %31 = tt.expand_dims %28 {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi1, #blocked1>
        %32 = tt.broadcast %31 : tensor<64x1xi1, #blocked1> -> tensor<64x32xi1, #blocked1>
        %33 = tt.expand_dims %23 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
        %34 = arith.muli %33, %cst_7 : tensor<64x1xi32, #blocked1>
        %35 = tt.addptr %12, %34 : tensor<64x1x!tt.ptr<bf16>, #blocked1>, tensor<64x1xi32, #blocked1>
        %36 = tt.broadcast %35 : tensor<64x1x!tt.ptr<bf16>, #blocked1> -> tensor<64x32x!tt.ptr<bf16>, #blocked1>
        %37 = tt.expand_dims %30 {axis = 0 : i32} : tensor<64xi1, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi1, #blocked>
        %38 = tt.broadcast %37 : tensor<1x64xi1, #blocked> -> tensor<32x64xi1, #blocked>
        %39 = tt.expand_dims %27 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
        %40 = tt.broadcast %39 : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked>
        %41 = ttg.local_alloc  : () -> !ttg.memdesc<2x64x32xbf16, #shared, #smem, mutable>
        %42 = ttg.local_alloc  : () -> !ttg.memdesc<2x32x64xbf16, #shared, #smem, mutable>
        %43 = arith.cmpi slt, %10, %cst_2 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
        %44 = arith.cmpi slt, %11, %cst_1 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %45 = tt.expand_dims %43 {axis = 0 : i32} : tensor<32xi1, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi1, #blocked1>
        %46 = tt.broadcast %45 : tensor<1x32xi1, #blocked1> -> tensor<64x32xi1, #blocked1>
        %47 = arith.andi %32, %46 : tensor<64x32xi1, #blocked1>
        %48 = tt.broadcast %13 : tensor<1x32xi32, #blocked1> -> tensor<64x32xi32, #blocked1>
        %49 = tt.addptr %36, %48 : tensor<64x32x!tt.ptr<bf16>, #blocked1>, tensor<64x32xi32, #blocked1>
        %50 = ttg.memdesc_subview %41[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x64x32xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xbf16, #shared, #smem, mutable, 2x64x32>
        %51 = ttg.async_copy_global_to_local %49, %50 mask %47 other %cst_5 : tensor<64x32x!tt.ptr<bf16>, #blocked1> -> <64x32xbf16, #shared, #smem, mutable, 2x64x32>
        %52 = ttg.async_commit_group %51
        %53 = tt.expand_dims %44 {axis = 1 : i32} : tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi1, #blocked>
        %54 = tt.broadcast %53 : tensor<32x1xi1, #blocked> -> tensor<32x64xi1, #blocked>
        %55 = arith.andi %54, %38 : tensor<32x64xi1, #blocked>
        %56 = arith.muli %14, %cst_8 : tensor<32x1xi32, #blocked>
        %57 = tt.addptr %15, %56 : tensor<32x1x!tt.ptr<bf16>, #blocked>, tensor<32x1xi32, #blocked>
        %58 = tt.broadcast %57 : tensor<32x1x!tt.ptr<bf16>, #blocked> -> tensor<32x64x!tt.ptr<bf16>, #blocked>
        %59 = tt.addptr %58, %40 : tensor<32x64x!tt.ptr<bf16>, #blocked>, tensor<32x64xi32, #blocked>
        %60 = ttg.memdesc_subview %42[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x32x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<32x64xbf16, #shared, #smem, mutable, 2x32x64>
        %61 = ttg.async_copy_global_to_local %59, %60 mask %55 other %cst_4 : tensor<32x64x!tt.ptr<bf16>, #blocked> -> <32x64xbf16, #shared, #smem, mutable, 2x32x64>
        %62 = ttg.async_commit_group %61
        %63 = arith.addi %13, %cst_0 : tensor<1x32xi32, #blocked1>
        %64 = tt.broadcast %63 : tensor<1x32xi32, #blocked1> -> tensor<64x32xi32, #blocked1>
        %65 = tt.addptr %36, %64 : tensor<64x32x!tt.ptr<bf16>, #blocked1>, tensor<64x32xi32, #blocked1>
        %66 = ttg.memdesc_subview %41[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x64x32xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xbf16, #shared, #smem, mutable, 2x64x32>
        %67 = ttg.async_copy_global_to_local %65, %66 mask %47 other %cst_5 : tensor<64x32x!tt.ptr<bf16>, #blocked1> -> <64x32xbf16, #shared, #smem, mutable, 2x64x32>
        %68 = ttg.async_commit_group %67
        %69 = arith.addi %14, %cst : tensor<32x1xi32, #blocked>
        %70 = arith.muli %69, %cst_8 : tensor<32x1xi32, #blocked>
        %71 = tt.addptr %15, %70 : tensor<32x1x!tt.ptr<bf16>, #blocked>, tensor<32x1xi32, #blocked>
        %72 = tt.broadcast %71 : tensor<32x1x!tt.ptr<bf16>, #blocked> -> tensor<32x64x!tt.ptr<bf16>, #blocked>
        %73 = tt.addptr %72, %40 : tensor<32x64x!tt.ptr<bf16>, #blocked>, tensor<32x64xi32, #blocked>
        %74 = ttg.memdesc_subview %42[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x32x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<32x64xbf16, #shared, #smem, mutable, 2x32x64>
        %75 = ttg.async_copy_global_to_local %73, %74 mask %55 other %cst_4 : tensor<32x64x!tt.ptr<bf16>, #blocked> -> <32x64xbf16, #shared, #smem, mutable, 2x32x64>
        %76 = ttg.async_commit_group %75
        %77:5 = scf.for %arg6 = %c0_i32 to %c256_i32 step %c32_i32 iter_args(%arg7 = %cst_3, %arg8 = %c1_i32, %arg9 = %c-1_i32, %arg10 = %62, %arg11 = %76) -> (tensor<64x64xf32, #blocked2>, i32, i32, !ttg.async.token, !ttg.async.token)  : i32 {
          %92 = arith.cmpi slt, %arg6, %c192_i32 : i32
          %93 = arith.addi %arg9, %c1_i32 : i32
          %94 = arith.cmpi slt, %93, %c2_i32 : i32
          %95 = arith.select %94, %93, %c0_i32 : i32
          %96 = ttg.async_wait %arg10 {num = 2 : i32}
          %97 = ttg.memdesc_subview %41[%95, %c0_i32, %c0_i32] : !ttg.memdesc<2x64x32xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xbf16, #shared, #smem, mutable, 2x64x32>
          %98 = ttg.local_load %97 token %96 : !ttg.memdesc<64x32xbf16, #shared, #smem, mutable, 2x64x32> -> tensor<64x32xbf16, #blocked1>
          %99 = ttg.memdesc_subview %42[%95, %c0_i32, %c0_i32] : !ttg.memdesc<2x32x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<32x64xbf16, #shared, #smem, mutable, 2x32x64>
          %100 = ttg.local_load %99 token %96 : !ttg.memdesc<32x64xbf16, #shared, #smem, mutable, 2x32x64> -> tensor<32x64xbf16, #blocked>
          %101 = arith.extf %98 : tensor<64x32xbf16, #blocked1> to tensor<64x32xf32, #blocked1>
          %102 = ttg.local_alloc %101 : (tensor<64x32xf32, #blocked1>) -> !ttg.memdesc<64x32xf32, #shared, #smem>
          %103 = arith.extf %100 : tensor<32x64xbf16, #blocked> to tensor<32x64xf32, #blocked>
          %104 = ttg.local_alloc %103 : (tensor<32x64xf32, #blocked>) -> !ttg.memdesc<32x64xf32, #shared, #smem>
          %105 = ttg.local_load %102 : !ttg.memdesc<64x32xf32, #shared, #smem> -> tensor<64x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
          %106 = ttg.local_load %104 : !ttg.memdesc<32x64xf32, #shared, #smem> -> tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>>
          %107 = tt.dot %105, %106, %arg7 : tensor<64x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> * tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<64x64xf32, #blocked2>
          %108 = arith.addi %arg8, %c1_i32 : i32
          %109 = arith.cmpi slt, %108, %c2_i32 : i32
          %110 = arith.select %109, %108, %c0_i32 : i32
          %111 = arith.subi %c192_i32, %arg6 : i32
          %112 = arith.minsi %111, %c32_i32 : i32
          %113 = tt.splat %112 : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
          %114 = tt.splat %112 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %115 = arith.cmpi slt, %10, %113 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
          %116 = arith.cmpi slt, %11, %114 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %117 = tt.expand_dims %115 {axis = 0 : i32} : tensor<32xi1, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi1, #blocked1>
          %118 = tt.broadcast %117 : tensor<1x32xi1, #blocked1> -> tensor<64x32xi1, #blocked1>
          %119 = arith.andi %32, %118 : tensor<64x32xi1, #blocked1>
          %120 = arith.addi %arg6, %c64_i32 : i32
          %121 = tt.splat %120 : i32 -> tensor<1x32xi32, #blocked1>
          %122 = arith.addi %121, %13 : tensor<1x32xi32, #blocked1>
          %123 = tt.broadcast %122 : tensor<1x32xi32, #blocked1> -> tensor<64x32xi32, #blocked1>
          %124 = tt.addptr %36, %123 : tensor<64x32x!tt.ptr<bf16>, #blocked1>, tensor<64x32xi32, #blocked1>
          %125 = ttg.memdesc_subview %41[%110, %c0_i32, %c0_i32] : !ttg.memdesc<2x64x32xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xbf16, #shared, #smem, mutable, 2x64x32>
          %126 = tt.splat %92 : i1 -> tensor<64x32xi1, #blocked1>
          %127 = arith.andi %126, %119 : tensor<64x32xi1, #blocked1>
          %128 = ttg.async_copy_global_to_local %124, %125 mask %127 other %cst_5 : tensor<64x32x!tt.ptr<bf16>, #blocked1> -> <64x32xbf16, #shared, #smem, mutable, 2x64x32>
          %129 = ttg.async_commit_group %128
          %130 = tt.expand_dims %116 {axis = 1 : i32} : tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi1, #blocked>
          %131 = tt.broadcast %130 : tensor<32x1xi1, #blocked> -> tensor<32x64xi1, #blocked>
          %132 = arith.andi %131, %38 : tensor<32x64xi1, #blocked>
          %133 = tt.splat %120 : i32 -> tensor<32x1xi32, #blocked>
          %134 = arith.addi %133, %14 : tensor<32x1xi32, #blocked>
          %135 = arith.muli %134, %cst_8 : tensor<32x1xi32, #blocked>
          %136 = tt.addptr %15, %135 : tensor<32x1x!tt.ptr<bf16>, #blocked>, tensor<32x1xi32, #blocked>
          %137 = tt.broadcast %136 : tensor<32x1x!tt.ptr<bf16>, #blocked> -> tensor<32x64x!tt.ptr<bf16>, #blocked>
          %138 = tt.addptr %137, %40 : tensor<32x64x!tt.ptr<bf16>, #blocked>, tensor<32x64xi32, #blocked>
          %139 = ttg.memdesc_subview %42[%110, %c0_i32, %c0_i32] : !ttg.memdesc<2x32x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<32x64xbf16, #shared, #smem, mutable, 2x32x64>
          %140 = tt.splat %92 : i1 -> tensor<32x64xi1, #blocked>
          %141 = arith.andi %140, %132 : tensor<32x64xi1, #blocked>
          %142 = ttg.async_copy_global_to_local %138, %139 mask %141 other %cst_4 : tensor<32x64x!tt.ptr<bf16>, #blocked> -> <32x64xbf16, #shared, #smem, mutable, 2x32x64>
          %143 = ttg.async_commit_group %142
          scf.yield %107, %110, %95, %arg11, %143 : tensor<64x64xf32, #blocked2>, i32, i32, !ttg.async.token, !ttg.async.token
        }
        %78 = ttg.async_wait  {num = 0 : i32}
        ttg.local_dealloc %42 : !ttg.memdesc<2x32x64xbf16, #shared, #smem, mutable>
        ttg.local_dealloc %41 : !ttg.memdesc<2x64x32xbf16, #shared, #smem, mutable>
        %79 = tt.expand_dims %29 {axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi1, #blocked>
        %80 = tt.broadcast %79 : tensor<64x1xi1, #blocked> -> tensor<64x64xi1, #blocked>
        %81 = tt.broadcast %37 : tensor<1x64xi1, #blocked> -> tensor<64x64xi1, #blocked>
        %82 = arith.andi %80, %81 : tensor<64x64xi1, #blocked>
        %83 = tt.expand_dims %24 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
        %84 = arith.muli %83, %cst_9 : tensor<64x1xi32, #blocked>
        %85 = tt.addptr %16, %84 : tensor<64x1x!tt.ptr<bf16>, #blocked>, tensor<64x1xi32, #blocked>
        %86 = tt.broadcast %85 : tensor<64x1x!tt.ptr<bf16>, #blocked> -> tensor<64x64x!tt.ptr<bf16>, #blocked>
        %87 = tt.broadcast %39 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
        %88 = tt.addptr %86, %87 : tensor<64x64x!tt.ptr<bf16>, #blocked>, tensor<64x64xi32, #blocked>
        %89 = arith.truncf %77#0 : tensor<64x64xf32, #blocked2> to tensor<64x64xbf16, #blocked2>
        %90 = ttg.convert_layout %89 : tensor<64x64xbf16, #blocked2> -> tensor<64x64xbf16, #blocked>
        tt.store %88, %90, %82 : tensor<64x64x!tt.ptr<bf16>, #blocked>
        %91 = arith.addi %arg5, %c16_i32 : i32
        scf.yield %91 : i32
      }
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
2025-03-10 09:23:57,609 - ERROR - Error in backward_x kernel: PassManager::run failed
2025-03-10 09:23:57,609 - INFO - Falling back to PyTorch for grad_x
2025-03-10 09:23:57,656 - INFO - Computing grad_w with triton kernel
2025-03-10 09:23:58,017 - INFO - grad_w computation successful with triton
2025-03-10 09:23:58,039 - INFO - Gradient shapes - grad_x: torch.Size([256, 64]), grad_w: torch.Size([256, 64])
2025-03-10 09:23:58,039 - INFO - Running PyTorch reference implementation
/home/less/.conda/envs/tritondev/lib/python3.12/site-packages/torch/autograd/graph.py:824: UserWarning: Attempting to run cuBLAS, but there was no current CUDA context! Attempting to set the primary context... (Triggered internally at /pytorch/aten/src/ATen/cuda/CublasHandlePool.cpp:181.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
2025-03-10 09:23:58,298 - INFO - Comparing gradients with PyTorch reference
2025-03-10 09:23:58,365 - INFO - grad W compare: grad_w=tensor([[ 3.0625e+01, -7.5312e+00,  2.1375e+01,  ...,  8.8125e+00,
          2.1875e+00,  4.9062e+00],
        [-2.6625e+01, -1.6172e+00, -7.9688e+00,  ..., -2.1875e+01,
          1.7750e+01, -4.2812e+00],
        [-1.2250e+01, -2.7625e+01,  1.1375e+01,  ...,  2.7500e+00,
         -2.3000e+01,  8.8125e+00],
        ...,
        [-3.8594e+00, -8.6250e+00, -1.6625e+01,  ..., -8.6250e+00,
         -1.7125e+01,  4.5000e+00],
        [-1.5812e+01,  1.2188e+01, -5.5000e+00,  ...,  4.1250e+00,
          7.6875e+00,  1.3938e+01],
        [-2.3000e+01,  8.3750e+00,  8.1250e+00,  ..., -8.1787e-03,
          3.6750e+01,  3.0125e+01]], device='cuda:0', dtype=torch.bfloat16), w_autograd=tensor([[ 4.7461e-01,  2.1191e-01, -1.1484e+00,  ...,  1.6211e-01,
          1.3203e+00, -2.8906e-01],
        [ 1.7422e+00, -1.2695e-01, -2.4219e-01,  ...,  1.7334e-02,
          2.0410e-01, -2.1582e-01],
        [ 1.0000e+00,  5.5469e-01, -8.2422e-01,  ...,  2.4805e-01,
         -5.9375e-01,  2.1250e+00],
        ...,
        [-7.3047e-01,  1.3516e+00,  3.1250e-01,  ..., -8.8120e-04,
         -5.4688e-01, -5.8594e-01],
        [ 5.5859e-01, -1.7578e+00,  7.8125e-01,  ...,  5.5469e-01,
          1.1108e-02,  1.0547e+00],
        [ 4.5117e-01, -7.0312e-01,  1.5156e+00,  ...,  8.5938e-01,
         -2.2344e+00, -6.2500e-01]], device='cuda:0', dtype=torch.bfloat16,
       requires_grad=True)
2025-03-10 09:23:58,367 - INFO - grad X compare: grad_x=tensor([[  5.5000, -36.5000, -14.0625,  ...,  25.1250,  -0.9492, -13.5625],
        [  6.7812,  -8.4375,   9.0000,  ...,  -0.6016,  31.5000, -22.6250],
        [-23.2500, -15.7500,  -9.5625,  ...,   4.2188,  -4.0938,   3.3281],
        ...,
        [ 11.3750,   2.6875, -14.6250,  ...,   1.1953,  16.2500,  -1.9844],
        [-24.8750,  -3.5000,  25.6250,  ...,  10.6250, -42.7500,  15.0625],
        [  1.9375, -12.6875,  24.0000,  ...,  -1.9531,  16.2500,   2.5625]],
       device='cuda:0', dtype=torch.bfloat16, grad_fn=<CopyBackwards>), x_autograd=tensor([[ 0.0449,  0.5312, -0.1426,  ...,  1.6250,  1.3438, -0.6250],
        [-1.3828,  0.4238, -1.7422,  ...,  0.9023,  0.9609,  1.0312],
        [-0.4746, -1.2109,  0.1797,  ..., -1.0000, -0.3047, -1.0625],
        ...,
        [-0.3457, -0.6367,  0.5312,  ..., -1.1484,  0.0058, -2.9844],
        [ 0.5938,  0.3555, -0.1562,  ...,  1.3906, -0.1602,  1.0703],
        [ 1.2812, -0.7500, -0.7031,  ...,  0.7109, -0.5547, -0.1650]],
       device='cuda:0', dtype=torch.bfloat16, requires_grad=True)
2025-03-10 09:23:58,367 - INFO - Maximum gradient error - grad_x: 0.0078125, grad_w: 0.0078125
2025-03-10 09:23:58,367 - INFO - ✓ Gradients match the PyTorch reference
2025-03-10 09:23:58,367 - INFO - Test succeeded

with all close:
    2025-03-10 09:29:43,633 - ERROR - Error in backward_x kernel: PassManager::run failed
2025-03-10 09:29:43,633 - INFO - Falling back to PyTorch for grad_x
2025-03-10 09:29:43,684 - INFO - Computing grad_w with triton kernel
2025-03-10 09:29:43,689 - INFO - grad_w computation successful with triton
2025-03-10 09:29:43,711 - INFO - Gradient shapes - grad_x: torch.Size([256, 64]), grad_w: torch.Size([256, 64])
2025-03-10 09:29:43,711 - INFO - Running PyTorch reference implementation
/home/less/.conda/envs/tritondev/lib/python3.12/site-packages/torch/autograd/graph.py:824: UserWarning: Attempting to run cuBLAS, but there was no current CUDA context! Attempting to set the primary context... (Triggered internally at /pytorch/aten/src/ATen/cuda/CublasHandlePool.cpp:181.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
2025-03-10 09:29:43,975 - INFO - Comparing gradients with PyTorch reference
2025-03-10 09:29:43,996 - INFO - Maximum gradient error - grad_x: 0.0625, grad_w: 0.0625
2025-03-10 09:29:44,015 - INFO - Gradients allclose check - grad_x: True, grad_w: True
2025-03-10 09:29:44,015 - INFO - ✓ Gradients match the PyTorch reference (allclose check passed)
2025-03-10 09:29:44,016 - INFO - Test succeeded
"""
