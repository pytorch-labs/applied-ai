"""
Running test_backward_pass
Test setup - G: 2, M: 256, N: 128, K: 128
Input x shape: torch.Size([256, 128])
2025-03-09 23:07:11,891 - INFO - Weight w shape: torch.Size([256, 128])
2025-03-09 23:07:11,896 - INFO - Group sizes: tensor([128, 128], device='cuda:0', dtype=torch.int32)
2025-03-09 23:07:11,896 - INFO - Running forward pass
2025-03-09 23:07:12,338 - INFO - Forward result shape: torch.Size([256, 256])
2025-03-09 23:07:12,338 - INFO - Created gradient with shape: torch.Size([256, 256])
2025-03-09 23:07:12,338 - INFO - Running backward pass directly
2025-03-09 23:07:12,338 - INFO - Starting grouped_gemm_backward with fixed configurations
2025-03-09 23:07:12,338 - INFO - Using PyTorch fallback for grouped GEMM backward
2025-03-09 23:07:12,338 - INFO - PyTorch fallback dims - G: 2, M: 256, N: 128, K_x: 128, K_w: 128
/data/users/less/applied-ai/dev/triton_groupGEMM/clean/tgrouped_gemm_backwards.py:402: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=False):
/data/users/less/applied-ai/dev/triton_groupGEMM/clean/tgrouped_gemm_backwards.py:462: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=False):
2025-03-09 23:07:12,535 - INFO - Gradient shapes - grad_x: torch.Size([256, 128]), grad_w: torch.Size([256, 128])
2025-03-09 23:07:12,535 - INFO - Running PyTorch reference implementation
2025-03-09 23:07:12,823 - INFO - Comparing gradients with PyTorch reference
2025-03-09 23:07:12,909 - INFO - grad W compare: grad_w=tensor([[-18.0000,   2.0625, -10.5625,  ...,  -4.0938,  -0.3359,  15.0000],
        [-20.5000,  -2.8750, -21.2500,  ...,   0.5117,   6.7500,  -5.5000],
        [  8.1875, -19.8750,  20.3750,  ...,  -6.9062, -16.3750,  -4.5312],
        ...,
        [ 15.1250,   9.9375,   1.5938,  ...,   5.3750,   1.8047,   0.7500],
        [ 13.2500,   7.8750,   5.2188,  ...,  10.1875,   3.5625,   4.2500],
        [  4.5000,  -4.9688,  -4.9375,  ...,  -1.9297,  -5.5625,  15.5625]],
       device='cuda:0', dtype=torch.bfloat16, grad_fn=<CopySlices>), w_autograd=tensor([[-0.9648, -1.2266,  0.7656,  ..., -0.5977, -0.4961, -0.3125],
        [ 0.1611, -2.2188, -0.4863,  ..., -0.7305,  1.2812,  1.5859],
        [-0.0300, -0.2480, -1.2812,  ...,  1.5938, -1.1719, -1.7109],
        ...,
        [-0.5703,  0.1963, -0.7734,  ..., -0.1240,  0.8984,  2.8906],
        [-2.1250, -0.0073, -0.0050,  ...,  1.1016, -0.6094,  0.0231],
        [-1.0547,  0.1396,  1.1250,  ...,  0.6562,  0.5547,  0.2334]],
       device='cuda:0', dtype=torch.bfloat16, requires_grad=True)
2025-03-09 23:07:12,911 - INFO - grad X compare: grad_x=tensor([[  1.0234, -12.4375,  10.2500,  ...,  -5.9688,   8.0625,   6.6562],
        [-14.1250,  -2.3438,  -2.9219,  ...,  11.3750,   2.1094,  25.2500],
        [ 11.6875,   5.5000, -12.1250,  ...,   8.3125,  11.9375,  -0.5703],
        ...,
        [  9.8125,  -4.8438,  -6.8750,  ...,  -4.4375,  -4.3125,   9.5000],
        [  4.5000,  -7.6250,   0.3516,  ...,  -6.9062,  18.0000,  -0.5742],
        [  3.5000,   2.1875,   6.5938,  ..., -11.1250,  -2.7344, -10.9375]],
       device='cuda:0', dtype=torch.bfloat16, grad_fn=<CopySlices>), x_autograd=tensor([[-1.6094,  0.3242, -0.4316,  ..., -0.3457,  1.6641, -0.6523],
        [-0.9023, -0.1738, -0.5859,  ..., -0.0762,  0.7344, -1.2812],
        [-0.6016, -0.4980, -0.2256,  ...,  0.3477,  0.6133,  1.0000],
        ...,
        [-1.1406,  0.7891, -0.3008,  ...,  0.6875, -0.0532,  0.7656],
        [ 0.3184, -0.2793, -1.8281,  ..., -1.9141,  0.9727,  0.0172],
        [ 0.7461,  0.2480,  0.0796,  ...,  0.7305,  1.1094,  1.2734]],
       device='cuda:0', dtype=torch.bfloat16, requires_grad=True)
2025-03-09 23:07:12,911 - INFO - Maximum gradient error - grad_x: 0.00048828125, grad_w: 0.0625
2025-03-09 23:07:12,911 - ERROR - ✗ Gradient mismatch above tolerance threshold
2025-03-09 23:07:12,911 - INFO - Test succeeded
"""

"""
Running test_backward_pass
Test setup - G: 2, M: 512, N: 256, K: 64
Input x shape: torch.Size([512, 64])
2025-03-09 23:08:24,561 - INFO - Weight w shape: torch.Size([512, 64])
2025-03-09 23:08:24,587 - INFO - Group sizes: tensor([256, 256], device='cuda:0', dtype=torch.int32)
2025-03-09 23:08:24,587 - INFO - Running forward pass
2025-03-09 23:08:25,120 - INFO - Forward result shape: torch.Size([512, 512])
2025-03-09 23:08:25,120 - INFO - Created gradient with shape: torch.Size([512, 512])
2025-03-09 23:08:25,120 - INFO - Running backward pass directly
2025-03-09 23:08:25,120 - INFO - Starting grouped_gemm_backward with fixed configurations
2025-03-09 23:08:25,120 - INFO - Using PyTorch fallback for grouped GEMM backward
2025-03-09 23:08:25,120 - INFO - PyTorch fallback dims - G: 2, M: 512, N: 256, K_x: 64, K_w: 64
/data/users/less/applied-ai/dev/triton_groupGEMM/clean/tgrouped_gemm_backwards.py:402: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=False):
/data/users/less/applied-ai/dev/triton_groupGEMM/clean/tgrouped_gemm_backwards.py:462: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=False):
2025-03-09 23:08:25,462 - INFO - Gradient shapes - grad_x: torch.Size([512, 64]), grad_w: torch.Size([512, 64])
2025-03-09 23:08:25,462 - INFO - Running PyTorch reference implementation
2025-03-09 23:08:25,796 - INFO - Comparing gradients with PyTorch reference
2025-03-09 23:08:25,900 - INFO - grad W compare: grad_w=tensor([[ -3.5000,   3.0781,  -1.4609,  ...,   2.0312,  26.1250, -17.5000],
        [ -7.9688,  -3.5469,  -1.8047,  ..., -12.4375, -21.5000,  28.2500],
        [ 31.2500,  10.0625, -13.0625,  ..., -28.0000, -41.2500,   3.7812],
        ...,
        [-15.5000,  30.3750,   8.2500,  ...,  -9.2500,   1.9531,   1.6484],
        [-13.5625,   0.6016,   3.0156,  ..., -15.3125, -12.3125,   7.2812],
        [  0.5742,  11.1875,  -8.0625,  ...,   8.6250,  -7.9062,  12.2500]],
       device='cuda:0', dtype=torch.bfloat16, grad_fn=<CopySlices>), w_autograd=tensor([[-0.8711, -0.5664,  0.9492,  ..., -0.5781, -0.7734,  1.4531],
        [-0.0233, -1.5938, -1.5938,  ..., -1.4375, -2.0000,  1.2891],
        [-0.3848,  1.0312,  0.2246,  ...,  1.0625,  0.7617, -0.3379],
        ...,
        [ 0.6914,  0.7344,  0.5156,  ..., -1.3359,  0.0747, -0.0635],
        [ 0.1416, -0.3652,  0.3301,  ..., -0.9180,  0.8281, -0.2285],
        [-1.2656, -0.3984, -1.6953,  ..., -0.1602,  0.5000, -0.8867]],
       device='cuda:0', dtype=torch.bfloat16, requires_grad=True)
2025-03-09 23:08:25,902 - INFO - grad X compare: grad_x=tensor([[ -2.5312,  -9.7500,   9.9375,  ..., -25.8750,  18.1250,   0.5742],
        [ 15.5625,  -4.0625, -18.3750,  ...,  -6.0625, -17.5000,  13.4375],
        [ -8.5625,  -1.3047,  -8.8125,  ...,   8.8750,   4.7188,  13.6875],
        ...,
        [ -6.0625,   2.2031,  46.2500,  ...,   0.6445,   5.0625, -22.7500],
        [ -7.5312,  20.8750,   9.4375,  ..., -26.3750,  14.4375,  -1.4688],
        [ 19.6250,  14.4375,   4.2500,  ...,  -9.5625,  -2.4844, -20.1250]],
       device='cuda:0', dtype=torch.bfloat16, grad_fn=<CopySlices>), x_autograd=tensor([[ 0.3926,  0.9883,  0.7188,  ...,  0.6289,  0.5977,  0.2461],
        [-1.9844,  0.0344, -0.3359,  ...,  0.2773,  0.4375,  0.2852],
        [ 1.4766, -0.1992, -0.6016,  ..., -1.2656, -0.3652, -0.7227],
        ...,
        [ 0.4180,  0.1738,  1.8047,  ...,  0.4277,  1.2500,  0.5430],
        [-1.3750,  0.1250,  0.3613,  ...,  0.2354,  0.2002, -1.2969],
        [ 2.1875, -1.7500,  0.3438,  ..., -0.1504,  0.0544,  0.8750]],
       device='cuda:0', dtype=torch.bfloat16, requires_grad=True)
2025-03-09 23:08:25,902 - INFO - Maximum gradient error - grad_x: 0.125, grad_w: 0.125
2025-03-09 23:08:25,902 - ERROR - ✗ Gradient mismatch above tolerance threshold
2025-03-09 23:08:25,902 - INFO - Test succeeded
"""
