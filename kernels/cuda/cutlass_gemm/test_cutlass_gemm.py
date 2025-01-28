import torch
from pingpong_gemm import cutlass_scaled_mm

m, k, n = 16, 4096, 4096
dtype = torch.float8_e4m3fn
out_dtype = torch.float16

a = torch.empty(m, k).normal_(mean=0.0, std=0.5).to(dtype=dtype, device='cuda')
bt = torch.empty(n, k).normal_(mean=0.0, std=0.5).to(dtype=dtype, device='cuda').t()
scale_a = torch.ones((1,)).to(dtype=torch.float32, device='cuda')
scale_b = torch.ones((1,)).to(dtype=torch.float32, device='cuda')
y = cutlass_scaled_mm(a, bt, scale_a, scale_b)
print(y)