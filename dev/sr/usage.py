import torch
import stochastic_round_cuda as sr_cuda

x = torch.randn(1000, device='cuda')
x_bf16 = sr_cuda.forward(x)
print(f"fp32: {x[0:10]}")
print(f"bf16: {x_bf16[0:10]}")
print(f"Success!")
