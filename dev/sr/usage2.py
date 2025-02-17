import torch
import stochastic_rounding_cuda

# Test tensor
x = torch.tensor([9.8751e-01, -8.5288e-01, 1.6775e+00], device='cuda')

# Compare with regular rounding
y_normal = x.to(torch.bfloat16)
y_stochastic = stochastic_rounding_cuda.stochastic_round_bf16(x)

print(f"Input: {x}")
print(f"Normal BF16: {y_normal}")
print(f"Stochastic BF16: {y_stochastic}")
