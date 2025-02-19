import torch
import stochastic_rounding_cuda

# Create input tensor
input_tensor = torch.randn(12, device='cuda', dtype=torch.float32)

# Apply stochastic rounding
output_tensor = stochastic_rounding_cuda.stochastic_round_bf16(input_tensor)
print(f"Input tensor: {input_tensor}")
print(f"Output tensor: {output_tensor}")
print(f"Output tensor dtype: {output_tensor.dtype}")
print(f"Success!")

'''
# Test tensor
x = torch.tensor([9.8751e-01, -8.5288e-01, 1.6775e+00, -1.3683e+00,
                  4.0467e-01, 1.0759e-03, 2.8418e-01, -4.9392e-01,
                  8.7239e-01, -9.0545e-01, 1.1134e+00, 0],  # -2.6872e+00
                device='cuda')

# Convert to BF16
y = stochastic_rounding_cuda.stochastic_round_bf16(x)
print(f"Input: {x}")
print(f"Output: {y}")
'''
