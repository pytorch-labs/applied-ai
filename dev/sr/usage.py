import torch
import stochastic_rounding_cuda

# Create input tensor
input_tensor = torch.randn(1000, device='cuda', dtype=torch.float32)

# Apply stochastic rounding
output_tensor = stochastic_rounding_cuda.stochastic_round_bf16(input_tensor)
print(f"Input tensor: {input_tensor}")
print(f"Output tensor: {output_tensor}")
print(f"Output tensor dtype: {output_tensor.dtype}")
print(f"Success!")
