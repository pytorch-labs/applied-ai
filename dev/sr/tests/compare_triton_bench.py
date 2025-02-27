import torch
import torchao
import triton
from tabulate import tabulate
from torchao.prototype.low_bit_optim.quant_utils import _fp32_to_bf16_sr
from stochastic_rounding_cuda import stochastic_round_bf16 as sr_bf16

compiled_sr_bf16 = torch.compile(_fp32_to_bf16_sr, fullgraph=True, dynamic=False)

results = []
for n in range(2, 10):
    size = 10 ** n
    x = torch.randn(size, device="cuda", dtype=torch.float32)

    torch_time = triton.testing.do_bench(lambda: compiled_sr_bf16(x))
    kernel_time = triton.testing.do_bench(lambda: sr_bf16(x))
    cast_time = triton.testing.do_bench(lambda: x.to(torch.bfloat16))

    results.append([
        f"{size:,}",
        torch_time,
        kernel_time,
        cast_time
    ])

print("\nPerformance Comparison (ms):")
print(tabulate(results,
              headers=['Size', 'Torch AO', 'Kernel', 'Direct Cast'],
              floatfmt='.4f'))
