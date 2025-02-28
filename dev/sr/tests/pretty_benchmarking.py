import torch
import torchao
import triton
from tabulate import tabulate

from torchao.prototype.low_bit_optim.quant_utils import _fp32_to_bf16_sr
from stochastic_rounding_cuda import stochastic_round_bf16 as sr_bf16

compiled_sr_bf16 = torch.compile(_fp32_to_bf16_sr, fullgraph=True, dynamic=False)


results = []
sizes = [1000, 100000, 10000000, 50000000, 1000000000, 1009000000, 1050000000, 2000000000, ]#  3000000000] # , 10 ** 11, 10 ** 12, 10 ** 13,
for size in sizes:
    #size = (10 ** 10)*n
    x = torch.randn(size, device="cuda", dtype=torch.float32)

    torch_time = triton.testing.do_bench(lambda: compiled_sr_bf16(x))
    kernel_time = triton.testing.do_bench(lambda: sr_bf16(x))
    cast_time = triton.testing.do_bench(lambda: x.to(torch.bfloat16))

    results.append([
    f"{size:,}",
    torch_time,
    kernel_time,
    cast_time,
    f"{(torch_time/cast_time):.3f}x",
    f"{(kernel_time/cast_time):.3f}x",
    f"{(torch_time/kernel_time):.3f}x",
    f"{(kernel_time/torch_time):.3f}x",
    f"{((torch_time-kernel_time)/torch_time*100):.1f}%"  # % speedup
])

print("\nPerformance Comparison:")
print(tabulate(results,
              headers=['Size', 'AO (ms)', 'Kernel (ms)', 'Cast (ms)',
                      'AO/Cast', 'Kernel/Cast', 'AO/Kernel', 'Kernel/AO',
                      'Kernel Speedup vs AO'],
              floatfmt='.3f'))
"""
    results.append([
    f"{size:,}",
    torch_time,
    kernel_time,
    cast_time,
    f"{(torch_time/cast_time):.3f}x",
    f"{(kernel_time/cast_time):.3f}x",
    f"{(torch_time/kernel_time):.3f}x",
    f"{(kernel_time/torch_time):.3f}x"  # Added Kernel/AO
])

print("\nPerformance Comparison:")
print(tabulate(results,
              headers=['Size', 'AO (ms)', 'Kernel (ms)', 'Cast (ms)',
                      'AO/Cast', 'Kernel/Cast', 'AO/Kernel', 'Kernel/AO'],
              floatfmt='.3f'))
"""
