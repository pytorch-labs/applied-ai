import torch
import torchao
import triton
from torchao.prototype.low_bit_optim.quant_utils import _fp32_to_bf16_sr
from stochastic_rounding_cuda import stochastic_round_bf16 as sr_bf16

compiled_sr_bf16 = torch.compile(_fp32_to_bf16_sr, fullgraph=True, dynamic=False)

print(f"\n ====== torch ao bf16 rounding: ======== \n")
for n in range(3, 10):
    x = torch.randn(10 ** n, device="cuda", dtype=torch.float32)

    def f():
        return compiled_sr_bf16(x)

    print(f"{x.size()}: \t\t{triton.testing.do_bench(f)}ms")
print(f"\n ====== kernel rounding: ======== \n")
for n in range(3, 10):
    x = torch.randn(10 ** n, device="cuda", dtype=torch.float32)

    def f():
        return sr_bf16(x)

    print(f"{x.size()}: \t\t{triton.testing.do_bench(f)}ms")

print(f"\n ====== direct conversion : ======== \n")
for n in range(3, 10):
    x = torch.randn(10 ** n, device="cuda", dtype=torch.float32)

    def f():
        return x.to(torch.bfloat16)

    print(f"{x.size()}: \t\t{triton.testing.do_bench(f)}ms")
