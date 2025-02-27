import torch
from torch.profiler import profile, record_function, ProfilerActivity
import stochastic_rounding_cuda

def profile_sr_bf16():
    x = torch.randn(100_000, device="cuda", dtype=torch.float32)

    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/sr_profile')
    ) as prof:
        with record_function("stochastic_round_bf16"):
            _ = stochastic_rounding_cuda.stochastic_round_bf16(x)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

if __name__ == '__main__':
    profile_sr_bf16()
