"""
Performance testing utilities for fused attention implementations.
from your_attention_module import attention
from attention_perf_test import run_benchmarks

# Run benchmarks with default settings
run_benchmarks(attention)

# Or customize the benchmark parameters
run_benchmarks(
    attention,
    batch=8,
    n_heads=16,
    head_dim=32,
    save_path="benchmark_results",
    print_data=True
)


"""

import torch
import triton
import triton.testing

# Check if Flash Attention is available
try:
    from flash_attn.flash_attn_interface import (
        flash_attn_qkvpacked_func as flash_attn_func,
    )

    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

# Check if PyTorch has FP8 support
TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")


def create_benchmark_configs(batch=4, n_heads=32, head_dim=64):
    """
    Creates benchmark configurations for different attention implementations.

    Args:
        batch (int): Batch size for testing
        n_heads (int): Number of attention heads
        head_dim (int): Dimension of each attention head

    Returns:
        list: List of benchmark configurations
    """
    configs = []
    for mode in ["fwd", "bwd"]:
        for causal in [True, False]:
            if mode == "bwd" and not causal:
                continue

            line_vals = ["triton-bf16"]
            line_names = ["Triton [BF16]"]
            styles = [("red", "-")]

            if TORCH_HAS_FP8:
                line_vals.append("triton-fp8")
                line_names.append("Triton [FP8]")
                styles.append(("blue", "-"))

            if HAS_FLASH:
                line_vals.append("flash")
                line_names.append("Flash-2")
                styles.append(("green", "-"))

            configs.append(
                triton.testing.Benchmark(
                    x_names=["N_CTX"],
                    x_vals=[2**i for i in range(10, 15)],
                    line_arg="provider",
                    line_vals=line_vals,
                    line_names=line_names,
                    styles=styles,
                    ylabel="TFLOPS",
                    plot_name=f"fused-attention-bf16-batch{batch}-head{n_heads}-d{head_dim}-{mode}-causal={causal}",
                    args={
                        "H": n_heads,
                        "BATCH": batch,
                        "HEAD_DIM": head_dim,
                        "mode": mode,
                        "causal": causal,
                    },
                )
            )
    return configs


@triton.testing.perf_report
def benchmark_attention(
    attention_fn, BATCH, H, N_CTX, HEAD_DIM, causal, mode, provider, device="cuda"
):
    """
    Benchmarks different attention implementations.

    Args:
        attention_fn: The attention implementation to benchmark
        BATCH (int): Batch size
        H (int): Number of heads
        N_CTX (int): Sequence length
        HEAD_DIM (int): Dimension of each head
        causal (bool): Whether to use causal attention
        mode (str): "fwd" for forward pass, "bwd" for backward pass
        provider (str): Implementation provider ("triton-fp16", "triton-fp8", or "flash")
        device (str): Device to run on

    Returns:
        float: Performance in TFLOPS
    """
    assert mode in ["fwd", "bwd"]
    dtype = torch.bfloat16

    if "triton" in provider:
        q = torch.randn(
            (BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True
        )
        k = torch.randn(
            (BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True
        )
        v = torch.randn(
            (BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True
        )

        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2).contiguous()
            v = v.permute(0, 1, 3, 2)
            v = v.to(torch.float8_e5m2)

        sm_scale = 1.3
        fn = lambda: attention_fn(q, k, v, causal, sm_scale)

        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)

    elif provider == "flash":
        qkv = torch.randn(
            (BATCH, N_CTX, 3, H, HEAD_DIM),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        fn = lambda: flash_attn_func(qkv, causal=causal)

        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)

    ms = triton.testing.do_bench(fn)

    # Calculate FLOPS
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul

    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)

    return total_flops * 1e-12 / (ms * 1e-3)


def run_benchmarks(attention_fn, save_path=".", print_data=True, **kwargs):
    """
    Runs all benchmarks for the given attention implementation.

    Args:
        attention_fn: The attention implementation to benchmark
        save_path (str): Path to save benchmark results
        print_data (bool): Whether to print benchmark data
        **kwargs: Additional arguments to pass to create_benchmark_configs
    """
    configs = create_benchmark_configs(**kwargs)
    benchmark_fn = triton.testing.perf_report(configs)(
        lambda *args, **kwargs: benchmark_attention(attention_fn, *args, **kwargs)
    )
    benchmark_fn.run(save_path=save_path, print_data=print_data)


if __name__ == "__main__":
    print(
        "This module provides performance testing utilities for fused attention implementations."
    )
    print("Import and use the provided functions in your own code.")
