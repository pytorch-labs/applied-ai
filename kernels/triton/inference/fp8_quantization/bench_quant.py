import time
from typing import Callable, Tuple

import numpy as np
import torch
from deepseek_quant import act_quant
from group_quant import verify_quantization  # act_quant_v2,


def generate_test_data(size: Tuple[int, ...], device: str = "cuda") -> torch.Tensor:
    """Generate test data with a mix of small and large values."""
    torch.manual_seed(2020)
    x = torch.randn(size, device=device)
    # Add some outliers to test scaling
    mask = torch.rand(size, device=device) > 0.95
    x[mask] *= 10
    return x


def measure_performance(
    func: Callable,
    input_tensor: torch.Tensor,
    num_warmup: int = 10,
    num_runs: int = 50,
) -> Tuple[float, float]:
    """
    Measure execution time and compute TFLOPS.

    Args:
        func: Quantization function to benchmark
        input_tensor: Input tensor
        num_warmup: Number of warmup iterations
        num_runs: Number of timing runs

    Returns:
        Tuple of (average_ms, TFLOPS)
    """
    # Warmup runs
    for _ in range(num_warmup):
        _ = func(input_tensor)

    torch.cuda.synchronize()

    # Timing runs
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = func(input_tensor)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time_ms = (end - start) * 1000 / num_runs

    # Calculate TFLOPS
    # For quantization, we roughly estimate:
    # - 1 abs() operation per element
    # - 1 division per element
    # - 1 comparison per element for the mask
    ops_per_element = 3
    total_ops = input_tensor.numel() * ops_per_element
    tflops = (total_ops * num_runs) / ((end - start) * 1e12)

    return avg_time_ms, tflops


def measure_accuracy(
    func: Callable, input_tensor: torch.Tensor, ref_func: Callable = None
) -> Tuple[float, float]:
    """
    Measure quantization accuracy metrics.

    Args:
        func: Quantization function to test
        input_tensor: Input tensor
        ref_func: Reference function for comparison (optional)

    Returns:
        Tuple of (RMSE, max_error)
    """
    y, s = func(input_tensor)

    # Dequantize
    if len(s.shape) == 1:
        s = s.view(-1, 1).expand(-1, 128)
    x_recon = y * s

    # Calculate error metrics
    abs_error = torch.abs(input_tensor - x_recon)
    rmse = torch.sqrt(torch.mean(torch.square(abs_error)))
    max_error = torch.max(abs_error)

    return rmse.item(), max_error.item()


def run_benchmark(sizes: list) -> None:
    """Run benchmark suite."""

    print("\nQuantization Benchmark Results")
    print("=" * 80)
    print(
        f"{'Size':>15} {'Method':>15} {'Time (ms)':>12} {'TFLOPS':>10} {'RMSE':>10} {'Max Err':>10}"
    )
    print("-" * 80)

    for size in sizes:
        # Generate test data
        x = generate_test_data(size)
        print(f"Testing {x=}, {str(size)}")

        # Test group_quant (chunked version)
        """try:
            time_chunk, tflops_chunk = measure_performance(act_quant_v2, x)
            rmse_chunk, max_err_chunk = measure_accuracy(act_quant_v2, x)
            print(
                f"{str(size):>15} {'Chunked':>15} {time_chunk:>12.3f} {tflops_chunk:>10.2f} "
                f"{rmse_chunk:>10.3e} {max_err_chunk:>10.3e}"
            )
        except Exception as e:
            print(f"{str(size):>15} {'Chunked':>15} Error: {str(e)}")
        """
        # Test deepseek_quant (baseline version)
        try:
            time_base, tflops_base = measure_performance(act_quant, x)
            rmse_base, max_err_base = measure_accuracy(act_quant, x)
            print(
                f"{str(size):>15} {'Baseline':>15} {time_base:>12.3f} {tflops_base:>10.2f} "
                f"{rmse_base:>10.3e} {max_err_base:>10.3e}"
            )
        except Exception as e:
            print(f"{str(size):>15} {'Baseline':>15} Error: {str(e)}")


if __name__ == "__main__":
    # Test various sizes
    sizes = [
        (1024, 1024),  # 1M elements
        (4096, 1024),  # 4M elements
        (16384, 1024),  # 16M elements
        (65536, 1024),  # 64M elements
    ]

    run_benchmark(sizes)
