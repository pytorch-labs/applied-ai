import time
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import triton
import triton.language as tl
from deepseek_quant import act_quant
from group_quant import fast_quant_128 as fast_act_quant


@dataclass
class BenchmarkResult:
    size: int
    original_ms: float
    optimized_ms: float
    speedup: float
    max_diff: float
    scale_diff: float
    throughput_orig: float  # GB/s
    throughput_opt: float  # GB/s


class QuantizationBenchmark:
    def __init__(self, sizes: List[int] = None):
        if sizes is None:
            sizes = [1024, 4096, 16384, 65536, 262144, 1048576]
        self.sizes = sizes
        self.results: List[BenchmarkResult] = []

    def run_single_benchmark(
        self, x: torch.Tensor, num_runs: int = 100
    ) -> BenchmarkResult:
        n = x.numel()
        bytes_processed = n * 4  # float32 input

        # Warm up
        y_orig, s_orig = act_quant(x)
        y_fast, s_fast = fast_act_quant(x)
        torch.cuda.synchronize()

        # Benchmark original
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(num_runs):
            y_orig, s_orig = act_quant(x)
        end.record()
        torch.cuda.synchronize()
        orig_time = start.elapsed_time(end) / num_runs

        # Benchmark optimized
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(num_runs):
            y_fast, s_fast = fast_act_quant(x)
        end.record()
        torch.cuda.synchronize()
        opt_time = start.elapsed_time(end) / num_runs

        # Calculate throughput
        throughput_orig = (bytes_processed / 1e9) / (orig_time / 1000)  # GB/s
        throughput_opt = (bytes_processed / 1e9) / (opt_time / 1000)  # GB/s

        # Convert FP8 tensors to FP32 for comparison
        y_orig_fp32 = y_orig.to(torch.float32)
        y_fast_fp32 = y_fast.to(torch.float32)

        # Verify results
        max_diff = torch.max(torch.abs(y_orig_fp32 - y_fast_fp32)).item()
        scale_diff = torch.max(torch.abs(s_orig - s_fast)).item()

        return BenchmarkResult(
            size=n,
            original_ms=orig_time,
            optimized_ms=opt_time,
            speedup=orig_time / opt_time,
            max_diff=max_diff,
            scale_diff=scale_diff,
            throughput_orig=throughput_orig,
            throughput_opt=throughput_opt,
        )

    def run_benchmarks(self, num_runs: int = 100):
        self.results.clear()

        for size in self.sizes:
            try:
                x = torch.randn(size, device="cuda")
                result = self.run_single_benchmark(x, num_runs)
                self.results.append(result)

                print(f"\nSize: {size:,}")
                print(
                    f"Original: {result.original_ms:.3f} ms ({result.throughput_orig:.1f} GB/s)"
                )
                print(
                    f"Optimized: {result.optimized_ms:.3f} ms ({result.throughput_opt:.1f} GB/s)"
                )
                print(f"Speedup: {result.speedup:.2f}x")
                print(f"Max output difference: {result.max_diff:.6f}")
                print(f"Max scale difference: {result.scale_diff:.6f}")
            except RuntimeError as e:
                print(f"\nError running benchmark for size {size}: {str(e)}")
                continue

    def plot_results(self):
        if not self.results:
            raise ValueError("No benchmark results available. Run benchmarks first.")

        sizes = [r.size for r in self.results]
        orig_times = [r.original_ms for r in self.results]
        opt_times = [r.optimized_ms for r in self.results]
        speedups = [r.speedup for r in self.results]
        throughput_orig = [r.throughput_orig for r in self.results]
        throughput_opt = [r.throughput_opt for r in self.results]

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Execution Time
        ax1.plot(sizes, orig_times, "o-", label="Original")
        ax1.plot(sizes, opt_times, "o-", label="Optimized")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel("Input Size")
        ax1.set_ylabel("Time (ms)")
        ax1.set_title("Execution Time vs Input Size")
        ax1.grid(True)
        ax1.legend()

        # Plot 2: Speedup
        ax2.plot(sizes, speedups, "o-")
        ax2.set_xscale("log")
        ax2.set_xlabel("Input Size")
        ax2.set_ylabel("Speedup (x)")
        ax2.set_title("Speedup vs Input Size")
        ax2.grid(True)

        # Plot 3: Throughput
        ax3.plot(sizes, throughput_orig, "o-", label="Original")
        ax3.plot(sizes, throughput_opt, "o-", label="Optimized")
        ax3.set_xscale("log")
        ax3.set_xlabel("Input Size")
        ax3.set_ylabel("Throughput (GB/s)")
        ax3.set_title("Memory Throughput vs Input Size")
        ax3.grid(True)
        ax3.legend()

        # Plot 4: Numerical Accuracy
        max_diffs = [r.max_diff for r in self.results]
        scale_diffs = [r.scale_diff for r in self.results]
        ax4.plot(sizes, max_diffs, "o-", label="Output Diff")
        ax4.plot(sizes, scale_diffs, "o-", label="Scale Diff")
        ax4.set_xscale("log")
        ax4.set_yscale("log")
        ax4.set_xlabel("Input Size")
        ax4.set_ylabel("Maximum Difference")
        ax4.set_title("Numerical Accuracy vs Input Size")
        ax4.grid(True)
        ax4.legend()

        plt.tight_layout()
        return fig


def run_complete_benchmark(save_path: str = "benchmark_results.png"):
    """
    Run benchmarks and save results to a file.

    Args:
        save_path: Path where to save the plot. Default is 'benchmark_results.png'
                  Supported formats include .png, .jpg, .pdf based on file extension

    Returns:
        Tuple of (benchmark_results, matplotlib_figure)
    """
    # Test sizes from 1K to 1M
    sizes = [2**n for n in range(10, 21)]

    try:
        # Create and run benchmarks
        benchmark = QuantizationBenchmark(sizes)
        benchmark.run_benchmarks()

        # Plot results
        fig = benchmark.plot_results()

        # Save plot with high DPI for better quality
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nBenchmark plot saved to: {save_path}")

        return benchmark.results, fig
    except Exception as e:
        print(f"Error running benchmark: {str(e)}")
        return None, None


if __name__ == "__main__":
    results, fig = run_complete_benchmark()
