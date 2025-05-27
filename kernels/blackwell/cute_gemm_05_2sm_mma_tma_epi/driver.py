#!/usr/bin/env python3
"""
Benchmark script for CuTe Blackwell GEMM vs PyTorch implementations
"""

import argparse
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    import sm100_gemm  # cute_blackwell_gemm

    CUTE_AVAILABLE = True
except ImportError:
    CUTE_AVAILABLE = False
    print(
        "Warning: cute_blackwell_gemm not available. Only PyTorch benchmarks will run."
    )

cute_blackwell_gemm = sm100_gemm.cute_blackwell_gemm


def check_gpu_capability():
    """Check if we have a compatible GPU"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    device_props = torch.cuda.get_device_properties(0)
    print(f"GPU: {device_props.name}")
    print(f"Compute Capability: {device_props.major}.{device_props.minor}")

    if device_props.major < 10:
        print("Warning: This benchmark is designed for Blackwell (SM100) GPUs")
        print("CuTe Blackwell GEMM may not work on this hardware")

    return device_props


def create_test_tensors(
    M: int, N: int, K: int, device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create test tensors with the required layouts"""
    # A: M×K, K-major (row-major), float16
    A = torch.randn(M, K, dtype=torch.float16, device=device)

    # B: N×K, K-major (column-major), float16
    B = torch.randn(N, K, dtype=torch.float16, device=device)

    # C: M×N, N-major (row-major), float32
    C = torch.randn(M, N, dtype=torch.float32, device=device)

    return A, B, C


def benchmark_pytorch_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    beta: float,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> Dict[str, float]:
    """Benchmark PyTorch GEMM implementations"""

    # Warmup
    for _ in range(num_warmup):
        _ = alpha * torch.mm(A.float(), B.float().t()) + beta * C

    torch.cuda.synchronize()

    # Benchmark torch.mm with manual scaling
    start_time = time.perf_counter()
    for _ in range(num_runs):
        result = alpha * torch.mm(A.float(), B.float().t()) + beta * C
    torch.cuda.synchronize()
    torch_mm_time = (time.perf_counter() - start_time) / num_runs

    # Benchmark torch.addmm (more optimized)
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(num_runs):
        result = torch.addmm(C, A.float(), B.float().t(), alpha=alpha, beta=beta)
    torch.cuda.synchronize()
    torch_addmm_time = (time.perf_counter() - start_time) / num_runs

    # Benchmark with mixed precision (if available)
    try:
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(num_runs):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                result = alpha * torch.mm(A, B.t()) + beta * C.half()
                result = result.float()
        torch.cuda.synchronize()
        torch_autocast_time = (time.perf_counter() - start_time) / num_runs
    except:
        torch_autocast_time = float("inf")

    return {
        "torch_mm": torch_mm_time,
        "torch_addmm": torch_addmm_time,
        "torch_autocast": torch_autocast_time,
    }


def benchmark_cute_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    beta: float,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> float:
    """Benchmark CuTe Blackwell GEMM"""
    if not CUTE_AVAILABLE:
        return float("inf")

    try:
        # Warmup
        for _ in range(num_warmup):
            _ = cute_blackwell_gemm(A, B, C, alpha, beta)

        torch.cuda.synchronize()

        # Benchmark
        start_time = time.perf_counter()
        for _ in range(num_runs):
            result = cute_blackwell_gemm(A, B, C, alpha, beta)
        torch.cuda.synchronize()

        return (time.perf_counter() - start_time) / num_runs

    except Exception as e:
        print(f"CuTe GEMM failed: {e}")
        return float("inf")


def verify_correctness(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    beta: float,
    use_cute_verification: bool = True,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    verbose: bool = False,
) -> bool:
    """Verify that CuTe GEMM produces correct results"""
    if not CUTE_AVAILABLE:
        return True

    try:
        if use_cute_verification:
            # Use the integrated CuTe verification
            _ = cute_blackwell_gemm.cute_blackwell_gemm_verify(
                A, B, C, alpha, beta, True, verbose
            )
            return True  # If no exception was thrown, verification passed
        else:
            # Fallback to PyTorch comparison
            ref_result = alpha * torch.mm(A.float(), B.float().t()) + beta * C
            cute_result = cute_blackwell_gemm.cute_blackwell_gemm(A, B, C, alpha, beta)

            is_close = torch.allclose(cute_result, ref_result, rtol=rtol, atol=atol)

            if not is_close and verbose:
                max_diff = torch.max(torch.abs(cute_result - ref_result)).item()
                mean_diff = torch.mean(torch.abs(cute_result - ref_result)).item()
                print(f"PyTorch verification failed!")
                print(f"Max difference: {max_diff:.6f}")
                print(f"Mean difference: {mean_diff:.6f}")

            return is_close

    except Exception as e:
        print(f"Correctness check failed with error: {e}")
        return False


def calculate_tflops(M: int, N: int, K: int, time_seconds: float) -> float:
    """Calculate TFLOPS for GEMM operation"""
    # GEMM operations: 2*M*N*K (multiply-add for each output element)
    flops = 2 * M * N * K
    tflops = flops / (time_seconds * 1e12)
    return tflops


def run_benchmark_suite(
    problem_sizes: List[Tuple[int, int, int]],
    alpha: float = 1.0,
    beta: float = 0.0,
    num_warmup: int = 10,
    num_runs: int = 100,
    verify: bool = True,
    verification_method: str = "cute",
    verbose_verify: bool = False,
) -> None:
    """Run comprehensive benchmark suite"""

    print("=" * 80)
    print("CuTe Blackwell GEMM Benchmark Suite")
    print("=" * 80)

    device_props = check_gpu_capability()
    print()

    # Check if tensors need to be aligned to tile sizes
    print("Note: Problem sizes should be multiples of 256 for optimal performance")
    print()

    results = []

    for M, N, K in problem_sizes:
        print(f"\nProblem Size: M={M}, N={N}, K={K}")
        print("-" * 50)

        # Create test tensors
        A, B, C = create_test_tensors(M, N, K)

        # Verify correctness first
        if verify and CUTE_AVAILABLE:
            print("Verifying correctness...", end=" ")
            verification_passed = False

            if verification_method in ["cute", "both"]:
                try:
                    is_correct = verify_correctness(
                        A,
                        B,
                        C,
                        alpha,
                        beta,
                        use_cute_verification=True,
                        verbose=verbose_verify,
                    )
                    print("✓ PASSED (CuTe)" if is_correct else "✗ FAILED (CuTe)")
                    verification_passed = is_correct

                    if not is_correct and verbose_verify:
                        print("Running detailed CuTe verification...")
                        verify_correctness(
                            A,
                            B,
                            C,
                            alpha,
                            beta,
                            use_cute_verification=True,
                            verbose=True,
                        )
                except Exception as e:
                    print(f"✗ FAILED (CuTe) with error: {e}")

            if verification_method in ["pytorch", "both"] or (
                verification_method == "cute" and not verification_passed
            ):
                if verification_method == "both" or not verification_passed:
                    print("Running PyTorch verification...", end=" ")

                is_correct = verify_correctness(
                    A,
                    B,
                    C,
                    alpha,
                    beta,
                    use_cute_verification=False,
                    verbose=verbose_verify,
                )
                print("✓ PASSED (PyTorch)" if is_correct else "✗ FAILED (PyTorch)")

                if verification_method != "both":
                    verification_passed = is_correct

            if not verification_passed:
                print("Skipping performance benchmark for this size")
                continue

        # Benchmark PyTorch implementations
        print("Benchmarking PyTorch implementations...")
        pytorch_times = benchmark_pytorch_gemm(
            A, B, C, alpha, beta, num_warmup, num_runs
        )

        # Benchmark CuTe implementation
        print("Benchmarking CuTe Blackwell GEMM...")
        cute_time = benchmark_cute_gemm(A, B, C, alpha, beta, num_warmup, num_runs)

        # Calculate performance metrics
        print("\nResults:")
        print(f"{'Method':<20} {'Time (ms)':<12} {'TFLOPS':<12} {'Speedup':<10}")
        print("-" * 60)

        best_pytorch_time = min(
            [t for t in pytorch_times.values() if t != float("inf")]
        )

        for method, exec_time in pytorch_times.items():
            if exec_time != float("inf"):
                tflops = calculate_tflops(M, N, K, exec_time)
                speedup = best_pytorch_time / exec_time
                print(
                    f"{method:<20} {exec_time*1000:<12.3f} {tflops:<12.1f} {speedup:<10.2f}x"
                )

        if cute_time != float("inf"):
            tflops = calculate_tflops(M, N, K, cute_time)
            speedup = best_pytorch_time / cute_time
            print(
                f"{'cute_blackwell':<20} {cute_time*1000:<12.3f} {tflops:<12.1f} {speedup:<10.2f}x"
            )
        else:
            print(f"{'cute_blackwell':<20} {'FAILED':<12} {'N/A':<12} {'N/A':<10}")

        results.append(
            {
                "problem_size": (M, N, K),
                "pytorch_times": pytorch_times,
                "cute_time": cute_time,
            }
        )

    # Summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    if CUTE_AVAILABLE:
        successful_runs = [r for r in results if r["cute_time"] != float("inf")]
        if successful_runs:
            avg_speedup = np.mean(
                [
                    min([t for t in r["pytorch_times"].values() if t != float("inf")])
                    / r["cute_time"]
                    for r in successful_runs
                ]
            )
            max_speedup = np.max(
                [
                    min([t for t in r["pytorch_times"].values() if t != float("inf")])
                    / r["cute_time"]
                    for r in successful_runs
                ]
            )
            min_speedup = np.min(
                [
                    min([t for t in r["pytorch_times"].values() if t != float("inf")])
                    / r["cute_time"]
                    for r in successful_runs
                ]
            )

            # Calculate best TFLOPS achieved
            best_tflops = np.max(
                [
                    calculate_tflops(*r["problem_size"], r["cute_time"])
                    for r in successful_runs
                ]
            )

            print(f"Successful runs: {len(successful_runs)}/{len(results)}")
            print(f"Average speedup over best PyTorch: {avg_speedup:.2f}x")
            print(f"Speedup range: {min_speedup:.2f}x - {max_speedup:.2f}x")
            print(f"Best TFLOPS achieved: {best_tflops:.1f}")

            # Show per-size breakdown
            print("\nPer-size performance breakdown:")
            print(f"{'Size (M×N×K)':<20} {'CuTe TFLOPS':<15} {'Speedup':<10}")
            print("-" * 50)
            for r in successful_runs:
                M, N, K = r["problem_size"]
                tflops = calculate_tflops(M, N, K, r["cute_time"])
                speedup = (
                    min([t for t in r["pytorch_times"].values() if t != float("inf")])
                    / r["cute_time"]
                )
                print(f"{M}×{N}×{K:<15} {tflops:<15.1f} {speedup:<10.2f}x")
        else:
            print("No successful CuTe runs")
    else:
        print("CuTe Blackwell GEMM not available for comparison")


def main():
    parser = argparse.ArgumentParser(description="Benchmark CuTe Blackwell GEMM")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=str,
        default=["512,1024,256", "1024,2048,512", "2048,4096,1024"],
        help="Problem sizes as M,N,K (e.g., 512,1024,256)",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha scaling factor")
    parser.add_argument("--beta", type=float, default=0.0, help="Beta scaling factor")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup runs")
    parser.add_argument(
        "--runs", type=int, default=100, help="Number of benchmark runs"
    )
    parser.add_argument(
        "--verification",
        choices=["cute", "pytorch", "both", "none"],
        default="none",
        help="Verification method: cute (CuTe native), pytorch (PyTorch comparison), both, none",
    )
    parser.add_argument(
        "--verbose-verify",
        action="store_true",
        help="Enable verbose verification output",
    )

    args = parser.parse_args()

    # Parse problem sizes
    problem_sizes = []
    for size_str in args.sizes:
        try:
            M, N, K = map(int, size_str.split(","))
            problem_sizes.append((M, N, K))
        except ValueError:
            print(f"Invalid problem size format: {size_str}. Use M,N,K format.")
            continue

    if not problem_sizes:
        print("No valid problem sizes provided")
        return

    # Run benchmark
    run_benchmark_suite(
        problem_sizes=problem_sizes,
        alpha=args.alpha,
        beta=args.beta,
        num_warmup=args.warmup,
        num_runs=args.runs,
        verify=args.verification != "none",
        verification_method=args.verification,
        verbose_verify=args.verbose_verify,
    )


if __name__ == "__main__":
    main()
