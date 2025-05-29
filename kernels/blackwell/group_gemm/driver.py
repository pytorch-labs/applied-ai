import numpy as np
import torch

try:
    import grouped_gemm_cuda
except ImportError:
    raise ImportError(
        "grouped_gemm_cuda C++ extension not found. Make sure it's properly installed."
    )


def test_grouped_gemm():
    """
    Test the grouped GEMM extension with sample data
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    # Check if we're on a Blackwell GPU
    props = torch.cuda.get_device_properties(0)
    if props.major != 10 or props.minor != 0:
        print(
            f"Warning: This extension requires Blackwell architecture (compute 10.0), "
            f"but found compute {props.major}.{props.minor}"
        )

    # Create test data for grouped GEMM
    num_groups = 5

    # Different problem sizes for each group
    problem_sizes = [
        (256, 512, 128),  # M, N, K
        (512, 256, 256),
        (128, 1024, 512),
        (1024, 128, 256),
        (256, 256, 256),
    ]

    A_tensors = []
    B_tensors = []
    C_tensors = []

    print("Creating test tensors...")
    for i, (M, N, K) in enumerate(problem_sizes):
        print(f"Group {i}: M={M}, N={N}, K={K}")

        # Create random tensors on GPU
        # Note: Using random values in the valid range for FP8 E4M3
        A = torch.randn(M, K, device=device, dtype=torch.float16) * 0.5
        A = A.to(torch.float8_e4m3fn)  # Convert to FP8 E4M3

        B = torch.randn(K, N, device=device, dtype=torch.float16) * 0.5
        B = B.to(torch.float8_e4m3fn)  # Convert to FP8 E4M3

        C = torch.randn(M, N, device=device, dtype=torch.float16) * 0.1

        A_tensors.append(A)
        B_tensors.append(B)
        C_tensors.append(C)

    # Test parameters
    alpha = 1.0
    beta = 0.5

    print("\nRunning grouped GEMM with 1SM config...")
    try:
        result_1sm = grouped_gemm_cuda.grouped_gemm(
            A_tensors,
            B_tensors,
            C_tensors,
            alpha=alpha,
            beta=beta,
            use_2sm_config=False,
            raster_order="M",
        )
        print(f"1SM result shape: {result_1sm.shape}")
        print(f"1SM result dtype: {result_1sm.dtype}")
    except Exception as e:
        print(f"1SM config failed: {e}")
        result_1sm = None

    print("\nRunning grouped GEMM with 2SM config...")
    try:
        result_2sm = grouped_gemm_cuda.grouped_gemm(
            A_tensors,
            B_tensors,
            C_tensors,
            alpha=alpha,
            beta=beta,
            use_2sm_config=True,
            raster_order="N",
        )
        print(f"2SM result shape: {result_2sm.shape}")
        print(f"2SM result dtype: {result_2sm.dtype}")
    except Exception as e:
        print(f"2SM config failed: {e}")
        result_2sm = None

    # Verify results by computing reference manually
    print("\nVerifying results...")
    if result_1sm is not None:
        offset = 0
        for i, (M, N, K) in enumerate(problem_sizes):
            # Extract result for this group
            group_result = result_1sm[offset : offset + M * N].view(M, N)

            # Compute reference: alpha * A @ B + beta * C
            A_fp16 = A_tensors[i].to(torch.float16)
            B_fp16 = B_tensors[i].to(torch.float16)
            reference = alpha * torch.matmul(A_fp16, B_fp16) + beta * C_tensors[i]

            # Check relative error
            rel_error = torch.max(
                torch.abs(group_result - reference) / (torch.abs(reference) + 1e-5)
            )
            print(f"Group {i} relative error: {rel_error.item():.6f}")

            offset += M * N

    print("\nGrouped GEMM test completed!")


def benchmark_grouped_gemm(num_groups=10, iterations=100):
    """
    Benchmark the grouped GEMM extension
    """
    device = torch.device("cuda")

    # Fixed problem size for benchmarking
    M, N, K = 1024, 1024, 1024

    A_tensors = []
    B_tensors = []
    C_tensors = []

    print(f"Creating benchmark tensors: {num_groups} groups of size {M}x{N}x{K}")

    for i in range(num_groups):
        A = torch.randn(M, K, device=device, dtype=torch.float16) * 0.5
        A = A.to(torch.float8_e4m3fn)

        B = torch.randn(K, N, device=device, dtype=torch.float16) * 0.5
        B = B.to(torch.float8_e4m3fn)

        C = torch.randn(M, N, device=device, dtype=torch.float16) * 0.1

        A_tensors.append(A)
        B_tensors.append(B)
        C_tensors.append(C)

    # Warmup
    print("Warming up...")
    for _ in range(10):
        _ = grouped_gemm_cuda.grouped_gemm(A_tensors, B_tensors, C_tensors)

    torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking {iterations} iterations...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iterations):
        _ = grouped_gemm_cuda.grouped_gemm(A_tensors, B_tensors, C_tensors)
    end_event.record()

    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_ms / iterations

    # Calculate TFLOPS
    total_flops = num_groups * 2 * M * N * K  # 2 FLOPs per multiply-add
    tflops = (total_flops * iterations / (elapsed_ms / 1000)) / 1e12

    print(f"Average time per iteration: {avg_time_ms:.3f} ms")
    print(f"Throughput: {tflops:.2f} TFLOPS")


if __name__ == "__main__":
    print("Testing Grouped GEMM Extension")
    print("=" * 40)

    # Run basic test
    test_grouped_gemm()

    # Run benchmark if requested
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        print("\n" + "=" * 40)
        print("Running Benchmark")
        print("=" * 40)
        benchmark_grouped_gemm()
import grouped_gemm_cuda
import numpy as np
import torch


def test_grouped_gemm():
    """
    Test the grouped GEMM extension with sample data
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    # Check if we're on a Blackwell GPU
    props = torch.cuda.get_device_properties(0)
    if props.major != 10 or props.minor != 0:
        print(
            f"Warning: This extension requires Blackwell architecture (compute 10.0), "
            f"but found compute {props.major}.{props.minor}"
        )

    # Create test data for grouped GEMM
    num_groups = 5

    # Different problem sizes for each group
    problem_sizes = [
        (256, 512, 128),  # M, N, K
        (512, 256, 256),
        (128, 1024, 512),
        (1024, 128, 256),
        (256, 256, 256),
    ]

    A_tensors = []
    B_tensors = []
    C_tensors = []

    print("Creating test tensors...")
    for i, (M, N, K) in enumerate(problem_sizes):
        print(f"Group {i}: M={M}, N={N}, K={K}")

        # Create random tensors on GPU
        # Note: Using random values in the valid range for FP8 E4M3
        A = torch.randn(M, K, device=device, dtype=torch.float16) * 0.5
        A = A.to(torch.float8_e4m3fn)  # Convert to FP8 E4M3

        B = torch.randn(K, N, device=device, dtype=torch.float16) * 0.5
        B = B.to(torch.float8_e4m3fn)  # Convert to FP8 E4M3

        C = torch.randn(M, N, device=device, dtype=torch.float16) * 0.1

        A_tensors.append(A)
        B_tensors.append(B)
        C_tensors.append(C)

    # Test parameters
    alpha = 1.0
    beta = 0.5

    print("\nRunning grouped GEMM with 1SM config...")
    try:
        result_1sm = grouped_gemm_cuda.grouped_gemm(
            A_tensors,
            B_tensors,
            C_tensors,
            alpha=alpha,
            beta=beta,
            use_2sm_config=False,
            raster_order="M",
        )
        print(f"1SM result shape: {result_1sm.shape}")
        print(f"1SM result dtype: {result_1sm.dtype}")
    except Exception as e:
        print(f"1SM config failed: {e}")
        result_1sm = None

    print("\nRunning grouped GEMM with 2SM config...")
    try:
        result_2sm = grouped_gemm_cuda.grouped_gemm(
            A_tensors,
            B_tensors,
            C_tensors,
            alpha=alpha,
            beta=beta,
            use_2sm_config=True,
            raster_order="N",
        )
        print(f"2SM result shape: {result_2sm.shape}")
        print(f"2SM result dtype: {result_2sm.dtype}")
    except Exception as e:
        print(f"2SM config failed: {e}")
        result_2sm = None

    # Verify results by computing reference manually
    print("\nVerifying results...")
    if result_1sm is not None:
        offset = 0
        for i, (M, N, K) in enumerate(problem_sizes):
            # Extract result for this group
            group_result = result_1sm[offset : offset + M * N].view(M, N)

            # Compute reference: alpha * A @ B + beta * C
            A_fp16 = A_tensors[i].to(torch.float16)
            B_fp16 = B_tensors[i].to(torch.float16)
            reference = alpha * torch.matmul(A_fp16, B_fp16) + beta * C_tensors[i]

            # Check relative error
            rel_error = torch.max(
                torch.abs(group_result - reference) / (torch.abs(reference) + 1e-5)
            )
            print(f"Group {i} relative error: {rel_error.item():.6f}")

            offset += M * N

    print("\nGrouped GEMM test completed!")


def benchmark_grouped_gemm(num_groups=10, iterations=100):
    """
    Benchmark the grouped GEMM extension
    """
    device = torch.device("cuda")

    # Fixed problem size for benchmarking
    M, N, K = 1024, 1024, 1024

    A_tensors = []
    B_tensors = []
    C_tensors = []

    print(f"Creating benchmark tensors: {num_groups} groups of size {M}x{N}x{K}")

    for i in range(num_groups):
        A = torch.randn(M, K, device=device, dtype=torch.float16) * 0.5
        A = A.to(torch.float8_e4m3fn)

        B = torch.randn(K, N, device=device, dtype=torch.float16) * 0.5
        B = B.to(torch.float8_e4m3fn)

        C = torch.randn(M, N, device=device, dtype=torch.float16) * 0.1

        A_tensors.append(A)
        B_tensors.append(B)
        C_tensors.append(C)

    # Warmup
    print("Warming up...")
    for _ in range(10):
        _ = grouped_gemm_cuda.grouped_gemm(A_tensors, B_tensors, C_tensors)

    torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking {iterations} iterations...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iterations):
        _ = grouped_gemm_cuda.grouped_gemm(A_tensors, B_tensors, C_tensors)
    end_event.record()

    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_ms / iterations

    # Calculate TFLOPS
    total_flops = num_groups * 2 * M * N * K  # 2 FLOPs per multiply-add
    tflops = (total_flops * iterations / (elapsed_ms / 1000)) / 1e12

    print(f"Average time per iteration: {avg_time_ms:.3f} ms")
    print(f"Throughput: {tflops:.2f} TFLOPS")


if __name__ == "__main__":
    print("Testing Grouped GEMM Extension")
    print("=" * 40)

    # Run basic test
    test_grouped_gemm()

    # Run benchmark if requested
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        print("\n" + "=" * 40)
        print("Running Benchmark")
        print("=" * 40)
        benchmark_grouped_gemm()
