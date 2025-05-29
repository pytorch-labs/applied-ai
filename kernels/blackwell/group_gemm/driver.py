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
        results_1sm = grouped_gemm_cuda.grouped_gemm(
            A_tensors,
            B_tensors,
            C_tensors,
            alpha=alpha,
            beta=beta,
            use_2sm_config=False,
            raster_order="M",
        )
        print(f"1SM returned {len(results_1sm)} result tensors")
        for i, result in enumerate(results_1sm):
            print(f"  Group {i} result shape: {result.shape}, dtype: {result.dtype}")
    except Exception as e:
        print(f"1SM config failed: {e}")
        results_1sm = None

    print("\nRunning grouped GEMM with 2SM config...")
    try:
        results_2sm = grouped_gemm_cuda.grouped_gemm(
            A_tensors,
            B_tensors,
            C_tensors,
            alpha=alpha,
            beta=beta,
            use_2sm_config=True,
            raster_order="N",
        )
        print(f"2SM returned {len(results_2sm)} result tensors")
        for i, result in enumerate(results_2sm):
            print(f"  Group {i} result shape: {result.shape}, dtype: {result.dtype}")
    except Exception as e:
        print(f"2SM config failed: {e}")
        results_2sm = None

    # Verify results by computing reference manually
    print("\nVerifying results...")
    if results_1sm is not None:
        max_errors = []
        for i, (M, N, K) in enumerate(problem_sizes):
            # Get result for this group
            group_result = results_1sm[i]

            # Verify shape
            assert group_result.shape == (M, N), f"Shape mismatch for group {i}"

            # Compute reference: alpha * A @ B + beta * C
            A_fp16 = A_tensors[i].to(torch.float16)
            B_fp16 = B_tensors[i].to(torch.float16)
            reference = alpha * torch.matmul(A_fp16, B_fp16) + beta * C_tensors[i]

            # Check relative error
            rel_error = torch.max(
                torch.abs(group_result - reference) / (torch.abs(reference) + 1e-5)
            )
            max_errors.append(rel_error.item())
            print(f"Group {i} relative error: {rel_error.item():.6f}")

        print(f"Average relative error: {np.mean(max_errors):.6f}")
        print(f"Max relative error: {np.max(max_errors):.6f}")

    # Test pre-allocated version if available
    if hasattr(grouped_gemm_cuda, "grouped_gemm_preallocated"):
        print("\nTesting pre-allocated version...")
        D_tensors = []
        for M, N, K in problem_sizes:
            D = torch.empty(M, N, device=device, dtype=torch.float16)
            D_tensors.append(D)

        try:
            results_prealloc = grouped_gemm_cuda.grouped_gemm_preallocated(
                A_tensors,
                B_tensors,
                C_tensors,
                D_tensors,
                alpha=alpha,
                beta=beta,
                use_2sm_config=False,
            )
            print("Pre-allocated version succeeded")
            # D_tensors should now contain the results
            for i, D in enumerate(D_tensors):
                print(f"  Group {i} result shape: {D.shape}")
        except Exception as e:
            print(f"Pre-allocated version failed: {e}")

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

    # Memory bandwidth estimate
    # Each group: read A (M*K), B (K*N), C (M*N), write D (M*N)
    # All in FP8 except C and D which are FP16
    bytes_per_group = (M * K + K * N) * 1 + (
        M * N
    ) * 2 * 2  # FP8 = 1 byte, FP16 = 2 bytes
    total_bytes = bytes_per_group * num_groups
    bandwidth_gb_s = (total_bytes * iterations / (elapsed_ms / 1000)) / 1e9
    print(f"Estimated memory bandwidth: {bandwidth_gb_s:.2f} GB/s")


def test_different_layouts():
    """
    Test different memory layouts and configurations
    """
    device = torch.device("cuda")

    # Test square matrices
    print("Testing square matrices...")
    sizes = [(128, 128, 128), (256, 256, 256), (512, 512, 512)]

    for M, N, K in sizes:
        A = torch.randn(M, K, device=device, dtype=torch.float16) * 0.5
        A = A.to(torch.float8_e4m3fn)

        B = torch.randn(K, N, device=device, dtype=torch.float16) * 0.5
        B = B.to(torch.float8_e4m3fn)

        C = torch.zeros(M, N, device=device, dtype=torch.float16)

        try:
            results = grouped_gemm_cuda.grouped_gemm([A], [B], [C])
            print(f"  {M}x{N}x{K}: Success, result shape: {results[0].shape}")
        except Exception as e:
            print(f"  {M}x{N}x{K}: Failed - {e}")

    # Test non-square matrices
    print("\nTesting non-square matrices...")
    sizes = [(128, 256, 512), (512, 128, 256), (256, 512, 128)]

    for M, N, K in sizes:
        A = torch.randn(M, K, device=device, dtype=torch.float16) * 0.5
        A = A.to(torch.float8_e4m3fn)

        B = torch.randn(K, N, device=device, dtype=torch.float16) * 0.5
        B = B.to(torch.float8_e4m3fn)

        C = torch.zeros(M, N, device=device, dtype=torch.float16)

        try:
            results = grouped_gemm_cuda.grouped_gemm([A], [B], [C])
            print(f"  {M}x{N}x{K}: Success, result shape: {results[0].shape}")
        except Exception as e:
            print(f"  {M}x{N}x{K}: Failed - {e}")


if __name__ == "__main__":
    print("Testing Grouped GEMM Extension")
    print("=" * 40)

    # Run basic test
    test_grouped_gemm()

    # Run additional tests
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "benchmark":
            print("\n" + "=" * 40)
            print("Running Benchmark")
            print("=" * 40)
            benchmark_grouped_gemm()
        elif sys.argv[1] == "layouts":
            print("\n" + "=" * 40)
            print("Testing Different Layouts")
            print("=" * 40)
            test_different_layouts()
