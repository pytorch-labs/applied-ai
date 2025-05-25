# driver.py - High-level Python interface with SM100 GEMM implementations

import torch

try:
    import sm100_gemm  # The compiled extension
except ImportError:
    print("❌ SM100 not ready!")
    raise ImportError(
        "SM100 not ready! Please build the extension using `python setup.py install`"
    )


def check_sm100_compatibility():
    """Check if SM100 is supported and available"""
    compile_support = sm100_gemm.is_sm100_supported()
    device_support = sm100_gemm.check_sm100_device()

    info = sm100_gemm.get_device_info()
    major, minor, compile_flag, device_flag = info.tolist()

    print(f"Device compute capability: {major}.{minor}")
    print(f"Compile-time SM100 support: {bool(compile_flag)}")
    print(f"Runtime SM100 device support: {bool(device_flag)}")

    if not compile_support:
        print(
            "❌ SM100 support not compiled in. Rebuild with CUTLASS_ARCH_MMA_SM100_SUPPORTED"
        )
    elif not device_support:
        print("❌ Current GPU does not support SM100 (need compute capability 10.0a)")
    else:
        print("✅ SM100 with TMA Multicast ready!")

    return compile_support and device_support


def sm100_gemm_f16(A, B, C=None, alpha=1.0, beta=0.0, check_alignment=True):
    """
    Perform GEMM using SM100 optimized kernel with TMA Multicast: D = alpha * A @ B^T + beta * C

    TMA Multicast enables efficient data sharing across multiple CTAs in a cluster:
    - A matrix data is multicast along the N dimension (reduces memory bandwidth)
    - B matrix data is multicast along the M dimension (improves cache efficiency)
    - Automatic cluster size optimization based on problem size

    Args:
        A (torch.Tensor): Input tensor A of shape (M, K), dtype=torch.float16
        B (torch.Tensor): Input tensor B of shape (N, K), dtype=torch.float16
        C (torch.Tensor, optional): Input tensor C of shape (M, N), dtype=torch.float32
                                   If None, creates zero tensor
        alpha (float): Scaling factor for A @ B^T
        beta (float): Scaling factor for C
        check_alignment (bool): Whether to check and suggest aligned dimensions

    Returns:
        torch.Tensor: Output tensor D of shape (M, N), dtype=torch.float32

    Note:
        - Uses TMA Multicast for maximum efficiency on large matrices
        - A and B are K-major (transposed in BLAS terms)
        - C and D are N-major (row-major)
        - All tensors must be on CUDA
        - M must be multiple of 128, N multiple of 256, K multiple of 64
    """

    # Input validation
    assert A.dtype == torch.float16, f"A must be float16, got {A.dtype}"
    assert B.dtype == torch.float16, f"B must be float16, got {B.dtype}"
    assert A.is_cuda and B.is_cuda, "A and B must be on CUDA"
    assert A.is_contiguous() and B.is_contiguous(), "A and B must be contiguous"

    M, K = A.shape
    N, K_B = B.shape
    assert K == K_B, f"Inner dimensions must match: A.shape[1]={K}, B.shape[1]={K_B}"

    # Check or fix alignment requirements
    if check_alignment:
        aligned_M, aligned_N, aligned_K = sm100_gemm.get_aligned_shape(M, N, K)

        if M != aligned_M or N != aligned_N or K != aligned_K:
            print(f"Warning: Dimensions ({M}, {N}, {K}) not aligned for SM100")
            print(
                f"Suggested aligned dimensions: ({aligned_M}, {aligned_N}, {aligned_K})"
            )
            print("Consider padding tensors or use create_aligned_tensors()")

    # Strict alignment check
    assert (
        M % sm100_gemm.MMA_TILE_M == 0
    ), f"M={M} must be multiple of {sm100_gemm.MMA_TILE_M}"
    assert (
        N % sm100_gemm.MMA_TILE_N == 0
    ), f"N={N} must be multiple of {sm100_gemm.MMA_TILE_N}"
    assert (
        K % sm100_gemm.MMA_TILE_K == 0
    ), f"K={K} must be multiple of {sm100_gemm.MMA_TILE_K}"

    # Create C if not provided
    if C is None:
        C = torch.zeros(M, N, dtype=torch.float32, device=A.device)
    else:
        assert C.dtype == torch.float32, f"C must be float32, got {C.dtype}"
        assert C.is_cuda, "C must be on CUDA"
        assert C.is_contiguous(), "C must be contiguous"
        assert C.shape == (
            M,
            N,
        ), f"C shape {C.shape} must match output shape ({M}, {N})"

    # Call the extension (uses TMA Multicast internally)
    return sm100_gemm.sm100_gemm_f16(A, B, C, alpha, beta)


def create_aligned_tensors(
    M, N, K, device="cuda", dtype_AB=torch.float16, dtype_C=torch.float32
):
    """
    Create properly aligned tensors for SM100 GEMM with TMA Multicast

    Returns:
        tuple: (A, B, C) tensors with aligned dimensions optimized for multicast
    """
    aligned_M, aligned_N, aligned_K = sm100_gemm.get_aligned_shape(M, N, K)

    A = torch.zeros(aligned_M, aligned_K, dtype=dtype_AB, device=device)
    B = torch.zeros(aligned_N, aligned_K, dtype=dtype_AB, device=device)
    C = torch.zeros(aligned_M, aligned_N, dtype=dtype_C, device=device)

    return A, B, C


def optimize_cluster_size_for_problem(M, N, K):
    """
    Suggest optimal cluster configuration for TMA multicast based on problem size

    Args:
        M, N, K: Matrix dimensions

    Returns:
        tuple: (cluster_m, cluster_n) optimal cluster shape
    """
    # Memory requirements in GB
    memory_gb = (M * K * 2 + N * K * 2 + M * N * 4 * 2) / (1024**3)

    # For very large problems, use larger clusters
    if memory_gb > 8.0 and M >= 4096 and N >= 4096:
        return (4, 4)  # 4x4 cluster for massive matrices
    elif memory_gb > 2.0 and M >= 2048 and N >= 2048:
        return (2, 4)  # 2x4 cluster for large matrices
    elif M >= 1024 and N >= 1024:
        return (2, 2)  # 2x2 cluster for medium matrices
    else:
        return (1, 1)  # Single CTA for small matrices


def benchmark_tma_multicast_scaling(base_size=512, max_size=4096, num_trials=20):
    """
    Benchmark TMA multicast performance scaling with different cluster sizes
    """
    print("=== TMA Multicast Scaling Analysis ===")
    print("Testing how TMA multicast performance scales with problem size")

    sizes = []
    current = base_size
    while current <= max_size:
        # Ensure alignment
        M = (
            (current + sm100_gemm.MMA_TILE_M - 1) // sm100_gemm.MMA_TILE_M
        ) * sm100_gemm.MMA_TILE_M
        N = (
            (current + sm100_gemm.MMA_TILE_N - 1) // sm100_gemm.MMA_TILE_N
        ) * sm100_gemm.MMA_TILE_N
        K = (
            (current + sm100_gemm.MMA_TILE_K - 1) // sm100_gemm.MMA_TILE_K
        ) * sm100_gemm.MMA_TILE_K
        sizes.append((M, N, K))
        current *= 2

    results = []

    for M, N, K in sizes:
        try:
            print(f"\nTesting size: ({M}, {N}, {K})")

            # Calculate memory requirements
            memory_gb = (M * K * 2 + N * K * 2 + M * N * 4 * 2) / (1024**3)
            print(f"Memory requirement: {memory_gb:.2f} GB")

            # Get optimal cluster size
            cluster_m, cluster_n = optimize_cluster_size_for_problem(M, N, K)
            print(f"Optimal cluster: {cluster_m}x{cluster_n}")

            if memory_gb > 16:  # Skip if too large
                print("⚠️  Skipping due to memory constraints")
                continue

            # Create tensors
            A, B, C = create_aligned_tensors(M, N, K)
            A[:M, :K].normal_(0, 0.1)
            B[:N, :K].normal_(0, 0.1)
            C[:M, :N].normal_(0, 0.1)

            # Warmup
            for _ in range(3):
                result = sm100_gemm_f16(A, B, C.clone(), check_alignment=False)

            torch.cuda.synchronize()

            # Benchmark
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(num_trials):
                result = sm100_gemm_f16(A, B, C.clone(), check_alignment=False)
            end.record()

            torch.cuda.synchronize()
            avg_time = start.elapsed_time(end) / num_trials

            # Calculate performance
            flops = 2 * M * N * K
            tflops = flops / (avg_time * 1e-3) / 1e12
            bandwidth = memory_gb / (avg_time * 1e-3)  # GB/s

            results.append(
                {
                    "size": (M, N, K),
                    "memory_gb": memory_gb,
                    "cluster": (cluster_m, cluster_n),
                    "time_ms": avg_time,
                    "tflops": tflops,
                    "bandwidth_gbs": bandwidth,
                }
            )

            print(f"✅ Time: {avg_time:.2f} ms")
            print(f"✅ Performance: {tflops:.2f} TFLOPS")
            print(f"✅ Bandwidth: {bandwidth:.1f} GB/s")
            print(
                f" TMA Multicast cluster {cluster_m}x{cluster_n} optimized for this size!"
            )

        except torch.cuda.OutOfMemoryError:
            print(f"❌ Out of memory for size ({M}, {N}, {K})")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            break

    # Print scaling analysis
    if len(results) >= 2:
        print(f"\n=== TMA Multicast Scaling Analysis ===")
        print(
            "Size (M,N,K) | Memory(GB) | Cluster | Time(ms) | TFLOPS | Bandwidth(GB/s)"
        )
        print("-" * 80)
        for r in results:
            M, N, K = r["size"]
            print(
                f"({M:4d},{N:4d},{K:3d}) | {r['memory_gb']:8.2f} | {r['cluster'][0]}x{r['cluster'][1]}     | {r['time_ms']:6.2f}  | {r['tflops']:6.2f} | {r['bandwidth_gbs']:12.1f}"
            )

        # Calculate scaling efficiency
        base_tflops = results[0]["tflops"]
        print(f"\nScaling efficiency (vs smallest size):")
        for r in results:
            scale_factor = (r["size"][0] * r["size"][1] * r["size"][2]) / (
                results[0]["size"][0] * results[0]["size"][1] * results[0]["size"][2]
            )
            efficiency = (r["tflops"] / base_tflops) / scale_factor * 100
            print(f"  Size {r['size']}: {efficiency:.1f}% efficiency")

    return results


def benchmark_sm100_vs_torch(M=1024, N=2048, K=512, num_warmup=10, num_trials=50):
    """
    Benchmark SM100 GEMM with TMA Multicast against PyTorch's native GEMM
    """
    # Ensure dimensions are aligned
    M = (
        (M + sm100_gemm.MMA_TILE_M - 1) // sm100_gemm.MMA_TILE_M
    ) * sm100_gemm.MMA_TILE_M
    N = (
        (N + sm100_gemm.MMA_TILE_N - 1) // sm100_gemm.MMA_TILE_N
    ) * sm100_gemm.MMA_TILE_N
    K = (
        (K + sm100_gemm.MMA_TILE_K - 1) // sm100_gemm.MMA_TILE_K
    ) * sm100_gemm.MMA_TILE_K

    cluster_m, cluster_n = optimize_cluster_size_for_problem(M, N, K)

    print(f"Benchmarking GEMM with TMA Multicast for shape: ({M}, {N}, {K})")
    print(f"Optimal cluster configuration: {cluster_m}x{cluster_n}")

    # Check SM100 availability
    if not check_sm100_compatibility():
        print("SM100 not available, skipping benchmark")
        return None

    # Create test tensors
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(N, K, dtype=torch.float16, device="cuda")
    C = torch.randn(M, N, dtype=torch.float32, device="cuda")

    # PyTorch baseline (using mixed precision)
    A_fp32 = A.float()
    B_fp32 = B.float()

    # Warmup
    for _ in range(num_warmup):
        # PyTorch GEMM
        torch_result = torch.addmm(C, A_fp32, B_fp32.T)

        # SM100 GEMM with TMA Multicast
        sm100_result = sm100_gemm_f16(A, B, C.clone(), check_alignment=False)

    torch.cuda.synchronize()

    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_trials):
        torch_result = torch.addmm(C, A_fp32, B_fp32.T)
    end.record()
    torch.cuda.synchronize()
    torch_time = start.elapsed_time(end) / num_trials

    # Benchmark SM100 with TMA Multicast
    start.record()
    for _ in range(num_trials):
        sm100_result = sm100_gemm_f16(A, B, C.clone(), check_alignment=False)
    end.record()
    torch.cuda.synchronize()
    sm100_time = start.elapsed_time(end) / num_trials

    # Check correctness
    max_diff = torch.max(torch.abs(torch_result - sm100_result))
    rel_error = max_diff / torch.max(torch.abs(torch_result))

    # Calculate FLOPS and bandwidth
    flops = 2 * M * N * K  # Multiply-add operations
    torch_tflops = flops / (torch_time * 1e-3) / 1e12
    sm100_tflops = flops / (sm100_time * 1e-3) / 1e12

    # Memory bandwidth analysis
    memory_bytes = M * K * 2 + N * K * 2 + M * N * 4 * 2  # Input + output tensors
    sm100_bandwidth = memory_bytes / (sm100_time * 1e-3) / 1e9  # GB/s

    print(f"PyTorch time: {torch_time:.3f} ms ({torch_tflops:.2f} TFLOPS)")
    print(f"SM100+TMA Multicast time: {sm100_time:.3f} ms ({sm100_tflops:.2f} TFLOPS)")
    print(f"Speedup: {torch_time/sm100_time:.2f}x")
    print(f"Memory bandwidth: {sm100_bandwidth:.1f} GB/s")
    print(f"Max difference: {max_diff:.6f}")
    print(f"Relative error: {rel_error:.6f}")

    return {
        "torch_time": torch_time,
        "sm100_time": sm100_time,
        "speedup": torch_time / sm100_time,
        "torch_tflops": torch_tflops,
        "sm100_tflops": sm100_tflops,
        "bandwidth_gbs": sm100_bandwidth,
        "cluster_shape": (cluster_m, cluster_n),
        "max_diff": max_diff.item(),
        "rel_error": rel_error.item(),
    }


# Linear layer implementation with TMA Multicast
class SM100Linear(torch.nn.Module):
    """
    Linear layer using SM100 GEMM with TMA Multicast for forward pass
    Optimized for large batch sizes and feature dimensions
    """

    def __init__(self, in_features, out_features, bias=True, device="cuda"):
        super().__init__()

        # Align dimensions
        self.orig_in_features = in_features
        self.orig_out_features = out_features

        aligned_in = (
            (in_features + sm100_gemm.MMA_TILE_K - 1) // sm100_gemm.MMA_TILE_K
        ) * sm100_gemm.MMA_TILE_K
        aligned_out = (
            (out_features + sm100_gemm.MMA_TILE_N - 1) // sm100_gemm.MMA_TILE_N
        ) * sm100_gemm.MMA_TILE_N

        self.in_features = aligned_in
        self.out_features = aligned_out

        # Parameters (with padding)
        self.weight = torch.nn.Parameter(
            torch.randn(aligned_out, aligned_in, dtype=torch.float16, device=device)
            * 0.1
        )

        if bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(aligned_out, dtype=torch.float32, device=device)
            )
        else:
            self.register_parameter("bias", None)

        print(
            f"SM100Linear: {in_features} -> {out_features} (aligned: {aligned_in} -> {aligned_out})"
        )

    def forward(self, x):
        # Pad input if necessary
        batch_size = x.size(0)

        # Align batch size
        aligned_batch = (
            (batch_size + sm100_gemm.MMA_TILE_M - 1) // sm100_gemm.MMA_TILE_M
        ) * sm100_gemm.MMA_TILE_M

        if x.size(1) != self.in_features or batch_size != aligned_batch:
            x_padded = torch.zeros(
                aligned_batch, self.in_features, dtype=torch.float16, device=x.device
            )
            x_padded[:batch_size, : self.orig_in_features] = x
            x = x_padded

        # Prepare bias
        if self.bias is not None:
            C = (
                self.bias.unsqueeze(0)
                .expand(aligned_batch, self.out_features)
                .contiguous()
            )
            beta = 1.0
        else:
            C = torch.zeros(
                aligned_batch, self.out_features, dtype=torch.float32, device=x.device
            )
            beta = 0.0

        # SM100 GEMM with TMA Multicast: output = x @ weight^T + bias
        output = sm100_gemm_f16(
            x, self.weight, C, alpha=1.0, beta=beta, check_alignment=False
        )

        # Remove padding
        return output[:batch_size, : self.orig_out_features]


def demonstrate_multicast_benefits():
    """
    Demonstrate the benefits of TMA multicast for different scenarios
    """
    print("=== TMA Multicast Benefits Demonstration ===")

    scenarios = [
        {
            "name": "Large Transformer FFN",
            "description": "Large feed-forward network typical in modern transformers",
            "M": 2048,
            "N": 8192,
            "K": 2048,
            "expected_benefit": "High - large matrices benefit most from multicast",
        },
        {
            "name": "Attention QKV Projection",
            "description": "Multi-head attention query/key/value projections",
            "M": 1024,
            "N": 3072,
            "K": 1024,
            "expected_benefit": "Medium - moderate size with good multicast potential",
        },
        {
            "name": "Small MLP Layer",
            "description": "Small MLP layer typical in smaller models",
            "M": 512,
            "N": 1024,
            "K": 512,
            "expected_benefit": "Low-Medium - smaller matrices benefit less from multicast",
        },
    ]

    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        print(f"Description: {scenario['description']}")
        print(
            f"Matrix dimensions: M={scenario['M']}, N={scenario['N']}, K={scenario['K']}"
        )
        print(f"Expected benefit: {scenario['expected_benefit']}")

        try:
            # Align dimensions
            M = (
                (scenario["M"] + sm100_gemm.MMA_TILE_M - 1) // sm100_gemm.MMA_TILE_M
            ) * sm100_gemm.MMA_TILE_M
            N = (
                (scenario["N"] + sm100_gemm.MMA_TILE_N - 1) // sm100_gemm.MMA_TILE_N
            ) * sm100_gemm.MMA_TILE_N
            K = (
                (scenario["K"] + sm100_gemm.MMA_TILE_K - 1) // sm100_gemm.MMA_TILE_K
            ) * sm100_gemm.MMA_TILE_K

            # Get optimal cluster size
            cluster_m, cluster_n = optimize_cluster_size_for_problem(M, N, K)
            print(f"Optimal cluster configuration: {cluster_m}x{cluster_n}")

            # Run benchmark
            result = benchmark_sm100_vs_torch(M, N, K, num_warmup=5, num_trials=20)

            if result:
                print(f"Speedup vs PyTorch: {result['speedup']:.2f}x")
                print(f"Performance: {result['sm100_tflops']:.2f} TFLOPS")
                print(f"Memory bandwidth: {result['bandwidth_gbs']:.1f} GB/s")

        except Exception as e:
            print(f"Error running benchmark: {e}")
            continue

    return scenarios


def main():
    """
    Main entry point for testing and benchmarking SM100 GEMM with TMA Multicast
    """
    print("=== SM100 GEMM with TMA Multicast ===")

    # Check if SM100 is available
    if not check_sm100_compatibility():
        print("SM100 not available, exiting")
        return

    # Run benchmarks
    print("\n1. Benchmarking SM100 GEMM with TMA Multicast vs PyTorch")
    benchmark_sm100_vs_torch()

    print("\n2. Analyzing TMA Multicast scaling with problem size")
    benchmark_tma_multicast_scaling()

    print("\n3. Demonstrating TMA Multicast benefits for different scenarios")
    demonstrate_multicast_benefits()


if __name__ == "__main__":
    main()
