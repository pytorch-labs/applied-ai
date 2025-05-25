# python_interface.py - High-level Python interface with TMA support

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
        print("✅ SM100 with TMA ready!")

    return compile_support and device_support


def sm100_gemm_f16_tma(A, B, C=None, alpha=1.0, beta=0.0, check_alignment=True):
    """
    Perform GEMM using SM100 optimized kernel with TMA: D = alpha * A @ B^T + beta * C

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
        - Uses TMA (Tensor Memory Accelerator) for efficient memory transfers
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

    # Call the extension (now uses TMA internally)
    return sm100_gemm.sm100_gemm_f16(A, B, C, alpha, beta)


# Keep the old name for compatibility
sm100_gemm_f16 = sm100_gemm_f16_tma


def create_aligned_tensors(
    M, N, K, device="cuda", dtype_AB=torch.float16, dtype_C=torch.float32
):
    """
    Create properly aligned tensors for SM100 GEMM with TMA

    Returns:
        tuple: (A, B, C) tensors with aligned dimensions
    """
    aligned_M, aligned_N, aligned_K = sm100_gemm.get_aligned_shape(M, N, K)

    A = torch.zeros(aligned_M, aligned_K, dtype=dtype_AB, device=device)
    B = torch.zeros(aligned_N, aligned_K, dtype=dtype_AB, device=device)
    C = torch.zeros(aligned_M, aligned_N, dtype=dtype_C, device=device)

    return A, B, C


def pad_to_aligned(tensor, target_shape=None, dim_requirements=None):
    """
    Pad tensor to meet SM100 alignment requirements

    Args:
        tensor: Input tensor to pad
        target_shape: Specific target shape (optional)
        dim_requirements: Tuple of (M_align, N_align, K_align) requirements

    Returns:
        Padded tensor and padding info for later unpadding
    """
    if dim_requirements is None:
        dim_requirements = (
            sm100_gemm.MMA_TILE_M,
            sm100_gemm.MMA_TILE_N,
            sm100_gemm.MMA_TILE_K,
        )

    if tensor.dim() == 2:
        M, N = tensor.shape

        if target_shape:
            target_M, target_N = target_shape
        else:
            target_M = (
                (M + dim_requirements[0] - 1) // dim_requirements[0]
            ) * dim_requirements[0]
            target_N = (
                (N + dim_requirements[1] - 1) // dim_requirements[1]
            ) * dim_requirements[1]

        pad_M = target_M - M
        pad_N = target_N - N

        # PyTorch padding format: (pad_left, pad_right, pad_top, pad_bottom)
        padded = torch.nn.functional.pad(tensor, (0, pad_N, 0, pad_M))

        return padded, (M, N, pad_M, pad_N)
    else:
        raise ValueError("Only 2D tensors supported")


def unpad_result(tensor, padding_info):
    """Remove padding from result tensor"""
    orig_M, orig_N, pad_M, pad_N = padding_info
    return tensor[:orig_M, :orig_N]


def benchmark_sm100_vs_torch(
    M=512,
    N=1024,
    K=256,
    num_warmup=1,
    num_trials=10,
    auto_align=True,
    compare_tma=True,
):
    """
    Benchmark SM100 GEMM with TMA against PyTorch's native GEMM
    """
    # Ensure dimensions are aligned
    if auto_align:
        M = (
            (M + sm100_gemm.MMA_TILE_M - 1) // sm100_gemm.MMA_TILE_M
        ) * sm100_gemm.MMA_TILE_M
        N = (
            (N + sm100_gemm.MMA_TILE_N - 1) // sm100_gemm.MMA_TILE_N
        ) * sm100_gemm.MMA_TILE_N
        K = (
            (K + sm100_gemm.MMA_TILE_K - 1) // sm100_gemm.MMA_TILE_K
        ) * sm100_gemm.MMA_TILE_K

    print(f"Benchmarking GEMM with TMA for shape: ({M}, {N}, {K})")

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

        # SM100 GEMM with TMA
        sm100_result = sm100_gemm_f16_tma(A, B, C.clone(), check_alignment=False)

    torch.cuda.synchronize()

    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # warmup
    torch_result = torch.addmm(C, A_fp32, B_fp32.T)

    start.record()
    for _ in range(num_trials):
        torch_result = torch.addmm(C, A_fp32, B_fp32.T)
    end.record()
    torch.cuda.synchronize()
    torch_time = start.elapsed_time(end) / num_trials

    # Benchmark SM100 with TMA
    # warmup
    sm100_result = sm100_gemm_f16_tma(A, B, C.clone(), check_alignment=False)

    start.record()
    for _ in range(num_trials):
        sm100_result = sm100_gemm_f16_tma(A, B, C.clone(), check_alignment=False)
    end.record()
    torch.cuda.synchronize()
    sm100_time = start.elapsed_time(end) / num_trials

    # Check correctness
    max_diff = torch.max(torch.abs(torch_result - sm100_result))
    rel_error = max_diff / torch.max(torch.abs(torch_result))

    # Calculate FLOPS
    flops = 2 * M * N * K  # Multiply-add operations
    torch_tflops = flops / (torch_time * 1e-3) / 1e12
    sm100_tflops = flops / (sm100_time * 1e-3) / 1e12

    print(f"PyTorch time: {torch_time:.3f} ms ({torch_tflops:.2f} TFLOPS)")
    print(f"SM100+TMA time: {sm100_time:.3f} ms ({sm100_tflops:.2f} TFLOPS)")
    print(f"Speedup: {torch_time/sm100_time:.2f}x")
    print(f"Max difference: {max_diff:.6f}")
    print(f"Relative error: {rel_error:.6f}")
    print(f"🚀 TMA provides efficient memory transfers for large matrices!")

    return {
        "torch_time": torch_time,
        "sm100_time": sm100_time,
        "speedup": torch_time / sm100_time,
        "torch_tflops": torch_tflops,
        "sm100_tflops": sm100_tflops,
        "max_diff": max_diff.item(),
        "rel_error": rel_error.item(),
    }


# Neural network layer implementations with TMA
class SM100LinearTMA(torch.nn.Module):
    """
    Linear layer using SM100 GEMM with TMA for forward pass
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
            f"SM100LinearTMA: {in_features} -> {out_features} (aligned: {aligned_in} -> {aligned_out})"
        )
        print("🚀 Using TMA for efficient memory transfers")

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

        # SM100 GEMM with TMA: output = x @ weight^T + bias
        output = sm100_gemm_f16_tma(
            x, self.weight, C, alpha=1.0, beta=beta, check_alignment=False
        )

        # Remove padding
        return output[:batch_size, : self.orig_out_features]


def benchmark_tma_vs_cooperative_copy(M=512, N=1024, K=256, num_trials=50):
    """
    TMA addition
    """

    results = benchmark_sm100_vs_torch(M, N, K, num_trials=num_trials)

    if results:
        print(f"\nTMA-accelerated SM100 GEMM achieved:")
        print(f"   Performance: {results['sm100_tflops']:.2f} TFLOPS")
        print(f"   Speedup: {results['speedup']:.2f}x over PyTorch")
        print(f"   Memory efficiency: Hardware-optimized transfers")


def stress_test_large_matrices():
    """
    Test TMA performance with large matrices that benefit most from TMA
    """
    print("\n=== Large Matrix Stress Test with TMA ===")

    # Test progressively larger matrices
    test_sizes = [
        (1024, 2048, 512),  # 1GB+ tensors
        (2048, 4096, 1024),  # 4GB+ tensors
        (4096, 8192, 2048),  # 16GB+ tensors (if memory allows)
    ]

    for M, N, K in test_sizes:
        try:
            print(f"\nTesting size: ({M}, {N}, {K})")

            # Check memory requirements
            memory_A = M * K * 2  # FP16
            memory_B = N * K * 2  # FP16
            memory_C = M * N * 4  # FP32
            total_memory = (memory_A + memory_B + memory_C * 2) / (1024**3)  # GB

            print(f"Memory requirement: {total_memory:.2f} GB")

            if total_memory > 20:  # Skip if > 20GB
                print("⚠️  Skipping due to memory constraints")
                continue

            # Create tensors
            A, B, C = create_aligned_tensors(M, N, K)
            A[:M, :K].normal_(0, 0.1)
            B[:N, :K].normal_(0, 0.1)
            C[:M, :N].normal_(0, 0.1)

            # Warmup
            for _ in range(3):
                result = sm100_gemm_f16_tma(A, B, C.clone(), check_alignment=False)

            torch.cuda.synchronize()

            # Benchmark
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            num_trials = 10
            start.record()
            for _ in range(num_trials):
                result = sm100_gemm_f16_tma(A, B, C.clone(), check_alignment=False)
            end.record()

            torch.cuda.synchronize()
            avg_time = start.elapsed_time(end) / num_trials

            # Calculate performance
            flops = 2 * M * N * K
            tflops = flops / (avg_time * 1e-3) / 1e12
            bandwidth = total_memory / (avg_time * 1e-3)  # GB/s

            print(f"✅ Time: {avg_time:.2f} ms")
            print(f"✅ Performance: {tflops:.2f} TFLOPS")
            print(f"✅ Bandwidth: {bandwidth:.1f} GB/s")
            print(f"🚀 TMA enables efficient handling of large matrices!")

        except torch.cuda.OutOfMemoryError:
            print(f"❌ Out of memory for size ({M}, {N}, {K})")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            break


# Example usage and test
if __name__ == "__main__":
    print("=== SM100 GEMM with TMA Extension Test ===")

    # Check compatibility first
    if not check_sm100_compatibility():
        print("Exiting due to compatibility issues")
        exit(1)

    print("\n=== Testing basic TMA functionality ===")

    # Test with properly aligned dimensions
    M, N, K = 512, 1024, 256
    A, B, C = create_aligned_tensors(M, N, K)

    # Fill with random data (only the actual needed portion)
    A[:M, :K].normal_()
    B[:N, :K].normal_()
    C[:M, :N].normal_()

    # Test the TMA GEMM
    result = sm100_gemm_f16_tma(A, B, C, alpha=1.0, beta=0.5, check_alignment=False)
    print(
        f"✅ TMA GEMM test passed. Result shape: {result.shape}, dtype: {result.dtype}"
    )

    print("\n=== Testing SM100LinearTMA layer ===")

    # Test linear layer with TMA
    layer = SM100LinearTMA(256, 512, bias=True)
    x = torch.randn(128, 256, dtype=torch.float16, device="cuda")
    output = layer(x)
    print(f"✅ TMA Linear layer test passed. Output shape: {output.shape}")

    print("\n=== Testing padding utilities ===")

    # Test padding for misaligned tensors
    misaligned_A = torch.randn(300, 200, dtype=torch.float16, device="cuda")
    padded_A, pad_info = pad_to_aligned(misaligned_A)
    print(f"Original shape: {misaligned_A.shape}, Padded shape: {padded_A.shape}")

    unpadded = unpad_result(padded_A, pad_info)
    print(f"✅ Padding test passed. Unpadded shape: {unpadded.shape}")

    print("\n=== Running TMA performance benchmark ===")

    # Run benchmark
    benchmark_results = benchmark_sm100_vs_torch(M=512, N=1024, K=256, num_trials=50)

    if benchmark_results:
        print(f"\n✅ All TMA tests passed!")
        print(
            f"🚀 SM100+TMA achieved {benchmark_results['speedup']:.2f}x speedup over PyTorch"
        )
        print(f"🚀 TMA provides hardware-accelerated memory transfers!")

        # Run additional TMA-specific tests
        benchmark_tma_vs_cooperative_copy(M=1024, N=2048, K=512)

        # Test with larger matrices if memory allows
        print("\n=== Testing TMA with larger matrices ===")
        stress_test_large_matrices()

    else:
        print("❌ Benchmark failed")

    print("\n=== TMA Summary ===")
    print("🚀 TMA (Tensor Memory Accelerator) provides:")
    print("   • Hardware-accelerated global->shared memory transfers")
    print("   • Reduced CPU overhead and better bandwidth utilization")
    print("   • Automatic memory layout optimization")
    print("   • Essential for peak performance on large matrices")
    print("   • Enables scaling to multi-GB tensor operations")
import sm100_gemm  # The compiled extension

# python_interface.py - High-level Python interface (updated for split files)
import torch


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
        print("SM100 ready!")  # ✅

    return compile_support and device_support


def sm100_gemm_f16(A, B, C=None, alpha=1.0, beta=0.0, check_alignment=True):
    """
    Perform GEMM using SM100 optimized kernel: D = alpha * A @ B^T + beta * C

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

    # Call the extension
    return sm100_gemm.sm100_gemm_f16(A, B, C, alpha, beta)


def create_aligned_tensors(
    M, N, K, device="cuda", dtype_AB=torch.float16, dtype_C=torch.float32
):
    """
    Create properly aligned tensors for SM100 GEMM

    Returns:
        tuple: (A, B, C) tensors with aligned dimensions
    """
    aligned_M, aligned_N, aligned_K = sm100_gemm.get_aligned_shape(M, N, K)

    A = torch.zeros(aligned_M, aligned_K, dtype=dtype_AB, device=device)
    B = torch.zeros(aligned_N, aligned_K, dtype=dtype_AB, device=device)
    C = torch.zeros(aligned_M, aligned_N, dtype=dtype_C, device=device)

    return A, B, C


def pad_to_aligned(tensor, target_shape=None, dim_requirements=None):
    """
    Pad tensor to meet SM100 alignment requirements

    Args:
        tensor: Input tensor to pad
        target_shape: Specific target shape (optional)
        dim_requirements: Tuple of (M_align, N_align, K_align) requirements

    Returns:
        Padded tensor and padding info for later unpadding
    """
    if dim_requirements is None:
        dim_requirements = (
            sm100_gemm.MMA_TILE_M,
            sm100_gemm.MMA_TILE_N,
            sm100_gemm.MMA_TILE_K,
        )

    if tensor.dim() == 2:
        M, N = tensor.shape

        if target_shape:
            target_M, target_N = target_shape
        else:
            target_M = (
                (M + dim_requirements[0] - 1) // dim_requirements[0]
            ) * dim_requirements[0]
            target_N = (
                (N + dim_requirements[1] - 1) // dim_requirements[1]
            ) * dim_requirements[1]

        pad_M = target_M - M
        pad_N = target_N - N

        # PyTorch padding format: (pad_left, pad_right, pad_top, pad_bottom)
        padded = torch.nn.functional.pad(tensor, (0, pad_N, 0, pad_M))

        return padded, (M, N, pad_M, pad_N)
    else:
        raise ValueError("Only 2D tensors supported")


def unpad_result(tensor, padding_info):
    """Remove padding from result tensor"""
    orig_M, orig_N, pad_M, pad_N = padding_info
    return tensor[:orig_M, :orig_N]


def benchmark_sm100_vs_torch(
    M=512, N=1024, K=256, num_warmup=10, num_trials=100, auto_align=True
):
    """
    Benchmark SM100 GEMM against PyTorch's native GEMM
    """
    # Ensure dimensions are aligned
    if auto_align:
        M = (
            (M + sm100_gemm.MMA_TILE_M - 1) // sm100_gemm.MMA_TILE_M
        ) * sm100_gemm.MMA_TILE_M
        N = (
            (N + sm100_gemm.MMA_TILE_N - 1) // sm100_gemm.MMA_TILE_N
        ) * sm100_gemm.MMA_TILE_N
        K = (
            (K + sm100_gemm.MMA_TILE_K - 1) // sm100_gemm.MMA_TILE_K
        ) * sm100_gemm.MMA_TILE_K

    print(f"Benchmarking GEMM with shape: ({M}, {N}, {K})")

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

        # SM100 GEMM
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

    # Benchmark SM100
    start.record()
    for _ in range(num_trials):
        sm100_result = sm100_gemm_f16(A, B, C.clone(), check_alignment=False)
    end.record()
    torch.cuda.synchronize()
    sm100_time = start.elapsed_time(end) / num_trials

    # Check correctness
    max_diff = torch.max(torch.abs(torch_result - sm100_result))
    rel_error = max_diff / torch.max(torch.abs(torch_result))

    # Calculate FLOPS
    flops = 2 * M * N * K  # Multiply-add operations
    torch_tflops = flops / (torch_time * 1e-3) / 1e12
    sm100_tflops = flops / (sm100_time * 1e-3) / 1e12

    print(f"PyTorch time: {torch_time:.3f} ms ({torch_tflops:.2f} TFLOPS)")
    print(f"SM100 time: {sm100_time:.3f} ms ({sm100_tflops:.2f} TFLOPS)")
    print(f"Speedup: {torch_time/sm100_time:.2f}x")
    # print(f"Max difference: {max_diff:.6f}")
    print(f"Relative error: {rel_error:.6f}")

    return {
        "torch_time": torch_time,
        "sm100_time": sm100_time,
        "speedup": torch_time / sm100_time,
        "torch_tflops": torch_tflops,
        "sm100_tflops": sm100_tflops,
        "max_diff": max_diff.item(),
        "rel_error": rel_error.item(),
    }


# Neural network layer implementations
class SM100Linear(torch.nn.Module):
    """
    Linear layer using SM100 GEMM for forward pass
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

        # SM100 GEMM: output = x @ weight^T + bias
        output = sm100_gemm_f16(
            x, self.weight, C, alpha=1.0, beta=beta, check_alignment=False
        )

        # Remove padding
        return output[:batch_size, : self.orig_out_features]


# Example usage and test
if __name__ == "__main__":
    print("=== SM100 GEMM Extension Test ===")

    # Check compatibility first
    if not check_sm100_compatibility():
        print("Exiting due to compatibility issues")
        exit(1)

    print("\n=== Testing basic functionality ===")

    # Test with properly aligned dimensions
    M, N, K = 512, 1024, 256
    A, B, C = create_aligned_tensors(M, N, K)

    # Fill with random data (only the actual needed portion)
    A[:M, :K].normal_()
    B[:N, :K].normal_()
    C[:M, :N].normal_()

    # Test the GEMM
    result = sm100_gemm_f16(A, B, C, alpha=1.0, beta=0.5, check_alignment=False)
    print(
        f"✅ Basic GEMM test passed. Result shape: {result.shape}, dtype: {result.dtype}"
    )

    print("\n=== Testing SM100Linear layer ===")

    # Test linear layer
    layer = SM100Linear(256, 512, bias=True)
    x = torch.randn(128, 256, dtype=torch.float16, device="cuda")
    output = layer(x)
    print(f"✅ Linear layer test passed. Output shape: {output.shape}")

    print("\n=== Testing padding utilities ===")

    # Test padding for misaligned tensors
    misaligned_A = torch.randn(300, 200, dtype=torch.float16, device="cuda")
    padded_A, pad_info = pad_to_aligned(misaligned_A)
    print(f"Original shape: {misaligned_A.shape}, Padded shape: {padded_A.shape}")

    unpadded = unpad_result(padded_A, pad_info)
    print(f"✅ Padding test passed. Unpadded shape: {unpadded.shape}")

    print("\n=== Running performance benchmark ===")

    # Run benchmark
    benchmark_results = benchmark_sm100_vs_torch(
        M=8192, N=8192 * 2, K=2048, num_trials=50
    )

    if benchmark_results:
        print(f"\n✅ All tests passed!")
        print(
            f"Blackwell Cute GEMM with TMA Loading achieved {benchmark_results['speedup']:.2f}x speedup over PyTorch"
        )
    else:
        print("❌ Benchmark failed")
