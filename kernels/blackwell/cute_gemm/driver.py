# ==============================================================================
# python_interface.py - High-level Python interface
# ==============================================================================


import torch

try:
    import sm100_gemm  # The compiled extension - this has to go after import torch...but auto-formatting is blocking
except ImportError:
    print("❌ SM100 not ready!")
    raise ImportError(
        "SM100 not ready! Please build the extension using `python setup.py install`"
    )


def sm100_gemm_f16(A, B, C=None, alpha=1.0, beta=0.0):
    """
    Perform GEMM using SM100 optimized kernel: D = alpha * A @ B^T + beta * C

    Args:
        A (torch.Tensor): Input tensor A of shape (M, K), dtype=torch.float16
        B (torch.Tensor): Input tensor B of shape (N, K), dtype=torch.float16
        C (torch.Tensor, optional): Input tensor C of shape (M, N), dtype=torch.float32
                                   If None, creates zero tensor
        alpha (float): Scaling factor for A @ B^T
        beta (float): Scaling factor for C

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

    # Check alignment requirements
    assert M % 128 == 0, f"M={M} must be multiple of 128"
    assert N % 256 == 0, f"N={N} must be multiple of 256"
    assert K % 64 == 0, f"K={K} must be multiple of 64"

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


def benchmark_sm100_vs_torch(M=512, N=1024, K=256, num_warmup=10, num_trials=100):
    """
    Benchmark SM100 GEMM against PyTorch's native GEMM
    """
    # Ensure dimensions are aligned
    M = ((M + 127) // 128) * 128
    N = ((N + 255) // 256) * 256
    K = ((K + 63) // 64) * 64

    print(f"Benchmarking GEMM with shape: ({M}, {N}, {K})")

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
        sm100_result = sm100_gemm_f16(A, B, C.clone())

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
        sm100_result = sm100_gemm_f16(A, B, C.clone())
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
    print(f"Max difference: {max_diff:.6f}")
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


# Example usage and test
if __name__ == "__main__":
    # Test basic functionality
    print("Testing SM100 GEMM...")

    M, N, K = 512, 1024, 256
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(N, K, dtype=torch.float16, device="cuda")
    C = torch.randn(M, N, dtype=torch.float32, device="cuda")

    # Test the GEMM
    result = sm100_gemm_f16(A, B, C, alpha=1.0, beta=0.5)
    print(f"Result shape: {result.shape}, dtype: {result.dtype}")

    # Run benchmark
    print("\nRunning benchmark...")
    benchmark_results = benchmark_sm100_vs_torch(M, N, K)

# ==============================================================================
# Makefile for easy building
# ==============================================================================

MAKEFILE_CONTENT = """
# Makefile for SM100 GEMM PyTorch Extension

# Set these paths according to your installation
CUTLASS_PATH ?= /path/to/cutlass
CUDA_HOME ?= $(shell python -c "import torch; print(torch.utils.cpp_extension.CUDA_HOME)")

# Build the extension
build:
	CUTLASS_PATH=$(CUTLASS_PATH) python setup.py build_ext --inplace

# Install the extension
install:
	CUTLASS_PATH=$(CUTLASS_PATH) pip install .

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info/ sm100_gemm*.so

# Test the installation
test:
	python python_interface.py

# Check CUDA device capability
check_device:
	python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name()}, Compute capability: {torch.cuda.get_device_capability()}')"

.PHONY: build install clean test check_device
"""

# Write Makefile
with open("Makefile", "w") as f:
    f.write(MAKEFILE_CONTENT)

print("Setup files created!")
print("To build:")
print("1. Set CUTLASS_PATH environment variable to your CUTLASS installation")
print("2. Run: make build")
print("3. Test: make test")
