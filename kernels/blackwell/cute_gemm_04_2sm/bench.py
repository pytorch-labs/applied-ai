import torch

try:
    import sm100_gemm
except ImportError as e:
    _SM100_AVAILABLE = False
    _IMPORT_ERROR = str(e)
    raise ImportError(_IMPORT_ERROR)

gemm_f16 = sm100_gemm.gemm_f16


def benchmark_sm100_vs_torch(
    M=4096, N=8192, K=2048, num_warmup=1, num_trials=10
):  # M=512, N=1024, K=256, num_warmup=10, num_trials=100):
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
    C = torch.randn(M, N, dtype=torch.float16, device="cuda")
    C32 = C.to(torch.float32).clone()

    # Keep A and B as FP16 for PyTorch
    A_fp16 = A
    B_fp16 = B

    # Warmup
    for _ in range(num_warmup):
        # PyTorch GEMM (using FP16)
        torch_result = torch.addmm(C, A_fp16, B_fp16.T)

        # SM100 GEMM
        sm100_result = gemm_f16(A_fp16, B_fp16, C32)

    torch.cuda.synchronize()

    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_trials):
        torch_result = torch.addmm(C, A_fp16, B_fp16.T)
    end.record()
    torch.cuda.synchronize()
    torch_time = start.elapsed_time(end) / num_trials

    # Benchmark SM100
    start.record()
    for _ in range(num_trials):
        sm100_result = gemm_f16(A, B, C32)
    end.record()
    torch.cuda.synchronize()
    sm100_time = start.elapsed_time(end) / num_trials

    # Check correctness
    max_diff = torch.max(torch.abs(torch_result - sm100_result.to(torch.float16)))
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

    M, N, K = 1024, 8192 * 2, 2048
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(N, K, dtype=torch.float16, device="cuda")
    C = torch.randn(M, N, dtype=torch.float32, device="cuda")

    # Test the GEMM
    result = gemm_f16(
        A,
        B,
        C,
    )  # alpha=1.0, beta=0.5)
    print(f"Result shape: {result.shape}, dtype: {result.dtype}")

    # Run benchmark
    print("\nRunning benchmark...")
    benchmark_results = benchmark_sm100_vs_torch(M, N, K)
