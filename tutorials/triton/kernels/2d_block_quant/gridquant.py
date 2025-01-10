from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from typing import Tuple
from triton import Config

# debug compare
from sijia_gemm import triton_quantize_fp8_block

# H100-specific constants
FP8_E4M3_MAX: tl.constexpr = 448.0


# from deepseek v3:


@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous()
    assert x.size(-1) % block_size == 0
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    assert x.is_contiguous() and s.is_contiguous()
    assert x.dim() == 2 and s.dim() == 2
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


fp8_gemm_configs = [
    Config({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': 128}, num_stages=num_stages, num_warps=8)
    for block_m in [16, 32, 64] for block_n in [32, 64, 128] for num_stages in [3, 4, 5, 6]
]

@triton.autotune(configs=fp8_gemm_configs, key=['N', 'K'])
@triton.jit
def fp8_gemm_kernel(a_ptr, b_ptr, c_ptr,
                    a_s_ptr, b_s_ptr,
                    M, N: tl.constexpr, K: tl.constexpr,
                    BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def ds_fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    fp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)
    return c
# end deepseek kernels




@dataclass
class GridQuantConfig:
    """Configuration for GridQuant."""

    # Larger block sizes for better parallelism on H100
    block_size_m: int = 128
    block_size_k: int = 128
    vec_size_m: int = 32  # Match warp size
    vec_size_k: int = 32
    min_scale: float = 1e-4
    dtype: torch.dtype = torch.float8_e4m3fn
    num_warps: int = 16  # H100 supports more warps per block
    num_stages: int = 4  # Pipeline depth


def get_optimal_h100_blocks(M: int, K: int) -> tuple[int, int]:
    """
    Calculate optimal block sizes for H100 based on matrix dimensions
    Note: atm we don't use this b/c we keep 256,256 for all sizes
    """
    block_size = 256 # THIS IS HARDCODED ATM AND NOT DYNAMIC...ADJUST IF NEEDED.
    # Optimize for H100's cache hierarchy and memory bandwidth
    if M * K <= 1024 * 1024:  # Small matrices
        return block_size, block_size
    elif M * K <= 4 * 1024 * 1024:  # Medium matrices
        return block_size, block_size
    else:  # Large matrices
        return block_size, block_size  # Rectangular blocks for better memory access patterns


@triton.jit
def gridquant_kernel(
    A,  # Input tensor (BF16)
    A_scale,  # Output scale tensor
    A_fp8,  # Output FP8 tensor
    M,
    K,  # Matrix dimensions
    stride_am,
    stride_ak,  # Input strides
    stride_ascale_m,
    stride_ascale_k,  # Scale strides
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MIN_SCALE: tl.constexpr,
    VEC_SIZE_M: tl.constexpr,
    VEC_SIZE_K: tl.constexpr,
) -> None:
    """
    Grid Stride Loop quantization kernel with tensor core support and vectorization
    """
    pid = tl.program_id(0)
    grid_k = tl.cdiv(K, BLOCK_K)
    block_m = pid // grid_k
    block_k = pid % grid_k

    # establish base offsets for the block
    offs_m_base = block_m * BLOCK_M
    offs_k_base = block_k * BLOCK_K

    # vectorized ranges for both dimensions
    vec_range_m = tl.arange(0, VEC_SIZE_M)
    vec_range_k = tl.arange(0, VEC_SIZE_K)

    # Shared memory for block aggregation
    max_vals = tl.zeros([VEC_SIZE_M, VEC_SIZE_K], dtype=tl.float32) - float("inf")

    # First pass: find maximum absolute value
    for m_idx in range(0, BLOCK_M, VEC_SIZE_M):
        m_offs = offs_m_base + m_idx + vec_range_m
        m_mask = m_offs < M

        for k_idx in range(0, BLOCK_K, VEC_SIZE_K):
            k_offs = offs_k_base + k_idx + vec_range_k
            k_mask = k_offs < K

            # matrix offsets for vectorized load
            offs_a = m_offs[:, None] * stride_am + k_offs[None, :] * stride_ak
            mask = m_mask[:, None] & k_mask[None, :]

            a_block = tl.load(
                A + offs_a, mask=mask, other=0.0, eviction_policy="evict_last"
            )
            a_block = tl.abs(a_block)
            max_vals = tl.maximum(max_vals, a_block)

    # Reduce maximum values
    block_max = tl.max(max_vals)

    # Compute scale with numerical stability
    scale = FP8_E4M3_MAX / tl.maximum(block_max, MIN_SCALE)

    # Store scale factor
    scale_offset = block_m * stride_ascale_m + block_k * stride_ascale_k
    tl.store(A_scale + scale_offset, scale)

    # Second pass: quantize with computed scale
    for m_idx in range(0, BLOCK_M, VEC_SIZE_M):
        m_offs = offs_m_base + m_idx + vec_range_m
        m_mask = m_offs < M

        for k_idx in range(0, BLOCK_K, VEC_SIZE_K):
            k_offs = offs_k_base + k_idx + vec_range_k
            k_mask = k_offs < K

            offs_a = m_offs[:, None] * stride_am + k_offs[None, :] * stride_ak
            mask = m_mask[:, None] & k_mask[None, :]

            a_block = tl.load(
                A + offs_a, mask=mask, other=0.0, eviction_policy="evict_last"
            )
            a_fp8 = tl.where(mask, (a_block * scale).to(tl.float8e4nv), 0.0)
            # Store new fp8 values
            tl.store(A_fp8 + offs_a, a_fp8, mask=mask)


def gridquant(
    x: torch.Tensor, config: GridQuantConfig = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Grid Stride quantization implementation
    """

    if x.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {x.dim()}D")

    if config is None:
        config = GridQuantConfig()

    M, K = x.shape

    # Warning - we don't actually do anything here..it's all 256,256 for now
    # Get optimal block sizes for current matrix dimensions
    block_m, block_k = get_optimal_h100_blocks(M, K)
    config.block_size_m = block_m
    config.block_size_k = block_k

    # Ensure input is contiguous
    if x.stride(-1) != 1:
        x = x.contiguous()

    # Calculate grid dimensions (why not cdiv? can't be used outside of kernel...)
    gridm = (M + block_m - 1) // block_m
    gridk = (K + block_k - 1) // block_k

    # Allocate output tensors with optimal memory layout
    x_scale = torch.empty(
        (gridm, gridk),
        device=x.device,
        dtype=torch.float32,
        memory_format=torch.contiguous_format,
    )

    x_fp8_out = torch.empty(
        (M, K),
        device=x.device,
        dtype=config.dtype,
        memory_format=torch.contiguous_format,
    )

    # Calculate optimal grid configuration
    grid = (gridm * gridk,)

    # Launch GridQuant kernel
    gridquant_kernel[grid](
        x,
        x_scale,
        x_fp8_out,
        M,
        K,
        x.stride(0),
        x.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        BLOCK_M=config.block_size_m,
        BLOCK_K=config.block_size_k,
        MIN_SCALE=config.min_scale,
        VEC_SIZE_M=config.vec_size_m,
        VEC_SIZE_K=config.vec_size_k,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )

    return x_fp8_out, x_scale


@triton.jit
def griddequant_kernel(
    Q,  # Quantized input
    Scale,  # Scale factors
    Output,  # Output tensor
    M,
    K,  # Matrix dimensions
    stride_q_m,
    stride_q_k,  # Input strides
    stride_o_m,
    stride_o_k,  # Output strides
    stride_scale_m,
    stride_scale_k,  # Scale strides
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    VEC_SIZE_M: tl.constexpr,
    VEC_SIZE_K: tl.constexpr,
):
    """
    GridQuant dequantization kernel
    """
    pid = tl.program_id(0)
    grid_k = tl.cdiv(K, BLOCK_K)
    block_m = pid // grid_k
    block_k = pid % grid_k

    # Calculate base offsets
    offs_m_base = block_m * BLOCK_M
    offs_k_base = block_k * BLOCK_K

    # Create vectorized ranges for both dimensions
    vec_range_m = tl.arange(0, VEC_SIZE_M)
    vec_range_k = tl.arange(0, VEC_SIZE_K)

    # Load scale factor with proper striding
    scale = tl.load(
        Scale + block_m * stride_scale_m + block_k * stride_scale_k,
        eviction_policy="evict_first",
    )

    # Process blocks with 2D vectorization
    for m_idx in range(0, BLOCK_M, VEC_SIZE_M):
        m_offs = offs_m_base + m_idx + vec_range_m
        m_mask = m_offs < M

        for k_idx in range(0, BLOCK_K, VEC_SIZE_K):
            k_offs = offs_k_base + k_idx + vec_range_k
            k_mask = k_offs < K

            # Compute matrix offsets for vectorized load
            offs_q = m_offs[:, None] * stride_q_m + k_offs[None, :] * stride_q_k
            offs_o = m_offs[:, None] * stride_o_m + k_offs[None, :] * stride_o_k
            mask = m_mask[:, None] & k_mask[None, :]

            q = tl.load(Q + offs_q, mask=mask, other=0.0, eviction_policy="evict_last")
            # Convert to FP32 for better numerical stability
            deq = tl.where(mask, (q.to(tl.float32) / scale).to(tl.bfloat16), 0.0)

            # Store with proper output striding
            tl.store(Output + offs_o, deq, mask=mask)


def griddequant(
    quantized: torch.Tensor, scale: torch.Tensor, config: GridQuantConfig = None
) -> torch.Tensor:
    """
    GridQuant dequantization

    Args:
        quantized: Input tensor in FP8 format
        scale: Scale factors tensor
        config: Configuration object for specific optimizations

    Returns:
        torch.Tensor: Dequantized tensor in BF16 format
    """
    if config is None:
        config = GridQuantConfig()

    # Input validation
    if quantized.dtype != config.dtype:
        raise ValueError(f"Expected {config.dtype} input tensor, got {quantized.dtype}")

    if scale.dtype != torch.float32:
        raise ValueError(f"Expected float32 scale tensor, got {scale.dtype}")

    if quantized.dim() != 2 or scale.dim() != 2:
        raise ValueError("Both quantized and scale tensors must be 2D")

    M, K = quantized.shape
    grid_m = triton.cdiv(M, config.block_size_m)
    grid_k = triton.cdiv(K, config.block_size_k)

    expected_scale_shape = (grid_m, grid_k)
    if scale.shape != expected_scale_shape:
        raise ValueError(
            f"Scale tensor shape {scale.shape} doesn't match expected shape {expected_scale_shape}"
        )

    # Ensure inputs are in optimal memory layout
    quantized = quantized.contiguous()
    scale = scale.contiguous()

    # Allocate output tensor with optimal memory layout
    output = torch.empty(
        (M, K),
        device=quantized.device,
        dtype=torch.bfloat16,
        memory_format=torch.contiguous_format,
    )

    # Calculate grid size
    grid = (grid_m * grid_k,)

    # Launch kernel with improved parameter passing
    griddequant_kernel[grid](
        quantized,
        scale,
        output,
        M,
        K,
        quantized.stride(0),
        quantized.stride(1),  # Input strides
        output.stride(0),
        output.stride(1),  # Output strides
        scale.stride(0),
        scale.stride(1),  # Scale strides
        BLOCK_M=config.block_size_m,
        BLOCK_K=config.block_size_k,
        VEC_SIZE_M=config.vec_size_m,
        VEC_SIZE_K=config.vec_size_k,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )

    return output


# benchmark function
def benchmark_gridquant(M: int = 4096, K: int = 4096, num_runs: int = 100):
    """
    Benchmark GridQuant implementation
    """
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    config = GridQuantConfig()

    # Warmup
    for _ in range(10):
        x_fp8, scales = gridquant(x, config)
        # _ = triton_quantize_fp8_block(x)

        # _ = griddequant(x_fp8, scales, config)

    # Benchmark
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_runs):
        x_fp8, scales = gridquant(x, config)
        # x_fp8, scales = triton_quantize_fp8_block(x)

    end.record()

    torch.cuda.synchronize()

    # dequantized = griddequant(x_fp8, scales, config)

    # Compute metrics
    """abs_error = torch.abs(x - dequantized)
    rel_error = abs_error / (torch.abs(x) + 1e-7)

    stats = {
        "max_abs_error": abs_error.max().item(),
        "mean_abs_error": abs_error.mean().item(),
        "max_rel_error": rel_error.max().item(),
        "mean_rel_error": rel_error.mean().item(),
    }
    """
    avg_time = start.elapsed_time(end) / num_runs
    throughput = (M * K * 4) / (avg_time * 1e6)  # GB/s

    print(f"Matrix size: {M}x{K}")
    print(f"Average time: {avg_time:.3f} ms")
    print(f"Throughput: {throughput:.2f} GB/s")
    # print(f"Max rel error: {stats['max_rel_error']:.3e}")

def benchmark_quantization(M: int = 4096, K: int = 4096, num_runs: int = 100):
    """
    Benchmark different quantization implementations
    """
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    config = GridQuantConfig()
    block_size = 128  # block size for act_quant

    # Store results
    results = {}

    def run_benchmark(name, func, *args):
        # Warmup
        for _ in range(10):
            _ = func(*args)
        
        # Benchmark
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(num_runs):
            _ = func(*args)
        end.record()
        
        torch.cuda.synchronize()
        avg_time = start.elapsed_time(end) / num_runs
        throughput = (M * K * 4) / (avg_time * 1e6)  # GB/s
        
        results[name] = {
            "avg_time": avg_time,
            "throughput": throughput
        }

    # Benchmark Triton implementation
    # run_benchmark("Triton", triton_quantize_fp8_block, x, block_size, block_size)

    # Benchmark act_quant
    # run_benchmark("act_quant", act_quant, x, block_size)

    # Benchmark GridQuant
    run_benchmark("GridQuant", gridquant, x, config)


    # Print results
    print(f"\nMatrix size: {M}x{K}")
    print("\nBenchmark Results:")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Average time: {metrics['avg_time']:.3f} ms")
        print(f"  Throughput: {metrics['throughput']:.2f} GB/s")

if __name__ == "__main__":
    # Compare performance across some random sizes
    sizes = [
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
        (16384, 16384),
        (32768, 32768),
        (6656, 16384),
        (16384, 13312),
    ]
    for M, K in sizes:
        #benchmark_gridquant(M, K)
        benchmark_quantization(M, K)
        print("\n")
