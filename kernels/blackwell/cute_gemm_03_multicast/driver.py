# driver.py - Enhanced high-level Python interface for SM100 GEMM

import time
import warnings
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

try:
    import sm100_gemm

    _SM100_AVAILABLE = True
except ImportError as e:
    _SM100_AVAILABLE = False
    _IMPORT_ERROR = str(e)


class SM100NotAvailableError(RuntimeError):
    """Raised when SM100 functionality is not available"""

    pass


class SM100ValidationError(ValueError):
    """Raised when input validation fails for SM100 operations"""

    pass


class SM100PerformanceWarning(UserWarning):
    """Warning for suboptimal SM100 performance conditions"""

    pass


def _ensure_sm100_available():
    """Ensure SM100 extension is available and raise descriptive error if not"""
    if not _SM100_AVAILABLE:
        raise SM100NotAvailableError(
            f"SM100 extension not available: {_IMPORT_ERROR}. "
            "Please build the extension using `python setup.py install`"
        )


@contextmanager
def cuda_timing():
    """Context manager for accurate CUDA timing"""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    yield lambda: start.elapsed_time(end)
    end.record()
    torch.cuda.synchronize()


class SM100DeviceInfo:
    """Enhanced device information and capability checking"""

    def __init__(self):
        _ensure_sm100_available()
        self._info = None
        self._compatibility = None
        self.refresh()

    def refresh(self):
        """Refresh device information"""
        self._info = sm100_gemm.get_device_info_enhanced()
        self._compatibility = sm100_gemm.check_sm100_compatibility_detailed()

    @property
    def compute_capability(self) -> Tuple[int, int]:
        """Get compute capability as (major, minor)"""
        return (int(self._info[0]), int(self._info[1]))

    @property
    def is_supported(self) -> bool:
        """Check if SM100 is fully supported"""
        return bool(self._compatibility[0])

    @property
    def support_message(self) -> str:
        """Get detailed support status message"""
        return self._compatibility[1]

    @property
    def total_memory_mb(self) -> int:
        """Total GPU memory in MB"""
        return int(self._info[4])

    @property
    def free_memory_mb(self) -> int:
        """Free GPU memory in MB"""
        return int(self._info[5])

    @property
    def multiprocessor_count(self) -> int:
        """Number of multiprocessors"""
        return int(self._info[6])

    @property
    def max_threads_per_block(self) -> int:
        """Maximum threads per block"""
        return int(self._info[7])

    def __str__(self) -> str:
        if not self.is_supported:
            return f"SM100 Status: Not Supported - {self.support_message}"

        return (
            f"SM100 Device Info:\n"
            f"  Compute Capability: {self.compute_capability[0]}.{self.compute_capability[1]}\n"
            f"  Multiprocessors: {self.multiprocessor_count}\n"
            f"  Memory: {self.free_memory_mb}/{self.total_memory_mb} MB (free/total)\n"
            f"  Max Threads/Block: {self.max_threads_per_block}\n"
            f"  Status: {self.support_message}"
        )


class SM100Config:
    """Configuration management for SM100 operations"""

    def __init__(self):
        self.validate_inputs = True
        self.auto_align = True
        self.warn_on_suboptimal = True
        self.profile_performance = False
        self.cache_tensors = False
        self.default_alpha = 1.0
        self.default_beta = 0.0

    @contextmanager
    def temporary_config(self, **kwargs):
        """Temporarily change configuration"""
        original = {key: getattr(self, key) for key in kwargs}
        try:
            for key, value in kwargs.items():
                setattr(self, key, value)
            yield
        finally:
            for key, value in original.items():
                setattr(self, key, value)


# Global configuration instance
config = SM100Config()


def check_sm100_compatibility() -> Tuple[bool, str]:
    """
    Check SM100 compatibility and return detailed status

    Returns:
        Tuple of (is_supported, status_message)
    """
    _ensure_sm100_available()

    device_info = SM100DeviceInfo()

    if not device_info.is_supported:
        return False, device_info.support_message

    # Additional checks for optimal performance
    warnings_list = []

    if device_info.multiprocessor_count < 8:
        warnings_list.append(
            f"Low multiprocessor count ({device_info.multiprocessor_count}). "
            "Performance may be suboptimal."
        )

    if device_info.free_memory_mb < 1024:
        warnings_list.append(
            f"Low available memory ({device_info.free_memory_mb} MB). "
            "Large matrix operations may fail."
        )

    if warnings_list and config.warn_on_suboptimal:
        warning_msg = "SM100 supported but with performance warnings:\n" + "\n".join(
            warnings_list
        )
        warnings.warn(warning_msg, SM100PerformanceWarning)

    return True, device_info.support_message


def validate_tensor_for_sm100(
    tensor: torch.Tensor, name: str, expected_dtype: torch.dtype, expected_dims: int = 2
) -> None:
    """Validate tensor for SM100 operations"""
    if not tensor.is_cuda:
        raise SM100ValidationError(
            f"Tensor {name} must be on CUDA device, got: {tensor.device}"
        )

    if tensor.dtype != expected_dtype:
        raise SM100ValidationError(
            f"Tensor {name} must be {expected_dtype}, got: {tensor.dtype}"
        )

    if tensor.dim() != expected_dims:
        raise SM100ValidationError(
            f"Tensor {name} must be {expected_dims}D, got: {tensor.dim()}D"
        )

    if not tensor.is_contiguous():
        raise SM100ValidationError(f"Tensor {name} must be contiguous")


def get_aligned_dimensions(M: int, N: int, K: int) -> Tuple[int, int, int]:
    """
    Get SM100-aligned dimensions

    Args:
        M, N, K: Original matrix dimensions

    Returns:
        Tuple of aligned (M, N, K) dimensions
    """
    _ensure_sm100_available()
    return tuple(sm100_gemm.get_aligned_shape_enhanced(M, N, K))


def create_aligned_tensors(
    M: int,
    N: int,
    K: int,
    device: Union[str, torch.device] = "cuda",
    dtype_AB: torch.dtype = torch.float16,
    dtype_C: torch.dtype = torch.float32,
    initialize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create properly aligned tensors for SM100 GEMM operations

    Args:
        M, N, K: Matrix dimensions
        device: CUDA device
        dtype_AB: Data type for A and B matrices
        dtype_C: Data type for C matrix
        initialize: Whether to initialize with random values

    Returns:
        Tuple of (A, B, C) tensors with optimal alignment
    """
    _ensure_sm100_available()

    if isinstance(device, str):
        device = torch.device(device)

    tensors = sm100_gemm.create_aligned_tensors_enhanced(
        M, N, K, device, dtype_AB, dtype_C
    )
    A, B, C = tensors

    if initialize:
        # Initialize with small random values for numerical stability
        A.normal_(0, 0.1)
        B.normal_(0, 0.1)
        C.normal_(0, 0.1)

    return A, B, C


def estimate_performance(
    M: int, N: int, K: int, cluster_m: int = 2, cluster_n: int = 2
) -> Dict[str, float]:
    """
    Estimate performance characteristics for given problem size

    Args:
        M, N, K: Matrix dimensions
        cluster_m, cluster_n: Cluster configuration

    Returns:
        Dictionary with performance estimates
    """
    _ensure_sm100_available()

    estimates = sm100_gemm.estimate_performance(M, N, K, cluster_m, cluster_n)

    return {
        "estimated_gflops": estimates[0],
        "estimated_bandwidth_gbs": estimates[1],
        "estimated_time_ms": estimates[2],
        "cluster_config": (cluster_m, cluster_n),
        "total_flops": 2 * M * N * K,
        "memory_mb": (M * K * 2 + N * K * 2 + M * N * 4 * 2) / (1024 * 1024),
    }


def optimize_cluster_size(M: int, N: int, K: int) -> Tuple[int, int]:
    """
    Determine optimal cluster configuration for given problem size

    Args:
        M, N, K: Matrix dimensions

    Returns:
        Optimal (cluster_m, cluster_n) configuration
    """
    # Calculate memory requirements
    memory_gb = (M * K * 2 + N * K * 2 + M * N * 4 * 2) / (1024**3)

    # Enhanced heuristics based on problem characteristics
    if memory_gb > 8.0 and M >= 4096 and N >= 4096:
        return (4, 4)  # Large matrices - maximize multicast benefits
    elif memory_gb > 2.0 and M >= 2048 and N >= 2048:
        return (2, 4)  # Medium-large matrices
    elif M >= 1024 and N >= 1024:
        return (2, 2)  # Medium matrices
    else:
        return (1, 1)  # Small matrices


def sm100_gemm_f16(
    A: torch.Tensor,
    B: torch.Tensor,
    C: Optional[torch.Tensor] = None,
    alpha: float = None,
    beta: float = None,
    validate_inputs: Optional[bool] = None,
) -> torch.Tensor:
    """
    Perform GEMM using SM100 optimized kernel with TMA Multicast

    Computes: D = alpha * A @ B^T + beta * C

    Args:
        A: Input tensor A of shape (M, K), dtype=torch.float16
        B: Input tensor B of shape (N, K), dtype=torch.float16
        C: Input tensor C of shape (M, N), dtype=torch.float32 (optional)
        alpha: Scaling factor for A @ B^T (default from config)
        beta: Scaling factor for C (default from config)
        validate_inputs: Whether to validate inputs (default from config)

    Returns:
        Output tensor D of shape (M, N), dtype=torch.float32

    Note:
        - Automatically uses optimal TMA multicast configuration
        - A and B are K-major (transposed in BLAS terms)
        - C and D are N-major (row-major)
        - All tensors must be on CUDA and properly aligned
    """
    _ensure_sm100_available()

    # Use config defaults if not specified
    if alpha is None:
        alpha = config.default_alpha
    if beta is None:
        beta = config.default_beta
    if validate_inputs is None:
        validate_inputs = config.validate_inputs

    # Enhanced input validation
    if validate_inputs:
        validate_tensor_for_sm100(A, "A", torch.float16)
        validate_tensor_for_sm100(B, "B", torch.float16)

        M, K = A.shape
        N, K_B = B.shape

        if K != K_B:
            raise SM100ValidationError(
                f"Inner dimensions must match: A.shape[1]={K}, B.shape[1]={K_B}"
            )

    # Handle C tensor
    if C is None:
        M, K = A.shape
        N, _ = B.shape
        C = torch.zeros(M, N, dtype=torch.float32, device=A.device)
        beta = 0.0
    else:
        if validate_inputs:
            validate_tensor_for_sm100(C, "C", torch.float32)
            M, N = C.shape
            if A.shape[0] != M or B.shape[0] != N:
                raise SM100ValidationError(
                    f"C shape {C.shape} incompatible with A shape {A.shape} and B shape {B.shape}"
                )

    # Auto-alignment check and warning
    if config.auto_align and validate_inputs:
        M, N, K = A.shape[0], B.shape[0], A.shape[1]
        aligned_M, aligned_N, aligned_K = get_aligned_dimensions(M, N, K)

        if (M, N, K) != (aligned_M, aligned_N, aligned_K):
            if config.warn_on_suboptimal:
                warnings.warn(
                    f"Tensors not optimally aligned. Current: ({M}, {N}, {K}), "
                    f"Optimal: ({aligned_M}, {aligned_N}, {aligned_K}). "
                    f"Consider using create_aligned_tensors() for better performance.",
                    SM100PerformanceWarning,
                )

    # Performance profiling
    if config.profile_performance:
        with cuda_timing() as get_time:
            result = sm100_gemm.sm100_gemm_f16_enhanced(
                A, B, C, alpha, beta, validate_inputs
            )

        # Calculate and display performance metrics
        elapsed_ms = get_time()
        M, N, K = A.shape[0], B.shape[0], A.shape[1]
        flops = 2 * M * N * K
        gflops = flops / (elapsed_ms * 1e-3) / 1e9

        print(f"SM100 Performance: {elapsed_ms:.3f} ms, {gflops:.2f} GFLOPS")

        return result
    else:
        return sm100_gemm.sm100_gemm_f16_enhanced(A, B, C, alpha, beta, validate_inputs)


class SM100Linear(torch.nn.Module):
    """
    Enhanced Linear layer using SM100 GEMM with automatic optimization
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Union[str, torch.device] = "cuda",
        auto_align: bool = True,
    ):
        super().__init__()

        _ensure_sm100_available()

        if isinstance(device, str):
            device = torch.device(device)

        self.orig_in_features = in_features
        self.orig_out_features = out_features
        self.auto_align = auto_align

        # Determine actual dimensions (with optional alignment)
        if auto_align:
            # Align to SM100 requirements for optimal performance
            self.in_features = ((in_features + 63) // 64) * 64
            self.out_features = ((out_features + 255) // 256) * 256
        else:
            self.in_features = in_features
            self.out_features = out_features

        # Initialize weight with proper scaling
        self.weight = torch.nn.Parameter(
            torch.empty(
                self.out_features, self.in_features, dtype=torch.float16, device=device
            )
        )

        if bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(self.out_features, dtype=torch.float32, device=device)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using improved scheme"""
        # Use fan-in initialization for better numerical stability
        fan_in = self.in_features
        std = (2.0 / fan_in) ** 0.5
        with torch.no_grad():
            self.weight.normal_(0, std)
            if self.bias is not None:
                self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # Handle input padding if necessary
        if x.size(1) != self.in_features or self.auto_align:
            # Determine aligned batch size
            aligned_batch = ((batch_size + 127) // 128) * 128

            x_padded = torch.zeros(
                aligned_batch, self.in_features, dtype=torch.float16, device=x.device
            )
            x_padded[:batch_size, : self.orig_in_features] = x.to(torch.float16)
            x = x_padded
        else:
            x = x.to(torch.float16)

        # Prepare bias tensor
        if self.bias is not None:
            C = self.bias.unsqueeze(0).expand(x.size(0), self.out_features).contiguous()
            beta = 1.0
        else:
            C = torch.zeros(
                x.size(0), self.out_features, dtype=torch.float32, device=x.device
            )
            beta = 0.0

        # Perform SM100 GEMM: output = x @ weight^T + bias
        output = sm100_gemm_f16(
            x, self.weight, C, alpha=1.0, beta=beta, validate_inputs=False
        )

        # Remove padding and return
        return output[:batch_size, : self.orig_out_features]

    def extra_repr(self) -> str:
        return (
            f"in_features={self.orig_in_features}, out_features={self.orig_out_features}, "
            f"bias={self.bias is not None}, aligned=({self.in_features}, {self.out_features})"
        )


class SM100Benchmark:
    """Comprehensive benchmarking suite for SM100 operations"""

    def __init__(self, device: Union[str, torch.device] = "cuda"):
        _ensure_sm100_available()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.results = []

    def benchmark_single(
        self,
        M: int,
        N: int,
        K: int,
        num_warmup: int = 10,
        num_trials: int = 50,
        cluster_config: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """Benchmark a single problem size"""

        # Ensure aligned dimensions
        M, N, K = get_aligned_dimensions(M, N, K)

        if cluster_config is None:
            cluster_config = optimize_cluster_size(M, N, K)

        print(f"Benchmarking size ({M}, {N}, {K}) with cluster {cluster_config}")

        # Create test tensors
        A, B, C = create_aligned_tensors(M, N, K, self.device)

        # Warmup
        for _ in range(num_warmup):
            _ = sm100_gemm_f16(A, B, C.clone(), validate_inputs=False)

        torch.cuda.synchronize()

        # Benchmark
        with cuda_timing() as get_time:
            for _ in range(num_trials):
                result = sm100_gemm_f16(A, B, C.clone(), validate_inputs=False)

        elapsed_ms = get_time() / num_trials

        # Calculate metrics
        flops = 2 * M * N * K
        gflops = flops / (elapsed_ms * 1e-3) / 1e9
        memory_bytes = M * K * 2 + N * K * 2 + M * N * 4 * 2
        bandwidth_gbs = memory_bytes / (elapsed_ms * 1e-3) / 1e9

        result_dict = {
            "size": (M, N, K),
            "cluster_config": cluster_config,
            "time_ms": elapsed_ms,
            "gflops": gflops,
            "bandwidth_gbs": bandwidth_gbs,
            "memory_mb": memory_bytes / (1024 * 1024),
            "efficiency": min(gflops / 1500, 1.0),  # Assuming ~1500 GFLOPS peak
        }

        self.results.append(result_dict)
        return result_dict

    def benchmark_scaling(
        self, base_size: int = 512, max_size: int = 4096, num_trials: int = 20
    ) -> List[Dict[str, Any]]:
        """Benchmark scaling characteristics across different problem sizes"""

        print("=== SM100 Scaling Benchmark ===")

        sizes = []
        current = base_size
        while current <= max_size:
            M, N, K = get_aligned_dimensions(current, current, current)
            sizes.append((M, N, K))
            current *= 2

        scaling_results = []

        for M, N, K in sizes:
            try:
                result = self.benchmark_single(
                    M, N, K, num_warmup=5, num_trials=num_trials
                )
                scaling_results.append(result)

                print(
                    f"  {M:4d}x{N:4d}x{K:4d}: {result['time_ms']:6.2f}ms, "
                    f"{result['gflops']:6.1f} GFLOPS, {result['bandwidth_gbs']:6.1f} GB/s"
                )

            except torch.cuda.OutOfMemoryError:
                print(f"  {M:4d}x{N:4d}x{K:4d}: Out of memory")
                break
            except Exception as e:
                print(f"  {M:4d}x{N:4d}x{K:4d}: Error - {e}")
                break

        return scaling_results

    def compare_with_torch(
        self, M: int, N: int, K: int, num_trials: int = 50
    ) -> Dict[str, Any]:
        """Compare SM100 performance with PyTorch's native GEMM"""

        # Ensure aligned dimensions
        M, N, K = get_aligned_dimensions(M, N, K)

        print(f"Comparing SM100 vs PyTorch for size ({M}, {N}, {K})")

        # Create test tensors
        A, B, C = create_aligned_tensors(M, N, K, self.device)
        A_fp32 = A.float()
        B_fp32 = B.float()

        # Warmup both implementations
        for _ in range(10):
            torch_result = torch.addmm(C, A_fp32, B_fp32.T)
            sm100_result = sm100_gemm_f16(A, B, C.clone(), validate_inputs=False)

        torch.cuda.synchronize()

        # Benchmark PyTorch
        with cuda_timing() as get_torch_time:
            for _ in range(num_trials):
                torch_result = torch.addmm(C, A_fp32, B_fp32.T)
        torch_time = get_torch_time() / num_trials

        # Benchmark SM100
        with cuda_timing() as get_sm100_time:
            for _ in range(num_trials):
                sm100_result = sm100_gemm_f16(A, B, C.clone(), validate_inputs=False)
        sm100_time = get_sm100_time() / num_trials

        # Calculate metrics
        flops = 2 * M * N * K
        torch_gflops = flops / (torch_time * 1e-3) / 1e9
        sm100_gflops = flops / (sm100_time * 1e-3) / 1e9
        speedup = torch_time / sm100_time

        # Check correctness
        max_diff = torch.max(torch.abs(torch_result - sm100_result))
        rel_error = max_diff / torch.max(torch.abs(torch_result))

        comparison = {
            "size": (M, N, K),
            "torch_time_ms": torch_time,
            "sm100_time_ms": sm100_time,
            "torch_gflops": torch_gflops,
            "sm100_gflops": sm100_gflops,
            "speedup": speedup,
            "max_diff": max_diff.item(),
            "rel_error": rel_error.item(),
            "correctness_ok": rel_error < 1e-3,
        }

        print(f"  PyTorch: {torch_time:.3f}ms ({torch_gflops:.1f} GFLOPS)")
        print(f"  SM100:   {sm100_time:.3f}ms ({sm100_gflops:.1f} GFLOPS)")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Error:   {rel_error:.2e} (max diff: {max_diff:.2e})")

        return comparison


def demonstrate_features():
    """Demonstrate key SM100 features and capabilities"""

    print("=== SM100 GEMM Feature Demonstration ===\n")

    # 1. Device compatibility check
    print("1. Device Compatibility Check:")
    device_info = SM100DeviceInfo()
    print(device_info)
    print()

    if not device_info.is_supported:
        print("SM100 not supported on this device. Demonstration cannot continue.")
        return

    # 2. Automatic tensor alignment
    print("2. Automatic Tensor Alignment:")
    M, N, K = 1000, 1500, 800
    aligned_M, aligned_N, aligned_K = get_aligned_dimensions(M, N, K)
    print(f"Original dimensions: ({M}, {N}, {K})")
    print(f"Aligned dimensions:  ({aligned_M}, {aligned_N}, {aligned_K})")
    print()

    # 3. Performance estimation
    print("3. Performance Estimation:")
    perf_est = estimate_performance(aligned_M, aligned_N, aligned_K)
    print(f"Estimated performance: {perf_est['estimated_gflops']:.1f} GFLOPS")
    print(f"Estimated bandwidth:   {perf_est['estimated_bandwidth_gbs']:.1f} GB/s")
    print(f"Estimated time:        {perf_est['estimated_time_ms']:.3f} ms")
    print(f"Memory requirement:    {perf_est['memory_mb']:.1f} MB")
    print()

    # 4. Basic GEMM operation
    print("4. Basic GEMM Operation:")
    A, B, C = create_aligned_tensors(aligned_M, aligned_N, aligned_K)

    with cuda_timing() as get_time:
        D = sm100_gemm_f16(A, B, C)
    elapsed = get_time()

    actual_gflops = (2 * aligned_M * aligned_N * aligned_K) / (elapsed * 1e-3) / 1e9
    print(f"Actual performance: {actual_gflops:.1f} GFLOPS in {elapsed:.3f} ms")
    print(f"Output shape: {D.shape}, dtype: {D.dtype}")
    print()

    # 5. Neural network layer demonstration
    print("5. Neural Network Layer:")
    layer = SM100Linear(512, 1024, bias=True)
    print(f"Layer: {layer}")

    batch_input = torch.randn(64, 512, device="cuda")
    with cuda_timing() as get_time:
        output = layer(batch_input)
    elapsed = get_time()

    print(f"Forward pass: {output.shape} in {elapsed:.3f} ms")
    print()

    # 6. Performance comparison
    print("6. Performance Comparison with PyTorch:")
    benchmark = SM100Benchmark()
    comparison = benchmark.compare_with_torch(1024, 1024, 1024, num_trials=20)
    print()

    print("=== Demonstration Complete ===")


def main():
    """Main entry point for SM100 GEMM demonstration and testing"""

    print("=== SM100 GEMM with Enhanced TMA Multicast ===\n")

    try:
        # Check availability and compatibility
        is_supported, message = check_sm100_compatibility()
        if not is_supported:
            print(f"❌ {message}")
            return 1

        print(f"✅ {message}\n")

        # Run feature demonstration
        demonstrate_features()

        # Run scaling benchmark
        print("\n=== Scaling Analysis ===")
        benchmark = SM100Benchmark()
        scaling_results = benchmark.benchmark_scaling(
            base_size=512, max_size=2048, num_trials=10
        )

        if len(scaling_results) >= 2:
            print("\nScaling Efficiency Analysis:")
            base_perf = scaling_results[0]["gflops"]
            print("Size        | Time(ms) | GFLOPS | Efficiency")
            print("-" * 45)

            for result in scaling_results:
                M, N, K = result["size"]
                efficiency = result["gflops"] / base_perf if base_perf > 0 else 0
                print(
                    f"{M:4d}x{N:4d}x{K:3d} | {result['time_ms']:6.2f}  | {result['gflops']:6.1f} | {efficiency:8.2f}"
                )

        print("\n✅ All demonstrations completed successfully!")
        return 0

    except SM100NotAvailableError as e:
        print(f"❌ SM100 not available: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback

        traceback.print_exc()
        return 1


# Convenience functions for common use cases


def quick_gemm(
    A: torch.Tensor, B: torch.Tensor, alpha: float = 1.0, beta: float = 0.0
) -> torch.Tensor:
    """
    Quick GEMM operation with minimal setup

    Args:
        A: Input tensor A (M, K), will be auto-converted to float16
        B: Input tensor B (N, K), will be auto-converted to float16
        alpha: Scaling factor for A @ B^T
        beta: Scaling factor (requires C tensor if non-zero)

    Returns:
        Result tensor D = alpha * A @ B^T + beta * C
    """
    # Auto-convert to appropriate types and devices
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()

    A = A.to(torch.float16)
    B = B.to(torch.float16)

    # Create zero C tensor
    M, K = A.shape
    N, _ = B.shape
    C = torch.zeros(M, N, dtype=torch.float32, device=A.device)

    return sm100_gemm_f16(A, B, C, alpha=alpha, beta=beta)


def benchmark_matmul_sizes(
    sizes: List[Tuple[int, int, int]], num_trials: int = 20
) -> Dict[str, List[float]]:
    """
    Benchmark multiple matrix sizes and return performance data

    Args:
        sizes: List of (M, N, K) tuples to benchmark
        num_trials: Number of timing trials per size

    Returns:
        Dictionary with performance metrics for each size
    """
    benchmark = SM100Benchmark()
    results = {
        "sizes": [],
        "times_ms": [],
        "gflops": [],
        "bandwidths_gbs": [],
        "memory_mbs": [],
    }

    for M, N, K in sizes:
        try:
            result = benchmark.benchmark_single(
                M, N, K, num_warmup=5, num_trials=num_trials
            )
            results["sizes"].append((M, N, K))
            results["times_ms"].append(result["time_ms"])
            results["gflops"].append(result["gflops"])
            results["bandwidths_gbs"].append(result["bandwidth_gbs"])
            results["memory_mbs"].append(result["memory_mb"])
        except Exception as e:
            print(f"Skipped size ({M}, {N}, {K}): {e}")

    return results


def auto_tune_cluster(
    M: int, N: int, K: int, max_trials: int = 10
) -> Tuple[int, int, float]:
    """
    Automatically find the best cluster configuration for given problem size

    Args:
        M, N, K: Matrix dimensions
        max_trials: Maximum number of cluster configurations to try

    Returns:
        Tuple of (best_cluster_m, best_cluster_n, best_gflops)
    """
    # Candidate cluster configurations
    cluster_configs = [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
        (1, 4),
        (4, 1),
        (2, 4),
        (4, 2),
        (4, 4),
    ]

    benchmark = SM100Benchmark()
    best_gflops = 0.0
    best_config = (2, 2)

    print(f"Auto-tuning cluster configuration for ({M}, {N}, {K})...")

    for cluster_m, cluster_n in cluster_configs[:max_trials]:
        if cluster_m * cluster_n > 16:  # Hardware limit
            continue

        try:
            result = benchmark.benchmark_single(
                M,
                N,
                K,
                num_warmup=3,
                num_trials=10,
                cluster_config=(cluster_m, cluster_n),
            )

            gflops = result["gflops"]
            print(f"  Cluster {cluster_m}x{cluster_n}: {gflops:.1f} GFLOPS")

            if gflops > best_gflops:
                best_gflops = gflops
                best_config = (cluster_m, cluster_n)

        except Exception as e:
            print(f"  Cluster {cluster_m}x{cluster_n}: Failed - {e}")

    print(
        f"Best configuration: {best_config[0]}x{best_config[1]} ({best_gflops:.1f} GFLOPS)"
    )
    return best_config[0], best_config[1], best_gflops


# Context managers for configuration


@contextmanager
def sm100_config(**kwargs):
    """Context manager for temporary SM100 configuration changes"""
    with config.temporary_config(**kwargs):
        yield


@contextmanager
def sm100_profiling():
    """Context manager to enable performance profiling"""
    with config.temporary_config(profile_performance=True):
        yield


@contextmanager
def sm100_fast_mode():
    """Context manager for maximum performance (minimal validation)"""
    with config.temporary_config(
        validate_inputs=False, warn_on_suboptimal=False, profile_performance=False
    ):
        yield


# Export public API
__all__ = [
    # Main functions
    "sm100_gemm_f16",
    "quick_gemm",
    # Device and compatibility
    "check_sm100_compatibility",
    "SM100DeviceInfo",
    # Tensor utilities
    "create_aligned_tensors",
    "get_aligned_dimensions",
    # Performance and optimization
    "estimate_performance",
    "optimize_cluster_size",
    "auto_tune_cluster",
    # Neural network layers
    "SM100Linear",
    # Benchmarking
    "SM100Benchmark",
    "benchmark_matmul_sizes",
    # Configuration
    "SM100Config",
    "config",
    "sm100_config",
    "sm100_profiling",
    "sm100_fast_mode",
    # Exceptions
    "SM100NotAvailableError",
    "SM100ValidationError",
    "SM100PerformanceWarning",
    # Main demo
    "main",
    "demonstrate_features",
]

if __name__ == "__main__":
    exit(main())
