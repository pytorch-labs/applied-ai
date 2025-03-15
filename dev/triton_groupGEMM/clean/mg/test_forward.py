import os
import sys
import time
from typing import List, Tuple

import numpy as np
import torch

from forward_group_gemm import group_gemm_forward

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from forward_group_gemm import group_gemm_forward as grouped_gemm_forward
except ImportError:
    print("Error: grouped_gemm_forward module not found.")
    print("Make sure the grouped GEMM implementation is in the current directory.")
    sys.exit(1)


def reference_grouped_gemm(
    x: torch.Tensor, w: torch.Tensor, m_sizes: torch.Tensor
) -> torch.Tensor:
    """
    Reference implementation using PyTorch's native matmul function.

    Args:
        x: Input tensor of shape [M_total, K] where M_total is sum of all group sizes
        w: Weight tensor of shape [N, K]
        m_sizes: Tensor of shape [G] containing the size of each group

    Returns:
        Output tensor of shape [M_total, N]
    """
    device = x.device
    dtype = x.dtype
    G = m_sizes.shape[0]
    M_total, K = x.shape
    N = w.shape[0]

    # Create output tensor
    y_ref = torch.empty((M_total, N), device=device, dtype=dtype)

    # Process each group
    start_idx = 0
    for g in range(G):
        m_size = m_sizes[g].item()
        if m_size > 0:
            # Extract current group input
            x_g = x[start_idx : start_idx + m_size]

            # Perform matrix multiplication
            y_g = torch.matmul(x_g, w.t())

            # Store result
            y_ref[start_idx : start_idx + m_size] = y_g

            # Update start index
            start_idx += m_size

    return y_ref


def generate_test_data(
    G: int,
    m_sizes: List[int],
    N: int,
    K: int,
    device: torch.device,
    dtype=torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate test data for grouped GEMM.

    Args:
        G: Number of groups
        m_sizes: List of sizes for each group
        N: Output feature dimension
        K: Input feature dimension
        device: Device to place tensors on
        dtype: Data type for tensors

    Returns:
        x: Input tensor of shape [M_total, K]
        w: Weight tensor of shape [N, K]
        m_sizes_tensor: Tensor of shape [G] containing the size of each group
    """
    assert len(m_sizes) == G, f"Expected {G} group sizes, got {len(m_sizes)}"

    # Calculate total M dimension
    M_total = sum(m_sizes)

    # Create tensors with controlled values for easier debugging
    x = torch.randn((M_total, K), device=device, dtype=dtype)
    w = torch.randn((N, K), device=device, dtype=dtype)
    m_sizes_tensor = torch.tensor(m_sizes, device=device, dtype=torch.int32)

    return x, w, m_sizes_tensor


def verify_grouped_gemm(
    G: int,
    m_sizes: List[int],
    N: int,
    K: int,
    device: torch.device = torch.device("cuda"),
    atol: float = 1e-2,
    rtol: float = 1e-2,
    dtype=torch.bfloat16,
    verbose: bool = True,
) -> bool:
    """
    Verify the correctness of the grouped GEMM implementation.

    Args:
        G: Number of groups
        m_sizes: List of sizes for each group
        N: Output feature dimension
        K: Input feature dimension
        device: Device to place tensors on
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison
        dtype: Data type for tensors
        verbose: Whether to print verbose output

    Returns:
        bool: True if the implementation is correct, False otherwise
    """
    torch.manual_seed(42)  # For reproducibility

    # Generate test data
    if verbose:
        print(f"Generating test data with G={G}, N={N}, K={K}, m_sizes={m_sizes}")
    x, w, m_sizes_tensor = generate_test_data(G, m_sizes, N, K, device, dtype)

    # Run reference implementation
    start_time = time.time()
    y_ref = reference_grouped_gemm(x, w, m_sizes_tensor)
    ref_time = time.time() - start_time

    # Run our implementation
    start_time = time.time()
    y = grouped_gemm_forward(x, w, m_sizes_tensor)
    impl_time = time.time() - start_time

    # Convert to float32 for accurate comparison
    y_ref = y_ref.to(torch.float32)
    y = y.to(torch.float32)

    # Compare outputs
    max_abs_diff = torch.max(torch.abs(y - y_ref)).item()
    max_rel_diff = torch.max(torch.abs((y - y_ref) / (y_ref + 1e-6))).item()

    if verbose:
        print(f"Reference implementation time: {ref_time:.6f}s")
        print(f"Our implementation time: {impl_time:.6f}s")
        print(f"Speedup: {ref_time / impl_time:.2f}x")
        print(f"Max absolute difference: {max_abs_diff:.6e}")
        print(f"Max relative difference: {max_rel_diff:.6e}")

    # Check if outputs match within tolerance
    is_close = torch.allclose(y, y_ref, atol=atol, rtol=rtol)

    if verbose:
        if is_close:
            print("✓ Outputs match within tolerance")
        else:
            print("✗ Outputs do not match within tolerance")

            # If not close, provide more debug information
            not_close = ~torch.isclose(y, y_ref, atol=atol, rtol=rtol)
            num_mismatched = torch.sum(not_close).item()
            percent_mismatched = 100.0 * num_mismatched / y.numel()
            print(
                f"Number of mismatched elements: {num_mismatched} ({percent_mismatched:.2f}%)"
            )

            if num_mismatched > 0:
                # Get indices of first few mismatches
                mismatched_indices = torch.nonzero(not_close, as_tuple=False)
                print("\nFirst few mismatches:")
                for idx in range(min(5, mismatched_indices.shape[0])):
                    i, j = mismatched_indices[idx]
                    i, j = i.item(), j.item()
                    print(
                        f"  Position [{i}, {j}]: Our={y[i, j].item():.6f}, Ref={y_ref[i, j].item():.6f}, "
                        f"Diff={y[i, j].item() - y_ref[i, j].item():.6f}"
                    )

    return is_close


def run_tests():
    """Run a series of tests to verify the grouped GEMM implementation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: Running on CPU. Tests will be slow.")

    print("=" * 80)
    print("Grouped GEMM Verification Tests")
    print("=" * 80)

    all_passed = True

    # Test 1: Small groups with equal sizes
    print("\nTest 1: Small groups with equal sizes")
    G, N, K = 4, 32, 64
    m_sizes = [128, 128, 128, 128]
    test_passed = verify_grouped_gemm(G, m_sizes, N, K, device)
    all_passed = all_passed and test_passed

    # Test 2: Groups with varying sizes
    print("\nTest 2: Groups with varying sizes")
    G, N, K = 3, 64, 128
    m_sizes = [32, 64, 16]
    test_passed = verify_grouped_gemm(G, m_sizes, N, K, device)
    all_passed = all_passed and test_passed

    # Test 3: Some empty groups
    print("\nTest 3: Some empty groups")
    G, N, K = 5, 32, 64
    m_sizes = [16, 0, 24, 0, 8]
    test_passed = verify_grouped_gemm(G, m_sizes, N, K, device)
    all_passed = all_passed and test_passed

    # Test 4: Larger dimensions
    print("\nTest 4: Larger dimensions")
    G, N, K = 2, 128, 256
    m_sizes = [96, 128]
    test_passed = verify_grouped_gemm(G, m_sizes, N, K, device)
    all_passed = all_passed and test_passed

    # Test 5: Single group
    print("\nTest 5: Single group")
    G, N, K = 1, 64, 128
    m_sizes = [48]
    test_passed = verify_grouped_gemm(G, m_sizes, N, K, device)
    all_passed = all_passed and test_passed

    # Summary
    print("\n" + "=" * 80)
    if all_passed:
        print("All tests passed! The grouped GEMM implementation is correct.")
    else:
        print("Some tests failed. Please check the implementation.")
    print("=" * 80)


def benchmark(num_runs=10):
    """
    Benchmark the performance of the grouped GEMM implementation against the reference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test configurations
    configs = [
        # G, m_sizes, N, K
        (4, [1024, 1024, 64, 64], 128, 256),
        (8, [64, 64, 32, 32, 32, 32, 32, 32], 128, 256),
        (2, [512, 512], 256, 512),
        (16, [32] * 16, 64, 128),
    ]

    print("\n" + "=" * 80)
    print("Performance Benchmark")
    print("=" * 80)

    for config_idx, (G, m_sizes, N, K) in enumerate(configs):
        print(f"\nConfiguration {config_idx+1}:")
        print(f"  G={G}, N={N}, K={K}, M_total={sum(m_sizes)}")

        # Generate test data
        x, w, m_sizes_tensor = generate_test_data(G, m_sizes, N, K, device)

        # Warmup
        for _ in range(5):
            reference_grouped_gemm(x, w, m_sizes_tensor)
            grouped_gemm_forward(x, w, m_sizes_tensor)

        # Time reference implementation
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_runs):
            y_ref = reference_grouped_gemm(x, w, m_sizes_tensor)
            torch.cuda.synchronize()
        ref_time = (time.time() - start_time) / num_runs

        # Time our implementation
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_runs):
            y = grouped_gemm_forward(x, w, m_sizes_tensor)
            torch.cuda.synchronize()
        impl_time = (time.time() - start_time) / num_runs

        # Check correctness
        # In the benchmark function, replace the correctness check with:
        # Check correctness
        y_float = y.to(torch.float32)
        y_ref_float = y_ref.to(torch.float32)
        is_close = torch.allclose(y_float, y_ref_float, atol=1e-1, rtol=1e-1)

        if not is_close:
            max_abs_diff = torch.max(torch.abs(y_float - y_ref_float)).item()
            max_rel_diff = torch.max(
                torch.abs((y_float - y_ref_float) / (y_ref_float + 1e-6))
            ).item()
            not_close = ~torch.isclose(y_float, y_ref_float, atol=1e-1, rtol=1e-1)
            num_mismatched = torch.sum(not_close).item()
            percent_mismatched = 100.0 * num_mismatched / y.numel()
            print(f"  Max absolute difference: {max_abs_diff:.6e}")
            print(f"  Max relative difference: {max_rel_diff:.6e}")
            print(
                f"  Mismatched elements: {num_mismatched} ({percent_mismatched:.2f}%)"
            )

        # Report results
        print(f"  Reference implementation: {ref_time*1000:.3f} ms")
        print(f"  Our implementation: {impl_time*1000:.3f} ms")
        print(f"  Speedup: {ref_time / impl_time:.2f}x")
        print(f"  Correct output: {'Yes' if is_close else 'No'}")


if __name__ == "__main__":
    run_tests()
    benchmark()
