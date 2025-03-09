# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import triton
import triton.language as tl
from tma_utils import TmaAutoTuneHelper
from triton.runtime import driver

"""
Current errors:
2025-03-09 12:36:54,566 - INFO - Testing GMM Backward with G=1, M=64
EFE2025-03-09 12:36:58,343 - INFO - Testing BF16 GMM with G=1, M=64
2025-03-09 12:36:58,485 - INFO - Testing BF16 GMM with G=1, M=512
2025-03-09 12:36:59,717 - INFO - Testing BF16 GMM with G=4, M=64
F
======================================================================
ERROR: test_grouped_gemm_backward (unit_tests_back.TestGroupedGEMM.test_grouped_gemm_backward)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/data/users/less/applied-ai/dev/triton_groupGEMM/clean/unit_tests_back.py", line 217, in test_grouped_gemm_backward
    _test_grouped_gemm_backward((G, M, 256, 256), torch.device("cuda"))
  File "/data/users/less/applied-ai/dev/triton_groupGEMM/clean/unit_tests_back.py", line 171, in _test_grouped_gemm_backward
    result.backward(grad_output)
  File "/home/less/.conda/envs/tritondev/lib/python3.12/site-packages/torch/_tensor.py", line 648, in backward
    torch.autograd.backward(
  File "/home/less/.conda/envs/tritondev/lib/python3.12/site-packages/torch/autograd/__init__.py", line 353, in backward
    _engine_run_backward(
  File "/home/less/.conda/envs/tritondev/lib/python3.12/site-packages/torch/autograd/graph.py", line 824, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/less/.conda/envs/tritondev/lib/python3.12/site-packages/torch/autograd/function.py", line 307, in apply
    return user_fn(self, *args)
           ^^^^^^^^^^^^^^^^^^^^
  File "/data/users/less/applied-ai/dev/triton_groupGEMM/clean/unit_tests_back.py", line 53, in backward
    grad_x, grad_w = grouped_gemm_backward(grad_output, x, w, m_sizes)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/users/less/applied-ai/dev/triton_groupGEMM/clean/tgrouped_gemm_backwards.py", line 580, in grouped_gemm_backward
    _kernel_grouped_gemm_backward_x[grid_x](
  File "/data/users/less/triton/python/triton/runtime/jit.py", line 336, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/users/less/triton/python/triton/runtime/autotuner.py", line 189, in run
    timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/users/less/triton/python/triton/runtime/autotuner.py", line 167, in _bench
    return self.do_bench(kernel_call, quantiles=(0.5, 0.2, 0.8))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/users/less/triton/python/triton/testing.py", line 145, in do_bench
    fn()
  File "/data/users/less/triton/python/triton/runtime/autotuner.py", line 153, in kernel_call
    self.fn.run(
  File "/data/users/less/triton/python/triton/runtime/jit.py", line 563, in run
    kernel = self.compile(src, target=target, options=options.__dict__)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/users/less/triton/python/triton/compiler/compiler.py", line 278, in compile
    module = src.make_ir(options, codegen_fns, module_map, context)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/users/less/triton/python/triton/compiler/compiler.py", line 81, in make_ir
    return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
triton.compiler.errors.UnsupportedLanguageConstruct: at 51:12:
    M_end_offset = 0
    iterated_tiles = 0

    for g in range(G):
        # For each group
        M_start_offset = M_end_offset
        m_size = tl.load(m_sizes + g)
        M_end_offset = M_start_offset + m_size

        # Skip empty groups
        if m_size <= 0:
            continue
            ^
unsupported AST node type: Continue

======================================================================
ERROR: test_grouped_gemm_backward_numerical_gradient (unit_tests_back.TestGroupedGEMM.test_grouped_gemm_backward_numerical_gradient)
Test backward pass using numerical gradients for verification
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/data/users/less/applied-ai/dev/triton_groupGEMM/clean/unit_tests_back.py", line 330, in test_grouped_gemm_backward_numerical_gradient
    output_bf16 = GroupedGEMMFunction.apply(a_bf16, b_bf16, m_sizes)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/less/.conda/envs/tritondev/lib/python3.12/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/users/less/applied-ai/dev/triton_groupGEMM/clean/unit_tests_back.py", line 40, in forward
    output = grouped_gemm(x, w, m_sizes)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/users/less/applied-ai/dev/triton_groupGEMM/groupgemm.py", line 521, in grouped_gemm
    return _grouped_gemm(x, w, m_sizes)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/users/less/applied-ai/dev/triton_groupGEMM/groupgemm.py", line 500, in _grouped_gemm
    _kernel_grouped_gemm[grid](
  File "/data/users/less/triton/python/triton/runtime/jit.py", line 336, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/users/less/triton/python/triton/runtime/autotuner.py", line 192, in run
    self.cache[key] = builtins.min(timings, key=timings.get)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: min() iterable argument is empty

======================================================================
FAIL: test_grouped_gemm_backward_nontrivial_groups (unit_tests_back.TestGroupedGEMM.test_grouped_gemm_backward_nontrivial_groups)
Test backward pass with specific non-trivial group sizes
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/data/users/less/applied-ai/dev/triton_groupGEMM/clean/unit_tests_back.py", line 291, in test_grouped_gemm_backward_nontrivial_groups
    _test_grouped_gemm_backward_custom_groups(
  File "/data/users/less/applied-ai/dev/triton_groupGEMM/clean/unit_tests_back.py", line 256, in _test_grouped_gemm_backward_custom_groups
    self.assertEqual(
AssertionError: torch.Size([128, 64]) != (128, 256) : Forward result shape mismatch, expected (128, 256), got torch.Size([128, 64])

======================================================================
FAIL: test_grouped_gemm_bf16 (unit_tests_back.TestGroupedGEMM.test_grouped_gemm_bf16)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/data/users/less/applied-ai/dev/triton_groupGEMM/clean/unit_tests_back.py", line 118, in test_grouped_gemm_bf16
    _test_grouped_gemm_bf16((G, M, 256, 256), torch.device("cuda"))
  File "/data/users/less/applied-ai/dev/triton_groupGEMM/clean/unit_tests_back.py", line 94, in _test_grouped_gemm_bf16
    self.assertEqual(result.shape, (M, N * G))
AssertionError: torch.Size([64, 256]) != (64, 1024)

----------------------------------------------------------------------
Ran 4 tests in 6.378s

FAILED (failures=2, errors=2)
"""

# NVIDIA configurations - with balanced block sizes for H100
_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": block_size_n,
            "BLOCK_SIZE_K": block_size_k,
        },
        num_stages=num_stages,
        num_warps=num_warps,
        num_ctas=num_ctas,
    )
    for block_size_m in [32, 64, 128]  # Balanced range for M dimension
    for block_size_n in [32, 64, 128]  # Balanced range for N dimension
    for block_size_k in [32, 64, 128]  # Balanced range for K dimension
    for num_stages in [2, 3]  # Reduced stages for memory efficiency
    for num_warps in [4, 8]  # Common warp counts
    for num_ctas in [1]  # Single CTA for simplicity
]


def early_config_prune(configs, name_args, **kwargs):
    """
    Prune configurations based on hardware constraints and problem size.

    Args:
        configs: List of triton configurations
        name_args: Dictionary of kernel arguments
        **kwargs: Additional keyword arguments

    Returns:
        List of pruned configurations that should work well for the given problem
    """
    device = torch.cuda.current_device()

    # Get element size for the tensor we're computing gradients for
    dtsize = (
        name_args["grad_x_ptr"].element_size()
        if "grad_x_ptr" in name_args
        else name_args["grad_w_ptr"].element_size()
    )

    pruned_configs = []

    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, BLOCK_K, num_stages = (
            kw["BLOCK_SIZE_M"],
            kw["BLOCK_SIZE_N"],
            kw["BLOCK_SIZE_K"],
            config.num_stages,
        )
        G, M, N, K = (
            name_args["G"],
            name_args["M_bucket"],
            name_args["N"],
            name_args["K"],
        )

        # Check shared memory requirements
        max_shared_memory = driver.active.utils.get_device_properties(device)[
            "max_shared_mem"
        ]
        required_shared_memory = (BLOCK_M + BLOCK_N) * BLOCK_K * num_stages * dtsize
        if required_shared_memory > max_shared_memory:
            continue

        # Check K dimension alignment - critical for performance
        if K % BLOCK_K != 0 and K > BLOCK_K:
            continue

        # Check if N dimension is suitable for the block size
        if N % BLOCK_N != 0 and N > BLOCK_N * 2:
            continue

        # Add this config to our pruned list
        pruned_configs.append(config)

    # Ensure we always have at least one config
    if not pruned_configs and configs:
        # Add the smallest configuration as a fallback
        smallest_config = min(
            configs,
            key=lambda c: (
                c.kwargs["BLOCK_SIZE_M"]
                * c.kwargs["BLOCK_SIZE_N"]
                * c.kwargs["BLOCK_SIZE_K"]
            ),
        )
        pruned_configs.append(smallest_config)

    return pruned_configs


# ======= Backwards for weights (d_W) ======================================
@triton.autotune(
    configs=_CONFIGS,
    key=["G", "M_bucket", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def _kernel_grouped_gemm_backward_w(
    x_t_ptr,  # x transposed [K, M]
    grad_y_ptr,  # grad of dl/dY [M, N*G]
    grad_w_ptr,  # output of kernel (grad_w) [N*G, K]
    workspace,
    m_sizes,
    # problem sizes
    G: tl.constexpr,
    M_bucket: tl.constexpr,
    N: tl.constexpr,  # N is per group
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
) -> None:
    """
    Compute gradients with respect to w (weights).

    grad_w = x_t (x transposed) @ grad_y (dl/dY)

    Here:
    - x_t is [K, M] (transposed from [M, K])
    - grad_y is [M, N*G] where N is output dim per group
    - grad_w is [N*G, K]
    """
    tidx = tl.program_id(0)
    dtype = grad_w_ptr.dtype.element_ty
    TMA_SIZE: tl.constexpr = 128

    # Initialize workspace pointer if using TMA store
    if USE_TMA_STORE:
        c_desc_ptr = workspace + tidx * TMA_SIZE
    else:
        c_desc_ptr = None

    M_end_offset = 0
    iterated_tiles = 0

    for g in range(G):
        # For each group
        M_start_offset = M_end_offset
        m_size = tl.load(m_sizes + g)
        M_end_offset = M_start_offset + m_size

        # Skip empty groups
        if m_size <= 0:
            continue

        # N_start_offset is where this group's weights start in the N*G dimension
        N_start_offset = g * N

        # Calculate tiles for this group
        num_m_tiles = tl.cdiv(N, BLOCK_SIZE_M)  # Tiles along N dimension
        num_n_tiles = tl.cdiv(K, BLOCK_SIZE_N)  # Tiles along K dimension
        num_tiles = num_m_tiles * num_n_tiles

        # Setup TMA descriptor for output if using TMA
        if USE_TMA_STORE:
            tl.static_assert(hasattr(tl.extra, "cuda"), "TMA requires CUDA support")
            tl.extra.cuda.experimental_device_tensormap_create2d(
                desc_ptr=c_desc_ptr,
                global_address=grad_w_ptr + N_start_offset * K,
                load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                global_size=[N, K],
                element_ty=dtype,
            )
            tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

        # Process tiles
        while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
            gidx = tidx - iterated_tiles

            # Calculate tile indices
            tile_m_idx = gidx % num_m_tiles  # Tile index along N dimension
            tile_n_idx = gidx // num_m_tiles  # Tile index along K dimension

            # Initialize accumulator for this tile
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            # Calculate offsets for better memory access
            offs_m = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

            # Create masks for boundary checking
            m_mask = offs_m < N
            n_mask = offs_n < K

            # Loop over the reduction dimension (M)
            for k_offset in range(0, m_size, BLOCK_SIZE_K):
                # Handle boundary conditions for the reduction dimension
                k_size = tl.minimum(BLOCK_SIZE_K, m_size - k_offset)
                offs_k = tl.arange(0, BLOCK_SIZE_K)
                k_mask = offs_k < k_size

                # Load grad_y [M, N*G] block
                # Shape: [BLOCK_SIZE_K, BLOCK_SIZE_M]
                grad_y_block = tl.load(
                    grad_y_ptr
                    + (M_start_offset + k_offset + offs_k[:, None]) * (N * G)
                    + (N_start_offset + offs_m[None, :]),
                    mask=k_mask[:, None] & m_mask[None, :],
                    other=0.0,
                )

                # Load x_t [K, M] block
                # Shape: [BLOCK_SIZE_N, BLOCK_SIZE_K]
                x_t_block = tl.load(
                    x_t_ptr
                    + offs_n[:, None] * M_bucket
                    + (M_start_offset + k_offset + offs_k[None, :]),
                    mask=n_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )

                # Matrix multiplication: (x_t_block.T @ grad_y_block.T).T
                # This computes a portion of x_t.T @ grad_y for the current tile
                accumulator += tl.dot(
                    x_t_block.to(tl.float32).T,
                    grad_y_block.to(tl.float32),
                    allow_tf32=True,
                ).T

            # Store result to grad_w
            if USE_TMA_STORE:
                # Use TMA to store the result
                m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)

                # Convert to output dtype and store
                tl._experimental_descriptor_store(
                    c_desc_ptr,
                    accumulator.to(dtype),
                    [m_offset, n_offset],
                )
            else:
                # Manual store with boundary checking
                tl.store(
                    grad_w_ptr
                    + (N_start_offset + offs_m[:, None]) * K
                    + offs_n[None, :],
                    accumulator.to(dtype),
                    mask=m_mask[:, None] & n_mask[None, :],
                )

            # Move to next tile
            tidx += NUM_SMS

        # Update tiles processed counter
        iterated_tiles += num_tiles


# ======= Backwards for inputs (d_X) ======================================
@triton.autotune(
    configs=_CONFIGS,
    key=["G", "M_bucket", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def _kernel_grouped_gemm_backward_x(
    grad_y_ptr,  # grad of dl/dY [M, N*G]
    w_t_ptr,  # w transposed [K, N*G]
    grad_x_ptr,  # output of kernel [M, K]
    workspace,
    m_sizes,
    # problem sizes
    G: tl.constexpr,
    M_bucket: tl.constexpr,
    N: tl.constexpr,  # N is per group
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
) -> None:
    """
    Compute gradients with respect to x (input).

    grad_x = grad_y (dl/dY) @ w_t (w transposed)

    Here:
    - grad_y is [M, N*G] where N is output dim per group
    - w_t is [K, N*G] (transposed from [N*G, K])
    - grad_x is [M, K]
    """
    tidx = tl.program_id(0)
    dtype = grad_x_ptr.dtype.element_ty
    TMA_SIZE: tl.constexpr = 128

    # Initialize workspace pointer if using TMA store
    if USE_TMA_STORE:
        c_desc_ptr = workspace + tidx * TMA_SIZE
    else:
        c_desc_ptr = None

    M_end_offset = 0
    iterated_tiles = 0

    for g in range(G):
        # For each group
        M_start_offset = M_end_offset
        m_size = tl.load(m_sizes + g)
        M_end_offset = M_start_offset + m_size

        # Skip empty groups
        if m_size <= 0:
            continue

        # N_start_offset is where this group's weights start in the N*G dimension
        N_start_offset = g * N

        # Calculate tiles for this group
        num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)  # Tiles along M dimension
        num_n_tiles = tl.cdiv(K, BLOCK_SIZE_N)  # Tiles along K dimension
        num_tiles = num_m_tiles * num_n_tiles

        # Setup TMA descriptor for output if using TMA
        if USE_TMA_STORE:
            tl.static_assert(hasattr(tl.extra, "cuda"), "TMA requires CUDA support")
            tl.extra.cuda.experimental_device_tensormap_create2d(
                desc_ptr=c_desc_ptr,
                global_address=grad_x_ptr + M_start_offset * K,
                load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                global_size=[m_size, K],
                element_ty=dtype,
            )
            tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

        # Process tiles
        while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
            gidx = tidx - iterated_tiles

            # Calculate tile indices
            tile_m_idx = gidx % num_m_tiles  # Tile index along M dimension
            tile_n_idx = gidx // num_m_tiles  # Tile index along K dimension

            # Initialize accumulator for this tile
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            # Calculate offsets for better memory access
            offs_m = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

            # Create masks for boundary checking
            m_mask = offs_m < m_size
            n_mask = offs_n < K

            # Loop over the reduction dimension (N)
            for k_offset in range(0, N, BLOCK_SIZE_K):
                # Handle boundary conditions for the reduction dimension
                k_size = tl.minimum(BLOCK_SIZE_K, N - k_offset)
                offs_k = tl.arange(0, BLOCK_SIZE_K)
                k_mask = offs_k < k_size

                # Load grad_y [M, N*G] block
                # Shape: [BLOCK_SIZE_M, BLOCK_SIZE_K]
                grad_y_block = tl.load(
                    grad_y_ptr
                    + (M_start_offset + offs_m[:, None]) * (N * G)
                    + (N_start_offset + k_offset + offs_k[None, :]),
                    mask=m_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )

                # Load w_t [K, N*G] block
                # Shape: [BLOCK_SIZE_N, BLOCK_SIZE_K]
                w_t_block = tl.load(
                    w_t_ptr
                    + offs_n[:, None] * (N * G)
                    + (N_start_offset + k_offset + offs_k[None, :]),
                    mask=n_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )

                # Matrix multiplication: grad_y @ w_t.T
                # This computes a portion of grad_y @ w_t.T for the current tile
                accumulator += tl.dot(
                    grad_y_block.to(tl.float32),
                    w_t_block.to(tl.float32).T,
                    allow_tf32=True,
                )

            # Store result to grad_x
            if USE_TMA_STORE:
                # Use TMA to store the result
                m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)

                # Convert to output dtype and store
                tl._experimental_descriptor_store(
                    c_desc_ptr,
                    accumulator.to(dtype),
                    [m_offset, n_offset],
                )
            else:
                # Manual store with boundary checking
                tl.store(
                    grad_x_ptr
                    + (M_start_offset + offs_m[:, None]) * K
                    + offs_n[None, :],
                    accumulator.to(dtype),
                    mask=m_mask[:, None] & n_mask[None, :],
                )

            # Move to next tile
            tidx += NUM_SMS

        # Update tiles processed counter
        iterated_tiles += num_tiles


def grouped_gemm_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for grouped matrix multiplication.

    Computes gradients with respect to x and w.

    Args:
        grad_output: Gradient with respect to output, shape [M, N*G]
        x: Input tensor from forward pass, shape [M, K]
        w: Weight tensor from forward pass, shape [N*G, K]
        m_sizes: Group sizes tensor, shape [G]

    Returns:
        Tuple of gradients with respect to x and w: (grad_x, grad_w)
    """
    if not hasattr(torch.cuda, "current_device"):
        raise RuntimeError("CUDA not available for backward pass")

    # Get GPU parameters
    device_props = torch.cuda.get_device_properties("cuda")
    if device_props.major < 9:
        raise RuntimeError("H100 or newer GPU required for grouped GEMM")

    if not hasattr(tl.extra, "cuda"):
        raise NotImplementedError("TMA support is required but not available")

    G = m_sizes.shape[0]

    # Ensure contiguous tensors for efficient memory access
    grad_output = grad_output.contiguous()
    x = x.contiguous()
    w = w.contiguous()
    m_sizes = m_sizes.contiguous()

    M, K = x.shape
    N_times_G, K_w = w.shape

    # Validate dimensions
    assert K == K_w, f"Input K ({K}) must match weight K ({K_w})"
    assert (
        N_times_G % G == 0
    ), f"Weight dim 0 ({N_times_G}) must be divisible by number of groups ({G})"

    # Calculate N - output dimension per group
    N = N_times_G // G

    # Verify grad_output shape
    expected_grad_output_shape = (M, N_times_G)
    assert grad_output.shape == expected_grad_output_shape, (
        f"grad_output shape mismatch: got {grad_output.shape}, "
        f"expected {expected_grad_output_shape}"
    )

    # Transpose x and w for backward computation
    x_t = x.T.contiguous()  # Shape: [K, M]
    w_t = w.T.contiguous()  # Shape: [K, N*G]

    # Allocate output tensors with correct shapes
    grad_x = torch.empty_like(x)
    grad_w = torch.empty_like(w)

    # Configure kernel parameters
    NUM_SMS = device_props.multi_processor_count
    USE_TMA_LOAD = True  # Use TMA for loading
    USE_TMA_STORE = False  # Disable TMA store for better compatibility

    # Setup TMA descriptors
    desc_helper = TmaAutoTuneHelper()

    # Create descriptors for all tensors
    desc_helper.init_tma_descriptor("grad_output")
    desc_helper.init_tma_descriptor("w_t")
    desc_helper.init_tma_descriptor("x_t")

    # Get pointers to descriptors
    grad_y_ptr = desc_helper.get_tma_descriptor_kernel_param("grad_output")
    w_t_ptr = desc_helper.get_tma_descriptor_kernel_param("w_t")
    x_t_ptr = desc_helper.get_tma_descriptor_kernel_param("x_t")

    # Allocate workspace for TMA descriptors
    workspace = torch.empty(
        (NUM_SMS * 128),  # TMA_SIZE
        device=x.device,
        dtype=torch.uint8,
    )

    # Setup grid for grad_x kernel
    def grid_x(META):
        # Configure TMA descriptor for grad_output
        desc_helper.fill_2d_tma_descriptor(
            "grad_output",
            grad_output.data_ptr(),
            M,
            N_times_G,
            META["BLOCK_SIZE_M"],
            META["BLOCK_SIZE_K"],
            grad_output.element_size(),
        )

        # Configure TMA descriptor for w_t
        desc_helper.fill_2d_tma_descriptor(
            "w_t",
            w_t.data_ptr(),
            K,
            N_times_G,
            META["BLOCK_SIZE_N"],
            META["BLOCK_SIZE_K"],
            w_t.element_size(),
        )

        return (NUM_SMS,)

    # Setup grid for grad_w kernel
    def grid_w(META):
        # Configure TMA descriptor for x_t
        desc_helper.fill_2d_tma_descriptor(
            "x_t",
            x_t.data_ptr(),
            K,
            M,
            META["BLOCK_SIZE_N"],
            META["BLOCK_SIZE_K"],
            x_t.element_size(),
        )

        # Configure TMA descriptor for grad_output
        desc_helper.fill_2d_tma_descriptor(
            "grad_output",
            grad_output.data_ptr(),
            M,
            N_times_G,
            META["BLOCK_SIZE_K"],
            META["BLOCK_SIZE_M"],
            grad_output.element_size(),
        )

        return (NUM_SMS,)

    # Use the next power of 2 for M to avoid alignment issues
    M_bucket = triton.next_power_of_2(M)

    # First compute grad_x: grad_y @ w_t.T
    _kernel_grouped_gemm_backward_x[grid_x](
        grad_y_ptr,
        w_t_ptr,
        grad_x,
        workspace,
        m_sizes,
        G,
        M_bucket,
        N,  # N per group
        K,
        NUM_SMS,
        USE_TMA_LOAD,
        USE_TMA_STORE,
    )

    # Then compute grad_w: x_t.T @ grad_y
    _kernel_grouped_gemm_backward_w[grid_w](
        x_t_ptr,
        grad_y_ptr,
        grad_w,
        workspace,
        m_sizes,
        G,
        M_bucket,
        N,  # N per group
        K,
        NUM_SMS,
        USE_TMA_LOAD,
        USE_TMA_STORE,
    )

    # Verify output shapes
    assert (
        grad_x.shape == x.shape
    ), f"grad_x shape mismatch: got {grad_x.shape}, expected {x.shape}"
    assert (
        grad_w.shape == w.shape
    ), f"grad_w shape mismatch: got {grad_w.shape}, expected {w.shape}"

    return grad_x, grad_w
