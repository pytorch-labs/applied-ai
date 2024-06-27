# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import logging
from typing import List, Optional, Tuple

import torch
import triton  # @manual

import triton.language as tl  # @manual
from torch._tensor import Tensor

from triton import Config  # @manual
from triton.ops.matmul_perf_model import (  # @manual
    early_config_prune,
    estimate_matmul_time,
)
from triton.runtime.jit import reinterpret as tl_reinterpret, TensorWrapper  # @manual

logger: logging.Logger = logging.getLogger(__name__)


def get_fp8_constants() -> Tuple[torch.dtype, tl.dtype, float, float]:
    """
    Helper function to get constant values for the current platform.

    Returns:
        pt_dtype (torch.dtype): The correct torch fp8 datatype.
        tl_dtype (tl.dtype): The correct triton fp8 datatype.
        max_fp8 (float): The maximum reprsentable value for the fp8 datatype.
        eps (float): Minimum clip value to prevent divide by zero.
    """
    if torch.version.hip is not None:
        pt_fp8_dtype = torch.float8_e4m3fnuz
        tl_fp8_dtype = tl.float8e4b8
    else:
        pt_fp8_dtype = torch.float8_e4m3fn
        tl_fp8_dtype = tl.float8e4nv
    return pt_fp8_dtype, tl_fp8_dtype, torch.finfo(pt_fp8_dtype).max, 1e-12


def convert_fp8_type(tensor, dtype) -> triton.TensorWrapper:
    """
    Converts tensor to triton fp8 type.

    Args:
        tensor (torch.Tensor): input tensor.
        dtype (tl.dtype): target triton dtype.

    Returns:
        triton.TensorWrapper: fp8 tensor.
    """
    return tl_reinterpret(tensor, dtype=dtype)


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def get_configs_io_bound() -> List[Config]:
    """
    Returns a list of configs for matmul that are IO bound.

    Returns:
        List[Config]: list of configs.
    """
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(
                        Config(
                            {
                                "BLOCK_M": block_m,
                                "BLOCK_N": block_n,
                                "BLOCK_K": block_k,
                                "SPLIT_K": 1,
                            },
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )
                    # split_k
                    for split_k in []:  # Disabled [2, 4, 8, 16]:
                        configs.append(
                            Config(
                                {
                                    "BLOCK_M": block_m,
                                    "BLOCK_N": block_n,
                                    "BLOCK_K": block_k,
                                    "SPLIT_K": split_k,
                                },
                                num_stages=num_stages,
                                num_warps=num_warps,
                                pre_hook=init_to_zero("C"),
                            )
                        )
    return configs


@triton.jit
def _kernel_matmul_fp8_row_tma_persistent(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    A_scale,
    B_scale,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    dot_out_dtype: tl.constexpr,
    allow_tf32: tl.constexpr,
    fp8_fast_accum: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    AB_DTYPE: tl.constexpr,
    NUM_SMS: tl.constexpr,
) -> None:
    """Matmul kernel of [M, K] @ [N, K] with row-wise scales

    performs swizzled matmul in [BLOCK_M, BLOCK_K] with [BLOCK_K, BLOCK_N] tiles.

    Args:
        A (TensorWrapper): [M, K] input tensor.
        B (TensorWrapper): [N, K] input tensor.
        C (TensorWrapper): [M, N] output tensor.
        M (int): M dimension of input tensor.
        N (int): N dimension of input tensor.
        K (int): K dimension of input tensor.
        A_scale (TensorWrapper): [M] reciprocal scale tensor per row. A * A_scale = original A
        B_scale (TensorWrapper): [N] reciprocal scale tensor per row. B * B_scale = original B
        stride_am (int): Stride of M dimension of A.
        stride_ak (int): Stride of K dimension of A.
        stride_bn (int): Stride of N dimension of B.
        stride_bk (int): Stride of K dimension of B.
        stride_cm (int): Stride of M dimension of C.
        stride_cn (int): Stride of N dimension of C.
        dot_out_dtype (torch.dtype): Output type of tensor core.
        allow_tf32 (bool): Whether to use TF32 for tensor core.
        fp8_fast_accum (bool): Whether to use fast accumulation for tensor core.
        BLOCK_M (int): Block size for M dimension.
        BLOCK_N (int): Block size for N dimension.
        BLOCK_K (int): Block size for K dimension.
        GROUP_M (int): Number of groups for M dimension swizzle.
        SPLIT_K (int): Number of SM's to launch per row.
        EVEN_K (bool): Whether K is evenly divisible by BLOCK_K * SPLIT_K.
        AB_DTYPE (bool): Wether to cast A and B to C.dtype before tensor core.
    """
    # Matrix multiplication.
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    num_pid_in_group = GROUP_M * num_pid_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)

    dtype_fp8 = tl.float8e4nv
    scale_dtype = tl.float32

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            offs_am = pid_m * BLOCK_M
            offs_bn = pid_n * BLOCK_N
            offs_am = tl.multiple_of(offs_am, BLOCK_M)
            offs_bn = tl.multiple_of(offs_bn, BLOCK_N)

        offs_k = ki * BLOCK_K

        a = tl._experimental_descriptor_load(
            A_ptr, [offs_am, offs_k], [BLOCK_M, BLOCK_K], dtype_fp8
        )
        b = tl._experimental_descriptor_load(
            B_ptr, [offs_bn, offs_k], [BLOCK_N, BLOCK_K], dtype_fp8
        )
        acc = tl.dot(a, b.T, acc, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)

        if ki == k_tiles - 1:
            # rematerialize rm and rn to save registers
            rm = pid_m * BLOCK_M
            rn = pid_n * BLOCK_N

            # # Invert scaling.
            a_scale = tl._experimental_descriptor_load(
                A_scale, [rm], [BLOCK_M], scale_dtype
            )
            b_scale = tl._experimental_descriptor_load(
                B_scale, [rn], [BLOCK_N], scale_dtype
            )
            # pyre-ignore[16]: Undefined attribute [16]: `float` has no attribute `__getitem__`.
            scale = a_scale[:, None] * b_scale[None, :]
            acc *= scale
            acc = acc.to(C_ptr.dtype.element_ty)

            tl._experimental_descriptor_store(C_ptr, acc, [rm, rn])
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)


def matmul_fp8_row(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    dot_out_dtype: Optional[torch.dtype] = None,
    allow_tf32: bool = True,
    fp8_fast_accum: bool = True,
    imprecise_acc: bool = False,
    tma_persistent: bool = False,
) -> torch.Tensor:
    """
    Performs matmul on [M, K] and [N, K] fp8 matrices with row-wise scalings [M], [N].

    Args:
        a (torch.Tensor): [M, K] input tensor.
        b (torch.Tensor): [N, K] input tensor.
        a_scale (torch.Tensor): [M] reciprocal scale tensor per row. A * a_scale = original A
        b_scale (torch.Tensor): [N] reciprocal scale tensor per row. B * b_scale = original B
        dot_out_dtype (torch.dtype): Output type of tensor core.
        allow_tf32 (bool): Whether to use TF32 for tensor core.
        fp8_fast_accum (bool): Whether to use fast accumulation for tensor core.
        tma_persistent (bool): Whether to use TMA persistent kernel impl.

    Returns:
        torch.Tensor: [M, N] Output tensor a @ b / (a_scale[:, None] * b_scale[None, :])
    """
    # Get datatypes and constants to use.
    _, tl_dtype, _, _ = get_fp8_constants()
    # Reinterpret inputs into proper triton fp8 dtype.
    a_tl = convert_fp8_type(a, tl_dtype)
    b_tl = convert_fp8_type(b, tl_dtype)
    M, N, K, m_key, n_key, k_key, c, dot_out_dtype_triton, device = prep_matmul(
        a_tl, b_tl, dot_out_dtype
    )
    # launch kernel
    if a.device == torch.device("cpu"):
        logger.info(
            "FP8 Row-wise Triton kernel not supported on cpu, fallback to torch"
        )
        return (
            torch.matmul(a.to(torch.bfloat16), b.to(torch.bfloat16).T)
            * (a_scale[:, None] * b_scale[None, :])
        ).to(dtype=c.dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
            META["SPLIT_K"],
        )

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    def persistent_grid(META):
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
            ),
        )

    if tma_persistent:
        # used by TMA persistent kernel
        TMA_SIZE = 128
        import numpy as np

        # autotune doesn't work with TMA
        # https://github.com/triton-lang/triton/blob/main/python/tutorials/09-persistent-matmul.py#L312

        BLOCK_M = 128
        BLOCK_N = 256
        BLOCK_K = 128
        GROUP_M = 8
        num_stages = 3
        num_warps = 8

        desc_a = np.empty(TMA_SIZE, dtype=np.int8)
        desc_b = np.empty(TMA_SIZE, dtype=np.int8)
        desc_c = np.empty(TMA_SIZE, dtype=np.int8)
        desc_a_scale = np.empty(TMA_SIZE, dtype=np.int8)
        desc_b_scale = np.empty(TMA_SIZE, dtype=np.int8)

        triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
            a_tl.data_ptr(),
            M,
            K,
            BLOCK_M,
            BLOCK_K,
            a_tl.element_size(),
            desc_a,
        )
        triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
            b_tl.data_ptr(),
            N,
            K,
            BLOCK_N,
            BLOCK_K,
            b_tl.element_size(),
            desc_b,
        )
        triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
            c.data_ptr(),
            M,
            N,
            BLOCK_M,
            BLOCK_N,
            c.element_size(),
            desc_c,
        )
        triton.runtime.driver.active.utils.fill_1d_tma_descriptor(
            a_scale.data_ptr(),
            M,
            BLOCK_M,
            a_scale.element_size(),
            desc_a_scale,
        )
        triton.runtime.driver.active.utils.fill_1d_tma_descriptor(
            b_scale.data_ptr(),
            N,
            BLOCK_N,
            b_scale.element_size(),
            desc_b_scale,
        )
        desc_a = torch.tensor(desc_a, device="cuda")
        desc_b = torch.tensor(desc_b, device="cuda")
        desc_c = torch.tensor(desc_c, device="cuda")
        desc_a_scale = torch.tensor(desc_a_scale, device="cuda")
        desc_b_scale = torch.tensor(desc_b_scale, device="cuda")

        # pyre-ignore[28]:
        _kernel_matmul_fp8_row_tma_persistent[persistent_grid](
            desc_a,
            desc_b,
            desc_c,
            M,
            N,
            K,
            desc_a_scale,
            desc_b_scale,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            dot_out_dtype=dot_out_dtype_triton,
            allow_tf32=allow_tf32,
            fp8_fast_accum=fp8_fast_accum,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M,
            AB_DTYPE=False,
            NUM_SMS=NUM_SMS,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        return c
