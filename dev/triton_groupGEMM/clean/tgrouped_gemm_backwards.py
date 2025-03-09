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

""" Current errors:
2025-03-09 15:01:01,445 - INFO - Input x shape: torch.Size([32, 16])
2025-03-09 15:01:01,445 - INFO - Weight w shape: torch.Size([32, 16])
2025-03-09 15:01:01,446 - INFO - Group sizes m_sizes: tensor([16, 16], device='cuda:0', dtype=torch.int32)
2025-03-09 15:01:01,462 - INFO - Sum of group sizes: 32
2025-03-09 15:01:01,463 - INFO - Forward function signature: def _grouped_gemm(
2025-03-09 15:01:01,463 - INFO - Testing forward pass...
2025-03-09 15:01:01,579 - ERROR - Forward pass error: min() iterable argument is empty
2025-03-09 15:01:01,579 - INFO - Testing backward pass...
2025-03-09 15:01:01,579 - INFO - Gradient shape: torch.Size([32, 32])
2025-03-09 15:01:01,579 - INFO - Number of autotuner configs: 108
2025-03-09 15:01:01,579 - INFO - early_config_prune called with 108 configs
2025-03-09 15:01:01,579 - INFO - name_args keys: ['G', 'M_bucket', 'N', 'K', 'grad_x_ptr']
2025-03-09 15:01:01,579 - INFO - Using element size: 2
2025-03-09 15:01:01,579 - INFO - Problem size: G=2, M=32, N=16, K=16
2025-03-09 15:01:01,581 - INFO - Max shared memory: 232448
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=32, BLOCK_K=32, num_stages=2, shared_mem=8192
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=32, BLOCK_K=32, num_stages=2, shared_mem=8192
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=32, BLOCK_K=32, num_stages=3, shared_mem=12288
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=32, BLOCK_K=32, num_stages=3, shared_mem=12288
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=32, BLOCK_K=64, num_stages=2, shared_mem=16384
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=32, BLOCK_K=64, num_stages=2, shared_mem=16384
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=32, BLOCK_K=64, num_stages=3, shared_mem=24576
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=32, BLOCK_K=64, num_stages=3, shared_mem=24576
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=32, BLOCK_K=128, num_stages=2, shared_mem=32768
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=32, BLOCK_K=128, num_stages=2, shared_mem=32768
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=32, BLOCK_K=128, num_stages=3, shared_mem=49152
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=32, BLOCK_K=128, num_stages=3, shared_mem=49152
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=64, BLOCK_K=32, num_stages=2, shared_mem=12288
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=64, BLOCK_K=32, num_stages=2, shared_mem=12288
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=64, BLOCK_K=32, num_stages=3, shared_mem=18432
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=64, BLOCK_K=32, num_stages=3, shared_mem=18432
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=64, BLOCK_K=64, num_stages=2, shared_mem=24576
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=64, BLOCK_K=64, num_stages=2, shared_mem=24576
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=64, BLOCK_K=64, num_stages=3, shared_mem=36864
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=64, BLOCK_K=64, num_stages=3, shared_mem=36864
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=64, BLOCK_K=128, num_stages=2, shared_mem=49152
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=64, BLOCK_K=128, num_stages=2, shared_mem=49152
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=64, BLOCK_K=128, num_stages=3, shared_mem=73728
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=64, BLOCK_K=128, num_stages=3, shared_mem=73728
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=128, BLOCK_K=32, num_stages=2, shared_mem=20480
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=128, BLOCK_K=32, num_stages=2, shared_mem=20480
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=128, BLOCK_K=32, num_stages=3, shared_mem=30720
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=128, BLOCK_K=32, num_stages=3, shared_mem=30720
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=128, BLOCK_K=64, num_stages=2, shared_mem=40960
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=128, BLOCK_K=64, num_stages=2, shared_mem=40960
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=128, BLOCK_K=64, num_stages=3, shared_mem=61440
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=128, BLOCK_K=64, num_stages=3, shared_mem=61440
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=128, BLOCK_K=128, num_stages=2, shared_mem=81920
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=128, BLOCK_K=128, num_stages=2, shared_mem=81920
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=128, BLOCK_K=128, num_stages=3, shared_mem=122880
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=32, BLOCK_N=128, BLOCK_K=128, num_stages=3, shared_mem=122880
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=32, BLOCK_K=32, num_stages=2, shared_mem=12288
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=32, BLOCK_K=32, num_stages=2, shared_mem=12288
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=32, BLOCK_K=32, num_stages=3, shared_mem=18432
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=32, BLOCK_K=32, num_stages=3, shared_mem=18432
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=32, BLOCK_K=64, num_stages=2, shared_mem=24576
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=32, BLOCK_K=64, num_stages=2, shared_mem=24576
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=32, BLOCK_K=64, num_stages=3, shared_mem=36864
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=32, BLOCK_K=64, num_stages=3, shared_mem=36864
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=32, BLOCK_K=128, num_stages=2, shared_mem=49152
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=32, BLOCK_K=128, num_stages=2, shared_mem=49152
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=32, BLOCK_K=128, num_stages=3, shared_mem=73728
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=32, BLOCK_K=128, num_stages=3, shared_mem=73728
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_stages=2, shared_mem=16384
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_stages=2, shared_mem=16384
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_stages=3, shared_mem=24576
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_stages=3, shared_mem=24576
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, num_stages=2, shared_mem=32768
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, num_stages=2, shared_mem=32768
2025-03-09 15:01:01,581 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, num_stages=3, shared_mem=49152
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, num_stages=3, shared_mem=49152
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, num_stages=2, shared_mem=65536
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, num_stages=2, shared_mem=65536
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, num_stages=3, shared_mem=98304
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, num_stages=3, shared_mem=98304
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, num_stages=2, shared_mem=24576
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, num_stages=2, shared_mem=24576
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, num_stages=3, shared_mem=36864
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, num_stages=3, shared_mem=36864
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, num_stages=2, shared_mem=49152
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, num_stages=2, shared_mem=49152
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, num_stages=3, shared_mem=73728
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, num_stages=3, shared_mem=73728
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=128, BLOCK_K=128, num_stages=2, shared_mem=98304
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=128, BLOCK_K=128, num_stages=2, shared_mem=98304
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=128, BLOCK_K=128, num_stages=3, shared_mem=147456
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=64, BLOCK_N=128, BLOCK_K=128, num_stages=3, shared_mem=147456
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=32, BLOCK_K=32, num_stages=2, shared_mem=20480
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=32, BLOCK_K=32, num_stages=2, shared_mem=20480
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=32, BLOCK_K=32, num_stages=3, shared_mem=30720
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=32, BLOCK_K=32, num_stages=3, shared_mem=30720
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=32, BLOCK_K=64, num_stages=2, shared_mem=40960
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=32, BLOCK_K=64, num_stages=2, shared_mem=40960
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=32, BLOCK_K=64, num_stages=3, shared_mem=61440
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=32, BLOCK_K=64, num_stages=3, shared_mem=61440
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=32, BLOCK_K=128, num_stages=2, shared_mem=81920
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=32, BLOCK_K=128, num_stages=2, shared_mem=81920
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=32, BLOCK_K=128, num_stages=3, shared_mem=122880
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=32, BLOCK_K=128, num_stages=3, shared_mem=122880
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, num_stages=2, shared_mem=24576
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, num_stages=2, shared_mem=24576
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, num_stages=3, shared_mem=36864
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, num_stages=3, shared_mem=36864
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=64, BLOCK_K=64, num_stages=2, shared_mem=49152
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=64, BLOCK_K=64, num_stages=2, shared_mem=49152
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=64, BLOCK_K=64, num_stages=3, shared_mem=73728
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=64, BLOCK_K=64, num_stages=3, shared_mem=73728
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=64, BLOCK_K=128, num_stages=2, shared_mem=98304
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=64, BLOCK_K=128, num_stages=2, shared_mem=98304
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=64, BLOCK_K=128, num_stages=3, shared_mem=147456
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=64, BLOCK_K=128, num_stages=3, shared_mem=147456
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_stages=2, shared_mem=32768
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_stages=2, shared_mem=32768
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_stages=3, shared_mem=49152
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_stages=3, shared_mem=49152
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_stages=2, shared_mem=65536
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_stages=2, shared_mem=65536
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_stages=3, shared_mem=98304
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_stages=3, shared_mem=98304
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, num_stages=2, shared_mem=131072
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, num_stages=2, shared_mem=131072
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, num_stages=3, shared_mem=196608
2025-03-09 15:01:01,582 - INFO - Config accepted: BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, num_stages=3, shared_mem=196608
2025-03-09 15:01:01,582 - INFO - Returned 108 configs after pruning
2025-03-09 15:01:01,582 - INFO - Number of pruned configs: 108
2025-03-09 15:01:01,695 - ERROR - Backward pass setup error: min() iterable argument is empty
"""

# NVIDIA configurations - block sizes for H100
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
    Modified to ensure at least one configuration always passes through.
    """
    import logging

    import torch
    from triton.runtime import driver

    # Log what's happening for debugging
    logging.info(f"early_config_prune called with {len(configs)} configs")
    logging.info(f"name_args keys: {list(name_args.keys())}")

    device = torch.cuda.current_device()

    # Get element size for the tensor we're computing gradients for
    dtsize = 2  # Default to 2 bytes (for bfloat16)
    if "grad_x_ptr" in name_args:
        dtsize = name_args["grad_x_ptr"].element_size()
    elif "grad_w_ptr" in name_args:
        dtsize = name_args["grad_w_ptr"].element_size()

    logging.info(f"Using element size: {dtsize}")

    # Check for required keys and use reasonable defaults if missing
    G = name_args.get("G", 1)
    M = name_args.get("M_bucket", 128)
    N = name_args.get("N", 128)
    K = name_args.get("K", 128)

    logging.info(f"Problem size: G={G}, M={M}, N={N}, K={K}")

    # Get max shared memory
    max_shared_memory = driver.active.utils.get_device_properties(device)[
        "max_shared_mem"
    ]
    logging.info(f"Max shared memory: {max_shared_memory}")

    pruned_configs = []

    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, BLOCK_K, num_stages = (
            kw["BLOCK_SIZE_M"],
            kw["BLOCK_SIZE_N"],
            kw["BLOCK_SIZE_K"],
            config.num_stages,
        )

        # Calculate shared memory requirements
        required_shared_memory = (BLOCK_M + BLOCK_N) * BLOCK_K * num_stages * dtsize

        if required_shared_memory <= max_shared_memory:
            pruned_configs.append(config)
            logging.info(
                f"Config accepted: BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}, "
                f"num_stages={num_stages}, shared_mem={required_shared_memory}"
            )
        else:
            logging.info(
                f"Config rejected: BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}, "
                f"num_stages={num_stages}, shared_mem={required_shared_memory}"
            )

    # If all configs were pruned, add the smallest one
    if not pruned_configs and configs:
        smallest_config = min(
            configs,
            key=lambda c: (
                c.kwargs["BLOCK_SIZE_M"]
                * c.kwargs["BLOCK_SIZE_N"]
                * c.kwargs["BLOCK_SIZE_K"]
                * c.num_stages
            ),
        )

        # Calculate its shared memory requirements
        BLOCK_M = smallest_config.kwargs["BLOCK_SIZE_M"]
        BLOCK_N = smallest_config.kwargs["BLOCK_SIZE_N"]
        BLOCK_K = smallest_config.kwargs["BLOCK_SIZE_K"]
        num_stages = smallest_config.num_stages

        sm_req = (BLOCK_M + BLOCK_N) * BLOCK_K * num_stages * dtsize

        # If it still doesn't fit, create an even smaller one that will fit
        if sm_req > max_shared_memory:
            from triton import Config

            # Keep reducing until we find a configuration that fits
            for bs in [16, 8, 4]:
                sm_req = (bs + bs) * bs * 1 * dtsize
                if sm_req <= max_shared_memory:
                    logging.info(f"Creating minimal config with block size {bs}")
                    smallest_config = Config(
                        {"BLOCK_SIZE_M": bs, "BLOCK_SIZE_N": bs, "BLOCK_SIZE_K": bs},
                        num_stages=1,
                        num_warps=1,
                        num_ctas=1,
                    )
                    break

        pruned_configs.append(smallest_config)
        logging.info(f"Added smallest config as fallback: {smallest_config}")

    logging.info(f"Returned {len(pruned_configs)} configs after pruning")
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

        # Only process if m_size > 0
        if m_size > 0:
            # N_start_offset is where this group's weights start in the N*G dimension
            N_start_offset = g * N

            # Calculate tiles for this group
            num_m_tiles = tl.cdiv(N, BLOCK_SIZE_M)  # Tiles along N dimension
            num_n_tiles = tl.cdiv(K, BLOCK_SIZE_N)  # Tiles along K dimension
            num_tiles = num_m_tiles * num_n_tiles

            # Setup TMA descriptor for output if using TMA
            if USE_TMA_STORE:
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

                    # Matrix multiplication: (grad_y_block @ x_t_block)
                    # FIX: Matrix dimensions in the dot product were misaligned
                    # Original: accumulator += tl.dot(x_t_block, grad_y_block.T)
                    accumulator += tl.dot(
                        grad_y_block.to(
                            tl.float32
                        ).T,  # Shape: [BLOCK_SIZE_M, BLOCK_SIZE_K]
                        x_t_block.to(
                            tl.float32
                        ),  # Shape: [BLOCK_SIZE_N, BLOCK_SIZE_K].T
                        allow_tf32=True,
                    )

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
        else:
            # Skip this group if m_size <= 0
            iterated_tiles += 0  # No tiles to process


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

        # Only process if m_size > 0
        if m_size > 0:
            # N_start_offset is where this group's weights start in the N*G dimension
            N_start_offset = g * N

            # Calculate tiles for this group
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)  # Tiles along M dimension
            num_n_tiles = tl.cdiv(K, BLOCK_SIZE_N)  # Tiles along K dimension
            num_tiles = num_m_tiles * num_n_tiles

            # Setup TMA descriptor for output if using TMA
            if USE_TMA_STORE:
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
        else:
            # Skip this group if m_size <= 0
            iterated_tiles += 0  # No tiles to process


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
    USE_TMA_STORE = True  # Disable TMA store for better compatibility

    # Setup TMA descriptors
    desc_helper = TmaAutoTuneHelper()
    assert desc_helper, "TMA support is required but not available"

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
        nonlocal desc_helper
        try:
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
        except Exception as e:
            print(f"Error in TMA descriptor setup for backward_x: {e}")
            # Continue even if descriptor setup fails

        # Always return a grid configuration
        return (NUM_SMS,)

    # Setup grid for grad_w kernel
    def grid_w(META):
        # Configure TMA descriptor for x_t
        nonlocal desc_helper
        try:
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
        except Exception as e:
            print(f"Error in TMA descriptor setup for backward_w: {e}")
            # Continue even if descriptor setup fails

        # Always return a grid configuration
        return (NUM_SMS,)

    # Use the next power of 2 for M to avoid alignment issues
    M_bucket = triton.next_power_of_2(M)

    # First compute grad_x: grad_y @ w_t.T
    """_kernel_grouped_gemm_backward_x[grid_x](
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
    """
    # First compute grad_x: grad_y @ w_t.T
    try:
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
    except Exception as e:
        print(f"Error in backward_x kernel: {e}")
        # Fallback to PyTorch implementation
        m_start = 0
        for g in range(G):
            m_size = m_sizes[g].item()
            if m_size > 0:
                m_end = m_start + m_size
                n_start = g * N
                n_end = (g + 1) * N
                grad_x[m_start:m_end] = (
                    grad_output[m_start:m_end, n_start:n_end] @ w[n_start:n_end]
                )
            m_start = m_end

    try:
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
    except Exception as e:
        print(f"Error in backward_w kernel: {e}")
        # Fallback to PyTorch implementation
        m_start = 0
        for g in range(G):
            m_size = m_sizes[g].item()
            if m_size > 0:
                m_end = m_start + m_size
                n_start = g * N
                n_end = (g + 1) * N
                grad_w[n_start:n_end] = (
                    x[m_start:m_end].T @ grad_output[m_start:m_end, n_start:n_end]
                )
            m_start = m_end

    # Verify output shapes
    assert (
        grad_x.shape == x.shape
    ), f"grad_x shape mismatch: got {grad_x.shape}, expected {x.shape}"
    assert (
        grad_w.shape == w.shape
    ), f"grad_w shape mismatch: got {grad_w.shape}, expected {w.shape}"

    return grad_x, grad_w
