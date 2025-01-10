# flash forward v2

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import triton
import triton.language as tl

BLOCK_SIZES = {
    "q": [64, 128],
    "kv": [32, 64],
}

NUM_STAGES = [3, 4, 7]
NUM_WARPS = [2, 4]


def check_device() -> bool:
    """Check if CUDA is available, and if the current device is a SM90+ device with FP8 support"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
        return False  # lol, I don't think this is needed

    is_SM90 = torch.cuda.get_device_capability("cuda") >= (9, 0)
    if not is_SM90:
        print("Warning: FlashAttention with FP8 is only supported on SM90+ devices")
    return True
