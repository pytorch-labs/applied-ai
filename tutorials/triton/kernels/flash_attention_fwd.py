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
