"""Fused MoE W4A16 Kernel."""

import torch
import triton
import triton.language as tl
from vllm._C import ops

@triton.jit
def print_tensor_dim(tensor, str_name):
    if tl.program_id(0) == 0 and tl.program_id(1) == 0:
        tl.static_print(str_name," ",tensor.shape," ",tensor.dtype)
@triton.jit
def print_value(value):
    if tl.program_id(0) == 0 and tl.program_id(1) == 0:
        tl.device_print(str(value))

@triton.jit()
def grouped_launch(pid,
                m, n,
                block_m: tl.constexpr, block_n: tl.constexpr, group_m: tl.constexpr):
    
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)

    width = group_m * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * group_m, group_m)

    pid_m = group_id * group_m + (pid % group_size)
    pid_n = (pid % width) // group_size

    return pid_m, pid_n


@triton.jit()
def col_major(pid,
              m, n, num_tokens_post_padded,
              block_m: tl.constexpr, block_n: tl.constexpr):
    
    grid_m = tl.cdiv(m, block_m)    
    grid_n = tl.cdiv(n, block_n)
    
    pid_m = (pid % grid_n) 
    pid_n = pid // grid_m

    return pid_m, pid_n


@triton.jit()
def w4a16_fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Quantization Scales and Zeros Ptr
    scales_ptr,
    zeros_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Quantization Scales and Zeros Strides
    stride_scales_e, stride_scales_g, stride_scales_n,
    stride_zeros_e, stride_zeros_g, stride_zeros_n,
    # Meta-parameters
    groupsize: tl.constexpr,
    top_k: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
):

    pid = tl.program_id(0)

    # GEMM Schedule
    pid_m, pid_n = grouped_launch(pid,
                                  EM, N,
                                  block_m, block_n, group_m)
    grid_k = tl.cdiv(K, block_k)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * block_m >= num_tokens_post_padded:
        return
    
    # Offset Calculations
    offs_token_id = pid_m*block_m + tl.arange(0, block_m)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_bn = (pid_n * block_n + tl.arange(0, block_n)) % N # NOTE: No change needed here since weights are packed along K dim
    offs_k = tl.arange(0, block_k)
    off_experts = tl.load(expert_ids_ptr + pid_m)

    # Mask for Activations
    token_mask = offs_token < num_valid_tokens 

    # Pointer Calculations
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak) #NOTE: offs_token[:, None] // top_k -> since each row of activations repeats top_k times
    b_ptrs = b_ptr + off_experts * stride_be + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn) #NOTE: offs_k[:, None] // 8 -> since B is packed along k dim is packed 

    # We need to handle the e dim of the scales and zeros pointers
    # We can do this in the same fashion that the stacked expert weight matrix is handled
    
    # off_experts = tl.load(expert_ids_ptr + pid_m)
    # b_ptr + off_experts * stride_be + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)

    scales_ptrs = scales_ptr + off_experts * stride_scales_e + offs_bn * stride_scales_n
    zeros_ptrs = zeros_ptr + off_experts * stride_zeros_e + ((offs_bn//8) * stride_zeros_n)

    shifter = (offs_k % 8) * 4
    zeros_shifter = (offs_bn % 8) * 4

    acc = tl.zeros([block_m, block_n], dtype=tl.float32)
    for k in range(0, grid_k):

        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * block_k), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * block_k, other=0.0)
        
        g_id = k // (groupsize // block_k)
        ptr = scales_ptrs + g_id * stride_scales_g
        
        scales = tl.load(ptr)
        ptr = zeros_ptrs + g_id * stride_zeros_g
        zeros = tl.load(ptr) 
        zeros = (zeros >> zeros_shifter) & 0xF
        zeros = (zeros + 1) * scales

        b = (b >> shifter[:, None]) & 0xF
        b = b * scales[None, :] - zeros[None, :]

        acc += tl.dot(a, b)

        a_ptrs += block_k * stride_ak
        b_ptrs += (block_k // 8) * stride_bk
    
    acc.to(tl.float16)

    offs_m = pid_m*block_m + tl.arange(0, block_m)
    offs_n = pid_n*block_n + tl.arange(0, block_n)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc)



def invoke_dequant_gemm_moe(activations: torch.Tensor, 
                            qweight: torch.Tensor, 
                            c: torch.Tensor,
                            scales: torch.Tensor, 
                            qzeros: torch.Tensor,
                            topk_ids: torch.Tensor, 
                            sorted_token_ids: torch.Tensor,
                            expert_ids: torch.Tensor,
                            num_tokens_post_padded: torch.Tensor,
                            topk: torch.Tensor,
                            ):
    
    EM = sorted_token_ids.shape[0]
    N = qweight.shape[1]
    K = qweight.shape[2]
    block_m = 32
    block_n = 32
    block_k = 32
    group_m = 8
    groupsize = 128
    topk = 2

    if topk_ids.numel() <= qweight.shape[0]:
            block_m = 16
            block_n = 128
            block_k = 128
            group_m = 8

    total_blocks_m = triton.cdiv(EM, block_m)
    total_blocks_n = triton.cdiv(N, block_n)

    grid = (total_blocks_m * total_blocks_n,)
    w4a16_fused_moe_kernel[grid](
        activations,
        qweight,
        c,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        scales,
        qzeros,
        N,
        K,
        EM,
        topk_ids.numel(),
        activations.stride(0), activations.stride(1),
        qweight.stride(0), qweight.stride(2), qweight.stride(1),
        c.stride(1), c.stride(2),
        scales.stride(0), scales.stride(1), scales.stride(2),
        qzeros.stride(0), qzeros.stride(1), qzeros.stride(2),
        groupsize=groupsize,
        top_k=topk,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        group_m=group_m,
    )

def moe_align_block_size(
        topk_ids: torch.Tensor, block_size: int,
        num_experts: int) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Aligns the token distribution across experts to be compatible with block size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding, ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process so that it is divisible by block_size. 
    Padding ensures that during block matrix multiplication, the dimensions align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]], block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts, with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12]. 
        Tokens 12 are non-existent (padding) and are ignored in the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible by block_size for proper block matrix operations.
    """
    sorted_ids = torch.empty(
        (topk_ids.numel() + num_experts * (block_size - 1), ),
        dtype=torch.int32,
        device=topk_ids.device)
    expert_ids = torch.empty((topk_ids.numel() + num_experts, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    num_tokens_post_pad = torch.empty((1),
                                      dtype=torch.int32,
                                      device=topk_ids.device)
    ops.moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids,
                             expert_ids, num_tokens_post_pad)
    return sorted_ids, expert_ids, num_tokens_post_pad

def dequant_gemm_moe(hidden_states: torch.Tensor,
                    qw1: torch.Tensor,
                    qw2: torch.Tensor,
                    scales_qw1: torch.Tensor,
                    scales_qw2: torch.Tensor,
                    zeros_qw1: torch.Tensor,
                    zeros_qw2: torch.Tensor,
                    topk_ids: torch.Tensor,
                    ):
    
    # Check constraints.
    # assert hidden_states.shape[1] == qw1.shape[2], "Incompatible dimensions"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert qw1.is_contiguous(), "Expert weights1 must be contiguous"
    assert qw2.is_contiguous(), "Expert weights2 must be contiguous"
    # assert hidden_states.dtype in [torch.float16, torch.bfloat16]
    M, _ = hidden_states.shape
    E, N, _ = qw1.shape

    block_m = 32
    if topk_ids.numel() <= qw1.shape[0]:
        block_m = 16

    intermediate_cache1 = torch.empty((M, topk_ids.shape[1], N),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    intermediate_cache2 = torch.empty((M * topk_ids.shape[1], N // 2),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    intermediate_cache3 = torch.empty((M, topk_ids.shape[1], qw2.shape[1]),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_m, E)

    invoke_dequant_gemm_moe(hidden_states, 
                            qw1, 
                            intermediate_cache1,
                            scales_qw1, 
                            zeros_qw1,
                            topk_ids, 
                            sorted_token_ids,
                            expert_ids, 
                            num_tokens_post_padded,
                            topk_ids.shape[1],)
    
    # return torch.sum(intermediate_cache1.view(*intermediate_cache1.shape), dim=1)

    ops.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))

    invoke_dequant_gemm_moe(intermediate_cache2, 
                            qw2, 
                            intermediate_cache3,
                            scales_qw2,
                            zeros_qw2,
                            topk_ids, 
                            sorted_token_ids,
                            expert_ids, 
                            num_tokens_post_padded, 
                            1,)
    
    return torch.sum(intermediate_cache3.view(*intermediate_cache3.shape),
                    dim=1)
    
