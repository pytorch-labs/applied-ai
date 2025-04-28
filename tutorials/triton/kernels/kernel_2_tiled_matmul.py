import torch
import triton
import triton.language as tl

@triton.jit
def outerk_matmul_kernel(
    # input pointers
    a_ptr,
    b_ptr,
    # output ptr
    c_ptr,
    # matrix dimensions
    M, N, K,
    # the stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am, stride_ak,
    stride_bk, stride_bn,  # b is transposed, so k is now row dimension
    stride_cm, stride_cn,
    # meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Compute the matrix multiplication C = A @ B

    A is of shape (M, K)
    B is of shape (K, N)  # note that B is transposed to achieve this
    C is of shape (M, N)
    """

    # map program ids to blocks in the matrices
    pid_m = tl.program_id(axis=0)  # row for A and C
    pid_n = tl.program_id(axis=1)  # col for B and C

    # calculate our starting position for this block
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    # create offsets for accessing elements within the block
    offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)

    # masking to handle non-multiple of block size cases
    mask_m = offsets_m < M
    mask_n = offsets_n < N

    # init our accumulator
    # it is the size of our C output block (BLOCK_SIZE_M, BLOCK_SIZE_N)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # now we iterate over the K dimension in blocks of BLOCK_SIZE_K
    for k in range(0, K, BLOCK_SIZE_K):
        # create offsets for the k dimension
        offsets_k = k + tl.arange(0, BLOCK_SIZE_K)

        # mask for K dimension
        mask_k = offsets_k < K

        # compute memory addresses for A and B blocks
        # note that we are using a column vector of M or K dimension offsets
        # and a row vector of K or N dimension offsets to create a 2D grid of all offsets for ptrs
        a_ptrs = a_ptr + (offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn)

        # load A and B blocks using our K mask.  We set other=0.0 to fill remaining spots
        a = tl.load(a_ptrs, mask= mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask = mask_k[:, None] & mask_n[None, :], other=0.0)

        # perform our current k block matmul and add it to the accumulator for C
        acc += tl.dot(a,b)

    # store result back to global memory
    c_ptrs = c_ptr + (offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask= mask_m[:,None] & mask_n[None, :])



# our triton kernel wrapper/interface function
def triton_outer_k_matmul(a, b):
    """
    Compute matmul of C = A @ B using Triton block tiled kernel

    Inputs:
    a: torch.tensor of shape (M, K)
    b: torch.tensor of shape (K, N)

    Returns:
    C: shape (M, N)

    """

    # verify our inputs
    assert a.is_cuda and b.is_cuda, "a and b must be on GPU"
    assert a.shape[1] == b.shape[0], "mismatch between inner dimensions"

    M = a.shape[0]
    K = a.shape[1]
    N = b.shape[1]

    # allocate our output C tensor
    c = torch.empty((M, N), device = a.device, dtype = torch.float32)

    # calculate the strides for our kernel
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = b.stride(0), b.stride(1)
    stride_cm, stride_cn = c.stride(0), c.stride(1)

    # define block sizes we will use to process the matmul
    # Note - we will tune this later with autotune, but for now we can hand tune
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64

    # calculate our grid
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    # launch our kernel
    outerk_matmul_kernel[grid] (
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )

    return c


# Example usage and performance comparison
def benchmark_matmul():
    # Create random matrices
    M, N, K = 8192, 8192, 4096
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)



    # Verify correctness
    torch_output = torch.matmul(a, b)
    triton_output = triton_outer_k_matmul(a, b)
    assert torch.allclose(torch_output, triton_output, rtol=1e-2, atol=1e-1), \
        "Triton and PyTorch matmul results don't match!"



    # Benchmark PyTorch matmul
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.matmul(a, b)  # warmup
    start.record()
    for _ in range(10):
        torch.matmul(a, b)
    end.record()
    torch.cuda.synchronize()
    pytorch_time = start.elapsed_time(end) / 10

    # Benchmark Triton matmul
    triton_outer_k_matmul(a, b)  # warmup
    torch.cuda.synchronize()
    start.record()
    for _ in range(10):
        triton_outer_k_matmul(a, b)
    end.record()
    torch.cuda.synchronize()
    triton_time = start.elapsed_time(end) / 10

    print(f"PyTorch matmul time: {pytorch_time:.2f} ms")
    print(f"Triton matmul time: {triton_time:.2f} ms")
    print(f"Speedup: {pytorch_time / triton_time:.2f}x")


if __name__ == "__main__":
    benchmark_matmul()
