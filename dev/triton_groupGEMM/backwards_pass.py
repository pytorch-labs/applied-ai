# Copyright (c) Meta Platforms, Inc. and affiliates.


import torch
import triton
import triton.language as tl


# backward for dx
@triton.jit
def _kernel_grouped_gemm_backward_x(
    grad_y_ptr, # grad output (dl/dy)
    w_t_ptr,
    grad_x_ptr, # output
    workspace,
    m_sizes,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
)-> None:
    """
    compute gradients wrt x (input)
    grad_x = grad_y @ w_t
    grad_y = [M,N]
    w_t = [N*G,K] transposed to [K,N*G]
    grad_x = [M,K]

    """
    tidx = tl.program_id(0)
    dtype = grad_x_ptr.dtype.element_ty
    TMA_SIZE: tl.constexpr = tl.constexpr(128)
    c_desc_ptr = workspace + tidx * TMA_SIZE

    M_end_offset = 0
    iterated_tiles = 0
    for g in tl.range(G):
        # move across groups
        M_start_offset = M_end_offset
        m_size = tl.load(m_sizes + g)
        M_end_offset = M_start_offset + m_size # advance

        if m_size <=0:
            continue

        N_start_offset = g.to(tl.int64) *N
        n_size = N
        num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(K, BLOCK_SIZE_N) # K is output dim in backward!
        num_tiles = num_m_tiles * num_n_tiles

        tl.extra.cuda.experimental_device_tensormap_create_2d(
            desc_ptr = c_desc_ptr,
            global_address = grad_x_ptr + M_start_offset * K,
            load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
            global_size = [m_size, K],
            element_ty = grad_x_ptr.dtype.element_ty,
        )
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

        # move across tiles
        while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
            gidx = tidx - iterated_tiles
            # Split M first and N second
            tile_m_idx = gidx % num_m_tiles
            tile_n_idx = gidx // num_m_tiles

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype =tl.float32)

            # for each K block in the innner dimensions (N in forward)
            for k_offset in range(0, N, BLOCK_SIZE_K):
                k_size = tl.minimum(BLOCK_SIZE_K, N-k_offset)

                # load grad_y block [M,K]
                offs_m = tile_m_idx* BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_k = tl.arange(0, BLOCK_SIZE_K)

                m_mask = offs_m < m_size
                k_mask = offs_k < k_size

                # load grad_y [M,N]
                grad_y_block = tl.load(
                    grad_y_ptr + (M_start_offset + offs_m[:,None] *N + (N_start_offset+k_offset + offs_k[None,:]),
                                  mask=mask[:,None] & k_mask[None,:]),
                                  other=0.0
                )

                # load w_t [K,N]
                offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                w_t_block = tl.load(
                    w_t_ptr + (N_start_offset + k_offset + offs_k[:,None]) *K + offs_n[None,:],
                    mask = k_mask[:,None] & (offs_n[None,:] < K),
                    other = 0.0
                )

                # compute grad contribution from this block
                accumulator += tl.dot(grad_y_block, w_t_block)

            # store result in grad_x
            offs_m = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

            m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
            n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
            tl.experimental_descriptor_store(
                c_desc_ptr,
                accumulator.to(grad_x_ptr.dtype.element_ty),
                [m_offset, n_offset]
            )

            tidx += NUM_SMS

        iterated_tiles += num_tiles

# backward for dx
@triton.jit
def _kernel_grouped_gemm_backward_w(
    x_t_ptr, #x transposed
    grad_y_ptr, # grad output (dl/dy)
    grad_w_ptr, # output for this kernel
    workspace,
    m_sizes,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
)-> None:
    """
    compute gradients with respect to w (weights)
    grad_w = x_t @ grad_y

    x_t is [M,K] but transposed to [K,M]
    grad_y = [M,N]
    grad_w = [N*G,K]
    """
    tidx = tl.program_id(0)
    dtype = grad_w_ptr.dtype.element_ty
    TMA_SIZE: tl.constexpr = tl.constexpr(128)

    c_desc_ptr = workspace + tidx * TMA_SIZE

    iterated_tiles = 0
    for g in tl.range(G):
        # for each group
        M_offsets = [0]
        for i in range(g):
            M_offsets.append(M_offsets[-1] + tl.load(m_sizes+i))

        M_start_offset = M_offsets[g]
        m_size = tl.load(m_sizes + g)
        N_start_offset = g.to(tl.int64) *N

        if m_size <=0:
            continue

        # for gradients, we're computing N rows of K columns each group
        num_m_tiles = tl.cdiv(N, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(K, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles

        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr = c_desc_ptr,
            global_address = grad_w_ptr + N_start_offset * K,
            load_size = [BLOCK_SIZE_M, BLOCK_SIZE_N],
            global_size = [N,K],
            element_ty = grad_w_ptr.dtype.element_ty,
        )
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

        # move across tiles
        while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
            gidx = tidx - iterated_tiles
            # Split M first, N Second for the output grad_w
            tile_m_idx = gidx % num_m_tiles
            tile_n_idx = gidx // num_m_tiles

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            # for each block M in grad_y
            for k_offset in range(0, m_size, BLOCK_SIZE_K):
                k_size = tl.minimum(BLOCK_SIZE_K, m_size - k_offset)
                # load x_t
                offs_k = tl.arange(0, BLOCK_SIZE_K)
                offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

                k_mask = offs_k < k_size
                n_mask = offs_n < K

                x_t_block = tl.load(
                    x_t_ptr + offs_n[None, :] * M + (M_start_offset + k_offset + offs_k[:, None]),
                    mask = k_mask[:,None] & n_mask[None,:],
                    other = 0.0
                )

                # load grad_y [M,N]
                offs_m = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                m_mask = offs_m < N

                grad_y_block = tl.load(
                    grad_y_ptr + (m_mask + k_offset + offs_k[:,None]) * N + (N_start_offset + offs_m[None, :]),
                    mask = k_mask[:,None] & m_mask[None,:],
                    other = 0.0
                )

                # add to grad_w fromthis block
                accumulator += tl.dot(x_t_block.T, grad_y_block.T)

            # store result
            offs_m = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

            # using TMA
            m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
            n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
            tl._experimental_descriptor_store(
                c_desc_ptr,
                accumulator.to(grad_w_ptr.dtype.element_ty),
                [m_offset, n_offset]
            )
        tidx += NUM_SMS
    iterated_tiles += num_tiles



# wrapper for bwd
def _grouped_gemm_backward (
        grad_output: torch.Tensor,
        x: torch.Tensor,
        w: torch.Tensor,
        m_sizes: torch.Tensor,
        x_scale: Optionsal[torch.Tensor]=None,
        w_scale: Optional[torch.Tensor]=None,
)-> Tuple[torch.Tensor, torch.Tensor]:

    """ bwd pass for grouped matmul"""
    if not utils.HAS_TMA_DESC:
        raise NotImplementedError("Grouped GEMM bwd requires TMA")

    G = m_sizes.shape[0]

    #verify
    grad_output = grad_output.contiguous()
    x = x.contiguous()
    w = w.contiguous()
    m_sizes = m_sizes.contiguous()

    M,K = x.shape
    N = w.shape[0] // G
    assert K = w.shape[1]  # confirm inner match

    # create output tensors for grads
    grad_x = torch.empty_like(x)
    grad_w = torch.empty_like(w)

    # prepare transposed matrices
    x_t = x.t().contigouous()
    w_t = w.t().contiguous()

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    USE_TMA_LOAD = True
    USE_TMA_STORE = True
    workspace = None

    # Setup TMA descriptors
    desc_helper = utils.TmaAutoTuneHelper()
    desc_helper.init_tma_descriptor('grad_output')
    desc_helper.init_tma_descriptor('w_t')
    desc_helper.init_tma_descriptor('x_t')

    grad_y_ptr = desc_helper.get_tma_descriptor_kernel_param("grad_output")
    w_t_ptr = desc_helper.get_tma_descriptor_kernel_param("w_t")
    x_t_ptr = desc_helper.get_tma_descriptor_kernel_param("grad_output")

    if USE_TMA_STORE:
        workspace = torch.empty(NUM_SMS * utils.TmaAutoTuneHelper.TMA_SIZE,
                                device = x.device,
                                dtype = torch.uint8)

    def grid_x(META):
        if USE_TMA_LOAD:
            desc_helper.fill_2d_tma_descriptor(
                "grad_output",
                grad_output.data_prt(),
                M,
                N*G,
                META["BLOCK_SIZE_M"],
                META["BLOCK_SIZE_K"],
                grad_output.element_size(),
            )
            desc_helper.fill_2d_tma_descriptor(
                "w_t",
                w_t.data_ptr(),
                K,
                N*G,
                META["BLOCK_SIZE_N"],
                META["BLOCK_SIZE_K"],
                w_t.element_size(),
            )
        return (NUM_SMS,)

    def grid_w(META):
        if USE_TMA_LOAD:
            # grad_w computation
            desc_helper.fill_2d_tma_descriptor(
                "x_t",
                x_t.data_prt(),
                K,
                M,
                META["BLOCK_SIZE_N"],
                META["BLOCK_SIZE_K"],
                x_t.element_size(),
            )
            desc_helper.fill_2d_tma_descriptor(
                "grad_output",
                grad_output.data_prt(),
                M,
                N*G,
                META["BLOCK_SIZE_K"],
                META["BLOCK_SIZE_M"],
                grad_output.element_size(),
            )
        return (NUM_SMS,)

    M_BUCKET = triton.next_power_of_2(M)

    # Compute grad_x
    _kernel_grouped_gemm_backward_x[grid_x](
            grad_y_ptr,
            w_t_ptr,
            grad_x,
            workspace,
            m_sizes,
            G,
            M_BUCKET,
            N,
            K,
            NUM_SMS,
            USE_TMA_LOAD,
            USE_TMA_STORE,
        )

    # Compute grad_w
    _kernel_grouped_gemm_backward_w[grid_w](
            x_t_ptr,
            grad_y_ptr,
            grad_w,
            workspace,
            m_sizes,
            G,
            M_BUCKET,
            N,
            K,
            NUM_SMS,
            USE_TMA_LOAD,
            USE_TMA_STORE,
        )
    return grad_x, grad_w








def grouped_gemm_backward(
        grad_output: torch.Tensor,
        x: torch.Tensor,
        w: torch.Tensor
        m_sizes: torch.Tensor

)-> Tuple[torch.Tensor, torch.Tensor]:

""" grad_output = gradient wrt output, shape [M, N*G]
x: Input tensor, shape [M, K]
w: weight tensor, shape [N*G, K]
m_sizes: Group sizes tensor

returns:
tuple of gradients with respect to X and W
"""

return _grouped_gemm_backward(grad_output, x, w, m_sizes)
