import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    is_causal_block: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    # Range determination based on causality
    if is_causal_block:
        # The block containing the diagonal - need causal masking
        lo = block_index_q * BLOCK_SIZE_Q
        hi = (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        # Blocks to the left of the diagonal - no masking needed
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # Process blocks
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)

        if is_causal_block:
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block * softmax_scale - m_ij[:, None]

        P_block = tl.math.exp(QK_block)
        l_ij = tl.sum(P_block, 1)

        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        V_block = tl.load(V_block_ptr)
        P_block = P_block.to(tl.float16)
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)

        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))

    return O_block, l_i, m_i


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [64, 128]
        for BLOCK_SIZE_KV in [32, 64]
        for num_stages in [3, 4, 7]
        for num_warps in [2, 4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    softmax_scale,
    M,
    O,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    # Program ID for block in sequence
    block_index_q = tl.program_id(0)
    index_batch_head = tl.program_id(1)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    qkv_offset = (
        index_batch.to(tl.int64) * stride_batch + index_head.to(tl.int64) * stride_head
    )

    # Make pointers for Q, K, V, O blocks
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_seq, stride_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_seq, stride_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(stride_dim, stride_seq),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O + qkv_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_seq, stride_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    # Initialize accumulator
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # Load Q block
    Q_block = tl.load(Q_block_ptr)

    # Process non-causal part (blocks to the left of diagonal)
    O_block, l_i, m_i = _attn_fwd_inner(
        O_block,
        l_i,
        m_i,
        Q_block,
        K_block_ptr,
        V_block_ptr,
        block_index_q,
        softmax_scale,
        BLOCK_SIZE_Q,
        BLOCK_SIZE_KV,
        False,  # non-causal block
        offs_q,
        offs_kv,
        SEQ_LEN,
    )

    # Process causal part (diagonal block)
    O_block, l_i, m_i = _attn_fwd_inner(
        O_block,
        l_i,
        m_i,
        Q_block,
        K_block_ptr,
        V_block_ptr,
        block_index_q,
        softmax_scale,
        BLOCK_SIZE_Q,
        BLOCK_SIZE_KV,
        True,  # causal block
        offs_q,
        offs_kv,
        SEQ_LEN,
    )

    # Finalize
    m_i += tl.math.log(l_i)
    O_block = O_block / l_i[:, None]

    # Store results
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))


@triton.jit
def _attn_bwd_dq(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    # Calculate base offset for all tensors
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(
        tl.int64
    )
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    # Apply offsets to all tensors
    base_offset = offset_batch_head
    Q += base_offset
    K += base_offset
    V += base_offset
    dO += base_offset
    dQ += base_offset
    dK += base_offset
    dV += base_offset

    M += offset_batch_head_seq
    D += offset_batch_head_seq

    # Setup dimensions
    offs_dim = tl.arange(0, HEAD_DIM)
    index_block_kv = tl.program_id(0)
    start_q = index_block_kv * BLOCK_Q
    offs_q = start_q + tl.arange(0, BLOCK_Q)

    # Load Q block and initialize dQ
    Q_block = tl.load(Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)
    dQ_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    dO_block = tl.load(
        dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )

    # Load M block
    M_block = tl.load(M + offs_q)
    M_block = M_block[:, None]

    offs_kv = tl.arange(0, BLOCK_KV)
    kT_ptrs = K + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    vT_ptrs = V + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim

    Di = tl.load(D + offs_q)

    # Process blocks
    curr_kv = 0
    num_steps = SEQ_LEN // BLOCK_KV
    for blk_idx in range(num_steps):
        # Load K and V blocks
        K_T_block = tl.load(kT_ptrs)
        V_T_block = tl.load(vT_ptrs)

        # Compute attention scores
        QK_block = softmax_scale * tl.dot(Q_block, K_T_block)
        P_block = tl.math.exp(QK_block - M_block)

        # Apply causal mask
        offs_kv = curr_kv + tl.arange(0, BLOCK_KV)
        mask_block = offs_q[:, None] >= offs_kv[None, :]
        P_block = tl.where(mask_block, P_block, 0.0)

        # Compute gradients
        dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
        dS_block = P_block * (dP_block - Di[:, None])
        dS_block = dS_block.to(tl.float16)
        dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block))

        # Move to next block
        curr_kv += BLOCK_KV
        kT_ptrs += BLOCK_KV * stride_seq
        vT_ptrs += BLOCK_KV * stride_seq

    # Store final dQ block
    dQ_block_ptrs = dQ + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dQ_block_ptrs, dQ_block)


@triton.jit
def _attn_bwd_dk_dv(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    # Calculate base offset for all tensors
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(
        tl.int64
    )
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    # Apply offsets to all tensors
    base_offset = offset_batch_head
    Q += base_offset
    K += base_offset
    V += base_offset
    dO += base_offset
    dQ += base_offset
    dK += base_offset
    dV += base_offset

    M += offset_batch_head_seq
    D += offset_batch_head_seq

    # Setup dimensions
    offs_dim = tl.arange(0, HEAD_DIM)
    index_block_kv = tl.program_id(0)
    start_kv = index_block_kv * BLOCK_KV
    offs_kv = start_kv + tl.arange(0, BLOCK_KV)

    # Initialize gradient blocks
    dV_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dK_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    # Load K and V blocks
    K_block = tl.load(
        K + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )
    V_block = tl.load(
        V + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )

    offs_q = tl.arange(0, BLOCK_Q)
    qT_ptrs = Q + offs_q[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    dO_ptrs = dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim

    # Process blocks
    curr_q = 0
    num_steps = SEQ_LEN // BLOCK_Q
    for blk_idx in range(num_steps):
        # Load Q block and M values
        qT_block = tl.load(qT_ptrs)
        offs_q = curr_q + tl.arange(0, BLOCK_Q)
        m = tl.load(M + offs_q)

        # Compute attention scores
        QK_T_block = softmax_scale * tl.dot(K_block, qT_block)
        P_T_block = tl.math.exp(QK_T_block - m[None, :])

        # Apply causal mask
        mask_block = offs_q[None, :] >= offs_kv[:, None]
        P_T_block = tl.where(mask_block, P_T_block, 0.0)

        # Load dO block
        dO_block = tl.load(dO_ptrs)

        # Compute dV
        dV_block += tl.dot(P_T_block.to(tl.float16), dO_block)

        # Compute dK
        Di = tl.load(D + offs_q)
        dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)
        dS_T_block = P_T_block * (dpT_block - Di[None, :])
        dS_T_block = dS_T_block.to(tl.float16)
        dK_block += softmax_scale * tl.dot(dS_T_block, tl.trans(qT_block))

        # Move to next block
        curr_q += BLOCK_Q
        qT_ptrs += BLOCK_Q * stride_seq
        dO_ptrs += BLOCK_Q * stride_seq

    # Store final gradients
    dV_block_ptrs = dV + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dV_block_ptrs, dV_block)

    dK_block_ptrs = dK + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dK_block_ptrs, dK_block)


@triton.jit
def _attn_bwd_preprocess(
    O,
    dO,
    D,
    SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    block_index_q = tl.program_id(0)
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    index_batch_head = tl.program_id(1)
    offs_dim = tl.arange(0, HEAD_DIM)

    # Load and process blocks
    O_block = tl.load(
        O
        + index_batch_head * HEAD_DIM * SEQ_LEN
        + offs_q[:, None] * HEAD_DIM
        + offs_dim[None, :]
    )
    dO_block = tl.load(
        dO
        + index_batch_head * HEAD_DIM * SEQ_LEN
        + offs_q[:, None] * HEAD_DIM
        + offs_dim[None, :]
    ).to(tl.float32)
    D_block = tl.sum(dO_block * O_block, axis=1)

    # Store result
    D_block_ptrs = D + index_batch_head * SEQ_LEN + offs_q
    tl.store(D_block_ptrs, D_block)


class CausalAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, softmax_scale=None):
        # Validate inputs
        assert Q.stride() == K.stride() == V.stride(), "Q, K, V must have same stride"
        assert (
            Q.shape[-1] == K.shape[-1] == V.shape[-1]
        ), "Q, K, V must have same HEAD_DIM"

        HEAD_DIM = Q.shape[-1]
        BATCH_SIZE, NUM_HEADS, SEQ_LEN = Q.shape[:3]

        if softmax_scale is None:
            softmax_scale = 1.0 / (HEAD_DIM**0.5)

    # Reference implementation
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
    P = P.masked_fill(MASK[None, None, :, :] == 0, float("-inf"))
    P = torch.softmax(P.float(), dim=-1).half()
    ref_O = torch.matmul(P, V)
    ref_O.backward(dO)
    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None

    # Triton implementation
    tri_out = CausalAttention.apply(Q, K, V, softmax_scale)
    tri_out.backward(dO)
    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None

    # Compare results
    rtol = 0.0
    atol = 1e-2
    assert torch.allclose(ref_O, tri_out, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)
    print("All tests passed!")


def test_causal_attention(
    BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=4096, HEAD_DIM=64, dtype=torch.float16
):
    """
    Test causal attention implementation against PyTorch reference implementation.

    Args:
        BATCH_SIZE: Number of sequences in batch
        NUM_HEADS: Number of attention heads
        SEQ_LEN: Length of input sequence
        HEAD_DIM: Dimension of each head
        dtype: Data type for tensors
    """
    # Create test inputs
    torch.manual_seed(42)
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    K = torch.empty_like(Q).normal_(0, 0.5).requires_grad_()
    V = torch.empty_like(Q).normal_(0, 0.5).requires_grad_()

    # Create gradient for backward pass
    dO = torch.randn_like(Q)
    softmax_scale = 1.0 / (HEAD_DIM**0.5)

    print("Testing forward/backward pass...")
    print(
        f"Shapes: BATCH_SIZE={BATCH_SIZE}, NUM_HEADS={NUM_HEADS}, SEQ_LEN={SEQ_LEN}, HEAD_DIM={HEAD_DIM}"
    )

    # Reference implementation
    with torch.cuda.amp.autocast(dtype=dtype):
        # Create causal mask
        MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
        # Compute attention scores
        P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
        # Apply causal mask
        P = P.masked_fill(MASK[None, None, :, :] == 0, float("-inf"))
        # Softmax and matmul with values
        P = torch.softmax(P.float(), dim=-1).to(dtype)
        ref_O = torch.matmul(P, V)

    # Backward pass for reference implementation
    ref_O.backward(dO)
    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None

    # Reset gradients for Triton implementation
    Q.grad = None
    K.grad = None
    V.grad = None

    # Triton implementation
    tri_out = CausalAttention.apply(Q, K, V, softmax_scale)
    tri_out.backward(dO)
    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None

    # Compare results
    rtol = 0.0
    atol = 1e-2

    print("\nChecking forward pass...")
    forward_match = torch.allclose(ref_O, tri_out, atol=atol, rtol=rtol)
    print(f"Forward pass {'matched' if forward_match else 'MISMATCH'}")
    if not forward_match:
        max_diff = (ref_O - tri_out).abs().max().item()
        print(f"Max difference in forward pass: {max_diff}")

    print("\nChecking backward pass...")
    dq_match = torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)
    dk_match = torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    dv_match = torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)

    print(f"dQ {'matched' if dq_match else 'MISMATCH'}")
    print(f"dK {'matched' if dk_match else 'MISMATCH'}")
    print(f"dV {'matched' if dv_match else 'MISMATCH'}")

    if not all([dq_match, dk_match, dv_match]):
        print("\nMax differences in backward pass:")
        print(f"dQ: {(ref_dQ - tri_dQ).abs().max().item()}")
        print(f"dK: {(ref_dK - tri_dK).abs().max().item()}")
        print(f"dV: {(ref_dV - tri_dV).abs().max().item()}")

    all_passed = all([forward_match, dq_match, dk_match, dv_match])
    if all_passed:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")

    return all_passed


if __name__ == "__main__":
    # Test with different configurations
    print("Testing small sequence...")
    test_causal_attention(BATCH_SIZE=2, NUM_HEADS=4, SEQ_LEN=128, HEAD_DIM=64)

    print("\nTesting medium sequence...")
    test_causal_attention(BATCH_SIZE=4, NUM_HEADS=8, SEQ_LEN=1024, HEAD_DIM=64)

    print("\nTesting large sequence...")
    test_causal_attention(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=4096, HEAD_DIM=64)
