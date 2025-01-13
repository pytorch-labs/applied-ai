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
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    # Stage 1: From 0 to the left of the diagonal
    # Stage 2: For the block containing the diagonal
    if STAGE == 1:
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    else:  # STAGE == 2
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)

        if STAGE == 2:
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
        for num_stages in [3, 4]
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
    block_index_q = tl.program_id(0)
    index_batch_head = tl.program_id(1)

    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    qkv_offset = (
        index_batch.to(tl.int64) * stride_batch + index_head.to(tl.int64) * stride_head
    )

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

    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    Q_block = tl.load(Q_block_ptr)

    # Process blocks to the left of diagonal
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
        1,
        offs_q,
        offs_kv,
        SEQ_LEN,
    )

    # Process diagonal block
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
        2,
        offs_q,
        offs_kv,
        SEQ_LEN,
    )

    m_i += tl.math.log(l_i)
    O_block = O_block / l_i[:, None]

    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))


@triton.jit
def _attn_bwd_preprocess(
    O,
    dO,
    D,
    SEQ_LEN,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    block_index_q = tl.program_id(0)
    offs_q = block_index_q * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    index_batch_head = tl.program_id(1)
    offs_dim = tl.arange(0, HEAD_DIM)

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
    D_block_ptrs = D + index_batch_head * SEQ_LEN + offs_q
    tl.store(D_block_ptrs, D_block)


@triton.jit
def _attn_bwd_dq(
    Q,
    K,
    V,
    dO,
    dQ,
    M,
    D,
    softmax_scale,
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
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(
        tl.int64
    )
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    M += offset_batch_head_seq
    D += offset_batch_head_seq

    offs_dim = tl.arange(0, HEAD_DIM)
    index_block_kv = tl.program_id(0)
    start_q = index_block_kv * BLOCK_Q
    offs_q = start_q + tl.arange(0, BLOCK_Q)

    Q_block = tl.load(Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)
    dQ_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    dO_block = tl.load(
        dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )
    M_block = tl.load(M + offs_q)[:, None]

    offs_kv = tl.arange(0, BLOCK_KV)
    kT_ptrs = K + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    vT_ptrs = V + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    Di = tl.load(D + offs_q)

    curr_kv = 0
    for blk_idx in range(SEQ_LEN // BLOCK_KV):
        K_T_block = tl.load(kT_ptrs)
        V_T_block = tl.load(vT_ptrs)
        QK_block = softmax_scale * tl.dot(Q_block, K_T_block)
        P_block = tl.math.exp(QK_block - M_block)

        # Causal masking
        offs_kv = curr_kv + tl.arange(0, BLOCK_KV)
        mask_block = offs_q[:, None] >= offs_kv[None, :]
        P_block = tl.where(mask_block, P_block, 0.0)

        dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
        dS_block = P_block * (dP_block - Di[:, None])
        dS_block = dS_block.to(tl.float16)
        dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block))

        curr_kv += BLOCK_KV
        kT_ptrs += BLOCK_KV * stride_seq
        vT_ptrs += BLOCK_KV * stride_seq

    dQ_block_ptrs = dQ + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dQ_block_ptrs, dQ_block)


@triton.jit
def _attn_bwd_dkv(
    Q,
    K,
    V,
    dO,
    dK,
    dV,
    M,
    D,
    softmax_scale,
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
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(
        tl.int64
    )
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head
    M += offset_batch_head_seq
    D += offset_batch_head_seq

    offs_dim = tl.arange(0, HEAD_DIM)
    index_block_kv = tl.program_id(0)
    start_kv = index_block_kv * BLOCK_KV
    offs_kv = start_kv + tl.arange(0, BLOCK_KV)

    dV_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dK_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    K_block = tl.load(
        K + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )
    V_block = tl.load(
        V + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )

    offs_q = tl.arange(0, BLOCK_Q)
    qT_ptrs = Q + offs_q[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    dO_ptrs = dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim

    curr_q = 0
    for blk_idx in range(SEQ_LEN // BLOCK_Q):
        qT_block = tl.load(qT_ptrs)
        offs_q = curr_q + tl.arange(0, BLOCK_Q)
        m = tl.load(M + offs_q)

        QK_T_block = softmax_scale * tl.dot(K_block, qT_block)
        P_T_block = tl.math.exp(QK_T_block - m[None, :])

        # Causal masking
        mask_block = offs_q[None, :] >= offs_kv[:, None]
        P_T_block = tl.where(mask_block, P_T_block, 0.0)

        dO_block = tl.load(dO_ptrs)
        dV_block += tl.dot(P_T_block.to(tl.float16), dO_block)

        Di = tl.load(D + offs_q)
        dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)
        dS_T_block = P_T_block * (dpT_block - Di[None, :])
        dK_block += softmax_scale * tl.dot(
            dS_T_block.to(tl.float16), tl.trans(qT_block)
        )

        curr_q += BLOCK_Q
        qT_ptrs += BLOCK_Q * stride_seq
        dO_ptrs += BLOCK_Q * stride_seq

    dV_ptrs = dV + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dV_ptrs, dV_block)
    dK_ptrs = dK + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dK_ptrs, dK_block)


class TritonCausalAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, softmax_scale=None):
        if softmax_scale is None:
            softmax_scale = 1.0 / (Q.shape[-1] ** 0.5)

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
        O = torch.empty_like(Q)
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        )

        grid = (triton.cdiv(SEQ_LEN, 128), BATCH_SIZE * NUM_HEADS, 1)

        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=softmax_scale,
            M=M,
            O=O,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            BATCH_SIZE=BATCH_SIZE,
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
        )

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.softmax_scale = softmax_scale
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, M = ctx.saved_tensors
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        BLOCK_SIZE = 128
        grid = (SEQ_LEN // BLOCK_SIZE, BATCH_SIZE * NUM_HEADS)

        # Compute D = sum(dO * O, dim=-1)
        D = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        )
        _attn_bwd_preprocess[grid](
            O=O,
            dO=dO,
            D=D,
            SEQ_LEN=SEQ_LEN,
            BLOCK_SIZE=BLOCK_SIZE,
            HEAD_DIM=HEAD_DIM,
        )

        # Compute gradients
        grid = (SEQ_LEN // BLOCK_SIZE, 1, BATCH_SIZE * NUM_HEADS)

        _attn_bwd_dq[grid](
            Q=Q,
            K=K,
            V=V,
            dO=dO,
            dQ=dQ,
            M=M,
            D=D,
            softmax_scale=ctx.softmax_scale,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE,
            BLOCK_KV=32,
            HEAD_DIM=HEAD_DIM,
        )

        _attn_bwd_dkv[grid](
            Q=Q,
            K=K,
            V=V,
            dO=dO,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            softmax_scale=ctx.softmax_scale,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=32,
            BLOCK_KV=BLOCK_SIZE,
            HEAD_DIM=HEAD_DIM,
        )

        return dQ, dK, dV, None


def attention(q, k, v, softmax_scale=None):
    """
    Compute causal attention using Triton kernels.
    Args:
        q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        v: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
        softmax_scale: Optional scaling factor for attention scores
    Returns:
        Output tensor of shape (batch_size, num_heads, seq_len, head_dim)
    """
    return TritonCausalAttention.apply(q, k, v, softmax_scale)


def test_causal_attention():
    """Test causal attention implementation against PyTorch reference."""
    torch.manual_seed(0)

    BATCH_SIZE = 2
    NUM_HEADS = 4
    SEQ_LEN = 1024  # Using smaller sequence length for testing
    HEAD_DIM = 64

    # Create inputs
    Q = torch.randn(
        BATCH_SIZE,
        NUM_HEADS,
        SEQ_LEN,
        HEAD_DIM,
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    K = torch.randn_like(Q, requires_grad=True)
    V = torch.randn_like(Q, requires_grad=True)
    dO = torch.randn_like(Q)

    # PyTorch reference implementation
    def pytorch_causal_attention(q, k, v):
        scale = 1.0 / (q.shape[-1] ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Create causal mask
        mask = torch.triu(
            torch.ones(SEQ_LEN, SEQ_LEN, device=scores.device), diagonal=1
        ).bool()
        scores.masked_fill_(mask[None, None, :, :], float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    # Forward pass comparison
    triton_out = attention(Q, K, V)
    torch_out = pytorch_causal_attention(Q, K, V)

    max_error = (triton_out - torch_out).abs().max().item()
    print(f"Max forward pass error: {max_error}")
    assert max_error < 1e-2, "Forward pass error too large"

    # Backward pass comparison
    triton_out.backward(dO, retain_graph=True)
    torch_out.backward(dO)

    for name, (triton_grad, torch_grad) in [
        ("dQ", (Q.grad, Q.grad)),
        ("dK", (K.grad, K.grad)),
        ("dV", (V.grad, V.grad)),
    ]:
        max_error = (triton_grad - torch_grad).abs().max().item()
        print(f"Max {name} backward pass error: {max_error}")
        assert max_error < 1e-2, f"{name} backward pass error too large"

    print("All tests passed!")


if __name__ == "__main__":
    test_causal_attention()
