import triton.language as tl
import triton
import torch


@triton.jit()
def stay_attention(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_b, stride_nh, 
    stride_qs, stride_qh,
    stride_ks, stride_kh,
    stride_vs, stride_vh,
    stride_os, stride_oh,
    seq_len, head_dim,
    sm_scale,
    BLOCK_SEQ: tl.constexpr, 
    BLOCK_HD: tl.constexpr, 
    NUM_SM: tl.constexpr,
):  

    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid = tl.program_id(2)
    
    qkv_offset = pid_b*stride_b + pid_h*stride_nh
    num_tiles_seq_len = tl.cdiv(seq_len, BLOCK_SEQ)

    tiles_per_SM = num_tiles_seq_len // NUM_SM
    if pid < num_tiles_seq_len % NUM_SM:
        tiles_per_SM += 1

    tile_id = pid - NUM_SM
    si = -1

    pid_seq_m = 0
    pid_seq_n = 0

    offs_seq_m = tl.arange(0, BLOCK_SEQ)
    offs_seq_n = tl.arange(0, BLOCK_SEQ)
    offs_head = tl.arange(0, BLOCK_HD)

    q_ptrs = q_ptr + qkv_offset + offs_seq_n[:, None]*stride_qs + offs_head[None, :]*stride_qh

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_SEQ], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SEQ], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504

    q = tl.load(q_ptrs)
    q = (q * qk_scale)
    
    pv = tl.zeros([BLOCK_SEQ, BLOCK_HD], dtype=tl.float32)
    for _ in range(0, num_tiles_seq_len * tiles_per_SM):

        si = tl.where(si == num_tiles_seq_len - 1, 0, si + 1)
        
        if si == 0:

            tile_id += NUM_SM

            pid_seq_m = pid // num_tiles_seq_len
            pid_seq_n = pid % num_tiles_seq_len

            offs_seq_m = pid_seq_m*BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
            offs_seq_n = pid_seq_n*BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
            offs_head = tl.arange(0, BLOCK_HD)

            q_ptrs = q_ptr + qkv_offset + offs_seq_n[:, None]*stride_qs + offs_head[None, :]*stride_qh
            
            qk_scale = sm_scale * 1.44269504
            q = tl.load(q_ptrs)
            q = (q * qk_scale)
        
        offs_seq_m = si*BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
        offs_head = tl.arange(0, BLOCK_HD)

        k_ptrs = k_ptr + qkv_offset + offs_seq_m[:, None]*stride_ks + offs_head[None, :]*stride_kh
        v_ptrs = v_ptr + qkv_offset + offs_seq_m[:, None]*stride_vs + offs_head[None, :]*stride_vh

        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        qk = tl.dot(q.to(tl.float16), k.T, out_dtype=tl.float32)

        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])

        # -- scale and update acc --
        pv *= alpha[:, None]
        pv += tl.dot(p.to(tl.float16), v, out_dtype=tl.float32)

        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

        if si == num_tiles_seq_len - 1:

            offs_seq_n = pid_seq_n*BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
            pv = pv / l_i[:, None]
            o_ptrs = o_ptr + qkv_offset + offs_seq_n[:, None]*stride_os + offs_head[None, :]*stride_oh
            tl.store(o_ptrs, pv)
            pv = tl.zeros([BLOCK_SEQ, BLOCK_HD], dtype=tl.float32)


def flash_fn(q, k, v):

    batch, num_heads, seq_len, head_dim = q.shape

    sm_scale = 0.5
    BLOCK_SEQ = 64
    BLOCK_HD = 128

    NUM_SM = torch.cuda.get_device_properties("cuda").multi_processor_count
    grid = (batch, num_heads, NUM_SM)
    o = torch.zeros(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')    
    stay_attention[grid](q, k, v, o,
                             q.stride(0), q.stride(1), 
                             q.stride(2), q.stride(3), 
                             k.stride(2), k.stride(3),
                             v.stride(2), v.stride(3),
                             o.stride(2), o.stride(3),
                             seq_len, head_dim,
                             sm_scale,
                             BLOCK_SEQ, BLOCK_HD, NUM_SM)
    return o 


if __name__ == '__main__':

    torch.manual_seed(0)

    batch, num_heads, seq_len, head_dim = 1, 32, 4096, 128

    q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda') // 10
    k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda') // 10
    v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda') // 10

    sm_scale = 0.5
    p = (q @ k.transpose(2, 3)) * sm_scale
    p = torch.softmax(p.float(), dim=-1)
    o_torch = torch.matmul(p.to(torch.float16), v)

    o_triton = flash_fn(q, k, v)

    print(f"{o_triton=}")
    print(f"{o_torch=}")

    torch.testing.assert_close(o_triton, o_torch, atol=1e-2, rtol=0)

