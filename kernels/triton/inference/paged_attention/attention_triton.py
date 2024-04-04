#from einops import rearrange
import torch
import triton
import triton.language as tl

# Expect block table to map
# logical bid (block id) -> (physical bid, # filled)
# In tests, it maps: logical pid -> physical bid

@triton.jit
def print_tensor_dim(tensor, str_name):
    if tl.program_id(0) == 0 and tl.program_id(1) == 0:
        tl.static_print(str_name," ",tensor.shape," ",tensor.dtype)
        #tl.static_print('*************** program id: ', tl.program_id(0), tl.program_id(1))

@triton.jit
def print_value(value):
    if tl.program_id(0) == 0 and tl.program_id(1) == 0:
        tl.device_print(str(value))
        #tl.static_print('*************** program id: ', tl.program_id(0), tl.program_id(1))
        #tl.static_print(str_name+" ")

@triton.jit
def print_line(str_line):
    if tl.program_id(0) == 0 and tl.program_id(1) == 0:
        print(str_line)

#Paged Attention V1: basic version, has a memory limitation error
@triton.jit
def paged_attention_v1(
    # need these b/c we can't use view/reshape
    scratchpad_key_ptr,  # [num_seqs, max_seq_len, num_heads, head_size]
    scratchpad_value_ptr,  # [num_seqs, max_seq_len, num_heads, head_size]
    output_ptr,  # [num_seqs, num_query_heads, head_size]
    query_ptr,  # [num_seqs, num_query_heads, head_size]
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size, block_size]
    value_cache_ptr,  # [num_blocks, num_kv_heads, head_size, block_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    context_lens_ptr,  # [num_seqs]
    scale,  # float32
    num_seqs,  # int
    num_heads,  # int
    cache_block_stride,  # int
    MAX_SEQ_LEN: tl.constexpr,  # int (same as max_seq_len)
    BLOCK_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    MAX_NUM_BLOCKS_PER_SEQ: tl.constexpr,  # int, must be power of 2
):
    seq_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
     
    #Compute the offsets of the query using the strides
    #TODO(amorari) use the strides as returned from tensor.stride() instead 
    query_offset = seq_idx * num_seqs + head_idx * HEAD_SIZE
    query_head = tl.load(query_ptr + query_offset + tl.arange(0, HEAD_SIZE))
    #print_tensor_dim(query_head, "query_head")
    
    block_table_offset = seq_idx * MAX_NUM_BLOCKS_PER_SEQ
    #load the context len for this q vector
    context_len = tl.load(context_lens_ptr + seq_idx)

    #print_tensor_dim(block_tables_ptr, "block_tables_ptr")
   
    #iterate on the tokens
    for tok_idx in range(0, context_len):
        logical_block_idx = tok_idx // BLOCK_SIZE
        
        #physical block starting pointer for token
        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + logical_block_idx
        )

        start_of_block_offset = (
            physical_block_idx.to(tl.int64) * cache_block_stride + head_idx * HEAD_SIZE * BLOCK_SIZE
        )
        tok_idx_within_block = tok_idx % BLOCK_SIZE
        tok_offsets = (
            start_of_block_offset
            + BLOCK_SIZE * tl.arange(0, HEAD_SIZE)
            + tok_idx_within_block
        )

        #Get all blocks for this token
        tok_key = tl.load(key_cache_ptr + tok_offsets)
        tok_value = tl.load(value_cache_ptr + tok_offsets)
        #print_tensor_dim(tok_key, "tok_key")
        #print_tensor_dim(tok_value, "tok_value")

        #Compute offsets to write in the scratchpad
        scratchpad_offset = (
            seq_idx.to(tl.int64) * (MAX_SEQ_LEN * num_heads.to(tl.int64) * HEAD_SIZE)
            + tok_idx.to(tl.int64) * (num_heads * HEAD_SIZE)
            + head_idx * HEAD_SIZE
        )
        tl.store(
            scratchpad_key_ptr + scratchpad_offset + tl.arange(0, HEAD_SIZE), tok_key
        )
        tl.store(
            scratchpad_value_ptr + scratchpad_offset + tl.arange(0, HEAD_SIZE),
            tok_value,
        )


    # TODO: Not sure if this is necessary
    tl.debug_barrier()

    # scratchpad_key_ptr,  # [num_seqs, max_seq_len, num_heads, head_size]
    start_seq_offset = (MAX_SEQ_LEN * num_heads * HEAD_SIZE) * seq_idx

    start_tok_offset = start_seq_offset + tl.arange(0, MAX_SEQ_LEN) \
        * (num_heads * HEAD_SIZE) + head_idx * HEAD_SIZE


    # [seq_len, head_size]
    # zero out keys that aren't part of the sequence

    mask = tl.arange(0, MAX_SEQ_LEN)[:, None] < context_len
    kv_offs = start_tok_offset[:, None] + tl.arange(0, HEAD_SIZE)[None, :]
    print_tensor_dim(kv_offs, "kv_offs_v1")
    keys = tl.load(scratchpad_key_ptr + kv_offs, mask=mask, other=0.0)
    print_tensor_dim(keys, "keys_v1")
    values = tl.load(scratchpad_value_ptr + kv_offs, mask=mask, other=0.0)
    print_tensor_dim(values, "values_v1")

    # keys shape  [seq_len x head_size], query shape = [head_size]
    # Can't do below b/c minimum size on all dimensions is 16
    # scores = tl.dot(query_head[None, :], keys.T)
    
    scores = tl.sum(scale * keys * query_head[None, :], axis=1)

    # This mask is necessary b/c even though we mask out the keys on load
    # that just results in 0s in the attention dot product, 
    # which then get softmaxed and result in non-zero values 
    # in the softmax output (which is wrong)
    # -inf guarantees that the softmax output will be 0 for masked values
    mask = tl.full([MAX_SEQ_LEN], -float('inf'), dtype=tl.float32)
    cond = tl.arange(0, MAX_SEQ_LEN) < context_len
    scores_masked = tl.where(cond, scores, mask)

    # do a numerically stable softmax on the scores
    scores_minus_max = scores_masked - tl.max(scores_masked, axis=0)

    
    numerator = tl.exp(scores_minus_max)
    denominator = tl.sum(numerator, axis=0) + float(1e-6)
    logits = numerator / denominator
    print_tensor_dim(logits, "logits_v1")

    weighted_values = tl.sum(values * logits[:, None], axis=0)
    print_tensor_dim(weighted_values, "weighted_values_v1")

    output_offset = seq_idx * (num_heads * HEAD_SIZE) + head_idx * HEAD_SIZE
    tl.store(output_ptr + output_offset + tl.arange(0, HEAD_SIZE), weighted_values)

def paged_attention_triton_v1(
            output,
            query,
            key_cache,
            value_cache,
            #head_mapping,
            scale,
            block_tables,
            context_lens,
            block_size,
            #max_seq_len,
            #alibi_slopes, 
            num_seqs,
            num_query_heads,
            max_seq_len,
            max_num_blocks_per_seq,
            head_size
):
    scratchpad_key = torch.zeros(
        (num_seqs, max_seq_len, num_query_heads, head_size),
        dtype=torch.float32,
        device="cuda",
    )
    
    scratchpad_value = torch.zeros_like(scratchpad_key)

    paged_attention_v1[(num_seqs, num_query_heads)](
        scratchpad_key_ptr=scratchpad_key,
        scratchpad_value_ptr=scratchpad_value,
        output_ptr=output,
        query_ptr=query,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        block_tables_ptr=block_tables,
        context_lens_ptr=context_lens,
        scale=scale,
        num_seqs=num_seqs,
        num_heads=num_query_heads,
        cache_block_stride=key_cache.stride(0),
        MAX_SEQ_LEN=max_seq_len,
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size,
        MAX_NUM_BLOCKS_PER_SEQ=max_num_blocks_per_seq,
    )


#Paged Attention V2: Iterate on kv vectors to avoid memory limitation error (sram)
@triton.jit
def paged_attention_v2(
    # need these b/c we can't use view/reshape
    scratchpad_key_ptr,  # [num_seqs, max_seq_len, num_heads, head_size]
    scratchpad_value_ptr,  # [num_seqs, max_seq_len, num_heads, head_size]
    partition_buf_ptr,
    output_ptr,  # [num_seqs, num_query_heads, head_size]
    query_ptr,  # [num_seqs, num_query_heads, head_size]
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size, block_size]
    value_cache_ptr,  # [num_blocks, num_kv_heads, head_size, block_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    context_lens_ptr,  # [num_seqs]
    scale,  # float32
    num_seqs,  # int
    num_heads,  # int
    cache_block_stride,  # int
    num_partitions, #int
    PARTITION_SIZE: tl.constexpr, #int
    MAX_SEQ_LEN: tl.constexpr,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    MAX_NUM_BLOCKS_PER_SEQ: tl.constexpr,  # int, must be power of 2
):
    seq_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    partition_idx = tl.program_id(2).to(tl.int64)
   
    #Compute the offsets of the query using the strides
    #TODO(amorari) use the strides as returned from tensor.stride() instead 
    query_offset = seq_idx * num_seqs + head_idx * HEAD_SIZE

    #load one q vector
    query_head = tl.load(query_ptr + query_offset + tl.arange(0, HEAD_SIZE))
    print_tensor_dim(query_head, "query_head")
    
    block_table_offset = seq_idx * MAX_NUM_BLOCKS_PER_SEQ
    #load the context len for this q vector
    context_len = tl.load(context_lens_ptr + seq_idx)
    assert(context_len <= MAX_SEQ_LEN)

    #iterate on the tokens in this partition
    token_start_idx = partition_idx * PARTITION_SIZE
    token_end_idx = min((partition_idx + 1) * PARTITION_SIZE, context_len)
    #NOTE: For some sequence, it is possible that context_len < token_start_idx
    for tok_idx in range(token_start_idx, token_end_idx):
        logical_block_offset = tok_idx // BLOCK_SIZE
        
        #physical block starting pointer for token
        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + logical_block_offset
        )

        start_of_block_offset = (
            physical_block_idx * cache_block_stride + head_idx * HEAD_SIZE * BLOCK_SIZE
        )

        tok_idx_within_block = tok_idx % BLOCK_SIZE
        tok_offsets = (
            start_of_block_offset
            + BLOCK_SIZE * tl.arange(0, HEAD_SIZE)
            + tok_idx_within_block
        )

        tok_key = tl.load(key_cache_ptr + tok_offsets)
        #print_tensor_dim(tok_key, "tok_key")
        tok_value = tl.load(value_cache_ptr + tok_offsets)
        #print_tensor_dim(tok_key, "tok_value")

        scratchpad_offset = (
            seq_idx.to(tl.int64) * (MAX_SEQ_LEN * num_heads.to(tl.int64) * HEAD_SIZE)
            + tok_idx.to(tl.int64) * (num_heads.to(tl.int64) * HEAD_SIZE)
            + head_idx * HEAD_SIZE
        )

        print_tensor_dim(scratchpad_key_ptr, "scratchpad_key_ptr")
        mask=tl.full([HEAD_SIZE], 1, dtype=tl.float32) > 0
        #store the keys in line
        tl.store(
            scratchpad_key_ptr + scratchpad_offset + tl.arange(0, HEAD_SIZE), tok_key, mask
        )
        #store the values in line
        tl.store(
            scratchpad_value_ptr + scratchpad_offset + tl.arange(0, HEAD_SIZE), 
            tok_value, mask
        )

    # TODO: Not sure if this is necessary
    tl.debug_barrier()

   
    #start of the sequence
    start_seq_offset = (MAX_SEQ_LEN * num_heads.to(tl.int64) * HEAD_SIZE) * seq_idx.to(tl.int64)
    #offsets with the start of the token
    start_tok_offsets = start_seq_offset.to(tl.int64) \
                    + tl.arange(0, PARTITION_SIZE) * (num_heads.to(tl.int64) * HEAD_SIZE) \
                    + head_idx.to(tl.int64) * HEAD_SIZE

    # [seq_len, head_size]
    # zero out keys that aren't part of the sequence

    mask = tl.arange(0, PARTITION_SIZE)[:, None] < context_len
    kv_offs = start_tok_offsets[:, None] + tl.arange(0, HEAD_SIZE)[None, :]
    print_tensor_dim(kv_offs, "kv_offs_v2")
    keys = tl.load(scratchpad_key_ptr + kv_offs, mask=mask, other=0.0)
    print_tensor_dim(keys, "keys_v2")

    # Can't do below b/c minimum size on all dimensions is 16
    # scores = tl.dot(query_head[None, :], keys.T)
    scores = tl.sum(scale * keys * query_head[None, :], axis=1)
    print_tensor_dim(keys, "scores_v2")

    partition_buf_offset = start_seq_offset \
        + head_idx.to(tl.int64) * HEAD_SIZE + partition_idx.to(tl.int64) * PARTITION_SIZE
    print_tensor_dim(partition_buf_offset, "partition_buf_offset_v2")

    tl.store(partition_buf_ptr + partition_buf_offset + tl.arange(0, PARTITION_SIZE), scores)
        
    #weighted_values = tl.zeros(HEAD_SIZE, dtype=tl.float32)

    # This mask is necessary b/c even though we mask out the keys on load
    # that just results in 0s in the attention dot product, 
    # which then get softmaxed and result in non-zero values 
    # in the softmax output (which is wrong)
    # -inf guarantees that the softmax output will be 0 for masked values
    mask = tl.full([PARTITION_SIZE], -float('inf'), dtype=tl.float32)
    cond = tl.arange(0, PARTITION_SIZE) < context_len
    scores_masked = tl.where(cond, scores, mask)

    # do a numerically stable softmax on the scores
    scores_minus_max = scores_masked - tl.max(scores_masked, axis=0)
    numerator = tl.exp(scores_minus_max)
    denominator = tl.sum(numerator, axis=0) + float(1e-6)

    logits = numerator / denominator
    print_tensor_dim(logits, "logits_v2")

    values = tl.load(scratchpad_value_ptr + kv_offs, mask=mask, other=0.0)
    print_tensor_dim(values, "values_v2")
    weighted_values += tl.sum(values * logits[:, None], axis=0)
    print_tensor_dim(weighted_values, "weighed_values_v2")

    #output_offset = seq_idx.to(tl.int64) * (num_heads.to(tl.int64) * HEAD_SIZE) \
    #    + head_idx.to(tl.int64) * HEAD_SIZE + seq_partition_idx.to(tl.int64) * PARTITION_SIZE

    #to_store_values=weighted_values.to(tl.float32)
    #mask = tl.full([HEAD_SIZE], 1, dtype=tl.float32) > 0
    #tl.store(output_ptr + output_offset + tl.arange(0, HEAD_SIZE), to_store_values, mask)


    output_offset = seq_idx * (num_heads * HEAD_SIZE) + head_idx * HEAD_SIZE
    tl.store(output_ptr + output_offset + tl.arange(0, HEAD_SIZE), weighted_values)


def paged_attention_triton_v2(
            output,
            query,
            key_cache,
            value_cache,
            #head_mapping,
            scale,
            block_tables,
            context_lens,
            block_size,
            partition_size,
            #alibi_slopes, 
            num_seqs,
            num_query_heads,
            max_seq_len,
            max_num_blocks_per_seq,
            head_size
):

    scratchpad_key = torch.zeros(
        (num_seqs, max_seq_len, num_query_heads, head_size),
        dtype=torch.float32,
        device="cuda",
    )

    scratchpad_value = torch.zeros_like(scratchpad_key)

    num_partitions = max_seq_len//partition_size
    assert(max_seq_len % partition_size == 0)

    partition_buf_ptr = torch.zeros((num_seqs,max_seq_len,num_query_heads,head_size),
                                    dtype=torch.float32,
                                    device="cuda")
   
    #print(f"started_v2 num_seqs: {num_seqs} num_query_heads: {num_query_heads}")
    paged_attention_v2[(num_seqs, num_query_heads, num_partitions)](
        scratchpad_key_ptr=scratchpad_key,
        scratchpad_value_ptr=scratchpad_value,
        partition_buf_ptr=partition_buf_ptr,
        output_ptr=output,
        query_ptr=query,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        block_tables_ptr=block_tables,
        context_lens_ptr=context_lens,
        scale=scale,
        num_seqs=num_seqs,
        num_heads=num_query_heads,
        cache_block_stride=key_cache.stride(0),
        num_partitions=num_partitions,
        PARTITION_SIZE=partition_size,
        MAX_SEQ_LEN=max_seq_len,
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size,
        MAX_NUM_BLOCKS_PER_SEQ=max_num_blocks_per_seq,
    )
    #print("finished_v2")

