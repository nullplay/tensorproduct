import torch

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton.jit
def my_coo_triton_old(
    imap0, # First dimension of the input indices in the COO format. Shape is [Po * Pi] (flattened)
    imap1, # Second dimension of the input indices in the COO format.
    omap0, # The output indices in the COO format. Shape is [Po].
           # BUT importantly, the elements in this array are spaced 4 elements apart in memory
           # because the previous kernel stored it that way.
    vals, # The values in the COO format. Shape is [Po * Pi]
    input0, # Input tensor 1. Shape is [B, L1]
    input1, # Input tensor 2. Shape is [B, L2]
    output, # Output tensor. Shape is [B, L3]
    #debug_out, # [Po, Pi]
    # Note that for our purposes L1, L2, and L3 are pre-defined constants.
    ynumel, # B
    xnumel, # Po
    YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # First define offsets and strides
    y_offset = tl.program_id(1) * YBLOCK
    x_offset = tl.program_id(0) * XBLOCK
    l1_dim = 36
    l2_dim = 36
    l3_dim = 1296
    y_stride = l1_dim
    x_stride = 1

    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    y_index = y_offset + tl.arange(0, YBLOCK)[None, :]
    x_mask = x_index < xnumel
    y_mask = y_index < ynumel

    i_idx_0 = tl.load(imap0 + x_index * 4,     x_mask)
    i_idx_1 = tl.load(imap0 + x_index * 4 + 1, x_mask)
    i_idx_2 = tl.load(imap0 + x_index * 4 + 2, x_mask)
    i_idx_3 = tl.load(imap0 + x_index * 4 + 3, x_mask)
    j_idx_0 = tl.load(imap1 + x_index * 4,     x_mask)
    j_idx_1 = tl.load(imap1 + x_index * 4 + 1, x_mask)
    j_idx_2 = tl.load(imap1 + x_index * 4 + 2, x_mask)
    j_idx_3 = tl.load(imap1 + x_index * 4 + 3, x_mask)
    w_vals_0 = tl.load(vals + x_index * 4,      x_mask)
    w_vals_1 = tl.load(vals + x_index * 4 + 1,  x_mask)
    w_vals_2 = tl.load(vals + x_index * 4 + 2,  x_mask)
    w_vals_3 = tl.load(vals + x_index * 4 + 3,  x_mask)
    k_idx = tl.load(omap0 + x_index * 4, x_mask)
    out = tl.zeros([XBLOCK, YBLOCK], dtype=tl.float32)

    # Now we need to load the input tensors
    input0_loaded = tl.load(input0 + i_idx_0 + y_index * y_stride, x_mask & y_mask)
    input1_loaded = tl.load(input1 + j_idx_0 + y_index * y_stride, x_mask & y_mask)
    out0 = input0_loaded * input1_loaded * w_vals_0
    #tl.store(debug_out + y_index * xnumel * 4 + x_index * 4, input0_loaded * input1_loaded * w_vals_0, x_mask & y_mask)
    # repeat for the other 3 indices
    input0_loaded = tl.load(input0 + i_idx_1 + y_index * y_stride, x_mask & y_mask)
    input1_loaded = tl.load(input1 + j_idx_1 + y_index * y_stride, x_mask & y_mask)
    out1 = out0 + input0_loaded * input1_loaded * w_vals_1
    #tl.store(debug_out + y_index * xnumel * 4 + x_index * 4 + 1, input0_loaded * input1_loaded * w_vals_1, x_mask & y_mask)
    input0_loaded = tl.load(input0 + i_idx_2 + y_index * y_stride, x_mask & y_mask)
    input1_loaded = tl.load(input1 + j_idx_2 + y_index * y_stride, x_mask & y_mask)
    out2 = out1 + input0_loaded * input1_loaded * w_vals_2
    #tl.store(debug_out + y_index * xnumel * 4 + x_index * 4 + 2, input0_loaded * input1_loaded * w_vals_2, x_mask & y_mask)
    input0_loaded = tl.load(input0 + i_idx_3 + y_index * y_stride, x_mask & y_mask)
    input1_loaded = tl.load(input1 + j_idx_3 + y_index * y_stride, x_mask & y_mask)
    out3 = out2 + input0_loaded * input1_loaded * w_vals_3
    #tl.store(debug_out + y_index * xnumel + x_index, out3, x_mask & y_mask)
    #tl.store(debug_out + x_index, k_idx, x_mask)
    
    tl.atomic_add(output + (tl.broadcast_to(k_idx + l3_dim * y_index, [XBLOCK, YBLOCK])), out3, x_mask & y_mask, sem = 'relaxed')


@triton.jit
def my_coo_triton(
    imap0, # First dimension of the input indices in the COO format. Shape is [Po * Pi] (flattened)
    imap1, # Second dimension of the input indices in the COO format.
    omap0, # The output indices in the COO format. Shape is [Po].
           # BUT importantly, the elements in this array are spaced 4 elements apart in memory
           # because the previous kernel stored it that way.
    vals, # The values in the COO format. Shape is [Po * Pi]
    input0, # Input tensor 1. Shape is [B, L1]
    input1, # Input tensor 2. Shape is [B, L2]
    output, # Output tensor. Shape is [B, L3]
    #debug_out, # [Po, Pi]
    # Note that for our purposes L1, L2, and L3 are pre-defined constants.
    ynumel, # B
    xnumel, # Po
    W : tl.constexpr, # Pi
    YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # First define offsets and strides
    y_offset = tl.program_id(1) * YBLOCK
    x_offset = tl.program_id(0) * XBLOCK
    l1_dim = 36
    l2_dim = 36
    l3_dim = 1296
    y_stride = l1_dim
    x_stride = 1

    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    y_index = y_offset + tl.arange(0, YBLOCK)[None, :]
    x_mask = x_index < xnumel
    y_mask = y_index < ynumel

    out = tl.zeros([XBLOCK, YBLOCK], dtype=tl.float32)

    for i in range(W):
        curr_i_idx = tl.load(imap0 + x_index * W + i, x_mask)
        curr_j_idx = tl.load(imap1 + x_index * W + i, x_mask)
        curr_w_val = tl.load(vals + x_index * W + i, x_mask)
        
        input0_loaded = tl.load(input0 + curr_i_idx + y_index * y_stride, x_mask & y_mask)
        input1_loaded = tl.load(input1 + curr_j_idx + y_index * y_stride, x_mask & y_mask)
        out = out + input0_loaded * input1_loaded * curr_w_val

    k_idx = tl.load(omap0 + x_index * W, x_mask)
    tl.atomic_add(output + (tl.broadcast_to(k_idx + l3_dim * y_index, [XBLOCK, YBLOCK])), out, x_mask & y_mask, sem = 'relaxed')

