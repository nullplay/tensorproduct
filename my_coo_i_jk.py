import torch
import numpy as np
# Input1_[b,c,p] = Input1[b,c,imap1[p]]
# Input2_[b,c,p] = Input2[b,c,imap2[p]]
# product[b,c,p] = W[p] * Input1_[b,c,p] * Input2_[b,c,p]
# Output[b,c,omap[p]] += product[b,c,p]

# Output[b,c,omap[po]] += W[po,pi] * Input1[b,c,imap1[po,pi]] * Input2[b,c,imap2[po,pi]]

# Padded COO
#def pad_coo(w, o, i, val, window_size):
#    w = np.asarray(w) #i
#    o = np.asarray(o) #j
#    i = np.asarray(i) #k
#    val = np.asarray(val) #val
#    sort_indices = np.lexsort((i,o,w))
#    w_sorted = w[sort_indices]
#    o_sorted = o[sort_indices]
#    i_sorted = i[sort_indices]
#    val_sorted = val[sort_indices]
#    unique_w, counts = np.unique(w_sorted, return_counts=True)
#    counts = counts.astype(int)
#    padded_counts = np.ceil(counts / window_size).astype(int) * window_size
#    positions = np.concatenate(([0], np.cumsum(padded_counts[:-1])))
#    w_padded = np.repeat(unique_w, padded_counts)
#    o_padded = np.full_like(w_padded, 0, dtype=o.dtype) # -1
#    i_padded = np.full_like(w_padded, 0, dtype=i.dtype) # -1
#    val_padded = np.full_like(w_padded, 0.0, dtype=val.dtype) # -1
#    positions_repeated = np.repeat(positions, counts)
#    group_offsets = np.repeat(np.arange(len(counts)), counts)
#    offsets_within_group = np.concatenate([
#        np.arange(count) for count in counts
#    ])
#    indices = positions_repeated + offsets_within_group
#    o_padded[indices] = o_sorted
#    i_padded[indices] = i_sorted
#    val_padded[indices] = val_sorted
#
#    res1 = torch.tensor(w_padded[::window_size], device="cuda")
#    res2 = torch.tensor(o_padded, device="cuda")
#    res3 = torch.tensor(i_padded, device="cuda")
#    res4 = torch.tensor(val_padded, device="cuda")
#    return res1, res2, res3, res4 
#
#import torch

def pad_coo(w, o, i, val, window_size):
    device = "cuda"
    w_max = w.max()
    o_max = o.max()
    i_max = i.max()
    o_multiplier = i_max + 1
    w_multiplier = (o_max + 1) * o_multiplier
    key = w * w_multiplier + o * o_multiplier + i

    # Perform the sort
    _, sort_indices = torch.sort(key)
    w_sorted = w[sort_indices]
    o_sorted = o[sort_indices]
    i_sorted = i[sort_indices]
    val_sorted = val[sort_indices]

    unique_w, counts = torch.unique_consecutive(w_sorted, return_counts=True)
    counts = counts.long()  # Ensure counts are integers
    padded_counts = ((counts.float() / window_size).ceil().long()) * window_size
    positions = torch.cat([torch.tensor([0], device=device), torch.cumsum(padded_counts[:-1], dim=0)])
    total_padded_length = padded_counts.sum()
    w_padded = unique_w.repeat_interleave(padded_counts)
    o_padded = torch.zeros(total_padded_length, dtype=o.dtype, device=device)
    i_padded = torch.zeros(total_padded_length, dtype=i.dtype, device=device)
    val_padded = torch.zeros(total_padded_length, dtype=val.dtype, device=device)
    total_counts = counts.sum()
    starts = torch.cumsum(counts, dim=0) - counts  # Start indices for each group in the sorted arrays
    starts_repeated = starts.repeat_interleave(counts)
    offsets_within_group = torch.arange(total_counts, device=device) - starts_repeated
    positions_repeated = positions.repeat_interleave(counts)
    indices = positions_repeated + offsets_within_group

    o_padded[indices] = o_sorted
    i_padded[indices] = i_sorted
    val_padded[indices] = val_sorted

    res1 = w_padded[::window_size]
    res2 = o_padded
    res3 = i_padded
    res4 = val_padded

    return res1, res2, res3, res4


@torch.compile(mode="max-autotune-no-cudagraphs")
def my_coo_i_jk(imap1, imap2, omap, W, Input1, Input2, output, B, C, Po, Pi):
  imap1_flat = imap1.reshape(-1)  # Shape: [Po*Pi]
  imap2_flat = imap2.reshape(-1)  # Shape: [Po*Pi]
  Input1_selected = torch.index_select(Input1, 2, imap1_flat)  # Shape: [B, C, Po*Pi]
  Input2_selected = torch.index_select(Input2, 2, imap2_flat)  # Shape: [B, C, Po*Pi]

  Input1_ = Input1_selected.view(B, C, Po, Pi)
  Input2_ = Input2_selected.view(B, C, Po, Pi)

  W_expanded = W.view(1, 1, Po, Pi)  # Shape: [1, 1, Po, Pi]
  #product[b,c,po] = w[po,pi] * input1[b,c,po,pi] * input2[b,c,po,pi]
  product_pi = W_expanded * Input1_ * Input2_  # Shape: [B, C, Po, Pi]
  product = product_pi.sum(dim=3)  # Sum over pi dimension, Shape: [B, C, Po]

  output.index_add_(2, omap, product)
  return output



