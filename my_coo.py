import torch
from torch._inductor.utils import fresh_inductor_cache

# Input1_[b,c,p] = Input1[b,c,imap1[p]]
# Input2_[b,c,p] = Input2[b,c,imap2[p]]
# product[b,c,p] = W[p] * Input1_[b,c,p] * Input2_[b,c,p]
# Output[b,c,omap[p]] += product[b,c,p]

#torch._inductor.config.max_autotune_pointwise=True
#torch._inductor.config.triton.prefer_nd_tiling=True

@fresh_inductor_cache()
@torch.compile(mode="max-autotune-no-cudagraphs")
def my_coo(coo, coovalue, Input1, Input2, output, B, C, P):
  imap1 = coo[:,0]
  imap2 = coo[:,1]
  omap = coo[:,2]
  # Gather the necessary slices using index_select
  Input1_selected = torch.index_select(Input1, 2, imap1)  # Shape: [B, C, P]
  Input2_selected = torch.index_select(Input2, 2, imap2)  # Shape: [B, C, P]

  # Reshape coovalue for broadcasting
  coovalue_expanded = coovalue.view(1, 1, -1)  # Shape: [1, 1, P]

  # Compute the intermediate product
  product = coovalue_expanded * Input1_selected * Input2_selected  # Shape: [B, C, P]

  # Scatter the product into the output tensor
  #omap_expanded = omap.expand(B, C, P)  # Shape: [B, C, P]
  #output.scatter_add_(2, omap_expanded, product)
  output.index_add_(2, omap, product)

  return output



