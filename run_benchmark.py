from ctypes import c_void_p, c_long
import torch
import numpy as np
import e3nn_jax as e3nn
import numpy as np
import jax
from jax_baseline_tp import tensor_product, benchmark_jax
import functools

def sparse_tensor_product(coo, coovalue, Input1, Input2, output, B, C, P):
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
  omap_expanded = omap.expand(B, C, P)  # Shape: [B, C, P]
  output.scatter_add_(2, omap_expanded, product)

  return output


def generate_cg(input1_irreps, input2_irreps):
    cg_matrices = []
    input1_l_dims = 2*np.array(input1_irreps.ls) + 1
    input2_l_dims = 2*np.array(input2_irreps.ls) + 1
    #print(len(input1_irreps))
    for i1, ((mul_1, ir_1), slice_1) in enumerate(zip(input1_irreps, input2_irreps.slices())):
        for i2, ((mul_2, ir_2), slice_2) in enumerate(zip(input1_irreps, input2_irreps.slices())):

            for ir_out in ir_1 * ir_2:
                cg = (e3nn.clebsch_gordan(ir_1.l, ir_2.l, ir_out.l) * np.sqrt(ir_out.dim))
                pad_width = [
                    (sum(input1_l_dims[:i1]), sum(input1_l_dims[i1+1:])),  # Pad first dimension
                    (sum(input2_l_dims[:i2]), sum(input2_l_dims[i2+1:])),  # Pad second dimension
                    (0, 0)  # Do not pad third dimension
                ]
                cg_pad = np.pad(cg, pad_width, mode='constant')
                
                nonzero_indices = np.nonzero(cg_pad)
                coo = np.vstack(nonzero_indices).T.astype(np.int32)
                #print(cg.shape, cg_pad.shape,coo)

                cg_matrices.append(cg_pad)
    mask = np.concatenate(cg_matrices, axis=-1)
    #print(mask.shape)
    return mask

def convert_to_coo(array):
  nonzero_indices = np.nonzero(array)
  coovalue = array[nonzero_indices].astype(np.float32)
  coo = np.vstack(nonzero_indices).T.astype(np.int64)
  return torch.tensor(coovalue,device="cuda"), torch.tensor(coo,device="cuda")

def generate_e3nn_buffers(lmax, channel, Batch):
  lmax1 = lmax
  lmax2 = lmax  
  input1_irreps = e3nn.Irreps("+".join(f"{l}e" for l in range(lmax1+1)))
  input2_irreps = e3nn.Irreps("+".join(f"{l}e" for l in range(lmax2+1)))

  input1_e3nn = e3nn.normal(input1_irreps, jax.random.PRNGKey(0), (Batch, channel))
  input2_e3nn = e3nn.normal(input2_irreps, jax.random.PRNGKey(1), (Batch, channel))

  input1 = torch.tensor(np.asarray(input1_e3nn.array), device="cuda")
  input2 = torch.tensor(np.asarray(input2_e3nn.array), device="cuda")
  
  KMax = e3nn.tensor_product(input1_e3nn.irreps, input2_e3nn.irreps).dim
  output = torch.zeros((Batch, channel, KMax), device="cuda")

  cg = generate_cg(input1_e3nn.irreps, input2_e3nn.irreps)
  coovalue, coo = convert_to_coo(cg)
  
  return input1_e3nn, input1, input2_e3nn, input2, output, coo, coovalue

if __name__ == "__main__":

    Bs = [1e4]
    lmaxs = range(2, 3)
    channel = 32
    
    for Batch in Bs:
        for lmax in lmaxs:
            Batch = ((int(Batch)+127)//128) * 128        
            input1_e3nn, input1, input2_e3nn, input2, output, coo, coovalue = generate_e3nn_buffers(lmax, channel, Batch)
            
            LMax = input1.shape[-1]
            KMax = output.shape[-1]
            UMax = channel
           
            input1 = input1.reshape(Batch, channel, -1)
            input2 = input2.reshape(Batch, channel, -1)
            result = sparse_tensor_product(coo, coovalue, input1, input2, output, Batch, channel, coo.shape[0])
           
            ref = tensor_product(input1_e3nn, input2_e3nn, sorted=False, regroup_output=False).array

            print("3. halide vs e3nn")
            np.testing.assert_allclose(result.cpu().numpy(), ref, atol=1e-2)
            print("PASS\n")

            
            tensor_product_jax = jax.jit(functools.partial(tensor_product, sorted=False, regroup_output=False))

            from torch._inductor.utils import print_performance
            print(f"ours - batch {Batch} lmax {lmax} channel {channel} {print_performance(lambda : sparse_tensor_product(coo, coovalue, input1, input2, output, Batch, channel, coo.shape[0]))*1000:.3f} ms")
            
            print(f"jax - batch {Batch} lmax {lmax} channel {channel} {benchmark_jax(tensor_product_jax, input1_e3nn, input2_e3nn):.3f} ms")
