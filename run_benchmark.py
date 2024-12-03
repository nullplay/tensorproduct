from ctypes import c_void_p, c_long
import torch
import numpy as np
import e3nn_jax as e3nn
import numpy as np
import jax
from jax_baseline_tp import tensor_product, benchmark_jax
import functools
from my_coo import my_coo
from my_coo_i_jk import pad_coo, my_coo_i_jk

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
  
  # Function to calculate avg nnz per axis
  def avg_nnz_per_axis(coo, axis):
      # Extract the desired axis
      indices = coo[:, axis]
      
      # Count occurrences of each unique index
      nnz_per_axis = torch.bincount(indices)
      
      # Compute the average
      avg_nnz = nnz_per_axis.float().mean()
      return avg_nnz

  avg_nnz_i = avg_nnz_per_axis(coo, axis=0)
  avg_nnz_j = avg_nnz_per_axis(coo, axis=1)
  avg_nnz_k = avg_nnz_per_axis(coo, axis=2)

  print(f"Average nnz per i: {avg_nnz_i}")
  print(f"Average nnz per j: {avg_nnz_j}")
  print(f"Average nnz per k: {avg_nnz_k}")
  print(f"COO shape: {coo.shape}")
  return input1_e3nn, input1, input2_e3nn, input2, output, coo, coovalue

if __name__ == "__main__":

    Bs = [1e4]
    lmaxs = range(5, 6)
    channel = 1 
    
    for Batch in Bs:
        for lmax in lmaxs:
            Batch = ((int(Batch)+127)//128) * 128        
            input1_e3nn, input1, input2_e3nn, input2, output, coo, coovalue = generate_e3nn_buffers(lmax, channel, Batch)
            
            LMax = input1.shape[-1]
            KMax = output.shape[-1]
            UMax = channel
            #Padding
            wsize = 4 
            i_,j_,k_,val_ = pad_coo(coo[:,2],coo[:,0],coo[:,1],coovalue,wsize)
            output2 = torch.zeros_like(output)

            result = my_coo(coo, coovalue, input1, input2, output, Batch, channel, coo.shape[0])
            result2 = my_coo_i_jk(j_, k_, i_, val_, input1, input2, output2, Batch, channel, i_.shape[0], wsize)
           
            ref = tensor_product(input1_e3nn, input2_e3nn, sorted=False, regroup_output=False).array

            print("correctness : torch vs e3nn")
            np.testing.assert_allclose(result.cpu().numpy(), ref, atol=1e-2)
            np.testing.assert_allclose(result2.cpu().numpy(), ref, atol=1e-2)
            print("PASS\n")

            
            tensor_product_jax = jax.jit(functools.partial(tensor_product, sorted=False, regroup_output=False))

            from torch._inductor.utils import print_performance
            print(f"ours - batch {Batch} lmax {lmax} channel {channel} {print_performance(lambda : my_coo(coo, coovalue, input1, input2, output, Batch, channel, coo.shape[0]))*1000:.3f} ms")
            print(f"ours2 - batch {Batch} lmax {lmax} channel {channel} {print_performance(lambda :  my_coo_i_jk(j_, k_, i_, val_, input1, input2, output2, Batch, channel, i_.shape[0], wsize))*1000:.3f} ms")
            
            print(f"jax - batch {Batch} lmax {lmax} channel {channel} {benchmark_jax(tensor_product_jax, input1_e3nn, input2_e3nn):.3f} ms")
