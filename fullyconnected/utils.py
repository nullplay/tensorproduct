import torch
from e3nn import o3
import e3nn_jax 
import numpy as np
import jax
import functools
class Instruction:
    """Defines an instruction for the tensor product."""
    def __init__(self, i_in1, i_in2, i_out, connection_mode, has_weight, path_weight=1.0, path_shape=None):
        self.i_in1 = i_in1
        self.i_in2 = i_in2
        self.i_out = i_out
        self.connection_mode = connection_mode
        self.has_weight = has_weight
        self.path_weight = path_weight
        self.path_shape = path_shape  # Include path_shape in the constructor


def generate_cg_widx(irreps_in1, irreps_in2, irreps_out, instructions):
    irreps_in1 = o3.Irreps(irreps_in1)
    irreps_in2 = o3.Irreps(irreps_in2)
    irreps_out = o3.Irreps(irreps_out)
    instructions = [Instruction(**instr) for instr in instructions]
  
    in1_ls = np.array([ir1.ir.l for ir1 in irreps_in1])
    in2_ls = np.array([ir2.ir.l for ir2 in irreps_in2])
    out_ls = np.array([ir3.ir.l for ir3 in irreps_out])
    input1_l_dims = 2*in1_ls + 1
    input2_l_dims = 2*in2_ls + 1
    output_l_dims = 2*out_ls + 1
    cg_matrices = np.zeros(
        (sum(input1_l_dims), sum(input2_l_dims), sum(output_l_dims)),
        dtype = np.float32
    )
    w_matrices = np.zeros(
        (sum(input1_l_dims), sum(input2_l_dims), sum(output_l_dims)),
        dtype = np.int64
    )
    for idx, instr in enumerate(instructions):
        i1 = instr.i_in1
        i2 = instr.i_in2
        i_out = instr.i_out
        mul_ir_in1 = irreps_in1[i1]
        mul_ir_in2 = irreps_in2[i2]
        mul_ir_out = irreps_out[i_out]

        cg = e3nn_jax.clebsch_gordan(
            mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l
        ) * instr.path_weight
        pad_width = [
            (sum(input1_l_dims[:i1]), sum(input1_l_dims[i1+1:])),  
            (sum(input2_l_dims[:i2]), sum(input2_l_dims[i2+1:])),  
            (sum(output_l_dims[:i_out]), sum(output_l_dims[i_out+1:])), 
        ]
        cg_pad = np.pad(cg, pad_width, mode='constant')
        cg_matrices += cg_pad
        
        w_pad = np.pad(np.full_like(cg, idx, dtype=np.int64), 
            pad_width, mode='constant')
        w_matrices += w_pad

    mask = cg_matrices 
    weightindices = w_matrices[np.nonzero(mask)]
    return mask, weightindices


def convert_to_coo(array):
  nonzero_indices = np.nonzero(array)
  coovalue = array[nonzero_indices].astype(np.float32)
  coo = np.vstack(nonzero_indices).T.astype(np.int64)
  return torch.tensor(coovalue,device="cuda"), torch.tensor(coo,device="cuda")


