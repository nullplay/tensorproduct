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
    return mask, torch.tensor(weightindices, device="cuda")


def convert_to_coo(array):
  nonzero_indices = np.nonzero(array)
  coovalue = array[nonzero_indices].astype(np.float32)
  coo = np.vstack(nonzero_indices).T.astype(np.int64)
  return torch.tensor(coovalue,device="cuda"), torch.tensor(coo,device="cuda")



############
# precompute
############

def split_into_2d(tensor, irreps, B, M):
    """
    Splits a 1D tensor into 2D slices according to irreps dimensions.

    Args:
        tensor: Input tensor of shape (B, total_length).
        irreps: List of irreducible representations.
        B: Batch size.
        M: Target second dimension of the reshaped slices.

    Returns:
        A 2D tensor of shape (B, M, -1).
    """
    curr = 0
    slices = []
    for ir in e3nn_jax.Irreps(irreps):
        dim = ir.dim
        slices.append(tensor[:, curr:curr + dim].reshape(B, M, -1))
        curr += dim
    return np.concatenate(slices, axis=-1)


def prepare_inputs(x1_torch, x2_torch, irreps_in, irreps_sh, irreps_out, ws_torch, B, U, V):
    """
    Prepares inputs by splitting tensors into 2D slices.

    Args:
        x1_torch: Input tensor 1 (B, length).
        x2_torch: Input tensor 2 (B, length).
        irreps_in: Irreps for x1_torch.
        irreps_sh: Irreps for x2_torch.
        irreps_out: Output irreps.
        ws_torch: Weight tensor.
        B: Batch size.
        U, V: Target reshape dimensions for x1 and x2.

    Returns:
        Prepared tensors x1_2d, x2_2d, ws_torch.
    """
    x1_2d = split_into_2d(x1_torch, irreps_in, B, U)
    x2_2d = split_into_2d(x2_torch, irreps_sh, B, V)

    x1_torch = torch.tensor(x1_2d, device="cuda")
    x2_torch = torch.tensor(x2_2d, device="cuda")
    ws_torch = torch.stack(ws_torch, dim=3).cuda()
    return x1_torch, x2_torch, ws_torch


def compute_output_shape(irreps_out, W):
    """
    Computes the output tensor shape based on irreps.

    Args:
        irreps_out: List of irreducible representations.
        W: Output dimension.

    Returns:
        Total output length.
    """
    return sum(ir.dim for ir in e3nn_jax.Irreps(irreps_out)) // W



##########
# epilogue
##########

def reconstruct_from_2d(tensor_2d, irreps, B, W):
    """
    Reconstructs a 1D tensor from a 2D tensor based on irreps.

    Args:
        tensor_2d: Input tensor of shape (B, W, K).
        irreps: List of irreducible representations.
        U: Reshape dimension.

    Returns:
        A 1D tensor of shape (B, total_length).
    """
    curr = 0
    tensor_1d = []
    for ir in e3nn_jax.Irreps(irreps):
        k = ir.dim // W
        column = tensor_2d[:, :, curr:curr + k].reshape(B, -1)
        tensor_1d.append(column)
        curr += k
    return np.concatenate(tensor_1d, axis=1)
