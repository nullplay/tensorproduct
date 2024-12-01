import torch
from e3nn import o3
import e3nn_jax as e3nn_jax
import numpy as np

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

class FunctionalTensorProductTorch:
    def __init__(self, irreps_in1, irreps_in2, irreps_out, instructions):
        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)
        self.instructions = [Instruction(**instr) for instr in instructions]

        # Precompute Clebsch-Gordan coefficients and set up paths
        self.cg_dict = {}
        for idx, instr in enumerate(self.instructions):
            mul_ir_in1 = self.irreps_in1[instr.i_in1]
            mul_ir_in2 = self.irreps_in2[instr.i_in2]
            mul_ir_out = self.irreps_out[instr.i_out]

            # Compute Clebsch-Gordan coefficients
            cg = e3nn_jax.clebsch_gordan(
                mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l
            )
            cg = torch.tensor(np.array(cg), dtype=torch.float32)
            self.cg_dict[idx] = cg * instr.path_weight  # Apply path weight

        # Initialize output buffers per output irrep index
        self.output_irreps = self.irreps_out

    def __call__(self, input1, input2, weights):
        outputs = [None] * len(self.output_irreps)
        weight_index = 0
        batch_shape = input1.shape[:-1]

        for idx, instr in enumerate(self.instructions):
            mul_ir_in1 = self.irreps_in1[instr.i_in1]
            mul_ir_in2 = self.irreps_in2[instr.i_in2]
            mul_ir_out = self.irreps_out[instr.i_out]

            # Extract inputs
            start1 = self.irreps_in1[:instr.i_in1].dim
            end1 = start1 + mul_ir_in1.dim
            x1 = input1[..., start1:end1]

            start2 = self.irreps_in2[:instr.i_in2].dim
            end2 = start2 + mul_ir_in2.dim
            x2 = input2[..., start2:end2]

            # Reshape inputs to separate multiplicities
            x1 = x1.view(*batch_shape, mul_ir_in1.mul, mul_ir_in1.ir.dim)
            x2 = x2.view(*batch_shape, mul_ir_in2.mul, mul_ir_in2.ir.dim)
            # Get Clebsch-Gordan coefficients
            cg = self.cg_dict[idx]  # [dim_in1, dim_in2, dim_out]
            w = weights[idx] # [mul_in1, mul_in2, mul_out]
            assert w.shape == instr.path_shape, f"Weight shape mismatch, expected {instr.path_shape}, got {w.shape}"
            # Compute tensor product
            tmp = torch.einsum('ijk,...ui,...vj,uvw->...wk', cg, x1, x2,w)

            # Reshape to [batch_shape..., mul_out, ir_dim]
            tmp = tmp.view(*batch_shape, mul_ir_out.mul, mul_ir_out.ir.dim)

            # Accumulate outputs per output irrep
            if outputs[instr.i_out] is None:
                outputs[instr.i_out] = tmp
            else:
                outputs[instr.i_out] += tmp

        # Concatenate outputs
        output_list = []
        for out_chunk, mul_ir in zip(outputs, self.output_irreps):
           #print(out_chunk, mul_ir)
           # print(out_chunk.reshape(*batch_shape, mul_ir.mul * mul_ir.ir.dim))

            if out_chunk is not None:
                out_chunk = out_chunk.reshape(*batch_shape, mul_ir.mul * mul_ir.ir.dim)
                output_list.append(out_chunk)

        output = torch.cat(output_list, dim=-1)
        return output


