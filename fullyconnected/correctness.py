# JAX imports
import jax
import jax.numpy as jnp
import e3nn_jax as e3nn_jax
from e3nn_jax.legacy import FunctionalFullyConnectedTensorProduct
import numpy as np

# PyTorch imports
import torch
from e3nn import o3

# Set the random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
key = jax.random.PRNGKey(seed)

# Define irreps
irreps_in = "3x0e + 3x1o"
irreps_sh = "2x0e + 2x1o"
irreps_out = "4x0e + 4x1o"

# Initialize the JAX tensor product
tp_jax = FunctionalFullyConnectedTensorProduct(
    e3nn_jax.Irreps(irreps_in),
    e3nn_jax.Irreps(irreps_sh),
    e3nn_jax.Irreps(irreps_out)
)

# Extract instructions
instructions = []
for ins in tp_jax.instructions:
    instruction = {
        'i_in1': ins.i_in1,
        'i_in2': ins.i_in2,
        'i_out': ins.i_out,
        'connection_mode': ins.connection_mode,
        'has_weight': ins.has_weight,
        'path_weight': float(ins.path_weight),
        'path_shape': ins.path_shape,
    }
    instructions.append(instruction)

# Generate random weights
ws = [np.random.normal(size=ins['path_shape']).astype(np.float32)
      for ins in instructions if ins['has_weight']]

# Generate random inputs
x1 = np.random.normal(size=(e3nn_jax.Irreps(irreps_in).dim,)).astype(np.float32)
x2 = np.random.normal(size=(e3nn_jax.Irreps(irreps_sh).dim,)).astype(np.float32)

# JAX Computation
ws_jax = [jnp.array(w) for w in ws]
x1_jax_irreps = e3nn_jax.IrrepsArray(tp_jax.irreps_in1, jnp.array(x1))
x2_jax_irreps = e3nn_jax.IrrepsArray(tp_jax.irreps_in2, jnp.array(x2))

a_jax = tp_jax.left_right(ws_jax, x1_jax_irreps, x2_jax_irreps, fused=False).array

# PyTorch Implementation

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

            w = weights[weight_index]
            weight_index += 1
            # w: [mul_in1, mul_in2, mul_out]
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
            if out_chunk is not None:
                out_chunk = out_chunk.reshape(*batch_shape, mul_ir.mul * mul_ir.ir.dim)
                output_list.append(out_chunk)

        output = torch.cat(output_list, dim=-1)
        return output




# PyTorch Computation
ws_torch = [torch.from_numpy(w) for w in ws]
x1_torch = torch.from_numpy(x1)
x2_torch = torch.from_numpy(x2)

tp_torch = FunctionalTensorProductTorch(irreps_in, irreps_sh, irreps_out, instructions)

a_torch = tp_torch(x1_torch, x2_torch, ws_torch)

# Compare outputs
a_jax_np = np.array(a_jax)
a_torch_np = a_torch.detach().numpy()

difference = np.abs(a_jax_np - a_torch_np)
max_diff = np.max(difference)
print("Max difference between JAX and PyTorch outputs:", max_diff)

# Check if outputs are close
if np.allclose(a_jax_np, a_torch_np, atol=1e-6):
    print("Outputs are close! Correctness check passed.")
else:
    print("Outputs are not close! There might be an error.")

