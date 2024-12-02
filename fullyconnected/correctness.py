import jax
import jax.numpy as jnp
import e3nn_jax as e3nn_jax
from e3nn_jax.legacy import FunctionalFullyConnectedTensorProduct
import numpy as np
import torch
from utils import  generate_cg_widx, convert_to_coo, prepare_inputs, compute_output_shape, reconstruct_from_2d
from mycoo import my_coo

# Set the random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
key = jax.random.PRNGKey(seed)

# Define irreps
B=16 
U=24
V=32
W=16
irreps_in = f"{U}x0e + {U}x1o + {U}x2e "  
irreps_sh = f"{V}x0e + {V}x1o + {V}x2e "  
irreps_out = f"{W}x0e + {W}x1o + {W}x2e + {W}x3o" 

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

# JAX Computation
ws_jax = [jnp.array(w) for w in ws]
input1_e3nn = e3nn_jax.normal(tp_jax.irreps_in1, jax.random.PRNGKey(0), (B,))
input2_e3nn = e3nn_jax.normal(tp_jax.irreps_in2, jax.random.PRNGKey(0), (B,))

def tensor_product_single(ws, x1, x2):
    return tp_jax.left_right(ws, x1, x2)

tensor_product_batched = jax.vmap(
    tensor_product_single,
    in_axes=(None, 0, 0),  # ws is shared, x1 and x2 are batched
    out_axes=0             # Output is batched along the first dimension
)

b_jax = tensor_product_batched(ws_jax, input1_e3nn, input2_e3nn).array


# PyTorch Computation
ws_torch = [torch.from_numpy(w) for w in ws]
x1_torch = torch.from_numpy(np.array(input1_e3nn.array)) # B x UL
x2_torch = torch.from_numpy(np.array(input2_e3nn.array)) # B x VL


# Setting up tensors
mask, widx = generate_cg_widx(irreps_in, irreps_sh, irreps_out, instructions)
coovalue, coo = convert_to_coo(mask)
x1_torch, x2_torch, ws_torch = prepare_inputs(x1_torch, x2_torch, irreps_in, irreps_sh, irreps_out, ws_torch, B, U, V)
output = torch.zeros((B, W, compute_output_shape(irreps_out, W)), device="cuda", dtype=torch.float32)

# Main compute
output = my_coo(coo, widx, coovalue, x1_torch, x2_torch, ws_torch, output, B, U, V, W)

# Epilogue
out_torch_np = output.cpu().detach().numpy()
tensor_1d = reconstruct_from_2d(out_torch_np, irreps_out, B, W)


if np.testing.assert_allclose(tensor_1d, np.array(b_jax), atol=1e-2) is None :
    print("Outputs are close! Correctness check passed.")
