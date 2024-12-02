import jax
import jax.numpy as jnp
import e3nn_jax as e3nn_jax
from e3nn_jax.legacy import FunctionalFullyConnectedTensorProduct
import numpy as np
import torch
from utils import  generate_cg_widx, convert_to_coo

# Set the random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
key = jax.random.PRNGKey(seed)

# Define irreps
U=32
V=32
W=32
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
B = 16 
input1_e3nn = e3nn_jax.normal(tp_jax.irreps_in1, jax.random.PRNGKey(0), (B,))
input2_e3nn = e3nn_jax.normal(tp_jax.irreps_in1, jax.random.PRNGKey(0), (B,))

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
x2_torch = torch.from_numpy(np.array(input1_e3nn.array)) # B x VL



mask, widx = generate_cg_widx(irreps_in, irreps_sh, irreps_out, instructions)
coovalue, coo = convert_to_coo(mask)

curri = 0
x1_2d = []
for ir in e3nn_jax.Irreps(irreps_in):
    i = ir.dim
    curr2d = x1_torch[:, curri:curri + i].reshape(B, U, -1)  
    x1_2d.append(curr2d)
    curri += i
x1_2d = np.concatenate(x1_2d, axis=-1)  # Concatenate along the last axis (-1)

currj = 0
x2_2d = []
for ir in e3nn_jax.Irreps(irreps_sh):
    j = ir.dim
    curr2d = x2_torch[:, currj:currj + j].reshape(B, V, -1)
    x2_2d.append(curr2d)
    currj += j
x2_2d = np.concatenate(x2_2d, axis=-1)  

x1_torch = torch.tensor(x1_2d, device="cuda") # B U L
x2_torch = torch.tensor(x2_2d, device="cuda") # B V L
ws_torch = torch.stack(ws_torch, dim=3).cuda()
widx_torch = torch.tensor(widx, device="cuda")

def my_coo(coo, widx, coovalue, Input1, Input2, Weight, output, B, U, V, W):
  imap1 = coo[:,0]
  imap2 = coo[:,1]
  omap = coo[:,2]
  Input1_selected = torch.index_select(Input1, 2, imap1).view(B,U,1,1,-1)  # Shape: [B, U, 1, 1, P]
  Input2_selected = torch.index_select(Input2, 2, imap2).view(B,1,V,1,-1)  # Shape: [B, 1, V, 1, P]
  Weight_selected = torch.index_select(Weight, 3, widx).view(1,U,V,W,-1)  # Shape: [B, U, V, W, P]

  coovalue_expanded = coovalue.view(1, 1, 1, -1)  # Shape: [1, 1, 1, 1, P]
  product = (coovalue_expanded * Input1_selected 
            * Input2_selected * Weight_selected)  # Shape: [B, U, V, W, P]
  product = torch.sum(product, dim=(1,2)) # Shape: [B,W,P]

  output.index_add_(2, omap, product) # Shape: [B,W,O]
  return output

K = sum([ir.dim for ir in e3nn_jax.Irreps(irreps_out)]) // W
output = torch.zeros((B,W,K), device="cuda", dtype=torch.float32)
my_coo(coo, widx_torch, coovalue, x1_torch, x2_torch, ws_torch, output, B, U, V, W)

out_torch_np = output.cpu().detach().numpy()

print(out_torch_np.shape)


currk = 0
tensor_1d = []
for ir in e3nn_jax.Irreps(irreps_out):
    k = ir.dim // U  # The number of columns per irreps
    column = out_torch_np[:, :, currk:currk + k].reshape(B, -1)  # Flatten within batch
    tensor_1d.append(column)
    currk += k
tensor_1d = np.concatenate(tensor_1d, axis=1)  # Concatenate along the second dimension


if np.testing.assert_allclose(tensor_1d, np.array(b_jax), atol=1e-2) is None :
    print("Outputs are close! Correctness check passed.")
