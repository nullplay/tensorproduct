import jax
import jax.numpy as jnp
import e3nn_jax as e3nn_jax
from e3nn_jax.legacy import FunctionalFullyConnectedTensorProduct
import numpy as np
import torch
import torchimpl 
from utils import  generate_cg_widx, convert_to_coo

# Set the random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
key = jax.random.PRNGKey(seed)

# Define irreps
U=3
V=4
W=5
irreps_in = f"{U}x0e + {U}x1o + {U}x2e "  # 3 = u
irreps_sh = f"{V}x0e + {V}x1o + {V}x2e "  # 2 = v
irreps_out = f"{W}x0e + {W}x1o + {W}x2e + {W}x3o" # 5 = w

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


# PyTorch Computation
ws_torch = [torch.from_numpy(w) for w in ws]
x1_torch = torch.from_numpy(x1)
x2_torch = torch.from_numpy(x2)
tp_torch = torchimpl.FunctionalTensorProductTorch(irreps_in, irreps_sh, irreps_out, instructions)
a_torch = tp_torch(x1_torch, x2_torch, ws_torch)

# Compare outputs
a_jax_np = np.array(a_jax)
a_torch_np = a_torch.detach().numpy()
print(a_torch_np.shape)
# Check if outputs are close
if np.allclose(a_jax_np, a_torch_np, atol=1e-6):
    print("Outputs are close! Correctness check passed.")
else:
    print("Outputs are not close! There might be an error.")



mask, widx = generate_cg_widx(irreps_in, irreps_sh, irreps_out, instructions)
coovalue, coo = convert_to_coo(mask)

curri = 0
x1_2d = []
for ir in e3nn_jax.Irreps(irreps_in):
    i = ir.dim
    curr2d = x1_torch[curri:curri + i].reshape(U, -1)
    x1_2d.append(curr2d)
    curri += i
x1_2d = np.hstack(x1_2d)

currj = 0
x2_2d = []
for ir in e3nn_jax.Irreps(irreps_sh):
    j = ir.dim
    curr2d = x2_torch[currj:currj + j].reshape(V, -1)
    x2_2d.append(curr2d)
    currj += j
x2_2d = np.hstack(x2_2d)

x1_torch = torch.tensor(x1_2d, device="cuda")
x2_torch = torch.tensor(x2_2d, device="cuda")
ws_torch = torch.stack(ws_torch, dim=3).cuda()
widx_torch = torch.tensor(widx, device="cuda")

def my_coo(coo, widx, coovalue, Input1, Input2, Weight, output, U, V, W):
  imap1 = coo[:,0]
  imap2 = coo[:,1]
  omap = coo[:,2]
  Input1_selected = torch.index_select(Input1, 1, imap1).view(U,1,1,-1)  # Shape: [U, 1, 1, P]
  Input2_selected = torch.index_select(Input2, 1, imap2).view(1,V,1,-1)  # Shape: [1, V, 1, P]
  Weight_selected = torch.index_select(Weight, 3, widx)#.view(U,V,W,-1)  # Shape: [U, V, W, P]

  coovalue_expanded = coovalue.view(1, 1, 1, -1)  # Shape: [1, 1, 1, P]
  product = (coovalue_expanded * Input1_selected 
            * Input2_selected * Weight_selected)  # Shape: [U, V, W, P]
  product = torch.sum(product, dim=(0,1)) # Shape: [W,P]

  output.index_add_(1, omap, product) # Shape: [W,O]
  return output

K = sum([ir.dim for ir in e3nn_jax.Irreps(irreps_out)]) // W
output = torch.zeros((W,K), device="cuda", dtype=torch.float32)
my_coo(coo, widx_torch, coovalue, x1_torch, x2_torch, ws_torch, output, U, V, W)

out_torch_np = output.cpu().detach().numpy()
currk = 0
tensor_1d = np.array([])
for ir in e3nn_jax.Irreps(irreps_out):
  k = ir.dim//W
  column = out_torch_np[:, currk:currk+k].flatten()
  tensor_1d = np.concatenate((tensor_1d,column))
  currk += k

if np.allclose(tensor_1d, a_torch_np, atol=1e-6):
    print("Outputs are close! Correctness check passed.")

