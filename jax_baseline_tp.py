from typing import List, Optional

import jax
import jax.numpy as jnp

from e3nn_jax._src.tensor_products import _prepare_inputs, _validate_filter_ir_out
from e3nn_jax._src.utils.decorators import overload_for_irreps_without_array
import e3nn_jax as e3nn
import time

def benchmark_jax(func, *args, warmup=10, repeat=100):
    for _ in range(warmup):
        result = func(*args)
        if isinstance(result, e3nn.IrrepsArray):
            result.array.block_until_ready()
        else:
            result.block_until_ready()

    start_time = time.time()
    for _ in range(repeat):
        result = func(*args)
        if isinstance(result, e3nn.IrrepsArray):
            result.array.block_until_ready()
        else:
            result.block_until_ready()
    end_time = time.time()

    return (end_time - start_time) / repeat * 1000  # Convert to milliseconds

import numpy as np
@overload_for_irreps_without_array((0, 1))
def tensor_product(
    input1: e3nn.IrrepsArray,
    input2: e3nn.IrrepsArray,
    *,
    filter_ir_out: Optional[List[e3nn.Irrep]] = None,
    irrep_normalization: Optional[str] = None,
    regroup_output: bool = True,
    sorted: bool = True,
) -> e3nn.IrrepsArray:
    input1, input2, leading_shape = _prepare_inputs(input1, input2)
    filter_ir_out = _validate_filter_ir_out(filter_ir_out)

    if irrep_normalization is None:
        irrep_normalization = e3nn.config("irrep_normalization")

    if regroup_output:
        input1 = input1.regroup()
        input2 = input2.regroup()

    input1_l_dims = [ir.dim for (m,ir) in input1.irreps]
    input2_l_dims = [ir.dim for (m,ir) in input2.irreps]
    
    irreps_out = []
    chunks = []
    for i1, ((mul_1, ir_1), x1) in enumerate(zip(input1.irreps, input1.chunks)):
        for i2, ((mul_2, ir_2), x2) in enumerate(zip(input2.irreps, input2.chunks)):
          #print("------")
            for ir_out in ir_1 * ir_2:


                if filter_ir_out is not None and ir_out not in filter_ir_out:
                    continue

                irreps_out.append((mul_1 * mul_2, ir_out))

                if x1 is not None and x2 is not None:
                    cg = e3nn.clebsch_gordan(ir_1.l, ir_2.l, ir_out.l)
                    if irrep_normalization == "component":
                        cg = cg * jnp.sqrt(ir_out.dim)
                    elif irrep_normalization == "norm":
                        cg = cg * jnp.sqrt(ir_1.dim * ir_2.dim)
                    elif irrep_normalization == "none":
                        pass
                    else:
                        raise ValueError(
                            f"irrep_normalization={irrep_normalization} not supported"
                        )
                    cg = cg.astype(x1.dtype)
                    chunk = jnp.einsum("...ui , ...vj , ijk -> ...uvk", x1, x2, cg)
                    chunk = jnp.reshape(
                        chunk, chunk.shape[:-3] + (mul_1 * mul_2, ir_out.dim)
                    )
                    
                    #print(f"mul1:{mul_1}, ir_1:{ir_1}, ir_1.l:{ir_1.l}, ir_1.dim:{ir_1.dim}, x1:{x1.shape}")
                    #print(f"mul2:{mul_2}, ir_2:{ir_2}, ir_2.l:{ir_2.l}, ir_2.dim:{ir_2.dim}, x2:{x2.shape}")
                    #print("real i : ", sum(input1_l_dims[:i1]) + np.arange(ir_1.dim))
                    #print("real j : ", sum(input2_l_dims[:i2]) + np.arange(ir_2.dim))
                    #print(f"{jnp.nonzero(cg)[0]+sum(input1_l_dims[:i1])} \n {jnp.nonzero(cg)[1]+sum(input2_l_dims[:i2])} \n {jnp.nonzero(cg)[2]}")
                    #print("x1[b,u,i] * x2[b,v,j] * cg[i,j,k] -> out[b,u,v,k]")
                    #print(x1.shape, x2.shape, cg.shape,  chunk.shape)
                    #jax.debug.print("Non-zero elements in CG: {nz}/{shape}", nz=jnp.count_nonzero(cg), shape=cg.shape[0]*cg.shape[1]*cg.shape[2])
                    #print()
                else:
                    chunk = None

                chunks.append(chunk)
    
    #print(len(chunks))
    #print(sum([chk.shape[2]*chk.shape[3] for chk in chunks ]))

    output = e3nn.from_chunks(irreps_out, chunks, leading_shape, input1.dtype)
    #print(output.shape)
    if sorted:
        output = output.sort()
    if regroup_output:
        output = output.regroup()
    #print(output.shape)
    return output
