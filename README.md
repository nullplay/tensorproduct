# 6.S894 final project

### What is e3nn and its Tensor Product?

e3nn is a Python library designed for building and accelerating neural networks that are equivariant to 3D rotations, translations, and other symmetries. At its core, it uses irreducible representations of the 3D rotation group (SO(3)) to process geometric data such as atomic structures, molecules, and point clouds. The **tensor product** in e3nn is a fundamental operation that combines features in these irreducible representations, allowing information to flow between different rotational symmetries while preserving equivariance.

### How Does the Tensor Product Work?

The tensor product operation combines input tensors `Input1[j, u]` and `Input2[k, v]` with a weight tensor `weight[i, j, k]` to produce an output `Out[i, u, v]`. The indices `(j, u)` and `(k, v)` correspond to structured, often ragged, representations of input data, while the weight tensor specifies sparse interactions. Mathematically, this involves an einsum-like operation: 

$$Out[i, u, v] = \sum_{j, k} Input1[j, u] \cdot Input2[k, v] \cdot weight[i, j, k]$$

The structured sparsity and raggedness stem from the nature of the irreducible representations, where different angular momentum states (e.g., \(j, k\)) dictate the ranges of \(u\) and \(v\), respectively. This hierarchical and sparse interaction pattern ensures the operation remains computationally efficient yet mathematically rigorous.

### Example of Tensor Product in Action

Consider a tensor product where:

- `Input1[j, u]` represents features for angular momentum states $j$ with the following ranges for $u$:  
  $j=0 \rightarrow u=0:16$  
  $j=1 \rightarrow u=0:256$  
  $j=2 \rightarrow u=0:32$

- `Input2[k, v]` represents features for angular momentum states $k$ with the following ranges for $v$:  
  $k=0 \rightarrow v=0:64$  
  $k=1 \rightarrow v=0:32$  
  $k=2 \rightarrow v=0:32$

If the weight tensor `weight[i, j, k]` specifies valid pairs like:  
- $(i=0, j=0, k=0)$  
- $(i=0, j=1, k=2)$

The contributions to $Out[0, u, v]$ are computed as follows:

1. For $(j=0, k=0)$:
   - Ranges: $u = 0:16$, $v = 0:64$  
   - Computation: $Out[0, 0:16, 0:64] += Input1[0, 0:16] \cdot Input2[0, 0:64] \cdot weight[0, 0, 0]$

2. For $(j=1, k=2)$:
   - Ranges: $u = 0:256$, $v = 0:32$  
   - Computation: $Out[0, 0:256, 0:32] += Input1[1, 0:256] \cdot Input2[2, 0:32] \cdot weight[0, 1, 2]$

The resulting range for $Out[0, u, v]$ spans the maximum ranges for $u$ and $v$:  
$$u = 0:256, \quad v = 0:64$$

For each $i$, $Out[i, u, v]$ forms a ragged tensor, where the size of the dense $(u, v)$ block depends on the sparse indices $(j, k)$ specified by `weight[i, j, k]`. 

This example highlights how the tensor product efficiently merges sparse, structured, and ragged data into a unified representation while maintaining computational efficiency.

### Proposal Objective

Accelerating this kernel involves leveraging the sparse structure of `weight[i, j, k]` and the ragged patterns of `Input1` and `Input2`. Techniques like structured sparsity optimization, custom memory layouts for ragged tensors, and GPU-friendly parallelization can significantly reduce redundant computations and improve overall throughput, making the tensor product more efficient for large-scale geometric data processing.


