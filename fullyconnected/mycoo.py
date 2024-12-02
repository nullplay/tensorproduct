import torch
##########
# main compute
##########

def my_coo(coo, widx, coovalue, Input1, Input2, Weight, output, B, U, V, W):
    """
    Performs the COO operation for sparse tensors.

    Args:
        coo: COO indices.
        widx: Weight indices.
        coovalue: COO values.
        Input1, Input2, Weight: Input tensors.
        output: Output tensor.
        B, U, V, W: Dimensions.

    Returns:
        Updated output tensor.
    """
    imap1 = coo[:, 0]
    imap2 = coo[:, 1]
    omap = coo[:, 2]
    Input1_selected = torch.index_select(Input1, 2, imap1).view(B, U, 1, 1, -1)
    Input2_selected = torch.index_select(Input2, 2, imap2).view(B, 1, V, 1, -1)
    Weight_selected = torch.index_select(Weight, 3, widx).view(1, U, V, W, -1)

    coovalue_expanded = coovalue.view(1, 1, 1, -1)
    product = coovalue_expanded * Input1_selected * Input2_selected * Weight_selected
    product = torch.sum(product, dim=(1, 2))
    output.index_add_(2, omap, product)
    return output


