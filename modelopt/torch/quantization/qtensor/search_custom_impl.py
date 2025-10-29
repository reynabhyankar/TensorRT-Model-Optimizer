import torch 
import torch.nn.functional as F
import numpy as np

def reduce_block_padding(input: torch.Tensor, block_sizes: dict, pad_value: float = 0):
    """Padding the input using block-based reduction for each dimension.

    Args:
        input_tensor (torch.Tensor): The input tensor.
        block_sizes (dict): A dictionary specifying the block size for padding each dimension.
                            Example: `{-1: 128, -2: 128}` pads the input over 2D blocks.
    """
    with torch.no_grad():
        padded_tensor = input
        num_dims = padded_tensor.dim()

        # Process each specified dimension independently
        for dim, block in block_sizes.items():
            # Convert negative dimension to positive index
            pos_dim = dim if dim >= 0 else num_dims + dim

            # Calculate how many elements are missing along that dimension
            current_size = padded_tensor.size(pos_dim)
            remainder = current_size % block
            pad_amt = 0 if remainder == 0 else block - remainder

            if pad_amt > 0:
                # F.pad expects a pad tuple of length 2*num_dims.
                pad = [0] * (2 * num_dims)
                # For dimension pos_dim, the right padding is at index: (num_dims - 1 - pos_dim)*2 + 1.
                pad_index = (num_dims - 1 - pos_dim) * 2
                pad[pad_index + 1] = (
                    pad_amt  # Set padding on the right side of the target dimension
                )

                padded_tensor = F.pad(padded_tensor, pad, value=pad_value)

        return padded_tensor

def reduce_block_amax(input_tensor: torch.Tensor, block_sizes: dict):
    """Computes the amax of the input tensor using block-based reduction for each dimension.

    Args:
        input_tensor (torch.Tensor): The input tensor.
        block_sizes (dict): A dictionary specifying the block size for each dimension.
                            Example: `{-1: 128, -2: 128}` reduces over 2D blocks.

    Returns:
        torch.Tensor: The reduced tensor with amax computed per block.

    Example:
        Input Shape: [256, 512]
        Block Sizes: {-1: 128, -2: 128}
        Process:
            - Block along last dim → Shape [256, 4, 128]
            - Compute block-wise amax → Shape [256, 4]
            - Block along second-to-last dim → Shape [2, 128, 4]
            - Compute block-wise amax → Shape [2, 4]
    """
    with torch.no_grad():
        amax = input_tensor.clone()

        for dim, block_size in block_sizes.items():
            # Convert negative dimensions to positive
            dim = dim if dim >= 0 else len(amax.shape) + dim
            assert amax.shape[dim] % block_size == 0, (
                f"Tensor dimension {amax.shape[dim]}, {amax.shape[dim]} is not divisible by {block_size}"
            )

            # Compute new shape for blocking
            outer_dim = amax.shape[dim] // block_size
            new_shape = [
                *list(amax.shape[:dim]),
                outer_dim,
                block_size,
                *list(amax.shape[dim + 1 :]),
            ]

            # Reshape into blocks
            amax = amax.reshape(new_shape)

            # Reduce along the newly created block dimension
            # Shift by 1 because we added an extra dimension
            amax = reduce_amax(amax, dim + 1, keepdims=False, squeeze_scalar=False)

        return amax


@torch.no_grad()
def reduce_amax(input, axis=None, keepdims=True, squeeze_scalar=True):
    """Compute the absolute maximum value of a tensor.

    Reduces input_tensor along the dimensions given in axis. Unless keepdims is true,
    the rank of the tensor is reduced by 1 for each entry in axis. If keepdims is true,
    the reduced dimensions are retained with length 1.

    .. note::
        Gradient computation is disabled as this function is never meant learning reduces amax

    Args:
        input: Input tensor
        axis: The dimensions to reduce. None or int or tuple of ints. If None (the default),
            reduces all dimensions. Must be in the range [-rank(input_tensor), rank(input_tensor)).
        keepdims: A boolean. If true, retains reduced dimensions with length 1. Default True

    Returns:
        The reduced tensor.
    """
    # A memory-efficient implementation that avoids copying input tensor
    if axis is None:
        max_val = torch.max(input)
        min_val = torch.min(input)
        output = torch.maximum(torch.abs(max_val), torch.abs(min_val))
    else:
        if isinstance(axis, int):
            axis = (axis,)
        max_val = torch.amax(input, dim=axis, keepdim=keepdims)
        min_val = torch.amin(input, dim=axis, keepdim=keepdims)
        output = torch.maximum(torch.abs(max_val), torch.abs(min_val))
        if squeeze_scalar and output.numel() == 1:
            output.squeeze_()
    return output

def get_weights_scaling_factor(
    input: torch.Tensor,
    block_size: int,
    weights_scaling_factor_2: torch.Tensor | None = None,
    keep_high_precision: bool = False,
):
    """Returns quantized per block weight scaling factor."""
    if weights_scaling_factor_2 is None:
        weights_scaling_factor_2 = get_weights_scaling_factor_2(input)

    # Get per_block amax
    assert block_size != 0, "Block size is zero. Cannot return per_block amax for given input."

    assert input.shape[-1] % block_size == 0, (
        "Weight shape is not divisible for block size for block quantization."
    )

    # Get per block amax
    per_block_amax = reduce_block_amax(input, block_sizes={-1: block_size}).float()
    # Get per-block-scale
    per_block_scale = per_block_amax / (
        6.0 * weights_scaling_factor_2.to(per_block_amax.device)
    )
    # Set all zero values in scale to 1.0
    per_block_scale[per_block_scale == 0] = 1.0
    # Convert to torch.float8_e4m3fn
    if not keep_high_precision:
        per_block_scale = per_block_scale.to(torch.float8_e4m3fn)
    return per_block_scale, weights_scaling_factor_2

def get_weights_scaling_factor_2( input: torch.Tensor):
    """Returns per tensor weight scaling factor."""
    return reduce_amax(input).float() / (6.0 * 448.0)

e2m1_bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])
e2m1_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])

def _cast_fp4(weight: torch.Tensor):
    """Converts tensor to uint4."""
    device = weight.device

    # Extract sign and compute absolute values in one pass
    sign_bit = (weight < 0).to(torch.uint8)
    weight_abs = weight.abs_()

    # Get bounds and compute ordinal values
    bounds = e2m1_bounds.to(device)
    ord = torch.searchsorted(bounds, weight_abs, out_int32=True).to(torch.uint8)

    # Efficiently check for rounding at odd-indexed bounds [0.75, 1.75, 2.5]
    # Only need to check bounds at indices 1, 3, 5
    odd_bounds = bounds[[1, 3, 5]]  # [0.75, 1.75, 2.5]
    equals_odd_bounds = torch.any(weight_abs.unsqueeze(-1) == odd_bounds, dim=-1).to(
        torch.uint8
    )

    # Combine sign, ordinal, and rounding adjustment
    return (sign_bit << 3) + ord + equals_odd_bounds

def quantize(
    input: torch.Tensor,
    block_size: int,
    weights_scaling_factor: torch.Tensor | None = None,
    weights_scaling_factor_2: torch.Tensor | None = None,
    keep_high_precision: bool = False,
    try_tensorrt: bool = False,
):
    """Converting a tensor to a quantized format based on NVFP4 quantization.

    Args:
        input (torch.Tensor): The input tensor to be quantized.
        block_size (int): The size of each block for quantization.
        weights_scaling_factor (torch.Tensor): The scaling factor for the weights.
        weights_scaling_factor_2 (torch.Tensor): The scaling factor for the weights.
        keep_high_precision (bool): Whether to keep output scales at high precision.

    Returns:
    tuple: Contains quantized data, quantized per block scaling factor, and per tensor scaling factor.
    """
    # Get original input shape
    input_shape = input.shape
    input_dtype = input.dtype

    # pad the input if needed
    input = reduce_block_padding(input, block_sizes={-1: block_size})

    if weights_scaling_factor_2 is None:
        weights_scaling_factor_2 = get_weights_scaling_factor_2(input)

    if weights_scaling_factor is None:
        weights_scaling_factor, _ = get_weights_scaling_factor(
            input, block_size, weights_scaling_factor_2
        )

    # Reshape the weight and scale factors
    original_shape = input.shape
    input = input.view((*tuple(input.shape[:-1]), -1, block_size))

    # Scale weights
    scaled_weight = input / (
        (weights_scaling_factor.to(torch.float32) * weights_scaling_factor_2).unsqueeze(-1)
    )

    # Reshape weights to original
    scaled_weight = scaled_weight.view(original_shape)

    if keep_high_precision:
        return scaled_weight
    # Cast weights to fp4
    q_weight = _cast_fp4(scaled_weight)
    # Pack weights
    packed_weight = (q_weight[..., 1::2] << 4) | q_weight[..., 0::2]
    return (
        input_shape, input_dtype, packed_weight,
        weights_scaling_factor,
        weights_scaling_factor_2,
    )

def _unpack_tensor(input: torch.Tensor, dtype: torch.dtype):
            # Initialize storage for unpacked tensor
            unpacked_shape = list(input.shape)
            unpacked_shape[-1] = unpacked_shape[-1] * 2
            unpacked = torch.empty(unpacked_shape, dtype=dtype, device=input.device)

            unpacked[..., 1::2] = input >> 4
            unpacked[..., 0::2] = input & 0x0F

            unpacked = unpacked.reshape(-1)
            unpacked = e2m1_values.to(input.device)[unpacked.long()]

            return unpacked.reshape(unpacked_shape)

def quantize_search(
    input: torch.Tensor,
    block_size: int,
    weights_scaling_factor: torch.Tensor | None = None,
    weights_scaling_factor_2: torch.Tensor | None = None,
    keep_high_precision: bool = False,
    search_range: tuple[int, int] = (-2, 6),
):
    """
    Converting a tensor to a quantized format using NVFP4 with SCALE SEARCH.
    
    This version searches for optimal per-block scales instead of using greedy selection.

    Args:
        input (torch.Tensor): The input tensor to be quantized.
        block_size (int): The size of each block for quantization (typically 16 for NVFP4).
        weights_scaling_factor (torch.Tensor): The per-block scaling factors (FP8). 
            If None, will be computed using search.
        weights_scaling_factor_2 (torch.Tensor): The global scaling factor. 
            If None, computed as max(abs(input)).
        keep_high_precision (bool): Whether to return pre-FP4 scaled values.
        search_range (tuple): Range of FP8 scale offsets to search (min_offset, max_offset).

    Returns:
        tuple: Contains quantized data, per-block scaling factors, and global scaling factor.
    """
    # Get original input shape
    input_shape = input.shape
    input_dtype = input.dtype

    # Pad the input if needed
    input = reduce_block_padding(input, block_sizes={-1: block_size})

    # Compute global scale (weights_scaling_factor_2) if not provided
    if weights_scaling_factor_2 is None:
        weights_scaling_factor_2 = get_weights_scaling_factor_2(input)
    
    global_scale = weights_scaling_factor_2.item() if isinstance(weights_scaling_factor_2, torch.Tensor) else weights_scaling_factor_2

    # Compute per-block scales using PARALLEL SEARCH if not provided
    if weights_scaling_factor is None:
        blocks = input.view(-1, block_size)  # [num_blocks, block_size]
        num_blocks = blocks.shape[0]
        
        # Greedy baseline for ALL blocks at once (using reduce_amax)
        block_max = reduce_amax(blocks, axis=1, keepdims=False)  # [num_blocks]
        greedy_scales = block_max / (6.0 * global_scale)  # Match get_weights_scaling_factor formula
        greedy_fp8 = greedy_scales.to(torch.float8_e4m3fn).view(torch.uint8)  # [num_blocks]
        
        # Initialize: best scales and losses for all blocks
        best_fp8 = greedy_fp8.clone()
        best_losses = torch.full((num_blocks,), float('inf'), device=input.device)
        
        # Search in parallel for ALL blocks
        for offset in range(search_range[0], search_range[1] + 1):
            # Candidate scales for all blocks
            candidate_fp8 = greedy_fp8 + offset
            
            # Mask: valid candidates (within FP8 range)
            valid = (candidate_fp8 >= 0) & (candidate_fp8 <= 255)
            if not valid.any():
                continue
            
            # Convert FP8 → float for all candidates
            scales = candidate_fp8.view(torch.uint8).view(torch.float8_e4m3fn).float()  # [num_blocks]
            
            # Compute divisor for scaling: scale * weights_scaling_factor_2
            divisor = (scales * weights_scaling_factor_2).unsqueeze(1)  # [num_blocks, 1]
            
            # Scale blocks (same as main quantize logic)
            scaled_blocks = blocks / divisor
            
            # Quantize using existing _cast_fp4
            fp4_indices = _cast_fp4(scaled_blocks)  # [num_blocks, block_size] - unpacked
            
            # Pack into bytes (2 FP4 values per byte)
            fp4_packed = (fp4_indices[..., 1::2] << 4) | fp4_indices[..., 0::2]
            
            # Dequantize using _unpack_tensor
            reconstructed = _unpack_tensor(fp4_packed, torch.float32) * divisor
            
            # Compute MSE per block
            losses = ((blocks - reconstructed) ** 2).sum(dim=1)  # [num_blocks]
    
            # Update best scales where loss improved AND candidate is valid
            improved = valid & (losses < best_losses)
            best_losses = torch.where(improved, losses, best_losses)
            best_fp8 = torch.where(improved, candidate_fp8, best_fp8)
        
        # Convert to FP8 tensor format
        weights_scaling_factor = best_fp8.view(torch.uint8).view(
            torch.float8_e4m3fn
        ).reshape(*input.shape[:-1], -1)
    
    # Now proceed with quantization using the computed (searched) scales
    # Reshape to expose blocks
    original_shape = input.shape
    input = input.view((*tuple(input.shape[:-1]), -1, block_size))

    # Scale weights using searched scales
    scaled_weight = input / (
        (weights_scaling_factor.to(torch.float32) * weights_scaling_factor_2).unsqueeze(-1)
    )

    # Reshape back to original
    scaled_weight = scaled_weight.view(original_shape)

    if keep_high_precision:
        return scaled_weight
    
    # Quantize using existing _cast_fp4
    q_weight = _cast_fp4(scaled_weight)
    
    # Pack into nibbles: 2 FP4 values per byte
    packed_weight = (q_weight[..., 1::2] << 4) | q_weight[..., 0::2]
    
    return (
        input_shape, 
        input_dtype, 
        packed_weight,
        weights_scaling_factor,
        weights_scaling_factor_2,
    )
