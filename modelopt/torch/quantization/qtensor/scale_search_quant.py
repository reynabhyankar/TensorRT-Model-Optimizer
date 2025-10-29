import torch
import triton
import triton.language as tl

@triton.jit
def f32_as_u32(x: tl.tensor):
    return x.to(tl.uint32, bitcast=True)

@triton.jit
def u32_as_f32(x: tl.tensor):
    return x.to(tl.float32, bitcast=True)

@triton.jit
def f32_to_f4_single(x: tl.tensor):
    SCALE: tl.constexpr = 1.1754943508222875e-38
    x = tl.clamp(x, -6.0, 6.0) * SCALE
    ux = f32_as_u32(x) + (1 << 21)
    ux = ((ux >> 22) & 7) | ((ux >> 28) & 8)
    return ux

@triton.jit
def f4_to_f32_single(x: tl.tensor):
    x = x.to(tl.uint32, bitcast=True)
    ISCALE: tl.constexpr = 8.507059173023462e+37
    ux = x & 15
    ux = ((ux & 7) << 22) | ((ux & 8) << 28)
    return u32_as_f32(ux) * ISCALE

@triton.jit
def pack_quantized_single(x, y):
    return x | (y << 4)

@triton.jit
def quantize_to_fp4_packed_search(x: tl.tensor, BLOCK_SIZE: tl.constexpr):
    best_quantized, best_xscale = quantize_to_fp4_single_search(x, BLOCK_SIZE)
    best_quantized = best_quantized.reshape((BLOCK_SIZE * 8, 2))
    best_quantized = tl.associative_scan(best_quantized, axis=-1, combine_fn=pack_quantized_single)
    _, quantized = tl.split(best_quantized)
    quantized = quantized.reshape((BLOCK_SIZE, 8))
    return quantized, best_xscale

@triton.jit
def add_fp8(x: tl.tensor, y):
    return (x.to(tl.uint8, bitcast=True) + y.to(tl.uint8)).to(tl.float8e4nv, bitcast=True)

@triton.jit
def quantize_to_fp4_single_search(x: tl.tensor, BLOCK_SIZE: tl.constexpr):
    max_abs_x = tl.maximum(tl.max(tl.abs(x), axis=-1), 1e-10) * (1.0 / 6.0)
    max_abs_x = tl.clamp(max_abs_x, 1e-10 / 6.0, 448.0)
    xscale_f8 = max_abs_x.to(tl.float8e4nv, fp_downcast_rounding="rtne")[:, None]
    best_loss = tl.full((BLOCK_SIZE,), float("inf"), dtype=tl.float32)
    best_quantized = tl.zeros((BLOCK_SIZE, 16), dtype=tl.uint8)
    best_xscale = tl.zeros((BLOCK_SIZE, 1), dtype=tl.float8e4nv)
    for offset_val in range(-7, 9, 1):
        xscale = (add_fp8(xscale_f8, offset_val)).to(tl.float32)
        xscale_inv = 1.0 / xscale
        quantized = f32_to_f4_single(x * xscale_inv)
        loss = (x - f4_to_f32_single(quantized) * xscale)
        loss = tl.sum(loss * loss, axis=-1)
        better = loss < best_loss
        best_loss = tl.where(better, loss, best_loss)
        best_quantized = tl.where(better[:, None], quantized.to(tl.uint8), best_quantized)
        new_xscale_fp8 = (xscale).to(tl.float8e4nv)
        best_xscale = tl.where(better[:, None], new_xscale_fp8, best_xscale)
    return best_quantized, best_xscale


## FIX BELOW SO I CAN CALL `quantize_to_fp4_packed_search` with any-dimensional tensor, not necessarily 4D 

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 16}),
        triton.Config({"BLOCK_SIZE": 32}),
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
    ],
    key=["num_blocks"],
)
@triton.jit
def quant_kernel(
    input: tl.tensor,
    output: tl.tensor,
    scales_output: tl.tensor,
    stride_batch_input: tl.int32, stride_block_input: tl.int32,
    stride_batch_output: tl.int32, stride_block_output: tl.int32,
    stride_batch_scales: tl.int32, stride_block_scales: tl.int32,
    num_blocks: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    # Get batch index and block index
    batch_idx = tl.program_id(0)
    block_group_idx = tl.program_id(1)

    # Compute which blocks we're processing (BLOCK_SIZE blocks at a time)
    block_indices = tl.arange(0, BLOCK_SIZE) + block_group_idx * BLOCK_SIZE
    mask = block_indices < num_blocks

    # Load input: shape (BLOCK_SIZE, 16)
    # Each row is one block of 16 elements to quantize
    input_offsets = batch_idx * stride_batch_input + block_indices[:, None] * stride_block_input + tl.arange(0, 16)[None, :]
    vals = tl.load(input + input_offsets, mask=mask[:, None])

    # Quantize: returns (BLOCK_SIZE, 8) quantized and (BLOCK_SIZE, 1) scales
    quantized_vals, scales = quantize_to_fp4_packed_search(vals, BLOCK_SIZE)

    # Store quantized output: shape (BLOCK_SIZE, 8)
    output_offsets = batch_idx * stride_batch_output + block_indices[:, None] * stride_block_output + tl.arange(0, 8)[None, :]
    tl.store(output + output_offsets, quantized_vals, mask=mask[:, None])

    # Store scales: shape (BLOCK_SIZE, 1)
    scales_offsets = batch_idx * stride_batch_scales + block_indices[:, None] * stride_block_scales + tl.arange(0, 1)[None, :]
    tl.store(scales_output + scales_offsets, scales, mask=mask[:, None])


def quantize_triton(x: torch.tensor):
    """
    Quantize a tensor of arbitrary dimensions to FP4.

    The quantization uses a fixed block size of 16 elements.

    Args:
        x: Input tensor of shape (..., last_dim) where last_dim % 16 == 0

    Returns:
        quantized: Tensor of shape (..., last_dim // 2)
        scales: Tensor of shape (..., last_dim // 16)
    """
    BLOCK_SIZE = 16
    original_shape = x.shape
    last_dim = original_shape[-1]

    assert last_dim % BLOCK_SIZE == 0, f"Last dimension {last_dim} must be divisible by 16"

    # Flatten all dimensions except the last one
    # Shape: (batch_size, last_dim)
    batch_size = 1
    for dim in original_shape[:-1]:
        batch_size *= dim

    x_flat = x.reshape(batch_size, last_dim)

    # Reshape to (batch_size, num_blocks, 16)
    num_blocks = last_dim // BLOCK_SIZE
    x_reshaped = x_flat.reshape(batch_size, num_blocks, BLOCK_SIZE)

    # Create output tensors
    scales = torch.empty((batch_size, num_blocks, 1), dtype=torch.float8_e4m3fn, device=x.device)
    quantized = torch.empty((batch_size, num_blocks, 8), dtype=torch.uint8, device=x.device)

    # Launch kernel
    grid = lambda META: (batch_size, triton.cdiv(num_blocks, META["BLOCK_SIZE"]))
    with torch.cuda.device(x.device.index):
        quant_kernel[grid](
            x_reshaped, quantized, scales,
            x_reshaped.stride(0), x_reshaped.stride(1),
            quantized.stride(0), quantized.stride(1),
            scales.stride(0), scales.stride(1),
            num_blocks=num_blocks,
        )

    # Reshape outputs back to original shape
    quantized_shape = list(original_shape[:-1]) + [last_dim // 2]
    scales_shape = list(original_shape[:-1]) + [last_dim // BLOCK_SIZE]

    quantized = quantized.reshape(quantized_shape)
    scales = scales.reshape(scales_shape)

    return quantized, scales


