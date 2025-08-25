import torch
import triton
import triton.language as tl
from flag_gems.utils.limits import get_dtype_min

@triton.jit
def maxpool2d_kernel(
    input_ptr,
    output_ptr,
    N, C, IH, IW,
    OH, OW,
    KH: tl.constexpr,
    KW: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    input_batch_stride,
    input_channel_stride,
    input_height_stride,
    input_width_stride,
    output_batch_stride,
    output_channel_stride,
    output_height_stride,
    output_width_stride,
    BLOCK_SIZE_C: tl.constexpr,
):

    pid = tl.program_id(0)
    n = pid // (OH * OW)
    ohow = pid % (OH * OW)
    oh = ohow // OW
    ow = ohow % OW
    c_block = tl.arange(0, BLOCK_SIZE_C)

    window_h = oh * stride_h - pad_h
    window_w = ow * stride_w - pad_w

    min_value = get_dtype_min(input_ptr.type.element_ty)
    max_vals = tl.full((BLOCK_SIZE_C,), min_value, dtype=tl.float32)
    channel_mask = c_block < C

    total_iters = KH * KW
    for k in range(total_iters):
        kh = k // KW
        kw = k % KW
        h = window_h + kh * dilation_h
        w = window_w + kw * dilation_w

        valid_h = (h >= 0) & (h < IH)
        valid_w = (w >= 0) & (w < IW)
        valid = valid_h & valid_w

        input_offset = (
            n * input_batch_stride +
            c_block * input_channel_stride +
            h * input_height_stride +
            w * input_width_stride
        )
        input_ptrs = input_ptr + input_offset
        total_mask = valid & channel_mask

        current = tl.load(input_ptrs, mask=total_mask, other=min_value)
        max_vals = tl.maximum(max_vals, current)
    max_vals = tl.where(max_vals == min_value, 0.0, max_vals)

    output_offset = (
        n * output_batch_stride +
        c_block * output_channel_stride +
        oh * output_height_stride +
        ow * output_width_stride
    )
    output_ptrs = output_ptr + output_offset
    tl.store(output_ptrs, max_vals, mask=channel_mask)

def maxpool2d(
    input: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    dilation: int = 1
) -> torch.Tensor:
    KH, KW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    stride_h, stride_w = (stride, stride) if stride is None or isinstance(stride, int) else stride
    pad_h, pad_w = (padding, padding) if isinstance(padding, int) else padding
    dil_h, dil_w = (dilation, dilation) if isinstance(dilation, int) else dilation
    N, C, IH, IW = input.shape
    OH = (IH + 2 * pad_h - dil_h * (KH - 1) - 1) // stride_h + 1
    OW = (IW + 2 * pad_w - dil_w * (KW - 1) - 1) // stride_w + 1
    output = torch.empty((N, C, OH, OW), dtype=input.dtype, device=input.device)

    (input_batch_stride, input_channel_stride,
     input_height_stride, input_width_stride) = input.stride()

    (output_batch_stride, output_channel_stride,
     output_height_stride, output_width_stride) = output.stride()

    BLOCK_SIZE_C = 128
    num_blocks_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid = (N * OH * OW * num_blocks_c,)

    maxpool2d_kernel[grid](
        input, output,
        N, C, IH, IW, OH, OW,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        input_batch_stride, input_channel_stride,
        input_height_stride, input_width_stride,
        output_batch_stride, output_channel_stride,
        output_height_stride, output_width_stride,
        BLOCK_SIZE_C,
    )
    return output