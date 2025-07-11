import logging

import torch
import triton
import triton.language as tl


@triton.jit
def maxpool2d_kernel(
    input_ptr,
    output_ptr,
    min_val_ptr,
    B, C, H, W,
    kernel_h, kernel_w,
    stride_h, stride_w,
    pad_h, pad_w,
    out_h, out_w,
    input_stride_b, input_stride_c, input_stride_h, input_stride_w,
    output_stride_b, output_stride_c, output_stride_h, output_stride_w,
    min_val_stride_b, min_val_stride_c, min_val_stride_h, min_val_stride_w,

):
    w_out = tl.program_id(0)
    h_out = tl.program_id(1)
    c_b = tl.program_id(2)

    batch_idx = c_b // C
    channel_idx = c_b % C

    h_start = h_out * stride_h - pad_h
    w_start = w_out * stride_w - pad_w

    min_val_block_ptr = tl.make_block_ptr(
        base=min_val_ptr,
        shape=(1, 1, 1, 1),
        strides=(min_val_stride_b, min_val_stride_c, min_val_stride_h, min_val_stride_w),
        offsets=(0, 0, 0, 0),
        block_shape=(1, 1, 1, 1),
        order=(3, 2, 1, 0),
    )

    current_max = tl.load(min_val_block_ptr, boundary_check=(0, 1, 2, 3))

    for h_win in range(kernel_h):
        h_in = h_start + h_win
        h_ok = (h_in >= 0) & (h_in < H)

        for w_win in range(kernel_w):
            w_in = w_start + w_win
            w_ok = (w_in >= 0) & (w_in < W)

            if h_ok & w_ok:
                input_block_ptr = tl.make_block_ptr(
                    base=input_ptr,
                    shape=(B, C, H, W),
                    strides=(input_stride_b, input_stride_c, input_stride_h, input_stride_w),
                    offsets=(batch_idx, channel_idx, h_in, w_in),
                    block_shape=(1, 1, 1, 1),
                    order=(3, 2, 1, 0),
                )
                val = tl.load(input_block_ptr, boundary_check=(0, 1, 2, 3))
                current_max = tl.maximum(current_max, val)

    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(B, C, out_h, out_w),
        strides=(output_stride_b, output_stride_c, output_stride_h, output_stride_w),
        offsets=(batch_idx, channel_idx, h_out, w_out),
        block_shape=(1, 1, 1, 1),
        order=(3, 2, 1, 0),
    )

    tl.store(output_block_ptr, current_max, boundary_check=(3, 2, 1, 0))



def maxpool2d(
    input,
    kernel_size,
    stride,
    padding,
):
    if stride is None:
        stride = kernel_size

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding

    B, C, H, W = input.shape

    out_h = (H + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (W + 2 * pad_w - kernel_w) // stride_w + 1
    assert out_h > 0 and out_w > 0, "Output dimensions must be positive"

    output = torch.empty((B, C, out_h, out_w),
                         device=input.device, dtype=input.dtype)

    min_val = torch.full((1, 1, 1, 1), float('-inf'))

    i_st = input.stride()
    o_st = output.stride()
    min_val_st = min_val.stride()
    grid = (out_w, out_h, B * C)

    maxpool2d_kernel[grid](
        input,
        output,
        min_val,
        B, C, H, W,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        out_h, out_w,
        i_st[0], i_st[1], i_st[2], i_st[3],
        o_st[0], o_st[1], o_st[2], o_st[3],
        min_val_st[0], min_val_st[1], min_val_st[2], min_val_st[3],
    )

    return output