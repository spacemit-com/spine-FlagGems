import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("im2col"),
    key=["IH", "IW", "OH", "OW", "KH", "KW"],
)

@triton.jit
def im2col_kernel(
    input_ptr,
    output_ptr,
    N, C, IH, IW,
    KH, KW,
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    OH, OW,
    input_batch_stride, input_height_stride, input_width_stride, input_channel_stride,
    output_row_stride, output_col_stride,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid = tl.program_id(0)

    n = pid // (OH * OW)
    ohow = pid % (OH * OW)
    oh = ohow // OW
    ow = ohow % OW

    window_h = oh * stride_h - pad_h
    window_w = ow * stride_w - pad_w

    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(N, IH, IW, C),
        strides=(input_batch_stride, input_height_stride, input_width_stride, input_channel_stride),
        offsets=(n, 0, 0, 0),
        block_shape=(1, 1, 1, BLOCK_SIZE_C),
        order=(3, 2, 1, 0)
    )

    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(N * OH * OW, C * KH * KW),
        strides=(output_row_stride, output_col_stride),
        offsets=(pid, 0),
        block_shape=(1, BLOCK_SIZE_C),
        order=(1, 0)
    )

    for kh in range(KH):
        for kw in range(KW):
            h = window_h + kh * dilation_h
            w = window_w + kw * dilation_w

            col_idx = (kh * KW + kw) * C

            output_block_ptr_col = tl.advance(output_block_ptr, (0, col_idx))

            valid_h = (h >= 0) & (h < IH)
            valid_w = (w >= 0) & (w < IW)
            valid = valid_h & valid_w

            for c_start in range(0, C, BLOCK_SIZE_C):
                if valid:
                    input_block_ptr_c = tl.advance(input_block_ptr, (0, h, w, c_start))
                    vals = tl.load(
                        input_block_ptr_c,
                        boundary_check=(0,1,2,3),
                    )
                    vals = tl.reshape(vals, (1, BLOCK_SIZE_C))
                else:
                    vals = tl.zeros((1, BLOCK_SIZE_C), dtype=tl.float32)
                output_block_ptr_final = tl.advance(output_block_ptr_col, (0, c_start))

                tl.store(
                    output_block_ptr_final,
                    vals,
                    boundary_check=(0,1)
                )


def im2col(input, N, C, IH, IW, KH, KW, stride, padding, dilation):
    str_h, str_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    OH = (IH + 2 * pad_h - dil_h * (KH - 1) - 1) // str_h + 1
    OW = (IW + 2 * pad_w - dil_w * (KW - 1) - 1) // str_w + 1

    GEMM_M = N * OH * OW
    GEMM_N = C * KH * KW

    im2col_input = torch.empty(
        (GEMM_M, GEMM_N),
        dtype=input.dtype,
        device=input.device
    )

    grid = (N * OH * OW,)

    input = input.permute(0,2,3,1).contiguous()

    im2col_kernel[grid](
        input,
        im2col_input,
        N, C, IH, IW,
        KH, KW,
        str_h, str_w,
        pad_h, pad_w,
        dil_h, dil_w,
        OH, OW,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        im2col_input.stride(0), im2col_input.stride(1),
    )

    return im2col_input

@libentry()
@libtuner(
    configs=runtime.get_tuned_config("bmm"),
    key=["M", "N", "K"],
)


@triton.jit
def bmm_kernel(
    A,
    B,
    O,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cb,
    stride_cn,
    stride_cm,
    dot_out_dtype: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
):

    pidx = tl.program_id(0)
    pidy = tl.program_id(1)
    pid_b = tl.program_id(2)

    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy


    block_m = pid_m * TILE_M
    block_n = pid_n * TILE_N


    offset_a = pid_b * stride_ab
    offset_o = pid_b * stride_cb


    a_ptr = tl.make_block_ptr(
        A + offset_a,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(block_m, 0),
        block_shape=(TILE_M, TILE_K),
        order=(1, 0),
    )

    b_ptr = tl.make_block_ptr(
        B,
        shape=(N, K),
        strides=(stride_bn, stride_bk),
        offsets=(block_n, 0),
        block_shape=(TILE_N, TILE_K),
        order=(1, 0),
    )

    o_ptr = tl.make_block_ptr(
        O + offset_o,
        shape=(N, M),
        strides=(stride_cn, stride_cm),
        offsets=(block_n, block_m),
        block_shape=(TILE_N, TILE_M),
        order=(1, 0),
    )


    acc = tl.zeros((TILE_N, TILE_M), dtype=tl.float32)


    if DIVISIBLE_K:
        for k in range(0, K, TILE_K):
            a_tile = tl.load(a_ptr, boundary_check=(0, 1))
            b_tile = tl.load(b_ptr, boundary_check=(0, 1))

            acc += tl.dot(b_tile, tl.trans(a_tile))

            a_ptr = tl.advance(a_ptr, [0, TILE_K])
            b_ptr = tl.advance(b_ptr, [0, TILE_K])
    else:
        a_tile = tl.load(a_ptr, boundary_check=(0, 1))
        b_tile = tl.load(b_ptr, boundary_check=(0, 1))
        acc += tl.dot(b_tile, tl.trans(a_tile))

    c = acc.to(dot_out_dtype)


    tl.store(o_ptr, c, boundary_check=(0, 1))


def bmm(A, B):
    batch, M, K = A.shape
    _, N, _ = B.shape

    if A.stride(0) > 1 and A.stride(1) > 1:
        A = A.contiguous()
    if B.stride(0) > 1 and B.stride(1) > 1:
        B = B.contiguous()

    out = torch.empty((batch, N, M), dtype=A.dtype, device=A.device)
    dot_out_dtype = tl.float32

    grid_fn = lambda meta: (
        triton.cdiv(meta["M"], meta["TILE_M"]),
        triton.cdiv(meta["N"], meta["TILE_N"]),
        batch,
    )
    with torch_device_fn.device(A.device):
        bmm_kernel[grid_fn](
            A,
            B,
            out,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(1),
            B.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            dot_out_dtype=dot_out_dtype,
        )
    return out


class Conv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, padding, stride, dilation, groups):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation

        N, C, H, W = input.shape
        OC, _, R, S = weight.shape
        str_h, str_w = stride
        pad_h, pad_w = padding
        dil_h, dil_w = dilation

        P = (H + 2 * pad_h - dil_h * (R - 1) - 1) // str_h + 1
        Q = (W + 2 * pad_w - dil_w * (S - 1) - 1) // str_w + 1

        # [N*P*Q, IC*R*S]
        input_col = im2col(input, N, C, H, W, R, S, (str_h, str_w), (pad_h, pad_w), (dil_h, dil_w))
        input_col = input_col.view(N*P*Q,R*S,C).permute(0,2,1).contiguous().view(N*P*Q,C*R*S)

        # [N, P*Q, IC*R*S]
        input_col = input_col.view(N, P*Q, C*R*S)

        # [1, OC, IC*R*S]
        weight = weight.view(1, OC, -1)

        output = bmm(input_col, weight)

        # output: [N, OC, P, Q]
        output = output.view(N, OC, P, Q)

        if bias is not None:
            output += bias[None, :, None, None]

        ctx.save_for_backward(input_col, weight, input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        pass

def conv2d(input, weight, bias=None, padding=0, stride=1, dilation=1, groups=1):
    return Conv2d.apply(input, weight, bias, padding, stride, dilation, groups)