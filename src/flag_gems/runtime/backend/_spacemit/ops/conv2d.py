import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("im2col"),
    key=["H", "W", "P", "Q", "R", "S"],
)

@triton.jit
def im2col_kernel(
    input_ptr,
    input_n, input_c, input_h, input_w,
    stride_n, stride_c, stride_h, stride_w,

    output_ptr,
    output_gemmm, output_gemmk,

    R, S,
    stride_h_conv, stride_w_conv,
    pad_h, pad_w,
    dil_h, dil_w,

    P, Q,

    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    gemm_m_offset = pid_m * BLOCK_M
    gemm_k_offset = pid_k * BLOCK_K

    gemm_m_idx = gemm_m_offset + tl.arange(0, BLOCK_M)
    gemm_k_idx = gemm_k_offset + tl.arange(0, BLOCK_K)

    gemm_m_mask = gemm_m_idx < output_gemmm
    gemm_k_mask = gemm_k_idx < output_gemmk
    active_mask = gemm_m_mask[:, None] & gemm_k_mask[None, :]

    pq = P * Q
    n_idx = tl.where(gemm_m_mask, gemm_m_idx // pq, 0)
    pq_residual = tl.where(gemm_m_mask, gemm_m_idx % pq, 0)
    p_idx = tl.where(gemm_m_mask, pq_residual // Q, 0)
    q_idx = tl.where(gemm_m_mask, pq_residual % Q, 0)

    rs = R * S
    c_idx = tl.where(gemm_k_mask, gemm_k_idx // rs, 0)
    rs_residual = tl.where(gemm_k_mask, gemm_k_idx % rs, 0)
    r_idx = tl.where(gemm_k_mask, rs_residual // S, 0)
    s_idx = tl.where(gemm_k_mask, rs_residual % S, 0)

    h_idx = p_idx[:, None] * stride_h_conv + r_idx[None, :] * dil_h - pad_h
    w_idx = q_idx[:, None] * stride_w_conv + s_idx[None, :] * dil_w - pad_w

    n_mask = n_idx[:, None] < input_n
    c_mask = c_idx[None, :] < input_c
    h_mask = (h_idx >= 0) & (h_idx < input_h)
    w_mask = (w_idx >= 0) & (w_idx < input_w)

    input_mask = active_mask & n_mask & c_mask & h_mask & w_mask

    n_off = n_idx[:, None] * stride_n
    c_off = c_idx[None, :] * stride_c
    h_off = h_idx * stride_h
    w_off = w_idx * stride_w

    input_offsets = n_off + c_off + h_off + w_off

    input_vals = tl.load(
        input_ptr + input_offsets,
        mask=input_mask,
        other=0.0
    )

    output_offsets = (gemm_m_idx[:, None] * output_gemmk + gemm_k_idx[None, :])

    tl.store(
        output_ptr + output_offsets,
        input_vals,
        mask=active_mask
    )

def im2col(input, N, C, H, W, R, S, stride, padding, dilation):
    str_h, str_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    P = (H + 2 * pad_h - dil_h * (R - 1) - 1) // str_h + 1
    Q = (W + 2 * pad_w - dil_w * (S - 1) - 1) // str_w + 1

    GEMM_M = N * P * Q
    GEMM_K = C * R * S

    im2col_input = torch.empty(
        (GEMM_M, GEMM_K),
        dtype=input.dtype,
        device=input.device
    )

    grid = lambda meta: (
        triton.cdiv(GEMM_M, meta['BLOCK_M']),
        triton.cdiv(GEMM_K, meta['BLOCK_K'])
    )

    im2col_kernel[grid](
        input,
        N, C, H, W,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),

        im2col_input,
        GEMM_M, GEMM_K,

        R, S,
        str_h, str_w,
        pad_h, pad_w,
        dil_h, dil_w,

        P, Q,
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