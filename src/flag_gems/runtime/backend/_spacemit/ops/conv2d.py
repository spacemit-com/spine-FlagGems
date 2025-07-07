import torch
import torch.nn.functional as F
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
    configs=runtime.get_tuned_config("mm"),
    key=["M", "N", "K"],
)

@triton.jit
def mm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    dot_out_dtype: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        offsets=[pid_m * BLOCK_SIZE_M, 0],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        order=[1, 0],
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        offsets=[0, pid_n * BLOCK_SIZE_N],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
        order=[1, 0],
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if EVEN_K:
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        accumulator += tl.dot(a, b, allow_tf32=False)
    else:
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_block_ptr, boundary_check=(0, 1))
            b = tl.load(b_block_ptr, boundary_check=(0, 1))
            accumulator += tl.dot(a, b, allow_tf32=False)
            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

    c = accumulator.to(dot_out_dtype)

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        offsets=[pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        order=[1, 0],
    )

    tl.store(c_block_ptr, c, boundary_check=(0, 1))


def mm(a, b):
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()

    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    dot_out_dtype = tl.float32

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    with torch.device(a.device):
        mm_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            dot_out_dtype=dot_out_dtype,
        )

    return c


class Conv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation

        N, C, H, W = input.shape
        K, C, R, S = weight.shape
        str_h, str_w = stride
        pad_h, pad_w = padding
        dil_h, dil_w = dilation

        P = (H + 2 * pad_h - dil_h * (R - 1) - 1) // str_h + 1
        Q = (W + 2 * pad_w - dil_w * (S - 1) - 1) // str_w + 1

        # [N*P*Q, C*R*S]
        input_col = im2col(input, N, C, H, W, R, S, (str_h, str_w), (pad_h, pad_w), (dil_h, dil_w))

        # [K, C*R*S]
        weight_reshaped = weight.view(K, -1)

        output = mm(input_col, weight_reshaped.t())
        # output = input_col@weight_reshaped.t()

        output = output.view(N, P, Q, K).permute(0, 3, 1, 2)

        if bias is not None:
            output += bias[None, :, None, None]

        ctx.save_for_backward(input_col, weight_reshaped, input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        pass

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1):
    return Conv2d.apply(input, weight, bias, stride, padding, dilation)