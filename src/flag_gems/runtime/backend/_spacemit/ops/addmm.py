import logging

import torch
import triton
import triton.language as tl
from flag_gems import runtime
from flag_gems.utils import libentry, libtuner
import triton.language.extra.deeplink as dl


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("addmm"),
    key=["M", "N", "K"],
    strategy=["log", "log", "log"],
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    alpha,
    beta,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    lhs_first: tl.constexpr,
    mr: tl.constexpr,
    nr: tl.constexpr,
    kr: tl.constexpr,
    mt: tl.constexpr,
    nt: tl.constexpr,
    kt: tl.constexpr,
    mb: tl.constexpr,
    nb: tl.constexpr,
    kb: tl.constexpr,
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
        dl.compile_hint(accumulator, "lhs_first", lhs_first)
        dl.compile_hint(accumulator, "mr", mr)
        dl.compile_hint(accumulator, "nr", nr)
        dl.compile_hint(accumulator, "kr", kr)
        dl.compile_hint(accumulator, "mt", mt)
        dl.compile_hint(accumulator, "nt", nt)
        dl.compile_hint(accumulator, "kt", kt)
        dl.compile_hint(accumulator, "mb", mb)
        dl.compile_hint(accumulator, "nb", nb)
        dl.compile_hint(accumulator, "kb", kb)
    else:
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_block_ptr, boundary_check=(0, 1))
            b = tl.load(b_block_ptr, boundary_check=(0, 1))
            accumulator += tl.dot(a, b, allow_tf32=False)

            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

    bias_block_ptr = tl.make_block_ptr(
        base=bias_ptr,
        shape=[M, N],
        strides=[stride_im, stride_in],
        offsets=[pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        order=[1, 0],
    )
    bias = tl.load(bias_block_ptr, boundary_check=(0, 1))
    accumulator = accumulator * alpha + bias * beta
    c = accumulator.to(bias.dtype)

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        offsets=[pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        order=[1, 0],
    )

    tl.store(c_block_ptr, c, boundary_check=(0, 1))


def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    logging.debug("GEMS ADDMM")
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    M, K = mat1.shape
    _, N = mat2.shape

    mat1 = mat1.contiguous()
    mat2 = mat2.contiguous()
    out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    bias = bias.broadcast_to(out.shape).contiguous()

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    BLOCK_SIZE_K = triton.next_power_of_2(K)
    EVEN_K = 1
    lhs_first = True
    kr = BLOCK_SIZE_K
    kt = 1
    with torch.device(mat1.device):
        addmm_kernel[grid](
            mat1,
            mat2,
            bias,
            out,
            alpha,
            beta,
            M,
            N,
            K,
            mat1.stride(0),
            mat1.stride(1),
            mat2.stride(0),
            mat2.stride(1),
            bias.stride(0),
            bias.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            EVEN_K=EVEN_K,
            lhs_first=lhs_first,
            kr=kr,
            kt=kt,
        )
    return out
