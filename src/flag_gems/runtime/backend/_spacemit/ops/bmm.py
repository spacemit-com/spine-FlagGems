import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


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
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
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
    offset_b = pid_b * stride_bb
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
        B + offset_b,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, block_n),
        block_shape=(TILE_K, TILE_N),
        order=(1, 0),
    )

    o_ptr = tl.make_block_ptr(
        O + offset_o,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(block_m, block_n),
        block_shape=(TILE_M, TILE_N),
        order=(1, 0),
    )

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    if DIVISIBLE_K:
        for k in range(0, K, TILE_K):
            a_tile = tl.load(a_ptr, boundary_check=(0, 1))
            b_tile = tl.load(b_ptr, boundary_check=(0, 1))

            acc += tl.dot(a_tile, b_tile)

            a_ptr = tl.advance(a_ptr, [0, TILE_K])
            b_ptr = tl.advance(b_ptr, [TILE_K, 0])
    else:
        a_tile = tl.load(a_ptr, boundary_check=(0, 1))
        b_tile = tl.load(b_ptr, boundary_check=(0, 1))
        acc += tl.dot(a_tile, b_tile)

    c = acc.to(o_ptr.dtype.element_ty)

    tl.store(o_ptr, c, boundary_check=(0, 1))


def bmm(A, B):
    logger.debug("GEMS BMM")
    batch, M, K = A.shape
    _, _, N = B.shape
    if A.stride(0) > 1 and A.stride(1) > 1:
        A = A.contiguous()
    if B.stride(0) > 1 and B.stride(1) > 1:
        B = B.contiguous()
    out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)

    grid_fn = lambda meta: (
        triton.cdiv(meta["M"], meta["TILE_M"]),
        triton.cdiv(meta["N"], meta["TILE_N"]),
        batch,
    )
    TILE_M = 128
    TILE_N = 128
    TILE_K = triton.next_power_of_2(K)
    GROUP_M = 1
    DIVISIBLE_K = 0
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
            B.stride(0),
            B.stride(1),
            B.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            TILE_K=TILE_K,
            GROUP_M=GROUP_M,
            DIVISIBLE_K=DIVISIBLE_K,
        )
    return out
