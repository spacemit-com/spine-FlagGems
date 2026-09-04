import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("mv"),
    key=["M", "N"],
)
@triton.jit
def mv_kernel(
    A,
    B,
    C,
    N,
    M,
    stride_an,
    stride_am,
    stride_bm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start_n = pid * BLOCK_N
    # f32 accumulator (matches mainline ops/mv.py): f16 partial-sum accumulation
    # over M blocks exceeds tolerance (test_mv 160-1024 fp16)
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for m in range(0, M, BLOCK_M):
        a_block_ptr = tl.make_block_ptr(
            base=A,
            shape=[N, M],
            strides=[stride_an, stride_am],
            offsets=[block_start_n, m],
            block_shape=[BLOCK_N, BLOCK_M],
            order=[1, 0],
        )
        a = tl.load(a_block_ptr, boundary_check=(0, 1)).to(tl.float32)

        b_block_ptr = tl.make_block_ptr(
            base=B,
            shape=[M],
            strides=[stride_bm],
            offsets=[m],
            block_shape=[BLOCK_M],
            order=[0],
        )
        b = tl.load(b_block_ptr, boundary_check=(0,)).to(tl.float32)
        acc += a * b[None, :]

    result = tl.sum(acc, axis=1)
    c_block_ptr = tl.make_block_ptr(
        base=C,
        shape=[N],
        strides=[stride_cn],
        offsets=[block_start_n],
        block_shape=[BLOCK_N],
        order=[0],
    )
    tl.store(c_block_ptr, result.to(C.dtype.element_ty), boundary_check=(0,))


def mv(inp, vec):
    logger.debug("GEMS_SPACEMIT MV")
    assert inp.shape[1] == vec.shape[0], "incompatible dimensions"
    N, M = inp.shape
    out = torch.empty((N,), device=inp.device, dtype=inp.dtype)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
    with torch_device_fn.device(inp.device):
        mv_kernel[grid](
            inp,
            vec,
            out,
            N,
            M,
            inp.stride(0),
            inp.stride(1),
            vec.stride(0),
            out.stride(0),
        )
    return out
