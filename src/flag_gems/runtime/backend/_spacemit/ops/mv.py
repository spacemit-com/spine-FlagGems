import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

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
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    a_block_ptr = tl.make_block_ptr(
        base=A,
        shape=[N, M],
        strides=[stride_an, stride_am],
        offsets=[pid * BLOCK_N, 0],
        block_shape=[BLOCK_N, BLOCK_M],
        order=[1, 0],
    )
    b_block_ptr = tl.make_block_ptr(
        base=B,
        shape=[M,],
        strides=[stride_bm,],
        offsets=[0,],
        block_shape=[BLOCK_M,],
        order=[0,],
    )
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=C.dtype.element_ty)
    for m in range(0, M, BLOCK_M):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, ))
        acc += a * b
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_M))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_M, ))

    acc = tl.sum(acc, axis=1)
    c_block_ptr = tl.make_block_ptr(
        base=C,
        shape=[N,],
        strides=[stride_cn,],
        offsets=[pid * BLOCK_N,],
        block_shape=[BLOCK_N,],
        order=[0],
    )
    tl.store(c_block_ptr, acc, boundary_check=(0, ))


def mv(inp, vec):
    logger.debug("GEMS MV")
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
