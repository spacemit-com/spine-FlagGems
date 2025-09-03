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
@triton.autotune(configs=runtime.get_tuned_config("triu"), key=["M", "N"])
@triton.jit(do_not_specialize=["diagonal"])
def triu_kernel(
    X,
    Y,
    M,
    N,
    diagonal,
    M_BLOCK_SIZE: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * M_BLOCK_SIZE

    x_block_ptr = tl.make_block_ptr(
        base=X,
        shape=(M, N),
        strides=(N, 1),
        offsets=(row_start, 0),
        block_shape=(M_BLOCK_SIZE, N_BLOCK_SIZE),
        order=(1, 0),
    )
    y_block_ptr = tl.make_block_ptr(
        base=Y,
        shape=(M, N),
        strides=(N, 1),
        offsets=(row_start, 0),
        block_shape=(M_BLOCK_SIZE, N_BLOCK_SIZE),
        order=(1, 0),
    )

    for n_offset in range(0, N, N_BLOCK_SIZE):
        x_block_ptr = tl.advance(x_block_ptr, [0, n_offset])
        y_block_ptr = tl.advance(y_block_ptr, [0, n_offset])
        x = tl.load(x_block_ptr, boundary_check=(0, 1))
        rows_in_block = tl.arange(0, M_BLOCK_SIZE) + row_start
        cols_in_block = tl.arange(0, N_BLOCK_SIZE) + n_offset
        row_idx = rows_in_block[:, None]
        col_idx = cols_in_block[None, :]
        triu_mask = row_idx + diagonal <= col_idx
        y = tl.where(triu_mask, x, 0.0)
        tl.store(y_block_ptr, y, boundary_check=(0, 1))


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("triu_batch"),
    key=["batch", "MN", "N", "diagonal"],
)
@triton.jit(do_not_specialize=["diagonal"])
def triu_batch_kernel(
    X,
    Y,
    batch,
    MN,
    N,
    diagonal,
    BATCH_BLOCK_SIZE: tl.constexpr,
    MN_BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    mn_id = tl.program_id(1)

    x_ptr = tl.make_block_ptr(
        base=X,
        shape=(batch, MN),
        strides=(MN, 1),
        offsets=(batch_id * BATCH_BLOCK_SIZE, mn_id * MN_BLOCK_SIZE),
        block_shape=(BATCH_BLOCK_SIZE, MN_BLOCK_SIZE),
        order=(1, 0),
    )
    y_ptr = tl.make_block_ptr(
        base=Y,
        shape=(batch, MN),
        strides=(MN, 1),
        offsets=(batch_id * BATCH_BLOCK_SIZE, mn_id * MN_BLOCK_SIZE),
        block_shape=(BATCH_BLOCK_SIZE, MN_BLOCK_SIZE),
        order=(1, 0),
    )

    x = tl.load(x_ptr, boundary_check=(0, 1))

    cols = tl.arange(0, MN_BLOCK_SIZE) + mn_id * MN_BLOCK_SIZE
    m = cols // N
    n = cols % N

    mask = m + diagonal <= n
    y = tl.where(mask, x, 0.0)
    tl.store(y_ptr, y, boundary_check=(0, 1))


INT32_MAX = torch.iinfo(torch.int32).max


def triu(A, diagonal=0):
    logger.debug("GEMS TRIU")
    A = A.contiguous()
    out = torch.empty_like(A)
    assert len(A.shape) > 1, "Input tensor must have at least 2 dimensions"
    M, N = A.shape[-2:]
    with torch_device_fn.device(A.device):
        if len(A.shape) == 2:
            grid = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]),)
            triu_kernel[grid](A, out, M, N, diagonal)
        else:
            batch = int(torch.numel(A) / M / N)
            B = A.view(batch, -1)
            grid = lambda meta: (
                triton.cdiv(batch, meta["BATCH_BLOCK_SIZE"]),
                triton.cdiv(M * N, meta["MN_BLOCK_SIZE"]),
            )
            triu_batch_kernel[grid](
                B,
                out,
                batch,
                M * N,
                N,
                diagonal,
            )
            out = out.view(A.shape)
    return out
