import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import tl_extra_shim
from flag_gems.ops.softmax import softmax as general_softmax
from flag_gems.ops.softmax import softmax_out as general_softmax_out

logger = logging.getLogger(__name__)
exp = tl_extra_shim.exp


@libentry()
@triton.heuristics(
    {
        # softmax_spacemit's tune config was static (ROW_SIZE=1, COL_SIZE=512);
        # both are made adaptive: ROW_SIZE keeps the grid within the 512
        # programs this stack's runtime executes normally, and COL_SIZE covers
        # a whole row so the kernel takes the single-pass branch -- the
        # two-pass online path carries its max/denominator reductions across
        # loop iterations, which lowers to per-iteration DDR scratch traffic
        # (~100x wall time at n_cols=2048)
        "ROW_SIZE": lambda args: max(1, triton.cdiv(args["n_rows"], 512)),
        "COL_SIZE": lambda args: min(triton.next_power_of_2(args["n_cols"]), 8192),
        "ONE_TILE_PER_ROW": lambda args: min(triton.next_power_of_2(args["n_cols"]), 8192)
        >= args["n_cols"],
    }
)
@triton.jit
def softmax_kernel_spacemit(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    ROW_SIZE: tl.constexpr,
    COL_SIZE: tl.constexpr,
    ONE_TILE_PER_ROW: tl.constexpr,
):
    row_start = tl.program_id(0) * ROW_SIZE
    element_ty = output_ptr.type.element_ty

    for row_idx in range(row_start, row_start + ROW_SIZE):
        if row_idx < n_rows:
            if ONE_TILE_PER_ROW:
                # Single-pass: load once, compute max/exp/sum/div in registers
                input_block_ptr = tl.make_block_ptr(
                    base=input_ptr + row_idx * input_row_stride,
                    shape=(n_cols,),
                    strides=(1,),
                    offsets=(0,),
                    block_shape=(COL_SIZE,),
                    order=(0,),
                )
                output_block_ptr = tl.make_block_ptr(
                    base=output_ptr + row_idx * output_row_stride,
                    shape=(n_cols,),
                    strides=(1,),
                    offsets=(0,),
                    block_shape=(COL_SIZE,),
                    order=(0,),
                )
                row = tl.load(
                    input_block_ptr, boundary_check=(0,), padding_option="neg_inf"
                ).to(tl.float32)
                row_max = tl.max(row, axis=0)
                numerator = exp(row - row_max)
                denominator = tl.sum(numerator, axis=0)
                result = (numerator / denominator).to(element_ty)
                tl.store(output_block_ptr, result, boundary_check=(0,))
            else:
                # Two-pass online softmax for large n_cols
                row_max_total = tl.full((1,), value=-float("inf"), dtype=tl.float32)
                denominator = tl.zeros((1,), dtype=tl.float32)

                for col_idx in range(0, n_cols, COL_SIZE):
                    input_block_ptr = tl.make_block_ptr(
                        base=input_ptr + row_idx * input_row_stride,
                        shape=(n_cols,),
                        strides=(1,),
                        offsets=(col_idx,),
                        block_shape=(COL_SIZE,),
                        order=(0,),
                    )
                    row = tl.load(
                        input_block_ptr, boundary_check=(0,), padding_option="neg_inf"
                    ).to(tl.float32)
                    block_max = tl.max(row, axis=0)
                    new_max = tl.maximum(row_max_total, block_max)
                    denominator = denominator * exp(row_max_total - new_max)
                    denominator += tl.sum(exp(row - new_max), axis=0)
                    row_max_total = new_max

                inv_denom = 1.0 / denominator

                for col_idx in range(0, n_cols, COL_SIZE):
                    input_block_ptr = tl.make_block_ptr(
                        base=input_ptr + row_idx * input_row_stride,
                        shape=(n_cols,),
                        strides=(1,),
                        offsets=(col_idx,),
                        block_shape=(COL_SIZE,),
                        order=(0,),
                    )
                    output_block_ptr = tl.make_block_ptr(
                        base=output_ptr + row_idx * output_row_stride,
                        shape=(n_cols,),
                        strides=(1,),
                        offsets=(col_idx,),
                        block_shape=(COL_SIZE,),
                        order=(0,),
                    )
                    row = tl.load(
                        input_block_ptr, boundary_check=(0,), padding_option="neg_inf"
                    ).to(tl.float32)
                    numerator = exp(row - row_max_total)
                    result = (numerator * inv_denom).to(element_ty)
                    tl.store(output_block_ptr, result, boundary_check=(0,))


def _spacemit_softmax_lastdim(inp, out):
    n_rows, n_cols = inp.shape
    grid = lambda meta: (triton.cdiv(n_rows, meta["ROW_SIZE"]),)
    with torch_device_fn.device(inp.device):
        softmax_kernel_spacemit[grid](
            out,
            inp,
            inp.stride(0),
            out.stride(0),
            n_rows,
            n_cols,
        )


def softmax(self, dim, half_to_float=False):
    logger.debug("GEMS_SPACEMIT SOFTMAX")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim

    # this port's kernel only reduces the last dim; anything else must go to
    # the general implementation (last-dim-only reduction silently produces
    # wrong values for dim < ndim-1, e.g. (1,256) dim=0)
    if dim != self.ndim - 1:
        return general_softmax(self, dim, half_to_float)

    if half_to_float:
        dtype = torch.float32
    else:
        dtype = self.dtype

    inp = self.contiguous()

    n_cols = inp.shape[-1]
    n_rows = inp.numel() // n_cols
    inp_2d = inp.view(n_rows, n_cols)
    out_2d = torch.empty_like(inp_2d, dtype=dtype)
    _spacemit_softmax_lastdim(inp_2d, out_2d)
    return out_2d.view_as(inp)


def softmax_out(self, dim, half_to_float=False, *, out):
    logger.debug("GEMS_SPACEMIT SOFTMAX_OUT")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim

    # same last-dim-only restriction as softmax(); the .out entry is what the
    # registration surface wires up for aten softmax on this branch
    if dim != self.ndim - 1 or self.numel() == 0:
        return general_softmax_out(self, dim, half_to_float, out=out)

    inp = self.contiguous()

    n_cols = inp.shape[-1]
    n_rows = inp.numel() // n_cols
    inp_2d = inp.view(n_rows, n_cols)

    if tuple(out.shape) != tuple(self.shape):
        out.resize_(self.shape)
    if out.is_contiguous():
        _spacemit_softmax_lastdim(inp_2d, out.view(n_rows, n_cols))
        return out

    out_2d = torch.empty_like(inp_2d, dtype=out.dtype)
    _spacemit_softmax_lastdim(inp_2d, out_2d)
    out.copy_(out_2d.view(self.shape))
    return out
