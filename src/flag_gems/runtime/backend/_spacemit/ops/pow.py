import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic

_pow = tl_extra_shim.pow
logger = logging.getLogger(__name__)


@triton.jit
def _pow_impl(x, exponent):
    """C `pow` semantics on fp32 operands.

    The backend lowers `linalg.powf` to exp2(y*log2(x)), which (a) is NaN for
    any negative base, even when the exponent is an integer, and (b) loses
    ~7e-6 relative accuracy for |exponent| ~ 100. Compute on |x| in fp64 and
    re-apply the sign of an integer exponent explicitly; a negative base with
    a non-integer exponent stays NaN.
    """
    xd = x.to(tl.float64)
    yd = exponent.to(tl.float64)
    r = _pow(tl.abs(xd), yd).to(tl.float32)
    r = tl.where((x == 0.0) & (exponent == 0.0), 1.0, r)
    is_int = (exponent - tl.floor(exponent)) == 0.0
    odd = (exponent - 2.0 * tl.floor(exponent * 0.5)) != 0.0
    sign = tl.where(odd, -1.0, 1.0)
    r_neg = tl.where(is_int, sign * r, float("nan"))
    return tl.where(x < 0.0, r_neg, r)


@pointwise_dynamic(promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func(x, exponent):
    return _pow_impl(x.to(tl.float32), exponent.to(tl.float32))


def pow_tensor_tensor(A, exponent):
    logger.debug("GEMS_SPACEMIT POW_TENSOR_TENSOR")
    return pow_func(A, exponent)


def pow_tensor_tensor_(A, exponent):
    logger.debug("GEMS_SPACEMIT POW_TENSOR_TENSOR_")
    return pow_func(A, exponent, out0=A)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_tensor_scalar(x, exponent):
    return _pow_impl(x.to(tl.float32), exponent.to(tl.float32))


@triton.jit
def pow_by_mul_kernel(
    X_ptr,
    Out_ptr,
    n_elements,
    exponent: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(X_ptr + offsets, mask=mask)
    result = x
    for _ in range(1, exponent):
        result = result * x
    tl.store(Out_ptr + offsets, result, mask=mask)


def _pow_by_mul(A, exponent):
    out = torch.empty_like(A)
    n_elements = A.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    pow_by_mul_kernel[grid](
        A,
        out,
        n_elements,
        exponent=exponent,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def pow_tensor_scalar(A, exponent):
    logger.debug("GEMS_SPACEMIT POW_TENSOR_SCALAR")
    has_neg = bool((A < 0).any().item())
    if has_neg and isinstance(exponent, int) and exponent > 0:
        return _pow_by_mul(A, exponent)
    return pow_func_tensor_scalar(A, exponent)


def pow_tensor_scalar_(A, exponent):
    logger.debug("GEMS_SPACEMIT POW_TENSOR_SCALAR_")
    has_neg = bool((A < 0).any().item())
    if has_neg and isinstance(exponent, int) and exponent > 0:
        result = _pow_by_mul(A, exponent)
        A.copy_(result)
        return A

    return pow_func_tensor_scalar(A, exponent, out0=A)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_scalar_tensor(x, exponent):
    return _pow_impl(x.to(tl.float32), exponent.to(tl.float32))


def pow_scalar(A, exponent):
    logger.debug("GEMS_SPACEMIT POW_SCALAR")
    return pow_func_scalar_tensor(A, exponent)
