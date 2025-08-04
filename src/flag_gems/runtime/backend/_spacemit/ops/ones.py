import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.shape_utils import volume

device_ = device
logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def ones_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(n_elements, ),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,)
    )
    value = 1.0

    tl.store(output_block_ptr, (value).to(output_block_ptr.dtype.element_ty), boundary_check=(0,))


def ones(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS ONES")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)

    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    with torch_device_fn.device(device):
        ones_kernel[grid](out, N, BLOCK_SIZE)
    return out
