import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)
rsqrt = tl.rsqrt


def make_3d_for_bn(input: Tensor) -> Tensor:
    """
    Converts the input to a 3D view for batch normalization.

    Args:
        input: Input to render 3D.

    Returns:
        Input's 3D view.
    """
    if input.ndim == 2:
        input = input.unsqueeze(-1)

    elif input.ndim >= 4:
        input = input.flatten(2, -1)

    return input


# NOTE: This part of the kernel code is copied and modified
# from the https://github.com/BobMcDear/attorch codebase.


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("batch_norm"),
    key=["batch_dim", "spatial_dim"],
    restore_value=["running_mean_pointer", "running_var_pointer"],
)
@triton.heuristics(runtime.get_heuristic_config("batch_norm"))
@triton.jit
def batch_norm_forward_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    mean_pointer,
    inv_std_pointer,
    output_pointer,
    running_mean_pointer,
    running_var_pointer,
    batch_dim,
    spatial_dim,
    input_batch_stride,
    input_feat_stride,
    input_spatial_stride,
    output_batch_stride,
    output_feat_stride,
    output_spatial_stride,
    momentum,
    eps,
    feat_dim,
    is_train: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    feat_pid = tl.program_id(axis=0)

    # traning mode default track_running_stat
    if is_train:
        mean = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        var = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        cnt = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

        m_num_steps = tl.cdiv(batch_dim, BLOCK_M)
        n_num_steps = tl.cdiv(spatial_dim, BLOCK_N)

        input_base_ptr = tl.make_block_ptr(
            base=input_pointer + input_feat_stride * feat_pid,
            shape=(batch_dim, spatial_dim),
            strides=(input_batch_stride, input_spatial_stride),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )

        for m_step in range(0, m_num_steps):
            for n_step in range(0, n_num_steps):
                spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
                spatial_mask = spatial_offset < spatial_dim

                batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
                batch_mask = batch_offset < batch_dim

                input_block_ptr = tl.advance(
                    input_base_ptr, (m_step * BLOCK_M, n_step * BLOCK_N)
                )

                curr_input = tl.load(input_block_ptr, boundary_check=(0, 1)).to(
                    tl.float32
                )
                mask = batch_mask[:, None] & spatial_mask[None, :]

                step = m_step * n_num_steps + n_step + 1
                new_mean = tl.where(mask, mean + (curr_input - mean) / step, mean)
                new_var = tl.where(
                    mask, var + (curr_input - new_mean) * (curr_input - mean), var
                )
                cnt += mask.to(tl.int32)
                mean = new_mean
                var = new_var

        final_mean = tl.sum(mean * cnt) / (batch_dim * spatial_dim)
        var = tl.sum(var + cnt * (mean - final_mean) * (mean - final_mean)) / (
            batch_dim * spatial_dim
        )
        inv_std = rsqrt(var + eps)
        mean = final_mean

        mean_ptr = tl.make_block_ptr(
            base=mean_pointer,
            shape=(feat_dim,),
            strides=(1,),
            offsets=(feat_pid,),
            block_shape=(1,),
            order=(0,),
        )

        tl.store(mean_ptr, mean, boundary_check=(0,))

        inv_std_ptr = tl.make_block_ptr(
            base=inv_std_pointer,
            shape=(feat_dim,),
            strides=(1,),
            offsets=(feat_pid,),
            block_shape=(1,),
            order=(0,),
        )

        tl.store(inv_std_ptr, inv_std, boundary_check=(0,))

        running_mean_ptr = tl.make_block_ptr(
            base=running_mean_pointer,
            shape=(feat_dim,),
            strides=(1,),
            offsets=(feat_pid,),
            block_shape=(1,),
            order=(0,),
        )

        running_var_ptr = tl.make_block_ptr(
            base=running_var_pointer,
            shape=(feat_dim,),
            strides=(1,),
            offsets=(feat_pid,),
            block_shape=(1,),
            order=(0,),
        )

        running_mean = tl.load(running_mean_ptr, boundary_check=(0,))
        running_var = tl.load(running_var_ptr, boundary_check=(0,))

        n = batch_dim * spatial_dim
        tl.store(running_mean_pointer, (1 - momentum) * running_mean + momentum * mean)
        tl.store(
            running_var_pointer,
            (1 - momentum) * running_var + momentum * var * n / (n - 1),
        )

    else:
        mean_ptr = tl.make_block_ptr(
            base=running_mean_pointer,
            shape=(feat_dim,),
            strides=(1,),
            offsets=(feat_pid,),
            block_shape=(1,),
            order=(0,),
        )
        mean = tl.load(mean_ptr, boundary_check=(0,))

        running_var_ptr = tl.make_block_ptr(
            base=running_var_pointer,
            shape=(feat_dim,),
            strides=(1,),
            offsets=(feat_pid,),
            block_shape=(1,),
            order=(0,),
        )
        running_var = tl.load(running_var_ptr, boundary_check=(0,))

        inv_std = rsqrt(running_var + eps)

    if weight_pointer:
        weight_ptr = tl.make_block_ptr(
            base=weight_pointer,
            shape=(feat_dim,),
            strides=(1,),
            offsets=(feat_pid,),
            block_shape=(1,),
            order=(0,),
        )
        weight = tl.load(weight_ptr, boundary_check=(0,)).to(tl.float32)
    else:
        weight = tl.full((1,), 1.0, dtype=tl.float32)
    if bias_pointer:
        bias_ptr = tl.make_block_ptr(
            base=bias_pointer,
            shape=(feat_dim,),
            strides=(1,),
            offsets=(feat_pid,),
            block_shape=(1,),
            order=(0,),
        )
        bias = tl.load(bias_ptr, boundary_check=(0,)).to(tl.float32)
    else:
        bias = tl.full((1,), 0.0, dtype=tl.float32)

    input_base_ptr = tl.make_block_ptr(
        base=input_pointer + input_feat_stride * feat_pid,
        shape=(batch_dim, spatial_dim),
        strides=(input_batch_stride, input_spatial_stride),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    output_base_ptr = tl.make_block_ptr(
        base=output_pointer + output_feat_stride * feat_pid,
        shape=(batch_dim, spatial_dim),
        strides=(output_batch_stride, output_spatial_stride),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    for m_step in range(0, tl.cdiv(batch_dim, BLOCK_M)):
        for n_step in range(0, tl.cdiv(spatial_dim, BLOCK_N)):
            batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
            batch_mask = batch_offset < batch_dim

            spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
            spatial_mask = spatial_offset < spatial_dim

            input_block_ptr = tl.advance(
                input_base_ptr, (m_step * BLOCK_M, n_step * BLOCK_N)
            )
            output_block_ptr = tl.advance(
                output_base_ptr, (m_step * BLOCK_M, n_step * BLOCK_N)
            )

            curr_input = tl.load(input_block_ptr, boundary_check=(0, 1)).to(tl.float32)

            output = weight * (curr_input - mean) * inv_std + bias

            tl.store(output_block_ptr, output, boundary_check=(0, 1))


def batch_norm(
    input: Tensor,
    weight=None,
    bias=None,
    running_mean=None,  # self.running_mean if not self.training or self.track_running_state else None
    running_var=None,
    training=False,  # (self.running_mean is None) and (self.running_var is None)
    momentum=0.1,
    eps=1e-05,
):
    logger.debug("GEMS BATCHNORM FORWARD")

    input_3d = make_3d_for_bn(input)

    batch_dim, feat_dim, spatial_dim = input_3d.shape
    output = torch.empty_like(input_3d)

    mean = torch.empty(feat_dim, device=input.device, dtype=input.dtype)
    inv_std = torch.empty(feat_dim, device=input.device, dtype=input.dtype)

    running_mean = input if running_mean is None else running_mean
    running_var = input if running_var is None else running_var

    # Launches 1D grid where each program operates over one feature.
    with torch_device_fn.device(input.device):
        batch_norm_forward_kernel[(feat_dim,)](
            input_3d,
            weight,
            bias,
            mean,
            inv_std,
            output,
            running_mean,
            running_var,
            batch_dim,
            spatial_dim,
            *input_3d.stride(),
            *output.stride(),
            momentum,
            eps,
            feat_dim,
            is_train=training,
        )

    return output.view_as(input), mean, inv_std
