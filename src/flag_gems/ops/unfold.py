import torch
import triton
import triton.language as tl

def unfold(
    input: torch.Tensor,
    kernel_size: tuple,
    stride: int = 1,
    dilation: int = 1,
    padding: int = 0
) -> torch.Tensor:
    pass