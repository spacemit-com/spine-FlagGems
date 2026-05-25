"""
SPACEMIT-specific embedding implementation without atomic operations.
Uses PyTorch fallback for operations requiring atomic_add.
"""
import logging

import torch


def embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    """
    SPACEMIT backend: embedding fallback to PyTorch.

    The original implementation uses tl.atomic_add which is not supported
    on SPACEMIT hardware. We fall back to PyTorch's native implementation.
    """
    logging.debug("SPACEMIT backend: using PyTorch fallback for embedding (atomic_add not supported)")
    return torch.nn.functional.embedding(
        indices, weight, padding_idx,
        max_norm=None, norm_type=2.0,
        scale_grad_by_freq=scale_grad_by_freq,
        sparse=sparse
    )
