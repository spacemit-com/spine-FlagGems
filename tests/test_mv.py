import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    MN_SHAPES = [
        (1, 32),
    ]
    FLOAT_DTYPES = [torch.float32]
else:
    MN_SHAPES = [
        (1, 32),
        (160, 1024),
        (5333, 497),
    ]
    FLOAT_DTYPES = utils.FLOAT_DTYPES


@pytest.mark.mv
@pytest.mark.parametrize("M, N", MN_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mv(M, N, dtype):
    matrix = torch.randn((N, M), dtype=dtype, device=flag_gems.device)
    vector = torch.randn((M,), dtype=dtype, device=flag_gems.device)
    ref_matrix = utils.to_reference(matrix, True)
    ref_vector = utils.to_reference(vector, True)

    ref_out = torch.mv(ref_matrix, ref_vector)
    with flag_gems.use_gems():
        res_out = torch.mv(matrix, vector)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=M)

# smt_rvisa eDSL path of the spacemit vendor mv (streaming svector GEMV,
# f16 only); on other backends this exercises the regular path. Shapes cover
# the narrowed-K tail (K not a multiple of 64) and the widened BLOCK needed
# to keep the grid within 512 programs.
@pytest.mark.mv
@pytest.mark.parametrize("N, K", [(64, 128), (100, 100), (8, 65), (260, 1024)])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_mv_smt_rvisa(N, K, dtype, monkeypatch):
    monkeypatch.setenv("SPACEMIT_SMT_RVISA", "1")
    matrix = torch.randn((N, K), dtype=dtype, device=flag_gems.device)
    vector = torch.randn((K,), dtype=dtype, device=flag_gems.device)
    ref_matrix = utils.to_reference(matrix, True)
    ref_vector = utils.to_reference(vector, True)

    ref_out = torch.mv(ref_matrix, ref_vector)
    with flag_gems.use_gems():
        res_out = torch.mv(matrix, vector)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)
