import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.mv import mv as general_mv
from flag_gems.runtime import torch_device_fn

from ._rvisa import HAS_SMT_RVISA, _sr_call, f16, f32
from ._rvisa import tle as srv

logger = logging.getLogger(__name__)

if HAS_SMT_RVISA:

    @srv.raw_kernel
    def _mv_rvisa(
        B: srv.mem(f16),
        A: srv.mem(f16),
        C: srv.mem(f32, out=True),
        K: srv.index,
        row_base: srv.index,
        row_end: srv.index,
    ):
        # streaming svector GEMV: 4 rows per iteration, f32 vmacc
        # accumulation, one horizontal vreduce_sum per row; 1.05-1.08x vs the
        # mm M=1 transposed-b decode path and up to 18x vs the general mv at
        # large N (K3 raw_vs_ops bench, 2026-10-23). Ported from spine-triton
        # python/tests/raw/test_raw_mv_svector.py style2.
        nvl = srv.vconfig(-1, 1)
        # strip-mine: the main loop takes full tiles (fast vload path), the
        # tail loop narrows via vconfig(K-ki) so padded lanes fill with 0 and
        # never affect the accumulation
        Kfloor = (K // nvl) * nvl
        for ni in srv.range(row_base, row_end, 4):
            acc0 = srv.vzero(f32)
            acc1 = srv.vzero(f32)
            acc2 = srv.vzero(f32)
            acc3 = srv.vzero(f32)
            for ki in srv.range(0, Kfloor, nvl):
                va = srv.vload(A, ki)
                vb0 = srv.vload(B, ni * K + ki)
                vb1 = srv.vload(B, (ni + 1) * K + ki)
                vb2 = srv.vload(B, (ni + 2) * K + ki)
                vb3 = srv.vload(B, (ni + 3) * K + ki)
                acc0 = srv.vmacc(acc0, vb0, va)
                acc1 = srv.vmacc(acc1, vb1, va)
                acc2 = srv.vmacc(acc2, vb2, va)
                acc3 = srv.vmacc(acc3, vb3, va)
            # tail loop: runs 0/1 times; independent temporaries so the
            # iter_arg detection does not pick up the main loop's names
            for ki in srv.range(Kfloor, K, nvl):
                nvl = srv.vconfig(K - ki, 1)
                ta = srv.vload(A, ki)
                tb0 = srv.vload(B, ni * K + ki)
                tb1 = srv.vload(B, (ni + 1) * K + ki)
                tb2 = srv.vload(B, (ni + 2) * K + ki)
                tb3 = srv.vload(B, (ni + 3) * K + ki)
                acc0 = srv.vmacc(acc0, tb0, ta)
                acc1 = srv.vmacc(acc1, tb1, ta)
                acc2 = srv.vmacc(acc2, tb2, ta)
                acc3 = srv.vmacc(acc3, tb3, ta)
            srv.sstore(C, ni, srv.vreduce_sum(acc0))
            srv.sstore(C, ni + 1, srv.vreduce_sum(acc1))
            srv.sstore(C, ni + 2, srv.vreduce_sum(acc2))
            srv.sstore(C, ni + 3, srv.vreduce_sum(acc3))

    @triton.jit(do_not_specialize=["K", "N"])
    def _mv_rvisa_host(B, A, C, K, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        row_base = pid * BLOCK
        row_end = min(row_base + BLOCK, N)
        _sr_call(_mv_rvisa, outputs=[], inputs=[B, A, C, K, row_base, row_end])


def mv(inp, vec):
    logger.debug("GEMS_SPACEMIT MV")
    assert inp.shape[1] == vec.shape[0], "incompatible dimensions"
    N, K = inp.shape

    # svector path constraints: f16 operands, N % 4 == 0 (the 4-row unroll
    # must not read past the last row) and a <=512-program grid (the runtime
    # silently drops programs beyond that); everything else goes to the
    # general implementation
    if (
        HAS_SMT_RVISA
        and inp.dtype == torch.float16
        and vec.dtype == torch.float16
        and N % 4 == 0
        and N > 0
    ):
        # BLOCK widens with N so ceil(N / BLOCK) stays within 512 programs;
        # 4 below N<=2048 and 16 above match the benchmarked configs
        block = 4 if N <= 2048 else 16
        while triton.cdiv(N, block) > 512:
            block *= 2
        # the unroll writes whole 4-row groups; pad C to a BLOCK multiple so
        # the final program's stores stay in bounds, then slice [:N]
        Np = ((N + block - 1) // block) * block
        inp_c = inp.contiguous().reshape(-1)
        vec_c = vec.contiguous()
        C = torch.empty(Np, dtype=torch.float32, device=inp.device)
        with torch_device_fn.device(inp.device):
            _mv_rvisa_host[(Np // block,)](inp_c, vec_c, C, K, N, BLOCK=block)
        return C[:N].to(inp.dtype)
    return general_mv(inp, vec)
