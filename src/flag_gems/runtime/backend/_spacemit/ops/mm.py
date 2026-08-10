import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner
import triton.language.extra.spine_raw as tle
from triton.language.extra.spine_raw import call as _sr_call
from triton.language.extra.spine_raw import mem, index

_f16 = tle.f16
_f32 = tle.f32
# NOTE: dtype args to tle.* must be bare names that match _DTYPE_NAMES
# ({"f16","f32",...}). Use the un-prefixed aliases f16/f32 inside the raw
# kernel body so the AST produces ast.Name("f32"), not ast.Name("_f32").
f16 = _f16
f32 = _f32


# ---------------------------------------------------------------------------
# M==1 GEMV fast path (true vfwmacc.vf intrinsic, pure llvm-direct sibling).
#   C[0, n:n+VL] = sum_k x[k] * W[k, n:n+VL]
#   - x[k]    : scalar f16 load (llvm.load)
#   - W[k, :] : contiguous row vector load (llvm.load, vector<[4]xf16> = 64 f16)
#   - acc     : vector<[4]xf32>, vfwmacc.vf (scalar·vector widening FMA) per k
# Pure llvm-direct → routes to LLVMDirectTextCodegen sibling path, bypasses
# spine-opt bufferization (which rejects llvm.call_intrinsic as "unknown memory
# side effects"). program_id resolved at runtime via spine_grid(ctx, axis).
# Output is f32 (vfwmacc accumulates in f32); host wrapper narrows to f16.
# Each program handles one NB=64 N-block; grid = (N // NB,).
# ---------------------------------------------------------------------------
@tle.raw_kernel
def mm_m1_raw(x_ptr: mem(f16), W_ptr: mem(f16), C_ptr: mem(f32, out=True),
              K: index, N: index, NB: index):
    vl = tle.llvm_const(64, "i64")
    zero_i64 = tle.llvm_const(0, "i64")
    zero_acc = tle.llvm_const(0.0, "vector<[4]xf32>")
    xbase = tle.llvm_base_ptr(x_ptr)
    wbase = tle.llvm_base_ptr(W_ptr)
    cbase = tle.llvm_base_ptr(C_ptr)
    n_start = tle.program_id(0) * NB
    acc = zero_acc
    for k in tle.range(0, K, 1):
        gx = tle.llvm_gep(xbase, k, "f16")
        xk = tle.call_intrinsic("llvm.load", [gx], result_type="f16")
        gw = tle.llvm_gep(wbase, k * N + n_start, "f16")
        wv = tle.call_intrinsic("llvm.load", [gw], result_type="vector<[4]xf16>")
        acc = tle.call_intrinsic("llvm.riscv.vfwmacc.vf", [acc, xk, wv, zero_i64, vl, zero_i64],
                                 result_type="vector<[4]xf32>")
    gc = tle.llvm_gep(cbase, n_start, "f32")
    tle.call_intrinsic("llvm.store", [acc, gc], result_type="()")


@triton.jit(do_not_specialize=["K", "N", "NB"])
def _mm_m1_host(x, W, C, K, N, NB):
    _sr_call(mm_m1_raw, outputs=[], inputs=[x, W, C, K, N, NB])


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm_spacemit"),
    key=["M", "N", "K"],
)
@triton.jit
def mm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    SPLIT_M: tl.constexpr,
    SPLIT_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    SUB_BLK_M: tl.constexpr,
    SUB_BLK_N: tl.constexpr,
    MICRO_M: tl.constexpr,
    MICRO_K: tl.constexpr,
    MICRO_N: tl.constexpr,
    SUB_BLK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        offsets=[pid_m * BLOCK_SIZE_M, 0],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        order=[1, 0],
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        offsets=[0, pid_n * BLOCK_SIZE_N],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
        order=[1, 0],
    )

    if EVEN_K:
        a_descriptor_load = smt.descriptor_load(a_block_ptr, (0, 0))
        a = smt.view(
            a_descriptor_load, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_K), (MICRO_M, MICRO_K)
        )
        b_descriptor_load = smt.descriptor_load(b_block_ptr, (0, 0))
        b = smt.view(
            b_descriptor_load, (0, 0), (BLOCK_SIZE_K, BLOCK_SIZE_N), (MICRO_K, MICRO_N)
        )
        accumulator = smt.dot(a, b)
        accumulator = smt.view(
            accumulator, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_N), (1, 1)
        )
        c = accumulator.to(c_ptr.dtype.element_ty)
        c_block_ptr = tl.make_block_ptr(
            base=c_ptr,
            shape=[M, N],
            strides=[stride_cm, stride_cn],
            offsets=[pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
            order=[1, 0],
        )
        tl.store(c_block_ptr, c, boundary_check=(0, 1))

    elif SPLIT_M:
        b_descriptor_load = smt.descriptor_load(b_block_ptr, (0, 0))
        b = smt.view(
            b_descriptor_load, (0, 0), (BLOCK_SIZE_K, BLOCK_SIZE_N), (MICRO_K, MICRO_N)
        )
        sub_num = (
            min(BLOCK_SIZE_M, M - BLOCK_SIZE_M * pid_m) + SUB_BLK_M - 1
        ) // SUB_BLK_M
        for s in smt.parallel(0, sub_num):
            a_descriptor_load = smt.descriptor_load(a_block_ptr, (0, 0))
            a = smt.view(
                a_descriptor_load,
                (s * SUB_BLK_M, 0),
                (SUB_BLK_M, BLOCK_SIZE_K),
                (MICRO_M, MICRO_K),
            )
            accumulator = smt.dot(a, b)
            accumulator = smt.view(
                accumulator, (0, 0), (SUB_BLK_M, BLOCK_SIZE_N), (1, 1)
            )
            c = accumulator.to(c_ptr.dtype.element_ty)
            c_block_ptr = tl.make_block_ptr(
                base=c_ptr,
                shape=[M, N],
                strides=[stride_cm, stride_cn],
                offsets=[pid_m * BLOCK_SIZE_M + s * SUB_BLK_M, pid_n * BLOCK_SIZE_N],
                block_shape=[SUB_BLK_M, BLOCK_SIZE_N],
                order=[1, 0],
            )
            tl.store(c_block_ptr, c, boundary_check=(0, 1))

    elif SPLIT_N:
        sub_num_m = (
            min(BLOCK_SIZE_M, M - BLOCK_SIZE_M * pid_m) + SUB_BLK_M - 1
        ) // SUB_BLK_M
        sub_num_n = (
            min(BLOCK_SIZE_N, N - BLOCK_SIZE_N * pid_n) + SUB_BLK_N - 1
        ) // SUB_BLK_N
        total_sub_blocks = sub_num_m * sub_num_n
        b_alloc_ptr = smt.alloc(shape=[BLOCK_SIZE_K, BLOCK_SIZE_N])
        b_alloc_view_ptr = smt.view(
            b_alloc_ptr, (0, 0), (BLOCK_SIZE_K, BLOCK_SIZE_N), (MICRO_K, MICRO_N)
        )
        bar = smt.mbarrier(flag=0, expect_count=sub_num_n)
        for s in smt.parallel(0, total_sub_blocks):
            s_m = s // sub_num_n
            s_n = s % sub_num_n
            a_descriptor_load = smt.descriptor_load(a_block_ptr, (0, 0))
            a = smt.view(
                a_descriptor_load,
                (s_m * SUB_BLK_M, 0),
                (SUB_BLK_M, BLOCK_SIZE_K),
                (MICRO_M, MICRO_K),
            )
            b_alloc_sub_ptr = smt.view(
                b_alloc_view_ptr, (0, s_n * SUB_BLK_N), (BLOCK_SIZE_K, SUB_BLK_N)
            )
            if s_m == 0:
                b_descriptor_load = smt.descriptor_load(b_block_ptr, (0, 0))
                b = smt.view(
                    b_descriptor_load,
                    (0, s_n * SUB_BLK_N),
                    (BLOCK_SIZE_K, SUB_BLK_N),
                    (MICRO_K, MICRO_N),
                )
                tl.store(b_alloc_sub_ptr, b, boundary_check=(0, 1, 2, 3))
                smt.barrier_arrive(bar)
            else:
                smt.barrier_wait(bar, flag=1)

            b_alloc = tl.load(b_alloc_sub_ptr, boundary_check=(0, 1, 2, 3))
            accumulator = smt.dot(a, b_alloc)
            accumulator = smt.view(accumulator, (0, 0), (SUB_BLK_M, SUB_BLK_N), (1, 1))
            c = accumulator.to(c_ptr.dtype.element_ty)
            c_block_ptr = tl.make_block_ptr(
                base=c_ptr,
                shape=[M, N],
                strides=[stride_cm, stride_cn],
                offsets=[
                    pid_m * BLOCK_SIZE_M + s_m * SUB_BLK_M,
                    pid_n * BLOCK_SIZE_N + s_n * SUB_BLK_N,
                ],
                block_shape=[SUB_BLK_M, SUB_BLK_N],
                order=[1, 0],
            )
            tl.store(c_block_ptr, c, boundary_check=(0, 1))

    elif SPLIT_K:
        accumulator = tl.zeros(
            (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=a_ptr.type.element_ty
        )
        accumulator = smt.view(
            accumulator, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_N), (MICRO_M, MICRO_N)
        )
        sub_num = (K + SUB_BLK_K - 1) // SUB_BLK_K
        for k in tl.range(0, sub_num):
            a_descriptor_load = smt.descriptor_load(a_block_ptr, (0, 0))
            a = smt.view(
                a_descriptor_load,
                (0, k * SUB_BLK_K),
                (BLOCK_SIZE_M, SUB_BLK_K),
                (MICRO_M, MICRO_K),
            )
            b_descriptor_load = smt.descriptor_load(b_block_ptr, (0, 0))
            b = smt.view(
                b_descriptor_load,
                (k * SUB_BLK_K, 0),
                (SUB_BLK_K, BLOCK_SIZE_N),
                (MICRO_K, MICRO_N),
            )
            accumulator += smt.dot(a, b)
        accumulator = smt.view(
            accumulator, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_N), (1, 1)
        )
        c = accumulator.to(c_ptr.dtype.element_ty)

        c_block_ptr = tl.make_block_ptr(
            base=c_ptr,
            shape=[M, N],
            strides=[stride_cm, stride_cn],
            offsets=[pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
            order=[1, 0],
        )
        tl.store(c_block_ptr, c, boundary_check=(0, 1))


def mm(a, b):
    if not a.is_contiguous():
        a = a.contiguous()
    # spestruct.pack lowering cannot handle a transposed source, so B's inner
    # (N) dim must be unit-stride. contiguous() is a no-op when N==1 (torch
    # treats size-1 dims as contiguous regardless of stride), so force a real
    # row-major copy whenever the N stride isn't already 1.
    if b.stride(1) != 1:
        b = torch.empty_like(b, memory_format=torch.contiguous_format).copy_(b)
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # M==1 GEMV fast path: true vfwmacc.vf intrinsic (pure llvm-direct sibling),
    # bypasses smt.dot (which SIGSEGVs on non-tile M). Qwen decode hits this 100%
    # (M=1, f16, N%64==0, K%64==0). Kernel writes f32 (vfwmacc accumulator);
    # narrow to f16 on host.
    if (M == 1 and a.dtype == torch.float16
            and a.is_contiguous() and b.is_contiguous()
            and b.stride(1) == 1
            and N % 64 == 0 and K % 64 == 0):
        NB = 64
        grid = (N // NB,)
        c_f32 = torch.empty((1, N), device=a.device, dtype=torch.float32)
        _mm_m1_host[grid](a, b, c_f32, K, N, NB=NB)
        c = c_f32.half()
        return c
    # M>1: mm_kernel uses smt.dot which SIGSEGVs on non-tile-aligned M (prefill
    # M=18, MICRO_M=16 → null store at vse16.v). Fall back to ATen native mm
    # (torch.addmm beta=0 bypasses aten::mm dispatch — see _arm/ops/mm.py:540).
    # Cannot call torch.mm() here (infinite recursion via torch.library override).
    return torch.addmm(c, a, b, beta=0, alpha=1)


def mm_out(a, b, *, out):
    if not a.is_contiguous():
        a = a.contiguous()
    # spestruct.pack lowering cannot handle a transposed source, so B's inner
    # (N) dim must be unit-stride. contiguous() is a no-op when N==1 (torch
    # treats size-1 dims as contiguous regardless of stride), so force a real
    # row-major copy whenever the N stride isn't already 1.
    if b.stride(1) != 1:
        b = torch.empty_like(b, memory_format=torch.contiguous_format).copy_(b)

    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape

    # M==1 GEMV fast path (see mm() for rationale). Kernel writes f32; narrow
    # to the out tensor's dtype (f16) on host.
    if (M == 1 and a.dtype == torch.float16
            and a.is_contiguous() and b.is_contiguous()
            and b.stride(1) == 1
            and N % 64 == 0 and K % 64 == 0):
        NB = 64
        grid = (N // NB,)
        c_f32 = torch.empty((1, N), device=a.device, dtype=torch.float32)
        _mm_m1_host[grid](a, b, c_f32, K, N, NB=NB)
        out.copy_(c_f32.half())
        return out

    # M>1 fallback to ATen native (see mm() for rationale). Use addmm with
    # out= to avoid infinite recursion via aten::mm.out dispatch.
    torch.addmm(out, a, b, beta=0, alpha=1, out=out)
    return out
