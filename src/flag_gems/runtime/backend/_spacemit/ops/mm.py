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
#   - fptrunc : f32→f16 vector narrowing (llvm.fptrunc), direct f16 output
# Pure llvm-direct → routes to LLVMDirectTextCodegen sibling path, bypasses
# spine-opt bufferization (which rejects llvm.call_intrinsic as "unknown memory
# side effects"). program_id resolved at runtime via spine_grid(ctx, axis).
# Direct f16 output — no f32 workspace, no host-side conversion. Each program
# handles one NB=64 N-block; grid = (N // NB,).
# ---------------------------------------------------------------------------
@tle.raw_kernel
def mm_m1_raw(x_ptr: mem(f16), W_ptr: mem(f16), C_ptr: mem(f16, out=True),
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
    acc_h = tle.call_intrinsic("llvm.fptrunc", [acc], result_type="vector<[4]xf16>")
    gc = tle.llvm_gep(cbase, n_start, "f16")
    tle.call_intrinsic("llvm.store", [acc_h, gc], result_type="()")


@triton.jit(do_not_specialize=["K", "N", "NB"])
def _mm_m1_host(x, W, C, K, N, NB):
    _sr_call(mm_m1_raw, outputs=[], inputs=[x, W, C, K, N, NB])


# ---------------------------------------------------------------------------
# M==1 GEMV transposed-b fast path (vfwmacc.vv, pure llvm-direct sibling).
#   For b = W.T where W is stored (N, K) row-major. b has shape (K, N) with
#   stride (1, K) — i.e. b.stride(0) == 1. The kernel interprets the raw
#   pointer as (N, K) row-major (W's actual layout) and computes
#   C[0, n] = sum_k x[k] * W[n, k] via vfwmacc.vv (vector·vector widening FMA).
#   Each program handles NB output rows, 8 rows unrolled per inner iteration
#   for ILP. Same pattern as test_raw_mv_three_stage.py::mv_vfwmacc_ci_parallel.
#   No host-side copy — reads W's storage directly via strided-free row-major
#   access (W[n, k:k+64] is contiguous because K is W's inner dim).
# ---------------------------------------------------------------------------
@tle.raw_kernel
def mm_m1_t_raw(x_ptr: mem(f16), W_ptr: mem(f16), C_ptr: mem(f16, out=True),
                K: index, N: index, NB: index):
    vl = tle.llvm_const(64, "i64")
    zero_i64 = tle.llvm_const(0, "i64")
    zero_acc = tle.llvm_const("0.000000e+00", "vector<[4]xf32>")
    zero_f = tle.llvm_const("0.000000e+00", "f32")
    xbase = tle.llvm_base_ptr(x_ptr)
    wbase = tle.llvm_base_ptr(W_ptr)
    cbase = tle.llvm_base_ptr(C_ptr)
    n_base = tle.program_id(0) * NB
    n_end = n_base + NB
    for ni in tle.range(n_base, n_end, 8):
        acc0 = zero_acc
        acc1 = zero_acc
        acc2 = zero_acc
        acc3 = zero_acc
        acc4 = zero_acc
        acc5 = zero_acc
        acc6 = zero_acc
        acc7 = zero_acc
        for ki in tle.range(zero_i64, K, vl):
            gx = tle.llvm_gep(xbase, ki, "f16")
            vx = tle.call_intrinsic("llvm.load", [gx], result_type="vector<[4]xf16>")
            gw0 = tle.llvm_gep(wbase, ni * K + ki, "f16")
            gw1 = tle.llvm_gep(wbase, (ni + 1) * K + ki, "f16")
            gw2 = tle.llvm_gep(wbase, (ni + 2) * K + ki, "f16")
            gw3 = tle.llvm_gep(wbase, (ni + 3) * K + ki, "f16")
            gw4 = tle.llvm_gep(wbase, (ni + 4) * K + ki, "f16")
            gw5 = tle.llvm_gep(wbase, (ni + 5) * K + ki, "f16")
            gw6 = tle.llvm_gep(wbase, (ni + 6) * K + ki, "f16")
            gw7 = tle.llvm_gep(wbase, (ni + 7) * K + ki, "f16")
            vw0 = tle.call_intrinsic("llvm.load", [gw0], result_type="vector<[4]xf16>")
            vw1 = tle.call_intrinsic("llvm.load", [gw1], result_type="vector<[4]xf16>")
            vw2 = tle.call_intrinsic("llvm.load", [gw2], result_type="vector<[4]xf16>")
            vw3 = tle.call_intrinsic("llvm.load", [gw3], result_type="vector<[4]xf16>")
            vw4 = tle.call_intrinsic("llvm.load", [gw4], result_type="vector<[4]xf16>")
            vw5 = tle.call_intrinsic("llvm.load", [gw5], result_type="vector<[4]xf16>")
            vw6 = tle.call_intrinsic("llvm.load", [gw6], result_type="vector<[4]xf16>")
            vw7 = tle.call_intrinsic("llvm.load", [gw7], result_type="vector<[4]xf16>")
            acc0 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc0, vx, vw0, zero_i64, vl, zero_i64],
                                      result_type="vector<[4]xf32>")
            acc1 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc1, vx, vw1, zero_i64, vl, zero_i64],
                                      result_type="vector<[4]xf32>")
            acc2 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc2, vx, vw2, zero_i64, vl, zero_i64],
                                      result_type="vector<[4]xf32>")
            acc3 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc3, vx, vw3, zero_i64, vl, zero_i64],
                                      result_type="vector<[4]xf32>")
            acc4 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc4, vx, vw4, zero_i64, vl, zero_i64],
                                      result_type="vector<[4]xf32>")
            acc5 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc5, vx, vw5, zero_i64, vl, zero_i64],
                                      result_type="vector<[4]xf32>")
            acc6 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc6, vx, vw6, zero_i64, vl, zero_i64],
                                      result_type="vector<[4]xf32>")
            acc7 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc7, vx, vw7, zero_i64, vl, zero_i64],
                                      result_type="vector<[4]xf32>")
        s0 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc0], result_type="f32")
        s1 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc1], result_type="f32")
        s2 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc2], result_type="f32")
        s3 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc3], result_type="f32")
        s4 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc4], result_type="f32")
        s5 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc5], result_type="f32")
        s6 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc6], result_type="f32")
        s7 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc7], result_type="f32")
        h0 = tle.call_intrinsic("llvm.fptrunc", [s0], result_type="f16")
        h1 = tle.call_intrinsic("llvm.fptrunc", [s1], result_type="f16")
        h2 = tle.call_intrinsic("llvm.fptrunc", [s2], result_type="f16")
        h3 = tle.call_intrinsic("llvm.fptrunc", [s3], result_type="f16")
        h4 = tle.call_intrinsic("llvm.fptrunc", [s4], result_type="f16")
        h5 = tle.call_intrinsic("llvm.fptrunc", [s5], result_type="f16")
        h6 = tle.call_intrinsic("llvm.fptrunc", [s6], result_type="f16")
        h7 = tle.call_intrinsic("llvm.fptrunc", [s7], result_type="f16")
        tle.call_intrinsic("llvm.store", [h0, tle.llvm_gep(cbase, ni, "f16")], result_type="()")
        tle.call_intrinsic("llvm.store", [h1, tle.llvm_gep(cbase, ni + 1, "f16")], result_type="()")
        tle.call_intrinsic("llvm.store", [h2, tle.llvm_gep(cbase, ni + 2, "f16")], result_type="()")
        tle.call_intrinsic("llvm.store", [h3, tle.llvm_gep(cbase, ni + 3, "f16")], result_type="()")
        tle.call_intrinsic("llvm.store", [h4, tle.llvm_gep(cbase, ni + 4, "f16")], result_type="()")
        tle.call_intrinsic("llvm.store", [h5, tle.llvm_gep(cbase, ni + 5, "f16")], result_type="()")
        tle.call_intrinsic("llvm.store", [h6, tle.llvm_gep(cbase, ni + 6, "f16")], result_type="()")
        tle.call_intrinsic("llvm.store", [h7, tle.llvm_gep(cbase, ni + 7, "f16")], result_type="()")


@triton.jit(do_not_specialize=["K", "N", "NB"])
def _mm_m1_t_host(x, W, C, K, N, NB):
    _sr_call(mm_m1_t_raw, outputs=[], inputs=[x, W, C, K, N, NB])


# ---------------------------------------------------------------------------
# M>1 mm transposed-b kernel (prefill). Extends mm_m1_t_raw with an M-row
# loop: C[m,n] = sum_k a[m,k]*W[n,k]. Each program handles NB columns × all
# M rows; 8 columns unrolled, M iterated. W rows reused across m via L1 cache.
# Same vfwmacc.vv pattern. Qwen prefill (M=18/32, b=W.T) hits this 100%.
# ---------------------------------------------------------------------------
@tle.raw_kernel
def mm_mN_t_raw(a_ptr: mem(f16), W_ptr: mem(f16), C_ptr: mem(f16, out=True),
                M: index, K: index, N: index, NB: index, CS: index):
    vl = tle.llvm_const(64, "i64")
    zero_i64 = tle.llvm_const(0, "i64")
    zero_acc = tle.llvm_const("0.000000e+00", "vector<[4]xf32>")
    zero_f = tle.llvm_const("0.000000e+00", "f32")
    abase = tle.llvm_base_ptr(a_ptr)
    wbase = tle.llvm_base_ptr(W_ptr)
    cbase = tle.llvm_base_ptr(C_ptr)
    n_base = tle.program_id(0) * NB
    n_end = n_base + NB
    for ni in tle.range(n_base, n_end, 8):
        for m in tle.range(0, M, 1):
            acc0 = zero_acc
            acc1 = zero_acc
            acc2 = zero_acc
            acc3 = zero_acc
            acc4 = zero_acc
            acc5 = zero_acc
            acc6 = zero_acc
            acc7 = zero_acc
            for ki in tle.range(zero_i64, K, vl):
                ga = tle.llvm_gep(abase, m * K + ki, "f16")
                va = tle.call_intrinsic("llvm.load", [ga], result_type="vector<[4]xf16>")
                gw0 = tle.llvm_gep(wbase, ni * K + ki, "f16")
                gw1 = tle.llvm_gep(wbase, (ni + 1) * K + ki, "f16")
                gw2 = tle.llvm_gep(wbase, (ni + 2) * K + ki, "f16")
                gw3 = tle.llvm_gep(wbase, (ni + 3) * K + ki, "f16")
                gw4 = tle.llvm_gep(wbase, (ni + 4) * K + ki, "f16")
                gw5 = tle.llvm_gep(wbase, (ni + 5) * K + ki, "f16")
                gw6 = tle.llvm_gep(wbase, (ni + 6) * K + ki, "f16")
                gw7 = tle.llvm_gep(wbase, (ni + 7) * K + ki, "f16")
                vw0 = tle.call_intrinsic("llvm.load", [gw0], result_type="vector<[4]xf16>")
                vw1 = tle.call_intrinsic("llvm.load", [gw1], result_type="vector<[4]xf16>")
                vw2 = tle.call_intrinsic("llvm.load", [gw2], result_type="vector<[4]xf16>")
                vw3 = tle.call_intrinsic("llvm.load", [gw3], result_type="vector<[4]xf16>")
                vw4 = tle.call_intrinsic("llvm.load", [gw4], result_type="vector<[4]xf16>")
                vw5 = tle.call_intrinsic("llvm.load", [gw5], result_type="vector<[4]xf16>")
                vw6 = tle.call_intrinsic("llvm.load", [gw6], result_type="vector<[4]xf16>")
                vw7 = tle.call_intrinsic("llvm.load", [gw7], result_type="vector<[4]xf16>")
                acc0 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc0, va, vw0, zero_i64, vl, zero_i64],
                                          result_type="vector<[4]xf32>")
                acc1 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc1, va, vw1, zero_i64, vl, zero_i64],
                                          result_type="vector<[4]xf32>")
                acc2 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc2, va, vw2, zero_i64, vl, zero_i64],
                                          result_type="vector<[4]xf32>")
                acc3 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc3, va, vw3, zero_i64, vl, zero_i64],
                                          result_type="vector<[4]xf32>")
                acc4 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc4, va, vw4, zero_i64, vl, zero_i64],
                                          result_type="vector<[4]xf32>")
                acc5 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc5, va, vw5, zero_i64, vl, zero_i64],
                                          result_type="vector<[4]xf32>")
                acc6 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc6, va, vw6, zero_i64, vl, zero_i64],
                                          result_type="vector<[4]xf32>")
                acc7 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc7, va, vw7, zero_i64, vl, zero_i64],
                                          result_type="vector<[4]xf32>")
            s0 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc0], result_type="f32")
            s1 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc1], result_type="f32")
            s2 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc2], result_type="f32")
            s3 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc3], result_type="f32")
            s4 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc4], result_type="f32")
            s5 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc5], result_type="f32")
            s6 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc6], result_type="f32")
            s7 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc7], result_type="f32")
            h0 = tle.call_intrinsic("llvm.fptrunc", [s0], result_type="f16")
            h1 = tle.call_intrinsic("llvm.fptrunc", [s1], result_type="f16")
            h2 = tle.call_intrinsic("llvm.fptrunc", [s2], result_type="f16")
            h3 = tle.call_intrinsic("llvm.fptrunc", [s3], result_type="f16")
            h4 = tle.call_intrinsic("llvm.fptrunc", [s4], result_type="f16")
            h5 = tle.call_intrinsic("llvm.fptrunc", [s5], result_type="f16")
            h6 = tle.call_intrinsic("llvm.fptrunc", [s6], result_type="f16")
            h7 = tle.call_intrinsic("llvm.fptrunc", [s7], result_type="f16")
            cn0 = m * CS + ni
            tle.call_intrinsic("llvm.store", [h0, tle.llvm_gep(cbase, cn0, "f16")], result_type="()")
            tle.call_intrinsic("llvm.store", [h1, tle.llvm_gep(cbase, cn0 + 1, "f16")], result_type="()")
            tle.call_intrinsic("llvm.store", [h2, tle.llvm_gep(cbase, cn0 + 2, "f16")], result_type="()")
            tle.call_intrinsic("llvm.store", [h3, tle.llvm_gep(cbase, cn0 + 3, "f16")], result_type="()")
            tle.call_intrinsic("llvm.store", [h4, tle.llvm_gep(cbase, cn0 + 4, "f16")], result_type="()")
            tle.call_intrinsic("llvm.store", [h5, tle.llvm_gep(cbase, cn0 + 5, "f16")], result_type="()")
            tle.call_intrinsic("llvm.store", [h6, tle.llvm_gep(cbase, cn0 + 6, "f16")], result_type="()")
            tle.call_intrinsic("llvm.store", [h7, tle.llvm_gep(cbase, cn0 + 7, "f16")], result_type="()")


@triton.jit(do_not_specialize=["M", "K", "N", "NB", "CS"])
def _mm_mN_t_host(a, W, C, M, K, N, NB, CS):
    _sr_call(mm_mN_t_raw, outputs=[], inputs=[a, W, C, M, K, N, NB, CS])


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


def _mm_transposed_tail(a, b, c, M, K, N):
    """vfwmacc fast path for transposed b (b.stride(0)==1, b=W.T).
    Splits N into main tile (multiple of NB, vfwmacc) + tail (vfwmacc with
    shifted pointer views). Writes into c (pre-allocated (M,N)). Returns True.

    NB auto-bumped to keep grid <= 512 (K3 runtime limit: grid>512 corrupts —
    programs silently drop or write zeros). n_main=151552 (lm_head) with NB=512
    gives grid=296. n_tail=384 handled by vfwmacc kernel with grid=(1,), NB=n_tail.

    Tail uses strided views b[:, n_main:] / c[:, n_main:] whose data_ptr is
    shifted by n_main elements — kernel reads W[n_main+ni, ki] and writes
    c[m, n_main+ni] correctly. Constraint: n_tail % 8 == 0 (kernel 8-col unroll).
    """
    GRID_LIMIT = 512
    NB = 256
    if N // NB > GRID_LIMIT and N >= 512:
        NB = 512
    n_main = (N // NB) * NB
    n_tail = N - n_main
    if n_main > 0:
        if M == 1:
            grid = (n_main // NB,)
            _mm_m1_t_host[grid](a, b, c, K, n_main, NB=NB)
        else:
            grid = (n_main // NB,)
            _mm_mN_t_host[grid](a, b, c, M, K, n_main, NB=NB, CS=N)
    if n_tail > 0:
        # tail: vfwmacc kernel with shifted pointer views (no temp, no native addmm).
        # b[:, n_main:] has stride (1, K) and data_ptr = b.data_ptr() + n_main*K*2,
        # which is exactly W[n_main:, :] row-major — kernel's wbase[ni*K+ki] reads
        # W[n_main+ni, ki]. c[:, n_main:] similarly shifts cbase so writes land at
        # c[m, n_main+ni]. CS=N (full row stride) keeps c layout correct.
        b_tail = b[:, n_main:]
        c_tail = c[:, n_main:]
        if n_tail % 8 == 0:
            if M == 1:
                _mm_m1_t_host[(1,)](a, b_tail, c_tail, K, n_tail, NB=n_tail)
            else:
                _mm_mN_t_host[(1,)](a, b_tail, c_tail, M, K, n_tail, NB=n_tail, CS=N)
        else:
            # n_tail not multiple of 8: fall back to native addmm (rare).
            tmp = torch.empty(M, n_tail, dtype=a.dtype, device=a.device)
            torch.addmm(tmp, a, b_tail, beta=0, alpha=1, out=tmp)
            c[:, n_main:].copy_(tmp)
    return True


def mm(a, b):
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # vfwmacc fast paths (pure llvm-direct sibling, no host-side copy):
    #   M==1 decode:  mm_m1_raw (vfwmacc.vf, contiguous b) or
    #                 mm_m1_t_raw (vfwmacc.vv, transposed b=W.T)
    #   M>1  prefill: mm_mN_t_raw (vfwmacc.vv, transposed b=W.T, M-row loop)
    # Qwen hits transposed b 100% (weights (out,in) row-major, mm(x, W.T)).
    # Forcing a copy costs ~1.6s/decode-step on K3 — never copy here.
    # Tail handling: N%8==0 suffices (main tile = (N//256)*256 via vfwmacc,
    #   tail <256 cols via native addmm). lm_head N=151936 (vocab) hits this.
    if (a.dtype == torch.float16 and a.is_contiguous()
            and K % 64 == 0):
        if M == 1:
            if b.is_contiguous() and b.stride(1) == 1 and N % 64 == 0:
                NB = 64
                grid = (N // NB,)
                _mm_m1_host[grid](a, b, c, K, N, NB=NB)
                return c
            if b.stride(0) == 1 and N % 8 == 0:
                _mm_transposed_tail(a, b, c, M, K, N)
                return c
        else:
            # M>1 prefill: vfwmacc.vv with M-row loop, transposed b only.
            if b.stride(0) == 1 and N % 8 == 0:
                _mm_transposed_tail(a, b, c, M, K, N)
                return c
    # Fallback: ATen native mm. torch.addmm beta=0 bypasses aten::mm dispatch
    # (see _arm/ops/mm.py:540). Cannot call torch.mm() (infinite recursion via
    # torch.library override). Handles non-f16, non-contiguous, non-tile-aligned.
    return torch.addmm(c, a, b, beta=0, alpha=1)


def mm_out(a, b, *, out):
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape

    if (a.dtype == torch.float16 and a.is_contiguous()
            and K % 64 == 0):
        if M == 1:
            if b.is_contiguous() and b.stride(1) == 1 and N % 64 == 0:
                NB = 64
                grid = (N // NB,)
                _mm_m1_host[grid](a, b, out, K, N, NB=NB)
                return out
            if b.stride(0) == 1 and N % 8 == 0:
                _mm_transposed_tail(a, b, out, M, K, N)
                return out
        else:
            if b.stride(0) == 1 and N % 8 == 0:
                _mm_transposed_tail(a, b, out, M, K, N)
                return out

    torch.addmm(out, a, b, beta=0, alpha=1, out=out)
    return out
