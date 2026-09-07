"""vfwmacc GQA attention kernels for SpacemiT K3 (arch 0xA064, vlen=1024).

Per-head grid (Hq,) with GQA via hk = hq // Nrep (skips repeat_kv).
D=128 handled by 2× 64-chunk (Qwen3 head_dim). f16 input → f32 acc
(vfwmacc widening) → fptrunc → f16 output.

LLVM-direct path (LLVMDirectTextCodegen) — bypasses spine-opt bufferization.
No math intrinsics (LLVM 23 removed llvm.intr.exp/maxnum); softmax uses
default spine_raw path with vexp instead.

Validated on Qwen3-0.6B (K3): QKT kernel 1.1ms vs native 15.4ms,
AV kernel 0.81ms vs native 15.4ms.
"""
import triton
import triton.language as tl
import triton.language.extra.spine_raw as tle
from triton.language.extra.spine_raw import call as _sr_call
from triton.language.extra.spine_raw import mem, index, f16, f32


# ── QKT: Q @ K^T, per-head, D=128 (2× 64-chunk) ────────────────────────────
# Q:(Hq,Mq,D) K:(Hkv,Mk,D) O:(Hq,Mq,Mk).  Hoist Q loads out of n loop.
@tle.raw_kernel
def attn_qkt_gqa_raw(Q_ptr: mem(f16), K_ptr: mem(f16), O_ptr: mem(f16, out=True),
                     Mq: index, Mk: index, D: index, Hq: index, Hkv: index, Nrep: index):
    vl = tle.llvm_const(64, "i64")
    zero_i64 = tle.llvm_const(0, "i64")
    zero_acc = tle.llvm_const(0.0, "vector<[4]xf32>")
    zero_f = tle.llvm_const(0.0, "f32")
    qbase = tle.llvm_base_ptr(Q_ptr)
    kbase = tle.llvm_base_ptr(K_ptr)
    obase = tle.llvm_base_ptr(O_ptr)
    hq = tle.program_id(0)
    hk = tle.call_intrinsic("llvm.sdiv", [hq, Nrep], result_type="i64")
    MqD = Mq * D
    MkD = Mk * D
    MqMk = Mq * Mk
    for m in tle.range(0, Mq, 1):
        gq0 = tle.llvm_gep(qbase, hq * MqD + m * D, "f16")
        qv0 = tle.call_intrinsic("llvm.load", [gq0], result_type="vector<[4]xf16>")
        gq1 = tle.llvm_gep(qbase, hq * MqD + m * D + 64, "f16")
        qv1 = tle.call_intrinsic("llvm.load", [gq1], result_type="vector<[4]xf16>")
        for n in tle.range(0, Mk, 1):
            gk0 = tle.llvm_gep(kbase, hk * MkD + n * D, "f16")
            kv0 = tle.call_intrinsic("llvm.load", [gk0], result_type="vector<[4]xf16>")
            prod0 = tle.call_intrinsic("llvm.riscv.vfwmacc", [zero_acc, qv0, kv0, zero_i64, vl, zero_i64],
                                       result_type="vector<[4]xf32>")
            dot = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, prod0], result_type="f32")
            gk1 = tle.llvm_gep(kbase, hk * MkD + n * D + 64, "f16")
            kv1 = tle.call_intrinsic("llvm.load", [gk1], result_type="vector<[4]xf16>")
            prod1 = tle.call_intrinsic("llvm.riscv.vfwmacc", [zero_acc, qv1, kv1, zero_i64, vl, zero_i64],
                                       result_type="vector<[4]xf32>")
            dot = tle.call_intrinsic("llvm.vector.reduce.fadd", [dot, prod1], result_type="f32")
            dot_h = tle.call_intrinsic("llvm.fptrunc", [dot], result_type="f16")
            go = tle.llvm_gep(obase, hq * MqMk + m * Mk + n, "f16")
            tle.call_intrinsic("llvm.store", [dot_h, go], result_type="()")


@triton.jit(do_not_specialize=["Mq", "Mk", "D", "Hq", "Hkv", "Nrep"])
def _attn_qkt_gqa_host(Q, K, O, Mq, Mk, D, Hq, Hkv, Nrep):
    _sr_call(attn_qkt_gqa_raw, outputs=[], inputs=[Q, K, O, Mq, Mk, D, Hq, Hkv, Nrep])


# ── AV: attn @ V, per-head, N=128 (2× 64-chunk) ────────────────────────────
# A:(Hq,M,K) V:(Hkv,K,N) O:(Hq,M,N).  Two independent accs for N=128.
@tle.raw_kernel
def attn_av_gqa_raw(A_ptr: mem(f16), V_ptr: mem(f16), O_ptr: mem(f16, out=True),
                    M: index, K: index, N: index, Hq: index, Hkv: index, Nrep: index):
    vl = tle.llvm_const(64, "i64")
    zero_i64 = tle.llvm_const(0, "i64")
    zero_acc = tle.llvm_const(0.0, "vector<[4]xf32>")
    abase = tle.llvm_base_ptr(A_ptr)
    vbase = tle.llvm_base_ptr(V_ptr)
    obase = tle.llvm_base_ptr(O_ptr)
    hq = tle.program_id(0)
    hk = tle.call_intrinsic("llvm.sdiv", [hq, Nrep], result_type="i64")
    MK = M * K
    KN = K * N
    MN = M * N
    for m in tle.range(0, M, 1):
        acc0 = zero_acc
        acc1 = zero_acc
        for k in tle.range(0, K, 1):
            ga = tle.llvm_gep(abase, hq * MK + m * K + k, "f16")
            a_scalar = tle.call_intrinsic("llvm.load", [ga], result_type="f16")
            gv0 = tle.llvm_gep(vbase, hk * KN + k * N, "f16")
            v_vec0 = tle.call_intrinsic("llvm.load", [gv0], result_type="vector<[4]xf16>")
            acc0 = tle.call_intrinsic("llvm.riscv.vfwmacc.vf", [acc0, a_scalar, v_vec0, zero_i64, vl, zero_i64],
                                      result_type="vector<[4]xf32>")
            gv1 = tle.llvm_gep(vbase, hk * KN + k * N + 64, "f16")
            v_vec1 = tle.call_intrinsic("llvm.load", [gv1], result_type="vector<[4]xf16>")
            acc1 = tle.call_intrinsic("llvm.riscv.vfwmacc.vf", [acc1, a_scalar, v_vec1, zero_i64, vl, zero_i64],
                                      result_type="vector<[4]xf32>")
        acc0_h = tle.call_intrinsic("llvm.fptrunc", [acc0], result_type="vector<[4]xf16>")
        acc1_h = tle.call_intrinsic("llvm.fptrunc", [acc1], result_type="vector<[4]xf16>")
        go0 = tle.llvm_gep(obase, hq * MN + m * N, "f16")
        tle.call_intrinsic("llvm.store", [acc0_h, go0], result_type="()")
        go1 = tle.llvm_gep(obase, hq * MN + m * N + 64, "f16")
        tle.call_intrinsic("llvm.store", [acc1_h, go1], result_type="()")


@triton.jit(do_not_specialize=["M", "K", "N", "Hq", "Hkv", "Nrep"])
def _attn_av_gqa_host(A, V, O, M, K, N, Hq, Hkv, Nrep):
    _sr_call(attn_av_gqa_raw, outputs=[], inputs=[A, V, O, M, K, N, Hq, Hkv, Nrep])


# ── Fused M=1: QKT + softmax + AV in one kernel (decode-only) ───────────────
# Default-path spine_raw (vexp/vmacc/vreduce). grid=(Hq,), host passes hq/hk
# as index scalars (multi-core via JIT wrapper, _to_handle casts i32→index).
# Decode Mq=1: no causal mask (query at pos Mk-1 attends to all keys 0..Mk-1).
# 2-phase softmax: dots→scratch (Hq,Mk) f32 → vreduce_max → vexp → normalize → AV.
# D=128 = 2× 64-chunk (Qwen3 head_dim). f16 input → f32 acc → f16 output.
@tle.raw_kernel
def attn_m1_fused_raw(Q_ptr: mem(f16), K_ptr: mem(f16), V_ptr: mem(f16),
                      Scratch_ptr: mem(f32), Scale_ptr: mem(f32),
                      O_ptr: mem(f16, out=True),
                      Mk: index, D: index, hq: index, hk: index):
    nvl = tle.vconfig(-1, 1)  # VL=64 (f16/f32 element count, lmul=1)

    scale = tle.sload(Scale_ptr, 0, dtype=f32)

    MkD = Mk * D

    # Load Q[hq, 0:64] and Q[hq, 64:128] as f32 (hoisted)
    qv0_h = tle.vload(Q_ptr, hq * D, dtype=f16)
    qv0 = tle.cast(qv0_h, f32)
    qv1_h = tle.vload(Q_ptr, hq * D + 64, dtype=f16)
    qv1 = tle.cast(qv1_h, f32)

    # Phase 1: QKT dots → scratch (default-path: cast + mul + vreduce_sum)
    for k in tle.range(0, Mk, 1):
        kv0_h = tle.vload(K_ptr, hk * MkD + k * D, dtype=f16)
        kv0 = tle.cast(kv0_h, f32)
        prod0 = qv0 * kv0
        dot = tle.vreduce_sum(prod0)
        kv1_h = tle.vload(K_ptr, hk * MkD + k * D + 64, dtype=f16)
        kv1 = tle.cast(kv1_h, f32)
        prod1 = qv1 * kv1
        dot = dot + tle.vreduce_sum(prod1)
        dot = dot * scale
        tle.sstore(Scratch_ptr, hq * Mk + k, dot)

    # Phase 2: global max (default-path, non-scalable f32)
    nvl2 = tle.vconfig(-1, 2)  # VL=128 for faster reduce on small Mk
    Kfloor = (Mk // nvl2) * nvl2
    max_vec = tle.vzero(f32) - 1.0e38
    for k in tle.range(0, Kfloor, nvl2):
        v = tle.vload(Scratch_ptr, hq * Mk + k, dtype=f32)
        max_vec = tle.vmax(max_vec, v)
    for k in tle.range(Kfloor, Mk, nvl2):
        _ = tle.vconfig(Mk - k, 2)
        vt = tle.vload(Scratch_ptr, hq * Mk + k, dtype=f32, fill=-1.0e38)
        max_vec = tle.vmax(max_vec, vt)
    _ = tle.vconfig(-1, 2)
    max_s = tle.vreduce_max(max_vec)

    # Phase 3a: exp → scratch, accumulate sum (default-path)
    sum_vec = tle.vzero(f32)
    for k in tle.range(0, Kfloor, nvl2):
        v = tle.vload(Scratch_ptr, hq * Mk + k, dtype=f32)
        ev = tle.vexp(v - max_s)
        tle.vstore(Scratch_ptr, hq * Mk + k, ev)
        sum_vec = sum_vec + ev
    for k in tle.range(Kfloor, Mk, nvl2):
        _ = tle.vconfig(Mk - k, 2)
        vt = tle.vload(Scratch_ptr, hq * Mk + k, dtype=f32, fill=-1.0e38)
        evt = tle.vexp(vt - max_s)
        tle.vstore(Scratch_ptr, hq * Mk + k, evt)
        sum_vec = sum_vec + evt
    _ = tle.vconfig(-1, 2)
    sum_exp = tle.vreduce_sum(sum_vec)
    inv = 1.0 / sum_exp

    # Phase 3b: normalize exp in scratch by inv (default-path)
    for k in tle.range(0, Kfloor, nvl2):
        ev = tle.vload(Scratch_ptr, hq * Mk + k, dtype=f32)
        tle.vstore(Scratch_ptr, hq * Mk + k, ev * inv)
    for k in tle.range(Kfloor, Mk, nvl2):
        _ = tle.vconfig(Mk - k, 2)
        ev = tle.vload(Scratch_ptr, hq * Mk + k, dtype=f32, fill=0.0)
        tle.vstore(Scratch_ptr, hq * Mk + k, ev * inv)
    _ = tle.vconfig(-1, 1)

    # Phase 4: AV (default-path: vv_f * ek scalar broadcast, accumulate)
    acc0 = tle.vzero(f32)
    acc1 = tle.vzero(f32)
    for k in tle.range(0, Mk, 1):
        ek = tle.sload(Scratch_ptr, hq * Mk + k, dtype=f32)
        vv0_h = tle.vload(V_ptr, hk * MkD + k * D, dtype=f16)
        vv0 = tle.cast(vv0_h, f32)
        acc0 = acc0 + vv0 * ek
        vv1_h = tle.vload(V_ptr, hk * MkD + k * D + 64, dtype=f16)
        vv1 = tle.cast(vv1_h, f32)
        acc1 = acc1 + vv1 * ek

    # Phase 5: trunc + store (default-path: cast f32→f16, vstore)
    out0 = tle.cast(acc0, f16)
    tle.vstore(O_ptr, hq * D, out0)
    out1 = tle.cast(acc1, f16)
    tle.vstore(O_ptr, hq * D + 64, out1)


@triton.jit(do_not_specialize=["Mk", "D"])
def _attn_m1_fused_host(Q, K, V, Scratch, Scale, O, Mk, D, Nrep: tl.constexpr, Hq: tl.constexpr):
    hq = tl.program_id(0)
    hk = hq // Nrep
    _sr_call(attn_m1_fused_raw, outputs=[], inputs=[Q, K, V, Scratch, Scale, O, Mk, D, hq, hk])


# ── 3-stage fuse for prefill (Mq>1): QKT + softmax + AV in ONE launch ───────
# Pattern: mix LLVMDirectTextCodegen (QKT, AV — fast vfwmacc) and default-path
# (softmax — has vexp). Each stage is a separate @tle.raw_kernel, called via
# _sr_call from one @triton.jit host. grid=(Hq,), each program handles one head.
#
# Stage 1 (LLVM-direct): QKT with scale + causal mask, output f32
# Stage 2 (default-path): softmax with vexp, f32 → f16
# Stage 3 (LLVM-direct): AV (reuse attn_av_gqa_raw), f16 attn @ f16 V → f16 out
#
# Causal mask precomputed on host (cached per shape), passed as mem(f32).
# Scale passed as 1-element mem(f32).
@tle.raw_kernel
def attn_qkt_fused_raw(Q_ptr: mem(f16), K_ptr: mem(f16), Mask_ptr: mem(f32),
                       Scale_ptr: mem(f32), QKT_ptr: mem(f32, out=True),
                       Mq: index, Mk: index, D: index, Nrep: index):
    vl = tle.llvm_const(64, "i64")
    zero_i64 = tle.llvm_const(0, "i64")
    zero_acc = tle.llvm_const(0.0, "vector<[4]xf32>")
    zero_f = tle.llvm_const(0.0, "f32")
    qbase = tle.llvm_base_ptr(Q_ptr)
    kbase = tle.llvm_base_ptr(K_ptr)
    mbase = tle.llvm_base_ptr(Mask_ptr)
    sbase = tle.llvm_base_ptr(Scale_ptr)
    obase = tle.llvm_base_ptr(QKT_ptr)
    hq = tle.program_id(0)
    hk = tle.call_intrinsic("llvm.sdiv", [hq, Nrep], result_type="i64")
    MqD = Mq * D
    MkD = Mk * D
    MqMk = Mq * Mk
    # Load scale (1-element memref)
    scale = tle.call_intrinsic("llvm.load", [sbase], result_type="f32")
    for m in tle.range(0, Mq, 1):
        gq0 = tle.llvm_gep(qbase, hq * MqD + m * D, "f16")
        qv0 = tle.call_intrinsic("llvm.load", [gq0], result_type="vector<[4]xf16>")
        gq1 = tle.llvm_gep(qbase, hq * MqD + m * D + 64, "f16")
        qv1 = tle.call_intrinsic("llvm.load", [gq1], result_type="vector<[4]xf16>")
        for n in tle.range(0, Mk, 1):
            gk0 = tle.llvm_gep(kbase, hk * MkD + n * D, "f16")
            kv0 = tle.call_intrinsic("llvm.load", [gk0], result_type="vector<[4]xf16>")
            prod0 = tle.call_intrinsic("llvm.riscv.vfwmacc", [zero_acc, qv0, kv0, zero_i64, vl, zero_i64],
                                       result_type="vector<[4]xf32>")
            dot = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, prod0], result_type="f32")
            gk1 = tle.llvm_gep(kbase, hk * MkD + n * D + 64, "f16")
            kv1 = tle.call_intrinsic("llvm.load", [gk1], result_type="vector<[4]xf16>")
            prod1 = tle.call_intrinsic("llvm.riscv.vfwmacc", [zero_acc, qv1, kv1, zero_i64, vl, zero_i64],
                                       result_type="vector<[4]xf32>")
            dot = tle.call_intrinsic("llvm.vector.reduce.fadd", [dot, prod1], result_type="f32")
            # Apply scale
            dot = tle.call_intrinsic("llvm.fmul", [dot, scale], result_type="f32")
            # Apply causal mask (0.0 or -1e38)
            gm = tle.llvm_gep(mbase, m * Mk + n, "f32")
            mask_val = tle.call_intrinsic("llvm.load", [gm], result_type="f32")
            dot = tle.call_intrinsic("llvm.fadd", [dot, mask_val], result_type="f32")
            # Store f32
            go = tle.llvm_gep(obase, hq * MqMk + m * Mk + n, "f32")
            tle.call_intrinsic("llvm.store", [dot, go], result_type="()")


# Stage 2: softmax (default-path, has vexp). Reads f32 QKT, writes f16 attn.
# Per program (one head, hq passed as index param from host).
# 3-phase softmax: max → exp+sum → normalize+store. Tail via 2nd range loop.
@tle.raw_kernel
def attn_softmax_fused_raw(QKT_ptr: mem(f32), Out_ptr: mem(f16, out=True),
                           Mq: index, Mk: index, hq: index):
    nvl = tle.vconfig(-1, 1)  # VL=32 for f32 (vlen=1024, f32=32 elem/reg)
    MqMk_base = hq * Mq * Mk
    for m in tle.range(0, Mq, 1):
        row_off = MqMk_base + m * Mk
        Kfloor = (Mk // nvl) * nvl
        # Phase 1: max
        max_vec = tle.vzero(f32) - 1.0e38
        for k in tle.range(0, Kfloor, nvl):
            v = tle.vload(QKT_ptr, row_off + k, dtype=f32)
            max_vec = tle.vmax(max_vec, v)
        for k in tle.range(Kfloor, Mk, nvl):
            _ = tle.vconfig(Mk - k, 1)
            vt = tle.vload(QKT_ptr, row_off + k, dtype=f32, fill=-1.0e38)
            max_vec = tle.vmax(max_vec, vt)
        _ = tle.vconfig(-1, 1)
        max_s = tle.vreduce_max(max_vec)
        # Phase 2: exp + sum (store exp back to QKT scratch in-place f32)
        sum_vec = tle.vzero(f32)
        for k in tle.range(0, Kfloor, nvl):
            v = tle.vload(QKT_ptr, row_off + k, dtype=f32)
            ev = tle.vexp(v - max_s)
            tle.vstore(QKT_ptr, row_off + k, ev)
            sum_vec = sum_vec + ev
        for k in tle.range(Kfloor, Mk, nvl):
            _ = tle.vconfig(Mk - k, 1)
            vt = tle.vload(QKT_ptr, row_off + k, dtype=f32, fill=-1.0e38)
            evt = tle.vexp(vt - max_s)
            tle.vstore(QKT_ptr, row_off + k, evt)
            sum_vec = sum_vec + evt
        _ = tle.vconfig(-1, 1)
        sum_exp = tle.vreduce_sum(sum_vec)
        inv = 1.0 / sum_exp
        # Phase 3: normalize + store f16
        for k in tle.range(0, Kfloor, nvl):
            ev = tle.vload(QKT_ptr, row_off + k, dtype=f32)
            out_h = tle.cast(ev * inv, f16)
            tle.vstore(Out_ptr, row_off + k, out_h)
        for k in tle.range(Kfloor, Mk, nvl):
            _ = tle.vconfig(Mk - k, 1)
            ev = tle.vload(QKT_ptr, row_off + k, dtype=f32, fill=0.0)
            out_h = tle.cast(ev * inv, f16)
            tle.vstore(Out_ptr, row_off + k, out_h)
        _ = tle.vconfig(-1, 1)


# 3-stage fuse host: single launch, grid=(Hq,). Mixes 2 codegen paths.
# Only attn_av_gqa_raw uses tle.program_id (LLVM-direct). QKT (LLVM-direct)
# and softmax (default-path) receive hq/hk as params from host to avoid
# spine_grid redefinition (2 LLVM-direct kernels with tle.program_id conflict).
@triton.jit(do_not_specialize=["Mq", "Mk", "D", "Nrep", "Hq", "Hkv"])
def _attn_fused_prefill_host(Q, K, V, Mask, Scale, QKT_scratch, Attn_scratch, O,
                             Mq, Mk, D, Nrep, Hq, Hkv):
    hq = tl.program_id(0)
    _sr_call(attn_qkt_fused_raw, outputs=[],
             inputs=[Q, K, Mask, Scale, QKT_scratch, Mq, Mk, D, Nrep])
    _sr_call(attn_softmax_fused_raw, outputs=[],
             inputs=[QKT_scratch, Attn_scratch, Mq, Mk, hq])
    _sr_call(attn_av_gqa_raw, outputs=[],
             inputs=[Attn_scratch, V, O, Mq, Mk, D, Hq, Hkv, Nrep])
