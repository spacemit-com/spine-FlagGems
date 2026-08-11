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
