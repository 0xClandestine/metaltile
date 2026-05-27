//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Unary elementwise benchmark — #[kernel] DSL vs MLX metal/unary.metal

use metaltile::kernel;

#[kernel]
pub fn mt_exp<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], exp(load(a[idx])));
}

#[kernel]
pub fn mt_log<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], log(load(a[idx])));
}

#[kernel]
pub fn mt_sqrt<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], sqrt(load(a[idx])));
}

#[kernel]
pub fn mt_rsqrt<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], rsqrt(load(a[idx])));
}

#[kernel]
pub fn mt_abs<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], abs(load(a[idx])));
}

#[kernel]
pub fn mt_silu<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], silu(load(a[idx])));
}

#[kernel]
pub fn mt_gelu<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], gelu(load(a[idx])));
}

#[kernel]
pub fn mt_relu<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], relu(load(a[idx])));
}

#[kernel]
pub fn mt_cos<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], cos(load(a[idx])));
}

#[kernel]
pub fn mt_sin<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], sin(load(a[idx])));
}

#[kernel]
pub fn mt_ceil<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], ceil(load(a[idx])));
}

#[kernel]
pub fn mt_floor<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], floor(load(a[idx])));
}

#[kernel]
pub fn mt_erf<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], erf(load(a[idx])));
}

#[kernel]
pub fn mt_exp2<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], exp2(load(a[idx])));
}

#[kernel]
pub fn mt_log2<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], log2(load(a[idx])));
}

#[kernel]
pub fn mt_sign<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], sign(load(a[idx])));
}

#[kernel]
pub fn mt_round<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], round(load(a[idx])));
}

#[kernel]
pub fn mt_neg<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], -load(a[idx]));
}

#[kernel]
pub fn mt_recip<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], 1.0f32.cast::<T>() / load(a[idx]));
}

#[kernel]
pub fn mt_square<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    let x = load(a[idx]);
    store(out[idx], x * x);
}

#[kernel]
pub fn mt_sigmoid<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    // Compute in f32 to match MLX precision (convert back to T at store).
    let x = load(a[idx]).cast::<f32>();
    let result = 1.0f32 / (1.0f32 + exp(-x));
    store(out[idx], result.cast::<T>());
}

#[kernel]
pub fn mt_log1p<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    let x = load(a[idx]);
    store(out[idx], log(1.0f32.cast::<T>() + x));
}

// ─── Transcendental ops shipped as discrete MLX kernels ───────────────
// Every op below has a matching `instantiate_unary_float` in MLX's
// unary.metal and produces a kernel named `v_<Op>{tn}{tn}`.

#[kernel]
pub fn mt_sinh<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], sinh(load(a[idx])));
}

#[kernel]
pub fn mt_cosh<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], cosh(load(a[idx])));
}

#[kernel]
pub fn mt_tan<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], tan(load(a[idx])));
}

#[kernel]
pub fn mt_tanh_op<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], tanh(load(a[idx])));
}

#[kernel]
pub fn mt_asin<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], asin(load(a[idx])));
}

#[kernel]
pub fn mt_atan<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], atan(load(a[idx])));
}

#[kernel]
pub fn mt_asinh<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], asinh(load(a[idx])));
}

#[kernel]
pub fn mt_acos<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], acos(load(a[idx])));
}

#[kernel]
pub fn mt_trunc<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], trunc(load(a[idx])));
}

#[kernel]
pub fn mt_acosh<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], acosh(load(a[idx])));
}

#[kernel]
pub fn mt_atanh<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], atanh(load(a[idx])));
}

#[kernel]
pub fn mt_expm1<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], expm1(load(a[idx])));
}

#[kernel]
pub fn mt_log10<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], log10(load(a[idx])));
}

#[kernel]
pub fn mt_erfinv<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], erfinv(load(a[idx])));
}

// Numerically-stable softplus: softplus(x) = max(x, 0) + log1p(exp(-|x|)).
// Avoids overflow at large positive x and underflow at large negative x —
// the naive `log(1 + exp(x))` blows up for x > ~80 (f32) / ~10 (f16).
// MLX has no dedicated softplus kernel (it composes log1p + exp at the
// graph layer); FFAI Ops.softplus calls this fused per-element variant
// directly because it lives on Mamba 2's hot path (`dt = softplus(dt_raw)`).
#[kernel]
pub fn mt_softplus<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    let x = load(a[idx]).cast::<f32>();
    let zero = 0.0f32;
    let pos = x > zero;
    let m = select(pos, x, zero);
    let ax = select(pos, x, zero - x);
    let r = m + log(1.0f32 + exp(zero - ax));
    store(out[idx], r.cast::<T>());
}

/// Cast a tensor of model dtype `T` to fp32 in-place per element. One
/// thread per element. Used by callers that need to mix fp32 state with
/// bf16 / f16 model activations on the GPU without a host round-trip —
/// the fused GDN prep step is the immediate consumer (its cache state
/// stays fp32 to avoid the 7-bit-mantissa drift over long decodes, but
/// the model activations into the kernel are bf16).
#[kernel]
pub fn mt_cast_to_f32<T>(input: Tensor<T>, out: Tensor<f32>) {
    let idx = program_id(0);
    store(out[idx], load(input[idx]).cast::<f32>());
}

/// Fused silu + cast-to-f32. Replaces the `silu(bf16) → cast_to_f32`
/// two-dispatch chain in FFAI's batched-prefill GDN inner loop with a
/// single dispatch: read bf16/f16, apply silu, write f32.
///
/// silu(x) = x · sigmoid(x) = x · (1 / (1 + exp(-x))) computed at f32
/// precision to match the bf16 → fp32 + silu → fp32 chain bit-for-bit
/// (modulo rounding mode on the final write — same as the standalone
/// silu kernel).
///
/// Saves T·30 ≈ 15k dispatches per Qwen3.6-A3B prefill at T=512 (one
/// silu + one cast per GDN-layer per-token iter → one fused dispatch).
#[kernel]
pub fn mt_silu_cast_to_f32<T>(input: Tensor<T>, out: Tensor<f32>) {
    let idx = program_id(0);
    let x = load(input[idx]).cast::<f32>();
    let sig = 1.0f32 / (1.0f32 + exp(0.0f32 - x));
    store(out[idx], x * sig);
}

/// Fused scalar-sigmoid fan-out + FMA. Computes
///   `out[i] = base[i] + sigmoid(gate[0]) * value[i]`
/// for `i in 0..hidden`, broadcasting the scalar `gate` across the
/// `[hidden]` vectors. One thread per output element; the scalar
/// re-loads through the GPU L1 cache so the broadcast is free.
///
/// Replaces FFAI's shared-expert host detour: `gateLogit.toFloatArray()`
/// + host `sigmoid()` + `Tensor.filled([hidden])` + `Ops.mul` + `Ops.add`
/// + a `commit + wait` to ensure the scalar is resident. With this
/// kernel the entire fan-out stays on the GPU and the command buffer
/// the gate was produced on no longer needs a host stall before the
/// next layer queues work.
///
/// Inputs are all in model dtype `T` (typically bf16 on Qwen3.6); the
/// internal accumulation widens to fp32 via the load-side `.cast` to
/// preserve sigmoid precision near saturation.
#[kernel]
pub fn mt_sigmoid_scalar_fma<T>(
    gate: Tensor<T>,
    value: Tensor<T>,
    base: Tensor<T>,
    out: Tensor<T>,
) {
    let idx = program_id(0);
    let gx = load(gate[0]).cast::<f32>();
    let g = 1.0f32 / (1.0f32 + exp(0.0f32 - gx));
    let v = load(value[idx]).cast::<f32>();
    let b = load(base[idx]).cast::<f32>();
    store(out[idx], (b + g * v).cast::<T>());
}

/// Fused elementwise sigmoid-scalar FMA WITH residual add. Computes
///   `out[i] = residual[i] + base[i] + sigmoid(gate[0]) * value[i]`
/// in one dispatch. Used by Qwen3.6-A3B's post-MoE-FFN site to
/// collapse the existing two-dispatch chain:
///   1. `mt_sigmoid_scalar_fma(gate, sharedOut, routed)` → ffnOut
///   2. `mt_add(postMix, ffnOut)`                       → result
/// into a single dispatch that reads `routed`, `sharedOut`, and
/// `postMix` once each and writes `result` once. Saves one full
/// `[hidden]` DRAM roundtrip on the intermediate `ffnOut` plus one
/// dispatch per MoE layer per token (×40 layers for Qwen3.6-A3B).
///
/// Same precision contract as `mt_sigmoid_scalar_fma`: model dtype
/// `T` on the read+write boundary, fp32 accumulation internally so
/// the sigmoid stays accurate at saturation.
#[kernel]
pub fn mt_sigmoid_scalar_fma_residual<T>(
    gate: Tensor<T>,
    value: Tensor<T>,
    base: Tensor<T>,
    residual: Tensor<T>,
    out: Tensor<T>,
) {
    let idx = program_id(0);
    let gx = load(gate[0]).cast::<f32>();
    let g = 1.0f32 / (1.0f32 + exp(0.0f32 - gx));
    let v = load(value[idx]).cast::<f32>();
    let b = load(base[idx]).cast::<f32>();
    let r = load(residual[idx]).cast::<f32>();
    store(out[idx], (r + b + g * v).cast::<T>());
}

/// Scalar-broadcast FMA. Computes
///   `out[i] = base[i] + scalar[0] * value[i]`
/// for `i in 0..n`, broadcasting a 1-element scalar buffer across the
/// `[n]` vectors. One thread per output element; the scalar re-loads
/// through the GPU L1 cache so the broadcast is free.
///
/// Replaces FFAI's MoE per-expert weighted-add chain at decode T=1:
/// instead of `Tensor.filled([hidden], weight)` (host alloc + memcpy)
/// + `Ops.mul(expertOut, broadcast)` + `Ops.add(accumulator, scaled)`,
/// we pack the routing weight into a 4-byte scalar buffer + dispatch
/// this kernel once. Saves 8 host allocations + 16 dispatches per MoE
/// layer × 40 layers = 320 allocations + 640 dispatches per
/// Qwen3.6-A3B decode token.
///
/// Numerical: accumulation widens to f32 via load-side `.cast` to keep
/// long sums of many small-weight expert outputs precise, then narrows
/// back to T on store.
#[kernel]
pub fn mt_scalar_fma<T>(scalar: Tensor<T>, value: Tensor<T>, base: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    let s = load(scalar[0]).cast::<f32>();
    let v = load(value[idx]).cast::<f32>();
    let b = load(base[idx]).cast::<f32>();
    store(out[idx], (b + s * v).cast::<T>());
}

/// 8-way fused scalar-FMA chain. Computes
///   `out[i] = sum_{k=0..8} scalar_k[0] * value_k[i]`
/// in a single dispatch. Replaces the topK=8 expert accumulator chain
/// in FFAI's MoE decode (8 sequential `mt_scalar_fma` dispatches +
/// 1 acc.zero) with one fused kernel that reads each value tensor
/// once and writes the output once — saving 7 acc reads + 1 zero
/// dispatch per MoE layer × 40 layers = 320 dispatches + ~660 KB of
/// L1/L2 traffic per Qwen3.6-A3B decode token.
///
/// Accumulation widens to f32 via load-side `.cast` to preserve
/// precision on long sums of small-weight expert outputs, then
/// narrows back to T on store. Bit-equivalent to the 8-call chain
/// modulo final-rounding mode.
#[kernel]
pub fn mt_scalar_fma_chain8<T>(
    scalar0: Tensor<T>,
    value0: Tensor<T>,
    scalar1: Tensor<T>,
    value1: Tensor<T>,
    scalar2: Tensor<T>,
    value2: Tensor<T>,
    scalar3: Tensor<T>,
    value3: Tensor<T>,
    scalar4: Tensor<T>,
    value4: Tensor<T>,
    scalar5: Tensor<T>,
    value5: Tensor<T>,
    scalar6: Tensor<T>,
    value6: Tensor<T>,
    scalar7: Tensor<T>,
    value7: Tensor<T>,
    out: Tensor<T>,
) {
    let idx = program_id(0);
    let sa = load(scalar0[0]).cast::<f32>();
    let sb = load(scalar1[0]).cast::<f32>();
    let sc = load(scalar2[0]).cast::<f32>();
    let sd = load(scalar3[0]).cast::<f32>();
    let se = load(scalar4[0]).cast::<f32>();
    let sf = load(scalar5[0]).cast::<f32>();
    let sg = load(scalar6[0]).cast::<f32>();
    let sh = load(scalar7[0]).cast::<f32>();
    let va = load(value0[idx]).cast::<f32>();
    let vb = load(value1[idx]).cast::<f32>();
    let vc = load(value2[idx]).cast::<f32>();
    let vd = load(value3[idx]).cast::<f32>();
    let ve = load(value4[idx]).cast::<f32>();
    let vf = load(value5[idx]).cast::<f32>();
    let vg = load(value6[idx]).cast::<f32>();
    let vh = load(value7[idx]).cast::<f32>();
    let sum = sa * va + sb * vb + sc * vc + sd * vd + se * ve + sf * vf + sg * vg + sh * vh;
    store(out[idx], sum.cast::<T>());
}

/// Fused elementwise sigmoid + mul. Computes
///   `out[i] = a[i] * sigmoid(b[i])`
/// in one dispatch. Used by Qwen3 attention layer's output gate:
///   attn_out = attn(x) * sigmoid(gate_proj(x))
/// Currently expressed as `Ops.mul(attnFlat, Ops.sigmoid(gate))` —
/// two dispatches. This fuses to one, saving 10 dispatches per
/// Qwen3.6-A3B decode token (1 per attn layer × 10 attn layers).
///
/// Sigmoid is computed at f32 precision via load-side cast to avoid
/// bf16 saturation drift near the asymptotes.
#[kernel]
pub fn mt_sigmoid_mul<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    let av = load(a[idx]).cast::<f32>();
    let bv = load(b[idx]).cast::<f32>();
    let sig = 1.0f32 / (1.0f32 + exp(0.0f32 - bv));
    store(out[idx], (av * sig).cast::<T>());
}

/// ⚠️ BROKEN — produces NaN/wrong output in FFAI integration test
/// (forwardManyEquivalence fails T=8 + T=128). Kept here as a stub
/// for future debug. Likely issue: multi-output codegen + the inlined
/// `mt_rms_inv_scalar` reduction call don't compose correctly in this
/// kernel shape. Possible fix paths:
///   1. Manually inline the reduction without `mt_rms_inv_scalar`
///   2. Use TWO compiled variants — one for residual write only, one
///      for residual+norm — and dispatch both in a shared encoder
///   3. Investigate the codegen MSL output for residual_out + normed_out
///      ordering vs the reduction-tg threadgroup layout
///
/// Fused residual-add + RMSNorm. For each row of `[n]` elements:
///   residual_out[i] = a[i] + b[i]
///   normed_out[i]   = (a[i] + b[i]) * w[i] / sqrt(mean((a+b)^2) + eps)
///
/// Standard transformer pattern at layer boundary:
///   h_new = h_old + mixer_out           (residual add)
///   normed = rms_norm(h_new, w)         (pre-norm for next mixer)
///
/// Both outputs are needed downstream: `residual_out` is the persistent
/// residual stream (input to the SECOND residual add of the same
/// layer); `normed_out` is the pre-FFN/pre-next-mixer input.
///
/// Same TG=n/4 contract as `mt_rms_norm`. Saves 1 dispatch per
/// residual-add+norm pair × 80 such pairs in Qwen3.6-A3B decode
/// (2 per layer × 40 layers) = ~1.4 ms / token at 17 µs encoder
/// overhead each.
#[kernel]
pub fn mt_add_rms_norm<T>(
    a: Tensor<T>,
    b: Tensor<T>,
    w: Tensor<T>,
    eps_buf: Tensor<f32>,
    mut residual_out: Tensor<T>,
    mut normed_out: Tensor<T>,
    #[constexpr] n: u32,
) {
    let row = program_id::<0>();
    let rs = row * n;
    let col = tid * 4u32;
    let in_bounds = col + 3u32 < n;
    let safe_col = select(in_bounds, col, 0u32);
    let safe_base = rs + safe_col;
    let base = rs + col;
    // Read a + b for 4 elements.
    let a0 = load(a[safe_base]).cast::<f32>();
    let a1 = load(a[safe_base + 1u32]).cast::<f32>();
    let a2 = load(a[safe_base + 2u32]).cast::<f32>();
    let a3 = load(a[safe_base + 3u32]).cast::<f32>();
    let b0 = load(b[safe_base]).cast::<f32>();
    let b1 = load(b[safe_base + 1u32]).cast::<f32>();
    let b2 = load(b[safe_base + 2u32]).cast::<f32>();
    let b3 = load(b[safe_base + 3u32]).cast::<f32>();
    let s0 = a0 + b0;
    let s1 = a1 + b1;
    let s2 = a2 + b2;
    let s3 = a3 + b3;
    let raw_ssq = s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3;
    let partial_ssq = select(in_bounds, raw_ssq, 0.0f32);
    // ITER 50 (Bagel 2): inline the reduction with `reduce_sum` directly
    // (no cross-kernel call to `mt_rms_inv_scalar`). The previous stub
    // used the cross-kernel call but the multi-output codegen path
    // didn't compose with the inlined reduction. Inlining matches the
    // working `mt_rms_norm` pattern exactly.
    let tg_ssq = reduce_sum(partial_ssq);
    let eps = load(eps_buf[0u32]);
    let rms = rsqrt(tg_ssq / n + eps);
    if in_bounds {
        // Write the residual stream (just the add).
        store(residual_out[base], s0.cast::<T>());
        store(residual_out[base + 1u32], s1.cast::<T>());
        store(residual_out[base + 2u32], s2.cast::<T>());
        store(residual_out[base + 3u32], s3.cast::<T>());
        // Write the normalized stream.
        let n0 = s0 * rms * load(w[col]).cast::<f32>();
        let n1 = s1 * rms * load(w[col + 1u32]).cast::<f32>();
        let n2 = s2 * rms * load(w[col + 2u32]).cast::<f32>();
        let n3 = s3 * rms * load(w[col + 3u32]).cast::<f32>();
        store(normed_out[base], n0.cast::<T>());
        store(normed_out[base + 1u32], n1.cast::<T>());
        store(normed_out[base + 2u32], n2.cast::<T>());
        store(normed_out[base + 3u32], n3.cast::<T>());
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

mod tests_support {
    #![allow(unused, dead_code)]
    use super::*;
    use metaltile::test_kernel;
    use metaltile_core::{
        DType,
        bench::{TestSetup, TestBuffer},
    };

    fn pack(vals: &[f32], dt: DType) -> Vec<u8> {
        match dt {
            DType::F32  => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
            DType::F16  => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            DType::BF16 => vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
            _           => panic!("unsupported dtype {dt:?}"),
        }
    }

    fn unpack(bytes: &[u8], dt: DType) -> Vec<f32> {
        match dt {
            DType::F32  => bytemuck::cast_slice::<u8, f32>(bytes).to_vec(),
            DType::F16  => bytes.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            DType::BF16 => bytes.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            _           => panic!("unsupported dtype {dt:?}"),
        }
    }

    fn round_dt(v: f32, dt: DType) -> f32 {
        match dt {
            DType::F32  => v,
            DType::F16  => half::f16::from_f32(v).to_f32(),
            DType::BF16 => half::bf16::from_f32(v).to_f32(),
            _           => v,
        }
    }

    // ── mlx/unary: exp ────────────────────────────────────────────────────

    #[test_kernel(name = "mlx/unary/exp", dtypes = [f32, f16], tol = 5e-3)]
    fn test_exp(dt: DType) -> TestSetup {
        let n = 512usize;
        let a: Vec<f32> = (0..n).map(|i| (i % 11) as f32 * 0.2 - 1.0).collect();
        let a_q: Vec<f32> = a.iter().map(|&v| round_dt(v, dt)).collect();
        let expected: Vec<f32> = a_q.iter().map(|x| x.exp()).collect();
        TestSetup::new(mt_exp::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a", pack(&a, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, DType::F32), DType::F32))
            .grid_1d(n, 256)
    }

    // ── mlx/unary: log ────────────────────────────────────────────────────

    #[test_kernel(name = "mlx/unary/log", dtypes = [f32], tol = 1e-4)]
    fn test_log(dt: DType) -> TestSetup {
        let n = 1024usize;
        let a: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.1 + 0.1).collect();
        let expected: Vec<f32> = a.iter().map(|x| x.ln()).collect();
        TestSetup::new(mt_log::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a", pack(&a, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, DType::F32), DType::F32))
            .grid_1d(n, 256)
    }

    // ── mlx/unary: sqrt ───────────────────────────────────────────────────

    #[test_kernel(name = "mlx/unary/sqrt", dtypes = [f32], tol = 1e-5)]
    fn test_sqrt(dt: DType) -> TestSetup {
        let n = 512usize;
        let a: Vec<f32> = (0..n).map(|i| (i % 19) as f32 * 0.1 + 0.05).collect();
        let expected: Vec<f32> = a.iter().map(|x| x.sqrt()).collect();
        TestSetup::new(mt_sqrt::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a", pack(&a, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, DType::F32), DType::F32))
            .grid_1d(n, 256)
    }

    // ── mlx/unary: rsqrt ──────────────────────────────────────────────────

    #[test_kernel(name = "mlx/unary/rsqrt", dtypes = [f32], tol = 1e-4)]
    fn test_rsqrt(dt: DType) -> TestSetup {
        let n = 512usize;
        let a: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.1 + 0.2).collect();
        let expected: Vec<f32> = a.iter().map(|x| 1.0 / x.sqrt()).collect();
        TestSetup::new(mt_rsqrt::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a", pack(&a, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, DType::F32), DType::F32))
            .grid_1d(n, 256)
    }

    // ── mlx/unary: abs ────────────────────────────────────────────────────

    #[test_kernel(name = "mlx/unary/abs", dtypes = [f32], tol = 1e-6)]
    fn test_abs(dt: DType) -> TestSetup {
        let n = 1024usize;
        let a: Vec<f32> = (0..n).map(|i| (i % 23) as f32 * 0.1 - 1.1).collect();
        let expected: Vec<f32> = a.iter().map(|x| x.abs()).collect();
        TestSetup::new(mt_abs::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a", pack(&a, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, DType::F32), DType::F32))
            .grid_1d(n, 256)
    }

    // ── mlx/unary: silu ───────────────────────────────────────────────────

    #[test_kernel(name = "mlx/unary/silu", dtypes = [f32, f16], tol = 5e-3)]
    fn test_silu(dt: DType) -> TestSetup {
        let n = 512usize;
        let a: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.3 - 2.0).collect();
        let a_q: Vec<f32> = a.iter().map(|&v| round_dt(v, dt)).collect();
        let expected: Vec<f32> = a_q.iter().map(|x| x / (1.0 + (-x).exp())).collect();
        TestSetup::new(mt_silu::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a", pack(&a, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, DType::F32), DType::F32))
            .grid_1d(n, 256)
    }

    // ── mlx/unary: relu ───────────────────────────────────────────────────

    #[test_kernel(name = "mlx/unary/relu", dtypes = [f32], tol = 1e-6)]
    fn test_relu(dt: DType) -> TestSetup {
        let n = 512usize;
        let a: Vec<f32> = (0..n).map(|i| (i % 19) as f32 * 0.15 - 1.4).collect();
        let expected: Vec<f32> = a.iter().map(|x| x.max(0.0)).collect();
        TestSetup::new(mt_relu::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a", pack(&a, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, DType::F32), DType::F32))
            .grid_1d(n, 256)
    }

    // ── mlx/unary: sigmoid ────────────────────────────────────────────────

    #[test_kernel(name = "mlx/unary/sigmoid", dtypes = [f32, bf16], tol = 2e-2)]
    fn test_sigmoid(dt: DType) -> TestSetup {
        let n = 512usize;
        let a: Vec<f32> = (0..n).map(|i| (i % 11) as f32 * 0.5 - 3.0).collect();
        let a_q: Vec<f32> = a.iter().map(|&v| round_dt(v, dt)).collect();
        let expected: Vec<f32> = a_q.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        TestSetup::new(mt_sigmoid::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a", pack(&a, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, DType::F32), DType::F32))
            .grid_1d(n, 256)
    }

    // ── mlx/cast_to_f32 ───────────────────────────────────────────────────

    #[test_kernel(name = "mlx/cast_to_f32/bf16", dtypes = [bf16], tol = 0.0)]
    fn test_cast_to_f32_bf16(dt: DType) -> TestSetup {
        let n = 1024usize;
        let vals: Vec<f32> = (0..n)
            .map(|i| {
                let x = (i as f32) * 0.137 - (n as f32) * 0.068;
                x * (1.0 + ((i % 5) as f32) * 0.01)
            })
            .collect();
        let expected: Vec<f32> = vals.iter().map(|&v| half::bf16::from_f32(v).to_f32()).collect();
        TestSetup::new(mt_cast_to_f32::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("input", pack(&vals, dt), dt))
            .expect(TestBuffer::from_vec("out", bytemuck::cast_slice::<f32, u8>(&expected).to_vec(), DType::F32))
            .grid_1d(n, 256)
    }

    #[test_kernel(name = "mlx/cast_to_f32/f16", dtypes = [f16], tol = 0.0)]
    fn test_cast_to_f32_f16(dt: DType) -> TestSetup {
        let n = 1024usize;
        let vals: Vec<f32> = (0..n).map(|i| (i as f32) * 0.041 - (n as f32) * 0.020).collect();
        let expected: Vec<f32> = vals.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
        TestSetup::new(mt_cast_to_f32::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("input", pack(&vals, dt), dt))
            .expect(TestBuffer::from_vec("out", bytemuck::cast_slice::<f32, u8>(&expected).to_vec(), DType::F32))
            .grid_1d(n, 256)
    }

    #[test_kernel(name = "mlx/cast_to_f32/f32_identity", dtypes = [f32], tol = 0.0)]
    fn test_cast_to_f32_f32(dt: DType) -> TestSetup {
        let n = 256usize;
        let vals: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 64.0).collect();
        TestSetup::new(mt_cast_to_f32::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("input", pack(&vals, dt), dt))
            .expect(TestBuffer::from_vec("out", bytemuck::cast_slice::<f32, u8>(&vals).to_vec(), DType::F32))
            .grid_1d(n, 256)
    }

    // ── mlx/sigmoid_scalar_fma ────────────────────────────────────────────

    fn sigmoid_fma_oracle(gate: f32, value: &[f32], base: &[f32], dt: DType) -> Vec<f32> {
        let g_q = round_dt(gate, dt);
        let s = 1.0 / (1.0 + (-g_q).exp());
        value.iter().zip(base.iter()).map(|(&v, &b)| {
            let vq = round_dt(v, dt);
            let bq = round_dt(b, dt);
            round_dt(bq + s * vq, dt)
        }).collect()
    }

    fn ramp(n: usize, modulus: usize, offset: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % modulus) as f32 - offset) * 0.05).collect()
    }

    #[test_kernel(name = "mlx/sigmoid_scalar_fma/f32", dtypes = [f32], tol = 1e-5)]
    fn test_sigmoid_scalar_fma_f32(dt: DType) -> TestSetup {
        let n = 2048usize;
        let gate = 0.5f32;
        let value: Vec<f32> = ramp(n, 11, 5.0).iter().map(|v| 0.2 * v - 1.0).collect();
        let base:  Vec<f32> = ramp(n, 17, 8.0).iter().map(|v| 0.1 * v).collect();
        let expected = sigmoid_fma_oracle(gate, &value, &base, dt);
        TestSetup::new(mt_sigmoid_scalar_fma::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("gate",  pack(&[gate, 0.0], dt), dt))
            .input(TestBuffer::from_vec("value", pack(&value, dt), dt))
            .input(TestBuffer::from_vec("base",  pack(&base, dt), dt))
            .expect(TestBuffer::from_vec("out",  pack(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(name = "mlx/sigmoid_scalar_fma/f16", dtypes = [f16], tol = 5e-4)]
    fn test_sigmoid_scalar_fma_f16(dt: DType) -> TestSetup {
        let n = 2048usize;
        let gate = 0.5f32;
        let value: Vec<f32> = ramp(n, 11, 5.0).iter().map(|v| 0.2 * v - 1.0).collect();
        let base:  Vec<f32> = ramp(n, 17, 8.0).iter().map(|v| 0.1 * v).collect();
        let expected = sigmoid_fma_oracle(gate, &value, &base, dt);
        TestSetup::new(mt_sigmoid_scalar_fma::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("gate",  pack(&[gate, 0.0], dt), dt))
            .input(TestBuffer::from_vec("value", pack(&value, dt), dt))
            .input(TestBuffer::from_vec("base",  pack(&base, dt), dt))
            .expect(TestBuffer::from_vec("out",  pack(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(name = "mlx/sigmoid_scalar_fma/bf16", dtypes = [bf16], tol = 5e-3)]
    fn test_sigmoid_scalar_fma_bf16(dt: DType) -> TestSetup {
        let n = 2048usize;
        let gate = 0.5f32;
        let value: Vec<f32> = ramp(n, 11, 5.0).iter().map(|v| 0.2 * v - 1.0).collect();
        let base:  Vec<f32> = ramp(n, 17, 8.0).iter().map(|v| 0.1 * v).collect();
        let expected = sigmoid_fma_oracle(gate, &value, &base, dt);
        TestSetup::new(mt_sigmoid_scalar_fma::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("gate",  pack(&[gate, 0.0], dt), dt))
            .input(TestBuffer::from_vec("value", pack(&value, dt), dt))
            .input(TestBuffer::from_vec("base",  pack(&base, dt), dt))
            .expect(TestBuffer::from_vec("out",  pack(&expected, dt), dt))
            .grid_1d(n, 256)
    }
}
