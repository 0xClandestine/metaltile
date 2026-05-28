//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Gated DeltaNet — **fused** prep + decode kernel.
//!
//! `mt_gated_delta_prep_step` extends the recurrence-only
//! [`mt_gated_delta_step`](super::gated_delta::mt_gated_delta_step) by
//! absorbing every host-side prep computation Qwen3.6 / Qwen3.5 currently
//! does between the conv1d and the GDN recurrence:
//!
//!   1. **Conv split** — `conv_out = [q (Hk·Dk), k (Hk·Dk), v (Hv·Dv)]`
//!      is split into q / k / v on the GPU instead of via a host
//!      `toFloatArray() + Swift slicing` roundtrip.
//!   2. **Per-head RMSNorm + scale of q, k** — replaces `perHeadRMSNormScale35`
//!      in `Qwen35GDNMixer.forward`. Scale and (optional) per-head_dim
//!      weights are folded into the same simd-sum over Dk that the kernel
//!      already pays for the recurrence.
//!   3. **g = exp(-exp(A_log) · softplus(a_raw + dt_bias))** — fused.
//!   4. **beta = sigmoid(b_raw)** — fused.
//!   5. The existing recurrence (state decay + delta + outer + read).
//!
//! Net effect on Qwen3.6 decode: one fused GDN kernel per layer instead
//! of `commit()/waitUntilCompleted()` → host arithmetic → `makeCommandBuffer()`
//! → `gatedDeltaStep` dispatch. 30 GDN layers per step × ≥2 host-sync
//! gaps per layer = the bandwidth recovery target for Iter FG2.
//!
//! Inputs that are now GPU-resident:
//!   - `conv_out`     : Tensor<T> [B, 2·Hk·Dk + Hv·Dv]
//!   - `a_log`        : Tensor<T> [Hv]   — per-Hv-head learnable
//!   - `dt_bias`      : Tensor<T> [Hv]
//!   - `a_raw`        : Tensor<T> [B, Hv]
//!   - `b_raw`        : Tensor<T> [B, Hv]
//!   - `q_norm_weight`: Tensor<T> [Hk·Dk]  — pass an all-1×scale vector to
//!     recover the unweighted `perHeadRMSNormScale35` path.
//!   - `k_norm_weight`: Tensor<T> [Hk·Dk]
//!   - `state_in`     : Tensor<T> [B, Hv, Dv, Dk]   (recurrence state)
//!
//! Outputs:
//!   - `state_out`    : Tensor<T> [B, Hv, Dv, Dk]
//!   - `y`            : Tensor<T> [B, Hv, Dv]
//!
//! ## DISPATCH INVARIANTS (identical to `mt_gated_delta_step`)
//!
//! - **Mode: Reduction.** Each TG is one simdgroup (32 threads).
//! - **Grid: `[Dv, B·Hv, 1]`, TG: `[32, 1, 1]`.**
//! - **`Dk % 32 == 0`.** Each lane owns `n_per_t = Dk / 32` contiguous
//!   slots via `s_idx = n_per_t · dk_idx + i`.
//! - **Hv divisible by Hk.** GQA: `hk_idx = hv_idx / (Hv/Hk)`.
//!
//! ## Per-head RMSNorm redundancy
//!
//! Each (Dv_idx, b, hv) TG re-computes the same q_normed / k_normed for
//! its Hk-group. Cost is `O(Dk)` ALU per TG and is already part of the
//! existing per-lane chunked load anyway — every lane reads its `n_per_t`
//! slice of q/k for the recurrence. The fused kernel just folds the
//! ssq + simd_sum + scale into that same pass and stashes the result on
//! the per-lane stack alongside `decayed` / `k_cache`. fp32 throughout.

use metaltile::kernel;

/// Fused GDN prep + recurrence step. See module doc for layout and
/// dispatch invariants. Drop-in replacement for the
/// `host-prep + mt_gated_delta_step` pair in `Qwen35GDNMixer.forward`.
#[kernel]
pub fn mt_gated_delta_prep_step<T>(
    conv_out: Tensor<T>,      // [B, 2·Hk·Dk + Hv·Dv]    q | k | v
    a_log: Tensor<T>,         // [Hv]
    dt_bias: Tensor<T>,       // [Hv]
    a_raw: Tensor<T>,         // [B, Hv]
    b_raw: Tensor<T>,         // [B, Hv]
    q_norm_weight: Tensor<T>, // [Hk·Dk]   pass 1.0×invKeyScale²  for unweighted q-scale path
    k_norm_weight: Tensor<T>, // [Hk·Dk]   pass 1.0×invKeyScale   for unweighted k-scale path
    state_in: Tensor<T>,      // [B, Hv, Dv, Dk]
    mut state_out: Tensor<T>, // [B, Hv, Dv, Dk]
    mut y: Tensor<T>,         // [B, Hv, Dv]
    #[constexpr] dk: u32,
    #[constexpr] dv: u32,
    #[constexpr] hv: u32,
    #[constexpr] hk: u32,
) {
    let dv_idx = tgid_x;
    let n = tgid_y;
    let dk_idx = tid;
    // GQA decomposition: n = b · Hv + hv_idx; hk_idx = hv_idx / (Hv/Hk).
    // Mirrors `mt_gated_delta_step` exactly.
    let hv_idx = n - (n / hv) * hv;
    let b = n / hv;
    let hk_per_hv = hv / hk;
    let hk_idx = hv_idx / hk_per_hv;
    let n_per_t = dk / 32u32;
    // Conv-output flat layout for batch `b`:
    //   q_base = b · (2·Hk·Dk + Hv·Dv)
    //   k_base = q_base + Hk·Dk
    //   v_base = q_base + 2·Hk·Dk
    let stride_b = 2u32 * hk * dk + hv * dv;
    let conv_base = b * stride_b;
    let q_off = conv_base + hk_idx * dk;
    let k_off = conv_base + hk * dk + hk_idx * dk;
    let v_off = conv_base + 2u32 * hk * dk + hv_idx * dv;
    // Per-head RMSNorm eps = 1e-6 (matches `perHeadRMSNormScale35`).
    let eps = 0.000001f32;
    let dk_f = dk.cast::<f32>();
    // ─── Phase 0a: Per-head RMSNorm of q / k ─────────────────────────────
    //
    // Each lane reads its `n_per_t` chunk of q and k (Dk-wide, per-head),
    // accumulates a partial ssq, then simd_sum to get the per-head total.
    // The same chunk is also weighted by `q_norm_weight` / `k_norm_weight`
    // and stashed on the per-lane stack so phase 1 / phase 2 read register
    // memory (no second load of conv_out).
    //
    // Cap = 8 (n_per_t @ Dk=256 / 32). At Dk=128, n_per_t=4 — upper 4 slots
    // simply go unread. Same convention as `mt_gated_delta_step`.
    stack_alloc("q_raw", 8u32, "f32");
    stack_alloc("k_raw", 8u32, "f32");
    stack_alloc("q_w", 8u32, "f32");
    stack_alloc("k_w", 8u32, "f32");
    let mut q_ssq = 0.0f32;
    let mut k_ssq = 0.0f32;
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * dk_idx + i;
        let qv = load(conv_out[q_off + s_idx]).cast::<f32>();
        let kv = load(conv_out[k_off + s_idx]).cast::<f32>();
        stack_store("q_raw", i, qv);
        stack_store("k_raw", i, kv);
        q_ssq = q_ssq + qv * qv;
        k_ssq = k_ssq + kv * kv;
        // Weights are layout `[Hk·Dk]` — same hk_idx slot the q/k row reads.
        let qw = load(q_norm_weight[hk_idx * dk + s_idx]).cast::<f32>();
        let kw = load(k_norm_weight[hk_idx * dk + s_idx]).cast::<f32>();
        stack_store("q_w", i, qw);
        stack_store("k_w", i, kw);
    }
    let q_ssq_sum = simd_sum(q_ssq);
    let k_ssq_sum = simd_sum(k_ssq);
    // rsqrt(ssq/Dk + eps) = 1 / sqrt(mean + eps). Folds the per-head
    // `scale` parameter directly: caller bakes it into `*_norm_weight`.
    let q_inv = rsqrt(q_ssq_sum / dk_f + eps);
    let k_inv = rsqrt(k_ssq_sum / dk_f + eps);
    // ─── Phase 0b: g / beta from a_log / dt_bias / a_raw / b_raw ─────────
    //
    // Math per Hv-head:
    //   dt   = softplus(a_raw + dt_bias)        (log(1 + exp(·)) form)
    //   g    = exp(-exp(a_log) · dt)
    //   beta = sigmoid(b_raw)
    //
    // softplus is not a DSL primitive — emit `log(exp(x) + 1)` directly.
    // Production values of `a_raw + dt_bias` for Qwen3.6 sit in
    // approximately [-6, +2] (see `Qwen3NextGatedDeltaNet` HF config), so
    // the un-clamped formula stays in fp32 dynamic range. The CPU oracle
    // uses the same formula so the GPU↔CPU diff is purely ULP.
    //
    // Every lane redundantly computes g / beta (scalar broadcast across
    // the simdgroup). The scalar load + 4 math ops cost much less than
    // burning a simd_broadcast plus a barrier.
    let a_log_val = load(a_log[hv_idx]).cast::<f32>();
    let dt_bias_val = load(dt_bias[hv_idx]).cast::<f32>();
    let a_raw_val = load(a_raw[n]).cast::<f32>();
    let b_raw_val = load(b_raw[n]).cast::<f32>();
    // softplus(x)   = log(1 + exp(x))  — un-clamped; production magnitudes
    //                  of (a_raw + dt_bias) sit in fp32 safe range.
    // sigmoid(x)    = 1 / (1 + exp(-x))  — inlined rather than using the
    //                  `Activation::Sigmoid` op because the standard pipeline
    //                  folds Activation into `FusedElementwise` and the
    //                  per-kernel feature analyzer (`needs_sigmoid`) does
    //                  not recurse into fused chains. Inlining keeps the
    //                  emitted MSL self-contained — no `mt_sigmoid` helper
    //                  required.
    let pre_softplus = a_raw_val + dt_bias_val;
    let dt_val = log(exp(pre_softplus) + 1.0f32);
    let g_val = exp(0.0f32 - exp(a_log_val) * dt_val);
    let beta_val = 1.0f32 / (1.0f32 + exp(0.0f32 - b_raw_val));
    // v reads once per Dv slot — no normalization, just dtype-cast.
    let v_val = load(conv_out[v_off + dv_idx]).cast::<f32>();
    // ─── Phase 1: decay + kv_mem reduction ───────────────────────────────
    //
    // Same shape as `mt_gated_delta_step::phase_1` but reads q/k from the
    // per-lane `*_normed` stash instead of global. `decayed` and `k_cache`
    // stay register-resident across phases 1/2, same convention.
    let state_base = n * dv * dk + dv_idx * dk;
    stack_alloc("decayed", 8u32, "f32");
    stack_alloc("k_cache", 8u32, "f32");
    let mut kv_mem = 0.0f32;
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * dk_idx + i;
        let s_decayed = load(state_in[state_base + s_idx]).cast::<f32>() * g_val;
        // Normed k = k_raw * q_inv * weight. RMSNorm formula:
        //   x_normed[d] = x[d] · rsqrt(mean(x²) + eps) · w[d]
        let k_normed = stack_load("k_raw", i) * k_inv * stack_load("k_w", i);
        stack_store("decayed", i, s_decayed);
        stack_store("k_cache", i, k_normed);
        kv_mem = kv_mem + s_decayed * k_normed;
    }
    let kv_mem_sum = simd_sum(kv_mem);
    let delta = (v_val - kv_mem_sum) * beta_val;
    // ─── Phase 2: rank-1 update + output projection ──────────────────────
    let mut out_acc = 0.0f32;
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * dk_idx + i;
        let s_decayed = stack_load("decayed", i);
        let k_normed = stack_load("k_cache", i);
        let s_new = s_decayed + k_normed * delta;
        store(state_out[state_base + s_idx], s_new.cast::<T>());
        let q_normed = stack_load("q_raw", i) * q_inv * stack_load("q_w", i);
        out_acc = out_acc + s_new * q_normed;
    }
    let out_sum = simd_sum(out_acc);
    // ─── Phase 3: lane 0 writes y[n, dv_idx] ────────────────────────────
    if dk_idx == 0u32 {
        store(y[n * dv + dv_idx], out_sum.cast::<T>());
    }
}

#[cfg(test)]
mod tests {
    use metaltile_core::{DType, ir::KernelMode};

    use super::*;

    /// Developer aid — dump the full generated MSL for inspection.
    /// `cargo test -p metaltile-std --lib --release -- ffai::gated_delta_prep::tests::dump --nocapture`
    #[test]
    fn dump() {
        use metaltile_codegen::msl::MslGenerator;
        let mut k = mt_gated_delta_prep_step::kernel_ir_for(DType::F32);
        k.mode = KernelMode::Reduction;
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        println!("===== BEGIN MSL =====\n{}\n===== END MSL =====", msl);
    }
}

#[cfg(target_os = "macos")]
pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::test_kernel;
    use metaltile_core::{
        DType,
        bench::{TestBuffer, TestSetup},
        ir::KernelMode,
    };

    use super::mt_gated_delta_prep_step;

    fn pack(vals: &[f32], dt: DType) -> Vec<u8> {
        match dt {
            DType::F32 => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
            DType::F16 => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            DType::BF16 =>
                vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
            _ => panic!("unsupported dtype {dt:?}"),
        }
    }

    fn u32_le(v: u32) -> Vec<u8> { v.to_le_bytes().to_vec() }

    fn softplus(x: f32) -> f32 { (x.exp() + 1.0).ln() }
    fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

    /// Fused prep + recurrence reference. Returns (y, state_out).
    fn cpu_prep_step(
        conv_out: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
        a_raw: &[f32],
        b_raw: &[f32],
        q_norm_weight: &[f32],
        k_norm_weight: &[f32],
        state_in: &[f32],
        b: usize,
        hv: usize,
        hk: usize,
        dv: usize,
        dk: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let eps = 1e-6_f32;
        let stride_b = 2 * hk * dk + hv * dv;
        let hk_per_hv = hv / hk;

        // Build q_normed, k_normed, v_flat, g, beta.
        let mut q_normed = vec![0.0_f32; b * hk * dk];
        let mut k_normed = vec![0.0_f32; b * hk * dk];
        let mut v_flat = vec![0.0_f32; b * hv * dv];
        let mut g = vec![0.0_f32; b * hv];
        let mut beta = vec![0.0_f32; b * hv];

        for batch in 0..b {
            let q_base = batch * stride_b;
            let k_base = q_base + hk * dk;
            let v_base = q_base + 2 * hk * dk;
            for hk_idx in 0..hk {
                let row_off = hk_idx * dk;
                let mut q_ssq = 0.0_f32;
                let mut k_ssq = 0.0_f32;
                for d in 0..dk {
                    let qv = conv_out[q_base + row_off + d];
                    let kv = conv_out[k_base + row_off + d];
                    q_ssq += qv * qv;
                    k_ssq += kv * kv;
                }
                let q_inv = 1.0 / ((q_ssq / dk as f32) + eps).sqrt();
                let k_inv = 1.0 / ((k_ssq / dk as f32) + eps).sqrt();
                for d in 0..dk {
                    let qv = conv_out[q_base + row_off + d];
                    let kv = conv_out[k_base + row_off + d];
                    q_normed[batch * hk * dk + row_off + d] =
                        qv * q_inv * q_norm_weight[hk_idx * dk + d];
                    k_normed[batch * hk * dk + row_off + d] =
                        kv * k_inv * k_norm_weight[hk_idx * dk + d];
                }
            }
            for hv_idx in 0..hv {
                for dv_idx in 0..dv {
                    v_flat[(batch * hv + hv_idx) * dv + dv_idx] =
                        conv_out[v_base + hv_idx * dv + dv_idx];
                }
            }
            for hv_idx in 0..hv {
                let n = batch * hv + hv_idx;
                let dt = softplus(a_raw[n] + dt_bias[hv_idx]);
                g[n] = (-a_log[hv_idx].exp() * dt).exp();
                beta[n] = sigmoid(b_raw[n]);
            }
        }

        // Recurrence.
        let mut y = vec![0.0_f32; b * hv * dv];
        let mut state_out = vec![0.0_f32; b * hv * dv * dk];
        for batch in 0..b {
            for hv_idx in 0..hv {
                let n = batch * hv + hv_idx;
                let hk_idx = hv_idx / hk_per_hv;
                let g_val = g[n];
                let beta_val = beta[n];
                let qk_base = (batch * hk + hk_idx) * dk;
                for dv_idx in 0..dv {
                    let v_val = v_flat[n * dv + dv_idx];
                    let s_base = n * dv * dk + dv_idx * dk;
                    let mut kv_mem = 0.0_f32;
                    let mut decayed = vec![0.0_f32; dk];
                    for s_idx in 0..dk {
                        let s = state_in[s_base + s_idx] * g_val;
                        decayed[s_idx] = s;
                        kv_mem += s * k_normed[qk_base + s_idx];
                    }
                    let delta = (v_val - kv_mem) * beta_val;
                    let mut out = 0.0_f32;
                    for s_idx in 0..dk {
                        let s_new = decayed[s_idx] + k_normed[qk_base + s_idx] * delta;
                        state_out[s_base + s_idx] = s_new;
                        out += s_new * q_normed[qk_base + s_idx];
                    }
                    y[n * dv + dv_idx] = out;
                }
            }
        }
        (y, state_out)
    }

    #[test_kernel(name = "ffai/gated_delta/prep_step", dtypes = [f32], tol = 1e-3)]
    fn test_gated_delta_prep_step(dt: DType) -> TestSetup {
        use super::mt_gated_delta_prep_step;
        let b = 1usize;
        let hv = 4usize;
        let hk = 2usize;
        let dv = 8usize;
        let dk = 32usize;
        let stride_b = 2 * hk * dk + hv * dv;
        let n_total = b * hv;

        let conv_out: Vec<f32> =
            (0..b * stride_b).map(|i| ((i as f32) * 0.0173).sin() * 0.5).collect();
        let a_log: Vec<f32> = (0..hv).map(|i| -1.0 + (i as f32) * 0.2).collect();
        let dt_bias: Vec<f32> = (0..hv).map(|i| (i as f32) * 0.05).collect();
        let a_raw: Vec<f32> = (0..b * hv).map(|i| ((i as f32) * 0.1).sin() * 0.3).collect();
        let b_raw: Vec<f32> = (0..b * hv).map(|i| ((i as f32) * 0.13).cos() * 0.4).collect();
        let q_norm_weight: Vec<f32> = vec![1.0_f32; hk * dk];
        let k_norm_weight: Vec<f32> = vec![1.0_f32; hk * dk];
        let state_in: Vec<f32> =
            (0..n_total * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.05).collect();

        let (expected_y, expected_state) = cpu_prep_step(
            &conv_out,
            &a_log,
            &dt_bias,
            &a_raw,
            &b_raw,
            &q_norm_weight,
            &k_norm_weight,
            &state_in,
            b,
            hv,
            hk,
            dv,
            dk,
        );

        let mut kernel_ir = mt_gated_delta_prep_step::kernel_ir_for(dt);
        kernel_ir.mode = metaltile_core::ir::KernelMode::Reduction;

        TestSetup::new(kernel_ir)
            .input(TestBuffer::from_vec("conv_out", pack(&conv_out, dt), dt))
            .input(TestBuffer::from_vec("a_log", pack(&a_log, dt), dt))
            .input(TestBuffer::from_vec("dt_bias", pack(&dt_bias, dt), dt))
            .input(TestBuffer::from_vec("a_raw", pack(&a_raw, dt), dt))
            .input(TestBuffer::from_vec("b_raw", pack(&b_raw, dt), dt))
            .input(TestBuffer::from_vec("q_norm_weight", pack(&q_norm_weight, dt), dt))
            .input(TestBuffer::from_vec("k_norm_weight", pack(&k_norm_weight, dt), dt))
            .input(TestBuffer::from_vec("state_in", pack(&state_in, dt), dt))
            .input(TestBuffer::from_vec("dk", u32_le(dk as u32), DType::U32))
            .input(TestBuffer::from_vec("dv", u32_le(dv as u32), DType::U32))
            .input(TestBuffer::from_vec("hv", u32_le(hv as u32), DType::U32))
            .input(TestBuffer::from_vec("hk", u32_le(hk as u32), DType::U32))
            .expect(TestBuffer::from_vec("y", pack(&expected_y, dt), dt))
            .expect(TestBuffer::from_vec("state_out", pack(&expected_state, dt), dt))
            .grid_3d(dv as u32, n_total as u32, 1, [32, 1, 1])
    }

    /// GPU correctness tests for `mt_gated_delta_prep_step`.
    fn cpu_fused_oracle(
        conv_out: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
        a_raw: &[f32],
        b_raw: &[f32],
        q_norm_weight: &[f32],
        k_norm_weight: &[f32],
        state_in: &[f32],
        b: usize,
        hv: usize,
        hk: usize,
        dv: usize,
        dk: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let eps = 1e-6_f32;
        let stride_b = 2 * hk * dk + hv * dv;
        let hk_per_hv = hv / hk;
        let mut q_normed = vec![0.0_f32; b * hk * dk];
        let mut k_normed = vec![0.0_f32; b * hk * dk];
        let mut v_flat = vec![0.0_f32; b * hv * dv];
        let mut g = vec![0.0_f32; b * hv];
        let mut beta_arr = vec![0.0_f32; b * hv];
        for batch in 0..b {
            let q_base = batch * stride_b;
            let k_base = q_base + hk * dk;
            let v_base = q_base + 2 * hk * dk;
            for hk_idx in 0..hk {
                let row_off = hk_idx * dk;
                let mut q_ssq = 0.0_f32;
                let mut k_ssq = 0.0_f32;
                for d in 0..dk {
                    let qv = conv_out[q_base + row_off + d];
                    let kv = conv_out[k_base + row_off + d];
                    q_ssq += qv * qv;
                    k_ssq += kv * kv;
                }
                let q_inv = 1.0 / ((q_ssq / dk as f32) + eps).sqrt();
                let k_inv = 1.0 / ((k_ssq / dk as f32) + eps).sqrt();
                for d in 0..dk {
                    q_normed[batch * hk * dk + row_off + d] =
                        conv_out[q_base + row_off + d] * q_inv * q_norm_weight[hk_idx * dk + d];
                    k_normed[batch * hk * dk + row_off + d] =
                        conv_out[k_base + row_off + d] * k_inv * k_norm_weight[hk_idx * dk + d];
                }
            }
            for hv_idx in 0..hv {
                for dv_idx in 0..dv {
                    v_flat[(batch * hv + hv_idx) * dv + dv_idx] =
                        conv_out[v_base + hv_idx * dv + dv_idx];
                }
            }
            for hv_idx in 0..hv {
                let n = batch * hv + hv_idx;
                let dt_v = softplus(a_raw[n] + dt_bias[hv_idx]);
                g[n] = (-a_log[hv_idx].exp() * dt_v).exp();
                beta_arr[n] = sigmoid(b_raw[n]);
            }
        }
        let mut y = vec![0.0_f32; b * hv * dv];
        let mut state_out = vec![0.0_f32; b * hv * dv * dk];
        for batch in 0..b {
            for hv_idx in 0..hv {
                let n = batch * hv + hv_idx;
                let hk_idx = hv_idx / hk_per_hv;
                let g_val = g[n];
                let beta_val = beta_arr[n];
                let qk_base = (batch * hk + hk_idx) * dk;
                for dv_idx in 0..dv {
                    let v_val = v_flat[n * dv + dv_idx];
                    let s_base = n * dv * dk + dv_idx * dk;
                    let mut kv_mem = 0.0_f32;
                    let mut decayed = vec![0.0_f32; dk];
                    for s_idx in 0..dk {
                        let s = state_in[s_base + s_idx] * g_val;
                        decayed[s_idx] = s;
                        kv_mem += s * k_normed[qk_base + s_idx];
                    }
                    let delta = (v_val - kv_mem) * beta_val;
                    let mut out = 0.0_f32;
                    for s_idx in 0..dk {
                        let s_new = decayed[s_idx] + k_normed[qk_base + s_idx] * delta;
                        state_out[s_base + s_idx] = s_new;
                        out += s_new * q_normed[qk_base + s_idx];
                    }
                    y[n * dv + dv_idx] = out;
                }
            }
        }
        (y, state_out)
    }

    fn make_inputs(
        b: usize,
        hv: usize,
        hk: usize,
        dv: usize,
        dk: usize,
        identity: bool,
        ws: f32,
        seed_offset: usize,
        dt: DType,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let stride_b = 2 * hk * dk + hv * dv;
        let round = |v: f32| match dt {
            DType::F16 => half::f16::from_f32(v).to_f32(),
            DType::BF16 => half::bf16::from_f32(v).to_f32(),
            _ => v,
        };
        let conv_out: Vec<f32> = (0..b * stride_b)
            .map(|i| round((((i + seed_offset) as f32) * 0.0131).sin() * 0.4))
            .collect();
        let a_log: Vec<f32> = (0..hv).map(|i| round(-1.5 - (i as f32) * 0.1)).collect();
        let dt_bias: Vec<f32> = (0..hv).map(|i| round(-0.5 + (i as f32) * 0.05)).collect();
        let a_raw: Vec<f32> = (0..b * hv).map(|i| round(-0.3 + (i as f32) * 0.04)).collect();
        let b_raw: Vec<f32> = (0..b * hv).map(|i| round(-0.2 + (i as f32) * 0.03)).collect();
        let q_norm_weight: Vec<f32> = if identity {
            vec![round(ws); hk * dk]
        } else {
            (0..hk * dk).map(|i| round(ws * (1.0 + ((i % 11) as f32) * 0.05))).collect()
        };
        let k_norm_weight: Vec<f32> = if identity {
            vec![round(ws); hk * dk]
        } else {
            (0..hk * dk).map(|i| round(ws * (1.0 + ((i % 13) as f32) * 0.04))).collect()
        };
        let state_in: Vec<f32> = (0..b * hv * dv * dk)
            .map(|i| round((((i + seed_offset) as f32) * 0.0073).cos() * 0.1))
            .collect();
        (conv_out, a_log, dt_bias, a_raw, b_raw, q_norm_weight, k_norm_weight, state_in)
    }

    fn build_setup(
        b: usize,
        hv: usize,
        hk: usize,
        dv: usize,
        dk: usize,
        identity: bool,
        ws: f32,
        seed_offset: usize,
        dt: DType,
    ) -> TestSetup {
        let (conv_out, a_log, dt_bias, a_raw, b_raw, q_norm_weight, k_norm_weight, state_in) =
            make_inputs(b, hv, hk, dv, dk, identity, ws, seed_offset, dt);
        let n_total = b * hv;
        let (expected_y, expected_state) = cpu_fused_oracle(
            &conv_out,
            &a_log,
            &dt_bias,
            &a_raw,
            &b_raw,
            &q_norm_weight,
            &k_norm_weight,
            &state_in,
            b,
            hv,
            hk,
            dv,
            dk,
        );
        let mut kernel_ir = mt_gated_delta_prep_step::kernel_ir_for(dt);
        kernel_ir.mode = KernelMode::Reduction;
        TestSetup::new(kernel_ir)
            .input(TestBuffer::from_vec("conv_out", pack(&conv_out, dt), dt))
            .input(TestBuffer::from_vec("a_log", pack(&a_log, dt), dt))
            .input(TestBuffer::from_vec("dt_bias", pack(&dt_bias, dt), dt))
            .input(TestBuffer::from_vec("a_raw", pack(&a_raw, dt), dt))
            .input(TestBuffer::from_vec("b_raw", pack(&b_raw, dt), dt))
            .input(TestBuffer::from_vec("q_norm_weight", pack(&q_norm_weight, dt), dt))
            .input(TestBuffer::from_vec("k_norm_weight", pack(&k_norm_weight, dt), dt))
            .input(TestBuffer::from_vec("state_in", pack(&state_in, dt), dt))
            .input(TestBuffer::from_vec("dk", u32_le(dk as u32), DType::U32))
            .input(TestBuffer::from_vec("dv", u32_le(dv as u32), DType::U32))
            .input(TestBuffer::from_vec("hv", u32_le(hv as u32), DType::U32))
            .input(TestBuffer::from_vec("hk", u32_le(hk as u32), DType::U32))
            .expect(TestBuffer::from_vec("y", pack(&expected_y, dt), dt))
            .expect(TestBuffer::from_vec("state_out", pack(&expected_state, dt), dt))
            .grid_3d(dv as u32, n_total as u32, 1, [32, 1, 1])
    }

    #[test_kernel(name = "ffai/gated_delta_prep/step_f32_qwen36_identity", dtypes = [f32], tol = 1e-3)]
    fn prep_step_f32_qwen36_shape_identity_weights(dt: DType) -> TestSetup {
        build_setup(1, 32, 16, 128, 128, true, 1.0, 0, dt)
    }

    #[test_kernel(name = "ffai/gated_delta_prep/step_f32_qwen36_weighted", dtypes = [f32], tol = 1e-3)]
    fn prep_step_f32_qwen36_shape_nonidentity_weights(dt: DType) -> TestSetup {
        build_setup(1, 32, 16, 128, 128, false, 0.5, 0, dt)
    }

    #[test_kernel(name = "ffai/gated_delta_prep/step_f32_no_gqa", dtypes = [f32], tol = 1e-3)]
    fn prep_step_f32_no_gqa(dt: DType) -> TestSetup {
        build_setup(1, 4, 4, 32, 64, true, 1.0, 0, dt)
    }

    #[test_kernel(name = "ffai/gated_delta_prep/step_f32_dk256", dtypes = [f32], tol = 1e-3)]
    fn prep_step_f32_dk_256_full_n_per_t_slot_usage(dt: DType) -> TestSetup {
        build_setup(1, 4, 2, 8, 256, true, 1.0, 0, dt)
    }

    #[test_kernel(name = "ffai/gated_delta_prep/step_f32_batch2", dtypes = [f32], tol = 1e-3)]
    fn prep_step_f32_batch_2(dt: DType) -> TestSetup {
        build_setup(2, 4, 2, 8, 64, false, 0.7, 0, dt)
    }

    #[test_kernel(name = "ffai/gated_delta_prep/step_f16_qwen36_identity", dtypes = [f16], tol = 1e-3)]
    fn prep_step_f16_qwen36_shape_identity_weights(dt: DType) -> TestSetup {
        build_setup(1, 32, 16, 128, 128, true, 1.0, 0, dt)
    }

    #[test_kernel(name = "ffai/gated_delta_prep/step_f16_qwen36_weighted", dtypes = [f16], tol = 1e-3)]
    fn prep_step_f16_qwen36_shape_nonidentity_weights(dt: DType) -> TestSetup {
        build_setup(1, 32, 16, 128, 128, false, 0.5, 0, dt)
    }

    #[test_kernel(name = "ffai/gated_delta_prep/step_f16_no_gqa", dtypes = [f16], tol = 1e-3)]
    fn prep_step_f16_no_gqa(dt: DType) -> TestSetup {
        build_setup(1, 4, 4, 32, 64, true, 1.0, 0, dt)
    }

    #[test_kernel(name = "ffai/gated_delta_prep/step_bf16_qwen36_identity", dtypes = [bf16], tol = 1e-3)]
    fn prep_step_bf16_qwen36_shape_identity_weights(dt: DType) -> TestSetup {
        build_setup(1, 32, 16, 128, 128, true, 1.0, 0, dt)
    }

    #[test_kernel(name = "ffai/gated_delta_prep/step_bf16_qwen36_weighted", dtypes = [bf16], tol = 1e-3)]
    fn prep_step_bf16_qwen36_shape_nonidentity_weights(dt: DType) -> TestSetup {
        build_setup(1, 32, 16, 128, 128, false, 0.5, 0, dt)
    }

    #[test_kernel(name = "ffai/gated_delta_prep/step_bf16_no_gqa", dtypes = [bf16], tol = 1e-3)]
    fn prep_step_bf16_no_gqa(dt: DType) -> TestSetup {
        build_setup(1, 4, 4, 32, 64, true, 1.0, 0, dt)
    }

    #[test_kernel(name = "ffai/gated_delta_prep/step_f32_unfused_equiv", dtypes = [f32], tol = 1e-3)]
    fn prep_step_f32_matches_unfused_path_when_weights_identity(dt: DType) -> TestSetup {
        build_setup(1, 8, 4, 16, 64, true, 1.0, 0, dt)
    }

    /// Multi-step recurrence test: runs 8 CPU steps to accumulate state, then
    /// verifies the GPU output on the final step against the CPU oracle.
    #[test_kernel(name = "ffai/gated_delta_prep/step_f32_multi_step8_final", dtypes = [f32], tol = 1e-3)]
    fn prep_step_f32_multi_step_8_consecutive(dt: DType) -> TestSetup {
        let b = 1usize;
        let hv = 4usize;
        let hk = 2usize;
        let dv = 8usize;
        let dk = 64usize;
        let n_steps = 8usize;
        let weights_q = vec![0.7_f32; hk * dk];
        let weights_k = vec![0.7_f32; hk * dk];
        let a_log: Vec<f32> = (0..hv).map(|i| -1.0 - (i as f32) * 0.1).collect();
        let dt_bias: Vec<f32> = (0..hv).map(|i| -0.3 + (i as f32) * 0.05).collect();
        let stride_b = 2 * hk * dk + hv * dv;
        let n_total = b * hv;

        // Advance CPU state through all steps; use the last step's inputs as
        // the GPU test case with the penultimate CPU state as state_in.
        let mut state_cpu = vec![0.0_f32; b * hv * dv * dk];
        let mut last_conv_out = vec![0.0_f32; b * stride_b];
        let mut last_a_raw = vec![0.0_f32; b * hv];
        let mut last_b_raw = vec![0.0_f32; b * hv];
        let mut penultimate_state = state_cpu.clone();

        for step in 0..n_steps {
            let conv_out: Vec<f32> = (0..b * stride_b)
                .map(|i| (((i + step * 17) as f32) * 0.0131).sin() * 0.4)
                .collect();
            let a_raw: Vec<f32> =
                (0..b * hv).map(|i| -0.3 + ((i + step) as f32) * 0.04).collect();
            let b_raw: Vec<f32> =
                (0..b * hv).map(|i| -0.2 + ((i + step) as f32) * 0.03).collect();
            penultimate_state = state_cpu.clone();
            let (_, state_new) = cpu_fused_oracle(
                &conv_out,
                &a_log,
                &dt_bias,
                &a_raw,
                &b_raw,
                &weights_q,
                &weights_k,
                &state_cpu,
                b,
                hv,
                hk,
                dv,
                dk,
            );
            last_conv_out = conv_out;
            last_a_raw = a_raw;
            last_b_raw = b_raw;
            state_cpu = state_new;
        }

        // Final step: GPU takes penultimate_state as input, should match state_cpu.
        let last_step = n_steps - 1;
        let final_conv_out: Vec<f32> = (0..b * stride_b)
            .map(|i| (((i + last_step * 17) as f32) * 0.0131).sin() * 0.4)
            .collect();
        let final_a_raw: Vec<f32> =
            (0..b * hv).map(|i| -0.3 + ((i + last_step) as f32) * 0.04).collect();
        let final_b_raw: Vec<f32> =
            (0..b * hv).map(|i| -0.2 + ((i + last_step) as f32) * 0.03).collect();
        let (expected_y, expected_state) = cpu_fused_oracle(
            &final_conv_out,
            &a_log,
            &dt_bias,
            &final_a_raw,
            &final_b_raw,
            &weights_q,
            &weights_k,
            &penultimate_state,
            b,
            hv,
            hk,
            dv,
            dk,
        );

        let mut kernel_ir = mt_gated_delta_prep_step::kernel_ir_for(dt);
        kernel_ir.mode = KernelMode::Reduction;
        TestSetup::new(kernel_ir)
            .input(TestBuffer::from_vec("conv_out", pack(&final_conv_out, dt), dt))
            .input(TestBuffer::from_vec("a_log", pack(&a_log, dt), dt))
            .input(TestBuffer::from_vec("dt_bias", pack(&dt_bias, dt), dt))
            .input(TestBuffer::from_vec("a_raw", pack(&final_a_raw, dt), dt))
            .input(TestBuffer::from_vec("b_raw", pack(&final_b_raw, dt), dt))
            .input(TestBuffer::from_vec("q_norm_weight", pack(&weights_q, dt), dt))
            .input(TestBuffer::from_vec("k_norm_weight", pack(&weights_k, dt), dt))
            .input(TestBuffer::from_vec("state_in", pack(&penultimate_state, dt), dt))
            .input(TestBuffer::from_vec("dk", u32_le(dk as u32), DType::U32))
            .input(TestBuffer::from_vec("dv", u32_le(dv as u32), DType::U32))
            .input(TestBuffer::from_vec("hv", u32_le(hv as u32), DType::U32))
            .input(TestBuffer::from_vec("hk", u32_le(hk as u32), DType::U32))
            .expect(TestBuffer::from_vec("y", pack(&expected_y, dt), dt))
            .expect(TestBuffer::from_vec("state_out", pack(&expected_state, dt), dt))
            .grid_3d(dv as u32, n_total as u32, 1, [32, 1, 1])
    }
}
