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
pub mod tests_support {
    //! GPU correctness tests for `mt_gated_delta_prep_step`.
    #![allow(clippy::too_many_arguments, clippy::type_complexity)]

    use std::collections::BTreeMap;

    use metaltile_core::{dtype::DType, ir::KernelMode};
    use metaltile_runtime::Context;

    use super::mt_gated_delta_prep_step;

    #[derive(Clone, Copy, Debug)]
    enum Dt { F32, F16, Bf16 }
    impl Dt {
        fn to_dtype(self) -> DType { match self { Dt::F32 => DType::F32, Dt::F16 => DType::F16, Dt::Bf16 => DType::BF16 } }
        fn round(self, v: f32) -> f32 { match self { Dt::F32 => v, Dt::F16 => half::f16::from_f32(v).to_f32(), Dt::Bf16 => half::bf16::from_f32(v).to_f32() } }
    }
    fn pack_bytes(vals: &[f32], dt: Dt) -> Vec<u8> {
        match dt { Dt::F32 => bytemuck::cast_slice::<f32,u8>(vals).to_vec(), Dt::F16 => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(), Dt::Bf16 => vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect() }
    }
    fn unpack_bytes(bytes: &[u8], dt: Dt) -> Vec<f32> {
        match dt { Dt::F32 => bytemuck::cast_slice::<u8,f32>(bytes).to_vec(), Dt::F16 => bytes.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0],c[1]]).to_f32()).collect(), Dt::Bf16 => bytes.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0],c[1]]).to_f32()).collect() }
    }
    fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(())).lock().unwrap_or_else(|e| e.into_inner())
    }
    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0_f64; let mut na = 0.0_f64; let mut nb = 0.0_f64;
        for (av, bv) in a.iter().zip(b.iter()) { dot += *av as f64 * *bv as f64; na += *av as f64 * *av as f64; nb += *bv as f64 * *bv as f64; }
        (dot / (na.sqrt() * nb.sqrt())) as f32
    }

    fn softplus_unclamped(x: f32) -> f32 { (x.exp() + 1.0).ln() }
    fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

    fn cpu_prep(conv_out: &[f32], a_log: &[f32], dt_bias: &[f32], a_raw: &[f32], b_raw: &[f32], q_norm_weight: &[f32], k_norm_weight: &[f32], b: usize, hv: usize, hk: usize, dv: usize, dk: usize) -> (Vec<f32>,Vec<f32>,Vec<f32>,Vec<f32>,Vec<f32>) {
        let eps = 1e-6_f32; let stride_b = 2*hk*dk+hv*dv;
        let mut q_normed = vec![0.0_f32; b*hk*dk]; let mut k_normed = vec![0.0_f32; b*hk*dk];
        let mut v_flat = vec![0.0_f32; b*hv*dv]; let mut g = vec![0.0_f32; b*hv]; let mut beta = vec![0.0_f32; b*hv];
        for batch in 0..b {
            let q_base = batch*stride_b; let k_base = q_base+hk*dk; let v_base = q_base+2*hk*dk;
            for hk_idx in 0..hk {
                let row_off = hk_idx*dk; let mut q_ssq = 0.0_f32; let mut k_ssq = 0.0_f32;
                for d in 0..dk { q_ssq += conv_out[q_base+row_off+d]*conv_out[q_base+row_off+d]; k_ssq += conv_out[k_base+row_off+d]*conv_out[k_base+row_off+d]; }
                let q_inv = 1.0/((q_ssq/dk as f32)+eps).sqrt(); let k_inv = 1.0/((k_ssq/dk as f32)+eps).sqrt();
                for d in 0..dk { q_normed[batch*hk*dk+row_off+d] = conv_out[q_base+row_off+d]*q_inv*q_norm_weight[hk_idx*dk+d]; k_normed[batch*hk*dk+row_off+d] = conv_out[k_base+row_off+d]*k_inv*k_norm_weight[hk_idx*dk+d]; }
            }
            for hv_idx in 0..hv { for dv_idx in 0..dv { v_flat[(batch*hv+hv_idx)*dv+dv_idx] = conv_out[v_base+hv_idx*dv+dv_idx]; } }
            for hv_idx in 0..hv { let n = batch*hv+hv_idx; let dt_v = softplus_unclamped(a_raw[n]+dt_bias[hv_idx]); g[n] = (-a_log[hv_idx].exp()*dt_v).exp(); beta[n] = sigmoid(b_raw[n]); }
        }
        (q_normed, k_normed, v_flat, g, beta)
    }

    fn cpu_step(q: &[f32], k: &[f32], v: &[f32], g: &[f32], beta: &[f32], state_in: &[f32], b: usize, hv: usize, hk: usize, dv: usize, dk: usize) -> (Vec<f32>, Vec<f32>) {
        let mut y = vec![0.0_f32; b*hv*dv]; let mut state_out = vec![0.0_f32; b*hv*dv*dk];
        let hk_per_hv = hv/hk;
        for batch in 0..b { for hv_idx in 0..hv {
            let n = batch*hv+hv_idx; let hk_idx = hv_idx/hk_per_hv;
            let g_val = g[n]; let beta_val = beta[n]; let qk_base = (batch*hk+hk_idx)*dk;
            for dv_idx in 0..dv {
                let v_val = v[n*dv+dv_idx]; let s_base = n*dv*dk+dv_idx*dk;
                let mut kv_mem = 0.0_f32; let mut decayed = vec![0.0_f32; dk];
                for s_idx in 0..dk { let s = state_in[s_base+s_idx]*g_val; decayed[s_idx] = s; kv_mem += s*k[qk_base+s_idx]; }
                let delta = (v_val - kv_mem)*beta_val; let mut out = 0.0_f32;
                for s_idx in 0..dk { let s_new = decayed[s_idx]+k[qk_base+s_idx]*delta; state_out[s_base+s_idx] = s_new; out += s_new*q[qk_base+s_idx]; }
                y[n*dv+dv_idx] = out;
            }
        }}
        (y, state_out)
    }

    fn cpu_fused_oracle(conv_out: &[f32], a_log: &[f32], dt_bias: &[f32], a_raw: &[f32], b_raw: &[f32], q_norm_weight: &[f32], k_norm_weight: &[f32], state_in: &[f32], b: usize, hv: usize, hk: usize, dv: usize, dk: usize) -> (Vec<f32>, Vec<f32>) {
        let (q, k, v, g, beta) = cpu_prep(conv_out, a_log, dt_bias, a_raw, b_raw, q_norm_weight, k_norm_weight, b, hv, hk, dv, dk);
        cpu_step(&q, &k, &v, &g, &beta, state_in, b, hv, hk, dv, dk)
    }

    struct Fixture { conv_out: Vec<f32>, a_log: Vec<f32>, dt_bias: Vec<f32>, a_raw: Vec<f32>, b_raw: Vec<f32>, q_norm_weight: Vec<f32>, k_norm_weight: Vec<f32>, state_in: Vec<f32> }

    fn make_fixture(b: usize, hv: usize, hk: usize, dv: usize, dk: usize, identity: bool, ws: f32, seed_offset: usize) -> Fixture {
        let stride_b = 2*hk*dk+hv*dv;
        let conv_out = (0..b*stride_b).map(|i| (((i+seed_offset) as f32)*0.0131).sin()*0.4).collect();
        let a_log = (0..hv).map(|i| -1.5-(i as f32)*0.1).collect();
        let dt_bias = (0..hv).map(|i| -0.5+(i as f32)*0.05).collect();
        let a_raw = (0..b*hv).map(|i| -0.3+(i as f32)*0.04).collect();
        let b_raw = (0..b*hv).map(|i| -0.2+(i as f32)*0.03).collect();
        let q_norm_weight = if identity { vec![ws; hk*dk] } else { (0..hk*dk).map(|i| ws*(1.0+((i%11) as f32)*0.05)).collect() };
        let k_norm_weight = if identity { vec![ws; hk*dk] } else { (0..hk*dk).map(|i| ws*(1.0+((i%13) as f32)*0.04)).collect() };
        let state_in = (0..b*hv*dv*dk).map(|i| (((i+seed_offset) as f32)*0.0073).cos()*0.1).collect();
        Fixture { conv_out, a_log, dt_bias, a_raw, b_raw, q_norm_weight, k_norm_weight, state_in }
    }

    fn round_fixture(f: &Fixture, dt: Dt) -> Fixture {
        let r = |xs: &[f32]| xs.iter().map(|&v| dt.round(v)).collect::<Vec<_>>();
        Fixture { conv_out: r(&f.conv_out), a_log: r(&f.a_log), dt_bias: r(&f.dt_bias), a_raw: r(&f.a_raw), b_raw: r(&f.b_raw), q_norm_weight: r(&f.q_norm_weight), k_norm_weight: r(&f.k_norm_weight), state_in: r(&f.state_in) }
    }

    fn run_gpu(f: &Fixture, dt: Dt, b: usize, hv: usize, hk: usize, dv: usize, dk: usize) -> (Vec<f32>, Vec<f32>) {
        assert!(dk.is_multiple_of(32));
        let n_total = b*hv;
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("conv_out".into(), pack_bytes(&f.conv_out, dt));
        buffers.insert("a_log".into(), pack_bytes(&f.a_log, dt));
        buffers.insert("dt_bias".into(), pack_bytes(&f.dt_bias, dt));
        buffers.insert("a_raw".into(), pack_bytes(&f.a_raw, dt));
        buffers.insert("b_raw".into(), pack_bytes(&f.b_raw, dt));
        buffers.insert("q_norm_weight".into(), pack_bytes(&f.q_norm_weight, dt));
        buffers.insert("k_norm_weight".into(), pack_bytes(&f.k_norm_weight, dt));
        buffers.insert("state_in".into(), pack_bytes(&f.state_in, dt));
        buffers.insert("state_out".into(), pack_bytes(&vec![0.0_f32; f.state_in.len()], dt));
        buffers.insert("y".into(), pack_bytes(&vec![0.0_f32; n_total*dv], dt));
        buffers.insert("dk".into(), (dk as u32).to_le_bytes().to_vec());
        buffers.insert("dv".into(), (dv as u32).to_le_bytes().to_vec());
        buffers.insert("hv".into(), (hv as u32).to_le_bytes().to_vec());
        buffers.insert("hk".into(), (hk as u32).to_le_bytes().to_vec());
        let ctx = Context::new().expect("Context::new");
        let mut kernel = mt_gated_delta_prep_step::kernel_ir_for(dt.to_dtype());
        kernel.mode = KernelMode::Reduction;
        let result = ctx.dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [dv, n_total, 1], [32, 1, 1]).expect("dispatch");
        let y = unpack_bytes(result.outputs.get("y").expect("y"), dt);
        let state_out = unpack_bytes(result.outputs.get("state_out").expect("state_out"), dt);
        (y, state_out)
    }

    fn run_cell(b: usize, hv: usize, hk: usize, dv: usize, dk: usize, dt: Dt, identity: bool, ws: f32) -> (f32, f32) {
        let _g = gpu_lock();
        let raw = make_fixture(b, hv, hk, dv, dk, identity, ws, 0);
        let f = round_fixture(&raw, dt);
        let (y_cpu, state_cpu) = cpu_fused_oracle(&f.conv_out,&f.a_log,&f.dt_bias,&f.a_raw,&f.b_raw,&f.q_norm_weight,&f.k_norm_weight,&f.state_in,b,hv,hk,dv,dk);
        let (y_gpu, state_gpu) = run_gpu(&f, dt, b, hv, hk, dv, dk);
        (cosine(&y_gpu, &y_cpu), cosine(&state_gpu, &state_cpu))
    }

    #[test] fn prep_step_f32_qwen36_shape_identity_weights() { let (cy,cs)=run_cell(1,32,16,128,128,Dt::F32,true,1.0); assert!(cy>=0.999,"f32 identity y={cy:.6}"); assert!(cs>=0.999,"f32 identity s={cs:.6}"); }
    #[test] fn prep_step_f32_qwen36_shape_nonidentity_weights() { let (cy,cs)=run_cell(1,32,16,128,128,Dt::F32,false,0.5); assert!(cy>=0.999,"f32 weighted y={cy:.6}"); assert!(cs>=0.999,"f32 weighted s={cs:.6}"); }
    #[test] fn prep_step_f32_no_gqa() { let (cy,cs)=run_cell(1,4,4,32,64,Dt::F32,true,1.0); assert!(cy>=0.999,"f32 no-GQA y={cy:.6}"); assert!(cs>=0.999,"f32 no-GQA s={cs:.6}"); }
    #[test] fn prep_step_f32_dk_256_full_n_per_t_slot_usage() { let (cy,cs)=run_cell(1,4,2,8,256,Dt::F32,true,1.0); assert!(cy>=0.999,"f32 Dk=256 y={cy:.6}"); assert!(cs>=0.999,"f32 Dk=256 s={cs:.6}"); }
    #[test] fn prep_step_f32_batch_2() { let (cy,cs)=run_cell(2,4,2,8,64,Dt::F32,false,0.7); assert!(cy>=0.999,"f32 B=2 y={cy:.6}"); assert!(cs>=0.999,"f32 B=2 s={cs:.6}"); }
    #[test] fn prep_step_f16_qwen36_shape_identity_weights() { let (cy,cs)=run_cell(1,32,16,128,128,Dt::F16,true,1.0); assert!(cy>=0.999,"f16 identity y={cy:.6}"); assert!(cs>=0.999,"f16 identity s={cs:.6}"); }
    #[test] fn prep_step_f16_qwen36_shape_nonidentity_weights() { let (cy,cs)=run_cell(1,32,16,128,128,Dt::F16,false,0.5); assert!(cy>=0.999,"f16 weighted y={cy:.6}"); assert!(cs>=0.999,"f16 weighted s={cs:.6}"); }
    #[test] fn prep_step_f16_no_gqa() { let (cy,cs)=run_cell(1,4,4,32,64,Dt::F16,true,1.0); assert!(cy>=0.999,"f16 no-GQA y={cy:.6}"); assert!(cs>=0.999,"f16 no-GQA s={cs:.6}"); }
    #[test] fn prep_step_bf16_qwen36_shape_identity_weights() { let (cy,cs)=run_cell(1,32,16,128,128,Dt::Bf16,true,1.0); assert!(cy>=0.999,"bf16 identity y={cy:.6}"); assert!(cs>=0.999,"bf16 identity s={cs:.6}"); }
    #[test] fn prep_step_bf16_qwen36_shape_nonidentity_weights() { let (cy,cs)=run_cell(1,32,16,128,128,Dt::Bf16,false,0.5); assert!(cy>=0.999,"bf16 weighted y={cy:.6}"); assert!(cs>=0.999,"bf16 weighted s={cs:.6}"); }
    #[test] fn prep_step_bf16_no_gqa() { let (cy,cs)=run_cell(1,4,4,32,64,Dt::Bf16,true,1.0); assert!(cy>=0.999,"bf16 no-GQA y={cy:.6}"); assert!(cs>=0.999,"bf16 no-GQA s={cs:.6}"); }
    #[test] fn prep_step_f32_matches_unfused_path_when_weights_identity() { let (cy,cs)=run_cell(1,8,4,16,64,Dt::F32,true,1.0); assert!(cy>=0.999,"f32 unfused-equiv y={cy:.6}"); assert!(cs>=0.999,"f32 unfused-equiv s={cs:.6}"); }

    #[test]
    fn prep_step_f32_multi_step_8_consecutive() {
        let _g = gpu_lock();
        let b=1; let hv=4; let hk=2; let dv=8; let dk=64; let n_steps=8;
        let weights_q = vec![0.7_f32; hk*dk]; let weights_k = vec![0.7_f32; hk*dk];
        let a_log: Vec<f32> = (0..hv).map(|i| -1.0-(i as f32)*0.1).collect();
        let dt_bias: Vec<f32> = (0..hv).map(|i| -0.3+(i as f32)*0.05).collect();
        let mut state_gpu = vec![0.0_f32; b*hv*dv*dk]; let mut state_cpu = state_gpu.clone();
        for step in 0..n_steps {
            let stride_b = 2*hk*dk+hv*dv;
            let conv_out: Vec<f32> = (0..b*stride_b).map(|i| (((i+step*17) as f32)*0.0131).sin()*0.4).collect();
            let a_raw: Vec<f32> = (0..b*hv).map(|i| -0.3+((i+step) as f32)*0.04).collect();
            let b_raw: Vec<f32> = (0..b*hv).map(|i| -0.2+((i+step) as f32)*0.03).collect();
            let (y_cpu, state_cpu_new) = cpu_fused_oracle(&conv_out,&a_log,&dt_bias,&a_raw,&b_raw,&weights_q,&weights_k,&state_cpu,b,hv,hk,dv,dk);
            let f = Fixture { conv_out: conv_out.clone(), a_log: a_log.clone(), dt_bias: dt_bias.clone(), a_raw: a_raw.clone(), b_raw: b_raw.clone(), q_norm_weight: weights_q.clone(), k_norm_weight: weights_k.clone(), state_in: state_gpu.clone() };
            let (y_gpu, state_gpu_new) = run_gpu(&f, Dt::F32, b, hv, hk, dv, dk);
            let cy = cosine(&y_gpu, &y_cpu); let cs = cosine(&state_gpu_new, &state_cpu_new);
            assert!(cy >= 0.999, "step {step} y cos = {cy:.6}");
            assert!(cs >= 0.999, "step {step} state cos = {cs:.6}");
            state_gpu = state_gpu_new; state_cpu = state_cpu_new;
        }
    }
}
