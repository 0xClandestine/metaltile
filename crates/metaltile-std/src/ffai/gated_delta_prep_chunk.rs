//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Gated DeltaNet — **fused** prep + chunked-prefill kernel.
//!
//! `mt_gated_delta_prep_chunk` extends
//! [`mt_gated_delta_prep_step`](super::gated_delta_prep::mt_gated_delta_prep_step)
//! over a chunk of `T` tokens, mirroring the relationship between
//! [`mt_gated_delta_step`](super::gated_delta::mt_gated_delta_step) and
//! [`mt_gated_delta_chunk`](super::gated_delta::mt_gated_delta_chunk).
//!
//! State stays register-resident across the entire `T`-loop — one
//! load_state at entry and one store_state at exit, regardless of `T`.
//! This collapses the dominant `mt_gated_delta_prep_step`-per-token T-loop
//! in `Qwen35GDNMixer.forwardMany` to a single dispatch per layer.
//!
//! Inputs (note the added `T` dimension on conv_out / a_raw / b_raw):
//!   - `conv_out`     : Tensor<T> [B, T, 2·Hk·Dk + Hv·Dv]   q | k | v slabs
//!   - `a_log`        : Tensor<T> [Hv]                      per-Hv learnable
//!   - `dt_bias`      : Tensor<T> [Hv]
//!   - `a_raw`        : Tensor<T> [B, T, Hv]
//!   - `b_raw`        : Tensor<T> [B, T, Hv]
//!   - `q_norm_weight`: Tensor<T> [Hk·Dk]   (pass 1.0×invKeyScale² for unweighted q-scale)
//!   - `k_norm_weight`: Tensor<T> [Hk·Dk]   (pass 1.0×invKeyScale for unweighted k-scale)
//!   - `state_in`     : Tensor<T> [B, Hv, Dv, Dk]           (one state per (b, hv))
//!   - `t_len`        : Tensor<u32> [1]                     runtime chunk length
//!
//! Outputs:
//!   - `state_out`    : Tensor<T> [B, Hv, Dv, Dk]
//!   - `y`            : Tensor<T> [B, T, Hv, Dv]
//!
//! ## DISPATCH INVARIANTS (identical to `mt_gated_delta_prep_step`)
//!
//! - **Mode: Reduction.** Each TG is one simdgroup (32 threads).
//! - **Grid: `[Dv, B·Hv, 1]`, TG: `[32, 1, 1]`.**
//! - **`Dk % 32 == 0`.** Each lane owns `n_per_t = Dk / 32` slots.
//! - **Hv divisible by Hk.** GQA: `hk_idx = hv_idx / (Hv/Hk)`.
//! - **`t_len` is runtime u32** so a single PSO compiles for every chunk size.
//!
//! ## Per-iter cost vs prep_step
//!
//! Prep-step pays:
//!   - 1× state-load + 1× state-store (Dk floats per lane)
//!   - prep math + recurrence math
//!
//! Prep-chunk pays:
//!   - 1× state-load + 1× state-store (Dk floats per lane), TOTAL — not per-t
//!   - T × (prep math + recurrence math)
//!
//! State traffic per layer drops by `T`× at the dispatch boundary. For
//! Qwen3.6-A3B (Dk=256, Dv=128, Hv=16, B=1): state size = 16·128·256·4 B =
//! 2 MiB per direction. At T=512 the per-token loop did `T × (state R+W) = 2
//! GiB device traffic per layer per direction × 30 GDN layers = 120 GiB
//! per prefill step in state traffic alone. The chunked variant does
//! 2 MiB × 30 = 60 MiB.

use metaltile::kernel;

#[kernel]
pub fn mt_gated_delta_prep_chunk<T>(
    conv_out: Tensor<T>,      // [B, T, 2·Hk·Dk + Hv·Dv]
    a_log: Tensor<T>,         // [Hv]
    dt_bias: Tensor<T>,       // [Hv]
    a_raw: Tensor<T>,         // [B, T, Hv]
    b_raw: Tensor<T>,         // [B, T, Hv]
    q_norm_weight: Tensor<T>, // [Hk·Dk]
    k_norm_weight: Tensor<T>, // [Hk·Dk]
    state_in: Tensor<T>,      // [B, Hv, Dv, Dk]
    mut state_out: Tensor<T>, // [B, Hv, Dv, Dk]
    mut y: Tensor<T>,         // [B, T, Hv, Dv]
    t_len: Tensor<u32>,       // [1] scalar
    #[constexpr] dk: u32,
    #[constexpr] dv: u32,
    #[constexpr] hv: u32,
    #[constexpr] hk: u32,
) {
    let dv_idx = tgid_x;
    let n = tgid_y;
    let dk_idx = tid;
    // GQA decomposition.
    let hv_idx = n - (n / hv) * hv;
    let b = n / hv;
    let hk_per_hv = hv / hk;
    let hk_idx = hv_idx / hk_per_hv;
    let n_per_t = dk / 32u32;
    let t_total = load(t_len[0]);
    let stride_b = 2u32 * hk * dk + hv * dv;
    let eps = 0.000001f32;
    let dk_f = dk.cast::<f32>();
    // Per-layer constants (loaded once per TG).
    let a_log_val = load(a_log[hv_idx]).cast::<f32>();
    let dt_bias_val = load(dt_bias[hv_idx]).cast::<f32>();
    let exp_a_log = exp(a_log_val);
    let state_base = n * dv * dk + dv_idx * dk;
    // ─── Load state into per-lane registers ONCE — persists across the T-loop.
    stack_alloc("state_reg", 8u32, "f32");
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * dk_idx + i;
        let val = load(state_in[state_base + s_idx]).cast::<f32>();
        stack_store("state_reg", i, val);
    }
    // q_w / k_w are static across the T-loop (one row of weights per
    // hk_idx); load them once into per-lane stack so the inner T-loop
    // doesn't re-read.
    stack_alloc("q_w", 8u32, "f32");
    stack_alloc("k_w", 8u32, "f32");
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * dk_idx + i;
        let qw = load(q_norm_weight[hk_idx * dk + s_idx]).cast::<f32>();
        let kw = load(k_norm_weight[hk_idx * dk + s_idx]).cast::<f32>();
        stack_store("q_w", i, qw);
        stack_store("k_w", i, kw);
    }
    // Stack arrays reused per-token: q_raw / k_raw / k_cache.
    stack_alloc("q_raw", 8u32, "f32");
    stack_alloc("k_raw", 8u32, "f32");
    stack_alloc("k_cache", 8u32, "f32");
    // ─── Inner T-loop: prep + recurrence per token ──────────────────────
    for t in range(0u32, t_total, 1u32) {
        let bt = b * t_total + t;
        let conv_base = bt * stride_b;
        let q_off = conv_base + hk_idx * dk;
        let k_off = conv_base + hk * dk + hk_idx * dk;
        let v_off = conv_base + 2u32 * hk * dk + hv_idx * dv;
        let gbeta_idx = bt * hv + hv_idx;
        // ─── Phase 0a: Per-head RMSNorm of q / k ─────────────────────────
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
        }
        let q_ssq_sum = simd_sum(q_ssq);
        let k_ssq_sum = simd_sum(k_ssq);
        let q_inv = rsqrt(q_ssq_sum / dk_f + eps);
        let k_inv = rsqrt(k_ssq_sum / dk_f + eps);
        // ─── Phase 0b: g / beta ──────────────────────────────────────────
        let a_raw_val = load(a_raw[gbeta_idx]).cast::<f32>();
        let b_raw_val = load(b_raw[gbeta_idx]).cast::<f32>();
        let pre_softplus = a_raw_val + dt_bias_val;
        let dt_val = log(exp(pre_softplus) + 1.0f32);
        let g_val = exp(0.0f32 - exp_a_log * dt_val);
        let beta_val = 1.0f32 / (1.0f32 + exp(0.0f32 - b_raw_val));
        // v: one read per Dv slot per token.
        let v_val = load(conv_out[v_off + dv_idx]).cast::<f32>();
        // ─── Phase 1: decay state + accumulate kv_mem; cache k_normed ────
        let mut kv_mem = 0.0f32;
        for i in range(0u32, n_per_t, 1u32) {
            let s_old = stack_load("state_reg", i);
            let s_decayed = s_old * g_val;
            stack_store("state_reg", i, s_decayed);
            let k_normed = stack_load("k_raw", i) * k_inv * stack_load("k_w", i);
            stack_store("k_cache", i, k_normed);
            kv_mem = kv_mem + s_decayed * k_normed;
        }
        let kv_mem_sum = simd_sum(kv_mem);
        let delta = (v_val - kv_mem_sum) * beta_val;
        // ─── Phase 2: rank-1 update + output projection ──────────────────
        let mut out_acc = 0.0f32;
        for i in range(0u32, n_per_t, 1u32) {
            let s_decayed = stack_load("state_reg", i);
            let k_normed = stack_load("k_cache", i);
            let s_new = s_decayed + k_normed * delta;
            stack_store("state_reg", i, s_new);
            let q_normed = stack_load("q_raw", i) * q_inv * stack_load("q_w", i);
            out_acc = out_acc + s_new * q_normed;
        }
        let out_sum = simd_sum(out_acc);
        // ─── Phase 3: lane 0 writes y[t, n, dv_idx] ──────────────────────
        if dk_idx == 0u32 {
            store(y[(bt * hv + hv_idx) * dv + dv_idx], out_sum.cast::<T>());
        }
    }
    // ─── Write final state ONCE at the end ──────────────────────────────
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * dk_idx + i;
        store(state_out[state_base + s_idx], stack_load("state_reg", i).cast::<T>());
    }
}

#[cfg(test)]
mod tests {
    use metaltile_core::{DType, ir::KernelMode};

    use super::*;

    /// Developer aid — dump the full generated MSL for inspection.
    /// `cargo test -p metaltile-std --lib --release -- ffai::gated_delta_prep_chunk::tests::dump --nocapture`
    #[test]
    fn dump() {
        use metaltile_codegen::msl::MslGenerator;
        let mut k = mt_gated_delta_prep_chunk::kernel_ir_for(DType::F32);
        k.mode = KernelMode::Reduction;
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        println!("===== BEGIN MSL =====\n{}\n===== END MSL =====", msl);
    }
}

#[cfg(target_os = "macos")]
pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    //! GPU correctness tests for `mt_gated_delta_prep_chunk`.

    use metaltile::test_kernel;
    use metaltile_core::{
        DType,
        bench::{TestBuffer, TestSetup},
        ir::KernelMode,
    };

    use super::mt_gated_delta_prep_chunk;

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

    fn cpu_chunk_oracle(
        conv_out: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
        a_raw: &[f32],
        b_raw: &[f32],
        q_norm_weight: &[f32],
        k_norm_weight: &[f32],
        state_in: &[f32],
        b: usize,
        t: usize,
        hv: usize,
        hk: usize,
        dv: usize,
        dk: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let eps = 1e-6_f32;
        let stride_b = 2 * hk * dk + hv * dv;
        let hk_per_hv = hv / hk;
        let mut state = state_in.to_vec();
        let mut y_all = vec![0.0_f32; b * t * hv * dv];
        for step in 0..t {
            for batch in 0..b {
                let bt = batch * t + step;
                let conv_step_base = bt * stride_b;
                for hv_idx in 0..hv {
                    let n = batch * hv + hv_idx;
                    let hk_idx = hv_idx / hk_per_hv;
                    let q_off = conv_step_base + hk_idx * dk;
                    let k_off = conv_step_base + hk * dk + hk_idx * dk;
                    let v_off = conv_step_base + 2 * hk * dk + hv_idx * dv;
                    let mut q_ssq = 0.0_f32;
                    let mut k_ssq = 0.0_f32;
                    for d in 0..dk {
                        q_ssq += conv_out[q_off + d] * conv_out[q_off + d];
                        k_ssq += conv_out[k_off + d] * conv_out[k_off + d];
                    }
                    let q_inv = 1.0 / ((q_ssq / dk as f32) + eps).sqrt();
                    let k_inv = 1.0 / ((k_ssq / dk as f32) + eps).sqrt();
                    let dt_v = softplus(a_raw[bt * hv + hv_idx] + dt_bias[hv_idx]);
                    let g_val = (-a_log[hv_idx].exp() * dt_v).exp();
                    let beta_val = sigmoid(b_raw[bt * hv + hv_idx]);
                    for dv_idx in 0..dv {
                        let v_val = conv_out[v_off + dv_idx];
                        let s_base = n * dv * dk + dv_idx * dk;
                        let mut kv_mem = 0.0_f32;
                        let mut decayed = vec![0.0_f32; dk];
                        let mut k_normed_arr = vec![0.0_f32; dk];
                        for s_idx in 0..dk {
                            let s = state[s_base + s_idx] * g_val;
                            decayed[s_idx] = s;
                            let k_normed = conv_out[k_off + s_idx]
                                * k_inv
                                * k_norm_weight[hk_idx * dk + s_idx];
                            k_normed_arr[s_idx] = k_normed;
                            kv_mem += s * k_normed;
                        }
                        let delta = (v_val - kv_mem) * beta_val;
                        let mut out = 0.0_f32;
                        for s_idx in 0..dk {
                            let s_new = decayed[s_idx] + k_normed_arr[s_idx] * delta;
                            state[s_base + s_idx] = s_new;
                            let q_normed = conv_out[q_off + s_idx]
                                * q_inv
                                * q_norm_weight[hk_idx * dk + s_idx];
                            out += s_new * q_normed;
                        }
                        y_all[(bt * hv + hv_idx) * dv + dv_idx] = out;
                    }
                }
            }
        }
        (y_all, state)
    }

    #[allow(clippy::type_complexity)]
    fn make_inputs(
        b: usize,
        t: usize,
        hv: usize,
        hk: usize,
        dv: usize,
        dk: usize,
        identity: bool,
        ws: f32,
        dt: DType,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let stride_b = 2 * hk * dk + hv * dv;
        let round = |v: f32| match dt {
            DType::F16 => half::f16::from_f32(v).to_f32(),
            DType::BF16 => half::bf16::from_f32(v).to_f32(),
            _ => v,
        };
        let conv_out: Vec<f32> =
            (0..b * t * stride_b).map(|i| round(((i as f32) * 0.0131).sin() * 0.1)).collect();
        let a_log: Vec<f32> = (0..hv).map(|i| round(-2.0 - (i as f32) * 0.05)).collect();
        let dt_bias: Vec<f32> = (0..hv).map(|i| round(-0.3 + (i as f32) * 0.02)).collect();
        let a_raw: Vec<f32> =
            (0..b * t * hv).map(|i| round(-0.2 + ((i as f32) * 0.04).sin() * 0.2)).collect();
        let b_raw: Vec<f32> =
            (0..b * t * hv).map(|i| round(-0.2 + ((i as f32) * 0.03).cos() * 0.2)).collect();
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
        let state_in: Vec<f32> =
            (0..b * hv * dv * dk).map(|i| round(((i as f32) * 0.0073).cos() * 0.02)).collect();
        (conv_out, a_log, dt_bias, a_raw, b_raw, q_norm_weight, k_norm_weight, state_in)
    }

    fn build_setup(
        b: usize,
        t: usize,
        hv: usize,
        hk: usize,
        dv: usize,
        dk: usize,
        identity: bool,
        ws: f32,
        dt: DType,
    ) -> TestSetup {
        let (conv_out, a_log, dt_bias, a_raw, b_raw, q_norm_weight, k_norm_weight, state_in) =
            make_inputs(b, t, hv, hk, dv, dk, identity, ws, dt);
        let n_total = b * hv;
        let (expected_y, expected_state) = cpu_chunk_oracle(
            &conv_out,
            &a_log,
            &dt_bias,
            &a_raw,
            &b_raw,
            &q_norm_weight,
            &k_norm_weight,
            &state_in,
            b,
            t,
            hv,
            hk,
            dv,
            dk,
        );
        let mut kernel_ir = mt_gated_delta_prep_chunk::kernel_ir_for(dt);
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
            .input(TestBuffer::from_vec("t_len", u32_le(t as u32), DType::U32))
            .input(TestBuffer::from_vec("dk", u32_le(dk as u32), DType::U32))
            .input(TestBuffer::from_vec("dv", u32_le(dv as u32), DType::U32))
            .input(TestBuffer::from_vec("hv", u32_le(hv as u32), DType::U32))
            .input(TestBuffer::from_vec("hk", u32_le(hk as u32), DType::U32))
            .expect(TestBuffer::from_vec("y", pack(&expected_y, dt), dt))
            .expect(TestBuffer::from_vec("state_out", pack(&expected_state, dt), dt))
            .grid_3d(dv as u32, n_total as u32, 1, [32, 1, 1])
    }

    #[test_kernel(name = "ffai/gated_delta_prep/chunk_f32_qwen36_t1", dtypes = [f32], tol = 1e-3)]
    fn prep_chunk_f32_qwen36_t1(dt: DType) -> TestSetup {
        build_setup(1, 1, 32, 16, 128, 128, true, 1.0, dt)
    }

    #[test_kernel(name = "ffai/gated_delta_prep/chunk_f32_qwen36_t8_identity", dtypes = [f32], tol = 1e-3)]
    fn prep_chunk_f32_qwen36_t8_identity(dt: DType) -> TestSetup {
        build_setup(1, 8, 32, 16, 128, 128, true, 1.0, dt)
    }

    #[test_kernel(name = "ffai/gated_delta_prep/chunk_f32_qwen36_t8_weighted", dtypes = [f32], tol = 1e-3)]
    fn prep_chunk_f32_qwen36_t8_weighted(dt: DType) -> TestSetup {
        build_setup(1, 8, 32, 16, 128, 128, false, 0.5, dt)
    }

    #[test_kernel(name = "ffai/gated_delta_prep/chunk_f32_long_t32_small", dtypes = [f32], tol = 1e-3)]
    fn prep_chunk_f32_long_t32_small_shape(dt: DType) -> TestSetup {
        build_setup(1, 32, 4, 2, 8, 32, true, 1.0, dt)
    }

    #[test_kernel(name = "ffai/gated_delta_prep/chunk_f32_dk256", dtypes = [f32], tol = 1e-3)]
    fn prep_chunk_f32_dk_256(dt: DType) -> TestSetup {
        build_setup(1, 4, 4, 2, 8, 256, true, 1.0, dt)
    }

    #[test_kernel(name = "ffai/gated_delta_prep/chunk_f16_t1_smoke", dtypes = [f16], tol = 1e-3)]
    fn prep_chunk_f16_t1_smoke(dt: DType) -> TestSetup {
        build_setup(1, 1, 4, 2, 8, 32, true, 1.0, dt)
    }

    #[test_kernel(name = "ffai/gated_delta_prep/chunk_bf16_qwen36_t8", dtypes = [bf16], tol = 1e-3)]
    fn prep_chunk_bf16_qwen36_t8(dt: DType) -> TestSetup {
        build_setup(1, 8, 32, 16, 128, 128, true, 1.0, dt)
    }

    #[test_kernel(name = "ffai/gated_delta_prep/chunk_bf16_small_t32", dtypes = [bf16], tol = 1e-3)]
    fn prep_chunk_bf16_small_shape_t32(dt: DType) -> TestSetup {
        build_setup(1, 32, 4, 2, 8, 32, true, 1.0, dt)
    }
}
