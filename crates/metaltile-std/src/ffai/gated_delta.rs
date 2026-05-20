//! GatedDeltaNet single-token decode recurrence — port of
//! `gated_delta.metal` from ekryski/mlx@alpha. Required for the
//! GDN-bearing hybrid models (Qwen 3.5 / 3.6).
//!
//! Recurrence (per (batch, v-head), one decode step over `t_val` tokens):
//!   S_t = g_t · S_{t-1} + β_t · k_t · (v_t − kᵀ_t · S_{t-1})ᵀ
//!   y_t = q_t · S_t
//!
//! Two kernels:
//!   - `gated_delta_step`        — pre-computed normalized q/k, g, beta.
//!   - `gated_delta_step_fused`  — raw q/k; computes RMSNorm(q), RMSNorm(k),
//!     `g = exp(−exp(a_log)·softplus(a + dt_bias))`, `beta = sigmoid(b)`
//!     internally.
//!
//! Threading: a 32-lane simdgroup splits the `Dk` state axis
//! (`n_per_t = Dk/32` per lane); `program_id::<1>()` = `Dv` index,
//! `program_id::<2>()` = `batch*Hv`. The recurrent state slice is held
//! in a per-thread `stack_alloc` array; `simd_sum` reduces the
//! `kᵀS` / `qS` dot products across the `Dk` lanes.
//!
//! `mask` (u32, 0/1) skips masked decode positions when the constexpr
//! `has_mask` flag is set.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Grid3D**, `grid = [1, Dv, batch*Hv]`, `tg = [32, 1, 1]` — one
//!   simdgroup per `(n, dv)`.
//! - `Dk` a multiple of 32.
//!
//! Codegen-only; correctness pinned by
//! `tests/gated_delta_gpu_correctness.rs`.

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

macro_rules! gdn_spec {
    ($name:ident, $subop:literal) => {
        inventory::submit! {
            BenchSpec {
                op: "gated_delta",
                subop: $subop,
                kernel_name: stringify!($name),
                kernel_ir: $name::kernel_ir_for,
                dtypes: &[DType::F32, DType::F16, DType::BF16],
                tol: 1e-3,
                mlx_src: None,
                mlx_pattern: None,
                shapes: &[],
                dispatch: BenchDispatch::Generic,
                kernel_mode: Some(KernelMode::Grid3D),
            }
        }
    };
}

// ── Standard GatedDelta: pre-normalized q/k, explicit g/beta ────────────────
macro_rules! gated_delta_standard {
    ($name:ident, $dk:literal, $dv:literal, $hk:literal, $hv:literal, $n_per_t:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            q: Tensor<T>,
            k: Tensor<T>,
            v: Tensor<T>,
            g: Tensor<T>,
            beta: Tensor<T>,
            state_in: Tensor<T>,
            mask: Tensor<u32>,
            mut y: Tensor<T>,
            mut state_out: Tensor<T>,
            #[constexpr] t_val: u32,
            #[constexpr] has_mask: u32,
        ) {
            let lane = program_id::<0>();
            let dv_idx = program_id::<1>();
            let n = program_id::<2>();
            let b_idx = n / $hv;
            let hv_idx = n - b_idx * $hv;
            let hk_idx = hv_idx / ($hv / $hk);
            let i_state_base = (n * $dv + dv_idx) * $dk;

            stack_alloc("state", $n_per_t, "f32");
            for i in range(0u32, $n_per_t, 1u32) {
                let v = load(state_in[i_state_base + $n_per_t * lane + i]).cast::<f32>();
                stack_store("state", i, v);
            }

            for t in range(0u32, t_val, 1u32) {
                let m = select(has_mask == 0u32, 1u32, load(mask[b_idx * t_val + t]));
                if m > 0u32 {
                    let qk_base = (b_idx * t_val + t) * $hk * $dk + hk_idx * $dk;
                    let v_base = (b_idx * t_val + t) * $hv * $dv + hv_idx * $dv;
                    let gb_idx = (b_idx * t_val + t) * $hv + hv_idx;
                    let g_val = load(g[gb_idx]).cast::<f32>();
                    let beta_val = load(beta[gb_idx]).cast::<f32>();

                    let mut kv_mem = 0.0f32;
                    for i in range(0u32, $n_per_t, 1u32) {
                        let s_idx = $n_per_t * lane + i;
                        let st = stack_load("state", i) * g_val;
                        stack_store("state", i, st);
                        kv_mem = kv_mem + st * load(k[qk_base + s_idx]).cast::<f32>();
                    }
                    let kv = simd_sum(kv_mem);
                    let delta = (load(v[v_base + dv_idx]).cast::<f32>() - kv) * beta_val;

                    let mut out_acc = 0.0f32;
                    for i in range(0u32, $n_per_t, 1u32) {
                        let s_idx = $n_per_t * lane + i;
                        let st =
                            stack_load("state", i) + load(k[qk_base + s_idx]).cast::<f32>() * delta;
                        stack_store("state", i, st);
                        out_acc = out_acc + st * load(q[qk_base + s_idx]).cast::<f32>();
                    }
                    let out_red = simd_sum(out_acc);
                    if lane == 0u32 {
                        store(y[v_base + dv_idx], out_red.cast::<T>());
                    }
                }
            }

            for i in range(0u32, $n_per_t, 1u32) {
                let st = stack_load("state", i);
                store(state_out[i_state_base + $n_per_t * lane + i], st.cast::<T>());
            }
        }
        gdn_spec!($name, $subop);
    };
}

// ── Fused GatedDelta: raw q/k, absorbs RMSNorm + g + beta ───────────────────
macro_rules! gated_delta_fused {
    ($name:ident, $dk:literal, $dk_f:literal, $dv:literal, $hk:literal, $hv:literal, $n_per_t:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            q_raw: Tensor<T>,
            k_raw: Tensor<T>,
            v: Tensor<T>,
            a: Tensor<T>,
            b_input: Tensor<T>,
            a_log: Tensor<T>,
            dt_bias: Tensor<T>,
            state_in: Tensor<T>,
            mask: Tensor<u32>,
            mut y: Tensor<T>,
            mut state_out: Tensor<T>,
            #[constexpr] t_val: u32,
            #[constexpr] has_mask: u32,
        ) {
            let lane = program_id::<0>();
            let dv_idx = program_id::<1>();
            let n = program_id::<2>();
            let b_idx = n / $hv;
            let hv_idx = n - b_idx * $hv;
            let hk_idx = hv_idx / ($hv / $hk);
            let i_state_base = (n * $dv + dv_idx) * $dk;

            // RMSNorm post-scales: q by 1/Dk, k by 1/sqrt(Dk).
            let dk_f = $dk_f;
            let inv_scale_sq = 1.0f32 / dk_f;
            let inv_scale_single = rsqrt(dk_f);
            let exp_a_log = exp(load(a_log[hv_idx]).cast::<f32>());
            let dt_bias_v = load(dt_bias[hv_idx]).cast::<f32>();

            stack_alloc("state", $n_per_t, "f32");
            for i in range(0u32, $n_per_t, 1u32) {
                let v = load(state_in[i_state_base + $n_per_t * lane + i]).cast::<f32>();
                stack_store("state", i, v);
            }
            stack_alloc("qn", $n_per_t, "f32");
            stack_alloc("kn", $n_per_t, "f32");

            for t in range(0u32, t_val, 1u32) {
                let m = select(has_mask == 0u32, 1u32, load(mask[b_idx * t_val + t]));
                if m > 0u32 {
                    let qk_base = (b_idx * t_val + t) * $hk * $dk + hk_idx * $dk;
                    let v_base = (b_idx * t_val + t) * $hv * $dv + hv_idx * $dv;
                    let ab_idx = (b_idx * t_val + t) * $hv + hv_idx;

                    // RMSNorm(q): sum-of-squares across the Dk lanes.
                    let mut q_ss = 0.0f32;
                    for i in range(0u32, $n_per_t, 1u32) {
                        let qv = load(q_raw[qk_base + $n_per_t * lane + i]).cast::<f32>();
                        stack_store("qn", i, qv);
                        q_ss = q_ss + qv * qv;
                    }
                    let q_rms = rsqrt(simd_sum(q_ss) / dk_f + 1e-6f32);
                    for i in range(0u32, $n_per_t, 1u32) {
                        stack_store("qn", i, stack_load("qn", i) * q_rms * inv_scale_sq);
                    }

                    // RMSNorm(k).
                    let mut k_ss = 0.0f32;
                    for i in range(0u32, $n_per_t, 1u32) {
                        let kv = load(k_raw[qk_base + $n_per_t * lane + i]).cast::<f32>();
                        stack_store("kn", i, kv);
                        k_ss = k_ss + kv * kv;
                    }
                    let k_rms = rsqrt(simd_sum(k_ss) / dk_f + 1e-6f32);
                    for i in range(0u32, $n_per_t, 1u32) {
                        stack_store("kn", i, stack_load("kn", i) * k_rms * inv_scale_single);
                    }

                    // g = exp(-exp(a_log) * softplus(a + dt_bias)).
                    let dt = load(a[ab_idx]).cast::<f32>() + dt_bias_v;
                    let sp = select(dt > 20.0f32, dt, log(1.0f32 + exp(dt)));
                    let g_val = exp(0.0f32 - exp_a_log * sp);
                    // beta = sigmoid(b).
                    let beta_val = sigmoid(load(b_input[ab_idx]).cast::<f32>());

                    let mut kv_mem = 0.0f32;
                    for i in range(0u32, $n_per_t, 1u32) {
                        let st = stack_load("state", i) * g_val;
                        stack_store("state", i, st);
                        kv_mem = kv_mem + st * stack_load("kn", i);
                    }
                    let kv = simd_sum(kv_mem);
                    let delta = (load(v[v_base + dv_idx]).cast::<f32>() - kv) * beta_val;

                    let mut out_acc = 0.0f32;
                    for i in range(0u32, $n_per_t, 1u32) {
                        let st = stack_load("state", i) + stack_load("kn", i) * delta;
                        stack_store("state", i, st);
                        out_acc = out_acc + st * stack_load("qn", i);
                    }
                    let out_red = simd_sum(out_acc);
                    if lane == 0u32 {
                        store(y[v_base + dv_idx], out_red.cast::<T>());
                    }
                }
            }

            for i in range(0u32, $n_per_t, 1u32) {
                let st = stack_load("state", i);
                store(state_out[i_state_base + $n_per_t * lane + i], st.cast::<T>());
            }
        }
        gdn_spec!($name, $subop);
    };
}

// Qwen 3.5-A3B: Dk=192, Dv=128, Hk=4, Hv=4.
gated_delta_standard!(
    gated_delta_step_d192_128_4_4,
    192u32,
    128u32,
    4u32,
    4u32,
    6u32,
    "step_d192_128_4_4"
);
gated_delta_fused!(
    gated_delta_step_fused_d192_128_4_4,
    192u32,
    192.0f32,
    128u32,
    4u32,
    4u32,
    6u32,
    "fused_d192_128_4_4"
);
// Qwen 3.5 dense: Dk=128, Dv=128, Hk=16, Hv=16.
gated_delta_standard!(
    gated_delta_step_d128_128_16_16,
    128u32,
    128u32,
    16u32,
    16u32,
    4u32,
    "step_d128_128_16_16"
);
gated_delta_fused!(
    gated_delta_step_fused_d128_128_16_16,
    128u32,
    128.0f32,
    128u32,
    16u32,
    16u32,
    4u32,
    "fused_d128_128_16_16"
);
