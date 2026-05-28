//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Mamba 2 (SSD-form) building blocks: the selective-scan single-token
//! decode step and the depthwise causal-conv streaming step. Plus
//! `ssm_step_a2d` — the Mamba 1 (Jamba) variant carrying a 2-D
//! per-(channel, state) `A_log` instead of the scalar-per-head `A`.
//!
//! `mt_ssm_step` is a faithful port of MLX's `ssm_step<T, Dh, Ds, H, G>`
//! from ekryski's `mlx` fork (`alpha` branch) — semantically MLX-aligned
//! but mainline MLX (pinned by `metaltile-std/build.rs`) doesn't ship
//! the `ssm.metal` source yet, so there's no side-by-side comparison
//! today. When the pin moves to a commit that ships `ssm.metal`, this
//! file (or just `mt_ssm_step` alone) graduates to `mlx/ssm.rs` and
//! picks up an MLX bench comparison via the standard `mlx=` /
//! `metal_file=` annotations.
//!
//! All three kernels run their `h`/state accumulators in fp32 — the
//! `exp(A*dt)*h + dt*B*x` recurrence in bf16 drifts in a few dozen
//! decode steps. Activation tensors stay in whatever dtype the model
//! runs at (typically bf16).
//!
//! Codegen-only. Correctness validated end-to-end in FFAI integration
//! tests against real Mamba/Nemotron decoding.

use metaltile::kernel;

mod tests_support {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::test_kernel;
    use metaltile_core::{
        DType,
        bench::{TestBuffer, TestSetup},
    };

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

    /// CPU oracle for ssm_step (Mamba 2 SSD-form single-token decode).
    fn naive_ssm_step(
        x: &[f32],
        a: &[f32],
        b_vec: &[f32],
        c_vec: &[f32],
        dt: &[f32],
        h_state: &mut [f32],
        n_heads: usize,
        head_dim: usize,
        state_dim: usize,
    ) -> Vec<f32> {
        let mut y = vec![0.0_f32; n_heads * head_dim];
        for h in 0..n_heads {
            let decay = (a[h] * dt[h]).exp();
            let h_base = h * state_dim * head_dim;
            for d in 0..head_dim {
                let x_d = x[h * head_dim + d];
                let mut y_d = 0.0_f32;
                for n in 0..state_dim {
                    let h_idx = h_base + n * head_dim + d;
                    let h_old = h_state[h_idx];
                    let new_h = decay * h_old + dt[h] * b_vec[n] * x_d;
                    h_state[h_idx] = new_h;
                    y_d += c_vec[n] * new_h;
                }
                y[h * head_dim + d] = y_d;
            }
        }
        y
    }

    /// CPU oracle for conv1d_causal_step.
    fn naive_conv1d_causal_step(
        x: &[f32],
        w: &[f32],
        b: &[f32],
        state: &mut [f32],
        n_channels: usize,
        kernel_size: usize,
    ) -> Vec<f32> {
        let mut y = vec![0.0_f32; n_channels];
        let k_last = kernel_size - 1;
        for d in 0..n_channels {
            let mut acc = b[d] + w[k_last * n_channels + d] * x[d];
            for k in 0..k_last {
                acc += w[k * n_channels + d] * state[k * n_channels + d];
            }
            y[d] = acc;
        }
        for d in 0..n_channels {
            for k in 0..kernel_size.saturating_sub(2) {
                state[k * n_channels + d] = state[(k + 1) * n_channels + d];
            }
            if kernel_size >= 2 {
                state[(kernel_size - 2) * n_channels + d] = x[d];
            }
        }
        y
    }

    #[test_kernel(name = "ffai/ssm/step", dtypes = [f32], tol = 1e-4)]
    fn test_ssm_step(dt: DType) -> TestSetup {
        use super::ssm_step;
        let n_heads = 4usize;
        let head_dim = 16usize;
        let state_dim = 8usize;

        let x: Vec<f32> =
            (0..n_heads * head_dim).map(|i| ((i as f32) * 0.013).sin() * 0.3).collect();
        let a: Vec<f32> = (0..n_heads).map(|i| -0.5 - (i as f32) * 0.1).collect();
        let b_vec: Vec<f32> = (0..state_dim).map(|i| 0.1 + (i as f32) * 0.05).collect();
        let c_vec: Vec<f32> = (0..state_dim).map(|i| 0.2 - (i as f32) * 0.02).collect();
        let dt_in: Vec<f32> = (0..n_heads).map(|i| 0.01 + (i as f32) * 0.003).collect();
        let h_initial: Vec<f32> =
            (0..n_heads * state_dim * head_dim).map(|i| ((i as f32) * 0.011).cos() * 0.1).collect();

        let mut h_oracle = h_initial.clone();
        let expected_y = naive_ssm_step(
            &x,
            &a,
            &b_vec,
            &c_vec,
            &dt_in,
            &mut h_oracle,
            n_heads,
            head_dim,
            state_dim,
        );

        let mut kernel_ir = ssm_step::kernel_ir_for(dt);
        kernel_ir.mode = metaltile_core::ir::KernelMode::Grid3D;
        let total = n_heads * head_dim;

        TestSetup::new(kernel_ir)
            .input(TestBuffer::from_vec("x", pack(&x, dt), dt))
            .input(TestBuffer::from_vec("a", pack(&a, dt), dt))
            .input(TestBuffer::from_vec("b", pack(&b_vec, dt), dt))
            .input(TestBuffer::from_vec("c", pack(&c_vec, dt), dt))
            .input(TestBuffer::from_vec("dt", pack(&dt_in, dt), dt))
            .input(TestBuffer::from_vec("h", pack(&h_initial, DType::F32), DType::F32))
            .input(TestBuffer::from_vec("head_dim", u32_le(head_dim as u32), DType::U32))
            .input(TestBuffer::from_vec("state_dim", u32_le(state_dim as u32), DType::U32))
            .expect(TestBuffer::from_vec("y", pack(&expected_y, dt), dt))
            .expect(TestBuffer::from_vec("h", pack(&h_oracle, DType::F32), DType::F32))
            .grid_3d(1, 1, 1, [total as u32, 1, 1])
    }

    #[test_kernel(name = "ffai/ssm/conv1d_causal_step", dtypes = [f32], tol = 1e-4)]
    fn test_conv1d_causal_step(dt: DType) -> TestSetup {
        use super::conv1d_causal_step;
        let n_channels = 64usize;
        let kernel_size = 4usize;

        let x: Vec<f32> = (0..n_channels).map(|i| ((i as f32) * 0.013).sin()).collect();
        let w: Vec<f32> =
            (0..kernel_size * n_channels).map(|i| 0.1 + ((i as f32) * 0.019).cos() * 0.2).collect();
        let b: Vec<f32> = (0..n_channels).map(|i| (i as f32) * 0.001 - 0.05).collect();
        let state_initial: Vec<f32> =
            (0..(kernel_size - 1) * n_channels).map(|i| ((i as f32) * 0.007).sin() * 0.5).collect();

        let mut state_oracle = state_initial.clone();
        let expected_y =
            naive_conv1d_causal_step(&x, &w, &b, &mut state_oracle, n_channels, kernel_size);

        let mut kernel_ir = conv1d_causal_step::kernel_ir_for(dt);
        kernel_ir.mode = metaltile_core::ir::KernelMode::Grid3D;

        TestSetup::new(kernel_ir)
            .input(TestBuffer::from_vec("x", pack(&x, dt), dt))
            .input(TestBuffer::from_vec("w", pack(&w, dt), dt))
            .input(TestBuffer::from_vec("b", pack(&b, dt), dt))
            .input(TestBuffer::from_vec("state", pack(&state_initial, dt), dt))
            .input(TestBuffer::from_vec("n_channels", u32_le(n_channels as u32), DType::U32))
            .input(TestBuffer::from_vec("kernel_size", u32_le(kernel_size as u32), DType::U32))
            .expect(TestBuffer::from_vec("y", pack(&expected_y, dt), dt))
            .grid_3d(1, 1, 1, [n_channels as u32, 1, 1])
    }

    /// CPU oracle for ssm_step_a2d (Mamba 1 / Jamba 2-D A_log variant).
    fn naive_ssm_step_a2d(
        x: &[f32],
        a_log: &[f32],
        b: &[f32],
        c: &[f32],
        dt: &[f32],
        h: &mut [f32],
        n_heads: usize,
        head_dim: usize,
        state_dim: usize,
    ) -> Vec<f32> {
        let mut y = vec![0.0_f32; n_heads * head_dim];
        for hi in 0..n_heads {
            let dt_val = dt[hi];
            let h_base = hi * state_dim * head_dim;
            for d in 0..head_dim {
                let x_d = x[hi * head_dim + d];
                let channel = hi * head_dim + d;
                let a_log_base = channel * state_dim;
                let mut y_d = 0.0_f32;
                for n in 0..state_dim {
                    let a_val = -a_log[a_log_base + n].exp();
                    let decay = (a_val * dt_val).exp();
                    let h_idx = h_base + n * head_dim + d;
                    let h_old = h[h_idx];
                    let new_h = decay * h_old + dt_val * b[n] * x_d;
                    h[h_idx] = new_h;
                    y_d += c[n] * new_h;
                }
                y[hi * head_dim + d] = y_d;
            }
        }
        y
    }

    #[test_kernel(name = "ffai/ssm/step_a2d", dtypes = [f32], tol = 1e-4)]
    fn test_ssm_step_a2d(dt: DType) -> TestSetup {
        use super::ssm_step_a2d;
        let n_heads = 4usize;
        let head_dim = 32usize;
        let state_dim = 16usize;

        let x: Vec<f32> =
            (0..n_heads * head_dim).map(|i| (((i % 11) as f32 - 5.0) * 0.02)).collect();
        let a_log: Vec<f32> =
            (0..n_heads * head_dim * state_dim).map(|i| -1.0 + 0.013 * (i as f32 % 19.0)).collect();
        let b: Vec<f32> = (0..state_dim).map(|i| ((i % 5) as f32 - 2.0) * 0.05).collect();
        let c: Vec<f32> = (0..state_dim).map(|i| ((i % 7) as f32 - 3.0) * 0.03).collect();
        let dt_in: Vec<f32> = (0..n_heads).map(|i| 0.1 + 0.05 * i as f32).collect();
        let h_initial: Vec<f32> =
            (0..n_heads * state_dim * head_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();

        let mut h_oracle = h_initial.clone();
        let expected_y = naive_ssm_step_a2d(
            &x,
            &a_log,
            &b,
            &c,
            &dt_in,
            &mut h_oracle,
            n_heads,
            head_dim,
            state_dim,
        );

        let mut kernel_ir = ssm_step_a2d::kernel_ir_for(dt);
        kernel_ir.mode = metaltile_core::ir::KernelMode::Grid3D;
        let total = n_heads * head_dim;

        TestSetup::new(kernel_ir)
            .input(TestBuffer::from_vec("x", pack(&x, dt), dt))
            .input(TestBuffer::from_vec("a_log", pack(&a_log, dt), dt))
            .input(TestBuffer::from_vec("b", pack(&b, dt), dt))
            .input(TestBuffer::from_vec("c", pack(&c, dt), dt))
            .input(TestBuffer::from_vec("dt", pack(&dt_in, dt), dt))
            .input(TestBuffer::from_vec("h", pack(&h_initial, DType::F32), DType::F32))
            .input(TestBuffer::from_vec("head_dim", u32_le(head_dim as u32), DType::U32))
            .input(TestBuffer::from_vec("state_dim", u32_le(state_dim as u32), DType::U32))
            .expect(TestBuffer::from_vec("y", pack(&expected_y, dt), dt))
            .expect(TestBuffer::from_vec("h", pack(&h_oracle, DType::F32), DType::F32))
            .grid_3d(1, 1, 1, [total as u32, 1, 1])
    }
}

// Mamba 2 / Mamba 1D depthwise causal-conv step — streaming-decode form.
//
//   y[d] = bias[d]
//        + w[K-1][d] * x[d]
//        + Σ_{k=0..K-2} w[k][d] * state[k][d]
//
// `state` holds the K-1 most recent inputs. After computing y the kernel
// shifts state in-place: state[k][d] = state[k+1][d], state[K-2][d] = x[d].
// Each channel d is owned by exactly one thread, so the read-then-write
// shift is safe within the thread without barriers.
//
// Grid: n_channels threads (one per channel). For Mamba 2 with conv_dim
// ~1500 channels and K=4 this is a tiny dispatch. Activation (Mamba 2
// follows the conv with SiLU) is the caller's concern — kept separate.
#[kernel]
pub fn conv1d_causal_step<T>(
    x: Tensor<T>,
    w: Tensor<T>,
    b: Tensor<T>,
    mut state: Tensor<T>,
    mut y: Tensor<T>,
    #[constexpr] n_channels: u32,
    #[constexpr] kernel_size: u32,
) {
    let d = program_id::<0>();
    let x_d = load(x[d]).cast::<f32>();
    let b_d = load(b[d]).cast::<f32>();
    // Convolution: w[K-1] pairs with current input x[d]; w[0]..w[K-2]
    // pair with state[0]..state[K-2].
    let w_last = load(w[(kernel_size - 1u32) * n_channels + d]).cast::<f32>();
    let mut acc = b_d + w_last * x_d;
    // `kernel_size` is contractually >= 2 (a causal conv with state).
    // Guard the unsigned subtraction anyway: a stray `kernel_size == 0`
    // would make `kernel_size - 1` underflow to ~4e9 — a GPU-pinning
    // loop. `select` clamps the trip count to 0 instead.
    let conv_taps = select(kernel_size > 1u32, kernel_size - 1u32, 0u32);
    for k in range(0u32, conv_taps, 1u32) {
        let s_kd = load(state[k * n_channels + d]).cast::<f32>();
        let w_kd = load(w[k * n_channels + d]).cast::<f32>();
        acc = acc + w_kd * s_kd;
    }
    store(y[d], acc.cast::<T>());
    // Shift state up by one (drop state[0], append x[d] at the tail).
    // Sequential within the thread → safe even though state[k] is read
    // after being written: we read state[k+1] each iteration, never
    // state[k].
    // Same underflow guard: `kernel_size - 2` would wrap to ~4e9 for
    // any `kernel_size < 2`.
    let shift_taps = select(kernel_size > 2u32, kernel_size - 2u32, 0u32);
    for k in range(0u32, shift_taps, 1u32) {
        let next = load(state[(k + 1u32) * n_channels + d]);
        store(state[k * n_channels + d], next);
    }
    store(state[(kernel_size - 2u32) * n_channels + d], load(x[d]));
}

// Mamba 2 selective-scan single-token decode step. One thread per
// (head, d) — no cross-thread sync needed because each (head, d)
// column of h is owned by exactly one thread.
//
// This is the decode form. Chunked prefill uses a parallel-scan
// variant — separate kernel, not in this drop.
#[kernel]
pub fn ssm_step<T>(
    x: Tensor<T>,
    a: Tensor<T>,
    b: Tensor<T>,
    c: Tensor<T>,
    dt: Tensor<T>,
    mut h: Tensor<f32>,
    mut y: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] state_dim: u32,
) {
    let idx = program_id::<0>();
    let h_id = idx / head_dim;
    let d = idx - h_id * head_dim;
    let dt_val = load(dt[h_id]).cast::<f32>();
    let a_val = load(a[h_id]).cast::<f32>();
    let decay = exp(a_val * dt_val);
    let x_d = load(x[h_id * head_dim + d]).cast::<f32>();
    let mut y_d = 0.0f32;
    let h_base = h_id * state_dim * head_dim;
    for n in range(0u32, state_dim, 1u32) {
        let h_idx = h_base + n * head_dim + d;
        let h_old = load(h[h_idx]);
        let b_n = load(b[n]).cast::<f32>();
        let new_h = decay * h_old + dt_val * b_n * x_d;
        store(h[h_idx], new_h);
        let c_n = load(c[n]).cast::<f32>();
        y_d = y_d + c_n * new_h;
    }
    store(y[h_id * head_dim + d], y_d.cast::<T>());
}

// Mamba 1 (Jamba) selective-scan single-token decode step — the
// 2D-`A_log` variant of `ssm_step` above.
//
// The scalar `ssm_step` bakes in a per-channel scalar `A` (`a[h_id]`),
// so the decay `exp(A·dt)` is constant across the state dimension.
// Jamba's Mamba 1 mixer instead carries a *2-D* `A_log` of shape
// `[n_heads*head_dim, state_dim]` — one decay coefficient per
// `(channel, state)` pair — so `decay` varies with `n` inside the
// state loop. Mainline Mamba 2 families (Mamba2, FalconH1, NemotronH,
// GraniteMoeHybrid) use the scalar-`A` kernel and are unaffected;
// this variant exists purely to move Jamba's selective scan onto the
// GPU (it otherwise runs host-side).
//
// `A_log` is the raw log-parameter; the kernel applies the canonical
// Mamba `A = -exp(A_log)` reparam (matching `mt_ssm_step`). Per state
// element `(h, d, n)`:
//
//   A      = -exp(A_log[(h*head_dim + d), n])
//   decay  = exp(A · dt[h])
//   h'     = decay · h_old + dt[h] · B[n] · x[h, d]
//   y[h,d] = Σ_n C[n] · h'[h, d, n]
//
// One thread per `(head, d)` — same Grid3D geometry as `ssm_step`; no
// cross-thread sync because each `(head, d)` column of `h` is owned by
// exactly one thread. The state `h` runs in fp32 (the recurrence
// drifts in bf16 within a few dozen decode steps).
#[kernel]
pub fn ssm_step_a2d<T>(
    x: Tensor<T>,
    a_log: Tensor<T>,
    b: Tensor<T>,
    c: Tensor<T>,
    dt: Tensor<T>,
    mut h: Tensor<f32>,
    mut y: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] state_dim: u32,
) {
    let idx = program_id::<0>();
    let h_id = idx / head_dim;
    let d = idx - h_id * head_dim;
    let dt_val = load(dt[h_id]).cast::<f32>();
    let x_d = load(x[h_id * head_dim + d]).cast::<f32>();
    // `A_log` row for this channel: channel = h_id*head_dim + d, the
    // same flat index `idx` already computed.
    let a_log_base = idx * state_dim;
    let mut y_d = 0.0f32;
    let h_base = h_id * state_dim * head_dim;
    for n in range(0u32, state_dim, 1u32) {
        // Per-(channel, state) decay — the 2-D `A_log` difference.
        let a_val = 0.0f32 - exp(load(a_log[a_log_base + n]).cast::<f32>());
        let decay = exp(a_val * dt_val);
        let h_idx = h_base + n * head_dim + d;
        let h_old = load(h[h_idx]);
        let b_n = load(b[n]).cast::<f32>();
        let new_h = decay * h_old + dt_val * b_n * x_d;
        store(h[h_idx], new_h);
        let c_n = load(c[n]).cast::<f32>();
        y_d = y_d + c_n * new_h;
    }
    store(y[h_id * head_dim + d], y_d.cast::<T>());
}

// Faithful port of MLX's `ssm_step<T, Dh, Ds, H, G>` (alpha branch). One
// threadgroup per `(d_idx, n)` output element, where `n ∈ [0, n_heads*batch)`
// and `d_idx ∈ [0, dh)`. Each threadgroup runs 32 threads (one simd-group)
// and reduces across the state dimension via `simd_sum`.
//
// Required: `ds % 32 == 0` (one thread handles `ds/32` state elements).
//
// `heads_per_group` is MLX's `G`: number of Q heads sharing one (B, C)
// slot. Total distinct (B, C) groups = n_heads / heads_per_group.
#[kernel]
pub fn mt_ssm_step<T>(
    x: Tensor<T>,             // [n_heads*batch, dh]
    a_log: Tensor<T>,         // [n_heads]
    b_mat: Tensor<T>,         // [batch, n_heads/heads_per_group, ds]
    c_mat: Tensor<T>,         // [batch, n_heads/heads_per_group, ds]
    d_skip: Tensor<T>,        // [n_heads]
    dt: Tensor<T>,            // [n_heads*batch]
    state_in: Tensor<T>,      // [n_heads*batch, dh, ds]
    mut state_out: Tensor<T>, // [n_heads*batch, dh, ds]
    mut out: Tensor<T>,       // [n_heads*batch, dh]
    #[constexpr] dh: u32,
    #[constexpr] ds: u32,
    #[constexpr] n_heads: u32,
    #[constexpr] heads_per_group: u32,
) {
    let d_idx = tgid_x;
    let n = tgid_y;
    let ds_idx = tid;
    // h_idx = n % n_heads (which head within the batch).
    // g_idx = n / heads_per_group (which (B, C) group this head reads from).
    let h_idx = n - (n / n_heads) * n_heads;
    let g_idx = n / heads_per_group;
    let dt_val = load(dt[n]).cast::<f32>();
    let a_val = 0.0f32 - exp(load(a_log[h_idx]).cast::<f32>());
    let da = exp(a_val * dt_val);
    let x_val = load(x[n * dh + d_idx]).cast::<f32>();
    let n_per_t = ds / 32u32;
    let bc_base = g_idx * ds;
    let state_base = n * dh * ds + d_idx * ds;
    let mut acc = 0.0f32;
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * ds_idx + i;
        let idx = state_base + s_idx;
        let db_by_x = x_val * dt_val * load(b_mat[bc_base + s_idx]).cast::<f32>();
        let new_state = da * load(state_in[idx]).cast::<f32>() + db_by_x;
        store(state_out[idx], new_state.cast::<T>());
        acc = acc + new_state * load(c_mat[bc_base + s_idx]).cast::<f32>();
    }
    let total = simd_sum(acc);
    if ds_idx == 0u32 {
        let d_val = load(d_skip[h_idx]).cast::<f32>();
        store(out[n * dh + d_idx], (total + x_val * d_val).cast::<T>());
    }
}

#[cfg(target_os = "macos")]
pub mod tests_support_ctx {
    use std::collections::BTreeMap;

    use metaltile_core::{dtype::DType, ir::KernelMode};
    use metaltile_runtime::Context;

    use super::{conv1d_causal_step, mt_ssm_step, ssm_step};

    #[derive(Clone, Copy, Debug)]
    enum Dt {
        F32,
        F16,
        Bf16,
    }

    impl Dt {
        fn to_dtype(self) -> DType {
            match self {
                Dt::F32 => DType::F32,
                Dt::F16 => DType::F16,
                Dt::Bf16 => DType::BF16,
            }
        }

        fn round(self, v: f32) -> f32 {
            match self {
                Dt::F32 => v,
                Dt::F16 => half::f16::from_f32(v).to_f32(),
                Dt::Bf16 => {
                    let bits = v.to_bits();
                    f32::from_bits((bits + 0x8000) & 0xffff0000)
                },
            }
        }
    }

    fn pack_bytes(vals: &[f32], dt: Dt) -> Vec<u8> {
        match dt {
            Dt::F32 => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
            Dt::F16 =>
                vals.iter().flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes()).collect(),
            Dt::Bf16 => vals
                .iter()
                .flat_map(|&v| {
                    let bits = v.to_bits();
                    let bf16 = ((bits + 0x8000) >> 16) as u16;
                    bf16.to_le_bytes()
                })
                .collect(),
        }
    }

    fn unpack_bytes(bytes: &[u8], dt: Dt) -> Vec<f32> {
        match dt {
            Dt::F32 => bytemuck::cast_slice::<u8, f32>(bytes).to_vec(),
            Dt::F16 => bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            Dt::Bf16 => bytes
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
        }
    }

    fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(())).lock().unwrap_or_else(|e| e.into_inner())
    }

    fn naive_conv1d_causal_step(
        x: &[f32],
        w: &[f32],
        b: &[f32],
        state: &mut [f32],
        n_channels: usize,
        kernel_size: usize,
    ) -> Vec<f32> {
        let mut y = vec![0.0_f32; n_channels];
        let k_last = kernel_size - 1;
        for d in 0..n_channels {
            let mut acc = b[d] + w[k_last * n_channels + d] * x[d];
            for k in 0..k_last {
                acc += w[k * n_channels + d] * state[k * n_channels + d];
            }
            y[d] = acc;
        }
        for d in 0..n_channels {
            for k in 0..kernel_size.saturating_sub(2) {
                state[k * n_channels + d] = state[(k + 1) * n_channels + d];
            }
            if kernel_size >= 2 {
                state[(kernel_size - 2) * n_channels + d] = x[d];
            }
        }
        y
    }

    fn run_conv1d_causal_step(
        x: &[f32],
        w: &[f32],
        b: &[f32],
        state: &[f32],
        dt: Dt,
        n_channels: usize,
        kernel_size: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("x".into(), pack_bytes(x, dt));
        buffers.insert("w".into(), pack_bytes(w, dt));
        buffers.insert("b".into(), pack_bytes(b, dt));
        buffers.insert("state".into(), pack_bytes(state, dt));
        buffers.insert("y".into(), pack_bytes(&vec![0.0_f32; n_channels], dt));
        buffers.insert("n_channels".into(), (n_channels as u32).to_le_bytes().to_vec());
        buffers.insert("kernel_size".into(), (kernel_size as u32).to_le_bytes().to_vec());
        let ctx = Context::new().expect("Context::new on macOS");
        let mut kernel = conv1d_causal_step::kernel_ir_for(dt.to_dtype());
        kernel.mode = KernelMode::Grid3D;
        let result = ctx
            .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, 1, 1], [n_channels, 1, 1])
            .expect("conv1d_causal_step dispatch");
        let y = unpack_bytes(result.outputs.get("y").expect("y"), dt);
        let state_out = unpack_bytes(result.outputs.get("state").expect("state"), dt);
        (y, state_out)
    }

    #[allow(clippy::too_many_arguments)]
    fn naive_ssm_step(
        x: &[f32],
        a: &[f32],
        b_vec: &[f32],
        c_vec: &[f32],
        dt: &[f32],
        h_state: &mut [f32],
        n_heads: usize,
        head_dim: usize,
        state_dim: usize,
    ) -> Vec<f32> {
        let mut y = vec![0.0_f32; n_heads * head_dim];
        for h in 0..n_heads {
            let decay = (a[h] * dt[h]).exp();
            let h_base = h * state_dim * head_dim;
            for d in 0..head_dim {
                let x_d = x[h * head_dim + d];
                let mut y_d = 0.0_f32;
                for n in 0..state_dim {
                    let h_idx = h_base + n * head_dim + d;
                    let h_old = h_state[h_idx];
                    let new_h = decay * h_old + dt[h] * b_vec[n] * x_d;
                    h_state[h_idx] = new_h;
                    y_d += c_vec[n] * new_h;
                }
                y[h * head_dim + d] = y_d;
            }
        }
        y
    }

    #[allow(clippy::too_many_arguments)]
    fn run_ssm_step(
        x: &[f32],
        a: &[f32],
        b_vec: &[f32],
        c_vec: &[f32],
        dt_in: &[f32],
        h_state: &[f32],
        dt: Dt,
        n_heads: usize,
        head_dim: usize,
        state_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("x".into(), pack_bytes(x, dt));
        buffers.insert("a".into(), pack_bytes(a, dt));
        buffers.insert("b".into(), pack_bytes(b_vec, dt));
        buffers.insert("c".into(), pack_bytes(c_vec, dt));
        buffers.insert("dt".into(), pack_bytes(dt_in, dt));
        buffers.insert("h".into(), pack_bytes(h_state, Dt::F32));
        buffers.insert("y".into(), pack_bytes(&vec![0.0_f32; n_heads * head_dim], dt));
        buffers.insert("head_dim".into(), (head_dim as u32).to_le_bytes().to_vec());
        buffers.insert("state_dim".into(), (state_dim as u32).to_le_bytes().to_vec());
        let ctx = Context::new().expect("Context::new on macOS");
        let mut kernel = ssm_step::kernel_ir_for(dt.to_dtype());
        kernel.mode = KernelMode::Grid3D;
        let total = n_heads * head_dim;
        let result = ctx
            .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, 1, 1], [total, 1, 1])
            .expect("ssm_step dispatch");
        let y = unpack_bytes(result.outputs.get("y").expect("y"), dt);
        let h_out = unpack_bytes(result.outputs.get("h").expect("h"), Dt::F32);
        (y, h_out)
    }

    #[allow(clippy::too_many_arguments)]
    fn naive_mt_ssm_step(
        x: &[f32],
        a_log: &[f32],
        b_mat: &[f32],
        c_mat: &[f32],
        d_skip: &[f32],
        dt_in: &[f32],
        state_in: &[f32],
        n_total: usize,
        dh: usize,
        ds: usize,
        n_heads: usize,
        heads_per_group: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut state_out = vec![0.0_f32; state_in.len()];
        let mut out = vec![0.0_f32; n_total * dh];
        for n in 0..n_total {
            let h_idx = n % n_heads;
            let g_idx = n / heads_per_group;
            let dt_val = dt_in[n];
            let a_val = -(a_log[h_idx].exp());
            let da = (a_val * dt_val).exp();
            for d_idx in 0..dh {
                let x_val = x[n * dh + d_idx];
                let mut acc = 0.0_f32;
                for s_idx in 0..ds {
                    let idx = n * dh * ds + d_idx * ds + s_idx;
                    let bc_idx = g_idx * ds + s_idx;
                    let db_by_x = x_val * dt_val * b_mat[bc_idx];
                    let new_state = da * state_in[idx] + db_by_x;
                    state_out[idx] = new_state;
                    acc += new_state * c_mat[bc_idx];
                }
                out[n * dh + d_idx] = acc + x_val * d_skip[h_idx];
            }
        }
        (state_out, out)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_mt_ssm_step(
        x: &[f32],
        a_log: &[f32],
        b_mat: &[f32],
        c_mat: &[f32],
        d_skip: &[f32],
        dt_in: &[f32],
        state_in: &[f32],
        dt: Dt,
        n_total: usize,
        dh: usize,
        ds: usize,
        n_heads: usize,
        heads_per_group: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let groups = n_total / heads_per_group;
        let bc_len = groups * ds;
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        buffers.insert("x".into(), pack_bytes(x, dt));
        buffers.insert("a_log".into(), pack_bytes(a_log, dt));
        assert_eq!(b_mat.len(), bc_len);
        assert_eq!(c_mat.len(), bc_len);
        buffers.insert("b_mat".into(), pack_bytes(b_mat, dt));
        buffers.insert("c_mat".into(), pack_bytes(c_mat, dt));
        buffers.insert("d_skip".into(), pack_bytes(d_skip, dt));
        buffers.insert("dt".into(), pack_bytes(dt_in, dt));
        buffers.insert("state_in".into(), pack_bytes(state_in, dt));
        buffers.insert("state_out".into(), pack_bytes(&vec![0.0_f32; state_in.len()], dt));
        buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; n_total * dh], dt));
        buffers.insert("dh".into(), (dh as u32).to_le_bytes().to_vec());
        buffers.insert("ds".into(), (ds as u32).to_le_bytes().to_vec());
        buffers.insert("n_heads".into(), (n_heads as u32).to_le_bytes().to_vec());
        buffers.insert("heads_per_group".into(), (heads_per_group as u32).to_le_bytes().to_vec());
        let ctx = Context::new().expect("Context::new on macOS");
        let mut kernel = mt_ssm_step::kernel_ir_for(dt.to_dtype());
        kernel.mode = KernelMode::Reduction;
        assert!(ds.is_multiple_of(32), "mt_ssm_step requires ds % 32 == 0");
        let result = ctx
            .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [dh, n_total, 1], [32, 1, 1])
            .expect("mt_ssm_step dispatch");
        let state_out = unpack_bytes(result.outputs.get("state_out").expect("state_out"), dt);
        let out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
        (state_out, out)
    }

    #[test]
    fn conv1d_causal_step_matches_oracle_f32() {
        let _g = gpu_lock();
        let n_channels = 128usize;
        let kernel_size = 4usize;
        let x: Vec<f32> = (0..n_channels).map(|i| ((i as f32) * 0.013).sin()).collect();
        let w: Vec<f32> =
            (0..kernel_size * n_channels).map(|i| 0.1 + ((i as f32) * 0.019).cos() * 0.2).collect();
        let b: Vec<f32> = (0..n_channels).map(|i| (i as f32) * 0.001 - 0.05).collect();
        let mut state_oracle: Vec<f32> =
            (0..(kernel_size - 1) * n_channels).map(|i| ((i as f32) * 0.007).sin() * 0.5).collect();
        let state_initial = state_oracle.clone();
        let y_expected =
            naive_conv1d_causal_step(&x, &w, &b, &mut state_oracle, n_channels, kernel_size);
        let (y_actual, state_actual) =
            run_conv1d_causal_step(&x, &w, &b, &state_initial, Dt::F32, n_channels, kernel_size);
        let mut max_diff = 0.0_f32;
        for (a, e) in y_actual.iter().zip(y_expected.iter()) {
            max_diff = max_diff.max((a - e).abs());
        }
        assert!(max_diff < 1e-5, "conv1d y max |diff| = {max_diff:.2e}");
        let mut max_state_diff = 0.0_f32;
        for (a, e) in state_actual.iter().zip(state_oracle.iter()) {
            max_state_diff = max_state_diff.max((a - e).abs());
        }
        assert!(max_state_diff < 1e-6, "conv1d state shift max |diff| = {max_state_diff:.2e}");
    }

    #[test]
    fn conv1d_causal_step_matches_oracle_f16() {
        let _g = gpu_lock();
        let n_channels = 64usize;
        let kernel_size = 4usize;
        let x_f32: Vec<f32> = (0..n_channels).map(|i| ((i as f32) * 0.013).sin()).collect();
        let w_f32: Vec<f32> =
            (0..kernel_size * n_channels).map(|i| 0.1 + ((i as f32) * 0.019).cos() * 0.2).collect();
        let b_f32: Vec<f32> = (0..n_channels).map(|i| (i as f32) * 0.001 - 0.05).collect();
        let state_f32: Vec<f32> =
            (0..(kernel_size - 1) * n_channels).map(|i| ((i as f32) * 0.007).sin() * 0.5).collect();
        let round = |v: &[f32]| v.iter().map(|&x| Dt::F16.round(x)).collect::<Vec<f32>>();
        let x = round(&x_f32);
        let w = round(&w_f32);
        let b = round(&b_f32);
        let mut state_oracle = round(&state_f32);
        let y_expected =
            naive_conv1d_causal_step(&x, &w, &b, &mut state_oracle, n_channels, kernel_size);
        let (y_actual, _state_actual) = run_conv1d_causal_step(
            &x,
            &w,
            &b,
            &round(&state_f32),
            Dt::F16,
            n_channels,
            kernel_size,
        );
        let mut max_rel = 0.0_f32;
        for (a, e) in y_actual.iter().zip(y_expected.iter()) {
            let rel = (a - e).abs() / e.abs().max(1e-3);
            max_rel = max_rel.max(rel);
        }
        assert!(max_rel < 5e-3, "conv1d f16 max rel = {max_rel:.2e}");
    }

    #[test]
    fn conv1d_causal_step_matches_oracle_bf16() {
        let _g = gpu_lock();
        let n_channels = 64usize;
        let kernel_size = 4usize;
        let round = |v: &[f32]| v.iter().map(|&x| Dt::Bf16.round(x)).collect::<Vec<f32>>();
        let x_f32: Vec<f32> = (0..n_channels).map(|i| ((i as f32) * 0.013).sin()).collect();
        let w_f32: Vec<f32> =
            (0..kernel_size * n_channels).map(|i| 0.1 + ((i as f32) * 0.019).cos() * 0.2).collect();
        let b_f32: Vec<f32> = (0..n_channels).map(|i| (i as f32) * 0.001 - 0.05).collect();
        let state_f32: Vec<f32> =
            (0..(kernel_size - 1) * n_channels).map(|i| ((i as f32) * 0.007).sin() * 0.5).collect();
        let x = round(&x_f32);
        let w = round(&w_f32);
        let b = round(&b_f32);
        let mut state_oracle = round(&state_f32);
        let y_expected =
            naive_conv1d_causal_step(&x, &w, &b, &mut state_oracle, n_channels, kernel_size);
        let (y_actual, _state_actual) = run_conv1d_causal_step(
            &x,
            &w,
            &b,
            &round(&state_f32),
            Dt::Bf16,
            n_channels,
            kernel_size,
        );
        let mut max_rel = 0.0_f32;
        for (a, e) in y_actual.iter().zip(y_expected.iter()) {
            let rel = (a - e).abs() / e.abs().max(1e-3);
            max_rel = max_rel.max(rel);
        }
        assert!(max_rel < 5e-2, "conv1d bf16 max rel = {max_rel:.2e}");
    }

    #[test]
    fn ssm_step_matches_oracle_f32() {
        let _g = gpu_lock();
        let n_heads = 4usize;
        let head_dim = 16usize;
        let state_dim = 8usize;
        let x: Vec<f32> =
            (0..n_heads * head_dim).map(|i| ((i as f32) * 0.013).sin() * 0.3).collect();
        let a: Vec<f32> = (0..n_heads).map(|i| -0.5 - (i as f32) * 0.1).collect();
        let b_vec: Vec<f32> = (0..state_dim).map(|i| 0.1 + (i as f32) * 0.05).collect();
        let c_vec: Vec<f32> = (0..state_dim).map(|i| 0.2 - (i as f32) * 0.02).collect();
        let dt_in: Vec<f32> = (0..n_heads).map(|i| 0.01 + (i as f32) * 0.003).collect();
        let mut h_state_oracle: Vec<f32> =
            (0..n_heads * state_dim * head_dim).map(|i| ((i as f32) * 0.011).cos() * 0.1).collect();
        let h_state_initial = h_state_oracle.clone();
        let y_expected = naive_ssm_step(
            &x,
            &a,
            &b_vec,
            &c_vec,
            &dt_in,
            &mut h_state_oracle,
            n_heads,
            head_dim,
            state_dim,
        );
        let (y_actual, h_actual) = run_ssm_step(
            &x,
            &a,
            &b_vec,
            &c_vec,
            &dt_in,
            &h_state_initial,
            Dt::F32,
            n_heads,
            head_dim,
            state_dim,
        );
        let mut max_y_diff = 0.0_f32;
        for (a, e) in y_actual.iter().zip(y_expected.iter()) {
            max_y_diff = max_y_diff.max((a - e).abs());
        }
        assert!(max_y_diff < 1e-5, "ssm_step y max |diff| = {max_y_diff:.2e}");
        let mut max_h_diff = 0.0_f32;
        for (a, e) in h_actual.iter().zip(h_state_oracle.iter()) {
            max_h_diff = max_h_diff.max((a - e).abs());
        }
        assert!(max_h_diff < 1e-5, "ssm_step h update max |diff| = {max_h_diff:.2e}");
    }

    #[test]
    fn ssm_step_state_decays_when_x_is_zero_f32() {
        let _g = gpu_lock();
        let n_heads = 2usize;
        let head_dim = 8usize;
        let state_dim = 4usize;
        let x = vec![0.0_f32; n_heads * head_dim];
        let a: Vec<f32> = vec![-1.0, -2.0];
        let b_vec: Vec<f32> = vec![0.5, 0.6, 0.7, 0.8];
        let c_vec: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
        let dt_in: Vec<f32> = vec![0.05, 0.05];
        let h_state_initial: Vec<f32> =
            (0..n_heads * state_dim * head_dim).map(|i| 0.5 + (i as f32) * 0.01).collect();
        let mut h_state_oracle = h_state_initial.clone();
        let y_expected = naive_ssm_step(
            &x,
            &a,
            &b_vec,
            &c_vec,
            &dt_in,
            &mut h_state_oracle,
            n_heads,
            head_dim,
            state_dim,
        );
        let (y_actual, _h_actual) = run_ssm_step(
            &x,
            &a,
            &b_vec,
            &c_vec,
            &dt_in,
            &h_state_initial,
            Dt::F32,
            n_heads,
            head_dim,
            state_dim,
        );
        let mut max_diff = 0.0_f32;
        for (a, e) in y_actual.iter().zip(y_expected.iter()) {
            max_diff = max_diff.max((a - e).abs());
        }
        assert!(max_diff < 1e-5, "x=0 decay invariant max |diff| = {max_diff:.2e}");
    }

    #[test]
    fn mt_ssm_step_matches_oracle_f32() {
        let _g = gpu_lock();
        let n_heads = 4usize;
        let heads_per_group = 2usize;
        let batch = 2usize;
        let n_total = n_heads * batch;
        let dh = 8usize;
        let ds = 32usize;
        let groups = n_total / heads_per_group;
        let x: Vec<f32> = (0..n_total * dh).map(|i| ((i as f32) * 0.017).sin() * 0.3).collect();
        let a_log: Vec<f32> = (0..n_heads).map(|i| -1.0 + (i as f32) * 0.2).collect();
        let b_mat: Vec<f32> = (0..groups * ds).map(|i| 0.05 + (i as f32) * 0.003).collect();
        let c_mat: Vec<f32> = (0..groups * ds).map(|i| 0.1 - (i as f32) * 0.001).collect();
        let d_skip: Vec<f32> = (0..n_heads).map(|i| 0.05 + (i as f32) * 0.01).collect();
        let dt_in: Vec<f32> = (0..n_total).map(|i| 0.02 + (i as f32) * 0.005).collect();
        let state_in: Vec<f32> =
            (0..n_total * dh * ds).map(|i| ((i as f32) * 0.009).cos() * 0.2).collect();
        let (state_expected, out_expected) = naive_mt_ssm_step(
            &x,
            &a_log,
            &b_mat,
            &c_mat,
            &d_skip,
            &dt_in,
            &state_in,
            n_total,
            dh,
            ds,
            n_heads,
            heads_per_group,
        );
        let (state_actual, out_actual) = run_mt_ssm_step(
            &x,
            &a_log,
            &b_mat,
            &c_mat,
            &d_skip,
            &dt_in,
            &state_in,
            Dt::F32,
            n_total,
            dh,
            ds,
            n_heads,
            heads_per_group,
        );
        let mut max_state_diff = 0.0_f32;
        for (a, e) in state_actual.iter().zip(state_expected.iter()) {
            max_state_diff = max_state_diff.max((a - e).abs());
        }
        assert!(max_state_diff < 1e-5, "mt_ssm_step state max |diff| = {max_state_diff:.2e}");
        let mut max_out_diff = 0.0_f32;
        for (a, e) in out_actual.iter().zip(out_expected.iter()) {
            max_out_diff = max_out_diff.max((a - e).abs());
        }
        assert!(max_out_diff < 5e-5, "mt_ssm_step out max |diff| = {max_out_diff:.2e}");
    }

    #[test]
    fn mt_ssm_step_matches_oracle_ds_128_f32() {
        let _g = gpu_lock();
        let n_heads = 4usize;
        let heads_per_group = 2usize;
        let batch = 1usize;
        let n_total = n_heads * batch;
        let dh = 4usize;
        let ds = 128usize;
        let groups = n_total / heads_per_group;
        let x: Vec<f32> = (0..n_total * dh).map(|i| ((i as f32) * 0.017).sin() * 0.3).collect();
        let a_log: Vec<f32> = (0..n_heads).map(|i| -1.0 + (i as f32) * 0.2).collect();
        let b_mat: Vec<f32> = (0..groups * ds).map(|i| 0.05 + (i as f32) * 0.003).collect();
        let c_mat: Vec<f32> = (0..groups * ds).map(|i| 0.1 - (i as f32) * 0.001).collect();
        let d_skip: Vec<f32> = (0..n_heads).map(|i| 0.05 + (i as f32) * 0.01).collect();
        let dt_in: Vec<f32> = (0..n_total).map(|i| 0.02 + (i as f32) * 0.005).collect();
        let state_in: Vec<f32> =
            (0..n_total * dh * ds).map(|i| ((i as f32) * 0.009).cos() * 0.2).collect();
        let (state_expected, out_expected) = naive_mt_ssm_step(
            &x,
            &a_log,
            &b_mat,
            &c_mat,
            &d_skip,
            &dt_in,
            &state_in,
            n_total,
            dh,
            ds,
            n_heads,
            heads_per_group,
        );
        let (state_actual, out_actual) = run_mt_ssm_step(
            &x,
            &a_log,
            &b_mat,
            &c_mat,
            &d_skip,
            &dt_in,
            &state_in,
            Dt::F32,
            n_total,
            dh,
            ds,
            n_heads,
            heads_per_group,
        );
        let mut max_state_diff = 0.0_f32;
        for (a, e) in state_actual.iter().zip(state_expected.iter()) {
            max_state_diff = max_state_diff.max((a - e).abs());
        }
        assert!(max_state_diff < 1e-5, "ds=128 state max |diff| = {max_state_diff:.2e}");
        let mut max_out_diff = 0.0_f32;
        for (a, e) in out_actual.iter().zip(out_expected.iter()) {
            max_out_diff = max_out_diff.max((a - e).abs());
        }
        assert!(max_out_diff < 1e-4, "ds=128 out max |diff| = {max_out_diff:.2e}");
    }

    #[test]
    fn mt_ssm_step_matches_oracle_bf16() {
        let _g = gpu_lock();
        let n_heads = 4usize;
        let heads_per_group = 2usize;
        let batch = 1usize;
        let n_total = n_heads * batch;
        let dh = 4usize;
        let ds = 32usize;
        let groups = n_total / heads_per_group;
        let round = |v: &[f32]| v.iter().map(|&x| Dt::Bf16.round(x)).collect::<Vec<f32>>();
        let x =
            round(&(0..n_total * dh).map(|i| ((i as f32) * 0.017).sin() * 0.3).collect::<Vec<_>>());
        let a_log = round(&(0..n_heads).map(|i| -1.0 + (i as f32) * 0.2).collect::<Vec<_>>());
        let b_mat = round(&(0..groups * ds).map(|i| 0.05 + (i as f32) * 0.003).collect::<Vec<_>>());
        let c_mat = round(&(0..groups * ds).map(|i| 0.1 - (i as f32) * 0.001).collect::<Vec<_>>());
        let d_skip = round(&(0..n_heads).map(|i| 0.05 + (i as f32) * 0.01).collect::<Vec<_>>());
        let dt_in = round(&(0..n_total).map(|i| 0.02 + (i as f32) * 0.005).collect::<Vec<_>>());
        let state_in = round(
            &(0..n_total * dh * ds).map(|i| ((i as f32) * 0.009).cos() * 0.2).collect::<Vec<_>>(),
        );
        let (_state_expected, out_expected) = naive_mt_ssm_step(
            &x,
            &a_log,
            &b_mat,
            &c_mat,
            &d_skip,
            &dt_in,
            &state_in,
            n_total,
            dh,
            ds,
            n_heads,
            heads_per_group,
        );
        let (_state_actual, out_actual) = run_mt_ssm_step(
            &x,
            &a_log,
            &b_mat,
            &c_mat,
            &d_skip,
            &dt_in,
            &state_in,
            Dt::Bf16,
            n_total,
            dh,
            ds,
            n_heads,
            heads_per_group,
        );
        let mut max_rel = 0.0_f32;
        for (a, e) in out_actual.iter().zip(out_expected.iter()) {
            let rel = (a - e).abs() / e.abs().max(1e-3);
            max_rel = max_rel.max(rel);
        }
        assert!(max_rel < 1e-1, "mt_ssm_step bf16 max rel = {max_rel:.2e}");
    }
}
