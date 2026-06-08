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

// ── Mamba 2 batched-prefill causal depthwise conv1d ─────────────────────
//
// Processes ALL S prompt tokens in one dispatch, with zero initial state
// (prefill starts from scratch). Each thread computes one output element
// y[ti, ch] = silu( bias[ch]
//                 + sum_{k=0..kc-1} w[k, ch] * xbc_in[ti - (kc-1-k), ch] )
// where out-of-bounds reads (ti < kc-1-k) are treated as 0 (zero initial
// state). Silu is applied inline — saves a second kernel dispatch.
//
// Grid: [s * conv_dim, 1, 1]; one thread per (token, channel).
// Replaces the host ring-conv loop in bench_nemotron's forward_batched.
// Gate: NEMOTRON_CONV_DEVICE=1 in bench_nemotron.
#[kernel]
pub fn conv1d_causal_prefill(
    xbc_in: Tensor<f32>,     // [s * conv_dim] flat row-major
    w: Tensor<f32>,          // [kc * conv_dim] reorganized same as decode step
    bias: Tensor<f32>,       // [conv_dim]
    mut y: Tensor<f32>,      // [s * conv_dim] output with silu applied
    #[constexpr] conv_dim: u32,
    #[constexpr] kc: u32,
) {
    let idx = program_id::<0>();
    let ti = idx / conv_dim;
    let ch = idx - ti * conv_dim;
    let b_ch = load(bias[ch]);
    // Accumulate: w[k, ch] pairs with xbc_in[ti - (kc-1-k), ch].
    // k=0 → lag (kc-1); k=kc-1 → current token (lag 0).
    let mut acc = b_ch;
    for k in range(0u32, kc, 1u32) {
        let lag = kc - 1u32 - k;
        // Only include this tap if it's within the valid prefix.
        if ti >= lag {
            let src_ti = ti - lag;
            let v = load(xbc_in[src_ti * conv_dim + ch]);
            let wk = load(w[k * conv_dim + ch]);
            acc = acc + wk * v;
        }
    }
    // Silu activation: y = acc / (1 + exp(-acc)).
    let sig = 1.0f32 / (1.0f32 + exp(0.0f32 - acc));
    store(y[idx], acc * sig);
}

// ── Strided column extraction ─────────────────────────────────────────────
//
// Extracts a contiguous subrange of columns from a row-major matrix:
//   dst[ti * width + ci] = src[ti * stride + col_off + ci]
//
// Used to carve z, xbc, and dt_raw out of the [s, in_proj_out] projection
// matrix without downloading it to host.
//
// Grid: [s * width, 1, 1]; one thread per output element.
#[kernel]
pub fn strided_col_copy(
    src: Tensor<f32>,    // [s * stride] flat row-major
    mut dst: Tensor<f32>, // [s * width] output
    #[constexpr] stride: u32,
    #[constexpr] col_off: u32,
    #[constexpr] width: u32,
) {
    let idx = program_id::<0>();
    let ti = idx / width;
    let ci = idx - ti * width;
    let v = load(src[ti * stride + col_off + ci]);
    store(dst[idx], v);
}

// ── Batched softplus + bias addition ─────────────────────────────────────
//
// Computes: dst[ti * n + hi] = softplus(src[ti * n + hi] + bias[hi])
// where softplus(x) = log(1 + exp(x)) ≈ x for x > 20.
//
// Used to convert the [s, m_nh] dt_raw tensor + dt_bias into dt_all on
// device, replacing the CPU softplus loop in forward_batched.
//
// Grid: [s * n, 1, 1]; one thread per output element.
#[kernel]
pub fn softplus_add_rows(
    src: Tensor<f32>,     // [s * n]
    bias: Tensor<f32>,    // [n]
    mut dst: Tensor<f32>, // [s * n]
    #[constexpr] n: u32,
) {
    let idx = program_id::<0>();
    let hi = idx - (idx / n) * n;
    let raw = load(src[idx]) + load(bias[hi]);
    // softplus: log(1 + exp(x)); numerically stable branch for large x.
    let sp = select(raw > 20.0f32, raw, log(1.0f32 + exp(raw)));
    store(dst[idx], sp);
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

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::{conv1d_causal_step, mt_ssm_step, ssm_step, ssm_step_a2d};
    use crate::utils::pack_f32;

    // ── SSD portable-scan kernels: cross-backend codegen smoke ────────────
    // The Mamba2 SSD chunked-matmul prefill scan (ffai-ops
    // `ssm_prefill_scan_ssd_portable`) is built from these `ssd_*` #[kernel]
    // ops + `ffai_gemm_batched`. The whole point is PORTABILITY — they must
    // codegen cleanly to MSL (Metal), CUDA (Nvidia), HIP (AMD/RDNA4) and
    // SPIR-V/GLSL (Vulkan), NOT raw-CUDA. This asserts every backend emits a
    // kernel definition under the declared name (catches a DSL construct that
    // only lowers on one target).
    #[test]
    fn ssd_portable_kernels_codegen_all_backends() {
        use metaltile_codegen::backend::CodegenBackend;
        use metaltile_codegen::msl::{MslConfig, MslGenerator};
        use metaltile_codegen::{CudaGenerator, GlslGenerator, HipGenerator};
        use metaltile_core::DType;
        use metaltile_core::ir::KernelMode;

        let kernels: Vec<(&str, metaltile_core::Kernel)> = vec![
            ("ffai_gemm_batched", {
                let mut k = super::super::gemm::ffai_gemm_batched::kernel_ir_for(DType::F32);
                k.mode = KernelMode::Reduction;
                k
            }),
            ("ssd_lcs", { let mut k = super::ssd_lcs::kernel_ir_for(); k.mode = KernelMode::Grid3D; k }),
            ("ssd_gather_bc", { let mut k = super::ssd_gather_bc::kernel_ir_for(); k.mode = KernelMode::Grid3D; k }),
            ("ssd_xt", { let mut k = super::ssd_xt::kernel_ir_for(); k.mode = KernelMode::Grid3D; k }),
            ("ssd_mmask", { let mut k = super::ssd_mmask::kernel_ir_for(); k.mode = KernelMode::Grid3D; k }),
            ("ssd_bdt", { let mut k = super::ssd_bdt::kernel_ir_for(); k.mode = KernelMode::Grid3D; k }),
            ("ssd_recur", { let mut k = super::ssd_recur::kernel_ir_for(); k.mode = KernelMode::Grid3D; k }),
            ("ssd_combine", { let mut k = super::ssd_combine::kernel_ir_for(); k.mode = KernelMode::Grid3D; k }),
            ("ssd_g1_cb", { let mut k = super::ssd_g1_cb::kernel_ir_for(); k.mode = KernelMode::Reduction; k }),
            ("ssd_g4_cs", { let mut k = super::ssd_g4_cs::kernel_ir_for(); k.mode = KernelMode::Reduction; k }),
        ];
        let msl = MslGenerator::new(MslConfig::default());
        let cuda = CudaGenerator::new();
        let hip = HipGenerator::new();
        let glsl = GlslGenerator::new();
        for (name, k) in &kernels {
            for (backend, src) in [
                ("MSL", msl.generate(k)),
                ("CUDA", cuda.generate(k)),
                ("HIP", hip.generate(k)),
                ("SPIRV/GLSL", glsl.generate(k)),
            ] {
                let s = src.unwrap_or_else(|e| panic!("{name}: {backend} codegen failed: {e:?}"));
                assert!(!s.is_empty(), "{name}: {backend} emitted empty source");
            }
        }
        eprintln!("✅ all SSD portable-scan kernels codegen on MSL/CUDA/HIP/SPIRV (portable to Apple/Nvidia/AMD/Vulkan)");
    }

    // ── conv1d_causal_step ──────────────────────────────────────────────

    /// CPU oracle: `y[d] = b[d] + w[K-1][d]·x[d] + Σ_{k<K-1} w[k][d]·state[k][d]`,
    /// then shift state up and append `x`. Returns `(y, shifted_state)`.
    fn conv1d_oracle(
        x: &[f32],
        w: &[f32],
        b: &[f32],
        state_in: &[f32],
        n_channels: usize,
        kernel_size: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut y = vec![0.0_f32; n_channels];
        let mut state = state_in.to_vec();
        let k_last = kernel_size - 1;
        for d in 0..n_channels {
            let mut acc = b[d] + w[k_last * n_channels + d] * x[d];
            for k in 0..k_last {
                acc += w[k * n_channels + d] * state_in[k * n_channels + d];
            }
            y[d] = acc;
        }
        for d in 0..n_channels {
            for k in 0..kernel_size.saturating_sub(2) {
                state[k * n_channels + d] = state_in[(k + 1) * n_channels + d];
            }
            if kernel_size >= 2 {
                state[(kernel_size - 2) * n_channels + d] = x[d];
            }
        }
        (y, state)
    }

    fn conv1d_setup(n_channels: usize, kernel_size: usize, dt: DType) -> TestSetup {
        let x: Vec<f32> = (0..n_channels).map(|i| ((i as f32) * 0.013).sin()).collect();
        let w: Vec<f32> =
            (0..kernel_size * n_channels).map(|i| 0.1 + ((i as f32) * 0.019).cos() * 0.2).collect();
        let b: Vec<f32> = (0..n_channels).map(|i| (i as f32) * 0.001 - 0.05).collect();
        let state_in: Vec<f32> =
            (0..(kernel_size - 1) * n_channels).map(|i| ((i as f32) * 0.007).sin() * 0.5).collect();

        let (y_exp, state_exp) = conv1d_oracle(&x, &w, &b, &state_in, n_channels, kernel_size);

        TestSetup::new(conv1d_causal_step::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("x", pack_f32(&x, dt), dt))
            .input(TestBuffer::from_vec("w", pack_f32(&w, dt), dt))
            .input(TestBuffer::from_vec("b", pack_f32(&b, dt), dt))
            .input(TestBuffer::from_vec("state", pack_f32(&state_in, dt), dt))
            .input(TestBuffer::zeros("y", n_channels, dt))
            .constexpr("n_channels", n_channels as u32)
            .constexpr("kernel_size", kernel_size as u32)
            .expect(TestBuffer::from_vec("y", pack_f32(&y_exp, dt), dt))
            .expect(TestBuffer::from_vec("state", pack_f32(&state_exp, dt), dt))
            .grid_3d(n_channels as u32, 1, 1, [1, 1, 1])
    }

    // Mamba 2 short-conv: kernel_size=4. One thread per channel.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-5, 5e-3, 5e-2])]
    fn test_conv1d_causal_step(dt: DType) -> TestSetup { conv1d_setup(128, 4, dt) }

    // ── ssm_step ────────────────────────────────────────────────────────

    /// CPU oracle for the scalar-A selective-scan decode step. `h` is f32;
    /// returns `(y, h_new)`.
    #[allow(clippy::too_many_arguments)]
    fn ssm_step_oracle(
        x: &[f32],
        a: &[f32],
        b_vec: &[f32],
        c_vec: &[f32],
        dt_in: &[f32],
        h_state: &[f32],
        n_heads: usize,
        head_dim: usize,
        state_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut y = vec![0.0_f32; n_heads * head_dim];
        let mut h = h_state.to_vec();
        for hh in 0..n_heads {
            let decay = (a[hh] * dt_in[hh]).exp();
            let h_base = hh * state_dim * head_dim;
            for d in 0..head_dim {
                let x_d = x[hh * head_dim + d];
                let mut y_d = 0.0_f32;
                for n in 0..state_dim {
                    let h_idx = h_base + n * head_dim + d;
                    let new_h = decay * h_state[h_idx] + dt_in[hh] * b_vec[n] * x_d;
                    h[h_idx] = new_h;
                    y_d += c_vec[n] * new_h;
                }
                y[hh * head_dim + d] = y_d;
            }
        }
        (y, h)
    }

    fn ssm_step_setup(n_heads: usize, head_dim: usize, state_dim: usize, dt: DType) -> TestSetup {
        let x: Vec<f32> =
            (0..n_heads * head_dim).map(|i| ((i as f32) * 0.013).sin() * 0.3).collect();
        let a: Vec<f32> = (0..n_heads).map(|i| -0.5 - (i as f32) * 0.1).collect();
        let b_vec: Vec<f32> = (0..state_dim).map(|i| 0.1 + (i as f32) * 0.05).collect();
        let c_vec: Vec<f32> = (0..state_dim).map(|i| 0.2 - (i as f32) * 0.02).collect();
        let dt_in: Vec<f32> = (0..n_heads).map(|i| 0.01 + (i as f32) * 0.003).collect();
        let h_state: Vec<f32> =
            (0..n_heads * state_dim * head_dim).map(|i| ((i as f32) * 0.011).cos() * 0.1).collect();

        let (y_exp, h_exp) =
            ssm_step_oracle(&x, &a, &b_vec, &c_vec, &dt_in, &h_state, n_heads, head_dim, state_dim);

        // `h` is always f32 in the kernel signature; `y` carries the tested dt.
        TestSetup::new(ssm_step::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("x", pack_f32(&x, dt), dt))
            .input(TestBuffer::from_vec("a", pack_f32(&a, dt), dt))
            .input(TestBuffer::from_vec("b", pack_f32(&b_vec, dt), dt))
            .input(TestBuffer::from_vec("c", pack_f32(&c_vec, dt), dt))
            .input(TestBuffer::from_vec("dt", pack_f32(&dt_in, dt), dt))
            .input(TestBuffer::from_vec("h", pack_f32(&h_state, DType::F32), DType::F32))
            .input(TestBuffer::zeros("y", n_heads * head_dim, dt))
            .constexpr("head_dim", head_dim as u32)
            .constexpr("state_dim", state_dim as u32)
            .expect(TestBuffer::from_vec("y", pack_f32(&y_exp, dt), dt))
            .expect(TestBuffer::from_vec("h", pack_f32(&h_exp, DType::F32), DType::F32))
            .grid_3d((n_heads * head_dim) as u32, 1, 1, [1, 1, 1])
    }

    // One thread per (head, d). `y` tolerance loosens for f16/bf16; `h` is
    // f32 so it must track tightly across all dtype runs.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-5, 5e-3, 5e-2])]
    fn test_ssm_step(dt: DType) -> TestSetup { ssm_step_setup(4, 16, 8, dt) }

    // ── ssm_step_a2d (Mamba 1 / Jamba: 2-D per-(channel,state) A_log) ────

    /// CPU oracle for the 2-D-A_log selective-scan step. Per state element
    /// `(h, d, n)`: `A = -exp(A_log[(h*head_dim+d), n])`, `decay = exp(A·dt[h])`,
    /// `h' = decay·h_old + dt[h]·B[n]·x[h,d]`, `y[h,d] = Σ_n C[n]·h'`.
    /// `h` is f32; returns `(y, h_new)`.
    #[allow(clippy::too_many_arguments)]
    fn ssm_step_a2d_oracle(
        x: &[f32],
        a_log: &[f32],
        b_vec: &[f32],
        c_vec: &[f32],
        dt_in: &[f32],
        h_state: &[f32],
        n_heads: usize,
        head_dim: usize,
        state_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut y = vec![0.0_f32; n_heads * head_dim];
        let mut h = h_state.to_vec();
        for hh in 0..n_heads {
            let h_base = hh * state_dim * head_dim;
            for d in 0..head_dim {
                let chan = hh * head_dim + d;
                let row_base = chan * state_dim;
                let x_d = x[hh * head_dim + d];
                let mut y_d = 0.0_f32;
                for n in 0..state_dim {
                    let a_val = -(a_log[row_base + n].exp());
                    let decay = (a_val * dt_in[hh]).exp();
                    let h_idx = h_base + n * head_dim + d;
                    let new_h = decay * h_state[h_idx] + dt_in[hh] * b_vec[n] * x_d;
                    h[h_idx] = new_h;
                    y_d += c_vec[n] * new_h;
                }
                y[hh * head_dim + d] = y_d;
            }
        }
        (y, h)
    }

    fn ssm_step_a2d_setup(
        n_heads: usize,
        head_dim: usize,
        state_dim: usize,
        dt: DType,
    ) -> TestSetup {
        let x: Vec<f32> =
            (0..n_heads * head_dim).map(|i| ((i as f32) * 0.013).sin() * 0.3).collect();
        // A_log is the raw log-parameter, one per (channel, state) pair; keep
        // it small/positive so `-exp(A_log)` stays in a stable decay range.
        let a_log: Vec<f32> = (0..n_heads * head_dim * state_dim)
            .map(|i| -1.0 + ((i as f32) * 0.017).sin() * 0.5)
            .collect();
        let b_vec: Vec<f32> = (0..state_dim).map(|i| 0.1 + (i as f32) * 0.05).collect();
        let c_vec: Vec<f32> = (0..state_dim).map(|i| 0.2 - (i as f32) * 0.02).collect();
        let dt_in: Vec<f32> = (0..n_heads).map(|i| 0.01 + (i as f32) * 0.003).collect();
        let h_state: Vec<f32> =
            (0..n_heads * state_dim * head_dim).map(|i| ((i as f32) * 0.011).cos() * 0.1).collect();

        let (y_exp, h_exp) = ssm_step_a2d_oracle(
            &x, &a_log, &b_vec, &c_vec, &dt_in, &h_state, n_heads, head_dim, state_dim,
        );

        TestSetup::new(ssm_step_a2d::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("x", pack_f32(&x, dt), dt))
            .input(TestBuffer::from_vec("a_log", pack_f32(&a_log, dt), dt))
            .input(TestBuffer::from_vec("b", pack_f32(&b_vec, dt), dt))
            .input(TestBuffer::from_vec("c", pack_f32(&c_vec, dt), dt))
            .input(TestBuffer::from_vec("dt", pack_f32(&dt_in, dt), dt))
            .input(TestBuffer::from_vec("h", pack_f32(&h_state, DType::F32), DType::F32))
            .input(TestBuffer::zeros("y", n_heads * head_dim, dt))
            .constexpr("head_dim", head_dim as u32)
            .constexpr("state_dim", state_dim as u32)
            .expect(TestBuffer::from_vec("y", pack_f32(&y_exp, dt), dt))
            .expect(TestBuffer::from_vec("h", pack_f32(&h_exp, DType::F32), DType::F32))
            .grid_3d((n_heads * head_dim) as u32, 1, 1, [1, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-5, 5e-3, 5e-2])]
    fn test_ssm_step_a2d(dt: DType) -> TestSetup { ssm_step_a2d_setup(4, 16, 8, dt) }

    // ── mt_ssm_step (MLX-aligned ssm_step<T,Dh,Ds,H,G>) ─────────────────
    //
    // Distinct from `ssm_step`: separate `state_in`/`state_out` buffers (not
    // in-place), a `d_skip` residual (`out = Σ C·state' + x·D`), GQA per-group
    // B/C sharing (`g = n/heads_per_group`), and a per-state simd_sum across
    // a 32-thread group (`ds % 32 == 0`, each thread owns `ds/32` states).
    // Grid: `(dh, n_heads*batch, 1)` threadgroups of 32.

    /// CPU oracle mirroring `mt_ssm_step` exactly (batch folded into `n`).
    #[allow(clippy::too_many_arguments)]
    fn mt_ssm_step_oracle(
        x: &[f32],
        a_log: &[f32],
        b_mat: &[f32],
        c_mat: &[f32],
        d_skip: &[f32],
        dt_in: &[f32],
        state_in: &[f32],
        n: usize,
        dh: usize,
        ds: usize,
        n_heads: usize,
        heads_per_group: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut out = vec![0.0_f32; n * dh];
        let mut state_out = state_in.to_vec();
        for nn in 0..n {
            let h_idx = nn % n_heads;
            let g_idx = nn / heads_per_group;
            let dt_val = dt_in[nn];
            let da = (-(a_log[h_idx].exp()) * dt_val).exp();
            let bc_base = g_idx * ds;
            for d in 0..dh {
                let x_val = x[nn * dh + d];
                let state_base = nn * dh * ds + d * ds;
                let mut acc = 0.0_f32;
                for s in 0..ds {
                    let idx = state_base + s;
                    let new_state = da * state_in[idx] + x_val * dt_val * b_mat[bc_base + s];
                    state_out[idx] = new_state;
                    acc += new_state * c_mat[bc_base + s];
                }
                out[nn * dh + d] = acc + x_val * d_skip[h_idx];
            }
        }
        (out, state_out)
    }

    fn mt_ssm_step_setup(
        n: usize,
        dh: usize,
        ds: usize,
        n_heads: usize,
        heads_per_group: usize,
        dt: DType,
    ) -> TestSetup {
        let n_groups = n_heads / heads_per_group;
        let x: Vec<f32> = (0..n * dh).map(|i| ((i as f32) * 0.013).sin() * 0.3).collect();
        let a_log: Vec<f32> = (0..n_heads).map(|i| -1.0 + (i as f32) * 0.1).collect();
        let b_mat: Vec<f32> = (0..n_groups * ds).map(|i| 0.1 + (i as f32) * 0.03).collect();
        let c_mat: Vec<f32> = (0..n_groups * ds).map(|i| 0.2 - (i as f32) * 0.01).collect();
        let d_skip: Vec<f32> = (0..n_heads).map(|i| 0.05 + (i as f32) * 0.02).collect();
        let dt_in: Vec<f32> = (0..n).map(|i| 0.01 + (i as f32) * 0.003).collect();
        let state_in: Vec<f32> =
            (0..n * dh * ds).map(|i| ((i as f32) * 0.011).cos() * 0.1).collect();

        let (out_exp, state_exp) = mt_ssm_step_oracle(
            &x,
            &a_log,
            &b_mat,
            &c_mat,
            &d_skip,
            &dt_in,
            &state_in,
            n,
            dh,
            ds,
            n_heads,
            heads_per_group,
        );

        TestSetup::new(mt_ssm_step::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&x, dt), dt))
            .input(TestBuffer::from_vec("a_log", pack_f32(&a_log, dt), dt))
            .input(TestBuffer::from_vec("b_mat", pack_f32(&b_mat, dt), dt))
            .input(TestBuffer::from_vec("c_mat", pack_f32(&c_mat, dt), dt))
            .input(TestBuffer::from_vec("d_skip", pack_f32(&d_skip, dt), dt))
            .input(TestBuffer::from_vec("dt", pack_f32(&dt_in, dt), dt))
            .input(TestBuffer::from_vec("state_in", pack_f32(&state_in, dt), dt))
            .input(TestBuffer::zeros("state_out", n * dh * ds, dt))
            .input(TestBuffer::zeros("out", n * dh, dt))
            .constexpr("dh", dh as u32)
            .constexpr("ds", ds as u32)
            .constexpr("n_heads", n_heads as u32)
            .constexpr("heads_per_group", heads_per_group as u32)
            .expect(TestBuffer::from_vec("state_out", pack_f32(&state_exp, dt), dt))
            .expect(TestBuffer::from_vec("out", pack_f32(&out_exp, dt), dt))
            .grid_3d(dh as u32, n as u32, 1, [32, 1, 1])
    }

    // batch folded into n: n_heads=4 (1 batch row), dh=16, ds=32 (=32 so one
    // thread per state, ds%32==0), heads_per_group=2.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_mt_ssm_step(dt: DType) -> TestSetup { mt_ssm_step_setup(4, 16, 32, 4, 2, dt) }

    // ── conv1d_causal_prefill ─────────────────────────────────────────────

    fn conv1d_causal_prefill_oracle(
        xbc: &[f32],
        w: &[f32],
        bias: &[f32],
        s: usize,
        conv_dim: usize,
        kc: usize,
    ) -> Vec<f32> {
        let mut y = vec![0.0f32; s * conv_dim];
        for ti in 0..s {
            for ch in 0..conv_dim {
                let mut acc = bias[ch];
                for k in 0..kc {
                    let lag = kc - 1 - k;
                    if ti >= lag {
                        acc += w[k * conv_dim + ch] * xbc[(ti - lag) * conv_dim + ch];
                    }
                }
                let sig = 1.0 / (1.0f32 + (-acc).exp());
                y[ti * conv_dim + ch] = acc * sig;
            }
        }
        y
    }

    fn conv1d_causal_prefill_setup(s: usize, conv_dim: usize, kc: usize) -> TestSetup {
        let dt = DType::F32;
        let xbc: Vec<f32> = (0..s * conv_dim).map(|i| ((i as f32) * 0.011).sin() * 0.5).collect();
        let w: Vec<f32> = (0..kc * conv_dim).map(|i| 0.1 + ((i as f32) * 0.019).cos() * 0.2).collect();
        let bias: Vec<f32> = (0..conv_dim).map(|i| (i as f32) * 0.001 - 0.05).collect();
        let y_exp = conv1d_causal_prefill_oracle(&xbc, &w, &bias, s, conv_dim, kc);
        use super::conv1d_causal_prefill;
        TestSetup::new(conv1d_causal_prefill::kernel_ir_for())
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("xbc_in", pack_f32(&xbc, dt), dt))
            .input(TestBuffer::from_vec("w", pack_f32(&w, dt), dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias, dt), dt))
            .input(TestBuffer::zeros("y", s * conv_dim, dt))
            .constexpr("conv_dim", conv_dim as u32)
            .constexpr("kc", kc as u32)
            .expect(TestBuffer::from_vec("y", pack_f32(&y_exp, dt), dt))
            .grid_3d((s * conv_dim) as u32, 1, 1, [1, 1, 1])
    }

    #[test_kernel(dtypes = [f32], tol = [1e-5])]
    fn test_conv1d_causal_prefill(_dt: DType) -> TestSetup { conv1d_causal_prefill_setup(8, 32, 4) }

    // ── strided_col_copy ──────────────────────────────────────────────────

    fn strided_col_copy_oracle(src: &[f32], s: usize, stride: usize, col_off: usize, width: usize) -> Vec<f32> {
        (0..s).flat_map(|ti| (0..width).map(move |ci| src[ti * stride + col_off + ci])).collect()
    }

    fn strided_col_copy_setup(s: usize, stride: usize, col_off: usize, width: usize) -> TestSetup {
        let dt = DType::F32;
        let src: Vec<f32> = (0..s * stride).map(|i| ((i as f32) * 0.017).sin()).collect();
        let exp_v = strided_col_copy_oracle(&src, s, stride, col_off, width);
        use super::strided_col_copy;
        TestSetup::new(strided_col_copy::kernel_ir_for())
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("src", pack_f32(&src, dt), dt))
            .input(TestBuffer::zeros("dst", s * width, dt))
            .constexpr("stride", stride as u32)
            .constexpr("col_off", col_off as u32)
            .constexpr("width", width as u32)
            .expect(TestBuffer::from_vec("dst", pack_f32(&exp_v, dt), dt))
            .grid_3d((s * width) as u32, 1, 1, [1, 1, 1])
    }

    #[test_kernel(dtypes = [f32], tol = [1e-6])]
    fn test_strided_col_copy(_dt: DType) -> TestSetup { strided_col_copy_setup(4, 10, 2, 3) }

    // ── softplus_add_rows ─────────────────────────────────────────────────

    fn softplus_add_rows_oracle(src: &[f32], bias: &[f32], n: usize) -> Vec<f32> {
        let s = src.len() / n;
        (0..s).flat_map(|ti| (0..n).map(move |hi| {
            let raw = src[ti * n + hi] + bias[hi];
            if raw > 20.0 { raw } else { (1.0f32 + raw.exp()).ln() }
        })).collect()
    }

    fn softplus_add_rows_setup(s: usize, n: usize) -> TestSetup {
        let dt = DType::F32;
        let src: Vec<f32> = (0..s * n).map(|i| ((i as f32) * 0.023).sin() * 2.0).collect();
        let bias: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.2).collect();
        let exp_v = softplus_add_rows_oracle(&src, &bias, n);
        use super::softplus_add_rows;
        TestSetup::new(softplus_add_rows::kernel_ir_for())
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("src", pack_f32(&src, dt), dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias, dt), dt))
            .input(TestBuffer::zeros("dst", s * n, dt))
            .constexpr("n", n as u32)
            .expect(TestBuffer::from_vec("dst", pack_f32(&exp_v, dt), dt))
            .grid_3d((s * n) as u32, 1, 1, [1, 1, 1])
    }

    #[test_kernel(dtypes = [f32], tol = [1e-5])]
    fn test_softplus_add_rows(_dt: DType) -> TestSetup { softplus_add_rows_setup(4, 8) }
}

/// New-syntax benchmarks for all four `ffai::ssm` kernels. `conv1d_causal_step`
/// and `ssm_step` are also correctness-tested above; `ssm_step_a2d` (2-D
/// per-(channel,state) A_log) and `mt_ssm_step` (MLX-aligned reduction form)
/// are bench-only — both carry recurrent state with no clean one-step oracle
/// inside this harness. All MLX-less (`class=GenericEmpty`), `Ref(GB/s)` blank.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{conv1d_causal_step, mt_ssm_step, ssm_step, ssm_step_a2d};

    // Mamba 2 short-conv at a realistic channel count, K=4. One thread/channel.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_conv1d_causal_step(dt: DType) -> BenchSetup {
        let (n_channels, kernel_size) = (1536usize, 4usize);
        BenchSetup::new(conv1d_causal_step::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("x", n_channels, dt))
            .buffer(BenchBuffer::random("w", kernel_size * n_channels, dt))
            .buffer(BenchBuffer::random("b", n_channels, dt))
            .buffer(BenchBuffer::random("state", (kernel_size - 1) * n_channels, dt).output())
            .buffer(BenchBuffer::zeros("y", n_channels, dt).output())
            .constexpr("n_channels", n_channels as u32)
            .constexpr("kernel_size", kernel_size as u32)
            .grid_3d(n_channels as u32, 1, 1, [1, 1, 1])
            .bytes_moved((kernel_size * n_channels * dt.size_bytes()) as u64)
    }

    // Scalar-A selective-scan decode. One thread per (head, d).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_ssm_step(dt: DType) -> BenchSetup {
        let (n_heads, head_dim, state_dim) = (32usize, 64usize, 16usize);
        BenchSetup::new(ssm_step::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("x", n_heads * head_dim, dt))
            .buffer(BenchBuffer::random("a", n_heads, dt))
            .buffer(BenchBuffer::random("b", state_dim, dt))
            .buffer(BenchBuffer::random("c", state_dim, dt))
            .buffer(BenchBuffer::random("dt", n_heads, dt))
            .buffer(BenchBuffer::random("h", n_heads * state_dim * head_dim, DType::F32).output())
            .buffer(BenchBuffer::zeros("y", n_heads * head_dim, dt).output())
            .constexpr("head_dim", head_dim as u32)
            .constexpr("state_dim", state_dim as u32)
            .grid_3d((n_heads * head_dim) as u32, 1, 1, [1, 1, 1])
            .bytes_moved((n_heads * state_dim * head_dim * 2 * 4) as u64)
    }

    // Mamba 1 (Jamba) 2-D A_log variant. `a_log` is [n_heads*head_dim, state_dim].
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_ssm_step_a2d(dt: DType) -> BenchSetup {
        let (n_heads, head_dim, state_dim) = (32usize, 64usize, 16usize);
        let channels = n_heads * head_dim;
        BenchSetup::new(ssm_step_a2d::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("x", channels, dt))
            .buffer(BenchBuffer::random("a_log", channels * state_dim, dt))
            .buffer(BenchBuffer::random("b", state_dim, dt))
            .buffer(BenchBuffer::random("c", state_dim, dt))
            .buffer(BenchBuffer::random("dt", n_heads, dt))
            .buffer(BenchBuffer::random("h", n_heads * state_dim * head_dim, DType::F32).output())
            .buffer(BenchBuffer::zeros("y", channels, dt).output())
            .constexpr("head_dim", head_dim as u32)
            .constexpr("state_dim", state_dim as u32)
            .grid_3d(channels as u32, 1, 1, [1, 1, 1])
            .bytes_moved((channels * state_dim * 4) as u64)
    }

    // MLX-aligned reduction form: one simdgroup per (d_idx, n) reduces the
    // state axis via simd_sum. Grid `[dh, n_heads*batch, 1]`, TG `[32,1,1]`.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mt_ssm_step(dt: DType) -> BenchSetup {
        let (n_heads, heads_per_group, batch, dh, ds) = (8usize, 2usize, 2usize, 64usize, 32usize);
        let n_total = n_heads * batch;
        let groups = n_total / heads_per_group;
        BenchSetup::new(mt_ssm_step::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", n_total * dh, dt))
            .buffer(BenchBuffer::random("a_log", n_heads, dt))
            .buffer(BenchBuffer::random("b_mat", groups * ds, dt))
            .buffer(BenchBuffer::random("c_mat", groups * ds, dt))
            .buffer(BenchBuffer::random("d_skip", n_heads, dt))
            .buffer(BenchBuffer::random("dt", n_total, dt))
            .buffer(BenchBuffer::random("state_in", n_total * dh * ds, dt))
            .buffer(BenchBuffer::zeros("state_out", n_total * dh * ds, dt).output())
            .buffer(BenchBuffer::zeros("out", n_total * dh, dt).output())
            .constexpr("dh", dh as u32)
            .constexpr("ds", ds as u32)
            .constexpr("n_heads", n_heads as u32)
            .constexpr("heads_per_group", heads_per_group as u32)
            .grid_3d(dh as u32, n_total as u32, 1, [32, 1, 1])
            .bytes_moved((n_total * dh * ds * 2 * dt.size_bytes()) as u64)
    }
}

// ── Fused Mamba projection split ─────────────────────────────────────────
//
// Replaces the 3 sequential `strided_col_copy` calls that carve z, xbc, and
// dt_raw out of the [s, in_proj_out] projection tensor.  One thread per
// output column × token: reads from the same source row once and writes to
// the appropriate output buffer.  Eliminates two round-trip dispatch launches.
//
// Layout (in_proj_out = di + conv_dim + m_nh = 4096 + 6144 + 64 = 10304):
//   z_raw    : cols [0        .. di           )  → [s * di]
//   xbc      : cols [di       .. di+conv_dim  )  → [s * conv_dim]
//   dt_raw   : cols [di+conv_dim .. in_proj_out)  → [s * m_nh]
//
// Grid: [s * in_proj_out, 1, 1]; one thread per source element.
// Each thread identifies which output slice it belongs to and writes there.
#[kernel]
pub fn mamba_split_proj(
    proj: Tensor<f32>,       // [s * in_proj_out] flat row-major
    mut z_out: Tensor<f32>,  // [s * di]
    mut xbc_out: Tensor<f32>, // [s * conv_dim]
    mut dt_out: Tensor<f32>, // [s * m_nh]
    #[constexpr] in_proj_out: u32,
    #[constexpr] di: u32,
    #[constexpr] conv_dim: u32,
    #[constexpr] m_nh: u32,
) {
    let idx = program_id::<0>();
    let ti = idx / in_proj_out;
    let ci = idx - ti * in_proj_out;
    let val = load(proj[idx]);
    if ci < di {
        store(z_out[ti * di + ci], val);
    } else if ci < di + conv_dim {
        store(xbc_out[ti * conv_dim + (ci - di)], val);
    } else {
        store(dt_out[ti * m_nh + (ci - di - conv_dim)], val);
    }
}

// ── Fused Mamba conv output split ────────────────────────────────────────
//
// Replaces the 3 sequential `strided_col_copy` calls that carve x_ssm, b,
// and c out of yc_silu [s, conv_dim].  Grid matches source size.
//
// Layout (conv_dim = di + 2*ng*ds = 4096 + 2*8*128 = 4096 + 2048 = 6144):
//   x_ssm : cols [0             .. di          )  → [s * di]
//   b     : cols [di            .. di+ng*ds    )  → [s * ng*ds]
//   c     : cols [di+ng*ds      .. conv_dim    )  → [s * ng*ds]
//
// Grid: [s * conv_dim, 1, 1]; one thread per source element.
#[kernel]
pub fn mamba_split_conv(
    yc: Tensor<f32>,          // [s * conv_dim] flat row-major
    mut x_out: Tensor<f32>,   // [s * di]
    mut b_out: Tensor<f32>,   // [s * ng_ds]
    mut c_out: Tensor<f32>,   // [s * ng_ds]
    #[constexpr] conv_dim: u32,
    #[constexpr] di: u32,
    #[constexpr] ng_ds: u32,  // ng * ds
) {
    let idx = program_id::<0>();
    let ti = idx / conv_dim;
    let ci = idx - ti * conv_dim;
    let val = load(yc[idx]);
    if ci < di {
        store(x_out[ti * di + ci], val);
    } else if ci < di + ng_ds {
        store(b_out[ti * ng_ds + (ci - di)], val);
    } else {
        store(c_out[ti * ng_ds + (ci - di - ng_ds)], val);
    }
}

// ── Batched gated group RMSNorm (Mamba2 Zamba2RMSNormGated, prefill) ──────
//
// Applies the per-group gated RMSNorm over S tokens in one dispatch,
// eliminating the host download+compute+upload round-trip in the CONV_DEVICE
// prefill path.
//
// For each token ti and group grp:
//   gate_i = y_i * silu(z_i)     (gated output, same as decode path)
//   rms    = 1/sqrt( mean(gate^2) + eps )  over the group
//   out_i  = gate_i * rms * w_i
//
// Grid: [s * ng, 1, 1], block: [gs/4, 1, 1].
//   One thread-group per (token, norm-group) pair.
//   Each thread in the block handles 4 consecutive elements.
#[kernel]
pub fn gated_group_rmsnorm_batched(
    y: Tensor<f32>,      // [s * di] flat
    z: Tensor<f32>,      // [s * di] flat
    w: Tensor<f32>,      // [di]     norm weights (shared across tokens)
    mut out: Tensor<f32>, // [s * di]
    eps_buf: Tensor<f32>, // [1]
    #[constexpr] gs: u32,  // group size (512 for Nemotron)
    #[constexpr] ng: u32,  // number of groups per token (8 for Nemotron)
) {
    // program_id::<0>() = token * ng + group
    let tg = program_id::<0>();
    let grp = tg - (tg / ng) * ng;
    let ti  = tg / ng;
    let rs  = ti * ng * gs + grp * gs;  // start offset in [s * di]
    let col = tid * 4u32;
    let in_bounds = col + 3u32 < gs;
    let safe_col = select(in_bounds, col, 0u32);
    let sb = rs + safe_col;
    let y0 = load(y[sb]);
    let y1 = load(y[sb + 1u32]);
    let y2 = load(y[sb + 2u32]);
    let y3 = load(y[sb + 3u32]);
    let z0 = load(z[sb]);
    let z1 = load(z[sb + 1u32]);
    let z2 = load(z[sb + 2u32]);
    let z3 = load(z[sb + 3u32]);
    let g0 = y0 * (z0 / (1.0f32 + exp(0.0f32 - z0)));
    let g1 = y1 * (z1 / (1.0f32 + exp(0.0f32 - z1)));
    let g2 = y2 * (z2 / (1.0f32 + exp(0.0f32 - z2)));
    let g3 = y3 * (z3 / (1.0f32 + exp(0.0f32 - z3)));
    let raw = g0 * g0 + g1 * g1 + g2 * g2 + g3 * g3;
    let partial = select(in_bounds, raw, 0.0f32);
    let ssq = reduce_sum(partial);
    let eps = load(eps_buf[0]);
    let rms = rsqrt(ssq / (gs.cast::<f32>()) + eps);
    if in_bounds {
        let base = rs + col;
        store(out[base],          (g0 * rms * load(w[grp * gs + col])));
        store(out[base + 1u32],   (g1 * rms * load(w[grp * gs + col + 1u32])));
        store(out[base + 2u32],   (g2 * rms * load(w[grp * gs + col + 2u32])));
        store(out[base + 3u32],   (g3 * rms * load(w[grp * gs + col + 3u32])));
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Mamba2 SSD chunked-matmul prefill scan — PORTABLE elementwise kernels.
//
// These are the portable (MSL/HIP/SPIRV-codegen) analogs of the raw-CUDA
// helper kernels in `ffai-ops/src/ssd_scan.rs`. They prepare the operands for
// the 4 batched GEMMs (run via `ffai_gemm_batched`) and combine the result.
// Everything runs in f32 (no f16 dependency) for portability + correctness.
//
// Fixed for NemotronH: dh=64, ds=128, H=64, G=8 (hpg=8). L (chunk len) is a
// runtime constexpr (128/256). nc = ceil(T/L). bhc = nc*H. The tail chunk is
// zero-padded (t = c*L + i ≥ T reads 0).
// ══════════════════════════════════════════════════════════════════════════

// Inclusive cumsum of A·dt within each chunk → Lcs. One thread per (chunk,head).
//   A = -exp(a_log[h]),  Lcs[bh, i] = Σ_{k≤i} A·dt[c*L+k]
// dt layout [T, H]; lcs layout [nc*H, L]. Grid: [nc*H, 1, 1].
#[kernel]
pub fn ssd_lcs(
    dt: Tensor<f32>,       // [T, H]
    a_log: Tensor<f32>,    // [H]
    mut lcs: Tensor<f32>,  // [nc*H, L]
    #[constexpr] t_total: u32,
    #[constexpr] n_heads: u32,
    #[constexpr] l: u32,
    #[constexpr] nc: u32,
) {
    let idx = program_id::<0>();      // bh = c*H + h
    let total = nc * n_heads;
    if idx < total {
        let c = idx / n_heads;
        let h = idx - c * n_heads;
        let a = 0.0f32 - exp(load(a_log[h]));
        let mut acc = 0.0f32;
        for i in range(0u32, l, 1u32) {
            let t = c * l + i;
            let dtv = select(t < t_total, load(dt[t * n_heads + h]), 0.0f32);
            acc = acc + a * dtv;
            store(lcs[idx * l + i], acc);
        }
    }
}

// Gather/broadcast B,C from [T,G,ds] into [nc*H, L, ds] (head h uses group
// h/hpg). One thread per output element. Grid: [nc*H*L*ds, 1, 1].
#[kernel]
pub fn ssd_gather_bc(
    b_mat: Tensor<f32>,    // [T, G, ds]
    c_mat: Tensor<f32>,    // [T, G, ds]
    mut b_out: Tensor<f32>, // [nc*H, L, ds]
    mut c_out: Tensor<f32>, // [nc*H, L, ds]
    #[constexpr] t_total: u32,
    #[constexpr] n_heads: u32,
    #[constexpr] n_groups: u32,
    #[constexpr] hpg: u32,
    #[constexpr] l: u32,
    #[constexpr] ds: u32,
    #[constexpr] nc: u32,
) {
    let e = program_id::<0>();
    let n = nc * n_heads * l * ds;
    if e < n {
        let s = e - (e / ds) * ds;
        let i = (e / ds) - (e / (ds * l)) * l;
        let bh = e / (ds * l);
        let c = bh / n_heads;
        let h = bh - c * n_heads;
        let g = h / hpg;
        let t = c * l + i;
        let valid = t < t_total;
        let t_safe = select(valid, t, 0u32);
        let src = (t_safe * n_groups + g) * ds + s;
        let bv = select(valid, load(b_mat[src]), 0.0f32);
        let cv = select(valid, load(c_mat[src]), 0.0f32);
        store(b_out[e], bv);
        store(c_out[e], cv);
    }
}

// Transpose x [T,H,dh] → xt [nc*H, dh, L]. One thread per output element.
// Grid: [nc*H*dh*L, 1, 1].
#[kernel]
pub fn ssd_xt(
    x: Tensor<f32>,        // [T, H, dh]
    mut xt: Tensor<f32>,   // [nc*H, dh, L]
    #[constexpr] t_total: u32,
    #[constexpr] n_heads: u32,
    #[constexpr] dh: u32,
    #[constexpr] l: u32,
    #[constexpr] nc: u32,
) {
    let e = program_id::<0>();
    let n = nc * n_heads * dh * l;
    if e < n {
        let i = e - (e / l) * l;              // position within chunk
        let p = (e / l) - (e / (l * dh)) * dh; // head_dim index
        let bh = e / (l * dh);
        let c = bh / n_heads;
        let h = bh - c * n_heads;
        let t = c * l + i;
        let valid = t < t_total;
        let t_safe = select(valid, t, 0u32);
        let v = select(valid, load(x[(t_safe * n_heads + h) * dh + p]), 0.0f32);
        store(xt[e], v);
    }
}

// M[i,j] = CB[i,j]·exp(Lcs[bh,i]-Lcs[bh,j])·dt[c*L+j,h], causal (i≥j) else 0.
// CB is the output of G1 ([nc*H, L, L]). One thread per element.
// Grid: [nc*H*L*L, 1, 1].
#[kernel]
pub fn ssd_mmask(
    cb: Tensor<f32>,       // [nc*H, L, L] = C·Bᵀ
    lcs: Tensor<f32>,      // [nc*H, L]
    dt: Tensor<f32>,       // [T, H]
    mut m_out: Tensor<f32>, // [nc*H, L, L]
    #[constexpr] t_total: u32,
    #[constexpr] n_heads: u32,
    #[constexpr] l: u32,
    #[constexpr] nc: u32,
) {
    let e = program_id::<0>();
    let n = nc * n_heads * l * l;
    if e < n {
        let j = e - (e / l) * l;
        let i = (e / l) - (e / (l * l)) * l;
        let bh = e / (l * l);
        // causal mask: i < j → 0
        if i < j {
            store(m_out[e], 0.0f32);
        } else {
            let c = bh / n_heads;
            let h = bh - c * n_heads;
            let tj = c * l + j;
            let dtj = select(tj < t_total, load(dt[tj * n_heads + h]), 0.0f32);
            let decay = exp(load(lcs[bh * l + i]) - load(lcs[bh * l + j]));
            store(m_out[e], load(cb[e]) * decay * dtj);
        }
    }
}

// BdT[s,j] = exp(Lcs[bh,L-1]-Lcs[bh,j])·dt[c*L+j,h]·B[c*L+j,g,s] → [nc*H, ds, L]
// (decayed, dt-weighted, transposed B for the chunk-state G3). One thread/elem.
// Grid: [nc*H*ds*L, 1, 1].
#[kernel]
pub fn ssd_bdt(
    b_mat: Tensor<f32>,    // [T, G, ds]
    lcs: Tensor<f32>,      // [nc*H, L]
    dt: Tensor<f32>,       // [T, H]
    mut bdt: Tensor<f32>,  // [nc*H, ds, L]
    #[constexpr] t_total: u32,
    #[constexpr] n_heads: u32,
    #[constexpr] n_groups: u32,
    #[constexpr] hpg: u32,
    #[constexpr] l: u32,
    #[constexpr] ds: u32,
    #[constexpr] nc: u32,
) {
    let e = program_id::<0>();
    let n = nc * n_heads * ds * l;
    if e < n {
        let j = e - (e / l) * l;             // position
        let s = (e / l) - (e / (l * ds)) * ds; // state index
        let bh = e / (l * ds);
        let c = bh / n_heads;
        let h = bh - c * n_heads;
        let g = h / hpg;
        let tj = c * l + j;
        let valid = tj < t_total;
        let tj_safe = select(valid, tj, 0u32);
        let bv = select(valid, load(b_mat[(tj_safe * n_groups + g) * ds + s]), 0.0f32);
        let dtj = select(valid, load(dt[tj_safe * n_heads + h]), 0.0f32);
        let decay = exp(load(lcs[bh * l + (l - 1u32)]) - load(lcs[bh * l + j]));
        store(bdt[e], decay * dtj * bv);
    }
}

// Serial inter-chunk recurrence. For each (head h, state s, head_dim p):
//   S_in[c]   = state at START of chunk c; S_in[0] = state_in[h,p,s]
//   S_in[c+1] = αc·S_in[c] + S_chunk[c],  αc = exp(Lcs[bh, L-1])
// Emits SinT[bh] = S_in[c]ᵀ as [nc*H, dh, ds] (transposed for G4) and the FINAL
// state (after chunk nc-1) → state_out [H, dh, ds].
// Grid: [H*ds*dh, 1, 1]; one thread per (head, s, p), loops nc serially.
#[kernel]
pub fn ssd_recur(
    s_chunk: Tensor<f32>,   // [nc*H, ds, dh]
    lcs: Tensor<f32>,       // [nc*H, L]
    state_in: Tensor<f32>,  // [H, dh, ds]
    mut sin_t: Tensor<f32>, // [nc*H, dh, ds]  (S_inᵀ per chunk)
    mut state_out: Tensor<f32>, // [H, dh, ds]
    #[constexpr] n_heads: u32,
    #[constexpr] dh: u32,
    #[constexpr] ds: u32,
    #[constexpr] l: u32,
    #[constexpr] nc: u32,
) {
    let idx = program_id::<0>();
    let tot = n_heads * ds * dh;
    if idx < tot {
        let p = idx - (idx / dh) * dh;       // head_dim
        let s = (idx / dh) - (idx / (dh * ds)) * ds; // state
        let h = idx / (dh * ds);
        let mut st = load(state_in[(h * dh + p) * ds + s]);
        for c in range(0u32, nc, 1u32) {
            let bh = c * n_heads + h;
            // emit S_inᵀ for this chunk BEFORE applying it
            store(sin_t[(bh * dh + p) * ds + s], st);
            let alpha = exp(load(lcs[bh * l + (l - 1u32)]));
            let sc = load(s_chunk[(bh * ds + s) * dh + p]);
            st = alpha * st + sc;
        }
        store(state_out[(h * dh + p) * ds + s], st);
    }
}

// Final combine. y[t,h,p] = y_intra[bh,i,p] + exp(Lcs[bh,i])·CS[bh,i,p]
//   + x[t,h,p]·D[h]. y_intra, CS are [nc*H, L, dh]; output y [T,H,dh].
// One thread per (bh, i, p) element; tail rows (t≥T) skipped.
// Grid: [nc*H*L*dh, 1, 1].
#[kernel]
pub fn ssd_combine(
    y_intra: Tensor<f32>,  // [nc*H, L, dh]
    cs: Tensor<f32>,       // [nc*H, L, dh]
    lcs: Tensor<f32>,      // [nc*H, L]
    x: Tensor<f32>,        // [T, H, dh]
    d_skip: Tensor<f32>,   // [H]
    mut y: Tensor<f32>,    // [T, H, dh]
    #[constexpr] t_total: u32,
    #[constexpr] n_heads: u32,
    #[constexpr] dh: u32,
    #[constexpr] l: u32,
    #[constexpr] nc: u32,
) {
    let e = program_id::<0>();
    let n = nc * n_heads * l * dh;
    if e < n {
        let p = e - (e / dh) * dh;
        let i = (e / dh) - (e / (dh * l)) * l;
        let bh = e / (dh * l);
        let c = bh / n_heads;
        let h = bh - c * n_heads;
        let t = c * l + i;
        if t < t_total {
            let yi = load(y_intra[e]);
            let decay = exp(load(lcs[bh * l + i]));
            let ci = load(cs[e]) * decay;
            let xv = load(x[(t * n_heads + h) * dh + p]);
            store(y[(t * n_heads + h) * dh + p], yi + ci + xv * load(d_skip[h]));
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════
// SSD fused gather-GEMMs — kill the 8× gather_bc materialization.
//
// The portable SSD scan used to materialize b_g / c_g `[nc*H, L, ds]` (B,C
// broadcast across the 8 heads-per-group) just to feed the G1 (CB = C·Bᵀ) and
// G4 (CS = C·Sᵀ) batched GEMMs. That's an 8× redundant HBM write+read (head h
// reads group h/hpg) and was the single largest elementwise cost in the scan
// (~70% of the `pre` bucket). These two GEMMs read B / C straight from their
// `[T, G, ds]` source with the broadcast folded into the tile-load index
// arithmetic — no `[nc*H, L, ds]` scratch, no 8× traffic.
//
// Same 32×32 / 16-K Reduction-mode tiling as `ffai_gemm_batched`; the batch
// index `bz = c*H + h` rides `tgid_z`, and the group `g = h/hpg` selects the
// B/C slice. Tail rows (t = c*L + row ≥ T) clamp-load 0 (zero-padded chunk).
// Portable (codegens MSL/CUDA/HIP/SPIR-V — pure index math, no raw intrinsics).
// ══════════════════════════════════════════════════════════════════════════

// G1 fused: CB[i,j] = Σ_s C[t_i,g,s] · B[t_j,g,s], then the mmask epilogue
//   M[i,j] = CB[i,j] · exp(Lcs[bh,i] - Lcs[bh,j]) · dt[t_j,h], causal (i≥j).
// → writes M [nc*H, L, L] directly. Fuses the separate `ssd_mmask` [L,L]
// HBM round-trip (read CB, write M) into the GEMM epilogue: the `cb` scratch
// is gone and the decay-mask multiply rides on the GEMM store.
//   weight role = B (col j → t_j), input role = C (row i → t_i), k = ds.
// Reads B,C directly from [T, G, ds]; equivalent to ffai_gemm_batched on
// (b_g, c_g) but without ever materializing them or CB.
#[kernel]
pub fn ssd_g1_cb(
    b_mat: Tensor<f32>,    // [T, G, ds]   (weight role)
    c_mat: Tensor<f32>,    // [T, G, ds]   (input role)
    lcs: Tensor<f32>,      // [nc*H, L]
    dt: Tensor<f32>,       // [T, H]
    mut out: Tensor<f32>,  // [nc*H, L, L]  (M = CB ⊙ decay-mask)
    #[constexpr] t_total: u32,
    #[constexpr] n_heads: u32,
    #[constexpr] n_groups: u32,
    #[constexpr] hpg: u32,
    #[constexpr] l: u32,
    #[constexpr] ds: u32,
) {
    let bz = program_id::<2>();          // batch = c*H + h
    let c = bz / n_heads;
    let h = bz - c * n_heads;
    let g = h / hpg;
    let tid = simd_id * 32u32 + simd_lane;
    let lr = tid / 32u32; // output row within tile (i, 0..31)
    let lo = tid % 32u32; // output col within tile (j, 0..31)
    threadgroup_alloc("g1cb_b", 512);
    threadgroup_alloc("g1cb_c", 512);
    let mut acc = 0.0f32;
    for k0 in range(0u32, ds, 16u32) {
        if tid < 512u32 {
            // B tile: col j = tgid_x*32 + s/16, state k = k0 + s%16.
            let s = tid;
            let j = tgid_x * 32u32 + s / 16u32;
            let tj = c * l + j;
            let valid = (j < l) & (tj < t_total);
            let tj_safe = select(valid, tj, 0u32);
            let kk = k0 + s - (s / 16u32) * 16u32;
            let bv = select(valid, load(b_mat[(tj_safe * n_groups + g) * ds + kk]), 0.0f32);
            threadgroup_store("g1cb_b", s, bv);
        }
        if tid >= 512u32 {
            // C tile: row i = tgid_y*32 + s/16, state k = k0 + s%16.
            let s = tid - 512u32;
            let i = tgid_y * 32u32 + s / 16u32;
            let ti = c * l + i;
            let valid = (i < l) & (ti < t_total);
            let ti_safe = select(valid, ti, 0u32);
            let kk = k0 + s - (s / 16u32) * 16u32;
            let cv = select(valid, load(c_mat[(ti_safe * n_groups + g) * ds + kk]), 0.0f32);
            threadgroup_store("g1cb_c", s, cv);
        }
        threadgroup_barrier();
        for k in range(0u32, 16u32, 1u32) {
            let bb = threadgroup_load("g1cb_b", lo * 16u32 + k);
            let cc = threadgroup_load("g1cb_c", lr * 16u32 + k);
            acc = acc + bb * cc;
        }
        threadgroup_barrier();
    }
    let i = tgid_y * 32u32 + lr;
    let j = tgid_x * 32u32 + lo;
    if i < l {
        if j < l {
            // mmask epilogue: causal (i<j → 0) · decay · dt[t_j, h].
            // `select` (not if/else) keeps it in the codegen-portable subset.
            let tj = c * l + j;
            let dtj = select(tj < t_total, load(dt[tj * n_heads + h]), 0.0f32);
            let decay = exp(load(lcs[bz * l + i]) - load(lcs[bz * l + j]));
            let m = select(i < j, 0.0f32, acc * decay * dtj);
            store(out[(bz * l + i) * l + j], m);
        }
    }
}

// G4 fused: CS[i,p] = Σ_s C[t_i,g,s] · SinT[bz,p,s]   →  [nc*H, L, dh].
//   weight role = sin_t[bz] [dh, ds] (col p), input role = C (row i → t_i),
//   k = ds. sin_t is already per-batch [nc*H, dh, ds]; only C is broadcast.
#[kernel]
pub fn ssd_g4_cs(
    sin_t: Tensor<f32>,    // [nc*H, dh, ds]   (weight role, per-batch)
    c_mat: Tensor<f32>,    // [T, G, ds]       (input role, broadcast)
    mut out: Tensor<f32>,  // [nc*H, L, dh]
    #[constexpr] t_total: u32,
    #[constexpr] n_heads: u32,
    #[constexpr] n_groups: u32,
    #[constexpr] hpg: u32,
    #[constexpr] l: u32,
    #[constexpr] ds: u32,
    #[constexpr] dh: u32,
) {
    let bz = program_id::<2>();          // batch = c*H + h
    let c = bz / n_heads;
    let h = bz - c * n_heads;
    let g = h / hpg;
    let w_base = bz * dh * ds;           // sin_t[bz] base
    let tid = simd_id * 32u32 + simd_lane;
    let lr = tid / 32u32; // output row within tile (i, 0..31)
    let lo = tid % 32u32; // output col within tile (p, 0..31)
    threadgroup_alloc("g4cs_w", 512);
    threadgroup_alloc("g4cs_c", 512);
    let mut acc = 0.0f32;
    for k0 in range(0u32, ds, 16u32) {
        if tid < 512u32 {
            // sin_t tile: col p = tgid_x*32 + s/16, state k = k0 + s%16.
            let s = tid;
            let p = tgid_x * 32u32 + s / 16u32;
            let valid = p < dh;
            let p_safe = select(valid, p, 0u32);
            let kk = k0 + s - (s / 16u32) * 16u32;
            let wv = select(valid, load(sin_t[w_base + p_safe * ds + kk]), 0.0f32);
            threadgroup_store("g4cs_w", s, wv);
        }
        if tid >= 512u32 {
            // C tile: row i = tgid_y*32 + s/16, state k = k0 + s%16.
            let s = tid - 512u32;
            let i = tgid_y * 32u32 + s / 16u32;
            let ti = c * l + i;
            let valid = (i < l) & (ti < t_total);
            let ti_safe = select(valid, ti, 0u32);
            let kk = k0 + s - (s / 16u32) * 16u32;
            let cv = select(valid, load(c_mat[(ti_safe * n_groups + g) * ds + kk]), 0.0f32);
            threadgroup_store("g4cs_c", s, cv);
        }
        threadgroup_barrier();
        for k in range(0u32, 16u32, 1u32) {
            let ww = threadgroup_load("g4cs_w", lo * 16u32 + k);
            let cc = threadgroup_load("g4cs_c", lr * 16u32 + k);
            acc = acc + ww * cc;
        }
        threadgroup_barrier();
    }
    let i = tgid_y * 32u32 + lr;
    let p = tgid_x * 32u32 + lo;
    if i < l {
        if p < dh {
            store(out[(bz * l + i) * dh + p], acc);
        }
    }
}
