//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Single-direction LSTM layer — the recurrent building block the StyleTTS2
//! / Kokoro prosody predictor, duration predictor, and text encoder need
//! (no FFAI model used an LSTM before this).
//!
//! Complements [`super::kokoro`]'s `lstm_cell`, which computes ONE timestep
//! and leaves the recurrence on the host (a per-step CPU↔GPU sync). This
//! kernel runs the **whole sequence** recurrence on the GPU in one dispatch
//! — the "GPU from the start" form — eliminating the per-timestep round
//! trip; the caller picks whichever fits its sequence lengths.
//!
//! Runs the full sequence recurrence on the GPU in **one threadgroup**:
//! thread `j` owns hidden unit `j`, with the hidden/cell state `h` / `c`
//! living in threadgroup memory across timesteps. Each step computes the
//! four gates for its unit and updates the state; a barrier between the
//! state read and write keeps the recurrence correct (every unit reads the
//! *previous* full `h` before any unit writes the new one). The time loop is
//! sequential (LSTM is inherently recurrent); the parallelism is across
//! hidden units + the per-gate matmuls.
//!
//! Per timestep `t` (PyTorch `nn.LSTM` cell, fused `b_ih + b_hh` into `bias`):
//!   `g = W_ih·x_t + W_hh·h_{t-1} + bias`   (4·hidden gate pre-activations)
//!   `i,f,o = σ(g_{i,f,o})`,  `g̃ = tanh(g_g)`
//!   `c_t = f ⊙ c_{t-1} + i ⊙ g̃`,  `h_t = o ⊙ tanh(c_t)`
//!
//! `reverse = 1` walks `t` from `seq_len-1 → 0` (the backward pass). A
//! **bidirectional** LSTM is two dispatches sharing one output buffer of
//! width `out_stride` (= 2·hidden): forward writes at `out_offset = 0`,
//! backward at `out_offset = hidden`, giving the concatenated `[seq_len,
//! 2·hidden]` result. A plain LSTM is one dispatch with `out_stride =
//! hidden`, `out_offset = 0`, `reverse = 0`.
//!
//! Layouts:
//!   x      `[seq_len, input_dim]`     T
//!   w_ih   `[4·hidden, input_dim]`    T   (gate order i, f, g, o)
//!   w_hh   `[4·hidden, hidden]`       T
//!   bias   `[4·hidden]`               f32 (b_ih + b_hh, precombined)
//!   out    `[seq_len, out_stride]`    T   (writes col `out_offset + j`)
//!
//! ## DISPATCH INVARIANTS
//!
//! Reduction, ONE threadgroup — dispatch with `grid_3d(1, 1, 1, [tpg, 1,
//! 1])` where `tpg = ceil(hidden / 32)·32` (≥ 32, ≤ 1024 → `hidden ≤
//! 1024`). Local thread id `j = simd_id·32 + simd_lane`; threads `j ≥
//! hidden` are idle (clamp-read, never write). Gate order in `w_ih`/`w_hh`/
//! `bias` is i, f, g, o.

use metaltile::kernel;

#[kernel(
    bench(
        op = "lstm",
        subop = "lstm",
        class = GenericEmpty,
        tol = 1e-3,
        kernel_mode = Reduction,
    )
)]
pub fn ffai_lstm<T>(
    x: Tensor<T>,
    w_ih: Tensor<T>,
    w_hh: Tensor<T>,
    bias: Tensor<f32>,
    out: Tensor<T>,
    #[constexpr] seq_len: u32,
    #[constexpr] input_dim: u32,
    #[constexpr] hidden: u32,
    #[constexpr] reverse: u32,
    #[constexpr] out_stride: u32,
    #[constexpr] out_offset: u32,
) {
    let sg = simd_id;
    let lane = simd_lane;
    let j = sg * 32u32 + lane;
    // Idle threads (j ≥ hidden) clamp their row index to 0 to stay in-bounds;
    // they compute garbage gates but never write, and still hit the barriers.
    let jj = select(j < hidden, j, 0u32);
    let active = j < hidden;
    // Gate rows for this unit (order i, f, g, o).
    let r_i = jj;
    let r_f = hidden + jj;
    let r_g = 2u32 * hidden + jj;
    let r_o = 3u32 * hidden + jj;

    // Fixed max allocation (threadgroup_alloc needs a literal size); only
    // the first `hidden` (≤ 1024) slots are used.
    threadgroup_alloc("h", 1024);
    threadgroup_alloc("c", 1024);
    if active {
        threadgroup_store("h", jj, 0.0f32);
        threadgroup_store("c", jj, 0.0f32);
    }
    threadgroup_barrier();

    for step in range(0u32, seq_len, 1u32) {
        // Forward walks t = step; backward walks t = seq_len-1-step.
        let t = select(reverse > 0u32, seq_len - 1u32 - step, step);
        let xb = t * input_dim;
        // ── Gate pre-activations: W_ih·x_t + W_hh·h_{t-1} + bias ──
        let mut gi = load(bias[r_i]);
        let mut gf = load(bias[r_f]);
        let mut gg = load(bias[r_g]);
        let mut go = load(bias[r_o]);
        for k in range(0u32, input_dim, 1u32) {
            let xk = load(x[xb + k]).cast::<f32>();
            gi = gi + load(w_ih[r_i * input_dim + k]).cast::<f32>() * xk;
            gf = gf + load(w_ih[r_f * input_dim + k]).cast::<f32>() * xk;
            gg = gg + load(w_ih[r_g * input_dim + k]).cast::<f32>() * xk;
            go = go + load(w_ih[r_o * input_dim + k]).cast::<f32>() * xk;
        }
        for m in range(0u32, hidden, 1u32) {
            let hm = threadgroup_load("h", m);
            gi = gi + load(w_hh[r_i * hidden + m]).cast::<f32>() * hm;
            gf = gf + load(w_hh[r_f * hidden + m]).cast::<f32>() * hm;
            gg = gg + load(w_hh[r_g * hidden + m]).cast::<f32>() * hm;
            go = go + load(w_hh[r_o * hidden + m]).cast::<f32>() * hm;
        }
        let c_old = threadgroup_load("c", jj);
        // Barrier: every unit has finished reading the previous `h` (and its
        // own `c`) before any unit overwrites the state below.
        threadgroup_barrier();
        // σ(x) = 1/(1+e^-x);  tanh(x) = 2/(1+e^-2x) − 1 (exact identity).
        let ig = 1.0f32 / (1.0f32 + exp(0.0f32 - gi));
        let fg = 1.0f32 / (1.0f32 + exp(0.0f32 - gf));
        let gt = 2.0f32 / (1.0f32 + exp(0.0f32 - 2.0f32 * gg)) - 1.0f32;
        let og = 1.0f32 / (1.0f32 + exp(0.0f32 - go));
        let c_new = fg * c_old + ig * gt;
        let tanh_c = 2.0f32 / (1.0f32 + exp(0.0f32 - 2.0f32 * c_new)) - 1.0f32;
        let h_new = og * tanh_c;
        if active {
            threadgroup_store("c", jj, c_new);
            threadgroup_store("h", jj, h_new);
            store(out[t * out_stride + out_offset + jj], h_new.cast::<T>());
        }
        // Barrier: new `h` is visible before the next step reads it.
        threadgroup_barrier();
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_lstm;
    use crate::utils::{pack_f32, unpack_f32};

    fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

    /// Reference single-direction LSTM (real libm tanh/sigmoid). Writes h_t
    /// into column `out_offset + j` of a `[seq_len, out_stride]` buffer.
    #[allow(clippy::too_many_arguments)]
    fn naive(
        x: &[f32],
        w_ih: &[f32],
        w_hh: &[f32],
        bias: &[f32],
        seq_len: usize,
        input_dim: usize,
        hidden: usize,
        reverse: bool,
        out_stride: usize,
        out_offset: usize,
        out: &mut [f32],
    ) {
        let mut h = vec![0.0f32; hidden];
        let mut c = vec![0.0f32; hidden];
        for step in 0..seq_len {
            let t = if reverse { seq_len - 1 - step } else { step };
            let mut hn = vec![0.0f32; hidden];
            let mut cn = vec![0.0f32; hidden];
            for j in 0..hidden {
                let rows = [j, hidden + j, 2 * hidden + j, 3 * hidden + j];
                let mut g = [bias[rows[0]], bias[rows[1]], bias[rows[2]], bias[rows[3]]];
                for (gi, &r) in g.iter_mut().zip(rows.iter()) {
                    for k in 0..input_dim {
                        *gi += w_ih[r * input_dim + k] * x[t * input_dim + k];
                    }
                    for m in 0..hidden {
                        *gi += w_hh[r * hidden + m] * h[m];
                    }
                }
                let i = sigmoid(g[0]);
                let f = sigmoid(g[1]);
                let gt = g[2].tanh();
                let o = sigmoid(g[3]);
                cn[j] = f * c[j] + i * gt;
                hn[j] = o * cn[j].tanh();
                out[t * out_stride + out_offset + j] = hn[j];
            }
            h = hn;
            c = cn;
        }
    }

    fn ramp(n: usize, step: f32, start: f32) -> Vec<f32> {
        (0..n).map(|i| ((start + i as f32 * step) % 2.0) - 1.0).collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn setup(
        dt: DType,
        seq_len: usize,
        input_dim: usize,
        hidden: usize,
        reverse: bool,
        tpg: u32,
    ) -> TestSetup {
        let x_f = ramp(seq_len * input_dim, 0.017, -0.5);
        let w_ih_f = ramp(4 * hidden * input_dim, 0.011, -0.4);
        let w_hh_f = ramp(4 * hidden * hidden, 0.009, -0.3);
        let bias: Vec<f32> = ramp(4 * hidden, 0.013, -0.2);
        let out_stride = hidden;
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);
        let w_ih_d = unpack_f32(&pack_f32(&w_ih_f, dt), dt);
        let w_hh_d = unpack_f32(&pack_f32(&w_hh_f, dt), dt);
        let mut expected = vec![0.0f32; seq_len * out_stride];
        naive(
            &x,
            &w_ih_d,
            &w_hh_d,
            &bias,
            seq_len,
            input_dim,
            hidden,
            reverse,
            out_stride,
            0,
            &mut expected,
        );
        TestSetup::new(ffai_lstm::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&x_f, dt), dt))
            .input(TestBuffer::from_vec("w_ih", pack_f32(&w_ih_f, dt), dt))
            .input(TestBuffer::from_vec("w_hh", pack_f32(&w_hh_f, dt), dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias, DType::F32), DType::F32))
            .input(TestBuffer::zeros("out", seq_len * out_stride, dt))
            .constexpr("seq_len", seq_len as u32)
            .constexpr("input_dim", input_dim as u32)
            .constexpr("hidden", hidden as u32)
            .constexpr("reverse", if reverse { 1u32 } else { 0u32 })
            .constexpr("out_stride", out_stride as u32)
            .constexpr("out_offset", 0u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(1, 1, 1, [tpg, 1, 1])
    }

    // Forward, hidden 4 (single simdgroup, 28 idle lanes).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [2e-3, 1e-2, 5e-2])]
    fn test_lstm_fwd(dt: DType) -> TestSetup { setup(dt, 8, 6, 4, false, 32) }

    // Backward pass (reverse time walk).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [2e-3, 1e-2, 5e-2])]
    fn test_lstm_bwd(dt: DType) -> TestSetup { setup(dt, 8, 6, 4, true, 32) }

    // Hidden 40 → two simdgroups (lanes 40..63 idle), exercises the
    // cross-simdgroup threadgroup recurrence.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [3e-3, 1e-2, 6e-2])]
    fn test_lstm_h40(dt: DType) -> TestSetup { setup(dt, 6, 8, 40, false, 64) }
}

/// New-syntax bench: Kokoro prosody-predictor LSTM (hidden 256, 200 frames).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_lstm;

    #[bench(name = "ffai/lstm/lstm", dtypes = [f32, f16, bf16])]
    fn bench_lstm(dt: DType) -> BenchSetup {
        let (seq_len, input_dim, hidden) = (200usize, 256usize, 256usize);
        let out_stride = hidden;
        BenchSetup::new(ffai_lstm::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", seq_len * input_dim, dt))
            .buffer(BenchBuffer::random("w_ih", 4 * hidden * input_dim, dt))
            .buffer(BenchBuffer::random("w_hh", 4 * hidden * hidden, dt))
            .buffer(BenchBuffer::random("bias", 4 * hidden, DType::F32))
            .buffer(BenchBuffer::zeros("out", seq_len * out_stride, dt).output())
            .constexpr("seq_len", seq_len as u32)
            .constexpr("input_dim", input_dim as u32)
            .constexpr("hidden", hidden as u32)
            .constexpr("reverse", 0u32)
            .constexpr("out_stride", out_stride as u32)
            .constexpr("out_offset", 0u32)
            .grid_3d(1, 1, 1, [256, 1, 1])
            .bytes_moved(
                ((seq_len * input_dim + 4 * hidden * (input_dim + hidden)) * dt.size_bytes())
                    as u64,
            )
    }
}
