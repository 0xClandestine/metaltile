//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! LSTM kernels — the recurrent building block style-vector TTS encoders /
//! prosody / duration predictors need (no FFAI model used an LSTM before).
//! This file holds two forms:
//!   * [`ffai_lstm`] — runs the **whole sequence** recurrence on the GPU in
//!     one dispatch (the "GPU from the start" form).
//!   * [`lstm_cell`] — ONE timestep, leaving the recurrence on the host (a
//!     per-step CPU↔GPU sync); use when per-step host control is wanted.
//!
//! ## `ffai_lstm`
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

#[kernel]
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
        // Gate activations via the DSL `sigmoid`/`tanh` intrinsics.
        let ig = sigmoid(gi);
        let fg = sigmoid(gf);
        let gt = tanh(gg);
        let og = sigmoid(go);
        let c_new = fg * c_old + ig * gt;
        let h_new = og * tanh(c_new);
        if active {
            threadgroup_store("c", jj, c_new);
            threadgroup_store("h", jj, h_new);
            store(out[t * out_stride + out_offset + jj], h_new.cast::<T>());
        }
        // Barrier: new `h` is visible before the next step reads it.
        threadgroup_barrier();
    }
}

/// One LSTM **timestep** across a whole batch (`grid_1d(batch*hidden, 256)`,
/// one thread per `(b, j)` unit) — the per-step form, leaving the recurrence
/// on the host (the host loops `t`, runs this forward `0..L` + backward
/// `L-1..0` with separate weights for a bidirectional layer). Use this when
/// per-step host control is wanted; use [`ffai_lstm`] above to run the whole
/// sequence on the GPU in one dispatch. Takes the `bias_ih` / `bias_hh`
/// split separately (vs. `ffai_lstm`'s precombined `bias`).
#[kernel]
pub fn lstm_cell<T>(
    x: Tensor<T>,
    h_prev: Tensor<T>,
    c_prev: Tensor<T>,
    weight_ih: Tensor<T>,
    weight_hh: Tensor<T>,
    bias_ih: Tensor<T>,
    bias_hh: Tensor<T>,
    mut h_out: Tensor<T>,
    mut c_out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] input_size: u32,
    #[constexpr] hidden: u32,
) {
    // One thread per (b, j) hidden unit.
    let idx = program_id::<0>();
    let j = idx % hidden;
    let b = idx / hidden;
    let x_base = b * input_size;
    let h_base = b * hidden;
    // Gate rows in the i|f|g|o stack.
    let row_i = j;
    let row_f = row_i + hidden;
    let row_g = row_f + hidden;
    let row_o = row_g + hidden;
    let w_ih_i = row_i * input_size;
    let w_ih_f = row_f * input_size;
    let w_ih_g = row_g * input_size;
    let w_ih_o = row_o * input_size;
    let w_hh_i = row_i * hidden;
    let w_hh_f = row_f * hidden;
    let w_hh_g = row_g * hidden;
    let w_hh_o = row_o * hidden;
    // Pre-activations seeded with both bias terms.
    let mut pi = load(bias_ih[row_i]).cast::<f32>() + load(bias_hh[row_i]).cast::<f32>();
    let mut pf = load(bias_ih[row_f]).cast::<f32>() + load(bias_hh[row_f]).cast::<f32>();
    let mut pg = load(bias_ih[row_g]).cast::<f32>() + load(bias_hh[row_g]).cast::<f32>();
    let mut po = load(bias_ih[row_o]).cast::<f32>() + load(bias_hh[row_o]).cast::<f32>();
    // Input projection: W_ih · x.
    for ii in range(0u32, input_size, 1u32) {
        let xv = load(x[x_base + ii]).cast::<f32>();
        pi = pi + xv * load(weight_ih[w_ih_i + ii]).cast::<f32>();
        pf = pf + xv * load(weight_ih[w_ih_f + ii]).cast::<f32>();
        pg = pg + xv * load(weight_ih[w_ih_g + ii]).cast::<f32>();
        po = po + xv * load(weight_ih[w_ih_o + ii]).cast::<f32>();
    }
    // Recurrent projection: W_hh · h_prev.
    for kk in range(0u32, hidden, 1u32) {
        let hv = load(h_prev[h_base + kk]).cast::<f32>();
        pi = pi + hv * load(weight_hh[w_hh_i + kk]).cast::<f32>();
        pf = pf + hv * load(weight_hh[w_hh_f + kk]).cast::<f32>();
        pg = pg + hv * load(weight_hh[w_hh_g + kk]).cast::<f32>();
        po = po + hv * load(weight_hh[w_hh_o + kk]).cast::<f32>();
    }
    let ig = sigmoid(pi);
    let fg = sigmoid(pf);
    let gg = tanh(pg);
    let og = sigmoid(po);
    let c_new = fg * load(c_prev[idx]).cast::<f32>() + ig * gg;
    let h_new = og * tanh(c_new);
    // Bounds guard: `grid_1d` rounds up to whole threadgroups, so threads
    // with `idx >= batch*hidden` are launched but must not store — with TWO
    // output buffers an unguarded OOB `h_out[idx]` write lands in the
    // adjacent `c_out` allocation and corrupts valid entries.
    if idx < batch * hidden {
        store(c_out[idx], c_new.cast::<T>());
        store(h_out[idx], h_new.cast::<T>());
    }
}

pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::{ffai_lstm, lstm_cell};
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

    // The real Kokoro `predictor.shared` shape: hidden 256 = 8 full
    // simdgroups, a long (65-step) sequence, wide input (640). Exercises the
    // cross-simdgroup threadgroup hand-off over many timesteps that the small
    // tests never reach. f32 only — f16/bf16 drift over 65 recurrent steps
    // exceeds a useful tolerance.
    #[test_kernel(dtypes = [f32], tol = [2e-3])]
    fn test_lstm_long_h256(dt: DType) -> TestSetup { setup(dt, 65, 640, 256, false, 256) }

    // ── Per-step lstm_cell tests (gate order i, f, g, o) ──
    fn ramp_p(n: usize, period: usize, amp: f32, start: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp + start).collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn naive_lstm_cell(
        x: &[f32],
        h_prev: &[f32],
        c_prev: &[f32],
        weight_ih: &[f32],
        weight_hh: &[f32],
        bias_ih: &[f32],
        bias_hh: &[f32],
        batch: usize,
        input_size: usize,
        hidden: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut h_out = vec![0.0f32; batch * hidden];
        let mut c_out = vec![0.0f32; batch * hidden];
        for b in 0..batch {
            for j in 0..hidden {
                let mut pre = [0.0f32; 4];
                for (g, p) in pre.iter_mut().enumerate() {
                    let row = g * hidden + j;
                    let mut acc = bias_ih[row] + bias_hh[row];
                    for ii in 0..input_size {
                        acc += x[b * input_size + ii] * weight_ih[row * input_size + ii];
                    }
                    for kk in 0..hidden {
                        acc += h_prev[b * hidden + kk] * weight_hh[row * hidden + kk];
                    }
                    *p = acc;
                }
                let ig = sigmoid(pre[0]);
                let fg = sigmoid(pre[1]);
                let gg = pre[2].tanh();
                let og = sigmoid(pre[3]);
                let c_new = fg * c_prev[b * hidden + j] + ig * gg;
                let h_new = og * c_new.tanh();
                c_out[b * hidden + j] = c_new;
                h_out[b * hidden + j] = h_new;
            }
        }
        (h_out, c_out)
    }

    fn lstm_cell_setup(
        kernel: Kernel,
        batch: usize,
        input_size: usize,
        hidden: usize,
        dt: DType,
    ) -> TestSetup {
        let x_f = ramp_p(batch * input_size, 13, 1.2, 0.0);
        let h_f = ramp_p(batch * hidden, 17, 0.8, 0.1);
        let c_f = ramp_p(batch * hidden, 11, 0.6, -0.1);
        let w_ih = ramp_p(4 * hidden * input_size, 23, 0.3, 0.0);
        let w_hh = ramp_p(4 * hidden * hidden, 29, 0.25, 0.0);
        let b_ih = ramp_p(4 * hidden, 7, 0.2, 0.0);
        let b_hh = ramp_p(4 * hidden, 5, 0.15, 0.0);
        let r = |v: &[f32]| unpack_f32(&pack_f32(v, dt), dt);
        let (h_exp, c_exp) = naive_lstm_cell(
            &r(&x_f),
            &r(&h_f),
            &r(&c_f),
            &r(&w_ih),
            &r(&w_hh),
            &r(&b_ih),
            &r(&b_hh),
            batch,
            input_size,
            hidden,
        );
        TestSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("x", pack_f32(&x_f, dt), dt))
            .input(TestBuffer::from_vec("h_prev", pack_f32(&h_f, dt), dt))
            .input(TestBuffer::from_vec("c_prev", pack_f32(&c_f, dt), dt))
            .input(TestBuffer::from_vec("weight_ih", pack_f32(&w_ih, dt), dt))
            .input(TestBuffer::from_vec("weight_hh", pack_f32(&w_hh, dt), dt))
            .input(TestBuffer::from_vec("bias_ih", pack_f32(&b_ih, dt), dt))
            .input(TestBuffer::from_vec("bias_hh", pack_f32(&b_hh, dt), dt))
            .input(TestBuffer::zeros("h_out", batch * hidden, dt))
            .input(TestBuffer::zeros("c_out", batch * hidden, dt))
            .constexpr("batch", batch as u32)
            .constexpr("input_size", input_size as u32)
            .constexpr("hidden", hidden as u32)
            .expect(TestBuffer::from_vec("h_out", pack_f32(&h_exp, dt), dt))
            .expect(TestBuffer::from_vec("c_out", pack_f32(&c_exp, dt), dt))
            .grid_1d(batch * hidden, 256)
    }

    // Text-encoder BiLSTM cell shape (per direction).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [2e-3, 2e-2, 1e-1])]
    fn test_lstm_cell(dt: DType) -> TestSetup {
        lstm_cell_setup(lstm_cell::kernel_ir_for(dt), 2, 128, 64, dt)
    }

    // Equal input/hidden size (prosody predictor LSTM).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [2e-3, 2e-2, 1e-1])]
    fn test_lstm_cell_square(dt: DType) -> TestSetup {
        lstm_cell_setup(lstm_cell::kernel_ir_for(dt), 3, 96, 96, dt)
    }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{ffai_lstm, lstm_cell};

    #[bench(dtypes = [f32, f16, bf16])]
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
            // 4 gates × (W_ih·x + W_hh·h) MACs per step, over the sequence:
            // 8·seq_len·hidden·(input_dim + hidden).
            .flops(
                8 * (seq_len as u64) * (hidden as u64) * (input_dim as u64 + hidden as u64),
            )
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_lstm_cell(dt: DType) -> BenchSetup {
        // BiLSTM cell: input 512, hidden 256, batch 8.
        let (batch, input_size, hidden) = (8usize, 512usize, 256usize);
        let n_out = batch * hidden;
        let bytes = (batch * input_size
            + 2 * batch * hidden
            + 4 * hidden * input_size
            + 4 * hidden * hidden)
            * dt.size_bytes();
        BenchSetup::new(lstm_cell::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("x", batch * input_size, dt))
            .buffer(BenchBuffer::random("h_prev", batch * hidden, dt))
            .buffer(BenchBuffer::random("c_prev", batch * hidden, dt))
            .buffer(BenchBuffer::random("weight_ih", 4 * hidden * input_size, dt))
            .buffer(BenchBuffer::random("weight_hh", 4 * hidden * hidden, dt))
            .buffer(BenchBuffer::random("bias_ih", 4 * hidden, dt))
            .buffer(BenchBuffer::random("bias_hh", 4 * hidden, dt))
            .buffer(BenchBuffer::zeros("h_out", n_out, dt).output())
            .buffer(BenchBuffer::zeros("c_out", n_out, dt).output())
            .constexpr("batch", batch as u32)
            .constexpr("input_size", input_size as u32)
            .constexpr("hidden", hidden as u32)
            .grid_1d(n_out, 256)
            .bytes_moved(bytes as u64)
            // 4 gates × (W_ih·x + W_hh·h) MACs per batch element:
            // 8·batch·hidden·(input_size + hidden).
            .flops(
                8 * (batch as u64) * (hidden as u64) * (input_size as u64 + hidden as u64),
            )
    }
}
