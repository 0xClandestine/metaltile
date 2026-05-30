//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! DSv4 Manifold-Constrained Hyper-Connections (mHC) — runtime kernels.
//!
//! mHC carries a per-token `n_hc=4`-channel residual state
//! `H[hidden, n_hc, n_tokens]`. At every sub-block boundary the state
//! is collapsed to a single-channel input via per-token `pre[n_hc]`
//! weights, the sub-block runs, then the output is expanded back into
//! the 4 channels via per-token `post[n_hc]` weights + a per-token
//! `comb[n_hc, n_hc]` (Sinkhorn-normalized) residual remix:
//!
//! ```text
//!   collapse:  x[d, t]              = sum_c pre[t, c] * H[d, c, t]
//!   expand:    H_new[d, dst, t]     = block_out[d, t] * post[t, dst]
//!                                   + sum_src comb[t, dst, src] * residual_H[d, src, t]
//! ```
//!
//! `pre`, `post`, and `comb` are DYNAMIC per-token tensors produced
//! by [`crate::ffai::dsv4_mhc_sinkhorn_split`] from the `hc_*_fn @
//! flatten(H)` 24-mix output. They are NOT stored model weights.
//!
//! ## Dispatch
//!
//! 1D grid over `hidden_dim`. Each thread loops over `n_tokens` and
//! the small `n_hc=4` inner — fine for decode (n_tokens=1) and OK
//! for short prefill chunks. Long-prefill rewrites would split the
//! grid over `(d, t)`.

use metaltile::kernel;

/// mHC collapse — `x[d, t] = sum_c pre[t, c] * H[d, c, t]`.
#[kernel]
pub fn ffai_dsv4_mhc_collapse<T>(
    state: Tensor<T>,
    pre: Tensor<f32>,
    mut out: Tensor<T>,
    #[constexpr] hidden_dim: u32,
    #[constexpr] n_hc: u32,
    #[constexpr] n_tokens: u32,
) {
    let d = tid;
    if d < hidden_dim {
        for _t in range(0u32, n_tokens, 1u32) {
            let state_token_base = _t * n_hc * hidden_dim;
            let pre_token_base = _t * n_hc;
            let mut acc = 0.0f32;
            for _c in range(0u32, n_hc, 1u32) {
                let w = load(pre[pre_token_base + _c]);
                let h = load(state[state_token_base + _c * hidden_dim + d]).cast::<f32>();
                acc = acc + w * h;
            }
            store(out[_t * hidden_dim + d], acc);
        }
    }
}

/// mHC expand — channel-wise residual remix:
///   `H_new[d, dst, t] = block_out[d, t] * post[t, dst]
///                     + sum_src comb[t, dst, src] * residual_H[d, src, t]`
///
/// Reads from `residual_state` and writes to `state` — caller passes
/// the per-channel pre-update residual in `residual_state` and an
/// allocated output buffer in `state`. `comb` layout matches the
/// split kernel output: `comb[t * n_hc * n_hc + dst * n_hc + src]`.
#[kernel]
pub fn ffai_dsv4_mhc_expand<T>(
    block_out: Tensor<T>,
    post: Tensor<f32>,
    comb: Tensor<f32>,
    residual_state: Tensor<T>,
    mut state: Tensor<T>,
    #[constexpr] hidden_dim: u32,
    #[constexpr] n_hc: u32,
    #[constexpr] n_tokens: u32,
) {
    let d = tid;
    if d < hidden_dim {
        for _t in range(0u32, n_tokens, 1u32) {
            let state_token_base = _t * n_hc * hidden_dim;
            let post_token_base = _t * n_hc;
            let comb_token_base = _t * n_hc * n_hc;
            let block_out_val = load(block_out[_t * hidden_dim + d]).cast::<f32>();
            for _dst in range(0u32, n_hc, 1u32) {
                let post_w = load(post[post_token_base + _dst]);
                let mut acc = block_out_val * post_w;
                for _src in range(0u32, n_hc, 1u32) {
                    let comb_w = load(comb[comb_token_base + _dst * n_hc + _src]);
                    let resid = load(residual_state[state_token_base + _src * hidden_dim + d])
                        .cast::<f32>();
                    acc = acc + comb_w * resid;
                }
                store(state[state_token_base + _dst * hidden_dim + d], acc);
            }
        }
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::{ffai_dsv4_mhc_collapse, ffai_dsv4_mhc_expand};
    use crate::utils::{pack_f32, unpack_f32};

    fn cpu_collapse(
        state: &[f32],
        pre: &[f32],
        hidden_dim: usize,
        n_hc: usize,
        n_tokens: usize,
    ) -> Vec<f32> {
        let mut out = vec![0f32; n_tokens * hidden_dim];
        for t in 0..n_tokens {
            for d in 0..hidden_dim {
                let mut acc = 0f32;
                for c in 0..n_hc {
                    acc += pre[t * n_hc + c] * state[t * n_hc * hidden_dim + c * hidden_dim + d];
                }
                out[t * hidden_dim + d] = acc;
            }
        }
        out
    }

    fn setup_collapse(hidden_dim: usize, n_hc: usize, n_tokens: usize, dt: DType) -> TestSetup {
        let state: Vec<f32> = (0..n_tokens * n_hc * hidden_dim)
            .map(|i| (i as f32 * 0.011 - 0.7).sin() * 1.4)
            .collect();
        let pre: Vec<f32> = (0..n_tokens * n_hc).map(|i| (i as f32 - 4.0) * 0.2 + 0.5).collect();
        let state_dt = unpack_f32(&pack_f32(&state, dt), dt);
        let expected = cpu_collapse(&state_dt, &pre, hidden_dim, n_hc, n_tokens);
        TestSetup::new(ffai_dsv4_mhc_collapse::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("state", pack_f32(&state, dt), dt))
            .input(TestBuffer::from_vec("pre", pack_f32(&pre, DType::F32), DType::F32))
            .input(TestBuffer::zeros("out", n_tokens * hidden_dim, dt))
            .constexpr("hidden_dim", hidden_dim as u32)
            .constexpr("n_hc", n_hc as u32)
            .constexpr("n_tokens", n_tokens as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(hidden_dim, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-5, 5e-3, 5e-2])]
    fn test_mhc_collapse_decode(dt: DType) -> TestSetup { setup_collapse(4096, 4, 1, dt) }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-5, 5e-3, 5e-2])]
    fn test_mhc_collapse_batch(dt: DType) -> TestSetup { setup_collapse(512, 4, 4, dt) }

    fn cpu_expand(
        block_out: &[f32],
        post: &[f32],
        comb: &[f32],
        residual: &[f32],
        hidden_dim: usize,
        n_hc: usize,
        n_tokens: usize,
    ) -> Vec<f32> {
        let mut out = vec![0f32; n_tokens * n_hc * hidden_dim];
        for t in 0..n_tokens {
            for d in 0..hidden_dim {
                let bo = block_out[t * hidden_dim + d];
                for dst in 0..n_hc {
                    let mut acc = bo * post[t * n_hc + dst];
                    for src in 0..n_hc {
                        acc += comb[t * n_hc * n_hc + dst * n_hc + src]
                            * residual[t * n_hc * hidden_dim + src * hidden_dim + d];
                    }
                    out[t * n_hc * hidden_dim + dst * hidden_dim + d] = acc;
                }
            }
        }
        out
    }

    fn setup_expand(hidden_dim: usize, n_hc: usize, n_tokens: usize, dt: DType) -> TestSetup {
        let block_out: Vec<f32> =
            (0..n_tokens * hidden_dim).map(|i| (i as f32 * 0.017 - 1.2).sin() * 0.7).collect();
        let post: Vec<f32> = (0..n_tokens * n_hc).map(|i| (i as f32 - 2.0) * 0.3 + 0.4).collect();
        let comb: Vec<f32> =
            (0..n_tokens * n_hc * n_hc).map(|i| (i as f32 * 0.05 - 0.1).cos() * 0.25).collect();
        let residual: Vec<f32> = (0..n_tokens * n_hc * hidden_dim)
            .map(|i| (i as f32 * 0.0083 - 0.1).cos() * 0.9)
            .collect();
        let block_out_dt = unpack_f32(&pack_f32(&block_out, dt), dt);
        let residual_dt = unpack_f32(&pack_f32(&residual, dt), dt);
        let expected =
            cpu_expand(&block_out_dt, &post, &comb, &residual_dt, hidden_dim, n_hc, n_tokens);
        TestSetup::new(ffai_dsv4_mhc_expand::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("block_out", pack_f32(&block_out, dt), dt))
            .input(TestBuffer::from_vec("post", pack_f32(&post, DType::F32), DType::F32))
            .input(TestBuffer::from_vec("comb", pack_f32(&comb, DType::F32), DType::F32))
            .input(TestBuffer::from_vec("residual_state", pack_f32(&residual, dt), dt))
            .input(TestBuffer::zeros("state", n_tokens * n_hc * hidden_dim, dt))
            .constexpr("hidden_dim", hidden_dim as u32)
            .constexpr("n_hc", n_hc as u32)
            .constexpr("n_tokens", n_tokens as u32)
            .expect(TestBuffer::from_vec("state", pack_f32(&expected, dt), dt))
            .grid_1d(hidden_dim, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-5, 5e-3, 5e-2])]
    fn test_mhc_expand_decode(dt: DType) -> TestSetup { setup_expand(4096, 4, 1, dt) }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-5, 5e-3, 5e-2])]
    fn test_mhc_expand_batch(dt: DType) -> TestSetup { setup_expand(512, 4, 4, dt) }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{ffai_dsv4_mhc_collapse, ffai_dsv4_mhc_expand};

    #[bench(name = "ffai/dsv4_mhc_collapse", dtypes = [f32, f16, bf16])]
    fn bench_collapse(dt: DType) -> BenchSetup {
        let (hidden_dim, n_hc, n_tokens) = (4096usize, 4usize, 1usize);
        BenchSetup::new(ffai_dsv4_mhc_collapse::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("state", n_tokens * n_hc * hidden_dim, dt))
            .buffer(BenchBuffer::random("pre", n_tokens * n_hc, DType::F32))
            .buffer(BenchBuffer::zeros("out", n_tokens * hidden_dim, dt).output())
            .constexpr("hidden_dim", hidden_dim as u32)
            .constexpr("n_hc", n_hc as u32)
            .constexpr("n_tokens", n_tokens as u32)
            .grid_1d(hidden_dim, 256)
            .bytes_moved(
                ((n_tokens * n_hc * hidden_dim + n_tokens * n_hc) * dt.size_bytes()
                    + n_tokens * hidden_dim * dt.size_bytes()) as u64,
            )
    }

    #[bench(name = "ffai/dsv4_mhc_expand", dtypes = [f32, f16, bf16])]
    fn bench_expand(dt: DType) -> BenchSetup {
        let (hidden_dim, n_hc, n_tokens) = (4096usize, 4usize, 1usize);
        BenchSetup::new(ffai_dsv4_mhc_expand::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("block_out", n_tokens * hidden_dim, dt))
            .buffer(BenchBuffer::random("post", n_tokens * n_hc, DType::F32))
            .buffer(BenchBuffer::random("comb", n_tokens * n_hc * n_hc, DType::F32))
            .buffer(BenchBuffer::random("residual_state", n_tokens * n_hc * hidden_dim, dt))
            .buffer(BenchBuffer::zeros("state", n_tokens * n_hc * hidden_dim, dt).output())
            .constexpr("hidden_dim", hidden_dim as u32)
            .constexpr("n_hc", n_hc as u32)
            .constexpr("n_tokens", n_tokens as u32)
            .grid_1d(hidden_dim, 256)
            .bytes_moved(
                ((2 * n_tokens * n_hc * hidden_dim + n_tokens * hidden_dim) * dt.size_bytes()
                    + n_tokens * (n_hc + n_hc * n_hc) * 4) as u64,
            )
    }
}
