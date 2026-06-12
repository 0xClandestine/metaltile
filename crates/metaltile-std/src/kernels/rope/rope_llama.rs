//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Llama-style RoPE with optional Llama-3 frequency-band scaling, generic over
//! T. One kernel serves both decode and prefill: position comes from a per-row
//! `positions: Tensor<u32>` and an outer grid axis selects the row, so decode
//! is just the `T = 1` case (a 1-element `positions` buffer, `z = 1` grid) and
//! prefill ropes all T tokens in ONE dispatch. (The old single-token
//! `mt_rope_llama` kernel was redundant — `#[constexpr]` is already a runtime
//! buffer read, so the scalar-position variant bought nothing over `T = 1`.)
//!
//! Different from `mt_rope` (in `kernels/rope/base.rs`):
//!   - generic dtype (mt_rope is f16-only)
//!   - supports Llama-3 wavelength banding (low / high / smoothed)
//!
//! For each (head, i in 0..head_dim/2):
//!
//!   base inv_freq = 1 / theta_base^(2i / head_dim)
//!   wavelen       = 2*pi / inv_freq
//!   if wavelen > low_freq_wavelen:        inv_freq /= scale_factor      (low-freq band)
//!   else if wavelen < high_freq_wavelen:  inv_freq                       (high-freq band)
//!   else (medium band):                   smoothed interpolation
//!
//! To turn scaling OFF, pass scale_factor=1, low_freq_factor=1,
//! high_freq_factor=1, original_max_position=very_large (e.g. 1e9).
//!
//! Layout: `qk`/`out` are `[T, n_heads, head_dim]`, `positions` is `[T]` (u32).
//! Grid3D: one thread per `(row, head, i)`, `i ∈ [0, half_dim)`.
//!
//! `row_stride` is the element stride between consecutive rows of `qk`/`out`
//! (`n_heads_q * head_dim` for Q, `n_heads_k * head_dim` for K) — supplied per
//! dispatch so one kernel handles both Q and K with their own KV-grouped head
//! counts. The paired `ropePartialTwo` semantics are intentionally NOT fused;
//! the host wraps two dispatches on a shared command encoder instead.
//!
//! Codegen-only. Validated end-to-end in FFAI integration tests.

use metaltile::kernel;

#[kernel]
pub fn mt_rope_llama<T>(
    qk: Tensor<T>,
    positions: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] half_dim: u32,
    #[constexpr] row_stride: u32,
    #[constexpr] theta_base: f32,
    #[constexpr] scale_factor: f32,
    #[constexpr] low_freq_factor: f32,
    #[constexpr] high_freq_factor: f32,
    #[constexpr] original_max_position: f32,
) {
    let r = program_id::<0>();
    let head = program_id::<1>();
    let i = program_id::<2>();
    // Per-row position lookup. Each row gets its own scalar `position` taken
    // from the positions vector (decode passes a 1-element vector).
    let position = load(positions[r]);
    let i_f = i.cast::<f32>();
    let half_f = half_dim.cast::<f32>();
    let inv_freq_base = exp2(-i_f * log2(theta_base) / half_f);
    let two_pi = 6.283185307179586f32;
    let wavelen = two_pi / inv_freq_base;
    let low_freq_wavelen = original_max_position / low_freq_factor;
    let high_freq_wavelen = original_max_position / high_freq_factor;
    let scaled = inv_freq_base / scale_factor;
    let smooth_num = original_max_position / wavelen - low_freq_factor;
    let smooth_den = high_freq_factor - low_freq_factor;
    let s = smooth_num / smooth_den;
    let smoothed = (1.0f32 - s) * scaled + s * inv_freq_base;
    let is_low_freq = wavelen > low_freq_wavelen;
    let is_high_freq = wavelen < high_freq_wavelen;
    let inv_freq = select(is_low_freq, scaled, select(is_high_freq, inv_freq_base, smoothed));
    let pos_f = position.cast::<f32>();
    let theta = pos_f * inv_freq;
    let cos_t = cos(theta);
    let sin_t = sin(theta);
    // Row-strided base offset. row_stride may exceed n_heads * head_dim if the
    // caller slices a wider packed tensor (fused QKV) — kept a separate
    // constexpr so the kernel never needs to know n_heads.
    let base = r * row_stride + head * head_dim;
    let i1 = base + i;
    let i2 = base + i + half_dim;
    let x1 = load(qk[i1]).cast::<f32>();
    let x2 = load(qk[i2]).cast::<f32>();
    let o1 = x1 * cos_t - x2 * sin_t;
    let o2 = x1 * sin_t + x2 * cos_t;
    store(out[i1], o1.cast::<T>());
    store(out[i2], o2.cast::<T>());
}

/// New-syntax correctness for `mt_rope_llama`. Grid3D `[T, n_heads, half_dim]`,
/// tpg `[1,1,1]` (one thread per (row, head, i); each writes the rotation pair
/// i / i+half_dim). Oracle replays the exact banded-inv_freq + rotation math in
/// f32, per row at that row's position. Covers decode (T=1) and prefill (T>1),
/// banding off and on.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::mt_rope_llama;
    use crate::utils::{pack_f32, unpack_f32};

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    /// Llama-3 banded inverse frequency for pair index `i` (mirrors the kernel).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn band_inv_freq(
        i: usize,
        half: usize,
        theta_base: f32,
        scale_factor: f32,
        low_ff: f32,
        high_ff: f32,
        orig_max: f32,
    ) -> f32 {
        let inv_base = (-(i as f32) * theta_base.log2() / half as f32).exp2();
        let two_pi = std::f32::consts::TAU;
        let wavelen = two_pi / inv_base;
        let low_wl = orig_max / low_ff;
        let high_wl = orig_max / high_ff;
        let scaled = inv_base / scale_factor;
        let s = (orig_max / wavelen - low_ff) / (high_ff - low_ff);
        let smoothed = (1.0 - s) * scaled + s * inv_base;
        if wavelen > low_wl {
            scaled
        } else if wavelen < high_wl {
            inv_base
        } else {
            smoothed
        }
    }

    /// Builds a `T`-row test. `positions[r] = 100 + 7*r` (nonzero, distinct per
    /// row so per-row indexing is exercised). Banding is selected by the
    /// frequency index, not the position, so the small positions keep the
    /// cos/sin precision tight while still hitting all three bands.
    #[allow(clippy::too_many_arguments)]
    fn rope_llama_setup(dt: DType, t_len: usize, sf: f32, lf: f32, hf: f32, omp: f32) -> TestSetup {
        let (n_heads, head_dim) = (4usize, 64usize);
        let half = head_dim / 2;
        let row_stride = n_heads * head_dim; // contiguous (no fused-QKV slack)
        let theta_base = 500_000.0f32;
        let positions: Vec<u32> = (0..t_len).map(|r| (100 + 7 * r) as u32).collect();
        let qk_f: Vec<f32> =
            (0..t_len * row_stride).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let qk = unpack_f32(&pack_f32(&qk_f, dt), dt);
        let mut exp = vec![0.0f32; t_len * row_stride];
        for (r, &posu) in positions.iter().enumerate() {
            let pos = posu as f32;
            for h in 0..n_heads {
                for i in 0..half {
                    let invf = band_inv_freq(i, half, theta_base, sf, lf, hf, omp);
                    let th = pos * invf;
                    let (c, s) = (th.cos(), th.sin());
                    let base = r * row_stride + h * head_dim;
                    let (x1, x2) = (qk[base + i], qk[base + i + half]);
                    exp[base + i] = x1 * c - x2 * s;
                    exp[base + i + half] = x1 * s + x2 * c;
                }
            }
        }
        TestSetup::new(mt_rope_llama::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("qk", pack_f32(&qk_f, dt), dt))
            .input(TestBuffer::from_vec("positions", u32_bytes(&positions), DType::U32))
            .input(TestBuffer::zeros("out", t_len * row_stride, dt))
            .constexpr("head_dim", head_dim as u32)
            .constexpr("half_dim", half as u32)
            .constexpr("row_stride", row_stride as u32)
            .constexpr("theta_base", theta_base)
            .constexpr("scale_factor", sf)
            .constexpr("low_freq_factor", lf)
            .constexpr("high_freq_factor", hf)
            .constexpr("original_max_position", omp)
            .expect(TestBuffer::from_vec("out", pack_f32(&exp, dt), dt))
            .grid_3d(t_len as u32, n_heads as u32, half as u32, [1, 1, 1])
    }

    // Decode (T=1): single-token degenerate case, banding OFF.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-2, 5e-2])]
    fn test_rope_llama_decode(dt: DType) -> TestSetup {
        rope_llama_setup(dt, 1, 1.0, 1.0, 1.0, 1.0e9)
    }

    // Prefill (T=4): banding OFF → high-freq branch for every pair.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-2, 5e-2])]
    fn test_rope_llama(dt: DType) -> TestSetup { rope_llama_setup(dt, 4, 1.0, 1.0, 1.0, 1.0e9) }

    // Prefill (T=4) with Llama-3 frequency-band scaling ON (scale 8, low/high
    // factors 1/4, original_max 8192) — exercises all three bands (low-freq
    // scaled, high-freq unscaled, smoothed-medium), selected by frequency index.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-2, 5e-2])]
    fn test_rope_llama_banding(dt: DType) -> TestSetup {
        rope_llama_setup(dt, 4, 8.0, 1.0, 4.0, 8192.0)
    }
}

/// New-syntax benchmarks for `mt_rope_llama`: decode (T=1) and prefill (T=512).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::mt_rope_llama;

    /// Builds a `T`-row bench at the given head geometry (Llama-3 banding on).
    fn rope_llama_bench(dt: DType, t_len: usize, n_heads: usize, head_dim: usize) -> BenchSetup {
        let half = head_dim / 2;
        let row_stride = n_heads * head_dim;
        BenchSetup::new(mt_rope_llama::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("qk", t_len * row_stride, dt))
            .buffer(BenchBuffer::random("positions", t_len, DType::U32))
            .buffer(BenchBuffer::zeros("out", t_len * row_stride, dt).output())
            .constexpr("head_dim", head_dim as u32)
            .constexpr("half_dim", half as u32)
            .constexpr("row_stride", row_stride as u32)
            .constexpr("theta_base", 500_000.0f32)
            .constexpr("scale_factor", 8.0f32)
            .constexpr("low_freq_factor", 1.0f32)
            .constexpr("high_freq_factor", 4.0f32)
            .constexpr("original_max_position", 8192.0f32)
            .with_shape_label(format!(
                "T{t_len} h{n_heads} d{head_dim} {}",
                crate::utils::dtype_label(dt)
            ))
            .grid_3d(t_len as u32, n_heads as u32, half as u32, [1, 1, 1])
            .bytes_moved((2 * t_len * row_stride * dt.size_bytes()) as u64)
    }

    /// Decode shape — one token, head_dim 128.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_rope_llama_decode(dt: DType) -> BenchSetup { rope_llama_bench(dt, 1, 32, 128) }

    /// Prefill shape — 512 tokens in one dispatch.
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_rope_llama(dt: DType) -> BenchSetup { rope_llama_bench(dt, 512, 32, 128) }
}
