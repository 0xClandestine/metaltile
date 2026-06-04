//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! BATCHED DSv4 partial RoPE — the per-token ffai_dsv4_partial_rope applied
//! to N tokens in ONE dispatch (grid z = token). Prefill ropes token t at
//! absolute position `base_position + t`; the old prefill looped this N times
//! (3N tiny dispatches/layer for q + kv + inverse — the #1 warm-prefill cost).
//! Identical math to ffai_dsv4_partial_rope (adjacent GPT-J pairs, YaRN).
//!
//! qk/out are [n_tokens, n_heads, head_dim]. Grid3D [n_heads, half_rot,
//! n_tokens], one thread per (head, rot-pair, token).

use metaltile::kernel;

#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ffai_dsv4_partial_rope_rows<T>(
    qk: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] n_nope: u32,
    #[constexpr] half_rot: u32,
    #[constexpr] n_heads: u32,
    #[constexpr] base_position: u32,
    #[constexpr] theta_base: f32,
    #[constexpr] inverse_flag: u32,
    #[constexpr] freq_scale: f32,
    #[constexpr] ext_factor: f32,
    #[constexpr] corr_low: f32,
    #[constexpr] corr_high: f32,
) {
    let head = program_id::<0>();
    let pair_idx = program_id::<1>();
    let token = program_id::<2>();
    let pair_f = pair_idx.cast::<f32>();
    let n_rot_f = (2u32 * half_rot).cast::<f32>();
    let inv_freq = exp2(0.0f32 - 2.0f32 * pair_f * log2(theta_base) / n_rot_f);
    let pos_f = (base_position + token).cast::<f32>();
    let theta_extrap = pos_f * inv_freq;
    let theta_interp = freq_scale * theta_extrap;
    let denom = select((corr_high - corr_low) > 0.001f32, corr_high - corr_low, 0.001f32);
    let y_raw = (pair_f - corr_low) / denom;
    let y_lo = select(y_raw > 0.0f32, y_raw, 0.0f32);
    let y_cl = select(y_lo < 1.0f32, y_lo, 1.0f32);
    let ramp_mix = (1.0f32 - y_cl) * ext_factor;
    let theta_raw = theta_interp * (1.0f32 - ramp_mix) + theta_extrap * ramp_mix;
    let theta_signed = select(inverse_flag == 0u32, theta_raw, 0.0f32 - theta_raw);
    let cos_t = cos(theta_signed);
    let sin_t = sin(theta_signed);
    let tok_base = token * n_heads * head_dim;
    let head_base = tok_base + head * head_dim;
    let dim_lo = head_base + n_nope + 2u32 * pair_idx;
    let dim_hi = head_base + n_nope + 2u32 * pair_idx + 1u32;
    let x_lo = load(qk[dim_lo]).cast::<f32>();
    let x_hi = load(qk[dim_hi]).cast::<f32>();
    let o_lo = x_lo * cos_t - x_hi * sin_t;
    let o_hi = x_lo * sin_t + x_hi * cos_t;
    store(out[dim_lo], o_lo.cast::<T>());
    store(out[dim_hi], o_hi.cast::<T>());
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_dsv4_partial_rope_rows;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_rope_rows(dt: DType) -> BenchSetup {
        let (n_heads, head_dim, n_nope, half_rot, n_tokens) =
            (64usize, 512usize, 448usize, 32usize, 256usize);
        BenchSetup::new(ffai_dsv4_partial_rope_rows::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("qk", n_tokens * n_heads * head_dim, dt))
            .buffer(BenchBuffer::zeros("out", n_tokens * n_heads * head_dim, dt).output())
            .constexpr("head_dim", head_dim as u32)
            .constexpr("n_nope", n_nope as u32)
            .constexpr("half_rot", half_rot as u32)
            .constexpr("n_heads", n_heads as u32)
            .constexpr("base_position", 0u32)
            .constexpr("theta_base", 10_000.0_f32)
            .constexpr("inverse_flag", 0u32)
            .constexpr("freq_scale", 1.0f32)
            .constexpr("ext_factor", 0.0f32)
            .constexpr("corr_low", 0.0f32)
            .constexpr("corr_high", 0.0f32)
            .grid_3d(n_heads as u32, half_rot as u32, n_tokens as u32, [1, 1, 1])
            .bytes_moved((4 * n_tokens * n_heads * half_rot * dt.size_bytes()) as u64)
    }
}
