//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Fused RMSNorm + RoPE — normalizes a Q/K head then applies the
//! rotary position embedding, in one dispatch. Saves a kernel launch
//! per Q and per K vs separate `rms_norm` + `rope` calls (the
//! post-projection q_norm/k_norm path in Qwen3-style models).
//!
//! Non-traditional (paired) RoPE layout: element `i` rotates with
//! element `i + half`. One threadgroup per row — a row is one
//! `(batch, seq_pos, head)` slice of length `axis_size`; thread `lid`
//! owns the pair `(lid, lid + half)`.
//!
//! Phase 1 — `inv_rms = rsqrt(mean(x²) + eps)` via `mt_rms_inv_scalar`
//! cross-kernel call with `partial_ssq = v1² + v2²` as the Value arg.
//! Phase 2 — `normed = w * x * inv_rms`, then rotate:
//!   `out[lid]      = normed_a·cos θ − normed_b·sin θ`
//!   `out[lid+half] = normed_a·sin θ + normed_b·cos θ`
//! where `θ = pos · inv_freqs[lid]` and the row's position is
//! `pos = offset + (row / n_heads) mod seq_len`.
//!
//! ## DISPATCH INVARIANTS
//!
//! Reduction-mode kernel.
//!
//! - **`TPG = axis_size / 2`** — one thread per rotation pair.
//! - **`axis_size` must be a multiple of 64** so `TPG` is a multiple
//!   of 32 (a whole number of simdgroups) and `TPG ≥ 32`. Common head
//!   dims 64 / 128 / 256 satisfy this.
//! - **`TPG ≤ 1024`** → `axis_size ≤ 2048`.
//! - **Grid: 1 threadgroup per row**, `program_id::<0>()` = row index.
//!
//! Codegen-only; correctness pinned by
//! `tests/rms_norm_rope_gpu_correctness.rs`.

use metaltile::kernel;

/// Fused RMSNorm + paired-layout RoPE for one Q/K head per threadgroup.
#[kernel]
pub fn ffai_rms_norm_rope<T>(
    x: Tensor<T>,
    w: Tensor<T>,
    inv_freqs: Tensor<f32>,
    out: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] axis_size: u32,
    #[constexpr] offset: u32,
    #[constexpr] n_heads: u32,
    #[constexpr] seq_len: u32,
) {
    let row = program_id::<0>();
    let half = axis_size / 2u32;
    let rs = row * axis_size;
    let lid = tid;
    // Phase 1: per-thread pair → threadgroup-wide inv_rms via cross-kernel call.
    // partial_ssq is a Value arg; eps_buf and axis_size are Tensor args whose
    // names are substituted into mt_rms_inv_scalar's callee loads.
    let v1 = load(x[rs + lid]).cast::<f32>();
    let v2 = load(x[rs + lid + half]).cast::<f32>();
    let partial_ssq = v1 * v1 + v2 * v2;
    let inv_rms = mt_rms_inv_scalar(partial_ssq, eps_buf, axis_size);
    // Phase 2: weight scale + RoPE rotation.
    let l = (row / n_heads) % seq_len;
    let pos = (offset + l).cast::<f32>();
    let theta = pos * load(inv_freqs[lid]);
    let cos_t = cos(theta);
    let sin_t = sin(theta);
    let normed_a = v1 * load(w[lid]).cast::<f32>() * inv_rms;
    let normed_b = v2 * load(w[lid + half]).cast::<f32>() * inv_rms;
    store(out[rs + lid], (normed_a * cos_t - normed_b * sin_t).cast::<T>());
    store(out[rs + lid + half], (normed_a * sin_t + normed_b * cos_t).cast::<T>());
}

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

use metaltile_core::{
        DType,
        bench::{TestBuffer, TestSetup},
    };
    use metaltile_macros::test_kernel;

    use super::*;

    fn pack(vals: &[f32], dt: DType) -> Vec<u8> {
        match dt {
            DType::F32 => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
            DType::F16 => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            DType::BF16 =>
                vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
            _ => panic!("unsupported dtype {dt:?}"),
        }
    }

    fn round(v: f32, dt: DType) -> f32 {
        match dt {
            DType::F16 => half::f16::from_f32(v).to_f32(),
            DType::BF16 => half::bf16::from_f32(v).to_f32(),
            _ => v,
        }
    }

    fn pack_u32_scalar(v: u32) -> Vec<u8> { v.to_le_bytes().to_vec() }
    fn pack_f32_scalar(v: f32) -> Vec<u8> { v.to_le_bytes().to_vec() }

    fn inv_freq_table(half: usize) -> Vec<f32> {
        (0..half).map(|i| 1.0 / 10000.0_f32.powf(i as f32 / half as f32)).collect()
    }

    /// CPU oracle: fused RMSNorm + paired-layout RoPE.
    fn naive_rms_norm_rope(
        x: &[f32],
        w: &[f32],
        inv_freqs: &[f32],
        rows: usize,
        axis: usize,
        n_heads: usize,
        seq_len: usize,
        offset: usize,
        eps: f32,
    ) -> Vec<f32> {
        let half = axis / 2;
        let mut out = vec![0.0f32; rows * axis];
        for r in 0..rows {
            let base = r * axis;
            let ssq: f32 = (0..axis).map(|i| x[base + i] * x[base + i]).sum();
            let inv_rms = 1.0 / (ssq / axis as f32 + eps).sqrt();
            let pos = (offset + (r / n_heads) % seq_len) as f32;
            for lid in 0..half {
                let theta = pos * inv_freqs[lid];
                let (s, c) = theta.sin_cos();
                let na = x[base + lid] * w[lid] * inv_rms;
                let nb = x[base + lid + half] * w[lid + half] * inv_rms;
                out[base + lid] = na * c - nb * s;
                out[base + lid + half] = na * s + nb * c;
            }
        }
        out
    }

    #[test_kernel(name = "rms_norm_rope/f32_axis128", dtypes = [f32], tol = 1e-3)]
    fn test_rms_norm_rope_f32(dt: DType) -> TestSetup {
        let (axis, n_heads, seq_len, offset, eps) = (128usize, 4usize, 8usize, 5usize, 1e-5_f32);
        let rows = n_heads * seq_len;
        let half = axis / 2;
        let x: Vec<f32> = (0..rows * axis).map(|i| ((i % 53) as f32) * 0.07 - 1.8).collect();
        let w: Vec<f32> = (0..axis).map(|i| 1.0 + 0.02 * ((i % 11) as f32 - 5.0)).collect();
        let inv_freqs = inv_freq_table(half);
        let expected =
            naive_rms_norm_rope(&x, &w, &inv_freqs, rows, axis, n_heads, seq_len, offset, eps);

        let mut k = ffai_rms_norm_rope::kernel_ir_for(dt);
        k.mode = metaltile_core::ir::KernelMode::Reduction;

        let tpg = (axis / 2) as u32;
        TestSetup::new(k)
            .input(TestBuffer::from_vec("x", pack(&x, dt), dt))
            .input(TestBuffer::from_vec("w", pack(&w, dt), dt))
            .input(TestBuffer::from_vec(
                "inv_freqs",
                bytemuck::cast_slice::<f32, u8>(&inv_freqs).to_vec(),
                DType::F32,
            ))
            .input(TestBuffer::from_vec("eps_buf", pack_f32_scalar(eps), DType::F32))
            .input(TestBuffer::from_vec("axis_size", pack_u32_scalar(axis as u32), DType::U32))
            .input(TestBuffer::from_vec("offset", pack_u32_scalar(offset as u32), DType::U32))
            .input(TestBuffer::from_vec("n_heads", pack_u32_scalar(n_heads as u32), DType::U32))
            .input(TestBuffer::from_vec("seq_len", pack_u32_scalar(seq_len as u32), DType::U32))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_3d(rows as u32, 1, 1, [tpg, 1, 1])
    }

    #[test_kernel(name = "rms_norm_rope/f32_axis64", dtypes = [f32], tol = 1e-3)]
    fn test_rms_norm_rope_f32_min_axis(dt: DType) -> TestSetup {
        let (axis, n_heads, seq_len, offset, eps) = (64usize, 2usize, 4usize, 0usize, 1e-5_f32);
        let rows = n_heads * seq_len;
        let half = axis / 2;
        let x: Vec<f32> = (0..rows * axis).map(|i| ((i % 41) as f32) * 0.09 - 1.5).collect();
        let w: Vec<f32> = (0..axis).map(|i| 1.0 + 0.03 * ((i % 7) as f32 - 3.0)).collect();
        let inv_freqs = inv_freq_table(half);
        let expected =
            naive_rms_norm_rope(&x, &w, &inv_freqs, rows, axis, n_heads, seq_len, offset, eps);

        let mut k = ffai_rms_norm_rope::kernel_ir_for(dt);
        k.mode = metaltile_core::ir::KernelMode::Reduction;

        let tpg = (axis / 2) as u32;
        TestSetup::new(k)
            .input(TestBuffer::from_vec("x", pack(&x, dt), dt))
            .input(TestBuffer::from_vec("w", pack(&w, dt), dt))
            .input(TestBuffer::from_vec(
                "inv_freqs",
                bytemuck::cast_slice::<f32, u8>(&inv_freqs).to_vec(),
                DType::F32,
            ))
            .input(TestBuffer::from_vec("eps_buf", pack_f32_scalar(eps), DType::F32))
            .input(TestBuffer::from_vec("axis_size", pack_u32_scalar(axis as u32), DType::U32))
            .input(TestBuffer::from_vec("offset", pack_u32_scalar(offset as u32), DType::U32))
            .input(TestBuffer::from_vec("n_heads", pack_u32_scalar(n_heads as u32), DType::U32))
            .input(TestBuffer::from_vec("seq_len", pack_u32_scalar(seq_len as u32), DType::U32))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_3d(rows as u32, 1, 1, [tpg, 1, 1])
    }

    #[test_kernel(name = "rms_norm_rope/bf16_axis128", dtypes = [bf16], tol = 1e-1)]
    fn test_rms_norm_rope_bf16(dt: DType) -> TestSetup {
        let (axis, n_heads, seq_len, offset, eps) = (128usize, 4usize, 8usize, 3usize, 1e-5_f32);
        let rows = n_heads * seq_len;
        let half = axis / 2;
        let x: Vec<f32> =
            (0..rows * axis).map(|i| round(((i % 53) as f32) * 0.07 - 1.8, dt)).collect();
        let w: Vec<f32> =
            (0..axis).map(|i| round(1.0 + 0.02 * ((i % 11) as f32 - 5.0), dt)).collect();
        let inv_freqs = inv_freq_table(half);
        let expected =
            naive_rms_norm_rope(&x, &w, &inv_freqs, rows, axis, n_heads, seq_len, offset, eps);

        let mut k = ffai_rms_norm_rope::kernel_ir_for(dt);
        k.mode = metaltile_core::ir::KernelMode::Reduction;

        let tpg = (axis / 2) as u32;
        TestSetup::new(k)
            .input(TestBuffer::from_vec("x", pack(&x, dt), dt))
            .input(TestBuffer::from_vec("w", pack(&w, dt), dt))
            .input(TestBuffer::from_vec(
                "inv_freqs",
                bytemuck::cast_slice::<f32, u8>(&inv_freqs).to_vec(),
                DType::F32,
            ))
            .input(TestBuffer::from_vec("eps_buf", pack_f32_scalar(eps), DType::F32))
            .input(TestBuffer::from_vec("axis_size", pack_u32_scalar(axis as u32), DType::U32))
            .input(TestBuffer::from_vec("offset", pack_u32_scalar(offset as u32), DType::U32))
            .input(TestBuffer::from_vec("n_heads", pack_u32_scalar(n_heads as u32), DType::U32))
            .input(TestBuffer::from_vec("seq_len", pack_u32_scalar(seq_len as u32), DType::U32))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_3d(rows as u32, 1, 1, [tpg, 1, 1])
    }
}
