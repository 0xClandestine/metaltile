//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! AURA fused encode — full per-vector encode pipeline:
//!
//!   1. L2 norm via `simd_sum` + cross-simdgroup reduction in
//!      threadgroup memory.
//!   2. Rotation Π via shared-memory matmul (each thread emits one
//!      rotated component).
//!   3. Quantize via branchless boundary comparison against the
//!      Lloyd-Max boundaries.
//!   4. Pack the codebook indices into a u32 stream via
//!      `atomic_or_tg` so threads racing on the same packed word
//!      (bit slots `[d*Bits, (d+1)*Bits)` may share a u32 word) are
//!      properly serialised.
//!   5. Norm correction — re-run the reduction over the dequantised
//!      centroid values to derive the per-vector norm-correction
//!      factor that the decoder multiplies back through.
//!
//! Port of `turbo_fused_encode` from
//! `ekryski/mlx@alpha:mlx/backend/metal/kernels/turbo_quant.metal`.
//! One threadgroup per input row; one thread per dim slot.
//!
//! ## Layout
//!
//! Inputs:
//! - `input      [rows, dim]`           f32
//! - `rotation   [dim, dim]`            f32
//! - `boundaries [2**bits - 1]`         f32  — Lloyd-Max thresholds.
//! - `codebook   [2**bits]`             f32  — centroid values.
//!
//! Outputs:
//! - `packed_out [rows, packed_width]`  u32
//! - `norms_out  [rows]`                f32  — norm-correction factor.
//!
//! ## Constexpr params
//!
//! - `bits`         — quant bit-width (2 / 3 / 4 / 8).
//! - `dim`          — vector length (64 / 80 / 96 / 128 / 256 / 512).
//! - `packed_width` — `ceil(dim * bits / 32)`.
//! - `levels`       — `1 << bits`.
//!
//! ## DISPATCH INVARIANTS
//!
//! This kernel is reduction-mode and has STRICT threadgroup-geometry
//! requirements. Violating any of these silently miscomputes the
//! encoded output (best case) or pins the GPU in an infinite loop
//! (worst case — see FFAI post-mortem 2026-05-19). Consumers MUST
//! encode these as preconditions in their wrappers.
//!
//! - **TPG = `dim`.** One thread per rotated coordinate. Each thread's
//!   slot in `shared_unit` is `tid`; loads/stores are unconditional.
//! - **`dim` must be a multiple of 32** (one full Apple simdgroup).
//!   The L2-norm `simd_sum` is only well-defined over full simdgroups;
//!   `dim < 32` produces undefined behaviour across lanes.
//! - **`dim ≤ 1024`** (Apple's max-threads-per-threadgroup cap, and
//!   matches the static `threadgroup_alloc("shared_unit", 1024)`).
//! - **`shared_norm[16]`** holds one partial per simdgroup. Adequate
//!   for `dim ≤ 16 * 32 = 512`; dims 513..1024 would overflow this
//!   buffer (currently bench-only — production AURA dims are 64/96/
//!   128/192/256).
//! - **Grid: 1 threadgroup per row.** Wrapper uses
//!   `grid = (dim, rows, 1)`, `tg = (dim, 1, 1)` so Metal slices that
//!   into `rows` threadgroups of `dim` threads.
//!
//! ## Macro structure
//!
//! `aura_encode_kernel!` wraps a single `#[kernel(bench(...))] pub fn …`
//! at module scope.  Bit-widths get separate invocations so the compiler
//! expands the outer macro before the `#[kernel]` proc-macro sees it —
//! required because the proc-macro does not expand inner declarative macros.

use metaltile::kernel;

macro_rules! aura_encode_kernel {
    ($name:ident, $bits:literal, $levels:literal, $subop:literal) => {
        // `input` is the model-dtype K or V row (typically bf16/f16 in
        // production, f32 in tests). All internal math runs in f32 —
        // we cast at the load. Everything else stays f32-only because
        // rotation, codebook, and norm-correction need the precision.
        #[kernel]
        pub fn $name<T>(
            input: Tensor<T>,
            rotation: Tensor<f32>,
            boundaries: Tensor<f32>,
            codebook: Tensor<f32>,
            mut packed_out: Tensor<u32>,
            mut norms_out: Tensor<f32>,
            #[constexpr] dim: u32,
            #[constexpr] packed_width: u32,
        ) {
            let d = tid;
            let row = tgid_x;

            // ── Stage 1: per-thread L2 norm via simd_sum + cross-simdgroup
            // reduction through threadgroup memory.  `shared_norm[16]`
            // holds one partial per simdgroup (16 is enough for dim ≤ 512).
            let val = load(input[row * dim + d]).cast::<f32>();
            let sq = val * val;
            let simd_norm_sq = simd_sum(sq);
            threadgroup_alloc("shared_norm", 16);
            let sg_id = d / 32u32;
            let lane = d & 31u32;
            if lane == 0u32 {
                threadgroup_store("shared_norm", sg_id, simd_norm_sq);
            }
            threadgroup_barrier();

            let mut total_norm_sq = 0.0f32;
            let num_groups = (dim + 31u32) / 32u32;
            for i in range(0u32, num_groups, 1u32) {
                total_norm_sq = total_norm_sq + threadgroup_load("shared_norm", i);
            }
            let norm_val = sqrt(total_norm_sq);
            let inv_norm = select(norm_val > 1.0e-8f32, 1.0f32 / norm_val, 0.0f32);
            let unit_val = val * inv_norm;

            // ── Stage 2: rotation Π via shared-memory matmul.  Each
            // thread stores its unit value, barriers, then reads the
            // full row to compute its rotated component.  `shared_unit`
            // sized at 1024 to match the MLX upstream's max dim.
            threadgroup_alloc("shared_unit", 1024);
            threadgroup_store("shared_unit", d, unit_val);
            threadgroup_barrier();
            let mut rotated = 0.0f32;
            for j in range(0u32, dim, 1u32) {
                rotated =
                    rotated + load(rotation[d * dim + j]) * threadgroup_load("shared_unit", j);
            }

            // ── Stage 3: branchless boundary comparison → codebook
            // index.  For LEVELS levels we have LEVELS-1 boundaries;
            // the index is the count of boundaries the rotated value
            // exceeds.
            let mut idx = 0u32;
            for b in range(0u32, $levels - 1u32, 1u32) {
                idx = idx + (rotated > load(boundaries[b])).cast::<u32>();
            }

            // ── Stage 4: pack the index into the u32 stream via
            // `atomic_or_tg` on threadgroup memory.  `shared_packed`
            // sized for the worst-case PackedWidth (D=512, bits=8 →
            // 128).  Other (dim, bits) combos use the prefix only.
            let bit_offset = d * $bits;
            let word_idx = bit_offset / 32u32;
            let shift = bit_offset & 31u32;
            let masked = idx & ((1u32 << $bits) - 1u32);

            threadgroup_alloc("shared_packed", 128, "u32");
            if d < packed_width {
                threadgroup_store("shared_packed", d, 0u32);
            }
            threadgroup_barrier();

            atomic_or_tg("shared_packed", word_idx, masked << shift);
            // Cross-word spill — write the high bits into the next u32
            // if the index straddles a word boundary.
            //
            // Use unsigned-only arithmetic.  The original formulation
            // `(shift + bits).cast::<i32>() - 32i32 > 0i32` lost its
            // signedness in codegen (lowered to `int v_spill_bits =
            // (int)(shift + bits) - 32; bool = v_spill_bits > 0u`),
            // making the comparison int-vs-uint.  C/MSL promotes the
            // int to uint, so -28 becomes ~4e9 and the spill branch
            // ran for EVERY thread, polluting `shared_packed[word+1]`
            // with garbage shifted by `masked >> 32` (which Apple
            // Silicon evaluates as `masked >> 0 = masked`).  Symptom:
            // low nibbles of subsequent words OR'd with unrelated dim
            // indices.  Caught by the aura_encode GPU correctness
            // test.  See FFAI post-mortem 2026-05-19 — a metaltile
            // codegen follow-up should emit i32 comparisons
            // faithfully when the DSL requests them.
            let total_bits = shift + $bits;
            if total_bits > 32u32 {
                let spill_u = total_bits - 32u32;
                atomic_or_tg("shared_packed", word_idx + 1u32, masked >> ($bits - spill_u));
            }
            threadgroup_barrier();

            if d < packed_width {
                store(packed_out[row * packed_width + d], threadgroup_load("shared_packed", d));
            }

            // ── Stage 5: norm correction.  Re-run the reduction over
            // the dequantised centroid values; `recon_norm` gives the
            // L2 norm of the reconstructed vector, and
            // `corrected = norm_val / recon_norm` is what the decoder
            // multiplies back through.
            let centroid_val = load(codebook[idx]);
            let recon_sq = centroid_val * centroid_val;
            let simd_recon_sq = simd_sum(recon_sq);
            if lane == 0u32 {
                threadgroup_store("shared_norm", sg_id, simd_recon_sq);
            }
            threadgroup_barrier();
            let mut total_recon_sq = 0.0f32;
            for i in range(0u32, num_groups, 1u32) {
                total_recon_sq = total_recon_sq + threadgroup_load("shared_norm", i);
            }
            let recon_norm = sqrt(total_recon_sq);
            let corrected_norm = select(recon_norm > 1.0e-8f32, norm_val / recon_norm, norm_val);

            if d == 0u32 {
                store(norms_out[row], corrected_norm);
            }
        }
    };
}

aura_encode_kernel!(aura_encode_int2, 2u32, 4u32, "encode_int2");
aura_encode_kernel!(aura_encode_int3, 3u32, 8u32, "encode_int3");
aura_encode_kernel!(aura_encode_int4, 4u32, 16u32, "encode_int4");
aura_encode_kernel!(aura_encode_int6, 6u32, 64u32, "encode_int6");
aura_encode_kernel!(aura_encode_int8, 8u32, 256u32, "encode_int8");

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::test_kernel;
    use metaltile_core::{
        DType,
        bench::{TestBuffer, TestSetup},
    };

    fn pack_f32(vals: &[f32]) -> Vec<u8> { bytemuck::cast_slice::<f32, u8>(vals).to_vec() }
    fn pack_u32(vals: &[u32]) -> Vec<u8> { bytemuck::cast_slice::<u32, u8>(vals).to_vec() }
    fn u32_le(v: u32) -> Vec<u8> { v.to_le_bytes().to_vec() }

    fn identity_rotation(dim: usize) -> Vec<f32> {
        let mut r = vec![0.0_f32; dim * dim];
        for d in 0..dim {
            r[d * dim + d] = 1.0;
        }
        r
    }

    /// CPU reference for aura_encode (identity rotation, int4 Lloyd-Max codebook).
    fn naive_aura_encode_f32(
        input: &[f32],
        rotation: &[f32],
        boundaries: &[f32],
        codebook: &[f32],
        rows: usize,
        dim: usize,
        bits: usize,
    ) -> (Vec<u32>, Vec<f32>) {
        let levels = 1usize << bits;
        let packed_width = (dim * bits).div_ceil(32);
        let mut packed_out = vec![0u32; rows * packed_width];
        let mut norms_out = vec![0.0_f32; rows];
        for r in 0..rows {
            let row_base = r * dim;
            let r_norm: f32 =
                input[row_base..row_base + dim].iter().map(|v| v * v).sum::<f32>().sqrt();
            let inv = if r_norm > 1e-8 { 1.0 / r_norm } else { 0.0 };
            let mut y = vec![0.0_f32; dim];
            for d in 0..dim {
                let mut acc = 0.0_f32;
                for j in 0..dim {
                    acc += rotation[d * dim + j] * input[row_base + j] * inv;
                }
                y[d] = acc;
            }
            let mut indices = vec![0u32; dim];
            for d in 0..dim {
                let mut idx = 0u32;
                for &bound in boundaries.iter().take(levels - 1) {
                    if y[d] > bound {
                        idx += 1;
                    }
                }
                indices[d] = idx;
            }
            for (d, &idx) in indices.iter().enumerate() {
                let bit_offset = d * bits;
                let word_idx = bit_offset / 32;
                let shift = bit_offset & 31;
                let masked = idx & ((1u32 << bits) - 1);
                packed_out[r * packed_width + word_idx] |= masked << shift;
                let spill_bits = (shift + bits) as i32 - 32;
                if spill_bits > 0 {
                    let spill = spill_bits as u32;
                    packed_out[r * packed_width + word_idx + 1] |= masked >> (bits as u32 - spill);
                }
            }
            let recon_sq: f32 = indices.iter().map(|&i| codebook[i as usize].powi(2)).sum();
            let recon_norm = recon_sq.sqrt();
            norms_out[r] = if recon_norm > 1e-8 { r_norm / recon_norm } else { r_norm };
        }
        (packed_out, norms_out)
    }

    #[test_kernel(name = "ffai/aura/encode_int4", dtypes = [f32], tol = 1e-4)]
    fn test_aura_encode_int4(dt: DType) -> TestSetup {
        use super::aura_encode_int4;
        let dim = 128usize;
        let bits = 4usize;
        let packed_width = (dim * bits).div_ceil(32);
        let rows = 2usize;
        let levels = 16usize;

        let codebook: Vec<f32> =
            (0..levels).map(|i| -1.0 + 2.0 * (i as f32) / (levels as f32 - 1.0)).collect();
        let boundaries: Vec<f32> =
            (0..levels - 1).map(|i| 0.5 * (codebook[i] + codebook[i + 1])).collect();
        let rotation = identity_rotation(dim);
        let input: Vec<f32> = (0..rows * dim).map(|i| (((i % 17) as f32 - 8.0) * 0.07)).collect();

        let (expected_packed, expected_norms) =
            naive_aura_encode_f32(&input, &rotation, &boundaries, &codebook, rows, dim, bits);

        let mut kernel_ir = aura_encode_int4::kernel_ir_for(dt);
        kernel_ir.mode = metaltile_core::ir::KernelMode::Reduction;

        // One TG per row, dim threads per TG.
        TestSetup::new(kernel_ir)
            .input(TestBuffer::from_vec("input", pack_f32(&input), DType::F32))
            .input(TestBuffer::from_vec("rotation", pack_f32(&rotation), DType::F32))
            .input(TestBuffer::from_vec("boundaries", pack_f32(&boundaries), DType::F32))
            .input(TestBuffer::from_vec("codebook", pack_f32(&codebook), DType::F32))
            .input(TestBuffer::from_vec("packed_width", u32_le(packed_width as u32), DType::U32))
            .expect(TestBuffer::from_vec("packed_out", pack_u32(&expected_packed), DType::U32))
            .expect(TestBuffer::from_vec("norms_out", pack_f32(&expected_norms), DType::F32))
            .grid_3d(rows as u32, 1, 1, [dim as u32, 1, 1])
    }
}
