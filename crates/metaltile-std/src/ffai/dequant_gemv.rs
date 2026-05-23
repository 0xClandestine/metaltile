//! MLX-format dequantizing GEMV kernels for int3 / int4 / int5 / int6 /
//! int8 weights. Reduction-mode kernels; one threadgroup per output row.
//!
//! Layouts (per dtype, with N = `in_dim`, G = `group_size`):
//!
//!   weight  [out_dim, N * bits / 32]   uint32  (bit-packed)
//!   scales  [out_dim, N / G]           T
//!   biases  [out_dim, N / G]           T
//!   input   [N]                        T
//!   output  [out_dim]                  T
//!
//! Two dispatch strategies, chosen per bit-width:
//!
//! **Pack-strided** (int4, int8) — threads stride over u32 packs; each pack
//! yields `32/bits` values.  One u32 load amortises across all values in
//! the pack; no extra bit-extraction arithmetic beyond a simple shift+mask.
//!
//!   n_packs = in_dim / (32/bits)
//!   per pack: 1 u32 load → (32/bits) shift+mask extractions → (32/bits) FMAs
//!
//! **Element-strided** (int3, int5, int6) — threads stride over individual
//! elements using the two-word bit-stream formula from `dequant_gather.rs`.
//! Odd-width packs don't align to u32 boundaries so pack-striding would
//! require complex cycle handling; element-striding is cleaner and achieves
//! the same cache behaviour (adjacent threads share the same u32 words →
//! L1 multicast) while avoiding the idle-thread problem of the old
//! group-strided approach.
//!
//!   n_iters = ceil(in_dim / lsize)
//!   per iter: 1-2 u32 loads + bit-extract (5 ops) + 1 FMA
//!
//! ## Macro structure
//!
//! Each bit-width gets a `#[kernel] pub fn dequant_gemv_int<bits><T>(…)` plus
//! its `BenchSpec` registration.  The body shape is identical across bit-
//! widths within a strategy, so an outer `macro_rules!` (`dequant_gemv_pow2!`
//! / `dequant_gemv_odd!`) emits the whole `#[kernel] fn …` + `inventory::
//! submit!` at module scope; the compiler expands those before the
//! `#[kernel]` proc-macro runs, so the body parser sees concrete tokens.
//! Embedding the body inside an *inner* `macro_rules!` invocation (the
//! previous shape of this file) silently produced empty kernels — the
//! proc-macro doesn't expand inner declarative macros.

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

const ALL_FLOAT_DTYPES: &[DType] = &[DType::F32, DType::F16, DType::BF16];

// ── Pack-strided kernel (int4, int8) ──────────────────────────────────────
//
// Each thread strides over u32 packs. One pack load → (32/$bits) extractions.
// `$bits` must divide 32 evenly (i.e. 4 or 8).
macro_rules! dequant_gemv_pow2 {
    ($name:ident, $bits:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            weight: Tensor<u32>,
            scales: Tensor<T>,
            biases: Tensor<T>,
            input: Tensor<T>,
            output: Tensor<T>,
            #[constexpr] in_dim: u32,
            #[constexpr] group_size: u32,
        ) {
            let vals_per_pack = 32u32 / $bits;
            let mask = (1u32 << $bits) - 1u32;
            let row = program_id::<0>();
            let n_packs_per_row = in_dim / vals_per_pack;
            let n_groups = in_dim / group_size;
            let packs_per_group = group_size / vals_per_pack;
            let row_pack_off = row * n_packs_per_row;
            let row_group_off = row * n_groups;

            let mut acc = 0.0f32;
            let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
            for p_iter in range(0u32, p_iters, 1u32) {
                let pack_idx = p_iter * lsize + tid;
                if pack_idx < n_packs_per_row {
                    let g = pack_idx / packs_per_group;
                    let scale = load(scales[row_group_off + g]).cast::<f32>();
                    let bias = load(biases[row_group_off + g]).cast::<f32>();

                    let packed = load(weight[row_pack_off + pack_idx]);
                    let p_off = pack_idx * vals_per_pack;

                    for i in range(0u32, vals_per_pack, 1u32) {
                        let q = (packed >> (i * $bits)) & mask;
                        acc = acc
                            + (q.cast::<f32>() * scale + bias)
                                * load(input[p_off + i]).cast::<f32>();
                    }
                }
            }

            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(output[row], total.cast::<T>());
            }
        }

        inventory::submit! {
            BenchSpec {
                op: "dequant_gemv",
                subop: $subop,
                kernel_name: stringify!($name),
                kernel_ir: $name::kernel_ir_for,
                dtypes: ALL_FLOAT_DTYPES,
                tol: 0.0,
                mlx_src: None,
                mlx_pattern: None,
                shapes: &[],
                dispatch: BenchDispatch::Generic,
                kernel_mode: Some(KernelMode::Reduction),
            }
        }
    };
}

// ── Element-strided kernel (int3, int5, int6) ────────────────────────────
//
// Odd-width packs don't divide u32s evenly, so pack-striding requires
// complex multi-u32 cycle handling. Element-striding with the two-word
// bit-stream formula is cleaner and avoids the idle-thread problem of the
// old group-strided approach (where threads past n_groups sat idle).
// Adjacent threads access adjacent elements → same u32 words → L1 multicast.
macro_rules! dequant_gemv_odd {
    ($name:ident, $bits:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            weight: Tensor<u32>,
            scales: Tensor<T>,
            biases: Tensor<T>,
            input: Tensor<T>,
            output: Tensor<T>,
            #[constexpr] in_dim: u32,
            #[constexpr] group_size: u32,
        ) {
            let row = program_id::<0>();
            let u32_per_row = in_dim * $bits / 32u32;
            let n_groups = in_dim / group_size;
            let row_u32_off = row * u32_per_row;
            let row_group_off = row * n_groups;

            let mut acc = 0.0f32;
            let n_iters = (in_dim + lsize - 1u32) / lsize;
            for _iter in range(0u32, n_iters, 1u32) {
                let d = _iter * lsize + tid;
                if d < in_dim {
                    let g = d / group_size;
                    let scale = load(scales[row_group_off + g]).cast::<f32>();
                    let bias = load(biases[row_group_off + g]).cast::<f32>();

                    let bit_off = d * $bits;
                    let word_idx = bit_off / 32u32;
                    let bit_in_w = bit_off & 31u32;
                    let bits_in_w0 = 32u32 - bit_in_w;
                    let lo_bits = select(bits_in_w0 >= $bits, $bits, bits_in_w0);
                    let spill = $bits - lo_bits;

                    let w0 = load(weight[row_u32_off + word_idx]);
                    let w1idx = select(spill > 0u32, word_idx + 1u32, word_idx);
                    let w1 = load(weight[row_u32_off + w1idx]);

                    let lo = (w0 >> bit_in_w) & ((1u32 << lo_bits) - 1u32);
                    let hi = (w1 & ((1u32 << spill) - 1u32)) << lo_bits;
                    let q = lo | hi;

                    acc = acc + (q.cast::<f32>() * scale + bias) * load(input[d]).cast::<f32>();
                }
            }

            let total = reduce_sum(acc);
            if tid == 0u32 {
                store(output[row], total.cast::<T>());
            }
        }

        inventory::submit! {
            BenchSpec {
                op: "dequant_gemv",
                subop: $subop,
                kernel_name: stringify!($name),
                kernel_ir: $name::kernel_ir_for,
                dtypes: ALL_FLOAT_DTYPES,
                tol: 0.0,
                mlx_src: None,
                mlx_pattern: None,
                shapes: &[],
                dispatch: BenchDispatch::Generic,
                kernel_mode: Some(KernelMode::Reduction),
            }
        }
    };
}

dequant_gemv_pow2!(dequant_gemv_int4, 4u32, "int4");
dequant_gemv_pow2!(dequant_gemv_int8, 8u32, "int8");
dequant_gemv_odd!(dequant_gemv_int3, 3u32, "int3");
dequant_gemv_odd!(dequant_gemv_int5, 5u32, "int5");
dequant_gemv_odd!(dequant_gemv_int6, 6u32, "int6");

// ── Perf-tuned int4 GEMV — 8 output rows per TG ─────────────────────────
//
// Mirrors `mt_qmv`'s geometry: tpg = 64 (2 simdgroups × 32 lanes);
// each simdgroup computes 4 output rows (indexed by `simd_id`); each
// lane caches 16 X values per 512-wide K-block. Uses the mask-without-
// shift trick + algebraic-split accumulator `s*q_dot + b*xs` from
// `mt_qmv` / MLX `qdot` (quantized.h:235-244).
//
// Kept separate from `dequant_gemv_int4` (the one-row-per-TG scalar)
// for backward compat — FFAI's GPU-router uses the indirect variant of
// the scalar kernel. The fast variant has no indirect consumer today;
// adding one is a one-line edit in `dequant_gemv_wants_indirect`.
//
// Dispatch:
//   Grid: [out_dim/8, 1, 1]  — one TG per 8-row tile.
//   TPG: 64                  — 2 SG × 32 lanes.
//   in_dim: multiple of 512 (block = 16 X × 32 lanes = 512 K elements).
//   out_dim: multiple of 8.
//   group_size: 64.

/// Perf-tuned int4 dequant GEMV — 8 rows per TG, `mt_qmv` geometry.
///
/// `output[row] = Σ_i (q[row,i]·scale_g + bias_g) · input[i]`
/// for 8 consecutive output rows per dispatch. Grid: `[out_dim/8, 1, 1]`,
/// TPG = 64, group_size = 64, in_dim a multiple of 512.
///
/// The existing `dequant_gemv_int4` is kept unchanged for backward compat
/// (FFAI's indirect-dispatch router uses that name). This variant is the
/// perf path for new callers that can guarantee the alignment constraints.
#[kernel]
pub fn dequant_gemv_int4_fast<T>(
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let tg = tgid_x;
    let sg = simd_id;
    let lane = simd_lane;

    // 8 rows per TG: SG 0 → rows 0-3, SG 1 → rows 4-7.
    let row0 = tg * 8u32 + sg * 4u32;
    let row1 = row0 + 1u32;
    let row2 = row0 + 2u32;
    let row3 = row0 + 3u32;

    let gs_per_row = in_dim / group_size;
    let packs_per_row = in_dim / 8u32; // 8 int4 values per u32

    let w_base0 = row0 * packs_per_row;
    let w_base1 = row1 * packs_per_row;
    let w_base2 = row2 * packs_per_row;
    let w_base3 = row3 * packs_per_row;

    let sb_base0 = row0 * gs_per_row;
    let sb_base1 = row1 * gs_per_row;
    let sb_base2 = row2 * gs_per_row;
    let sb_base3 = row3 * gs_per_row;

    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;

    let lane_x_off = lane * 16u32;
    let lane_pack_off = lane * 2u32;

    // Mask-without-shift constants — eliminates 56 shifts per block.
    // Matches `mt_qmv` / MLX `qdot` (quantized.h:235-244).
    let s_16 = 0.0625f32;
    let s_256 = 0.00390625f32;
    let s_4096 = 0.000244140625f32;

    for _b in range(0u32, in_dim, 512u32) {
        let xb = _b + lane_x_off;
        let xi0 = xb;
        let xi1 = xb + 1u32;
        let xi2 = xb + 2u32;
        let xi3 = xb + 3u32;
        let xi4 = xb + 4u32;
        let xi5 = xb + 5u32;
        let xi6 = xb + 6u32;
        let xi7 = xb + 7u32;
        let xi8 = xb + 8u32;
        let xi9 = xb + 9u32;
        let xi10 = xb + 10u32;
        let xi11 = xb + 11u32;
        let xi12 = xb + 12u32;
        let xi13 = xb + 13u32;
        let xi14 = xb + 14u32;
        let xi15 = xb + 15u32;

        let x0 = load(input[xi0]).cast::<f32>();
        let x1_raw = load(input[xi1]).cast::<f32>();
        let x2_raw = load(input[xi2]).cast::<f32>();
        let x3_raw = load(input[xi3]).cast::<f32>();
        let x4 = load(input[xi4]).cast::<f32>();
        let x5_raw = load(input[xi5]).cast::<f32>();
        let x6_raw = load(input[xi6]).cast::<f32>();
        let x7_raw = load(input[xi7]).cast::<f32>();
        let x8 = load(input[xi8]).cast::<f32>();
        let x9_raw = load(input[xi9]).cast::<f32>();
        let x10_raw = load(input[xi10]).cast::<f32>();
        let x11_raw = load(input[xi11]).cast::<f32>();
        let x12 = load(input[xi12]).cast::<f32>();
        let x13_raw = load(input[xi13]).cast::<f32>();
        let x14_raw = load(input[xi14]).cast::<f32>();
        let x15_raw = load(input[xi15]).cast::<f32>();

        // Algebraic-split: acc = scale * q_dot + bias * xs, where
        // xs = Σ input[i] over the 16-element block.
        let xs = x0
            + x1_raw
            + x2_raw
            + x3_raw
            + x4
            + x5_raw
            + x6_raw
            + x7_raw
            + x8
            + x9_raw
            + x10_raw
            + x11_raw
            + x12
            + x13_raw
            + x14_raw
            + x15_raw;

        // Pre-scale at nibble positions 1/2/3 for mask-without-shift.
        let x1 = x1_raw * s_16;
        let x2 = x2_raw * s_256;
        let x3 = x3_raw * s_4096;
        let x5 = x5_raw * s_16;
        let x6 = x6_raw * s_256;
        let x7 = x7_raw * s_4096;
        let x9 = x9_raw * s_16;
        let x10 = x10_raw * s_256;
        let x11 = x11_raw * s_4096;
        let x13 = x13_raw * s_16;
        let x14 = x14_raw * s_256;
        let x15 = x15_raw * s_4096;

        let g = xb / group_size;
        let pack_off = _b / 8u32 + lane_pack_off;

        // ── Row 0 ──
        let p00 = load(weight[w_base0 + pack_off]);
        let p01 = load(weight[w_base0 + pack_off + 1u32]);
        let p00_hi = p00 >> 16u32;
        let p01_hi = p01 >> 16u32;
        let s0 = load(scales[sb_base0 + g]).cast::<f32>();
        let bi0 = load(biases[sb_base0 + g]).cast::<f32>();
        let q00 = (p00 & 15u32).cast::<f32>();
        let q01 = (p00 & 240u32).cast::<f32>();
        let q02 = (p00 & 3840u32).cast::<f32>();
        let q03 = (p00 & 61440u32).cast::<f32>();
        let q04 = (p00_hi & 15u32).cast::<f32>();
        let q05 = (p00_hi & 240u32).cast::<f32>();
        let q06 = (p00_hi & 3840u32).cast::<f32>();
        let q07 = (p00_hi & 61440u32).cast::<f32>();
        let q08 = (p01 & 15u32).cast::<f32>();
        let q09 = (p01 & 240u32).cast::<f32>();
        let q010 = (p01 & 3840u32).cast::<f32>();
        let q011 = (p01 & 61440u32).cast::<f32>();
        let q012 = (p01_hi & 15u32).cast::<f32>();
        let q013 = (p01_hi & 240u32).cast::<f32>();
        let q014 = (p01_hi & 3840u32).cast::<f32>();
        let q015 = (p01_hi & 61440u32).cast::<f32>();
        let qd0 = q00 * x0
            + q01 * x1
            + q02 * x2
            + q03 * x3
            + q04 * x4
            + q05 * x5
            + q06 * x6
            + q07 * x7
            + q08 * x8
            + q09 * x9
            + q010 * x10
            + q011 * x11
            + q012 * x12
            + q013 * x13
            + q014 * x14
            + q015 * x15;
        acc0 = acc0 + s0 * qd0 + bi0 * xs;

        // ── Row 1 ──
        let p10 = load(weight[w_base1 + pack_off]);
        let p11 = load(weight[w_base1 + pack_off + 1u32]);
        let p10_hi = p10 >> 16u32;
        let p11_hi = p11 >> 16u32;
        let s1 = load(scales[sb_base1 + g]).cast::<f32>();
        let bi1 = load(biases[sb_base1 + g]).cast::<f32>();
        let q10 = (p10 & 15u32).cast::<f32>();
        let q11 = (p10 & 240u32).cast::<f32>();
        let q12 = (p10 & 3840u32).cast::<f32>();
        let q13 = (p10 & 61440u32).cast::<f32>();
        let q14 = (p10_hi & 15u32).cast::<f32>();
        let q15 = (p10_hi & 240u32).cast::<f32>();
        let q16 = (p10_hi & 3840u32).cast::<f32>();
        let q17 = (p10_hi & 61440u32).cast::<f32>();
        let q18 = (p11 & 15u32).cast::<f32>();
        let q19 = (p11 & 240u32).cast::<f32>();
        let q110 = (p11 & 3840u32).cast::<f32>();
        let q111 = (p11 & 61440u32).cast::<f32>();
        let q112 = (p11_hi & 15u32).cast::<f32>();
        let q113 = (p11_hi & 240u32).cast::<f32>();
        let q114 = (p11_hi & 3840u32).cast::<f32>();
        let q115 = (p11_hi & 61440u32).cast::<f32>();
        let qd1 = q10 * x0
            + q11 * x1
            + q12 * x2
            + q13 * x3
            + q14 * x4
            + q15 * x5
            + q16 * x6
            + q17 * x7
            + q18 * x8
            + q19 * x9
            + q110 * x10
            + q111 * x11
            + q112 * x12
            + q113 * x13
            + q114 * x14
            + q115 * x15;
        acc1 = acc1 + s1 * qd1 + bi1 * xs;

        // ── Row 2 ──
        let p20 = load(weight[w_base2 + pack_off]);
        let p21 = load(weight[w_base2 + pack_off + 1u32]);
        let p20_hi = p20 >> 16u32;
        let p21_hi = p21 >> 16u32;
        let s2 = load(scales[sb_base2 + g]).cast::<f32>();
        let bi2 = load(biases[sb_base2 + g]).cast::<f32>();
        let q20 = (p20 & 15u32).cast::<f32>();
        let q21 = (p20 & 240u32).cast::<f32>();
        let q22 = (p20 & 3840u32).cast::<f32>();
        let q23 = (p20 & 61440u32).cast::<f32>();
        let q24 = (p20_hi & 15u32).cast::<f32>();
        let q25 = (p20_hi & 240u32).cast::<f32>();
        let q26 = (p20_hi & 3840u32).cast::<f32>();
        let q27 = (p20_hi & 61440u32).cast::<f32>();
        let q28 = (p21 & 15u32).cast::<f32>();
        let q29 = (p21 & 240u32).cast::<f32>();
        let q210 = (p21 & 3840u32).cast::<f32>();
        let q211 = (p21 & 61440u32).cast::<f32>();
        let q212 = (p21_hi & 15u32).cast::<f32>();
        let q213 = (p21_hi & 240u32).cast::<f32>();
        let q214 = (p21_hi & 3840u32).cast::<f32>();
        let q215 = (p21_hi & 61440u32).cast::<f32>();
        let qd2 = q20 * x0
            + q21 * x1
            + q22 * x2
            + q23 * x3
            + q24 * x4
            + q25 * x5
            + q26 * x6
            + q27 * x7
            + q28 * x8
            + q29 * x9
            + q210 * x10
            + q211 * x11
            + q212 * x12
            + q213 * x13
            + q214 * x14
            + q215 * x15;
        acc2 = acc2 + s2 * qd2 + bi2 * xs;

        // ── Row 3 ──
        let p30 = load(weight[w_base3 + pack_off]);
        let p31 = load(weight[w_base3 + pack_off + 1u32]);
        let p30_hi = p30 >> 16u32;
        let p31_hi = p31 >> 16u32;
        let s3 = load(scales[sb_base3 + g]).cast::<f32>();
        let bi3 = load(biases[sb_base3 + g]).cast::<f32>();
        let q30 = (p30 & 15u32).cast::<f32>();
        let q31 = (p30 & 240u32).cast::<f32>();
        let q32 = (p30 & 3840u32).cast::<f32>();
        let q33 = (p30 & 61440u32).cast::<f32>();
        let q34 = (p30_hi & 15u32).cast::<f32>();
        let q35 = (p30_hi & 240u32).cast::<f32>();
        let q36 = (p30_hi & 3840u32).cast::<f32>();
        let q37 = (p30_hi & 61440u32).cast::<f32>();
        let q38 = (p31 & 15u32).cast::<f32>();
        let q39 = (p31 & 240u32).cast::<f32>();
        let q310 = (p31 & 3840u32).cast::<f32>();
        let q311 = (p31 & 61440u32).cast::<f32>();
        let q312 = (p31_hi & 15u32).cast::<f32>();
        let q313 = (p31_hi & 240u32).cast::<f32>();
        let q314 = (p31_hi & 3840u32).cast::<f32>();
        let q315 = (p31_hi & 61440u32).cast::<f32>();
        let qd3 = q30 * x0
            + q31 * x1
            + q32 * x2
            + q33 * x3
            + q34 * x4
            + q35 * x5
            + q36 * x6
            + q37 * x7
            + q38 * x8
            + q39 * x9
            + q310 * x10
            + q311 * x11
            + q312 * x12
            + q313 * x13
            + q314 * x14
            + q315 * x15;
        acc3 = acc3 + s3 * qd3 + bi3 * xs;
    }

    // Cross-lane reduce: each row's partial → one value per simdgroup.
    let r0 = simd_sum(acc0);
    let r1 = simd_sum(acc1);
    let r2 = simd_sum(acc2);
    let r3 = simd_sum(acc3);
    if lane == 0u32 {
        store(output[row0], r0.cast::<T>());
        store(output[row1], r1.cast::<T>());
        store(output[row2], r2.cast::<T>());
        store(output[row3], r3.cast::<T>());
    }
}

inventory::submit! {
    BenchSpec {
        op: "dequant_gemv",
        subop: "int4_fast",
        kernel_name: "dequant_gemv_int4_fast",
        kernel_ir: dequant_gemv_int4_fast::kernel_ir_for,
        dtypes: ALL_FLOAT_DTYPES,
        tol: 0.0,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
    }
}

/// Per-kernel opt-in for the indirect Swift-wrapper variant. FFAI's
/// GPU-router dispatches the int4 dequant-GEMV indirectly so the GPU
/// can drive the per-MoE-layer grid shape from a buffer; the other
/// dequant-GEMV bit-widths have no indirect consumer today.
///
/// Lives here (next to the kernel definitions) rather than in
/// `metaltile-codegen` so that adding a new kernel that wants the
/// indirect variant is a one-line edit in the same file as the
/// kernel, not a special-case match buried in the codegen pass.
/// The `tile emit` driver consumes this on the way to setting
/// `Kernel::wants_indirect_variant` before codegen runs.
pub fn dequant_gemv_wants_indirect(kernel_name: &str) -> bool {
    matches!(kernel_name, "dequant_gemv_int4_f16" | "dequant_gemv_int4_bf16")
}
