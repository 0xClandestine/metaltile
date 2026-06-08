//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Consolidated 1D convolution — see `../PLAN.md` for the full migration plan.
//!
//! Two `#[kernel(variants(...))]` blocks cover all 1D conv kernels in this
//! crate:
//!
//! * [`mt_conv1d_dense`](self::mt_conv1d_dense) — the four dense 1D convs
//!   (direct with/without dilation, transpose full/depthwise) in one
//!   function. Four named variants via `VARIANT` axis:
//!   `mt_conv1d_dense_direct` (audio_conv1d),
//!   `mt_conv1d_dense_dilated` (conv1d_dilated),
//!   `mt_conv1d_dense_transpose` (conv1d_transpose),
//!   `mt_conv1d_dense_depthwise` (ffai_conv1d_transpose_depthwise).
//!   The `dilation` constexpr is OPTIONAL — stripped from the MSL signature
//!   for the direct variant where the body does not reference it, so the
//!   host doesn't need to bind a buffer slot for that one kernel.
//!
//! * [`mt_conv1d_quant`](self::mt_conv1d_quant) — the 19
//!   block-scaled quantised-weight formats applied to both the direct
//!   (audio) and dilated (fishspeech) paths. Axes `DILATED × FMT`
//!   (38-row) generate the 38 kernels
//!   `mt_conv1d_quant_{audio,fishspeech}_{mxfp4,nvfp4,…,int8}`. The `dilation` constexpr is
//!   OPTIONAL — stripped for DILATED=audio and kept for DILATED=fishspeech
//!   (fishspeech). FMT/BITS/WT/ST co-vary across the 19 formats; type
//!   variants cannot gate compile-time `if` (per the variants macro's
//!   design), so the FMT int is the gating discriminant and the body has a
//!   deep `if FMT == N { }` tree pruned at variants-substitution time.
//!
//! The decode primitives (`mt_decode_e2m1`, `mt_decode_e4m3`,
//! `mt_decode_e5m2`, `mt_decode_int8`, `mt_decode_e8m0`,
//! `mt_unpack_nbit`) live in `super::primitives` and are inlined at
//! codegen by `KernelInlinePass`. The body parser lowers `if` to
//! `Op::If` (no value), so the active branch updates a pre-declared
//! mutable accumulator (`let mut acc: f32 = 0.0; ... acc = ...;`) rather
//! than binding `if` to a `let`.
//!
//! ## Format map (block-scaled only)
//!
//! | FMT | Format         | Weight | Scale | Decode element            | Decode scale           |
//! |----:|:---------------|:-------|:------|:--------------------------|:-----------------------|
//! |   0 | mxfp4          | u32    | u8    | nibble E2M1               | E8M0                   |
//! |   1 | nvfp4          | u32    | u8    | nibble E2M1               | E4M3 × global          |
//! | 2-6 | mxint{2..6}    | u32    | u8    | sub-byte bitstream sign   | E8M0                   |
//! |   7 | fp4            | u32    | f32   | nibble E2M1               | direct f32             |
//! | 8-12| int{2..6}      | u32    | f32   | sub-byte bitstream sign   | direct f32             |
//! |  13 | mxfp8\_e4m3    | u8     | u8    | byte E4M3                 | E8M0                   |
//! |  14 | mxfp8\_e5m2    | u8     | u8    | byte E5M2                 | E8M0                   |
//! |  15 | mxint8         | u8     | u8    | byte int8                 | E8M0                   |
//! |  16 | fp8\_e5m2      | u8     | f32   | byte E5M2                 | direct f32             |
//! |  17 | nvfp8          | u8     | f32   | byte E4M3                 | direct f32             |
//! |  18 | int8           | u8     | f32   | byte int8                 | direct f32             |
//!
//! Sub-byte ints (FMT 2-6, 8-12) and nibble formats (FMT 0,1,7) all share the
//! same bitstream view: a per-row word base of `oc * (c_dim * BITS / 32)` with
//! `bit_off = col * BITS`. Nibbles are BITS=4 byte-aligned codes — `spill` is
//! always 0 in that case. The straddle-aware `mt_unpack_nbit` primitive
//! handles the generic int2/3/5/6 case where codes straddle word
//! boundaries.

use metaltile::kernel;

// ─── § Dense / float ──────────────────────────────────────────────────────────
// One variants block covers all four dense 1D convs via a 3-axis cartesian
// product (manually enumerated to avoid duplicate-equivalent variants):
//
//   (TRANSPOSE=0, DILATED=0, DEPTHWISE=0) → audio_conv1d                  (direct, no dilation)
//   (TRANSPOSE=0, DILATED=1, DEPTHWISE=0) → conv1d_dilated              (direct, with dilation)
//   (TRANSPOSE=1, DILATED=0, DEPTHWISE=0) → conv1d_transpose            (transpose, full)
//   (TRANSPOSE=1, DILATED=0, DEPTHWISE=1) → ffai_conv1d_transpose_depthwise (transpose, depthwise)
//
// The body has a top-level `if TRANSPOSE == 0u32` branch for the direct
// gather pattern (anchor + walk) vs the transpose gather form
// (`(opp − tap) / stride`). Within each branch, the sub-axis gates the
// per-path differences:
//   * direct: DILATED=0 → `p0 + kx`, DILATED=1 → `p0 + kx * dilation`
//   * transpose: DEPTHWISE=0 → full `for ic { for kx }`,
//               DEPTHWISE=1 → depthwise `for kx` (no ic loop, per-channel weight)
//
// The `dilation` constexpr is OPTIONAL via the new
// `#[constexpr(only_when = "…")]` attribute syntax: it is stripped from
// the MSL signature for the single (T=0, D=0) row where the body does
// not reference it, so the host doesn't need to provide a buffer slot
// for that one kernel. All other rows keep `dilation`.
//
// The original source files (`audio_conv1d.rs`, `conv1d_dilated_transpose.rs`,
// `conv1d_transpose_depthwise.rs`) have been removed; this consolidated module
// is the sole implementation of all four dense 1D conv variants.

/// Combined dense 1D conv: direct (with/without dilation) and transpose
/// (full / depthwise) in one function. The body dispatches on TRANSPOSE,
/// then on DILATED (direct path) or DEPTHWISE (transpose path).
#[kernel(variants(
    // 4-row cartesian product: (T, D, W) = (0,0,0), (0,1,0), (1,0,0), (1,0,1).
    // The skipped (T, D, W) cells are equivalent to a kept row (e.g.
    // (0,0,1) ≡ (0,0,0) since DEPTHWISE doesn't apply to direct), so we
    // skip them to keep the variant count minimal.
    TRANSPOSE = [0u32,   0u32,    1u32,       1u32     ],
    DILATED   = [0u32,   1u32,    0u32,       0u32     ],
    DEPTHWISE = [0u32,   0u32,    0u32,       1u32     ],
    VARIANT   = [direct, dilated, transpose,  depthwise],
    suffix = "{VARIANT}",
))]
#[allow(clippy::too_many_arguments)]
pub fn mt_conv1d_dense<T>(
    input: Tensor<T>,
    weight: Tensor<T>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] in_ch: u32,
    #[constexpr] channels: u32,
    #[constexpr] in_len: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_len: u32,
    #[constexpr] k: u32,
    #[constexpr] stride: u32,
    #[constexpr] pad: u32,
    // Stripped only for the (T=0, D=0) row — direct non-dilated.
    // All other rows reference `dilation` (direct-dilated uses `kx * dilation`,
    // both transpose paths use `kx * dilation` for the gather arithmetic).
    #[constexpr(only_when = "DILATED == 1u32 || TRANSPOSE == 1u32")] dilation: u32,
) {
    let idx = program_id::<0>();
    let op = idx % out_len;
    // Body parser lowers `if` to `Op::If` (no value); the active branch
    // updates the pre-declared `acc` accumulator. After FMT-pruning only
    // one branch survives per variant.
    let mut acc: f32 = 0.0f32;
    if TRANSPOSE == 0u32 {
        // Direct form: anchor+walk, output position op walks forward through
        // input positions covered by the kernel.
        let t1 = idx / out_len;
        let oc = t1 % out_ch;
        let n = t1 / out_ch;
        let p0 = op * stride;
        let in_n_stride = in_ch * in_len;
        let w_oc_stride = in_ch * k;
        acc = load(bias[oc]).cast::<f32>();
        for ic in range(0u32, in_ch, 1u32) {
            let in_ic_base = n * in_n_stride + ic * in_len;
            let w_ic_base = oc * w_oc_stride + ic * k;
            for kx in range(0u32, k, 1u32) {
                let p = if DILATED == 0u32 { p0 + kx } else { p0 + kx * dilation };
                let valid = (p >= pad) & (p < pad + in_len);
                let ix = select(valid, p - pad, 0u32);
                let x = load(input[in_ic_base + ix]).cast::<f32>();
                let x_m = select(valid, x, 0.0f32);
                let wt = load(weight[w_ic_base + kx]).cast::<f32>();
                acc = acc + x_m * wt;
            }
        }
    } else {
        // Transpose form: gather (opp − tap) / stride. `opp = op + pad`.
        let opp = op + pad;
        if DEPTHWISE == 0u32 {
            // Full transpose: oc, n from idx, sum over input channels.
            let t1 = idx / out_len;
            let oc = t1 % out_ch;
            let n = t1 / out_ch;
            let in_n_stride = in_ch * in_len;
            let w_in_stride = out_ch * k;
            acc = load(bias[oc]).cast::<f32>();
            for ic in range(0u32, in_ch, 1u32) {
                let in_ic_base = n * in_n_stride + ic * in_len;
                let w_ic_base = ic * w_in_stride + oc * k;
                for kx in range(0u32, k, 1u32) {
                    let tap = kx * dilation;
                    let has = opp >= tap;
                    let num = select(has, opp - tap, 0u32);
                    let on_grid = (num % stride) == 0u32;
                    let ip = num / stride;
                    let valid = has & on_grid & (ip < in_len);
                    let ix = select(valid, ip, 0u32);
                    let x = load(input[in_ic_base + ix]).cast::<f32>();
                    let x_m = select(valid, x, 0.0f32);
                    let wt = load(weight[w_ic_base + kx]).cast::<f32>();
                    acc = acc + x_m * wt;
                }
            }
        } else {
            // Depthwise transpose: c from idx, no input-channel sum,
            // single per-channel weight of length k.
            let c = idx / out_len;
            let in_base = c * in_len;
            let w_base = c * k;
            acc = load(bias[c]).cast::<f32>();
            for kx in range(0u32, k, 1u32) {
                let tap = kx * dilation;
                let has = opp >= tap;
                let num = select(has, opp - tap, 0u32);
                let on_grid = (num % stride) == 0u32;
                let ip = num / stride;
                let valid = has & on_grid & (ip < in_len);
                let ix = select(valid, ip, 0u32);
                let x = load(input[in_base + ix]).cast::<f32>();
                let x_m = select(valid, x, 0.0f32);
                let wt = load(weight[w_base + kx]).cast::<f32>();
                acc = acc + x_m * wt;
            }
        }
    }
    store(out[idx], acc.cast::<T>());
}

// ─── § Block-scaled audio_conv1d + fishspeech_conv1d dispatch ──────────────
// A single variants block covers all 19 FMTs and both DILATED values
// (DILATED=0 → audio_conv1d-style, DILATED=1 → fishspeech_conv1d-style).
// The `dilation` constexpr is OPTIONAL via the new `#[constexpr(only_when)]`
// attribute syntax: it's stripped from the signature for DILATED=0 variants
// so the host doesn't need to provide a buffer slot, and the body parser
// never sees a reference to it (the `if DILATED == 0u32` branch is
// FMT-pruned away). The body is otherwise identical for both axes — the
// only difference is `p = p0 + kx` vs `p = p0 + kx * dilation`. `BITS`,
// `WT`, `ST` co-vary with FMT and are tracked by the macro for use in
// expressions (type-params cannot appear in compile-time `if`, so the FMT
// int is the gate).

#[kernel(variants(
    // (FMT, BITS, WT, ST) co-vary; DILATED is a cross axis producing 19×2=38 kernels.
    // Named labels give readable names: mt_conv1d_quant_audio_mxfp4, …fishspeech_int8.
    // FMT integer (0–18, position-based) gates the compile-time if dispatch tree.
    // DILATED: audio=0, fishspeech=1 — matches `if DILATED == 0u32 / 1u32` in body.
    (FMT,         BITS,  WT,  ST ) = [
        (mxfp4,      4u32, u32, u8 ),
        (nvfp4,      4u32, u32, u8 ),
        (mxint2,     2u32, u32, u8 ),
        (mxint3,     3u32, u32, u8 ),
        (mxint4,     4u32, u32, u8 ),
        (mxint5,     5u32, u32, u8 ),
        (mxint6,     6u32, u32, u8 ),
        (fp4,        4u32, u32, f32),
        (int2,       2u32, u32, f32),
        (int3,       3u32, u32, f32),
        (int4,       4u32, u32, f32),
        (int5,       5u32, u32, f32),
        (int6,       6u32, u32, f32),
        (mxfp8,      8u32, u8,  u8 ),
        (mxfp8_e5m2, 8u32, u8,  u8 ),
        (mxint8,     8u32, u8,  u8 ),
        (fp8_e5m2,   8u32, u8,  f32),
        (nvfp8,      8u32, u8,  f32),
        (int8,       8u32, u8,  f32),
    ],
    DILATED = cross[audio, fishspeech],
    suffix = "{DILATED}_{FMT}",
))]
/// Combined quantised-weight 1D conv: the 19 block-scaled formats applied
/// to both the direct (audio) and dilated (fishspeech) paths. The
/// `dilation` constexpr is OPTIONAL — stripped for DILATED=audio and
/// kept for DILATED=fishspeech. The body has a nested `if FMT == N { }`
/// tree pruned at variants-substitution time since type-params (WT, ST)
/// cannot appear in compile-time `if` conditions.
#[allow(clippy::too_many_arguments)]
pub fn mt_conv1d_quant<T>(
    input: Tensor<T>,
    weight: Tensor<WT>,
    scales: Tensor<ST>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] in_ch: u32,
    #[constexpr] in_len: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_len: u32,
    #[constexpr] k: u32,
    #[constexpr] stride: u32,
    #[constexpr] pad: u32,
    #[constexpr(only_when = "DILATED == 1u32")] dilation: u32,
    #[constexpr] block_size: u32,
    // nvfp4 (FMT=1) uses a per-tensor global scale factor multiplied onto each
    // per-block E4M3 micro-scale.  Stripped for all other FMTs.
    #[constexpr(only_when = "FMT == 1u32")] global: f32,
) {
    let idx = program_id::<0>();
    let op = idx % out_len;
    let t1 = idx / out_len;
    let oc = t1 % out_ch;
    let n = t1 / out_ch;
    let p0 = op * stride;
    let in_n_stride = in_ch * in_len;
    let c_dim = in_ch * k;
    let w_row_word_base = oc * (c_dim * BITS / 32u32);
    let w_row_blk = oc * (c_dim / block_size);
    let mut acc = load(bias[oc]).cast::<f32>();
    // Sign-extend constants for sub-byte ints (FMT 2-6, 8-12). For nibble/byte
    // formats (FMT 0,1,7 BITS=4; FMT 13-18 BITS=8) the sign-extend path is
    // dead — those formats bypass the bitstream decode and go through the
    // dedicated `mt_decode_e2m1` / `mt_decode_e4m3` / `mt_decode_e5m2` /
    // `mt_decode_int8` primitives instead. The half/full terms remain live but
    // the optimizer DCEs them.
    let half = 1u32 << (BITS - 1u32);
    let full = (1u32 << BITS).cast::<f32>();
    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * in_len;
        let col_ic = ic * k;
        for kx in range(0u32, k, 1u32) {
            let p = if DILATED == 0u32 { p0 + kx } else { p0 + kx * dilation };
            let valid = (p >= pad) & (p < pad + in_len);
            let ix = select(valid, p - pad, 0u32);
            let x = load(input[in_ic_base + ix]).cast::<f32>();
            let x_m = select(valid, x, 0.0f32);
            let col = col_ic + kx;

            // Element decode — two families selected by FMT:
            //   * FMT 0..12: u32 weight, BITS-bit codes (nibbles for BITS=4,
            //     sub-byte ints for BITS=2..6). The straddle-aware
            //     `mt_unpack_nbit` extracts the BITS-wide code; nibble formats
            //     (FMT 0,1,7) then go through `mt_decode_e2m1`, sub-byte ints
            //     sign-extend via `half`/`full` below.
            //   * FMT 13..18: u8 weight, one code per byte. Routed to
            //     `mt_decode_e4m3` / `mt_decode_e5m2` / `mt_decode_int8` by
            //     FMT.
            let elem = if FMT <= 12u32 {
                let bit_off = col * BITS;
                let word_idx = bit_off / 32u32;
                let bit_in_w = bit_off & 31u32;
                let lo_bits = select(32u32 - bit_in_w >= BITS, BITS, 32u32 - bit_in_w);
                let spill = BITS - lo_bits;
                let w0 = load(weight[w_row_word_base + word_idx]);
                let w1 =
                    load(weight[w_row_word_base + select(spill > 0u32, word_idx + 1u32, word_idx)]);
                let q = mt_unpack_nbit(w0, w1, bit_in_w, lo_bits, spill);
                if FMT <= 1u32 || FMT == 7u32 {
                    // Nibble E2M1 (BITS=4, byte-aligned, spill=0).
                    mt_decode_e2m1(q)
                } else {
                    // Sub-byte int: sign-extend via half/full.
                    let qf = q.cast::<f32>();
                    select(q >= half, qf - full, qf)
                }
            } else {
                // FMT 13: mxfp8_e4m3, FMT 14: mxfp8_e5m2, FMT 15: mxint8,
                // FMT 16: fp8_e5m2, FMT 17: nvfp8, FMT 18: int8.
                // WT=u8, BITS=8 — one code per byte.  The byte-level row base
                // is `oc * c_dim`, NOT `w_row_word_base` which uses BITS=8 in
                // the formula `c_dim * BITS / 32 = c_dim / 4` and is 4× too small.
                let raw = load(weight[oc * c_dim + col]).cast::<u32>();
                if FMT == 13u32 || FMT == 17u32 {
                    mt_decode_e4m3(raw)
                } else if FMT == 14u32 || FMT == 16u32 {
                    mt_decode_e5m2(raw)
                } else {
                    mt_decode_int8(raw)
                }
            };

            // Scale decode — ST=u8 (E8M0 or E4M3) vs ST=f32 (direct load).
            //   * ST=u8, FMT 1 (nvfp4): E4M3 micro-scale × `global` constexpr.
            //   * ST=u8, FMT 0 or 2-6 (mxfp4 / mxintN): E8M0 pow-2 scale
            //     (`mt_decode_e8m0` → 2^(b-127)).
            //   * ST=u8, FMT 13-15 (mxfp8 / mxint8): E8M0.
            //   * ST=f32 (FMT 7-12, 16-18): direct load, no decode.
            let scale = if FMT <= 6u32 {
                if FMT == 1u32 {
                    mt_decode_e4m3(load(scales[w_row_blk + col / block_size]).cast::<u32>())
                        * global
                } else {
                    mt_decode_e8m0(load(scales[w_row_blk + col / block_size]).cast::<u32>())
                }
            } else if FMT >= 13u32 && FMT <= 15u32 {
                mt_decode_e8m0(load(scales[w_row_blk + col / block_size]).cast::<u32>())
            } else {
                load(scales[w_row_blk + col / block_size])
            };

            acc = acc + x_m * (elem * scale);
        }
    }
    store(out[idx], acc.cast::<T>());
}

// ─── § Shared block-scaled test helper ────────────────────────────────────────
// Helper that mirrors the original `audio_conv1d_block_scaled::conv1d_setup` /
// `conv1d_bench` logic. Used by both `kernel_tests` and `kernel_benches` so
// they share a single source of truth for buffer types / constexprs.
//
// This function is plain Rust (not a #[kernel] body) — it runs on the CPU
// during test setup, building the buffers that get fed into the GPU
// kernel. Variants substitution in `#[test_kernel(variants(...))]` and
// `#[bench(variants(...))]` rewrites the kernel-module name in the caller
// before this helper sees it.
#[allow(clippy::too_many_arguments)]
fn blockscaled_setup(
    kernel: metaltile::core::ir::Kernel,
    fmt: crate::quant::format::QFormat,
    batch: usize,
    in_ch: usize,
    in_len: usize,
    out_ch: usize,
    k: usize,
    stride: usize,
    pad: usize,
    dilation: usize,
    dt: metaltile::core::DType,
) -> metaltile::harness::test::TestSetup {
    use metaltile::{
        core::{DType, ir::KernelMode},
        harness::test::{TestBuffer, TestSetup},
    };
    let out_len = (in_len + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
    let n_out = batch * out_ch * out_len;
    let c_dim = in_ch * k;
    let input_f: Vec<f32> =
        (0..(batch * in_ch * in_len)).map(|i| ((i % 13) as f32 / 13.0 - 0.5) * 6.0).collect();
    let bias_f: Vec<f32> = (0..out_ch).map(|i| ((i % 5) as f32 / 5.0 - 0.5) * 2.0).collect();
    let weight_f: Vec<f32> =
        (0..(out_ch * c_dim)).map(|i| ((i % 11) as f32 / 11.0 - 0.5) * 4.0).collect();
    let p = crate::quant::format::pack(fmt, &weight_f, out_ch, c_dim);
    let wdq = crate::quant::format::dequant(fmt, &p, out_ch, c_dim);
    let input = crate::utils::unpack_f32(&crate::utils::pack_f32(&input_f, dt), dt);
    let bias = crate::utils::unpack_f32(&crate::utils::pack_f32(&bias_f, dt), dt);
    let expected = {
        // CPU oracle: dense direct conv with dequantized weights.
        let mut out = vec![0.0f32; batch * out_ch * out_len];
        for n in 0..batch {
            for oc in 0..out_ch {
                for op in 0..out_len {
                    let mut acc = bias[oc];
                    for ic in 0..in_ch {
                        for kx in 0..k {
                            let p = op * stride + kx * dilation;
                            if p < pad || p >= pad + in_len {
                                continue;
                            }
                            let ix = p - pad;
                            let in_idx = (n * in_ch + ic) * in_len + ix;
                            let col = ic * k + kx;
                            let w_idx = oc * c_dim + col;
                            acc += input[in_idx] * wdq[w_idx];
                        }
                    }
                    out[(n * out_ch + oc) * out_len + op] = acc;
                }
            }
        }
        out
    };
    let weight_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
    let scales_dt = match fmt.scale_kind() {
        crate::quant::format::ScaleKind::F32 => DType::F32,
        crate::quant::format::ScaleKind::F16 => DType::F16,
        _ => DType::U8,
    };
    let mut s = TestSetup::new(kernel)
        .mode(KernelMode::Grid3D)
        .input(TestBuffer::from_vec("input", crate::utils::pack_f32(&input_f, dt), dt))
        .input(TestBuffer::from_vec("weight", p.codes, weight_dt))
        .input(TestBuffer::from_vec("scales", p.scales, scales_dt))
        .input(TestBuffer::from_vec("bias", crate::utils::pack_f32(&bias_f, dt), dt))
        .input(TestBuffer::zeros("out", n_out, dt))
        .constexpr("batch", batch as u32)
        .constexpr("in_ch", in_ch as u32)
            // `channels` is read only by the DEPTHWISE variant, but it stays
            // in every variant's signature — the CUDA/HIP/Vulkan dispatchers
            // bind constexprs strictly by the kernel's declared list and
            // error on a missing one (Metal tolerates the gap).
            .constexpr("channels", in_ch as u32)
        .constexpr("in_len", in_len as u32)
        .constexpr("out_ch", out_ch as u32)
        .constexpr("out_len", out_len as u32)
        .constexpr("k", k as u32)
        .constexpr("stride", stride as u32)
        .constexpr("pad", pad as u32)
        .constexpr("block_size", fmt.block_size() as u32);
    // DILATED=1 kernels keep `dilation` in their MSL signature; DILATED=0 strips it.
    if dilation > 1 {
        s = s.constexpr("dilation", dilation as u32);
    }
    if matches!(fmt, crate::quant::format::QFormat::Nvfp4) {
        s = s.constexpr("global", p.global);
    }
    s.expect(TestBuffer::from_vec("out", crate::utils::pack_f32(&expected, dt), dt))
        .grid_1d(n_out, 256)
}

// ─── § tile tests (variants syntax) ───────────────────────────────────────────
// One #[test_kernel(variants(...))] per test pattern. The variants
// expansion emits one test registration per row, with the variant params
// (T/D/W or DILATED/FMT) substituted into the function body via FMT-pruning.
// The kernel module name embeds the same variant values.

pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    // `use super::*;` brings the kernel modules from the parent scope
    // (`mt_conv1d_dense_fmt*`, `mt_conv1d_block_scaled_*`) into the test
    // setup's namespace so the per-variant kernel references below resolve.
    use super::*;

    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    /// Direct 1D conv oracle (NCL input, OIK weight) — dilation=1 special case.
    #[allow(clippy::too_many_arguments)]
    fn naive_direct(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        batch: usize,
        in_ch: usize,
        in_len: usize,
        out_ch: usize,
        k: usize,
        stride: usize,
        pad: usize,
    ) -> Vec<f32> {
        let out_len = (in_len + 2 * pad - k) / stride + 1;
        let mut out = vec![0.0f32; batch * out_ch * out_len];
        for n in 0..batch {
            for oc in 0..out_ch {
                for op in 0..out_len {
                    let mut acc = bias[oc];
                    for ic in 0..in_ch {
                        for kx in 0..k {
                            let p = op * stride + kx;
                            if p < pad || p >= pad + in_len {
                                continue;
                            }
                            let ix = p - pad;
                            acc += input[(n * in_ch + ic) * in_len + ix]
                                * weight[(oc * in_ch + ic) * k + kx];
                        }
                    }
                    out[(n * out_ch + oc) * out_len + op] = acc;
                }
            }
        }
        out
    }

    /// Direct 1D conv (NCL input, OIK weight) — generic dilation.
    #[allow(clippy::too_many_arguments)]
    fn naive_dilated(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        batch: usize,
        in_ch: usize,
        in_len: usize,
        out_ch: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
    ) -> Vec<f32> {
        let out_len = (in_len + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
        let mut out = vec![0.0f32; batch * out_ch * out_len];
        for n in 0..batch {
            for oc in 0..out_ch {
                for op in 0..out_len {
                    let mut acc = bias[oc];
                    for ic in 0..in_ch {
                        for kx in 0..k {
                            let p = op * stride + kx * dilation;
                            if p < pad || p >= pad + in_len {
                                continue;
                            }
                            let ix = p - pad;
                            acc += input[(n * in_ch + ic) * in_len + ix]
                                * weight[(oc * in_ch + ic) * k + kx];
                        }
                    }
                    out[(n * out_ch + oc) * out_len + op] = acc;
                }
            }
        }
        out
    }

    /// Transposed 1D conv (NCL input, IOK weight) — full (per-channel sum).
    #[allow(clippy::too_many_arguments)]
    fn naive_transpose(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        batch: usize,
        in_ch: usize,
        in_len: usize,
        out_ch: usize,
        out_len: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; batch * out_ch * out_len];
        for n in 0..batch {
            for oc in 0..out_ch {
                for op in 0..out_len {
                    let mut acc = bias[oc];
                    let opp = op + pad;
                    for ic in 0..in_ch {
                        for kx in 0..k {
                            let tap = kx * dilation;
                            if opp < tap {
                                continue;
                            }
                            let num = opp - tap;
                            if !num.is_multiple_of(stride) {
                                continue;
                            }
                            let ip = num / stride;
                            if ip >= in_len {
                                continue;
                            }
                            acc += input[(n * in_ch + ic) * in_len + ip]
                                * weight[(ic * out_ch + oc) * k + kx];
                        }
                    }
                    out[(n * out_ch + oc) * out_len + op] = acc;
                }
            }
        }
        out
    }

    /// Depthwise transposed 1D conv (per-channel, no cross-channel sum).
    #[allow(clippy::too_many_arguments)]
    fn naive_transpose_depthwise(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        channels: usize,
        in_len: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
    ) -> Vec<f32> {
        let out_len = (in_len - 1) * stride + dilation * (k - 1) + 1 - 2 * pad;
        let mut out = vec![0.0f32; channels * out_len];
        for c in 0..channels {
            for op in 0..out_len {
                let mut acc = bias[c];
                let opp = op + pad;
                for kx in 0..k {
                    let tap = kx * dilation;
                    if opp < tap {
                        continue;
                    }
                    let num = opp - tap;
                    if !num.is_multiple_of(stride) {
                        continue;
                    }
                    let ip = num / stride;
                    if ip < in_len {
                        acc += input[c * in_len + ip] * weight[c * k + kx];
                    }
                }
                out[c * out_len + op] = acc;
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn direct_setup(
        kernel: Kernel,
        batch: usize,
        in_ch: usize,
        in_len: usize,
        out_ch: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dt: DType,
    ) -> TestSetup {
        let out_len = (in_len + 2 * pad - k) / stride + 1;
        let n_out = batch * out_ch * out_len;
        let input_f = ramp(batch * in_ch * in_len, 13, 6.0);
        let weight_f = ramp(out_ch * in_ch * k, 11, 4.0);
        let bias_f = ramp(out_ch, 5, 2.0);
        let input = crate::utils::unpack_f32(&crate::utils::pack_f32(&input_f, dt), dt);
        let weight = crate::utils::unpack_f32(&crate::utils::pack_f32(&weight_f, dt), dt);
        let bias = crate::utils::unpack_f32(&crate::utils::pack_f32(&bias_f, dt), dt);
        let expected =
            naive_direct(&input, &weight, &bias, batch, in_ch, in_len, out_ch, k, stride, pad);
        TestSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", crate::utils::pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("weight", crate::utils::pack_f32(&weight_f, dt), dt))
            .input(TestBuffer::from_vec("bias", crate::utils::pack_f32(&bias_f, dt), dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("batch", batch as u32)
            .constexpr("in_ch", in_ch as u32)
            // `channels` is read only by the DEPTHWISE variant, but it stays
            // in every variant's signature — the CUDA/HIP/Vulkan dispatchers
            // bind constexprs strictly by the kernel's declared list and
            // error on a missing one (Metal tolerates the gap).
            .constexpr("channels", in_ch as u32)
            .constexpr("in_len", in_len as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_len", out_len as u32)
            .constexpr("k", k as u32)
            .constexpr("stride", stride as u32)
            .constexpr("pad", pad as u32)
            // dilation is OPTIONAL — for (T=0, D=0) the host doesn't bind it.
            .expect(TestBuffer::from_vec("out", crate::utils::pack_f32(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }

    #[allow(clippy::too_many_arguments)]
    fn dilated_setup(
        kernel: Kernel,
        batch: usize,
        in_ch: usize,
        in_len: usize,
        out_ch: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
        dt: DType,
    ) -> TestSetup {
        let out_len = (in_len + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
        let n_out = batch * out_ch * out_len;
        let input_f = ramp(batch * in_ch * in_len, 13, 6.0);
        let weight_f = ramp(out_ch * in_ch * k, 11, 4.0);
        let bias_f = ramp(out_ch, 5, 2.0);
        let input = crate::utils::unpack_f32(&crate::utils::pack_f32(&input_f, dt), dt);
        let weight = crate::utils::unpack_f32(&crate::utils::pack_f32(&weight_f, dt), dt);
        let bias = crate::utils::unpack_f32(&crate::utils::pack_f32(&bias_f, dt), dt);
        let expected = naive_dilated(
            &input, &weight, &bias, batch, in_ch, in_len, out_ch, k, stride, pad, dilation,
        );
        TestSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", crate::utils::pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("weight", crate::utils::pack_f32(&weight_f, dt), dt))
            .input(TestBuffer::from_vec("bias", crate::utils::pack_f32(&bias_f, dt), dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("batch", batch as u32)
            .constexpr("in_ch", in_ch as u32)
            // `channels` is read only by the DEPTHWISE variant, but it stays
            // in every variant's signature — the CUDA/HIP/Vulkan dispatchers
            // bind constexprs strictly by the kernel's declared list and
            // error on a missing one (Metal tolerates the gap).
            .constexpr("channels", in_ch as u32)
            .constexpr("in_len", in_len as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_len", out_len as u32)
            .constexpr("k", k as u32)
            .constexpr("stride", stride as u32)
            .constexpr("pad", pad as u32)
            .constexpr("dilation", dilation as u32)
            .expect(TestBuffer::from_vec("out", crate::utils::pack_f32(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }

    #[allow(clippy::too_many_arguments)]
    fn transpose_setup(
        kernel: Kernel,
        batch: usize,
        in_ch: usize,
        in_len: usize,
        out_ch: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
        output_padding: usize,
        dt: DType,
    ) -> TestSetup {
        let out_len = (in_len - 1) * stride + dilation * (k - 1) + output_padding + 1 - 2 * pad;
        let n_out = batch * out_ch * out_len;
        let input_f = ramp(batch * in_ch * in_len, 13, 6.0);
        let weight_f = ramp(in_ch * out_ch * k, 11, 4.0);
        let bias_f = ramp(out_ch, 5, 2.0);
        let input = crate::utils::unpack_f32(&crate::utils::pack_f32(&input_f, dt), dt);
        let weight = crate::utils::unpack_f32(&crate::utils::pack_f32(&weight_f, dt), dt);
        let bias = crate::utils::unpack_f32(&crate::utils::pack_f32(&bias_f, dt), dt);
        let expected = naive_transpose(
            &input, &weight, &bias, batch, in_ch, in_len, out_ch, out_len, k, stride, pad, dilation,
        );
        TestSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", crate::utils::pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("weight", crate::utils::pack_f32(&weight_f, dt), dt))
            .input(TestBuffer::from_vec("bias", crate::utils::pack_f32(&bias_f, dt), dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("batch", batch as u32)
            .constexpr("in_ch", in_ch as u32)
            // `channels` is read only by the DEPTHWISE variant, but it stays
            // in every variant's signature — the CUDA/HIP/Vulkan dispatchers
            // bind constexprs strictly by the kernel's declared list and
            // error on a missing one (Metal tolerates the gap).
            .constexpr("channels", in_ch as u32)
            .constexpr("in_len", in_len as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_len", out_len as u32)
            .constexpr("k", k as u32)
            .constexpr("stride", stride as u32)
            .constexpr("pad", pad as u32)
            .constexpr("dilation", dilation as u32)
            .expect(TestBuffer::from_vec("out", crate::utils::pack_f32(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }

    fn depthwise_setup(
        kernel: Kernel,
        channels: usize,
        in_len: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dt: DType,
    ) -> TestSetup {
        let dilation = 1usize;
        let out_len = (in_len - 1) * stride + dilation * (k - 1) + 1 - 2 * pad;
        let n_out = channels * out_len;
        let input_f = ramp(channels * in_len, 17, 4.0);
        let weight_f = ramp(channels * k, 7, 2.0);
        let bias_f = ramp(channels, 5, 0.5);
        let input = crate::utils::unpack_f32(&crate::utils::pack_f32(&input_f, dt), dt);
        let weight = crate::utils::unpack_f32(&crate::utils::pack_f32(&weight_f, dt), dt);
        let bias = crate::utils::unpack_f32(&crate::utils::pack_f32(&bias_f, dt), dt);
        let expected = naive_transpose_depthwise(
            &input, &weight, &bias, channels, in_len, k, stride, pad, dilation,
        );
        TestSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", crate::utils::pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("weight", crate::utils::pack_f32(&weight_f, dt), dt))
            .input(TestBuffer::from_vec("bias", crate::utils::pack_f32(&bias_f, dt), dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("channels", channels as u32)
            // batch/in_ch/out_ch are read only by the non-depthwise variants
            // but stay in the shared signature — strict-binding dispatchers
            // (CUDA/HIP/Vulkan) need values for them too.
            .constexpr("batch", 1u32)
            .constexpr("in_ch", channels as u32)
            .constexpr("out_ch", channels as u32)
            .constexpr("in_len", in_len as u32)
            .constexpr("out_len", out_len as u32)
            .constexpr("k", k as u32)
            .constexpr("stride", stride as u32)
            .constexpr("pad", pad as u32)
            .constexpr("dilation", dilation as u32)
            .expect(TestBuffer::from_vec("out", crate::utils::pack_f32(&expected, dt), dt))
            .grid_1d(n_out, 256)
    }

    // The body dispatches on the variant params (TRANSPOSE, DILATED,
    // DEPTHWISE). After variants-substitution the unused branches are
    // FMT-pruned away. We use multi-character param names (vs single-letter
    // `T`/`D`/`W`) so the ident-embedder doesn't accidentally substitute
    // inside unrelated idents like `TestSetup` or `DType`.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 8e-3, 4e-2],
                  variants(
                      TRANSPOSE = [0u32,   0u32,    1u32,       1u32     ],
                      DILATED   = [0u32,   1u32,    0u32,       0u32     ],
                      DEPTHWISE = [0u32,   0u32,    0u32,       1u32     ],
                      VARIANT   = [direct, dilated, transpose,  depthwise],
                      suffix = "{VARIANT}",
                  ))]
    fn test_dense_conv1d(dt: DType) -> TestSetup {
        if TRANSPOSE == 0u32 {
            if DILATED == 0u32 {
                // direct — no dilation. Whisper stem conv #1.
                direct_setup(mt_conv1d_dense_direct::kernel_ir_for(dt), 1, 8, 50, 16, 3, 1, 1, dt)
            } else {
                // dilated — MRF ResBlock, dilation 3.
                dilated_setup(
                    mt_conv1d_dense_dilated::kernel_ir_for(dt),
                    1,
                    12,
                    60,
                    12,
                    3,
                    1,
                    3,
                    3,
                    dt,
                )
            }
        } else if DEPTHWISE == 0u32 {
            // transpose — HiFi-GAN 8× upsample.
            transpose_setup(
                mt_conv1d_dense_transpose::kernel_ir_for(dt),
                1,
                8,
                16,
                6,
                16,
                8,
                4,
                1,
                0,
                dt,
            )
        } else {
            // depthwise transpose — StyleTTS2 pool.
            depthwise_setup(mt_conv1d_dense_depthwise::kernel_ir_for(dt), 6, 13, 3, 2, 1, dt)
        }
    }

    // Block-scaled path: 4 representative FMTs (mxfp4 / nvfp4 / fp8_e5m2 /
    // int8) each on a different DILATED value. The 4-row axis enumerates 4
    // kernel registrations with one body. Multi-character param names
    // (`DIL`, `FMT`) avoid substring collisions with idents like `DType`
    // / `TestSetup` / `fmt` etc.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1],
                  variants(
                      DIL = [audio,      fishspeech, audio,     fishspeech],
                      FMT = [mxfp4,      nvfp4,      fp8_e5m2,  int8      ],
                      suffix = "{DIL}_{FMT}",
                  ))]
    fn test_blockscaled_conv1d(dt: DType) -> TestSetup {
        // Named FMT values in this mini-list: mxfp4=0, nvfp4=1, fp8_e5m2=2, int8=3.
        // Named DIL values: audio=0, fishspeech=1.
        let fmt = if FMT == 0u32 {
            crate::quant::format::QFormat::Mxfp4
        } else if FMT == 1u32 {
            crate::quant::format::QFormat::Nvfp4
        } else if FMT == 2u32 {
            crate::quant::format::QFormat::Fp8E5m2
        } else {
            crate::quant::format::QFormat::Int8
        };
        let kernel = if DIL == 0u32 {
            if FMT == 0u32 {
                mt_conv1d_quant_audio_mxfp4::kernel_ir_for(dt)
            } else if FMT == 1u32 {
                mt_conv1d_quant_audio_nvfp4::kernel_ir_for(dt)
            } else if FMT == 2u32 {
                mt_conv1d_quant_audio_fp8_e5m2::kernel_ir_for(dt)
            } else {
                mt_conv1d_quant_audio_int8::kernel_ir_for(dt)
            }
        } else {
            if FMT == 0u32 {
                mt_conv1d_quant_fishspeech_mxfp4::kernel_ir_for(dt)
            } else if FMT == 1u32 {
                mt_conv1d_quant_fishspeech_nvfp4::kernel_ir_for(dt)
            } else if FMT == 2u32 {
                mt_conv1d_quant_fishspeech_fp8_e5m2::kernel_ir_for(dt)
            } else {
                mt_conv1d_quant_fishspeech_int8::kernel_ir_for(dt)
            }
        };
        let dilation = if DIL == 1u32 { 2usize } else { 1usize };
        super::blockscaled_setup(kernel, fmt, 1, 8, 32, 8, 8, 1, 1, dilation, dt)
    }
}

// ─── § tile benches (variants syntax) ─────────────────────────────────────────
// Same pattern as tests: one #[bench(variants(...))] per bench shape,
// enumerating the per-variant kernel module name.

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    // `use super::*;` brings the per-variant kernel modules from the file's
    // top level into the bench setup's namespace.
    use super::*;

    // Whisper-large stem conv #2 — 1280 ch, len 1500, k=3, stride 2, pad 1.
    // No dilation → direct variant (dilation stripped from MSL signature).
    #[bench(dtypes = [f32, f16, bf16],
           variants(
               TRANSPOSE = [0u32,   0u32,    1u32,       1u32     ],
               DILATED   = [0u32,   1u32,    0u32,       0u32     ],
               DEPTHWISE = [0u32,   0u32,    0u32,       1u32     ],
               VARIANT   = [direct, dilated, transpose,  depthwise],
               suffix = "{VARIANT}",
           ))]
    fn bench_dense_conv1d(dt: DType) -> BenchSetup {
        let (batch, ch, in_len, k, stride, pad) =
            (1usize, 1280usize, 1500usize, 3usize, 2usize, 1usize);
        let out_len = (in_len + 2 * pad - k) / stride + 1;
        let n_out = batch * ch * out_len;
        let kernel = if TRANSPOSE == 0u32 {
            if DILATED == 0u32 {
                mt_conv1d_dense_direct::kernel_ir_for(dt)
            } else {
                mt_conv1d_dense_dilated::kernel_ir_for(dt)
            }
        } else if DEPTHWISE == 0u32 {
            mt_conv1d_dense_transpose::kernel_ir_for(dt)
        } else {
            mt_conv1d_dense_depthwise::kernel_ir_for(dt)
        };
        let s = BenchSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", batch * ch * in_len, dt))
            .buffer(BenchBuffer::random("weight", ch * ch * k, dt))
            .buffer(BenchBuffer::random("bias", ch, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("batch", batch as u32)
            .constexpr("in_ch", ch as u32)
            .constexpr("in_len", in_len as u32)
            .constexpr("out_ch", ch as u32)
            .constexpr("out_len", out_len as u32)
            .constexpr("k", k as u32)
            .constexpr("stride", stride as u32)
            .constexpr("pad", pad as u32);
        // dilation is OPTIONAL — only bind for the variants that keep it.
        let s =
            if !(TRANSPOSE == 0u32 && DILATED == 0u32) { s.constexpr("dilation", 1u32) } else { s };
        s.grid_1d(n_out, 256)
            .bytes_moved((n_out * dt.size_bytes()) as u64)
            .flops(2 * (batch as u64) * (ch as u64) * (out_len as u64) * (ch as u64) * (k as u64))
    }

    // Block-scaled bench — same 4-row DILATED×FMT axis as the test, but a
    // larger Whisper-large-like shape.
    #[bench(dtypes = [f32, f16, bf16],
           variants(
               DIL = [audio,      fishspeech, audio,     fishspeech],
               FMT = [mxfp4,      nvfp4,      fp8_e5m2,  int8      ],
               suffix = "{DIL}_{FMT}",
           ))]
    fn bench_blockscaled_conv1d(dt: DType) -> BenchSetup {
        let (batch, in_ch, in_len, out_ch, k, stride, pad) =
            (1usize, 128usize, 1024usize, 128usize, 8usize, 2usize, 1usize);
        let out_len = (in_len + 2 * pad - k) / stride + 1;
        let c_dim = in_ch * k;
        let n_out = batch * out_ch * out_len;
        // Named FMT values in this mini-list: mxfp4=0, nvfp4=1, fp8_e5m2=2, int8=3.
        // Named DIL values: audio=0, fishspeech=1.
        let fmt = if FMT == 0u32 {
            crate::quant::format::QFormat::Mxfp4
        } else if FMT == 1u32 {
            crate::quant::format::QFormat::Nvfp4
        } else if FMT == 2u32 {
            crate::quant::format::QFormat::Fp8E5m2
        } else {
            crate::quant::format::QFormat::Int8
        };
        let (codes_dt, codes_len) = if fmt.element_bits() == 8 {
            (DType::U8, out_ch * c_dim)
        } else {
            (DType::U32, crate::quant::format::bitstream_words(out_ch * c_dim, fmt.element_bits()))
        };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let n_blocks = out_ch * (c_dim / fmt.block_size());
        let sz = dt.size_bytes();
        let bytes = batch * in_ch * in_len * sz
            + codes_len * codes_dt.size_bytes()
            + n_blocks * scales_dt.size_bytes()
            + out_ch * sz
            + n_out * sz;
        let kernel = if DIL == 0u32 {
            if FMT == 0u32 {
                mt_conv1d_quant_audio_mxfp4::kernel_ir_for(dt)
            } else if FMT == 1u32 {
                mt_conv1d_quant_audio_nvfp4::kernel_ir_for(dt)
            } else if FMT == 2u32 {
                mt_conv1d_quant_audio_fp8_e5m2::kernel_ir_for(dt)
            } else {
                mt_conv1d_quant_audio_int8::kernel_ir_for(dt)
            }
        } else {
            if FMT == 0u32 {
                mt_conv1d_quant_fishspeech_mxfp4::kernel_ir_for(dt)
            } else if FMT == 1u32 {
                mt_conv1d_quant_fishspeech_nvfp4::kernel_ir_for(dt)
            } else if FMT == 2u32 {
                mt_conv1d_quant_fishspeech_fp8_e5m2::kernel_ir_for(dt)
            } else {
                mt_conv1d_quant_fishspeech_int8::kernel_ir_for(dt)
            }
        };
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", batch * in_ch * in_len, dt))
            .buffer(BenchBuffer::random("weight", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::random("bias", out_ch, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("batch", batch as u32)
            .constexpr("in_ch", in_ch as u32)
            // `channels` is read only by the DEPTHWISE variant, but it stays
            // in every variant's signature — the CUDA/HIP/Vulkan dispatchers
            // bind constexprs strictly by the kernel's declared list and
            // error on a missing one (Metal tolerates the gap).
            .constexpr("channels", in_ch as u32)
            .constexpr("in_len", in_len as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_len", out_len as u32)
            .constexpr("k", k as u32)
            .constexpr("stride", stride as u32)
            .constexpr("pad", pad as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if DIL == 1u32 {
            s = s.constexpr("dilation", 1u32);
        }
        if matches!(fmt, crate::quant::format::QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_1d(n_out, 256)
            .bytes_moved(bytes as u64)
            .flops(2 * (out_ch as u64) * (out_len as u64) * (c_dim as u64))
            .with_shape_label(format!(
                "{} oc={out_ch} lo={out_len} c={c_dim}{}",
                fmt.name(),
                if DIL == 1u32 { " dilated" } else { "" }
            ))
    }
}
