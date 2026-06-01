//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Standalone dequant kernels for the spec-conformant block-scaled formats
//! (nvfp4 / mxfp4 / mxfp8 / nvfp8 — see `docs/BENCH_METRICS_SPEC.md` Appendix B).
//!
//! Each kernel reads packed element codes + per-block scales and writes the
//! reconstructed `[rows, cols]` matrix. The decode math mirrors
//! [`crate::quant::codec`] exactly, so the GPU output is checked against the
//! host [`crate::quant::format::dequant`] oracle — same reference, no drift.
//!
//! ## DISPATCH INVARIANTS (all kernels here)
//!
//! - **Mode: Grid3D (elementwise), one thread per output element.** Pure
//!   per-element decode with no cross-thread cooperation — `program_id::<0>()`
//!   is the global thread index. Dispatch `grid = [ceil(n/256), 1, 1]`,
//!   `tpg = [256, 1, 1]`; the `if i < n` guard covers the tail. Being Grid3D
//!   (not Reduction) it is *not* exposed to the `n_simd == 0` freeze hazard.
//! - **`block_size`** is the format's K-block (16 or 32) and must divide `cols`.
//! - 4-bit codes pack 8 nibbles per `u32` (little-endian: element `i` → word
//!   `i/8`, nibble shift `(i & 7) * 4`). 8-bit codes are one `uchar` each.

use metaltile::kernel;

/// mxfp4 — E2M1 elements (block 32), E8M0 pow-2 block scale.
/// `scales[b]` is the biased exponent; effective scale `2^(bits - 127)`.
#[kernel]
pub fn mt_mxfp4_dequant<T>(
    codes: Tensor<u32>,
    scales: Tensor<u8>,
    out: Tensor<T>,
    #[constexpr] n: u32,
    #[constexpr] block_size: u32,
) {
    let i = program_id::<0>();
    if i < n {
        let word = load(codes[i / 8u32]);
        let nib = (word >> ((i & 7u32) * 4u32)) & 0xFu32;
        let m = nib & 0x7u32;
        // E2M1 codebook {0, .5, 1, 1.5, 2, 3, 4, 6} indexed by the low 3 bits.
        let mag = select(
            m < 1u32,
            0.0f32,
            select(
                m < 2u32,
                0.5f32,
                select(
                    m < 3u32,
                    1.0f32,
                    select(
                        m < 4u32,
                        1.5f32,
                        select(
                            m < 5u32,
                            2.0f32,
                            select(m < 6u32, 3.0f32, select(m < 7u32, 4.0f32, 6.0f32)),
                        ),
                    ),
                ),
            ),
        );
        let val = select((nib & 0x8u32) > 0u32, -mag, mag);
        let sbits = load(scales[i / block_size]).cast::<f32>();
        let scale = exp2(sbits - 127.0f32); // E8M0: 2^(bits-127), exact for integer bits
        store(out[i], (val * scale).cast::<T>());
    }
}

pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{quant::format::QFormat, utils::pack_f32};

    /// Deterministic f32 weights with magnitude varying along K (so per-block
    /// scales differ) and mixed signs.
    fn weights(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| {
                let r = (i / cols) as f32;
                let c = (i % cols) as f32;
                let mag = (1.0 + r * 0.5) * (0.1 + (c % 11.0) * 0.25);
                if (i % 3) == 0 { -mag } else { mag }
            })
            .collect()
    }

    /// Shared setup: pack `[rows, cols]` weights in `fmt`, dispatch the dequant
    /// kernel, and expect the host oracle's reconstruction. Kernel and oracle
    /// share `quant::codec`, so the match is near-exact.
    fn dequant_setup(
        kernel: Kernel,
        fmt: QFormat,
        rows: usize,
        cols: usize,
        dt: DType,
    ) -> TestSetup {
        let w = weights(rows, cols);
        let p = crate::quant::format::pack(fmt, &w, rows, cols);
        let oracle = crate::quant::format::dequant(fmt, &p, rows, cols);
        let n = rows * cols;
        const TPG: u32 = 256;
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("codes", p.codes, DType::U32))
            .input(TestBuffer::from_vec("scales", p.scales, DType::U8))
            .input(TestBuffer::zeros("out", n, dt))
            .constexpr("n", n as u32)
            .constexpr("block_size", fmt.block_size() as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&oracle, dt), dt))
            .grid_3d((n as u32).div_ceil(TPG), 1, 1, [TPG, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 5e-2, 2e-1])]
    fn test_mxfp4_dequant(dt: DType) -> TestSetup {
        dequant_setup(mt_mxfp4_dequant::kernel_ir_for(dt), QFormat::Mxfp4, 4, 64, dt)
    }
}
