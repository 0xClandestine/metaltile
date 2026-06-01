//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Block-scaled quantization **formats** — the host-side packer + dequant
//! oracle built on the [`codec`](super::codec) bit primitives.
//!
//! A weight matrix `[rows, cols]` (row-major; `rows` = output dim `N`, `cols` =
//! contraction dim `K`) is quantized in contiguous **blocks along K**. Each
//! block stores: per-element codes (E2M1 / E4M3 / E5M2) + one block scale
//! (E8M0 / E4M3 / FP32). Two-level formats (nvfp4) additionally carry one global
//! FP32 so the per-block E4M3 micro-scales fit their range.
//!
//! | format   | element | block | block scale | global |
//! |----------|---------|-------|-------------|--------|
//! | nvfp4    | E2M1    | 16    | E4M3 (1 B)  | FP32   |
//! | mxfp4    | E2M1    | 32    | E8M0 (1 B)  | —      |
//! | mxfp8_e4 | E4M3    | 32    | E8M0 (1 B)  | —      |
//! | mxfp8_e5 | E5M2    | 32    | E8M0 (1 B)  | —      |
//! | nvfp8    | E4M3    | 16    | FP32 (4 B)  | —      |
//!
//! [`pack`] quantizes f32 weights → this layout; [`dequant`] reconstructs the
//! f32 matrix (the CPU correctness oracle). They share [`codec`], so the GPU
//! kernel — which emits the same `element_decode(code) * block_scale * global` —
//! is checked against a spec-exact reference, not a re-derivation that could
//! share a bug.

use super::codec;

/// E4M3's max finite magnitude (used as the nvfp4 micro-scale range).
const E4M3_MAX: f32 = 448.0;

/// A spec-conformant block-scaled weight format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QFormat {
    /// E2M1, block 16, E4M3 micro-scale + global FP32 (NVIDIA NVFP4).
    Nvfp4,
    /// E2M1, block 32, E8M0 pow-2 scale (OCP MXFP4).
    Mxfp4,
    /// E4M3, block 32, E8M0 pow-2 scale (OCP MXFP8).
    Mxfp8E4,
    /// E5M2, block 32, E8M0 pow-2 scale (OCP MXFP8).
    Mxfp8E5,
    /// E4M3, block 16, per-block FP32 scale (NVIDIA-style fp8).
    Nvfp8,
}

/// How a format stores its per-block scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScaleKind {
    /// 1 byte/block, pow-2 exponent. Effective scale `2^(bits-127)`.
    E8M0,
    /// 1 byte/block, E4M3 micro-scale; multiplied by the global FP32.
    E4M3,
    /// 4 bytes/block, raw little-endian f32.
    F32,
}

use QFormat::*;

impl QFormat {
    /// Elements per block along the K dimension.
    pub fn block_size(self) -> usize {
        match self {
            Nvfp4 | Nvfp8 => 16,
            Mxfp4 | Mxfp8E4 | Mxfp8E5 => 32,
        }
    }

    /// Bits per quantized element (4 for E2M1, 8 for E4M3/E5M2).
    pub fn element_bits(self) -> usize {
        match self {
            Nvfp4 | Mxfp4 => 4,
            Mxfp8E4 | Mxfp8E5 | Nvfp8 => 8,
        }
    }

    /// Short label for bench rows / shape strings.
    pub fn name(self) -> &'static str {
        match self {
            Nvfp4 => "nvfp4",
            Mxfp4 => "mxfp4",
            Mxfp8E4 => "mxfp8_e4m3",
            Mxfp8E5 => "mxfp8_e5m2",
            Nvfp8 => "nvfp8",
        }
    }

    /// Largest finite element magnitude — the block scale maps a block's amax to
    /// (roughly) this so the block uses the element's full dynamic range.
    fn element_max(self) -> f32 {
        match self {
            Nvfp4 | Mxfp4 => 6.0,        // E2M1 max codebook value
            Mxfp8E4 | Nvfp8 => E4M3_MAX, // E4M3 max
            Mxfp8E5 => 57344.0,          // E5M2 max
        }
    }

    fn scale_kind(self) -> ScaleKind {
        match self {
            Nvfp4 => ScaleKind::E4M3,
            Mxfp4 | Mxfp8E4 | Mxfp8E5 => ScaleKind::E8M0,
            Nvfp8 => ScaleKind::F32,
        }
    }

    /// Whether the format carries one global FP32 (two-level scaling).
    fn has_global(self) -> bool { matches!(self, Nvfp4) }

    fn element_encode(self, x: f32) -> u8 {
        match self {
            Nvfp4 | Mxfp4 => codec::e2m1_encode(x),
            Mxfp8E4 | Nvfp8 => codec::e4m3_encode(x),
            Mxfp8E5 => codec::e5m2_encode(x),
        }
    }

    fn element_decode(self, code: u8) -> f32 {
        match self {
            Nvfp4 | Mxfp4 => codec::e2m1_decode(code),
            Mxfp8E4 | Nvfp8 => codec::e4m3_decode(code),
            Mxfp8E5 => codec::e5m2_decode(code),
        }
    }
}

/// A quantized weight tensor in one [`QFormat`]'s byte layout.
#[derive(Debug, Clone)]
pub struct PackedTensor {
    /// Element codes. 4-bit formats pack 2 codes/byte (element `i` → byte `i/2`,
    /// low nibble for even `i`); 8-bit formats are 1 code/byte.
    pub codes: Vec<u8>,
    /// Per-block scales: 1 byte/block for E8M0/E4M3, 4 LE bytes/block for FP32.
    pub scales: Vec<u8>,
    /// Global FP32 scale (1.0 for single-level formats).
    pub global: f32,
}

/// Quantize a row-major `[rows, cols]` f32 weight matrix to `fmt`'s layout.
/// `cols` must be a multiple of `fmt.block_size()`.
pub fn pack(fmt: QFormat, w: &[f32], rows: usize, cols: usize) -> PackedTensor {
    assert_eq!(w.len(), rows * cols, "weight length must be rows*cols");
    let bs = fmt.block_size();
    assert_eq!(cols % bs, 0, "{}: cols {cols} not a multiple of block {bs}", fmt.name());
    let blocks_per_row = cols / bs;
    let nblocks = rows * blocks_per_row;

    // Per-block amax → the f32 block scale that maps amax to the element max.
    let mut block_scale = vec![0f32; nblocks];
    for r in 0..rows {
        for b in 0..blocks_per_row {
            let start = r * cols + b * bs;
            let amax = w[start..start + bs].iter().fold(0f32, |m, &v| m.max(v.abs()));
            block_scale[r * blocks_per_row + b] = amax / fmt.element_max();
        }
    }

    // Two-level (nvfp4): one global FP32 so the E4M3 micro-scales fit ±448.
    let global = if fmt.has_global() {
        let smax = block_scale.iter().fold(0f32, |m, &v| m.max(v));
        if smax > 0.0 { smax / E4M3_MAX } else { 1.0 }
    } else {
        1.0
    };

    let mut codes = vec![0u8; if fmt.element_bits() == 4 { rows * cols / 2 } else { rows * cols }];
    let mut scales =
        Vec::with_capacity(nblocks * if fmt.scale_kind() == ScaleKind::F32 { 4 } else { 1 });

    for r in 0..rows {
        for b in 0..blocks_per_row {
            let blk = r * blocks_per_row + b;
            // Store the scale, and recover the *effective* scale the dequant will
            // see (encoding is lossy for E8M0/E4M3 — quantize against what the
            // kernel will actually read, so codes + scale are self-consistent).
            let eff = match fmt.scale_kind() {
                ScaleKind::E8M0 => {
                    let bits = codec::e8m0_encode(block_scale[blk]);
                    scales.push(bits);
                    codec::e8m0_decode(bits)
                },
                ScaleKind::E4M3 => {
                    let bits = codec::e4m3_encode(block_scale[blk] / global);
                    scales.push(bits);
                    codec::e4m3_decode(bits) * global
                },
                ScaleKind::F32 => {
                    scales.extend_from_slice(&block_scale[blk].to_le_bytes());
                    block_scale[blk]
                },
            };
            let inv = if eff > 0.0 { 1.0 / eff } else { 0.0 };
            for e in 0..bs {
                let idx = r * cols + b * bs + e;
                let code = fmt.element_encode(w[idx] * inv);
                if fmt.element_bits() == 4 {
                    let byte = idx / 2;
                    if idx.is_multiple_of(2) {
                        codes[byte] = (codes[byte] & 0xF0) | (code & 0x0F);
                    } else {
                        codes[byte] = (codes[byte] & 0x0F) | ((code & 0x0F) << 4);
                    }
                } else {
                    codes[idx] = code;
                }
            }
        }
    }
    PackedTensor { codes, scales, global }
}

/// Decode a [`PackedTensor`] back to a row-major `[rows, cols]` f32 matrix — the
/// CPU correctness oracle. Mirrors exactly what a dequant kernel computes:
/// `element_decode(code) * block_scale * global`.
pub fn dequant(fmt: QFormat, p: &PackedTensor, rows: usize, cols: usize) -> Vec<f32> {
    let bs = fmt.block_size();
    let blocks_per_row = cols / bs;
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for b in 0..blocks_per_row {
            let blk = r * blocks_per_row + b;
            let eff = match fmt.scale_kind() {
                ScaleKind::E8M0 => codec::e8m0_decode(p.scales[blk]),
                ScaleKind::E4M3 => codec::e4m3_decode(p.scales[blk]) * p.global,
                ScaleKind::F32 => {
                    let o = blk * 4;
                    f32::from_le_bytes([
                        p.scales[o],
                        p.scales[o + 1],
                        p.scales[o + 2],
                        p.scales[o + 3],
                    ])
                },
            };
            for e in 0..bs {
                let idx = r * cols + b * bs + e;
                let code = if fmt.element_bits() == 4 {
                    let byte = p.codes[idx / 2];
                    if idx.is_multiple_of(2) { byte & 0x0F } else { byte >> 4 }
                } else {
                    p.codes[idx]
                };
                out[idx] = fmt.element_decode(code) * eff;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [QFormat; 5] = [Nvfp4, Mxfp4, Mxfp8E4, Mxfp8E5, Nvfp8];

    /// Deterministic weight matrix with per-row varying magnitude — exercises
    /// the per-block scaling (different blocks see different amax).
    fn weights(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| {
                let r = (i / cols) as f32;
                let c = (i % cols) as f32;
                // Sign + magnitude that varies along K and scales with the row.
                let mag = (1.0 + r * 0.5) * (0.1 + (c % 7.0) * 0.3);
                if (i % 3) == 0 { -mag } else { mag }
            })
            .collect()
    }

    /// Cosine similarity between two equal-length vectors.
    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0f64;
        let mut na = 0f64;
        let mut nb = 0f64;
        for (&x, &y) in a.iter().zip(b) {
            dot += x as f64 * y as f64;
            na += (x as f64).powi(2);
            nb += (y as f64).powi(2);
        }
        (dot / (na.sqrt() * nb.sqrt())) as f32
    }

    #[test]
    fn pack_dequant_preserves_direction() {
        // Block quantization is judged by *aggregate* fidelity (cosine
        // similarity), not per-element worst case — a small value sharing a
        // high-amax block rounds coarsely by design, but the dequantized matrix
        // still points the same way. This mirrors the cosine floor the GPU A/B
        // correctness checks use (DEFAULT_MIN_COSINE_SIM).
        let (rows, cols) = (8usize, 128usize); // divisible by 16 and 32
        let w = weights(rows, cols);
        for fmt in ALL {
            let p = pack(fmt, &w, rows, cols);
            let d = dequant(fmt, &p, rows, cols);
            assert_eq!(d.len(), w.len());
            let cos = cosine(&w, &d);
            // Floors reflect the format's precision: 4-bit E2M1 is coarse; the
            // mxfp8 E8M0 pow-2 scale leaves up to ~2× of the element range
            // unused (so it's looser than nvfp8's exact FP32 scale).
            let floor = match fmt {
                Nvfp4 | Mxfp4 => 0.97,     // 4-bit element
                Mxfp8E4 | Mxfp8E5 => 0.99, // 8-bit element + pow-2 scale
                Nvfp8 => 0.999,            // 8-bit element + exact FP32 scale
            };
            assert!(cos >= floor, "{}: cosine {cos} < {floor}", fmt.name());
        }
    }

    #[test]
    fn packed_byte_sizes_match_layout() {
        let (rows, cols) = (2usize, 32usize);
        let w = weights(rows, cols);
        for fmt in ALL {
            let p = pack(fmt, &w, rows, cols);
            let elems = rows * cols;
            let expected_codes = if fmt.element_bits() == 4 { elems / 2 } else { elems };
            assert_eq!(p.codes.len(), expected_codes, "{} codes", fmt.name());
            let nblocks = rows * (cols / fmt.block_size());
            let per_block = if fmt.scale_kind() == ScaleKind::F32 { 4 } else { 1 };
            assert_eq!(p.scales.len(), nblocks * per_block, "{} scales", fmt.name());
            if !fmt.has_global() {
                assert_eq!(p.global, 1.0, "{} global", fmt.name());
            }
        }
    }

    #[test]
    fn all_zero_block_dequants_to_zero() {
        let (rows, cols) = (1usize, 32usize);
        let w = vec![0f32; rows * cols];
        for fmt in ALL {
            let p = pack(fmt, &w, rows, cols);
            let d = dequant(fmt, &p, rows, cols);
            assert!(d.iter().all(|&v| v == 0.0), "{} zero block", fmt.name());
        }
    }
}
