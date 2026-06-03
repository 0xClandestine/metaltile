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
    /// E2M1, group 32, per-group FP32 scale (legacy float-scale fp4).
    Fp4,
    /// E4M3, group 32, per-group FP32 scale (legacy float-scale fp8).
    Fp8E4m3,
    /// E5M2, group 32, per-group FP32 scale (legacy float-scale fp8).
    Fp8E5m2,
    /// Symmetric int8, group 64, per-group FP32 scale (affine, scale-only).
    Int8,
    // ── Symmetric integers, group 64, per-group FP32 scale ──────────────────
    // The integer counterpart of the legacy float-scale family: a plain signed
    // N-bit element × an exact FP32 group scale, no zero-point. Distinct from
    // the affine track (Track 2), which carries a zero-point/bias for MLX
    // checkpoint interop. int8 above is the N=8 member.
    /// Symmetric int2, group 64, per-group FP32 scale.
    Int2,
    /// Symmetric int3, group 64, per-group FP32 scale.
    Int3,
    /// Symmetric int4, group 64, per-group FP32 scale.
    Int4,
    /// Symmetric int5, group 64, per-group FP32 scale.
    Int5,
    /// Symmetric int6, group 64, per-group FP32 scale.
    Int6,
    // ── MXINT: symmetric integers, block 32, E8M0 pow-2 scale (OCP MX-style) ─
    // The integer members of the MX family — a signed N-bit element × an E8M0
    // power-of-two block scale (block 32), hardware-mappable to tensor-core
    // block-scaling units. MXINT8 is OCP-ratified; the other widths follow the
    // same construction.
    /// Symmetric int2, block 32, E8M0 pow-2 scale (MXINT2).
    Mxint2,
    /// Symmetric int3, block 32, E8M0 pow-2 scale (MXINT3).
    Mxint3,
    /// Symmetric int4, block 32, E8M0 pow-2 scale (MXINT4).
    Mxint4,
    /// Symmetric int5, block 32, E8M0 pow-2 scale (MXINT5).
    Mxint5,
    /// Symmetric int6, block 32, E8M0 pow-2 scale (MXINT6).
    Mxint6,
    /// Symmetric int8, block 32, E8M0 pow-2 scale (MXINT8, OCP-ratified).
    Mxint8,
    // ── FP16-scale twins of the FP32-scaled formats ─────────────────────────
    // Same element + block size as their FP32-scaled twin, but the per-block
    // scale is stored as an IEEE half (2 B vs 4 B) — the layout real checkpoints
    // use, halving scale-buffer traffic at negligible scale precision cost.
    /// E4M3, block 16, per-block FP16 scale (nvfp8 twin).
    Nvfp8F16,
    /// E2M1, group 32, per-group FP16 scale (legacy fp4 twin).
    Fp4F16,
    /// E4M3, group 32, per-group FP16 scale (legacy fp8 e4m3 twin).
    Fp8E4m3F16,
    /// E5M2, group 32, per-group FP16 scale (legacy fp8 e5m2 twin).
    Fp8E5m2F16,
    /// Symmetric int2, group 64, per-group FP16 scale.
    Int2F16,
    /// Symmetric int3, group 64, per-group FP16 scale.
    Int3F16,
    /// Symmetric int4, group 64, per-group FP16 scale.
    Int4F16,
    /// Symmetric int5, group 64, per-group FP16 scale.
    Int5F16,
    /// Symmetric int6, group 64, per-group FP16 scale.
    Int6F16,
    /// Symmetric int8, group 64, per-group FP16 scale.
    Int8F16,
}

/// A format's quantized **element** type — one of the three orthogonal axes
/// (element · scale · zero-point) the formats vary along.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Element {
    /// Signed N-bit integer (the element is the integer itself).
    Int(u32),
    /// E2M1 micro-float (4-bit codebook).
    E2m1,
    /// E4M3 micro-float (8-bit).
    E4m3,
    /// E5M2 micro-float (8-bit).
    E5m2,
}

/// How a format stores its per-block scale — the scale axis of the
/// element · scale · zero-point model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleKind {
    /// 1 byte/block, pow-2 exponent. Effective scale `2^(bits-127)`.
    E8M0,
    /// 1 byte/block, E4M3 micro-scale; multiplied by the global FP32.
    E4M3,
    /// 4 bytes/block, raw little-endian f32.
    F32,
    /// 2 bytes/block, IEEE half (little-endian) — the memory-halving FP32 twin.
    F16,
}

impl ScaleKind {
    /// Bytes one block's scale occupies in the packed buffer.
    pub fn bytes(self) -> usize {
        match self {
            ScaleKind::E8M0 | ScaleKind::E4M3 => 1,
            ScaleKind::F16 => 2,
            ScaleKind::F32 => 4,
        }
    }
}

use QFormat::*;

impl QFormat {
    /// The element axis: integer width or micro-float codebook.
    pub fn element(self) -> Element {
        match self {
            Nvfp4 | Mxfp4 | Fp4 | Fp4F16 => Element::E2m1,
            Mxfp8E4 | Nvfp8 | Fp8E4m3 | Nvfp8F16 | Fp8E4m3F16 => Element::E4m3,
            Mxfp8E5 | Fp8E5m2 | Fp8E5m2F16 => Element::E5m2,
            Int2 | Mxint2 | Int2F16 => Element::Int(2),
            Int3 | Mxint3 | Int3F16 => Element::Int(3),
            Int4 | Mxint4 | Int4F16 => Element::Int(4),
            Int5 | Mxint5 | Int5F16 => Element::Int(5),
            Int6 | Mxint6 | Int6F16 => Element::Int(6),
            Int8 | Mxint8 | Int8F16 => Element::Int(8),
        }
    }

    /// Whether the element is a signed integer (vs a micro-float codebook).
    pub fn is_integer(self) -> bool { matches!(self.element(), Element::Int(_)) }

    /// The zero-point axis. Every Track-1 (block-scaled / float-scale) format is
    /// **symmetric** — no zero-point. The asymmetric integer track (zero-point
    /// for MLX-checkpoint interop) lives separately in `mlx/quantized.rs`.
    pub fn symmetric(self) -> bool { true }

    /// Elements per block (mx*) / group (nv*, legacy fp, FP32-scaled int) along K.
    pub fn block_size(self) -> usize {
        match self {
            Nvfp4 | Nvfp8 | Nvfp8F16 => 16,
            Mxfp4 | Mxfp8E4 | Mxfp8E5 | Fp4 | Fp8E4m3 | Fp8E5m2 => 32,
            Fp4F16 | Fp8E4m3F16 | Fp8E5m2F16 => 32,
            // MXINT shares the MX block size (32); FP32-scaled symmetric ints
            // use the int group size (64, matching int8 + the affine track).
            Mxint2 | Mxint3 | Mxint4 | Mxint5 | Mxint6 | Mxint8 => 32,
            Int2 | Int3 | Int4 | Int5 | Int6 | Int8 => 64,
            Int2F16 | Int3F16 | Int4F16 | Int5F16 | Int6F16 | Int8F16 => 64,
        }
    }

    /// Bits per quantized element (2–6/8 for ints, 4 for E2M1, 8 for E4M3/E5M2).
    pub fn element_bits(self) -> usize {
        match self.element() {
            Element::Int(n) => n as usize,
            Element::E2m1 => 4,
            Element::E4m3 | Element::E5m2 => 8,
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
            Fp4 => "fp4",
            Fp8E4m3 => "fp8_e4m3",
            Fp8E5m2 => "fp8_e5m2",
            Int2 => "int2",
            Int3 => "int3",
            Int4 => "int4",
            Int5 => "int5",
            Int6 => "int6",
            Int8 => "int8",
            Mxint2 => "mxint2",
            Mxint3 => "mxint3",
            Mxint4 => "mxint4",
            Mxint5 => "mxint5",
            Mxint6 => "mxint6",
            Mxint8 => "mxint8",
            Nvfp8F16 => "nvfp8_f16",
            Fp4F16 => "fp4_f16",
            Fp8E4m3F16 => "fp8_e4m3_f16",
            Fp8E5m2F16 => "fp8_e5m2_f16",
            Int2F16 => "int2_f16",
            Int3F16 => "int3_f16",
            Int4F16 => "int4_f16",
            Int5F16 => "int5_f16",
            Int6F16 => "int6_f16",
            Int8F16 => "int8_f16",
        }
    }

    /// Largest finite element magnitude — the block/group scale maps a block's
    /// amax to (roughly) this so the block uses the element's full range.
    fn element_max(self) -> f32 {
        match self.element() {
            Element::Int(n) => codec::intn_max(n) as f32, // 2^(N-1) − 1
            Element::E2m1 => 6.0,                         // E2M1 max codebook value
            Element::E4m3 => E4M3_MAX,                    // E4M3 max
            Element::E5m2 => 57344.0,                     // E5M2 max
        }
    }

    /// The scale axis: how the per-block scale is stored.
    pub fn scale_kind(self) -> ScaleKind {
        match self {
            Nvfp4 => ScaleKind::E4M3,
            Mxfp4 | Mxfp8E4 | Mxfp8E5 => ScaleKind::E8M0,
            Mxint2 | Mxint3 | Mxint4 | Mxint5 | Mxint6 | Mxint8 => ScaleKind::E8M0,
            // Legacy fp4/fp8 + FP32-scaled symmetric ints store a raw per-group
            // FP32 scale, like nvfp8.
            Nvfp8 | Fp4 | Fp8E4m3 | Fp8E5m2 => ScaleKind::F32,
            Int2 | Int3 | Int4 | Int5 | Int6 | Int8 => ScaleKind::F32,
            // FP16-scale twins.
            Nvfp8F16 | Fp4F16 | Fp8E4m3F16 | Fp8E5m2F16 => ScaleKind::F16,
            Int2F16 | Int3F16 | Int4F16 | Int5F16 | Int6F16 | Int8F16 => ScaleKind::F16,
        }
    }

    /// Whether the format carries one global FP32 (two-level scaling).
    fn has_global(self) -> bool { matches!(self, Nvfp4) }

    fn element_encode(self, x: f32) -> u8 {
        match self.element() {
            Element::Int(n) => codec::intn_encode(x, n) as u8, // low N bits, fits a byte
            Element::E2m1 => codec::e2m1_encode(x),
            Element::E4m3 => codec::e4m3_encode(x),
            Element::E5m2 => codec::e5m2_encode(x),
        }
    }

    fn element_decode(self, code: u8) -> f32 {
        match self.element() {
            Element::Int(n) => codec::intn_decode(code as u32, n),
            Element::E2m1 => codec::e2m1_decode(code),
            Element::E4m3 => codec::e4m3_decode(code),
            Element::E5m2 => codec::e5m2_decode(code),
        }
    }
}

/// A quantized weight tensor in one [`QFormat`]'s byte layout.
#[derive(Debug, Clone)]
pub struct PackedTensor {
    /// Element codes. 8-bit formats store one code/byte. Sub-byte formats
    /// (int2/3/4/5/6, E2M1) tight-bit-pack codes LSB-first into `u32` words
    /// (element `i` at bit `i·bits`), so a 4-bit stream is byte-identical to the
    /// classic 2-nibbles-per-byte layout. One guard word is appended so a
    /// straddling odd-width code's second-word read stays in bounds.
    pub codes: Vec<u8>,
    /// Per-block scales: 1 byte/block for E8M0/E4M3, 4 LE bytes/block for FP32.
    pub scales: Vec<u8>,
    /// Global FP32 scale (1.0 for single-level formats).
    pub global: f32,
}

/// `u32` words a tight bit-stream of `n` codes of `bits` each occupies. Widths
/// that divide 32 (2/4) never straddle a word boundary, so they pack to an exact
/// word count — this keeps per-row / per-expert concatenation (which the 4-bit
/// MoE kernels rely on) byte-aligned. Straddling widths (3/5/6) reserve one guard
/// word so a last-element second-word read can't run off the end.
pub fn bitstream_words(n: usize, bits: usize) -> usize {
    let words = (n * bits).div_ceil(32);
    if 32 % bits == 0 { words } else { words + 1 }
}

/// Write a code's low `bits` bits into the packed-code buffer for element `idx`.
/// 8-bit codes occupy one byte; sub-byte codes are OR'd bit-by-bit into the tight
/// bit-stream (LSB-first), so partial / straddling codes pack contiguously and
/// read back identically when the buffer is bound as `u32` (little-endian).
fn write_code(codes: &mut [u8], idx: usize, code: u8, bits: usize) {
    if bits == 8 {
        codes[idx] = code;
        return;
    }
    let bit_off = idx * bits;
    for bk in 0..bits {
        if (code >> bk) & 1 == 1 {
            let bit = bit_off + bk;
            codes[bit / 8] |= 1u8 << (bit % 8);
        }
    }
}

/// Read element `idx`'s low `bits` bits back out of the packed-code buffer (the
/// inverse of [`write_code`]).
fn read_code(codes: &[u8], idx: usize, bits: usize) -> u8 {
    if bits == 8 {
        return codes[idx];
    }
    let bit_off = idx * bits;
    let mut c = 0u8;
    for bk in 0..bits {
        let bit = bit_off + bk;
        c |= ((codes[bit / 8] >> (bit % 8)) & 1) << bk;
    }
    c
}

/// Quantize a row-major `[rows, cols]` f32 weight matrix to `fmt`'s layout.
///
/// Blocks tile K in `fmt.block_size()`-element groups. A `cols` that isn't a
/// multiple of the block size (e.g. int8 group 64 over a d96 head) gets a
/// shorter **trailing block** rather than being rejected — `blocks_per_row`
/// rounds up and each block is clamped to the row's remaining columns. The
/// kernel's `d / block_size` indexing maps every element to the right block,
/// the partial tail included, so codes + scales stay self-consistent.
pub fn pack(fmt: QFormat, w: &[f32], rows: usize, cols: usize) -> PackedTensor {
    assert_eq!(w.len(), rows * cols, "weight length must be rows*cols");
    let bs = fmt.block_size();
    let blocks_per_row = cols.div_ceil(bs);
    let nblocks = rows * blocks_per_row;

    // Per-block amax → the f32 block scale that maps amax to the element max.
    let mut block_scale = vec![0f32; nblocks];
    for r in 0..rows {
        for b in 0..blocks_per_row {
            let start = r * cols + b * bs;
            let len = bs.min(cols - b * bs); // clamp the ragged trailing block
            let amax = w[start..start + len].iter().fold(0f32, |m, &v| m.max(v.abs()));
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

    let bits = fmt.element_bits();
    // 8-bit codes are one byte each; sub-byte codes (2/3/4/5/6) tight-bit-pack
    // into u32 words (+1 guard word for straddling reads).
    let mut codes = if bits == 8 {
        vec![0u8; rows * cols]
    } else {
        vec![0u8; bitstream_words(rows * cols, bits) * 4]
    };
    let mut scales = Vec::with_capacity(nblocks * fmt.scale_kind().bytes());

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
                ScaleKind::F16 => {
                    let bits = codec::f16_scale_encode(block_scale[blk]);
                    scales.extend_from_slice(&bits.to_le_bytes());
                    codec::f16_scale_decode(bits)
                },
            };
            let inv = if eff > 0.0 { 1.0 / eff } else { 0.0 };
            let len = bs.min(cols - b * bs); // clamp the ragged trailing block
            for e in 0..len {
                let idx = r * cols + b * bs + e;
                let code = fmt.element_encode(w[idx] * inv);
                write_code(&mut codes, idx, code, bits);
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
    let bits = fmt.element_bits();
    let blocks_per_row = cols.div_ceil(bs); // ragged trailing block, mirrors `pack`
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
                ScaleKind::F16 => {
                    let o = blk * 2;
                    codec::f16_scale_decode(u16::from_le_bytes([p.scales[o], p.scales[o + 1]]))
                },
            };
            let len = bs.min(cols - b * bs); // clamp the ragged trailing block
            for e in 0..len {
                let idx = r * cols + b * bs + e;
                out[idx] = fmt.element_decode(read_code(&p.codes, idx, bits)) * eff;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [QFormat; 30] = [
        Nvfp4, Mxfp4, Mxfp8E4, Mxfp8E5, Nvfp8, Fp4, Fp8E4m3, Fp8E5m2, Int8, Int2, Int3, Int4, Int5,
        Int6, Mxint2, Mxint3, Mxint4, Mxint5, Mxint6, Mxint8, Nvfp8F16, Fp4F16, Fp8E4m3F16,
        Fp8E5m2F16, Int2F16, Int3F16, Int4F16, Int5F16, Int6F16, Int8F16,
    ];

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
                Nvfp4 | Mxfp4 => 0.97,              // 4-bit element
                Fp4 => 0.98,                        // 4-bit element + exact FP32 group scale
                Mxfp8E4 | Mxfp8E5 => 0.99,          // 8-bit element + pow-2 scale
                Nvfp8 | Fp8E4m3 | Fp8E5m2 => 0.999, // 8-bit element + exact FP32 scale
                Int8 => 0.9999,                     // int8 + FP32 scale is very tight
                // Symmetric ints, exact FP32 group scale — fidelity tracks the
                // bit width (more levels → tighter direction).
                Int2 => 0.80,
                Int3 => 0.94,
                Int4 => 0.97,
                Int5 => 0.995,
                Int6 => 0.998,
                // MXINT: same elements but an E8M0 pow-2 block scale leaves up to
                // ~2× of the range unused, so each is a touch looser than its
                // FP32-scaled int twin.
                Mxint2 => 0.70,
                Mxint3 => 0.90,
                Mxint4 => 0.95,
                Mxint5 => 0.98,
                Mxint6 => 0.99,
                Mxint8 => 0.997,
                // FP16-scale twins: same element fidelity as the FP32 twin; the
                // half scale's ~3-digit precision costs almost nothing, so floors
                // sit a notch below each twin to stay non-flaky.
                Fp4F16 => 0.97,
                Nvfp8F16 | Fp8E4m3F16 | Fp8E5m2F16 => 0.998,
                Int2F16 => 0.79,
                Int3F16 => 0.93,
                Int4F16 => 0.96,
                Int5F16 => 0.99,
                Int6F16 => 0.996,
                Int8F16 => 0.999,
            };
            assert!(cos >= floor, "{}: cosine {cos} < {floor}", fmt.name());
        }
    }

    #[test]
    fn packed_byte_sizes_match_layout() {
        let (rows, cols) = (2usize, 64usize); // 64 divisible by every block/group (16/32/64)
        let w = weights(rows, cols);
        for fmt in ALL {
            let p = pack(fmt, &w, rows, cols);
            let elems = rows * cols;
            // 8-bit: one byte/code. Sub-byte: tight bit-stream in u32 words (+guard).
            let expected_codes = if fmt.element_bits() == 8 {
                elems
            } else {
                bitstream_words(elems, fmt.element_bits()) * 4
            };
            assert_eq!(p.codes.len(), expected_codes, "{} codes", fmt.name());
            let nblocks = rows * (cols / fmt.block_size());
            assert_eq!(p.scales.len(), nblocks * fmt.scale_kind().bytes(), "{} scales", fmt.name());
            if !fmt.has_global() {
                assert_eq!(p.global, 1.0, "{} global", fmt.name());
            }
        }
    }

    #[test]
    fn ragged_trailing_block_round_trips() {
        // A dim that isn't a multiple of the block size (int8 group 64 over a
        // d96 head: a 64-block + a 32-block) must pack to a rounded-up block
        // count and still round-trip with full fidelity — this is what unblocks
        // int8 flash-SDPA KV at d96 (GPT-NeoX).
        let (rows, cols) = (4usize, 96usize);
        let w = weights(rows, cols);
        let p = pack(Int8, &w, rows, cols);
        // 96 / 64 rounds up to 2 blocks/row; F32 scales are 4 bytes each.
        let blocks_per_row = cols.div_ceil(Int8.block_size());
        assert_eq!(blocks_per_row, 2, "d96/64 should be 2 blocks");
        assert_eq!(p.scales.len(), rows * blocks_per_row * 4, "scale bytes");
        assert_eq!(p.codes.len(), rows * cols, "one code byte per int8 element");
        let d = dequant(Int8, &p, rows, cols);
        assert_eq!(d.len(), w.len());
        assert!(cosine(&w, &d) >= 0.9999, "ragged int8 cosine {}", cosine(&w, &d));
    }

    #[test]
    fn all_zero_block_dequants_to_zero() {
        let (rows, cols) = (1usize, 64usize); // divisible by every block/group
        let w = vec![0f32; rows * cols];
        for fmt in ALL {
            let p = pack(fmt, &w, rows, cols);
            let d = dequant(fmt, &p, rows, cols);
            assert!(d.iter().all(|&v| v == 0.0), "{} zero block", fmt.name());
        }
    }
}
