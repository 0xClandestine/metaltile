//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! GGUF block-quant **formats** — the llama.cpp `ggml-quants.c` super-block
//! family (q8_0, q2_k, …) — host-side packer + dequant oracle, built on the
//! shared [`codec`](super::codec) bit primitives.
//!
//! ## Why this is separate from [`QFormat`](super::format::QFormat)
//!
//! [`QFormat`] is metaltile's **own** block-scaled layout: *planar*
//! (`codes[]` + `scales[]`), *symmetric* (`element · block_scale · global`, no
//! zero-point), packed end-to-end by metaltile. GGUF formats are
//! **externally dictated** by the `ggml` on-disk spec and don't fit that mould:
//!
//! * **q8_0** — int8 over a block of 32 with one fp16 super-scale; the int8 is
//!   *asymmetric* (full `[-128, 127]`, unlike `QFormat`'s symmetric ±127).
//! * **q2_k** — a k-quant *super-block* of 256 (16 sub-blocks of 16) with a
//!   *two-level* scale (`d`/`dmin` fp16 super-scales × per-sub-block 4-bit
//!   scale + 4-bit **min**). The min term makes it *asymmetric*; the two-level
//!   hierarchy has nowhere to live in [`PackedTensor`](super::format::PackedTensor).
//!
//! This mirrors the codebase's existing split: the asymmetric MLX affine track
//! likewise lives outside `QFormat` (see [`format`](super::format) and
//! `mlx/quantized.rs`). What every track **shares** is [`codec`] — the
//! single source of truth for element/scale bit-decode.
//!
//! ## Single source of truth
//!
//! The GGUF dequant kernels (`ffai/gguf_dequant_*`, `ffai/moe_*_{q2k,iq2xxs}`,
//! …) emit their decode math inline in MSL. Their CPU correctness oracles call
//! the [`dequant_q8_0`] / [`dequant_q2_k`] functions here (and the canonical
//! [`q2_k_qpos`] index map), so the oracle can't silently drift from the kernel.
//! That drift is exactly the class of bug fixed in PR #264 (a hand-rolled q2_k
//! oracle used the wrong 2-bit byte layout and reported ~0.71 error against a
//! *correct* kernel); routing every oracle through one shared decoder retires it.

use half::f16;

use super::codec;

/// Elements per block / super-block, by format.
const Q8_0_BLOCK: usize = 32;
const Q2_K_SUPERBLOCK: usize = 256;
/// q2_k sub-block size: 16 sub-blocks of 16 values share one (scale, min) pair.
const Q2_K_SUBBLOCK: usize = 16;

/// A GGUF (`ggml`) block-quant weight format.
///
/// Provenance: these are the llama.cpp / GGUF super-block layouts, distinct from
/// the OCP/NVIDIA microscaling formats in [`QFormat`](super::format::QFormat) and
/// the MLX affine track in `mlx/quantized.rs`. They are listed in this one enum so
/// callers have a single registry of the GGUF precisions metaltile supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufFormat {
    /// `Q8_0`: int8 element, block 32, one fp16 super-scale per block.
    /// `out[i] = qs[i] · d[i/32]`. Asymmetric int8 (`[-128, 127]`).
    Q8_0,
    /// `Q2_K`: 2-bit element, super-block 256 = 16 sub-blocks of 16, two-level
    /// scale (`d`/`dmin` fp16 super-scales × per-sub-block 4-bit scale + 4-bit
    /// min). `out[i] = d·scale₄·q₂ − dmin·min₄`. Asymmetric (the `min` term).
    Q2K,
}

impl GgufFormat {
    /// Short label for bench rows / shape strings.
    pub fn name(self) -> &'static str {
        match self {
            GgufFormat::Q8_0 => "q8_0",
            GgufFormat::Q2K => "q2_k",
        }
    }

    /// Elements per block (q8_0) / super-block (q2_k) along K.
    pub fn block_size(self) -> usize {
        match self {
            GgufFormat::Q8_0 => Q8_0_BLOCK,
            GgufFormat::Q2K => Q2_K_SUPERBLOCK,
        }
    }

    /// Bits per quantized element.
    pub fn element_bits(self) -> usize {
        match self {
            GgufFormat::Q8_0 => 8,
            GgufFormat::Q2K => 2,
        }
    }
}

// ── Q8_0 ────────────────────────────────────────────────────────────────────
// int8 over a block of 32, one fp16 super-scale. The GPU loader splits a block
// into a planar `qs:u8` (int8 reinterpreted) + a host-fp32 `scales` (the fp16
// super-scale converted once at load). Decode reuses `codec::int8_decode`
// verbatim — the *encode* is GGUF-specific (full [-128, 127] range, vs codec's
// symmetric ±127), so the packer below does not call `codec::int8_encode`.

/// Host-resident Q8_0 tensors the kernel ABI expects: `(qs, scales)`.
/// `qs[block·32 + lane]` is the int8 quant (as `u8`); `scales[block]` is the
/// fp16 super-scale already converted to f32 by the loader.
#[derive(Debug, Clone)]
pub struct PackedQ8_0 {
    /// int8 quants reinterpreted as `u8`, one byte per value.
    pub qs: Vec<u8>,
    /// Per-block fp16 super-scale, host-converted to f32.
    pub scales: Vec<f32>,
}

/// Quantize a multiple-of-32 slice to Q8_0 (mirrors `quantize_row_q8_0`):
/// per-block `d = amax/127` rounded to fp16, then `q = round(v/d)` clamped to
/// `[-128, 127]`. The fp16 round on `d` matches what the on-disk super-scale
/// gives the kernel, so codes + scale stay self-consistent.
pub fn pack_q8_0(values: &[f32]) -> PackedQ8_0 {
    assert_eq!(values.len() % Q8_0_BLOCK, 0, "Q8_0 needs a multiple-of-32 length");
    let n_blocks = values.len() / Q8_0_BLOCK;
    let mut qs = Vec::with_capacity(n_blocks * Q8_0_BLOCK);
    let mut scales = Vec::with_capacity(n_blocks);
    for block in values.chunks_exact(Q8_0_BLOCK) {
        let amax = block.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        let d = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        // Round the super-scale through fp16, the on-disk precision.
        let d = f16::from_f32(d).to_f32();
        scales.push(d);
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        for &v in block {
            // GGUF q8_0 keeps the full int8 range, so we clamp to [-128, 127]
            // here rather than via `codec::int8_encode` (which is symmetric ±127).
            let q = (v * inv).round().clamp(-128.0, 127.0) as i8;
            qs.push(q as u8);
        }
    }
    PackedQ8_0 { qs, scales }
}

/// Decode Q8_0 back to f32 — the CPU oracle. `out[i] = int8_decode(qs[i]) ·
/// scales[i/32]`, sharing [`codec::int8_decode`] with every other int8 path.
pub fn dequant_q8_0(p: &PackedQ8_0) -> Vec<f32> {
    let mut out = Vec::with_capacity(p.qs.len());
    for (i, &q) in p.qs.iter().enumerate() {
        out.push(codec::int8_decode(q) * p.scales[i / Q8_0_BLOCK]);
    }
    out
}

// ── Q2_K ──────────────────────────────────────────────────────────────────────
// k-quant: 2-bit weights, super-block of 256, two-level scale. The GPU loader
// splits a block into `qs_packed:u32` (64 packed-quant bytes → 16 LE words),
// `scales:u8` (16 bytes, low nibble = 4-bit scale, high nibble = 4-bit min),
// and `d`/`dmin:f32` (the fp16 super-scales, host-converted).

/// Canonical Q2_K output-index → `(qs byte 0..63, 2-bit shift)`. The 256 values
/// are **not** 4-consecutive-per-byte: they split into 2 halves of 128; each
/// half into 4 j-groups of 32; each j-group into two runs of 16 values indexing
/// 16 consecutive `qs` bytes at a shared `jg·2` shift (the llama.cpp
/// `dequantize_row_q2_K` order). The canonical scale index for output `i` is
/// simply `i / 16`. This is the **one** definition the kernel, the quantizer,
/// and the oracle all read — getting it wrong was the PR #264 bug.
pub fn q2_k_qpos(i: usize) -> (usize, u32) {
    let half = i / 128; // 0..1  → qs byte base half*32
    let yh = i % 128; // 0..127
    let jg = yh / 32; // 0..3  → shift = jg*2
    let yg = yh % 32; // 0..31
    let sub_half = yg / 16; // 0..1
    let l = yg % 16; // 0..15 → byte within the 16-run
    (half * 32 + sub_half * 16 + l, (jg * 2) as u32)
}

/// Host-resident Q2_K tensors the kernel ABI expects.
#[derive(Debug, Clone)]
pub struct PackedQ2K {
    /// 64 packed-quant bytes/block, re-laid as 16 LE `u32` words/block.
    pub qs_packed: Vec<u32>,
    /// 16 scale/min bytes/block (low nibble scale, high nibble min).
    pub scales: Vec<u8>,
    /// Per-block fp16 super-scale for the sub-block scales, host-converted.
    pub d: Vec<f32>,
    /// Per-block fp16 super-scale for the sub-block mins, host-converted.
    pub dmin: Vec<f32>,
}

/// Quantize a multiple-of-256 slice to the Q2_K GPU-resident split. A
/// *non-trained* quantizer (per-sub-block min/max with uniform 2-bit bucketing):
/// sufficient to exercise the dequant kernel, not the perplexity-tuned
/// `quantize_row_q2_K` search. `d`/`dmin` are reconstructed so dequant exactly
/// recovers the per-sub-block scale (`d = 1/15`, `dmin = 1`, fp16-rounded).
pub fn pack_q2_k(values: &[f32]) -> PackedQ2K {
    assert_eq!(values.len() % Q2_K_SUPERBLOCK, 0, "Q2_K needs a multiple-of-256 length");
    let n_blocks = values.len() / Q2_K_SUPERBLOCK;
    let n_sub = Q2_K_SUPERBLOCK / Q2_K_SUBBLOCK; // 16
    let mut qs_packed = Vec::with_capacity(n_blocks * 16);
    let mut scales = Vec::with_capacity(n_blocks * 16);
    let mut d = Vec::with_capacity(n_blocks);
    let mut dmin = Vec::with_capacity(n_blocks);

    for block in values.chunks_exact(Q2_K_SUPERBLOCK) {
        let mut sub_scales = [0u8; 16];
        let mut sub_mins = [0u8; 16];
        let mut qs_bytes = [0u8; 64];
        for s in 0..n_sub {
            let sub = &block[s * Q2_K_SUBBLOCK..(s + 1) * Q2_K_SUBBLOCK];
            let mn = sub.iter().cloned().fold(f32::INFINITY, f32::min);
            let mx = sub.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let scale = ((mx - mn) / 3.0).max(1e-30);
            let mn_q = (-mn / scale).round().clamp(0.0, 15.0) as u8;
            let scale_q = (scale * 15.0).round().clamp(0.0, 15.0) as u8;
            sub_scales[s] = scale_q;
            sub_mins[s] = mn_q;
            // Quantize against the *reconstructed* scale/min so dequant inverts
            // exactly: out = round((x − recon_min) / recon_scale).
            let recon_scale = (scale_q as f32) / 15.0;
            let recon_min = -(mn_q as f32) * recon_scale;
            for (i, &v) in sub.iter().enumerate() {
                let q = ((v - recon_min) / recon_scale.max(1e-30)).round().clamp(0.0, 3.0) as u8;
                let (q_byte, shift) = q2_k_qpos(s * Q2_K_SUBBLOCK + i);
                qs_bytes[q_byte] |= q << shift;
            }
        }
        for s in 0..n_sub {
            scales.push((sub_mins[s] << 4) | sub_scales[s]);
        }
        for w in 0..16 {
            let bs = w * 4;
            qs_packed.push(u32::from_le_bytes([
                qs_bytes[bs],
                qs_bytes[bs + 1],
                qs_bytes[bs + 2],
                qs_bytes[bs + 3],
            ]));
        }
        // Materialize the super-scales the way the GGUF loader does (fp16 → f32).
        d.push(f16::from_f32(1.0 / 15.0).to_f32());
        dmin.push(f16::from_f32(1.0).to_f32());
    }
    PackedQ2K { qs_packed, scales, d, dmin }
}

/// Decode Q2_K back to f32 — the CPU oracle. Mirrors the kernel exactly:
/// `out[i] = d·scale₄·q₂ − dmin·min₄`, with the byte/shift from [`q2_k_qpos`]
/// and scale index `i/16`.
pub fn dequant_q2_k(p: &PackedQ2K) -> Vec<f32> {
    let n_blocks = p.d.len();
    let mut out = Vec::with_capacity(n_blocks * Q2_K_SUPERBLOCK);
    for b in 0..n_blocks {
        let d = p.d[b];
        let dmin = p.dmin[b];
        for i in 0..Q2_K_SUPERBLOCK {
            let sub = i / Q2_K_SUBBLOCK; // canonical scale index
            let (q_byte, shift) = q2_k_qpos(i);
            let word = p.qs_packed[b * 16 + q_byte / 4];
            let byte_in_word = q_byte & 3;
            let qs_byte = (word >> (byte_in_word * 8)) & 0xff;
            let q_2bit = (qs_byte >> shift) & 0x3;
            let scale_byte = p.scales[b * 16 + sub] as u32;
            let scale_4bit = scale_byte & 0xf;
            let min_4bit = (scale_byte >> 4) & 0xf;
            out.push(d * (scale_4bit as f32) * (q_2bit as f32) - dmin * (min_4bit as f32));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic weights with magnitude varying along K (exercises the
    /// per-block / per-sub-block scaling).
    fn weights(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let c = (i % 37) as f32;
                let mag = 0.1 + (c % 7.0) * 0.3;
                if i % 3 == 0 { -mag } else { mag }
            })
            .collect()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (&x, &y) in a.iter().zip(b) {
            dot += x as f64 * y as f64;
            na += (x as f64).powi(2);
            nb += (y as f64).powi(2);
        }
        (dot / (na.sqrt() * nb.sqrt())) as f32
    }

    #[test]
    fn q8_0_round_trips() {
        let w = weights(32 * 8);
        let p = pack_q8_0(&w);
        assert_eq!(p.qs.len(), w.len());
        assert_eq!(p.scales.len(), w.len() / 32);
        let d = dequant_q8_0(&p);
        // int8 + fp16 scale is tight.
        assert!(cosine(&w, &d) >= 0.999, "q8_0 cosine {}", cosine(&w, &d));
    }

    #[test]
    fn q8_0_keeps_full_int8_range() {
        // A block whose amax forces a −128 code: codec::int8_decode must round-trip it.
        let mut w = vec![0.5f32; 32];
        w[0] = -1.0; // amax → d = 1/127; -1.0/d rounds to -127, and a slightly
        w[1] = -1.008; // larger magnitude reaches -128.
        let p = pack_q8_0(&w);
        // The most-negative code present decodes via the shared int8 decoder.
        let d = dequant_q8_0(&p);
        assert_eq!(d.len(), w.len());
        assert!(cosine(&w, &d) >= 0.99, "q8_0 range cosine {}", cosine(&w, &d));
    }

    #[test]
    fn q2_k_round_trips() {
        let w = weights(256 * 4);
        let p = pack_q2_k(&w);
        assert_eq!(p.qs_packed.len(), p.d.len() * 16);
        assert_eq!(p.scales.len(), p.d.len() * 16);
        let dq = dequant_q2_k(&p);
        assert_eq!(dq.len(), w.len());
        // 2-bit k-quant is coarse but must keep direction.
        assert!(cosine(&w, &dq) >= 0.85, "q2_k cosine {}", cosine(&w, &dq));
    }

    #[test]
    fn q2_k_qpos_is_a_bijection_over_256() {
        // Every output index maps to a distinct (byte, shift) slot — the 64
        // bytes × 4 shifts cover the 256 values exactly once.
        let mut seen = std::collections::HashSet::new();
        for i in 0..256 {
            let (byte, shift) = q2_k_qpos(i);
            assert!(byte < 64, "byte {byte} out of range at i={i}");
            assert!(shift < 8 && shift % 2 == 0, "shift {shift} invalid at i={i}");
            assert!(seen.insert((byte, shift)), "collision at i={i}: ({byte}, {shift})");
        }
        assert_eq!(seen.len(), 256);
    }
}
