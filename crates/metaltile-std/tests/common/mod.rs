//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Shared test helpers for metaltile-std GPU integration tests.

#![allow(dead_code)]

use std::sync::{Mutex, MutexGuard, OnceLock};

use metaltile::core::dtype::DType;

/// Serialise GPU dispatches across all integration tests that pull in
/// this module. cargo runs integration tests in parallel by default;
/// concurrent dispatches on the shared Metal pipeline race the PSO
/// cache + library compilation path and surface as cross-test numeric
/// corruption (caught e.g. when an f16 test ran after an f32 test in
/// a single `cargo test` invocation and produced output ≈ 0.45× the
/// expected magnitude). Lighter than requiring `--test-threads=1` at
/// the command line.
///
/// Tests that grab this lock at the top of their body serialise across
/// the entire integration-test binary they're linked into. Mutex
/// poisoning unwraps to `into_inner()` so a panic in one test still
/// lets the others run.
pub fn gpu_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().unwrap_or_else(|e| e.into_inner())
}

#[derive(Clone, Copy, Debug)]
pub enum Dt {
    F32,
    F16,
    Bf16,
}

impl Dt {
    pub fn bytes(self) -> usize {
        match self {
            Dt::F32 => 4,
            Dt::F16 | Dt::Bf16 => 2,
        }
    }
    pub fn to_dtype(self) -> DType {
        match self {
            Dt::F32 => DType::F32,
            Dt::F16 => DType::F16,
            Dt::Bf16 => DType::BF16,
        }
    }
    /// Round-trip a value through this dtype's precision. Used by
    /// per-dtype correctness oracles so the CPU reference sees the
    /// same load-cast quantisation the kernel does (no-op for f32,
    /// 10-bit mantissa for f16, 7-bit for bf16).
    pub fn round(self, v: f32) -> f32 {
        match self {
            Dt::F32 => v,
            Dt::F16 => half::f16::from_f32(v).to_f32(),
            Dt::Bf16 => half::bf16::from_f32(v).to_f32(),
        }
    }
}

pub fn pack_bytes(vals: &[f32], dt: Dt) -> Vec<u8> {
    match dt {
        // Host is little-endian on all current Metal targets — single
        // memcpy beats `flat_map(to_le_bytes)`'s per-element iter churn.
        // Noticeable on the SWA perf bench's 4M-element K/V ramps.
        Dt::F32 => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
        Dt::F16 => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
        Dt::Bf16 => vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
    }
}

/// u32 vec → LE bytes — for packed-quantized-weight buffers (the
/// `weight` input of `dequant_gather` / `dequant_gemv`).
pub fn pack_u32_bytes(vals: &[u32]) -> Vec<u8> { bytemuck::cast_slice::<u32, u8>(vals).to_vec() }

/// LE bytes → u32 vec (output readback for u32-typed kernel outputs).
pub fn unpack_u32_bytes(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

pub fn unpack_bytes(bytes: &[u8], dt: Dt) -> Vec<f32> {
    match dt {
        Dt::F32 => bytemuck::cast_slice::<u8, f32>(bytes).to_vec(),
        Dt::F16 =>
            bytes.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        Dt::Bf16 => bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
    }
}

pub struct SdpaShape {
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub n_kv: usize,
    pub scale: f32,
}

/// Naive triple-loop SDPA reference: `O = softmax(Q · Kᵀ · scale) · V`
/// per Q head, GQA via `kv_head = q_head / heads_per_group`, fp32.
pub fn naive_sdpa_f32(q: &[f32], k: &[f32], v: &[f32], s: &SdpaShape) -> Vec<f32> {
    naive_sdpa_swa_f32(q, k, v, s, 0, 0)
}

/// Sliding-window + sink-token SDPA reference. Attended positions are
/// `[0, sink_end) ∪ [window_start, n_kv)`; masked positions contribute
/// nothing (no score, no softmax weight). Caller must satisfy
/// `window_start >= sink_end` and `window_start <= n_kv`, the same
/// preconditions the GPU kernel enforces. With `sink_end = 0` and
/// `window_start = 0` this is the dense reference (used by
/// [`naive_sdpa_f32`]).
pub fn naive_sdpa_swa_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    s: &SdpaShape,
    sink_end: usize,
    window_start: usize,
) -> Vec<f32> {
    assert!(s.n_q_heads.is_multiple_of(s.n_kv_heads));
    assert!(
        window_start >= sink_end,
        "window_start must be >= sink_end (overlap would double-count)"
    );
    assert!(window_start <= s.n_kv && sink_end <= s.n_kv);
    let gqa = s.n_q_heads / s.n_kv_heads;
    let mut out = vec![0.0f32; s.n_q_heads * s.head_dim];
    let attended = |t: usize| t < sink_end || t >= window_start;
    for qh in 0..s.n_q_heads {
        let kvh = qh / gqa;
        let q_off = qh * s.head_dim;
        let kv_slab = kvh * s.n_kv * s.head_dim;
        let mut scores = vec![f32::NEG_INFINITY; s.n_kv];
        for (t, score) in scores.iter_mut().enumerate() {
            if !attended(t) {
                continue;
            }
            let k_off = kv_slab + t * s.head_dim;
            let mut dot = 0.0f32;
            for d in 0..s.head_dim {
                dot += q[q_off + d] * k[k_off + d];
            }
            *score = dot * s.scale;
        }
        let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for score in scores.iter_mut() {
            if score.is_finite() {
                *score = (*score - m).exp();
                sum += *score;
            } else {
                *score = 0.0;
            }
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for d in 0..s.head_dim {
            let mut acc = 0.0f32;
            for (t, score) in scores.iter().enumerate() {
                acc += *score * inv * v[kv_slab + t * s.head_dim + d];
            }
            out[q_off + d] = acc;
        }
    }
    out
}

/// Causal-prefix SDPA reference for batched-Q decode (M7 prefill-tile arm).
///
/// Q layout `[n_q_heads, q_len, head_dim]`, K/V `[n_kv_heads, k_len, head_dim]`,
/// out `[n_q_heads, q_len, head_dim]`. For each Q row `qi` in `0..q_len`, the
/// attended KV range is `[0, q_len_off + qi + 1)` — the same mask the
/// `mt_sdpa_prefill_mma` kernel applies via
/// `q_abs = q_tile_first + fm + q_len_off`. With `q_len_off = k_len - q_len`,
/// this is the standard chunked-prefill / speculative-decode-verify pattern.
/// GQA via `kv_head = q_head / (n_q_heads / n_kv_heads)`.
#[allow(clippy::too_many_arguments)]
pub fn naive_sdpa_causal_prefix_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_q_heads: usize,
    n_kv_heads: usize,
    q_len: usize,
    k_len: usize,
    head_dim: usize,
    q_len_off: usize,
    scale: f32,
) -> Vec<f32> {
    assert!(n_q_heads.is_multiple_of(n_kv_heads));
    assert_eq!(q.len(), n_q_heads * q_len * head_dim);
    assert_eq!(k.len(), n_kv_heads * k_len * head_dim);
    assert_eq!(v.len(), n_kv_heads * k_len * head_dim);
    let gqa = n_q_heads / n_kv_heads;
    let mut out = vec![0.0f32; n_q_heads * q_len * head_dim];
    for qh in 0..n_q_heads {
        let kvh = qh / gqa;
        let kv_slab = kvh * k_len * head_dim;
        let q_head_off = qh * q_len * head_dim;
        for qi in 0..q_len {
            let q_off = q_head_off + qi * head_dim;
            let visible_end = (q_len_off + qi + 1).min(k_len);
            let mut scores = vec![f32::NEG_INFINITY; k_len];
            for (t, score) in scores.iter_mut().enumerate().take(visible_end) {
                let k_off = kv_slab + t * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_off + d] * k[k_off + d];
                }
                *score = dot * scale;
            }
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for score in scores.iter_mut() {
                if score.is_finite() {
                    *score = (*score - m).exp();
                    sum += *score;
                } else {
                    *score = 0.0;
                }
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (t, score) in scores.iter().enumerate() {
                    acc += *score * inv * v[kv_slab + t * head_dim + d];
                }
                out[q_off + d] = acc;
            }
        }
    }
    out
}

/// Build a small SRHT-style orthogonal rotation `[dim, dim]` for the
/// AURA codec tests: `H · diag(±1) / √dim` where `H` is the
/// Sylvester–Hadamard matrix. `dim` must be a power of two.
///
/// The production AURA path rotates each K/V vector with exactly this
/// shape (a sub-sampled randomised Hadamard transform). It is
/// orthogonal — `Rᵀ R = I` — so the encoder's norm bookkeeping holds,
/// while every output coordinate mixes every input coordinate, which
/// exercises the encode kernel's rotation matmul stage that an identity
/// Π leaves dormant.
///
/// The `±1` sign flips are derived deterministically from a small LCG
/// seeded by `seed`, so the rotation is reproducible across runs.
pub fn srht_rotation(dim: usize, seed: u64) -> Vec<f32> {
    assert!(dim.is_power_of_two(), "SRHT rotation requires a power-of-two dim");

    // Sylvester–Hadamard: H[i][j] = (-1)^popcount(i & j).
    let mut h = vec![0.0_f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let sign = if (i & j).count_ones() % 2 == 0 { 1.0 } else { -1.0 };
            h[i * dim + j] = sign;
        }
    }

    // Deterministic ±1 diagonal sign vector via a small LCG.
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let signs: Vec<f32> = (0..dim)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            if (state >> 33) & 1 == 0 { 1.0 } else { -1.0 }
        })
        .collect();

    // R = H · diag(signs) / √dim → column j scaled by signs[j].
    let scale = 1.0 / (dim as f32).sqrt();
    let mut r = vec![0.0_f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            r[i * dim + j] = h[i * dim + j] * signs[j] * scale;
        }
    }
    r
}

/// Deterministic init pattern — small repeating modulus avoids both
/// degenerate all-zero softmax and uniform-value short-circuits.
pub fn ramp(n: usize, modulus: usize, offset: f32) -> Vec<f32> {
    (0..n).map(|i| ((i % modulus) as f32 - offset) * 0.05).collect()
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
}

/// Naive RMSNorm reference: `out = x * w / sqrt(mean(x²) + eps)`.
/// Was used by the legacy `tests/rms_norm_gpu_correctness.rs` (removed in
/// #240). f32 throughout.
pub fn naive_rms_norm_f32(x: &[f32], w: &[f32], n: usize, eps: f32) -> Vec<f32> {
    assert_eq!(x.len() % n, 0, "x len must be multiple of row width n");
    assert_eq!(w.len(), n, "w length must equal row width n");
    let rows = x.len() / n;
    let mut out = vec![0.0_f32; x.len()];
    for r in 0..rows {
        let base = r * n;
        let ssq: f32 = x[base..base + n].iter().map(|v| v * v).sum();
        let rms = (ssq / (n as f32) + eps).sqrt().recip();
        for d in 0..n {
            out[base + d] = x[base + d] * rms * w[d];
        }
    }
    out
}

/// Naive AURA encode reference (single bit-width, identity rotation
/// or caller-supplied dense rotation). Mirrors the GPU pipeline:
///   1. r        = ||x||_2
///   2. u        = x / r
///   3. y        = rotation @ u
///   4. indices  = quantise(y) against `boundaries` (Lloyd-Max)
///   5. packed   = bitpack(indices, bits)
///   6. r̂        = ||codebook[indices]||_2
///   7. corrected = r / r̂  (or r if r̂ near zero)
///
/// `dim` must match rotation row/col count. `packed_width = ceil(dim * bits / 32)`.
/// `boundaries` has `2^bits - 1` entries; `codebook` has `2^bits`.
///
/// Returns `(packed_out [rows * packed_width], norms_out [rows])`.
pub fn naive_aura_encode_f32(
    input: &[f32],
    rotation: &[f32],
    boundaries: &[f32],
    codebook: &[f32],
    rows: usize,
    dim: usize,
    bits: usize,
) -> (Vec<u32>, Vec<f32>) {
    let levels = 1usize << bits;
    assert_eq!(boundaries.len(), levels - 1, "boundaries must have 2^bits-1 entries");
    assert_eq!(codebook.len(), levels, "codebook must have 2^bits entries");
    assert_eq!(rotation.len(), dim * dim, "rotation must be [dim, dim]");
    assert_eq!(input.len(), rows * dim, "input must be [rows, dim]");

    // packed_width = ceil(dim * bits / 32). Mirrors what the kernel
    // uses via its `packed_width` constexpr argument; the caller in
    // FFAI also computes it this way.
    let packed_width = (dim * bits).div_ceil(32);

    let mut packed_out = vec![0u32; rows * packed_width];
    let mut norms_out = vec![0.0_f32; rows];

    for r in 0..rows {
        let row_base = r * dim;

        // Step 1: r = ||x||_2
        let r_norm: f32 = input[row_base..row_base + dim].iter().map(|v| v * v).sum::<f32>().sqrt();
        let inv = if r_norm > 1e-8 { 1.0 / r_norm } else { 0.0 };

        // Step 2 + 3: y = rotation @ (x / r)
        let mut y = vec![0.0_f32; dim];
        for d in 0..dim {
            let mut acc = 0.0_f32;
            for j in 0..dim {
                acc += rotation[d * dim + j] * input[row_base + j] * inv;
            }
            y[d] = acc;
        }

        // Step 4: per-coord Lloyd-Max quantisation. idx = count of
        // boundaries the rotated value strictly exceeds (matches the
        // kernel's `idx + (rotated > boundaries[b]).cast::<u32>()`).
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

        // Step 5: bitpack. Mirrors the kernel's `bit_offset = d*bits`,
        // `word_idx = bit_offset/32`, `shift = bit_offset & 31`, plus
        // cross-word spill for non-aligned widths (int3, int5).
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

        // Steps 6 + 7: r̂ from codebook, corrected = r / r̂.
        let recon_sq: f32 = indices.iter().map(|&i| codebook[i as usize].powi(2)).sum();
        let recon_norm = recon_sq.sqrt();
        let corrected = if recon_norm > 1e-8 { r_norm / recon_norm } else { r_norm };
        norms_out[r] = corrected;
    }

    (packed_out, norms_out)
}
