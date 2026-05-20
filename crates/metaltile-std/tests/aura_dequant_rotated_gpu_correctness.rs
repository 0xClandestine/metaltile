//! End-to-end correctness test for `ffai::aura_dequant_rotated_int4`
//! on real Metal.
//!
//! Grid3D kernel — `(packed_width, tokens, B*H)` threads, one thread
//! per packed word per token per (batch * head). Reads `packed` +
//! `norms` + `codebook`, writes `out[bh, t, d] = codebook[q[d]] * norm[bh, t]`.
//!
//! Test approach: build a packed buffer with known codebook indices,
//! pre-compute the expected output, dispatch, compare.
//!
//! macOS-gated.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, max_abs_diff, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::ffai::aura_dequant_rotated::aura_dequant_rotated_int4;

fn f32_slice_to_bytes(vals: &[f32]) -> Vec<u8> { pack_bytes(vals, Dt::F32) }
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> { unpack_bytes(bytes, Dt::F32) }

/// Bit-pack a flat `[bh, t, dim]` index array into `[bh, t, packed_width]`
/// u32 words. Mirrors what `aura_encode` produces.
fn pack_int4_indices(indices: &[u32], bh: usize, tokens: usize, dim: usize) -> Vec<u32> {
    let bits = 4;
    let packed_width = (dim * bits).div_ceil(32);
    let mut packed = vec![0u32; bh * tokens * packed_width];
    for b in 0..bh {
        for t in 0..tokens {
            for d in 0..dim {
                let idx = indices[(b * tokens + t) * dim + d];
                let bit_offset = d * bits;
                let word_idx = bit_offset / 32;
                let shift = bit_offset & 31;
                let masked = idx & 0xf;
                packed[(b * tokens + t) * packed_width + word_idx] |= masked << shift;
            }
        }
    }
    packed
}

fn naive_aura_dequant(
    indices: &[u32],
    norms: &[f32],
    codebook: &[f32],
    bh: usize,
    tokens: usize,
    dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; bh * tokens * dim];
    for b in 0..bh {
        for t in 0..tokens {
            let norm_val = norms[b * tokens + t];
            for d in 0..dim {
                let q = indices[(b * tokens + t) * dim + d];
                out[(b * tokens + t) * dim + d] = codebook[q as usize] * norm_val;
            }
        }
    }
    out
}

#[test]
fn aura_dequant_rotated_int4_matches_naive_reference_f32() {
    // bits=4, dim=128, packed_width=16 u32, 2 heads × 3 tokens.
    let bits = 4usize;
    let dim = 128usize;
    let packed_width = (dim * bits).div_ceil(32); // 16
    let bh = 2usize;
    let tokens = 3usize;

    // 16-level symmetric codebook in [-1, 1].
    let codebook: Vec<f32> = (0..16).map(|i| -1.0 + 2.0 * i as f32 / 15.0).collect();

    // Pseudo-random indices in [0, 16).
    let indices: Vec<u32> = (0..bh * tokens * dim).map(|i| ((i * 7 + 3) % 16) as u32).collect();
    let packed = pack_int4_indices(&indices, bh, tokens, dim);
    let norms: Vec<f32> = (0..bh * tokens).map(|i| 0.5 + 0.1 * i as f32).collect();

    let expected = naive_aura_dequant(&indices, &norms, &codebook, bh, tokens, dim);

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("packed".into(), pack_u32_bytes(&packed));
    buffers.insert("norms".into(), f32_slice_to_bytes(&norms));
    buffers.insert("codebook".into(), f32_slice_to_bytes(&codebook));
    buffers.insert("out".into(), vec![0u8; bh * tokens * dim * 4]);
    buffers.insert("dim".into(), (dim as u32).to_le_bytes().to_vec());
    buffers.insert("packed_width".into(), (packed_width as u32).to_le_bytes().to_vec());
    buffers.insert("tokens".into(), (tokens as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = aura_dequant_rotated_int4::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Grid3D;

    // Grid3D: gid.[xyz] = thread_position_in_grid in each axis. For total
    // threads (packed_width, tokens, bh) we use tg = full N on x and
    // grid_groups = 1 on x; y/z axes carry the remaining extent via
    // grid_groups since tg.y/z = 1. (Spawning [N_x,N_y,N_z]/[N_x,1,1]
    // = N_x² × N_y × N_z threads previously passed only by virtue of
    // the kernel's `if d < dim` guard skipping illegitimate writes.)
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, tokens, bh], [
            packed_width,
            1,
            1,
        ])
        .expect("dispatch_with_grid should succeed");

    let out_bytes = result.outputs.get("out").expect("`out` buffer in dispatch result");
    let actual = bytes_to_f32_vec(out_bytes);

    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 1e-5, "aura_dequant_rotated int4: max |diff| = {diff:.2e}");
}
