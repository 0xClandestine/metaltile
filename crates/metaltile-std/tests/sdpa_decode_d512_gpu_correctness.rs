//! End-to-end correctness for `ffai_sdpa_decode_d512` — the head_dim=512
//! specialization needed for Gemma 4's global (`full_attention`) layers.
//! Validates that the four-phase output reduction produces the same
//! answer as a naive CPU SDPA reference. macOS-gated.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, SdpaShape, naive_sdpa_f32, pack_bytes, ramp, unpack_bytes};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::ffai::sdpa_decode_d512::ffai_sdpa_decode_d512;

fn f32_slice_to_bytes(vals: &[f32]) -> Vec<u8> { pack_bytes(vals, Dt::F32) }
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> { unpack_bytes(bytes, Dt::F32) }

#[test]
fn sdpa_decode_d512_matches_naive_cpu_reference_f32() {
    let n_q_heads = 4usize;
    let n_kv_heads = 1usize; // GQA fan-out 4 (Gemma 4 E2B global layout)
    let head_dim = 512usize;
    let n_kv = 8usize;
    let kv_stride = 8usize;
    let heads_per_group = n_q_heads / n_kv_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q = ramp(n_q_heads * head_dim, 19, 9.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 23, 7.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 29, 5.0);

    let shape = SdpaShape { n_q_heads, n_kv_heads, head_dim, n_kv, scale };
    let expected = naive_sdpa_f32(&q, &k, &v, &shape);

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("q".into(), f32_slice_to_bytes(&q));
    buffers.insert("k".into(), f32_slice_to_bytes(&k));
    buffers.insert("v".into(), f32_slice_to_bytes(&v));
    buffers.insert("out".into(), vec![0u8; n_q_heads * head_dim * 4]);
    buffers.insert("head_dim".into(), (head_dim as u32).to_le_bytes().to_vec());
    buffers.insert("n_kv".into(), (n_kv as u32).to_le_bytes().to_vec());
    buffers.insert("kv_stride".into(), (kv_stride as u32).to_le_bytes().to_vec());
    buffers.insert("heads_per_group".into(), (heads_per_group as u32).to_le_bytes().to_vec());
    buffers.insert("scale".into(), scale.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = ffai_sdpa_decode_d512::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    // TPG = 512 (16 simdgroups), not 1024: the 16-wide per-lane register
    // footprint pushes the kernel's maxTotalThreadsPerThreadgroup below
    // 1024, so a 1024-thread dispatch silently no-ops (output stays
    // zero). See the kernel's DISPATCH INVARIANTS.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n_q_heads, 1, 1], [512, 1, 1])
        .expect("dispatch_with_grid should succeed");

    let out_bytes = result.outputs.get("out").expect("`out` buffer");
    let actual = bytes_to_f32_vec(out_bytes);

    assert_eq!(actual.len(), expected.len(), "output element count");

    let mut max_diff = 0.0_f32;
    let mut max_at = 0usize;
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        let diff = (e - a).abs();
        if diff > max_diff {
            max_diff = diff;
            max_at = i;
        }
    }

    assert!(
        max_diff < 1e-3,
        "sdpa_decode_d512 diverges from CPU reference: max |diff| = {max_diff:.2e} at {max_at} \
         (expected[{max_at}] = {:.4}, actual[{max_at}] = {:.4})",
        expected[max_at],
        actual[max_at],
    );
}
