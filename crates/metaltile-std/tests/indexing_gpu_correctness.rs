//! GPU correctness for the strided `indexing` kernels —
//! `mt_gather_front`, `mt_scatter`, `mt_masked_scatter`.
//!
//! Each kernel is dispatched on a real Metal device and compared to a
//! straight-loop CPU reference. The along-an-axis forms (`gather_axis`
//! / `scatter_axis`) have their own test files; this covers the three
//! remaining `indexing/` ops.
//!
//! macOS-gated.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::mlx::indexing::{mt_gather_front, mt_masked_scatter, mt_scatter};

// ── gather_front: out[r, :] = src[indices[r], :] ─────────────────────

#[test]
fn gather_front_matches_naive_f32() {
    let _g = gpu_lock();
    let (n_src_rows, n_out_rows, row_width) = (6usize, 9usize, 5usize);
    let src: Vec<f32> = (0..n_src_rows * row_width).map(|i| i as f32 * 0.5 - 3.0).collect();
    // Varied source-row picks, with repeats (a row gathered twice).
    let indices: Vec<u32> = (0..n_out_rows).map(|r| ((r * 5 + 1) % n_src_rows) as u32).collect();

    let mut expected = vec![0.0f32; n_out_rows * row_width];
    for r in 0..n_out_rows {
        let s = indices[r] as usize;
        for i in 0..row_width {
            expected[r * row_width + i] = src[s * row_width + i];
        }
    }

    let total = n_out_rows * row_width;
    let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b.insert("src".into(), pack_bytes(&src, Dt::F32));
    b.insert("indices".into(), pack_u32_bytes(&indices));
    b.insert("out".into(), pack_bytes(&vec![0.0; expected.len()], Dt::F32));
    b.insert("row_width".into(), (row_width as u32).to_le_bytes().to_vec());
    b.insert("n_elems".into(), (total as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_gather_front::kernel_ir_for(Dt::F32.to_dtype());
    kernel.mode = KernelMode::Grid3D;
    let result = ctx
        .dispatch_with_grid(&kernel, &b, &BTreeMap::new(), [total.div_ceil(64), 1, 1], [64, 1, 1])
        .expect("gather_front dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), Dt::F32);
    out.truncate(total);
    for (i, (a, e)) in out.iter().zip(&expected).enumerate() {
        assert!((a - e).abs() < 1e-6, "gather_front elem {i}: got {a}, want {e}");
    }
}

// ── scatter: out[indices[r], :] = updates[r, :] ──────────────────────

#[test]
fn scatter_matches_naive_f16() {
    let _g = gpu_lock();
    let (n_upd_rows, n_out_rows, row_width) = (4usize, 7usize, 6usize);
    let updates: Vec<f32> = (0..n_upd_rows * row_width).map(|i| i as f32 * 0.25 - 1.0).collect();
    // Distinct target rows (assignment form — collisions would race).
    let indices: Vec<u32> = vec![5, 1, 6, 2];
    // `out` pre-initialized with a recognisable base pattern.
    let mut out_init: Vec<f32> = (0..n_out_rows * row_width).map(|i| 100.0 + i as f32).collect();

    let mut expected = out_init.clone();
    for (r, &tgt) in indices.iter().enumerate() {
        for i in 0..row_width {
            expected[tgt as usize * row_width + i] = updates[r * row_width + i];
        }
    }

    // One thread per update element.
    let total = n_upd_rows * row_width;
    let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b.insert("updates".into(), pack_bytes(&updates, Dt::F16));
    b.insert("indices".into(), pack_u32_bytes(&indices));
    b.insert("out".into(), pack_bytes(&out_init, Dt::F16));
    b.insert("row_width".into(), (row_width as u32).to_le_bytes().to_vec());
    b.insert("n_elems".into(), (total as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_scatter::kernel_ir_for(Dt::F16.to_dtype());
    kernel.mode = KernelMode::Grid3D;
    let result = ctx
        .dispatch_with_grid(&kernel, &b, &BTreeMap::new(), [total.div_ceil(64), 1, 1], [64, 1, 1])
        .expect("scatter dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), Dt::F16);
    out.truncate(n_out_rows * row_width);
    // f16-round the oracle so unmasked rows compare exactly.
    out_init.iter_mut().for_each(|v| *v = Dt::F16.round(*v));
    for (r, &tgt) in indices.iter().enumerate() {
        for i in 0..row_width {
            expected[tgt as usize * row_width + i] = Dt::F16.round(updates[r * row_width + i]);
        }
    }
    for (i, (a, e)) in out.iter().zip(&expected).enumerate() {
        assert!((a - e).abs() < 1e-2, "scatter elem {i}: got {a}, want {e}");
    }
}

// ── masked_scatter: out[i] = mask[i] ? src[offsets[i]] : out[i] ───────

#[test]
fn masked_scatter_matches_naive_f32() {
    let _g = gpu_lock();
    let n = 32usize;
    let n_src = 16usize;
    let src: Vec<f32> = (0..n_src).map(|i| i as f32 * 2.0 - 10.0).collect();
    // Every 3rd slot is masked-in.
    let mask: Vec<u32> = (0..n).map(|i| u32::from(i % 3 == 0)).collect();
    // Offsets: masked slots pull a varied src index; unmasked slots
    // carry a harmless in-bounds index (value is discarded).
    let offsets: Vec<u32> =
        (0..n).map(|i| if i % 3 == 0 { ((i * 7 + 2) % n_src) as u32 } else { 0 }).collect();
    let out_init: Vec<f32> = (0..n).map(|i| 1000.0 + i as f32).collect();

    let mut expected = out_init.clone();
    for i in 0..n {
        if mask[i] != 0 {
            expected[i] = src[offsets[i] as usize];
        }
    }

    let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b.insert("mask".into(), pack_u32_bytes(&mask));
    b.insert("offsets".into(), pack_u32_bytes(&offsets));
    b.insert("src".into(), pack_bytes(&src, Dt::F32));
    b.insert("out".into(), pack_bytes(&out_init, Dt::F32));
    b.insert("n_elems".into(), (n as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_masked_scatter::kernel_ir_for(Dt::F32.to_dtype());
    kernel.mode = KernelMode::Grid3D;
    let result = ctx
        .dispatch_with_grid(&kernel, &b, &BTreeMap::new(), [n.div_ceil(64), 1, 1], [64, 1, 1])
        .expect("masked_scatter dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), Dt::F32);
    out.truncate(n);
    for (i, (a, e)) in out.iter().zip(&expected).enumerate() {
        assert!((a - e).abs() < 1e-6, "masked_scatter elem {i}: got {a}, want {e}");
    }
}

#[test]
fn masked_scatter_all_unmasked_is_identity_f32() {
    let _g = gpu_lock();
    // No masked slots — output must equal the pre-initialized `out`.
    let n = 24usize;
    let src: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let mask: Vec<u32> = vec![0u32; n];
    let offsets: Vec<u32> = vec![0u32; n];
    let out_init: Vec<f32> = (0..n).map(|i| 7.0 + i as f32 * 0.5).collect();

    let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b.insert("mask".into(), pack_u32_bytes(&mask));
    b.insert("offsets".into(), pack_u32_bytes(&offsets));
    b.insert("src".into(), pack_bytes(&src, Dt::F32));
    b.insert("out".into(), pack_bytes(&out_init, Dt::F32));
    b.insert("n_elems".into(), (n as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_masked_scatter::kernel_ir_for(Dt::F32.to_dtype());
    kernel.mode = KernelMode::Grid3D;
    let result = ctx
        .dispatch_with_grid(&kernel, &b, &BTreeMap::new(), [n.div_ceil(64), 1, 1], [64, 1, 1])
        .expect("masked_scatter dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), Dt::F32);
    out.truncate(n);
    for (i, (a, e)) in out.iter().zip(&out_init).enumerate() {
        assert!((a - e).abs() < 1e-6, "identity elem {i}: got {a}, want {e}");
    }
}
