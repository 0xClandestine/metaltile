//! GPU correctness for `mlx::scatter_axis` — contiguous scatter along
//! an axis: `out[o, indices[o,a,i], i] = updates[o,a,i]`.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::mlx::scatter_axis::mt_scatter_axis;

#[test]
fn scatter_axis_matches_naive_f32() {
    let _g = gpu_lock();
    let (outer, axis_size, axis_upd, inner) = (3usize, 7usize, 5usize, 4usize);
    // Distinct indices per (o, i) column: a permutation prefix so no
    // two updates collide — required for a deterministic no-reduce
    // scatter.
    let mut indices = vec![0u32; outer * axis_upd * inner];
    for o in 0..outer {
        for i in 0..inner {
            for a in 0..axis_upd {
                // shift the permutation per (o,i) so it varies.
                indices[(o * axis_upd + a) * inner + i] = ((a + o + i) % axis_size) as u32;
            }
        }
    }
    let updates: Vec<f32> = (0..outer * axis_upd * inner).map(|i| i as f32 * 0.25 + 1.0).collect();
    let init: Vec<f32> = (0..outer * axis_size * inner).map(|i| -(i as f32)).collect();

    let mut expected = init.clone();
    for o in 0..outer {
        for a in 0..axis_upd {
            for i in 0..inner {
                let ui = (o * axis_upd + a) * inner + i;
                let s = indices[ui] as usize;
                expected[(o * axis_size + s) * inner + i] = updates[ui];
            }
        }
    }

    let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b.insert("updates".into(), pack_bytes(&updates, Dt::F32));
    b.insert("indices".into(), pack_u32_bytes(&indices));
    b.insert("out".into(), pack_bytes(&init, Dt::F32));
    b.insert("axis_upd".into(), (axis_upd as u32).to_le_bytes().to_vec());
    b.insert("axis_size".into(), (axis_size as u32).to_le_bytes().to_vec());
    b.insert("inner".into(), (inner as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_scatter_axis::kernel_ir_for(Dt::F32.to_dtype());
    kernel.mode = KernelMode::Grid3D;
    let total = outer * axis_upd * inner;
    let result = ctx
        .dispatch_with_grid(&kernel, &b, &BTreeMap::new(), [total.div_ceil(64), 1, 1], [64, 1, 1])
        .expect("scatter_axis dispatch");
    let out = unpack_bytes(result.outputs.get("out").expect("out"), Dt::F32);
    for (i, (a, e)) in out.iter().take(expected.len()).zip(&expected).enumerate() {
        assert!((a - e).abs() < 1e-6, "elem {i}: got {a}, want {e}");
    }
}
