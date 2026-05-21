//! GPU correctness for `mlx::gather_axis` — contiguous gather along an
//! axis: `out[o,a,i] = src[o, indices[o,a,i], i]`.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::mlx::gather_axis::mt_gather_axis;

#[test]
fn gather_axis_matches_naive_f32() {
    let _g = gpu_lock();
    let (outer, axis_size, axis_out, inner) = (3usize, 7usize, 5usize, 4usize);
    let src: Vec<f32> = (0..outer * axis_size * inner).map(|i| i as f32 * 0.5 - 2.0).collect();
    // Deterministic varied indices in [0, axis_size).
    let indices: Vec<u32> =
        (0..outer * axis_out * inner).map(|i| ((i * 3 + 1) % axis_size) as u32).collect();

    let mut expected = vec![0.0_f32; outer * axis_out * inner];
    for o in 0..outer {
        for a in 0..axis_out {
            for i in 0..inner {
                let oi = (o * axis_out + a) * inner + i;
                let g = indices[oi] as usize;
                expected[oi] = src[(o * axis_size + g) * inner + i];
            }
        }
    }

    let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b.insert("src".into(), pack_bytes(&src, Dt::F32));
    b.insert("indices".into(), pack_u32_bytes(&indices));
    b.insert("out".into(), pack_bytes(&vec![0.0; expected.len()], Dt::F32));
    b.insert("axis_out".into(), (axis_out as u32).to_le_bytes().to_vec());
    b.insert("axis_size".into(), (axis_size as u32).to_le_bytes().to_vec());
    b.insert("inner".into(), (inner as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_gather_axis::kernel_ir_for(Dt::F32.to_dtype());
    kernel.mode = KernelMode::Grid3D;
    let total = outer * axis_out * inner;
    let result = ctx
        .dispatch_with_grid(&kernel, &b, &BTreeMap::new(), [total.div_ceil(64), 1, 1], [64, 1, 1])
        .expect("gather_axis dispatch");
    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), Dt::F32);
    out.truncate(total);
    for (i, (a, e)) in out.iter().zip(&expected).enumerate() {
        assert!((a - e).abs() < 1e-6, "elem {i}: got {a}, want {e}");
    }
}
