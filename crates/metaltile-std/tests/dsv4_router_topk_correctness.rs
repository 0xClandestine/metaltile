//! Copyright 2026 TheTom
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for `ffai::mt_dsv4_router_topk` — top-K by biased
//! score, weights = unbiased[chosen] renormalised to sum 1.
#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes, unpack_u32_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::dsv4_router_topk::{mt_dsv4_router_topk, mt_remap_u32};

#[test]
fn dsv4_router_topk_f32() {
    let _g = gpu_lock();
    let n = 256usize;
    let k = 6usize;
    // Deterministic distinct biased scores so the top-K is unambiguous.
    let biased: Vec<f32> = (0..n).map(|i| ((i * 37 + 11) % 251) as f32 * 0.1).collect();
    let unbiased: Vec<f32> = (0..n).map(|i| ((i * 53 + 7) % 199) as f32 * 0.01 + 0.05).collect();

    // CPU reference.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| biased[b].partial_cmp(&biased[a]).unwrap());
    let chosen: Vec<usize> = order.iter().take(k).copied().collect();
    let wsum: f32 = chosen.iter().map(|&e| unbiased[e]).sum();
    let want_w: Vec<f32> = chosen.iter().map(|&e| unbiased[e] / wsum).collect();

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("score_biased".into(), pack_bytes(&biased, Dt::F32));
    buffers.insert("score_unbiased".into(), pack_bytes(&unbiased, Dt::F32));
    buffers.insert("indices_out".into(), vec![0u8; k * 4]);
    buffers.insert("weights_out".into(), pack_bytes(&vec![0.0f32; k], Dt::F32));
    buffers.insert("n_experts".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("ctx");
    let mut kernel = mt_dsv4_router_topk::kernel_ir_for(Dt::F32.to_dtype());
    kernel.mode = KernelMode::Reduction;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, 1, 1], [32, 1, 1])
        .expect("dispatch");
    let got_idx = unpack_u32_bytes(result.outputs.get("indices_out").expect("idx"));
    let got_w = unpack_bytes(result.outputs.get("weights_out").expect("w"), Dt::F32);

    assert_eq!(got_idx.iter().map(|&x| x as usize).collect::<Vec<_>>(), chosen, "indices");
    for (i, (g, w)) in got_w.iter().zip(want_w.iter()).enumerate() {
        assert!((g - w).abs() < 1e-4, "weight {i}: got={g} want={w}");
    }
}

/// `mt_remap_u32`: out[i] = table[idx[i]] — a plain u32 gather over `n`
/// elements. Non-generic Grid3D kernel (one thread per output), so it
/// dispatches via `kernel_ir()` and a [n,1,1] grid with [1,1,1] tg.
#[test]
fn remap_u32_matches_cpu() {
    let _g = gpu_lock();
    let n = 6usize;
    let table_len = 256usize;
    let table: Vec<u32> = (0..table_len).map(|e| ((e * 37 + 11) % table_len) as u32).collect();
    let idx: Vec<u32> = vec![3, 200, 0, 255, 128, 64];
    let want: Vec<u32> = idx.iter().map(|&e| table[e as usize]).collect();

    let to_bytes = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("table".into(), to_bytes(&table));
    buffers.insert("idx".into(), to_bytes(&idx));
    buffers.insert("out".into(), vec![0u8; n * 4]);
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("ctx");
    let kernel = mt_remap_u32::kernel_ir();
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n, 1, 1], [1, 1, 1])
        .expect("dispatch");
    let got = unpack_u32_bytes(result.outputs.get("out").expect("out"));

    eprintln!("want={want:?} got={got:?}");
    assert_eq!(got, want, "remap_u32 gather mismatch");
}
