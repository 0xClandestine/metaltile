//! Copyright 2026 TheTom
//! SPDX-License-Identifier: Apache-2.0
//! Isolated check: ffai_sdpa_decode_d512_sink with n_kv=1 (single visible
//! KV — the first decode token), many q heads (MQA), per-head sink. The
//! in-source test only covers n_kv=64; DSv4 token 0 hits n_kv=1.
#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile::core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::sdpa_decode_d512_sink::ffai_sdpa_decode_d512_sink;

fn xorshift(s: &mut u32) -> u32 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    x
}
fn frand(s: &mut u32) -> f32 { (xorshift(s) as f32 / u32::MAX as f32) * 0.2 - 0.1 }

#[test]
fn sdpa_decode_d512_sink_nkv1_64heads() {
    let _g = gpu_lock();
    let dt = Dt::F32;
    let hd = 512usize;
    let n_q = 64usize;
    let n_kv = 1usize;
    let kv_stride = 128usize; // like the real sliding-window cache
    let heads_per_group = n_q; // MQA: 1 kv head
    let scale = 1.0f32 / (hd as f32).sqrt();
    let mut st = 0x5D9A_2026u32;

    let q: Vec<f32> = (0..n_q * hd).map(|_| frand(&mut st)).collect();
    let k: Vec<f32> = (0..kv_stride * hd).map(|_| frand(&mut st)).collect();
    let v: Vec<f32> = (0..kv_stride * hd).map(|_| frand(&mut st)).collect();
    let sink: Vec<f32> = (0..n_q).map(|_| frand(&mut st)).collect();

    // Oracle.
    let mut want = vec![0.0f32; n_q * hd];
    #[allow(clippy::needless_range_loop)]
    for h in 0..n_q {
        let qo = h * hd;
        let mut s = 0.0f32;
        for d in 0..hd {
            s += q[qo + d] * scale * k[d];
        } // single key at row 0
        let m = s.max(sink[h]);
        let denom = (s - m).exp() + (sink[h] - m).exp();
        let w = (s - m).exp() / denom;
        for d in 0..hd {
            want[qo + d] = w * v[d];
        }
    }

    let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b.insert("q".into(), pack_bytes(&q, dt));
    b.insert("k".into(), pack_bytes(&k, dt));
    b.insert("v".into(), pack_bytes(&v, dt));
    b.insert("sink_logit".into(), pack_bytes(&sink, Dt::F32));
    b.insert("out".into(), pack_bytes(&vec![0.0f32; n_q * hd], dt));
    b.insert("head_dim".into(), (hd as u32).to_le_bytes().to_vec());
    b.insert("n_kv".into(), (n_kv as u32).to_le_bytes().to_vec());
    b.insert("kv_stride".into(), (kv_stride as u32).to_le_bytes().to_vec());
    b.insert("heads_per_group".into(), (heads_per_group as u32).to_le_bytes().to_vec());
    b.insert("scale".into(), scale.to_le_bytes().to_vec());

    let ctx = Context::new().unwrap();
    let mut kern = ffai_sdpa_decode_d512_sink::kernel_ir_for(dt.to_dtype());
    kern.mode = KernelMode::Reduction;
    let r = ctx.dispatch_with_grid(&kern, &b, &BTreeMap::new(), [n_q, 1, 1], [512, 1, 1]).unwrap();
    let got = unpack_bytes(r.outputs.get("out").unwrap(), dt);

    let mut max_rel = 0.0f32;
    for i in 0..(n_q * hd) {
        let denom = want[i].abs().max(0.05);
        let rel = (got[i] - want[i]).abs() / denom;
        if rel > max_rel {
            max_rel = rel;
        }
    }
    eprintln!("nkv1 64head max_rel={max_rel}");
    // Spot-check head 5 specifically (the FFAI case that zeroed).
    eprintln!("head5: got={:?} want={:?}", &got[5 * hd..5 * hd + 4], &want[5 * hd..5 * hd + 4]);
    assert!(max_rel < 1e-3, "max_rel={max_rel}");
}
