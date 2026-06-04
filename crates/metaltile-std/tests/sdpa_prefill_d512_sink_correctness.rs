//! Copyright 2026 TheTom
//! SPDX-License-Identifier: Apache-2.0
//! GPU correctness for `ffai::ffai_sdpa_prefill_d512_sink` — multi-query
//! causal sliding-window SDPA (d512, attn sink, MQA). Oracle: per-(q_pos,
//! q_head) causal softmax with sink over the KV window.
#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile::core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::sdpa_prefill_d512_sink::ffai_sdpa_prefill_d512_sink;

fn xorshift(s: &mut u32) -> u32 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    x
}
fn frand(s: &mut u32) -> f32 { (xorshift(s) as f32 / u32::MAX as f32) * 0.2 - 0.1 }

fn run_case(dt: Dt, n_query: usize, kv: usize, window: usize, tol: f32) {
    run_case_nq(dt, 4, n_query, kv, window, tol);
}

fn run_case_nq(dt: Dt, n_q: usize, n_query: usize, kv: usize, window: usize, tol: f32) {
    let _g = gpu_lock();
    let hd = 512usize;
    let heads_per_group = n_q; // MQA: 1 kv head
    let scale = 1.0f32 / (hd as f32).sqrt();
    let mut st = 0x5D9A_2026u32;

    let q: Vec<f32> = (0..n_query * n_q * hd).map(|_| frand(&mut st)).collect();
    let k: Vec<f32> = (0..kv * hd).map(|_| frand(&mut st)).collect();
    let v: Vec<f32> = (0..kv * hd).map(|_| frand(&mut st)).collect();
    let sink: Vec<f32> = (0..n_q).map(|_| frand(&mut st)).collect();

    // Oracle (kv_base = 0).
    let mut want = vec![0.0f32; n_query * n_q * hd];
    for qp in 0..n_query {
        let p1 = qp + 1;
        let lo = p1.saturating_sub(window);
        #[allow(clippy::needless_range_loop)]
        for h in 0..n_q {
            let qo = (qp * n_q + h) * hd;
            let mut scores = vec![0.0f32; p1 - lo];
            for (i, t) in (lo..p1).enumerate() {
                let mut s = 0.0f32;
                for d in 0..hd {
                    s += q[qo + d] * scale * k[t * hd + d];
                }
                scores[i] = s;
            }
            let mut m = sink[h];
            for &s in &scores {
                if s > m {
                    m = s;
                }
            }
            let mut denom = (sink[h] - m).exp();
            for &s in &scores {
                denom += (s - m).exp();
            }
            for d in 0..hd {
                let mut acc = 0.0f32;
                for (i, t) in (lo..p1).enumerate() {
                    acc += ((scores[i] - m).exp() / denom) * v[t * hd + d];
                }
                want[qo + d] = acc;
            }
        }
    }

    let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    b.insert("q".into(), pack_bytes(&q, dt));
    b.insert("k".into(), pack_bytes(&k, dt));
    b.insert("v".into(), pack_bytes(&v, dt));
    b.insert("sink_logit".into(), pack_bytes(&sink, Dt::F32));
    b.insert("out".into(), pack_bytes(&vec![0.0f32; n_query * n_q * hd], dt));
    b.insert("head_dim".into(), (hd as u32).to_le_bytes().to_vec());
    b.insert("n_q_heads".into(), (n_q as u32).to_le_bytes().to_vec());
    b.insert("kv_stride".into(), (kv as u32).to_le_bytes().to_vec());
    b.insert("heads_per_group".into(), (heads_per_group as u32).to_le_bytes().to_vec());
    b.insert("window".into(), (window as u32).to_le_bytes().to_vec());
    b.insert("kv_base".into(), 0u32.to_le_bytes().to_vec());
    b.insert("scale".into(), scale.to_le_bytes().to_vec());

    let ctx = Context::new().unwrap();
    let mut kern = ffai_sdpa_prefill_d512_sink::kernel_ir_for(dt.to_dtype());
    kern.mode = KernelMode::Reduction;
    let r =
        ctx.dispatch_with_grid(&kern, &b, &BTreeMap::new(), [n_q, n_query, 1], [32, 1, 1]).unwrap();
    let got = unpack_bytes(r.outputs.get("out").unwrap(), dt);

    for i in 0..(n_query * n_q * hd) {
        let denom = want[i].abs().max(0.05);
        assert!((got[i] - want[i]).abs() / denom < tol, "i={i}: got={} want={}", got[i], want[i]);
    }
}

#[test]
fn sdpa_prefill_d512_sink_f32() {
    run_case(Dt::F32, 8, 8, 128, 1e-3); // full causal (window > kv)
    run_case(Dt::F32, 40, 40, 16, 1e-3); // sliding window active
}

#[test]
fn sdpa_prefill_d512_sink_single_token_64heads() {
    // DSv4 decode-parity: 1 query, 1 KV, 64 q heads (MQA), per-head sink.
    run_case_nq(Dt::F32, 64, 1, 1, 128, 1e-3);
    run_case_nq(Dt::F32, 64, 4, 4, 128, 1e-3);
}

#[test]
fn sdpa_prefill_d512_sink_f16() { run_case(Dt::F16, 8, 8, 128, 2e-2); }
