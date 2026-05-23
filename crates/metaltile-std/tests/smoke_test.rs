//! Fast "heartbeat" smoke test for the critical FFAI kernel path.
//!
//! One binary, one `gpu_lock()`, ~6 kernels with tiny shapes.
//! Finishes in < 30 s on GHA macOS runners (vs. 5–10 min for the
//! full 130+ binary correctness suite).
//!
//! Coverage:
//!   - `gather`        — embedding lookup (Grid3D)
//!   - `rms_norm`      — pre-attention normalisation (Reduction)
//!   - `kv_cache_update` — single-token cache append (Grid3D)
//!   - `dequant_gemv_int4` — quantized decode linear (Reduction)
//!   - `gemm`          — prefill matmul (Reduction, 2-D grid)
//!   - `sdpa_decode`   — single-token attention (Reduction, strict TPG)
//!
//! If any of these fail, the full coherence suite would fail too —
//! but this gives feedback in seconds rather than minutes.
//!
//! macOS-gated: needs a real Metal device.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, naive_rms_norm_f32, pack_bytes, pack_u32_bytes, ramp, unpack_bytes};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::{
    ffai::{
        dequant_gemv::dequant_gemv_int4,
        gather::ffai_gather,
        gemm::ffai_gemm,
        kv_cache::kv_cache_update,
        sdpa_decode::ffai_sdpa_decode,
    },
    mlx::rms_norm::mt_rms_norm,
};

fn f32_slice_to_bytes(vals: &[f32]) -> Vec<u8> { pack_bytes(vals, Dt::F32) }
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> { unpack_bytes(bytes, Dt::F32) }

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: element count");
    let mut max_diff = 0.0f32;
    let mut at = 0usize;
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let d = (a - e).abs();
        if d > max_diff {
            max_diff = d;
            at = i;
        }
    }
    assert!(
        max_diff < tol,
        "{label}: max |diff| = {max_diff:.2e} at {at} (expected {:.6}, got {:.6})",
        expected[at],
        actual[at]
    );
}

// ── 1. gather ─────────────────────────────────────────────────────────

fn smoke_gather(ctx: &Context) {
    let vocab = 8usize;
    let dim = 4usize;
    let n_tokens = 2usize;

    let table: Vec<f32> = (0..vocab * dim).map(|i| ((i / dim) * 1000 + (i % dim)) as f32).collect();
    let indices: Vec<u32> = vec![3, 0];

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("table".into(), f32_slice_to_bytes(&table));
    buffers.insert("indices".into(), pack_u32_bytes(&indices));
    buffers.insert("out".into(), f32_slice_to_bytes(&vec![0.0f32; n_tokens * dim]));
    buffers.insert("dim".into(), (dim as u32).to_le_bytes().to_vec());

    let mut kernel = ffai_gather::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Grid3D;

    let total = n_tokens * dim;
    let tpg = 256usize;
    let groups = total.div_ceil(tpg);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [groups, 1, 1], [tpg, 1, 1])
        .expect("gather dispatch");

    let out = bytes_to_f32_vec(result.outputs.get("out").expect("out"));
    for (token_i, &id) in indices.iter().enumerate() {
        for d in 0..dim {
            let expected = (id as usize * 1000 + d) as f32;
            let got = out[token_i * dim + d];
            assert!(
                (got - expected).abs() < 1e-6,
                "gather token {token_i} d={d}: expected {expected}, got {got}"
            );
        }
    }
}

// ── 2. rms_norm ───────────────────────────────────────────────────────

fn smoke_rms_norm(ctx: &Context) {
    let n = 128usize;
    let rows = 1usize;
    let eps = 1e-5_f32;
    let tpg = n / 4; // kernel invariant: N = TPG * 4

    let x = ramp(rows * n, 13, 5.0);
    let w = ramp(n, 7, 2.0);
    let expected = naive_rms_norm_f32(&x, &w, n, eps);

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), f32_slice_to_bytes(&x));
    buffers.insert("w".into(), f32_slice_to_bytes(&w));
    buffers.insert("out".into(), vec![0u8; rows * n * 4]);
    buffers.insert("eps_buf".into(), eps.to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());

    let mut kernel = mt_rms_norm::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [rows, 1, 1], [tpg, 1, 1])
        .expect("rms_norm dispatch");

    let actual = bytes_to_f32_vec(result.outputs.get("out").expect("out"));
    assert_close(&actual, &expected, 1e-3, "rms_norm");
}

// ── 3. kv_cache_update ──────────────────────────────────────────────

fn smoke_kv_cache_update(ctx: &Context) {
    let n_kv_heads = 2usize;
    let head_dim = 8usize;
    let max_seq = 4usize;
    let position = 1usize;

    let sentinel = 999.0_f32;
    let cache = vec![sentinel; n_kv_heads * max_seq * head_dim];
    let src: Vec<f32> = (0..n_kv_heads * head_dim).map(|i| 10.0 + i as f32).collect();

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("src".into(), f32_slice_to_bytes(&src));
    buffers.insert("out".into(), f32_slice_to_bytes(&cache));
    buffers.insert("head_dim".into(), (head_dim as u32).to_le_bytes().to_vec());
    buffers.insert("max_seq".into(), (max_seq as u32).to_le_bytes().to_vec());
    buffers.insert("position".into(), (position as u32).to_le_bytes().to_vec());

    let mut kernel = kv_cache_update::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Grid3D;

    let total = n_kv_heads * head_dim;
    let tpg = 256usize;
    let groups = total.div_ceil(tpg);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [groups, 1, 1], [tpg, 1, 1])
        .expect("kv_cache_update dispatch");

    let out = bytes_to_f32_vec(result.outputs.get("out").expect("out"));
    for h in 0..n_kv_heads {
        for s in 0..max_seq {
            for d in 0..head_dim {
                let idx = h * max_seq * head_dim + s * head_dim + d;
                let expected =
                    if s == position { 10.0 + (h * head_dim + d) as f32 } else { sentinel };
                assert!(
                    (out[idx] - expected).abs() < 1e-6,
                    "kv_cache_update h={h} s={s} d={d}: expected {expected}, got {}",
                    out[idx]
                );
            }
        }
    }
}

// ── 4. dequant_gemv_int4 ────────────────────────────────────────────

/// Per-group affine quantize a row to 4-bit values, packed as u32.
fn quantize_row_int4(row: &[f32], group_size: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let in_dim = row.len();
    let n_groups = in_dim / group_size;
    let n_u32 = in_dim * 4 / 32;
    let mut packed = vec![0u32; n_u32];
    let mut scales = vec![0.0_f32; n_groups];
    let mut biases = vec![0.0_f32; n_groups];
    let max_q = 15u32;

    for g in 0..n_groups {
        let g_slice = &row[g * group_size..(g + 1) * group_size];
        let mn = g_slice.iter().copied().fold(f32::INFINITY, f32::min);
        let mx = g_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = mx - mn;
        let scale = if range.abs() < 1e-10 { 1.0 } else { range / max_q as f32 };
        scales[g] = scale;
        biases[g] = mn;
        for (i, &v) in g_slice.iter().enumerate() {
            let q = ((v - mn) / scale).round().clamp(0.0, max_q as f32) as u32;
            let bit_off = ((g * group_size + i) * 4) as u32;
            let word = (bit_off / 32) as usize;
            let in_w = bit_off & 31;
            packed[word] |= q << in_w;
        }
    }
    (packed, scales, biases)
}

#[allow(clippy::needless_range_loop)]
fn naive_dequant_gemv_int4(
    weight: &[u32],
    scales: &[f32],
    biases: &[f32],
    input: &[f32],
    in_dim: usize,
    group_size: usize,
    out_dim: usize,
) -> Vec<f32> {
    let u32_per_row = in_dim * 4 / 32;
    let n_groups = in_dim / group_size;
    let mut out = vec![0.0_f32; out_dim];
    for row in 0..out_dim {
        let mut acc = 0.0_f32;
        let row_w = &weight[row * u32_per_row..(row + 1) * u32_per_row];
        let row_s = &scales[row * n_groups..(row + 1) * n_groups];
        let row_b = &biases[row * n_groups..(row + 1) * n_groups];
        for d in 0..in_dim {
            let g = d / group_size;
            let bit_off = (d * 4) as u32;
            let word = (bit_off / 32) as usize;
            let in_w = bit_off & 31;
            let q = ((row_w[word] as u64) >> in_w) & 0xF;
            let w_real = (q as f32) * row_s[g] + row_b[g];
            acc += w_real * input[d];
        }
        out[row] = acc;
    }
    out
}

fn smoke_dequant_gemv_int4(ctx: &Context) {
    let in_dim = 32usize;
    let out_dim = 2usize;
    let group_size = 8usize;

    let input = ramp(in_dim, 11, 3.0);
    let mut all_packed = Vec::new();
    let mut all_scales = Vec::new();
    let mut all_biases = Vec::new();
    for _ in 0..out_dim {
        let row = ramp(in_dim, 17, 4.0);
        let (packed, scales, biases) = quantize_row_int4(&row, group_size);
        all_packed.extend_from_slice(&packed);
        all_scales.extend_from_slice(&scales);
        all_biases.extend_from_slice(&biases);
    }

    let expected = naive_dequant_gemv_int4(
        &all_packed,
        &all_scales,
        &all_biases,
        &input,
        in_dim,
        group_size,
        out_dim,
    );

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("weight".into(), pack_u32_bytes(&all_packed));
    buffers.insert("scales".into(), f32_slice_to_bytes(&all_scales));
    buffers.insert("biases".into(), f32_slice_to_bytes(&all_biases));
    buffers.insert("input".into(), f32_slice_to_bytes(&input));
    buffers.insert("output".into(), f32_slice_to_bytes(&vec![0.0f32; out_dim]));
    buffers.insert("in_dim".into(), (in_dim as u32).to_le_bytes().to_vec());
    buffers.insert("out_dim".into(), (out_dim as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());

    let mut kernel = dequant_gemv_int4::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    let tpg = 128usize;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [out_dim, 1, 1], [tpg, 1, 1])
        .expect("dequant_gemv_int4 dispatch");

    let actual = bytes_to_f32_vec(result.outputs.get("output").expect("output"));
    assert_close(&actual, &expected, 5e-2, "dequant_gemv_int4");
}

// ── 5. gemm ───────────────────────────────────────────────────────────

fn naive_gemm(
    weight: &[f32],
    input: &[f32],
    n_rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_rows * out_dim];
    for r in 0..n_rows {
        for o in 0..out_dim {
            let mut acc = 0.0f32;
            for k in 0..in_dim {
                acc += weight[o * in_dim + k] * input[r * in_dim + k];
            }
            out[r * out_dim + o] = acc;
        }
    }
    out
}

fn smoke_gemm(ctx: &Context) {
    let n_rows = 8usize;
    let in_dim = 16usize;
    let out_dim = 16usize;

    let weight = ramp(out_dim * in_dim, 31, 14.0);
    let input = ramp(n_rows * in_dim, 23, 9.0);
    let expected = naive_gemm(&weight, &input, n_rows, in_dim, out_dim);

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("weight".into(), f32_slice_to_bytes(&weight));
    buffers.insert("input".into(), f32_slice_to_bytes(&input));
    buffers.insert("out".into(), f32_slice_to_bytes(&vec![0.0f32; n_rows * out_dim]));
    buffers.insert("in_dim".into(), (in_dim as u32).to_le_bytes().to_vec());
    buffers.insert("out_dim".into(), (out_dim as u32).to_le_bytes().to_vec());
    buffers.insert("n_rows".into(), (n_rows as u32).to_le_bytes().to_vec());

    let mut kernel = ffai_gemm::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    let n_tiles = out_dim.div_ceil(32);
    let m_tiles = n_rows.div_ceil(32);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n_tiles, m_tiles, 1], [
            1024, 1, 1,
        ])
        .expect("gemm dispatch");

    let actual = bytes_to_f32_vec(result.outputs.get("out").expect("out"));
    assert_close(&actual, &expected, 1e-3, "gemm");
}

// ── 6. sdpa_decode ────────────────────────────────────────────────────

#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
fn naive_sdpa_decode(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_q_heads: usize,
    head_dim: usize,
    n_kv: usize,
    kv_stride: usize,
    heads_per_group: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_q_heads * head_dim];
    for qh in 0..n_q_heads {
        let kvh = qh / heads_per_group;
        let q_off = qh * head_dim;
        let kv_slab = kvh * kv_stride * head_dim;
        let mut scores = vec![0.0f32; n_kv];
        for t in 0..n_kv {
            let k_off = kv_slab + t * head_dim;
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[q_off + d] * k[k_off + d];
            }
            scores[t] = dot * scale;
        }
        let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - m).exp();
            sum += *s;
        }
        let inv = 1.0 / sum;
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for t in 0..n_kv {
                let v_off = kv_slab + t * head_dim + d;
                acc += scores[t] * inv * v[v_off];
            }
            out[q_off + d] = acc;
        }
    }
    out
}

fn smoke_sdpa_decode(ctx: &Context) {
    let n_q_heads = 2usize;
    let n_kv_heads = 1usize;
    let head_dim = 128usize;
    let n_kv = 4usize;
    let kv_stride = 4usize;
    let heads_per_group = n_q_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q = ramp(n_q_heads * head_dim, 11, 5.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 3.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 17, 7.0);
    let expected =
        naive_sdpa_decode(&q, &k, &v, n_q_heads, head_dim, n_kv, kv_stride, heads_per_group, scale);

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("q".into(), f32_slice_to_bytes(&q));
    buffers.insert("k".into(), f32_slice_to_bytes(&k));
    buffers.insert("v".into(), f32_slice_to_bytes(&v));
    buffers.insert("out".into(), vec![0u8; n_q_heads * head_dim * 4]);
    buffers.insert("head_dim".into(), (head_dim as u32).to_le_bytes().to_vec());
    buffers.insert("n_kv".into(), (n_kv as u32).to_le_bytes().to_vec());
    buffers.insert("kv_stride".into(), (kv_stride as u32).to_le_bytes().to_vec());
    buffers.insert("heads_per_group".into(), (heads_per_group as u32).to_le_bytes().to_vec());
    buffers.insert("sink_end".into(), 0u32.to_le_bytes().to_vec());
    buffers.insert("window_start".into(), 0u32.to_le_bytes().to_vec());
    buffers.insert("has_sink".into(), 0u32.to_le_bytes().to_vec());
    buffers.insert("sink_logit".into(), 0.0f32.to_le_bytes().to_vec());
    buffers.insert("scale".into(), scale.to_le_bytes().to_vec());

    let mut kernel = ffai_sdpa_decode::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n_q_heads, 1, 1], [1024, 1, 1])
        .expect("sdpa_decode dispatch");

    let actual = bytes_to_f32_vec(result.outputs.get("out").expect("out"));
    assert_close(&actual, &expected, 5e-2, "sdpa_decode");
}

// ── Orchestrator ─────────────────────────────────────────────────────

#[test]
fn ffai_kernel_smoke() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new should succeed on macOS");

    smoke_gather(&ctx);
    smoke_rms_norm(&ctx);
    smoke_kv_cache_update(&ctx);
    smoke_dequant_gemv_int4(&ctx);
    smoke_gemm(&ctx);
    smoke_sdpa_decode(&ctx);
}
