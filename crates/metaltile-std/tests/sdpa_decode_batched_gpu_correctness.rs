//! End-to-end GPU correctness for `sdpa_decode_batched_q2` (M7).
//!
//! Strategy: build two independent Q vectors (`Q_a` and `Q_b`) per head,
//! interleave them into the batched layout `[n_q_heads, 2, head_dim]`,
//! dispatch the batched kernel once, then assert that
//! `out[h, 0, :] ≈ naive_sdpa(Q_a)[h, :]` and
//! `out[h, 1, :] ≈ naive_sdpa(Q_b)[h, :]`.
//!
//! This catches:
//!   * Wrong Q indexing in `q_off_0` / `q_off_1` (both outputs would
//!     match the same reference).
//!   * Wrong KV-walk interleaving (factor/weight applied to the wrong
//!     stream).
//!   * Cross-phase tg-buffer aliasing (Phase B reading Q[0]'s residual
//!     state).
//!   * Wrong output offsets (Q[1] writes landing in Q[0]'s slot).
//!
//! macOS-gated: needs an actual Metal device. Mirrors the layout of
//! `sdpa_decode_gpu_correctness.rs`.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, SdpaShape, gpu_lock, naive_sdpa_f32, pack_bytes, ramp, unpack_bytes};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::ffai::sdpa_decode_batched::{sdpa_decode_batched_q2, sdpa_decode_batched_q4};

fn f32_slice_to_bytes(vals: &[f32]) -> Vec<u8> { pack_bytes(vals, Dt::F32) }
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> { unpack_bytes(bytes, Dt::F32) }

/// Interleave two `[n_q_heads, head_dim]` arrays into `[n_q_heads, 2,
/// head_dim]`: for each head, slot 0 is `q_a[head]`, slot 1 is
/// `q_b[head]`. This matches the kernel's Q layout where
/// `q_off_0 = q_head * 2 * head_dim` and `q_off_1 = q_off_0 + head_dim`.
fn interleave_q(q_a: &[f32], q_b: &[f32], n_q_heads: usize, head_dim: usize) -> Vec<f32> {
    assert_eq!(q_a.len(), n_q_heads * head_dim);
    assert_eq!(q_b.len(), n_q_heads * head_dim);
    let mut out = Vec::with_capacity(n_q_heads * 2 * head_dim);
    for h in 0..n_q_heads {
        out.extend_from_slice(&q_a[h * head_dim..(h + 1) * head_dim]);
        out.extend_from_slice(&q_b[h * head_dim..(h + 1) * head_dim]);
    }
    out
}

/// Take an output buffer in the batched layout `[n_q_heads, 2,
/// head_dim]` and split it into per-Q-slot views. Returns `(out_0,
/// out_1)` where each is `[n_q_heads, head_dim]`.
fn split_batched_out(batched: &[f32], n_q_heads: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(batched.len(), n_q_heads * 2 * head_dim);
    let mut out_0 = Vec::with_capacity(n_q_heads * head_dim);
    let mut out_1 = Vec::with_capacity(n_q_heads * head_dim);
    for h in 0..n_q_heads {
        let base = h * 2 * head_dim;
        out_0.extend_from_slice(&batched[base..base + head_dim]);
        out_1.extend_from_slice(&batched[base + head_dim..base + 2 * head_dim]);
    }
    (out_0, out_1)
}

#[allow(clippy::too_many_arguments)]
fn run_sdpa_decode_batched_q2_f32(
    ctx: &Context,
    kernel: &metaltile_core::ir::Kernel,
    q_batched: &[f32],
    k: &[f32],
    v: &[f32],
    n_q_heads: usize,
    head_dim: usize,
    n_kv: usize,
    kv_stride: usize,
    heads_per_group: usize,
    scale: f32,
) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("q".into(), f32_slice_to_bytes(q_batched));
    buffers.insert("k".into(), f32_slice_to_bytes(k));
    buffers.insert("v".into(), f32_slice_to_bytes(v));
    // Out layout is [n_q_heads, 2, head_dim].
    buffers.insert("out".into(), vec![0u8; n_q_heads * 2 * head_dim * 4]);
    buffers.insert("head_dim".into(), (head_dim as u32).to_le_bytes().to_vec());
    buffers.insert("n_kv".into(), (n_kv as u32).to_le_bytes().to_vec());
    buffers.insert("kv_stride".into(), (kv_stride as u32).to_le_bytes().to_vec());
    buffers.insert("heads_per_group".into(), (heads_per_group as u32).to_le_bytes().to_vec());
    buffers.insert("scale".into(), scale.to_le_bytes().to_vec());

    let result = ctx
        .dispatch_with_grid(kernel, &buffers, &BTreeMap::new(), [n_q_heads, 1, 1], [1024, 1, 1])
        .expect("dispatch_with_grid should succeed");
    let out_bytes = result.outputs.get("out").expect("`out` buffer in dispatch result");
    bytes_to_f32_vec(out_bytes)
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: element count");
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
        max_diff < tol,
        "{label}: max |diff| = {max_diff:.2e} at index {max_at} (expected {:.6}, got {:.6})",
        expected[max_at],
        actual[max_at],
    );
}

// ── Tolerance derivation ────────────────────────────────────────────────
//
// The K=2/4 decode-form kernels run online softmax in fp32 throughout
// (storage in T). The dominant numerical error source is `simd_sum`'s
// reorder noise across the n_kv-length walk: each KV position contributes
// one `simd_sum(partial)` which floats up through a tree reduction whose
// ordering is implementation-defined. The expected upper bound on the
// per-output element error is approximately:
//
//   tol ≈ C · n_kv · eps_f32 · max_abs_value
//
// For the test inputs (ramp values bounded by ~1 in absolute value) and
// f32 (eps ≈ 1.19e-7), this gives:
//
//   n_kv =   4: tol_theory ≈ 4 × 1.19e-7 ≈ 5e-7   → 1e-4 chosen (200× margin)
//   n_kv = 1024: tol_theory ≈ 1024 × 1.19e-7 ≈ 1.2e-4 → 5e-4 chosen (~4× margin)
//
// The tighter `1e-6` for the identical-Q sanity check is what fp32
// gives when both sides ran the same simd_sum tree on the same inputs
// — i.e. only the rescale `exp(run_max - g_max) / g_sum` ULP drift
// shows up.

/// Small shape — n_kv=4, n_q_heads=2 — picks up the basic layout/index
/// bugs without burying them under floating-point reduction noise.
#[test]
fn sdpa_decode_batched_q2_matches_two_independent_sdpa_decodes_f32() {
    let _g = gpu_lock();
    let n_q_heads = 2usize;
    let n_kv_heads = 1usize;
    let head_dim = 128usize;
    let n_kv = 4usize;
    let kv_stride = 4usize;
    let heads_per_group = n_q_heads / n_kv_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    // Two visibly-different Q tensors so Q[0]'s output can't match
    // Q[1]'s reference (and vice-versa) by accident.
    let q_a = ramp(n_q_heads * head_dim, 17, 8.0);
    let q_b = ramp(n_q_heads * head_dim, 11, 5.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 19, 9.0);

    let shape = SdpaShape { n_q_heads, n_kv_heads, head_dim, n_kv, scale };
    let expected_a = naive_sdpa_f32(&q_a, &k, &v, &shape);
    let expected_b = naive_sdpa_f32(&q_b, &k, &v, &shape);

    let q_batched = interleave_q(&q_a, &q_b, n_q_heads, head_dim);

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = sdpa_decode_batched_q2::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    let batched_out = run_sdpa_decode_batched_q2_f32(
        &ctx,
        &kernel,
        &q_batched,
        &k,
        &v,
        n_q_heads,
        head_dim,
        n_kv,
        kv_stride,
        heads_per_group,
        scale,
    );
    let (out_0, out_1) = split_batched_out(&batched_out, n_q_heads, head_dim);

    assert_close(&out_0, &expected_a, 1e-4, "sdpa_decode_batched_q2 Q[0] vs naive_sdpa(Q_a)");
    assert_close(&out_1, &expected_b, 1e-4, "sdpa_decode_batched_q2 Q[1] vs naive_sdpa(Q_b)");
}

/// Larger shape — n_kv=1024, n_q_heads=8 (GQA factor 4) — exercises the
/// cross-simdgroup reduction at the scale where 32 simdgroups each
/// stride-walk 32 KV positions. If the two-phase output reduction got
/// the tg-buffer reuse wrong, the larger shape is where it surfaces
/// (more sg's contributing, more rescale magnitudes).
#[test]
fn sdpa_decode_batched_q2_matches_at_larger_n_kv_f32() {
    let _g = gpu_lock();
    let n_q_heads = 8usize;
    let n_kv_heads = 2usize; // gqa_factor = 4
    let head_dim = 128usize;
    let n_kv = 1024usize;
    let kv_stride = 1024usize;
    let heads_per_group = n_q_heads / n_kv_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q_a = ramp(n_q_heads * head_dim, 23, 11.0);
    let q_b = ramp(n_q_heads * head_dim, 29, 14.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 19, 9.0);

    let shape = SdpaShape { n_q_heads, n_kv_heads, head_dim, n_kv, scale };
    let expected_a = naive_sdpa_f32(&q_a, &k, &v, &shape);
    let expected_b = naive_sdpa_f32(&q_b, &k, &v, &shape);

    let q_batched = interleave_q(&q_a, &q_b, n_q_heads, head_dim);

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = sdpa_decode_batched_q2::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    let batched_out = run_sdpa_decode_batched_q2_f32(
        &ctx,
        &kernel,
        &q_batched,
        &k,
        &v,
        n_q_heads,
        head_dim,
        n_kv,
        kv_stride,
        heads_per_group,
        scale,
    );
    let (out_0, out_1) = split_batched_out(&batched_out, n_q_heads, head_dim);

    // Tolerance bumped to 5e-4 for the larger reduction depth — 1024
    // KV positions stack up `simd_sum` reorder noise that the n_kv=4
    // test doesn't see.
    assert_close(&out_0, &expected_a, 5e-4, "sdpa_decode_batched_q2 Q[0] @ n_kv=1024");
    assert_close(&out_1, &expected_b, 5e-4, "sdpa_decode_batched_q2 Q[1] @ n_kv=1024");
}

/// Sanity check: when Q[0] == Q[1] the two output slots must be
/// bit-identical (within fp32 noise). Catches phase B accidentally
/// reading Q[0]'s residual state by aliasing rather than reset.
#[test]
fn sdpa_decode_batched_q2_identical_qs_produce_identical_outputs_f32() {
    let _g = gpu_lock();
    let n_q_heads = 2usize;
    let n_kv_heads = 1usize;
    let head_dim = 128usize;
    let n_kv = 8usize;
    let kv_stride = 8usize;
    let heads_per_group = n_q_heads / n_kv_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q = ramp(n_q_heads * head_dim, 17, 8.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 19, 9.0);

    let q_batched = interleave_q(&q, &q, n_q_heads, head_dim);

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = sdpa_decode_batched_q2::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    let batched_out = run_sdpa_decode_batched_q2_f32(
        &ctx,
        &kernel,
        &q_batched,
        &k,
        &v,
        n_q_heads,
        head_dim,
        n_kv,
        kv_stride,
        heads_per_group,
        scale,
    );
    let (out_0, out_1) = split_batched_out(&batched_out, n_q_heads, head_dim);

    // Both phases ran the SAME math on the SAME inputs. Any divergence
    // here points at non-deterministic ordering inside the two-phase
    // reduction (e.g. lingering tg-buffer state, missing barrier).
    assert_close(&out_0, &out_1, 1e-6, "Q[0] vs Q[1] when Q vectors are equal");
}

// ── K=4 GPU correctness ─────────────────────────────────────────────────

fn interleave_q4(qs: [&[f32]; 4], n_q_heads: usize, head_dim: usize) -> Vec<f32> {
    for q in &qs {
        assert_eq!(q.len(), n_q_heads * head_dim);
    }
    let mut out = Vec::with_capacity(n_q_heads * 4 * head_dim);
    for h in 0..n_q_heads {
        for &q in &qs {
            out.extend_from_slice(&q[h * head_dim..(h + 1) * head_dim]);
        }
    }
    out
}

fn split_batched_out_q4(batched: &[f32], n_q_heads: usize, head_dim: usize) -> [Vec<f32>; 4] {
    assert_eq!(batched.len(), n_q_heads * 4 * head_dim);
    let mut outs: [Vec<f32>; 4] = Default::default();
    for o in &mut outs {
        o.reserve(n_q_heads * head_dim);
    }
    for h in 0..n_q_heads {
        let base = h * 4 * head_dim;
        for (i, o) in outs.iter_mut().enumerate() {
            o.extend_from_slice(&batched[base + i * head_dim..base + (i + 1) * head_dim]);
        }
    }
    outs
}

#[allow(clippy::too_many_arguments)]
fn run_sdpa_decode_batched_q4_f32(
    ctx: &Context,
    kernel: &metaltile_core::ir::Kernel,
    q_batched: &[f32],
    k: &[f32],
    v: &[f32],
    n_q_heads: usize,
    head_dim: usize,
    n_kv: usize,
    kv_stride: usize,
    heads_per_group: usize,
    scale: f32,
) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("q".into(), f32_slice_to_bytes(q_batched));
    buffers.insert("k".into(), f32_slice_to_bytes(k));
    buffers.insert("v".into(), f32_slice_to_bytes(v));
    buffers.insert("out".into(), vec![0u8; n_q_heads * 4 * head_dim * 4]);
    buffers.insert("head_dim".into(), (head_dim as u32).to_le_bytes().to_vec());
    buffers.insert("n_kv".into(), (n_kv as u32).to_le_bytes().to_vec());
    buffers.insert("kv_stride".into(), (kv_stride as u32).to_le_bytes().to_vec());
    buffers.insert("heads_per_group".into(), (heads_per_group as u32).to_le_bytes().to_vec());
    buffers.insert("scale".into(), scale.to_le_bytes().to_vec());

    // K=4's per-lane register pressure (~40 fp32 persistent + transients)
    // caps Metal's `maxTotalThreadsPerThreadgroup` at 768 on M1; dispatching
    // at 1024 is undefined behavior (silently produces all-zero outputs).
    // 512 threads = 16 simdgroups × 32 lanes leaves headroom; the kernel's
    // `n_simd` / `lane` math handles any 32-multiple cleanly. See module
    // docstring in `ffai/sdpa_decode_batched.rs`.
    let result = ctx
        .dispatch_with_grid(kernel, &buffers, &BTreeMap::new(), [n_q_heads, 1, 1], [512, 1, 1])
        .expect("dispatch_with_grid should succeed");
    let out_bytes = result.outputs.get("out").expect("`out` buffer in dispatch result");
    bytes_to_f32_vec(out_bytes)
}

#[test]
fn sdpa_decode_batched_q4_matches_four_independent_sdpa_decodes_f32() {
    let _g = gpu_lock();
    let n_q_heads = 2usize;
    let n_kv_heads = 1usize;
    let head_dim = 128usize;
    let n_kv = 4usize;
    let kv_stride = 4usize;
    let heads_per_group = n_q_heads / n_kv_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    // Four visibly-distinct Q tensors. If any phase's output state
    // leaks into another phase's reduction, the per-Q comparison
    // surfaces it.
    let q_a = ramp(n_q_heads * head_dim, 17, 8.0);
    let q_b = ramp(n_q_heads * head_dim, 11, 5.0);
    let q_c = ramp(n_q_heads * head_dim, 23, 11.0);
    let q_d = ramp(n_q_heads * head_dim, 29, 14.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 19, 9.0);

    let shape = SdpaShape { n_q_heads, n_kv_heads, head_dim, n_kv, scale };
    let expected = [
        naive_sdpa_f32(&q_a, &k, &v, &shape),
        naive_sdpa_f32(&q_b, &k, &v, &shape),
        naive_sdpa_f32(&q_c, &k, &v, &shape),
        naive_sdpa_f32(&q_d, &k, &v, &shape),
    ];

    let q_batched = interleave_q4([&q_a, &q_b, &q_c, &q_d], n_q_heads, head_dim);

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = sdpa_decode_batched_q4::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    let batched_out = run_sdpa_decode_batched_q4_f32(
        &ctx,
        &kernel,
        &q_batched,
        &k,
        &v,
        n_q_heads,
        head_dim,
        n_kv,
        kv_stride,
        heads_per_group,
        scale,
    );
    let actual = split_batched_out_q4(&batched_out, n_q_heads, head_dim);
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_close(a, e, 1e-4, &format!("sdpa_decode_batched_q4 Q[{i}] vs naive_sdpa"));
    }
}

#[test]
fn sdpa_decode_batched_q4_matches_at_larger_n_kv_f32() {
    let _g = gpu_lock();
    let n_q_heads = 8usize;
    let n_kv_heads = 2usize;
    let head_dim = 128usize;
    let n_kv = 1024usize;
    let kv_stride = 1024usize;
    let heads_per_group = n_q_heads / n_kv_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q_a = ramp(n_q_heads * head_dim, 23, 11.0);
    let q_b = ramp(n_q_heads * head_dim, 29, 14.0);
    let q_c = ramp(n_q_heads * head_dim, 31, 15.0);
    let q_d = ramp(n_q_heads * head_dim, 37, 18.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 19, 9.0);

    let shape = SdpaShape { n_q_heads, n_kv_heads, head_dim, n_kv, scale };
    let expected = [
        naive_sdpa_f32(&q_a, &k, &v, &shape),
        naive_sdpa_f32(&q_b, &k, &v, &shape),
        naive_sdpa_f32(&q_c, &k, &v, &shape),
        naive_sdpa_f32(&q_d, &k, &v, &shape),
    ];

    let q_batched = interleave_q4([&q_a, &q_b, &q_c, &q_d], n_q_heads, head_dim);

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = sdpa_decode_batched_q4::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    let batched_out = run_sdpa_decode_batched_q4_f32(
        &ctx,
        &kernel,
        &q_batched,
        &k,
        &v,
        n_q_heads,
        head_dim,
        n_kv,
        kv_stride,
        heads_per_group,
        scale,
    );
    let actual = split_batched_out_q4(&batched_out, n_q_heads, head_dim);
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_close(a, e, 5e-4, &format!("sdpa_decode_batched_q4 Q[{i}] @ n_kv=1024"));
    }
}

#[test]
fn sdpa_decode_batched_q4_identical_qs_produce_identical_outputs_f32() {
    let _g = gpu_lock();
    let n_q_heads = 2usize;
    let n_kv_heads = 1usize;
    let head_dim = 128usize;
    let n_kv = 8usize;
    let kv_stride = 8usize;
    let heads_per_group = n_q_heads / n_kv_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q = ramp(n_q_heads * head_dim, 17, 8.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 19, 9.0);

    let q_batched = interleave_q4([&q, &q, &q, &q], n_q_heads, head_dim);

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = sdpa_decode_batched_q4::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    let batched_out = run_sdpa_decode_batched_q4_f32(
        &ctx,
        &kernel,
        &q_batched,
        &k,
        &v,
        n_q_heads,
        head_dim,
        n_kv,
        kv_stride,
        heads_per_group,
        scale,
    );
    let actual = split_batched_out_q4(&batched_out, n_q_heads, head_dim);

    // All four phases ran the same math on the same inputs. Any
    // divergence here points at non-deterministic ordering inside the
    // four-phase reduction (lingering tg-buffer state, missing barrier,
    // wrong rescale read).
    assert_close(&actual[1], &actual[0], 1e-6, "Q[1] vs Q[0] when Q's are equal");
    assert_close(&actual[2], &actual[0], 1e-6, "Q[2] vs Q[0] when Q's are equal");
    assert_close(&actual[3], &actual[0], 1e-6, "Q[3] vs Q[0] when Q's are equal");
}

/// Regression test for the K=4-at-tpg=1024 hazard documented in
/// `ffai/sdpa_decode_batched.rs:DISPATCH INVARIANTS`. Metal's
/// pipeline-state compiler caps `maxTotalThreadsPerThreadgroup` at 768
/// for this kernel's register pressure on M1 Max. Dispatching at 1024
/// is undefined behavior and the empirically observed symptom is that
/// the output buffer stays at its zero-init state.
///
/// If this test starts FAILING (i.e., 1024-tpg output stops diverging
/// from 512-tpg output by ≥1e-2), the register-pressure cap has
/// changed — update the DISPATCH INVARIANTS doc, the
/// `tpg <= 768 || batch_q < 4` runtime assert, and the inventory submit
/// row's `tpg` accordingly.
#[allow(clippy::too_many_arguments)]
fn dispatch_q4_at_tpg(
    ctx: &Context,
    kernel: &metaltile_core::ir::Kernel,
    tpg: usize,
    q_batched: &[f32],
    k: &[f32],
    v: &[f32],
    n_q_heads: usize,
    head_dim: usize,
    n_kv: usize,
    kv_stride: usize,
    heads_per_group: usize,
    scale: f32,
) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("q".into(), f32_slice_to_bytes(q_batched));
    buffers.insert("k".into(), f32_slice_to_bytes(k));
    buffers.insert("v".into(), f32_slice_to_bytes(v));
    buffers.insert("out".into(), vec![0u8; n_q_heads * 4 * head_dim * 4]);
    buffers.insert("head_dim".into(), (head_dim as u32).to_le_bytes().to_vec());
    buffers.insert("n_kv".into(), (n_kv as u32).to_le_bytes().to_vec());
    buffers.insert("kv_stride".into(), (kv_stride as u32).to_le_bytes().to_vec());
    buffers.insert("heads_per_group".into(), (heads_per_group as u32).to_le_bytes().to_vec());
    buffers.insert("scale".into(), scale.to_le_bytes().to_vec());
    let result = ctx
        .dispatch_with_grid(kernel, &buffers, &BTreeMap::new(), [n_q_heads, 1, 1], [tpg, 1, 1])
        .expect("dispatch_with_grid should succeed");
    let out_bytes = result.outputs.get("out").expect("`out` buffer in dispatch result");
    bytes_to_f32_vec(out_bytes)
}

#[test]
fn sdpa_decode_batched_q4_at_tpg_1024_diverges_from_tpg_512_f32() {
    let _g = gpu_lock();
    let n_q_heads = 2usize;
    let n_kv_heads = 1usize;
    let head_dim = 128usize;
    let n_kv = 8usize;
    let kv_stride = 8usize;
    let heads_per_group = n_q_heads / n_kv_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let q_a = ramp(n_q_heads * head_dim, 17, 8.0);
    let q_b = ramp(n_q_heads * head_dim, 11, 5.0);
    let q_c = ramp(n_q_heads * head_dim, 23, 11.0);
    let q_d = ramp(n_q_heads * head_dim, 29, 14.0);
    let k = ramp(n_kv_heads * kv_stride * head_dim, 13, 6.0);
    let v = ramp(n_kv_heads * kv_stride * head_dim, 19, 9.0);
    let q_batched = interleave_q4([&q_a, &q_b, &q_c, &q_d], n_q_heads, head_dim);

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = sdpa_decode_batched_q4::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Reduction;

    // Dispatch at the safe tpg=512 — produces correct attention output.
    let out_512 = dispatch_q4_at_tpg(
        &ctx,
        &kernel,
        512,
        &q_batched,
        &k,
        &v,
        n_q_heads,
        head_dim,
        n_kv,
        kv_stride,
        heads_per_group,
        scale,
    );
    // Dispatch at the over-cap tpg=1024 — Metal silently writes zeros
    // (or otherwise undefined output) because the kernel's PSO max is
    // 768 on M1 Max.
    let out_1024 = dispatch_q4_at_tpg(
        &ctx,
        &kernel,
        1024,
        &q_batched,
        &k,
        &v,
        n_q_heads,
        head_dim,
        n_kv,
        kv_stride,
        heads_per_group,
        scale,
    );

    // The M1 Max hazard this test pins: K=4 at tpg=1024 silently writes
    // all-zero output because the kernel's PSO `maxTotalThreadsPerThreadgroup`
    // is capped at 768 for K=4's register pressure on M1. The previous
    // version of this test asserted unconditional divergence — which
    // fails on chips with larger register files (M4 / M5 Max) where
    // 1024 dispatches cleanly and matches tpg=512 within fp32 noise.
    //
    // Runtime probe: if tpg=1024 produced near-zero output, we're on
    // an M1-class GPU and the divergence assertion still applies. If
    // tpg=1024 produced legitimate output, we're on a chip where the
    // cap is ≥ 1024 and the dispatch is safe — assert outputs match
    // instead of diverge.
    //
    // Issue #113 — replaces the cross-chip-broken empirical test.
    let max_abs_1024 = out_1024.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let max_abs_512 = out_512.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let max_diff =
        out_512.iter().zip(out_1024.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);

    // "Near-zero" = max abs at tpg=1024 is < 10% of max abs at tpg=512.
    // Well below any plausible fp32 noise + well above the all-zero case
    // (0 vs ~1e-1).
    let tpg_1024_appears_broken = max_abs_1024 < max_abs_512 * 0.1;

    if tpg_1024_appears_broken {
        // M1-class GPU — assert divergence is real, not numerical noise.
        assert!(
            max_diff > 1e-2,
            "K=4 at tpg=1024 appears broken (max_abs={max_abs_1024:.2e}) but \
             diverges from tpg=512 by only {max_diff:.2e}. Expected ≥ 1e-2.",
        );
    } else {
        // M4 / M5 / future chip with `maxTotalThreadsPerThreadgroup` ≥ 1024
        // for this kernel — tpg=1024 dispatches cleanly. Verify the
        // outputs MATCH instead of diverge. fp32 reduction-order noise
        // at n_kv=8 is ~1e-4, so 5e-4 leaves headroom.
        assert!(
            max_diff < 5e-4,
            "K=4 at tpg=1024 produced valid output (max_abs={max_abs_1024:.2e}) \
             but diverges from tpg=512 by {max_diff:.2e}. Expected < 5e-4 \
             (fp32 reduction-order noise).",
        );
    }
}
