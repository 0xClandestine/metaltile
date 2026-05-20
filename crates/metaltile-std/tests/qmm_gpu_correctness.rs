//! End-to-end correctness test for `mt_qmm` — quantized matmul (B>1
//! prefill path). Dispatches the kernel on the Metal pipeline and
//! compares against a straight-translation CPU reference that mirrors
//! the same int4 dequant + algebraic-split math the kernel uses.
//!
//! The reference is intentionally a faithful re-statement of the
//! kernel body (not a separate independent algorithm): both walk K in
//! groups of `group_size = 64`, dequant each int4 nibble via
//! `(packed >> (i*4)) & 0xF`, and accumulate
//! `acc += s_g · Σ q·x + bias_g · Σ x`. That makes correctness here
//! mean "MSL emit + dispatch wiring + index math match the IR" — not
//! "matches a separate dense matmul oracle." The dense-oracle check
//! belongs at the bench-runner layer once mt_qmm graduates from `mlx/`
//! to a `BenchDispatch::QuantizedMatMul` variant with an MLX `qmm_t`
//! comparison kernel.
//!
//! Shape is intentionally small (m=8, n=16, k=128 = 2 groups) so the
//! CPU reference runs instantly + the comparison is easy to eyeball.

#![cfg(target_os = "macos")]

use std::collections::BTreeMap;

mod common;

use common::gpu_lock;
use metaltile_core::dtype::DType;
use metaltile_runtime::{Context, DispatchSpec, ResidentBuffer};
use metaltile_std::mlx::quantized::mt_qmm;

#[allow(clippy::too_many_arguments)]
fn run_qmm(
    ctx: &Context,
    dtype: DType,
    w: &[u32],
    scales_bytes: &[u8],
    biases_bytes: &[u8],
    x_bytes: &[u8],
    m: usize,
    n: usize,
    k: usize,
    gs_per_row: usize,
    out_bytes_per_elem: usize,
) -> Vec<u8> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("w".into(), w.iter().flat_map(|v| v.to_le_bytes()).collect());
    buffers.insert("scales".into(), scales_bytes.to_vec());
    buffers.insert("biases".into(), biases_bytes.to_vec());
    buffers.insert("x".into(), x_bytes.to_vec());
    buffers.insert("out".into(), vec![0u8; m * n * out_bytes_per_elem]);
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("gs_per_row".into(), (gs_per_row as u32).to_le_bytes().to_vec());

    let mut kernel = mt_qmm::kernel_ir_for(dtype);
    // Reduction mode is required so the codegen emits the
    // `tgid_x`/`tgid_y` aliases the kernel body references. Same
    // dispatch contract as `mt_qmv`.
    kernel.mode = metaltile_core::ir::KernelMode::Reduction;
    // Grid: (n/8 N-tiles, m M-rows, 1). 2 simdgroups × 32 lanes = 64
    // threads per group. Each TG produces 8 outputs at one (m_row).
    // Caller assertion: n % 8 == 0, k % 512 == 0 (mt_qmm preconditions
    // inherited from mt_qmv).
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n / 8, m, 1], [64, 1, 1])
        .expect("dispatch_with_grid should succeed");
    result.outputs.get("out").expect("`out` buffer in dispatch result").clone()
}

#[allow(clippy::too_many_arguments)]
fn cpu_qmm_reference(
    w: &[u32],
    scales: &[f32],
    biases: &[f32],
    x: &[f32],
    m: usize,
    n: usize,
    k: usize,
    gs_per_row: usize,
    group_size: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for m_row in 0..m {
        for n_col in 0..n {
            let mut acc = 0.0f32;
            for g in 0..gs_per_row {
                let s = scales[n_col * gs_per_row + g];
                let bias = biases[n_col * gs_per_row + g];
                let mut q_dot = 0.0f32;
                let mut x_sum = 0.0f32;
                for p in 0..8usize {
                    let packed = w[n_col * k / 8 + g * 8 + p];
                    for bit in 0..8u32 {
                        let q = ((packed >> (bit * 4)) & 0xF) as f32;
                        let xv = x[m_row * k + g * group_size + p * 8 + bit as usize];
                        q_dot += q * xv;
                        x_sum += xv;
                    }
                }
                acc += s * q_dot + bias * x_sum;
            }
            out[m_row * n + n_col] = acc;
        }
    }
    out
}

#[test]
fn mt_qmm_matches_cpu_reference_f32() {
    let m = 8usize;
    let n = 16usize;
    let k = 512usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    // Deterministic q4 weights. Per-pack pattern lifted from the qmv
    // correctness oracle in run_spec.rs so both paths exercise the
    // same packed bit layout.
    let w: Vec<u32> = (0..n * k / 8)
        .map(|i| {
            let mut v = 0u32;
            for bit in 0..8u32 {
                v |= ((i as u32 + bit) & 0xF) << (bit * 4);
            }
            v
        })
        .collect();
    let scales: Vec<f32> = (0..n * gs_per_row).map(|i| 0.1 + (i as f32) * 0.001).collect();
    let biases: Vec<f32> = (0..n * gs_per_row).map(|i| (i as f32) * 0.0001).collect();
    let x: Vec<f32> = (0..m * k).map(|i| 1.0 + (i as f32) * 0.001).collect();

    let mut expected = vec![0.0f32; m * n];
    for m_row in 0..m {
        for n_col in 0..n {
            let mut acc = 0.0f32;
            for g in 0..gs_per_row {
                let s = scales[n_col * gs_per_row + g];
                let bias = biases[n_col * gs_per_row + g];
                let mut q_dot = 0.0f32;
                let mut x_sum = 0.0f32;
                for p in 0..8usize {
                    let packed = w[n_col * k / 8 + g * 8 + p];
                    for bit in 0..8u32 {
                        let q = ((packed >> (bit * 4)) & 0xF) as f32;
                        let xv = x[m_row * k + g * group_size + p * 8 + bit as usize];
                        q_dot += q * xv;
                        x_sum += xv;
                    }
                }
                acc += s * q_dot + bias * x_sum;
            }
            expected[m_row * n + n_col] = acc;
        }
    }

    let scales_bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
    let biases_bytes: Vec<u8> = biases.iter().flat_map(|v| v.to_le_bytes()).collect();
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let out_bytes = run_qmm(
        &ctx,
        DType::F32,
        &w,
        &scales_bytes,
        &biases_bytes,
        &x_bytes,
        m,
        n,
        k,
        gs_per_row,
        4,
    );
    let actual: Vec<f32> =
        out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

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
        "max |diff| = {max_diff:.2e} at index {max_at} (expected {:.6}, got {:.6})",
        expected[max_at],
        actual[max_at],
    );
}

#[test]
fn mt_qmm_matches_cpu_reference_f16() {
    // f16 path: inputs round-tripped through half-precision so the
    // oracle and the kernel agree to within f16's 3-digit precision.
    let m = 8usize;
    let n = 16usize;
    let k = 512usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let w: Vec<u32> = (0..n * k / 8)
        .map(|i| {
            let mut v = 0u32;
            for bit in 0..8u32 {
                v |= ((i as u32 + bit) & 0xF) << (bit * 4);
            }
            v
        })
        .collect();
    let scales_f32: Vec<f32> = (0..n * gs_per_row).map(|i| 0.1 + (i as f32) * 0.001).collect();
    let biases_f32: Vec<f32> = (0..n * gs_per_row).map(|i| (i as f32) * 0.0001).collect();
    let x_f32: Vec<f32> = (0..m * k).map(|i| 1.0 + (i as f32) * 0.001).collect();

    // Round inputs through f16 so the oracle reflects what the kernel
    // sees after the f16 → f32 cast on load.
    let round_f16 = |v: f32| -> f32 { half::f16::from_f32(v).to_f32() };
    let scales: Vec<f32> = scales_f32.iter().map(|&v| round_f16(v)).collect();
    let biases: Vec<f32> = biases_f32.iter().map(|&v| round_f16(v)).collect();
    let x: Vec<f32> = x_f32.iter().map(|&v| round_f16(v)).collect();
    let expected = cpu_qmm_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let scales_bytes: Vec<u8> =
        scales.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect();
    let biases_bytes: Vec<u8> =
        biases.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect();
    let x_bytes: Vec<u8> =
        x.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect();

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let out_bytes = run_qmm(
        &ctx,
        DType::F16,
        &w,
        &scales_bytes,
        &biases_bytes,
        &x_bytes,
        m,
        n,
        k,
        gs_per_row,
        2,
    );
    let actual: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();

    assert_eq!(actual.len(), expected.len(), "output element count");

    // f16 has 10 bits of mantissa → ULP ≈ |v| * 2^-10. At our K=512
    // output magnitudes (~3-4k for accumulated dequant·dot products),
    // absolute ULP is ~4. Use relative tolerance: 0.5% of expected
    // magnitude. f16 round-to-nearest rounding at the output store +
    // simd_sum reordering of partial f32 → f16 narrowings stays well
    // inside this envelope.
    let mut max_rel = 0.0_f32;
    let mut max_at = 0usize;
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        let rel = (e - a).abs() / e.abs().max(1.0);
        if rel > max_rel {
            max_rel = rel;
            max_at = i;
        }
    }
    assert!(
        max_rel < 5e-3,
        "max relative diff = {max_rel:.2e} at index {max_at} (expected {:.6}, got {:.6})",
        expected[max_at],
        actual[max_at],
    );
}

#[test]
fn mt_qmm_runs_on_qwen3_attention_proj_shape() {
    // Qwen3-8B/14B attention projection (Q/K/V/O): n=5120, k=5120.
    // Use m=4 tokens to keep the test fast while still B>1. This isn't
    // a numeric check — random weights make the oracle expensive. It's
    // a "kernel dispatches at production shape without faulting" smoke
    // check on the actual hot-path size.
    let m = 4usize;
    let n = 5120usize;
    let k = 5120usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let w: Vec<u32> = (0..n * k / 8).map(|i| (i as u32).wrapping_mul(2654435761u32)).collect();
    let scales: Vec<f32> = (0..n * gs_per_row).map(|i| 0.01 + (i % 13) as f32 * 0.001).collect();
    let biases: Vec<f32> = (0..n * gs_per_row).map(|i| (i % 7) as f32 * 0.0001).collect();
    let x: Vec<f32> = (0..m * k).map(|i| 0.1 + ((i % 31) as f32) * 0.01).collect();

    let scales_bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
    let biases_bytes: Vec<u8> = biases.iter().flat_map(|v| v.to_le_bytes()).collect();
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let out_bytes = run_qmm(
        &ctx,
        DType::F32,
        &w,
        &scales_bytes,
        &biases_bytes,
        &x_bytes,
        m,
        n,
        k,
        gs_per_row,
        4,
    );

    // Sanity: all outputs finite. NaN/inf would indicate a real
    // dispatch fault (e.g., out-of-bounds load) we want to catch.
    let actual: Vec<f32> =
        out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(actual.len(), m * n);
    for (i, &v) in actual.iter().enumerate() {
        assert!(v.is_finite(), "non-finite output at index {i}: {v}");
    }
}

/// All five Qwen3 hot-path shapes from `mlx/quantized.rs:QUANTIZED_SHAPES`.
/// Returned as `(n, k, label)` for the Qwen3-8B/14B + Qwen3-coder-30B
/// MoE expert sizes.
const QWEN3_SHAPES: &[(usize, usize, &str)] = &[
    (4096, 4096, "baseline 4096²"),
    (5120, 5120, "Qwen3-8B/14B attn proj"),
    (14336, 5120, "Qwen3-8B/14B MLP up_proj"),
    (5120, 14336, "Qwen3-8B/14B MLP down_proj"),
    (27648, 5120, "Qwen3-coder-30B MoE expert up_proj"),
];

#[test]
fn mt_qmm_m1_byte_identical_to_qmv_dispatch_path() {
    // mt_qmm at M=1 should produce the same outputs as a 1-row qmm
    // dispatch — the kernel body is qmv's body parameterised by an
    // outer M-row grid axis, and `tgid_y=0` makes the M offset
    // collapse to zero. This is the smallest non-trivial witness
    // that the M-axis lift didn't disturb the qmv inner loop.
    let _g = gpu_lock();

    let m = 1usize;
    let n = 32usize;
    let k = 512usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let w: Vec<u32> = (0..n * k / 8)
        .map(|i| {
            let mut v = 0u32;
            for bit in 0..8u32 {
                v |= ((i as u32 + bit) & 0xF) << (bit * 4);
            }
            v
        })
        .collect();
    let scales: Vec<f32> = (0..n * gs_per_row).map(|i| 0.1 + (i as f32) * 0.001).collect();
    let biases: Vec<f32> = (0..n * gs_per_row).map(|i| (i as f32) * 0.0001).collect();
    let x: Vec<f32> = (0..m * k).map(|i| 1.0 + (i as f32) * 0.001).collect();

    let expected = cpu_qmm_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let scales_bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
    let biases_bytes: Vec<u8> = biases.iter().flat_map(|v| v.to_le_bytes()).collect();
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let out_bytes = run_qmm(
        &ctx,
        DType::F32,
        &w,
        &scales_bytes,
        &biases_bytes,
        &x_bytes,
        m,
        n,
        k,
        gs_per_row,
        4,
    );
    let actual: Vec<f32> =
        out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let max_diff =
        expected.iter().zip(actual.iter()).map(|(e, a)| (e - a).abs()).fold(0.0_f32, f32::max);
    assert!(max_diff < 1e-3, "max |diff| = {max_diff:.2e}");
}

#[test]
fn mt_qmm_matches_cpu_reference_bf16_small_shape() {
    // bf16 is generic `<T>` over the kernel — same code path as f16
    // modulo the load cast and the output store narrowing. mt_qmv
    // doesn't bench-wire bf16 because MLX's `affine_qmv_fast` only
    // ships f32/f16 host bindings at our pinned commit; mt_qmm bench
    // wiring will hit the same constraint. Coverage anyway: bf16's
    // 7-bit mantissa drifts faster than f16's 10-bit, so the cast
    // path needs an explicit oracle to catch any subtle silent
    // truncation in the accumulator chain.
    let _g = gpu_lock();

    let m = 4usize;
    let n = 16usize;
    let k = 512usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let w: Vec<u32> = (0..n * k / 8)
        .map(|i| {
            let mut v = 0u32;
            for bit in 0..8u32 {
                v |= ((i as u32 + bit) & 0xF) << (bit * 4);
            }
            v
        })
        .collect();
    let scales_f32: Vec<f32> = (0..n * gs_per_row).map(|i| 0.1 + (i as f32) * 0.001).collect();
    let biases_f32: Vec<f32> = (0..n * gs_per_row).map(|i| (i as f32) * 0.0001).collect();
    let x_f32: Vec<f32> = (0..m * k).map(|i| 1.0 + (i as f32) * 0.001).collect();

    let round_bf16 = |v: f32| -> f32 { half::bf16::from_f32(v).to_f32() };
    let scales: Vec<f32> = scales_f32.iter().map(|&v| round_bf16(v)).collect();
    let biases: Vec<f32> = biases_f32.iter().map(|&v| round_bf16(v)).collect();
    let x: Vec<f32> = x_f32.iter().map(|&v| round_bf16(v)).collect();
    let expected = cpu_qmm_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let scales_bytes: Vec<u8> =
        scales.iter().flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes()).collect();
    let biases_bytes: Vec<u8> =
        biases.iter().flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes()).collect();
    let x_bytes: Vec<u8> =
        x.iter().flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes()).collect();

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let out_bytes = run_qmm(
        &ctx,
        DType::BF16,
        &w,
        &scales_bytes,
        &biases_bytes,
        &x_bytes,
        m,
        n,
        k,
        gs_per_row,
        2,
    );
    let actual: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();

    // bf16 has 7-bit mantissa → ULP ≈ |v| * 2^-7 ≈ |v| / 128.
    // 2% relative envelope covers store rounding + simd_sum reordering.
    let mut max_rel = 0.0_f32;
    let mut max_at = 0usize;
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        let rel = (e - a).abs() / e.abs().max(1.0);
        if rel > max_rel {
            max_rel = rel;
            max_at = i;
        }
    }
    assert!(
        max_rel < 2e-2,
        "max relative diff = {max_rel:.2e} at index {max_at} (expected {:.6}, got {:.6})",
        expected[max_at],
        actual[max_at],
    );
}

#[test]
fn mt_qmm_dispatches_all_qwen3_shapes_at_b4_f16() {
    // Finite-output smoke check across every Qwen3 hot-path shape at
    // M=4 (typical batched prefill). Random-ish weights make the
    // CPU oracle too expensive to compute — instead verify the
    // dispatch succeeds + outputs are finite, which catches
    // address-arithmetic bugs that only show up at production sizes.
    let _g = gpu_lock();

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let m = 4usize;
    let group_size = 64usize;

    for &(n, k, label) in QWEN3_SHAPES {
        let gs_per_row = k / group_size;
        let w: Vec<u32> = (0..n * k / 8).map(|i| (i as u32).wrapping_mul(2654435761u32)).collect();
        let scales: Vec<f32> =
            (0..n * gs_per_row).map(|i| 0.01 + (i % 13) as f32 * 0.001).collect();
        let biases: Vec<f32> = (0..n * gs_per_row).map(|i| (i % 7) as f32 * 0.0001).collect();
        let x: Vec<f32> = (0..m * k).map(|i| 0.1 + ((i % 31) as f32) * 0.01).collect();

        let scales_bytes: Vec<u8> =
            scales.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect();
        let biases_bytes: Vec<u8> =
            biases.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect();
        let x_bytes: Vec<u8> =
            x.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect();

        let out_bytes = run_qmm(
            &ctx,
            DType::F16,
            &w,
            &scales_bytes,
            &biases_bytes,
            &x_bytes,
            m,
            n,
            k,
            gs_per_row,
            2,
        );
        let actual: Vec<f32> = out_bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect();
        assert_eq!(actual.len(), m * n, "{label}: output length");
        for (i, &v) in actual.iter().enumerate() {
            assert!(v.is_finite(), "{label}: non-finite output at {i}: {v}");
        }
    }
}

#[test]
#[ignore = "perf bench, run via --ignored --nocapture"]
fn mt_qmm_perf_bench_qwen3_shapes_f16_m_sweep() {
    // Per-shape × per-M-row throughput probe. Reports median GPU µs +
    // effective GB/s (q4 weight bytes + per-group scale/bias + X + Y
    // streams). M-sweep: 1, 4, 8, 32 tokens — covers the spectrum
    // from single-prompt prefill chunk to batched serving. Intent:
    // expose where the per-M-row TG dispatch starts losing to W
    // bandwidth saturation (the v3 BM-tile W-reuse rationale).
    //
    // Canonical perf measurement target is M2 mini (see
    // `feedback_metaltile_bench_on_m2_mini`); this bench surfaces the
    // same numbers on the dev host for iteration.
    //
    // ## Resident-buffer pattern
    //
    // `w`, `scales`, `biases` are static across all iterations within
    // a shape (only `x` / `out` vary per-M, and only the kernel runs
    // per-iter). Upload them once per shape via
    // `Context::upload_resident` and bind through
    // `DispatchSpec::resident` — the dispatch skips the host→GPU
    // memcpy + per-iter buffer-pool allocation that would otherwise
    // re-stream ~70 MB of q4 weights at the largest Qwen3 shape
    // (27648 × 5120 / 2) every dispatch. Same pattern used by
    // `tests/sdpa_decode_2pass_gpu.rs:171-172` for K/V residency.
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let group_size = 64usize;
    const WARMUP: usize = 20;
    const ITERS: usize = 50;

    println!();
    println!(
        "mt_qmm f16 — Apple M-series (median of {ITERS} iters)\n  {:>30}  {:>5}  {:>10}  {:>10}",
        "shape (n × k)", "M", "µs", "GB/s"
    );

    let mut kernel = mt_qmm::kernel_ir_for(DType::F16);
    kernel.mode = metaltile_core::ir::KernelMode::Reduction;
    let empty_fn_consts: BTreeMap<String, u32> = BTreeMap::new();
    let no_output_resident: BTreeMap<String, ResidentBuffer> = BTreeMap::new();

    for &(n, k, label) in QWEN3_SHAPES {
        let gs_per_row = k / group_size;
        let w: Vec<u32> = (0..n * k / 8).map(|i| (i as u32).wrapping_mul(2654435761u32)).collect();
        let scales: Vec<f32> =
            (0..n * gs_per_row).map(|i| 0.01 + (i % 13) as f32 * 0.001).collect();
        let biases: Vec<f32> = (0..n * gs_per_row).map(|i| (i % 7) as f32 * 0.0001).collect();

        let w_bytes_vec: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let scales_bytes: Vec<u8> =
            scales.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect();
        let biases_bytes: Vec<u8> =
            biases.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect();

        // Upload the per-shape static buffers ONCE. ResidentBuffer
        // holds an Rc to a Metal buffer in private storage mode; the
        // dispatch skips the per-call host→GPU memcpy + buffer-pool
        // alloc for these inputs.
        let w_res = ctx.upload_resident(&w_bytes_vec).expect("upload w");
        let scales_res = ctx.upload_resident(&scales_bytes).expect("upload scales");
        let biases_res = ctx.upload_resident(&biases_bytes).expect("upload biases");
        let mut residents: BTreeMap<String, ResidentBuffer> = BTreeMap::new();
        residents.insert("w".into(), w_res);
        residents.insert("scales".into(), scales_res);
        residents.insert("biases".into(), biases_res);

        for &m in &[1usize, 4, 8, 32] {
            let x: Vec<f32> = (0..m * k).map(|i| 0.1 + ((i % 31) as f32) * 0.01).collect();
            let x_bytes: Vec<u8> =
                x.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect();

            // Per-iter buffers: only the live-data ones (x + out) and
            // the constexpr scalars. Static weights / scales / biases
            // stay GPU-resident via the `residents` map above.
            let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
            buffers.insert("x".into(), x_bytes);
            buffers.insert("out".into(), vec![0u8; m * n * 2]);
            buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
            buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
            buffers.insert("gs_per_row".into(), (gs_per_row as u32).to_le_bytes().to_vec());

            let mut samples = Vec::with_capacity(ITERS);
            for i in 0..(WARMUP + ITERS) {
                let r = ctx
                    .dispatch_chain(&[DispatchSpec {
                        kernel: &kernel,
                        buffers: &buffers,
                        fn_consts: &empty_fn_consts,
                        grid_groups: [n / 8, m, 1],
                        threads_per_group: [64, 1, 1],
                        resident: &residents,
                        output_resident: &no_output_resident,
                    }])
                    .expect("dispatch");
                if i >= WARMUP {
                    samples.push(r[0].elapsed_us);
                }
            }
            // O(n) median via `select_nth_unstable_by` — same pattern
            // `tests/sdpa_decode_2pass_gpu.rs` uses; we don't need a
            // fully sorted samples vector, only the midpoint.
            let mid = samples.len() / 2;
            samples.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
            let median_us = samples[mid];

            // Bytes touched per kernel: W (q4) + scales + biases + X + Y.
            // q4 weights: n × k / 2 bytes. Scales/biases: 2 bytes each
            // per group, total n × gs_per_row × 2 × 2 bytes. X + Y in T.
            let bytes = (n * k / 2 + 2 * n * gs_per_row * 2 + m * k * 2 + m * n * 2) as f64;
            let gbps = bytes / (median_us * 1e-6) / 1e9;
            println!("  {label:>30}  {m:>5}  {median_us:>10.2}  {gbps:>10.1}");
        }
    }
}
