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
use metaltile_std::mlx::quantized::{mt_qmm, mt_qmm_bm2, mt_qmm_bm4, mt_qmm_mma, mt_qmm_mma_m16};

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

// ── mt_qmm_bm2 (BM=2 W-reuse variant) ──────────────────────────────────
//
// Same dispatch contract as mt_qmm but each TG owns 2 M-rows → grid
// Y halves. Mirrors the v2 correctness suite to pin the doubled
// inner-loop math + 8-store epilogue against the same CPU oracle.

#[allow(clippy::too_many_arguments)]
fn run_qmm_bm2(
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
    assert!(m.is_multiple_of(2), "mt_qmm_bm2 requires m %% 2 == 0 (BM=2 tile)");
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("w".into(), w.iter().flat_map(|v| v.to_le_bytes()).collect());
    buffers.insert("scales".into(), scales_bytes.to_vec());
    buffers.insert("biases".into(), biases_bytes.to_vec());
    buffers.insert("x".into(), x_bytes.to_vec());
    buffers.insert("out".into(), vec![0u8; m * n * out_bytes_per_elem]);
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("gs_per_row".into(), (gs_per_row as u32).to_le_bytes().to_vec());

    let mut kernel = mt_qmm_bm2::kernel_ir_for(dtype);
    kernel.mode = metaltile_core::ir::KernelMode::Reduction;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n / 8, m / 2, 1], [64, 1, 1])
        .expect("dispatch_with_grid should succeed");
    result.outputs.get("out").expect("`out` buffer in dispatch result").clone()
}

#[test]
fn mt_qmm_bm2_matches_cpu_reference_f32() {
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
    let scales: Vec<f32> = (0..n * gs_per_row).map(|i| 0.1 + (i as f32) * 0.001).collect();
    let biases: Vec<f32> = (0..n * gs_per_row).map(|i| (i as f32) * 0.0001).collect();
    let x: Vec<f32> = (0..m * k).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let expected = cpu_qmm_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let scales_bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
    let biases_bytes: Vec<u8> = biases.iter().flat_map(|v| v.to_le_bytes()).collect();
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let out_bytes = run_qmm_bm2(
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
fn mt_qmm_bm2_matches_cpu_reference_f16() {
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
    let out_bytes = run_qmm_bm2(
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
fn mt_qmm_bm2_matches_mt_qmm_at_same_shape_f32() {
    // Strongest practical pin on the BM=2 hand-unroll: numeric
    // agreement with the v2 BM=1 kernel at the same shape. Strict
    // bit-equivalence isn't reachable — the BM=2 body inlines two
    // qdot accumulators per N-row in alternation, and fp32 add isn't
    // associative so the per-K-block FMAs sequence differently.
    // Observed reorder noise on Apple M-series is ≤ 2 ULP at our
    // output magnitudes (~2-3k → ulp ≈ 2.5e-4). Cap at 1e-3 to match
    // the v2 oracle tolerance; anything larger means an actual bug.
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
    let scales: Vec<f32> = (0..n * gs_per_row).map(|i| 0.1 + (i as f32) * 0.001).collect();
    let biases: Vec<f32> = (0..n * gs_per_row).map(|i| (i as f32) * 0.0001).collect();
    let x: Vec<f32> = (0..m * k).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let scales_bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
    let biases_bytes: Vec<u8> = biases.iter().flat_map(|v| v.to_le_bytes()).collect();
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let out_v2 = run_qmm(
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
    let out_bm2 = run_qmm_bm2(
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
    let a_v2: Vec<f32> =
        out_v2.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let a_bm2: Vec<f32> =
        out_bm2.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let mut max_diff = 0.0_f32;
    let mut max_at = 0usize;
    for (i, (v2, b)) in a_v2.iter().zip(a_bm2.iter()).enumerate() {
        let d = (v2 - b).abs();
        if d > max_diff {
            max_diff = d;
            max_at = i;
        }
    }
    assert!(
        max_diff < 1e-3,
        "v2 vs bm2 diverge: max |diff| = {max_diff:.2e} at index {max_at} (v2 {:.6}, bm2 {:.6})",
        a_v2[max_at],
        a_bm2[max_at],
    );
}

#[test]
fn mt_qmm_bm2_runs_on_qwen3_attention_proj_shape() {
    // Same prod shape as the v2 smoke (n=k=5120) but at m=8 so we
    // exercise 4 BM=2 tiles in Y. Goal: dispatch + index math hold
    // at production-scale shape.
    let m = 8usize;
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
    let out_bytes = run_qmm_bm2(
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
    for (i, v) in actual.iter().enumerate() {
        assert!(v.is_finite(), "non-finite output at index {i}: {v}");
    }
}

// ── mt_qmm_bm4 (BM=4 W-reuse variant) ──────────────────────────────────
//
// Same dispatch contract as mt_qmm/bm2 but each TG owns 4 M-rows → grid Y
// quartered. 32 outputs per TG (vs bm2's 16, v2's 8).

#[allow(clippy::too_many_arguments)]
fn run_qmm_bm4(
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
    assert!(m.is_multiple_of(4), "mt_qmm_bm4 requires m %% 4 == 0 (BM=4 tile)");
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("w".into(), w.iter().flat_map(|v| v.to_le_bytes()).collect());
    buffers.insert("scales".into(), scales_bytes.to_vec());
    buffers.insert("biases".into(), biases_bytes.to_vec());
    buffers.insert("x".into(), x_bytes.to_vec());
    buffers.insert("out".into(), vec![0u8; m * n * out_bytes_per_elem]);
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("gs_per_row".into(), (gs_per_row as u32).to_le_bytes().to_vec());

    let mut kernel = mt_qmm_bm4::kernel_ir_for(dtype);
    kernel.mode = metaltile_core::ir::KernelMode::Reduction;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n / 8, m / 4, 1], [64, 1, 1])
        .expect("dispatch_with_grid should succeed");
    result.outputs.get("out").expect("`out` buffer in dispatch result").clone()
}

#[test]
fn mt_qmm_bm4_matches_cpu_reference_f32() {
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
    let scales: Vec<f32> = (0..n * gs_per_row).map(|i| 0.1 + (i as f32) * 0.001).collect();
    let biases: Vec<f32> = (0..n * gs_per_row).map(|i| (i as f32) * 0.0001).collect();
    let x: Vec<f32> = (0..m * k).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let expected = cpu_qmm_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let scales_bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
    let biases_bytes: Vec<u8> = biases.iter().flat_map(|v| v.to_le_bytes()).collect();
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let out_bytes = run_qmm_bm4(
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
fn mt_qmm_bm4_matches_cpu_reference_f16() {
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
    let out_bytes = run_qmm_bm4(
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
fn mt_qmm_bm4_matches_mt_qmm_at_same_shape_f32() {
    // Same numeric-agreement pin as bm2 — bm4 dual qdot chains in
    // alternation will reorder fp32 adds, but cell tolerance 1e-3
    // catches hand-unroll typos while accepting reorder noise.
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
    let scales: Vec<f32> = (0..n * gs_per_row).map(|i| 0.1 + (i as f32) * 0.001).collect();
    let biases: Vec<f32> = (0..n * gs_per_row).map(|i| (i as f32) * 0.0001).collect();
    let x: Vec<f32> = (0..m * k).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let scales_bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
    let biases_bytes: Vec<u8> = biases.iter().flat_map(|v| v.to_le_bytes()).collect();
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let out_v2 = run_qmm(
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
    let out_bm4 = run_qmm_bm4(
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
    let a_v2: Vec<f32> =
        out_v2.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let a_bm4: Vec<f32> =
        out_bm4.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let mut max_diff = 0.0_f32;
    let mut max_at = 0usize;
    for (i, (v2, b)) in a_v2.iter().zip(a_bm4.iter()).enumerate() {
        let d = (v2 - b).abs();
        if d > max_diff {
            max_diff = d;
            max_at = i;
        }
    }
    assert!(
        max_diff < 1e-3,
        "v2 vs bm4 diverge: max |diff| = {max_diff:.2e} at index {max_at} (v2 {:.6}, bm4 {:.6})",
        a_v2[max_at],
        a_bm4[max_at],
    );
}

#[test]
fn mt_qmm_bm4_runs_on_qwen3_attention_proj_shape() {
    // M=16 at prod shape — 4 TGs in Y axis. Pinned smoke for the
    // headline M=16 cell.
    let m = 16usize;
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
    let out_bytes = run_qmm_bm4(
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
    for (i, v) in actual.iter().enumerate() {
        assert!(v.is_finite(), "non-finite output at index {i}: {v}");
    }
}

// ── mt_qmm_mma (simdgroup-matrix MMA variant) ─────────────────────────
//
// 32×32 output tile, 128 tpg = 4 SG × 32 lanes. Same int4 layout as
// v2/bm2/bm4 — geometry is BM=BN=BK=32, grid `[n/32, m/32, 1]`.

#[allow(clippy::too_many_arguments)]
fn run_qmm_mma(
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
    assert!(m.is_multiple_of(32), "mt_qmm_mma requires m %% 32 == 0 (BM=32 tile)");
    assert!(n.is_multiple_of(32), "mt_qmm_mma requires n %% 32 == 0 (BN=32 tile)");
    assert!(k.is_multiple_of(32), "mt_qmm_mma requires k %% 32 == 0 (BK=32 step)");
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("w".into(), w.iter().flat_map(|v| v.to_le_bytes()).collect());
    buffers.insert("scales".into(), scales_bytes.to_vec());
    buffers.insert("biases".into(), biases_bytes.to_vec());
    buffers.insert("x".into(), x_bytes.to_vec());
    buffers.insert("out".into(), vec![0u8; m * n * out_bytes_per_elem]);
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("gs_per_row".into(), (gs_per_row as u32).to_le_bytes().to_vec());

    let mut kernel = mt_qmm_mma::kernel_ir_for(dtype);
    // Apply Fix 1 (dtype-aware TG skew) — see patch_qmm_mma_dtype_aware_skew.
    metaltile_std::mlx::quantized::patch_qmm_mma_dtype_aware_skew(&mut kernel, dtype);
    kernel.mode = metaltile_core::ir::KernelMode::Reduction;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n / 32, m / 32, 1], [128, 1, 1])
        .expect("dispatch_with_grid should succeed");
    result.outputs.get("out").expect("`out` buffer in dispatch result").clone()
}

#[test]
fn mt_qmm_mma_matches_cpu_reference_f32() {
    // Smallest valid MMA shape: m=32 (1 TG in M), n=32 (1 TG in N),
    // k=64 (2 K-blocks). gs_per_row=1 (group_size=64, k=64).
    let m = 32usize;
    let n = 32usize;
    let k = 64usize;
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

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let out_bytes = run_qmm_mma(
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
fn mt_qmm_mma_matches_cpu_reference_f16() {
    let m = 32usize;
    let n = 32usize;
    let k = 64usize;
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
    let out_bytes = run_qmm_mma(
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

    let mut max_rel = 0.0_f32;
    let mut max_at = 0usize;
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        let rel = (e - a).abs() / e.abs().max(1.0);
        if rel > max_rel {
            max_rel = rel;
            max_at = i;
        }
    }
    // Looser tolerance than f32: MMA frags + dequant-into-TG-fp16
    // round to half precision twice (once on Ws load, once on MMA
    // output cast). 1.5e-2 vs bm4's 5e-3.
    assert!(
        max_rel < 1.5e-2,
        "max relative diff = {max_rel:.2e} at index {max_at} (expected {:.6}, got {:.6})",
        expected[max_at],
        actual[max_at],
    );
}

#[test]
fn mt_qmm_mma_matches_mt_qmm_at_same_shape_f32() {
    // Pinned numeric-agreement check: MMA must match v2 reference within
    // 1e-3 tolerance. v2 requires k %% 512 == 0; pick k=512 (which is
    // also a multiple of BK=32). m=32, n=32 = single TG for MMA.
    let m = 32usize;
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
    let scales_bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
    let biases_bytes: Vec<u8> = biases.iter().flat_map(|v| v.to_le_bytes()).collect();
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let out_v2 = run_qmm(
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
    let out_mma = run_qmm_mma(
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
    let a_v2: Vec<f32> =
        out_v2.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let a_mma: Vec<f32> =
        out_mma.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    // MMA accumulation order (frag-broadcast) differs from v2's scalar
    // simd_sum reduce; allow relative diff < 1e-5 — at k=512 with values
    // ~1e4, absolute diff of ~1e-2 is normal fp32 reorder noise.
    let mut max_rel = 0.0_f32;
    let mut max_at = 0usize;
    for (i, (v2, m_)) in a_v2.iter().zip(a_mma.iter()).enumerate() {
        let rel = (v2 - m_).abs() / v2.abs().max(1.0);
        if rel > max_rel {
            max_rel = rel;
            max_at = i;
        }
    }
    assert!(
        max_rel < 1e-4,
        "v2 vs mma diverge: max rel diff = {max_rel:.2e} at index {max_at} (v2 {:.6}, mma {:.6})",
        a_v2[max_at],
        a_mma[max_at],
    );
}

#[test]
fn mt_qmm_mma_runs_on_qwen3_attention_proj_shape() {
    // M=32 at prod shape — 1 TG in M, n/32 = 160 TGs in N axis. The
    // headline cell where MMA needs to beat bm4 + match MLX.
    let m = 32usize;
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
    let out_bytes = run_qmm_mma(
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
    for (i, v) in actual.iter().enumerate() {
        assert!(v.is_finite(), "non-finite output at index {i}: {v}");
    }
}

// ── mt_qmm_mma_m16 (half-height simdgroup-matrix MMA, M=16 cell) ─────
//
// 16×32 output tile, 64 tpg = 2 SG × 32 lanes (WM=1, WN=2). Same int4
// weight layout — geometry is BM=16, BN=BK=32, grid `[n/32, m/16, 1]`.
// Each SG owns a 16×16 sub-tile = 2 frag_m × 2 frag_n = 4 8×8 frags.

#[allow(clippy::too_many_arguments)]
fn run_qmm_mma_m16(
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
    assert!(m.is_multiple_of(16), "mt_qmm_mma_m16 requires m %% 16 == 0 (BM=16 tile)");
    assert!(n.is_multiple_of(32), "mt_qmm_mma_m16 requires n %% 32 == 0 (BN=32 tile)");
    assert!(k.is_multiple_of(32), "mt_qmm_mma_m16 requires k %% 32 == 0 (BK=32 step)");
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("w".into(), w.iter().flat_map(|v| v.to_le_bytes()).collect());
    buffers.insert("scales".into(), scales_bytes.to_vec());
    buffers.insert("biases".into(), biases_bytes.to_vec());
    buffers.insert("x".into(), x_bytes.to_vec());
    buffers.insert("out".into(), vec![0u8; m * n * out_bytes_per_elem]);
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
    buffers.insert("gs_per_row".into(), (gs_per_row as u32).to_le_bytes().to_vec());

    let mut kernel = mt_qmm_mma_m16::kernel_ir_for(dtype);
    kernel.mode = metaltile_core::ir::KernelMode::Reduction;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n / 32, m / 16, 1], [64, 1, 1])
        .expect("dispatch_with_grid should succeed");
    result.outputs.get("out").expect("`out` buffer in dispatch result").clone()
}

#[test]
fn mt_qmm_mma_m16_matches_cpu_reference_f32() {
    // Smallest valid M=16 shape: m=16 (1 TG in M), n=32 (1 TG in N),
    // k=64 (2 K-blocks). gs_per_row=1 (group_size=64, k=64).
    let m = 16usize;
    let n = 32usize;
    let k = 64usize;
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

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let out_bytes = run_qmm_mma_m16(
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
fn mt_qmm_mma_m16_matches_cpu_reference_f16() {
    let m = 16usize;
    let n = 32usize;
    let k = 64usize;
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
    let out_bytes = run_qmm_mma_m16(
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

    let mut max_rel = 0.0_f32;
    let mut max_at = 0usize;
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        let rel = (e - a).abs() / e.abs().max(1.0);
        if rel > max_rel {
            max_rel = rel;
            max_at = i;
        }
    }
    // Same tolerance as mt_qmm_mma_f16 — frags + dequant-into-TG-fp16
    // round to half precision twice.
    assert!(
        max_rel < 1.5e-2,
        "max relative diff = {max_rel:.2e} at index {max_at} (expected {:.6}, got {:.6})",
        expected[max_at],
        actual[max_at],
    );
}

#[test]
fn mt_qmm_mma_m16_matches_mt_qmm_at_same_shape_f32() {
    // Pinned numeric-agreement check: M=16 MMA must match v2 reference
    // within 1e-4 relative tolerance. v2 requires k %% 512 == 0; pick
    // k=512. m=16, n=32 = single TG for M=16 MMA.
    let m = 16usize;
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
    let scales_bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
    let biases_bytes: Vec<u8> = biases.iter().flat_map(|v| v.to_le_bytes()).collect();
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let out_v2 = run_qmm(
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
    let out_mma = run_qmm_mma_m16(
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
    let a_v2: Vec<f32> =
        out_v2.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let a_mma: Vec<f32> =
        out_mma.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    // MMA accumulation order (frag-broadcast) differs from v2's scalar
    // simd_sum reduce; allow relative diff < 1e-4 (same tol as mt_qmm_mma).
    let mut max_rel = 0.0_f32;
    let mut max_at = 0usize;
    for (i, (v2, m_)) in a_v2.iter().zip(a_mma.iter()).enumerate() {
        let rel = (v2 - m_).abs() / v2.abs().max(1.0);
        if rel > max_rel {
            max_rel = rel;
            max_at = i;
        }
    }
    assert!(
        max_rel < 1e-4,
        "v2 vs mma_m16 diverge: max rel diff = {max_rel:.2e} at index {max_at} (v2 {:.6}, mma {:.6})",
        a_v2[max_at],
        a_mma[max_at],
    );
}

#[test]
fn mt_qmm_mma_m16_runs_on_qwen3_attention_proj_shape() {
    // M=16 at prod shape — 1 TG in M, n/32 = 160 TGs in N axis. The
    // headline cell where M=16 MMA needs to beat bm4 + match MLX on
    // wide-N M2 cases.
    let m = 16usize;
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
    let out_bytes = run_qmm_mma_m16(
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
    for (i, v) in actual.iter().enumerate() {
        assert!(v.is_finite(), "non-finite output at index {i}: {v}");
    }
}

#[test]
#[ignore = "perf bench, run via --ignored --nocapture"]
fn mt_qmm_v2_vs_bm2_head_to_head_f16_m_sweep() {
    // Head-to-head v2 (`mt_qmm`) vs BM=2 W-reuse (`mt_qmm_bm2`) across
    // the Qwen3 hot-path shapes and the M-rows the selector
    // `mt_qmm_for` discriminates on. Goal: prove the selector's
    // `(4..=12).contains(&m)` route actually corresponds to where bm2
    // wins, and that v2 still beats bm2 at M ∈ {1, 2, 16, 32}.
    //
    // Bench mechanics mirror `mt_qmm_perf_bench_qwen3_shapes_f16_m_sweep`:
    //   * resident-buffer pattern for w / scales / biases (uploaded
    //     once per shape, kept GPU-resident across the M-sweep so the
    //     v2 vs bm2 comparison sees identical memory residency)
    //   * WARMUP = 20, ITERS = 50 (one median per kernel per cell)
    //   * O(n) median via `select_nth_unstable_by` at the midpoint
    //
    // BM=2 grid is `[n/8, m/2, 1]` — requires `m % 2 == 0`. M=1 is v2
    // only (bm2 would assert at run_qmm_bm2 anyway).
    //
    // Output is two tables per shape (v2 first, bm2 second) plus a
    // per-shape join with `bm2/v2 speedup` column, then a closing
    // selector-route accuracy summary. Numbers are stable enough for
    // ratio reads at 1 dp; for variance across reruns, drive the
    // outer `cargo test` invocation 5 times back-to-back (see test
    // doc-comment in commit body).
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let group_size = 64usize;
    const WARMUP: usize = 20;
    const ITERS: usize = 50;
    const M_SWEEP: &[usize] = &[1, 2, 4, 6, 8, 12, 16, 32];

    println!();
    println!(
        "mt_qmm v2 vs mt_qmm_bm2 head-to-head — f16, Apple M-series (median of {ITERS} iters)"
    );

    let mut kernel_v2 = mt_qmm::kernel_ir_for(DType::F16);
    kernel_v2.mode = metaltile_core::ir::KernelMode::Reduction;
    let mut kernel_bm2 = mt_qmm_bm2::kernel_ir_for(DType::F16);
    kernel_bm2.mode = metaltile_core::ir::KernelMode::Reduction;
    let empty_fn_consts: BTreeMap<String, u32> = BTreeMap::new();

    // Track per-M aggregate winners across all shapes — used for the
    // closing selector-accuracy table.
    #[derive(Default, Clone, Copy)]
    struct CellAgg {
        bm2_wins: u32,
        v2_wins: u32,
        ratio_sum: f64, // Σ bm2_us / v2_us  (bm2 faster ⇒ <1)
        ratio_n: u32,
    }
    let mut per_m: BTreeMap<usize, CellAgg> = BTreeMap::new();
    for &m in M_SWEEP {
        per_m.insert(m, CellAgg::default());
    }

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

        // Upload static buffers once per shape. Both kernels read the
        // same w/scales/biases — sharing the resident map makes the
        // comparison see identical GPU memory residency.
        let w_res = ctx.upload_resident(&w_bytes_vec).expect("upload w");
        let scales_res = ctx.upload_resident(&scales_bytes).expect("upload scales");
        let biases_res = ctx.upload_resident(&biases_bytes).expect("upload biases");
        let mut residents: BTreeMap<String, ResidentBuffer> = BTreeMap::new();
        residents.insert("w".into(), w_res);
        residents.insert("scales".into(), scales_res);
        residents.insert("biases".into(), biases_res);

        println!();
        println!("  shape = {label}  (n={n}, k={k})");
        println!(
            "    {:>5}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
            "M", "v2 µs", "v2 GB/s", "bm2 µs", "bm2 GB/s", "bm2/v2"
        );

        for &m in M_SWEEP {
            let x: Vec<f32> = (0..m * k).map(|i| 0.1 + ((i % 31) as f32) * 0.01).collect();
            let x_bytes: Vec<u8> =
                x.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect();

            // Per-iter buffer set — same for both kernels (same scalars,
            // same x, same out shape). Cloned per dispatch so each
            // kernel sees a fresh `out`.
            let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
            buffers.insert("x".into(), x_bytes);
            buffers.insert("out".into(), vec![0u8; m * n * 2]);
            buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
            buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());
            buffers.insert("gs_per_row".into(), (gs_per_row as u32).to_le_bytes().to_vec());

            // ── v2 (mt_qmm): grid [n/8, m, 1] ──
            let mut samples_v2 = Vec::with_capacity(ITERS);
            for i in 0..(WARMUP + ITERS) {
                let r = ctx
                    .dispatch_chain(&[DispatchSpec {
                        kernel: &kernel_v2,
                        buffers: &buffers,
                        fn_consts: &empty_fn_consts,
                        grid_groups: [n / 8, m, 1],
                        threads_per_group: [64, 1, 1],
                        resident: &residents,
                    }])
                    .expect("dispatch v2");
                if i >= WARMUP {
                    samples_v2.push(r[0].elapsed_us);
                }
            }
            let mid = samples_v2.len() / 2;
            samples_v2.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
            let v2_us = samples_v2[mid];

            // Bytes touched: same memory traffic model the existing
            // single-kernel bench uses (W + scales + biases + X + Y).
            let bytes = (n * k / 2 + 2 * n * gs_per_row * 2 + m * k * 2 + m * n * 2) as f64;
            let v2_gbps = bytes / (v2_us * 1e-6) / 1e9;

            // ── bm2 (mt_qmm_bm2): grid [n/8, m/2, 1], requires m % 2 == 0 ──
            let bm2_cell: Option<(f64, f64, f64)> = if m % 2 == 0 {
                let mut samples_bm2 = Vec::with_capacity(ITERS);
                for i in 0..(WARMUP + ITERS) {
                    let r = ctx
                        .dispatch_chain(&[DispatchSpec {
                            kernel: &kernel_bm2,
                            buffers: &buffers,
                            fn_consts: &empty_fn_consts,
                            grid_groups: [n / 8, m / 2, 1],
                            threads_per_group: [64, 1, 1],
                            resident: &residents,
                        }])
                        .expect("dispatch bm2");
                    if i >= WARMUP {
                        samples_bm2.push(r[0].elapsed_us);
                    }
                }
                let mid_b = samples_bm2.len() / 2;
                samples_bm2.select_nth_unstable_by(mid_b, |a, b| a.partial_cmp(b).unwrap());
                let bm2_us = samples_bm2[mid_b];
                let bm2_gbps = bytes / (bm2_us * 1e-6) / 1e9;
                let ratio = bm2_us / v2_us; // <1 ⇒ bm2 faster
                Some((bm2_us, bm2_gbps, ratio))
            } else {
                None
            };

            match bm2_cell {
                Some((bm2_us, bm2_gbps, ratio)) => {
                    println!(
                        "    {m:>5}  {v2_us:>10.2}  {v2_gbps:>10.1}  {bm2_us:>10.2}  \
                         {bm2_gbps:>10.1}  {ratio:>10.3}"
                    );
                    let agg = per_m.get_mut(&m).unwrap();
                    if ratio < 1.0 {
                        agg.bm2_wins += 1;
                    } else {
                        agg.v2_wins += 1;
                    }
                    agg.ratio_sum += ratio;
                    agg.ratio_n += 1;
                },
                None => {
                    println!(
                        "    {m:>5}  {v2_us:>10.2}  {v2_gbps:>10.1}  {:>10}  {:>10}  {:>10}",
                        "n/a", "n/a", "skip"
                    );
                },
            }
        }
    }

    // ── selector-route accuracy summary ──
    println!();
    println!("selector-route accuracy (across all {} shapes)", QWEN3_SHAPES.len());
    println!(
        "  {:>5}  {:>10}  {:>10}  {:>14}  {:>14}  {:>10}",
        "M", "bm2_wins", "v2_wins", "mean bm2/v2", "selector→", "matches?"
    );
    for &m in M_SWEEP {
        let agg = per_m[&m];
        let mean_ratio =
            if agg.ratio_n > 0 { agg.ratio_sum / agg.ratio_n as f64 } else { f64::NAN };
        let selector_route = if (4..=12).contains(&(m as u32)) { "bm2" } else { "v2" };
        // Selector route matches data when:
        //   * route = bm2 AND bm2 wins majority of shapes
        //   * route = v2 AND v2 wins majority of shapes (or bm2 cell skipped at M=1)
        let data_winner = if agg.ratio_n == 0 {
            "v2"
        } else if agg.bm2_wins > agg.v2_wins {
            "bm2"
        } else {
            "v2"
        };
        let matches = if selector_route == data_winner { "YES" } else { "NO" };
        let mean_disp =
            if mean_ratio.is_nan() { "n/a".to_string() } else { format!("{mean_ratio:.3}") };
        println!(
            "  {m:>5}  {:>10}  {:>10}  {:>14}  {:>14}  {:>10}",
            agg.bm2_wins, agg.v2_wins, mean_disp, selector_route, matches
        );
    }
    println!();
    println!("legend: bm2/v2 < 1.0 ⇒ bm2 faster than v2 at that cell");
}

// ── Extended coverage — bf16 dtype + production-shape full oracles ──────
//
// PR #56/#57/#60 bar-raising additions. Each new qmm kernel (bm2, bm4,
// mma, mma_m16) gets:
//   - bf16 dtype correctness pin (round-trip oracle)
//   - Production Qwen3 shape (n=k=5120) with full CPU oracle (not just
//     "non-NaN" smoke). Catches numerical bugs at scale that small
//     shapes hide.

fn f32_to_bf16_bits(v: f32) -> u16 { half::bf16::from_f32(v).to_bits() }
fn bf16_bits_to_f32(b: u16) -> f32 { half::bf16::from_bits(b).to_f32() }

#[allow(clippy::type_complexity)]
fn build_bf16_inputs(
    n: usize,
    k: usize,
    gs_per_row: usize,
    m: usize,
) -> (Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<u8>, Vec<u8>, Vec<u8>) {
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
    let scales: Vec<f32> =
        scales_f32.iter().map(|&v| bf16_bits_to_f32(f32_to_bf16_bits(v))).collect();
    let biases: Vec<f32> =
        biases_f32.iter().map(|&v| bf16_bits_to_f32(f32_to_bf16_bits(v))).collect();
    let x: Vec<f32> = x_f32.iter().map(|&v| bf16_bits_to_f32(f32_to_bf16_bits(v))).collect();
    let scales_bytes: Vec<u8> =
        scales_f32.iter().flat_map(|v| f32_to_bf16_bits(*v).to_le_bytes()).collect();
    let biases_bytes: Vec<u8> =
        biases_f32.iter().flat_map(|v| f32_to_bf16_bits(*v).to_le_bytes()).collect();
    let x_bytes: Vec<u8> = x_f32.iter().flat_map(|v| f32_to_bf16_bits(*v).to_le_bytes()).collect();
    (w, scales, biases, x, scales_bytes, biases_bytes, x_bytes)
}

fn check_bf16_outputs(out_bytes: &[u8], expected: &[f32], tol_rel: f32) {
    let actual: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    assert_eq!(actual.len(), expected.len(), "output element count");
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
        max_rel < tol_rel,
        "max rel diff = {max_rel:.2e} at index {max_at} (expected {:.6}, got {:.6})",
        expected[max_at],
        actual[max_at],
    );
}

#[test]
fn mt_qmm_bm2_matches_cpu_reference_bf16() {
    let m = 8usize;
    let n = 16usize;
    let k = 512usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;
    let (w, scales, biases, x, scales_bytes, biases_bytes, x_bytes) =
        build_bf16_inputs(n, k, gs_per_row, m);
    let expected = cpu_qmm_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let _g = gpu_lock();
    let ctx = Context::new().unwrap();
    let out_bytes = run_qmm_bm2(
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
    // bf16 has 7-bit mantissa + same expo range as fp32. Wider rel tolerance.
    check_bf16_outputs(&out_bytes, &expected, 5e-2);
}

#[test]
fn mt_qmm_bm4_matches_cpu_reference_bf16() {
    let m = 8usize;
    let n = 16usize;
    let k = 512usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;
    let (w, scales, biases, x, scales_bytes, biases_bytes, x_bytes) =
        build_bf16_inputs(n, k, gs_per_row, m);
    let expected = cpu_qmm_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);
    let _g = gpu_lock();
    let ctx = Context::new().unwrap();
    let out_bytes = run_qmm_bm4(
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
    check_bf16_outputs(&out_bytes, &expected, 5e-2);
}

#[test]
fn mt_qmm_mma_matches_cpu_reference_bf16() {
    let m = 32usize;
    let n = 32usize;
    let k = 512usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;
    let (w, scales, biases, x, scales_bytes, biases_bytes, x_bytes) =
        build_bf16_inputs(n, k, gs_per_row, m);
    let expected = cpu_qmm_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);
    let _g = gpu_lock();
    let ctx = Context::new().unwrap();
    let out_bytes = run_qmm_mma(
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
    check_bf16_outputs(&out_bytes, &expected, 5e-2);
}

#[test]
fn mt_qmm_mma_m16_matches_cpu_reference_bf16() {
    let m = 16usize;
    let n = 32usize;
    let k = 512usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;
    let (w, scales, biases, x, scales_bytes, biases_bytes, x_bytes) =
        build_bf16_inputs(n, k, gs_per_row, m);
    let expected = cpu_qmm_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);
    let _g = gpu_lock();
    let ctx = Context::new().unwrap();
    let out_bytes = run_qmm_mma_m16(
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
    check_bf16_outputs(&out_bytes, &expected, 5e-2);
}

#[test]
fn mt_qmm_mma_matches_cpu_reference_f32_qwen3_shape_full_oracle() {
    // Production cell with FULL CPU oracle (not smoke). M=32, n=k=5120
    // — Qwen3 attention projection size at the M=32 prefill cell where
    // mma is the routed kernel.
    let m = 32usize;
    let n = 5120usize;
    let k = 5120usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    // Smaller deterministic ranges to keep CPU oracle reasonable (~1.3M ops).
    let w: Vec<u32> =
        (0..n * k / 8).map(|i| ((i as u32) % 17).wrapping_mul(0xDEADBEEFu32)).collect();
    let scales: Vec<f32> = (0..n * gs_per_row).map(|i| 0.005 + (i % 7) as f32 * 0.0007).collect();
    let biases: Vec<f32> = (0..n * gs_per_row).map(|i| (i % 5) as f32 * 0.00005).collect();
    let x: Vec<f32> = (0..m * k).map(|i| 0.05 + ((i % 23) as f32) * 0.003).collect();
    let expected = cpu_qmm_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let scales_bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
    let biases_bytes: Vec<u8> = biases.iter().flat_map(|v| v.to_le_bytes()).collect();
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();

    let _g = gpu_lock();
    let ctx = Context::new().unwrap();
    let out_bytes = run_qmm_mma(
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
    assert_eq!(actual.len(), expected.len());
    let mut max_diff = 0.0f32;
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        let d = (e - a).abs();
        // K=5120 = 10 groups → accumulation can drift ~10 ULPs in fp32.
        // Allow 5e-3 absolute (output magnitudes ~10-1000 at this shape).
        if d > max_diff {
            max_diff = d;
            assert!(
                d < 5e-3,
                "production-shape mma divergence at [{i}]: {a:.6} vs {e:.6} (diff {d:.2e})"
            );
        }
    }
}

#[test]
fn mt_qmm_mma_m16_at_qwen3_shape_full_oracle_f32() {
    // M=16 production cell with FULL CPU oracle. n=k=5120.
    let m = 16usize;
    let n = 5120usize;
    let k = 5120usize;
    let group_size = 64usize;
    let gs_per_row = k / group_size;

    let w: Vec<u32> =
        (0..n * k / 8).map(|i| ((i as u32) % 13).wrapping_mul(0xC0FFEE00u32)).collect();
    let scales: Vec<f32> = (0..n * gs_per_row).map(|i| 0.003 + (i % 11) as f32 * 0.0005).collect();
    let biases: Vec<f32> = (0..n * gs_per_row).map(|i| (i % 9) as f32 * 0.00003).collect();
    let x: Vec<f32> = (0..m * k).map(|i| 0.07 + ((i % 19) as f32) * 0.002).collect();
    let expected = cpu_qmm_reference(&w, &scales, &biases, &x, m, n, k, gs_per_row, group_size);

    let scales_bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
    let biases_bytes: Vec<u8> = biases.iter().flat_map(|v| v.to_le_bytes()).collect();
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();

    let _g = gpu_lock();
    let ctx = Context::new().unwrap();
    let out_bytes = run_qmm_mma_m16(
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
    let mut max_diff = 0.0f32;
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        let d = (e - a).abs();
        if d > max_diff {
            max_diff = d;
            assert!(d < 5e-3, "production-shape mma_m16 divergence at [{i}]: {a:.6} vs {e:.6}");
        }
    }
}
