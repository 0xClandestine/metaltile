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

use std::{
    collections::BTreeMap,
    sync::{Mutex, MutexGuard, OnceLock},
};

use metaltile_core::dtype::DType;
use metaltile_runtime::Context;
use metaltile_std::mlx::quantized::mt_qmm;

/// Serialise GPU dispatches across the tests in this file. cargo runs
/// integration tests in parallel by default; concurrent dispatches on
/// the shared Metal pipeline race the PSO cache + library compilation
/// path and surface as cross-test numeric corruption (caught when the
/// f16 test ran after the f32 test in a single `cargo test` invocation
/// and produced output ≈ 0.45× the expected magnitude). Same pattern
/// other gpu integration suites in this crate use; lighter than
/// requiring `--test-threads=1` at the command line.
fn gpu_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().unwrap_or_else(|e| e.into_inner())
}

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

    let kernel = mt_qmm::kernel_ir_for(dtype);
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [m * n, 1, 1], [64, 1, 1])
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
    let k = 128usize;
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
    let k = 128usize;
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

    // f16 output: ~3 decimal digits of precision at our value magnitudes
    // (outputs land in the 10–50 range with q ∈ [0,15]). Tolerance set
    // to 0.5 to cover f16 rounding + the f16 ULP at this magnitude.
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
        max_diff < 0.5,
        "max |diff| = {max_diff:.2e} at index {max_at} (expected {:.6}, got {:.6})",
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
