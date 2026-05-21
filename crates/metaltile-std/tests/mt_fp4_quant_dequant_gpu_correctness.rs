//! GPU correctness for `mlx::fp_quantized::mt_fp4_quant_dequant`.
//!
//! This is a **non-generic** kernel (`fn mt_fp4_quant_dequant(...)` — no `<T>`).
//! Calling `kernel_ir_for(DType)` on it would be wrong — the generated fn is
//! `kernel_ir_for() -> Kernel` with no arguments. Use `mt_fp4_quant_dequant::kernel_ir_for()`.
//!
//! ## Algorithm
//!
//! Each simdgroup (32 threads) processes 32 consecutive elements:
//!   1. Each thread loads its element `x`.
//!   2. `group_max = simd_max(|x|)` — shared across the 32-lane simdgroup.
//!   3. `inv_scale = 6 / group_max`  (0 when group_max == 0).
//!   4. `norm = |x| * inv_scale`.
//!   5. Map norm to a 3-bit fp4 code (thresholds at 0.25, 0.75, 1.25, 1.75,
//!      2.5, 3.5, 5.0 → levels 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0).
//!   6. Restore sign: `result = sign(x) * q * (group_max / 6.0)`.
//!
//! ## DISPATCH INVARIANTS (from BenchSpec)
//!
//! - **Reduction mode**, `tpg = 32` (one simdgroup per threadgroup).
//! - **Grid**: `[n / tpg, 1, 1]` threadgroups.
//! - `n` must be a multiple of 32.
//! - Constexpr `n: u32` is passed in the buffer map as 4 LE bytes.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::mlx::fp_quantized::mt_fp4_quant_dequant;

/// CPU oracle: exact replica of the kernel's per-simdgroup (group_size=32) logic.
///
/// MLX uses `nvfp4` format with group_size=16 lanes for its reference kernel,
/// but the metaltile port uses `simd_max` across 32 threads (one full simdgroup).
/// This oracle mirrors the metaltile kernel, not the MLX kernel.
fn oracle_fp4_quant_dequant(data: &[f32]) -> Vec<f32> {
    assert_eq!(data.len() % 32, 0, "data length must be a multiple of 32");
    let mut out = vec![0.0f32; data.len()];
    for chunk in data.chunks(32) {
        let base = chunk.as_ptr() as usize - data.as_ptr() as usize;
        let group_start = base / std::mem::size_of::<f32>();

        // Step 2: group_max = max of |x| across 32 elements.
        let group_max = chunk.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

        // Step 3: inv_scale = 6 / group_max (0 when group_max == 0).
        let inv_scale = if group_max > 0.0 { 6.0 / group_max } else { 0.0 };
        let scale = group_max / 6.0; // dequant scale

        for (i, &x) in chunk.iter().enumerate() {
            let ax = x.abs();
            let norm = ax * inv_scale;

            // Step 5: lloyd-max fp4 thresholds — match the kernel's select chain.
            let q = if norm < 0.25 {
                0.0
            } else if norm < 0.75 {
                0.5
            } else if norm < 1.25 {
                1.0
            } else if norm < 1.75 {
                1.5
            } else if norm < 2.5 {
                2.0
            } else if norm < 3.5 {
                3.0
            } else if norm < 5.0 {
                4.0
            } else {
                6.0
            };

            // Step 6: sign * q * scale.
            let sign = if x < 0.0 { -1.0 } else { 1.0 };
            out[group_start + i] = sign * q * scale;
        }
    }
    out
}

/// Dispatch `mt_fp4_quant_dequant` for `n` f32 elements.
///
/// The kernel is non-generic — call `kernel_ir_for()` with no DType argument.
/// Reduction mode: `tpg = 32`, `grid = [n / 32, 1, 1]`.
/// Constexpr `n` is passed as a LE-encoded u32 buffer.
fn run_fp4_quant_dequant(data: &[f32]) -> Vec<f32> {
    let n = data.len();
    assert_eq!(n % 32, 0, "n must be a multiple of 32 (one simdgroup per TG)");

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    // Kernel params: inp (f32), out (f32), constexpr n (u32).
    buffers.insert("inp".into(), pack_bytes(data, Dt::F32));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; n], Dt::F32));
    // Constexpr "n" — u32 LE bytes.
    buffers.insert("n".into(), (n as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    // Non-generic: kernel_ir_for() takes no arguments.
    let mut kernel = mt_fp4_quant_dequant::kernel_ir_for();
    // The bench runner treats this as Reduction mode (simd_max). Override here
    // so dispatch_with_grid emits the correct threadgroup position attribute.
    kernel.mode = KernelMode::Reduction;

    let tpg = 32usize; // one simdgroup
    let groups = n / tpg;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [groups, 1, 1], [tpg, 1, 1])
        .expect("fp4_quant_dequant dispatch");

    unpack_bytes(result.outputs.get("out").expect("out"), Dt::F32)
}

#[test]
fn fp4_quant_dequant_matches_oracle_small_f32() {
    let _g = gpu_lock();
    // 64 elements = 2 simdgroups.
    let data: Vec<f32> = (0..64usize).map(|i| (i % 256) as f32 * 0.01 - 1.28).collect();
    let expected = oracle_fp4_quant_dequant(&data);
    let actual = run_fp4_quant_dequant(&data);

    let diff = max_abs_diff(&actual, &expected);
    // Tolerance 0.5: fp4 quant → 3-bit level, max quantisation error is
    // half the largest level gap (6 - 4 = 2; half = 1.0). In practice
    // our oracle is bit-exact with the kernel so diff << 0.5.
    assert!(
        diff < 0.5,
        "fp4_quant_dequant small: max |diff| = {diff:.4} > 0.5 (oracle mismatch, not quant noise)",
    );
}

#[test]
fn fp4_quant_dequant_oracle_is_exact_f32() {
    let _g = gpu_lock();
    // Oracle should match the kernel exactly (not just within quant tolerance).
    // Use a 128-element batch that exercises all 8 fp4 levels.
    let data: Vec<f32> = vec![
        // Group 1 (32 elems): covers each quantisation bin.
        0.0,   0.1,  0.3,  0.6,  1.0,  1.3,  1.6,  2.0,
        2.3,   3.0,  3.5,  4.5,  5.5,  6.5,  7.0,  7.5,
        -0.1, -0.3, -0.6, -1.0, -1.3, -1.6, -2.0, -2.3,
        -3.0, -3.5, -4.5, -5.5, -6.5, -7.0, -7.5, -8.0,
        // Groups 2–4: repeat with different scale.
        0.05,  0.15, 0.25, 0.45, 0.65, 0.85, 1.05, 1.25,
        1.45,  1.65, 1.85, 2.05, 2.25, 2.45, 2.75, 3.25,
        -0.05,-0.15,-0.25,-0.45,-0.65,-0.85,-1.05,-1.25,
        -1.45,-1.65,-1.85,-2.05,-2.25,-2.45,-2.75,-3.25,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    ];
    let expected = oracle_fp4_quant_dequant(&data);
    let actual = run_fp4_quant_dequant(&data);
    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-5, "fp4_quant_dequant oracle exact: max |diff| = {diff:.2e}");
}

#[test]
fn fp4_quant_dequant_all_zero_group_f32() {
    let _g = gpu_lock();
    // A group of all zeros: group_max = 0 → inv_scale = 0 → all output 0.
    let data = vec![0.0f32; 64];
    let expected = oracle_fp4_quant_dequant(&data);
    let actual = run_fp4_quant_dequant(&data);
    assert!(
        actual.iter().all(|&v| v == 0.0),
        "fp4_quant_dequant all-zero group: output should be all zeros",
    );
    let diff = max_abs_diff(&actual, &expected);
    assert!(diff < 1e-6, "fp4_quant_dequant all-zero: diff = {diff}");
}

#[test]
fn fp4_quant_dequant_output_not_all_zeros_nonzero_input_f32() {
    let _g = gpu_lock();
    // Smoke: non-zero input must produce non-zero output.
    let data: Vec<f32> = (0..32usize).map(|i| (i + 1) as f32).collect();
    let actual = run_fp4_quant_dequant(&data);
    assert!(
        actual.iter().any(|&v| v != 0.0),
        "fp4_quant_dequant: non-zero input produced all-zero output (empty kernel body?)",
    );
}
