//! GPU correctness for the multi-bit-width quantized matvec / vecmat /
//! matmul family in `mlx::quantized`:
//!
//!   - `mt_qmv_b{3,4,5,6,8}`  — `y = W · x`,  W [N, K]
//!   - `mt_qvm_b{3,4,5,6,8}`  — `y = xᵀ · W`, W [K, N]
//!   - `mt_qmm_b{3,4,5,6,8}`  — batched `y = W · x` over M rows
//!
//! All kernels are Reduction-mode, one simdgroup (TPG = 32) per output
//! element. The CPU oracle dequantizes in f32 and runs the naive
//! triple loop — the contract.
//!
//! Bit-packing:
//!   - pow2 widths (4, 8): pack-aligned, `32/bits` codes per u32.
//!   - odd widths (3, 5, 6): a contiguous bit-stream; a code may
//!     straddle a u32 boundary.
//!
//! macOS-gated. Shared gpu_lock.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, max_abs_diff, pack_bytes, pack_u32_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::mlx::quantized::{
    mt_qmm_b3,
    mt_qmm_b4,
    mt_qmm_b5,
    mt_qmm_b6,
    mt_qmm_b8,
    mt_qmv_b3,
    mt_qmv_b4,
    mt_qmv_b5,
    mt_qmv_b6,
    mt_qmv_b8,
    mt_qvm_b3,
    mt_qvm_b4,
    mt_qvm_b5,
    mt_qvm_b6,
    mt_qvm_b8,
};

/// Pack a row of unsigned codes (`< 2^bits`) into a u32 bit-stream.
/// For pow2 widths this coincides with the pack-aligned layout; for odd
/// widths a code may straddle two u32 words. Matches the kernel's
/// two-word extract.
fn pack_codes(codes: &[u32], bits: u32) -> Vec<u32> {
    let total_bits = codes.len() as u32 * bits;
    let n_words = total_bits.div_ceil(32) as usize;
    let mut words = vec![0u32; n_words];
    for (i, &c) in codes.iter().enumerate() {
        let bit_off = i as u32 * bits;
        let word = (bit_off / 32) as usize;
        let shift = bit_off % 32;
        words[word] |= (c & ((1u32 << bits) - 1)) << shift;
        let lo = 32 - shift;
        if lo < bits {
            // Spill into the next word.
            words[word + 1] |= c >> lo;
        }
    }
    words
}

/// Affine-quantize a group of f32 values to `bits`-bit codes.
/// Returns (codes, scale, bias) with `value ≈ code * scale + bias`.
fn quantize_group(group: &[f32], bits: u32) -> (Vec<u32>, f32, f32) {
    let max_code = ((1u32 << bits) - 1) as f32;
    let mn = group.iter().copied().fold(f32::INFINITY, f32::min);
    let mx = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let scale = if (mx - mn).abs() < 1e-10 { 1.0 } else { (mx - mn) / max_code };
    let bias = mn;
    let codes =
        group.iter().map(|&v| ((v - bias) / scale).round().clamp(0.0, max_code) as u32).collect();
    (codes, scale, bias)
}

/// Quantize a `[rows, cols]` row-major f32 matrix, grouping along the
/// `cols` (contraction) axis. Returns (packed words flattened over
/// rows, scales [rows, cols/group], biases [rows, cols/group]).
fn quantize_matrix(
    mat: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
    bits: u32,
) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let n_groups = cols / group_size;
    let words_per_row = (cols as u32 * bits).div_ceil(32) as usize;
    let mut packed = vec![0u32; rows * words_per_row];
    let mut scales = vec![0.0f32; rows * n_groups];
    let mut biases = vec![0.0f32; rows * n_groups];
    for r in 0..rows {
        let mut codes = vec![0u32; cols];
        for g in 0..n_groups {
            let off = g * group_size;
            let slice = &mat[r * cols + off..r * cols + off + group_size];
            let (c, s, b) = quantize_group(slice, bits);
            scales[r * n_groups + g] = s;
            biases[r * n_groups + g] = b;
            codes[off..off + group_size].copy_from_slice(&c);
        }
        let w = pack_codes(&codes, bits);
        packed[r * words_per_row..r * words_per_row + w.len()].copy_from_slice(&w);
    }
    (packed, scales, biases)
}

/// Dequantize the matrix back to f32 — the oracle's view of the weights.
fn dequant_matrix(
    packed: &[u32],
    scales: &[f32],
    biases: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
    bits: u32,
) -> Vec<f32> {
    let n_groups = cols / group_size;
    let words_per_row = (cols as u32 * bits).div_ceil(32) as usize;
    let mask = (1u64 << bits) - 1;
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let bit_off = c as u32 * bits;
            let word = (bit_off / 32) as usize;
            let shift = bit_off % 32;
            let base = r * words_per_row;
            let lo = (packed[base + word] as u64) >> shift;
            let hi = if 32 - shift < bits {
                (packed[base + word + 1] as u64) << (32 - shift)
            } else {
                0
            };
            let code = ((lo | hi) & mask) as f32;
            let g = c / group_size;
            out[r * cols + c] = code * scales[r * n_groups + g] + biases[r * n_groups + g];
        }
    }
    out
}

/// Dispatch one of the family kernels. `grid_n` / `grid_m` are the grid
/// extents; `out_len` the output element count.
#[allow(clippy::too_many_arguments)]
fn dispatch_family(
    kernel: metaltile_core::ir::Kernel,
    packed: &[u32],
    scales: &[f32],
    biases: &[f32],
    x: &[f32],
    dt: Dt,
    k: u32,
    n: u32,
    group_size: u32,
    grid_n: usize,
    grid_m: usize,
    out_len: usize,
) -> Vec<f32> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("w".into(), pack_u32_bytes(packed));
    buffers.insert("scales".into(), pack_bytes(scales, dt));
    buffers.insert("biases".into(), pack_bytes(biases, dt));
    buffers.insert("x".into(), pack_bytes(x, dt));
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; out_len], dt));
    buffers.insert("k".into(), k.to_le_bytes().to_vec());
    buffers.insert("n".into(), n.to_le_bytes().to_vec());
    buffers.insert("group_size".into(), group_size.to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = kernel;
    kernel.mode = KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [grid_n, grid_m, 1], [32, 1, 1])
        .expect("family dispatch");

    let mut out = unpack_bytes(result.outputs.get("out").expect("out"), dt);
    out.truncate(out_len);
    out
}

/// CPU oracle — naive matvec `y = W · x` with W dequantized in f32.
fn oracle_matvec(w_deq: &[f32], x: &[f32], n: usize, k: usize) -> Vec<f32> {
    (0..n).map(|row| (0..k).map(|d| w_deq[row * k + d] * x[d]).sum()).collect()
}

/// CPU oracle — naive vecmat `y = xᵀ · W` with W [K, N] dequantized.
fn oracle_vecmat(w_deq: &[f32], x: &[f32], n: usize, k: usize) -> Vec<f32> {
    (0..n).map(|col| (0..k).map(|d| x[d] * w_deq[d * n + col]).sum()).collect()
}

// ─── qmv — matvec, W [N, K] ──────────────────────────────────────────────

fn run_qmv_test(bits: u32, kernel: metaltile_core::ir::Kernel, dt: Dt) {
    let n = 64usize; // out_dim
    let k = 256usize; // in_dim — multiple of 32
    let group_size = 64usize;

    // Deterministic synthetic weight matrix + input vector.
    let w: Vec<f32> = (0..n * k).map(|i| (((i * 31 + 7) % 53) as f32 - 26.0) * 0.03).collect();
    let x: Vec<f32> = (0..k).map(|d| (((d * 17 + 3) % 19) as f32 - 9.0) * 0.05).collect();

    let (packed, scales, biases) = quantize_matrix(&w, n, k, group_size, bits);
    let w_deq = dequant_matrix(&packed, &scales, &biases, n, k, group_size, bits);
    let expected = oracle_matvec(&w_deq, &x, n, k);

    let actual = dispatch_family(
        kernel,
        &packed,
        &scales,
        &biases,
        &x,
        dt,
        k as u32,
        n as u32,
        group_size as u32,
        n,
        1,
        n,
    );
    assert!(actual.iter().any(|&v| v != 0.0), "qmv b{bits}: all-zero output (empty body?)");
    let diff = max_abs_diff(&actual, &expected);
    // Oracle dequantizes from the same packed words — disagreement is
    // pure GPU-vs-CPU arithmetic drift, not quant error.
    let tol = if matches!(dt, Dt::F32) { 1e-3 } else { 2e-2 };
    assert!(diff < tol, "qmv b{bits} {dt:?}: max |diff| = {diff:.2e}");
}

#[test]
fn qmv_b4_matches_oracle_f32() {
    let _g = gpu_lock();
    run_qmv_test(4, mt_qmv_b4::kernel_ir_for(Dt::F32.to_dtype()), Dt::F32);
}

#[test]
fn qmv_b8_matches_oracle_f32() {
    let _g = gpu_lock();
    run_qmv_test(8, mt_qmv_b8::kernel_ir_for(Dt::F32.to_dtype()), Dt::F32);
}

#[test]
fn qmv_b3_matches_oracle_f32() {
    let _g = gpu_lock();
    run_qmv_test(3, mt_qmv_b3::kernel_ir_for(Dt::F32.to_dtype()), Dt::F32);
}

#[test]
fn qmv_b5_matches_oracle_f32() {
    let _g = gpu_lock();
    run_qmv_test(5, mt_qmv_b5::kernel_ir_for(Dt::F32.to_dtype()), Dt::F32);
}

#[test]
fn qmv_b6_matches_oracle_f32() {
    let _g = gpu_lock();
    run_qmv_test(6, mt_qmv_b6::kernel_ir_for(Dt::F32.to_dtype()), Dt::F32);
}

#[test]
fn qmv_b4_matches_oracle_bf16() {
    let _g = gpu_lock();
    run_qmv_test(4, mt_qmv_b4::kernel_ir_for(Dt::Bf16.to_dtype()), Dt::Bf16);
}

#[test]
fn qmv_b8_matches_oracle_f16() {
    let _g = gpu_lock();
    run_qmv_test(8, mt_qmv_b8::kernel_ir_for(Dt::F16.to_dtype()), Dt::F16);
}

// ─── qvm — vecmat, W [K, N] ──────────────────────────────────────────────

fn run_qvm_test(bits: u32, kernel: metaltile_core::ir::Kernel, dt: Dt) {
    let n = 64usize; // out_dim (columns of W)
    let k = 256usize; // in_dim (rows of W) — multiple of 32
    let group_size = 64usize;

    // W is [K, N] row-major; groups run along K, so quantize the
    // *transpose* view: each "row" of the quantizer is a length-K
    // column of W. Build W column-major-quantized by quantizing the
    // transposed [N, K] matrix, then transposing the dequant back.
    let w_t: Vec<f32> = (0..n * k).map(|i| (((i * 29 + 5) % 47) as f32 - 23.0) * 0.025).collect(); // logical [N, K]
    let x: Vec<f32> = (0..k).map(|d| (((d * 11 + 2) % 23) as f32 - 11.0) * 0.04).collect();

    // Quantize the [N, K] view (groups along K), then the kernel reads
    // W as [K, N]: code (d, col) of the [K,N] matrix == code (col, d)
    // of the [N,K] matrix. We pack W as [K, N] explicitly to match.
    let n_groups = k / group_size;
    let words_per_krow = (n as u32 * bits).div_ceil(32) as usize;
    let mut packed = vec![0u32; k * words_per_krow];
    // scales/biases laid out [K/group, N].
    let mut scales = vec![0.0f32; n_groups * n];
    let mut biases = vec![0.0f32; n_groups * n];
    // Quantize per (column, group) — the contraction axis is K.
    for col in 0..n {
        for g in 0..n_groups {
            let off = g * group_size;
            let slice: Vec<f32> = (0..group_size).map(|i| w_t[col * k + off + i]).collect();
            let (codes, s, b) = quantize_group(&slice, bits);
            scales[g * n + col] = s;
            biases[g * n + col] = b;
            // Store each code at (d = off+i, col) of the [K, N] stream.
            for (i, &c) in codes.iter().enumerate() {
                let d = off + i;
                let bit_off = col as u32 * bits;
                let word = (bit_off / 32) as usize;
                let shift = bit_off % 32;
                let base = d * words_per_krow;
                packed[base + word] |= (c & ((1u32 << bits) - 1)) << shift;
                let lo = 32 - shift;
                if lo < bits {
                    packed[base + word + 1] |= c >> lo;
                }
            }
        }
    }
    // Dequantize W back as [K, N] for the oracle.
    let mask = (1u64 << bits) - 1;
    let mut w_deq = vec![0.0f32; k * n];
    for d in 0..k {
        for col in 0..n {
            let bit_off = col as u32 * bits;
            let word = (bit_off / 32) as usize;
            let shift = bit_off % 32;
            let base = d * words_per_krow;
            let lo = (packed[base + word] as u64) >> shift;
            let hi = if 32 - shift < bits {
                (packed[base + word + 1] as u64) << (32 - shift)
            } else {
                0
            };
            let code = ((lo | hi) & mask) as f32;
            let g = d / group_size;
            w_deq[d * n + col] = code * scales[g * n + col] + biases[g * n + col];
        }
    }
    let expected = oracle_vecmat(&w_deq, &x, n, k);

    let actual = dispatch_family(
        kernel,
        &packed,
        &scales,
        &biases,
        &x,
        dt,
        k as u32,
        n as u32,
        group_size as u32,
        n,
        1,
        n,
    );
    assert!(actual.iter().any(|&v| v != 0.0), "qvm b{bits}: all-zero output (empty body?)");
    let diff = max_abs_diff(&actual, &expected);
    let tol = if matches!(dt, Dt::F32) { 1e-3 } else { 2e-2 };
    assert!(diff < tol, "qvm b{bits} {dt:?}: max |diff| = {diff:.2e}");
}

#[test]
fn qvm_b4_matches_oracle_f32() {
    let _g = gpu_lock();
    run_qvm_test(4, mt_qvm_b4::kernel_ir_for(Dt::F32.to_dtype()), Dt::F32);
}

#[test]
fn qvm_b8_matches_oracle_f32() {
    let _g = gpu_lock();
    run_qvm_test(8, mt_qvm_b8::kernel_ir_for(Dt::F32.to_dtype()), Dt::F32);
}

#[test]
fn qvm_b3_matches_oracle_f32() {
    let _g = gpu_lock();
    run_qvm_test(3, mt_qvm_b3::kernel_ir_for(Dt::F32.to_dtype()), Dt::F32);
}

#[test]
fn qvm_b5_matches_oracle_f32() {
    let _g = gpu_lock();
    run_qvm_test(5, mt_qvm_b5::kernel_ir_for(Dt::F32.to_dtype()), Dt::F32);
}

#[test]
fn qvm_b6_matches_oracle_f32() {
    let _g = gpu_lock();
    run_qvm_test(6, mt_qvm_b6::kernel_ir_for(Dt::F32.to_dtype()), Dt::F32);
}

#[test]
fn qvm_b4_matches_oracle_bf16() {
    let _g = gpu_lock();
    run_qvm_test(4, mt_qvm_b4::kernel_ir_for(Dt::Bf16.to_dtype()), Dt::Bf16);
}

// ─── qmm — batched matvec, W [N, K], M rows ──────────────────────────────

fn run_qmm_test(bits: u32, kernel: metaltile_core::ir::Kernel, dt: Dt) {
    let n = 64usize;
    let k = 256usize;
    let m = 4usize; // batch rows
    let group_size = 64usize;

    let w: Vec<f32> = (0..n * k).map(|i| (((i * 31 + 7) % 53) as f32 - 26.0) * 0.03).collect();
    // M independent input rows.
    let x: Vec<f32> = (0..m * k).map(|i| (((i * 17 + 3) % 19) as f32 - 9.0) * 0.05).collect();

    let (packed, scales, biases) = quantize_matrix(&w, n, k, group_size, bits);
    let w_deq = dequant_matrix(&packed, &scales, &biases, n, k, group_size, bits);

    // Expected: [M, N] row-major — each M-row is an independent matvec.
    let mut expected = vec![0.0f32; m * n];
    for mr in 0..m {
        let row_out = oracle_matvec(&w_deq, &x[mr * k..mr * k + k], n, k);
        expected[mr * n..mr * n + n].copy_from_slice(&row_out);
    }

    let actual = dispatch_family(
        kernel,
        &packed,
        &scales,
        &biases,
        &x,
        dt,
        k as u32,
        n as u32,
        group_size as u32,
        n,
        m,
        m * n,
    );
    assert!(actual.iter().any(|&v| v != 0.0), "qmm b{bits}: all-zero output (empty body?)");
    let diff = max_abs_diff(&actual, &expected);
    let tol = if matches!(dt, Dt::F32) { 1e-3 } else { 2e-2 };
    assert!(diff < tol, "qmm b{bits} {dt:?}: max |diff| = {diff:.2e}");
}

#[test]
fn qmm_b4_matches_oracle_f32() {
    let _g = gpu_lock();
    run_qmm_test(4, mt_qmm_b4::kernel_ir_for(Dt::F32.to_dtype()), Dt::F32);
}

#[test]
fn qmm_b8_matches_oracle_f32() {
    let _g = gpu_lock();
    run_qmm_test(8, mt_qmm_b8::kernel_ir_for(Dt::F32.to_dtype()), Dt::F32);
}

#[test]
fn qmm_b3_matches_oracle_f32() {
    let _g = gpu_lock();
    run_qmm_test(3, mt_qmm_b3::kernel_ir_for(Dt::F32.to_dtype()), Dt::F32);
}

#[test]
fn qmm_b5_matches_oracle_f32() {
    let _g = gpu_lock();
    run_qmm_test(5, mt_qmm_b5::kernel_ir_for(Dt::F32.to_dtype()), Dt::F32);
}

#[test]
fn qmm_b6_matches_oracle_f32() {
    let _g = gpu_lock();
    run_qmm_test(6, mt_qmm_b6::kernel_ir_for(Dt::F32.to_dtype()), Dt::F32);
}

#[test]
fn qmm_b4_matches_oracle_bf16() {
    let _g = gpu_lock();
    run_qmm_test(4, mt_qmm_b4::kernel_ir_for(Dt::Bf16.to_dtype()), Dt::Bf16);
}
