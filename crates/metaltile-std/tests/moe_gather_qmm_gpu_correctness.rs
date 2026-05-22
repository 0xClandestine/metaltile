//! GPU correctness for `ffai::moe::mt_moe_gather_qmm_int4`.
//!
//! Matches MLX's `gatherQuantizedMM` (called by `SwitchLinear` →
//! `SwitchGLU` in mlx-swift-lm) at the cell level: per-row expert routing
//! + int4-quantized per-expert weight matmul. Verifies against a
//! full-precision CPU oracle that does the same routing + matmul in f32.
//!
//! Tolerance is wide (1e-1 abs) because the int4 quantization itself is a
//! lossy approximation of the f32 reference; the kernel just has to match
//! its own dequant + matmul (cosine ≥ 0.99 vs the dequant-and-matmul CPU
//! oracle).

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::moe::{
    mt_moe_gather_qmm_b3, mt_moe_gather_qmm_b5, mt_moe_gather_qmm_b6, mt_moe_gather_qmm_b8,
    mt_moe_gather_qmm_int4,
};

/// Pack a row of int4 weights into uint32s (8 per uint, LSB-first per nibble).
fn pack_int4_row(weights: &[u32]) -> Vec<u32> {
    assert!(weights.len() % 8 == 0);
    weights
        .chunks_exact(8)
        .map(|chunk| {
            let mut packed = 0u32;
            for (i, &q) in chunk.iter().enumerate() {
                packed |= (q & 0xf) << (i * 4);
            }
            packed
        })
        .collect()
}

/// CPU oracle: per-row, look up expert via expert_offsets, dequantize
/// weight row, dot against input row.
#[allow(clippy::too_many_arguments)]
fn cpu_gather_qmm_int4(
    x: &[f32],
    weight_packed: &[u32],
    scales: &[f32],
    biases: &[f32],
    expert_offsets: &[u32],
    t_rows: usize,
    k_in: usize,
    m_out: usize,
    n_experts: usize,
    group_size: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; t_rows * m_out];
    let weight_stride_m = k_in / 8;
    let groups_per_row = k_in / group_size;

    for row in 0..t_rows {
        // Resolve expert: first e where row < expert_offsets[e+1].
        let mut expert = 0;
        for e in 0..n_experts {
            if (row as u32) < expert_offsets[e + 1] {
                expert = e;
                break;
            }
        }

        for m in 0..m_out {
            let weight_row_base = expert * m_out * weight_stride_m + m * weight_stride_m;
            let scale_row_base = expert * m_out * groups_per_row + m * groups_per_row;
            let x_row_base = row * k_in;

            let mut acc = 0.0_f32;
            for pack_idx in 0..(k_in / 8) {
                let packed = weight_packed[weight_row_base + pack_idx];
                let k_first = pack_idx * 8;
                let g = k_first / group_size;
                let scale = scales[scale_row_base + g];
                let bias = biases[scale_row_base + g];
                for nib in 0..8 {
                    let q = (packed >> (nib * 4)) & 0xf;
                    let w = q as f32 * scale + bias;
                    let xv = x[x_row_base + k_first + nib];
                    acc += w * xv;
                }
            }
            out[row * m_out + m] = acc;
        }
    }
    out
}

/// Make a small but realistic test case: 3 experts, hidden=32, m_out=8,
/// group_size=32 (one group per row), 6 rows distributed across experts.
#[test]
fn moe_gather_qmm_int4_matches_cpu_oracle_f32() {
    let _g = gpu_lock();
    let n_experts = 3usize;
    let k_in = 32usize;
    let m_out = 8usize;
    let group_size = 32usize;
    let t_rows = 6usize;

    // Expert offsets: rows [0..2) → expert 0, [2..5) → expert 1, [5..6) → expert 2.
    let expert_offsets: Vec<u32> = vec![0, 2, 5, 6];

    // Random-ish quantized weights, scales, biases.
    let mut weight_unpacked = vec![0u32; n_experts * m_out * k_in];
    for (i, w) in weight_unpacked.iter_mut().enumerate() {
        *w = (((i as u32) * 7 + 3) & 0xf) as u32;
    }
    let weight_packed: Vec<u32> =
        weight_unpacked.chunks_exact(k_in).flat_map(pack_int4_row).collect();

    let scales: Vec<f32> =
        (0..n_experts * m_out * (k_in / group_size)).map(|i| 0.01 + 0.001 * (i as f32)).collect();
    let biases: Vec<f32> =
        (0..n_experts * m_out * (k_in / group_size)).map(|i| -0.05 + 0.002 * (i as f32)).collect();

    let x: Vec<f32> = (0..t_rows * k_in).map(|i| 0.1 * ((i as f32 * 0.17).sin())).collect();

    let y_cpu = cpu_gather_qmm_int4(
        &x,
        &weight_packed,
        &scales,
        &biases,
        &expert_offsets,
        t_rows,
        k_in,
        m_out,
        n_experts,
        group_size,
    );

    // Run on GPU.
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(&x, Dt::F32));
    buffers.insert("weight_packed".into(), {
        weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect()
    });
    buffers.insert("scales".into(), pack_bytes(&scales, Dt::F32));
    buffers.insert("biases".into(), pack_bytes(&biases, Dt::F32));
    buffers.insert("expert_offsets".into(), {
        expert_offsets.iter().flat_map(|o| o.to_le_bytes()).collect()
    });
    buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * m_out], Dt::F32));
    buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    buffers.insert("m_out".into(), (m_out as u32).to_le_bytes().to_vec());
    buffers.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_moe_gather_qmm_int4::kernel_ir_for(Dt::F32.to_dtype());
    kernel.mode = KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [m_out, t_rows, 1], [32, 1, 1])
        .expect("mt_moe_gather_qmm_int4 dispatch");

    let y_gpu = unpack_bytes(result.outputs.get("out").expect("out"), Dt::F32);

    let max_diff = y_cpu.iter().zip(&y_gpu).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    assert!(max_diff < 1e-4, "max |y_cpu - y_gpu| = {max_diff:.2e}");
}

/// Same small case as the f32 oracle test but exercised at f16 / bf16.
/// The kernel casts every operand to f32 and accumulates in f32 — only
/// the input quantisation of `x` / `scales` / `biases` and the final
/// output store carry `T` precision, so the CPU oracle rounds those
/// operands through the dtype to match.
fn run_small_case_dtype(dt: Dt) {
    let _g = gpu_lock();
    let n_experts = 3usize;
    let k_in = 32usize;
    let m_out = 8usize;
    let group_size = 32usize;
    let t_rows = 6usize;

    let expert_offsets: Vec<u32> = vec![0, 2, 5, 6];

    let mut weight_unpacked = vec![0u32; n_experts * m_out * k_in];
    for (i, w) in weight_unpacked.iter_mut().enumerate() {
        *w = ((i as u32) * 7 + 3) & 0xf;
    }
    let weight_packed: Vec<u32> =
        weight_unpacked.chunks_exact(k_in).flat_map(pack_int4_row).collect();

    // Round operands through the dtype so the oracle sees the same
    // quantised values the kernel loads.
    let scales: Vec<f32> = (0..n_experts * m_out * (k_in / group_size))
        .map(|i| dt.round(0.01 + 0.001 * (i as f32)))
        .collect();
    let biases: Vec<f32> = (0..n_experts * m_out * (k_in / group_size))
        .map(|i| dt.round(-0.05 + 0.002 * (i as f32)))
        .collect();
    let x: Vec<f32> =
        (0..t_rows * k_in).map(|i| dt.round(0.1 * ((i as f32 * 0.17).sin()))).collect();

    let y_cpu = cpu_gather_qmm_int4(
        &x,
        &weight_packed,
        &scales,
        &biases,
        &expert_offsets,
        t_rows,
        k_in,
        m_out,
        n_experts,
        group_size,
    );

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(&x, dt));
    buffers.insert("weight_packed".into(), {
        weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect()
    });
    buffers.insert("scales".into(), pack_bytes(&scales, dt));
    buffers.insert("biases".into(), pack_bytes(&biases, dt));
    buffers.insert("expert_offsets".into(), {
        expert_offsets.iter().flat_map(|o| o.to_le_bytes()).collect()
    });
    buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * m_out], dt));
    buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    buffers.insert("m_out".into(), (m_out as u32).to_le_bytes().to_vec());
    buffers.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_moe_gather_qmm_int4::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [m_out, t_rows, 1], [32, 1, 1])
        .expect("mt_moe_gather_qmm_int4 dispatch");

    let y_gpu = unpack_bytes(result.outputs.get("out").expect("out"), dt);

    // The output store rounds to `T`; allow one ulp of that dtype plus
    // a small f32-accumulation margin.
    let (tol, label) = match dt {
        Dt::F32 => (1e-4, "f32"),
        Dt::F16 => (5e-3, "f16"),
        Dt::Bf16 => (3e-2, "bf16"),
    };
    let max_diff = y_cpu.iter().zip(&y_gpu).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    assert!(max_diff < tol, "{label}: max |y_cpu - y_gpu| = {max_diff:.2e} (tol {tol:.0e})");
}

// ─── wider-precision gather matmul: int3 / int5 / int6 / int8 ────────────

/// Pack a row of unsigned codes into a u32 bit-stream at `bits` width.
/// Pow2 widths land pack-aligned; odd widths may straddle a u32 word.
fn pack_codes_bits(codes: &[u32], bits: u32) -> Vec<u32> {
    let n_words = (codes.len() as u32 * bits).div_ceil(32) as usize;
    let mut words = vec![0u32; n_words];
    for (i, &c) in codes.iter().enumerate() {
        let bit_off = i as u32 * bits;
        let word = (bit_off / 32) as usize;
        let shift = bit_off % 32;
        words[word] |= (c & ((1u32 << bits) - 1)) << shift;
        let lo = 32 - shift;
        if lo < bits {
            words[word + 1] |= c >> lo;
        }
    }
    words
}

/// CPU oracle for gather-qmm at an arbitrary bit width. `weight_packed`
/// is the concatenation, over (expert, m_out) rows, of each row's
/// bit-stream-packed codes.
#[allow(clippy::too_many_arguments)]
fn cpu_gather_qmm_bits(
    x: &[f32],
    weight_packed: &[u32],
    scales: &[f32],
    biases: &[f32],
    expert_offsets: &[u32],
    t_rows: usize,
    k_in: usize,
    m_out: usize,
    n_experts: usize,
    group_size: usize,
    bits: u32,
) -> Vec<f32> {
    let words_per_row = (k_in as u32 * bits).div_ceil(32) as usize;
    let groups_per_row = k_in / group_size;
    let mask = (1u64 << bits) - 1;
    let mut out = vec![0.0f32; t_rows * m_out];
    for row in 0..t_rows {
        let mut expert = 0;
        for e in 0..n_experts {
            if (row as u32) < expert_offsets[e + 1] {
                expert = e;
                break;
            }
        }
        for m in 0..m_out {
            let wbase = (expert * m_out + m) * words_per_row;
            let sbase = (expert * m_out + m) * groups_per_row;
            let mut acc = 0.0f32;
            for d in 0..k_in {
                let bit_off = d as u32 * bits;
                let word = (bit_off / 32) as usize;
                let shift = bit_off % 32;
                let lo = (weight_packed[wbase + word] as u64) >> shift;
                let hi = if 32 - shift < bits {
                    (weight_packed[wbase + word + 1] as u64) << (32 - shift)
                } else {
                    0
                };
                let q = ((lo | hi) & mask) as f32;
                let g = d / group_size;
                let w = q * scales[sbase + g] + biases[sbase + g];
                acc += w * x[row * k_in + d];
            }
            out[row * m_out + m] = acc;
        }
    }
    out
}

/// Build the small 3-expert / k=64 / m_out=8 case at `bits` width and
/// assert the kernel matches the f32 dequant oracle.
fn run_gather_qmm_bits(bits: u32, kernel: metaltile_core::ir::Kernel) {
    let n_experts = 3usize;
    let k_in = 64usize; // multiple of 32 and of 32/8 — works for all widths
    let m_out = 8usize;
    let group_size = 64usize;
    let t_rows = 6usize;

    let expert_offsets: Vec<u32> = vec![0, 2, 5, 6];
    let max_code = (1u32 << bits) - 1;

    // Per-(expert, m_out) row of k_in codes, bit-stream packed.
    let n_rows = n_experts * m_out;
    let mut weight_packed: Vec<u32> = Vec::new();
    for r in 0..n_rows {
        let codes: Vec<u32> =
            (0..k_in).map(|d| ((r as u32 * 13 + d as u32 * 7 + 3) % (max_code + 1))).collect();
        weight_packed.extend(pack_codes_bits(&codes, bits));
    }

    let groups_total = n_rows * (k_in / group_size);
    let scales: Vec<f32> = (0..groups_total).map(|i| 0.01 + 0.001 * (i as f32)).collect();
    let biases: Vec<f32> = (0..groups_total).map(|i| -0.05 + 0.002 * (i as f32)).collect();
    let x: Vec<f32> = (0..t_rows * k_in).map(|i| 0.1 * ((i as f32 * 0.17).sin())).collect();

    let y_cpu = cpu_gather_qmm_bits(
        &x, &weight_packed, &scales, &biases, &expert_offsets, t_rows, k_in, m_out, n_experts,
        group_size, bits,
    );

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(&x, Dt::F32));
    buffers.insert(
        "weight_packed".into(),
        weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect(),
    );
    buffers.insert("scales".into(), pack_bytes(&scales, Dt::F32));
    buffers.insert("biases".into(), pack_bytes(&biases, Dt::F32));
    buffers.insert(
        "expert_offsets".into(),
        expert_offsets.iter().flat_map(|o| o.to_le_bytes()).collect(),
    );
    buffers.insert("out".into(), pack_bytes(&vec![0.0f32; t_rows * m_out], Dt::F32));
    buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    buffers.insert("m_out".into(), (m_out as u32).to_le_bytes().to_vec());
    buffers.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = kernel;
    kernel.mode = KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [m_out, t_rows, 1], [32, 1, 1])
        .expect("gather_qmm bits dispatch");

    let y_gpu = unpack_bytes(result.outputs.get("out").expect("out"), Dt::F32);
    assert!(y_gpu.iter().any(|&v| v != 0.0), "gather_qmm b{bits}: all-zero output (empty body?)");
    let max_diff = y_cpu.iter().zip(&y_gpu).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_diff < 1e-3, "gather_qmm b{bits}: max |y_cpu - y_gpu| = {max_diff:.2e}");
}

#[test]
fn moe_gather_qmm_b8_matches_oracle_f32() {
    let _g = gpu_lock();
    run_gather_qmm_bits(8, mt_moe_gather_qmm_b8::kernel_ir_for(Dt::F32.to_dtype()));
}

#[test]
fn moe_gather_qmm_b3_matches_oracle_f32() {
    let _g = gpu_lock();
    run_gather_qmm_bits(3, mt_moe_gather_qmm_b3::kernel_ir_for(Dt::F32.to_dtype()));
}

#[test]
fn moe_gather_qmm_b5_matches_oracle_f32() {
    let _g = gpu_lock();
    run_gather_qmm_bits(5, mt_moe_gather_qmm_b5::kernel_ir_for(Dt::F32.to_dtype()));
}

#[test]
fn moe_gather_qmm_b6_matches_oracle_f32() {
    let _g = gpu_lock();
    run_gather_qmm_bits(6, mt_moe_gather_qmm_b6::kernel_ir_for(Dt::F32.to_dtype()));
}

#[test]
fn moe_gather_qmm_int4_matches_cpu_oracle_f16() { run_small_case_dtype(Dt::F16); }

#[test]
fn moe_gather_qmm_int4_matches_cpu_oracle_bf16() { run_small_case_dtype(Dt::Bf16); }

/// Realistic Qwen3.6-35B-A3B layer shape: K_in=2048, M_out=256,
/// N_experts=128, group_size=64. 4 routed tokens across 3 experts (token
/// count tiny for test speed; kernel handles arbitrary T via grid_y).
#[test]
fn moe_gather_qmm_int4_qwen36_shape_f32() {
    let _g = gpu_lock();
    let n_experts = 128usize;
    let k_in = 2048usize;
    let m_out = 256usize;
    let group_size = 64usize;
    let t_rows = 4usize;

    // Most experts have zero rows; a handful own all the work.
    let mut expert_offsets: Vec<u32> = vec![0; n_experts + 1];
    // Rows 0..2 → expert 7, rows 2..3 → expert 42, row 3 → expert 100.
    for e in 0..=n_experts {
        let off = if e <= 7 {
            0
        } else if e <= 42 {
            2
        } else if e <= 100 {
            3
        } else {
            t_rows as u32
        };
        expert_offsets[e] = off;
    }

    let total_weights = n_experts * m_out * k_in;
    let weight_unpacked: Vec<u32> =
        (0..total_weights).map(|i| ((i as u32) * 7 + 3) & 0xf).collect();
    let weight_packed: Vec<u32> =
        weight_unpacked.chunks_exact(k_in).flat_map(pack_int4_row).collect();

    let groups_total = n_experts * m_out * (k_in / group_size);
    let scales: Vec<f32> =
        (0..groups_total).map(|i| 0.005 + 0.0001 * ((i as f32 * 0.03).sin())).collect();
    let biases: Vec<f32> =
        (0..groups_total).map(|i| -0.02 + 0.0005 * ((i as f32 * 0.07).cos())).collect();
    let x: Vec<f32> = (0..t_rows * k_in).map(|i| 0.05 * ((i as f32 * 0.013).sin())).collect();

    let y_cpu = cpu_gather_qmm_int4(
        &x,
        &weight_packed,
        &scales,
        &biases,
        &expert_offsets,
        t_rows,
        k_in,
        m_out,
        n_experts,
        group_size,
    );

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_bytes(&x, Dt::F32));
    buffers.insert("weight_packed".into(), {
        weight_packed.iter().flat_map(|w| w.to_le_bytes()).collect()
    });
    buffers.insert("scales".into(), pack_bytes(&scales, Dt::F32));
    buffers.insert("biases".into(), pack_bytes(&biases, Dt::F32));
    buffers.insert("expert_offsets".into(), {
        expert_offsets.iter().flat_map(|o| o.to_le_bytes()).collect()
    });
    buffers.insert("out".into(), pack_bytes(&vec![0.0_f32; t_rows * m_out], Dt::F32));
    buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());
    buffers.insert("m_out".into(), (m_out as u32).to_le_bytes().to_vec());
    buffers.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
    buffers.insert("group_size".into(), (group_size as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_moe_gather_qmm_int4::kernel_ir_for(Dt::F32.to_dtype());
    kernel.mode = KernelMode::Reduction;

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [m_out, t_rows, 1], [32, 1, 1])
        .expect("mt_moe_gather_qmm_int4 qwen36-shape dispatch");

    let y_gpu = unpack_bytes(result.outputs.get("out").expect("out"), Dt::F32);

    // Tighter cosine check at production shape — large reductions can
    // accumulate fp noise.
    let mut dot = 0.0_f64;
    let mut nc = 0.0_f64;
    let mut ng = 0.0_f64;
    for (a, b) in y_cpu.iter().zip(&y_gpu) {
        dot += (*a as f64) * (*b as f64);
        nc += (*a as f64) * (*a as f64);
        ng += (*b as f64) * (*b as f64);
    }
    let cos = dot / (nc.sqrt() * ng.sqrt() + 1e-12);
    assert!(cos >= 0.999, "cosine vs CPU oracle = {cos:.6} (want ≥ 0.999)");
}
