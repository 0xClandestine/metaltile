//! End-to-end correctness for MoE orchestration kernels — router
//! top-k (plus future permute / unpermute).
//!
//! Compares GPU output to a straight CPU reference. The reference is
//! a faithful re-statement of the kernel algorithm: k iterative
//! argmax passes with mask of previously-chosen indices, then softmax
//! over the chosen k values.

#![cfg(target_os = "macos")]

use std::collections::BTreeMap;

mod common;

use common::gpu_lock;
use metaltile_core::dtype::DType;
use metaltile_runtime::Context;
use metaltile_std::ffai::moe::mt_moe_router_topk;

#[allow(clippy::too_many_arguments)]
fn run_topk(
    ctx: &Context,
    dtype: DType,
    router_logits_bytes: &[u8],
    n_rows: usize,
    n_experts: usize,
    k: usize,
    out_w_bytes_per_elem: usize,
) -> (Vec<u32>, Vec<u8>) {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("router_logits".into(), router_logits_bytes.to_vec());
    buffers.insert("indices_out".into(), vec![0u8; n_rows * k * 4]);
    buffers.insert("weights_out".into(), vec![0u8; n_rows * k * out_w_bytes_per_elem]);
    buffers.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());

    let mut kernel = mt_moe_router_topk::kernel_ir_for(dtype);
    kernel.mode = metaltile_core::ir::KernelMode::Reduction;
    // Grid: one TG per token row. tpg=32 (single simdgroup).
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n_rows, 1, 1], [32, 1, 1])
        .expect("dispatch_with_grid should succeed");
    let idx_bytes = result.outputs.get("indices_out").expect("indices_out").clone();
    let w_bytes = result.outputs.get("weights_out").expect("weights_out").clone();
    let indices: Vec<u32> =
        idx_bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    (indices, w_bytes)
}

fn cpu_topk_reference(
    router_logits: &[f32],
    n_rows: usize,
    n_experts: usize,
    k: usize,
) -> (Vec<u32>, Vec<f32>) {
    let mut indices = vec![0u32; n_rows * k];
    let mut weights = vec![0.0f32; n_rows * k];
    for row in 0..n_rows {
        let row_base = row * n_experts;
        let mut chosen = Vec::with_capacity(k);
        let mut chosen_vals = Vec::with_capacity(k);
        for _ in 0..k {
            let mut best_val = f32::NEG_INFINITY;
            let mut best_idx = 0usize;
            for j in 0..n_experts {
                if chosen.contains(&(j as u32)) {
                    continue;
                }
                let v = router_logits[row_base + j];
                if v > best_val {
                    best_val = v;
                    best_idx = j;
                }
            }
            chosen.push(best_idx as u32);
            chosen_vals.push(best_val);
        }
        // Softmax over chosen.
        let max_v = chosen_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = chosen_vals.iter().map(|&v| (v - max_v).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        for j in 0..k {
            indices[row * k + j] = chosen[j];
            weights[row * k + j] = exp_vals[j] / sum_exp;
        }
    }
    (indices, weights)
}

#[test]
fn mt_moe_router_topk_matches_cpu_reference_f32() {
    // Small shape covering the simdgroup edge case (n_experts > 32
    // so each lane scans 2+ entries) and exercising the chosen-mask
    // logic with k > 1.
    let n_rows = 8usize;
    let n_experts = 64usize;
    let k = 4usize;

    // Deterministic logits — distinct values so top-k is unambiguous.
    let logits: Vec<f32> = (0..n_rows * n_experts)
        .map(|i| ((i as f32 * 0.13) % 7.0) - 3.5 + (i as f32 * 0.001))
        .collect();
    let (ref_idx, ref_w) = cpu_topk_reference(&logits, n_rows, n_experts, k);

    let logits_bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new on macOS");
    let (gpu_idx, gpu_w_bytes) = run_topk(&ctx, DType::F32, &logits_bytes, n_rows, n_experts, k, 4);
    let gpu_w: Vec<f32> =
        gpu_w_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    // Indices must match exactly.
    assert_eq!(
        gpu_idx, ref_idx,
        "indices mismatch — GPU vs CPU reference\nGPU: {gpu_idx:?}\nCPU: {ref_idx:?}",
    );

    // Weights match within fp32 softmax tolerance.
    let mut max_diff = 0.0f32;
    for (i, (&g, &r)) in gpu_w.iter().zip(ref_w.iter()).enumerate() {
        let d = (g - r).abs();
        if d > max_diff {
            max_diff = d;
            assert!(d < 1e-5, "weight[{i}] diverges: gpu={g:.6} ref={r:.6} diff={d:.2e}");
        }
    }
}
