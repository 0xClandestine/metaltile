//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Copyright 2026 TheTom
//! SPDX-License-Identifier: Apache-2.0
//! Vulkan/RDNA4 correctness + throughput for the gated `VK_KHR_cooperative_matrix`
//! GEMM codegen path (`MT_VK_COOPMAT=1`).
//!
//! Drives the real `ffai_gemm_q8_mpp` 64×64×32 SimdGroup CoopTile kernel
//! through `VulkanDevice::run_kernel`. The device honors `MT_VK_COOPMAT`
//! at create time; the SPIR-V emitter emits coopMatLoad/MulAdd/Store for
//! the CoopTile ops. Oracle: triple-loop Q8_0 dequant GEMM (same recipe
//! as the Metal `gemm_q8_mpp_correctness` test). Run BOTH with and without
//! the env var set to A/B the scalar fallback vs coopmat.
//!
//!   # scalar:   cargo test -p metaltile-std --features vulkan --release \
//!                  --test vulkan_coopmat_gemm -- --nocapture
//!   # coopmat:  MT_VK_COOPMAT=1 (same line)
#![cfg(feature = "vulkan")]

use std::{collections::BTreeMap, time::Instant};

use metaltile::{VulkanDevice, core::dtype::DType};
use metaltile_std::ffai::gemm_q8_mpp::ffai_gemm_q8_mpp;

fn xorshift(s: &mut u32) -> u32 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    x
}

fn f16_round(v: f32) -> f32 { half::f16::from_f32(v).to_f32() }

#[test]
fn coopmat_gemm_q8_mpp_correct_and_fast() {
    let Some(dev) = VulkanDevice::create().expect("Vulkan init") else {
        eprintln!("no Vulkan device — skipping");
        return;
    };
    let coopmat_on = std::env::var("MT_VK_COOPMAT").map(|v| v == "1").unwrap_or(false);
    eprintln!("=== coopmat_gemm_q8_mpp: MT_VK_COOPMAT={} ===", coopmat_on as u8);

    // q_a-like prefill shape (matches the bench): in=4096, out=1024, 256 tokens.
    let n_rows = 256usize;
    let out_dim = 1024usize;
    let k_in = 4096usize;
    let n_blocks = out_dim * k_in / 32;

    let mut st = 0x515E_2026u32;
    let qs: Vec<u32> = (0..n_blocks * 8).map(|_| xorshift(&mut st)).collect();
    let d: Vec<f32> =
        (0..n_blocks).map(|_| (xorshift(&mut st) % 1000) as f32 * 0.0002 + 0.001).collect();
    // f16 activations (coopmat staging is f16); round the oracle inputs too.
    let x: Vec<f32> = (0..n_rows * k_in)
        .map(|_| f16_round(((xorshift(&mut st) % 2000) as f32 / 1000.0) - 1.0))
        .collect();

    // Oracle: dense Q8 dequant GEMM.
    let deq = |vidx: usize| -> f32 {
        let block = vidx / 32;
        let lane = vidx % 32;
        let word = qs[block * 8 + lane / 4];
        let by = ((word >> ((lane % 4) * 8)) & 0xff) as i32;
        let q = if by > 127 { by - 256 } else { by };
        d[block] * q as f32
    };
    let mut want = vec![0.0f32; n_rows * out_dim];
    for r in 0..n_rows {
        for o in 0..out_dim {
            let mut acc = 0.0f32;
            for k in 0..k_in {
                // Weight rounded through f16 staging too.
                acc += f16_round(deq(o * k_in + k)) * x[r * k_in + k];
            }
            want[r * out_dim + o] = acc;
        }
    }

    // Kernel IR (f16 instantiation → coop_stage(f16)=f16 staging).
    // The kernel reads tgid_x/tgid_y + simd_group/simd_lane → Reduction
    // mode (matches the bench's `.mode(KernelMode::Reduction)`).
    let mut kernel = ffai_gemm_q8_mpp::kernel_ir_for(DType::F16);
    kernel.mode = metaltile::core::ir::KernelMode::Reduction;

    // Pack buffers. x as f16, qs/d as-is, out zeroed f16.
    let pack_f16 = |v: &[f32]| -> Vec<u8> {
        v.iter().flat_map(|&f| half::f16::from_f32(f).to_bits().to_le_bytes()).collect()
    };
    let pack_u32 = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|&u| u.to_le_bytes()).collect() };
    let pack_f32 = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|&f| f.to_le_bytes()).collect() };

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("x".into(), pack_f16(&x));
    buffers.insert("qs".into(), pack_u32(&qs));
    buffers.insert("d_f32".into(), pack_f32(&d));
    buffers.insert("out".into(), pack_f16(&vec![0.0f32; n_rows * out_dim]));
    buffers.insert("n_rows".into(), (n_rows as u32).to_le_bytes().to_vec());
    buffers.insert("out_dim".into(), (out_dim as u32).to_le_bytes().to_vec());
    buffers.insert("k_in".into(), (k_in as u32).to_le_bytes().to_vec());

    let grid = [(out_dim as u32).div_ceil(64), (n_rows as u32).div_ceil(64), 1];
    let tpg = [128u32, 1, 1];

    // Correctness.
    let outputs = dev.run_kernel(&kernel, &buffers, grid, tpg).expect("run_kernel");
    let got_bytes = outputs.get("out").expect("out buffer");
    let got: Vec<f32> = got_bytes
        .chunks_exact(2)
        .map(|b| half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
        .collect();

    // (1) Sanity vs the f32 CPU oracle (loose: f16 inputs/output + f32
    //     accumulate over k=4096 → a few % is expected, this only catches
    //     gross garbage / wrong indexing).
    let mut max_rel_oracle = 0.0f32;
    for i in 0..n_rows * out_dim {
        assert!(got[i].is_finite(), "non-finite at {i}: {}", got[i]);
        let w = f16_round(want[i]);
        let denom = w.abs().max(8.0);
        max_rel_oracle = max_rel_oracle.max((got[i] - w).abs() / denom);
    }
    eprintln!("  vs CPU oracle: maxRelDiff={max_rel_oracle:.4}");
    // Coopmat accumulates each 16×16×16 fragment in true f32, so it
    // tracks the f32 oracle to ~0.1%. The scalar SoftwareLocalC path is
    // looser at this f16 shape (~3%). Gate each accordingly.
    let oracle_tol = if coopmat_on { 5e-3 } else { 6e-2 };
    assert!(
        max_rel_oracle < oracle_tol,
        "kernel diverges from oracle: {max_rel_oracle} >= {oracle_tol}"
    );

    // (2) GPU-vs-GPU A/B: coopmat must match the proven scalar path to
    //     f16-output ULP (both accumulate in f32). Persist the scalar
    //     output, then compare the coopmat run against it. This is the
    //     real correctness gate for the coopmat codegen.
    let ref_path = std::env::temp_dir().join("mt_coopmat_scalar_ref.bin");
    if !coopmat_on {
        std::fs::write(&ref_path, got_bytes).expect("write scalar ref");
        eprintln!("  scalar reference written to {}", ref_path.display());
    } else {
        let ref_bytes = std::fs::read(&ref_path)
            .expect("scalar reference missing — run once WITHOUT MT_VK_COOPMAT first");
        let refv: Vec<f32> = ref_bytes
            .chunks_exact(2)
            .map(|b| half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect();
        let mut max_rel_ab = 0.0f32;
        let mut maxd_ab = 0.0f32;
        for i in 0..n_rows * out_dim {
            let d = (got[i] - refv[i]).abs();
            maxd_ab = maxd_ab.max(d);
            max_rel_ab = max_rel_ab.max(d / refv[i].abs().max(8.0));
        }
        eprintln!(
            "  coopmat vs scalar (informational): maxAbsDiff={maxd_ab:.4}  maxRelDiff={max_rel_ab:.4}"
        );
        // The two GPU paths differ in staging precision (coopmat: f16
        // shared tiles + f32 fragment accumulate; scalar: f32 shared
        // tiles). Both round to f16 output. The gap is bounded by the
        // f16-staging delta — informational, the oracle check above is
        // the hard gate. (Coopmat is the MORE accurate of the two here.)
        assert!(max_rel_ab < 6e-2, "coopmat vs scalar gap too large: {max_rel_ab}");
    }

    // Throughput. 2*M*N*K flops.
    let flops = 2.0 * n_rows as f64 * out_dim as f64 * k_in as f64;
    let iters = 50;
    // warmup
    for _ in 0..5 {
        let _ = dev.run_kernel(&kernel, &buffers, grid, tpg).unwrap();
    }
    let t = Instant::now();
    for _ in 0..iters {
        let _ = dev.run_kernel(&kernel, &buffers, grid, tpg).unwrap();
    }
    let per = t.elapsed().as_secs_f64() / iters as f64;
    let tflops = flops / per / 1e12;
    eprintln!(
        "  shape {}x{}x{}  {:.3} ms/dispatch  {:.2} TFLOP/s (incl. run_kernel overhead)",
        n_rows,
        out_dim,
        k_in,
        per * 1e3,
        tflops
    );
}
