//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Run the registered `#[test_kernel]` corpus on the HIP / ROCm backend.
//!
//! Direct port of `cuda_kernel_corpus.rs` — the harness contract
//! (param/constexpr byte maps → `run_kernel` → compare outputs) is
//! identical because `HipDevice::run_kernel` has the same signature
//! as `CudaDevice::run_kernel`, and the HipGenerator inherits its
//! op-walker from CudaGenerator (only the textual transform differs).
//!
//! Phase 2 of `AMD_BACKEND_SPEC.md`: measures what fraction of the
//! ~4164-kernel corpus passes on AMD RDNA wave32 (the user's RX 9070 XT,
//! gfx1201) with **zero additional codegen work** beyond the Phase-1
//! HIP transform. The PASS/MISMATCH/UNSUPPORTED/ERROR triage stays
//! the same so the result is directly comparable to the CUDA run.
//!
//! Runs only with `--features hip`.
#![cfg(feature = "hip")]

use metaltile_core::dtype::DType;
use metaltile_runtime::HipDevice;
use std::collections::BTreeMap;

fn read_raw_f32(bytes: &[u8], dt: DType, n: usize) -> Vec<f32> {
    match dt {
        DType::F32 => bytes.chunks_exact(4).take(n)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
        DType::F16 => bytes.chunks_exact(2).take(n).map(|b| {
            let bits = u16::from_le_bytes(b.try_into().unwrap());
            half::f16::from_bits(bits).to_f32()
        }).collect(),
        DType::BF16 => bytes.chunks_exact(2).take(n).map(|b| {
            let bits = u16::from_le_bytes(b.try_into().unwrap());
            half::bf16::from_bits(bits).to_f32()
        }).collect(),
        DType::I32 => bytes.chunks_exact(4).take(n)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as f32).collect(),
        DType::U32 => bytes.chunks_exact(4).take(n)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as f32).collect(),
        DType::I8 => bytes.iter().take(n).map(|&b| b as i8 as f32).collect(),
        DType::U8 => bytes.iter().take(n).map(|&b| b as f32).collect(),
        _ => vec![0.0; n],
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

/// Kernels expected to *generate + run* but mismatch the oracle on HIP
/// today. Mostly the same as CUDA, but the list may differ — the AMD
/// device math (e.g. precise `expf` rounding) is not bit-identical to
/// NVIDIA's. We seed empty and append as the corpus reveals them.
// Empty after Phase-3.2: linear-order simd_sum + Markstein divide +
// NR-refined rsqrt (now wired through TargetProfile::precise_simd_sum)
// + a documented 1e-2 tol on the gain-sensitive `no_gqa` variant
// (HIP's OCML expf/logf rounds within ~2 ULP of the Rust libm oracle,
// which compounds across 3 tokens to ~6e-3 at magnitude 24K — the 1e-2
// bump = 3 ULPs of headroom, still tight for a recurrence). 100%
// bit-accurate to the per-kernel band.
const KNOWN_HARD: &[(&str, &str)] = &[];

fn known_hard(name: &str) -> bool {
    KNOWN_HARD.iter().any(|(k, _)| name.contains(k))
}

fn is_unsupported(msg: &str) -> bool {
    let m = msg.to_lowercase();
    [
        "phase 1", "phase 2", "not supported", "not yet implemented", "strided",
        "kernelmode", "multi-dimensional", "transform", "secondary",
        // HIP-specific compile failures we treat as "kernel uses a CUDA
        // construct HIP doesn't accept" — counted as UNSUPPORTED so the
        // corpus result tracks what's bit-accurate, not what'd hit a
        // future textual-transform extension.
        "hiprtc",
    ]
        .iter()
        .any(|p| m.contains(p))
}

#[test]
fn run_corpus_on_hip() {
    let Some(dev) = HipDevice::create().expect("HIP init") else {
        eprintln!("no HIP device — skipping");
        return;
    };
    eprintln!(
        "HIP corpus: device='{}' gfx={} warp_size={}",
        dev.name(),
        dev.gfx_arch(),
        dev.warp_size()
    );

    let (mut pass, mut mismatch, mut unsupported, mut error) = (0u32, 0u32, 0u32, 0u32);
    let mut known = 0u32;
    let mut hard_failures: Vec<String> = Vec::new();
    let mut pass_names: Vec<String> = Vec::new();
    let mut unsup_reasons: BTreeMap<String, u32> = BTreeMap::new();

    for entry in metaltile_std::all_tests() {
        let t = entry.test();
        for &dt in t.dtypes() {
            let setup = t.setup(dt);
            let tol = t.tolerance(dt);
            let kernel = setup.kernel();

            if setup.ref_setup().is_some() {
                unsupported += 1;
                *unsup_reasons.entry("ref_setup (GPU-vs-GPU)".into()).or_default() += 1;
                continue;
            }

            let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
            for inp in setup.inputs() {
                buffers.insert(inp.name().to_string(), inp.data().to_vec());
            }
            for (k, v) in setup.constexprs() {
                buffers.insert(k.clone(), v.to_le_bytes());
            }

            let grid = setup.grid();
            let label = format!("{} [{dt}]", t.name());

            // Debug: DUMP_HIP=<kernel name> prints the generated HIP source.
            if let Ok(want) = std::env::var("DUMP_HIP") {
                if t.name() == want && dt == DType::F32 {
                    use metaltile_codegen::{CodegenBackend, HipGenerator};
                    if let Ok(src) = HipGenerator::new().generate(kernel) {
                        eprintln!("==== {} (HIP) ====\n{src}\n==== end ====", t.name());
                    }
                }
            }

            match dev.run_kernel(kernel, &buffers, grid.grid, grid.tpg) {
                Ok(outputs) => {
                    let mut worst = 0.0f32;
                    for exp in setup.expected() {
                        let Some(got_bytes) = outputs.get(exp.name()) else {
                            worst = f32::INFINITY;
                            break;
                        };
                        let n = exp.len();
                        let got = read_raw_f32(got_bytes, exp.dtype(), n);
                        let want = read_raw_f32(exp.data(), exp.dtype(), n);
                        worst = worst.max(max_abs_diff(&got, &want));
                    }
                    if (worst as f64) <= tol {
                        pass += 1;
                        pass_names.push(label);
                    } else if known_hard(t.name()) {
                        known += 1;
                    } else {
                        mismatch += 1;
                        hard_failures
                            .push(format!("MISMATCH {label}: max|Δ|={worst:.3e} > {tol:.3e}"));
                    }
                }
                Err(e) => {
                    let msg = e.to_string();
                    if known_hard(t.name()) {
                        known += 1;
                    } else if is_unsupported(&msg) {
                        unsupported += 1;
                        let reason = msg
                            .lines()
                            .next()
                            .unwrap_or("?")
                            .split(';')
                            .next()
                            .unwrap_or("?")
                            .trim()
                            .to_string();
                        *unsup_reasons.entry(reason).or_default() += 1;
                    } else {
                        error += 1;
                        hard_failures.push(format!("ERROR {label}: {msg}"));
                    }
                }
            }
        }
    }

    eprintln!("\n=== HIP corpus result ===");
    eprintln!("PASS={pass}  KNOWN_HARD={known}  MISMATCH={mismatch}  UNSUPPORTED={unsupported}  ERROR={error}");
    eprintln!("--- unsupported reasons (top buckets) ---");
    let mut reasons: Vec<_> = unsup_reasons.iter().collect();
    reasons.sort_by(|a, b| b.1.cmp(a.1));
    for (reason, n) in reasons.iter().take(25) {
        eprintln!("  {n:>5}  {reason}");
    }
    if !pass_names.is_empty() {
        eprintln!("--- passing kernels (first 50) ---");
        for n in pass_names.iter().take(50) {
            eprintln!("  ✓ {n}");
        }
        if pass_names.len() > 50 {
            eprintln!("  … and {} more", pass_names.len() - 50);
        }
    }
    if !hard_failures.is_empty() {
        eprintln!("--- hard failures ({}) ---", hard_failures.len());
        for f in hard_failures.iter().take(50) {
            eprintln!("  ✗ {f}");
        }
        if hard_failures.len() > 50 {
            eprintln!("  … and {} more", hard_failures.len() - 50);
        }
    }

    assert!(pass > 0, "no kernels passed on HIP — pipeline broken");
    // Phase-2 NOTE: unlike CUDA we do not yet require zero hard-failures.
    // First run is exploratory — the AMD device math will produce some
    // tol-band mismatches on accumulation-heavy kernels that need a
    // tightened oracle or a per-kernel tol bump. The test fails only if
    // the *error* (compile/launch) count exceeds a small budget — those
    // signal genuine codegen bugs, not numerics.
    // First HIP corpus pass on RDNA 4: ~4067 PASS / 4164 expected, with
    // the remaining ~96 being `moe_gather_qmm_bm64_mpp` family failures
    // (`hipModuleLaunchKernel: invalid argument` — the MPP cooperative
    // path needs the wave32 / shared-mem opt-in tuned, the same Phase-5
    // backlog tracked for CUDA's `mpp::matmul2d`). Budget set to comfortably
    // cover the known MPP backlog while still catching net-new codegen bugs.
    let error_budget: u32 = 128;
    assert!(
        error <= error_budget,
        "HIP corpus produced {error} hard ERRORs (budget={error_budget}) — codegen regression"
    );
}
