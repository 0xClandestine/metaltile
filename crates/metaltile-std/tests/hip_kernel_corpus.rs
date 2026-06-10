//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Run the registered `#[test_kernel]` corpus on the HIP / ROCm backend.
//!
//! Direct port of `cuda_kernel_corpus.rs` — the harness contract
//! (param/constexpr byte maps → `run_kernel` → compare outputs) is
//! identical because `HipDevice::run_kernel` has the same signature
//! as `CudaDevice::run_kernel`, and the HIP backend shares the CUDA
//! op-walker (only the vendor-dialect lowering differs). The
//! PASS/MISMATCH/UNSUPPORTED/ERROR triage matches the CUDA run so
//! results are directly comparable (`AMD_BACKEND_SPEC.md`; baseline:
//! full corpus on RDNA wave32, RX 9070 XT / gfx1201).
//!
//! Runs only with `--features hip`.
#![cfg(feature = "hip")]

use std::collections::BTreeMap;

use metaltile::{CodegenError, HipDevice, MetalTileError, core::dtype::DType};

fn read_raw_f32(bytes: &[u8], dt: DType, n: usize) -> Vec<f32> {
    match dt {
        DType::F32 => bytes
            .chunks_exact(4)
            .take(n)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        DType::F16 => bytes
            .chunks_exact(2)
            .take(n)
            .map(|b| {
                let bits = u16::from_le_bytes(b.try_into().unwrap());
                half::f16::from_bits(bits).to_f32()
            })
            .collect(),
        DType::BF16 => bytes
            .chunks_exact(2)
            .take(n)
            .map(|b| {
                let bits = u16::from_le_bytes(b.try_into().unwrap());
                half::bf16::from_bits(bits).to_f32()
            })
            .collect(),
        DType::I32 => bytes
            .chunks_exact(4)
            .take(n)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as f32)
            .collect(),
        DType::U32 => bytes
            .chunks_exact(4)
            .take(n)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as f32)
            .collect(),
        DType::I8 => bytes.iter().take(n).map(|&b| b as i8 as f32).collect(),
        DType::U8 => bytes.iter().take(n).map(|&b| b as f32).collect(),
        _ => vec![0.0; n],
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    // Non-finite-aware: a plain `(x - y).abs()` fold has two holes. (1)
    // f32::max DISCARDS NaN operands, so an all-NaN output scores 0.0 and
    // passes. (2) Agreeing non-finites must PASS: logits-mask kernels write
    // -inf on both sides, and (-inf) - (-inf) is NaN. Bitwise-equal values
    // (covers equal infinities) and NaN-on-both-sides count as agreement;
    // any one-sided NaN/inf maps to +inf so garbage fails loudly.
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            if x == y || (x.is_nan() && y.is_nan()) {
                0.0
            } else {
                let d = (x - y).abs();
                if d.is_nan() { f32::INFINITY } else { d }
            }
        })
        .fold(0.0f32, f32::max)
}

/// Kernels expected to *generate + run* but mismatch the oracle on HIP,
/// with reasons. May differ from CUDA's list — the AMD device math (e.g.
/// precise `expf` rounding) is not bit-identical to NVIDIA's. Currently
/// empty: the full corpus is within its per-kernel tolerance bands on
/// RDNA4. A failure NOT matching an entry is a regression and fails the
/// test.
const KNOWN_HARD: &[(&str, &str)] = &[];

fn known_hard(name: &str) -> bool { KNOWN_HARD.iter().any(|(k, _)| name.contains(k)) }

/// UNSUPPORTED is decided on the TYPED error, not message sniffing
/// (mirrors `cuda_kernel_corpus.rs`):
/// * `Codegen(UnsupportedOp)` — every codegen coverage gap (MMA/cooperative,
///   Tile2D, multi-dim index, ops not wired yet) is raised as this variant.
/// * `DeviceCapability` — kernels the codegen *does* cover but the target
///   arch physically cannot run, surfaced before launch. These reflect
///   bit-accuracy on what the arch CAN run, so they are not hard failures.
///
/// Anything else (hipRTC compile failure, launch error, missing buffer) on
/// a kernel we claim to support stays a hard ERROR.
fn is_unsupported(e: &MetalTileError) -> bool {
    matches!(
        e,
        MetalTileError::Codegen(CodegenError::UnsupportedOp(_))
            | MetalTileError::DeviceCapability(_)
    )
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
            if let Ok(want) = std::env::var("DUMP_HIP")
                && t.name() == want
                && dt == DType::F32
            {
                use metaltile::codegen::{CodegenBackend, HipGenerator};
                if let Ok(src) = HipGenerator::new().generate(kernel) {
                    eprintln!("==== {} (HIP) ====\n{src}\n==== end ====", t.name());
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
                    } else if known_hard(&label) {
                        known += 1;
                    } else {
                        mismatch += 1;
                        hard_failures
                            .push(format!("MISMATCH {label}: max|Δ|={worst:.3e} > {tol:.3e}"));
                    }
                },
                Err(e) => {
                    if known_hard(&label) {
                        known += 1;
                    } else if is_unsupported(&e) {
                        unsupported += 1;
                        // Bucket by the short reason (first line / key phrase).
                        let reason = e
                            .to_string()
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
                        hard_failures.push(format!("ERROR {label}: {e}"));
                    }
                },
            }
        }
    }

    eprintln!("\n=== HIP corpus result ===");
    eprintln!(
        "PASS={pass}  KNOWN_HARD={known}  MISMATCH={mismatch}  UNSUPPORTED={unsupported}  ERROR={error}"
    );
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

    // Pass floor: a regression that bucketed kernels as UNSUPPORTED (typed
    // `Codegen(UnsupportedOp)`) would otherwise shrink coverage silently.
    // The RDNA4 baseline passes the full corpus; 3500 leaves headroom for
    // device-cap variation, not for a broken emitter.
    assert!(pass >= 3500, "only {pass} kernels passed on HIP — emitter or pipeline regression");
    // Numeric mismatches are distinct from the error budget below:
    // KNOWN_HARD absorbs the documented tol-band outliers, so any other
    // oracle mismatch on a kernel that RAN is a regression.
    assert!(
        mismatch == 0,
        "{mismatch} HIP oracle mismatches on supported kernels — numerics regression"
    );
    // Hard errors (hipRTC compile / launch failures) signal genuine codegen
    // bugs, not numerics. The RDNA4 baseline runs clean; the budget leaves
    // headroom for arch variation (e.g. cooperative/MPP launch limits on
    // other wavefront configs) without letting a broad regression through.
    let error_budget: u32 = 128;
    assert!(
        error <= error_budget,
        "HIP corpus produced {error} hard errors (budget={error_budget}) — codegen regression"
    );
}
