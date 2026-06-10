//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Run the registered `#[test_kernel]` corpus on the Vulkan / SPIR-V backend.
//!
//! Direct port of `hip_kernel_corpus.rs` / `cuda_kernel_corpus.rs`. The
//! `VulkanDevice::run_kernel` signature matches the CUDA / HIP one (3-D
//! grid × 3-D block, `BTreeMap` of param bytes), so the iteration loop is
//! the same; only the device handle differs. The triage matches the CUDA
//! run so results are directly comparable (`VULKAN_BACKEND_SPEC.md`;
//! baseline: full corpus on RDNA4 via the portable subgroup-width-agnostic
//! reductions).
//!
//! Runs only with `--features vulkan`.
#![cfg(feature = "vulkan")]

use std::collections::BTreeMap;

use metaltile::{CodegenError, MetalTileError, VulkanDevice, core::dtype::DType};

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

/// Kernels expected to *generate + run* but mismatch the oracle on Vulkan,
/// with reasons. A failure NOT matching an entry is a regression and fails
/// the test.
const KNOWN_HARD: &[(&str, &str)] = &[
    // f32-only, 1.9x over its 1.5e-2 tol: the mel filterbank accumulates
    // hundreds of sin/cos twiddle products, and GLSL's transcendental
    // rounding drifts a few ULPs from the libm oracle per term. f16/bf16
    // variants (looser tols) and every other dtype pass on RDNA4.
    ("test_mel_spectrogram_magnitude [f32]", "transcendental rounding accumulation"),
];

fn known_hard(name: &str) -> bool { KNOWN_HARD.iter().any(|(k, _)| name.contains(k)) }

/// UNSUPPORTED is decided on the TYPED error, not message sniffing
/// (mirrors `cuda_kernel_corpus.rs`):
/// * `Codegen(UnsupportedOp)` — every codegen coverage gap (cooperative
///   MMA, multi-dim index, dtype gaps, ops not wired yet) is raised as
///   this variant by the GLSL/SPIR-V emitter.
/// * `DeviceCapability` — kernels the codegen *does* cover but the target
///   physically cannot run, surfaced before launch.
///
/// Anything else (shaderc compile failure, VkResult rejection, missing
/// buffer) on a kernel we claim to support stays a hard ERROR.
fn is_unsupported(e: &MetalTileError) -> bool {
    matches!(
        e,
        MetalTileError::Codegen(CodegenError::UnsupportedOp(_))
            | MetalTileError::DeviceCapability(_)
    )
}

#[test]
fn run_corpus_on_vulkan() {
    let Some(dev) = VulkanDevice::create().expect("Vulkan init") else {
        eprintln!("no Vulkan device — skipping");
        return;
    };
    eprintln!("Vulkan corpus: queue_family={}", dev.queue_family());

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

            if let Ok(want) = std::env::var("DUMP_VK")
                && t.name() == want
                && dt == DType::F32
            {
                use metaltile::codegen::{CodegenBackend, GlslGenerator};
                if let Ok(src) = GlslGenerator::new().with_local_size_3d(grid.tpg).generate(kernel)
                {
                    eprintln!("==== {} (Vulkan/GLSL) ====\n{src}\n==== end ====", t.name());
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

    eprintln!("\n=== Vulkan corpus result ===");
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
        for f in hard_failures.iter() {
            eprintln!("  ✗ {f}");
        }
    }
    // Mismatch surface by kernel-base name (strips trailing `_<dtype>`).
    let unique_mm_kernels: std::collections::BTreeSet<String> = hard_failures
        .iter()
        .filter(|f| f.starts_with("MISMATCH"))
        .map(|f| {
            let s = f.trim_start_matches("MISMATCH ");
            s.split(" [").next().unwrap_or("").to_string()
        })
        .collect();
    eprintln!("--- unique MISMATCH kernel-base count: {} ---", unique_mm_kernels.len());
    for k in &unique_mm_kernels {
        eprintln!("  · {k}");
    }

    // Pass floor: a regression that bucketed kernels as UNSUPPORTED (typed
    // `Codegen(UnsupportedOp)`) would otherwise shrink coverage silently.
    // The RDNA4 baseline passes the full corpus; 3500 leaves headroom for
    // device-cap variation, not for a broken emitter.
    assert!(pass >= 3500, "only {pass} kernels passed on Vulkan — emitter or pipeline regression");
    // The corpus is bit-accurate on RDNA4 (phase-3 baseline): any oracle
    // mismatch on a supported kernel is a regression, not noise.
    assert!(
        mismatch == 0,
        "{mismatch} Vulkan oracle mismatches on supported kernels — numerics regression"
    );
    // Hard errors (shaderc compile / VkResult failures) signal genuine
    // codegen bugs, not numerics. The RDNA4 baseline runs clean; the budget
    // leaves headroom for driver/device variation without letting a broad
    // regression through.
    let error_budget: u32 = 64;
    assert!(
        error <= error_budget,
        "Vulkan corpus produced {error} hard errors (budget={error_budget}) — codegen regression"
    );
}
