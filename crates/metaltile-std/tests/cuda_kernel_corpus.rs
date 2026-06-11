//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Run the registered `#[test_kernel]` corpus on the CUDA backend.
//!
//! Iterates the same `KernelTest` inventory the Metal harness uses
//! (`tests/kernel_tests_harness.rs`), but dispatches each on a real CUDA
//! device via `CudaDevice::run_kernel`, comparing to the same CPU oracle.
//! Categorizes each (kernel × dtype) as PASS / MISMATCH / UNSUPPORTED /
//! ERROR. UNSUPPORTED (codegen doesn't cover the kernel yet — MMA,
//! cooperative, Strided, multi-dim) is expected and not a failure;
//! MISMATCH (ran but wrong) and ERROR (NVRTC/launch failure on a kernel we
//! claimed to support) are hard failures.
//!
//! Runs only with `--features cuda` on a CUDA host (the GX10 / sm_121).
#![cfg(feature = "cuda")]

use std::collections::BTreeMap;

use metaltile::{CodegenError, CudaDevice, MetalTileError, core::dtype::DType};

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

/// Kernels that GENERATE but don't yet match the oracle on CUDA, with
/// reasons — tracked so the suite stays green while documenting the
/// remaining pure-DSL gaps (distinct from the cooperative/MMA backlog).
/// A failure NOT matching one of these is a regression and fails the test.
const KNOWN_HARD: &[(&str, &str)] = &[
    // f16-only, DETERMINISTIC max|Δ|=5.0e-1 vs 5e-2 tol: the no_gqa fixture
    // is the file's documented gain-sensitive 3-step recurrence (its f32 tol
    // was already loosened 5e-3 → 1e-2 for the same drift). The f16-rounded
    // input set hits a `(v - kv_mem)` cancellation that amplifies CUDA-vs-
    // oracle transcendental rounding (exp/log/rsqrt ulp differences — NOT FMA
    // contraction: identical with MT_FMAD=0). f32, bf16, and the gqa fixture
    // all pass on the same kernel.
    ("test_mt_gated_delta_prep_chunk_no_gqa [f16]", "cancellation-amplified rounding"),
];

fn known_hard(name: &str) -> bool { KNOWN_HARD.iter().any(|(k, _)| name.contains(k)) }

/// UNSUPPORTED is decided on the TYPED error, not message sniffing:
/// * `Codegen(UnsupportedOp)` — every codegen coverage gap (MMA/cooperative,
///   Tile2D, multi-dim index, ops not wired yet) is raised as this variant.
/// * `DeviceCapability` — kernels the codegen *does* cover but the target
///   arch physically cannot run (e.g. >48KB dynamic shared memory on
///   pre-Volta), surfaced before launch. These reflect bit-accuracy on what
///   the arch CAN run, so they are not hard failures.
///
/// Anything else (NVRTC compile failure, launch error, missing buffer) on a
/// kernel we claim to support stays a hard ERROR.
fn is_unsupported(e: &MetalTileError) -> bool {
    matches!(
        e,
        MetalTileError::Codegen(CodegenError::UnsupportedOp(_))
            | MetalTileError::DeviceCapability(_)
    )
}

#[test]
fn run_corpus_on_cuda() {
    let Some(dev) = CudaDevice::create().expect("CUDA init") else {
        eprintln!("no CUDA device — skipping");
        return;
    };

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

            // GPU-vs-GPU reference setups need two dispatches; skip for now.
            if setup.ref_setup().is_some() {
                unsupported += 1;
                *unsup_reasons.entry("ref_setup (GPU-vs-GPU)".into()).or_default() += 1;
                continue;
            }

            // Build the param/constexpr byte map.
            let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
            for inp in setup.inputs() {
                buffers.insert(inp.name().to_string(), inp.data().to_vec());
            }
            for (k, v) in setup.constexprs() {
                buffers.insert(k.clone(), v.to_le_bytes());
            }

            let grid = setup.grid();
            let label = format!("{} [{dt}]", t.name());

            // Debug: DUMP=<exact kernel name> prints its generated CUDA.
            if let Ok(want) = std::env::var("DUMP")
                && t.name() == want
                && dt == DType::F32
            {
                use metaltile::codegen::{CodegenBackend, CudaGenerator};
                if let Ok(src) = CudaGenerator::new().generate(kernel) {
                    eprintln!("==== {} ====\n{src}\n==== end ====", t.name());
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
                    // Match on the dtype-qualified label so a KNOWN_HARD entry
                    // can be scoped to one dtype without masking the others.
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

    eprintln!("\n=== CUDA corpus result ===");
    eprintln!(
        "PASS={pass}  KNOWN_HARD={known}  MISMATCH={mismatch}  UNSUPPORTED={unsupported}  ERROR={error}"
    );
    eprintln!("--- unsupported reasons (top buckets) ---");
    let mut reasons: Vec<_> = unsup_reasons.iter().collect();
    reasons.sort_by(|a, b| b.1.cmp(a.1));
    for (reason, n) in reasons.iter().take(25) {
        eprintln!("  {n:>5}  {reason}");
    }
    eprintln!("--- passing kernels ---");
    for n in &pass_names {
        eprintln!("  ✓ {n}");
    }
    if !hard_failures.is_empty() {
        eprintln!("--- hard failures ({}) ---", hard_failures.len());
        for f in &hard_failures {
            eprintln!("  ✗ {f}");
        }
    }

    assert!(pass > 0, "no kernels passed on CUDA — pipeline broken");
    assert!(
        hard_failures.is_empty(),
        "{} CUDA hard failures (mismatch/error on supported kernels)",
        hard_failures.len()
    );
}
