//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Per-device peak hardware ceilings for roofline / %-of-peak reporting.
//!
//! Metal exposes no API for a GPU's peak FLOP/s (and none at all for the M5
//! Neural-Accelerator matmul throughput), so the ceilings are a hand-maintained
//! table keyed by `MTLDevice.name()`. An **unknown device returns `None`** — the
//! roofline columns simply stay blank, so a new chip (or CI's
//! `Apple Paravirtual device`) never breaks a bench run.
//!
//! Numbers and where they come from:
//! - Bandwidth + FP32 TFLOP/s are the verified figures in
//!   `sam/planning/performance-notes/gpu-model-specs.md`, covering the full
//!   Apple line: base / Pro / Max for M1–M5, plus Ultra for M1/M2/M3 (no M4
//!   Ultra shipped; M5 Ultra is still an estimate, not yet seeded). Where that
//!   table lumps variants into a range, we split per chip and note it inline
//!   (M2/M3 Max 15.8–17.5 → M2 Max 15.8, M3 Max 17.5; base M1/M2/M3 2.6–5.1 →
//!   M1 2.6, M2 3.6, M3 4.1).
//! - **FP16 SIMD peak is 2× FP32** on Apple GPUs (half-precision runs at double
//!   rate on the SIMD ALUs), so `peak_f16 = 2 × peak_f32`. The
//!   `gpu-model-specs.md` table reports FP32 only.
//! - The M5 **Neural Accelerator** FP16 ceilings (`na_f16_tflops`) come from a
//!   separate per-core spec (~1.75 TFLOPS/core; ~70 TFLOPS on the 40-core M5
//!   Max, ~17.5 on the ≤10-core base M5) in `docs/BENCH_METRICS_SPEC.md`
//!   Appendix C — not the `gpu-model-specs.md` table. The M5 Pro NA ceiling has
//!   no confirmed per-core count yet, so it scores against the 2× SIMD pipe
//!   (`na_f16_tflops: None`) until its GPU-core count is known. On M5 the NA
//!   accelerates **FP16 only** — not bf16, not fp8/fp4 — so bf16 matmuls score
//!   against the 2× SIMD pipe.

use metaltile_core::dtype::DType;

/// Peak hardware ceilings for one GPU, used to turn measured GB/s and GFLOP/s
/// into %-of-peak (roofline) figures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeviceSpecs {
    /// Peak DRAM (unified-memory) bandwidth in GB/s.
    pub peak_bw_gbps: f64,
    /// Peak FP32 compute on the SIMD pipe, in TFLOP/s.
    pub peak_f32_tflops: f64,
    /// Peak FP16 compute on the SIMD pipe, in TFLOP/s (≈ FP32 on Apple GPUs).
    pub peak_f16_tflops: f64,
    /// Peak FP16 matmul on the M5+ per-core Neural Accelerator, in TFLOP/s.
    /// `None` on pre-M5 chips that have no NA.
    pub na_f16_tflops: Option<f64>,
    /// Peak INT8 matmul on the M5+ Neural Accelerator, in TOP/s. `None` where no
    /// reliable public figure exists (the NA accelerates INT8 but lags FP16).
    pub na_int8_tops: Option<f64>,
}

impl DeviceSpecs {
    /// The compute ceiling (TFLOP/s) to divide a kernel of dtype `dt` by:
    /// FP16 matmuls use the Neural-Accelerator path when present (M5+), else the
    /// SIMD pipe. bf16 stays on the SIMD pipe even on M5 (the first-gen NA does
    /// not accelerate bf16). FP32 always uses the SIMD pipe.
    pub fn peak_tflops_for(&self, dt: DType) -> f64 {
        match dt {
            DType::F32 => self.peak_f32_tflops,
            DType::F16 => self.na_f16_tflops.unwrap_or(self.peak_f16_tflops),
            // bf16 (NA-unaccelerated on M5) and any other dtype: SIMD f16 pipe.
            _ => self.peak_f16_tflops,
        }
    }
}

/// Look up peak specs for a Metal device name (e.g. `"Apple M1 Max"`).
/// Returns `None` for any device not in the table — callers leave the roofline
/// columns blank rather than failing.
pub fn lookup(device_name: &str) -> Option<DeviceSpecs> {
    let n = device_name.to_ascii_lowercase();
    // Helper: SIMD-only device (no Neural Accelerator). `f32` is the verified
    // FP32 TFLOP/s; the Apple-GPU SIMD pipe runs half-precision at 2× FP32, so
    // `peak_f16 = 2 × f32`. Used for every pre-M5 chip and the (NA-spec-less)
    // M5 Pro.
    let simd = |bw: f64, f32: f64| DeviceSpecs {
        peak_bw_gbps: bw,
        peak_f32_tflops: f32,
        peak_f16_tflops: f32 * 2.0,
        na_f16_tflops: None,
        na_int8_tops: None,
    };
    // Helper for M5-class chips with a Neural Accelerator: SIMD f16 = 2× f32,
    // plus the NA FP16 matmul ceiling.
    let m5 = |bw: f64, f32: f64, na: f64| DeviceSpecs {
        peak_bw_gbps: bw,
        peak_f32_tflops: f32,
        peak_f16_tflops: f32 * 2.0,
        na_f16_tflops: Some(na),
        na_int8_tops: None,
    };
    // FP32 + bandwidth are the verified `gpu-model-specs.md` figures. Match
    // most-specific first within each generation: "m5 max" / "m5 pro" must beat
    // the bare "m5" substring (same for m1–m4). bf16 always scores against the
    // 2× SIMD pipe (peak_f16); the NA path is FP16-only.
    if n.contains("m5 max") {
        Some(m5(614.0, 24.0, 70.0)) // 40-core M5 Max; NA ~70 TFLOPS FP16.
    } else if n.contains("m5 pro") {
        // M5 Pro (est. ~12.0 FP32, 307 GB/s). NA per-core count unconfirmed →
        // SIMD-only scoring (na_f16: None) until GPU cores are known.
        Some(simd(307.0, 12.0))
    } else if n.contains("m5") {
        Some(m5(153.6, 5.8, 17.5)) // base M5 (≤10 cores); NA ~17.5 TFLOPS FP16.
    } else if n.contains("m4 max") {
        Some(simd(546.0, 21.1))
    } else if n.contains("m4 pro") {
        Some(simd(273.0, 10.4))
    } else if n.contains("m4") {
        Some(simd(120.0, 4.4))
    } else if n.contains("m3 max") {
        Some(simd(400.0, 17.5)) // M2/M3 Max range upper bound.
    } else if n.contains("m3 ultra") {
        Some(simd(800.0, 31.8))
    } else if n.contains("m3 pro") {
        Some(simd(150.0, 6.2))
    } else if n.contains("m3") {
        Some(simd(100.0, 4.1)) // base M1/M2/M3 range: M3.
    } else if n.contains("m2 max") {
        Some(simd(400.0, 15.8)) // M2/M3 Max range lower bound.
    } else if n.contains("m2 ultra") {
        Some(simd(800.0, 27.2))
    } else if n.contains("m2 pro") {
        Some(simd(200.0, 6.8))
    } else if n.contains("m2") {
        Some(simd(100.0, 3.6)) // base M1/M2/M3 range: M2.
    } else if n.contains("m1 max") {
        Some(simd(400.0, 10.4))
    } else if n.contains("m1 ultra") {
        Some(simd(800.0, 21.2))
    } else if n.contains("m1 pro") {
        Some(simd(200.0, 5.2))
    } else if n.contains("m1") {
        Some(simd(68.0, 2.6)) // base M1/M2/M3 range: M1.
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_devices_resolve_most_specific_first() {
        // Within a generation, "max"/"pro" must not be swallowed by the bare arm.
        let m5_max = lookup("Apple M5 Max").unwrap();
        assert_eq!(m5_max.peak_bw_gbps, 614.0);
        assert_eq!(m5_max.na_f16_tflops, Some(70.0));
        let m5_pro = lookup("Apple M5 Pro").unwrap();
        assert_eq!(m5_pro.peak_bw_gbps, 307.0);
        assert!(m5_pro.na_f16_tflops.is_none()); // NA core count TBD → SIMD-only.
        let m5 = lookup("Apple M5").unwrap();
        assert_eq!(m5.peak_bw_gbps, 153.6);
        assert_eq!(m5.na_f16_tflops, Some(17.5));

        // Pre-M5: base / Pro / Max all resolve distinctly, no NA.
        let m4_max = lookup("Apple M4 Max").unwrap();
        assert_eq!((m4_max.peak_bw_gbps, m4_max.peak_f32_tflops), (546.0, 21.1));
        let m4_pro = lookup("Apple M4 Pro").unwrap();
        assert_eq!((m4_pro.peak_bw_gbps, m4_pro.peak_f32_tflops), (273.0, 10.4));
        let m4 = lookup("Apple M4").unwrap();
        assert_eq!((m4.peak_bw_gbps, m4.peak_f32_tflops), (120.0, 4.4));
        assert!(m4.na_f16_tflops.is_none());

        // Bare base chips resolve (not swallowed by, nor swallowing, max/pro).
        assert_eq!(lookup("Apple M1").unwrap().peak_f32_tflops, 2.6);
        assert_eq!(lookup("Apple M2").unwrap().peak_f32_tflops, 3.6);
        assert_eq!(lookup("Apple M3").unwrap().peak_f32_tflops, 4.1);
        assert_eq!(lookup("Apple M1 Pro").unwrap().peak_f32_tflops, 5.2);
        assert_eq!(lookup("Apple M3 Max").unwrap().peak_f32_tflops, 17.5);

        // Ultra tier (M1/M2/M3 only): 800 GB/s, no NA; distinct from the bare arm.
        let m1u = lookup("Apple M1 Ultra").unwrap();
        assert_eq!((m1u.peak_bw_gbps, m1u.peak_f32_tflops), (800.0, 21.2));
        assert!(m1u.na_f16_tflops.is_none());
        assert_eq!(lookup("Apple M2 Ultra").unwrap().peak_f32_tflops, 27.2);
        assert_eq!(lookup("Apple M3 Ultra").unwrap().peak_f32_tflops, 31.8);
        // f16 = 2× f32 holds for Ultra too.
        assert_eq!(lookup("Apple M3 Ultra").unwrap().peak_f16_tflops, 63.6);
    }

    #[test]
    fn unknown_device_returns_none() {
        // CI's virtualized GPU (and any unseeded chip) → blank roofline, no panic.
        assert!(lookup("Apple Paravirtual device").is_none());
        assert!(lookup("Some Future GPU").is_none());
    }

    #[test]
    fn fp16_simd_peak_is_double_fp32() {
        // Apple-GPU half-precision runs at 2× FP32 on the SIMD pipe.
        for name in ["Apple M1 Max", "Apple M4 Pro", "Apple M2", "Apple M5 Pro"] {
            let s = lookup(name).unwrap();
            assert_eq!(s.peak_f16_tflops, s.peak_f32_tflops * 2.0, "{name}");
        }
    }

    #[test]
    fn peak_tflops_picks_na_for_f16_simd_for_bf16_and_f32() {
        let m5 = lookup("Apple M5 Max").unwrap();
        // FP16 matmul rides the Neural Accelerator.
        assert_eq!(m5.peak_tflops_for(DType::F16), 70.0);
        // bf16 is NOT NA-accelerated on first-gen NA → 2× SIMD pipe (2 × 24.0).
        assert_eq!(m5.peak_tflops_for(DType::BF16), 48.0);
        // FP32 → SIMD pipe (verified gpu-model-specs figure).
        assert_eq!(m5.peak_tflops_for(DType::F32), 24.0);
        // Pre-M5: f16 falls back to the 2× SIMD pipe (no NA): 2 × 10.4.
        let m1 = lookup("Apple M1 Max").unwrap();
        assert_eq!(m1.peak_tflops_for(DType::F16), 20.8);
        assert_eq!(m1.peak_tflops_for(DType::F32), 10.4);
    }
}
