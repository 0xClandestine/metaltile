//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Per-device peak hardware ceilings for roofline / %-of-peak reporting.
//!
//! Metal exposes no API for a GPU's peak FLOP/s (nor for the matrix-engine
//! throughput), so the ceilings are a hand-maintained table keyed by
//! `MTLDevice.name()`. An **unknown device returns `None`** — the roofline
//! columns simply stay blank, so a new chip (or CI's `Apple Paravirtual device`)
//! never breaks a bench run.
//!
//! ## Source — the "Apple Silicon GPU Vector, Matrix, and ANE Compute" table
//!
//! Published per-chip specs for the whole Apple line: base / Pro / Max for
//! M1–M5, plus Ultra for M1 / M2 / M3 / M5. Two caveats baked in:
//! **no M4 Ultra ever shipped** (so there is no M4 Ultra row), and the
//! **M5 Ultra is a projection** (it follows the M1→M3 pattern of Ultra = 2× Max
//! across vector, matrix, and bandwidth).
//!
//! Field ↔ table column:
//! - **`peak_f32_tflops`** ← "Peak FP32 (Standard GPU Vector)".
//! - **`peak_f16_tflops`** — not tabulated; Apple GPUs run half-precision at
//!   **2× FP32** on the SIMD ALUs, so `peak_f16 = 2 × peak_f32`.
//! - **`peak_bw_gbps`** ← "Peak Memory Bandwidth"; where the spec lists a range
//!   (M3 Max 300–400, M4 Max 410–546, M5 Max 460–614) we take the **upper** bound.
//! - **`na_f16_tflops`** ← "Peak Matrix Compute Block" **when it is embedded in
//!   the GPU cores** — i.e. the **GPU Neural Accelerator**, which is **M5+ only**
//!   (M5: base 16.8 / Pro 33.2 / Max 66.4 / Ultra 132.8 TFLOPS FP16). This is the
//!   ceiling the NAX / MPP `mpp::tensor_ops::matmul2d` path scores against. `None`
//!   on pre-M5 chips: their GPUs have no embedded matrix engine, so a cooperative
//!   matmul rides the 2× SIMD f16 pipe.
//! - **`ane_tops`** ← "Peak Matrix Compute Block" **when it lives in a Standalone
//!   / Dual ANE Block** — i.e. the separate **Apple Neural Engine** (M1–M4:
//!   11 / 15.8 / 18 / 38 TOPS, Ultra parts double it for their dual ANE). Not yet
//!   consumed by the GPU roofline; recorded so the upcoming ANE kernels/benches
//!   have a ceiling. On M5 the table reports the GPU-embedded matrix instead of
//!   the chip's separate ANE, so M5 `ane_tops` is `None`.
//!
//! On M5 the Neural Accelerator accelerates **FP16 matmul** only (not bf16 / fp8
//! / fp4), so bf16 matmuls score against the 2× SIMD pipe.

use metaltile_core::dtype::DType;

/// Peak hardware ceilings for one GPU, used to turn measured GB/s and GFLOP/s
/// into %-of-peak (roofline) figures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeviceSpecs {
    /// Peak DRAM (unified-memory) bandwidth in GB/s (upper bound of any range).
    pub peak_bw_gbps: f64,
    /// Peak FP32 compute on the GPU SIMD pipe ("GPU Vector"), in TFLOP/s.
    pub peak_f32_tflops: f64,
    /// Peak FP16 compute on the GPU SIMD pipe, in TFLOP/s (≈ 2× FP32 on Apple GPUs).
    pub peak_f16_tflops: f64,
    /// Peak FP16 matmul on the **GPU Neural Accelerator** — the M5+ matrix engine
    /// embedded in the GPU cores, in TFLOP/s. `None` on pre-M5 chips, whose GPUs
    /// have no embedded matrix engine.
    pub na_f16_tflops: Option<f64>,
    /// Peak INT8 matmul on the GPU Neural Accelerator, in TOP/s. `None` where no
    /// reliable public figure exists (the NA accelerates INT8 but lags FP16).
    pub na_int8_tops: Option<f64>,
    /// Peak throughput of the **Apple Neural Engine** (the separate NPU), in
    /// TOP/s — the "Matrix Compute Block" on M1–M4 (Standalone / Dual ANE Block).
    /// Not yet consumed by the GPU roofline; recorded for the upcoming ANE
    /// kernels/benches. `None` on M5 (the reference table reports the chip's
    /// GPU-embedded matrix rather than its separate ANE).
    pub ane_tops: Option<f64>,
}

impl DeviceSpecs {
    /// The compute ceiling (TFLOP/s) to divide a kernel of dtype `dt` by:
    /// FP16 matmuls use the GPU Neural-Accelerator path when present (M5+), else
    /// the SIMD pipe. bf16 stays on the SIMD pipe even on M5 (the first-gen NA
    /// does not accelerate bf16). FP32 always uses the SIMD pipe.
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
    // Pre-M5 chip: GPU SIMD pipe only (no GPU-embedded matrix engine), plus a
    // standalone Apple Neural Engine. `f32` = FP32 GPU-vector peak; f16 runs at
    // 2× on the SIMD ALUs; `ane` = the ANE's "Matrix Compute Block" TOPS.
    let pre_m5 = |bw: f64, f32: f64, ane: f64| DeviceSpecs {
        peak_bw_gbps: bw,
        peak_f32_tflops: f32,
        peak_f16_tflops: f32 * 2.0,
        na_f16_tflops: None,
        na_int8_tops: None,
        ane_tops: Some(ane),
    };
    // M5-class chip: the matrix engine is embedded in the GPU cores. `na` = its
    // FP16 matmul ceiling ("Matrix Compute Block" TFLOPS); the SIMD f16 pipe is
    // still 2× f32. The table reports this GPU matrix for M5 rather than the
    // separate ANE, so `ane_tops` is None.
    let m5 = |bw: f64, f32: f64, na: f64| DeviceSpecs {
        peak_bw_gbps: bw,
        peak_f32_tflops: f32,
        peak_f16_tflops: f32 * 2.0,
        na_f16_tflops: Some(na),
        na_int8_tops: None,
        ane_tops: None,
    };
    // Match most-specific first within each generation: "m5 max" / "m5 pro" /
    // "m5 ultra" must beat the bare "m5" substring (same for m1–m4). bf16 always
    // scores against the 2× SIMD pipe (peak_f16); the NA path is FP16-only.
    if n.contains("m5 max") {
        Some(m5(614.0, 16.6, 66.4)) // M5 Max; NA 66.4 TFLOPS FP16 (BW 460–614 → 614).
    } else if n.contains("m5 pro") {
        Some(m5(307.0, 8.3, 33.2)) // M5 Pro; NA 33.2 TFLOPS FP16.
    } else if n.contains("m5 ultra") {
        Some(m5(1228.0, 33.2, 132.8)) // M5 Ultra (projected 2× Max); NA 132.8 TFLOPS FP16.
    } else if n.contains("m5") {
        Some(m5(153.6, 4.2, 16.8)) // base M5; NA 16.8 TFLOPS FP16.
    } else if n.contains("m4 max") {
        Some(pre_m5(546.0, 17.2, 38.0)) // M4 Max; ANE 38 TOPS (BW 410–546 → 546). No M4 Ultra.
    } else if n.contains("m4 pro") {
        Some(pre_m5(273.0, 8.6, 38.0)) // M4 Pro; ANE 38 TOPS.
    } else if n.contains("m4") {
        Some(pre_m5(120.0, 4.3, 38.0)) // base M4; ANE 38 TOPS.
    } else if n.contains("m3 max") {
        Some(pre_m5(400.0, 12.8, 18.0)) // M3 Max; ANE 18 TOPS (BW 300–400 → 400).
    } else if n.contains("m3 ultra") {
        Some(pre_m5(800.0, 28.4, 36.0)) // M3 Ultra (Dual ANE); ANE 36 TOPS.
    } else if n.contains("m3 pro") {
        Some(pre_m5(150.0, 7.1, 18.0)) // M3 Pro; ANE 18 TOPS.
    } else if n.contains("m3") {
        Some(pre_m5(100.0, 3.5, 18.0)) // base M3; ANE 18 TOPS.
    } else if n.contains("m2 max") {
        Some(pre_m5(400.0, 13.6, 15.8)) // M2 Max; ANE 15.8 TOPS.
    } else if n.contains("m2 ultra") {
        Some(pre_m5(800.0, 27.2, 31.6)) // M2 Ultra (Dual ANE); ANE 31.6 TOPS.
    } else if n.contains("m2 pro") {
        Some(pre_m5(200.0, 6.8, 15.8)) // M2 Pro; ANE 15.8 TOPS.
    } else if n.contains("m2") {
        Some(pre_m5(100.0, 3.6, 15.8)) // base M2; ANE 15.8 TOPS.
    } else if n.contains("m1 max") {
        Some(pre_m5(400.0, 10.6, 11.0)) // M1 Max; ANE 11 TOPS.
    } else if n.contains("m1 ultra") {
        Some(pre_m5(800.0, 21.2, 22.0)) // M1 Ultra (Dual ANE); ANE 22 TOPS.
    } else if n.contains("m1 pro") {
        Some(pre_m5(200.0, 5.3, 11.0)) // M1 Pro; ANE 11 TOPS.
    } else if n.contains("m1") {
        Some(pre_m5(68.0, 2.6, 11.0)) // base M1; ANE 11 TOPS.
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_devices_resolve_most_specific_first() {
        // Within a generation, "max"/"pro"/"ultra" must not be swallowed by the
        // bare arm. M5-class chips carry the GPU Neural Accelerator (na_f16).
        let m5_max = lookup("Apple M5 Max").unwrap();
        assert_eq!((m5_max.peak_bw_gbps, m5_max.peak_f32_tflops), (614.0, 16.6));
        assert_eq!(m5_max.na_f16_tflops, Some(66.4));
        assert!(m5_max.ane_tops.is_none()); // M5 matrix is GPU-embedded, not the ANE.
        let m5_pro = lookup("Apple M5 Pro").unwrap();
        assert_eq!((m5_pro.peak_bw_gbps, m5_pro.na_f16_tflops), (307.0, Some(33.2)));
        let m5 = lookup("Apple M5").unwrap();
        assert_eq!((m5.peak_bw_gbps, m5.na_f16_tflops), (153.6, Some(16.8)));
        let m5_ultra = lookup("Apple M5 Ultra").unwrap();
        assert_eq!((m5_ultra.peak_bw_gbps, m5_ultra.peak_f32_tflops), (1228.0, 33.2));
        assert_eq!(m5_ultra.na_f16_tflops, Some(132.8));

        // Pre-M5: no GPU NA (na_f16 None), but a standalone ANE (ane_tops Some).
        let m4_max = lookup("Apple M4 Max").unwrap();
        assert_eq!((m4_max.peak_bw_gbps, m4_max.peak_f32_tflops), (546.0, 17.2));
        assert!(m4_max.na_f16_tflops.is_none());
        assert_eq!(m4_max.ane_tops, Some(38.0));
        assert_eq!(lookup("Apple M4 Pro").unwrap().peak_f32_tflops, 8.6);
        assert_eq!(lookup("Apple M4").unwrap().peak_f32_tflops, 4.3);

        // Bare base chips resolve (not swallowed by, nor swallowing, max/pro).
        assert_eq!(lookup("Apple M1").unwrap().peak_f32_tflops, 2.6);
        assert_eq!(lookup("Apple M2").unwrap().peak_f32_tflops, 3.6);
        assert_eq!(lookup("Apple M3").unwrap().peak_f32_tflops, 3.5);
        assert_eq!(lookup("Apple M1 Pro").unwrap().peak_f32_tflops, 5.3);
        assert_eq!(lookup("Apple M3 Max").unwrap().peak_f32_tflops, 12.8);

        // Ultra tier (M1/M2/M3): 800 GB/s, dual-ANE TOPS, no GPU NA.
        let m1u = lookup("Apple M1 Ultra").unwrap();
        assert_eq!(
            (m1u.peak_bw_gbps, m1u.peak_f32_tflops, m1u.ane_tops),
            (800.0, 21.2, Some(22.0))
        );
        assert!(m1u.na_f16_tflops.is_none());
        assert_eq!(lookup("Apple M2 Ultra").unwrap().peak_f32_tflops, 27.2);
        assert_eq!(lookup("Apple M3 Ultra").unwrap().peak_f32_tflops, 28.4);
        assert_eq!(lookup("Apple M3 Ultra").unwrap().ane_tops, Some(36.0));
        // f16 = 2× f32 holds for Ultra too (2 × 28.4).
        assert_eq!(lookup("Apple M3 Ultra").unwrap().peak_f16_tflops, 56.8);
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
        // FP16 matmul rides the GPU Neural Accelerator.
        assert_eq!(m5.peak_tflops_for(DType::F16), 66.4);
        // bf16 is NOT NA-accelerated on first-gen NA → 2× SIMD pipe (2 × 16.6).
        assert_eq!(m5.peak_tflops_for(DType::BF16), 33.2);
        // FP32 → SIMD pipe.
        assert_eq!(m5.peak_tflops_for(DType::F32), 16.6);
        // Pre-M5: f16 falls back to the 2× SIMD pipe (no GPU NA): 2 × 10.6.
        let m1 = lookup("Apple M1 Max").unwrap();
        assert_eq!(m1.peak_tflops_for(DType::F16), 21.2);
        assert_eq!(m1.peak_tflops_for(DType::F32), 10.6);
    }

    #[test]
    fn ane_tops_recorded_pre_m5_absent_on_m5() {
        // M1–M4 expose the standalone-ANE "Matrix Compute Block" TOPS.
        assert_eq!(lookup("Apple M1").unwrap().ane_tops, Some(11.0));
        assert_eq!(lookup("Apple M2 Max").unwrap().ane_tops, Some(15.8));
        assert_eq!(lookup("Apple M3 Pro").unwrap().ane_tops, Some(18.0));
        assert_eq!(lookup("Apple M4").unwrap().ane_tops, Some(38.0));
        // M5's matrix block is GPU-embedded (na_f16); its separate ANE isn't tabulated.
        assert!(lookup("Apple M5 Max").unwrap().ane_tops.is_none());
    }
}
