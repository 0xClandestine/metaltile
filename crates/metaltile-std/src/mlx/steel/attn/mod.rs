pub mod steel_attention;
pub mod steel_attention_mma;
pub mod steel_attention_mma_bf16;
pub mod steel_attention_nax;

use metaltile_core::{dtype::DType, ir::Kernel};

/// Auto-select the best SDPA-prefill MMA kernel for the given dtype + GPU
/// family. Returns the kernel IR ready to dispatch.
///
/// Heuristic:
/// - bf16 + Apple gen-8 (M2): use `mt_sdpa_prefill_mma_bf16` — single-Q
///   dd-loop variant; reduces simdgroup-matrix frag count 22 → 7, freeing
///   register-file room for M2's emulated bf16-MMA path. +14pts vs the
///   16-Q-preload sibling at bf16 on M2.
/// - bf16 + Apple gen-9+ (M3+): use `mt_sdpa_prefill_mma` — both variants
///   tie on bf16 on M5 (native bf16 MMA, no emulation tax), but the
///   sibling wins f32/f16 by 1pt on idle so we stick with it.
/// - f32 / f16 (any family): use `mt_sdpa_prefill_mma`.
///
/// `family` should be the `Context::chip_family()` value (`None` means
/// "unknown / non-Apple-Silicon target" — fall back to the sibling kernel
/// which has the broadest perf profile).
///
/// Composite numbers via this selector (median of 5 idle runs):
///
/// | Machine | dtype | Selected | MT% MLX |
/// |---------|-------|----------|--------:|
/// | M2 mini | f32   | mma       | 127% |
/// | M2 mini | f16   | mma       |  99% |
/// | M2 mini | bf16  | mma_bf16  |  99% |
/// | M5 Max  | f32   | mma       | 116% |
/// | M5 Max  | f16   | mma       | 107% |
/// | M5 Max  | bf16  | mma       | 107% |
///
/// M2 mini f16 closed the 93% gap to 99% via the `kv_ld = 136` bank-skew
/// re-tune (was 132 — tuned for bf16 8-byte loads; f16's 4-byte loads
/// prefer +8 pad over +4 to dodge a different bank pattern). M5 Max
/// gained +1-2pts across all dtypes as a side benefit; no regression.
///
/// # Untested hardware
///
/// Heuristic was validated on M2 mini (Apple8/gen-8) and M5 Max
/// (Apple10/gen-17+). The other Apple GPU families are inferred:
///
/// - **M1 (Apple7/gen-7)**: same architectural class as M2 (no native bf16
///   MMA, emulates via fp32). Selector routes bf16 → `mma_bf16` here too,
///   which *should* be the right call but is not measured. If perf is
///   off, suspect the kv_ld=132 bank-skew pad (M1 has different TG memory
///   bank geometry) or barrier density.
/// - **M3 / M4 (Apple9/gen-17)**: native bf16 MMA hardware. Selector
///   routes bf16 → `mma` (16-Q-preload sibling), inferred by analogy to
///   M5. Worth confirming `mma` wins on these too; if not, the `family
///   ≤ 8` cutoff should be tightened to `family ≤ 7`.
/// - **A17/A18 mobile GPUs** (gen-17, gen-18): same family as M3/M4 on
///   paper but TG memory limits and L1 sizes differ; unmeasured.
///
/// Track results in PR notes or a follow-up; nudge the cutoff if M1
/// bf16 regresses or if M3/M4 bf16 prefers `mma_bf16`.
pub fn sdpa_prefill_mma_for(dtype: DType, family: Option<u32>) -> Kernel {
    let is_pre_m3_bf16 = dtype == DType::BF16 && matches!(family, Some(f) if f <= 8);
    let mut k = if is_pre_m3_bf16 {
        steel_attention_mma_bf16::mt_sdpa_prefill_mma_bf16::kernel_ir_for(dtype)
    } else {
        steel_attention_mma::mt_sdpa_prefill_mma::kernel_ir_for(dtype)
    };
    // Opt in to the MFA-style f32→bf16 reinterpret cast. The MMA
    // kernels accumulate in f32 throughout and emit a single
    // narrowing cast at output store; the ≤1 ULP truncation drift is
    // absorbed by SDPA's heavy-tailed attention mass and stays
    // inside the `tol=2e-2` bench envelope. The codegen default is
    // off (rms_norm / arange would fail their tighter tolerances);
    // see the `Kernel::bfloat_reinterpret_cast` field doc.
    k.bfloat_reinterpret_cast = true;
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_select_picks_bf16_variant_for_m2_bf16() {
        let k = sdpa_prefill_mma_for(DType::BF16, Some(8));
        assert_eq!(k.name, "mt_sdpa_prefill_mma_bf16");
    }

    #[test]
    fn auto_select_picks_sibling_for_m5_bf16() {
        let k = sdpa_prefill_mma_for(DType::BF16, Some(10));
        assert_eq!(k.name, "mt_sdpa_prefill_mma");
    }

    #[test]
    fn auto_select_picks_sibling_for_f32_and_f16_on_any_family() {
        for family in [None, Some(7), Some(8), Some(9), Some(10)] {
            for dt in [DType::F32, DType::F16] {
                let k = sdpa_prefill_mma_for(dt, family);
                assert_eq!(k.name, "mt_sdpa_prefill_mma", "dt={dt:?} family={family:?}");
            }
        }
    }

    #[test]
    fn auto_select_opts_in_to_bfloat_reinterpret_cast() {
        // The MMA prefill kernels accumulate in f32 and only narrow
        // at the output store; the MFA-style reinterpret-cast
        // truncation is bench-tolerable for them. Codegen default is
        // off (rms_norm / arange need round-to-nearest), so the
        // selector explicitly opts in. Every selected kernel must
        // have the flag set regardless of dtype × family.
        for family in [None, Some(7), Some(8), Some(9), Some(10)] {
            for dt in [DType::F32, DType::F16, DType::BF16] {
                let k = sdpa_prefill_mma_for(dt, family);
                assert!(
                    k.bfloat_reinterpret_cast,
                    "kernel-side bfloat_reinterpret_cast must be set for dt={dt:?} family={family:?}",
                );
            }
        }
    }

    #[test]
    fn auto_select_falls_back_to_sibling_when_family_unknown() {
        // Non-Apple-Silicon hosts (or unidentified GPUs) get the sibling
        // kernel — broadest perf profile across all dtypes.
        let k = sdpa_prefill_mma_for(DType::BF16, None);
        assert_eq!(k.name, "mt_sdpa_prefill_mma");
    }
}
