//! MSL generation configuration.

use crate::passes::tile_lowering::TileSchedule;

#[derive(Debug, Clone)]
pub struct MslConfig {
    pub simd_size: u32,
    /// Emit `simdgroup_multiply_accumulate` (requires Metal GPU family 7+ / M1+).
    /// On Apple9 (M3+) this uses dedicated matrix hardware; on Apple7/8 (M1/M2)
    /// it is emulated via FMA on the simdgroup.
    pub use_simd_matrix: bool,
    pub debug_comments: bool,
    /// Use native `bfloat` type (Metal 3.1+, M3+). When false, uses `bfloat16_t` struct.
    pub native_bfloat: bool,
    /// MFA-style raw upper-16-bit reinterpret for f32→bf16 casts
    /// (`as_type<bfloat2>(fp32)[1]`). Bypasses Metal's IEEE-compliant
    /// `__bf16_to_f32` builtin which is slow on M2 (gen-8 lacks the M3+
    /// tensor unit).
    ///
    /// **Trades numeric correctness for speed:** the reinterpret is a
    /// straight truncation of the lower 16 bits of fp32, NOT round-to-
    /// nearest-even. Drift is ≤ 1 ULP per cast (≈0.4% relative for bf16).
    /// Tolerable for SDPA-style kernels with heavy-tailed attention mass;
    /// fails tight-tolerance quality checks on kernels that store many
    /// small post-normalised values (e.g. `rms_norm`).
    ///
    /// Default is **off** — opt in per kernel/MslGenerator when the cast
    /// site can prove the 1 ULP drift is acceptable.
    pub bfloat_reinterpret_cast: bool,
    /// Emit `async_copy` prefetch (requires Metal 3 / M2+).
    pub async_copy: bool,
    /// Target Apple GPU family level (7 = M1, 8 = M2, 9 = M3/M4, 10 = M5),
    /// when known. `None` leaves codegen in its conservative mode and is
    /// the default. Kernels that have family-specific fast paths read this
    /// to pick between variants — e.g. on `Some(10)` the SDPA path can
    /// route around the M5 Neural Accelerator's bf16-disabled fallback.
    pub apple_family: Option<u32>,
    /// Expected threadgroup size at dispatch (in threads). When known, the
    /// codegen specializes paths that depend on `lsize` — most notably the
    /// Reduction-mode `Op::Reduce` emit, which collapses to a single
    /// `simd_*(value)` call when `expected_tpg <= simd_size` (one simdgroup,
    /// no second-level reduction needed) and emits the full two-level
    /// threadgroup path otherwise. `None` leaves codegen in the conservative
    /// two-level path that is correct at any TPG ≥ 32. Bench dispatch sets
    /// this from `ShapeSpec.tpg` so each (kernel × dtype × tpg-bucket)
    /// compiles to optimal MSL.
    ///
    /// The compiled-kernel cache in `run_generic` (`metaltile-std`) keys on
    /// a 1-bit bucket of this value: `None` and `Some(n > simd_size)` share
    /// the slow-path PSO slot; `Some(n <= simd_size)` gets its own. Two
    /// shapes that differ only in TPG-bucket therefore compile separately
    /// instead of colliding on one PSO.
    pub expected_tpg: Option<u32>,
    pub tile_schedule: TileSchedule,
}

impl Default for MslConfig {
    fn default() -> Self {
        MslConfig {
            simd_size: 32,
            use_simd_matrix: false,
            debug_comments: false,
            native_bfloat: true,
            bfloat_reinterpret_cast: false,
            async_copy: false,
            apple_family: None,
            expected_tpg: None,
            tile_schedule: TileSchedule::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_apple_family_is_none() {
        assert!(MslConfig::default().apple_family.is_none());
    }

    #[test]
    fn apple_family_round_trips() {
        let cfg = MslConfig { apple_family: Some(10), ..MslConfig::default() };
        assert_eq!(cfg.apple_family, Some(10));
        // Other fields retain their defaults.
        assert!(cfg.native_bfloat);
        assert!(!cfg.use_simd_matrix);
    }
}
