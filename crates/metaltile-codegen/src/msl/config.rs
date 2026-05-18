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
    /// Emit `async_copy` prefetch (requires Metal 3 / M2+).
    pub async_copy: bool,
    /// Target Apple GPU family level (7 = M1, 8 = M2, 9 = M3/M4, 10 = M5),
    /// when known. `None` leaves codegen in its conservative mode and is
    /// the default. Kernels that have family-specific fast paths read this
    /// to pick between variants — e.g. on `Some(10)` the SDPA path can
    /// route around the M5 Neural Accelerator's bf16-disabled fallback.
    pub apple_family: Option<u32>,
    pub tile_schedule: TileSchedule,
}

impl Default for MslConfig {
    fn default() -> Self {
        MslConfig {
            simd_size: 32,
            use_simd_matrix: false,
            debug_comments: false,
            native_bfloat: true,
            async_copy: false,
            apple_family: None,
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
