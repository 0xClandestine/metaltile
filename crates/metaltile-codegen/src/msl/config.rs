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
            tile_schedule: TileSchedule::default(),
        }
    }
}
