//! Static, GPU-free cost model + TuneConfig→ScheduleConfig adapter.

use metaltile::{MetalTileError, autotune::TuneConfig};
use metaltile_codegen::passes::{
    self,
    Pass,
    occupancy::{self, Bottleneck},
    schedule::ScheduleConfig,
};
use metaltile_core::ir::KernelMode;

/// Static, GPU-free cost: lower occupancy → higher cost. Stable and
/// deterministic; useful as a fallback when --measure isn't requested
/// (or when the candidate config rejects the kernel at compile time).
pub(crate) fn static_cost(
    kernel_template: &metaltile_core::ir::Kernel,
    mode: KernelMode,
    cfg: &TuneConfig,
) -> Result<f64, MetalTileError> {
    let mut k = kernel_template.clone();
    k.mode = mode;
    passes::run_passes(&mut k, &passes::standard_pipeline())
        .map_err(|e| MetalTileError::Autotune(format!("pipeline failed: {e}")))?;
    let sched: ScheduleConfig = tune_to_schedule(cfg);
    passes::schedule::SchedulePass::new(sched)
        .run(&mut k)
        .map_err(|e| MetalTileError::Autotune(format!("schedule failed: {e}")))?;

    let est = occupancy::estimate_occupancy(&k, cfg.threads.0, None);
    let mut us = 100.0 - est.occupancy_pct;
    if matches!(est.bottleneck, Bottleneck::RegisterLimited) {
        us += 20.0;
    }
    Ok(us)
}

pub(crate) fn tune_to_schedule(cfg: &TuneConfig) -> ScheduleConfig {
    let tile = if cfg.tile_dims.len() == 3 {
        (cfg.tile_dims[0] as u32, cfg.tile_dims[1] as u32, cfg.tile_dims[2] as u32)
    } else {
        (32, 32, 16)
    };
    ScheduleConfig {
        threads_per_threadgroup: cfg.threads,
        threadgroups_per_grid: (1, 1, 1),
        tile_dims: tile,
        simd_size: 32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tune_to_schedule_picks_up_3d_tile_dims() {
        let cfg = TuneConfig {
            tile_dims: vec![64, 32, 16],
            threads: (256, 1, 1),
            unroll_factor: 4,
            use_simd_matrix: true,
            use_async_copy: false,
        };
        let s = tune_to_schedule(&cfg);
        assert_eq!(s.tile_dims, (64, 32, 16));
        assert_eq!(s.threads_per_threadgroup, (256, 1, 1));
    }

    #[test]
    fn tune_to_schedule_falls_back_when_tile_dims_missing() {
        let cfg = TuneConfig {
            tile_dims: vec![],
            threads: (512, 1, 1),
            unroll_factor: 4,
            use_simd_matrix: false,
            use_async_copy: false,
        };
        let s = tune_to_schedule(&cfg);
        assert_eq!(s.tile_dims, (32, 32, 16));
        assert_eq!(s.threads_per_threadgroup, (512, 1, 1));
    }

    #[test]
    fn tune_to_schedule_uses_default_when_tile_dims_too_short() {
        let cfg = TuneConfig {
            tile_dims: vec![16],
            threads: (1024, 1, 1),
            unroll_factor: 4,
            use_simd_matrix: false,
            use_async_copy: false,
        };
        let s = tune_to_schedule(&cfg);
        assert_eq!(s.tile_dims, (32, 32, 16));
    }
}
