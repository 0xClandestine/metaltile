//! Pass infrastructure and optimization passes.

pub mod const_fold;
pub mod cse;
pub mod fusion;
pub mod licm;
pub mod schedule;
pub mod tile_lowering;
pub mod type_check;
pub mod vectorize;

use std::time::Instant;

use metaltile_core::ir::Kernel;

/// A transformation pass on the IR.
pub trait Pass {
    fn name(&self) -> &str;
    fn run(&self, kernel: &mut Kernel) -> metaltile_core::error::Result<()>;
}

/// Run a sequence of passes on a kernel.
pub fn run_passes(
    kernel: &mut Kernel,
    passes: &[Box<dyn Pass>],
) -> metaltile_core::error::Result<()> {
    for pass in passes {
        pass.run(kernel)?;
    }
    Ok(())
}

/// Statistics for a single pass execution.
#[derive(Debug, Clone)]
pub struct PassStats {
    pub name: String,
    pub ops_before: usize,
    pub ops_after: usize,
    pub wall_us: u64,
}

/// Run a sequence of passes on a kernel, collecting statistics.
/// When `METALTILE_PASS_DEBUG=1` is set, prints a summary table.
pub fn run_passes_with_stats(
    kernel: &mut Kernel,
    passes: &[Box<dyn Pass>],
) -> metaltile_core::error::Result<Vec<PassStats>> {
    let mut stats = Vec::with_capacity(passes.len());
    let debug = std::env::var("METALTILE_PASS_DEBUG").as_deref() == Ok("1");

    for pass in passes {
        let ops_before = count_total_ops(kernel);
        let start = Instant::now();
        pass.run(kernel)?;
        let elapsed = start.elapsed();
        let ops_after = count_total_ops(kernel);
        stats.push(PassStats {
            name: pass.name().to_string(),
            ops_before,
            ops_after,
            wall_us: elapsed.as_micros() as u64,
        });
    }

    if debug {
        eprintln!("pass          ops_before  ops_after  delta  time_us");
        eprintln!("-----------   ----------  ---------  -----  -------");
        for s in &stats {
            let delta = s.ops_after as isize - s.ops_before as isize;
            eprintln!(
                "{:<12}  {:>10}  {:>9}  {:>+5}  {:>7}",
                s.name, s.ops_before, s.ops_after, delta, s.wall_us
            );
        }
    }

    Ok(stats)
}

/// Count all ops across the kernel body and all nested blocks.
fn count_total_ops(kernel: &Kernel) -> usize {
    let mut total = kernel.body.ops.len();
    for block in kernel.blocks.values() {
        total += block.ops.len();
    }
    total
}

// ---------------------------------------------------------------------------
// PipelineBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing an optimization pipeline with optional overrides.
pub struct PipelineBuilder {
    passes: Vec<Box<dyn Pass>>,
}

impl PipelineBuilder {
    /// Create a builder with the standard 9-pass pipeline.
    pub fn standard() -> Self {
        PipelineBuilder {
            passes: vec![
                Box::new(type_check::TypeCheckPass),
                Box::new(const_fold::ConstFoldPass::new()),
                Box::new(cse::CsePass),
                Box::new(licm::LicmPass),
                Box::new(tile_lowering::TileLoweringPass::default()),
                Box::new(fusion::FusionPass),
                Box::new(schedule::SchedulePass::default()),
                Box::new(vectorize::VectorizePass),
            ],
        }
    }

    /// Remove a pass by name from the pipeline.
    pub fn without(mut self, name: &str) -> Self {
        self.passes.retain(|p| p.name() != name);
        self
    }

    /// Override the unroll factor (available after LoopUnroll pass is added).
    pub fn with_unroll_factor(self, _factor: u32) -> Self {
        // TODO: wire up UnrollPass when it's added in PR 2.
        self
    }

    /// Build the final pass list.
    pub fn build(self) -> Vec<Box<dyn Pass>> { self.passes }
}

/// Standard optimization pipeline.
///
/// Order (CODEGEN_OVERHAUL.md §3):
///   TypeCheck -> ConstFold -> CSE -> LICM -> TileLowering -> Fusion -> Schedule -> Vectorize
///
/// (LoopUnroll will be added between Fusion and Schedule in a follow-up PR.)
pub fn standard_pipeline() -> Vec<Box<dyn Pass>> { PipelineBuilder::standard().build() }
