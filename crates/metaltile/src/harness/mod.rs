pub mod filters;
pub mod protocol;
pub mod row;
pub mod runner;

pub use filters::Filters;
pub use protocol::ProtocolRecord;
pub use row::{BenchRow, BuildRow, TestRow};
pub use runner::{run_bench, run_build, run_test};

/// Result of a single correctness check.
#[derive(Debug, Clone)]
pub struct CorrectnessResult {
    pub op_name: String,
    pub dtype: String,
    pub passed: bool,
    pub max_err: f32,
    pub cosine_sim: f32,
}

/// Alias used by external harness consumers.
pub type HarnessMessage = ProtocolRecord;
