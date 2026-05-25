pub mod helpers;
pub mod run;

use metaltile_core::bench::spec::BenchSpec;

/// Collect all bench specs registered via inventory across all kernel crates.
pub fn bench_specs() -> Vec<&'static BenchSpec> {
    let mut specs: Vec<&'static BenchSpec> = inventory::iter::<BenchSpec>().collect();
    specs.sort_by_key(|s| (s.op, s.subop));
    specs
}
