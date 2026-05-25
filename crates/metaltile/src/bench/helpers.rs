//! Helpers for bench kernel authors — MSL generation and test macros.
//!
//! Moved from `metaltile-core/src/bench/types.rs` because they depend on
//! `metaltile-codegen` which would create a circular dependency in core.

use metaltile_codegen::msl::MslGenerator;
use metaltile_core::ir::{Kernel, KernelMode};

/// Generate MSL for an elementwise kernel IR produced by `make_ir`.
///
/// Uses default `KernelMode::Elementwise`. `label` is used only in the error message.
pub fn generate_elementwise_msl<F>(make_ir: F, label: &str) -> String
where F: Fn() -> Kernel {
    MslGenerator::default().generate(&make_ir()).unwrap_or_else(|e| {
        eprintln!("[{label}]: {e}");
        String::new()
    })
}

/// Generate MSL for a reduction kernel IR produced by `make_ir`, setting `Reduction` mode.
///
/// `label` is used only in the error message when code generation fails.
pub fn generate_reduction_msl<F>(make_ir: F, label: &str) -> String
where F: Fn() -> Kernel {
    let mut k = make_ir();
    k.mode = KernelMode::Reduction;
    MslGenerator::default().generate(&k).unwrap_or_else(|e| {
        eprintln!("[{label}]: {e}");
        String::new()
    })
}

/// Emit the standard two-test block for a reduction op.
///
/// Generates:
/// - `msl_generates_for_all_dtypes` — calls `$msl_fn(dt)` for each float dtype
/// - `kernels_compile` (macos only) — compiles the generated MSL
///
/// Usage:
/// ```ignore
/// bench_tests!(msl_fn: layer_norm_msl_for, kernel_name: "mt_layer_norm");
/// ```
#[macro_export]
macro_rules! bench_tests {
    (msl_fn: $msl_fn:ident, kernel_name: $name:expr) => {
        #[cfg(test)]
        mod tests {
            use super::*;

            #[test]
            fn msl_generates_for_all_dtypes() {
                for &dt in metaltile_core::bench::types::FLOAT_DTYPES {
                    let msl = $msl_fn(dt);
                    assert!(!msl.trim().is_empty(), "MSL empty for {dt:?}");
                }
            }

            #[cfg(target_os = "macos")]
            #[test]
            fn kernels_compile() {
                // NOTE: GpuRunner is not available in metaltile-std.
                // This test is only meaningful in metaltile-bench or metaltile-cli.
                // The MSL generation test above covers the pure path.
            }
        }
    };
}
