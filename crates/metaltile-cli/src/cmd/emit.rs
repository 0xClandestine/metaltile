//! `tile emit` — emit a `kernels.metallib` + manifest + Swift wrappers.
//!
//! Iterates every `BenchSpec` registered via `inventory::submit!` and
//! produces artifacts under `<out>/`:
//!
//!   Resources/kernels/<name>.metal   — MSL source per kernel
//!   Resources/kernels.metallib       — compiled Metal library
//!   Resources/manifest.json          — per-kernel metadata
//!   Generated/MetalTileKernels.swift — typed Swift dispatch wrappers
//!
//! Usage:
//!   tile emit --out <swift-package-dir> [--sdk macosx] [--no-compile]

use std::{collections::BTreeMap, fs, path::PathBuf};

use metaltile_codegen::{
    MslGenerator,
    emit::{compile_metallib, dtype_suffix, write_manifest, write_msl, write_swift_wrappers},
};
use metaltile_core::ir::Kernel;
use metaltile_std::{
    bench_types::DType,
    spec::{BenchSpec, effective_mode},
};

use crate::{CliError, EmitArgs};

pub fn run(args: &EmitArgs) -> Result<(), CliError> {
    let out = PathBuf::from(&args.out);
    let resources_dir = out.join("Resources");
    let kernels_dir = resources_dir.join("kernels");
    let generated_dir = out.join("Generated");

    fs::create_dir_all(&kernels_dir)?;
    fs::create_dir_all(&generated_dir)?;

    let kernels = collect_kernels();
    println!("tile emit: {} kernels", kernels.len());

    let generator = MslGenerator::default();
    let mut metal_paths = Vec::with_capacity(kernels.len());

    for kernel in &kernels {
        let path = write_msl(kernel, &kernels_dir, &generator)
            .map_err(|e| CliError::Other(format!("MSL for {}: {e}", kernel.name)))?;
        println!("  wrote {}", path.display());
        metal_paths.push(path);
    }

    write_manifest(&kernels, &resources_dir.join("manifest.json"))
        .map_err(|e| CliError::Other(e.to_string()))?;
    println!("  wrote {}", resources_dir.join("manifest.json").display());

    write_swift_wrappers(&kernels, &generated_dir.join("MetalTileKernels.swift"))
        .map_err(|e| CliError::Other(e.to_string()))?;
    println!("  wrote {}", generated_dir.join("MetalTileKernels.swift").display());

    if args.no_compile {
        println!("--no-compile: skipping metallib build");
    } else {
        let metallib = resources_dir.join("kernels.metallib");
        let air_dir = std::env::var("CARGO_TARGET_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("target"))
            .join("tile-emit-air");
        compile_metallib(&metal_paths, &metallib, &args.sdk, &air_dir)
            .map_err(|e| CliError::Other(e.to_string()))?;
        println!("  wrote {}", metallib.display());
    }

    println!("tile emit: done");
    Ok(())
}

/// Collect all kernels from the inventory, one per (kernel_name, dtype) pair.
///
/// Multiple `BenchSpec`s can share the same `kernel_name` (e.g. different
/// shapes for the same kernel). We deduplicate by name and union their dtypes,
/// then call `kernel_ir(dt)` once per unique (name, dt) combination.
fn collect_kernels() -> Vec<Kernel> {
    let mut by_name: BTreeMap<&str, (&BenchSpec, Vec<DType>)> = BTreeMap::new();
    for spec in inventory::iter::<BenchSpec> {
        let entry = by_name.entry(spec.kernel_name).or_insert_with(|| (spec, Vec::new()));
        for &dt in spec.dtypes {
            if !entry.1.contains(&dt) {
                entry.1.push(dt);
            }
        }
    }

    let total: usize = by_name.values().map(|(_, dtypes)| dtypes.len()).sum();
    let mut kernels = Vec::with_capacity(total);
    for (kernel_name, (spec, dtypes)) in &by_name {
        let mode = effective_mode(spec);
        for &dt in dtypes {
            let mut k = (spec.kernel_ir)(dt);
            k.name = format!("{}_{}", kernel_name, dtype_suffix(dt));
            k.mode = mode;
            // Per-kernel opt-in for the `_indirect` Swift wrapper.
            // Replaces the previous hardcoded kernel-name allowlist in
            // `metaltile-codegen::emit` — kernels now declare their own
            // indirect-dispatch eligibility (see
            // `dequant_gemv::dequant_gemv_wants_indirect`) and the
            // codegen pass just reads `Kernel::wants_indirect_variant`.
            if metaltile_std::ffai::dequant_gemv::dequant_gemv_wants_indirect(&k.name) {
                k.wants_indirect_variant = true;
            }
            kernels.push(k);
        }
    }
    kernels
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression guard for the `_indirect` Swift wrappers.
    ///
    /// FFAI's GPU-router dispatches `dequant_gemv_int4` indirectly (grid
    /// shape from an `MTLBuffer` rather than a host `MTLSize`). When emit
    /// moved out of the `metaltile-emit` crate into `tile emit` (#145),
    /// the indirect-wrapper generator was dropped — `tile emit` produced
    /// only the direct wrappers and the FFAI build broke.
    ///
    /// This exercises the full `tile emit` pipeline — registry inventory
    /// → per-dtype naming → `render_swift_wrappers` — so a future drop of
    /// the indirect path, an unregistered/renamed `dequant_gemv_int4`, or
    /// a lost f16/bf16 dtype all fail here rather than in a downstream
    /// FFAI build.
    #[test]
    fn tile_emit_keeps_indirect_wrappers_for_dequant_gemv_int4() {
        let kernels = collect_kernels();
        assert!(
            kernels.iter().any(|k| k.name == "dequant_gemv_int4_f16"),
            "dequant_gemv_int4_f16 missing from the `tile emit` kernel set"
        );
        assert!(
            kernels.iter().any(|k| k.name == "dequant_gemv_int4_bf16"),
            "dequant_gemv_int4_bf16 missing from the `tile emit` kernel set"
        );

        let swift = metaltile_codegen::emit::render_swift_wrappers(&kernels);
        assert!(
            swift.contains("func dequant_gemv_int4_f16_indirect("),
            "indirect Swift wrapper for dequant_gemv_int4_f16 dropped from `tile emit`"
        );
        assert!(
            swift.contains("func dequant_gemv_int4_bf16_indirect("),
            "indirect Swift wrapper for dequant_gemv_int4_bf16 dropped from `tile emit`"
        );
        assert!(
            swift.contains("dispatchThreadgroups(indirectBuffer:"),
            "indirect wrappers must dispatch from an indirect buffer"
        );
    }
}
