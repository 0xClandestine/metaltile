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
            kernels.push(k);
        }
    }
    kernels
}
