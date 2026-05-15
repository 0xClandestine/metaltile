//! metaltile-emit
//!
//! Build-time codegen tool. Walks a registry of `#[kernel]` definitions and
//! produces three artifacts under `<out>/`:
//!
//!   Resources/kernels/<name>.metal   — MSL source per kernel (debug aid)
//!   Resources/kernels.metallib       — compiled Metal library
//!   Resources/manifest.json          — per-kernel metadata
//!   Generated/MetalTileKernels.swift — typed Swift dispatch wrappers
//!
//! Phase 0 ships two kernels: `vector_add` (proof-of-life) and `rms_norm`
//! across f32/f16/bf16. Add more in `register_kernels()` as later phases land.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context as _, Result, bail};
use clap::Parser;
use metaltile::kernel;
use metaltile_codegen::msl::MslGenerator;
use metaltile_core::{
    dtype::DType,
    ir::{Kernel, KernelMode, Param, ParamKind},
};
use serde::Serialize;

// ─── CLI ──────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "metaltile-emit", about = "Emit metallib + manifest + Swift wrappers")]
struct Cli {
    /// Output directory (typically `Sources/MetalTileSwift/` of a Swift package).
    #[arg(long)]
    out: PathBuf,

    /// SDK to use for `xcrun metal` invocation.
    #[arg(long, default_value = "macosx")]
    sdk: String,

    /// Skip the metallib compile step (still emit .metal + manifest + Swift).
    /// Useful when running on a host without the Metal toolchain.
    #[arg(long)]
    no_compile: bool,
}

// ─── Kernel definitions ───────────────────────────────────────────────────
//
// These are the kernels emitted into the Phase 0 metallib. To add a kernel:
//   1. Define it here with `#[kernel]`
//   2. Register it in `register_kernels()` below
//   3. Re-run `cargo run -p metaltile-emit -- --out <dir>`

#[kernel]
fn vector_add(a: Tensor<f32>, b: Tensor<f32>, c: Tensor<f32>) {
    let idx = program_id::<0>();
    store(c[idx], load(a[idx]) + load(b[idx]));
}

#[kernel]
fn mt_rms_norm<T>(
    x: Tensor<T>,
    w: Tensor<T>,
    out: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] n: u32,
) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    let ssq = strided_reduce_dot(x, x, rs, 0, re);
    let tg_ssq = reduce_sum(ssq);
    let eps = load(eps_buf[0]);
    let rms = rsqrt(tg_ssq / n + eps);
    let n_full = n / (lsize * 4u32);
    for _r in range(0u32, n_full, 1u32) {
        let base = rs + (_r * lsize + tid) * 4u32;
        let col = base - rs;
        let n0 = load(x[base]).cast::<f32>() * rms * load(w[col]).cast::<f32>();
        let n1 = load(x[base + 1u32]).cast::<f32>() * rms * load(w[col + 1u32]).cast::<f32>();
        let n2 = load(x[base + 2u32]).cast::<f32>() * rms * load(w[col + 2u32]).cast::<f32>();
        let n3 = load(x[base + 3u32]).cast::<f32>() * rms * load(w[col + 3u32]).cast::<f32>();
        store(out[base], n0.cast::<T>());
        store(out[base + 1u32], n1.cast::<T>());
        store(out[base + 2u32], n2.cast::<T>());
        store(out[base + 3u32], n3.cast::<T>());
    }
    for _i in range(rs + n_full * lsize * 4u32 + tid, re, lsize) {
        let ni = load(x[_i]).cast::<f32>() * rms * load(w[_i - rs]).cast::<f32>();
        store(out[_i], ni.cast::<T>());
    }
}

// ─── Registry ─────────────────────────────────────────────────────────────

/// Build the list of kernels to emit. Each entry is a fully-named IR ready
/// for codegen.
fn register_kernels() -> Vec<Kernel> {
    let mut kernels: Vec<Kernel> = Vec::new();

    // vector_add — single dtype (f32), no generics.
    kernels.push(vector_add::kernel_ir());

    // rms_norm — generic over T, specialize for f32 / f16 / bf16. Rename
    // each instantiation so MSL function names don't collide. Reduction
    // mode is required so the codegen emits `lsize`/`tid`/`tgid` aliases
    // used inside the kernel body.
    for &dt in &[DType::F32, DType::F16, DType::BF16] {
        let mut k = mt_rms_norm::kernel_ir_for(dt);
        k.name = format!("rms_norm_{}", dtype_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);
    }

    kernels
}

// ─── Manifest schema (v1) ─────────────────────────────────────────────────

#[derive(Serialize)]
struct Manifest {
    /// Manifest schema version. Bump on breaking changes.
    version: u32,
    metaltile_emit_version: String,
    kernels: Vec<KernelManifest>,
}

#[derive(Serialize)]
struct KernelManifest {
    /// Public kernel name (also the MSL function name).
    name: String,
    /// Path to the MSL source file relative to the manifest.
    source: String,
    /// Thread-indexing mode — informs default grid/threadgroup sizing.
    kernel_mode: String,
    /// Buffer-bound parameters in slot order.
    params: Vec<ParamManifest>,
    /// Constexpr scalars bound as `setBytes` after `params`.
    constexprs: Vec<ConstExprManifest>,
}

#[derive(Serialize)]
struct ParamManifest {
    name: String,
    /// "Tensor", "Strided", or "Scalar".
    kind: String,
    /// "f32", "f16", "bf16", "u32", "i32", etc.
    dtype: String,
    is_output: bool,
}

#[derive(Serialize)]
struct ConstExprManifest {
    name: String,
    dtype: String,
}

// ─── Main flow ────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();

    let resources_dir = cli.out.join("Resources");
    let kernels_dir = resources_dir.join("kernels");
    let generated_dir = cli.out.join("Generated");

    fs::create_dir_all(&kernels_dir).context("create Resources/kernels")?;
    fs::create_dir_all(&generated_dir).context("create Generated")?;

    let kernels = register_kernels();
    println!("metaltile-emit: registered {} kernels", kernels.len());

    let mut manifest_entries: Vec<KernelManifest> = Vec::new();
    let mut metal_files: Vec<PathBuf> = Vec::new();
    let generator = MslGenerator::default();

    for kernel in &kernels {
        let msl = generator
            .generate(kernel)
            .map_err(|e| anyhow::anyhow!("generate MSL for {}: {:?}", kernel.name, e))?;

        let metal_path = kernels_dir.join(format!("{}.metal", kernel.name));
        fs::write(&metal_path, &msl)
            .with_context(|| format!("write {}", metal_path.display()))?;
        println!("  wrote {}", metal_path.display());

        manifest_entries.push(kernel_to_manifest(kernel));
        metal_files.push(metal_path);
    }

    // Manifest
    let manifest = Manifest {
        version: 1,
        metaltile_emit_version: env!("CARGO_PKG_VERSION").to_string(),
        kernels: manifest_entries,
    };
    let manifest_path = resources_dir.join("manifest.json");
    fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)
        .with_context(|| format!("write {}", manifest_path.display()))?;
    println!("  wrote {}", manifest_path.display());

    // Generated Swift wrappers
    let swift = generate_swift_wrappers(&manifest);
    let swift_path = generated_dir.join("MetalTileKernels.swift");
    fs::write(&swift_path, swift).with_context(|| format!("write {}", swift_path.display()))?;
    println!("  wrote {}", swift_path.display());

    // Compile metallib (unless explicitly skipped)
    if cli.no_compile {
        println!("--no-compile: skipping metallib build");
    } else {
        let metallib_path = resources_dir.join("kernels.metallib");
        compile_metallib(&metal_files, &metallib_path, &cli.sdk)?;
        println!("  wrote {}", metallib_path.display());
    }

    println!("metaltile-emit: done");
    Ok(())
}

// ─── Helpers ──────────────────────────────────────────────────────────────

fn dtype_suffix(dt: DType) -> &'static str {
    match dt {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::I32 => "i32",
        DType::U32 => "u32",
        DType::I8 => "i8",
        DType::U8 => "u8",
        DType::I64 => "i64",
        DType::U64 => "u64",
        DType::I4 => "i4",
        DType::Bool => "bool",
    }
}

fn param_kind_str(k: &ParamKind) -> &'static str {
    match k {
        ParamKind::Tensor => "Tensor",
        ParamKind::Strided => "Strided",
        ParamKind::Scalar => "Scalar",
    }
}

fn kernel_mode_str(m: KernelMode) -> &'static str {
    match m {
        KernelMode::Elementwise => "Elementwise",
        KernelMode::Reduction => "Reduction",
        KernelMode::Grid3D => "Grid3D",
        KernelMode::Tile2D => "Tile2D",
    }
}

fn kernel_to_manifest(k: &Kernel) -> KernelManifest {
    KernelManifest {
        name: k.name.clone(),
        source: format!("kernels/{}.metal", k.name),
        kernel_mode: kernel_mode_str(k.mode).to_string(),
        params: k
            .params
            .iter()
            .map(|p: &Param| ParamManifest {
                name: p.name.clone(),
                kind: param_kind_str(&p.kind).to_string(),
                dtype: dtype_suffix(p.dtype).to_string(),
                is_output: p.is_output,
            })
            .collect(),
        constexprs: k
            .constexprs
            .iter()
            .map(|c| ConstExprManifest {
                name: c.name.name().to_string(),
                dtype: dtype_suffix(c.dtype).to_string(),
            })
            .collect(),
    }
}

// ─── Swift wrapper generation ─────────────────────────────────────────────
//
// One static function per kernel. Caller supplies MTLBuffers (+ offsets),
// constexpr scalars, grid + threadgroup sizes, and a command buffer. The
// wrapper looks up the PSO from `PSOCache.shared`, encodes the dispatch,
// and ends the encoder. PSOCache lives in MetalTileSwift hand-written code.

fn generate_swift_wrappers(manifest: &Manifest) -> String {
    let mut out = String::new();
    out.push_str(
        "// AUTOGENERATED by metaltile-emit. DO NOT EDIT.\n\
         //\n\
         // Each function dispatches a single Metal kernel from kernels.metallib.\n\
         // Looks up the pre-compiled PSO from PSOCache.shared, encodes the\n\
         // dispatch on the supplied command buffer, ends the encoder.\n\n\
         import Metal\n\n\
         public enum MetalTileKernels {\n",
    );

    for k in &manifest.kernels {
        emit_swift_wrapper(&mut out, k);
    }

    out.push_str("}\n");
    out
}

fn emit_swift_wrapper(out: &mut String, k: &KernelManifest) {
    use std::fmt::Write as _;
    let fn_name = swift_safe_name(&k.name);

    writeln!(out, "    /// Dispatches `{}` from kernels.metallib.", k.name).ok();
    writeln!(out, "    public static func {fn_name}(").ok();

    // Buffer params (Tensor / Strided / Scalar all bind as buffers in Phase 0)
    for p in &k.params {
        let label = swift_safe_name(&p.name);
        writeln!(out, "        {label}: MTLBuffer, {label}Offset: Int = 0,").ok();
    }
    // Constexpr scalars (bound via setBytes after the param buffers)
    for c in &k.constexprs {
        let label = swift_safe_name(&c.name);
        let swift_ty = swift_scalar_type(&c.dtype);
        writeln!(out, "        {label}: {swift_ty},").ok();
    }
    // Grid + threadgroup sizing
    writeln!(out, "        gridSize: MTLSize,").ok();
    writeln!(out, "        threadgroupSize: MTLSize,").ok();
    writeln!(out, "        on commandBuffer: MTLCommandBuffer").ok();
    writeln!(out, "    ) {{").ok();
    writeln!(
        out,
        "        let pso = PSOCache.shared.pipelineState(for: \"{}\")",
        k.name
    )
    .ok();
    writeln!(
        out,
        "        guard let enc = commandBuffer.makeComputeCommandEncoder() else {{ return }}"
    )
    .ok();
    writeln!(out, "        enc.setComputePipelineState(pso)").ok();

    let mut slot = 0usize;
    for p in &k.params {
        let label = swift_safe_name(&p.name);
        writeln!(
            out,
            "        enc.setBuffer({label}, offset: {label}Offset, index: {slot})"
        )
        .ok();
        slot += 1;
    }
    for c in &k.constexprs {
        let label = swift_safe_name(&c.name);
        let len = swift_scalar_size(&c.dtype);
        writeln!(out, "        var {label}_v = {label}").ok();
        writeln!(
            out,
            "        enc.setBytes(&{label}_v, length: {len}, index: {slot})"
        )
        .ok();
        slot += 1;
    }
    writeln!(
        out,
        "        enc.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)"
    )
    .ok();
    writeln!(out, "        enc.endEncoding()").ok();
    writeln!(out, "    }}\n").ok();
}

fn swift_safe_name(s: &str) -> String {
    // For Phase 0 just snake-case → snake-case. We may want camelCase later
    // for idiomatic Swift; revisit when we have more kernels.
    s.replace('-', "_")
}

fn swift_scalar_type(dtype: &str) -> &'static str {
    match dtype {
        "f32" => "Float",
        "f16" => "Float16",
        "bf16" => "Float", // no native Swift bfloat16; pass widened, kernel reads narrow
        "i32" => "Int32",
        "u32" => "UInt32",
        "i64" => "Int64",
        "u64" => "UInt64",
        "i8" => "Int8",
        "u8" => "UInt8",
        "bool" => "Bool",
        _ => "UInt32",
    }
}

fn swift_scalar_size(dtype: &str) -> usize {
    match dtype {
        "f32" | "i32" | "u32" => 4,
        "f16" | "bf16" | "i16" | "u16" => 2,
        "i8" | "u8" | "bool" => 1,
        "i64" | "u64" => 8,
        _ => 4,
    }
}

// ─── Metal toolchain invocation ───────────────────────────────────────────

fn compile_metallib(metal_files: &[PathBuf], output: &Path, sdk: &str) -> Result<()> {
    if metal_files.is_empty() {
        bail!("no .metal files to compile");
    }

    let air_dir = tempdir_in_target()?;
    let mut air_files: Vec<PathBuf> = Vec::new();

    println!("compiling {} .metal files...", metal_files.len());
    for metal in metal_files {
        let stem = metal
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow::anyhow!("bad metal filename: {}", metal.display()))?;
        let air = air_dir.join(format!("{stem}.air"));

        let status = Command::new("xcrun")
            .args(["-sdk", sdk, "metal", "-c"])
            .arg(metal)
            .arg("-o")
            .arg(&air)
            .status()
            .with_context(|| format!("invoke xcrun metal for {}", metal.display()))?;
        if !status.success() {
            bail!("xcrun metal failed for {}", metal.display());
        }
        air_files.push(air);
    }

    println!("linking metallib {}", output.display());
    let status = Command::new("xcrun")
        .args(["-sdk", sdk, "metallib"])
        .args(&air_files)
        .arg("-o")
        .arg(output)
        .status()
        .context("invoke xcrun metallib")?;
    if !status.success() {
        bail!("xcrun metallib failed");
    }

    Ok(())
}

fn tempdir_in_target() -> Result<PathBuf> {
    // Use cargo's target/ so we don't pollute /tmp on every build.
    let dir = std::env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target"))
        .join("metaltile-emit-air");
    fs::create_dir_all(&dir).context("create air tempdir")?;
    Ok(dir)
}
