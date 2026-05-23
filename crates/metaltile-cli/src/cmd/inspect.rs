//! `tile inspect` — Print IR and/or MSL for kernels and TOML models.
//!
//! Two modes:
//!   1. **Single kernel** (default): iterates `inventory::iter::<BenchSpec>`.
//!   2. **TOML model** (`--toml <path>`): compiles a TOML model definition
//!      and inspects the full pipeline — pre-fusion IR, fused host kernels,
//!      and generated MSL for each dispatch node / fused group.
//!
//! Usage:
//!   tile inspect                           # list all registered kernels
//!   tile inspect <kernel>                  # print final MSL (default)
//!   tile inspect <kernel> --ir             # print raw IR
//!   tile inspect <kernel> --stats          # print per-pass op-count table
//!   tile inspect <kernel> -o /tmp/out      # write .metal file
//!   tile inspect --all -o /tmp/out         # dump every kernel to disk
//!   tile inspect --toml models/llama_decode.toml
//!   tile inspect --toml models/llama_decode.toml --no-fuse
//!   tile inspect --toml models/llama_decode.toml --ir
//!   tile inspect --toml models/llama_decode.toml --pass fusion
//!   tile inspect --toml models/llama_decode.toml --pass all
//!   tile inspect --toml models/llama_decode.toml --stats

use std::{collections::BTreeMap, str::FromStr};

use metaltile_codegen::generator_for_mode;
use metaltile_model::{
    CompileParams, ExecutionPlan, FusionMode, KernelRegistry, ModelDef,
    compile,
};
use metaltile_std::{
    bench_types::DType,
    spec::{BenchSpec, effective_mode},
};

use crate::{
    CliError,
    InspectArgs,
    matches_filter,
    term::{Color, Style, paint_stdout},
};

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn run(args: &InspectArgs) -> Result<(), CliError> {
    // ── TOML model inspection path ──────────────────────────────────
    if let Some(toml_path) = &args.toml {
        return run_toml_inspect(toml_path, args);
    }

    // ── Single-kernel inspection path (original) ────────────────────
    let filter_val = args.filter.as_ref().or(args.kernel.as_ref());
    let _span = tracing::info_span!(
        "inspect",
        filter = ?filter_val,
        ir = args.ir,
        stats = args.stats,
    )
    .entered();
    let dir = &args.dir;
    let filter = filter_val;
    let all_flag = args.all;
    let ir_flag = args.ir;
    let stats_flag = args.stats;
    let pass_arg = &args.pass;
    let dtype_override: Option<DType> = args.dtype.as_deref().and_then(|s| DType::from_str(s).ok());

    // Collect all specs and group by kernel_name.
    let mut kernels: BTreeMap<&str, (&BenchSpec, Vec<DType>)> = BTreeMap::new();
    for spec in inventory::iter::<BenchSpec> {
        let entry = kernels.entry(spec.kernel_name).or_insert_with(|| (spec, Vec::new()));
        for &dt in spec.dtypes {
            if !entry.1.contains(&dt) {
                entry.1.push(dt);
            }
        }
    }

    if kernels.is_empty() {
        eprintln!("No kernels registered.");
        return Ok(());
    }

    let mut sorted: Vec<(&str, (&BenchSpec, Vec<DType>))> = kernels.into_iter().collect();
    sorted.sort_unstable_by_key(|(name, _)| *name);

    // --all flag: dump every kernel
    if all_flag {
        for (name, (spec, dtypes)) in &sorted {
            let dt = dtypes.first().copied().unwrap_or(DType::F32);
            if ir_flag {
                let k = (spec.kernel_ir)(dt);
                if let Some(d) = dir {
                    let path = format!("{}/{}.ir", d, name);
                    std::fs::create_dir_all(d).map_err(CliError::Io)?;
                    std::fs::write(&path, format!("{k}")).map_err(CliError::Io)?;
                    println!("wrote {path}");
                } else {
                    println!("{k}");
                }
            } else {
                let msl = generate_msl(spec, dtypes);
                if let Some(d) = dir {
                    let path = format!("{}/{}.metal", d, name);
                    std::fs::create_dir_all(d).map_err(CliError::Io)?;
                    std::fs::write(&path, &msl).map_err(CliError::Io)?;
                    println!("wrote {path}");
                } else {
                    let mode_str = effective_mode(spec).to_string();
                    println!("// ═══════════════════════════════════════════════════════");
                    println!("// kernel: {}  mode: {}", name, mode_str);
                    println!("// ═══════════════════════════════════════════════════════");
                    println!("{}", msl);
                }
            }
        }
        return Ok(());
    }

    // No filter: list all kernels
    let Some(filter) = filter else {
        eprintln!("{}", paint_stdout("tile inspect", Style::new().fg(Color::Cyan).bold()),);
        eprintln!();
        for (name, (spec, dtypes)) in &sorted {
            let dtype_str = dtypes.iter().map(|dt| dt.label()).collect::<Vec<_>>().join("/");
            let mode_str = effective_mode(spec).to_string();
            eprintln!(
                "  {}   {}   {dtype_str}",
                paint_stdout(format!("{name:<20}"), Style::new().fg(Color::Cyan).bold()),
                paint_stdout(mode_str, Style::new().fg(Color::BrightBlack)),
            );
        }
        let sep = paint_stdout("·", Style::new().fg(Color::BrightBlack).dim());
        eprintln!();
        eprintln!(
            "  {} {sep} {}",
            paint_stdout(format!("{} kernels", sorted.len()), Style::new().fg(Color::BrightBlack)),
            paint_stdout("<kernel> for MSL", Style::new().fg(Color::BrightBlack)),
        );
        return Ok(());
    };

    // Filter by kernel name
    let matched: Vec<_> =
        sorted.iter().filter(|(name, _)| matches_filter(Some(filter), name)).collect();

    if matched.is_empty() {
        eprintln!(
            "{} {}",
            paint_stdout("error:", Style::new().fg(Color::Red).bold()),
            paint_stdout(
                format!("no kernel matched '{filter}'"),
                Style::new().fg(Color::BrightWhite),
            ),
        );
        eprintln!(
            "\n{} {}",
            paint_stdout("Available:", Style::new().fg(Color::BrightBlack)),
            paint_stdout(
                sorted.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", "),
                Style::new().fg(Color::BrightWhite),
            ),
        );
        return Err(CliError::Other(format!("no kernel matched '{filter}'")));
    }

    for (name, (spec, dtypes)) in &matched {
        let dt = dtype_override.unwrap_or_else(|| dtypes.first().copied().unwrap_or(DType::F32));

        if ir_flag {
            // Print raw IR via Display impl
            let k = (spec.kernel_ir)(dt);
            if let Some(d) = dir {
                let path = format!("{}/{}.ir", d, name);
                std::fs::create_dir_all(d).map_err(CliError::Io)?;
                std::fs::write(&path, format!("{k}")).map_err(CliError::Io)?;
                println!("wrote {path}");
            } else {
                println!("{k}");
            }
        } else if stats_flag {
            let mut k = (spec.kernel_ir)(dt);
            k.mode = effective_mode(spec);
            let expected_tpg =
                spec.shapes.first().map(|s| s.tpg as u32).or_else(|| spec.dispatch.tpg_hint());
            let generator = generator_for_mode(effective_mode(spec), expected_tpg);
            match generator.generate_with_stats(&k) {
                Ok((_, stats)) => print_stats_table(&stats),
                Err(e) => eprintln!("error: {e}"),
            }
        } else if let Some(pass) = pass_arg {
            // --pass flag: print IR after a specific pass (or 'all' for every stage)
            let mut k = (spec.kernel_ir)(dt);
            let mode = effective_mode(spec);
            k.mode = mode;

            match pass.as_str() {
                "all" => {
                    println!("// ── BEFORE PASSES ───────────────────────────");
                    println!("{k}");
                    run_all_passes_and_print(&mut k);
                },
                name => match metaltile_codegen::passes::PassRegistry::get(name) {
                    Some(pass_obj) => {
                        if let Err(e) = pass_obj.run(&mut k) {
                            eprintln!("Pass {name} failed: {e}");
                            return Ok(());
                        }
                        println!("// ── AFTER {name} ────────────────────────");
                        println!("{k}");
                    },
                    None => {
                        let valid: Vec<_> = metaltile_codegen::passes::PassRegistry::names();
                        eprintln!("Unknown pass: {name}. Valid: {} all", valid.join(", "));
                        return Ok(());
                    },
                },
            }
        } else {
            // Default: print MSL
            let eff_dt =
                dtype_override.unwrap_or_else(|| dtypes.first().copied().unwrap_or(DType::F32));
            let msl = generate_msl_dt(spec, eff_dt);
            if let Some(d) = dir {
                let path = format!("{}/{}.metal", d, name);
                std::fs::create_dir_all(d).map_err(CliError::Io)?;
                std::fs::write(&path, &msl).map_err(CliError::Io)?;
                println!("wrote {path}");
            } else {
                let mode_str = effective_mode(spec).to_string();
                println!("// ═══════════════════════════════════════════════════════");
                println!("// kernel: {}  mode: {}", name, mode_str);
                println!("// ═══════════════════════════════════════════════════════");
                println!("{}", msl);
            }
        }
    }
    Ok(())
}

// ===========================================================================
// TOML model inspection
// ===========================================================================

/// Inspect a TOML-defined model pipeline.
///
/// Compiles the model definition and displays the full pipeline structure,
/// individual node IRs, fused host kernels, and generated MSL — with user
/// control over fusion mode and pass inspection.
fn run_toml_inspect(toml_path: &str, args: &InspectArgs) -> Result<(), CliError> {
    // ── Read and parse TOML ──────────────────────────────────────────
    let toml_src =
        std::fs::read_to_string(toml_path).map_err(|e| CliError::Other(format!("{e}")))?;
    let def: ModelDef =
        toml::from_str(&toml_src).map_err(|e| CliError::Other(format!("TOML parse: {e}")))?;

    // ── Resolve model params ────────────────────────────────────────
    let params = resolve_model_params(args, &def)?;
    let dtype = resolve_dtype(args);
    let fusion_mode = resolve_fusion_mode(args);

    // ── Build registry and compile ──────────────────────────────────
    let reg = KernelRegistry::build();

    let state_keys = vec![
        "token_id".to_string(),
        "position".to_string(),
        "n_kv".to_string(),
        "rms_eps".to_string(),
        "temperature".to_string(),
        "uniform".to_string(),
    ];
    let mut all_state_keys = state_keys.clone();
    let n_layers = *params.params.get("n_layers").unwrap_or(&1) as usize;
    for layer in 0..n_layers {
        all_state_keys.push(format!("kv_cache.{layer}.k"));
        all_state_keys.push(format!("kv_cache.{layer}.v"));
    }

    let compile_params = CompileParams {
        params: params.params,
        float_params: std::collections::HashMap::new(),
        activation_dtype: dtype,
        n_layers,
        state_keys: all_state_keys,
    };

    eprintln!(
        "Compiling {} with {n_layers} layers, {dtype}, fusion={fusion_mode:?}",
        def.model.name
    );
    let plan = compile(&def, &compile_params, &reg, fusion_mode)
        .map_err(|e| CliError::Other(format!("compile: {e}")))?;

    // ── Display ─────────────────────────────────────────────────────
    let dir = &args.dir;

    let ir_flag = args.ir;
    let stats_flag = args.stats;
    let pass_arg = &args.pass;

    // Determine whether fusion happened at the plan level.
    let has_fusion = plan.nodes.iter().any(|n| n.fuse_group.is_some())
        || (fusion_mode != FusionMode::None && plan.nodes.len() < plan.nodes.len()); // fused groups reduce count

    if ir_flag {
        // --ir: show pre-fusion IR for each node (before plan-level fusion)
        print_toml_pipeline_summary(&plan, def.model.name.as_str(), &compile_params, dtype);
        print_toml_nodes_ir(&plan, &compile_params, dtype, dir)?;
    } else if stats_flag {
        // --stats: run pass pipeline on each (fused) kernel, show stats
        if has_fusion {
            eprintln!("\nPer-kernel pass statistics after fusion:");
        }
        print_toml_kernel_stats(&plan, dir)?;
    } else if let Some(pass) = pass_arg {
        // --pass: show IR after a specific pass on each fused kernel
        print_toml_pipeline_summary(&plan, def.model.name.as_str(), &compile_params, dtype);
        match pass.as_str() {
            "all" => print_toml_passes_all(&plan, dtype, dir)?,
            name => print_toml_pass_single(&plan, name, dtype, dir)?,
        }
    } else {
        // Default: show pipeline summary + final MSL for each (fused) kernel
        print_toml_pipeline_summary(&plan, def.model.name.as_str(), &compile_params, dtype);
        print_toml_kernels_msl(&plan, dtype, dir)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// TOML display helpers
// ---------------------------------------------------------------------------

/// Print a compact summary of the compiled pipeline.
fn print_toml_pipeline_summary(
    plan: &ExecutionPlan,
    model_name: &str,
    params: &CompileParams,
    _dtype: DType,
) {
    let n_fused = plan.nodes.iter().filter(|n| n.fuse_group.is_some()).count();
    let n_standalone = plan.nodes.len() - n_fused;
    let n_groups = plan
        .nodes
        .iter()
        .filter_map(|n| n.fuse_group)
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);

    let bold_cyan = Style::new().fg(Color::Cyan).bold();
    let bright_black = Style::new().fg(Color::BrightBlack);
    let sep = paint_stdout("·", Style::new().fg(Color::BrightBlack).dim());

    eprintln!();
    eprintln!("{}", paint_stdout(format!("═══ {model_name} ═══"), bold_cyan));
    eprintln!(
        "  {} {sep} {} hidden, {} vocab, {} heads, {} kv-heads",
        paint_stdout(format!("{} layers", params.n_layers), bright_black),
        paint_stdout(
            params.params.get("hidden_dim").map(|v| v.to_string()).unwrap_or_default(),
            bright_black
        ),
        paint_stdout(
            params.params.get("vocab_size").map(|v| v.to_string()).unwrap_or_default(),
            bright_black
        ),
        paint_stdout(
            params.params.get("n_heads").map(|v| v.to_string()).unwrap_or_default(),
            bright_black
        ),
        paint_stdout(
            params.params.get("n_kv_heads").map(|v| v.to_string()).unwrap_or_default(),
            bright_black
        ),
    );
    eprintln!(
        "  {} {sep} {} slots, {} cached kernels",
        paint_stdout(format!("{} nodes", plan.nodes.len()), bright_black),
        paint_stdout(plan.slots.len().to_string(), bright_black),
        paint_stdout(plan.cached_kernels.len().to_string(), bright_black),
    );
    if n_groups > 0 {
        eprintln!(
            "  {} {sep} {} standalone, {} fused groups",
            paint_stdout(format!("fusion enabled"), Style::new().fg(Color::Green).bold()),
            paint_stdout(n_standalone.to_string(), bright_black),
            paint_stdout(n_groups.to_string(), bright_black),
        );
    } else {
        eprintln!(
            "  {}",
            paint_stdout("fusion disabled (each node dispatched separately)", bright_black)
        );
    }
    eprintln!();

    // Show node list with fuse groups
    let mut current_group: Option<usize> = None;
    for (i, node) in plan.nodes.iter().enumerate() {
        let group_tag = if let Some(gid) = node.fuse_group {
            if current_group != Some(gid) {
                current_group = Some(gid);
                format!(" ╭─ fuse[{}]", gid)
            } else {
                format!(" │  fuse[{}]", gid)
            }
        } else {
            current_group = None;
            String::new()
        };
        let mode_str = node.mode.to_string();
        eprintln!(
            "  {:>3}{}  {:<35}  {}  {}",
            i,
            paint_stdout(group_tag, Style::new().fg(Color::Yellow)),
            paint_stdout(&node.label, bold_cyan),
            paint_stdout(&node.kernel_name, Style::new().fg(Color::Green)),
            paint_stdout(mode_str, bright_black),
        );
    }
    eprintln!();
}

/// Print raw IR for each pre-fusion node.
fn print_toml_nodes_ir(
    plan: &ExecutionPlan,
    _params: &CompileParams,
    _dtype: DType,
    dir: &Option<String>,
) -> Result<(), CliError> {
    for (i, kernel) in plan.cached_kernels.iter().enumerate() {
        let label = &plan.nodes[i].label;
        let safe_name: String = label.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect();
        let filename = format!("{:03}_{}", i, safe_name);

        if let Some(d) = dir {
            let path = format!("{}/{}.ir", d, filename);
            std::fs::create_dir_all(d).map_err(CliError::Io)?;
            std::fs::write(&path, format!("{kernel}")).map_err(CliError::Io)?;
            println!("wrote {path}");
        } else {
            println!("// ── [{:>3}] {} ─────────────────────", i, label);
            println!("// kernel: {}  mode: {}", plan.nodes[i].kernel_name, plan.nodes[i].mode);
            println!("{kernel}");
        }
    }
    Ok(())
}

/// Generate and print MSL for each (fused) kernel.
fn print_toml_kernels_msl(
    plan: &ExecutionPlan,
    _dtype: DType,
    dir: &Option<String>,
) -> Result<(), CliError> {
    for (i, kernel) in plan.cached_kernels.iter().enumerate() {
        let node = &plan.nodes[i];
        let safe_name: String = node
            .label
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
            .collect();
        let filename = format!("{:03}_{}", i, safe_name);

        let mut k = kernel.clone();
        k.mode = node.mode;
        let generator = generator_for_mode(node.mode, None);
        let msl = generator.generate(&k).unwrap_or_else(|e| format!("// ERROR: {e}\n"));

        if let Some(d) = dir {
            let path = format!("{}/{}.metal", d, filename);
            std::fs::create_dir_all(d).map_err(CliError::Io)?;
            std::fs::write(&path, &msl).map_err(CliError::Io)?;
            println!("wrote {path}");
        } else {
            let fuse_tag = node
                .fuse_group
                .map(|g| format!(" fuse_group={g}"))
                .unwrap_or_default();
            println!("// ═══════════════════════════════════════════════════════");
            println!(
                "// [{:>3}] {}    mode: {}{}",
                i, node.label, node.mode, fuse_tag
            );
            println!("// kernel: {}", node.kernel_name);
            println!("// ═══════════════════════════════════════════════════════");
            println!("{msl}");
        }
    }
    Ok(())
}

/// Run the full pass pipeline on each (fused) kernel and print per-pass stats.
fn print_toml_kernel_stats(
    plan: &ExecutionPlan,
    dir: &Option<String>,
) -> Result<(), CliError> {
    for (i, kernel) in plan.cached_kernels.iter().enumerate() {
        let node = &plan.nodes[i];
        let mut k = kernel.clone();
        k.mode = node.mode;
        let generator = generator_for_mode(node.mode, None);

        match generator.generate_with_stats(&k) {
            Ok((_, ref stats)) => {
                let fuse_tag = node
                    .fuse_group
                    .map(|g| format!(" [fuse={g}]"))
                    .unwrap_or_default();
                println!("// ── [{:>3}] {}{} ─────────────────────", i, node.label, fuse_tag);
                print_stats_table(stats);
                println!();

                if let Some(d) = dir {
                    let safe_name: String = node
                        .label
                        .chars()
                        .map(|c| {
                            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                                c
                            } else {
                                '_'
                            }
                        })
                        .collect();
                    let filename = format!("{:03}_{}", i, safe_name);
                    let path = format!("{}/{}.stats.txt", d, filename);
                    std::fs::create_dir_all(d).map_err(CliError::Io)?;
                    std::fs::write(&path, format!("{stats:#?}")).map_err(CliError::Io)?;
                    println!("wrote {path}");
                }
            }
            Err(e) => eprintln!("error: [{i}] {} — {e}", node.label),
        }
    }
    Ok(())
}

/// Run the full pass pipeline on each kernel, printing IR after each pass.
fn print_toml_passes_all(
    plan: &ExecutionPlan,
    _dtype: DType,
    dir: &Option<String>,
) -> Result<(), CliError> {
    use metaltile_codegen::msl::MslGenerator;

    for (i, kernel) in plan.cached_kernels.iter().enumerate() {
        let node = &plan.nodes[i];
        let mut k = kernel.clone();
        k.mode = node.mode;

        let fuse_tag = node
            .fuse_group
            .map(|g| format!(" [fuse={g}]"))
            .unwrap_or_default();

        if let Some(d) = dir {
            let safe_name: String = node
                .label
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect();
            let filename = format!("{:03}_{}", i, safe_name);
            let base = format!("{}/{}.pass", d, filename);
            std::fs::create_dir_all(d).map_err(CliError::Io)?;
            std::fs::write(
                format!("{base}_00_before.ir"),
                format!("{k}"),
            )
            .map_err(CliError::Io)?;

            let passes = metaltile_codegen::passes::PassRegistry::standard_with_names();
            for (pi, (name, pass)) in passes.iter().enumerate() {
                if let Err(e) = pass.run(&mut k) {
                    eprintln!("Pass {name} failed on [{i}] {}: {e}", node.label);
                    break;
                }
                let path = format!("{base}_{:02}_{name}.ir", pi + 1);
                std::fs::write(&path, format!("{k}")).map_err(CliError::Io)?;
                println!("wrote {path}");
            }

            // Final MSL
            let generator = MslGenerator::default();
            match generator.generate(&mut k) {
                Ok(msl) => {
                    let path = format!("{base}_final.msl");
                    std::fs::write(&path, &msl).map_err(CliError::Io)?;
                    println!("wrote {path}");
                }
                Err(e) => eprintln!("MSL error on [{i}] {}: {e}", node.label),
            }
        } else {
            println!("// ── [{:>3}] {}{} ─────────────────────", i, node.label, fuse_tag);
            println!("// ── BEFORE PASSES ───────────────────────────");
            println!("{k}");
            run_all_passes_and_print(&mut k);
        }
    }
    Ok(())
}

/// Run a single pass on each kernel and print the IR after it.
fn print_toml_pass_single(
    plan: &ExecutionPlan,
    pass_name: &str,
    _dtype: DType,
    dir: &Option<String>,
) -> Result<(), CliError> {
    for (i, kernel) in plan.cached_kernels.iter().enumerate() {
        let node = &plan.nodes[i];
        let mut k = kernel.clone();
        k.mode = node.mode;

        let fuse_tag = node
            .fuse_group
            .map(|g| format!(" [fuse={g}]"))
            .unwrap_or_default();

        match metaltile_codegen::passes::PassRegistry::get(pass_name) {
            Some(pass_obj) => {
                if let Err(e) = pass_obj.run(&mut k) {
                    eprintln!("Pass {pass_name} failed on [{i}] {}: {e}", node.label);
                    continue;
                }

                if let Some(d) = dir {
                    let safe_name: String = node
                        .label
                        .chars()
                        .map(|c| {
                            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                                c
                            } else {
                                '_'
                            }
                        })
                        .collect();
                    let filename = format!("{:03}_{}", i, safe_name);
                    let path = format!("{}/{}.after_{}.ir", d, filename, pass_name);
                    std::fs::create_dir_all(d).map_err(CliError::Io)?;
                    std::fs::write(&path, format!("{k}")).map_err(CliError::Io)?;
                    println!("wrote {path}");
                } else {
                    println!("// ── [{:>3}] {}{} ─────────────────────", i, node.label, fuse_tag);
                    println!("// ── AFTER {pass_name} ────────────────────────");
                    println!("{k}");

                    // If the pass was fusion, also indicate which ops were fused
                    if pass_name == "fusion" {
                        let n_fused = k
                            .body
                            .ops
                            .iter()
                            .filter(|op| matches!(op, metaltile_core::ir::Op::FusedElementwise { .. }))
                            .count();
                        if n_fused > 0 {
                            println!(
                                "//   → {n_fused} FusedElementwise chain{} created",
                                if n_fused == 1 { "" } else { "s" }
                            );
                        }
                    }
                }
            }
            None => {
                if i == 0 {
                    let valid: Vec<_> = metaltile_codegen::passes::PassRegistry::names();
                    eprintln!(
                        "Unknown pass: {pass_name}. Valid: {} all",
                        valid.join(", ")
                    );
                    return Err(CliError::Other(format!("unknown pass '{pass_name}'")));
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Param / fusion / dtype resolution
// ---------------------------------------------------------------------------

/// Resolved model parameters from CLI flags (with defaults for Llama 3 8B).
struct ResolvedParams {
    params: std::collections::HashMap<String, u32>,
}

/// Resolve model parameters from CLI flags, a config.json, or defaults.
fn resolve_model_params(args: &InspectArgs, _def: &ModelDef) -> Result<ResolvedParams, CliError> {
    // Try --config-json first (like the infer command)
    if let Some(config_path) = &args.config_json {
        return resolve_from_config_json(config_path);
    }

    // Otherwise use CLI flags with defaults (Llama 3 8B-ish)
    let mut params = std::collections::HashMap::new();
    params.insert("n_layers".to_string(), args.n_layers.unwrap_or(32));
    params.insert("n_heads".to_string(), args.n_heads.unwrap_or(32));
    params.insert("n_kv_heads".to_string(), args.n_kv_heads.unwrap_or(8));
    params.insert("head_dim".to_string(), args.head_dim.unwrap_or(128));
    params.insert("hidden_dim".to_string(), args.hidden_dim.unwrap_or(4096));
    params.insert("ffn_dim".to_string(), args.ffn_dim.unwrap_or(14336));
    params.insert("vocab_size".to_string(), args.vocab_size.unwrap_or(128256));
    params.insert("max_seq_len".to_string(), args.max_seq_len.unwrap_or(8192));
    Ok(ResolvedParams { params })
}

/// Resolve model parameters from a HuggingFace-style config.json.
fn resolve_from_config_json(path: &str) -> Result<ResolvedParams, CliError> {
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| {
            CliError::Other(format!("reading config.json: {e}"))
        })?)
        .map_err(|e| CliError::Other(format!("parsing config.json: {e}")))?;

    let mut params = std::collections::HashMap::new();

    // HuggingFace Llama config keys
    let get_u32 = |key: &str, fallback: u32| -> u32 {
        config
            .get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(fallback)
    };

    let n_layers = get_u32("num_hidden_layers", 32);
    let n_heads = get_u32("num_attention_heads", 32);
    let n_kv_heads = get_u32("num_key_value_heads", n_heads);
    let hidden_dim = get_u32("hidden_size", 4096);
    let head_dim = config
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .unwrap_or_else(|| hidden_dim / n_heads);
    let ffn_dim = get_u32("intermediate_size", 14336);
    let vocab_size = get_u32("vocab_size", 128256);
    let max_seq_len = get_u32("max_position_embeddings", 8192);

    params.insert("n_layers".to_string(), n_layers);
    params.insert("n_heads".to_string(), n_heads);
    params.insert("n_kv_heads".to_string(), n_kv_heads);
    params.insert("head_dim".to_string(), head_dim);
    params.insert("hidden_dim".to_string(), hidden_dim);
    params.insert("ffn_dim".to_string(), ffn_dim);
    params.insert("vocab_size".to_string(), vocab_size);
    params.insert("max_seq_len".to_string(), max_seq_len);

    eprintln!(
        "Config: {} layers, {} heads ({} kv), dim={}, ffn={}, vocab={}",
        n_layers, n_heads, n_kv_heads, hidden_dim, ffn_dim, vocab_size,
    );
    Ok(ResolvedParams { params })
}

/// Resolve activation dtype from CLI flags.
fn resolve_dtype(args: &InspectArgs) -> DType {
    args.dtype.as_deref().and_then(|s| DType::from_str(s).ok()).unwrap_or(DType::F16)
}

/// Resolve the fusion mode from CLI flags.
fn resolve_fusion_mode(args: &InspectArgs) -> FusionMode {
    if args.no_fuse {
        FusionMode::None
    } else if args.graph_fuse {
        FusionMode::GraphDriven
    } else {
        FusionMode::TomlDriven
    }
}

// ===========================================================================
// Original helpers (unchanged)
// ===========================================================================

/// Run all compilation passes and print IR after each stage.
fn run_all_passes_and_print(k: &mut metaltile_core::ir::Kernel) {
    use metaltile_codegen::msl::MslGenerator;

    let passes = metaltile_codegen::passes::PassRegistry::standard_with_names();

    for (name, pass) in &passes {
        if let Err(e) = pass.run(k) {
            println!("\n// ── AFTER {name} ──────── ERROR ──");
            println!("// {e}");
            return;
        }
        println!("\n// ── AFTER {name} ────────────────────────");
        println!("{k}");
    }

    // Generate final MSL
    let generator = MslGenerator::default();
    match generator.generate(k) {
        Ok(msl) => {
            println!("\n// ── FINAL MSL ───────────────────────────────");
            println!("{msl}");
        },
        Err(e) => {
            println!("\n// ── MSL ERROR ───────────────────────────────");
            println!("// {e}");
        },
    }
}

fn generate_msl(spec: &BenchSpec, dtypes: &[DType]) -> String {
    generate_msl_dt(spec, dtypes.first().copied().unwrap_or(DType::F32))
}

fn generate_msl_dt(spec: &BenchSpec, dt: DType) -> String {
    let mut k = (spec.kernel_ir)(dt);
    // Mirror bench-side mt_qmm_mma dtype-aware-skew patch so `tile inspect`
    // shows the same MSL the bench compiles.
    if spec.kernel_name == "mt_qmm_mma" {
        metaltile_std::mlx::quantized::patch_qmm_mma_dtype_aware_skew(&mut k, dt);
    }
    let mode = effective_mode(spec);
    k.mode = mode;
    let expected_tpg =
        spec.shapes.first().map(|s| s.tpg as u32).or_else(|| spec.dispatch.tpg_hint());
    generator_for_mode(mode, expected_tpg)
        .generate(&k)
        .unwrap_or_else(|e| format!("// ERROR: {e}\n"))
}

fn print_stats_table(stats: &[metaltile_codegen::passes::PassStats]) {
    println!(
        "{:<20}  {:>10}  {:>9}  {:>6}  {:>7}",
        "pass", "ops_before", "ops_after", "delta", "time_us"
    );
    println!("{:-<20}  {:->10}  {:->9}  {:->6}  {:->7}", "", "", "", "", "");
    for s in stats {
        let delta = s.ops_after as isize - s.ops_before as isize;
        let delta_str = if delta == 0 { "  +0".to_string() } else { format!("{:>+4}", delta) };
        println!(
            "{:<20}  {:>10}  {:>9}  {:>6}  {:>7}",
            s.name, s.ops_before, s.ops_after, delta_str, s.wall_us
        );
    }
}
