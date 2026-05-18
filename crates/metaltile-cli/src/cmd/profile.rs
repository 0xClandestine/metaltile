//! `tile profile` — Estimate GPU occupancy and register pressure for kernels.
//!
//! Runs the standard optimization pipeline followed by liveness analysis and
//! occupancy estimation across a sweep of threadgroup sizes. Reports the
//! optimal threadgroup size and the limiting bottleneck.
//!
//! Usage:
//!   tile profile                        # all kernels — per-kernel sub-tables
//!   tile profile <kernel>               # one kernel — full sweep table
//!   tile profile <kernel> --sweep       # show full per-size sweep
//!   tile profile --filter <glob>        # filter by name substring

use std::collections::BTreeMap;

use metaltile_codegen::passes::{
    self,
    occupancy::{self, Bottleneck},
};
use metaltile_std::{bench_types::DType, spec::BenchSpec};

use crate::{
    ProfileArgs,
    matches_filter,
    term::{Color, Style, paint_stdout},
};

/// Threadgroup sizes to sweep (total threads).
const TG_SWEEP: &[u32] = &[64, 128, 256, 512, 1024];

pub fn run(args: &ProfileArgs) {
    let filter = args.filter.as_ref().or(args.kernel.as_ref());
    let sweep_flag = args.sweep;

    // Collect all specs.
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
        return;
    }

    let mut sorted: Vec<(&str, (&BenchSpec, Vec<DType>))> = kernels.into_iter().collect();
    sorted.sort_unstable_by_key(|(name, _)| *name);

    // Apply filter if given.
    let matched: Vec<_> = if let Some(f) = filter {
        sorted.iter().filter(|(name, _)| matches_filter(Some(f), name)).collect()
    } else {
        sorted.iter().collect()
    };

    if matched.is_empty() {
        eprintln!(
            "{} {}",
            paint_stdout("error:", Style::new().fg(Color::Red).bold()),
            paint_stdout(
                format!("no kernel matched '{}'", filter.map_or("", |v| v)),
                Style::new().fg(Color::BrightWhite),
            ),
        );
        return;
    }

    let single = matched.len() == 1;

    if single {
        let (name, (spec, dtypes)) = matched[0];
        let dt = dtypes.first().copied().unwrap_or(DType::F32);
        let mut k = (spec.kernel_ir)(dt);
        k.mode = spec.dispatch.default_mode(spec.shapes);

        if let Err(e) = passes::run_passes(&mut k, &passes::standard_pipeline()) {
            eprintln!("Pipeline failed: {e}");
            return;
        }

        print_single_kernel(name, &k, sweep_flag);
    } else {
        print_multi_kernel(&matched);
    }
    println!();
}

// ── Single-kernel (verbose) mode ──────────────────────────────────────

fn print_single_kernel(name: &str, k: &metaltile_core::ir::Kernel, show_sweep: bool) {
    let reg_est = passes::register_estimate::estimate_registers(k);
    let bold = Style::new().fg(Color::BrightWhite).bold();
    let dim = Style::new().fg(Color::BrightBlack).dim();

    // Banner.
    println!(
        "{}  {}",
        paint_stdout("tile profile", Style::new().fg(Color::Cyan).bold()),
        paint_stdout(format!("{name} · max-tg=1024  tg-mem=32KB"), dim),
    );
    println!();

    // Kernel-level stats.
    println!(
        "  max-live    {}",
        paint_stdout(format!("{}", reg_est.max_live), bold),
    );
    println!(
        "  regs/thr    {}  (heuristic: max_live × 1.5)",
        paint_stdout(format!("{}", reg_est.regs_per_thread), bold),
    );
    println!();

    if show_sweep {
        // Full per-size sub-table.
        let sep = paint_stdout("│", dim);

        println!(
            "  {}  {} {} {} {} {} {}",
            paint_stdout("tg_size", bold),
            sep,
            paint_stdout(" occ%", bold),
            sep,
            paint_stdout("~max_tgs", bold),
            sep,
            paint_stdout("bottleneck", bold),
        );
        println!("  {}", paint_stdout("───────  ─────  ────────  ─────────", dim));

        for &tg_size in TG_SWEEP {
            let est = occupancy::estimate_occupancy(k, tg_size, None);
            let pct = paint_stdout(
                format!("{:5.1}", est.occupancy_pct),
                occ_color(est.occupancy_pct),
            );
            let tgs = est.max_tgs_per_cu.map(|n| format!("~{n}")).unwrap_or_else(|| "—".into());
            let tgs = paint_stdout(format!("{tgs:>8}"), bold);
            let bn = bottle_label(est.bottleneck);
            let tg_size = tg_size;
            println!("  {tg_size:>7}  {sep}  {pct}  {sep}  {tgs}  {sep}  {bn}");
        }
        println!();
    }

    // Best pick.
    let candidates: Vec<_> = TG_SWEEP.iter().map(|&s| (s, None)).collect();
    if let Some((best_tg, best_est)) = occupancy::best_threadgroup_size(k, &candidates) {
        println!(
            "  {}  tg_size={}  occ={}%  bottleneck={}",
            paint_stdout("best →", Style::new().fg(Color::Green).bold()),
            paint_stdout(format!("{best_tg}"), Style::new().fg(Color::Cyan).bold()),
            paint_stdout(format!("{:.1}", best_est.occupancy_pct), bold),
            bottle_label(best_est.bottleneck),
        );
    }
}

/// Wraps a profile result for a single kernel in multi-kernel mode.
struct ProfileRow<'a> {
    name: &'a str,
    best_tg: u32,
    occ_pct: f64,
    regs_per_thread: usize,
    bottleneck: Bottleneck,
}

// ── Multi-kernel (compact) mode ──────────────────────────────────────

fn print_multi_kernel(matched: &[&(&str, (&BenchSpec, Vec<DType>))]) {
    let dim = Style::new().fg(Color::BrightBlack).dim();
    let bold = Style::new().fg(Color::BrightWhite).bold();

    // Banner.
    println!(
        "{}  {}",
        paint_stdout("tile profile", Style::new().fg(Color::Cyan).bold()),
        paint_stdout("max-tg=1024  tg-mem=32KB", dim),
    );
    println!();

    // Collect all rows, skip pipeline failures.
    let mut rows: Vec<ProfileRow> = Vec::new();
    let mut errors: usize = 0;
    for (name, (spec, dtypes)) in matched {
        let dt = dtypes.first().copied().unwrap_or(DType::F32);
        let mut k = (spec.kernel_ir)(dt);
        k.mode = spec.dispatch.default_mode(spec.shapes);

        if let Err(_e) = passes::run_passes(&mut k, &passes::standard_pipeline()) {
            errors += 1;
            continue;
        }

        let reg_est = passes::register_estimate::estimate_registers(&k);
        let candidates: Vec<_> = TG_SWEEP.iter().map(|&s| (s, None)).collect();
        let (best_tg, best_est) = occupancy::best_threadgroup_size(&k, &candidates)
            .unwrap_or((0, occupancy::estimate_occupancy(&k, 256, None)));

        rows.push(ProfileRow {
            name,
            best_tg,
            occ_pct: best_est.occupancy_pct,
            regs_per_thread: reg_est.regs_per_thread,
            bottleneck: best_est.bottleneck,
        });
    }

    // Compute column widths.
    let name_w = rows.iter().map(|r| r.name.len()).max().unwrap_or(6).max(12);
    let sep = paint_stdout("│", dim);

    // Column headers.
    println!(
        "  {}  {} {} {} {} {} {} {} {}",
        paint_stdout(&pad_right("kernel", name_w), bold),
        sep,
        paint_stdout("tg_size", bold),
        sep,
        paint_stdout(" occ%", bold),
        sep,
        paint_stdout("regs/th", bold),
        sep,
        paint_stdout("bottleneck", bold),
    );

    // Separator.
    let tg_w = 7;
    let occ_w = 6;
    let reg_w = 7;
    let bn_w = 22;
    let total = 2 + name_w + 2 + tg_w + 3 + occ_w + 3 + reg_w + 3 + bn_w;
    println!("  {}", paint_stdout("─".repeat(total), dim));

    // Data rows.
    for row in &rows {
        let name_cell = paint_stdout(&pad_right(row.name, name_w), Style::new().fg(Color::Cyan));
        let tg_cell = format!("{:>7}", row.best_tg);
        let occ_cell = paint_stdout(format!("{:>6.1}", row.occ_pct), occ_color(row.occ_pct));
        let reg_cell = paint_stdout(format!("{:>7}", row.regs_per_thread), bold);
        let bn_cell = bottle_label(row.bottleneck);
        println!("  {name_cell}  {sep}  {tg_cell}  {sep}  {occ_cell}  {sep}  {reg_cell}  {sep}  {bn_cell}");
    }

    // Footer.
    let dot = paint_stdout("·", dim);
    println!();
    let tot = rows.len();
    if errors > 0 {
        println!(
            "  {} {dot} {} {dot} {} {dot} {}",
            paint_stdout(format!("{tot} kernels"), dim),
            paint_stdout(format!("{errors} errors"), Style::new().fg(Color::Red)),
            paint_stdout("'tile profile <kernel>' for detail", dim),
            paint_stdout("--sweep for breakdown", dim),
        );
    } else {
        println!(
            "  {} {dot} {} {dot} {}",
            paint_stdout(format!("{tot} kernels"), dim),
            paint_stdout("'tile profile <kernel>' for detail", dim),
            paint_stdout("--sweep for breakdown", dim),
        );
    }
}

// ── Helpers ────────────────────────────────────────────────────────

fn pad_right(text: &str, width: usize) -> String {
    format!("{text:<width$}")
}

fn occ_color(pct: f64) -> Style {
    if pct >= 80.0 {
        Style::new().fg(Color::Green)
    } else if pct >= 50.0 {
        Style::new().fg(Color::Yellow)
    } else {
        Style::new().fg(Color::Red)
    }
}

fn bottle_label(bn: Bottleneck) -> String {
    let style = match bn {
        Bottleneck::RegisterLimited => Style::new().fg(Color::Yellow),
        Bottleneck::MemoryLimited => Style::new().fg(Color::Magenta),
        Bottleneck::CachePressure => Style::new().fg(Color::Magenta),
        Bottleneck::ThreadLimited => Style::new().fg(Color::Green),
    };
    paint_stdout(&bn.to_string(), style)
}
