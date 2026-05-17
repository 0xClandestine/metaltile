//! `tile profile` — Estimate GPU occupancy and register pressure for kernels.
//!
//! Runs the standard optimization pipeline followed by liveness analysis and
//! occupancy estimation across a sweep of threadgroup sizes. Reports the
//! optimal threadgroup size and the limiting bottleneck.
//!
//! Usage:
//!   tile profile                        # profile all kernels, compact table
//!   tile profile <kernel>               # profile one kernel, verbose
//!   tile profile <kernel> --sweep       # show full per-size sweep
//!   tile profile --filter <glob>        # filter by name substring

use std::collections::BTreeMap;

use metaltile_codegen::passes::{
    self,
    occupancy::{self, Bottleneck},
};
use metaltile_std::{bench_types::DType, spec::BenchSpec};

use crate::{
    flag_present,
    flag_val,
    kernel_utils::first_mode,
    matches_filter,
    positional,
    term::{Color, Style, paint_stdout},
};

/// Threadgroup sizes to sweep (total threads).
const TG_SWEEP: &[u32] = &[64, 128, 256, 512, 1024];

pub fn run(args: &[String]) {
    let filter = flag_val(args, "--filter").or_else(|| positional(args));
    let sweep_flag = flag_present(args, "--sweep");

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
    let matched: Vec<_> = if let Some(ref f) = filter {
        sorted.iter().filter(|(name, _)| matches_filter(Some(f), name)).collect()
    } else {
        sorted.iter().collect()
    };

    if matched.is_empty() {
        eprintln!(
            "{} {}",
            paint_stdout("error:", Style::new().fg(Color::Red).bold()),
            paint_stdout(
                format!("no kernel matched '{}'", filter.as_deref().unwrap_or("")),
                Style::new().fg(Color::BrightWhite),
            ),
        );
        return;
    }

    let single = matched.len() == 1;

    // Header.
    if !sweep_flag || single {
        println!();
        println!(
            "{}",
            paint_stdout("MetalTile Occupancy Profile", Style::new().fg(Color::BrightWhite).bold(),),
        );
        println!(
            "{}  max-threads/tg=1024  tg-mem=32KB  max-regs=128",
            paint_stdout("GPU limits:", Style::new().fg(Color::BrightBlack),),
        );
        println!();
    }

    if single {
        // Verbose mode: full sweep for one kernel.
        let (name, (spec, dtypes)) = matched[0];
        let dt = dtypes.first().copied().unwrap_or(DType::F32);
        let mut k = (spec.kernel_ir)(dt);
        k.mode = first_mode(spec);

        // Run the pipeline.
        if let Err(e) = passes::run_passes(&mut k, &passes::standard_pipeline()) {
            eprintln!("Pipeline failed: {e}");
            return;
        }

        // Register estimate.
        let reg_est = passes::register_estimate::estimate_registers(&k);
        println!("  kernel     {}", paint_stdout(*name, Style::new().fg(Color::Cyan).bold()),);
        println!(
            "  max-live   {}",
            paint_stdout(format!("{}", reg_est.max_live), Style::new().fg(Color::BrightWhite)),
        );
        println!(
            "  regs/thr   {}  (heuristic: max_live × 1.5)",
            paint_stdout(
                format!("{}", reg_est.regs_per_thread),
                Style::new().fg(Color::BrightWhite),
            ),
        );
        println!();

        if sweep_flag {
            // Full per-size table.
            println!("  {:<10} {:>6}  {:>6}  {:<22}", "tg_size", "occ%", "max_tgs", "bottleneck");
            println!("  {:-<10} {:-<6}  {:-<6}  {:-<22}", "", "", "", "");
            for &tg_size in TG_SWEEP {
                let est = occupancy::estimate_occupancy(&k, tg_size, None);
                let pct = paint_stdout(
                    format!("{:5.1}", est.occupancy_pct),
                    if est.occupancy_pct >= 80.0 {
                        Style::new().fg(Color::Green)
                    } else if est.occupancy_pct >= 50.0 {
                        Style::new().fg(Color::Yellow)
                    } else {
                        Style::new().fg(Color::Red)
                    },
                );
                let tgs = est.max_tgs_per_cu.map(|n| n.to_string()).unwrap_or_else(|| "—".into());
                let bn = bottle_label(est.bottleneck);
                println!("  {:<10} {}  {:>6}  {}", tg_size, pct, tgs, bn);
            }
        }

        // Best pick.
        println!();
        let candidates: Vec<_> = TG_SWEEP.iter().map(|&s| (s, None)).collect();
        if let Some((best_tg, best_est)) = occupancy::best_threadgroup_size(&k, &candidates) {
            println!(
                "  {}  tg_size={}  occ={}%  bottleneck={}",
                paint_stdout("best →", Style::new().fg(Color::Green).bold()),
                paint_stdout(format!("{}", best_tg), Style::new().fg(Color::Cyan).bold(),),
                paint_stdout(
                    format!("{:.1}", best_est.occupancy_pct),
                    Style::new().fg(Color::BrightWhite).bold(),
                ),
                bottle_label(best_est.bottleneck),
            );
        }
    } else {
        // Compact table: best result per kernel.
        println!(
            "  {:<24} {:>5}  {:>6}  {:>7}  {}",
            "kernel", "tg", "occ%", "regs/th", "bottleneck"
        );
        println!("  {:-<24} {:-<5}  {:-<6}  {:-<7}  {:-<22}", "", "", "", "", "");

        for (name, (spec, dtypes)) in &matched {
            let dt = dtypes.first().copied().unwrap_or(DType::F32);
            let mut k = (spec.kernel_ir)(dt);
            k.mode = first_mode(spec);

            if let Err(e) = passes::run_passes(&mut k, &passes::standard_pipeline()) {
                println!(
                    "  {:<24} {}",
                    paint_stdout(*name, Style::new().fg(Color::Cyan)),
                    paint_stdout(format!("pipeline error: {e}"), Style::new().fg(Color::Red),),
                );
                continue;
            }

            let reg_est = passes::register_estimate::estimate_registers(&k);
            let candidates: Vec<_> = TG_SWEEP.iter().map(|&s| (s, None)).collect();
            let (best_tg, best_est) = occupancy::best_threadgroup_size(&k, &candidates)
                .unwrap_or((0, occupancy::estimate_occupancy(&k, 256, None)));

            let pct = paint_stdout(
                format!("{:5.1}", best_est.occupancy_pct),
                if best_est.occupancy_pct >= 80.0 {
                    Style::new().fg(Color::Green)
                } else if best_est.occupancy_pct >= 50.0 {
                    Style::new().fg(Color::Yellow)
                } else {
                    Style::new().fg(Color::Red)
                },
            );

            println!(
                "  {:<24} {:>5}  {}  {:>7}  {}",
                paint_stdout(*name, Style::new().fg(Color::Cyan)),
                best_tg,
                pct,
                reg_est.regs_per_thread,
                bottle_label(best_est.bottleneck),
            );
        }

        println!();
        println!(
            "  {}",
            paint_stdout(
                format!("{} kernels profiled", matched.len()),
                Style::new().fg(Color::BrightBlack),
            ),
        );
        println!(
            "  {}",
            paint_stdout(
                "'tile profile <kernel>' for detailed view.",
                Style::new().fg(Color::BrightBlack),
            ),
        );
        if !sweep_flag {
            println!(
                "  {}",
                paint_stdout(
                    "'tile profile <kernel> --sweep' for per-tg-size breakdown.",
                    Style::new().fg(Color::BrightBlack),
                ),
            );
        }
    }
    println!();
}

fn bottle_label(bn: Bottleneck) -> String {
    match bn {
        Bottleneck::RegisterLimited =>
            paint_stdout("register-limited", Style::new().fg(Color::Yellow)).to_string(),
        Bottleneck::MemoryLimited =>
            paint_stdout("memory-limited", Style::new().fg(Color::Magenta)).to_string(),
        Bottleneck::ThreadLimited =>
            paint_stdout("thread-limited", Style::new().fg(Color::Green)).to_string(),
    }
}
