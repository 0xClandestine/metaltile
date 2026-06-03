//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `tile inspect` — Print IR and/or MSL for kernels.
//!
//! Spawns `__tile_runner inspect` as a subprocess. All kernel discovery and
//! MSL/IR generation happens in the runner process where `metaltile-std` is
//! linked.
//!
//! Usage:
//!   tile inspect                           # list all registered kernels
//!   tile inspect <kernel>                  # print final MSL (default)
//!   tile inspect <kernel> --ir             # print raw IR
//!   tile inspect <kernel> --stats          # print per-pass op-count table
//!   tile inspect -o /tmp/out               # write .metal file
//!   tile inspect --all -o /tmp/out         # dump every kernel to disk

use metaltile_core::protocol::{InspectKind, ProtocolMessage};

use crate::{
    CliError,
    InspectArgs,
    project_runner::{ProjectRunner, RunnerInvocation},
    term::{Color, Style, paint_stdout},
};

/// `TileCommand` wrapper for `tile inspect`.
pub struct InspectCommand<'a>(pub &'a InspectArgs);

impl<'a> super::TileCommand for InspectCommand<'a> {
    fn run(&self, harness: &crate::harness::Harness) -> Result<(), crate::CliError> {
        run(self.0, harness)
    }
}

pub fn run(args: &InspectArgs, harness: &crate::harness::Harness) -> Result<(), CliError> {
    let filter_val = args.filter_args.filter.as_ref().or(args.kernel.as_ref());
    let _span = tracing::info_span!(
        "inspect",
        filter = ?filter_val,
        ir = args.ir,
        stats = args.stats,
    )
    .entered();

    if args.pass.is_some() {
        eprintln!(
            "{} --pass requires the kernel registry to be linked in-process. \
             Use `tile inspect --ir` to view the final IR, or file a feature request.",
            paint_stdout("error:", Style::new().fg(Color::Red).bold()),
        );
        return Err(CliError::Other("--pass not supported in project mode".into()));
    }

    let kind_str = if args.ir {
        "ir"
    } else if args.stats {
        "stats"
    } else {
        "msl"
    };
    let filter = filter_val.cloned();
    // When no filter and no --all, we're in list-only mode.
    let list_only = filter.is_none() && !args.all;

    let inv = RunnerInvocation {
        command: "inspect".into(),
        filter: filter.clone(),
        dtype: args.dtype.clone(),
        inspect_kind: Some(kind_str.to_string()),
        ..Default::default()
    };

    let dir = args.dir.clone();
    let mut listed_names: Vec<String> = Vec::new();
    let mut got_any = false;
    let mut printed_list_header = false;

    let runner_ok = ProjectRunner::new(harness).run_streaming(&inv, |msg| match msg {
        ProtocolMessage::Start { total, .. } =>
            if list_only {
                if !printed_list_header {
                    println!(
                        "{}",
                        paint_stdout("tile inspect", Style::new().fg(Color::Cyan).bold())
                    );
                    println!();
                    printed_list_header = true;
                }
                let _ = total;
            },
        ProtocolMessage::Inspect { name, kind, content } => {
            got_any = true;
            if list_only {
                listed_names.push(name.clone());
            } else {
                let ext = match kind {
                    InspectKind::Msl => "metal",
                    InspectKind::Ir => "ir",
                    _ => "txt",
                };
                if let Some(d) = &dir {
                    let path = format!("{}/{}.{}", d, name, ext);
                    let _ = std::fs::create_dir_all(d);
                    match std::fs::write(&path, &content) {
                        Ok(()) => println!("wrote {path}"),
                        Err(e) => eprintln!("write {path}: {e}"),
                    }
                } else {
                    println!("// ═══════════════════════════════════════════════════════");
                    println!("// kernel: {name}  ({kind_str})");
                    println!("// ═══════════════════════════════════════════════════════");
                    println!("{content}");
                }
            }
        },
        ProtocolMessage::ProtocolError { name, dtype, message } => {
            eprintln!(
                "{} {} [{}]: {}",
                paint_stdout("error:", Style::new().fg(Color::Red).bold()),
                name,
                dtype,
                message,
            );
        },
        _ => {},
    });

    if list_only {
        for name in &listed_names {
            println!("  {}", paint_stdout(name, Style::new().fg(Color::Cyan).bold()));
        }
        if !listed_names.is_empty() {
            let sep = paint_stdout("·", Style::new().fg(Color::BrightBlack).dim());
            println!();
            println!(
                "  {} {sep} {}",
                paint_stdout(
                    format!("{} kernels", listed_names.len()),
                    Style::new().fg(Color::BrightBlack),
                ),
                paint_stdout("<kernel> for MSL", Style::new().fg(Color::BrightBlack)),
            );
        } else if runner_ok {
            eprintln!("No kernels registered.");
        }
        return Ok(());
    }

    if !got_any {
        let filter_desc = filter.as_deref().unwrap_or("<filter>");
        eprintln!(
            "{} no kernel matched '{filter_desc}'",
            paint_stdout("error:", Style::new().fg(Color::Red).bold()),
        );
        return Err(CliError::Other(format!("no kernel matched '{filter_desc}'")));
    }

    Ok(())
}
