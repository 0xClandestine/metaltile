//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `tile device` — Show GPU device info and supported feature flags.

use metaltile_core::GpuFamily;
use metaltile_runtime::runner::GpuRunner;

use crate::{
    CliError,
    term::{Color, Style, paint_stdout},
};

#[derive(clap::Args, Debug)]
pub struct DeviceArgs {
    /// Print device info as JSON.
    #[arg(long = "json")]
    pub json: bool,
}

impl DeviceArgs {
    pub fn run(&self) -> Result<(), CliError> {
        let args = self;
        let _span = tracing::info_span!("device", json = args.json).entered();

        let runner = match GpuRunner::new() {
            Ok(r) => r,
            Err(e) => {
                if args.json {
                    println!("{{\"error\":{:?}}}", e);
                    return Ok(());
                }
                return Err(CliError::GpuInit(e));
            },
        };

        let device_name = &runner.device_name;
        let simd = runner.supports_simd_matrix();
        let gpu_family = GpuFamily::from_device_name(device_name);
        let apple9_or_later = gpu_family.is_apple9_or_later();
        let tpg_mem = gpu_family.threadgroup_mem_kb();
        let max_tpg = gpu_family.max_threads_per_threadgroup();

        if args.json {
            println!(
                "{{\"device\":{:?},\"gpu_family\":{:?},\"simdgroup_hw\":{},\"native_bfloat\":{},\"threadgroup_mem_kb\":{},\"max_tpg\":{}}}",
                device_name,
                gpu_family.code().unwrap_or("unknown"),
                simd,
                apple9_or_later,
                tpg_mem,
                max_tpg,
            );
            return Ok(());
        }

        let label_style = Style::new().fg(Color::BrightBlack).bold();

        println!("{}", paint_stdout("tile device", Style::new().fg(Color::Cyan).bold()));
        println!();
        println!(
            "  {}  {}",
            paint_stdout(format!("{:<16}", "Device"), label_style),
            paint_stdout(device_name, Style::new().fg(Color::BrightWhite)),
        );
        println!(
            "  {}  {}",
            paint_stdout(format!("{:<16}", "GPU family"), label_style),
            paint_stdout(gpu_family.display_label(), Style::new().fg(Color::BrightWhite)),
        );
        println!("  {}", paint_stdout("─".repeat(42), Style::new().fg(Color::BrightBlack).dim()));

        let check = |label: &str, supported: bool, note: &str| {
            let sym = if supported {
                paint_stdout("✓", Style::new().fg(Color::Green).bold())
            } else {
                paint_stdout("✗", Style::new().fg(Color::Red).bold())
            };
            println!(
                "  {}  {sym}   {}",
                paint_stdout(format!("{label:<16}"), label_style),
                paint_stdout(note, Style::new().fg(Color::BrightBlack).dim()),
            );
        };

        check("native_bfloat", apple9_or_later, "Metal 3.1+ bfloat type");
        check("simdgroup_hw", simd, "simdgroup matrix multiply");
        check("async_copy", apple9_or_later, "async threadgroup copy (M3+)");

        println!("  {}", paint_stdout("─".repeat(42), Style::new().fg(Color::BrightBlack).dim()));

        println!(
            "  {}  {}",
            paint_stdout(format!("{:<16}", "Threadgroup"), label_style),
            paint_stdout(format!("{tpg_mem} KB"), Style::new().fg(Color::BrightWhite)),
        );
        println!(
            "  {}  {}",
            paint_stdout(format!("{:<16}", "Max TPG"), label_style),
            paint_stdout(format!("{max_tpg}"), Style::new().fg(Color::BrightWhite)),
        );
        println!(
            "  {}  {}",
            paint_stdout(format!("{:<16}", "SLC"), label_style),
            paint_stdout(GpuFamily::slc_label(device_name), Style::new().fg(Color::BrightWhite)),
        );
        println!();
        Ok(())
    }
}
