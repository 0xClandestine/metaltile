//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `tile clean` — Remove build artifacts and cache.

use std::path::Path;

use crate::{
    CliError,
    project::{air_cache_dir, resolve_out_dir},
    term::{Color, Style, paint_stdout},
};

#[derive(clap::Args, Debug)]
pub struct CleanArgs {
    /// Remove only the intermediate build cache, keeping emitted artifacts.
    #[arg(long = "cache-only")]
    pub cache_only: bool,
}

impl CleanArgs {
    pub fn run(&self) -> Result<(), CliError> {
        let args = self;
        let out_dir = resolve_out_dir();
        let out_path = Path::new(&out_dir);
        let air_dir = air_cache_dir();

        let removed = if args.cache_only {
            remove_dir(&air_dir, "air cache")?
                + remove_dir(&out_path.join(".cache"), "compile cache")?
        } else {
            remove_dir(out_path, &out_dir)? + remove_dir(&air_dir, "air cache")?
        };

        if removed == 0 {
            println!(
                "  {} nothing to clean",
                paint_stdout("✓", Style::new().fg(Color::Green).bold())
            );
        }
        Ok(())
    }
}

fn remove_dir(dir: &Path, label: &str) -> Result<u32, CliError> {
    if dir.exists() {
        std::fs::remove_dir_all(dir).map_err(CliError::Io)?;
        println!(
            "  {} {}",
            paint_stdout("✓", Style::new().fg(Color::Green).bold()),
            paint_stdout(
                format!("removed {label} ({})", dir.display()),
                Style::new().fg(Color::BrightWhite)
            ),
        );
        Ok(1)
    } else {
        Ok(0)
    }
}
