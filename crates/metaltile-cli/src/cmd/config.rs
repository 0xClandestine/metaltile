//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `tile config` — display the effective merged configuration.
//!
//! Prints the fully-resolved `TileConfig` as TOML (default) or JSON
//! (`--json` / global `--json`), showing the result of merging:
//!   defaults → extends base → tile.toml → `[profiles.<name>]` → env vars

use crate::{
    CliError,
    ConfigArgs,
    harness::Harness,
    term::{Color, Style, paint_stderr},
};

pub fn run(args: &ConfigArgs, harness: &Harness) -> Result<(), CliError> {
    let json_out = args.json || harness.json_output();

    if json_out {
        let json = serde_json::to_string_pretty(&harness.config).map_err(CliError::Json)?;
        println!("{json}");
    } else {
        match harness.config.to_string_pretty() {
            Ok(toml) => println!("{toml}"),
            Err(e) => return Err(CliError::Other(format!("serialize config: {e}"))),
        }
    }

    // Show where tile.toml was found (helpful for debugging parent-dir walk).
    if let Some(path) = crate::config::ConfigLoader::find_tile_toml() {
        if !harness.is_quiet() && !json_out {
            eprintln!(
                "\n{}  {}",
                paint_stderr("tile.toml found at", Style::new().fg(Color::BrightBlack)),
                paint_stderr(path.display().to_string(), Style::new().fg(Color::BrightWhite)),
            );
            if let Some(p) = &harness.global.profile {
                eprintln!(
                    "{}  {}",
                    paint_stderr("active profile  ", Style::new().fg(Color::BrightBlack)),
                    paint_stderr(p, Style::new().fg(Color::Cyan)),
                );
            }
        }
    } else if !json_out {
        eprintln!(
            "{}",
            paint_stderr(
                "note: no tile.toml found; showing built-in defaults",
                Style::new().fg(Color::BrightBlack),
            ),
        );
    }

    Ok(())
}
