//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `tile config` — Print resolved Tile.toml configuration.

use std::path::Path;

use crate::{
    CliError,
    project::config::TileConfig,
    term::{Color, Style, paint_stdout},
};

#[derive(clap::Args, Debug)]
pub struct ConfigArgs {
    /// Profile to display (default: $TILE_PROFILE or "default").
    #[arg(long = "profile")]
    pub profile: Option<String>,
}

impl ConfigArgs {
    pub fn run(&self) -> Result<(), CliError> {
        let args = self;
        let profile_env = std::env::var("TILE_PROFILE").ok();
        let profile_name =
            args.profile.as_deref().or_else(|| profile_env.as_deref()).unwrap_or("default");

        let tile_toml_path = Path::new("Tile.toml");
        if !tile_toml_path.exists() {
            return Err(CliError::Config("Tile.toml not found".into()));
        }

        let cfg = TileConfig::load(Path::new("."))
            .map_err(|e| CliError::Config(e))?
            .ok_or_else(|| CliError::Config("Tile.toml not found".into()))?;

        let profile = cfg.resolved_profile(profile_name);
        let bench = cfg.resolved_bench(profile_name);
        let tol = cfg.resolved_tol(profile_name);

        println!(
            "{} {} (from Tile.toml)",
            paint_stdout("tile config", Style::new().fg(Color::Cyan).bold()),
            paint_stdout(
                format!("· profile={}", profile_name),
                Style::new().fg(Color::BrightBlack)
            ),
        );
        println!("  {:19} {}", paint_stdout("src", style_key()), profile.src);
        println!("  {:19} {}", paint_stdout("test", style_key()), profile.test);
        println!("  {:19} {}", paint_stdout("bench", style_key()), profile.bench);
        println!("  {:19} {}", paint_stdout("out", style_key()), profile.out);
        println!("  {:19} {}", paint_stdout("baselines", style_key()), profile.baselines);
        println!("  {:19} {}", paint_stdout("dtypes", style_key()), profile.dtypes.join(", "));
        println!("  {:19} {}", paint_stdout("bench.n", style_key()), bench.n);
        println!("  {:19} {}", paint_stdout("bench.iters", style_key()), bench.iters);
        println!("  {:19} {:.4}", paint_stdout("tol.f32", style_key()), tol.f32);
        println!("  {:19} {:.4}", paint_stdout("tol.f16", style_key()), tol.f16);
        println!("  {:19} {:.4}", paint_stdout("tol.bf16", style_key()), tol.bf16);

        Ok(())
    }
}

fn style_key() -> Style { Style::new().fg(Color::BrightWhite).bold() }
