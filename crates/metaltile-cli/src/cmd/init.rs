//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `tile init` — Bootstrap a new MetalTile project.

use std::path::PathBuf;

use crate::{
    CliError,
    term::{Color, Style, paint_stdout},
};

#[derive(clap::Args, Debug)]
pub struct InitArgs {
    /// Project name (default: my-kernels).
    pub name: Option<String>,
    /// Project template: kernel (default), library, swift.
    #[arg(long = "template", default_value = "kernel")]
    pub template: String,
    /// Skip git repository initialization.
    #[arg(long = "no-git")]
    pub no_git: bool,
    /// Generate .vscode/ settings for Metal shader development.
    #[arg(long = "vscode")]
    pub vscode: bool,
}

impl InitArgs {
    pub fn run(&self) -> Result<(), CliError> {
        let args = self;
        let name = args.name.clone().unwrap_or_else(|| "my-kernels".to_string());
        let project_dir = PathBuf::from(&name);

        if project_dir.exists() {
            return Err(CliError::Scaffold(format!("directory '{name}' already exists")));
        }

        let scaffold = ProjectScaffold::new(&name, &project_dir);
        scaffold.create()?;

        println!(
            "  {} {} {}",
            paint_stdout("✓", Style::new().fg(Color::Green).bold()),
            paint_stdout("Initialized", Style::new().fg(Color::Cyan).bold()),
            paint_stdout(&name, Style::new().fg(Color::BrightWhite)),
        );
        println!();
        println!("  {} cd {name}", paint_stdout("$", Style::new().fg(Color::BrightBlack).dim()));
        println!("  {} tile build", paint_stdout("$", Style::new().fg(Color::BrightBlack).dim()));
        println!("  {} tile test", paint_stdout("$", Style::new().fg(Color::BrightBlack).dim()));
        println!("  {} tile bench", paint_stdout("$", Style::new().fg(Color::BrightBlack).dim()));

        Ok(())
    }
}

// ── Project scaffold builder ─────────────────────────────────────────────

struct ProjectScaffold {
    files: Vec<(PathBuf, String)>,
}

impl ProjectScaffold {
    fn new(name: &str, project_dir: &PathBuf) -> Self {
        let kernels_dir = project_dir.join("kernels");
        let benches_dir = project_dir.join("benches");
        let tests_dir = project_dir.join("tests");

        let mut scaffold = Self { files: Vec::new() };

        scaffold = scaffold
            .file(project_dir.join("Tile.toml"), tile_toml(name))
            .file(project_dir.join("Cargo.toml"), cargo_toml(name))
            .file(kernels_dir.join("lib.rs"), kernels_lib_rs())
            .file(benches_dir.join("kernels.rs"), benches_kernels_rs(name))
            .file(tests_dir.join("vector_add.t.rs"), tests_vector_add_t_rs(name))
            .file(project_dir.join(".gitignore"), gitignore());

        scaffold
    }

    fn file(mut self, path: PathBuf, content: String) -> Self {
        self.files.push((path, content));
        self
    }

    fn create(&self) -> Result<(), CliError> {
        // Create all needed directories
        for (path, _) in &self.files {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(CliError::Io)?;
            }
        }
        // Write all files
        for (path, content) in &self.files {
            std::fs::write(path, content).map_err(CliError::Io)?;
        }
        Ok(())
    }
}

fn tile_toml(_name: &str) -> String {
    r#"[profile.default]
src       = "kernels"
test      = "tests"
bench     = "benches"
out       = "tile-out"
baselines = "baselines"
dtypes    = ["f32", "f16", "bf16"]

[bench]
n     = 67108864
iters = 10

[tol]
f32  = 1e-4
f16  = 1.5e-2
bf16 = 1.3e-1

[profile.ci.bench]
n     = 4194304
iters = 3

[profile.release.bench]
iters = 20
"#
    .to_string()
}

fn cargo_toml(name: &str) -> String {
    format!(
        r#"[package]
name = "{name}"
version = "0.1.0"
edition = "2024"
license = "Apache-2.0"

[dependencies]
metaltile = {{ git = "https://github.com/metaltile/metaltile" }}
metaltile-std = {{ git = "https://github.com/metaltile/metaltile" }}

[[bin]]
name = "{name}"
path = "benches/kernels.rs"
"#
    )
}

fn kernels_lib_rs() -> String {
    r#"use metaltile::prelude::*;

#[bench_kernel(
    op    = "vector_add",
    class = Binary,
    input = Signed,
    tol   = 1e-4,
)]
#[kernel]
pub fn mt_vector_add<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], load(a[idx]) + load(b[idx]));
}

pub fn bench_specs() -> &'static [&'static metaltile::BenchSpec] {
    &[&MT_VECTOR_ADD_SPEC]
}
"#
    .to_string()
}

fn benches_kernels_rs(name: &str) -> String { format!("tile_harness!({name}::bench_specs);\n") }

fn tests_vector_add_t_rs(name: &str) -> String {
    format!(
        r#"#[cfg(test)]
mod tests {{
    use metaltile::harness::GpuContext;
    use metaltile::harness::run_correctness_check;
    use metaltile_core::bench::types::DType;
    use {name}::bench_specs;

    #[test]
    fn vector_add_f32_correctness() {{
        let ctx = GpuContext::new().unwrap();
        let result = run_correctness_check(&ctx, bench_specs(), "vector_add", DType::F32);
        assert!(result.passed, "vector_add f32: {{result}}");
    }}

    #[test]
    fn vector_add_f16_correctness() {{
        let ctx = GpuContext::new().unwrap();
        let result = run_correctness_check(&ctx, bench_specs(), "vector_add", DType::F16);
        assert!(result.passed, "vector_add f16: {{result}}");
    }}

    #[test]
    fn vector_add_bf16_correctness() {{
        let ctx = GpuContext::new().unwrap();
        let result = run_correctness_check(&ctx, bench_specs(), "vector_add", DType::BF16);
        assert!(result.passed, "vector_add bf16: {{result}}");
    }}
}}
"#
    )
}

fn gitignore() -> String { "target/\ntile-out/\n".to_string() }
