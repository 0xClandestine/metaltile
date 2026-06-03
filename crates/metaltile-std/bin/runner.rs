//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `__tile_runner` entry point for the metaltile workspace.
//!
//! User projects get their own copy scaffolded by `tile init`. This copy
//! serves the metaltile workspace itself (e.g. `make bench` / `make test`).
fn main() {
    let args = match metaltile::runner::RunnerArgs::from_env_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("__tile_runner: {e}");
            std::process::exit(2);
        },
    };
    std::process::exit(if metaltile::runner::RunnerHarness::run(&args) { 0 } else { 1 });
}
