//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `__tile_runner` entry point for the metaltile workspace.
//!
//! User projects get their own copy scaffolded by `tile init`. This copy
//! serves the metaltile workspace itself (e.g. `make bench` / `make test`).

// Force the linker to include all `inventory::submit!` statics from the
// metaltile-std library so that kernel/bench/test registrations are populated.
extern crate metaltile_std;

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
