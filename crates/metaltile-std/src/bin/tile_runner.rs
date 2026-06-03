//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `__tile_runner` — subprocess entry point for the `tile` CLI.
//!
//! This binary must live in `metaltile-std` so that all `#[bench]` and
//! `#[test_kernel]` inventory statics are linked in. The `tile` CLI has no
//! direct dep on `metaltile-std`; it spawns this process and consumes its
//! `ProtocolMessage` JSON-lines output.

fn main() {
    let args = match metaltile::runner::RunnerArgs::from_env_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("__tile_runner: {e}");
            std::process::exit(2);
        },
    };
    let ok = metaltile::runner::RunnerHarness::run(&args);
    std::process::exit(if ok { 0 } else { 1 });
}
