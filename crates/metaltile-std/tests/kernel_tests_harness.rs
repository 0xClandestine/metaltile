//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Cargo bridge for new-syntax (`#[test_kernel]`) correctness tests.
//!
//! Iterates the `KernelTest` inventory and runs every registered test on the
//! GPU through the shared in-process runner, asserting each passes within its
//! tolerance. This makes the new test path part of `cargo test --workspace`
//! (the commit gate) without requiring `tile test` — this harness is the
//! replacement for the former `tests/*_gpu_correctness.rs` files (removed in
//! #240; per-kernel coverage now lives in in-source `#[test_kernel]`s).
//!
//! macOS-gated; shares the global `gpu_lock` so it serialises with the other
//! GPU integration tests.

#![cfg(target_os = "macos")]

mod common;

use common::gpu_lock;
use metaltile::runner::run_kernel_test;
use metaltile_runtime::Context;

#[test]
fn all_registered_kernel_tests_pass() {
    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new on macOS");

    let mut total = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for entry in metaltile::harness::registry::all_tests() {
        let t = entry.test();
        for &dt in t.dtypes() {
            total += 1;
            let setup = t.setup(dt);
            let tol = t.tolerance(dt);
            match run_kernel_test(&ctx, &setup, tol) {
                Ok(o) if o.passed => {},
                Ok(o) => failures.push(format!(
                    "{} [{dt}]: max|Δ|={:.3e} > tol {:.3e} (n_checked={})",
                    t.name(),
                    o.max_abs_err,
                    tol,
                    o.n_checked,
                )),
                Err(e) => failures.push(format!("{} [{dt}]: {e}", t.name())),
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{}/{} #[test_kernel] checks failed:\n  {}",
        failures.len(),
        total,
        failures.join("\n  "),
    );
}
