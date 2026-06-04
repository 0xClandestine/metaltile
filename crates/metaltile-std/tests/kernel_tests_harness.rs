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

    // Anchor the metaltile-std kernel translation units so the linker keeps
    // them in this test binary. `#[test_kernel]` registers each test via an
    // `inventory::submit!` static that lives in the *same* TU as the kernel
    // it covers. Under `--gc-sections` (macOS `-dead_strip`) the linker drops
    // every kernel TU that nothing in this binary references — taking the
    // submissions with it — so `all_tests()` would iterate zero entries and
    // this test would pass *vacuously*. Touching the kernel-IR registry (the
    // same anchor `kernel_registry_consistency` relies on) keeps the TUs
    // alive. The `tile` runner avoids this because it dispatches kernels
    // directly; this cargo bridge has no such reference of its own.
    let n_kernels = metaltile_std::all_kernels().count();
    assert!(
        n_kernels > 0,
        "all_kernels() is empty — the metaltile-std kernel object files were \
         stripped from this test binary, so the #[test_kernel] inventory \
         cannot be read",
    );

    let mut total = 0usize;
    let mut failures: Vec<String> = Vec::new();

    // NB: iterate via `metaltile_std::all_tests()` (not `metaltile::harness::
    // registry::all_tests()`). Per the `metaltile-std` lib docs, importing the
    // registry accessor through `metaltile_std` is what pulls the std rlib into
    // this integration-test link so the `#[test_kernel]` inventory statics are
    // retained — going through `metaltile::…` directly leaves them dead-code-
    // eliminated and the harness silently iterates an EMPTY set.
    for entry in metaltile_std::all_tests() {
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

    // Guard against silent regression: a populated kernel registry but an
    // empty test registry means the `#[test_kernel]` submissions stopped
    // linking (or registering) — the exact failure mode that let this
    // harness pass while exercising nothing.
    assert!(
        total > 0,
        "all_tests() iterated zero #[test_kernel] entries despite {n_kernels} \
         registered kernels — link / registration regression",
    );

    assert!(
        failures.is_empty(),
        "{}/{} #[test_kernel] checks failed:\n  {}",
        failures.len(),
        total,
        failures.join("\n  "),
    );
}
