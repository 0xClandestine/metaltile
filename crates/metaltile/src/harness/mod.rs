//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Harness infrastructure for `#[bench]` and `#[test_kernel]` kernels.
//!
//! This module is the home of the bench/test setup types and the in-process
//! registries consumed by the `__tile_runner` subprocess.
//!
//! # Structure
//!
//! - [`bench`] — [`KernelBench`](bench::KernelBench) trait, [`BenchSetup`](bench::BenchSetup),
//!   [`BenchBuffer`](bench::BenchBuffer), [`RefKernel`](bench::RefKernel), [`KernelBenchEntry`](bench::KernelBenchEntry)
//! - [`test`]  — [`KernelTest`](test::KernelTest) trait, [`TestSetup`](test::TestSetup),
//!   [`TestBuffer`](test::TestBuffer), [`KernelTestEntry`](test::KernelTestEntry)
//! - [`registry`] — [`all_benches`](registry::all_benches), [`all_tests`](registry::all_tests),
//!   [`all_kernels`](registry::all_kernels) accessors

pub mod bench;
pub mod registry;
pub mod test;
