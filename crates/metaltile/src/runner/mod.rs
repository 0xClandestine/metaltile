//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Runner infrastructure for the `__tile_runner` subprocess.
//!
//! This module is the GPU-side half of the subprocess architecture.  The
//! `tile` CLI never imports anything from here — it only reads the
//! JSON Lines produced by the runner binary.
//!
//! # Sub-modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`gpu`] | [`GpuRunner`], [`CompiledKernel`], [`GpuBuffer`], timing helpers |
//! | [`emit`] | Write [`ProtocolMessage`] lines to stdout |
//! | [`args`] | [`RunnerArgs`] — arg-parsing for the subprocess |
//! | [`harness`] | [`RunnerHarness`] — orchestrates bench/test/build/inspect |
//! | [`profile`] | CPU occupancy / register / bottleneck estimation |
//! | [`device_specs`] | Per-GPU peak bandwidth + compute specs for roofline |

pub mod args;
pub mod device_specs;
pub mod emit;
pub mod gpu;
pub mod harness;
pub mod profile;

pub use args::{RunnerArgs, RunnerCommand};
pub use emit::{emit, emit_stdout};
pub use gpu::{
    BENCH_ITERS,
    BENCH_WARMUP,
    BenchStats,
    CompiledKernel,
    GpuBuffer,
    GpuRunner,
    bench_gbps,
    bench_gbps_only,
    bench_gbps_with,
    buffer_typed,
    elem_bytes,
    read_typed,
    run_f16_once_as_f32,
    run_typed_once,
    to_gbps,
    to_gflops,
    zeros_typed,
};
pub use harness::{RunnerHarness, TestOutcome, run_kernel_test};
// Re-export GpuFamily so CLI crates can reach it without a direct
// metaltile-runtime dep.  Device-capability queries go through here.
pub use metaltile_runtime::GpuFamily;
