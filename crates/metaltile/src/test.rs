//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Prelude for `kernel_tests` and `kernel_benches` modules.
//!
//! Import with `use metaltile::test::*;` at the top of a kernel file.
//! This single glob covers every type and macro needed to write both test
//! and bench setups — the only item intentionally absent is the `bench`
//! attribute macro, which conflicts with Rust's built-in `#[bench]` when
//! brought in through a glob and must be imported explicitly:
//!
//! ```rust,ignore
//! pub mod kernel_benches {
//!     use super::*;
//!     use metaltile::bench; // explicit — glob would be ambiguous with std
//!     ...
//! }
//! ```

pub use metaltile_core::{
    BenchBuffer,
    BenchSetup,
    DType,
    TestBuffer,
    TestSetup,
    bench::{Grid, RefKernel},
};
pub use metaltile_macros::test_kernel;
