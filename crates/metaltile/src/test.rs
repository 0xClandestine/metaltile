//! Author-facing surface for `#[bench]` / `#[test_kernel]` setups.
//!
//! Glob-import this in a kernel file to bring the builder types and `DType`
//! into scope:
//!
//! ```ignore
//! pub use metaltile::test::*;
//!
//! #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-6)]
//! fn test_my_kernel(dt: DType) -> TestSetup { /* ... */ }
//! ```
//!
//! These are re-exports of the canonical definitions in
//! [`metaltile_core::bench`]; importing them from here keeps kernel files
//! depending only on the `metaltile` umbrella crate.

pub use metaltile_core::{
    DType,
    bench::{
        BenchBuffer,
        BenchSetup,
        ConstValue,
        Grid,
        KernelBench,
        KernelTest,
        RefKernel,
        TestBuffer,
        TestSetup,
    },
    ir::KernelMode,
};
