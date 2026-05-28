# metaltile-std kernel file conventions

Every file under `src/mlx/` and `src/ffai/` follows the same three-section
layout.  Keep the sections in order and separated by a blank line.

---

## 1 — Kernel definitions (file body)

Kernel functions live at the top of the file, outside any module.

```rust
//! Copyright 2026 …
//! SPDX-License-Identifier: Apache-2.0
//! One-line description of what this file contains.

use metaltile::kernel;

#[kernel]
pub fn mt_exp<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], exp(load(a[idx])));
}

#[kernel]
pub fn mt_log<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], log(load(a[idx])));
}
```

Rules:
- One `use metaltile::kernel;` at the top — the proc-macro prelude brings
  `Tensor`, `program_id`, `store`, `load`, and all builtins into scope.
- Kernels are generic over `T` unless a concrete dtype is required.
- No bench or test imports here; keep the top-level clean.

---

## 2 — `pub mod kernel_tests`

Correctness tests live in a single sub-module.  The module is `pub` so the
test runner discovers it via `metaltile-std`'s public surface.

All imports are declared with `pub use` at the **file top** so that the inner
modules need only `use super::*` — no per-module import lists.

```rust
// ── file top (section 1) ──────────────────────────────────────────────────────
use metaltile::kernel;
pub use metaltile::test::*;         // test_kernel + DType + TestBuffer/Setup + BenchBuffer/Setup
pub use crate::utils::{pack_f32, scalar_bytes};

// … kernel definitions …

// ── section 2 ────────────────────────────────────────────────────────────────
pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]
    use super::*;

    /// Keep N small (≤ 8 192) — tests run on every CI push.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-3)]
    fn test_exp(dt: DType) -> TestSetup {
        let n = 512usize;
        let inp: Vec<f32> = (0..n).map(|i| (i % 32) as f32 * 0.1 - 1.6).collect();
        let exp_out: Vec<f32> = inp.iter().map(|&v| v.exp()).collect();
        TestSetup::new(mt_exp::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a",   pack_f32(&inp,     dt), dt))
            .expect(TestBuffer::from_vec("out", pack_f32(&exp_out, dt), dt))
            .grid_1d(n, 1024)
    }
}
```

Rules:
- `#![allow(unused, dead_code, clippy::too_many_arguments)]` at the top of
  the module — auto-generated impls trigger these lints.
- `use super::*` is the only import line needed — it pulls in the `pub use`
  re-exports declared at the file top, plus the kernel modules (`mt_exp`, …).
- `dtypes = [f32]` / `[f16, bf16]` / `[f32, f16, bf16]` — list only the
  dtypes actually tested; the macro expands one entry per dtype.
- `tol` is the max element-wise absolute error. Three forms are accepted:
  - **Scalar** `tol = 1e-4` — same threshold for every dtype.
  - **Array** `tol = [1e-6, 1e-3, 1e-2]` — one value per dtype, same order as `dtypes`.
  - **Table** `tol = { f32: 1e-6, f16: 1e-3, bf16: 1e-2 }` — keyed by dtype name.
  Typical values:
  - f32 exact copies: `1e-6`
  - f32 transcendentals: `1e-4`
  - f16: `1e-3`
  - bf16: `1e-2`
- Keep `N ≤ 8_192` in tests.  The runtime pools `MTLBuffer` allocations
  across dispatches so there are no per-test alloc/free cycles, but very
  large pool buckets still stress the unified DRAM bandwidth shared between
  the GPU and the display compositor on Apple Silicon.  Small N → small
  pool buckets → no display pressure.
- Use deterministic, bounded inputs (not `TestBuffer::random`) so failures
  reproduce reliably.
- Use `compare_against(ref_setup)` when comparing a new kernel against an
  existing reference kernel rather than computing CPU expected values.  See
  the section below for the full pattern.

---

## 3 — `pub mod kernel_benches`

Performance benchmarks live in a second sub-module.

```rust
pub mod kernel_benches {
    #![allow(unused, dead_code, clippy::too_many_arguments)]
    use super::*;
    use metaltile::bench; // explicit: `bench` conflicts with std's built-in #[bench] via glob

    /// Bench sizes should match MLX for a fair side-by-side GB/s comparison.
    /// `name` maps to `"op/subop"` in the baseline JSON (baselines/*.json).
    #[bench(name = "unary/exp", dtypes = [f32, f16, bf16])]
    fn bench_exp(dt: DType) -> BenchSetup {
        let n = 64 * 1024 * 1024usize;   // 64 M elements — matches MLX default
        BenchSetup::new(mt_exp::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("a",   n, dt))
            .buffer(BenchBuffer::zeros("out",  n, dt).output())
            .grid_1d(n, 1024)
    }
}
```

Rules:
- `#![allow(unused, dead_code, clippy::too_many_arguments)]` at the top.
- Bench setup is written **inline** — do not call `crate::mlx::benches::*`
  helpers.  The `benches/` sub-module is being removed; write the
  `BenchSetup` construction directly here.
- `name` must match the `"op/subop"` field in `baselines/apple-*.json` so
  the CLI can display the MLX reference GB/s alongside the MetalTile result.
- Use `BenchBuffer::random` for inputs, `BenchBuffer::zeros(...).output()`
  for write-back buffers.
- Mark outputs with `.output()` so the runner reads them back and the
  runtime-computed bandwidth is accurate.
- Use `grid_1d(n, tpg)` for elementwise, `grid_3d(rows, 1, 1, [tpg,1,1])`
  for row-parallel reductions, matching the dispatch used by the kernel.
- **When an MLX reference kernel exists, attach it with `.with_reference()`**
  so the bench runner times both kernels live and reports `ref_gbps` /
  `mt_pct` without relying on a static JSON baseline.  See the section below.

---

## Live reference benchmarking (`.with_reference()`)

For every op that has a corresponding kernel in the reference source tree,
attach a `RefKernel` so `tile bench` can show a live side-by-side comparison:

```
bench  copy/copy              f32    278.2 GB/s    3.6 µs
                              ref    274.1 GB/s  +1.5%
```

The runner reads the `.metal` source from the directory set by
`[bench] reference_metal_path` in `tile.toml`, compiles it (PSO is cached
after the first run), and times it with the same warmup / iteration count as
the MetalTile kernel.

The path is **project-level config** — nothing is hardcoded in kernel files.
For MetalTile the `tile.toml` points at the checked-out MLX source tree:

```toml
[bench]
reference_metal_path = ".cache/mlx/mlx/backend/metal/kernels"
```

Any project can set this to its own reference kernel directory instead.

### Pattern

```rust
use metaltile::core::bench::{BenchBuffer, BenchSetup, RefKernel};

pub mod kernel_benches {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::bench;
    use metaltile::core::{DType, bench::{BenchBuffer, BenchSetup, RefKernel}};

    use super::*;

    #[bench(name = "copy/copy", dtypes = [f32, f16, bf16])]
    fn bench_mt_copy(dt: DType) -> BenchSetup {
        let n = 64 * 1024 * 1024usize;

        // ── MLX reference kernel ─────────────────────────────────────────
        // fn_name: the exact Metal kernel function name in the .metal file.
        // metal_file: path relative to reference_metal_path in tile.toml.
        // buffers: positional — index 0 = [[buffer(0)]], etc.
        let ref_kernel = RefKernel {
            fn_name: format!("copy{}", dt.mlx_type_suffix()),
            metal_file: "copy.metal".to_string(),
            buffers: vec![
                BenchBuffer::random("src", n, dt),
                BenchBuffer::zeros("dst", n, dt).output(),
            ],
            grid: metaltile::core::bench::Grid::new_1d(n, 256),
        };

        // ── MetalTile DSL kernel ─────────────────────────────────────────
        BenchSetup::new(mt_copy::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("a",   n, dt))
            .buffer(BenchBuffer::zeros("out",  n, dt).output())
            .grid_1d(n, 1024)
            .with_reference(ref_kernel)   // ← runner times both, reports pct
    }
}
```

### Rules

- `fn_name` must be the **exact** Metal compute kernel function name as it
  appears in the `.metal` source (not a C++ template instantiation name —
  check the `.metal` file directly).
- `metal_file` is relative to `reference_metal_path`.  For MLX kernels this
  is usually the filename at the top of the kernels directory, e.g.
  `"unary.metal"`, `"binary.metal"`, `"copy.metal"`.
- Buffers are bound **positionally**.  Match the order of `[[buffer(N)]]`
  arguments in the Metal function signature exactly.
- Use the **same N** as the main kernel so bandwidth figures are comparable.
- If `tile.toml` has no `reference_metal_path`, the reference is silently
  skipped — `ref_gbps` stays `None` and the CLI omits the ref line.
- Omit `.with_reference()` for ops with no reference equivalent (fused ops,
  custom ffai kernels, stubs).  The runner still reports `mt_gbps`.

---

## GPU-vs-GPU reference testing (`compare_against`)

When computing expected values on the CPU is expensive or impractical, use
`.compare_against(ref_setup)` to run a known-correct reference kernel on the
GPU and compare its output against the kernel under test.

The two canonical use-cases are:

1. **Optimised variant vs scalar baseline** — an MMA / batched / tiled
   variant tested against the scalar m1 kernel that is already known-correct.
2. **New kernel vs existing kernel** — a new MetalTile kernel for an op that
   already has a working kernel; validate the new one matches the old before
   switching over.

### Rules

- The `ref_setup` uses `.input()` for **all** buffers, including the zeroed
  output buffer.  Do not call `.expect()` on either setup — the runner uses
  the reference kernel's GPU outputs as the expected values.
- Both setups must receive **identical input data** (same bytes, same
  shapes).  Build the input vecs once and share them.
- The reference kernel must itself be correct.  If the reference is broken,
  the test will always pass vacuously — pick the simplest, most-audited
  variant as the reference (usually m1 / scalar / f32).
- Reference kernels may use a different dtype from the kernel under test.
  Convert the inputs accordingly (e.g. pack as f32 for the reference, as
  bf16 for the main kernel).

### Example

```rust
#[test_kernel(dtypes = [f16, bf16], tol = 1e-2)]
fn test_moe_mma_m8(dt: DType) -> TestSetup {
    let n = 512usize;
    let x: Vec<f32> = (0..n).map(|i| 0.05 * (i as f32 * 0.013).sin()).collect();

    // ── reference: scalar m1 kernel running in f32 ──────────────────────
    let mut ref_kernel = mt_moe_m1::kernel_ir_for(DType::F32);
    ref_kernel.mode = KernelMode::Reduction;
    let ref_setup = TestSetup::new(ref_kernel)
        .input(TestBuffer::from_vec("x",   pack_f32(&x, DType::F32), DType::F32))
        .input(TestBuffer::from_vec("out", vec![0u8; n * 4],          DType::F32))
        .constexpr("n", n as u32)
        .grid_3d(n as u32, 1, 1, [32, 1, 1]);

    // ── kernel under test: MMA m8 variant ───────────────────────────────
    let mut kernel = mt_moe_mma_m8::kernel_ir_for(dt);
    kernel.mode = KernelMode::Reduction;
    TestSetup::new(kernel)
        .input(TestBuffer::from_vec("x",   pack_f32(&x, dt),         dt))
        .input(TestBuffer::from_vec("out", vec![0u8; n * dt.size_bytes()], dt))
        .constexpr("n", n as u32)
        .grid_3d(n as u32, 1, 1, [256, 1, 1])
        .compare_against(ref_setup)    // ← runner dispatches ref first,
                                       //   uses its outputs as expected
}
```

---

## Complete annotated example

`src/mlx/copy.rs` is the canonical reference implementation.  It has all
three sections with working tests and a bench that matches the MLX baseline.

---

## `name` → baseline mapping

| Baseline `op`     | Expected `name` prefix in `#[bench]` |
|-------------------|--------------------------------------|
| `unary`           | `unary/<subop>` e.g. `unary/exp`     |
| `binary`          | `binary/<subop>`                     |
| `all_reduce`      | `all_reduce/<subop>`                 |
| `row_reduce`      | `row_reduce/<subop>`                 |
| `rms_norm`        | `rms_norm/<subop>`                   |
| `layer_norm`      | `layer_norm/layer_norm`              |
| `softmax`         | `softmax/softmax`                    |
| `gemv`            | `gemv/gemv`                          |
| `quantized`       | `quantized/<subop>`                  |
| `steel_gemm_fused`| `steel_gemm_fused/<tile_config>`     |
| `sdpa`            | `sdpa/<subop>`                       |
| `copy`            | `copy/copy`                          |
| `scan`            | `scan/scan`                          |
| `sort`            | `sort/sort`                          |

---

## What to avoid

- **Do not** add `#[bench]` or `#[test_kernel]` outside these two modules.
- **Do not** use `macro_rules!` inside a `#[kernel]` body — declarative
  macros are silently dropped by the proc-macro.
- **Do not** use `while` or `return` inside `#[kernel]` — silently dropped.
- **Do not** reference `crate::mlx::benches` — that module is being removed.
- **Do not** allocate buffers larger than ~32 M elements in `kernel_tests` —
  this causes GPU memory pressure and display compositor flicker on Apple
  Silicon while tests run.
